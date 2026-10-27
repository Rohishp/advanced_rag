from ast import List
import re
from typing import List, TypedDict

import hashlib
from langchain_chroma import Chroma
from pathlib import Path
import pymupdf4llm
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path("./documents")

def clean(text: str) -> str:
    # strip lone surrogates from PDF extraction
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")

def load_pdf(path: Path) -> list[Document]:
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    return [
        Document(
            page_content=clean(p["text"]),
            metadata={
                "source": path.name,
                "page": p["metadata"].get("page") or p["metadata"].get("page_number") or n,
            },
        )
        for n, p in enumerate(pages, 1)
        if p["text"].strip()
    ]

DB_DIR = "./chroma_db"

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=64
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name="books",
    embedding_function=embeddings,
    persist_directory=DB_DIR,
)

def chunk_id(doc: Document) -> str:
    """Make a unique, repeatable ID from source + page + text.
    Same chunk -> same ID every run, which lets us skip re-embedding it."""
    key = f"{doc.metadata['source']}|{doc.metadata['page']}|{doc.page_content}"
    return hashlib.sha256(key.encode()).hexdigest()

MANIFEST = Path(DB_DIR) / "manifest.json"   # lives inside chroma_db, so deleting the folder resets it too


def file_hash(path: Path) -> str:
    """Fingerprint of the PDF file itself: changes only if the file changes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_folder(folder: Path, batch_size: int = 500) -> None:
    """Index new or changed PDFs; skip unchanged ones without loading them."""
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

    for pdf in sorted(folder.glob("*.pdf")):
        fingerprint = file_hash(pdf)

        # Unchanged since last indexing -> skip the slow load/split entirely
        if manifest.get(pdf.name) == fingerprint:
            print(f"{pdf.name}: unchanged, skipped")
            continue

        # File was indexed before but has changed -> remove its old chunks first
        if pdf.name in manifest:
            old_ids = vector_store.get(where={"source": pdf.name})["ids"]
            for start in range(0, len(old_ids), batch_size):
                vector_store.delete(ids=old_ids[start:start + batch_size])

        chunks = splitter.split_documents(load_pdf(pdf))

        # Remove identical chunks (repeated headers etc.)
        unique: dict[str, Document] = {}
        for c in chunks:
            unique.setdefault(chunk_id(c), c)
        ids = list(unique)

        existing = set(vector_store.get(ids=ids)["ids"])
        new_ids = [i for i in ids if i not in existing]
        for start in range(0, len(new_ids), batch_size):
            batch = new_ids[start:start + batch_size]
            vector_store.add_documents([unique[i] for i in batch], ids=batch)

        print(f"{pdf.name}: {len(chunks)} chunks, {len(ids)} unique, "
              f"{len(new_ids)} newly embedded")

        # Record this file as indexed (saved after each book, so a crash
        # halfway doesn't lose progress)
        manifest[pdf.name] = fingerprint
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2))



# The retriever finds chunks for a question.
# MMR = fetch the 20 most similar chunks, then pick 4 that are relevant
# AND different from each other (avoids 4 near-duplicate passages).
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20},
)


# 4) LLM + prompt
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

# 5) GRAPH
class State(TypedDict, total=False):
    """Shared data that flows through the graph. Each node reads what
    it needs and returns only the fields it changes.
    total=False -> fields are optional, so we can start with just 'question'."""
    question: str
    docs: List[Document]       # chunks returned by the retriever
    strips: List[str]          # all sentences cut from those chunks
    kept_strips: List[str]     # sentences the filter judged relevant
    refined_context: str       # kept sentences joined into one text
    answer: str                # final answer
class State(TypedDict, total=False):
    question: str
    docs: list[Document]
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    answer: str

def retrieve(state: State) -> State:
    q = state["question"]
    return {"docs": retriever.invoke(q)}

# ---------- Helper: DECOMPOSE ----------
def decompose_to_sentences(text: str) -> List[str]:
    """Split a block of text into sentences ("strips")."""
    # Collapse newlines/tabs/multiple spaces into single spaces
    text = re.sub(r"\s+", " ", text).strip()
    # Split after . ! or ? followed by whitespace.
    # (?<=...) is a lookbehind: the punctuation stays with its sentence.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # Drop tiny fragments (page numbers, headings, formula bits)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


class KeepOrDrop(BaseModel):
    keep: bool


filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the sentence directly helps answer the question.\n"
            "Use ONLY the sentence. Output JSON only.",
        ),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ]
)

# prompt | model: fill the template, send it to the LLM, and parse the
# reply into a KeepOrDrop object (because of with_structured_output).
filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)

# ---------- Node 2: REFINE (decompose -> filter -> recompose) ----------
def refine(state: State) -> State:
    q = state["question"]

    # Merge the retrieved chunks into one big text
    context = "\n\n".join(d.page_content for d in state["docs"]).strip()

    # 1) DECOMPOSE: text -> list of sentences
    strips = decompose_to_sentences(context)

    # 2) FILTER: ask the LLM about every sentence.
    # .batch() sends the requests in parallel (up to 10 at a time)
    # instead of one after another, which is much faster.
    results = filter_chain.batch(
        [{"question": q, "sentence": s} for s in strips],
        config={"max_concurrency": 10},
    )
    # Pair each sentence with its verdict and keep the "true" ones
    kept = [s for s, r in zip(strips, results) if r.keep]

    # 3) RECOMPOSE: join the survivors into the context for the answer
    refined_context = "\n".join(kept).strip()

    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context,
    }




answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful ML tutor. Answer ONLY using the provided refined bullets.\n"
            "If the bullets are empty or insufficient, say: 'I don't know based on the provided books.'",
        ),
        ("human", "Question: {question}\n\nRefined context:\n{refined_context}"),
    ]
)


def generate(state: State) -> State:
    """Write the final answer using only the filtered sentences."""
    out = (answer_prompt | llm).invoke(
        {"question": state["question"], "refined_context": state["refined_context"]}
    )
    return {"answer": out.content}  # .content = the text of the reply


g = StateGraph(State)
g.add_node("retrieve", retrieve)
g.add_node("refine", refine)
g.add_node("generate", generate)

# Straight line: START -> retrieve -> refine -> generate -> END
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "refine")
g.add_edge("refine", "generate")
g.add_edge("generate", END)

app = g.compile()  # turn the definition into a runnable app

# 5) Run
res = app.invoke({"question": "Explain the bias–variance tradeoff"})

print(res["answer"])
print()

print(f"Kept {len(res['kept_strips'])} of {len(res['strips'])} sentences:")
for s in res["kept_strips"]:
    print("-", s)
print("*" * 100)

for i, d in enumerate(res["docs"], 1):
    print(f"[{i}] {d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}")
    print(d.page_content[:500])
    print("*" * 100)