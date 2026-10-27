import re
from typing import List, Literal, TypedDict
from langchain_tavily import TavilySearch
import json
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


index_folder(DOCS_DIR)
# The retriever finds chunks for a question.
# MMR = fetch the 20 most similar chunks, then pick 4 that are relevant
# AND different from each other (avoids 4 near-duplicate passages).
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20},
)


# 4) LLM + prompt
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
# Relevance thresholds for the retrieval evaluator
UPPER_TH = 0.7   # above this: chunk is clearly relevant
LOWER_TH = 0.3   # below this: chunk is irrelevant

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
    doc_scores: List[float]     # relevance score per retrieved chunk
    good_docs: List[Document]   # chunks scoring above LOWER_TH
    verdict: Literal["CORRECT", "INCORRECT", "AMBIGUOUS"]
    reason: str                 # why that verdict was chosen
    doc_reasons: List[str]      # evaluator's justification per chunk
    web_docs: List[Document]    # results from web search


def retrieve(state: State) -> State:
    q = state["question"]
    return {"docs": retriever.invoke(q)}

# ---------- Node: EVALUATE retrieved chunks ----------
class DocEvalScore(BaseModel):
    """Structured output: relevance score plus a short justification."""
    score: float
    reason: str


doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.\n"
            "Output JSON only.",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
)

doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)


def eval_each_doc(state: State) -> State:
    """Score every retrieved chunk and decide CORRECT / INCORRECT / AMBIGUOUS."""
    q = state["question"]
    docs = state["docs"]

    # Score all chunks in parallel instead of one by one
    results = doc_eval_chain.batch(
        [{"question": q, "chunk": d.page_content} for d in docs],
        config={"max_concurrency": 10},
    )
    scores = [r.score for r in results]
    reasons = [r.reason for r in results]

    # Chunks worth refining: anything above the lower threshold
    good = [d for d, s in zip(docs, scores) if s > LOWER_TH]

    # CORRECT: at least one chunk is clearly relevant
    if any(s > UPPER_TH for s in scores):
        return {
            "doc_scores": scores,
            "good_docs": good,
            "verdict": "CORRECT",
            "doc_reasons": reasons,
            "reason": f"At least one retrieved chunk scored > {UPPER_TH}.",
        }

    # INCORRECT: nothing retrieved, or every chunk is irrelevant
    if not scores or all(s < LOWER_TH for s in scores):
        return {
            "doc_scores": scores,
            "good_docs": [],
            "verdict": "INCORRECT",
            "doc_reasons": reasons,
            "reason": f"All retrieved chunks scored < {LOWER_TH}. No chunk was sufficient.",
        }

    # AMBIGUOUS: some signal, but nothing convincing
    return {
        "doc_scores": scores,
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "doc_reasons": reasons,
        "reason": f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}. Mixed relevance signals.",
    }

# ---------- Node: WEB SEARCH (when the books aren't enough) ----------
tavily = TavilySearch(max_results=5, topic="general")


def web_search(state: State) -> State:
    """Search the web and turn the results into Documents."""
    try:
        response = tavily.invoke({"query": state["question"]})
    except Exception as e:  # e.g. missing API key, network error
        print(f"Web search failed: {e}")
        return {"web_docs": []}

    results = response.get("results", []) if isinstance(response, dict) else []
    web_docs = [
        Document(
            page_content=r.get("content", ""),  # only the text gets refined
            metadata={"source": r.get("title", "web"), "url": r.get("url", "")},
        )
        for r in results
        if r.get("content")
    ]
    return {"web_docs": web_docs}

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

    # Book chunks the evaluator kept + web results (either may be empty)
    docs = state.get("good_docs", []) + state.get("web_docs", [])
    context = "\n\n".join(d.page_content for d in docs).strip()
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



def route_after_eval(state: State) -> str:
    """Pick the next node based on the evaluator's verdict."""
    if state["verdict"] == "CORRECT":
        return "refine"      # books are enough
    return "web_search"      # INCORRECT: web only; AMBIGUOUS: web + partial book chunks

g = StateGraph(State)
g.add_node("retrieve", retrieve)
g.add_node("eval_each_doc", eval_each_doc)
g.add_node("web_search", web_search)
g.add_node("refine", refine)
g.add_node("generate", generate)

# START -> retrieve -> eval_each_doc -> (refine | web_search -> refine) -> generate -> END
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "eval_each_doc")
g.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {"refine": "refine", "web_search": "web_search"},
)
g.add_edge("web_search", "refine")
g.add_edge("refine", "generate")
g.add_edge("generate", END)

app = g.compile()  # turn the definition into a runnable app

# Save a diagram of the graph (needs internet: rendered via mermaid.ink)
try:
    Path("graph.png").write_bytes(app.get_graph().draw_mermaid_png())
except Exception:
    print(app.get_graph().draw_mermaid())   # offline fallback: Mermaid text
    
# 5) Run
# 
QUESTIONS = [
    "Explain the bias–variance tradeoff",                                     # expect CORRECT
    "What are attention mechanisms and why are they important in current models?",  # likely AMBIGUOUS
    "AI news from last week",                                                 # expect INCORRECT
]

for q in QUESTIONS:
    res = app.invoke({"question": q})
    print("=" * 100)
    print("QUESTION:", q)
    print(f"VERDICT: {res['verdict']} ({res['reason']})")
    for d, s, why in zip(res["docs"], res["doc_scores"], res["doc_reasons"]):
        print(f"  {s:.2f}  {d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}: {why}")
    for d in res.get("web_docs", []):
        print(f"  web: {d.metadata['source']} ({d.metadata['url']})")
    print("\nANSWER:\n", res["answer"])
    
# print(res["answer"])
# print()

# print(f"Verdict: {res['verdict']} ({res['reason']})")
# for d, s in zip(res["docs"], res["doc_scores"]):
#     print(f"  {s:.2f}  {d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}")
# print("*" * 100)

# print(f"Kept {len(res['kept_strips'])} of {len(res['strips'])} sentences:")
# for s in res["kept_strips"]:
#     print("-", s)
# print("*" * 100)

# for i, d in enumerate(res["docs"], 1):
#     print(f"[{i}] {d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}")
#     print(d.page_content[:500])
#     print("*" * 100)

# for d, s, why in zip(res["docs"], res["doc_scores"], res["doc_reasons"]):
#     print(f"  {s:.2f}  {d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}: {why}")

