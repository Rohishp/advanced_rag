"""Self-RAG over company PDFs (NexaAI Solutions), with web search fallback.

Built step by step.
Step 1: configuration, PDF loading and persistent indexing.
Step 2: retrieval decision -> (generate_direct | retrieve).
"""

# ---------- Standard library ----------
import hashlib
import json
from pathlib import Path
from typing import List, Literal, TypedDict

# ---------- Third-party ----------
import pymupdf4llm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ---------- LangChain / LangGraph ----------
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

load_dotenv()


# CONFIG
DOCS_DIR = Path("./company_docs")          # company PDFs to index
DB_DIR = Path("./chroma_company")          # Chroma storage (separate from rag.py)
COLLECTION = "company"
MANIFEST = DB_DIR / "manifest.json"        # file fingerprint per indexed PDF

EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-5.4-mini"

CHUNK_SIZE = 300        # tokens per chunk (short documents -> small chunks)
CHUNK_OVERLAP = 50      # tokens shared between neighbouring chunks
RETRIEVER_K = 4         # chunks returned per query
MAX_CONCURRENCY = 10    # parallel LLM calls in .batch()
INDEX_BATCH = 500       # max IDs per Chroma add/delete call

RECURSION_LIMIT = 50    # max graph steps per question
GRAPH_PNG = "self_rag_graph.png"

TEST_QUESTIONS = [
    "What is machine learning?",                                 # direct
    "How many days of annual leave do employees get?",           # docs
    "What does the Growth plan cost per user?",                  # docs (pricing)
    "Who is the CEO of NexaAI?",                                 # docs
    "Describe NexaAI's company culture.",                        # IsSUP strictness
    "What is the refund policy of NexaAI?",                      # not in docs
    "Who won the latest ICC Men's T20 World Cup match?",         # web fallback
]


# 1) LOAD PDFs
def clean(text: str) -> str:
    """Strip lone surrogates that PDF extraction sometimes produces."""
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")


def load_pdf(path: Path) -> list[Document]:
    """Read one PDF as Markdown, one Document per non-empty page."""
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    return [
        Document(
            page_content=clean(p["text"]),
            metadata={
                "source": path.name,
                # key differs between pymupdf4llm versions; never store None
                # (Chroma silently drops None metadata values)
                "page": p["metadata"].get("page") or p["metadata"].get("page_number") or n,
            },
        )
        for n, p in enumerate(pages, 1)
        if p["text"].strip()
    ]


# INDEX (persistent; unchanged PDFs are skipped without loading)
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)

vector_store = Chroma(
    collection_name=COLLECTION,
    embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
    persist_directory=str(DB_DIR),
)


def chunk_id(doc: Document) -> str:
    """Stable ID from source + page + text, so re-runs skip existing chunks."""
    key = f"{doc.metadata['source']}|{doc.metadata['page']}|{doc.page_content}"
    return hashlib.sha256(key.encode()).hexdigest()


def file_hash(path: Path) -> str:
    """Fingerprint of the PDF file itself."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def delete_source(source: str) -> None:
    """Remove all chunks that came from one PDF."""
    old_ids = vector_store.get(where={"source": source}, include=[])["ids"]
    for start in range(0, len(old_ids), INDEX_BATCH):
        vector_store.delete(ids=old_ids[start:start + INDEX_BATCH])


def save_manifest(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def index_folder(folder: Path) -> None:
    """Index new or changed PDFs, skip unchanged ones, drop removed ones."""
    if not folder.is_dir():
        raise FileNotFoundError(f"PDF folder not found: {folder.resolve()}")

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    pdfs = sorted(folder.glob("*.pdf"))

    # PDFs that were indexed before but no longer exist: remove their chunks
    for name in set(manifest) - {p.name for p in pdfs}:
        delete_source(name)
        del manifest[name]
        save_manifest(manifest)
        print(f"{name}: removed from folder, chunks deleted")

    for pdf in pdfs:
        fingerprint = file_hash(pdf)
        if manifest.get(pdf.name) == fingerprint:
            print(f"{pdf.name}: unchanged, skipped")
            continue

        if pdf.name in manifest:  # changed file: drop its old chunks first
            delete_source(pdf.name)

        chunks = splitter.split_documents(load_pdf(pdf))

        # De-duplicate identical chunks (Chroma raises DuplicateIDError otherwise)
        unique: dict[str, Document] = {}
        for c in chunks:
            unique.setdefault(chunk_id(c), c)
        ids = list(unique)

        existing = set(vector_store.get(ids=ids, include=[])["ids"]) if ids else set()
        new_ids = [i for i in ids if i not in existing]
        for start in range(0, len(new_ids), INDEX_BATCH):
            batch = new_ids[start:start + INDEX_BATCH]
            vector_store.add_documents([unique[i] for i in batch], ids=batch)

        print(f"{pdf.name}: {len(chunks)} chunks, {len(ids)} unique, "
              f"{len(new_ids)} newly embedded")

        manifest[pdf.name] = fingerprint   # save after each PDF to keep progress
        save_manifest(manifest)

    total = len(vector_store.get(include=[])["ids"])
    print(f"Collection '{COLLECTION}': {total} chunks in total")


retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)


# GRAPH STATE
class State(TypedDict, total=False):
    """Data flowing through the graph; each node returns only what it changes."""
    question: str
    # decide_retrieval
    need_retrieval: bool
    # retrieve
    docs: List[Document]
    # generate_direct
    answer: str
    # grade_documents
    relevant_docs: List[Document]


# 4) NODES (in execution order)

# ---------- Node: DECIDE whether retrieval is needed ----------
class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        description="True if the question needs the company knowledge base or "
                    "other external/up-to-date sources; False for general knowledge."
    )


decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether a question needs retrieval before answering.\n\n"
            "The internal knowledge base contains documents about the company "
            "NexaAI Solutions: HR policies and leave policies, the company profile "
            "and leadership team, and its products and pricing plans.\n\n"
            "Set should_retrieve=true if:\n"
            "- the question is about NexaAI (its people, policies, products, plans, "
            "prices, culture, or anything else company-specific), or\n"
            "- it needs recent, current or fast-changing information (news, results, "
            "latest events), or specific facts you cannot answer reliably.\n"
            "Set should_retrieve=false only for general, timeless knowledge "
            "(definitions, concepts, explanations).\n"
            "If unsure, choose true.",
        ),
        ("human", "Question: {question}"),
    ]
)
decide_retrieval_chain = decide_retrieval_prompt | llm.with_structured_output(RetrieveDecision)


def decide_retrieval(state: State) -> State:
    decision = decide_retrieval_chain.invoke({"question": state["question"]})
    return {"need_retrieval": decision.should_retrieve}


# ---------- Node: DIRECT answer (no retrieval) ----------
direct_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only your general knowledge. Be concise.\n"
            "If the answer requires specific, company-internal or recent information "
            "you do not reliably know, say: "
            "'I don't know based on my general knowledge.'",
        ),
        ("human", "{question}"),
    ]
)


def generate_direct(state: State) -> State:
    out = (direct_prompt | llm).invoke({"question": state["question"]})
    return {"answer": out.content}


# ---------- Node: RETRIEVE ----------
def retrieve(state: State) -> State:
    return {"docs": retriever.invoke(state["question"])}

# ---------- Node: GRADE retrieved chunks (topic-level, lenient) ----------
class RelevanceGrade(BaseModel):
    is_relevant: bool = Field(
        description="True if the chunk is about the same entity or topic area as the question."
    )


relevance_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You judge whether a retrieved document chunk is relevant to a question, "
            "at the TOPIC level.\n\n"
            "A chunk is relevant if it discusses the same entity or topic area as the "
            "question. It does NOT need to contain the exact answer; whether the answer "
            "is actually supported is checked later.\n\n"
            "Examples:\n"
            "- HR or leave policies are relevant to questions about leave, probation, "
            "notice periods, benefits or working conditions.\n"
            "- Product and pricing chunks are relevant to questions about plans, prices, "
            "trials, billing or refunds.\n"
            "- Company profile chunks are relevant to questions about leadership, "
            "history, size, vision or culture.\n\n"
            "A chunk is NOT relevant if it is about a clearly different subject "
            "(e.g. company documents for a question about sports or world news).\n"
            "When unsure, set is_relevant=true.",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
)
relevance_chain = relevance_prompt | llm.with_structured_output(RelevanceGrade)


def grade_documents(state: State) -> State:
    """Grade all retrieved chunks in parallel; keep the relevant ones."""
    docs = state.get("docs", [])
    grades = relevance_chain.batch(
        [{"question": state["question"], "chunk": d.page_content} for d in docs],
        config={"max_concurrency": MAX_CONCURRENCY},
    )
    return {"relevant_docs": [d for d, g in zip(docs, grades) if g.is_relevant]}

# ---------- Routing ----------
def route_after_decide(state: State) -> Literal["retrieve", "generate_direct"]:
    return "retrieve" if state["need_retrieval"] else "generate_direct"


# BUILD GRAPH
def build_graph():
    """START -> decide_retrieval -> (generate_direct | retrieve -> grade_documents) -> END"""
    g = StateGraph(State)
    g.add_node("decide_retrieval", decide_retrieval)
    g.add_node("generate_direct", generate_direct)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)

    g.add_edge(START, "decide_retrieval")
    g.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {"retrieve": "retrieve", "generate_direct": "generate_direct"},
    )
    g.add_edge("generate_direct", END)
    g.add_edge("retrieve", "grade_documents")
    g.add_edge("grade_documents", END)  # temporary end of the retrieval path
    return g.compile()


def save_graph_diagram(app, path: str = GRAPH_PNG) -> None:
    """Render the graph via mermaid.ink (needs internet); print Mermaid text offline."""
    try:
        Path(path).write_bytes(app.get_graph().draw_mermaid_png())
    except Exception:
        print(app.get_graph().draw_mermaid())



#RUN

def print_result(res: State) -> None:
    print("=" * 100)
    print("QUESTION:", res["question"])
    print("NEED RETRIEVAL:", res.get("need_retrieval"))

    if res.get("docs"):
        relevant = res.get("relevant_docs", [])
        print(f"\nRETRIEVED CHUNKS ({len(relevant)}/{len(res['docs'])} relevant):")
        for d in res["docs"]:
            mark = "keep" if d in relevant else "drop"
            preview = d.page_content[:80].replace("\n", " ")
            print(f"  [{mark}] {d.metadata.get('source', '?')} "
                  f"p.{d.metadata.get('page', '?')}: {preview}")

    print("\nANSWER:\n" + res.get("answer", "(none yet)"))


def main() -> None:
    index_folder(DOCS_DIR)
    app = build_graph()
    save_graph_diagram(app)
    for q in TEST_QUESTIONS:
        print_result(app.invoke({"question": q}, config={"recursion_limit": RECURSION_LIMIT}))


if __name__ == "__main__":
    main()