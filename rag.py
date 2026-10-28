"""Corrective RAG over local PDF books, with web search fallback.

Flow: retrieve -> evaluate chunks -> (refine | rewrite query -> web search -> refine)
      -> generate
"""

# ---------- Standard library ----------
import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import List, Literal, Optional, TypedDict

# ---------- Third-party ----------
import pymupdf4llm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ---------- LangChain / LangGraph ----------
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

load_dotenv()


# ======================================================================
# CONFIG
# ======================================================================
DOCS_DIR = Path("./documents")             # PDFs to index
DB_DIR = "./chroma_db"                     # Chroma storage
MANIFEST = Path(DB_DIR) / "manifest.json"  # fingerprints of indexed PDFs

EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-5.4-mini"

CHUNK_SIZE = 512        # tokens per chunk
CHUNK_OVERLAP = 64      # tokens shared between neighbouring chunks
RETRIEVER_K = 4         # chunks returned per question
RETRIEVER_FETCH_K = 20  # candidates considered by MMR

UPPER_TH = 0.7          # evaluator score above this: clearly relevant
LOWER_TH = 0.3          # evaluator score below this: irrelevant

WEB_MAX_RESULTS = 5
MAX_CONCURRENCY = 10    # parallel LLM calls in .batch()

QUESTIONS = [
    "Explain the bias–variance tradeoff",                                           # in the books
    "What are attention mechanisms and why are they important in current models?",  # in the books
    "AI news from last week",                                                       # needs the web
]


# ======================================================================
# 1) LOAD PDFs
# ======================================================================
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
                # key differs between pymupdf4llm versions; fall back to position
                "page": p["metadata"].get("page") or p["metadata"].get("page_number") or n,
            },
        )
        for n, p in enumerate(pages, 1)
        if p["text"].strip()
    ]


# ======================================================================
# 2) INDEX (persistent; only new or changed PDFs are processed)
# ======================================================================
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)

vector_store = Chroma(
    collection_name="books",
    embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
    persist_directory=DB_DIR,
)


def chunk_id(doc: Document) -> str:
    """Stable ID from source + page + text, so re-runs skip existing chunks."""
    key = f"{doc.metadata['source']}|{doc.metadata['page']}|{doc.page_content}"
    return hashlib.sha256(key.encode()).hexdigest()


def file_hash(path: Path) -> str:
    """Fingerprint of the PDF file itself."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_folder(folder: Path, batch_size: int = 500) -> None:
    """Index new or changed PDFs; skip unchanged ones without loading them."""
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

    for pdf in sorted(folder.glob("*.pdf")):
        fingerprint = file_hash(pdf)
        if manifest.get(pdf.name) == fingerprint:
            print(f"{pdf.name}: unchanged, skipped")
            continue

        # Changed file: remove its old chunks first
        if pdf.name in manifest:
            old_ids = vector_store.get(where={"source": pdf.name})["ids"]
            for start in range(0, len(old_ids), batch_size):
                vector_store.delete(ids=old_ids[start:start + batch_size])

        chunks = splitter.split_documents(load_pdf(pdf))

        # De-duplicate identical chunks (repeated headers etc.)
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

        # Save after each book so a crash halfway keeps progress
        manifest[pdf.name] = fingerprint
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2))


# MMR: fetch FETCH_K similar chunks, keep K that are relevant AND diverse
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_FETCH_K},
)

llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
tavily = TavilySearch(max_results=WEB_MAX_RESULTS, topic="general")


# ======================================================================
# 3) GRAPH STATE
# ======================================================================
class State(TypedDict, total=False):
    """Data flowing through the graph; each node returns only what it changes."""
    question: str
    # retrieve
    docs: List[Document]
    # eval_each_doc
    doc_scores: List[float]
    doc_reasons: List[str]
    good_docs: List[Document]
    verdict: Literal["CORRECT", "INCORRECT", "AMBIGUOUS"]
    reason: str
    # rewrite_query + web_search
    web_query: str
    web_time_range: Optional[str]
    web_docs: List[Document]
    # refine
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    # generate
    answer: str


# ======================================================================
# 4) NODES (in execution order)
# ======================================================================

# ---------- Node: RETRIEVE ----------
def retrieve(state: State) -> State:
    return {"docs": retriever.invoke(state["question"])}


# ---------- Node: EVALUATE retrieved chunks ----------
class DocEvalScore(BaseModel):
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
    """Score every chunk, then decide CORRECT / INCORRECT / AMBIGUOUS."""
    docs = state["docs"]
    results = doc_eval_chain.batch(
        [{"question": state["question"], "chunk": d.page_content} for d in docs],
        config={"max_concurrency": MAX_CONCURRENCY},
    )
    scores = [r.score for r in results]
    good = [d for d, s in zip(docs, scores) if s > LOWER_TH]
    base = {"doc_scores": scores, "doc_reasons": [r.reason for r in results]}

    if any(s > UPPER_TH for s in scores):
        return {**base, "verdict": "CORRECT", "good_docs": good,
                "reason": f"At least one retrieved chunk scored > {UPPER_TH}."}

    if not scores or all(s < LOWER_TH for s in scores):
        return {**base, "verdict": "INCORRECT", "good_docs": [],
                "reason": f"All retrieved chunks scored < {LOWER_TH}."}

    return {**base, "verdict": "AMBIGUOUS", "good_docs": good,
            "reason": f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}."}


# ---------- Node: REWRITE question into a web search query ----------
class WebQuery(BaseModel):
    query: str = Field(description="Keyword web search query, 6-14 words.")
    time_range: Optional[Literal["day", "week", "month", "year"]] = Field(
        default=None,
        description="Recency filter if the question asks for recent information, otherwise null.",
    )


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6-14 words).\n"
            "- Today's date is {today}. Replace relative dates (e.g. 'last week') "
            "with explicit dates or months.\n"
            "- If the question asks for recent information (recent/latest/last week/"
            "last month), set time_range to day, week, month or year. Otherwise leave it null.\n"
            "- Do NOT answer the question.",
        ),
        ("human", "Question: {question}"),
    ]
)
rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)


def rewrite_query(state: State) -> State:
    out = rewrite_chain.invoke(
        {"question": state["question"], "today": date.today().isoformat()}
    )
    return {"web_query": out.query, "web_time_range": out.time_range}


# ---------- Node: WEB SEARCH ----------
def web_search(state: State) -> State:
    """Search the web; keep only page text in page_content (title/URL in metadata)."""
    args = {"query": state.get("web_query") or state["question"]}
    if state.get("web_time_range"):
        args["time_range"] = state["web_time_range"]

    try:
        response = tavily.invoke(args)
    except Exception as e:  # missing API key, network error, ...
        print(f"Web search failed: {e}")
        return {"web_docs": []}

    results = response.get("results", []) if isinstance(response, dict) else []
    return {
        "web_docs": [
            Document(
                page_content=r["content"],
                metadata={"source": r.get("title", "web"), "url": r.get("url", "")},
            )
            for r in results
            if r.get("content")
        ]
    }


# ---------- Node: REFINE (decompose -> filter -> recompose) ----------
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
filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)


def decompose_to_sentences(text: str) -> List[str]:
    """Split text into sentences; drop fragments of 20 characters or fewer."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def refine(state: State) -> State:
    """Keep only the sentences (from books and/or web) that help answer."""
    docs = state.get("good_docs", []) + state.get("web_docs", [])
    strips = decompose_to_sentences("\n\n".join(d.page_content for d in docs))

    results = filter_chain.batch(
        [{"question": state["question"], "sentence": s} for s in strips],
        config={"max_concurrency": MAX_CONCURRENCY},
    )
    kept = [s for s, r in zip(strips, results) if r.keep]

    return {"strips": strips, "kept_strips": kept, "refined_context": "\n".join(kept)}


# ---------- Node: GENERATE ----------
answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful ML tutor. Answer ONLY using the provided refined bullets.\n"
            "If the bullets are empty or insufficient, say: "
            "'I don't know based on the available sources.'",
        ),
        ("human", "Question: {question}\n\nRefined context:\n{refined_context}"),
    ]
)


def generate(state: State) -> State:
    out = (answer_prompt | llm).invoke(
        {"question": state["question"], "refined_context": state["refined_context"]}
    )
    return {"answer": out.content}


# ---------- Routing ----------
def route_after_eval(state: State) -> str:
    """CORRECT: books only. INCORRECT: web only. AMBIGUOUS: web + partial book chunks."""
    return "refine" if state["verdict"] == "CORRECT" else "web_search"


# ======================================================================
# 5) BUILD GRAPH
# ======================================================================
def build_graph():
    """START -> retrieve -> eval_each_doc -> (refine | rewrite_query -> web_search -> refine)
    -> generate -> END"""
    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("eval_each_doc", eval_each_doc)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("web_search", web_search)
    g.add_node("refine", refine)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "eval_each_doc")
    g.add_conditional_edges(
        "eval_each_doc",
        route_after_eval,
        {"refine": "refine", "web_search": "rewrite_query"},
    )
    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search", "refine")
    g.add_edge("refine", "generate")
    g.add_edge("generate", END)
    return g.compile()


def save_graph_diagram(app, path: str = "graph.png") -> None:
    """Render the graph via mermaid.ink (needs internet); print Mermaid text offline."""
    try:
        Path(path).write_bytes(app.get_graph().draw_mermaid_png())
    except Exception:
        print(app.get_graph().draw_mermaid())


# ======================================================================
# 6) RUN
# ======================================================================
def print_result(res: State) -> None:
    print("=" * 100)
    print("QUESTION:", res["question"])
    print(f"VERDICT:  {res['verdict']} ({res['reason']})")

    print("\nBOOK CHUNKS:")
    for d, s, why in zip(res["docs"], res["doc_scores"], res["doc_reasons"]):
        print(f"  {s:.2f}  {d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}: {why}")

    if res.get("web_query"):
        print(f"\nWEB QUERY: {res['web_query']} (time_range={res.get('web_time_range')})")
        for d in res.get("web_docs", []):
            print(f"  web: {d.metadata['source']} ({d.metadata['url']})")

    print("\nANSWER:\n" + res["answer"])


def main() -> None:
    index_folder(DOCS_DIR)
    app = build_graph()
    save_graph_diagram(app)
    for q in QUESTIONS:
        print_result(app.invoke({"question": q}))


if __name__ == "__main__":
    main()