"""Self-RAG over a folder of PDFs, with a one-time web search fallback.

Configure the corpus in the CONFIG block; the graph flow is documented in build_graph().
"""

# ---------- Standard library ----------
import hashlib
import json
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
DOCS_DIR = Path("./company_docs")          # PDFs to index
DB_DIR = Path("./chroma_company")          # Chroma storage
COLLECTION = "company"
MANIFEST = DB_DIR / "manifest.json"        # fingerprint per indexed PDF

# What the knowledge base contains; used by the prompts that need to know.
KB_DESCRIPTION = (
    "internal documents of the company NexaAI Solutions: HR and leave policies, "
    "workplace conduct, the company profile and leadership team, and its products, "
    "pricing plans and billing"
)
CHAR_REPLACEMENTS = {"■": "₹"}             # PDF extraction artifacts (wrong -> right)

EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-5.4-mini"

CHUNK_SIZE = 300        # tokens per chunk (short documents -> small chunks)
CHUNK_OVERLAP = 50      # tokens shared between neighbouring chunks
RETRIEVER_K = 4         # chunks returned per query
INDEX_BATCH = 500       # max IDs per Chroma add/delete call
WEB_MAX_RESULTS = 5     # Tavily results for the web fallback

MAX_CONCURRENCY = 10    # parallel LLM calls in .batch()
MAX_REVISIONS = 2       # support-driven rewrites per answer
MAX_REWRITES = 2        # retrieval retries with a rewritten query
RECURSION_LIMIT = 50    # max graph steps per question (worst case uses 41)

NO_ANSWER = "I couldn't find this in the available sources."
GRAPH_PNG = "self_rag_graph.png"

TEST_QUESTIONS = [
    "What is machine learning?",                                 # direct
    "How many days of annual leave do employees get?",           # docs
    "What does the Growth plan cost per user?",                  # docs (pricing)
    "Who is the CEO of NexaAI?",                                 # docs
    "Describe NexaAI's company culture.",                        # support strictness
    "What is the refund policy of NexaAI?",                      # not in docs
    "Who won the latest ICC Men's T20 World Cup match?",         # web fallback
]


# ======================================================================
# CLIENTS
# ======================================================================
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
vector_store = Chroma(
    collection_name=COLLECTION,
    embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
    persist_directory=str(DB_DIR),
)
retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
tavily = TavilySearch(max_results=WEB_MAX_RESULTS, topic="general", search_depth="advanced")


# ======================================================================
# LOAD PDFs
# ======================================================================
def clean(text: str) -> str:
    """Strip lone surrogates and fix known extraction artifacts."""
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    for wrong, right in CHAR_REPLACEMENTS.items():
        text = text.replace(wrong, right)
    return text


def load_pdf(path: Path) -> List[Document]:
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


# ======================================================================
# INDEX (persistent; unchanged PDFs are skipped without loading)
# ======================================================================
def chunk_id(doc: Document) -> str:
    """Stable ID from source + page + text, so re-runs skip existing chunks."""
    key = f"{doc.metadata['source']}|{doc.metadata['page']}|{doc.page_content}"
    return hashlib.sha256(key.encode()).hexdigest()


def pdf_fingerprint(path: Path) -> str:
    """Hash of the PDF plus the settings that shape its chunks (changing them re-indexes)."""
    settings = json.dumps(
        {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
         "char_replacements": CHAR_REPLACEMENTS},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(path.read_bytes() + settings.encode()).hexdigest()


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

    for name in set(manifest) - {p.name for p in pdfs}:   # PDFs removed from the folder
        delete_source(name)
        del manifest[name]
        save_manifest(manifest)
        print(f"{name}: removed from folder, chunks deleted")

    for pdf in pdfs:
        fingerprint = pdf_fingerprint(pdf)
        if manifest.get(pdf.name) == fingerprint:
            print(f"{pdf.name}: unchanged, skipped")
            continue

        if pdf.name in manifest:  # changed file or settings: drop its old chunks first
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


# ======================================================================
# GRAPH STATE
# ======================================================================
class State(TypedDict, total=False):
    """Data flowing through the graph; each node returns only what it changes."""
    question: str
    # decide_retrieval
    need_retrieval: bool
    # rewrite_question
    retrieval_query: str
    rewrites: int
    # rewrite_web_query, web_search
    web_query: str
    web_time_range: Optional[Literal["day", "week", "month", "year"]]
    web_searched: bool
    # retrieve, web_search
    docs: List[Document]
    # grade_documents
    relevant_docs: List[Document]
    # generate, generate_direct, revise_answer, no_answer
    context: str
    answer: str
    # check_support
    support: Optional[Literal["fully_supported", "partially_supported", "no_support"]]
    evidence: List[str]
    support_reason: str
    # revise_answer
    revisions: int
    # check_usefulness
    usefulness: Optional[Literal["useful", "not_useful"]]
    usefulness_reason: str


# ======================================================================
# NODES (in execution order)
# ======================================================================

# ---------- Node: DECIDE whether retrieval is needed ----------
class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        description="True if the question needs the knowledge base or up-to-date sources."
    )


decide_retrieval_chain = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Decide whether a question needs retrieval before answering.\n"
            "The knowledge base contains {kb_description}.\n"
            "Retrieve if the question is about anything the knowledge base covers, or "
            "needs recent or specific facts you cannot answer reliably. Do not retrieve "
            "for general, timeless knowledge. If unsure, retrieve.",
        ),
        ("human", "Question: {question}"),
    ]
).partial(kb_description=KB_DESCRIPTION) | llm.with_structured_output(RetrieveDecision)


def decide_retrieval(state: State) -> State:
    decision = decide_retrieval_chain.invoke({"question": state["question"]})
    return {"need_retrieval": decision.should_retrieve}


# ---------- Node: DIRECT answer (no retrieval) ----------
direct_chain = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer from your general knowledge. Be concise. If you do not reliably "
            "know the answer, say: 'I don't know based on my general knowledge.'",
        ),
        ("human", "{question}"),
    ]
) | llm


def generate_direct(state: State) -> State:
    return {"answer": direct_chain.invoke({"question": state["question"]}).content}


# ---------- Node: RETRIEVE ----------
def retrieve(state: State) -> State:
    # the rewritten query is only a search key; graders always use the original question
    return {"docs": retriever.invoke(state.get("retrieval_query") or state["question"])}


# ---------- Node: GRADE retrieved chunks (topic-level, lenient) ----------
class RelevanceGrade(BaseModel):
    is_relevant: bool


relevance_chain = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Judge whether a chunk is relevant to a question, at the topic level.\n"
            "Relevant: it discusses the same entity or topic; it need not contain the "
            "answer.\n"
            "Not relevant: a different subject, or a different organization, person or "
            "product than the one asked about, even with a similar name.\n"
            "When unsure, it is relevant.",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
) | llm.with_structured_output(RelevanceGrade)


def grade_documents(state: State) -> State:
    """Grade all chunks in parallel; keep the relevant ones."""
    docs = state.get("docs", [])
    grades = relevance_chain.batch(
        [{"question": state["question"], "chunk": d.page_content} for d in docs],
        config={"max_concurrency": MAX_CONCURRENCY},
    )
    return {"relevant_docs": [d for d, g in zip(docs, grades) if g.is_relevant]}


# ---------- Node: GENERATE from relevant chunks ----------
def source_label(doc: Document) -> str:
    """'file p.N' for PDF chunks, 'web: title (url)' for web results."""
    if doc.metadata.get("url"):
        return f"web: {doc.metadata.get('source', '?')} ({doc.metadata['url']})"
    return f"{doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}"


def format_context(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(f"[{source_label(d)}]\n{d.page_content}" for d in docs)


generate_chain = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using ONLY the context, which comes from the knowledge "
            "base ({kb_description}) or from web search results.\n"
            "- Do not add facts, opinions or interpretations that are not in the context.\n"
            "- Only use information about the entity the question asks about; never "
            "attribute information about a similarly named one to it.\n"
            "- Cite the source label of every fact, exactly as it appears in the context.\n"
            "- If the context answers only part of the question, answer that part and "
            "say what is missing.\n"
            "- If it does not answer the question at all, reply exactly: \"{no_answer}\"",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
).partial(kb_description=KB_DESCRIPTION, no_answer=NO_ANSWER) | llm


def generate(state: State) -> State:
    context = format_context(state.get("relevant_docs", []))
    answer = generate_chain.invoke({"question": state["question"], "context": context})
    return {"answer": answer.content, "context": context}


# ---------- Node: CHECK SUPPORT (IsSUP) ----------
def is_not_found(answer: str) -> bool:
    """True for the NO_ANSWER sentence (curly apostrophes and case ignored)."""
    def normalize(text: str) -> str:
        return text.strip().replace("\u2019", "'").lower()
    return normalize(answer) == normalize(NO_ANSWER)


class SupportCheck(BaseModel):
    # evidence and explanation come before the verdict so the model reasons first
    evidence: List[str] = Field(description="Up to 3 short verbatim quotes from the context.")
    explanation: str = Field(description="One or two sentences naming any unsupported claim.")
    support: Literal["fully_supported", "partially_supported", "no_support"]


support_chain = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Check whether the ANSWER is supported by the CONTEXT. Use only the context.\n"
            "- fully_supported: every claim is stated in the context (close paraphrases "
            "are fine).\n"
            "- partially_supported: the core facts are in the context, but the answer "
            "adds interpretive wording, inferred links or generalizations that are not.\n"
            "- no_support: the key claims are not in the context.\n"
            "Source labels and statements that the sources do not cover something are "
            "not claims.",
        ),
        # no question: whether the answer addresses it is the usefulness check's job
        ("human", "Answer:\n{answer}\n\nContext:\n{context}"),
    ]
) | llm.with_structured_output(SupportCheck)


def check_support(state: State) -> State:
    if is_not_found(state["answer"]):  # nothing claimed, nothing to verify
        return {"support": "fully_supported", "evidence": [],
                "support_reason": "The answer states that nothing was found."}
    out = support_chain.invoke({"answer": state["answer"], "context": state["context"]})
    return {"support": out.support, "evidence": out.evidence,
            "support_reason": out.explanation}


# ---------- Node: REVISE an answer that is not fully supported ----------
revise_chain = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the answer so every statement is supported by the CONTEXT.\n"
            "- Remove what the reviewer flagged and any other interpretation.\n"
            "- Stay close to the context's wording and keep the source labels.\n"
            "- If the sources do not describe what was asked directly, say so, then "
            "list what they do state.\n"
            "- If nothing in the context answers the question, reply exactly: "
            "\"{no_answer}\"",
        ),
        (
            "human",
            "Question: {question}\n\nAnswer:\n{answer}\n\n"
            "Reviewer feedback:\n{feedback}\n\nContext:\n{context}",
        ),
    ]
).partial(no_answer=NO_ANSWER) | llm


def revise_answer(state: State) -> State:
    out = revise_chain.invoke(
        {"question": state["question"], "answer": state["answer"],
         "feedback": state["support_reason"], "context": state["context"]}
    )
    return {"answer": out.content, "revisions": state.get("revisions", 0) + 1}


# ---------- Node: CHECK USEFULNESS (IsUSE) ----------
class UsefulnessCheck(BaseModel):
    reason: str = Field(description="One short line.")
    usefulness: Literal["useful", "not_useful"]


usefulness_chain = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Judge whether the ANSWER gives the user what the QUESTION asks for.\n"
            "- useful: it provides the requested information, even if it notes a "
            "limitation.\n"
            "- not_useful: it says the information was not found (in any wording), or "
            "it is generic or off-topic.\n"
            "Do not judge correctness; that was checked before.",
        ),
        ("human", "Question: {question}\n\nAnswer:\n{answer}"),
    ]
) | llm.with_structured_output(UsefulnessCheck)


def check_usefulness(state: State) -> State:
    if is_not_found(state["answer"]):
        return {"usefulness": "not_useful",
                "usefulness_reason": "The answer states that nothing was found."}
    out = usefulness_chain.invoke({"question": state["question"], "answer": state["answer"]})
    return {"usefulness": out.usefulness, "usefulness_reason": out.reason}


# ---------- Node: REWRITE the question for internal retrieval ----------
def fresh_pass() -> State:
    """Reset per-pass fields: each new search gets its own revision budget."""
    return {"revisions": 0, "support": None, "evidence": [], "support_reason": "",
            "usefulness": None, "usefulness_reason": ""}


class RetrievalQuery(BaseModel):
    query: str = Field(description="4-12 search keywords, not a sentence.")


rewrite_chain = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the question into a keyword query for a vector store containing "
            "{kb_description}. Keep key names, add likely synonyms, make it different "
            "from the previous query, and never answer the question.",
        ),
        ("human", "Question: {question}\n\nPrevious query: {previous}"),
    ]
).partial(kb_description=KB_DESCRIPTION) | llm.with_structured_output(RetrievalQuery)


def rewrite_question(state: State) -> State:
    out = rewrite_chain.invoke(
        {"question": state["question"],
         "previous": state.get("retrieval_query") or state["question"]}
    )
    return {"retrieval_query": out.query, "rewrites": state.get("rewrites", 0) + 1,
            **fresh_pass()}


# ---------- Node: REWRITE the question for web search ----------
class WebQuery(BaseModel):
    query: str = Field(description="Keyword web search query, 6-14 words.")
    time_range: Optional[Literal["day", "week", "month", "year"]] = Field(
        default=None, description="Only if the question names a recent period."
    )


web_query_chain = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the question into a keyword web search query (6-14 words). "
            "Today's date is {today}: replace relative dates with explicit ones, and for "
            "'latest' questions add the current year. Set time_range only if the "
            "question names a recent period (e.g. 'this week'). Never answer the question.",
        ),
        ("human", "Question: {question}"),
    ]
) | llm.with_structured_output(WebQuery)


def rewrite_web_query(state: State) -> State:
    out = web_query_chain.invoke(
        {"question": state["question"], "today": date.today().isoformat()}
    )
    return {"web_query": out.query, "web_time_range": out.time_range, **fresh_pass()}


# ---------- Node: WEB SEARCH (at most once per question) ----------
def web_search(state: State) -> State:
    """Page text goes into page_content; title and URL into metadata."""
    args = {"query": state["web_query"]}
    if state.get("web_time_range"):
        args["time_range"] = state["web_time_range"]
    try:
        response = tavily.invoke(args)
    except Exception as e:  # missing API key, network error, ...
        print(f"Web search failed: {e}")
        response = {}

    results = response.get("results", []) if isinstance(response, dict) else []
    docs = [
        Document(page_content=r["content"],
                 metadata={"source": r.get("title") or "web", "url": r.get("url", "")})
        for r in results if r.get("content")
    ]
    return {"docs": docs, "web_searched": True}


# ---------- Node: NO ANSWER ----------
def no_answer(state: State) -> State:
    return {"answer": NO_ANSWER, "context": ""}


# ======================================================================
# ROUTING
# ======================================================================
def route_after_decide(state: State) -> Literal["retrieve", "generate_direct"]:
    return "retrieve" if state["need_retrieval"] else "generate_direct"


def next_attempt(state: State) -> Literal["rewrite_question", "rewrite_web_query", "no_answer"]:
    """Internal rewrites first, then one web search, then give up."""
    if state.get("rewrites", 0) < MAX_REWRITES:
        return "rewrite_question"
    if not state.get("web_searched"):
        return "rewrite_web_query"
    return "no_answer"


def route_after_grade(state: State) -> str:
    return "generate" if state.get("relevant_docs") else next_attempt(state)


def route_after_support(state: State) -> Literal["revise_answer", "accept"]:
    if state["support"] == "fully_supported" or state.get("revisions", 0) >= MAX_REVISIONS:
        return "accept"
    return "revise_answer"


def route_after_usefulness(state: State) -> str:
    return "done" if state["usefulness"] == "useful" else next_attempt(state)


# ======================================================================
# BUILD GRAPH
# ======================================================================
def build_graph():
    """START -> decide_retrieval
         -> generate_direct -> END
         -> retrieve -> grade_documents
              -> generate -> check_support <-> revise_answer -> check_usefulness
                   -> END                          (useful)
                   -> next attempt                 (not useful)
              -> next attempt                      (no relevant chunks)

    next attempt, in order:
         rewrite_question -> retrieve                           (up to MAX_REWRITES)
         rewrite_web_query -> web_search -> grade_documents     (once)
         no_answer -> END
    """
    g = StateGraph(State)
    for name, node in [
        ("decide_retrieval", decide_retrieval), ("generate_direct", generate_direct),
        ("retrieve", retrieve), ("grade_documents", grade_documents),
        ("generate", generate), ("check_support", check_support),
        ("revise_answer", revise_answer), ("check_usefulness", check_usefulness),
        ("rewrite_question", rewrite_question), ("rewrite_web_query", rewrite_web_query),
        ("web_search", web_search), ("no_answer", no_answer),
    ]:
        g.add_node(name, node)

    retry = {"rewrite_question": "rewrite_question",
             "rewrite_web_query": "rewrite_web_query",
             "no_answer": "no_answer"}

    g.add_edge(START, "decide_retrieval")
    g.add_conditional_edges("decide_retrieval", route_after_decide,
                            {"retrieve": "retrieve", "generate_direct": "generate_direct"})
    g.add_edge("generate_direct", END)

    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", route_after_grade,
                            {"generate": "generate", **retry})

    g.add_edge("generate", "check_support")
    g.add_conditional_edges("check_support", route_after_support,
                            {"revise_answer": "revise_answer", "accept": "check_usefulness"})
    g.add_edge("revise_answer", "check_support")
    g.add_conditional_edges("check_usefulness", route_after_usefulness,
                            {"done": END, **retry})

    g.add_edge("rewrite_question", "retrieve")
    g.add_edge("rewrite_web_query", "web_search")
    g.add_edge("web_search", "grade_documents")
    g.add_edge("no_answer", END)
    return g.compile()


def save_graph_diagram(app, path: str = GRAPH_PNG) -> None:
    """Render the graph via mermaid.ink (needs internet); print Mermaid text offline."""
    try:
        Path(path).write_bytes(app.get_graph().draw_mermaid_png())
    except Exception:
        print(app.get_graph().draw_mermaid())


# ======================================================================
# RUN
# ======================================================================
def print_result(res: State) -> None:
    print("=" * 100)
    print("QUESTION:", res["question"])
    print("NEED RETRIEVAL:", res.get("need_retrieval"))
    if res.get("rewrites"):
        print(f"REWRITES: {res['rewrites']}/{MAX_REWRITES} (last query: {res['retrieval_query']})")
    if res.get("web_searched"):
        print(f"WEB QUERY: {res.get('web_query')} (time_range={res.get('web_time_range')})")

    if res.get("docs"):
        relevant = res.get("relevant_docs", [])
        print(f"\nSOURCES ({len(relevant)}/{len(res['docs'])} relevant):")
        for d in res["docs"]:
            mark = "keep" if d in relevant else "drop"
            print(f"  [{mark}] {source_label(d)}: {d.page_content[:80].replace(chr(10), ' ')}")

    if res.get("support"):
        print(f"\nSUPPORT: {res['support']} (revisions: {res.get('revisions', 0)})")
        print(f"  {res['support_reason']}")
        for quote in res.get("evidence", []):
            print(f'  > "{quote}"')
    if res.get("usefulness"):
        print(f"\nUSEFULNESS: {res['usefulness']}\n  {res['usefulness_reason']}")

    print("\nANSWER:\n" + res.get("answer", "(none)"))


def main() -> None:
    index_folder(DOCS_DIR)
    app = build_graph()
    save_graph_diagram(app)
    for q in TEST_QUESTIONS:
        print_result(app.invoke({"question": q}, config={"recursion_limit": RECURSION_LIMIT}))


if __name__ == "__main__":
    main()