"""Self-RAG over company PDFs, with web search fallback.

The graph flow is documented in build_graph().
"""

# ---------- Standard library ----------
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import List, Literal, Optional, TypedDict

# ---------- Third-party ----------
import pymupdf4llm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ---------- LangChain / LangGraph ----------
from langchain_tavily import TavilySearch
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
MAX_REVISIONS = 2       # support-driven rewrites per answer
MAX_REWRITES = 2        # retrieval retries with a rewritten query
WEB_MAX_RESULTS = 5     # Tavily results for the web fallback
INDEX_BATCH = 500       # max IDs per Chroma add/delete call

RECURSION_LIMIT = 50    # max graph steps per question
NO_ANSWER = "I couldn't find this in the available sources."
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
tavily = TavilySearch(max_results=WEB_MAX_RESULTS, topic="general")


# GRAPH STATE
class State(TypedDict, total=False):
    """Data flowing through the graph; each node returns only what it changes."""
    question: str
    # decide_retrieval
    need_retrieval: bool
    # rewrite_question
    retrieval_query: str
    rewrites: int
    # rewrite_web_query
    web_query: str
    web_time_range: Optional[Literal["day", "week", "month", "year"]]
    # web_search
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
    return {"docs": retriever.invoke(state.get("retrieval_query") or state["question"])}

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

# ---------- Node: GENERATE from relevant chunks ----------
def source_label(doc: Document) -> str:
    """[file p.N] for PDF chunks, [web: title (url)] for web results."""
    if doc.metadata.get("url"):
        return f"web: {doc.metadata.get('source', '?')} ({doc.metadata['url']})"
    return f"{doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}"


def format_context(docs: List[Document]) -> str:
    """Label each chunk with its source so the answer can cite it."""
    return "\n\n---\n\n".join(f"[{source_label(d)}]\n{d.page_content}" for d in docs)


rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer the question using ONLY the context below. It comes either from "
            "internal documents of the company NexaAI Solutions or from web search "
            "results.\n\n"
            "Rules:\n"
            "- Do not use outside knowledge and do not add facts, opinions or "
            "interpretations that are not stated in the context.\n"
            "- Cite the source label of every fact you use, e.g. "
            "[Company_Policies.pdf p.1] or [web: Title (https://...)].\n"
            "- If the context only partly answers the question, answer that part and "
            "say what is missing.\n"
            "- If the context does not answer the question at all, reply exactly: "
            "\"{no_answer}\"\n"
            "- Do not mention 'the context' in your answer.\n"
            "- The symbol ■ stands for the Indian rupee sign (₹).",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)


def generate(state: State) -> State:
    context = format_context(state.get("relevant_docs", []))
    out = (rag_prompt | llm).invoke(
        {"question": state["question"], "context": context, "no_answer": NO_ANSWER}
    )
    return {"answer": out.content, "context": context}



# ---------- Node: NO ANSWER ----------
def no_answer(state: State) -> State:
    return {"answer": NO_ANSWER, "context": ""}


# ---------- Node: CHECK SUPPORT (IsSUP) ----------
class SupportCheck(BaseModel):
    # evidence and explanation come before the verdict so the model reasons first
    evidence: List[str] = Field(
        description="Up to 3 short verbatim quotes from the context that support the answer."
    )
    explanation: str = Field(
        description="One or two sentences naming any claim or wording not stated in the "
                    "context. Never flag source labels or sentences saying the documents "
                    "do not cover something."
    )
    
    support: Literal["fully_supported", "partially_supported", "no_support"]


support_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You verify whether an ANSWER is supported by the CONTEXT it was generated "
            "from. Use only the context, never outside knowledge.\n\n"
            "- fully_supported: every factual claim is stated in the context. Close "
            "paraphrases are fine. Sentences saying the documents do not cover something "
            "(e.g. 'the documents do not directly describe the culture') are honest "
            "limitations, not claims, and count as supported. Source labels like "
            "[file p.N] are not claims either.\n"
            "- partially_supported: the core facts are in the context, but the answer "
            "adds anything that is not: interpretive or qualitative words (e.g. "
            "'generous', 'emphasizes', 'strong', 'employee-first'), inferred causes or "
            "links (e.g. 'supports X through Y'), generalizations, or claiming the "
            "documents describe something they do not (e.g. presenting a policy as "
            "the company culture). Noting that the documents do not describe something "
            "directly and then listing what they do state is fine.\n"
            "- no_support: the key claims are not in the context.\n\n"
            "Be strict about claims: a single unsupported word or link makes it "
            "partially_supported.\n"
            "An answer that only says the information could not be found is "
            "fully_supported.",
        ),
        ("human", "Question: {question}\n\nAnswer:\n{answer}\n\nContext:\n{context}"),
    ]
)
support_chain = support_prompt | llm.with_structured_output(SupportCheck)

def normalize(text: str) -> str:
    """Lower-case and straighten curly apostrophes for robust comparisons."""
    return text.strip().replace("\u2019", "'").lower()


def is_not_found(answer: str) -> bool:
    return normalize(answer) == normalize(NO_ANSWER)

def check_support(state: State) -> State:
    if is_not_found(state["answer"]):  # nothing claimed, nothing to verify
        return {"support": "fully_supported", "evidence": [],
                "support_reason": "The answer states that the information was not found."}

    out = support_chain.invoke(
        {"question": state["question"], "answer": state["answer"],
         "context": state["context"]}
    )
    return {"support": out.support, "evidence": out.evidence,
            "support_reason": out.explanation}


# ---------- Node: REVISE an answer that is not fully supported ----------
revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite an answer so that it is fully supported by the CONTEXT.\n\n"
            "Rules:\n"
            "- Keep only facts stated in the context; remove everything the reviewer "
            "flagged and any other interpretation, judgement or inferred link.\n"
            "- Stay close to the wording of the context; quote it where helpful.\n"
            "- If the question asks for something the documents do not describe "
            "directly (e.g. 'culture'), say so, then list what they do state.\n"
            "- Put the source label after each fact, e.g. [Company_Policies.pdf p.1].\n"
            "- The symbol ■ stands for the Indian rupee sign (₹).\n"
            "- If nothing in the context answers the question, reply exactly: "
            "\"{no_answer}\"",
        ),
        (
            "human",
            "Question: {question}\n\nCurrent answer:\n{answer}\n\n"
            "Reviewer feedback:\n{feedback}\n\nContext:\n{context}",
        ),
    ]
)


def revise_answer(state: State) -> State:
    out = (revise_prompt | llm).invoke(
        {"question": state["question"], "answer": state["answer"],
         "feedback": state["support_reason"], "context": state["context"],
         "no_answer": NO_ANSWER}
    )
    return {"answer": out.content, "revisions": state.get("revisions", 0) + 1}


# ---------- Node: CHECK USEFULNESS (IsUSE) ----------
class UsefulnessCheck(BaseModel):
    # reason comes before the verdict so the model reasons first
    reason: str = Field(description="One short line explaining the verdict.")
    usefulness: Literal["useful", "not_useful"]


usefulness_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You judge whether an ANSWER is useful for the QUESTION, i.e. whether it "
            "actually gives the user what they asked for.\n\n"
            "- useful: the answer provides the requested information. An answer that "
            "notes a limitation (e.g. 'the documents do not describe this directly') "
            "but still gives relevant, specific information is useful.\n"
            "- not_useful: the answer's main content is that the information could not "
            "be found or is not available, in any wording (e.g. 'No refund policy was "
            "found'). Mentioning the topic does not make it useful. Also not_useful: "
            "generic, off-topic, or only background without answering.\n\n"
            "Do NOT check whether the answer is correct or grounded; that was verified "
            "before. Only judge: does it give the user the requested information?",
        ),
        ("human", "Question: {question}\n\nAnswer:\n{answer}"),
    ]
)
usefulness_chain = usefulness_prompt | llm.with_structured_output(UsefulnessCheck)


def check_usefulness(state: State) -> State:
    if is_not_found(state["answer"]):  # "not found" never answers the question
        return {"usefulness": "not_useful",
                "usefulness_reason": "The answer states that the information was not found."}

    out = usefulness_chain.invoke(
        {"question": state["question"], "answer": state["answer"]}
    )
    return {"usefulness": out.usefulness, "usefulness_reason": out.reason}


# ---------- Node: REWRITE the question into a better retrieval query ----------
class RetrievalQuery(BaseModel):
    query: str = Field(description="Keyword search query, 4-12 words, not a sentence.")


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite a question into a search query for a vector store of internal "
            "NexaAI Solutions documents: HR and leave policies, workplace conduct, "
            "company profile and leadership, products, pricing plans, billing and "
            "discounts.\n\n"
            "Rules:\n"
            "- Output 4-12 keywords, not a full sentence.\n"
            "- Keep key entities (NexaAI, plan or product names).\n"
            "- Add terms and synonyms likely to appear in such documents "
            "(e.g. refund -> refund cancellation money-back billing invoice).\n"
            "- The query must differ from the previous query: use different "
            "vocabulary or a different angle.\n"
            "- Never answer the question.",
        ),
        (
            "human",
            "Question: {question}\n\nPrevious query: {previous}\n\n"
            "Why the previous attempt failed: {failure}",
        ),
    ]
)
rewrite_chain = rewrite_prompt | llm.with_structured_output(RetrievalQuery)

def fresh_pass() -> State:
    """Reset per-pass fields before retrieving or searching again."""
    return {"revisions": 0,
            "support": None, "evidence": [], "support_reason": "",
            "usefulness": None, "usefulness_reason": ""}

def rewrite_question(state: State) -> State:
    if state.get("usefulness") == "not_useful":
        failure = state["usefulness_reason"]
    else:
        failure = "None of the retrieved chunks was relevant to the question."

    out = rewrite_chain.invoke(
        {"question": state["question"],
         "previous": state.get("retrieval_query") or state["question"],
         "failure": failure}
    )
    # new retrieval pass: reset the revision budget and clear the old verdicts
    return {"retrieval_query": out.query, "rewrites": state.get("rewrites", 0) + 1,
            **fresh_pass()}


# ---------- Node: REWRITE the question into a web search query ----------
class WebQuery(BaseModel):
    query: str = Field(description="Keyword web search query, 6-14 words.")
    time_range: Optional[Literal["day", "week", "month", "year"]] = Field(
        default=None,
        description="Recency filter only if the question names a recent period; otherwise null.",
    )


web_query_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6-14 words).\n"
            "- Today's date is {today}. Replace relative dates (e.g. 'last week', "
            "'this year') with explicit dates, months or years.\n"
            "- Set time_range only if the question asks about a specific recent period "
            "(e.g. 'today', 'this week', 'last month'). Words like 'latest' or "
            "'current' alone do not justify a filter, because the latest event may be "
            "months old. Otherwise leave it null.\n"
            "- Do NOT answer the question.",
        ),
        ("human", "Question: {question}"),
    ]
)
web_query_chain = web_query_prompt | llm.with_structured_output(WebQuery)


def rewrite_web_query(state: State) -> State:
    out = web_query_chain.invoke(
        {"question": state["question"], "today": date.today().isoformat()}
    )
    return {"web_query": out.query, "web_time_range": out.time_range, **fresh_pass()}


# ---------- Node: WEB SEARCH (runs at most once per question) ----------
def web_search(state: State) -> State:
    """Search the web; keep only page text in page_content (title/URL in metadata)."""
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
        Document(
            page_content=r["content"],
            metadata={"source": r.get("title") or "web", "url": r.get("url", "")},
        )
        for r in results
        if r.get("content")
    ]
    return {"docs": docs, "web_searched": True}


# ---------- Routing ----------
def route_after_decide(state: State) -> Literal["retrieve", "generate_direct"]:
    return "retrieve" if state["need_retrieval"] else "generate_direct"


def next_attempt(state: State) -> Literal["rewrite_question", "rewrite_web_query", "no_answer"]:
    """Internal rewrites first, then one web search, then give up."""
    if state.get("rewrites", 0) < MAX_REWRITES:
        return "rewrite_question"
    if not state.get("web_searched"):
        return "rewrite_web_query"
    return "no_answer"


def route_after_grade(
    state: State,
) -> Literal["generate", "rewrite_question", "rewrite_web_query", "no_answer"]:
    return "generate" if state.get("relevant_docs") else next_attempt(state)


def route_after_support(state: State) -> Literal["revise_answer", "accept"]:
    if state["support"] == "fully_supported":
        return "accept"
    if state.get("revisions", 0) >= MAX_REVISIONS:
        return "accept"   # give up revising; usefulness is judged next
    return "revise_answer"


def route_after_usefulness(
    state: State,
) -> Literal["done", "rewrite_question", "rewrite_web_query", "no_answer"]:
    return "done" if state["usefulness"] == "useful" else next_attempt(state)


# BUILD GRAPH
def build_graph():
    """START -> decide_retrieval
         -> generate_direct -> END
         -> retrieve -> grade_documents
              -> generate -> check_support <-> revise_answer -> check_usefulness
                   -> END                                         (useful)
                   -> next attempt                                (not useful)
              -> next attempt                                     (no relevant chunks)

    next attempt, in order:
         rewrite_question -> retrieve                 (up to MAX_REWRITES times)
         rewrite_web_query -> web_search -> grade_documents   (once)
         no_answer -> END
    """
    g = StateGraph(State)
    g.add_node("decide_retrieval", decide_retrieval)
    g.add_node("generate_direct", generate_direct)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("generate", generate)
    g.add_node("check_support", check_support)
    g.add_node("revise_answer", revise_answer)
    g.add_node("check_usefulness", check_usefulness)
    g.add_node("rewrite_question", rewrite_question)
    g.add_node("rewrite_web_query", rewrite_web_query)
    g.add_node("web_search", web_search)
    g.add_node("no_answer", no_answer)

    next_attempt_targets = {
        "rewrite_question": "rewrite_question",
        "rewrite_web_query": "rewrite_web_query",
        "no_answer": "no_answer",
    }

    g.add_edge(START, "decide_retrieval")
    g.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {"retrieve": "retrieve", "generate_direct": "generate_direct"},
    )
    g.add_edge("generate_direct", END)

    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {"generate": "generate", **next_attempt_targets},
    )

    g.add_edge("generate", "check_support")
    g.add_conditional_edges(
        "check_support",
        route_after_support,
        {"revise_answer": "revise_answer", "accept": "check_usefulness"},
    )
    g.add_edge("revise_answer", "check_support")

    g.add_conditional_edges(
        "check_usefulness",
        route_after_usefulness,
        {"done": END, **next_attempt_targets},
    )

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



#RUN

def print_result(res: State) -> None:
    print("=" * 100)
    print("QUESTION:", res["question"])
    print("NEED RETRIEVAL:", res.get("need_retrieval"))

    if res.get("rewrites"):
        print(f"REWRITES: {res['rewrites']}/{MAX_REWRITES} "
              f"(last query: {res['retrieval_query']})")

    if res.get("web_searched"):
        print(f"WEB QUERY: {res.get('web_query')} "
              f"(time_range={res.get('web_time_range')})")

    if res.get("docs"):
        relevant = res.get("relevant_docs", [])
        print(f"\nSOURCES ({len(relevant)}/{len(res['docs'])} relevant):")
        for d in res["docs"]:
            mark = "keep" if d in relevant else "drop"
            preview = d.page_content[:80].replace("\n", " ")
            print(f"  [{mark}] {source_label(d)}: {preview}")

    if res.get("support"):
        print(f"\nSUPPORT: {res['support']} (revisions: {res.get('revisions', 0)})")
        print(f"  {res['support_reason']}")
        for quote in res.get("evidence", []):
            print(f'  > "{quote}"')
    
    if res.get("usefulness"):
        print(f"\nUSEFULNESS: {res['usefulness']}")
        print(f"  {res['usefulness_reason']}")

    print("\nANSWER:\n" + res.get("answer", "(none)"))


def main() -> None:
    index_folder(DOCS_DIR)
    app = build_graph()
    save_graph_diagram(app)
    for q in TEST_QUESTIONS:
        print_result(app.invoke({"question": q}, config={"recursion_limit": RECURSION_LIMIT}))


if __name__ == "__main__":
    main()