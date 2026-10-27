from typing import List, TypedDict
import time

import hashlib
from langchain_chroma import Chroma
from pathlib import Path
import pymupdf4llm
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field
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
    key = f"{doc.metadata['source']}|{doc.metadata['page']}|{doc.page_content}"
    return hashlib.sha256(key.encode()).hexdigest()

def index_folder(folder: Path, batch_size: int = 500) -> None:
    for pdf in sorted(folder.glob("*.pdf")):
        chunks = splitter.split_documents(load_pdf(pdf))

        # de-duplicate identical chunks (same source, page, and text)
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

index_folder(DOCS_DIR)



retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20},
)


# 4) LLM + prompt
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

class CitedAnswer(BaseModel):
    answer: str = Field(description="Answer based only on the context.")
    citations: list[int] = Field(description="Numbers of the context passages used.")

structured_llm = llm.with_structured_output(CitedAnswer)

class State(TypedDict):
    question: str
    docs: List[Document]
    answer: str
    citations: List[int]


def retrieve(state):
    q = state["question"]
    return {"docs": retriever.invoke(q)}


prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer only from the numbered context passages. "
     "If the answer is not in the context, say you don't know. "
     "Cite the passage numbers you used."),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])

def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] ({d.metadata.get('source', '?')}, p.{d.metadata.get('page', '?')})\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )

def generate(state):
    out: CitedAnswer = (prompt | structured_llm).invoke(
        {"question": state["question"], "context": format_context(state["docs"])}
    )
    return {"answer": out.answer, "citations": out.citations}


g = StateGraph(State)
g.add_node("retrieve", retrieve)
g.add_node("generate", generate)
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "generate")
g.add_edge("generate", END)
app = g.compile()

# 5) Run
res = app.invoke({"question": "WHat is a transformer in deep learning.", "docs": [], "answer": ""})
print(res["answer"], "\n")
for n in res["citations"]:
    d = res["docs"][n - 1]
    print(f"[{n}] {d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}")
    print(d.page_content[:500])
    print("*" * 100)