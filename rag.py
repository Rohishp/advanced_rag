from typing import List, TypedDict
import time

from pathlib import Path
import pymupdf4llm
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

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

docs = [d for pdf in sorted(DOCS_DIR.glob("*.pdf")) for d in load_pdf(pdf)]

# 2) Chunk
chunks =  RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=64
    ).split_documents(docs)


# 3) Index (fresh collection each run)
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
vector_store = FAISS.from_documents(chunks, embeddings)


retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})

# 4) LLM + prompt
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

class State(TypedDict):
    question: str
    docs: List[Document]
    answer: str


def retrieve(state):
    q = state["question"]
    return {"docs": retriever.invoke(q)}


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer only from the context. If not in context, say you don't know."),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)

def generate(state):
    context = "\n\n".join(d.page_content for d in state["docs"])
    out = (prompt | llm).invoke({"question": state["question"], "context": context})
    return {"answer": out.content}


g = StateGraph(State)
g.add_node("retrieve", retrieve)
g.add_node("generate", generate)
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "generate")
g.add_edge("generate", END)
app = g.compile()

# 5) Run
res = app.invoke({"question": "WHat is a transformer in deep learning.", "docs": [], "answer": ""})
print(res["answer"])

print(res['docs'][0].page_content)
print('*'*100)
print(res['docs'][1].page_content)
print('*'*100)
print(res['docs'][2].page_content)
print('*'*100)
print(res['docs'][3].page_content)