"""Stub chains for graph tests: each makes one call to a fake chat model (so LLM callbacks
fire like in the real pipeline) and returns a scripted result."""
import self_rag as sr
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

_fake = FakeListChatModel(responses=["x"] * 100000)

# Scripted behaviour; tests change these between runs (reset() restores the defaults).
DEFAULTS = {"scope": "knowledge_base", "keep": True, "support": "fully_supported",
            "useful": "useful", "cite": None, "passages": [1]}
S = dict(DEFAULTS)
INTERNAL_DOCS = [Document(page_content=f"chunk {i}", metadata={"source": "a.pdf", "page": i})
                 for i in range(4)]
GRADED = []   # source labels of every graded text, in order


def reset(**overrides):
    S.clear()
    S.update(DEFAULTS, **overrides)
    GRADED.clear()


def _chain(fn):
    def run(x, config=None):
        _fake.invoke("x", config)
        return fn(x)
    return RunnableLambda(run)


def _grade(x):
    GRADED.append(x["source"])
    keep = S["keep"] or x["source"].startswith("web:")
    return sr.RelevanceGrade(question_entity="none", same_entity=keep, on_topic=keep)


def _grade_passages(x):
    g = _grade(x)
    return sr.RelevancePassages(**g.model_dump(), passages=list(S["passages"]))


def _answer(x):
    label = S["cite"] or x["context"].split("]")[0][1:]
    return AIMessage(content=f"Answer [{label}]")


def install():
    sr.decide_retrieval_chain = _chain(lambda x: sr.ScopeDecision(reason="r", scope=S["scope"]))
    sr.direct_chain = _chain(lambda x: AIMessage(content="direct answer"))
    sr.relevance_chain = _chain(_grade)
    sr.relevance_passages_chain = _chain(_grade_passages)
    sr.generate_chain = _chain(_answer)
    sr.revise_chain = _chain(_answer)
    sr.support_chain = _chain(lambda x: sr.SupportCheck(evidence=[], explanation="e",
                                                        support=S["support"]))
    sr.usefulness_chain = _chain(lambda x: sr.UsefulnessCheck(reason="r",
                                                              usefulness=S["useful"]))
    sr.rewrite_chain = _chain(lambda x: sr.RetrievalQuery(query="q2"))
    sr.web_query_chain = _chain(lambda x: sr.WebQuery(query="q"))


CONFIG = {"recursion_limit": sr.RECURSION_LIMIT,
          "configurable": {"kb_description": "an invented test corpus",
                           "retriever": RunnableLambda(lambda q: list(INTERNAL_DOCS))}}
