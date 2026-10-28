"""eval_self_rag.py --local: same evaluators and summary, nothing sent to LangSmith
(decisions S9.1-S9.5)."""
import os
import time
from types import SimpleNamespace

from _support import Checks, FakeServer

srv = FakeServer()
# Tracing switched on and pointed at the fake server: local mode must still send nothing.
os.environ.update(LANGSMITH_ENDPOINT=srv.url, LANGSMITH_API_KEY="test-key",
                  LANGSMITH_TRACING="true")

import eval_self_rag as E  # noqa: E402
import self_rag as sr  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

check = Checks("local eval mode")
fake_chain = ChatPromptTemplate.from_messages([("human", "{q}")]) \
    | FakeListChatModel(responses=["x"] * 1000)


def correctness_grade(x):
    fake_chain.invoke({"q": "judge"})   # a traced LLM call inside an evaluator
    facts = [line[2:] for line in x["facts"].splitlines() if line.startswith("- ")]
    return E.CorrectnessGrade(reasoning="r",
                              fact_checks=[E.FactCheck(fact=f, present=True) for f in facts],
                              declines="(none" in x["facts"], wrong_claims=False)


E.correctness_chain = RunnableLambda(correctness_grade)
E.faithfulness_chain = RunnableLambda(
    lambda x: E.FaithfulnessGrade(claims=[E.ClaimCheck(claim="c", supported=True)]))

# Invented cases (no corpus data); one crashes in the target.
CASES = [
    {"id": "fact", "question": "How long is the invented warranty?", "expected_path": "docs",
     "facts": ["The warranty lasts two years"]},
    {"id": "gap", "question": "Who is the invented CFO?", "expected_path": ["no_answer", "docs"]},
    {"id": "crash", "question": "CRASH", "expected_path": "docs", "facts": ["x"]},
]


def target(inputs):
    if inputs["question"] == "CRASH":
        raise RuntimeError("boom")
    fake_chain.invoke({"q": inputs["question"]})   # a traced LLM call inside the target
    found = "warranty" in inputs["question"]
    return {"answer": "Two years [a.pdf p.1]" if found else sr.NO_ANSWER,
            "path": "docs" if found else "no_answer", "scope": "knowledge_base",
            "context": "[a.pdf p.1]\nTwo years." if found else "",
            "sources": ["a.pdf p.1"] if found else [], "kept": [], "dropped": [],
            "llm_calls": 3, "tokens": {}, "cost_usd": 0.001, "unpriced_models": [],
            "latency_s": 0.1}


args = SimpleNamespace(prefix="test", reps=2, concurrency=2)
experiment, rows = E.run_local(target, CASES, args)
summary = E.summarize(rows, experiment, "invented", 2, "local")
c = summary["cases"]
check("experiment name has the LangSmith form <prefix>-<8 hex>",
      experiment.startswith("test-") and len(experiment) == len("test-") + 8, experiment)
check("every case x rep ran", len(rows) == 6 and all(len(c[i]["runs"]) == 2 for i in c))
check("scores: fact case", c["fact"]["correctness"] == 1.0 and c["fact"]["citations"] == 1.0
      and c["fact"]["path_match"] == 1.0)
check("scores: no_answer case accepted", c["gap"]["correctness"] == 1.0
      and c["gap"]["path_match"] == 1.0 and c["gap"]["faithfulness"] is None)
check("crashed run counts as failed, error kept", c["crash"]["correctness"] == 0.0
      and c["crash"]["path_match"] == 0.0 and "boom" in c["crash"]["runs"][0]["error"])
check("summary records the mode", summary["mode"] == "local")
# LangSmith uploads in a background thread: wait for anything pending before checking.
from langsmith.run_trees import get_cached_client  # noqa: E402
get_cached_client().flush()
time.sleep(1)
check("nothing sent to LangSmith", not any(p.startswith("/runs") or p.startswith("/datasets")
                                           for p in srv.requests), str(srv.requests))
check.finish()
