"""Evaluate self_rag.py on a corpus's eval set with LangSmith.

    python eval_self_rag.py --corpus corpora/nexaai --prefix baseline --reps 3
    python eval_self_rag.py --corpus corpora/nexaai --compare eval_results/nexaai/<run>.json

The eval set is <corpus>/eval.yaml. It is synced to the LangSmith dataset
"self-rag-<corpus name>" (cases matched by id), then every case is run through the
graph and scored. A JSON summary is saved under eval_results/<corpus name>/.

Metrics (higher is better, except llm_calls and latency_s):
    correctness  expected facts stated; for no_answer cases: declines without false claims
    faithfulness share of answer claims supported by the context it was given
    citations    share of cited labels that exist in the context (deterministic)
    path_match   direct / docs / web / no_answer as expected (deterministic)
    llm_calls    LLM calls made by the pipeline for the question
    latency_s    wall-clock seconds per question
"""

# ---------- Standard library ----------
import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ---------- Third-party ----------
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ---------- LangChain / LangSmith ----------
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client, evaluate

# ---------- Local ----------
import self_rag
from rag_common import CorpusConfig, PdfIndex

load_dotenv()

EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-5.4")   # judge; stronger than the pipeline's model
RESULTS_DIR = Path("eval_results")
METRICS = ["correctness", "faithfulness", "citations", "path_match", "llm_calls", "latency_s"]

judge = ChatOpenAI(model=EVAL_MODEL, temperature=0)


# ======================================================================
# DATASET (eval.yaml -> LangSmith dataset, matched by case id)
# ======================================================================
def load_cases(corpus_dir: Path) -> List[dict]:
    path = corpus_dir / "eval.yaml"
    if not path.is_file():
        sys.exit(f"No eval set: {path.resolve()}")
    cases = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("cases") or []
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        sys.exit(f"{path}: duplicate case ids")
    for c in cases:
        if c.get("expected_path") not in {"direct", "docs", "web", "no_answer"}:
            sys.exit(f"{path}: case {c['id']}: invalid expected_path {c.get('expected_path')!r}")
    return cases


def sync_dataset(client: Client, name: str, cases: List[dict]) -> None:
    """Create, update or delete examples so the dataset matches eval.yaml."""
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(name, description="Self-RAG eval set (from eval.yaml)")
    existing = {(ex.metadata or {}).get("case_id"): ex
                for ex in client.list_examples(dataset_id=dataset.id)}

    created = updated = 0
    for c in cases:
        inputs = {"question": c["question"]}
        outputs = {"expected_path": c["expected_path"], "facts": list(c.get("facts") or [])}
        metadata = {"case_id": c["id"], "tags": list(c.get("tags") or [])}
        ex = existing.pop(c["id"], None)
        if ex is None:
            client.create_example(inputs=inputs, outputs=outputs, metadata=metadata,
                                  dataset_id=dataset.id)
            created += 1
        elif (ex.inputs, ex.outputs, {k: (ex.metadata or {}).get(k) for k in metadata}) \
                != (inputs, outputs, metadata):   # LangSmith adds its own metadata keys
            client.update_example(ex.id, inputs=inputs, outputs=outputs, metadata=metadata)
            updated += 1
    for ex in existing.values():            # cases removed from eval.yaml
        client.delete_example(ex.id)
    print(f"Dataset '{name}': {len(cases)} cases "
          f"({created} created, {updated} updated, {len(existing)} deleted)")


# ======================================================================
# TARGET (one pipeline run per case)
# ======================================================================
class LlmCallCounter(BaseCallbackHandler):
    """Counts chat-model and LLM calls, including those inside .batch()."""

    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()

    def _count(self):
        with self._lock:
            self.calls += 1

    def on_chat_model_start(self, *args, **kwargs):
        self._count()

    def on_llm_start(self, *args, **kwargs):
        self._count()


def make_target(index: PdfIndex):
    app = self_rag.build_graph()
    base_config = self_rag.run_config(index, self_rag.kb_description_for(index))

    def target(inputs: dict) -> dict:
        counter = LlmCallCounter()
        start = time.perf_counter()
        res = app.invoke({"question": inputs["question"]},
                         config={**base_config, "callbacks": [counter]})
        path = self_rag.answer_path(res)
        grounded = path in ("docs", "web")
        return {
            "answer": res.get("answer", ""),
            "path": path,
            "context": res.get("context", "") if grounded else "",
            "sources": [self_rag.source_label(d) for d in res.get("relevant_docs", [])]
                       if grounded else [],
            "support": res.get("support"),
            "rewrites": res.get("rewrites", 0),
            "revisions": res.get("revisions", 0),
            "web_searched": bool(res.get("web_searched")),
            "llm_calls": counter.calls,
            "latency_s": round(time.perf_counter() - start, 2),
        }

    return target


# ======================================================================
# EVALUATORS
# ======================================================================
class FactCheck(BaseModel):
    fact: str
    present: bool = Field(description="The answer states this fact (paraphrase is fine).")


class CorrectnessGrade(BaseModel):
    reasoning: str = Field(description="Two or three sentences.")
    fact_checks: List[FactCheck] = Field(description="One entry per expected fact, in order.")
    declines: bool = Field(description="The answer says the information is not available.")
    wrong_claims: bool = Field(
        description="The answer contradicts an expected fact, or presents information about "
                    "a different organization, person or product as if it answered the question."
    )


correctness_chain = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade a question-answering system against a reference.\n"
            "- For each expected fact, decide whether the answer states it. Paraphrases, "
            "other formats and extra detail are fine; vague or partial mentions are not.\n"
            "- declines: the answer says it could not find or does not have the information.\n"
            "- wrong_claims: the answer contradicts an expected fact, or answers with "
            "information about a different entity than the one asked about (for example a "
            "similarly named company).",
        ),
        ("human", "Question: {question}\n\nExpected facts:\n{facts}\n\nAnswer:\n{answer}"),
    ]
) | judge.with_structured_output(CorrectnessGrade)


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    facts = reference_outputs.get("facts") or []
    grade = correctness_chain.invoke({
        "question": inputs["question"],
        "facts": "\n".join(f"- {f}" for f in facts) or "(none: the correct answer is that "
                                                         "the information is not available)",
        "answer": outputs["answer"],
    })
    if reference_outputs["expected_path"] == "no_answer":
        score = float(grade.declines and not grade.wrong_claims)
    elif grade.wrong_claims or not facts:
        score = 0.0 if grade.wrong_claims else 1.0
    else:
        checks = grade.fact_checks[:len(facts)]
        score = sum(c.present for c in checks) / len(facts)
    return {"key": "correctness", "score": round(score, 3), "comment": grade.reasoning}


class ClaimCheck(BaseModel):
    claim: str
    supported: bool


class FaithfulnessGrade(BaseModel):
    claims: List[ClaimCheck] = Field(
        description="Every factual claim in the answer. Source labels and statements that "
                    "something is not covered are not claims."
    )


faithfulness_chain = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Split the ANSWER into its factual claims and mark each as supported if the "
            "CONTEXT states it (close paraphrases count) and unsupported otherwise. Use only "
            "the context, not your own knowledge.",
        ),
        ("human", "Answer:\n{answer}\n\nContext:\n{context}"),
    ]
) | judge.with_structured_output(FaithfulnessGrade)


def faithfulness(outputs: dict) -> dict:
    if not outputs.get("context"):
        return {"key": "faithfulness", "score": None, "comment": "n/a: no retrieved context"}
    grade = faithfulness_chain.invoke({"answer": outputs["answer"],
                                       "context": outputs["context"]})
    if not grade.claims:
        return {"key": "faithfulness", "score": None, "comment": "n/a: no claims"}
    unsupported = [c.claim for c in grade.claims if not c.supported]
    score = 1 - len(unsupported) / len(grade.claims)
    return {"key": "faithfulness", "score": round(score, 3),
            "comment": "unsupported: " + "; ".join(unsupported) if unsupported else "all supported"}


CITATION = re.compile(r"\[([^\[\]\n]+)\](?!\()")   # [label], but not markdown [text](url)


def citations(outputs: dict) -> dict:
    """Share of cited labels that exist in the context; uncited grounded answers score 0."""
    if outputs["path"] not in ("docs", "web"):
        return {"key": "citations", "score": None, "comment": "n/a: not a grounded answer"}
    known = set(outputs.get("sources") or [])
    cited = [c.strip() for c in CITATION.findall(outputs["answer"])]
    if not cited:
        return {"key": "citations", "score": 0.0, "comment": "no citations"}
    invalid = [c for c in cited if c not in known]
    return {"key": "citations", "score": round(1 - len(invalid) / len(cited), 3),
            "comment": "invalid: " + "; ".join(invalid) if invalid else "all valid"}


def path_match(outputs: dict, reference_outputs: dict) -> dict:
    expected = reference_outputs["expected_path"]
    return {"key": "path_match", "score": float(outputs["path"] == expected),
            "comment": f"expected {expected}, got {outputs['path']}"}


def llm_calls(outputs: dict) -> dict:
    return {"key": "llm_calls", "score": outputs["llm_calls"]}


def latency_s(outputs: dict) -> dict:
    return {"key": "latency_s", "score": outputs["latency_s"]}


EVALUATORS = [correctness, faithfulness, citations, path_match, llm_calls, latency_s]


# ======================================================================
# SUMMARY
# ======================================================================
def mean(values: List[Optional[float]]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def summarize(results, experiment: str, corpus: str, reps: int) -> dict:
    per_case: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    paths: Dict[str, list] = defaultdict(list)
    questions: Dict[str, str] = {}
    for row in results:
        case_id = row["example"].metadata["case_id"]
        questions[case_id] = row["example"].inputs["question"]
        paths[case_id].append(row["run"].outputs["path"])
        for r in row["evaluation_results"]["results"]:
            per_case[case_id][r.key].append(r.score)

    cases = {
        cid: {"question": questions[cid], "paths": paths[cid],
              **{m: mean(per_case[cid][m]) for m in METRICS}}
        for cid in sorted(per_case)
    }
    overall = {m: mean([c[m] for c in cases.values()]) for m in METRICS}
    return {"experiment": experiment, "corpus": corpus, "created": datetime.now().isoformat(
        timespec="seconds"), "reps": reps, "llm_model": self_rag.LLM_MODEL,
        "eval_model": EVAL_MODEL, "overall": overall, "cases": cases}


def fmt(value) -> str:
    return "  -  " if value is None else f"{value:5.2f}" if value < 10 else f"{value:5.1f}"


def print_summary(summary: dict, baseline: Optional[dict] = None) -> None:
    head = f"{'case':28}" + "".join(f"{m[:11]:>12}" for m in METRICS)
    print("\n" + head + "\n" + "-" * len(head))
    for cid, c in summary["cases"].items():
        line = f"{cid[:28]:28}" + "".join(f"{fmt(c[m]):>12}" for m in METRICS)
        if baseline and cid in baseline["cases"]:
            old, new = baseline["cases"][cid]["correctness"], c["correctness"]
            if old is not None and new is not None and abs(new - old) >= 0.01:
                line += f"   correctness {old:.2f} -> {new:.2f}"
        print(line)
    print("-" * len(head))
    print(f"{'OVERALL':28}" + "".join(f"{fmt(summary['overall'][m]):>12}" for m in METRICS))
    if baseline:
        print(f"{'BASELINE':28}" + "".join(f"{fmt(baseline['overall'][m]):>12}" for m in METRICS)
              + f"   ({baseline['experiment']})")


# ======================================================================
# CLI
# ======================================================================
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate self_rag.py with LangSmith.")
    parser.add_argument("--corpus", required=True, help="corpus folder containing eval.yaml")
    parser.add_argument("--prefix", default="self-rag", help="experiment name prefix")
    parser.add_argument("--reps", type=int, default=1, help="runs per case (answers vary)")
    parser.add_argument("--concurrency", type=int, default=2, help="cases run in parallel")
    parser.add_argument("--only", help="comma-separated case ids to run")
    parser.add_argument("--compare", type=Path, help="earlier summary JSON to compare with")
    args = parser.parse_args(argv)

    if not os.getenv("LANGSMITH_API_KEY"):
        sys.exit("LANGSMITH_API_KEY is not set (.env).")
    baseline = json.loads(args.compare.read_text(encoding="utf-8")) if args.compare else None

    corpus_dir = Path(args.corpus)
    cfg = CorpusConfig.load(corpus_dir)
    index = PdfIndex(cfg, self_rag.embeddings)
    if index.count() == 0:
        sys.exit(f"Corpus '{cfg.name}' is not indexed: run self_rag.py index first.")

    cases = load_cases(corpus_dir if corpus_dir.is_dir() else corpus_dir.parent)
    client = Client()
    dataset = f"self-rag-{cfg.name}"
    sync_dataset(client, dataset, cases)

    data = dataset
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        data = [ex for ex in client.list_examples(dataset_name=dataset)
                if (ex.metadata or {}).get("case_id") in wanted]
        if not data:
            sys.exit(f"No cases match --only {args.only}")

    results = evaluate(
        make_target(index),
        data=data,
        evaluators=EVALUATORS,
        experiment_prefix=args.prefix,
        num_repetitions=args.reps,
        max_concurrency=args.concurrency,
        metadata={"corpus": cfg.name, "llm_model": self_rag.LLM_MODEL,
                  "eval_model": EVAL_MODEL, "reps": args.reps},
        client=client,
    )

    summary = summarize(results, results.experiment_name, cfg.name, args.reps)
    out_dir = RESULTS_DIR / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{results.experiment_name}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print_summary(summary, baseline)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
