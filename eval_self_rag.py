"""Evaluate self_rag.py on a corpus's eval set, with LangSmith or fully local.

    python eval_self_rag.py --corpus corpora/nexaai --prefix baseline --reps 3
    python eval_self_rag.py --corpus corpora/nexaai --compare eval_results/nexaai/<run>.json
    python eval_self_rag.py --corpus corpora/nexaai --local --reps 3 --compare <run>.json

The eval set is <corpus>/eval.yaml. It is synced to the LangSmith dataset
"self-rag-<corpus name>" (cases matched by id), then every case is run through the
graph and scored. A JSON summary is saved under eval_results/<corpus name>/.

--local runs the same cases, evaluators and summary without LangSmith: no dataset sync,
no experiment, tracing off, no LANGSMITH_API_KEY needed.

Metrics (higher is better, except llm_calls and latency_s):
    correctness  expected facts stated; for no_answer cases: declines without false claims
    faithfulness share of answer claims supported by the context it was given
    citations    share of cited labels that exist in the context (deterministic)
    path_match   direct / docs / web / no_answer as expected (deterministic)
    llm_calls    LLM calls made by the pipeline for the question
    cost_per_1k  pipeline cost in USD per 1000 such questions (prices: model_prices.yaml)
    latency_s    wall-clock seconds per question
"""

# ---------- Standard library ----------
import argparse
import inspect
import json
import logging
import os
import statistics
import sys
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar, copy_context
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
from langchain_core.tracers.context import register_configure_hook
from langsmith import Client, evaluate, tracing_context

# ---------- Local ----------
import self_rag
from rag_common import CorpusConfig, PdfIndex, chat_model, setup_logging

load_dotenv()

# Status and progress go to the log; the results table is printed.
log = logging.getLogger("rag.eval")

# Judge: stronger than the pipeline's model. Defaults chosen in step 6 (decision S6.11).
EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-6.1-sol")
EVAL_REASONING_EFFORT = os.getenv("EVAL_REASONING_EFFORT", "low") or None
RESULTS_DIR = Path("eval_results")
METRICS = ["correctness", "faithfulness", "citations", "path_match", "llm_calls",
           "cost_per_1k", "latency_s"]
PRICES_FILE = Path(os.getenv("MODEL_PRICES", "model_prices.yaml"))

judge = chat_model(EVAL_MODEL, EVAL_REASONING_EFFORT)


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
        if not c.get("expected_path") or not set(expected_paths(c["expected_path"])) <= PATHS:
            sys.exit(f"{path}: case {c['id']}: invalid expected_path {c.get('expected_path')!r}")
    return cases


PATHS = {"direct", "docs", "web", "no_answer"}


def expected_paths(value) -> List[str]:
    """expected_path is one path or a list of accepted paths."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


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
    log.info("Dataset '%s': %d cases (%d created, %d updated, %d deleted)",
             name, len(cases), created, updated, len(existing))


# ======================================================================
# TARGET (one pipeline run per case)
# ======================================================================
class LlmCallCounter(BaseCallbackHandler):
    """Counts LLM calls and their token usage per model (thread-safe)."""

    def __init__(self):
        self.calls = 0
        self.tokens: Dict[str, Dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})
        self._lock = threading.Lock()

    def _count(self):
        with self._lock:
            self.calls += 1

    def on_chat_model_start(self, *args, **kwargs):
        self._count()

    def on_llm_start(self, *args, **kwargs):
        self._count()

    def on_llm_end(self, response, **kwargs):
        for generations in response.generations:
            for gen in generations:
                message = getattr(gen, "message", None)
                usage = getattr(message, "usage_metadata", None) or {}
                if not usage:
                    continue
                model = ((getattr(message, "response_metadata", None) or {}).get("model_name")
                         or (response.llm_output or {}).get("model_name") or "unknown")
                with self._lock:
                    self.tokens[model]["input"] += usage.get("input_tokens", 0)
                    self.tokens[model]["output"] += usage.get("output_tokens", 0)


def load_prices(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.is_file():
        log.warning("No price table at %s: cost_per_1k will be empty.", path)
        return {}
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("models") or {}


def price_for(model: str, prices: Dict[str, Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Longest prefix match, so dated API names (e.g. '<model>-2026-09-22') still match."""
    matches = [name for name in prices if model.startswith(name)]
    return prices[max(matches, key=len)] if matches else None


def cost_usd(tokens: Dict[str, Dict[str, int]], prices) -> tuple:
    """(total USD or None if any model is unpriced, list of unpriced models)."""
    total, unpriced = 0.0, []
    for model, t in tokens.items():
        price = price_for(model, prices)
        if price is None:
            unpriced.append(model)
            continue
        total += (t["input"] * price["input"] + t["output"] * price["output"]) / 1_000_000
    return (None if unpriced else total), unpriced


# Registered like LangChain's own usage tracker: every callback manager created while the
# variable is set picks the counter up, so it never has to be passed in the invoke config
# (an explicit callbacks list there can interfere with the LangSmith tracer).
_call_counter: ContextVar[Optional[LlmCallCounter]] = ContextVar("llm_call_counter", default=None)
register_configure_hook(_call_counter, inheritable=True)


def make_target(index: PdfIndex):
    prices = load_prices(PRICES_FILE)
    app = self_rag.build_graph()
    base_config = self_rag.run_config(index, self_rag.kb_description_for(index))

    def target(inputs: dict) -> dict:
        counter = LlmCallCounter()
        token = _call_counter.set(counter)
        start = time.perf_counter()
        try:
            res = app.invoke({"question": inputs["question"]}, config=base_config)
        finally:
            _call_counter.reset(token)
        path = self_rag.answer_path(res)
        cost, unpriced = cost_usd(counter.tokens, prices)
        if res.get("llm_calls", 0) != counter.calls:   # a node that misreports its calls
            log.warning("%r: pipeline counted %s LLM calls, callback counted %d",
                        inputs["question"], res.get("llm_calls", 0), counter.calls)
        grounded = path in ("docs", "web")
        return {
            "answer": res.get("answer", ""),
            "path": path,
            "scope": res.get("scope"),
            "context": res.get("context", "") if grounded else "",
            "sources": [self_rag.source_label(d) for d in res.get("relevant_docs", [])]
                       if grounded else [],
            "kept": [self_rag.source_label(d) for d in res.get("relevant_docs", [])],
            "dropped": [self_rag.source_label(d) for d in res.get("docs", [])
                        if d not in res.get("relevant_docs", [])],
            "support": res.get("support"),
            "usefulness": res.get("usefulness"),
            "usefulness_reason": res.get("usefulness_reason", ""),
            "rewrites": res.get("rewrites", 0),
            "revisions": res.get("revisions", 0),
            "web_searched": bool(res.get("web_searched")),
            "web_error": res.get("web_error") or "",
            "llm_calls": counter.calls,
            "pipeline_llm_calls": res.get("llm_calls", 0),
            "budget_reached": self_rag.budget_reached(res),
            "tokens": {m: dict(t) for m, t in counter.tokens.items()},
            "cost_usd": cost,
            "unpriced_models": unpriced,
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
    if "no_answer" in expected_paths(reference_outputs["expected_path"]) and not facts:
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


def citations(outputs: dict) -> dict:
    """Share of cited labels that exist in the context; uncited grounded answers score 0."""
    if outputs["path"] not in ("docs", "web"):
        return {"key": "citations", "score": None, "comment": "n/a: not a grounded answer"}
    known = set(outputs.get("sources") or [])
    cited = self_rag.cited_labels(outputs["answer"])   # same parser as the pipeline
    if not cited:
        return {"key": "citations", "score": 0.0, "comment": "no citations"}
    invalid = [c for c in cited if c not in known]
    return {"key": "citations", "score": round(1 - len(invalid) / len(cited), 3),
            "comment": "invalid: " + "; ".join(invalid) if invalid else "all valid"}


def path_match(outputs: dict, reference_outputs: dict) -> dict:
    expected = expected_paths(reference_outputs["expected_path"])
    return {"key": "path_match", "score": float(outputs["path"] in expected),
            "comment": f"expected {' or '.join(expected)}, got {outputs['path']}"}


def llm_calls(outputs: dict) -> dict:
    return {"key": "llm_calls", "score": outputs["llm_calls"]}


def cost_per_1k(outputs: dict) -> dict:
    if outputs.get("cost_usd") is None:
        return {"key": "cost_per_1k", "score": None,
                "comment": f"unpriced models: {outputs.get('unpriced_models')}"}
    return {"key": "cost_per_1k", "score": round(outputs["cost_usd"] * 1000, 3)}


def latency_s(outputs: dict) -> dict:
    return {"key": "latency_s", "score": outputs["latency_s"]}


EVALUATORS = [correctness, faithfulness, citations, path_match, llm_calls, cost_per_1k,
              latency_s]


# ======================================================================
# RUNNERS (both return rows: case_id, question, outputs, results [{key, score, comment}])
# ======================================================================
def run_langsmith(target, client: Client, data, args, metadata: dict):
    """LangSmith evaluate(): experiment and traces are uploaded."""
    results = evaluate(
        target,
        data=data,
        evaluators=EVALUATORS,
        experiment_prefix=args.prefix,
        num_repetitions=args.reps,
        max_concurrency=args.concurrency,
        metadata=metadata,
        client=client,
    )
    rows = [{"case_id": row["example"].metadata["case_id"],
             "question": row["example"].inputs["question"],
             "outputs": row["run"].outputs or {},
             "results": [{"key": r.key, "score": r.score, "comment": r.comment}
                         for r in row["evaluation_results"]["results"]]}
            for row in results]
    return results.experiment_name, rows


def run_evaluators(inputs: dict, outputs: dict, reference_outputs: dict) -> List[dict]:
    """Call each evaluator with the arguments it declares, as evaluate() does."""
    available = {"inputs": inputs, "outputs": outputs, "reference_outputs": reference_outputs}
    results = []
    for ev in EVALUATORS:
        try:
            r = ev(**{p: available[p] for p in inspect.signature(ev).parameters})
        except Exception as e:
            r = {"key": ev.__name__, "score": None, "comment": f"evaluator error: {e!r}"}
        results.append({"key": r["key"], "score": r.get("score"), "comment": r.get("comment")})
    return results


def run_local(target, cases: List[dict], args) -> tuple:
    """Same cases, evaluators and repetitions as run_langsmith, without LangSmith.

    Not evaluate(upload_results=False): in langsmith 0.14.13 that still uploads the
    LangChain runs inside the target and the evaluators (decision S9.1).
    """
    experiment = f"{args.prefix}-{uuid.uuid4().hex[:8]}"   # same form as LangSmith's names
    jobs = [(c, rep) for rep in range(args.reps) for c in cases]
    log.info("Local eval '%s': %d cases x %d reps, LangSmith off",
             experiment, len(cases), args.reps)

    def one(case: dict, rep: int) -> dict:
        inputs = {"question": case["question"]}
        reference = {"expected_path": case["expected_path"],
                     "facts": list(case.get("facts") or [])}
        try:
            outputs = target(inputs)
        except Exception as e:   # a crashed run is a failed answer, not a skipped one
            outputs = {"path": "error", "answer": "", "error": repr(e)}
            results = [{"key": m, "score": 0.0 if m in ("correctness", "path_match") else None,
                        "comment": f"target error: {e!r}"} for m in METRICS]
        else:
            results = run_evaluators(inputs, outputs, reference)
        return {"case_id": case["id"], "question": case["question"], "rep": rep,
                "outputs": outputs, "results": results}

    rows = []
    # tracing_context(enabled=False) overrides LANGSMITH_TRACING; copy_context() carries it
    # (and the LLM call counter hook) into the worker threads.
    with tracing_context(enabled=False), \
            ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(copy_context().run, one, c, rep) for c, rep in jobs]
        for done, f in enumerate(as_completed(futures), 1):
            row = f.result()
            scores = {r["key"]: r["score"] for r in row["results"]}
            error = row["outputs"].get("error")
            (log.warning if error else log.info)(
                "[%d/%d] %s (rep %d): path %s, correctness %s%s", done, len(jobs),
                row["case_id"], row["rep"] + 1, row["outputs"].get("path"),
                scores.get("correctness"), f", {error}" if error else "")
            rows.append(row)
    rows.sort(key=lambda r: (r["case_id"], r["rep"]))
    return experiment, rows


# ======================================================================
# SUMMARY
# ======================================================================
def mean(values: List[Optional[float]]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def summarize(rows: List[dict], experiment: str, corpus: str, reps: int, mode: str) -> dict:
    per_case: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    paths: Dict[str, list] = defaultdict(list)
    scopes: Dict[str, list] = defaultdict(list)
    runs: Dict[str, list] = defaultdict(list)
    questions: Dict[str, str] = {}
    for row in rows:
        case_id = row["case_id"]
        questions[case_id] = row["question"]
        out = row["outputs"]
        paths[case_id].append(out.get("path"))
        scopes[case_id].append(out.get("scope"))
        detail = {k: out.get(k) for k in ("path", "scope", "rewrites", "web_searched",
                                          "web_error", "kept",
                                          "dropped", "usefulness", "usefulness_reason",
                                          "llm_calls", "budget_reached", "tokens",
                                          "cost_usd")}
        detail["answer"] = (out.get("answer") or "")[:600]
        if out.get("error"):
            detail["error"] = out["error"]
        detail["comments"] = {}
        for r in row["results"]:
            per_case[case_id][r["key"]].append(r["score"])
            if r["comment"] and r["key"] in ("correctness", "faithfulness", "citations"):
                detail["comments"][r["key"]] = r["comment"]
        runs[case_id].append(detail)

    cases = {
        cid: {"question": questions[cid], "paths": paths[cid], "scopes": scopes[cid],
              **{m: mean(per_case[cid][m]) for m in METRICS}, "runs": runs[cid]}
        for cid in sorted(per_case)
    }
    overall = {m: mean([c[m] for c in cases.values()]) for m in METRICS}
    return {"experiment": experiment, "corpus": corpus, "created": datetime.now().isoformat(
        timespec="seconds"), "mode": mode, "reps": reps, "llm_model": self_rag.LLM_MODEL,
        "llm_reasoning_effort": self_rag.LLM_REASONING_EFFORT, "eval_model": EVAL_MODEL,
        "eval_reasoning_effort": EVAL_REASONING_EFFORT, "overall": overall, "cases": cases}


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
        print(f"{'BASELINE':28}"
              + "".join(f"{fmt(baseline['overall'].get(m)):>12}" for m in METRICS)
              + f"   ({baseline['experiment']})")


# ======================================================================
# CLI
# ======================================================================
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate self_rag.py (LangSmith or local).")
    parser.add_argument("--corpus", required=True, help="corpus folder containing eval.yaml")
    parser.add_argument("--prefix", default="self-rag", help="experiment name prefix")
    parser.add_argument("--reps", type=int, default=1, help="runs per case (answers vary)")
    parser.add_argument("--concurrency", type=int, default=2, help="cases run in parallel")
    parser.add_argument("--only", help="comma-separated case ids to run")
    parser.add_argument("--compare", type=Path, help="earlier summary JSON to compare with")
    parser.add_argument("--local", action="store_true",
                        help="run without LangSmith (no dataset sync, no uploads, tracing off)")
    args = parser.parse_args(argv)
    setup_logging()

    if not args.local and not os.getenv("LANGSMITH_API_KEY"):
        sys.exit("LANGSMITH_API_KEY is not set (.env). Use --local to run without LangSmith.")
    baseline = json.loads(args.compare.read_text(encoding="utf-8")) if args.compare else None

    corpus_dir = Path(args.corpus)
    cfg = CorpusConfig.load(corpus_dir)
    index = PdfIndex(cfg, self_rag.embeddings)
    if index.count() == 0:
        sys.exit(f"Corpus '{cfg.name}' is not indexed: run self_rag.py index first.")

    cases = load_cases(corpus_dir if corpus_dir.is_dir() else corpus_dir.parent)
    wanted = {s.strip() for s in args.only.split(",")} if args.only else None
    if wanted and not any(c["id"] in wanted for c in cases):
        sys.exit(f"No cases match --only {args.only}")
    target = make_target(index)

    if args.local:
        experiment, rows = run_local(
            target, [c for c in cases if not wanted or c["id"] in wanted], args)
    else:
        client = Client()
        dataset = f"self-rag-{cfg.name}"
        sync_dataset(client, dataset, cases)
        data = dataset
        if wanted:
            data = [ex for ex in client.list_examples(dataset_name=dataset)
                    if (ex.metadata or {}).get("case_id") in wanted]
        metadata = {"corpus": cfg.name, "llm_model": self_rag.LLM_MODEL,
                    "llm_reasoning_effort": self_rag.LLM_REASONING_EFFORT or "default",
                    "eval_model": EVAL_MODEL,
                    "eval_reasoning_effort": EVAL_REASONING_EFFORT or "default",
                    "reps": args.reps}
        experiment, rows = run_langsmith(target, client, data, args, metadata)

    summary = summarize(rows, experiment, cfg.name, args.reps,
                        "local" if args.local else "langsmith")
    out_dir = RESULTS_DIR / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{experiment}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print_summary(summary, baseline)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
