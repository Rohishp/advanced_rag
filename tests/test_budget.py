"""Per-question LLM call budget (decisions S9.29-S9.33)."""
import contextlib
import io
import subprocess
import sys

from _support import ROOT, Checks

import _graph_stubs as stubs  # noqa: E402
import self_rag as sr  # noqa: E402
from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402

stubs.install()
sr.search_web = lambda q, t=None: {"results": [
    {"title": f"Page {i}", "url": f"https://example.test/{i}", "content": "text"}
    for i in range(3)]}
app = sr.build_graph()
check = Checks("LLM call budget")
k = len(stubs.INTERNAL_DOCS)


class Count(BaseCallbackHandler):
    def __init__(self):
        self.n = 0

    def on_chat_model_start(self, *args, **kwargs):
        self.n += 1


def ask(budget=30, **script):
    stubs.reset(**script)
    sr.MAX_LLM_CALLS = budget
    counter = Count()
    r = app.invoke({"question": "Q?"}, config={**stubs.CONFIG, "callbacks": [counter]})
    return r, counter.n


r, n = ask()
check("docs path: state count = callback count = 1+k+3", r["llm_calls"] == n == 1 + k + 3,
      f"{r['llm_calls']}/{n}")
r, n = ask(keep=False)
check("knowledge-base gap: counts match, no budget hit", r["llm_calls"] == n
      == 1 + k + 2 * (1 + k) and r["rewrites"] == 2 and not sr.budget_reached(r),
      f"{r['llm_calls']}/{n}")
r, n = ask(scope="general")
check("direct: 2 calls", r["llm_calls"] == n == 2)
r, n = ask(scope="external", keep=False)
check("external, web path: counts match", sr.answer_path(r) == "web" and r["llm_calls"] == n
      == 1 + k + 1 + 3 + 3, f"{r['llm_calls']}/{n}")
r, n = ask(support="partially_supported")
check("revisions to the limit: counts match", r["revisions"] == sr.MAX_REVISIONS
      and r["llm_calls"] == n, f"{r['llm_calls']}/{n}")

r, n = ask(budget=k + 2, keep=False)
check("budget on a gap: stops after one rewrite", r["rewrites"] == 1
      and sr.answer_path(r) == "no_answer" and sr.budget_reached(r) and r["llm_calls"] == n,
      f"rewrites {r.get('rewrites')}, {r['llm_calls']}/{n}")
r, n = ask(budget=k + 4, support="partially_supported")
check("budget mid-revision: one revision, then accept", r["revisions"] == 1
      and sr.answer_path(r) == "docs" and r["llm_calls"] == n == k + 6,
      f"revisions {r.get('revisions')}, {r['llm_calls']}/{n}")
r, n = ask(budget=k + 2, cite="missing.pdf p.9")
check("budget reached with citation errors: support still LLM-checked",
      r["support"] == "fully_supported" and r["citation_errors"] and r.get("revisions", 0) == 0
      and r["llm_calls"] == n == k + 4, f"{r.get('support')}, {r['llm_calls']}/{n}")
r, n = ask(cite="missing.pdf p.9")
check("within budget, citation errors: revised to the limit (S8.3)",
      r["revisions"] == sr.MAX_REVISIONS and r["support"] == "fully_supported")
r, n = ask(budget=k + 1, scope="external", keep=False)
check("budget on an external question: no web search", not r.get("web_searched")
      and sr.answer_path(r) == "no_answer")
r, n = ask(budget=1)
check("budget 1: the started attempt finishes", sr.answer_path(r) == "docs"
      and sr.budget_reached(r))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sr.print_result(r)
check("printout shows calls and budget",
      f"LLM CALLS: {r['llm_calls']} (budget 1, reached: no further attempts)" in buf.getvalue())

code = ("import os, sys; sys.path.insert(0, 'tests'); import _support; "
        "os.environ['MAX_LLM_CALLS'] = sys.argv[1]; import self_rag; print(self_rag.MAX_LLM_CALLS)")
for value, ok in (("abc", False), ("0", False), ("12", True)):
    p = subprocess.run([sys.executable, "-c", code, value], capture_output=True, text=True,
                       cwd=ROOT)
    passed = (p.returncode == 0 and p.stdout.strip() == value) if ok else \
        (p.returncode != 0 and "MAX_LLM_CALLS" in p.stderr)
    check(f"MAX_LLM_CALLS={value} {'accepted' if ok else 'rejected'}", passed, p.stderr[-200:])
check.finish()
