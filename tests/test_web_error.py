"""A failed web search is recorded in web_error and ends in no_answer (decisions S9.15-S9.19)."""
import contextlib
import functools
import io
import os

from _support import Checks, FakeServer

srv = FakeServer()
os.environ["API_MAX_RETRIES"] = "1"

import _graph_stubs as stubs  # noqa: E402
import self_rag as sr  # noqa: E402
from tavily import TavilyClient  # noqa: E402

sr.TavilyClient = functools.partial(TavilyClient, api_base_url=srv.url)
stubs.install()
app = sr.build_graph()
check = Checks("web_error routing")


def ask(**script):
    stubs.reset(**{"scope": "external", "keep": False, **script})
    return app.invoke({"question": "Who won the invented cup?"}, config=stubs.CONFIG)


os.environ["TAVILY_API_KEY"] = ""
r = ask()
check("no key: path no_answer", sr.answer_path(r) == "no_answer")
check("no key: web_error recorded", r.get("web_error") == "RuntimeError: TAVILY_API_KEY is not set",
      repr(r.get("web_error")))
check("no key: only the internal pass graded", all(not g.startswith("web:") for g in stubs.GRADED))
check("no key: answer is NO_ANSWER", r["answer"] == sr.NO_ANSWER)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sr.print_result(r)
check("printout shows WEB ERROR", "WEB ERROR: RuntimeError: TAVILY_API_KEY is not set"
      in buf.getvalue())

os.environ["TAVILY_API_KEY"] = "test-key"
srv.reset({"/search": [401]})
r = ask()
check("401: no_answer with InvalidAPIKeyError", sr.answer_path(r) == "no_answer"
      and r["web_error"].startswith("InvalidAPIKeyError"), repr(r.get("web_error")))
srv.reset({"/search": [503]})
r = ask()
check("503 after retries: no_answer with HTTPError", sr.answer_path(r) == "no_answer"
      and r["web_error"].startswith("HTTPError"), repr(r.get("web_error")))
srv.reset()
r = ask()
check("search works: web path, empty web_error", sr.answer_path(r) == "web"
      and r["web_error"] == "" and any(g.startswith("web:") for g in stubs.GRADED))

sr.search_web, real_search = (lambda q, t=None: {"results": []}), sr.search_web
r = ask()
check("no results is not an error: no_answer, empty web_error",
      sr.answer_path(r) == "no_answer" and r["web_error"] == "")
sr.search_web = real_search

os.environ["TAVILY_API_KEY"] = ""
r = ask(keep=True, useful="not_useful")
check("not useful, then failed search: no stale sources", sr.answer_path(r) == "no_answer"
      and r["relevant_docs"] == [] and r["web_error"])
r = ask(scope="knowledge_base", keep=False)
check("knowledge_base never searches the web", not r.get("web_searched")
      and not r.get("web_error") and r.get("rewrites") == sr.MAX_REWRITES)
check.finish()
