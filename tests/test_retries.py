"""Timeouts and retries for LLM, embedding and web search calls (decisions S9.7-S9.12)."""
import functools
import os

from _support import Checks, FakeServer

srv = FakeServer()
os.environ["OPENAI_BASE_URL"] = srv.url + "/v1"

import openai  # noqa: E402
import rag_common as rc  # noqa: E402
import self_rag as sr  # noqa: E402
from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from tavily import TavilyClient  # noqa: E402

sr.TavilyClient = functools.partial(TavilyClient, api_base_url=srv.url)
check = Checks("timeouts and retries")


def run(script, fn, env=None):
    """(result, error, requests made, seconds) with temporary env overrides."""
    import time
    srv.reset(script)
    old = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    start = time.perf_counter()
    try:
        return fn(), None, len(srv.requests), time.perf_counter() - start
    except Exception as e:
        return None, e, len(srv.requests), time.perf_counter() - start
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


# ---------- settings ----------
check("defaults", [rc.api_setting(k) for k in rc.API_DEFAULTS] == [120.0, 60.0, 60.0, 3])
for name, value in [("API_MAX_RETRIES", "x"), ("API_MAX_RETRIES", "-1"),
                    ("LLM_TIMEOUT_S", "0"), ("LLM_TIMEOUT_S", "-5")]:
    os.environ[name] = value
    try:
        rc.api_setting(name)
        ok = False
    except ValueError:
        ok = True
    os.environ.pop(name)
    check(f"rejects {name}={value}", ok)
os.environ["API_MAX_RETRIES"] = "0"
check("API_MAX_RETRIES=0 allowed", rc.api_setting("API_MAX_RETRIES") == 0)
os.environ.pop("API_MAX_RETRIES")
m = rc.chat_model("model-a", "none")
check("chat model gets timeout and retries", (m.request_timeout, m.max_retries, m.temperature)
      == (120.0, 3, 0))
m = rc.chat_model("model-b", "low")
check("effort low: no temperature", m.temperature is None and m.max_retries == 3)
e = rc.embedding_model("embed-a")
check("embedding model gets timeout and retries", (e.request_timeout, e.max_retries) == (60.0, 3))

# ---------- call_with_retries ----------
calls, slept = [], []
real_sleep, rc.time.sleep = rc.time.sleep, slept.append


def flaky(failures, exc):
    def fn():
        calls.append(1)
        if len(calls) <= failures:
            raise exc
        return "ok"
    return fn


calls.clear()
check("retry then success", rc.call_with_retries(flaky(2, OSError()), lambda e: True, "t", 3)
      == "ok" and len(calls) == 3)
calls.clear()
try:
    rc.call_with_retries(flaky(9, OSError()), lambda e: True, "t", 3)
    ok = False
except OSError:
    ok = len(calls) == 4
check("gives up after the retries", ok)
calls.clear()
try:
    rc.call_with_retries(flaky(9, ValueError()), lambda e: False, "t", 3)
    ok = False
except ValueError:
    ok = len(calls) == 1
check("non-retryable error raised at once", ok)
check("backoff between 0.25 and 8 s", all(0.25 <= s <= 8 for s in slept), str(slept))
rc.time.sleep = real_sleep

# ---------- Tavily ----------
r, err, n, _ = run({"/search": [429, 200]}, lambda: sr.search_web("q"))
check("web: 429 retried", err is None and n == 2, f"{n} requests, {err!r}")
sent = r["_sent"] if r else {}
check("web: request parameters", sent.get("search_depth") == "advanced"
      and sent.get("topic") == "general" and sent.get("max_results") == sr.WEB_MAX_RESULTS
      and "time_range" not in sent, str(sent))
r, err, n, _ = run({"/search": [500, 502, 200]}, lambda: sr.search_web("q", "week"))
check("web: 5xx retried, time_range sent", err is None and n == 3
      and r["_sent"]["time_range"] == "week", f"{n} requests, {err!r}")
for code in (400, 401, 403, 432, 433):
    r, err, n, _ = run({"/search": [code, 200]}, lambda: sr.search_web("q"))
    check(f"web: {code} not retried", err is not None and n == 1, f"{n} requests")
r, err, n, _ = run({"/search": [503]}, lambda: sr.search_web("q"))
check("web: gives up after 3 retries", err is not None and n == 4, f"{n} requests")
r, err, n, dt = run({"/search": ["hang"]}, lambda: sr.search_web("q"),
                    {"WEB_TIMEOUT_S": "1", "API_MAX_RETRIES": "1"})
check("web: timeout per request, retried", type(err).__name__ == "TimeoutError" and n == 2
      and dt < 5, f"{n} requests, {dt:.1f} s, {err!r}")
for key in ("", "   "):
    os.environ["TAVILY_API_KEY"] = key
    r, err, n, _ = run({}, lambda: sr.search_web("q"))
    check(f"web: key {key!r} counts as missing, no request (never keyless)",
          isinstance(err, RuntimeError) and n == 0, repr(err))
os.environ["TAVILY_API_KEY"] = "test-key"
r, err, n, _ = run({"/search": [500, 200]},
                   lambda: sr.web_search({"web_query": "q", "web_time_range": None}))
check("web_search node: docs after a retry, brackets made safe", err is None
      and len(r["docs"]) == 1 and sr.source_label(r["docs"][0])
      == "web: Page (1) (https://example.test/a)", repr(r))


# ---------- OpenAI chat and embeddings ----------
class Count(BaseCallbackHandler):
    def __init__(self):
        self.n = 0

    def on_chat_model_start(self, *args, **kwargs):
        self.n += 1


counter = Count()
r, err, n, _ = run({"/v1/chat/completions": [500, 429, 200]},
                   lambda: sr.llm.invoke("hi", {"callbacks": [counter]}))
check("llm: 500 and 429 retried by the SDK", err is None and r.content == "ok" and n == 3,
      f"{n} requests, {err!r}")
check("llm: retries count as one LLM call", counter.n == 1, str(counter.n))
r, err, n, _ = run({"/v1/chat/completions": [400]}, lambda: sr.llm.invoke("hi"))
check("llm: 400 not retried", err is not None and n == 1, f"{n} requests")
r, err, n, _ = run({"/v1/chat/completions": [503]}, lambda: sr.llm.invoke("hi"))
check("llm: gives up after 3 retries", err is not None and n == 4, f"{n} requests")
r, err, n, dt = run({"/v1/chat/completions": ["hang"]},
                    lambda: rc.chat_model("model-a", "none").invoke("hi"),
                    {"LLM_TIMEOUT_S": "1", "API_MAX_RETRIES": "1"})
check("llm: timeout per request, retried", isinstance(err, openai.APITimeoutError) and n == 2,
      f"{n} requests, {dt:.1f} s, {err!r}")


def embed(model):
    model.check_embedding_ctx_length = False   # no tiktoken download
    return model.embed_query("hello")


r, err, n, _ = run({"/v1/embeddings": [500, 200]}, lambda: embed(sr.embeddings))
check("embeddings: 500 retried", err is None and r == [0.1, 0.2, 0.3] and n == 2,
      f"{n} requests, {err!r}")
r, err, n, dt = run({"/v1/embeddings": ["hang"]}, lambda: embed(rc.embedding_model("embed-a")),
                    {"EMBED_TIMEOUT_S": "1", "API_MAX_RETRIES": "1"})
check("embeddings: timeout per request, retried", isinstance(err, openai.APITimeoutError)
      and n == 2, f"{n} requests, {err!r}")
check.finish()
