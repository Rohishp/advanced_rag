"""Shared setup for the offline tests: no .env, no network, no real API keys.

Import this module before self_rag, rag_common or eval_self_rag.
"""
import json
import os
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Never load the real .env (keys, tracing, limits): self_rag imports load_dotenv at import.
import dotenv  # noqa: E402
dotenv.load_dotenv = lambda *a, **k: False

# Fake keys, tracing off, every setting at its default; tests override what they need.
os.environ.update(OPENAI_API_KEY="test-key", OPENAI_BASE_URL="http://127.0.0.1:9/v1",
                  TAVILY_API_KEY="test-key", LANGSMITH_TRACING="false", LANGSMITH_API_KEY="")
for name in ("LLM_TIMEOUT_S", "EMBED_TIMEOUT_S", "WEB_TIMEOUT_S", "API_MAX_RETRIES",
             "MAX_LLM_CALLS", "LOG_LEVEL", "LANGSMITH_ENDPOINT", "LANGCHAIN_TRACING_V2"):
    os.environ.pop(name, None)

try:  # PDFs are never loaded here; stub the extractor only if it is not installed
    __import__("pymupdf4llm")   # only checks that it is installed
except Exception:
    sys.modules["pymupdf4llm"] = types.ModuleType("pymupdf4llm")


class Checks:
    """Collects PASS/FAIL lines; finish() exits non-zero if anything failed."""

    def __init__(self, title: str):
        self.failed = []
        print(f"== {title}")

    def __call__(self, name: str, ok: bool, info: str = "") -> None:
        print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info and not ok else ""))
        if not ok:
            self.failed.append(name)

    def finish(self) -> None:
        print(f"-- {'all passed' if not self.failed else 'FAILED: ' + ', '.join(self.failed)}\n")
        sys.exit(1 if self.failed else 0)


class FakeServer:
    """A local HTTP server for OpenAI, Tavily and LangSmith endpoints.

    script[path] is a list of actions for successive requests (the last one repeats):
    an HTTP status code, or "hang" (sleep 3 s, then 200). requests lists every path hit.
    """

    def __init__(self):
        self.script, self.requests = {}, []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self):
                n = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(n) if n else b""
                path = self.path.split("?")[0]
                server.requests.append(path)
                actions = server.script.get(path) or [200]
                action = actions.pop(0) if len(actions) > 1 else actions[0]
                if action == "hang":
                    import time
                    time.sleep(3)
                    action = 200
                try:
                    body = json.loads(raw or b"{}")
                except ValueError:
                    body = {}
                status, out = (action, {"error": {"message": f"fake {action}"},
                                        "detail": {"error": f"fake {action}"}}) \
                    if action != 200 else (200, server.response(path, body))
                data = json.dumps(out).encode()
                try:
                    self.send_response(status)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(data)))
                    self.send_header("retry-after-ms", "10")
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass   # the client already timed out

            do_GET = do_POST = do_PATCH = do_PUT = _handle

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"

    def reset(self, script=None):
        self.script = dict(script or {})
        self.requests.clear()

    @staticmethod
    def response(path: str, body: dict) -> dict:
        if path.endswith("/chat/completions"):
            return {"id": "x", "object": "chat.completion", "created": 0,
                    "model": body.get("model", "m"),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}
        if path.endswith("/embeddings"):
            inputs = body.get("input") or [""]
            return {"object": "list", "model": body.get("model", "m"),
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    "data": [{"object": "embedding", "index": i, "embedding": [0.1, 0.2, 0.3]}
                             for i in range(len(inputs))]}
        if path.endswith("/search"):
            return {"query": body.get("query"), "_sent": body,
                    "results": [{"title": "Page [1]", "url": "https://example.test/a",
                                 "content": "text"}]}
        return {}
