"""Status goes to rag.* loggers on stderr, results to stdout (decisions S9.23-S9.26)."""
import subprocess
import sys

from _support import ROOT, Checks

check = Checks("logging")
CODE = """
import logging, os, sys
sys.path.insert(0, 'tests')
import _support
os.environ.update(dict(a.split('=', 1) for a in sys.argv[1:]))
import rag_common as rc, self_rag as sr
rc.setup_logging()
sr.report_tracing()
logging.getLogger('httpx').info('HTTP Request: POST hidden')
logging.getLogger('rag.common').debug('debug line')
rc.log.info('%s: unchanged, skipped', 'a.pdf')
print('ANSWER on stdout')
"""


def run(*env):
    p = subprocess.run([sys.executable, "-c", CODE, *env], capture_output=True, text=True,
                       cwd=ROOT)
    return p.stdout, p.stderr, p.returncode


out, err, code = run()
check("result on stdout only", out.strip() == "ANSWER on stdout" and "ANSWER" not in err, out)
check("status on stderr from rag.* loggers", "INFO rag.self_rag: LangSmith tracing: off" in err
      and "INFO rag.common: a.pdf: unchanged, skipped" in err, err)
check("httpx INFO and DEBUG hidden by default", "HTTP Request" not in err
      and "debug line" not in err)
out, err, code = run("LOG_LEVEL=warning")
check("LOG_LEVEL=warning hides INFO", "unchanged" not in err and code == 0, err)
out, err, code = run("LOG_LEVEL=debug")
check("LOG_LEVEL=debug shows DEBUG, still no httpx", "debug line" in err
      and "HTTP Request" not in err, err)
out, err, code = run("LOG_LEVEL=loud")
check("invalid LOG_LEVEL rejected", code != 0 and "LOG_LEVEL='LOUD'" in err, err[-200:])
out, err, code = run("LANGSMITH_TRACING=true", "LANGSMITH_API_KEY=")
check("tracing without key is a WARNING", "WARNING rag.self_rag: LangSmith tracing: "
      "LANGSMITH_TRACING is true" in err, err)
check.finish()
