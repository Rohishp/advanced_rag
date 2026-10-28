"""Refinement of long sources: grade and pick passages in one call (plan Step 11, S11.5)."""
import contextlib
import io
import subprocess
import sys

from _support import ROOT, Checks

import self_rag as sr  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

real_short = sr.relevance_chain.first        # real prompts, before the stubs replace the chains
real_long = sr.relevance_passages_chain.first
import _graph_stubs as stubs  # noqa: E402

stubs.install()
check = Checks("refinement of long sources")

# ---- the short-source grader prompt must stay exactly as before Step 11 ----
OLD_SYSTEM = (
    "Judge whether a text can help answer a question.\n"
    "The knowledge base contains KB. A text whose source is a file comes "
    "from it and is about the knowledge base's own organization unless it says "
    "otherwise. A text whose source starts with 'web:' comes from a web search.\n"
    "1. question_entity: the organization or person the question is about. If it "
    "asks about a product, plan, policy or role, name the organization it belongs "
    "to. 'none' if it is about no specific one.\n"
    "2. same_entity: the text is about that organization or person, including any of "
    "its own products, plans, people and policies. A different entity with a similar "
    "or overlapping name (e.g. 'Acme Labs' vs 'Acme Systems') is not the same. A web "
    "text counts only if it names the entity.\n"
    "3. on_topic: the text has information for at least one part of the question; "
    "it need not answer all of it.\n" + sr.CONTEXT_IS_DATA
)
args = {"question": "Q?", "source": "s", "chunk": "c", "kb_description": "KB"}
check("short sources: grader prompt unchanged",
      real_short.format_messages(**args)[0].content == OLD_SYSTEM)
long_system = real_long.format_messages(**args)[0].content
check("long sources: same criteria plus passages",
      long_system.startswith(OLD_SYSTEM[:-len(sr.CONTEXT_IS_DATA)]) and "4. passages:" in long_system
      and long_system.endswith(sr.CONTEXT_IS_DATA))

# ---- passages ----
text = ("Intro line without a stop\n" + " ".join(f"Sentence {i} is here." for i in range(120))
        + "\nTable row | a | b\n" + "x" * 900)
ps = sr.split_passages(text, 120)
check("passages respect the size limit (except one over-long sentence)",
      all(len(p) <= 120 for p in ps[:-1]) and ps[-1] == "x" * 900, str([len(p) for p in ps]))
check("passages keep every word in order", " ".join(" ".join(ps).split()) == " ".join(text.split()))
check("line breaks are kept inside passages", any("\n" in p for p in ps))
check("short text: one passage", sr.split_passages("One. Two.", 400) == ["One. Two."])
row = " | ".join(f"cell {i}" for i in range(300))          # a long line without full stops
ps_row = sr.split_passages(row, 120)
check("long line without full stops is cut at spaces",
      len(ps_row) > 1 and all(len(p) <= 120 for p in ps_row)
      and " ".join(ps_row) == row, str([len(p) for p in ps_row]))
check("empty text: no passages", sr.split_passages("  \n ", 400) == [])

long_doc = Document(page_content=text, metadata={"source": "Long page", "url": "https://e.test/l"})
ps = sr.split_passages(long_doc.page_content)
r = sr.refined(long_doc, ps, [3, 2, 3, 99, 0])
check("picked passages verbatim, in order, duplicates and invalid numbers ignored",
      r.page_content == "\n".join(["…", ps[1], ps[2], "…"]) and r.metadata["passages"] == f"2/{len(ps)}"
      and r.metadata["original_chars"] == len(text) and r.metadata["url"] == "https://e.test/l",
      r.page_content[:200])
r = sr.refined(long_doc, ps, [1, len(ps)])
check("gap marks only where text was left out",
      r.page_content == "\n".join([ps[0], "…", ps[-1]]) if len(ps) > 2 else True)
check("no valid passage picked: the source is used whole", sr.refined(long_doc, ps, [99]) is long_doc)
check("source label unchanged by refinement",
      sr.source_label(sr.refined(long_doc, ps, [1])) == sr.source_label(long_doc))
check("threshold: long refined, short not",
      sr.needs_refining(long_doc) and not sr.needs_refining(stubs.INTERNAL_DOCS[0]))

# ---- through the graph ----
seen = []
grade_passages = sr.relevance_passages_chain
sr.relevance_passages_chain = stubs._chain(lambda x: (seen.append(x["chunk"]),
                                                      stubs._grade_passages(x))[1])
short_web = {"title": "Short page", "url": "https://e.test/s", "content": "Short web text."}
sr.search_web = lambda q, t=None: {"results": [
    {"title": "Long page", "url": "https://e.test/l", "content": text}, short_web]}
app = sr.build_graph()


def ask(**script):
    stubs.reset(**{"scope": "external", "keep": False, **script})
    seen.clear()
    return app.invoke({"question": "Q?"}, config=stubs.CONFIG)


r = ask(passages=[2, 3])
rel = r["relevant_docs"]
check("long web page graded as numbered passages", len(seen) == 1 and seen[0].startswith("#1: "))
check("one LLM call per source (no extra refinement calls)",
      r["llm_calls"] == 1 + len(stubs.INTERNAL_DOCS) + 1 + 2 + 3, str(r["llm_calls"]))
check("refined page kept with its passages, short page whole",
      len(rel) == 2 and rel[0].metadata.get("passages") == f"2/{len(ps)}"
      and rel[1].page_content == "Short web text.")
check("context holds only the picked passages", ps[1] in r["context"] and ps[0] not in r["context"]
      and "Short web text." in r["context"])
check("answer still cites the full web label", sr.answer_path(r) == "web"
      and r["answer"] == f"Answer [{sr.source_label(long_doc)}]", r["answer"])
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sr.print_result(r)
out = buf.getvalue()
check("printout marks the refined page as kept, with its passages",
      f"[keep S1, 2/{len(ps)} passages] {sr.source_label(long_doc)}" in out
      and "[keep S2] web: Short page" in out, out[:600])

saved, sr.REFINE_MIN_CHARS = sr.REFINE_MIN_CHARS, 0
r = ask(passages=[2])
check("REFINE_MIN_CHARS=0: no refinement, whole page", not seen
      and r["relevant_docs"][0].page_content == text)
sr.REFINE_MIN_CHARS = saved
check("default threshold 3000 chars", sr.REFINE_MIN_CHARS == 3000)

# ---- settings ----
code = ("import os, sys; sys.path.insert(0, 'tests'); import _support; "
        "os.environ[sys.argv[1]] = sys.argv[2]; import self_rag; "
        "print(getattr(self_rag, sys.argv[1]))")
for name, value, ok in (("REFINE_MIN_CHARS", "abc", False), ("REFINE_MIN_CHARS", "-1", False),
                        ("REFINE_MIN_CHARS", "0", True), ("PASSAGE_CHARS", "10", False),
                        ("PASSAGE_CHARS", "300", True)):
    p = subprocess.run([sys.executable, "-c", code, name, value], capture_output=True, text=True,
                       cwd=ROOT)
    passed = (p.returncode == 0 and p.stdout.strip() == value) if ok else \
        (p.returncode != 0 and name in p.stderr)
    check(f"{name}={value} {'accepted' if ok else 'rejected'}", passed, p.stderr[-200:])
check.finish()
