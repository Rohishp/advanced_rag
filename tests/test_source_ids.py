"""Short source ids in the context and web de-duplication by URL (plan Step 10)."""
import contextlib
import io

from _support import Checks

import _graph_stubs as stubs  # noqa: E402
import self_rag as sr  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

stubs.install()
check = Checks("source ids and web de-duplication")

pdf = Document(page_content="Section: Handbook > Leave\nTwo years.",
               metadata={"source": "acme.pdf", "page": 2, "section": "Handbook > Leave"})
web = Document(page_content="The invented cup was won by Acme FC.",
               metadata={"source": "Invented Cup final - Wiki [1]",
                         "url": "https://example.test/Acme%27s_Cup"})
docs = [pdf, web]
PDF_LABEL, WEB_LABEL = sr.source_label(pdf), sr.source_label(web)

# ---- context and citation check (pure functions) ----
ctx = sr.format_context(docs)
check("context labels sources [S1], [S2] with their full labels",
      ctx.startswith(f"[S1] Source: {PDF_LABEL}\n") and f"[S2] Source: {WEB_LABEL}\n" in ctx, ctx)
check("valid ids: no errors", sr.citation_errors("A [S1]. B [S2].", docs) == [])
check("grouped and lower-case ids are valid", sr.citation_errors("A [S1, s2]. B [S2; S1]", docs)
      == [])
check("unknown id is an error", sr.citation_errors("A [S3].", docs)
      == ["[S3] is not a source id in the context"])
check("a full label is no longer a valid citation", sr.citation_errors(f"A [{PDF_LABEL}]", docs)
      == [f"[{PDF_LABEL}] is not a source id"])
check("a web title without id is an error",
      sr.citation_errors("A [web: Invented Cup final - Wiki]", docs)
      == ["[web: Invented Cup final - Wiki] is not a source id"])
check("no citation is an error", sr.citation_errors("A.", docs) == ["the answer cites no sources"])
check("markdown links are not citations", sr.citation_errors("[x](https://e.test) [S1]", docs)
      == [])
check("repeated errors listed once", sr.citation_errors("A [S9]. B [S9].", docs)
      == ["[S9] is not a source id in the context"])

# ---- ids mapped back to full labels ----
check("ids resolve to full labels",
      sr.resolve_citations("A [S1]. B [S2].", docs) == f"A [{PDF_LABEL}]. B [{WEB_LABEL}].")
check("grouped ids resolve to one bracket each",
      sr.resolve_citations("A [S2, s1].", docs) == f"A [{WEB_LABEL}][{PDF_LABEL}].")
check("unknown ids and other brackets stay as written",
      sr.resolve_citations("A [S1, S7]. B [note]. [x](https://e.test)", docs)
      == "A [S1, S7]. B [note]. [x](https://e.test)")
check("resolved web label keeps its URL",
      "(https://example.test/Acme%27s_Cup)" in sr.resolve_citations("[S2]", docs))
check("resolved labels pass the eval's label check",
      set(sr.cited_labels(sr.resolve_citations("A [S1]. B [S2].", docs)))
      == {PDF_LABEL, WEB_LABEL})

# ---- URL normalization and de-duplication ----
same = ["https://example.test/Acme%27s_Cup", "https://example.test/Acme's_Cup",
        "HTTPS://Example.TEST/Acme's_Cup/", "https://example.test/Acme%27s_Cup#Final"]
check("percent-encoding, host case, trailing slash and fragment are ignored",
      len({sr.normalize_url(u) for u in same}) == 1, str({sr.normalize_url(u) for u in same}))
check("different paths stay different",
      sr.normalize_url("https://example.test/Cup") != sr.normalize_url("https://example.test/Cup_final"))
check("different queries stay different",
      sr.normalize_url("https://e.test/p?id=1") != sr.normalize_url("https://e.test/p?id=2"))
check("path case is kept", sr.normalize_url("https://e.test/A") != sr.normalize_url("https://e.test/a"))

results = [{"title": f"Page {i}", "url": u, "content": f"text {i}"} for i, u in enumerate(same)]
results.append({"title": "Other", "url": "https://example.test/other", "content": "other"})
results.append({"title": "No URL 1", "url": "", "content": "a"})
results.append({"title": "No URL 2", "url": "", "content": "b"})
sr.search_web = lambda q, t=None: {"results": list(results)}
out = sr.web_search({"web_query": "q"})
check("web_search keeps the first result per URL",
      [d.metadata["source"] for d in out["docs"]] == ["Page 0", "Other", "No URL 1", "No URL 2"],
      str([d.metadata["source"] for d in out["docs"]]))

# ---- through the graph ----
seen_by_usefulness = []
stubs_useful = sr.usefulness_chain
sr.usefulness_chain = stubs._chain(lambda x: (seen_by_usefulness.append(x["answer"]),
                                              sr.UsefulnessCheck(reason="r",
                                                                 usefulness=stubs.S["useful"]))[1])
app = sr.build_graph()


def ask(**script):
    stubs.reset(**script)
    seen_by_usefulness.clear()
    return app.invoke({"question": "Q?"}, config=stubs.CONFIG)


r = ask()
first = sr.source_label(stubs.INTERNAL_DOCS[0])
check("docs path: model cites [S1], answer shows the full label, no revision",
      r["answer"] == f"Answer [{first}]" and r["citation_errors"] == []
      and r.get("revisions", 0) == 0, r["answer"])
check("docs path: context carries ids", r["context"].startswith("[S1] Source: "))
check("usefulness check sees the full label", seen_by_usefulness == [f"Answer [{first}]"],
      str(seen_by_usefulness))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sr.print_result(r)
check("printout marks kept sources with their id", f"[keep S1] {first}:" in buf.getvalue())

r = ask(cite=first)
check("citing the full label instead of the id: revised to the limit, errors kept",
      r["revisions"] == sr.MAX_REVISIONS and r["citation_errors"] == [f"[{first}] is not a source id"]
      and r["support"] == "fully_supported", str(r.get("citation_errors")))

r = ask(cite="S9")
check("unknown id: stays [S9] in the answer and is reported",
      r["answer"] == "Answer [S9]" and r["citation_errors"]
      == ["[S9] is not a source id in the context"], r["answer"])

feedback = sr.check_support({"answer": "A [x]", "relevant_docs": docs, "context": ctx,
                             "revisions": 0, "llm_calls": 0})
check("citation feedback lists the valid ids", feedback["support_reason"].endswith(
      "the context's source ids are S1, S2.") and feedback.get("llm_calls", 0) == 0,
      feedback["support_reason"])

sr.search_web = lambda q, t=None: {"results": results[:5]}   # 4 spellings of one URL + 1
r = ask(scope="external", keep=False)
check("web path: duplicates dropped before grading",
      [g for g in stubs.GRADED if g.startswith("web:")]
      == [sr.source_label(d) for d in out["docs"][:2]], str(stubs.GRADED))
check("web path: answer cites the full web label with its URL",
      sr.answer_path(r) == "web" and r["answer"] == f"Answer [{sr.source_label(r['relevant_docs'][0])}]"
      and "(https://example.test/Acme%27s_Cup)" in r["answer"], r["answer"])
check.finish()
