"""The usefulness check is told the question's scope (decisions S10.11, S11.7)."""
from _support import Checks

import self_rag as sr  # noqa: E402

real_prompt = sr.usefulness_chain.first   # the real prompt, before the stubs replace the chain
import _graph_stubs as stubs  # noqa: E402

stubs.install()
check = Checks("usefulness check by scope")
KB = stubs.CONFIG["configurable"]["kb_description"]

# The knowledge-base prompt must stay exactly as before this change.
OLD_SYSTEM = (
    "Judge whether the ANSWER gives the user what the QUESTION asks for.\n"
    f"The knowledge base contains {KB}.\n"
    "- useful: it provides the requested information, even if it notes a "
    "limitation.\n"
    "- not_useful: it says the information was not found (in any wording); it is "
    "generic or off-topic; or it answers about a different organization, person or "
    "product than the one asked about (e.g. 'Acme Labs' vs 'Acme Systems').\n"
    "A short and a full form of one name (e.g. 'Acme' and 'Acme Labs Ltd.'), or the "
    "name the knowledge base uses for the entity asked about, are the same entity.\n"
    "Do not judge correctness; that was checked before."
)


def system_text(scope):
    note = sr.usefulness_source_note({"scope": scope} if scope else {}, stubs.CONFIG)
    msgs = real_prompt.format_messages(question="Q?", answer="A", source_note=note)
    return msgs[0].content


check("knowledge_base: prompt unchanged", system_text("knowledge_base") == OLD_SYSTEM)
check("no scope in the state: knowledge-base prompt", system_text(None) == OLD_SYSTEM)
ext = system_text("external")
check("external: no knowledge-base description", KB not in ext and "knowledge base contains"
      not in ext, ext)
check("external: told the question is outside the knowledge base",
      sr.EXTERNAL_SOURCE_NOTE in ext)
check("external: namesake rule still in the prompt", "different organization, person or "
      "product than the one asked about" in ext)

seen = []
sr.usefulness_chain = stubs._chain(lambda x: (seen.append(x), sr.UsefulnessCheck(
    reason="r", usefulness=stubs.S["useful"]))[1])
sr.search_web = lambda q, t=None: {"results": [
    {"title": "Page", "url": "https://example.test/p", "content": "text"}]}
app = sr.build_graph()


def ask(**script):
    stubs.reset(**script)
    seen.clear()
    return app.invoke({"question": "Q?"}, config=stubs.CONFIG)


r = ask()
check("graph, knowledge_base question: knowledge-base note",
      [x["source_note"] for x in seen] == [f"The knowledge base contains {KB}."], str(seen))
r = ask(scope="external", keep=False)
check("graph, external question on the web path: external note",
      sr.answer_path(r) == "web" and [x["source_note"] for x in seen]
      == [sr.EXTERNAL_SOURCE_NOTE], str(seen))
check.finish()
