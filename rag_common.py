"""Shared PDF loading and persistent Chroma indexing for the RAG scripts."""

import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

import pymupdf4llm
import yaml
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger("rag.common")

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging() -> None:
    """For the CLIs: the project's loggers ('rag.*') at LOG_LEVEL (.env, default INFO),
    everything else at WARNING (httpx and openai log every request at INFO).

    Logs go to stderr; the CLIs print their results (answers, eval tables) to stdout.
    """
    name = (os.getenv("LOG_LEVEL") or "INFO").strip().upper()
    level = logging.getLevelName(name)
    if not isinstance(level, int):
        raise ValueError(f"LOG_LEVEL={name!r}: expected DEBUG, INFO, WARNING or ERROR")
    logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT, datefmt="%H:%M:%S")
    logging.getLogger("rag").setLevel(level)

INDEX_BATCH = 500  # max IDs per Chroma add/delete call
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


# ======================================================================
# API SETTINGS (timeouts and retries; .env overrides the defaults)
# ======================================================================
# Read when a client is built, not at import, so a later load_dotenv() still applies.
API_DEFAULTS = {
    "LLM_TIMEOUT_S": 120.0,     # per LLM request (chat models, incl. the eval judge)
    "EMBED_TIMEOUT_S": 60.0,    # per embedding request
    "WEB_TIMEOUT_S": 60.0,      # per web search request (Tavily caps it at 120)
    "API_MAX_RETRIES": 3,       # retries after a failed request, for every API
}


def api_setting(name: str):
    """A timeout (> 0 seconds) or retry count (>= 0) from .env, else its default."""
    default = API_DEFAULTS[name]
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = type(default)(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r}: expected a {type(default).__name__}") from None
    if value < 0 or (value == 0 and isinstance(value, float)):
        raise ValueError(f"{name}={raw!r}: out of range")
    return value


T = TypeVar("T")


def call_with_retries(fn: Callable[[], T], retryable: Callable[[Exception], bool],
                      what: str, retries: Optional[int] = None) -> T:
    """fn() with exponential backoff and jitter for errors that retryable() accepts.

    Delays follow the OpenAI SDK's (0.5 s doubling, capped at 8 s). For clients without
    built-in retries; the OpenAI clients retry on their own (max_retries).
    """
    retries = api_setting("API_MAX_RETRIES") if retries is None else retries
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == retries or not retryable(e):
                raise
            delay = min(8.0, 0.5 * 2 ** attempt) * random.uniform(0.5, 1.0)
            log.warning("%s failed (%r); retry %d/%d in %.1f s",
                        what, e, attempt + 1, retries, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def chat_model(model: str, reasoning_effort: Optional[str] = None) -> ChatOpenAI:
    """ChatOpenAI with only the sampling settings the model accepts.

    No effort set: temperature 0 (models without reasoning settings).
    'none': no reasoning, temperature 0.
    Any other effort: no temperature; reasoning models reject it unless effort is 'none'.
    Timeout and retries: LLM_TIMEOUT_S, API_MAX_RETRIES. The OpenAI SDK retries connection
    errors, timeouts, 408/409/429 and 5xx with backoff; a retry is not a new LLM call.
    """
    if reasoning_effort and reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"Unknown reasoning effort {reasoning_effort!r}; "
                         f"expected one of {sorted(REASONING_EFFORTS)}")
    limits = {"timeout": api_setting("LLM_TIMEOUT_S"),
              "max_retries": api_setting("API_MAX_RETRIES")}
    if reasoning_effort and reasoning_effort != "none":
        return ChatOpenAI(model=model, reasoning_effort=reasoning_effort, **limits)
    extra = {"reasoning_effort": "none"} if reasoning_effort == "none" else {}
    return ChatOpenAI(model=model, temperature=0, **extra, **limits)


def embedding_model(model: str) -> OpenAIEmbeddings:
    """OpenAIEmbeddings with EMBED_TIMEOUT_S and API_MAX_RETRIES (SDK retries, as above)."""
    return OpenAIEmbeddings(model=model, timeout=api_setting("EMBED_TIMEOUT_S"),
                            max_retries=api_setting("API_MAX_RETRIES"))


TextFix = Tuple[str, str]  # (regex pattern, replacement)
CHUNK_MODES = {"sections", "pages"}
CONFIG_NAME = "corpus.yaml"

CONFIG_TEMPLATE = """\
# Corpus config for self_rag.py. Every key is optional; the values shown are the
# defaults. Relative paths are resolved against this folder.

# name: {name}                # defaults to the folder name
# docs_dir: docs              # PDFs to index
# db_dir: index               # Chroma storage + manifest
# collection: {name}          # defaults to name

# What the knowledge base contains; completes "The knowledge base contains ...".
# Leave it out to have it generated from the PDFs at index time.
# kb_description: >-
#   ...

# Regex fixes for PDF extraction artifacts, applied in order before chunking.
# text_fixes:
#   - pattern: '(?<![A-Za-z0-9])I(?=\\d)'
#     replacement: '\u20b9'

# chunking:
#   mode: sections            # sections: split at Markdown headings first; pages: old behaviour
#   size: 300                 # tokens per chunk (longer sections are split further)
#   overlap: 50               # tokens shared between neighbouring chunks

# retriever_k: 4              # chunks returned per query

# Questions for the demo command.
# test_questions:
#   - ...
"""


@dataclass(frozen=True)
class CorpusConfig:
    """Everything that defines one corpus: where it lives, how it is indexed and described."""
    name: str
    docs_dir: Path
    db_dir: Path
    collection: str
    chunk_mode: str = "sections"            # "sections" (headings first) or "pages"
    chunk_size: int = 300                   # tokens per chunk
    chunk_overlap: int = 50                 # tokens shared between neighbouring chunks
    text_fixes: Tuple[TextFix, ...] = ()    # regex fixes for PDF extraction artifacts
    retriever_k: int = 4                    # chunks returned per query
    kb_description: Optional[str] = None    # None: generated at index time
    test_questions: Tuple[str, ...] = ()

    @property
    def manifest(self) -> Path:
        return self.db_dir / "manifest.json"

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CorpusConfig":
        """Read a corpus folder (its corpus.yaml is optional) or a YAML file.

        Relative paths are resolved against the folder of the YAML file.
        A folder without corpus.yaml uses the defaults.
        """
        path = Path(path)
        if path.is_dir():
            base, path = path, path / CONFIG_NAME
            raw = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                   if path.is_file() else {})
        elif path.is_file():
            base = path.parent
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        else:
            raise FileNotFoundError(f"Corpus folder or config not found: {path.resolve()}")

        unknown = set(raw) - {"name", "docs_dir", "db_dir", "collection", "kb_description",
                              "text_fixes", "chunking", "retriever_k", "test_questions"}
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")

        fixes = []
        for fix in raw.get("text_fixes") or []:
            re.compile(fix["pattern"])  # fail early on a bad pattern
            fixes.append((fix["pattern"], fix["replacement"]))

        chunking = raw.get("chunking") or {}
        mode = chunking.get("mode", cls.chunk_mode)
        if mode not in CHUNK_MODES:
            raise ValueError(f"{path}: chunking.mode must be one of {sorted(CHUNK_MODES)}")
        name = raw.get("name") or base.resolve().name
        description = (raw.get("kb_description") or "").strip() or None
        return cls(
            name=name,
            docs_dir=(base / raw.get("docs_dir", "docs")).resolve(),
            db_dir=(base / raw.get("db_dir", "index")).resolve(),
            collection=raw.get("collection", name),
            chunk_mode=mode,
            chunk_size=int(chunking.get("size", cls.chunk_size)),
            chunk_overlap=int(chunking.get("overlap", cls.chunk_overlap)),
            text_fixes=tuple(fixes),
            retriever_k=int(raw.get("retriever_k", cls.retriever_k)),
            kb_description=description,
            test_questions=tuple(raw.get("test_questions") or ()),
        )


# ======================================================================
# LOAD PDFs
# ======================================================================
def clean(text: str, fixes: Tuple[TextFix, ...]) -> str:
    """Strip lone surrogates and apply the corpus's regex fixes in order."""
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    for pattern, replacement in fixes:
        text = re.sub(pattern, replacement, text)
    return text


def load_pdf(path: Path, fixes: Tuple[TextFix, ...]) -> List[Document]:
    """Read one PDF as Markdown, one Document per non-empty page."""
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    return [
        Document(
            page_content=clean(p["text"], fixes),
            metadata={
                "source": path.name,
                # key differs between pymupdf4llm versions; never store None
                # (Chroma silently drops None metadata values)
                "page": p["metadata"].get("page") or p["metadata"].get("page_number") or n,
            },
        )
        for n, p in enumerate(pages, 1)
        if p["text"].strip()
    ]


def chunk_id(doc: Document) -> str:
    """Stable ID from source + page + text, so re-runs skip existing chunks."""
    key = f"{doc.metadata['source']}|{doc.metadata['page']}|{doc.page_content}"
    return hashlib.sha256(key.encode()).hexdigest()


# ======================================================================
# SPLIT (Markdown sections first, then tokens within long sections)
# ======================================================================
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def heading_title(raw: str) -> str:
    """'## **Core Values**' -> 'Core Values'."""
    return re.sub(r"[*_`]+", "", raw).strip()


def split_sections(pages: List[Document]) -> List[Document]:
    """One Document per Markdown section of a PDF; sections may span pages.

    metadata: source, page (where the section starts), section (heading path,
    ' > '-joined), and page_offsets (char offset -> page, for later sub-splits).
    Text before the first heading becomes a section with an empty path.
    """
    sections: List[Document] = []
    path: List[Tuple[int, str]] = []          # (level, title) of the open headings
    lines: List[str] = []
    offsets: List[Tuple[int, int]] = []       # (char offset in section text, page)
    source = pages[0].metadata["source"] if pages else ""

    def close():
        text = "\n".join(lines).strip()
        body = "\n".join(l for l in lines if not HEADING.match(l)).strip()
        if body:                              # skip heading-only sections
            sections.append(Document(page_content=text, metadata={
                "source": source, "page": offsets[0][1],
                "section": " > ".join(t for _, t in path),
                "page_offsets": list(offsets)}))

    for doc in pages:
        page = doc.metadata["page"]
        for line in doc.page_content.splitlines():
            m = HEADING.match(line)
            if m:
                close()
                level = len(m.group(1))
                path = [(l, t) for l, t in path if l < level] + [(level, heading_title(m.group(2)))]
                lines, offsets = [], []
            if not offsets or offsets[-1][1] != page:
                offsets.append((len("\n".join(lines)) + (1 if lines else 0), page))
            lines.append(line)
    close()
    return sections


def chunk_sections(sections: List[Document], splitter) -> List[Document]:
    """Split long sections by tokens; prefix each chunk with its heading path."""
    chunks: List[Document] = []
    for sec in sections:
        meta = {k: v for k, v in sec.metadata.items() if k != "page_offsets"}
        parts = splitter.create_documents([sec.page_content])
        for n, part in enumerate(parts, 1):
            start = part.metadata.get("start_index", 0)
            page = [pg for off, pg in sec.metadata["page_offsets"] if off <= max(start, 0)][-1]
            prefix = f"Section: {meta['section']}\n\n" if meta["section"] else ""
            chunks.append(Document(
                page_content=prefix + part.page_content,
                metadata={**meta, "page": page, "part": n if len(parts) > 1 else 0},
            ))
    return chunks


# ======================================================================
# INDEX
# ======================================================================
class PdfIndex:
    """A persistent Chroma collection kept in sync with a folder of PDFs."""

    def __init__(self, cfg: CorpusConfig, embeddings: Embeddings):
        self.cfg = cfg
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap, add_start_index=True
        )
        self.store = Chroma(
            collection_name=cfg.collection,
            embedding_function=embeddings,
            persist_directory=str(cfg.db_dir),
        )

    def fingerprint(self, path: Path) -> str:
        """Hash of the PDF plus the settings that shape its chunks (changing them re-indexes)."""
        settings = json.dumps(
            {"chunk_mode": self.cfg.chunk_mode,
             "chunk_size": self.cfg.chunk_size, "chunk_overlap": self.cfg.chunk_overlap,
             "text_fixes": [list(f) for f in self.cfg.text_fixes]},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(path.read_bytes() + settings.encode()).hexdigest()

    def chunk(self, pages: List[Document]) -> List[Document]:
        if self.cfg.chunk_mode == "pages":
            return [Document(page_content=d.page_content,
                             metadata={k: v for k, v in d.metadata.items() if k != "start_index"})
                    for d in self.splitter.split_documents(pages)]
        return chunk_sections(split_sections(pages), self.splitter)

    def source_ids(self, source: str) -> List[str]:
        return self.store.get(where={"source": source}, include=[])["ids"]

    def delete(self, ids: List[str]) -> None:
        for start in range(0, len(ids), INDEX_BATCH):
            self.store.delete(ids=ids[start:start + INDEX_BATCH])

    def count(self) -> int:
        return len(self.store.get(include=[])["ids"])

    def fingerprints(self) -> Dict[str, str]:
        """Fingerprint per indexed PDF, as recorded by the last sync()."""
        return self._load_manifest()

    def _load_manifest(self) -> dict:
        m = self.cfg.manifest
        return json.loads(m.read_text()) if m.exists() else {}

    def _save_manifest(self, manifest: dict) -> None:
        self.cfg.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.manifest.write_text(json.dumps(manifest, indent=2))

    def sync(self) -> None:
        """Index new or changed PDFs, skip unchanged ones, drop removed ones.

        For a changed PDF only stale chunks are deleted and only new chunks embedded.
        """
        folder = self.cfg.docs_dir
        if not folder.is_dir():
            raise FileNotFoundError(f"PDF folder not found: {folder.resolve()}")

        manifest = self._load_manifest()
        pdfs = sorted(folder.glob("*.pdf"))

        for name in set(manifest) - {p.name for p in pdfs}:   # PDFs removed from the folder
            self.delete(self.source_ids(name))
            del manifest[name]
            self._save_manifest(manifest)
            log.info("%s: removed from folder, chunks deleted", name)

        for pdf in pdfs:
            fingerprint = self.fingerprint(pdf)
            if manifest.get(pdf.name) == fingerprint:
                log.info("%s: unchanged, skipped", pdf.name)
                continue

            chunks = self.chunk(load_pdf(pdf, self.cfg.text_fixes))

            # De-duplicate identical chunks (Chroma raises DuplicateIDError otherwise)
            unique: Dict[str, Document] = {}
            for c in chunks:
                unique.setdefault(chunk_id(c), c)

            old_ids = set(self.source_ids(pdf.name))
            stale = [i for i in old_ids if i not in unique]
            new_ids = [i for i in unique if i not in old_ids]
            self.delete(stale)
            for start in range(0, len(new_ids), INDEX_BATCH):
                batch = new_ids[start:start + INDEX_BATCH]
                self.store.add_documents([unique[i] for i in batch], ids=batch)

            log.info("%s: %d chunks, %d unique, %d newly embedded, %d stale removed",
                     pdf.name, len(chunks), len(unique), len(new_ids), len(stale))

            manifest[pdf.name] = fingerprint   # save after each PDF to keep progress
            self._save_manifest(manifest)

        log.info("Collection '%s': %d chunks in total", self.cfg.collection, self.count())


def init_corpus(folder: Union[str, Path]) -> Path:
    """Create a corpus folder with an empty docs/ and a commented corpus.yaml."""
    folder = Path(folder)
    config = folder / CONFIG_NAME
    if config.exists():
        raise FileExistsError(f"Already exists, not overwritten: {config.resolve()}")
    (folder / "docs").mkdir(parents=True, exist_ok=True)
    config.write_text(CONFIG_TEMPLATE.format(name=folder.resolve().name), encoding="utf-8")
    return config
