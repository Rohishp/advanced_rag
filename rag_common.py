"""Shared PDF loading and persistent Chroma indexing for the RAG scripts."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pymupdf4llm
import yaml
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

INDEX_BATCH = 500  # max IDs per Chroma add/delete call


TextFix = Tuple[str, str]  # (regex pattern, replacement)
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
#   size: 300                 # tokens per chunk
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
        name = raw.get("name") or base.resolve().name
        description = (raw.get("kb_description") or "").strip() or None
        return cls(
            name=name,
            docs_dir=(base / raw.get("docs_dir", "docs")).resolve(),
            db_dir=(base / raw.get("db_dir", "index")).resolve(),
            collection=raw.get("collection", name),
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
# INDEX
# ======================================================================
class PdfIndex:
    """A persistent Chroma collection kept in sync with a folder of PDFs."""

    def __init__(self, cfg: CorpusConfig, embeddings: Embeddings):
        self.cfg = cfg
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
        )
        self.store = Chroma(
            collection_name=cfg.collection,
            embedding_function=embeddings,
            persist_directory=str(cfg.db_dir),
        )

    def fingerprint(self, path: Path) -> str:
        """Hash of the PDF plus the settings that shape its chunks (changing them re-indexes)."""
        settings = json.dumps(
            {"chunk_size": self.cfg.chunk_size, "chunk_overlap": self.cfg.chunk_overlap,
             "text_fixes": [list(f) for f in self.cfg.text_fixes]},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(path.read_bytes() + settings.encode()).hexdigest()

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
            print(f"{name}: removed from folder, chunks deleted")

        for pdf in pdfs:
            fingerprint = self.fingerprint(pdf)
            if manifest.get(pdf.name) == fingerprint:
                print(f"{pdf.name}: unchanged, skipped")
                continue

            chunks = self.splitter.split_documents(load_pdf(pdf, self.cfg.text_fixes))

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

            print(f"{pdf.name}: {len(chunks)} chunks, {len(unique)} unique, "
                  f"{len(new_ids)} newly embedded, {len(stale)} stale removed")

            manifest[pdf.name] = fingerprint   # save after each PDF to keep progress
            self._save_manifest(manifest)

        print(f"Collection '{self.cfg.collection}': {self.count()} chunks in total")


def init_corpus(folder: Union[str, Path]) -> Path:
    """Create a corpus folder with an empty docs/ and a commented corpus.yaml."""
    folder = Path(folder)
    config = folder / CONFIG_NAME
    if config.exists():
        raise FileExistsError(f"Already exists, not overwritten: {config.resolve()}")
    (folder / "docs").mkdir(parents=True, exist_ok=True)
    config.write_text(CONFIG_TEMPLATE.format(name=folder.resolve().name), encoding="utf-8")
    return config
