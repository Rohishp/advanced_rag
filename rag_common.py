"""Shared PDF loading and persistent Chroma indexing for the RAG scripts."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pymupdf4llm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

INDEX_BATCH = 500  # max IDs per Chroma add/delete call


@dataclass(frozen=True)
class CorpusConfig:
    """Everything that defines one indexed PDF collection."""
    docs_dir: Path
    db_dir: Path
    collection: str
    chunk_size: int                # tokens per chunk
    chunk_overlap: int             # tokens shared between neighbouring chunks
    char_replacements: Dict[str, str] = field(default_factory=dict)  # extraction fixes

    @property
    def manifest(self) -> Path:
        return self.db_dir / "manifest.json"


# ======================================================================
# LOAD PDFs
# ======================================================================
def clean(text: str, replacements: Dict[str, str]) -> str:
    """Strip lone surrogates and fix known extraction artifacts."""
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return text


def load_pdf(path: Path, replacements: Dict[str, str]) -> List[Document]:
    """Read one PDF as Markdown, one Document per non-empty page."""
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    return [
        Document(
            page_content=clean(p["text"], replacements),
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
             "char_replacements": self.cfg.char_replacements},
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

            chunks = self.splitter.split_documents(load_pdf(pdf, self.cfg.char_replacements))

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