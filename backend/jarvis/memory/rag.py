"""RAG service: document ingestion, embedding, retrieval with citations."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from jarvis.config import settings
from jarvis.db.vector import VectorStore
from jarvis.providers.llm import get_embed_provider
from jarvis.runtime.event_bus import bus

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def extract_text(path: str) -> str:
    """Extract text from supported document formats."""
    p = Path(path)
    suffix = p.suffix.lower()
    try:
        if suffix in {".txt", ".md", ".json", ".csv", ".log"}:
            return p.read_text(encoding="utf-8", errors="replace")
        if suffix == ".pdf":
            import pdfplumber
            text_parts = []
            with pdfplumber.open(str(p)) as pdf:
                for i, page in enumerate(pdf.pages):
                    t = page.extract_text() or ""
                    text_parts.append(f"[Page {i+1}]\n{t}")
            return "\n\n".join(text_parts)
        if suffix == ".docx":
            from docx import Document
            doc = Document(str(p))
            return "\n\n".join(par.text for par in doc.paragraphs)
        if suffix == ".html":
            import trafilatura
            raw = p.read_text(encoding="utf-8", errors="replace")
            return trafilatura.extract(raw) or raw
    except Exception as exc:
        logger.warning("Extract failed for %s: %s", path, exc)
    return ""


class RAGService:
    def __init__(self):
        self.store = VectorStore(str(settings.VECTOR_STORE_PATH))
        self._embed_provider = None

    async def _embed(self, text: str) -> list[float]:
        if self._embed_provider is None:
            self._embed_provider = get_embed_provider()
        if self._embed_provider is None:
            return []
        try:
            return await self._embed_provider.embed(text)
        except Exception as exc:
            logger.error("Embed failed: %s", exc)
            return []

    async def ingest(self, path: str, source_label: str = None) -> dict:
        """Ingest a document into the vector store."""
        p = Path(path)
        if not p.exists():
            return {"success": False, "error": "File not found", "path": path}
        text = extract_text(str(p))
        if not text.strip():
            return {"success": False, "error": "No text extracted", "path": path}
        source = source_label or p.name
        # remove old chunks for this source
        self.store.delete_source(source)
        chunks = chunk_text(text)
        added = 0
        for i, chunk in enumerate(chunks):
            emb = await self._embed(chunk)
            if not emb:
                continue
            self.store.add(source, chunk, emb, {"chunk_index": i, "path": str(p)}, chunk_index=i)
            added += 1
        bus.publish("research.source_found", {"source": source, "chunks": added, "path": str(p)})
        return {"success": True, "source": source, "chunks": added, "text_length": len(text)}

    async def query(self, question: str, top_k: int = None, source_filter: str = None) -> list[dict]:
        """Retrieve relevant chunks for a question."""
        k = top_k or settings.RAG_TOP_K
        emb = await self._embed(question)
        if not emb:
            return []
        return self.store.search(emb, top_k=k, source_filter=source_filter)

    def list_sources(self) -> list[str]:
        return self.store.list_sources()

    def delete_source(self, source: str) -> int:
        return self.store.delete_source(source)

    def clear(self) -> None:
        self.store.clear()


rag = RAGService()