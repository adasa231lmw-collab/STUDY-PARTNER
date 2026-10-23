"""Local vector store using SQLite + numpy cosine search."""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from jarvis.config import settings

logger = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class VectorStore:
    """SQLite-backed vector store with cosine similarity search."""

    def __init__(self, path: str = ""):
        self.path = Path(path or settings.VECTOR_STORE_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    embedding TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON documents(source)")
            self._conn.commit()

    def add(self, source: str, text: str, embedding: list[float], metadata: dict = None, chunk_index: int = 0) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO documents (source, chunk_index, text, metadata, embedding) VALUES (?, ?, ?, ?, ?)",
                (source, chunk_index, text, json.dumps(metadata or {}), json.dumps(embedding)),
            )
            self._conn.commit()
            return cur.lastrowid

    def search(self, query_embedding: list[float], top_k: int = 5, source_filter: str = None) -> list[dict]:
        with self._lock:
            if source_filter:
                rows = self._conn.execute("SELECT * FROM documents WHERE source = ?", (source_filter,)).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM documents").fetchall()
        results = []
        for row in rows:
            doc_id, source, chunk_index, text, metadata, embedding, created_at = row
            emb = json.loads(embedding)
            score = _cosine(query_embedding, emb)
            results.append({
                "id": doc_id,
                "source": source,
                "chunk_index": chunk_index,
                "text": text,
                "metadata": json.loads(metadata) if metadata else {},
                "score": round(score, 4),
                "created_at": created_at,
            })
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def delete_source(self, source: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM documents WHERE source = ?", (source,))
            self._conn.commit()
            return cur.rowcount

    def list_sources(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT DISTINCT source FROM documents").fetchall()
            return [r[0] for r in rows]

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM documents")
            self._conn.commit()