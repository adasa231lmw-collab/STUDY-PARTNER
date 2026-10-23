"""Persistent user memory (preferences, projects, notes) with JSON storage."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from jarvis.config import settings

logger = logging.getLogger(__name__)


class MemoryStore:
    """JSON-backed long-term memory store."""

    def __init__(self, path: str = ""):
        self.path = Path(path or settings.MEMORY_STORE_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data = {"preferences": [], "projects": [], "notes": [], "facts": [], "exclusions": []}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Memory load failed: %s", exc)

    def _save(self) -> None:
        with self._lock:
            self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")

    def add(self, category: str, item: dict) -> dict:
        item.setdefault("id", f"{category}_{len(self._data.get(category, []))}")
        item.setdefault("enabled", True)
        self._data.setdefault(category, []).append(item)
        self._save()
        return item

    def get(self, category: str) -> list[dict]:
        return [i for i in self._data.get(category, []) if i.get("enabled", True)]

    def delete(self, category: str, item_id: str) -> bool:
        items = self._data.get(category, [])
        for i in items:
            if i.get("id") == item_id:
                i["enabled"] = False
                self._save()
                return True
        return False

    def update(self, category: str, item_id: str, updates: dict) -> bool:
        for i in self._data.get(category, []):
            if i.get("id") == item_id:
                i.update(updates)
                self._save()
                return True
        return False

    def search(self, query: str) -> list[dict]:
        q = query.lower()
        results = []
        for cat, items in self._data.items():
            for i in items:
                if not i.get("enabled", True):
                    continue
                hay = json.dumps(i, ensure_ascii=False).lower()
                if q in hay:
                    results.append({"category": cat, **i})
        return results

    def all(self) -> dict:
        return {k: [i for i in v if i.get("enabled", True)] for k, v in self._data.items()}

    def clear(self) -> None:
        self._data = {"preferences": [], "projects": [], "notes": [], "facts": [], "exclusions": []}
        self._save()