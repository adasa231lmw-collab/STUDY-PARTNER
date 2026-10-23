"""In-process event bus supporting SSE consumers and WebSocket subscribers."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class Event:
    event_id: str
    conversation_id: str
    task_id: Optional[str]
    timestamp: str
    event_type: str
    payload: dict = field(default_factory=dict)

    def to_sse(self) -> str:
        return f"data: {json.dumps(self.to_dict(), ensure_ascii=False)}\n\n"

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "conversation_id": self.conversation_id,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "event": self.event_type,
            "payload": self.payload,
        }


class EventBus:
    """Pub/sub event bus with SSE streaming support."""

    def __init__(self, history_size: int = 500):
        self.history: list[Event] = []
        self.history_size = history_size
        self._subscribers: list[asyncio.Queue] = []
        self._lock = asyncio.Lock()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def publish(self, event_type: str, payload: dict, conversation_id: str = "", task_id: Optional[str] = None) -> Event:
        ev = Event(
            event_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            task_id=task_id,
            timestamp=self._now(),
            event_type=event_type,
            payload=payload,
        )
        self.history.append(ev)
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]
        # non-blocking dispatch
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(self._dispatch(ev), loop)
        except RuntimeError:
            pass
        return ev

    async def _dispatch(self, ev: Event) -> None:
        async with self._lock:
            dead: list[asyncio.Queue] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(ev)
                except asyncio.QueueFull:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)

    async def subscribe(self, last_event_id: Optional[str] = None) -> AsyncIterator[Event]:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._subscribers.append(q)
        # replay missed events
        if last_event_id:
            for ev in self.history:
                if ev.event_id == last_event_id:
                    break
                yield ev
        try:
            while True:
                ev = await q.get()
                yield ev
        except asyncio.CancelledError:
            pass
        finally:
            async with self._lock:
                if q in self._subscribers:
                    self._subscribers.remove(q)

    def get_history(self, conversation_id: Optional[str] = None, limit: int = 100) -> list[Event]:
        items = self.history
        if conversation_id:
            items = [e for e in items if e.conversation_id == conversation_id]
        return items[-limit:]


bus = EventBus()