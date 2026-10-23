"""Task manager for orchestrator: tracks tasks, dependencies, concurrency, cancellation."""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from jarvis.agents.base import Task, TaskStatus
from jarvis.runtime.event_bus import bus

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskManager:
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.tasks: dict[str, Task] = {}
        self._sem = asyncio.Semaphore(max_concurrent)
        self._cancel_events: dict[str, asyncio.Event] = {}

    def create_task(self, objective: str, parent_task_id: Optional[str] = None, **kwargs) -> Task:
        task = Task(task_id=str(uuid.uuid4()), parent_task_id=parent_task_id, objective=objective, created_at=_now(), **kwargs)
        self.tasks[task.task_id] = task
        self._cancel_events[task.task_id] = asyncio.Event()
        bus.publish("task.queued", task.to_dict(), task.parent_task_id or "", task.task_id)
        return task

    def get(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def cancel(self, task_id: str) -> None:
        task = self.tasks.get(task_id)
        if not task:
            return
        task.status = TaskStatus.CANCELLED
        ev = self._cancel_events.get(task_id)
        if ev:
            ev.set()
        # cancel children
        for t in self.tasks.values():
            if t.parent_task_id == task_id and t.status in (TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.WAITING):
                self.cancel(t.task_id)
        bus.publish("task.cancelled", task.to_dict(), task.parent_task_id or "", task_id)

    async def run(self, task: Task, coro_factory) -> Task:
        """Execute a task with concurrency limit and cancellation support."""
        task.status = TaskStatus.RUNNING
        task.started_at = _now()
        cancel_event = self._cancel_events.get(task.task_id)
        bus.publish("task.started", task.to_dict(), task.parent_task_id or "", task.task_id)
        try:
            async with self._sem:
                if cancel_event and cancel_event.is_set():
                    task.status = TaskStatus.CANCELLED
                    return task
                result = await coro_factory(task, cancel_event)
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = _now()
                bus.publish("task.completed", task.to_dict(), task.parent_task_id or "", task.task_id)
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            bus.publish("task.cancelled", task.to_dict(), task.parent_task_id or "", task.task_id)
        except Exception as exc:
            task.error = str(exc)
            task.status = TaskStatus.FAILED
            task.completed_at = _now()
            bus.publish("task.failed", task.to_dict(), task.parent_task_id or "", task.task_id)
            logger.exception("Task %s failed", task.task_id)
        return task

    def children(self, parent_task_id: str) -> list[Task]:
        return [t for t in self.tasks.values() if t.parent_task_id == parent_task_id]

    def summary(self, parent_task_id: str) -> dict:
        kids = self.children(parent_task_id)
        return {
            "total": len(kids),
            "completed": sum(1 for k in kids if k.status == TaskStatus.COMPLETED),
            "failed": sum(1 for k in kids if k.status == TaskStatus.FAILED),
            "cancelled": sum(1 for k in kids if k.status == TaskStatus.CANCELLED),
            "running": sum(1 for k in kids if k.status == TaskStatus.RUNNING),
        }