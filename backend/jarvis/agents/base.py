"""Agent base class and task delegation protocol."""
from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from jarvis.config import settings
from jarvis.providers.llm import get_provider
from jarvis.runtime.event_bus import bus
from jarvis.tools.registry import registry
from jarvis.tools.permissions import PermissionLevel

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    task_id: str
    parent_task_id: Optional[str]
    objective: str
    acceptance_criteria: str = ""
    context: dict = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    expected_output: str = "text"
    status: TaskStatus = TaskStatus.QUEUED
    result: Any = None
    error: Optional[str] = None
    evidence: list[dict] = field(default_factory=list)
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "objective": self.objective,
            "acceptance_criteria": self.acceptance_criteria,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "evidence": self.evidence,
        }


@dataclass
class AgentConfig:
    name: str
    description: str
    system_prompt: str
    tool_allowlist: list[str] = field(default_factory=list)
    model: Optional[str] = None
    max_steps: int = 10


class BaseAgent(ABC):
    """Base class for all subagents."""

    def __init__(self, config: AgentConfig, conversation_id: str = ""):
        self.config = config
        self.conversation_id = conversation_id
        self.task_history: list[Task] = []

    @abstractmethod
    async def run(self, task: Task) -> Task:
        ...

    def can_use(self, tool_name: str) -> bool:
        if not self.config.tool_allowlist:
            return True
        return tool_name in self.config.tool_allowlist

    async def _call_model(self, messages: list[dict], tools: list[dict] = None, stream: bool = True):
        provider = get_provider()
        return provider.chat(messages, tools, stream=stream)

    def _emit(self, event_type: str, payload: dict) -> None:
        bus.publish(event_type, payload, self.conversation_id, task.task_id if hasattr(self, "task") else None)

    async def execute_tool(self, name: str, args: dict, confirmed: bool = False):
        if not self.can_use(name):
            return {"success": False, "error": f"Tool not allowed for agent {self.config.name}: {name}"}
        result = await registry.execute(name, args, confirmed)
        return result.to_dict()