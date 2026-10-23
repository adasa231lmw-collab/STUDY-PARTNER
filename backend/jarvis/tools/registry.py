"""Tool registry with schema validation and permission-aware execution."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from jarvis.tools.permissions import PermissionLevel, permissions

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    success: bool
    output: Any = None
    error: Optional[str] = None
    requires_confirmation: bool = False

    def to_dict(self) -> dict:
        return {"success": self.success, "output": self.output, "error": self.error, "requires_confirmation": self.requires_confirmation}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], Any]
    level: PermissionLevel = PermissionLevel.READ
    requires_confirmation: bool = False

    def to_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(self, args: dict, confirmed: bool = False) -> ToolResult:
        decision = permissions.decide(self.name, args, self.level)
        if not decision.allowed:
            return ToolResult(False, error=decision.reason)
        if decision.requires_confirmation and not confirmed:
            return ToolResult(False, error="Confirmation required", requires_confirmation=True)
        try:
            result = self.handler(args)
            if hasattr(result, "__await__"):
                result = await result
            return ToolResult(True, output=result)
        except Exception as exc:
            logger.exception("Tool %s failed", self.name)
            return ToolResult(False, error=str(exc))


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_schemas(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    async def execute(self, name: str, args: dict, confirmed: bool = False) -> ToolResult:
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(False, error=f"Unknown tool: {name}")
        return await tool.execute(args, confirmed)


registry = ToolRegistry()