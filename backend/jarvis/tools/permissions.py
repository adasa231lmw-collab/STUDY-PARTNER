"""Permission manager — enforces least-privilege tool access."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from jarvis.config import settings

logger = logging.getLogger(__name__)


class PermissionLevel(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    DESTRUCTIVE = "destructive"


@dataclass
class PermissionDecision:
    allowed: bool
    requires_confirmation: bool
    reason: str = ""
    level: PermissionLevel = PermissionLevel.READ


class PermissionManager:
    """Central permission enforcement for all tool calls."""

    def __init__(self):
        self._allowed_workspaces: list[Path] = [Path(settings.ALLOWED_WORKSPACE).resolve()]
        self._allow_shell = settings.ALLOWED_SHELL
        self._allow_browser = settings.ALLOWED_BROWSER
        self._confirm_destructive = settings.CONFIRM_DESTRUCTIVE
        self._audit: list[dict] = []

    def add_workspace(self, path: str) -> None:
        self._allowed_workspaces.append(Path(path).resolve())

    def check_path(self, path: str) -> bool:
        try:
            p = Path(path).resolve()
        except Exception:
            return False
        for ws in self._allowed_workspaces:
            try:
                p.relative_to(ws)
                return True
            except ValueError:
                continue
        return False

    def decide(
        self,
        tool: str,
        args: dict,
        level: PermissionLevel = PermissionLevel.READ,
    ) -> PermissionDecision:
        self._audit.append({"tool": tool, "args": {k: v for k, v in args.items() if k not in {"password", "token", "key", "secret"}}, "level": level.value})

        if level == PermissionLevel.EXECUTE and not self._allow_shell:
            return PermissionDecision(False, False, "Shell execution is disabled", level)
        if level == PermissionLevel.NETWORK:
            return PermissionDecision(True, False, "Network allowed", level)
        if level == PermissionLevel.DESTRUCTIVE and self._confirm_destructive:
            return PermissionDecision(True, True, "Destructive action requires confirmation", level)
        if level == PermissionLevel.WRITE:
            path = args.get("path") or args.get("file")
            if path and not self.check_path(path):
                return PermissionDecision(False, False, f"Path outside allowed workspace: {path}", level)
        return PermissionDecision(True, False, "Allowed", level)

    def audit(self) -> list[dict]:
        return list(self._audit)

    def clear_audit(self) -> None:
        self._audit.clear()


permissions = PermissionManager()