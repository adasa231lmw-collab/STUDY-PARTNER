"""Tool registry package — registers all built-in tools."""
from __future__ import annotations

import logging

from jarvis.tools.registry import registry, ToolRegistry, Tool, ToolResult
from jarvis.tools.permissions import PermissionManager, permissions, PermissionLevel, PermissionDecision

logger = logging.getLogger(__name__)


def register_builtin_tools() -> None:
    """Register all built-in tools into the global registry."""
    from jarvis.tools import web, files, shell, calc
    web.register_web_tools(registry)
    files.register_file_tools(registry)
    shell.register_shell_tools(registry)
    calc.register_calc_tools(registry)
    logger.info("Registered %d tools: %s", len(registry.list_names()), registry.list_names())


__all__ = ["registry", "ToolRegistry", "Tool", "ToolResult", "PermissionManager", "permissions", "PermissionLevel", "PermissionDecision", "register_builtin_tools"]