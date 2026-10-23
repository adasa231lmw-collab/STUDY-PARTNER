"""Controlled shell execution tool."""
from __future__ import annotations

import asyncio
import logging
import shlex
from typing import Any

from jarvis.config import settings
from jarvis.tools.permissions import PermissionLevel

logger = logging.getLogger(__name__)


async def run_command(args: dict) -> dict:
    if not settings.ALLOWED_SHELL:
        return {"success": False, "error": "Shell execution is disabled by configuration"}
    cmd = args.get("command", "")
    timeout = min(args.get("timeout", 30), 120)
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"success": False, "error": f"Command timed out after {timeout}s"}
        return {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": stdout.decode("utf-8", errors="replace")[:10000],
            "stderr": stderr.decode("utf-8", errors="replace")[:5000],
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def register_shell_tools(registry) -> None:
    from jarvis.tools.registry import Tool
    registry.register(Tool(
        name="shell_run",
        description="Run a shell command in the approved workspace. Disabled by default.",
        parameters={"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer", "default": 30}}},
        handler=run_command,
        level=PermissionLevel.EXECUTE,
    ))