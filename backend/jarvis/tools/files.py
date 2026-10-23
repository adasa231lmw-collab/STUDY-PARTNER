"""File system tools restricted to approved workspaces."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from jarvis.config import settings
from jarvis.tools.permissions import PermissionLevel

logger = logging.getLogger(__name__)


def _safe_path(path: str) -> Path:
    p = Path(path).resolve()
    allowed = Path(settings.ALLOWED_WORKSPACE).resolve()
    try:
        p.relative_to(allowed)
    except ValueError:
        raise ValueError(f"Path outside allowed workspace: {path}")
    return p


def read_file(args: dict) -> dict:
    path = _safe_path(args.get("path", ""))
    if not path.exists():
        return {"success": False, "error": "File not found", "path": str(path)}
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        return {"success": True, "path": str(path), "content": content[:20000], "truncated": len(content) > 20000}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def list_files(args: dict) -> dict:
    path = _safe_path(args.get("path", "."))
    if not path.exists() or not path.is_dir():
        return {"success": False, "error": "Directory not found", "path": str(path)}
    try:
        entries = []
        for entry in sorted(path.iterdir()):
            entries.append({"name": entry.name, "is_dir": entry.is_dir(), "size": entry.stat().st_size if entry.is_file() else 0})
        return {"success": True, "path": str(path), "entries": entries[:200]}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def write_file(args: dict) -> dict:
    path = _safe_path(args.get("path", ""))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args.get("content", ""), encoding="utf-8")
        return {"success": True, "path": str(path), "bytes": len(args.get("content", ""))}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def search_files(args: dict) -> dict:
    query = args.get("query", "").lower()
    root = _safe_path(args.get("path", "."))
    matches = []
    try:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if query in fn.lower():
                    matches.append(str(Path(dirpath) / fn))
                if len(matches) >= 50:
                    break
            if len(matches) >= 50:
                break
        return {"success": True, "matches": matches, "count": len(matches)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def register_file_tools(registry) -> None:
    from jarvis.tools.registry import Tool
    registry.register(Tool(
        name="file_read",
        description="Read a text file from the approved workspace.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=read_file,
        level=PermissionLevel.READ,
    ))
    registry.register(Tool(
        name="file_list",
        description="List files and folders in a directory.",
        parameters={"type": "object", "properties": {"path": {"type": "string", "default": "."}}},
        handler=list_files,
        level=PermissionLevel.READ,
    ))
    registry.register(Tool(
        name="file_write",
        description="Write text to a file in the approved workspace.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
        handler=write_file,
        level=PermissionLevel.WRITE,
    ))
    registry.register(Tool(
        name="file_search",
        description="Search for files by name pattern in the workspace.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string", "default": "."}}},
        handler=search_files,
        level=PermissionLevel.READ,
    ))