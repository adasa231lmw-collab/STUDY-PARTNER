"""Utility helpers."""
from __future__ import annotations

import asyncio
import functools
import logging
import os
import re
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def run_async(coro):
    """Run a coroutine in the current event loop or a new one."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return coro
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def strip_markdown(text: str) -> str:
    """Remove markdown syntax for TTS-friendly output."""
    if not text:
        return ""
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_~#]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate(text: str, max_len: int = 4000) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def is_destructive(action: str, args: dict) -> bool:
    destructive_keywords = ["delete", "remove", "rm ", "format", "shutdown", "reboot", "kill", "install", "uninstall"]
    cmd = str(args.get("command", "")).lower()
    return any(k in cmd for k in destructive_keywords)


def safe_json(text: str, default: Any = None) -> Any:
    import json
    try:
        return json.loads(text)
    except Exception:
        return default