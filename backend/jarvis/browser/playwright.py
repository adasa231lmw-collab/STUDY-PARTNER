"""Playwright-based browser automation tool harness."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from jarvis.config import settings

logger = logging.getLogger(__name__)

_browser = None
_context = None
_page = None


async def _ensure_browser():
    global _browser, _context, _page
    if _browser is None:
        from playwright.async_api import async_playwright
        _pw = await async_playwright().start()
        _browser = await _pw.chromium.launch(headless=True)
        _context = await _browser.new_context()
        _page = await _context.new_page()
    return _page


async def browser_navigate(args: dict) -> dict:
    url = args.get("url", "")
    if not url:
        return {"success": False, "error": "URL required"}
    try:
        page = await _ensure_browser()
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        title = await page.title()
        text = await page.inner_text("body")
        return {"success": True, "url": url, "title": title, "text": text[:5000]}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def browser_click(args: dict) -> dict:
    selector = args.get("selector", "")
    try:
        page = await _ensure_browser()
        await page.click(selector, timeout=10000)
        return {"success": True, "clicked": selector}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def browser_extract(args: dict) -> dict:
    selector = args.get("selector", "body")
    try:
        page = await _ensure_browser()
        text = await page.inner_text(selector)
        return {"success": True, "text": text[:5000]}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def browser_screenshot(args: dict) -> dict:
    try:
        page = await _ensure_browser()
        path = args.get("path", "screenshot.png")
        await page.screenshot(path=path)
        return {"success": True, "path": path}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def browser_close() -> dict:
    global _browser, _context, _page
    try:
        if _browser:
            await _browser.close()
        _browser = _context = _page = None
        return {"success": True}
    except Exception as exc:
        return {"success": False, "error": str(exc)}