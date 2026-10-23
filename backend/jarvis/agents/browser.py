"""Browser subagent — Playwright-based web automation."""
from __future__ import annotations

import logging

from jarvis.agents.base import BaseAgent, AgentConfig, Task

logger = logging.getLogger(__name__)


BROWSER_PROMPT = """You are a browser automation agent.

Rules:
- Use the browser tool to navigate, inspect, click, fill forms, and extract data.
- Return structured results with source URLs.
- Require confirmation before submitting forms, making purchases, or modifying account settings.
- Never bypass authentication or CAPTCHAs.
- Never expose passwords, cookies, or auth tokens to the model.
"""


def make_browser_agent(conversation_id: str = "") -> BaseAgent:
    return BaseAgent(AgentConfig(
        name="browser",
        description="Web browser automation via Playwright",
        system_prompt=BROWSER_PROMPT,
        tool_allowlist=["browser_navigate", "browser_click", "browser_extract", "browser_screenshot"],
    ), conversation_id)


async def browser_agent_run(agent: BaseAgent, task: Task, cancel_event) -> Any:
    from jarvis.browser.playwright import browser_navigate, browser_click, browser_extract, browser_screenshot
    messages = [
        {"role": "system", "content": agent.config.system_prompt},
        {"role": "user", "content": f"Task: {task.objective}\nContext: {task.context}"},
    ]
    try:
        provider = agent._call_model(messages)
        full = ""
        async for token in provider:
            full += token
        return {"success": True, "response": full, "agent": agent.config.name}
    except Exception as exc:
        return {"success": False, "error": str(exc), "agent": agent.config.name}