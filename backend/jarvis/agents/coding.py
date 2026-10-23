"""Coding subagent — inspects workspaces, debugs, proposes changes."""
from __future__ import annotations

import logging

from jarvis.agents.base import BaseAgent, AgentConfig, Task

logger = logging.getLogger(__name__)


CODING_PROMPT = """You are a coding assistant.

Rules:
- Only read/write files inside the approved workspace.
- Explain code and debug errors.
- Propose changes and review diffs.
- Run only authorized commands.
- Report changed files and test results.
- Never execute arbitrary model-generated commands silently.
- Require confirmation for destructive operations.
"""


def make_coding_agent(conversation_id: str = "") -> BaseAgent:
    return BaseAgent(AgentConfig(
        name="coding",
        description="Code explanation, debugging, and project work",
        system_prompt=CODING_PROMPT,
        tool_allowlist=["file_read", "file_list", "file_search", "file_write", "shell_run"],
    ), conversation_id)


async def coding_agent_run(agent: BaseAgent, task: Task, cancel_event) -> Any:
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