"""Verification subagent — independently checks facts, citations, code, and task completion."""
from __future__ import annotations

import logging

from jarvis.agents.base import BaseAgent, AgentConfig, Task

logger = logging.getLogger(__name__)


VERIFY_PROMPT = """You are a verification agent.

Rules:
- Independently check factual claims, citations, calculations, code changes, and task completion.
- Identify missing evidence and inconsistencies.
- Never simply approve another agent's answer without examining available evidence.
- Flag anything that cannot be verified.
- Be skeptical: assume claims are unverified until evidence supports them.
"""


def make_verification_agent(conversation_id: str = "") -> BaseAgent:
    return BaseAgent(AgentConfig(
        name="verification",
        description="Independent verification of facts, citations, and task completion",
        system_prompt=VERIFY_PROMPT,
        tool_allowlist=["web_search", "web_fetch", "file_read", "shell_run", "calculator"],
    ), conversation_id)


async def verification_agent_run(agent: BaseAgent, task: Task, cancel_event) -> Any:
    messages = [
        {"role": "system", "content": agent.config.system_prompt},
        {"role": "user", "content": f"Verify the following:\n{task.objective}\n\nContext: {task.context}"},
    ]
    try:
        provider = agent._call_model(messages, stream=False)
        result = await provider.__anext__()
        return {"success": True, "response": result, "agent": agent.config.name}
    except Exception as exc:
        return {"success": False, "error": str(exc), "agent": agent.config.name}