"""Planning subagent — breaks goals into tasks, tracks progress."""
from __future__ import annotations

import logging

from jarvis.agents.base import BaseAgent, AgentConfig, Task

logger = logging.getLogger(__name__)


PLANNING_PROMPT = """You are a planning agent.

Rules:
- Break large goals into manageable tasks.
- Identify dependencies, milestones, and blockers.
- Track progress and propose next steps.
- Return a clear task list with priorities and estimated effort.
"""


def make_planning_agent(conversation_id: str = "") -> BaseAgent:
    return BaseAgent(AgentConfig(
        name="planning",
        description="Goal decomposition and progress tracking",
        system_prompt=PLANNING_PROMPT,
        tool_allowlist=["file_read", "file_write"],
    ), conversation_id)


async def planning_agent_run(agent: BaseAgent, task: Task, cancel_event) -> Any:
    messages = [
        {"role": "system", "content": agent.config.system_prompt},
        {"role": "user", "content": f"Goal: {task.objective}\nContext: {task.context}"},
    ]
    try:
        provider = agent._call_model(messages)
        full = ""
        async for token in provider:
            full += token
        return {"success": True, "response": full, "agent": agent.config.name}
    except Exception as exc:
        return {"success": False, "error": str(exc), "agent": agent.config.name}