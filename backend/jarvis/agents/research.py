"""Research subagent — searches web, extracts evidence, returns citations."""
from __future__ import annotations

import logging
from typing import Any

from jarvis.agents.base import BaseAgent, AgentConfig, Task, TaskStatus
from jarvis.tools import web as web_tools
from jarvis.tools.registry import registry

logger = logging.getLogger(__name__)


RESEARCH_PROMPT = """You are a research agent. Your job is to investigate a topic and return well-cited findings.

Rules:
- Use web_search to find sources, then web_fetch to read the most promising pages.
- Always preserve source URLs and titles.
- Distinguish verified information from assumptions.
- Note disagreements between sources.
- Never fabricate citations or claim to have searched when you did not.
- Return a structured summary with findings, sources, and confidence levels.

Output format:
FINDINGS:
- [finding] (source: URL)

SOURCES:
1. Title - URL

CONFIDENCE: [high/medium/low]
NOTES: [any caveats]
"""


def make_research_agent(conversation_id: str = "") -> BaseAgent:
    return BaseAgent(AgentConfig(
        name="research",
        description="Web research with citations and source comparison",
        system_prompt=RESEARCH_PROMPT,
        tool_allowlist=["web_search", "web_fetch"],
    ), conversation_id)


async def research_agent_run(agent: BaseAgent, task: Task, cancel_event) -> Any:
    messages = [
        {"role": "system", "content": agent.config.system_prompt},
        {"role": "user", "content": f"Objective: {task.objective}\n\nAcceptance criteria: {task.acceptance_criteria}"},
    ]
    tools = [t for t in registry.list_schemas() if t["function"]["name"] in agent.config.tool_allowlist]
    findings = []
    sources = []
    steps = 0
    while steps < agent.config.max_steps and not cancel_event.is_set():
        steps += 1
        try:
            provider = agent._call_model(messages, tools)
            full = ""
            tool_calls = []
            async for token in provider:
                if token.startswith("__tool_calls__"):
                    import json
                    tool_calls = json.loads(token)["__tool_calls__"]
                else:
                    full += token
            if tool_calls:
                messages.append({"role": "assistant", "content": full or None, "tool_calls": tool_calls})
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name")
                    args = fn.get("arguments", "{}")
                    import json
                    try:
                        args = json.loads(args) if isinstance(args, str) else args
                    except Exception:
                        args = {}
                    result = await agent.execute_tool(name, args)
                    messages.append({"role": "tool", "tool_call_id": tc.get("id", str(steps)), "content": str(result)})
            else:
                messages.append({"role": "assistant", "content": full})
                findings.append(full)
                break
        except Exception as exc:
            return {"success": False, "error": str(exc), "findings": findings, "sources": sources}
    return {"success": True, "findings": findings, "sources": sources, "steps": steps, "agent": agent.config.name}