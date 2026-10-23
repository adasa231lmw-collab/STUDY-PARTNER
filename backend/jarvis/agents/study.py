"""Study subagent — explains concepts, solves problems, uses RAG."""
from __future__ import annotations

import logging

from jarvis.agents.base import BaseAgent, AgentConfig, Task

logger = logging.getLogger(__name__)


STUDY_PROMPT = """You are a study companion. Explain concepts clearly and simply.

Rules:
- Adapt explanations to the user's requested difficulty level.
- Work through problems step by step.
- Offer hints before revealing answers when asked.
- Distinguish textbook facts from your own additional explanations.
- Use the uploaded study material (retrieved via RAG) when relevant.
- Generate revision notes, quizzes, and flashcards on request.
- Never present assumptions as verified facts.

Curriculum: CBSE/NCERT supported when requested.
"""


def make_study_agent(conversation_id: str = "") -> BaseAgent:
    return BaseAgent(AgentConfig(
        name="study",
        description="Study companion for explanations, problems, and revision",
        system_prompt=STUDY_PROMPT,
        tool_allowlist=["file_read", "file_search"],
    ), conversation_id)


async def study_agent_run(agent: BaseAgent, task: Task, cancel_event) -> Any:
    from jarvis.memory.rag import rag
    context_chunks = await rag.query(task.objective, top_k=5)
    context = "\n\n".join(f"[{c['source']} p.{c['chunk_index']}]\n{c['text']}" for c in context_chunks) if context_chunks else ""
    messages = [
        {"role": "system", "content": agent.config.system_prompt},
        {"role": "user", "content": f"Study material context:\n{context}\n\nTask: {task.objective}"},
    ]
    try:
        provider = agent._call_model(messages)
        full = ""
        async for token in provider:
            full += token
        return {"success": True, "response": full, "sources_used": [c["source"] for c in context_chunks], "agent": agent.config.name}
    except Exception as exc:
        return {"success": False, "error": str(exc), "agent": agent.config.name}