"""Memory and knowledge subagent — searches approved memories and documents."""
from __future__ import annotations

import logging

from jarvis.agents.base import BaseAgent, AgentConfig, Task

logger = logging.getLogger(__name__)


MEMORY_PROMPT = """You are a memory and knowledge agent.

Rules:
- Search approved memories, notes, and indexed documents.
- Identify relevant context for other agents.
- Suggest useful memory updates but never store everything silently.
- Never access passwords, secrets, or sensitive data.
- Respect user exclusions.
"""


def make_memory_agent(conversation_id: str = "") -> BaseAgent:
    return BaseAgent(AgentConfig(
        name="memory",
        description="Memory and knowledge retrieval",
        system_prompt=MEMORY_PROMPT,
        tool_allowlist=["file_read", "file_search"],
    ), conversation_id)


async def memory_agent_run(agent: BaseAgent, task: Task, cancel_event) -> Any:
    from jarvis.memory.rag import rag
    from jarvis.db.memory import MemoryStore
    from jarvis.config import settings
    chunks = await rag.query(task.objective, top_k=5)
    store = MemoryStore(str(settings.MEMORY_STORE_PATH))
    memories = store.search(task.objective)
    context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
    mem_text = "\n\n".join(f"[{m.get('category')}]: {m.get('text', '')}" for m in memories[:5])
    messages = [
        {"role": "system", "content": agent.config.system_prompt},
        {"role": "user", "content": f"Retrieved context:\n{context}\n\nMemories:\n{mem_text}\n\nTask: {task.objective}"},
    ]
    try:
        provider = agent._call_model(messages)
        full = ""
        async for token in provider:
            full += token
        return {"success": True, "response": full, "sources": [c["source"] for c in chunks], "agent": agent.config.name}
    except Exception as exc:
        return {"success": False, "error": str(exc), "agent": agent.config.name}