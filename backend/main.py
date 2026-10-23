"""JARVIS backend entry point (minimal vertical slice)."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from jarvis.config import settings
from jarvis.providers.llm import get_provider
from jarvis.utils.logging import setup_logging

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = get_provider()
    yield
    await app.state.llm.client.aclose()


app = FastAPI(title="JARVIS", lifespan=lifespan)


@app.get("/health")
async def health():
    ok = await app.state.llm.health()
    return {"status": "ok" if ok else "degraded", "model": settings.OLLAMA_MODEL, "ollama": ok}


@app.post("/chat")
async def chat(payload: dict):
    messages = payload.get("messages") or [{"role": "user", "content": payload.get("message", "")}]
    chunks = []
    async for t in app.state.llm.chat(messages, stream=True):
        chunks.append(t)
    return {"reply": "".join(chunks), "model": settings.OLLAMA_MODEL}


@app.post("/embed")
async def embed(payload: dict):
    text = payload.get("text", "")
    vec = await app.state.llm.embed(text)
    return {"dim": len(vec), "embedding": vec[:8]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=False)
