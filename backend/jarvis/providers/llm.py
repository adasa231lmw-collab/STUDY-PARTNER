"""LLM provider abstraction supporting Ollama and OpenAI-compatible APIs."""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Optional

import httpx

from jarvis.config import settings

logger = logging.getLogger(__name__)


class LLMProvider:
    """Base interface for chat completions with streaming support."""

    def __init__(self, model: str, **kwargs: Any):
        self.model = model
        self.extra = kwargs

    async def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        stream: bool = True,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        raise NotImplementedError

    async def chat_complete(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> dict:
        chunks: list[str] = []
        async for token in self.chat(messages, tools, stream=True, **kwargs):
            chunks.append(token)
        return {"content": "".join(chunks)}


class OllamaProvider(LLMProvider):
    """Ollama-compatible local model provider."""

    def __init__(self, model: str, host: Optional[str] = None, **kwargs: Any):
        super().__init__(model, **kwargs)
        self.host = host or settings.OLLAMA_HOST
        self.client = httpx.AsyncClient(base_url=self.host, timeout=settings.OLLAMA_TIMEOUT)

    async def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        stream: bool = True,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_gpu": 1 if settings.OLLAMA_USE_GPU else 0,
            },
        }
        if tools:
            payload["tools"] = tools
        payload["options"].update(kwargs.pop("options", {}))
        payload.update(kwargs)

        try:
            async with self.client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    raise RuntimeError(f"Ollama error {resp.status_code}: {text.decode()[:300]}")
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = data.get("message", {})
                    content = msg.get("content", "")
                    if content:
                        yield content
                    if data.get("done"):
                        # emit tool calls if present
                        tc = msg.get("tool_calls")
                        if tc:
                            yield json.dumps({"__tool_calls__": tc})
                        break
        except httpx.ConnectError as exc:
            raise RuntimeError(f"Cannot connect to Ollama at {self.host}: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

    async def health(self) -> bool:
        try:
            r = await self.client.get("/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    async def embed(self, text: str) -> list[float]:
        r = await self.client.post(
            "/api/embeddings",
            json={"model": settings.OLLAMA_EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("embedding", [])


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible chat provider."""

    def __init__(self, model: str, api_key: str = "", base_url: str = "", **kwargs: Any):
        super().__init__(model, **kwargs)
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url or settings.OPENAI_BASE_URL or "https://api.openai.com/v1"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120,
        )

    async def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        stream: bool = True,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": stream}
        if tools:
            payload["tools"] = tools
        payload.update(kwargs)
        try:
            async with self.client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    raise RuntimeError(f"OpenAI error {resp.status_code}: {text.decode()[:300]}")
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choice = (obj.get("choices") or [{}])[0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
        except httpx.ConnectError as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc


def get_provider() -> LLMProvider:
    """Return the configured LLM provider."""
    if settings.OPENAI_API_KEY and not settings.OPENAI_BASE_URL:
        return OpenAIProvider(settings.OPENAI_MODEL)
    if settings.OPENAI_BASE_URL and settings.OPENAI_API_KEY:
        return OpenAIProvider(settings.OPENAI_MODEL, settings.OPENAI_API_KEY, settings.OPENAI_BASE_URL)
    return OllamaProvider(settings.OLLAMA_MODEL)


def get_embed_provider() -> Optional[LLMProvider]:
    """Return an embedding-capable provider (Ollama by default)."""
    try:
        return OllamaProvider(settings.OLLAMA_EMBED_MODEL)
    except Exception:
        return None