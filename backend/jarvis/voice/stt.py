"""Speech-to-text providers: Vosk (local) with fallbacks."""
from __future__ import annotations

import asyncio
import io
import logging
import os
import wave
from pathlib import Path
from typing import Optional

from jarvis.config import settings

logger = logging.getLogger(__name__)


def _write_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)
    return buf.getvalue()


class VoskSTT:
    """Local Vosk speech recognition."""

    def __init__(self, model_path: str = ""):
        self.model_path = model_path or settings.VOSK_MODEL_PATH
        self._model = None
        self._rec = None

    def _ensure_model(self):
        if self._model is not None:
            return True
        try:
            from vosk import Model, KaldiRecognizer
            if not Path(self.model_path).exists():
                logger.warning("Vosk model not found at %s", self.model_path)
                return False
            self._model = Model(self.model_path)
            return True
        except Exception as exc:
            logger.error("Vosk model load failed: %s", exc)
            return False

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        if not self._ensure_model():
            return ""
        from vosk import KaldiRecognizer
        wav = _write_wav(audio_bytes, sample_rate)
        rec = KaldiRecognizer(self._model, sample_rate)
        rec.SetWords(True)
        import json
        text_parts = []
        with wave.open(io.BytesIO(wav)) as wf:
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    text_parts.append(res.get("text", ""))
            res = json.loads(rec.FinalResult())
            text_parts.append(res.get("text", ""))
        return " ".join(p for p in text_parts if p).strip()


class OpenAISTT:
    """OpenAI Whisper-compatible transcription."""

    def __init__(self, api_key: str = "", base_url: str = ""):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url or settings.OPENAI_BASE_URL

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        if not self.api_key:
            return ""
        import httpx
        wav = _write_wav(audio_bytes, sample_rate)
        try:
            r = httpx.post(
                f"{self.base_url.rstrip('/')}/v1/audio/transcriptions" if self.base_url else "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": ("audio.wav", wav, "audio/wav")},
                data={"model": "whisper-1"},
                timeout=60,
            )
            r.raise_for_status()
            return r.json().get("text", "").strip()
        except Exception as exc:
            logger.error("OpenAI STT failed: %s", exc)
            return ""


def get_stt():
    provider = settings.STT_PROVIDER
    if provider == "openai":
        return OpenAISTT()
    if provider == "vosk":
        return VoskSTT()
    # auto: try vosk first
    vosk = VoskSTT()
    return vosk


async def transcribe_async(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    stt = get_stt()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, stt.transcribe, audio_bytes, sample_rate)