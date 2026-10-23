"""Text-to-speech providers: Edge TTS (local cloud), with fallbacks."""
from __future__ import annotations

import asyncio
import io
import logging
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

from jarvis.config import settings

logger = logging.getLogger(__name__)


def _decode_audio(audio_bytes: bytes) -> Tuple[bytes, int]:
    """Decode MP3/OGG to int16 PCM and sample rate."""
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        pcm = (np.clip(data, -1.0, 1.0) * 32767).astype("<i2")
        return pcm.tobytes(), sr
    except Exception as exc:
        logger.error("Audio decode failed: %s", exc)
        return b"", 24000


class EdgeTTS:
    """Microsoft Edge TTS via edge-tts library."""

    def __init__(self, voice: str = "", rate: str = "", pitch: str = ""):
        self.voice = voice or settings.TTS_VOICE
        self.rate = rate or settings.TTS_RATE
        self.pitch = pitch or settings.TTS_PITCH

    async def speak(self, text: str) -> bytes:
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)
            audio = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio += chunk["data"]
            return audio
        except Exception as exc:
            logger.error("Edge TTS failed: %s", exc)
            return b""

    async def stream(self, text: str):
        """Yield audio chunks as they become available."""
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
        except Exception as exc:
            logger.error("Edge TTS stream failed: %s", exc)


class PiperTTS:
    """Local Piper TTS."""

    async def speak(self, text: str) -> bytes:
        try:
            import piper
            voice = piper.PiperVoice.load("en_US-lessac-medium.onnx")
            buf = io.BytesIO()
            voice.speak(text, buf)
            return buf.getvalue()
        except Exception as exc:
            logger.error("Piper TTS failed: %s", exc)
            return b""


class SystemTTS:
    """Windows built-in SAPI TTS fallback."""

    async def speak(self, text: str) -> bytes:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            return b""
        except Exception as exc:
            logger.error("System TTS failed: %s", exc)
            return b""


def get_tts():
    provider = settings.TTS_PROVIDER
    if provider == "edge":
        return EdgeTTS()
    if provider == "piper":
        return PiperTTS()
    if provider == "system":
        return SystemTTS()
    return EdgeTTS()


async def speak_async(text: str, playback) -> None:
    """Generate TTS audio and play it through the playback object."""
    tts = get_tts()
    audio = await tts.speak(text)
    if audio:
        pcm, sr = _decode_audio(audio)
        if pcm:
            playback.play_pcm(pcm, sample_rate=sr)