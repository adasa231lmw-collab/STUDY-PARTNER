"""Audio capture and playback using sounddevice."""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import io

from jarvis.config import settings

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captures microphone audio into a numpy buffer with VAD gating."""

    def __init__(self, device_index: int = -1, sample_rate: int = 16000, chunk_ms: int = 30):
        self.device_index = device_index if device_index >= 0 else None
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.chunk_size = int(sample_rate * chunk_ms / 1000)
        self._stream: Optional[sd.InputStream] = None
        self._buffer: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._running = False
        self._level = 0.0
        self._vad_silence_ms = 0
        self._speech_started = False

    def _callback(self, indata, frames, time_info, status):
        nonlocal_max = float(np.max(np.abs(indata))) if len(indata) else 0.0
        with self._lock:
            self._level = max(self._level * 0.7, nonlocal_max)
            self._buffer.append(indata.copy())
            # simple energy-based VAD
            if nonlocal_max > 0.02:
                self._speech_started = True
                self._vad_silence_ms = 0
            elif self._speech_started:
                self._vad_silence_ms += self.chunk_ms

    def start(self) -> None:
        if self._running:
            return
        try:
            self._stream = sd.InputStream(
                device=self.device_index,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self._callback,
                dtype="float32",
            )
            self._stream.start()
            self._running = True
            self._buffer.clear()
            self._speech_started = False
            self._vad_silence_ms = 0
            logger.info("Audio capture started")
        except Exception as exc:
            logger.error("Audio capture failed: %s", exc)
            raise

    def stop(self) -> bytes:
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._running = False
        with self._lock:
            if not self._buffer:
                return b""
            data = np.concatenate(self._buffer, axis=0)
            self._buffer.clear()
            # convert float32 [-1,1] to int16 PCM
            pcm = (np.clip(data, -1.0, 1.0) * 32767).astype("<i2")
            return pcm.tobytes()

    def is_speech_done(self, silence_ms: int = 600) -> bool:
        return self._speech_started and self._vad_silence_ms >= silence_ms

    @property
    def level(self) -> float:
        with self._lock:
            return self._level


class AudioPlayback:
    """Plays back audio from bytes or file."""

    def __init__(self, device_index: int = -1, sample_rate: int = 24000):
        self.device_index = device_index if device_index >= 0 else None
        self.sample_rate = sample_rate
        self._stream: Optional[sd.OutputStream] = None
        self._lock = threading.Lock()
        self._cancelled = False

    def play_pcm(self, pcm_bytes: bytes, sample_rate: int = 24000) -> None:
        """Play raw int16 PCM bytes synchronously."""
        self.stop()
        data = np.frombuffer(pcm_bytes, dtype="<i2").astype("float32") / 32767.0
        if data.size == 0:
            return
        self._cancelled = False
        try:
            with sd.OutputStream(device=self.device_index, channels=1, samplerate=sample_rate, dtype="float32") as stream:
                stream.write(data)
        except Exception as exc:
            logger.error("Playback failed: %s", exc)

    def play_file(self, path: str) -> None:
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            self.stop()
            self._cancelled = False
            with sd.OutputStream(device=self.device_index, channels=1, samplerate=sr, dtype="float32") as stream:
                stream.write(data)
        except Exception as exc:
            logger.error("File playback failed: %s", exc)

    def stop(self) -> None:
        self._cancelled = True
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


def list_devices() -> dict:
    try:
        devices = sd.query_devices()
        return {
            "input": [{"index": i, "name": d["name"], "channels": d["max_input_channels"]} for i, d in enumerate(devices) if d["max_input_channels"] > 0],
            "output": [{"index": i, "name": d["name"], "channels": d["max_output_channels"]} for i, d in enumerate(devices) if d["max_output_channels"] > 0],
        }
    except Exception as exc:
        return {"input": [], "output": [], "error": str(exc)}