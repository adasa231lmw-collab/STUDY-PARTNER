"""Voice session manager — full-duplex conversation with barge-in support."""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from jarvis.config import settings
from jarvis.runtime.event_bus import bus
from jarvis.voice.audio import AudioCapture, AudioPlayback
from jarvis.voice.stt import transcribe_async
from jarvis.voice.tts import speak_async, get_tts, _decode_audio

logger = logging.getLogger(__name__)


class VoiceState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    RECONNECTING = "reconnecting"


@dataclass
class VoiceSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: VoiceState = VoiceState.IDLE
    capture: Optional[AudioCapture] = None
    playback: Optional[AudioPlayback] = None
    conversation_id: str = ""
    turn_id: str = ""
    cancelled: bool = False
    continuous: bool = True

    def __post_init__(self):
        self.capture = AudioCapture(
            device_index=settings.MIC_DEVICE_INDEX,
            sample_rate=settings.SAMPLE_RATE,
            chunk_ms=settings.CHUNK_MS,
        )
        self.playback = AudioPlayback(
            device_index=settings.SPEAKER_DEVICE_INDEX,
            sample_rate=24000,
        )


class VoiceSessionManager:
    """Manages continuous voice conversation with interruption support."""

    def __init__(self):
        self.active: Optional[VoiceSession] = None
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    async def start(self, conversation_id: str = "", continuous: bool = True) -> VoiceSession:
        async with self._lock:
            if self.active:
                await self.stop()
            session = VoiceSession(conversation_id=conversation_id, continuous=continuous)
            session.capture.start()
            self.active = session
            session.state = VoiceState.LISTENING
            bus.publish("voice.listening", {"session_id": session.session_id, "continuous": continuous}, conversation_id)
            logger.info("Voice session started: %s", session.session_id)
            return session

    async def stop(self) -> None:
        session = self.active
        if not session:
            return
        self.active = None
        session.cancelled = True
        if session.capture:
            try:
                session.capture.stop()
            except Exception:
                pass
        if session.playback:
            try:
                session.playback.stop()
            except Exception:
                pass
        session.state = VoiceState.IDLE
        bus.publish("voice.stopped", {"session_id": session.session_id}, session.conversation_id)

    async def interrupt(self) -> None:
        """Stop current playback and start listening again."""
        session = self.active
        if not session:
            return
        if session.playback:
            session.playback.stop()
        session.cancelled = True
        session.turn_id = str(uuid.uuid4())
        session.state = VoiceState.LISTENING
        bus.publish("assistant.interrupted", {"session_id": session.session_id, "turn_id": session.turn_id}, session.conversation_id)

    async def capture_turn(self, session: VoiceSession) -> bytes:
        """Capture audio until VAD detects end of speech or timeout."""
        if not session.capture:
            return b""
        elapsed = 0.0
        max_seconds = settings.MAX_RECORD_SECONDS
        while elapsed < max_seconds:
            await asyncio.sleep(0.05)
            elapsed += 0.05
            if session.capture.is_silence_done(settings.SILENCE_MS):
                break
            if session.cancelled:
                return b""
        return session.capture.stop()

    async def run_continuous(self, session: VoiceSession, orchestrator) -> None:
        """Run a continuous voice loop: listen → transcribe → respond → speak."""
        while self.active and session.active and not session.cancelled:
            try:
                # listen for speech
                session.state = VoiceState.LISTENING
                bus.publish("voice.listening", {"session_id": session.session_id}, session.conversation_id)
                audio = await self.capture_turn(session)
                if not audio or len(audio) < 400:  # too short / silence
                    continue
                # transcribe
                session.state = VoiceState.PROCESSING
                bus.publish("voice.transcribing", {"session_id": session.session_id, "audio_bytes": len(audio)}, session.conversation_id)
                text = await transcribe_async(audio, settings.SAMPLE_RATE)
                if not text or not text.strip():
                    continue
                bus.publish("voice.transcription", {"session_id": session.session_id, "text": text}, session.conversation_id)
                # respond
                session.turn_id = str(uuid.uuid4())
                response = await orchestrator.handle_user_input(text, session.conversation_id, session.turn_id)
                # speak
                if response and response.get("text"):
                    session.state = VoiceState.SPEAKING
                    bus.publish("assistant.speaking", {"session_id": session.session_id, "text": response["text"], "turn_id": session.turn_id}, session.conversation_id)
                    await self.speak_response(session, response["text"], session.turn_id)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Voice loop error: %s", exc)
                session.state = VoiceState.RECONNECTING
                bus.publish("error", {"error": str(exc), "session_id": session.session_id}, session.conversation_id)
                await asyncio.sleep(0.5)

    async def speak_response(self, session: VoiceSession, text: str, turn_id: str) -> None:
        """Stream TTS audio to playback, allowing interruption."""
        if not text:
            return
        try:
            tts = get_tts()
            async for chunk in tts.stream(text):
                if session.cancelled or self.active != session:
                    return
                if session.playback and chunk:
                    pcm, sr = _decode_audio(chunk)
                    if pcm:
                        session.playback.play_pcm(pcm, sample_rate=sr)
        except Exception as exc:
            logger.error("TTS playback failed: %s", exc)