"""Application configuration loaded from environment and config files."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    _BASE = Path(__file__).resolve().parents[3]
    load_dotenv(_BASE / ".env")
except Exception:
    pass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


BASE_DIR = Path(__file__).resolve().parents[3]
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = Path(_env("JARVIS_DATA_DIR", str(BASE_DIR / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Ollama / LLM ---
_raw_host = _env("OLLAMA_HOST", "http://127.0.0.1:11434")
if not _raw_host.startswith("http"):
    _raw_host = f"http://{_raw_host}"
OLLAMA_HOST = _raw_host.replace("0.0.0.0", "127.0.0.1")
OLLAMA_MODEL = _env("JARVIS_MODEL", "qwen3:4b")
OLLAMA_EMBED_MODEL = _env("JARVIS_EMBED_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT = _env_int("JARVIS_OLLAMA_TIMEOUT", 120)
OLLAMA_USE_GPU = _env_bool("JARVIS_OLLAMA_USE_GPU", True)

# --- OpenAI-compatible fallback ---
OPENAI_API_KEY = _env("OPENAI_API_KEY", "")
OPENAI_BASE_URL = _env("OPENAI_BASE_URL", "")
OPENAI_MODEL = _env("OPENAI_MODEL", "")

# --- Voice ---
STT_PROVIDER = _env("JARVIS_STT_PROVIDER", "auto")  # auto | vosk | faster_whisper | openai
TTS_PROVIDER = _env("JARVIS_TTS_PROVIDER", "edge")  # edge | piper | openai | system
TTS_VOICE = _env("JARVIS_TTS_VOICE", "en-US-AriaNeural")
TTS_RATE = _env("JARVIS_TTS_RATE", "+0%")
TTS_PITCH = _env("JARVIS_TTS_PITCH", "+0Hz")
VOSK_MODEL_PATH = _env("JARVIS_VOSK_MODEL_PATH", str(BASE_DIR / "models" / "vosk-model-small-en-us-0.15"))
WHISPER_MODEL = _env("JARVIS_WHISPER_MODEL", "base")
MIC_DEVICE_INDEX = _env_int("JARVIS_MIC_DEVICE", -1)
SPEAKER_DEVICE_INDEX = _env_int("JARVIS_SPEAKER_DEVICE", -1)
SAMPLE_RATE = _env_int("JARVIS_SAMPLE_RATE", 16000)
CHUNK_MS = _env_int("JARVIS_CHUNK_MS", 30)
SILENCE_MS = _env_int("JARVIS_SILENCE_MS", 600)
MAX_RECORD_SECONDS = _env_int("JARVIS_MAX_RECORD_SECONDS", 25)
VAD_AGGRESSIVENESS = _env_int("JARVIS_VAD_AGGRESSIVENESS", 2)

# --- Web search ---
SEARCH_PROVIDER = _env("JARVIS_SEARCH_PROVIDER", "duckduckgo")  # duckduckgo | brave | none
BRAVE_API_KEY = _env("BRAVE_API_KEY", "")
SEARCH_MAX_RESULTS = _env_int("JARVIS_SEARCH_MAX_RESULTS", 8)

# --- Memory / RAG ---
MEMORY_ENABLED = _env_bool("JARVIS_MEMORY_ENABLED", True)
MEMORY_STORE_PATH = Path(_env("JARVIS_MEMORY_STORE_PATH", str(DATA_DIR / "memory.json")))
VECTOR_STORE_PATH = Path(_env("JARVIS_VECTOR_STORE_PATH", str(DATA_DIR / "vector_store.json")))
RAG_TOP_K = _env_int("JARVIS_RAG_TOP_K", 5)
AUTO_SAVE_MEMORY = _env_bool("JARVIS_AUTO_SAVE_MEMORY", False)

# --- Permissions ---
CONFIRM_DESTRUCTIVE = _env_bool("JARVIS_CONFIRM_DESTRUCTIVE", True)
ALLOWED_WORKSPACE = _env("JARVIS_ALLOWED_WORKSPACE", str(BASE_DIR))
ALLOWED_SHELL = _env_bool("JARVIS_ALLOWED_SHELL", False)
ALLOWED_BROWSER = _env_bool("JARVIS_ALLOWED_BROWSER", True)

# --- Server ---
HOST = _env("JARVIS_HOST", "127.0.0.1")
PORT = _env_int("JARVIS_PORT", 8765)
SSE_PATH = _env("JARVIS_SSE_PATH", "/events")

# --- Hotkey ---
HOTKEY = _env("JARVIS_HOTKEY", "RSHIFT")
HOTKEY_CONTINUOUS = _env_bool("JARVIS_HOTKEY_CONTINUOUS", True)

# --- Personality ---
PERSONALITY = _env("JARVIS_PERSONALITY", "jarvis")

# --- Paths ---
SKILLS_DIR = BASE_DIR / "skills"
DOCS_DIR = BASE_DIR / "docs"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "jarvis.log"