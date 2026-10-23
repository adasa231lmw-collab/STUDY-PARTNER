# JARVIS — Status

## Done
- Backend scaffold: config/settings (Ollama qwen3:4b + nomic-embed-text), LLM provider, voice (audio/stt/tts/session), tools (calc/files/permissions/registry/shell), agents (base + research/study/coding/browser/memory/planning/verification), runtime (event_bus, task_manager), db (memory, vector), RAG, browser (playwright), utils (logging).
- `backend/main.py`: FastAPI entry point with /health, /chat, /embed; verified Ollama connectivity and model response.
- Fixed OLLAMA_HOST normalization (missing protocol / 0.0.0.0).
- Frontend package.json with Electron/React/Vite/Tailwind deps.
- .gitignore, initial commit pushed to GitHub (main).

## To Do
- Frontend source: notch overlay UI, chat, SSE client, voice controls.
- Global RSHIFT hotkey (native Windows hook).
- SSE event streaming endpoint wiring.
- Tool registry permission enforcement + confirmation flow.
- Browser automation integration tests.
- MCP server support.
- Agent Skills discovery/loader.
- RAG ingestion pipeline end-to-end with nomic-embed-text.
- Voice STT/TTS provider verification (Vosk/Piper/edge-tts).
- Tests and packaging.
- Study/research/coding agent orchestration vertical slice.
