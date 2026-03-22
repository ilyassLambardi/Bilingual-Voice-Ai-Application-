"""
FastAPI WebSocket server for the Speech-to-Speech pipeline.

Supports two modes:
  cloud  — Groq API for LLM + ASR, per-session managers (multi-user)
  local  — local GPU models, shared manager (single-user dev)

Protocol
--------
Client → Server:
    binary   Raw Int16 PCM, 16 kHz mono, 512-sample frames (~32 ms each).
    json     {"type": "clear"}   — reset conversation.
             {"type": "config", ...} — runtime config overrides.

Server → Client:
    json     {"type": "state",    "state": "idle|listening|thinking|speaking"}
    json     {"type": "transcript","role":"user|assistant","text":"..."}
    json     {"type": "audio_config", "sample_rate": 24000}
    json     {"type": "audio_end"}
    json     {"type": "interrupt"}
    binary   Raw Int16 PCM audio at the sample_rate from audio_config.
"""

import json
import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from config import config
from pipeline.manager import PipelineManager

log = logging.getLogger("s2s")

_MAX_SESSIONS = 10  # connection limit

# Built frontend directory (created by `npm run build` in frontend/)
_STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

# For local mode: shared manager (single-user, models loaded once)
# For cloud mode: managers are per-session (lightweight, API-based)
_shared_manager: PipelineManager | None = None
_sessions: dict[str, PipelineManager] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load models at startup."""
    global _shared_manager

    if config.mode == "local":
        print("[Startup] Local mode -- loading models (this may take a minute) ...")
        _shared_manager = PipelineManager(config)
        await _shared_manager.load_models()
    else:
        # Cloud mode: do a quick preload of VAD + TTS (shared, lightweight)
        # LLM and ASR are API calls, loaded per-session
        print("[Startup] Cloud mode -- preloading VAD + TTS ...")
        warmup = PipelineManager(config)
        await warmup.load_models()
        _shared_manager = warmup  # keep for sharing TTS cache

    print("[Startup] Ready -- accepting connections.")
    yield
    print("[Shutdown] Cleaning up sessions ...")
    _sessions.clear()
    print("[Shutdown] Server stopped.")


app = FastAPI(title="S2S Voice Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mode": config.mode,
        "active_sessions": len(_sessions),
        "config": {
            "sample_rate_in": config.sample_rate_in,
            "tts_sample_rate": config.tts_sample_rate,
        },
    }


@app.get("/api/system")
async def system_info():
    """Return system configuration for frontend display."""
    return {
        "mode": config.mode,
        "llm_model": config.groq_llm_model if config.mode == "cloud" else "local",
        "asr_model": config.groq_asr_model if config.mode == "cloud" else config.asr_model,
        "tts_speaker_en": config.tts_speaker_en,
        "tts_speaker_de": config.tts_speaker_de,
        "max_tokens": config.llm_max_tokens,
        "temperature": config.llm_temperature,
        "vad_threshold": config.vad_threshold,
        "sample_rate": config.sample_rate_in,
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    # Connection limit
    if len(_sessions) >= _MAX_SESSIONS:
        await ws.close(code=1013, reason="Too many connections")
        return

    await ws.accept()
    session_id = str(uuid.uuid4())[:8]
    log.info(f"[WS] Client {session_id} connected.")

    # Get or create a session manager
    if config.mode == "cloud":
        mgr = PipelineManager(config)
        await mgr.load_models()
        _sessions[session_id] = mgr
    else:
        mgr = _shared_manager

    # Create send closure ONCE per connection (not per-chunk)
    send = _make_send(ws)

    await ws.send_text(json.dumps({
        "type": "state",
        "state": "idle" if mgr._models_ready else "loading",
    }))

    try:
        while True:
            message = await ws.receive()

            if "bytes" in message and message["bytes"]:
                try:
                    await mgr.handle_audio_chunk(message["bytes"], send)
                except Exception as e:
                    log.error(f"[WS:{session_id}] Audio error: {e}")

            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")
                if msg_type == "clear":
                    await mgr.clear(send)
                elif msg_type == "chat":
                    text = data.get("text", "").strip()
                    if text:
                        await mgr.handle_text_chat(text, send)
                elif msg_type == "config":
                    _apply_session_config(data, mgr)

    except WebSocketDisconnect:
        log.info(f"[WS:{session_id}] Client disconnected.")
    except Exception as e:
        log.error(f"[WS:{session_id}] Error: {e}")
    finally:
        # Cancel any running pipeline tasks to avoid orphaned work
        await _cleanup_session(mgr)
        _sessions.pop(session_id, None)
        log.info(f"[WS:{session_id}] Session ended. Active: {len(_sessions)}")


def _make_send(ws: WebSocket):
    """Return an async callable that sends text or bytes."""
    async def _send(payload: str | bytes):
        try:
            if isinstance(payload, bytes):
                await ws.send_bytes(payload)
            else:
                await ws.send_text(payload)
        except Exception:
            pass  # client may have disconnected
    return _send


async def _cleanup_session(mgr: PipelineManager):
    """Cancel running tasks and clean up session resources."""
    try:
        if mgr._gen_task and not mgr._gen_task.done():
            mgr._interrupt.set()
            mgr._gen_task.cancel()
            try:
                await mgr._gen_task
            except (asyncio.CancelledError, Exception):
                pass
        mgr._generating = False
        mgr._audio_buffer.clear()
        if mgr._accumulation_timer is not None:
            mgr._accumulation_timer.cancel()
            mgr._accumulation_timer = None
    except Exception as e:
        log.warning(f"[WS] Cleanup error: {e}")


def _apply_session_config(data: dict, mgr: PipelineManager):
    """Apply runtime config overrides per-session (not global)."""
    if "language" in data:
        lang = data["language"]
        mgr.config.asr_language = lang if lang != "auto" else None
    if "vad_threshold" in data:
        val = float(data["vad_threshold"])
        if 0.1 <= val <= 0.99:  # validate bounds
            mgr.config.vad_threshold = val


# ── Static file serving (production: built React app) ──────────────
if _STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=_STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve built React app — SPA catch-all."""
        file = _STATIC_DIR / full_path
        if file.exists() and file.is_file():
            return FileResponse(file)
        return FileResponse(_STATIC_DIR / "index.html")
else:
    print("[Static] No frontend/dist found -- run 'npm run build' in frontend/ for production.")


if __name__ == "__main__":
    print("=" * 60)
    print(f"  Speech-to-Speech Backend ({config.mode.upper()} mode)")
    print(f"  WebSocket: ws://{config.host}:{config.port}/ws")
    if _STATIC_DIR.exists():
        print(f"  Frontend: serving from {_STATIC_DIR}")
    print("=" * 60)
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )
