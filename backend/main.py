"""
FastAPI WebSocket server for the Speech-to-Speech pipeline.

This is the application entry point that wires together the four
architectural modules:

    Module 1 — I/O:         ``pipeline/io_handler.py``
    Module 2 — Processing:  ``pipeline/vad.py``, ``asr*.py``, ``llm*.py``, ``tts*.py``
    Module 3 — State:       ``pipeline/session_state.py``, ``pipeline/memory.py``
    Module 4 — Scheduling:  ``pipeline/manager.py``

Supports two modes:
  cloud  — Groq API for LLM + ASR, per-session managers (multi-user)
  local  — local GPU models, shared manager (single-user dev)

Protocol (defined in Module 1 — io_handler)
--------------------------------------------
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
from pipeline.io_handler import (
    parse_inbound,
    InboundMessageType,
    make_safe_send,
    build_state_message,
)

log = logging.getLogger("s2s")

# ── Ring buffer for pipeline logs (viewable via /api/logs) ───────────
import collections

_log_buf = collections.deque(maxlen=300)
_original_print = print

def _capturing_print(*args, **kwargs):
    """Wrapper around print() that also captures to ring buffer."""
    try:
        _original_print(*args, **kwargs)
    except UnicodeEncodeError:
        # Windows charmap can't handle some Unicode — print ASCII-safe version
        safe = " ".join(str(a) for a in args).encode("ascii", "replace").decode()
        _original_print(safe, **{k: v for k, v in kwargs.items() if k != "sep"})
    msg = " ".join(str(a) for a in args)
    if msg.strip():
        _log_buf.append(msg.rstrip())

import builtins
builtins.print = _capturing_print

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


@app.get("/api/diagnose")
async def diagnose():
    """Test every pipeline component and return detailed results."""
    import numpy as np
    import time
    import traceback

    results = {"timestamp": time.time(), "mode": config.mode, "tests": {}}

    # Test 1: Config
    results["tests"]["config"] = {
        "status": "ok",
        "groq_key_set": bool(config.groq_api_key and not config.groq_api_key.startswith("gsk_your")),
        "groq_key_length": len(config.groq_api_key) if config.groq_api_key else 0,
        "min_silence_ms": config.min_silence_ms,
        "vad_threshold": config.vad_threshold,
        "tts_engine": config.tts_engine,
    }

    # Test 2: Shared manager state
    if _shared_manager:
        mgr = _shared_manager
        results["tests"]["manager"] = {
            "status": "ok",
            "models_ready": mgr._models_ready,
            "has_vad": mgr._vad is not None,
            "has_asr": mgr._asr is not None,
            "has_llm": mgr._llm is not None,
            "has_tts": mgr._tts is not None,
            "asr_type": type(mgr._asr).__name__ if mgr._asr else "None",
            "llm_type": type(mgr._llm).__name__ if mgr._llm else "None",
            "tts_type": type(mgr._tts).__name__ if mgr._tts else "None",
            "generating": mgr._generating,
            "state": mgr.state,
        }
    else:
        results["tests"]["manager"] = {"status": "error", "detail": "No shared manager"}

    # Test 3: VAD with synthetic audio
    try:
        from pipeline.vad import VADProcessor
        vad = VADProcessor(threshold=0.45, min_speech_ms=200, min_silence_ms=700, sample_rate=16000)
        noise = np.random.randn(512).astype(np.float32) * 0.001
        is_speaking, utt = vad.process_chunk(noise)
        results["tests"]["vad"] = {"status": "ok", "model_loaded": True}
    except Exception as e:
        results["tests"]["vad"] = {"status": "error", "detail": str(e)}

    # Test 4: ASR (quick Groq API check)
    if _shared_manager and _shared_manager._asr:
        asr = _shared_manager._asr
        try:
            # Create 1s of silence to test API connectivity
            test_audio = np.random.randn(16000).astype(np.float32) * 0.01
            t0 = time.time()
            result = await asr.transcribe(test_audio)
            t1 = time.time()
            results["tests"]["asr"] = {
                "status": "ok",
                "type": type(asr).__name__,
                "latency_ms": round((t1 - t0) * 1000),
                "result": result.get("text", "")[:100],
            }
        except Exception as e:
            results["tests"]["asr"] = {
                "status": "error",
                "type": type(asr).__name__,
                "detail": str(e),
                "traceback": traceback.format_exc()[-500:],
            }
    else:
        results["tests"]["asr"] = {"status": "error", "detail": "No ASR loaded"}

    # Test 5: LLM (quick generation test)
    if _shared_manager and _shared_manager._llm:
        llm = _shared_manager._llm
        try:
            t0 = time.time()
            tokens = []
            async for tok in llm.stream("Say hello in one word", lang="en"):
                tokens.append(tok)
                if len(tokens) > 5:
                    break
            t1 = time.time()
            results["tests"]["llm"] = {
                "status": "ok",
                "type": type(llm).__name__,
                "latency_ms": round((t1 - t0) * 1000),
                "first_tokens": "".join(tokens)[:50],
            }
        except Exception as e:
            results["tests"]["llm"] = {
                "status": "error",
                "type": type(llm).__name__,
                "detail": str(e),
                "traceback": traceback.format_exc()[-500:],
            }
    else:
        results["tests"]["llm"] = {"status": "error", "detail": "No LLM loaded (echo mode)"}

    # Test 6: TTS
    if _shared_manager and _shared_manager._tts:
        tts = _shared_manager._tts
        try:
            t0 = time.time()
            audio_bytes = await tts.synthesize("Hello", "en")
            t1 = time.time()
            results["tests"]["tts"] = {
                "status": "ok" if len(audio_bytes) > 0 else "error",
                "type": type(tts).__name__,
                "latency_ms": round((t1 - t0) * 1000),
                "output_bytes": len(audio_bytes),
            }
        except Exception as e:
            results["tests"]["tts"] = {
                "status": "error",
                "type": type(tts).__name__,
                "detail": str(e),
                "traceback": traceback.format_exc()[-500:],
            }
    else:
        results["tests"]["tts"] = {"status": "error", "detail": "No TTS loaded"}

    # Test 7: Sessions (with per-session state)
    session_details = {}
    for sid, smgr in _sessions.items():
        session_details[sid] = {
            "state": smgr.state,
            "generating": smgr._generating,
            "models_ready": smgr._models_ready,
            "buffer_fragments": len(smgr._audio_buffer),
            "has_timer": smgr._accumulation_timer is not None,
            "interrupt_set": smgr._interrupt.is_set(),
            "asr_type": type(smgr._asr).__name__ if smgr._asr else "None",
            "llm_type": type(smgr._llm).__name__ if smgr._llm else "None",
        }
    results["tests"]["sessions"] = {
        "active": len(_sessions),
        "max": _MAX_SESSIONS,
        "details": session_details,
    }

    # Overall
    all_ok = all(
        t.get("status") == "ok"
        for k, t in results["tests"].items()
        if k != "sessions"
    )
    results["overall"] = "ALL OK" if all_ok else "ISSUES FOUND"

    return results


@app.get("/api/logs")
async def get_logs(n: int = 100):
    """Return recent pipeline log lines for remote debugging."""
    return {"lines": list(_log_buf)[-min(n, 300):]}


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

    # Module 1 (I/O): Create send closure ONCE per connection
    send = make_safe_send(ws)

    await ws.send_text(build_state_message(
        "idle" if mgr._models_ready else "loading"
    ))

    try:
        while True:
            raw_message = await ws.receive()

            # Module 1 (I/O): Parse inbound message
            msg = parse_inbound(raw_message)
            if msg is None:
                continue

            if msg.type == InboundMessageType.AUDIO:
                try:
                    await mgr.handle_audio_chunk(msg.audio_bytes, send)
                except Exception as e:
                    log.error(f"[WS:{session_id}] Audio error: {e}")

            elif msg.type == InboundMessageType.CLEAR:
                await mgr.clear(send)

            elif msg.type == InboundMessageType.CHAT:
                await mgr.handle_text_chat(msg.text, send)

            elif msg.type == InboundMessageType.CONFIG:
                _apply_session_config(msg.config_data, mgr)

    except WebSocketDisconnect:
        log.info(f"[WS:{session_id}] Client disconnected.")
    except Exception as e:
        log.error(f"[WS:{session_id}] Error: {e}")
    finally:
        # Cancel any running pipeline tasks to avoid orphaned work
        await _cleanup_session(mgr)
        _sessions.pop(session_id, None)
        log.info(f"[WS:{session_id}] Session ended. Active: {len(_sessions)}")


# NOTE: _make_send moved to Module 1 (pipeline/io_handler.py) as make_safe_send


async def _cleanup_session(mgr: PipelineManager):
    """Cancel running tasks and clean up session resources."""
    try:
        # Cancel running generation task
        gen_task = getattr(mgr, '_gen_task', None)
        if gen_task and not gen_task.done():
            mgr._interrupt.set()
            gen_task.cancel()
            try:
                await gen_task
            except (asyncio.CancelledError, Exception):
                pass
        mgr._generating = False

        # Cancel accumulation timer
        timer = getattr(mgr, '_accumulation_timer', None)
        if timer is not None:
            timer.cancel()
            mgr._accumulation_timer = None

        # Clear audio buffer
        if hasattr(mgr, '_audio_buffer'):
            mgr._audio_buffer.clear()

        # Close LTM connection to prevent SQLite leaks
        ltm = getattr(mgr, '_ltm', None)
        if ltm:
            try:
                ltm.close()
            except Exception:
                pass
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
