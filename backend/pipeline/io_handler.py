"""
Module 1: Input/Output Handler — WebSocket I/O and protocol layer.

This module encapsulates all input/output operations for the real-time
Speech-to-Speech pipeline:

  - **Inbound**: Parses binary audio frames (Int16 PCM) and JSON control
    messages (clear, config, chat) arriving from the WebSocket client.
  - **Outbound**: Formats and sends structured JSON messages (state updates,
    transcripts, audio config, interrupts) and binary PCM audio back to
    the client.
  - **Protocol**: Defines message types, payload schemas, and the
    bidirectional communication contract between frontend and backend.

Architecture role:
    The I/O handler sits between the WebSocket transport (main.py) and
    the Control Flow module (manager.py).  It normalises raw WebSocket
    frames into typed Python objects and provides a safe, async send
    interface that the rest of the pipeline uses.
"""

import json
import logging
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

log = logging.getLogger("s2s.io")

# ── Type aliases ─────────────────────────────────────────────────────
SendFn = Callable[[str | bytes], None]


# ── Inbound message types ────────────────────────────────────────────

class InboundMessageType(Enum):
    """All message types the server can receive from the client."""
    AUDIO = "audio"        # binary PCM frames
    CLEAR = "clear"        # reset conversation
    CHAT = "chat"          # typed text input
    CONFIG = "config"      # runtime config overrides


@dataclass
class InboundMessage:
    """Parsed inbound message from the WebSocket client."""
    type: InboundMessageType
    audio_bytes: Optional[bytes] = None    # for AUDIO type
    text: Optional[str] = None             # for CHAT type
    config_data: Optional[dict] = None     # for CONFIG type


# ── Outbound message types ───────────────────────────────────────────

class OutboundMessageType(Enum):
    """All message types the server can send to the client."""
    STATE = "state"
    TRANSCRIPT = "transcript"
    PARTIAL_TRANSCRIPT = "partial_transcript"
    AUDIO_CONFIG = "audio_config"
    AUDIO_END = "audio_end"
    INTERRUPT = "interrupt"
    BACKCHANNEL = "backchannel"
    GHOST_TEXT = "ghost_text"
    LANGUAGE_SHIFT = "language_shift"
    ERROR = "error"


# ── Inbound parsing ─────────────────────────────────────────────────

def parse_inbound(message: dict) -> Optional[InboundMessage]:
    """Parse a raw WebSocket message dict into a typed InboundMessage.

    Parameters
    ----------
    message : dict
        Raw message from ``await ws.receive()``.  Contains either
        ``"bytes"`` (binary audio) or ``"text"`` (JSON control).

    Returns
    -------
    InboundMessage or None if the message is malformed / empty.
    """
    # Binary audio frame
    if "bytes" in message and message["bytes"]:
        raw = message["bytes"]
        # Validate: Int16 PCM requires even byte length, minimum 2 bytes
        if len(raw) < 2 or len(raw) % 2 != 0:
            log.warning(f"[IO] Invalid audio frame: {len(raw)} bytes (must be even, >= 2)")
            return None
        return InboundMessage(
            type=InboundMessageType.AUDIO,
            audio_bytes=raw,
        )

    # JSON control message
    if "text" in message and message["text"]:
        try:
            data = json.loads(message["text"])
        except json.JSONDecodeError:
            log.warning("[IO] Received invalid JSON, ignoring")
            return None

        msg_type = data.get("type", "")

        if msg_type == "clear":
            return InboundMessage(type=InboundMessageType.CLEAR)

        if msg_type == "chat":
            text = data.get("text", "").strip()
            if text:
                return InboundMessage(
                    type=InboundMessageType.CHAT,
                    text=text,
                )
            return None

        if msg_type == "config":
            return InboundMessage(
                type=InboundMessageType.CONFIG,
                config_data=data,
            )

        log.debug(f"[IO] Unknown message type: {msg_type}")
        return None

    return None


def audio_bytes_to_numpy(raw_bytes: bytes) -> np.ndarray:
    """Convert raw Int16 PCM bytes to a numpy array.

    Parameters
    ----------
    raw_bytes : bytes
        Raw Int16 PCM audio at 16 kHz mono.

    Returns
    -------
    np.ndarray of dtype int16.
    """
    return np.frombuffer(raw_bytes, dtype=np.int16)


# ── Outbound message builders ───────────────────────────────────────

def build_state_message(state: str) -> str:
    """Build a JSON state update message."""
    return json.dumps({"type": "state", "state": state})


def build_transcript_message(
    role: str,
    text: str,
    language: str,
    time_s: Optional[float] = None,
) -> str:
    """Build a JSON transcript message (user or assistant)."""
    msg = {
        "type": "transcript",
        "role": role,
        "text": text,
        "language": language,
    }
    if time_s is not None:
        msg["time"] = round(time_s, 3)
    return json.dumps(msg)


def build_partial_transcript(
    text: str,
    language: str,
    index: int,
) -> str:
    """Build a partial (streaming) assistant transcript message."""
    return json.dumps({
        "type": "partial_transcript",
        "role": "assistant",
        "text": text,
        "language": language,
        "index": index,
    })


def build_audio_config(sample_rate: int) -> str:
    """Build audio configuration message for the client."""
    return json.dumps({
        "type": "audio_config",
        "sample_rate": sample_rate,
    })


def build_audio_end() -> str:
    """Build audio end marker."""
    return json.dumps({"type": "audio_end"})


def build_interrupt() -> str:
    """Build interrupt notification."""
    return json.dumps({"type": "interrupt"})


def build_backchannel() -> str:
    """Build backchannel notification."""
    return json.dumps({"type": "backchannel"})


def build_ghost_text(text: str) -> str:
    """Build ghost text (partial ASR while user is speaking)."""
    return json.dumps({"type": "ghost_text", "text": text})


def build_language_shift(from_lang: str, to_lang: str) -> str:
    """Build language shift notification."""
    return json.dumps({
        "type": "language_shift",
        "from": from_lang,
        "to": to_lang,
    })


def build_error(message: str) -> str:
    """Build error message."""
    return json.dumps({"type": "error", "message": message})


# ── Safe sender wrapper ──────────────────────────────────────────────

def make_safe_send(ws) -> SendFn:
    """Create a safe async send function that never raises on disconnect.

    Parameters
    ----------
    ws : WebSocket
        The FastAPI WebSocket connection.

    Returns
    -------
    An async callable accepting str (JSON) or bytes (PCM audio).
    """
    async def _send(payload: str | bytes):
        try:
            if isinstance(payload, bytes):
                await ws.send_bytes(payload)
            else:
                await ws.send_text(payload)
        except Exception:
            pass  # client may have disconnected
    return _send
