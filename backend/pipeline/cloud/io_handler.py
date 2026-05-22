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
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional

log = logging.getLogger("s2s.io")

# ── Type aliases ─────────────────────────────────────────────────────
SendFn = Callable[[str | bytes], Awaitable[None]]


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
            log.warning("[IO] Invalid audio frame: %d bytes (must be even, >= 2)", len(raw))
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

        log.debug("[IO] Unknown message type: %s", msg_type)
        return None

    return None


# ── Outbound message builders ───────────────────────────────────────

def build_state_message(state: str) -> str:
    """Build a JSON state update message."""
    return json.dumps({"type": "state", "state": state})


def build_audio_end() -> str:
    """Build audio end marker."""
    return json.dumps({"type": "audio_end"})


def build_interrupt() -> str:
    """Build interrupt notification."""
    return json.dumps({"type": "interrupt"})


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
