"""WebSocket I/O — inbound parsing and outbound message builders."""
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional

log = logging.getLogger("s2s.io")
SendFn = Callable[[str | bytes], Awaitable[None]]

class InboundMessageType(Enum):
    AUDIO = "audio"
    CLEAR = "clear"
    CHAT = "chat"
    CONFIG = "config"

@dataclass
class InboundMessage:
    type: InboundMessageType
    audio_bytes: Optional[bytes] = None
    text: Optional[str] = None
    config_data: Optional[dict] = None

def parse_inbound(message: dict) -> Optional[InboundMessage]:
    if "bytes" in message and message["bytes"]:
        raw = message["bytes"]
        if len(raw) < 2 or len(raw) % 2 != 0:
            return None
        return InboundMessage(type=InboundMessageType.AUDIO, audio_bytes=raw)
    if "text" in message and message["text"]:
        try:
            data = json.loads(message["text"])
        except json.JSONDecodeError:
            return None
        t = data.get("type", "")
        if t == "clear":
            return InboundMessage(type=InboundMessageType.CLEAR)
        if t == "chat" and data.get("text", "").strip():
            return InboundMessage(type=InboundMessageType.CHAT, text=data["text"].strip())
        if t == "config":
            return InboundMessage(type=InboundMessageType.CONFIG, config_data=data)
    return None

def build_state_message(state: str) -> str:
    return json.dumps({"type": "state", "state": state})

def build_audio_end() -> str:
    return json.dumps({"type": "audio_end"})

def build_interrupt() -> str:
    return json.dumps({"type": "interrupt"})

def make_safe_send(ws) -> SendFn:
    async def _send(payload: str | bytes):
        try:
            await (ws.send_bytes(payload) if isinstance(payload, bytes) else ws.send_text(payload))
        except Exception:
            pass
    return _send
