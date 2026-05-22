"""Local pipeline — on-device VAD → ASR → LLM → TTS."""

from .manager import PipelineManager
from .vad import VADProcessor
from .asr import ASRProcessor
from .llm import LLMProcessor, FallbackLLM
from .tts import TTSProcessor
from .memory import LongTermMemory
from .io_handler import (
    parse_inbound, InboundMessageType, make_safe_send, build_state_message,
)

__all__ = [
    "PipelineManager",
    "VADProcessor", "ASRProcessor",
    "LLMProcessor", "FallbackLLM",
    "TTSProcessor",
    "LongTermMemory",
    "parse_inbound", "InboundMessageType", "make_safe_send", "build_state_message",
]
