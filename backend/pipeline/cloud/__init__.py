"""
Cloud pipeline — uses external APIs (Groq, Edge TTS) with local fallbacks.

Contains:
    - manager.py:    Pipeline orchestrator (VAD -> ASR -> LLM -> TTS)
    - io_handler.py: WebSocket I/O protocol layer
    - memory.py:     SQLite long-term memory
    - asr_groq.py:   Groq Whisper API for speech recognition
    - llm_groq.py:   Groq API for LLM chat completions
    - tts_edge.py:   Microsoft Edge Neural TTS
    - hallucination_filter.py: ASR hallucination detection
    - llm_common.py: LLM response cleanup + context hints
    - tts_common.py: TTS text sanitization + audio fades
    - lang_detect.py: Text-based EN/DE language detection

Cross-dependencies: VAD always from local (lightweight, CPU-based).
Fallbacks to local ASR/LLM/TTS when cloud APIs fail.
"""

from .manager import PipelineManager
from .asr_groq import GroqASR
from .llm_groq import GroqLLM
from .tts_edge import EdgeTTSProcessor
from .memory import LongTermMemory
from .io_handler import (
    parse_inbound, InboundMessageType, make_safe_send, build_state_message,
)

__all__ = [
    "PipelineManager",
    "GroqASR", "GroqLLM", "EdgeTTSProcessor",
    "LongTermMemory",
    "parse_inbound", "InboundMessageType", "make_safe_send", "build_state_message",
]
