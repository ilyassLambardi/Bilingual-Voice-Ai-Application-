"""
Module 3: Session State — Data storage and state management.

This module owns **all mutable session state** for a single client
connection.  It is the single source of truth for:

  - **Pipeline state machine**: idle → listening → thinking → speaking
  - **Audio buffer**: accumulated VAD utterance fragments waiting to be
    processed as one contiguous block.
  - **Interrupt state**: flags and counters that coordinate the
    interrupt / backchannel mechanism between VAD input and the
    running LLM+TTS pipeline.
  - **Language history**: recent detected languages for shift detection.
  - **Rate limiting**: rolling window of API request timestamps.
  - **Conversation memory**: interface to the persistent SQLite-backed
    Long-Term Memory (LTM) store.

Architecture role:
    SessionState is instantiated once per WebSocket connection.  The
    Control Flow module (PipelineManager / Scheduler) reads and mutates
    it; the Processing modules (VAD, ASR, LLM, TTS) are stateless
    workers that receive data and return results.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np

from .memory import LongTermMemory


# ── Pipeline state machine ────────────────────────────────────────────

VALID_STATES = {"idle", "listening", "thinking", "speaking"}


@dataclass
class InterruptState:
    """Tracks all interrupt-related counters and flags."""
    event: asyncio.Event = field(default_factory=asyncio.Event)
    generating: bool = False
    gen_task: Optional[asyncio.Task] = None
    speech_frames: int = 0                 # consecutive speaking frames during generation
    threshold: int = 4                     # ~128ms sustained speech to trigger interrupt
    backchannel_max_frames: int = 12       # ~384ms — shorter = backchannel, longer = interrupt
    backchannel_cooldown: int = 0          # cooldown frames after backchannel detection

    def reset(self):
        """Reset interrupt counters (not the event flag)."""
        self.speech_frames = 0
        self.backchannel_cooldown = 0

    def clear(self):
        """Full reset including event flag and generating state."""
        self.event.clear()
        self.generating = False
        self.gen_task = None
        self.speech_frames = 0
        self.backchannel_cooldown = 0


@dataclass
class AudioBuffer:
    """Manages accumulated audio fragments and the flush timer."""
    fragments: list = field(default_factory=list)
    timer: Optional[asyncio.TimerHandle] = None
    delay: float = 3.0           # seconds of silence before flushing
    send_fn: Optional[Callable] = None    # cached send ref for timer callback

    @property
    def total_samples(self) -> int:
        return sum(len(f) for f in self.fragments)

    @property
    def has_data(self) -> bool:
        return len(self.fragments) > 0

    def append(self, utterance: np.ndarray):
        """Add an utterance fragment to the buffer."""
        self.fragments.append(utterance)

    def flush(self) -> Optional[np.ndarray]:
        """Concatenate all fragments and clear the buffer.

        Returns None if the buffer is empty.
        """
        if not self.fragments:
            return None
        combined = np.concatenate(self.fragments)
        n = len(self.fragments)
        self.fragments.clear()
        return combined

    def clear(self):
        """Discard all buffered audio."""
        self.fragments.clear()

    def cancel_timer(self):
        """Cancel the pending flush timer if active."""
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None


class LanguageTracker:
    """Tracks recent detected languages for shift detection."""

    def __init__(self, max_history: int = 5):
        self._history: list[str] = []
        self._max = max_history

    @property
    def history(self) -> list[str]:
        return self._history

    def add(self, lang: str):
        """Record a detected language."""
        self._history.append(lang)
        if len(self._history) > self._max:
            self._history = self._history[-self._max:]

    def detect_shift(self, new_lang: str) -> Optional[str]:
        """Check if there's a language shift.

        Returns the previous language if a shift is detected, None otherwise.
        """
        prev = self._history[-3:] if len(self._history) >= 2 else []
        if prev and all(l != new_lang for l in prev) and len(prev) >= 2:
            return prev[-1]
        return None

    def clear(self):
        self._history.clear()


class RateLimiter:
    """Rolling-window rate limiter for API calls."""

    def __init__(self, max_requests: int = 25, window_seconds: float = 60.0):
        self._timestamps: list[float] = []
        self._max = max_requests
        self._window = window_seconds

    def check(self) -> bool:
        """Return True if the request is within limits, False if exceeded."""
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < self._window]
        if len(self._timestamps) >= self._max:
            return False
        self._timestamps.append(now)
        return True

    def clear(self):
        self._timestamps.clear()


class SessionState:
    """Complete mutable state for one client session.

    This is the central data store that the Control Flow / Scheduling
    module reads and writes.  All state transitions go through this
    object, making it easy to inspect, serialise, or reset.

    Attributes
    ----------
    pipeline_state : str
        Current state of the pipeline: idle | listening | thinking | speaking.
    interrupt : InterruptState
        All interrupt-related flags and counters.
    audio_buffer : AudioBuffer
        Accumulated VAD utterance fragments.
    language : LanguageTracker
        Recent language detection history.
    rate_limiter : RateLimiter
        API call rate limiting.
    ltm : LongTermMemory or None
        Persistent conversational memory.
    models_ready : bool
        Whether all ML models have finished loading.
    """

    def __init__(self, accumulation_delay: float = 3.0, rate_limit: int = 25):
        self.pipeline_state: str = "idle"
        self.interrupt = InterruptState()
        self.audio_buffer = AudioBuffer(delay=accumulation_delay)
        self.language = LanguageTracker()
        self.rate_limiter = RateLimiter(max_requests=rate_limit)
        self.ltm: Optional[LongTermMemory] = None
        self.models_ready: bool = False

        # Ghost texting counters
        self.ghost_counter: int = 0
        self.ghost_interval: int = 25

    # ── State transitions ──────────────────────────────────────────────

    def set_state(self, new_state: str):
        """Transition to a new pipeline state (with validation)."""
        assert new_state in VALID_STATES, f"Invalid state: {new_state}"
        self.pipeline_state = new_state

    # ── Full session reset ─────────────────────────────────────────────

    def reset(self):
        """Reset all session state (e.g., on "clear" command)."""
        self.pipeline_state = "idle"
        self.interrupt.clear()
        self.audio_buffer.cancel_timer()
        self.audio_buffer.clear()
        self.ghost_counter = 0
