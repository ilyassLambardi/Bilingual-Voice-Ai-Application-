"""
Voice Activity Detection — Silero VAD (streaming mode).

Processes 512-sample chunks (32 ms @ 16 kHz) and emits speech
start / end events.  Everything stays in RAM — no file I/O.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch


class VADProcessor:
    """Streaming VAD using Silero VAD from torch.hub."""

    # Class-level model cache — loaded once, shared across all sessions
    _shared_model = None

    def __init__(
        self,
        threshold: float = 0.45,
        min_speech_ms: int = 250,
        min_silence_ms: int = 600,
        sample_rate: int = 16_000,
    ):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_speech_samples = int(min_speech_ms * sample_rate / 1000)
        self.min_silence_samples = int(min_silence_ms * sample_rate / 1000)

        # Load Silero VAD — cached at class level for multi-session
        if VADProcessor._shared_model is None:
            print("[VAD] Loading Silero VAD ...")
            _local_jit = Path(__file__).resolve().parent.parent.parent / "models" / "silero_vad.jit"
            if _local_jit.exists():
                print(f"[VAD] Using local: {_local_jit.name}")
                VADProcessor._shared_model = torch.jit.load(str(_local_jit))
            else:
                VADProcessor._shared_model, _utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,
                )
            VADProcessor._shared_model.eval()
            print("[VAD] Ready.")
        else:
            print("[VAD] Using cached model (shared).")
        self.model = VADProcessor._shared_model

        # State
        self._in_speech = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._buffer: list[np.ndarray] = []

    # ── Public API ────────────────────────────────────────────────────────

    def process_chunk(
        self, chunk: np.ndarray
    ) -> tuple[bool, Optional[np.ndarray]]:
        """Feed a 512-sample int16/float32 chunk.

        Returns
        -------
        (is_speaking, completed_utterance)
            is_speaking : True while user is talking.
            completed_utterance : numpy float32 array of the full
                                  utterance once silence is confirmed,
                                  else None.
        """
        # Ensure float32 in [-1, 1]
        if chunk.dtype == np.int16:
            chunk_f = chunk.astype(np.float32) / 32768.0
        else:
            chunk_f = chunk.astype(np.float32)

        tensor = torch.from_numpy(chunk_f)
        prob = self.model(tensor, self.sample_rate).item()

        if prob >= self.threshold:
            # ── Speech detected ──────────────────────────────────────
            self._silence_samples = 0
            self._buffer.append(chunk_f)
            self._speech_samples += len(chunk_f)

            if not self._in_speech and self._speech_samples >= self.min_speech_samples:
                self._in_speech = True

            return self._in_speech, None

        else:
            # ── Silence ──────────────────────────────────────────────
            if self._in_speech:
                self._silence_samples += len(chunk_f)
                self._buffer.append(chunk_f)  # keep trailing silence

                if self._silence_samples >= self.min_silence_samples:
                    # Utterance complete
                    utterance = np.concatenate(self._buffer)
                    self.reset()
                    return False, utterance

                return True, None  # still waiting for more silence

            # Not in speech — discard
            self._speech_samples = 0
            self._buffer.clear()
            return False, None

    def reset(self):
        """Clear all state for a fresh utterance."""
        self._in_speech = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._buffer.clear()
        self.model.reset_states()

    @property
    def is_speaking(self) -> bool:
        return self._in_speech
