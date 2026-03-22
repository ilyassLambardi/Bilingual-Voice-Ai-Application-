"""
Module 2 (Processing/Core): ASR — openai-whisper with local large-v3-turbo model, or faster-whisper fallback.

Transcribes float32 numpy audio → text.  No file I/O.
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .hallucination_filter import filter_hallucination, VALID_SHORT

try:
    import whisper
except ImportError:
    whisper = None  # only needed for local .pt fallback

_pool = ThreadPoolExecutor(max_workers=2)

ALLOWED_LANGUAGES = {"en", "de"}

# ── Text-based German detection (more reliable than Whisper's lang prob) ──
_DE_CHARS = set("äöüß")
_DE_WORDS = {
    "ich", "du", "er", "sie", "es", "wir", "ihr", "und", "oder", "aber",
    "das", "der", "die", "ein", "eine", "ist", "sind", "war", "hat",
    "haben", "wird", "mit", "von", "zu", "auf", "nicht", "auch", "noch",
    "dass", "wenn", "weil", "dann", "schon", "sehr", "hier", "dort",
    "kann", "muss", "soll", "will", "denn", "nur", "mehr", "wie",
    "über", "unter", "nach", "vor", "für", "immer", "hallo", "danke",
    "bitte", "guten", "morgen", "abend", "nacht", "geht", "gut",
    "ja", "nein", "heute", "gestern", "morgen", "warum", "was",
    "wo", "wer", "wann", "dir", "mir", "mich", "dich", "uns",
    "bin", "bist", "wirklich", "gerade", "eigentlich", "natürlich",
    "vielleicht", "alles", "nichts", "müde", "freut", "schlecht",
}


def _detect_lang_from_text(text: str) -> str:
    """Detect language from transcribed text using characters + keywords.

    English is the strong default.  Only returns 'de' when the text
    is *clearly* German (special chars OR >= 40 % German keywords).
    """
    lower = text.lower()
    words = set(lower.split())
    # German-specific characters are a strong signal, but require
    # at least one German keyword too (avoids false positives on names)
    has_de_chars = any(ch in lower for ch in _DE_CHARS)
    de_hits = len(words & _DE_WORDS)
    if has_de_chars and de_hits >= 1:
        return "de"
    # Pure keyword ratio — must be clearly German (>= 40 %)
    if len(words) >= 2 and de_hits / len(words) >= 0.4:
        return "de"
    return "en"

# Path to local .pt model (if present)
_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class ASRProcessor:
    """Async-friendly Whisper ASR.  EN + DE only.

    Prefers local .pt model from models/ folder (large-v3-turbo),
    falls back to faster-whisper base if not found.
    """

    # Hallucination filtering is now handled by the shared
    # hallucination_filter module — see pipeline/hallucination_filter.py

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 3,
        language: Optional[str] = None,
    ):
        self.beam_size = beam_size
        self.language = language
        self._backend = None  # "whisper" or "faster-whisper"
        self.model = None

        # Use faster-whisper (CTranslate2) as primary
        try:
            from faster_whisper import WhisperModel
            print(f"[ASR] Loading faster-whisper ({model_size}, {compute_type}) on {device} ...", flush=True)
            try:
                self.model = WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=8)
                self._device = device
            except Exception as gpu_err:
                if device == "cuda":
                    print(f"[ASR] CUDA failed ({gpu_err}), falling back to CPU ...", flush=True)
                    self.model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=8)
                    self._device = "cpu"
                else:
                    raise
            self._backend = "faster-whisper"
            print(f"[ASR] Ready -- faster-whisper {model_size} on {self._device}.", flush=True)
        except Exception as e:
            # Fallback to openai-whisper with local .pt
            print(f"[ASR] faster-whisper failed ({e}), trying local .pt ...", flush=True)
            local_pt = self._find_local_model()
            if local_pt:
                self.model = whisper.load_model(str(local_pt), device=device)
                self._backend = "whisper"
                print(f"[ASR] Ready -- {local_pt.name} on {device} (openai-whisper).", flush=True)
            else:
                raise RuntimeError("No ASR model available")

        print(f"[ASR] Languages: EN + DE only.", flush=True)

    @staticmethod
    def _find_local_model() -> Optional[Path]:
        """Find the best .pt whisper model in the models/ folder."""
        if not _MODELS_DIR.exists():
            return None
        # Prefer larger models
        priority = ["large-v3-turbo", "large-v3", "large", "medium", "small", "base", "tiny"]
        pt_files = list(_MODELS_DIR.glob("*.pt"))
        for name in priority:
            for f in pt_files:
                if name in f.stem:
                    return f
        # Return any .pt file
        return pt_files[0] if pt_files else None

    # ── Async entry point ─────────────────────────────────────────────────

    async def transcribe(self, audio: np.ndarray) -> dict:
        """Transcribe audio (float32, 16 kHz) -> {text, language}."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_pool, self._transcribe_sync, audio)

    async def transcribe_fast(self, audio: np.ndarray) -> dict:
        """Fast single-pass transcription (for ghost texting, no dual-pass)."""
        loop = asyncio.get_running_loop()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return await loop.run_in_executor(_pool, self._transcribe_fast, audio)

    # ── Sync workers ──────────────────────────────────────────────────────

    def _transcribe_sync(self, audio: np.ndarray) -> dict:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Pre-check: reject audio that's too short or too quiet
        duration = len(audio) / 16000.0
        if duration < 0.3:
            print(f"[ASR] Audio too short ({duration:.2f}s), skipping")
            return {"text": "", "language": "en", "language_prob": 0.0}

        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.003:
            print(f"[ASR] Audio too quiet (RMS={rms:.4f}), skipping")
            return {"text": "", "language": "en", "language_prob": 0.0}

        if self._backend == "whisper":
            return self._transcribe_whisper(audio)
        else:
            return self._transcribe_faster_whisper(audio)

    def _transcribe_whisper(self, audio: np.ndarray) -> dict:
        """Transcribe using openai-whisper (local .pt model)."""
        # Detect language first
        mel = whisper.log_mel_spectrogram(
            whisper.pad_or_trim(audio), n_mels=self.model.dims.n_mels
        ).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        detected = max(probs, key=probs.get)
        lang_prob = probs.get(detected, 0.0)

        # Restrict to EN + DE
        if detected not in ALLOWED_LANGUAGES:
            detected = "en"

        # Transcribe with forced language
        # Use beam_size=1 (greedy) for speed on CPU with large models
        result = self.model.transcribe(
            audio,
            language=detected,
            beam_size=1,
            without_timestamps=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.5,
            logprob_threshold=-0.8,
            fp16=False,
        )

        text = result["text"].strip()
        duration = len(audio) / 16000.0

        # Filter hallucinations and short noise
        text = self._filter_text(text, duration_s=duration)

        return {
            "text": text,
            "language": detected,
            "language_prob": round(lang_prob, 3),
        }

    def _transcribe_faster_whisper(self, audio: np.ndarray) -> dict:
        """Transcribe using faster-whisper — bilingual EN/DE optimized.

        Uses initial_prompt to prime Whisper for bilingual detection.
        Falls back to forced-DE pass if auto-detect confidence is very low.
        """
        # Default to English — only switch to German based on
        # text-based post-detection (more reliable than auto-detect)
        segments, info = self.model.transcribe(
            audio,
            beam_size=self.beam_size,
            language="en",  # English default
            vad_filter=True,
            without_timestamps=True,
            no_speech_threshold=0.4,
            log_prob_threshold=-1.0,
            condition_on_previous_text=False,
        )
        parts = []
        nsp_values = []
        for seg in segments:
            nsp_values.append(seg.no_speech_prob)
            if seg.no_speech_prob < 0.7:
                parts.append(seg.text)
        text = " ".join(parts).strip()
        avg_nsp = sum(nsp_values) / len(nsp_values) if nsp_values else 0.0
        whisper_lang = info.language if info.language in ALLOWED_LANGUAGES else "en"
        lang_prob = info.language_probability
        duration = len(audio) / 16000.0

        # Text-based language detection — only switch to German if clearly German
        text_lang = _detect_lang_from_text(text)
        if text_lang == "de":
            # Re-transcribe with German for better accuracy
            print(f"[ASR] German detected in text, re-transcribing with lang=de")
            try:
                de_segments, de_info = self.model.transcribe(
                    audio,
                    beam_size=self.beam_size,
                    language="de",
                    vad_filter=True,
                    without_timestamps=True,
                    no_speech_threshold=0.4,
                    log_prob_threshold=-1.0,
                    condition_on_previous_text=False,
                )
                de_parts = []
                for seg in de_segments:
                    if seg.no_speech_prob < 0.7:
                        de_parts.append(seg.text)
                de_text = " ".join(de_parts).strip()
                if de_text:
                    text = de_text
            except Exception as e:
                print(f"[ASR] German re-transcribe failed ({e}), using English result")
            detected = "de"
            print(f"[ASR] DE (text-detect)")
        else:
            detected = "en"
            print(f"[ASR] EN (default)")

        text = self._filter_text(text, no_speech_prob=avg_nsp, duration_s=duration)
        return {"text": text, "language": detected, "language_prob": round(lang_prob, 3)}

    def _transcribe_fast(self, audio: np.ndarray) -> dict:
        """Fast single-pass transcription for ghost texting (no dual-pass)."""
        if self._backend == "whisper":
            return self._transcribe_whisper(audio)
        segments, info = self.model.transcribe(
            audio,
            beam_size=1,
            language="en",  # English default for ghost text too
            vad_filter=False,
            without_timestamps=True,
            no_speech_threshold=0.5,
            log_prob_threshold=-0.8,
            condition_on_previous_text=False,
        )
        nsp_vals = []
        parts = []
        for seg in segments:
            nsp_vals.append(seg.no_speech_prob)
            if seg.no_speech_prob < 0.7:
                parts.append(seg.text)
        detected = info.language if info.language in ALLOWED_LANGUAGES else "en"
        avg_nsp = sum(nsp_vals) / len(nsp_vals) if nsp_vals else 0.0
        text = self._filter_text(" ".join(parts).strip(), no_speech_prob=avg_nsp)
        return {"text": text, "language": detected, "language_prob": 0.0}

    @staticmethod
    def _filter_text(
        text: str,
        no_speech_prob: float = 0.0,
        log_prob: float = 0.0,
        duration_s: float = 0.0,
    ) -> str:
        """Filter hallucinations using the shared advanced filter."""
        return filter_hallucination(
            text,
            no_speech_prob=no_speech_prob if no_speech_prob else None,
            log_prob=log_prob if log_prob else None,
            duration_s=duration_s if duration_s else None,
        )
