"""
ASR — openai-whisper with local large-v3-turbo model, or faster-whisper fallback.

Transcribes float32 numpy audio → text.  No file I/O.
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import whisper

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
    """Detect language from transcribed text using characters + keywords."""
    lower = text.lower()
    # German-specific characters are a very strong signal
    if any(ch in lower for ch in _DE_CHARS):
        return "de"
    words = set(lower.split())
    de_hits = len(words & _DE_WORDS)
    if len(words) > 0 and de_hits / len(words) >= 0.25:
        return "de"
    return "en"

# Path to local .pt model (if present)
_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class ASRProcessor:
    """Async-friendly Whisper ASR.  EN + DE only.

    Prefers local .pt model from models/ folder (large-v3-turbo),
    falls back to faster-whisper base if not found.
    """

    # Common Whisper hallucination patterns (noise → fake text)
    _HALLUCINATIONS = {
        "thank you", "thanks for watching", "subscribe",
        "like and subscribe", "bye", "you", "the",
        ".", "..", "...", "um", "uh",
        "thanks", "thank you for watching",
        "danke", "danke schön", "tschüss",
    }

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
            print(f"[ASR] Ready — faster-whisper {model_size} on {self._device}.", flush=True)
        except Exception as e:
            # Fallback to openai-whisper with local .pt
            print(f"[ASR] faster-whisper failed ({e}), trying local .pt ...", flush=True)
            local_pt = self._find_local_model()
            if local_pt:
                self.model = whisper.load_model(str(local_pt), device=device)
                self._backend = "whisper"
                print(f"[ASR] Ready — {local_pt.name} on {device} (openai-whisper).", flush=True)
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

        # Filter hallucinations and short noise
        text = self._filter_text(text)

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
        # Bilingual prompt primes the model for EN+DE detection
        _BILINGUAL_PROMPT = (
            "Hello, how are you? Hallo, wie geht es dir? "
            "That's interesting. Das ist interessant. "
            "I think so too. Ich denke auch so."
        )

        segments, info = self.model.transcribe(
            audio,
            beam_size=self.beam_size,
            language=None,
            initial_prompt=_BILINGUAL_PROMPT,
            vad_filter=True,
            without_timestamps=True,
            no_speech_threshold=0.4,
            log_prob_threshold=-1.0,
            condition_on_previous_text=True,
        )
        parts = []
        for seg in segments:
            if seg.no_speech_prob < 0.7:
                parts.append(seg.text)
        text = " ".join(parts).strip()
        whisper_lang = info.language if info.language in ALLOWED_LANGUAGES else "en"
        lang_prob = info.language_probability

        # Override with text-based detection (much more reliable for bilingual)
        text_lang = _detect_lang_from_text(text)
        if text_lang != whisper_lang:
            detected = text_lang
            print(f"[ASR] {detected.upper()} (text-detect override, whisper said {whisper_lang.upper()} {lang_prob:.0%})")
        else:
            detected = whisper_lang
            print(f"[ASR] {detected.upper()} (conf={lang_prob:.0%})")

        text = self._filter_text(text)
        return {"text": text, "language": detected, "language_prob": round(lang_prob, 3)}

    def _transcribe_fast(self, audio: np.ndarray) -> dict:
        """Fast single-pass transcription for ghost texting (no dual-pass)."""
        if self._backend == "whisper":
            return self._transcribe_whisper(audio)
        segments, info = self.model.transcribe(
            audio,
            beam_size=1,
            language=None,
            vad_filter=False,
            without_timestamps=True,
            no_speech_threshold=0.5,
            log_prob_threshold=-0.8,
            condition_on_previous_text=False,
        )
        parts = [seg.text for seg in segments if seg.no_speech_prob < 0.7]
        detected = info.language if info.language in ALLOWED_LANGUAGES else "en"
        text = self._filter_text(" ".join(parts).strip())
        return {"text": text, "language": detected, "language_prob": 0.0}

    def _filter_text(self, text: str) -> str:
        """Filter hallucinations and noise."""
        if text.lower().strip("., ") in self._HALLUCINATIONS:
            print(f"[ASR] Filtered hallucination: '{text}'")
            return ""
        if len(text.split()) < 2 and len(text) < 5:
            print(f"[ASR] Filtered too-short: '{text}'")
            return ""
        return text
