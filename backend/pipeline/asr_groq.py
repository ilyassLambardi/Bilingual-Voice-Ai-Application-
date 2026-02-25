"""
ASR provider — Groq Whisper API (large-v3).

Uses Groq's hosted Whisper large-v3 for transcription.
Free tier, extremely fast (~0.3s for most utterances), near-perfect accuracy.

Same async interface as the local ASR provider.
"""

import asyncio
import io
import os
import struct
import wave
from typing import Optional

import numpy as np

ALLOWED_LANGUAGES = {"en", "de"}

# ── Text-based German detection (backup for language routing) ──
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
    if any(ch in lower for ch in _DE_CHARS):
        return "de"
    words = set(lower.split())
    de_hits = len(words & _DE_WORDS)
    if len(words) > 0 and de_hits / len(words) >= 0.25:
        return "de"
    return "en"


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert float32 numpy audio to WAV bytes for API upload."""
    if audio.dtype == np.float32:
        pcm = (audio * 32767).astype(np.int16)
    else:
        pcm = audio.astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# Common hallucination patterns
_HALLUCINATIONS = {
    "thank you", "thanks for watching", "subscribe",
    "like and subscribe", "bye", "you", "the",
    ".", "..", "...", "um", "uh",
    "thanks", "thank you for watching",
    "danke", "danke schön", "tschüss",
}


class GroqASR:
    """Async ASR via Groq Whisper API — large-v3, bilingual EN+DE."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-large-v3",
    ):
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self._api_key or self._api_key.startswith("gsk_your"):
            raise ValueError(
                "GROQ_API_KEY not set. Get one free at https://console.groq.com/keys"
            )

        from groq import Groq
        self._client = Groq(api_key=self._api_key)
        self._model = model
        print(f"[ASR] Groq Whisper API -> {model}")
        print(f"[ASR] Languages: EN + DE only.")

    async def transcribe(self, audio: np.ndarray) -> dict:
        """Transcribe float32 16kHz audio -> {text, language, language_prob}."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> dict:
        """Synchronous transcription via Groq API."""
        import time
        t0 = time.time()

        # Convert numpy audio to WAV bytes
        wav_bytes = _audio_to_wav_bytes(audio, sample_rate=16000)

        try:
            # Groq Whisper API call
            transcription = self._client.audio.transcriptions.create(
                file=("audio.wav", wav_bytes),
                model=self._model,
                language=None,  # auto-detect
                response_format="verbose_json",
                prompt="Hello, how are you? Hallo, wie geht es dir?",
            )

            text = transcription.text.strip() if transcription.text else ""
            # Get detected language from API response
            api_lang = getattr(transcription, 'language', 'en') or 'en'

            # Map to allowed languages
            if api_lang not in ALLOWED_LANGUAGES:
                api_lang = "en"

            # Override with text-based detection for reliability
            text_lang = _detect_lang_from_text(text)
            if text_lang != api_lang:
                detected = text_lang
                print(f"[ASR] {detected.upper()} (text-detect override, API said {api_lang.upper()})")
            else:
                detected = api_lang
                print(f"[ASR] {detected.upper()} (API)")

            # Filter hallucinations
            text = self._filter_text(text)

            return {
                "text": text,
                "language": detected,
                "language_prob": 0.95,  # API doesn't return confidence, assume high
            }

        except Exception as e:
            print(f"[ASR] Groq API error: {e}")
            return {"text": "", "language": "en", "language_prob": 0.0}

    @staticmethod
    def _filter_text(text: str) -> str:
        """Filter hallucinations and noise."""
        if text.lower().strip("., ") in _HALLUCINATIONS:
            print(f"[ASR] Filtered hallucination: '{text}'")
            return ""
        if len(text.split()) < 2 and len(text) < 5:
            print(f"[ASR] Filtered too-short: '{text}'")
            return ""
        return text
