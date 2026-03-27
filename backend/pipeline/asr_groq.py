"""
Module 2 (Processing/Core): ASR provider — Groq Whisper API (large-v3).

Uses Groq's hosted Whisper large-v3 for transcription.
Free tier, extremely fast (~0.3s for most utterances), near-perfect accuracy.

Same async interface as the local ASR provider.
"""

import asyncio
import io
import logging
import os
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from .hallucination_filter import filter_hallucination, VALID_SHORT

log = logging.getLogger("s2s.asr")
_pool = ThreadPoolExecutor(max_workers=2)

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
    "ja", "nein", "heute",  "gestern", "morgen", "warum", "was",
    "wo", "wer", "wann", "dir", "mir", "mich", "dich", "uns",
    "bin", "bist", "wirklich", "gerade", "eigentlich", "natürlich",
    "vielleicht", "alles", "nichts", "müde", "freut", "schlecht", "wie",
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
        return await loop.run_in_executor(_pool, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> dict:
        """Synchronous transcription via Groq API with retry."""
        # ── Pre-check: reject audio that's too short or too quiet ──
        duration = len(audio) / 16000.0
        if duration < 0.3:
            log.info(f"[ASR] Audio too short ({duration:.2f}s), skipping")
            return {"text": "", "language": "en", "language_prob": 0.0}

        rms = float(np.sqrt(np.mean(audio ** 2)))
        print(f"[ASR] Input: {duration:.2f}s, RMS={rms:.4f}, dtype={audio.dtype}")
        if rms < 0.001:
            print(f"[ASR] Audio too quiet (RMS={rms:.4f}), skipping")
            return {"text": "", "language": "en", "language_prob": 0.0}

        wav_bytes = _audio_to_wav_bytes(audio, sample_rate=16000)
        max_retries = 2

        for attempt in range(max_retries + 1):
            try:
                # Default to English — only switch to German based on
                # text-based post-detection (more reliable than Whisper's
                # auto-detect which often misclassifies short English)
                transcription = self._client.audio.transcriptions.create(
                    file=("audio.wav", wav_bytes),
                    model=self._model,
                    language="en",  # English default
                    response_format="verbose_json",
                )

                # ── Parse segments for no_speech_prob filtering ──
                segments = getattr(transcription, 'segments', None)
                avg_nsp = 0.0
                if segments and isinstance(segments, list):
                    # Filter out segments with high no_speech_prob
                    good_parts = []
                    nsp_values = []
                    for seg in segments:
                        nsp = getattr(seg, 'no_speech_prob', 0.0)
                        seg_text = getattr(seg, 'text', '').strip()
                        nsp_values.append(nsp)
                        if nsp < 0.6 and seg_text:
                            good_parts.append(seg_text)
                        else:
                            log.info(f"[ASR] Dropped segment (no_speech={nsp:.2f}): '{seg_text}'")
                    text = " ".join(good_parts).strip()
                    avg_nsp = sum(nsp_values) / len(nsp_values) if nsp_values else 0.0
                else:
                    # Fallback: use full text
                    text = transcription.text.strip() if transcription.text else ""

                # Text-based language detection (more reliable than Whisper auto)
                text_lang = _detect_lang_from_text(text)

                # If text is clearly German but we transcribed as English,
                # re-transcribe with language="de" for better accuracy.
                # Only re-transcribe if we have strong German signals (avoid
                # doubling API calls on false positives).
                words = set(text.lower().split())
                de_word_count = len(words & _DE_WORDS)
                if text_lang == "de" and de_word_count >= 2:
                    log.info(f"[ASR] German detected ({de_word_count} keywords), re-transcribing with lang=de")
                    try:
                        de_transcription = self._client.audio.transcriptions.create(
                            file=("audio.wav", wav_bytes),
                            model=self._model,
                            language="de",
                            response_format="verbose_json",
                        )
                        de_segments = getattr(de_transcription, 'segments', None)
                        if de_segments and isinstance(de_segments, list):
                            de_parts = []
                            for seg in de_segments:
                                nsp = getattr(seg, 'no_speech_prob', 0.0)
                                seg_text = getattr(seg, 'text', '').strip()
                                if nsp < 0.6 and seg_text:
                                    de_parts.append(seg_text)
                            de_text = " ".join(de_parts).strip()
                        else:
                            de_text = de_transcription.text.strip() if de_transcription.text else ""
                        if de_text:
                            text = de_text
                    except Exception as de_err:
                        log.warning(f"[ASR] German re-transcribe failed ({de_err}), using English result")
                    detected = "de"
                    log.info(f"[ASR] DE (text-detect)")
                else:
                    detected = "en"
                    log.info(f"[ASR] EN (default)")

                raw_text = text
                text = self._filter_text(text, no_speech_prob=avg_nsp, duration_s=duration)
                if raw_text and not text:
                    print(f"[ASR] Hallucination filter removed: '{raw_text}' (nsp={avg_nsp:.2f})")
                elif text:
                    print(f"[ASR] Result [{detected}]: '{text}'")

                return {
                    "text": text,
                    "language": detected,
                    "language_prob": 0.95,
                }

            except Exception as e:
                is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                if attempt < max_retries and is_rate_limit:
                    wait = 1.0 * (2 ** attempt)
                    log.warning(f"[ASR] Groq rate-limited, retry {attempt+1} in {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    log.error(f"[ASR] Groq API error: {e}")
                    return {"text": "", "language": "en", "language_prob": 0.0}

        return {"text": "", "language": "en", "language_prob": 0.0}

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
