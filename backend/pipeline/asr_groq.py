"""
ASR provider — Groq Whisper API (large-v3).

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


# ── Consolidated hallucination patterns (lowercased, stripped) ──────────
_HALLUCINATIONS = {
    # English Whisper noise hallucinations
    "thank you", "thanks", "thanks for watching", "thank you for watching",
    "subscribe", "like and subscribe", "like subscribe",
    "please subscribe", "hit the bell",
    "bye", "goodbye", "you", "the", "the end",
    "um", "uh", "hmm", "huh",
    "oh", "ah",
    "subtitles by", "amara.org", "subtitles made by",
    "copyright", "all rights reserved",
    "music", "applause", "laughter",
    # German Whisper noise hallucinations
    "danke", "danke schön", "danke schon", "tschüss", "tschuss",
    "wie geht's die", "wie gehts die", "wie geht es dir",
    "wie geht's", "wie gehts", "hallo wie geht's",
    "guten tag", "auf wiedersehen", "bis bald",
    "vielen dank", "herzlich willkommen",
    "untertitel von", "untertitelung", "untertitel",
    "musik",
}

# Substrings that indicate hallucination (catches partial matches)
_HALLUCINATION_SUBSTRINGS = [
    "thanks for watching", "thank you for watching",
    "subscribe", "subtitles by", "amara.org",
    "untertitel", "copyright",
    "please like", "hit the bell",
]

# Short words that are valid conversational input (never filter these)
_VALID_SHORT = {
    "yes", "no", "ok", "okay", "hi", "hey", "why", "how",
    "what", "when", "who", "where", "help", "stop", "go",
    "ja", "nein", "gut", "naja", "ach", "doch", "klar",
    "wow", "cool", "nice", "sure", "fine", "yep", "nah",
    "hallo", "hello", "bitte", "genau",
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
        return await loop.run_in_executor(_pool, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> dict:
        """Synchronous transcription via Groq API with retry."""
        # ── Pre-check: reject audio that's too short or too quiet ──
        duration = len(audio) / 16000.0
        if duration < 0.3:
            log.info(f"[ASR] Audio too short ({duration:.2f}s), skipping")
            return {"text": "", "language": "en", "language_prob": 0.0}

        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.003:
            log.info(f"[ASR] Audio too quiet (RMS={rms:.4f}), skipping")
            return {"text": "", "language": "en", "language_prob": 0.0}

        wav_bytes = _audio_to_wav_bytes(audio, sample_rate=16000)
        max_retries = 2

        for attempt in range(max_retries + 1):
            try:
                # No prompt — avoids Whisper hallucinating the prompt itself
                transcription = self._client.audio.transcriptions.create(
                    file=("audio.wav", wav_bytes),
                    model=self._model,
                    language=None,  # auto-detect
                    response_format="verbose_json",
                )

                # ── Parse segments for no_speech_prob filtering ──
                segments = getattr(transcription, 'segments', None)
                if segments and isinstance(segments, list):
                    # Filter out segments with high no_speech_prob
                    good_parts = []
                    for seg in segments:
                        nsp = getattr(seg, 'no_speech_prob', 0.0)
                        seg_text = getattr(seg, 'text', '').strip()
                        if nsp < 0.6 and seg_text:
                            good_parts.append(seg_text)
                        else:
                            log.info(f"[ASR] Dropped segment (no_speech={nsp:.2f}): '{seg_text}'")
                    text = " ".join(good_parts).strip()
                else:
                    # Fallback: use full text
                    text = transcription.text.strip() if transcription.text else ""

                api_lang = getattr(transcription, 'language', 'en') or 'en'

                if api_lang not in ALLOWED_LANGUAGES:
                    api_lang = "en"

                text_lang = _detect_lang_from_text(text)
                if text_lang != api_lang:
                    detected = text_lang
                    log.info(f"[ASR] {detected.upper()} (text-detect override, API said {api_lang.upper()})")
                else:
                    detected = api_lang
                    log.info(f"[ASR] {detected.upper()} (API)")

                text = self._filter_text(text)

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
    def _filter_text(text: str) -> str:
        """Filter hallucinations and noise — aggressive but safe."""
        raw_lower = text.lower().strip()

        # Catch pure punctuation / dots
        if not raw_lower or all(ch in '., !?…-–—' for ch in raw_lower):
            log.info(f"[ASR] Filtered noise/punctuation: '{text}'")
            return ""

        cleaned = raw_lower.strip("., !?…-–—")
        if not cleaned:
            log.info(f"[ASR] Filtered empty after strip: '{text}'")
            return ""

        # Exact hallucination match
        if cleaned in _HALLUCINATIONS:
            log.info(f"[ASR] Filtered hallucination: '{text}'")
            return ""

        # Substring hallucination match (catches "Thanks for watching everyone!")
        for sub in _HALLUCINATION_SUBSTRINGS:
            if sub in cleaned:
                log.info(f"[ASR] Filtered hallucination substring '{sub}' in: '{text}'")
                return ""

        # Repetition detection: noise produces "Thank you. Thank you. Thank you."
        words = cleaned.split()
        if len(words) >= 3:
            unique = set(words)
            # All same word
            if len(unique) == 1:
                log.info(f"[ASR] Filtered repetition: '{text}'")
                return ""
            # Mostly same word (>70%)
            from collections import Counter
            most_common_count = Counter(words).most_common(1)[0][1]
            if most_common_count / len(words) > 0.7:
                log.info(f"[ASR] Filtered dominant repetition: '{text}'")
                return ""

        # Repeated sentence pattern: "Thank you. Thank you."
        sentences = [s.strip().strip('.,!?').lower() for s in text.split('.') if s.strip()]
        if len(sentences) >= 2:
            unique_sentences = set(sentences)
            if len(unique_sentences) == 1 and sentences[0] in _HALLUCINATIONS:
                log.info(f"[ASR] Filtered repeated hallucination sentence: '{text}'")
                return ""

        # Too-short filter: block ≤2 char single words UNLESS they're known valid
        if len(words) == 1 and len(cleaned) <= 2 and cleaned not in _VALID_SHORT:
            log.info(f"[ASR] Filtered too-short: '{text}'")
            return ""

        return text
