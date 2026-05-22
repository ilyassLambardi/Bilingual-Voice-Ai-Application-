"""
Module 2 (Processing/Core): ASR provider — Groq Whisper API (large-v3).

Uses Groq's hosted Whisper large-v3 for transcription.
Free tier, extremely fast (~0.3s for most utterances), near-perfect accuracy.

Same async interface as the local ASR provider.
"""

import asyncio
import functools
import io
import logging
import os
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import numpy as np
from .hallucination_filter import filter_hallucination
from .lang_detect import detect_lang_from_text as _detect_lang_from_text

log = logging.getLogger("s2s.asr")
_pool = ThreadPoolExecutor(max_workers=2)

ALLOWED_LANGUAGES = {"en", "de"}

# ── Whisper language name → ISO code mapping ──
_LANG_MAP = {
    "english": "en", "german": "de", "deutsch": "de",
    "en": "en", "de": "de",
}

def _normalize_lang(raw: str) -> str:
    """Normalize Whisper language output to ISO 639-1 code.
    
    Groq/Whisper may return full names like 'german' or codes like 'de'.
    """
    return _LANG_MAP.get(raw.lower().strip(), raw.lower().strip())


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
        log.info("[ASR] Groq Whisper API -> %s", model)
        log.info("[ASR] Languages: EN + DE only.")

    async def transcribe(self, audio: np.ndarray, lang_hint: Optional[str] = None) -> dict:
        """Transcribe float32 16kHz audio -> {text, language, language_prob}.
        
        Parameters
        ----------
        lang_hint : str or None
            If 'de' or 'en', force Groq Whisper to use that language.
            If None, let Whisper auto-detect (may fail on short utterances).
        """
        loop = asyncio.get_running_loop()
        fn = functools.partial(self._transcribe_sync, audio, lang_hint=lang_hint)
        return await loop.run_in_executor(_pool, fn)

    def _transcribe_sync(self, audio: np.ndarray, lang_hint: Optional[str] = None) -> dict:
        """Synchronous transcription via Groq API with retry."""
        # ── Pre-check: reject audio that's too short or too quiet ──
        duration = len(audio) / 16000.0
        if duration < 0.3:
            log.info(f"[ASR] Audio too short ({duration:.2f}s), skipping")
            return {"text": "", "language": "en", "language_prob": 0.0}

        rms = float(np.sqrt(np.mean(audio ** 2)))
        log.info("[ASR] Input: %.2fs, RMS=%.4f, dtype=%s, hint=%s", duration, rms, audio.dtype, lang_hint)
        if rms < 0.001:
            log.info("[ASR] Audio too quiet (RMS=%.4f), skipping", rms)
            return {"text": "", "language": "en", "language_prob": 0.0}

        wav_bytes = _audio_to_wav_bytes(audio, sample_rate=16000)
        max_retries = 2

        # Normalize the language hint
        effective_hint = _normalize_lang(lang_hint) if lang_hint else None
        if effective_hint and effective_hint not in ALLOWED_LANGUAGES:
            effective_hint = None  # unknown hint, fall back to auto

        for attempt in range(max_retries + 1):
            try:
                # Build API call kwargs
                api_kwargs = {
                    "file": ("audio.wav", wav_bytes),
                    "model": self._model,
                    "response_format": "verbose_json",
                }
                # If user explicitly selected a language, tell Whisper
                if effective_hint:
                    api_kwargs["language"] = effective_hint
                    log.info("[ASR] Using forced language: %s", effective_hint)

                transcription = self._client.audio.transcriptions.create(**api_kwargs)

                # ── Always use transcription.text as primary source ──
                # Groq Whisper often returns empty segment texts but populates
                # the top-level text field correctly.
                text = transcription.text.strip() if transcription.text else ""
                raw_lang = getattr(transcription, 'language', 'en') or 'en'
                whisper_lang = _normalize_lang(raw_lang)
                log.info("[ASR] Groq raw text: '%s' (whisper_lang=%s, raw=%s)", text, whisper_lang, raw_lang)

                # Use segments only for no_speech_prob validation
                segments = getattr(transcription, 'segments', None)
                avg_nsp = 0.0
                if segments and isinstance(segments, list):
                    nsp_values = [getattr(seg, 'no_speech_prob', 0.0) for seg in segments]
                    avg_nsp = sum(nsp_values) / len(nsp_values) if nsp_values else 0.0
                    # Reject only if ALL segments have very high no_speech_prob
                    if avg_nsp > 0.85 and text:
                        log.info("[ASR] Rejected: avg no_speech_prob=%.2f too high", avg_nsp)
                        text = ""

                # ── Bilingual language decision ──────────────────────────
                # Whisper auto-detect is unreliable for German: it often
                # returns "russian", "english" (hallucinated), or other
                # wrong languages. Strategy:
                #   1. If auto-detect says DE → accept German immediately.
                #   2. Otherwise → cross-check with a forced-DE pass.
                #      If the DE text reads as German → it was German.
                #      If not → keep the auto-detect result.
                text_lang = _detect_lang_from_text(text)

                if whisper_lang == 'de' or text_lang == 'de':
                    # Auto-detect or text clearly German — accept
                    detected = 'de'
                    # If auto didn't use German decoding, re-transcribe
                    if whisper_lang != 'de' and text:
                        log.info("[ASR] Text-detected German, re-transcribing with lang=de")
                        try:
                            de_tr = self._client.audio.transcriptions.create(
                                file=("audio.wav", wav_bytes),
                                model=self._model,
                                language="de",
                                response_format="verbose_json",
                            )
                            de_text = de_tr.text.strip() if de_tr.text else ""
                            if de_text:
                                text = de_text
                        except Exception as de_err:
                            log.warning("[ASR] German re-transcribe failed: %s", de_err)
                    log.info("[ASR] -> DE (whisper=%s, text_detect=%s)", whisper_lang, text_lang)
                else:
                    # Auto-detect did NOT say German — cross-check with
                    # a forced-DE pass to catch misdetections.
                    auto_text = text  # save the auto-detect result
                    auto_lang = whisper_lang if whisper_lang in ALLOWED_LANGUAGES else 'en'
                    de_text = ""
                    try:
                        de_tr = self._client.audio.transcriptions.create(
                            file=("audio.wav", wav_bytes),
                            model=self._model,
                            language="de",
                            response_format="verbose_json",
                        )
                        de_text = de_tr.text.strip() if de_tr.text else ""
                    except Exception:
                        pass

                    de_text_lang = _detect_lang_from_text(de_text) if de_text else "en"
                    log.info("[ASR] Cross-check: auto(%s)='%s', forced-DE='%s' (detect=%s)",
                             whisper_lang, auto_text[:50], de_text[:50], de_text_lang)

                    if de_text_lang == "de" and de_text:
                        # Forced-DE produced real German text → user spoke German
                        detected = "de"
                        text = de_text
                        log.info("[ASR] -> DE (cross-check confirmed German)")
                    else:
                        # Cross-check confirms not German → use auto result
                        detected = auto_lang
                        # If auto was a non-allowed language, re-transcribe as EN
                        if whisper_lang not in ALLOWED_LANGUAGES and auto_text:
                            try:
                                en_tr = self._client.audio.transcriptions.create(
                                    file=("audio.wav", wav_bytes),
                                    model=self._model,
                                    language="en",
                                    response_format="verbose_json",
                                )
                                en_text = en_tr.text.strip() if en_tr.text else ""
                                if en_text:
                                    text = en_text
                            except Exception:
                                pass
                        detected = "en"
                        log.info("[ASR] -> EN (whisper=%s, cross-check=%s)", whisper_lang, de_text_lang)

                raw_text = text
                text = filter_hallucination(
                    text,
                    no_speech_prob=avg_nsp if avg_nsp else None,
                    duration_s=duration if duration else None,
                )
                if raw_text and not text:
                    log.info("[ASR] Hallucination filter removed: '%s' (nsp=%.2f)", raw_text, avg_nsp)
                elif text:
                    log.info("[ASR] Result [%s]: '%s'", detected, text)

                return {
                    "text": text,
                    "language": detected,
                    "language_prob": 0.95,
                }

            except Exception as e:
                err_str = str(e).lower()
                is_retryable = (
                    "429" in err_str or "rate" in err_str
                    or "connection" in err_str or "timeout" in err_str
                )
                if attempt < max_retries and is_retryable:
                    wait = 1.0 * (2 ** attempt)
                    log.warning(f"[ASR] Groq error ({type(e).__name__}), retry {attempt+1} in {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    log.error(f"[ASR] Groq API error: {e}")
                    return {"text": "", "language": "en", "language_prob": 0.0}

        return {"text": "", "language": "en", "language_prob": 0.0}

