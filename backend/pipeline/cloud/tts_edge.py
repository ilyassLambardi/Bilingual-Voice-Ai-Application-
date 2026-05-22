"""
Module 2 (Processing/Core): TTS — Microsoft Edge Neural TTS.

Uses edge-tts to access Microsoft's high-quality neural voices.
A single multilingual voice speaks both English and German naturally,
solving the "two different people" problem of dual-model approaches.

MP3 → PCM conversion via subprocess + imageio-ffmpeg bundled binary.
"""

import asyncio
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

from .tts_common import sanitize_text, apply_fades

_pool = ThreadPoolExecutor(max_workers=2)

# ── Clear proxy env vars so aiohttp connects directly ────────────────
# edge-tts uses aiohttp which reads these; proxy causes connection timeouts
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)

# ── Locate ffmpeg binary from imageio-ffmpeg ──────────────────────────
try:
    import imageio_ffmpeg
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    _FFMPEG = "ffmpeg"  # hope it's on PATH

import edge_tts

# ── Voice mapping ─────────────────────────────────────────────────────
# Using Andrew Multilingual — ONE voice that speaks both EN and DE
# naturally with the same timbre, pitch, and style.
_VOICE_MAP = {
    "en": "en-US-AndrewMultilingualNeural",
    "de": "en-US-AndrewMultilingualNeural",  # same voice for both = consistent persona
}
_DEFAULT_VOICE = "en-US-AndrewMultilingualNeural"


def _mp3_bytes_to_pcm(mp3_data: bytes, target_sr: int = 24000) -> np.ndarray:
    """Convert MP3 bytes → int16 PCM numpy array at target sample rate.

    Uses ffmpeg subprocess (from imageio-ffmpeg) for reliable decoding.
    """
    try:
        proc = subprocess.run(
            [_FFMPEG, "-loglevel", "error",
             "-i", "pipe:0",
             "-f", "s16le", "-acodec", "pcm_s16le",
             "-ar", str(target_sr), "-ac", "1",
             "pipe:1"],
            input=mp3_data, capture_output=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        logger.warning("[EdgeTTS] ffmpeg decode timed out (30s)")
        return np.array([], dtype=np.int16)
    if proc.returncode != 0:
        logger.warning("[EdgeTTS] ffmpeg decode error: %s", proc.stderr.decode()[:200])
        return np.array([], dtype=np.int16)
    return np.frombuffer(proc.stdout, dtype=np.int16)


class EdgeTTSProcessor:
    """Async-friendly Microsoft Edge Neural TTS wrapper.

    Single multilingual voice for all languages — no model switching.
    """

    _initialized: bool = False

    def __init__(self, sample_rate: int = 24_000, device: str = "cpu"):
        self.sample_rate = sample_rate
        # device param accepted for API compat but not used (cloud TTS)

        if not EdgeTTSProcessor._initialized:
            logger.info("[EdgeTTS] Microsoft Edge Neural TTS ready.")
            logger.info("[EdgeTTS] EN voice: %s", _VOICE_MAP['en'])
            logger.info("[EdgeTTS] DE voice: %s", _VOICE_MAP['de'])
            logger.info("[EdgeTTS] Sample rate: %dHz", sample_rate)
            EdgeTTSProcessor._initialized = True

    # ── Public async API ──────────────────────────────────────────────

    @staticmethod
    async def _fetch_edge_audio(text: str, voice: str, rate: str, pitch: str) -> bytes:
        """Single edge-tts call — returns MP3 bytes or empty bytes."""
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
        mp3_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_chunks.append(chunk["data"])
        return b''.join(mp3_chunks) if mp3_chunks else b''

    async def _stream_with_retry(
        self, text: str, voice: str, rate: str, pitch: str, max_retries: int = 3,
    ) -> bytes:
        """Call edge-tts with retry + short backoff for transient timeouts."""
        for attempt in range(max_retries):
            try:
                mp3_data = await asyncio.wait_for(
                    self._fetch_edge_audio(text, voice, rate, pitch),
                    timeout=8,
                )
                if mp3_data:
                    return mp3_data
                logger.warning("[EdgeTTS] No audio returned (attempt %d)", attempt + 1)
            except (TimeoutError, asyncio.TimeoutError, Exception) as e:
                wait = 0.5 * (attempt + 1)  # 0.5s, 1.0s, 1.5s
                if attempt < max_retries - 1:
                    logger.warning("[EdgeTTS] Attempt %d failed (%s), retrying in %.1fs...", attempt + 1, type(e).__name__, wait)
                    await asyncio.sleep(wait)
                else:
                    logger.error("[EdgeTTS] All %d attempts failed: %s", max_retries, e)
        return b''

    async def synthesize(
        self,
        text: str,
        lang: str = "en",
        speaker: Optional[str] = None,
        prosody: bool = True,
    ) -> bytes:
        """Convert text -> raw Int16 PCM bytes at self.sample_rate."""
        text = sanitize_text(text)
        if not text:
            return b''

        voice = speaker or _VOICE_MAP.get(lang, _DEFAULT_VOICE)

        try:
            rate = "+0%"
            pitch = "+0Hz"

            if prosody:
                if text.rstrip().endswith('?'):
                    pitch = "+2Hz"
                    rate = "-3%"
                elif text.rstrip().endswith('!'):
                    rate = "+5%"
                    pitch = "+1Hz"

            mp3_data = await self._stream_with_retry(text, voice, rate, pitch)
            if not mp3_data:
                return b''

            loop = asyncio.get_running_loop()
            pcm = await loop.run_in_executor(
                _pool, _mp3_bytes_to_pcm, mp3_data, self.sample_rate
            )

            if len(pcm) == 0:
                return b''

            # RMS normalization for consistent volume across sentences (P6 fix)
            pcm_f = pcm.astype(np.float32)
            rms = np.sqrt(np.mean(pcm_f ** 2))
            if rms > 1.0:  # avoid div-by-zero on silence
                target_rms = 0.25 * 32767  # ~25% of full scale
                gain = min(target_rms / rms, 5.0)  # cap gain to prevent over-amplification
                pcm_f = np.clip(pcm_f * gain, -32767, 32767)
                pcm = pcm_f.astype(np.int16)

            # Fade in/out to prevent clicks
            pcm = apply_fades(pcm, fade_ms=10, sample_rate=self.sample_rate)

            # Trailing silence after sentence-ending punctuation (shorter = smoother flow)
            trail_ms = 60 if text.rstrip()[-1:] in '.!?' else 30
            pad = np.zeros(int(self.sample_rate * trail_ms / 1000), dtype=np.int16)
            pcm = np.concatenate([pcm, pad])

            return pcm.tobytes()

        except Exception as e:
            logger.error("[EdgeTTS] Synthesis error: %s", e, exc_info=True)
            return b''

    def get_sample_rate(self) -> int:
        return self.sample_rate
