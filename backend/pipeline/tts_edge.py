"""
Module 2 (Processing/Core): TTS — Microsoft Edge Neural TTS.

Uses edge-tts to access Microsoft's high-quality neural voices.
A single multilingual voice speaks both English and German naturally,
solving the "two different people" problem of dual-model approaches.

MP3 → PCM conversion via subprocess + imageio-ffmpeg bundled binary.
"""

import asyncio
import os
import random
import re
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

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

# Filler phrases for "thinking" cues
_FILLERS = {
    "en": ["Hmm, let me think...", "One moment...", "Let me see..."],
    "de": ["Moment mal...", "Lass mich überlegen...", "Einen Augenblick..."],
}


def _sanitize_text(text: str) -> str:
    """Clean text for TTS — remove unspeakable characters."""
    text = re.sub(r'[\*\_\#\~\`\[\]\(\)\{\}\|]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^\w\s.,!?;:\'\"()\-/&$€%+@À-ɏßäöüÄÖÜ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not re.search(r'[a-zA-ZÀ-ɏ]', text):
        return ""
    return text


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
        print("[EdgeTTS] ffmpeg decode timed out (30s)")
        return np.array([], dtype=np.int16)
    if proc.returncode != 0:
        print(f"[EdgeTTS] ffmpeg decode error: {proc.stderr.decode()[:200]}")
        return np.array([], dtype=np.int16)
    return np.frombuffer(proc.stdout, dtype=np.int16)


def _apply_fades(pcm: np.ndarray, fade_ms: int = 8, sample_rate: int = 24000) -> np.ndarray:
    """Apply short fade-in/out to eliminate clicks at chunk boundaries."""
    fade_samples = int(fade_ms * sample_rate / 1000)
    if len(pcm) < fade_samples * 2:
        return pcm
    pcm = pcm.astype(np.float32)
    pcm[:fade_samples] *= np.linspace(0, 1, fade_samples)
    pcm[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return pcm.astype(np.int16)


class EdgeTTSProcessor:
    """Async-friendly Microsoft Edge Neural TTS wrapper.

    Single multilingual voice for all languages — no model switching.
    """

    _filler_cache: dict[str, list[bytes]] = {}
    _initialized: bool = False

    def __init__(self, sample_rate: int = 24_000, device: str = "cpu"):
        self.sample_rate = sample_rate
        # device param accepted for API compat but not used (cloud TTS)

        if not EdgeTTSProcessor._initialized:
            print("[EdgeTTS] Microsoft Edge Neural TTS ready.")
            print(f"[EdgeTTS] EN voice: {_VOICE_MAP['en']}")
            print(f"[EdgeTTS] DE voice: {_VOICE_MAP['de']}")
            print(f"[EdgeTTS] Sample rate: {sample_rate}Hz")
            EdgeTTSProcessor._initialized = True

            # Pre-cache fillers in background (best-effort)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._cache_fillers_async())
            except RuntimeError:
                pass  # no running loop yet; fillers will be None until first call

    # ── Public async API ──────────────────────────────────────────────

    async def _stream_with_retry(
        self, text: str, voice: str, rate: str, pitch: str, max_retries: int = 3,
    ) -> bytes:
        """Call edge-tts with retry + exponential backoff for transient timeouts."""
        for attempt in range(max_retries):
            try:
                communicate = edge_tts.Communicate(
                    text=text, voice=voice, rate=rate, pitch=pitch,
                )
                mp3_chunks = []
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        mp3_chunks.append(chunk["data"])
                if mp3_chunks:
                    return b''.join(mp3_chunks)
                print(f"[EdgeTTS] No audio returned (attempt {attempt+1})")
            except (TimeoutError, Exception) as e:
                wait = 0.5 * (2 ** attempt)
                if attempt < max_retries - 1:
                    print(f"[EdgeTTS] Attempt {attempt+1} failed ({type(e).__name__}), retrying in {wait:.1f}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"[EdgeTTS] All {max_retries} attempts failed: {e}")
        return b''

    async def synthesize(
        self,
        text: str,
        lang: str = "en",
        speaker: Optional[str] = None,
        prosody: bool = True,
    ) -> bytes:
        """Convert text -> raw Int16 PCM bytes at self.sample_rate."""
        text = _sanitize_text(text)
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
            pcm = _apply_fades(pcm, fade_ms=10, sample_rate=self.sample_rate)

            # Trailing silence after sentence-ending punctuation (shorter = smoother flow)
            trail_ms = 60 if text.rstrip()[-1:] in '.!?' else 30
            pad = np.zeros(int(self.sample_rate * trail_ms / 1000), dtype=np.int16)
            pcm = np.concatenate([pcm, pad])

            return pcm.tobytes()

        except Exception as e:
            print(f"[EdgeTTS] Synthesis error: {e}")
            traceback.print_exc()
            return b''

    # ── Filler support ────────────────────────────────────────────────

    async def _cache_fillers_async(self):
        """Pre-cache filler phrases for instant playback."""
        for lang in _FILLERS:
            EdgeTTSProcessor._filler_cache[lang] = []
            for phrase in _FILLERS[lang]:
                try:
                    pcm = await self.synthesize(phrase, lang=lang, prosody=False)
                    if pcm:
                        EdgeTTSProcessor._filler_cache[lang].append(pcm)
                except Exception as e:
                    print(f"[EdgeTTS] Filler cache failed for '{phrase}': {e}")
            print(f"[EdgeTTS] Cached {len(EdgeTTSProcessor._filler_cache[lang])} fillers for {lang.upper()}")

    def get_filler(self, lang: str = "en") -> Optional[bytes]:
        """Return a random pre-cached filler PCM."""
        fillers = EdgeTTSProcessor._filler_cache.get(lang, [])
        if not fillers:
            return None
        return random.choice(fillers)

    def get_sample_rate(self) -> int:
        return self.sample_rate
