"""
TTS — XTTSv2 (Coqui) unified multilingual voice.

Single voice latent speaks both English and German in the same voice.
Falls back to Silero if Coqui TTS is not installed.
"""

import asyncio
import re
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

_pool = ThreadPoolExecutor(max_workers=2)
_HERE = Path(__file__).resolve().parent
_MODELS_DIR = _HERE.parent / "models"
_VOICE_DIR = _MODELS_DIR / "voice_reference"

# Ensure voice reference directory exists
_VOICE_DIR.mkdir(parents=True, exist_ok=True)

# Try importing Coqui TTS
_XTTS_AVAILABLE = False
try:
    from TTS.api import TTS as CoquiTTS
    _XTTS_AVAILABLE = True
except ImportError:
    CoquiTTS = None


def _sanitize_text(text: str) -> str:
    """Clean text for TTS — remove unspeakable characters."""
    text = re.sub(r'[\*\_\#\~\`\[\]\(\)\{\}\|]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^\w\s.,!?;:\'\'\-À-ɏßäöüÄÖÜ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not re.search(r'[a-zA-ZÀ-ɏ]', text):
        return ""
    return text


def _apply_fades(pcm: np.ndarray, fade_ms: int = 8, sample_rate: int = 24000) -> np.ndarray:
    """Apply short fade-in/out to eliminate clicks at chunk boundaries."""
    fade_samples = int(fade_ms * sample_rate / 1000)
    if len(pcm) < fade_samples * 2:
        return pcm
    pcm = pcm.astype(np.float32)
    pcm[:fade_samples] *= np.linspace(0, 1, fade_samples)
    pcm[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return pcm.astype(np.int16)


class XTTSProcessor:
    """Async-friendly XTTSv2 TTS wrapper. Single voice for all languages."""

    _shared_model = None
    _initialized = False

    def __init__(self, sample_rate: int = 24_000, device: str = "cpu"):
        self.sample_rate = sample_rate
        self.device = device
        self._model = None
        self._reference_wav = None

        if not _XTTS_AVAILABLE:
            print("[XTTS] Coqui TTS not installed. Install with: pip install TTS")
            print("[XTTS] Falling back to Silero TTS.")
            self._fallback = True
            self._init_silero_fallback()
            return

        self._fallback = False

        if XTTSProcessor._initialized and XTTSProcessor._shared_model:
            self._model = XTTSProcessor._shared_model
            print("[XTTS] Using cached XTTSv2 model (shared).")
        else:
            self._load_xtts()
            XTTSProcessor._shared_model = self._model
            XTTSProcessor._initialized = True

        # Find reference voice wav
        self._find_reference_voice()

    def _load_xtts(self):
        """Load the XTTSv2 model."""
        print("[XTTS] Loading XTTSv2 model (this may take a moment)...")
        try:
            self._model = CoquiTTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=True,
            )
            if self.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self._model = self._model.to(self.device)
            print("[XTTS] XTTSv2 model loaded successfully!")
        except Exception as e:
            print(f"[XTTS] Failed to load XTTSv2: {e}")
            print("[XTTS] Falling back to Silero TTS.")
            self._fallback = True
            self._init_silero_fallback()

    def _find_reference_voice(self):
        """Find a reference WAV file for voice cloning."""
        # Look for any .wav file in the voice_reference directory
        wav_files = list(_VOICE_DIR.glob("*.wav"))
        if wav_files:
            self._reference_wav = str(wav_files[0])
            print(f"[XTTS] Using voice reference: {wav_files[0].name}")
        else:
            # Create a default reference using Silero (bootstrap)
            print(f"[XTTS] No voice reference found in {_VOICE_DIR}")
            print("[XTTS] Place a 6-10 second .wav file of your preferred voice in:")
            print(f"       {_VOICE_DIR}")
            print("[XTTS] Using default XTTSv2 voice for now.")
            self._reference_wav = None

    def _init_silero_fallback(self):
        """Initialize Silero as fallback."""
        from .tts import TTSProcessor
        self._silero = TTSProcessor(self.sample_rate, self.device)

    # ── Public async API ──────────────────────────────────────────────────

    async def synthesize(
        self,
        text: str,
        lang: str = "en",
        speaker: Optional[str] = None,
        prosody: bool = True,
    ) -> bytes:
        """Convert text → raw Int16 PCM bytes at self.sample_rate."""
        if self._fallback:
            return await self._silero.synthesize(text, lang, speaker, prosody)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _pool, self._synth_sync, text, lang
        )

    def _synth_sync(self, text: str, lang: str) -> bytes:
        """Synchronous XTTSv2 synthesis."""
        text = _sanitize_text(text)
        if not text:
            return b''

        # Map language codes
        xtts_lang = "en" if lang == "en" else "de"

        try:
            if self._reference_wav:
                wav = self._model.tts(
                    text=text,
                    speaker_wav=self._reference_wav,
                    language=xtts_lang,
                )
            else:
                wav = self._model.tts(
                    text=text,
                    language=xtts_lang,
                )

            # Convert to int16 PCM
            wav_np = np.array(wav, dtype=np.float32)
            peak = np.max(np.abs(wav_np))
            if peak > 0:
                wav_np = wav_np / peak * 0.95
            pcm = (wav_np * 32767).astype(np.int16)

            # Resample if needed (XTTS outputs at 24kHz by default)
            if self.sample_rate != 24000:
                n_out = int(len(pcm) * self.sample_rate / 24000)
                x_old = np.linspace(0, 1, len(pcm))
                x_new = np.linspace(0, 1, n_out)
                pcm = np.interp(x_new, x_old, pcm.astype(np.float32)).astype(np.int16)

            pcm = _apply_fades(pcm, fade_ms=8, sample_rate=self.sample_rate)

            # Sentence-end padding
            pad = np.zeros(int(self.sample_rate * 0.08), dtype=np.int16)
            pcm = np.concatenate([pcm, pad])

            return pcm.tobytes()

        except Exception as e:
            print(f"[XTTS] Synthesis error: {e}")
            return b''

    def get_filler(self, lang: str = "en") -> Optional[bytes]:
        """Return a filler phrase. Falls back to Silero for fillers."""
        if self._fallback:
            return self._silero.get_filler(lang)
        # For XTTS, synthesize a quick filler on demand
        # (not pre-cached since XTTS is slower)
        return None

    def get_sample_rate(self) -> int:
        return self.sample_rate
