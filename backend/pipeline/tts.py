"""
TTS — Silero TTS v3 (VITS architecture, torch.hub).

Synthesises text → raw Int16 PCM bytes entirely in RAM.
Supports English and German with lazy-loaded per-language models.
"""

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "silero-models"

_pool = ThreadPoolExecutor(max_workers=2)

# language → (hub model id, default speaker)
_LANG_MAP: dict[str, tuple[str, str]] = {
    "en": ("v3_en", "en_21"),
    "de": ("v3_de", "eva_k"),
}


# Filler phrases for thinking cues (short, natural)
_FILLERS = {
    "en": ["hmm", "let me think", "well"],
    "de": ["hmm", "moment mal", "also"],
}

# Sentiment keywords for prosody adjustment
_POSITIVE_KW = {
    "amazing", "awesome", "great", "love", "fantastic", "excited",
    "happy", "wonderful", "cool", "nice", "fun", "incredible",
    "toll", "super", "wunderbar", "geil", "klasse", "freue",
}
_NEGATIVE_KW = {
    "sorry", "sad", "unfortunately", "difficult", "hard", "tough",
    "miss", "lost", "worried", "afraid", "terrible", "awful",
    "traurig", "leider", "schwer", "schlimm", "angst", "sorge",
}


# Common German words for language detection
_DE_MARKERS = {
    "ich", "du", "er", "sie", "es", "wir", "ihr", "und", "oder", "aber",
    "das", "der", "die", "ein", "eine", "ist", "sind", "war", "hat",
    "haben", "wird", "mit", "von", "zu", "auf", "nicht", "auch", "noch",
    "dass", "wenn", "weil", "dann", "schon", "sehr", "hier", "dort",
    "kann", "muss", "soll", "will", "denn", "nur", "mehr", "wie",
    "über", "unter", "nach", "vor", "zwischen", "durch", "für",
    "immer", "vielleicht", "eigentlich", "natürlich", "glaube",
    "denke", "finde", "meine", "würde", "könnte", "sollte",
}


def _detect_sentence_lang(text: str) -> str:
    """Detect if a sentence is German or English using keyword + character heuristic."""
    lower = text.lower()
    # German-specific characters are a strong signal
    if any(ch in lower for ch in "\u00e4\u00f6\u00fc\u00df"):  # ä ö ü ß
        return "de"
    words = set(lower.split())
    de_count = len(words & _DE_MARKERS)
    if len(words) > 0 and de_count / len(words) > 0.15:
        return "de"
    return "en"


def _detect_sentiment(text: str) -> str:
    """Lightweight keyword sentiment: positive / negative / neutral."""
    lower = text.lower()
    words = set(lower.split())
    pos = len(words & _POSITIVE_KW)
    neg = len(words & _NEGATIVE_KW)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def _adjust_speed(pcm_int16: np.ndarray, factor: float) -> np.ndarray:
    """Resample PCM to adjust speed using linear interpolation (artifact-free)."""
    if abs(factor - 1.0) < 0.01:
        return pcm_int16
    n_in = len(pcm_int16)
    n_out = int(n_in / factor)
    if n_out < 1:
        return pcm_int16
    # Linear interpolation — much cleaner than scipy.signal.resample
    x_old = np.linspace(0, 1, n_in)
    x_new = np.linspace(0, 1, n_out)
    resampled = np.interp(x_new, x_old, pcm_int16.astype(np.float32))
    return resampled.astype(np.int16)


def _apply_fades(pcm: np.ndarray, fade_ms: int = 8, sample_rate: int = 48000) -> np.ndarray:
    """Apply short fade-in/out to eliminate clicks at chunk boundaries."""
    fade_samples = int(fade_ms * sample_rate / 1000)
    if len(pcm) < fade_samples * 2:
        return pcm
    pcm = pcm.astype(np.float32)
    # Fade in
    pcm[:fade_samples] *= np.linspace(0, 1, fade_samples)
    # Fade out
    pcm[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return pcm.astype(np.int16)


class TTSProcessor:
    """Async-friendly Silero TTS wrapper.  No file I/O."""

    # Class-level cache — models loaded once, shared across all sessions
    _shared_models: dict[str, torch.nn.Module] = {}
    _shared_fillers: dict[str, list[bytes]] = {}
    _initialized: bool = False

    def __init__(self, sample_rate: int = 24_000, device: str = "cpu"):
        self.sample_rate = sample_rate
        self.device = device
        self._models = TTSProcessor._shared_models
        self._fillers = TTSProcessor._shared_fillers
        if not TTSProcessor._initialized:
            # First time: load models and cache fillers
            for lang in _LANG_MAP:
                self._ensure_model(lang)
            self._cache_fillers()
            TTSProcessor._initialized = True
            print("[TTS] Silero TTS ready (EN + DE models loaded, fillers cached).")
        else:
            print("[TTS] Using cached models (shared).")

    # ── Public async API ──────────────────────────────────────────────────

    async def synthesize(
        self,
        text: str,
        lang: str = "en",
        speaker: Optional[str] = None,
        prosody: bool = True,
    ) -> bytes:
        """Convert text → raw Int16 PCM bytes at self.sample_rate.

        If prosody=True, adjusts speech rate based on detected sentiment.
        Safe to call from the async event loop (runs in a thread).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _pool, self._synth_sync, text, lang, speaker, prosody
        )

    # ── Sync worker ───────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Clean text for Silero TTS — remove unspeakable characters."""
        # Remove markdown/formatting
        text = re.sub(r'[\*\_\#\~\`\[\]\(\)\{\}\|]', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove emojis and non-latin unicode (keep basic accented chars)
        text = re.sub(r'[^\w\s.,!?;:\'\'\-À-ɏßäöüÄÖÜ]', '', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Must have at least one letter
        if not re.search(r'[a-zA-ZÀ-ɏ]', text):
            return ""
        return text

    def _synth_segment(self, text: str, lang: str, speaker: str) -> np.ndarray:
        """Synthesize a single text segment → int16 numpy array."""
        model = self._ensure_model(lang)
        try:
            audio_tensor = model.apply_tts(
                text=text, speaker=speaker, sample_rate=self.sample_rate
            )
        except (ValueError, RuntimeError) as e:
            print(f"[TTS] Skipping unspeakable text: '{text[:50]}' ({e})")
            return np.array([], dtype=np.int16)

        wav = audio_tensor.squeeze().numpy()
        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = wav / peak * 0.95
        return (wav * 32767).astype(np.int16)

    def _inject_breath_pauses(self, text: str, lang: str, speaker: str) -> np.ndarray:
        """Split text at commas/dashes/colons, synthesize segments, stitch with
        micro-silence gaps to simulate natural breathing and pacing."""
        # Split at breath points: commas, semicolons, dashes, colons
        segments = re.split(r'([,;:\u2014\u2013\-]+\s*)', text)
        segments = [s for s in segments if s.strip()]

        if len(segments) <= 1:
            # No breath points — synthesize directly
            return self._synth_segment(text, lang, speaker)

        parts = []
        for i, seg in enumerate(segments):
            clean = self._sanitize_text(seg)
            if not clean or re.fullmatch(r'[,;:\u2014\u2013\-\s]+', clean):
                # This is a punctuation segment — insert micro-pause
                pause_ms = 60 if ',' in seg else 90  # comma=60ms, semicolon/dash=90ms
                silence = np.zeros(int(self.sample_rate * pause_ms / 1000), dtype=np.int16)
                parts.append(silence)
            else:
                pcm = self._synth_segment(clean, lang, speaker)
                if len(pcm) > 0:
                    parts.append(pcm)

        if not parts:
            return np.array([], dtype=np.int16)
        return np.concatenate(parts)

    def _synth_sync(
        self, text: str, lang: str, speaker: Optional[str],
        prosody: bool = True,
    ) -> bytes:
        text = self._sanitize_text(text)
        if not text:
            return b''  # empty audio for unspeakable text

        spk = speaker or _LANG_MAP.get(lang, _LANG_MAP["en"])[1]

        # Use breath-pause injection for natural pacing
        has_breath_points = bool(re.search(r'[,;:\u2014\u2013\-]', text))
        if has_breath_points and len(text) > 20:
            pcm = self._inject_breath_pauses(text, lang, spk)
        else:
            pcm = self._synth_segment(text, lang, spk)

        if len(pcm) == 0:
            return b''

        # Emotional prosody: adjust speech rate based on sentiment
        if prosody:
            sentiment = _detect_sentiment(text)
            if sentiment == "positive":
                pcm = _adjust_speed(pcm, 1.06)   # slightly faster, energetic
            elif sentiment == "negative":
                pcm = _adjust_speed(pcm, 0.94)   # slightly slower, empathetic

        # Apply fade-in/out to prevent clicks at chunk edges
        pcm = _apply_fades(pcm, fade_ms=8, sample_rate=self.sample_rate)

        # Add short silence padding at end for natural sentence spacing
        pad = np.zeros(int(self.sample_rate * 0.08), dtype=np.int16)  # 80ms gap
        pcm = np.concatenate([pcm, pad])

        return pcm.tobytes()

    # ── Lazy model loading ────────────────────────────────────────────────

    def _ensure_model(self, lang: str) -> torch.nn.Module:
        if lang not in _LANG_MAP:
            lang = "en"
        if lang in self._models:
            return self._models[lang]

        model_id = _LANG_MAP[lang][0]
        local_pt = _MODELS_DIR.parent / f"{model_id}.pt"
        print(f"[TTS] Loading silero {model_id} ...")
        if local_pt.exists():
            print(f"[TTS] Using local: {local_pt.name}")
            model = torch.package.PackageImporter(str(local_pt)).load_pickle(
                "tts_models", "model"
            )
        else:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language=lang,
                speaker=model_id,
                trust_repo=True,
            )
        model.to(self.device)
        self._models[lang] = model
        print(f"[TTS] {lang.upper()} model ready.")
        return model

    # ── Filler / thinking cue support ────────────────────────────────────

    def _cache_fillers(self):
        """Pre-synthesize filler phrases at startup for instant playback."""
        import random
        for lang in _FILLERS:
            self._fillers[lang] = []
            for phrase in _FILLERS[lang]:
                try:
                    pcm = self._synth_sync(phrase, lang, None)
                    self._fillers[lang].append(pcm)
                except Exception as e:
                    print(f"[TTS] Filler cache failed for '{phrase}' ({lang}): {e}")
            print(f"[TTS] Cached {len(self._fillers[lang])} fillers for {lang.upper()}")

    def get_filler(self, lang: str = "en") -> Optional[bytes]:
        """Return a random pre-cached filler PCM (instant, no inference)."""
        import random
        fillers = self._fillers.get(lang, self._fillers.get("en", []))
        if not fillers:
            return None
        return random.choice(fillers)

    def get_sample_rate(self) -> int:
        return self.sample_rate
