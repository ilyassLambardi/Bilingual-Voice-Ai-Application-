"""TTS — Silero TTS v3 (EN + DE), async-friendly."""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import re
import numpy as np
import torch
from .paths import MODELS_DIR

def sanitize_text(text: str) -> str:
    """Clean text for TTS — remove unspeakable characters."""
    text = re.sub(r'[\*\_\#\~\`\[\]\(\)\{\}\|]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r"[^\w\s.,!?;:'\'\-À-ɏ]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not re.search(r'[a-zA-ZÀ-ɏ]', text):
        return ""
    return text

def apply_fades(pcm: np.ndarray, fade_ms: int = 8, sample_rate: int = 24000) -> np.ndarray:
    """Apply short fade-in/out to eliminate clicks at chunk boundaries."""
    fade_samples = int(fade_ms * sample_rate / 1000)
    if len(pcm) < fade_samples * 2:
        return pcm
    pcm = pcm.astype(np.float32)
    pcm[:fade_samples] *= np.linspace(0, 1, fade_samples)
    pcm[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return pcm.astype(np.int16)

logger = logging.getLogger(__name__)
_pool = ThreadPoolExecutor(max_workers=2)
_LANG_MAP = {"en": ("v3_en", "en_0"), "de": ("v3_de", "eva_k")}

class TTSProcessor:
    _shared_models: dict[str, torch.nn.Module] = {}
    _initialized = False

    def __init__(self, sample_rate=48_000, device="cpu"):
        self.sample_rate, self.device = sample_rate, device
        self._models = TTSProcessor._shared_models
        if not TTSProcessor._initialized:
            for lang in _LANG_MAP: self._ensure_model(lang)
            TTSProcessor._initialized = True
            logger.info("[TTS] Silero ready (EN+DE)")

    async def synthesize(self, text: str, lang="en", speaker=None, prosody=True) -> bytes:
        return await asyncio.get_running_loop().run_in_executor(
            _pool, self._synth_sync, text, lang, speaker, prosody)

    def _synth_sync(self, text: str, lang: str, speaker=None, prosody=True) -> bytes:
        text = sanitize_text(text)
        if not text: return b''
        spk = speaker or _LANG_MAP.get(lang, _LANG_MAP["en"])[1]
        model = self._ensure_model(lang)
        try:
            wav = model.apply_tts(text=text, speaker=spk, sample_rate=self.sample_rate).squeeze().numpy()
        except (ValueError, RuntimeError):
            return b''
        peak = np.max(np.abs(wav))
        if peak > 0: wav = wav / peak * 0.92
        pcm = (wav * 32767).astype(np.int16)
        pcm = apply_fades(pcm, fade_ms=12, sample_rate=self.sample_rate)
        pad = np.zeros(int(self.sample_rate * 0.1), dtype=np.int16)
        return np.concatenate([pcm, pad]).tobytes()

    def _ensure_model(self, lang: str) -> torch.nn.Module:
        if lang not in _LANG_MAP: lang = "en"
        if lang in self._models: return self._models[lang]
        mid = _LANG_MAP[lang][0]
        local_pt = MODELS_DIR / f"{mid}.pt"
        if local_pt.exists():
            model = torch.package.PackageImporter(str(local_pt)).load_pickle("tts_models", "model")
        else:
            model, _ = torch.hub.load("snakers4/silero-models", "silero_tts",
                                       language=lang, speaker=mid, trust_repo=True)
        model.to(self.device)
        self._models[lang] = model
        return model

    def get_sample_rate(self) -> int:
        return self.sample_rate
