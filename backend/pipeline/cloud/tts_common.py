"""
TTS utilities — text sanitization, audio fades, and language detection.
"""

import re

import numpy as np


def sanitize_text(text: str) -> str:
    """Clean text for TTS — remove unspeakable characters."""
    text = re.sub(r'[\*\_\#\~\`\[\]\(\)\{\}\|]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r"[^\w\s.,!?;:'\-À-ɏ]", '', text)
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
