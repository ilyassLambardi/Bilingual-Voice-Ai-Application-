"""ASR — faster-whisper (primary) with openai-whisper fallback."""
import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, TypedDict
import numpy as np
from .hallucination_filter import filter_hallucination
from .paths import MODELS_DIR
from .lang_detect import detect_lang_from_text

try:
    import whisper
except ImportError:
    whisper = None

logger = logging.getLogger(__name__)
_pool = ThreadPoolExecutor(max_workers=2)
_ALLOWED = {"en", "de"}
_SR = 16_000

class TranscriptionResult(TypedDict):
    text: str
    language: str
    language_prob: float

class ASRProcessor:
    def __init__(self, model_size="base", device="cpu", compute_type="int8", beam_size=3, language=None):
        self.beam_size, self.language, self._backend, self.model = beam_size, language, None, None
        try:
            from faster_whisper import WhisperModel
            try:
                self.model = WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=8)
                # Verify CUDA actually works (cuBLAS may be missing at runtime)
                if device == "cuda":
                    self.model.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)
            except Exception as e:
                if device == "cuda":
                    logger.warning("[ASR] CUDA failed (%s), falling back to CPU", e)
                    self.model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=8)
                else: raise
            self._backend = "faster-whisper"
            
        except Exception:
            pt = self._find_local_model()
            if pt and whisper:
                self.model = whisper.load_model(str(pt), device=device)
                self._backend = "whisper"
            else: raise RuntimeError("No ASR model available")
        logger.info("[ASR] Ready (%s)", self._backend)

    @staticmethod
    def _find_local_model() -> Optional[Path]:
        if not MODELS_DIR.exists(): 
            return None
        
        for name in ["large-v3-turbo","large-v3","large","medium","small","base","tiny"]:
            for f in MODELS_DIR.glob("*.pt"):
                if name in f.stem: 
                    return f
        pts = list(MODELS_DIR.glob("*.pt"))
        return pts[0] if pts else None

    async def transcribe(self, audio: np.ndarray, lang_hint: Optional[str] = None) -> TranscriptionResult:
        return await asyncio.get_running_loop().run_in_executor(
            _pool, functools.partial(self._transcribe_sync, audio, lang_hint=lang_hint))

    def _transcribe_sync(self, audio: np.ndarray, lang_hint=None) -> TranscriptionResult:
        audio = audio.astype(np.float32) if audio.dtype != np.float32 else audio
        dur = len(audio) / _SR
        if dur < 0.3 or float(np.sqrt(np.mean(audio**2))) < 0.003:
            return {"text": "", "language": "en", "language_prob": 0.0}
        return self._transcribe_whisper(audio) if self._backend == "whisper" else self._transcribe_fw(audio, lang_hint)

    def _transcribe_whisper(self, audio: np.ndarray) -> TranscriptionResult:
        assert whisper is not None, "whisper backend unavailable"
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=self.model.dims.n_mels).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        lang = max(probs, key=probs.get)

        if lang not in _ALLOWED: lang = "en"
        result = self.model.transcribe(audio, language=lang, beam_size=1, without_timestamps=True, condition_on_previous_text=False)
        text = filter_hallucination(result["text"].strip(), duration_s=len(audio)/_SR)
        return {"text": text, "language": lang, "language_prob": round(probs.get(lang, 0.0), 3)}

    def _fw_run(self, audio, lang=None):
        kw = dict(beam_size=self.beam_size, vad_filter=True, without_timestamps=True,
                  no_speech_threshold=0.4, log_prob_threshold=-1.0, condition_on_previous_text=False)
        if lang: kw["language"] = lang
        segs, info = self.model.transcribe(audio, **kw)
        parts, nsps, lps = [], [], []
        for s in segs:
            nsps.append(s.no_speech_prob); lps.append(s.avg_logprob)
            if s.no_speech_prob < 0.7: parts.append(s.text)
        text = " ".join(parts).strip()
        return (text, sum(nsps)/len(nsps) if nsps else 1.0,
                sum(lps)/len(lps) if lps else -999.0,
                info.language, getattr(info, "language_probability", 0.0))

    def _transcribe_fw(self, audio, lang_hint=None) -> TranscriptionResult:
        dur = len(audio) / _SR
        # Force EN first — small model auto-detect is unreliable (picks ar/ja/sa)
        text_en, nsp_en, lp_en, _, _ = self._fw_run(audio, lang="en")
        logger.info("[ASR] EN pass: nsp=%.2f lp=%.2f text='%s'", nsp_en, lp_en, text_en[:60])
        # If English transcription seems poor, try German
        if nsp_en > 0.5 or lp_en < -0.7 or not text_en.strip():
            text_de, nsp_de, lp_de, _, _ = self._fw_run(audio, lang="de")
            logger.info("[ASR] DE pass: nsp=%.2f lp=%.2f text='%s'", nsp_de, lp_de, text_de[:60])
            if text_de.strip() and lp_de > lp_en:
                text = filter_hallucination(text_de, no_speech_prob=nsp_de, duration_s=dur)
                wlang = detect_lang_from_text(text) if text.strip() else "de"
                return {"text": text, "language": wlang, "language_prob": 0.8}
        text = filter_hallucination(text_en, no_speech_prob=nsp_en, duration_s=dur)
        wlang = detect_lang_from_text(text) if text.strip() else "en"
        return {"text": text, "language": wlang, "language_prob": round(max(1.0 - nsp_en, 0.1), 2)}
