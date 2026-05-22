"""VAD — Silero VAD with adaptive noise gating and noise cancellation."""
import copy
import logging
from collections import deque
from typing import Optional
import noisereduce as nr
import numpy as np
import torch
from scipy.signal import butter, sosfilt
from .paths import MODELS_DIR

logger = logging.getLogger(__name__)

def _normalize_rms(sig, target=0.1, max_gain=10.0):
    rms = float(np.sqrt(np.mean(sig**2)))
    if rms < 1e-8: return sig
    return np.clip(sig * min(target/rms, max_gain), -1.0, 1.0).astype(np.float32)

class VADProcessor:
    _shared_model = None

    def __init__(self, threshold=0.45, min_speech_ms=250, min_silence_ms=600,
                 sample_rate=16_000, energy_threshold=0.005):
        self.threshold = self._base_threshold = threshold
        self.sample_rate = sample_rate
        self.min_speech_samples = int(min_speech_ms * sample_rate / 1000)
        self.min_silence_samples = int(min_silence_ms * sample_rate / 1000)
        self._energy_thr = energy_threshold
        self._noise_rms, self._noise_alpha, self._noise_frames, self._noise_cal = 0.0, 0.03, 0, 50
        self._gate_factor = 2.5
        self._timeout = sample_rate * 20
        self._cont = 0
        self._hangover = 0
        self._hangover_max = 6
        self._preroll: deque[np.ndarray] = deque(maxlen=max(int(0.2*sample_rate/512), 1))
        self._bufsz = 0
        self._min_snr = 3.0
        if VADProcessor._shared_model is None:
            jit = MODELS_DIR / "silero_vad.jit"
            if jit.exists():
                VADProcessor._shared_model = torch.jit.load(str(jit))
            else:
                VADProcessor._shared_model, _ = torch.hub.load(
                    "snakers4/silero-vad", "silero_vad", force_reload=False, trust_repo=True)
            VADProcessor._shared_model.eval()
            logger.info("[VAD] Model loaded")
        self.model = copy.deepcopy(VADProcessor._shared_model)
        self._in_speech = False
        self._speech_samples = self._silence_samples = 0
        self._buffer: list[np.ndarray] = []
        self._max_buf = sample_rate * 30

    def _upd_noise(self, rms):
        if self._noise_frames < self._noise_cal:
            self._noise_rms = (self._noise_rms*self._noise_frames+rms)/(self._noise_frames+1)
            self._noise_frames += 1
        elif not self._in_speech:
            self._noise_rms += self._noise_alpha*(rms-self._noise_rms)

    def _gate(self):
        if self._noise_frames < 10: return self._energy_thr
        return max(self._noise_rms*self._gate_factor, self._energy_thr)

    def _dyn_thr(self):
        if self._noise_rms > 0.03: return min(self._base_threshold+0.15, 0.75)
        if self._noise_rms > 0.015: return min(self._base_threshold+0.08, 0.65)
        return self._base_threshold

    def _snr(self, u):
        return 20.0*np.log10(max(float(np.sqrt(np.mean(u**2))),1e-8)/max(self._noise_rms,1e-8))

    def clean_audio(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) < 512: return audio
        try:
            sos = butter(4, 80, btype='highpass', fs=self.sample_rate, output='sos')
            audio = sosfilt(sos, audio).astype(np.float32)
            audio = nr.reduce_noise(y=audio, sr=self.sample_rate, stationary=True, prop_decrease=0.6)
            return _normalize_rms(audio, target=0.06)
        except Exception: return audio

    def process_chunk(self, chunk: np.ndarray) -> tuple[bool, Optional[np.ndarray]]:
        if len(chunk) == 0: return False, None
        cf = (chunk.astype(np.float32)/32768.0) if chunk.dtype==np.int16 else np.clip(chunk.astype(np.float32),-1,1)
        cf -= np.mean(cf)
        rms = float(np.sqrt(np.mean(cf**2)))
        gate = self._gate()

        def _emit():
            utt = np.concatenate(self._buffer); snr = self._snr(utt); self.reset()
            return (False, utt) if snr >= self._min_snr else (False, None)

        if rms < gate:
            self._upd_noise(rms)
            if self._in_speech:
                self._silence_samples += len(cf); self._buffer.append(cf)
                return _emit() if self._silence_samples >= self.min_silence_samples else (True, None)
            if self._speech_samples > 0:
                self._speech_samples = self._bufsz = self._hangover = 0; self._buffer.clear()
            self._preroll.append(cf)
            return False, None

        prob = self.model(torch.from_numpy(cf), self.sample_rate).item()
        thr = self._dyn_thr()

        if prob >= thr or (self._in_speech and self._hangover > 0):
            self._hangover = self._hangover_max if prob >= thr else self._hangover-1
            self._silence_samples = 0
            self._buffer.append(cf)
            self._speech_samples += len(cf); self._bufsz += len(cf); self._cont += len(cf)
            if not self._in_speech and self._speech_samples >= self.min_speech_samples:
                self._in_speech = True
                if self._preroll:
                    pre = list(self._preroll); self._buffer = pre+self._buffer
                    self._bufsz += sum(len(c) for c in pre); self._preroll.clear()
            if self._in_speech and self._cont >= self._timeout: return _emit()
            if self._in_speech and self._bufsz >= self._max_buf:
                utt = np.concatenate(self._buffer); self.reset(); return False, utt
            return self._in_speech, None
        else:
            self._upd_noise(rms)
            if self._in_speech:
                self._silence_samples += len(cf); self._buffer.append(cf)
                return _emit() if self._silence_samples >= self.min_silence_samples else (True, None)
            self._speech_samples = self._bufsz = self._hangover = 0
            self._buffer.clear(); self._preroll.append(cf)
            return False, None

    def reset(self):
        self._in_speech = False
        self._speech_samples = self._silence_samples = self._cont = self._hangover = self._bufsz = 0
        self._buffer.clear(); self._preroll.clear()
        self.model.reset_states()
        self.threshold = self._base_threshold

    @property
    def is_speaking(self): return self._in_speech

