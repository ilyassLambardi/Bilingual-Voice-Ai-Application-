"""Module 2 (Processing/Core): Voice Activity Detection — Silero VAD with noise cancellation.

Processes 512-sample chunks (32 ms @ 16 kHz) and emits speech
start / end events.  Includes:
  • Adaptive noise floor estimation
  • Spectral noise suppression (spectral subtraction)
  • SNR-based utterance validation
  • Speech timeout to prevent stuck-in-listening
  • Dynamic VAD threshold adjustment

Everything stays in RAM — no file I/O.
"""

import copy
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════
# Spectral noise suppression helpers
# ═══════════════════════════════════════════════════════════════════════

def _spectral_subtract(signal: np.ndarray, noise_profile: np.ndarray,
                       alpha: float = 2.0, beta: float = 0.02) -> np.ndarray:
    """Remove stationary noise via spectral subtraction.

    Parameters
    ----------
    signal : float32 audio (any length)
    noise_profile : estimated noise magnitude spectrum
    alpha : over-subtraction factor (higher = more aggressive)
    beta : spectral floor (prevents musical noise artifacts)

    Returns
    -------
    Cleaned float32 audio, same length as input.
    """
    n_fft = 512
    hop = 256
    window = np.hanning(n_fft).astype(np.float32)

    # Pad signal to fit full frames
    pad_len = (n_fft - len(signal) % hop) % hop
    padded = np.concatenate([signal, np.zeros(pad_len, dtype=np.float32)])

    n_frames = max(1, (len(padded) - n_fft) // hop + 1)
    output = np.zeros_like(padded)
    win_sum = np.zeros_like(padded)

    # Resize noise profile if needed
    freq_bins = n_fft // 2 + 1
    if len(noise_profile) != freq_bins:
        noise_profile = np.interp(
            np.linspace(0, 1, freq_bins),
            np.linspace(0, 1, len(noise_profile)),
            noise_profile,
        ).astype(np.float32)

    for i in range(n_frames):
        start = i * hop
        frame = padded[start:start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Subtract noise, keep spectral floor
        clean_mag = np.maximum(mag - alpha * noise_profile, beta * mag)
        clean_frame = np.fft.irfft(clean_mag * np.exp(1j * phase))

        output[start:start + n_fft] += clean_frame[:n_fft] * window
        win_sum[start:start + n_fft] += window ** 2

    # Normalize overlap-add
    win_sum = np.maximum(win_sum, 1e-8)
    output /= win_sum
    return output[:len(signal)]


def _highpass_fir(signal: np.ndarray, cutoff: float = 80.0,
                  sample_rate: int = 16000, num_taps: int = 255) -> np.ndarray:
    """FIR high-pass filter via windowed-sinc method.

    Removes DC offset, low-frequency rumble (HVAC, fans), and
    AC mains hum (50/60 Hz).  Pure numpy — no scipy needed.
    """
    fc = cutoff / sample_rate
    n = np.arange(num_taps)
    mid = (num_taps - 1) / 2.0

    # Low-pass sinc kernel
    with np.errstate(invalid="ignore"):
        h = np.sinc(2.0 * fc * (n - mid)).astype(np.float32)
    # Blackman window
    h *= np.blackman(num_taps).astype(np.float32)
    h_sum = np.sum(h)
    if abs(h_sum) > 1e-8:
        h /= h_sum
    # Spectral inversion → high-pass
    h = -h
    h[int(mid)] += 1.0
    return np.convolve(signal, h, mode="same").astype(np.float32)


def _wiener_filter(signal: np.ndarray, noise_spectrum: np.ndarray,
                   n_fft: int = 512, hop: int = 256,
                   floor: float = 0.05) -> np.ndarray:
    """Wiener filter — SNR-based gain, far less musical noise
    than spectral subtraction.

    Gain per bin: H = max(1 − noise²/signal², floor)
    """
    window = np.hanning(n_fft).astype(np.float32)
    freq_bins = n_fft // 2 + 1

    if len(noise_spectrum) != freq_bins:
        noise_spectrum = np.interp(
            np.linspace(0, 1, freq_bins),
            np.linspace(0, 1, len(noise_spectrum)),
            noise_spectrum,
        ).astype(np.float32)

    noise_power = noise_spectrum ** 2

    pad_len = (n_fft - len(signal) % hop) % hop
    padded = np.concatenate([signal, np.zeros(pad_len, dtype=np.float32)])
    n_frames = max(1, (len(padded) - n_fft) // hop + 1)
    output = np.zeros_like(padded)
    win_sum = np.zeros_like(padded)

    for i in range(n_frames):
        start = i * hop
        frame = padded[start:start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        power = np.abs(spectrum) ** 2

        gain = np.maximum(1.0 - noise_power / np.maximum(power, 1e-10), floor)
        clean_frame = np.fft.irfft(spectrum * gain)

        output[start:start + n_fft] += clean_frame[:n_fft] * window
        win_sum[start:start + n_fft] += window ** 2

    win_sum = np.maximum(win_sum, 1e-8)
    output /= win_sum
    return output[:len(signal)].astype(np.float32)


def _spectral_gate(signal: np.ndarray, noise_spectrum: np.ndarray,
                   n_fft: int = 1024, hop: int = 512,
                   threshold_factor: float = 1.5) -> np.ndarray:
    """Per-band spectral gate with smooth sigmoid transition.

    Catches residual noise that the Wiener filter missed.
    Uses a larger FFT (1024) for finer frequency resolution.
    """
    window = np.hanning(n_fft).astype(np.float32)
    freq_bins = n_fft // 2 + 1

    if len(noise_spectrum) != freq_bins:
        noise_spectrum = np.interp(
            np.linspace(0, 1, freq_bins),
            np.linspace(0, 1, len(noise_spectrum)),
            noise_spectrum,
        ).astype(np.float32)

    threshold = noise_spectrum * threshold_factor

    pad_len = (n_fft - len(signal) % hop) % hop
    padded = np.concatenate([signal, np.zeros(pad_len, dtype=np.float32)])
    n_frames = max(1, (len(padded) - n_fft) // hop + 1)
    output = np.zeros_like(padded)
    win_sum = np.zeros_like(padded)

    for i in range(n_frames):
        start = i * hop
        frame = padded[start:start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Smooth sigmoid gate: 0→1 transition around threshold
        gate = 1.0 / (1.0 + np.exp(-6.0 * (mag / np.maximum(threshold, 1e-8) - 1.0)))

        clean_frame = np.fft.irfft(mag * gate * np.exp(1j * phase))
        output[start:start + n_fft] += clean_frame[:n_fft] * window
        win_sum[start:start + n_fft] += window ** 2

    win_sum = np.maximum(win_sum, 1e-8)
    output /= win_sum
    return output[:len(signal)].astype(np.float32)


def _normalize_rms(signal: np.ndarray, target_rms: float = 0.1,
                   max_gain: float = 10.0) -> np.ndarray:
    """RMS-based auto-gain normalization for consistent ASR input level."""
    rms = float(np.sqrt(np.mean(signal ** 2)))
    if rms < 1e-8:
        return signal
    gain = min(target_rms / rms, max_gain)
    return np.clip(signal * gain, -1.0, 1.0).astype(np.float32)


class VADProcessor:
    """Streaming VAD using Silero VAD from torch.hub with noise cancellation."""

    # Class-level model cache — loaded once, shared across all sessions
    _shared_model = None

    def __init__(
        self,
        threshold: float = 0.45,
        min_speech_ms: int = 250,
        min_silence_ms: int = 600,
        sample_rate: int = 16_000,
        energy_threshold: float = 0.005,
    ):
        self.threshold = threshold
        self._base_threshold = threshold   # original, before dynamic adjustment
        self.sample_rate = sample_rate
        self.min_speech_samples = int(min_speech_ms * sample_rate / 1000)
        self.min_silence_samples = int(min_silence_ms * sample_rate / 1000)
        self._energy_threshold = energy_threshold  # absolute minimum floor

        # ── Adaptive noise floor estimation ────────────────────────────
        self._noise_rms = 0.0               # running estimate of background noise RMS
        self._noise_alpha = 0.03            # EMA smoothing factor (slow adaptation)
        self._noise_frames = 0              # frames used for initial calibration
        self._noise_calibration = 50        # calibrate for first ~50 chunks (~1.6s)
        self._adaptive_gate_factor = 2.5    # gate = noise_rms * this factor

        # ── Spectral noise profile ─────────────────────────────────────
        self._noise_spectrum: Optional[np.ndarray] = None
        self._noise_spec_alpha = 0.05       # EMA for spectral profile update
        self._spectral_enabled = True       # can disable if too CPU heavy

        # ── Speech timeout (prevents stuck-in-listening) ──────────────
        self._speech_timeout_samples = sample_rate * 8  # 8s max continuous speech
        self._continuous_speech_samples = 0

        # ── Hangover smoothing (prevents flickering at speech boundaries) ──
        self._hangover_frames = 0           # frames remaining in hangover
        self._hangover_max = 4              # ~128ms hangover after VAD drops

        # ── Pre-roll buffer (captures word onsets before VAD triggers) ──
        self._preroll_chunks = int(0.2 * sample_rate / 512)  # ~200ms
        self._preroll: deque[np.ndarray] = deque(maxlen=max(self._preroll_chunks, 1))

        # ── Cached buffer size (avoids O(n) sum every chunk) ──────────
        self._buffer_samples = 0

        # ── SNR validation ────────────────────────────────────────────
        self._min_snr_db = 3.0              # minimum SNR to accept an utterance

        # Load Silero VAD — cached at class level for multi-session
        if VADProcessor._shared_model is None:
            print("[VAD] Loading Silero VAD ...")
            _local_jit = Path(__file__).resolve().parent.parent.parent / "models" / "silero_vad.jit"
            if _local_jit.exists():
                print(f"[VAD] Using local: {_local_jit.name}")
                VADProcessor._shared_model = torch.jit.load(str(_local_jit))
            else:
                VADProcessor._shared_model, _utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,
                )
            VADProcessor._shared_model.eval()
            print("[VAD] Ready (with noise cancellation).")
        else:
            print("[VAD] Using cached model (shared).")
        # Deep-copy so each session has independent LSTM hidden states
        self.model = copy.deepcopy(VADProcessor._shared_model)

        # State
        self._in_speech = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._buffer: list[np.ndarray] = []
        self._max_buffer_samples = sample_rate * 15  # 15s hard cap (reduced from 30s)

    # ── Adaptive noise floor ────────────────────────────────────────────

    def _update_noise_estimate(self, rms: float, chunk_f: np.ndarray):
        """Update the adaptive noise floor with a new chunk.
        Only updates when we believe this is a noise-only frame."""
        if self._noise_frames < self._noise_calibration:
            # Initial calibration: average all frames
            self._noise_rms = (
                self._noise_rms * self._noise_frames + rms
            ) / (self._noise_frames + 1)
            self._noise_frames += 1
            if self._noise_frames == self._noise_calibration:
                print(f"[VAD] Noise floor calibrated: RMS={self._noise_rms:.4f}")
        else:
            # Ongoing: slow EMA update (only when not in speech)
            if not self._in_speech:
                self._noise_rms += self._noise_alpha * (rms - self._noise_rms)

        # Update spectral noise profile
        if self._spectral_enabled and not self._in_speech:
            n_fft = 512
            if len(chunk_f) >= n_fft:
                spectrum = np.abs(np.fft.rfft(chunk_f[:n_fft]))
            else:
                padded = np.zeros(n_fft, dtype=np.float32)
                padded[:len(chunk_f)] = chunk_f
                spectrum = np.abs(np.fft.rfft(padded))

            if self._noise_spectrum is None:
                self._noise_spectrum = spectrum.astype(np.float32)
            else:
                self._noise_spectrum += self._noise_spec_alpha * (
                    spectrum - self._noise_spectrum
                )

    def _get_adaptive_gate(self) -> float:
        """Return the current adaptive energy gate."""
        if self._noise_frames < 10:
            return self._energy_threshold  # not enough data yet
        adaptive = self._noise_rms * self._adaptive_gate_factor
        # Never go below the absolute minimum floor
        return max(adaptive, self._energy_threshold)

    def _get_dynamic_threshold(self) -> float:
        """Raise VAD probability threshold in noisy environments."""
        if self._noise_rms > 0.03:
            # Very noisy: be stricter
            return min(self._base_threshold + 0.15, 0.75)
        elif self._noise_rms > 0.015:
            # Moderately noisy: slightly stricter
            return min(self._base_threshold + 0.08, 0.65)
        return self._base_threshold

    def _compute_snr(self, utterance: np.ndarray) -> float:
        """Compute signal-to-noise ratio in dB."""
        sig_rms = float(np.sqrt(np.mean(utterance ** 2)))
        noise = max(self._noise_rms, 1e-8)
        return 20.0 * np.log10(max(sig_rms, 1e-8) / noise)

    def clean_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply advanced 4-stage noise cancellation pipeline to an utterance.

        Stages:
            1. High-pass FIR (80 Hz) — removes DC, rumble, AC hum
            2. Wiener filter — primary noise reduction (SNR-based gain)
            3. Spectral gate — catches residual noise with smooth sigmoid
            4. RMS normalization — consistent level for ASR
        """
        if not self._spectral_enabled or self._noise_spectrum is None:
            return audio
        try:
            # Stage 1: High-pass filter
            audio = _highpass_fir(audio, cutoff=80.0, sample_rate=self.sample_rate)
            # Stage 2: Wiener filter (primary denoising)
            audio = _wiener_filter(audio, self._noise_spectrum)
            # Stage 3: Spectral gate (residual noise)
            audio = _spectral_gate(audio, self._noise_spectrum)
            # Stage 4: Normalize for consistent ASR input
            audio = _normalize_rms(audio, target_rms=0.1)
            return audio
        except Exception as e:
            print(f"[VAD] Advanced denoise failed ({e}), falling back to spectral subtract")
            try:
                return _spectral_subtract(audio, self._noise_spectrum)
            except Exception:
                return audio

    # ── Public API ────────────────────────────────────────────────────────

    def process_chunk(
        self, chunk: np.ndarray
    ) -> tuple[bool, Optional[np.ndarray]]:
        """Feed a 512-sample int16/float32 chunk.

        Returns
        -------
        (is_speaking, completed_utterance)
            is_speaking : True while user is talking.
            completed_utterance : numpy float32 array of the full
                                  utterance once silence is confirmed,
                                  else None.  Audio is noise-suppressed.
        """
        # ── Input validation ──────────────────────────────────────
        if len(chunk) == 0:
            return False, None

        # Ensure float32 in [-1, 1]
        if chunk.dtype == np.int16:
            chunk_f = chunk.astype(np.float32) / 32768.0
        else:
            chunk_f = chunk.astype(np.float32)
            # Clamp to valid range (prevents distortion from bad input)
            chunk_f = np.clip(chunk_f, -1.0, 1.0)

        # ── Per-chunk DC removal (improves VAD accuracy) ──────
        chunk_f -= np.mean(chunk_f)

        rms = float(np.sqrt(np.mean(chunk_f ** 2)))

        # ── Adaptive noise gate ───────────────────────────────────────
        gate = self._get_adaptive_gate()

        if rms < gate:
            # Below adaptive gate — treat as silence/noise
            self._update_noise_estimate(rms, chunk_f)

            if self._in_speech:
                self._silence_samples += len(chunk_f)
                self._buffer.append(chunk_f)
                if self._silence_samples >= self.min_silence_samples:
                    utterance = np.concatenate(self._buffer)
                    snr = self._compute_snr(utterance)
                    self.reset()
                    if snr < self._min_snr_db:
                        print(f"[VAD] Utterance rejected: SNR={snr:.1f}dB < {self._min_snr_db}dB")
                        return False, None
                    return False, utterance
                return True, None
            # Not in speech — clean up any pre-speech accumulation
            if self._speech_samples > 0:
                self._speech_samples = 0
                self._buffer.clear()
                self._buffer_samples = 0
                self._hangover_frames = 0
            # Accumulate pre-roll for word onset capture
            self._preroll.append(chunk_f)
            return False, None

        # ── Above gate — run VAD model ────────────────────────────────
        tensor = torch.from_numpy(chunk_f)
        prob = self.model(tensor, self.sample_rate).item()
        dyn_threshold = self._get_dynamic_threshold()

        if prob >= dyn_threshold or (self._in_speech and self._hangover_frames > 0):
            # ── Speech detected (or hangover active) ─────────────────
            if prob >= dyn_threshold:
                self._hangover_frames = self._hangover_max  # refresh hangover
            else:
                self._hangover_frames -= 1  # consuming hangover
            self._silence_samples = 0
            self._buffer.append(chunk_f)
            self._speech_samples += len(chunk_f)
            self._buffer_samples += len(chunk_f)
            self._continuous_speech_samples += len(chunk_f)

            if not self._in_speech and self._speech_samples >= self.min_speech_samples:
                self._in_speech = True
                # Prepend pre-roll chunks to capture word onsets
                if self._preroll:
                    preroll_list = list(self._preroll)
                    self._buffer = preroll_list + self._buffer
                    self._buffer_samples += sum(len(c) for c in preroll_list)
                    self._preroll.clear()
                print(f"[VAD] Speech start (gate={gate:.4f}, noise={self._noise_rms:.4f}, thr={dyn_threshold:.2f})")

            # ── Speech timeout: prevent stuck-in-listening ─────────
            if self._in_speech and self._continuous_speech_samples >= self._speech_timeout_samples:
                utterance = np.concatenate(self._buffer)
                snr = self._compute_snr(utterance)
                print(f"[VAD] Speech timeout ({self._continuous_speech_samples/self.sample_rate:.1f}s), SNR={snr:.1f}dB")
                self.reset()
                if snr < self._min_snr_db:
                    print(f"[VAD] Timeout utterance rejected: low SNR")
                    return False, None
                return False, utterance

            # Hard cap: force-emit if buffer exceeds max (prevents OOM)
            if self._in_speech and self._buffer_samples >= self._max_buffer_samples:
                utterance = np.concatenate(self._buffer)
                self.reset()
                return False, utterance

            return self._in_speech, None

        else:
            # ── Below VAD threshold (silence or noise) ────────────────
            # Update noise profile with this non-speech frame
            self._update_noise_estimate(rms, chunk_f)

            if self._in_speech:
                self._silence_samples += len(chunk_f)
                self._buffer.append(chunk_f)  # keep trailing silence

                if self._silence_samples >= self.min_silence_samples:
                    # Utterance complete — validate SNR
                    utterance = np.concatenate(self._buffer)
                    snr = self._compute_snr(utterance)
                    self.reset()
                    if snr < self._min_snr_db:
                        print(f"[VAD] Utterance rejected: SNR={snr:.1f}dB < {self._min_snr_db}dB")
                        return False, None
                    print(f"[VAD] Utterance accepted: SNR={snr:.1f}dB")
                    return False, utterance

                return True, None  # still waiting for more silence

            # Not in speech — discard pre-speech accumulation
            self._speech_samples = 0
            self._buffer.clear()
            self._buffer_samples = 0
            self._hangover_frames = 0
            # Accumulate pre-roll for word onset capture
            self._preroll.append(chunk_f)
            # NOTE: Do NOT reset_states() here — preserves LSTM context
            # for better onset detection. Only reset on explicit reset().
            return False, None

    def reset(self):
        """Clear all state for a fresh utterance."""
        self._in_speech = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._continuous_speech_samples = 0
        self._hangover_frames = 0
        self._buffer.clear()
        self._buffer_samples = 0
        self._preroll.clear()
        self.model.reset_states()  # only place LSTM states are reset
        # Restore base threshold (dynamic adjustment recomputes per-chunk)
        self.threshold = self._base_threshold

    @property
    def is_speaking(self) -> bool:
        return self._in_speech

    @property
    def noise_level(self) -> float:
        """Current estimated noise floor RMS."""
        return self._noise_rms
