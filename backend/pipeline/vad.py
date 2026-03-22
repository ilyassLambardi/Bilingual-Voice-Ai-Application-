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

    def _clean_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply spectral noise suppression to an utterance."""
        if not self._spectral_enabled or self._noise_spectrum is None:
            return audio
        try:
            return _spectral_subtract(audio, self._noise_spectrum)
        except Exception:
            return audio  # fallback: return original

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
        # Ensure float32 in [-1, 1]
        if chunk.dtype == np.int16:
            chunk_f = chunk.astype(np.float32) / 32768.0
        else:
            chunk_f = chunk.astype(np.float32)

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
                    # Apply noise suppression before returning
                    return False, self._clean_audio(utterance)
                return True, None
            # Not in speech — clean up any pre-speech accumulation
            if self._speech_samples > 0:
                self._speech_samples = 0
                self._buffer.clear()
                self.model.reset_states()
            return False, None

        # ── Above gate — run VAD model ────────────────────────────────
        # Update noise estimate only if we're not in speech
        if not self._in_speech:
            # Cautious: only update noise if probability is low
            pass  # we'll update below based on VAD probability

        tensor = torch.from_numpy(chunk_f)
        prob = self.model(tensor, self.sample_rate).item()
        dyn_threshold = self._get_dynamic_threshold()

        if prob >= dyn_threshold:
            # ── Speech detected ──────────────────────────────────────
            self._silence_samples = 0
            self._buffer.append(chunk_f)
            self._speech_samples += len(chunk_f)
            self._continuous_speech_samples += len(chunk_f)

            if not self._in_speech and self._speech_samples >= self.min_speech_samples:
                self._in_speech = True
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
                return False, self._clean_audio(utterance)

            # Hard cap: force-emit if buffer exceeds max (prevents OOM)
            total = sum(len(b) for b in self._buffer)
            if self._in_speech and total >= self._max_buffer_samples:
                utterance = np.concatenate(self._buffer)
                self.reset()
                return False, self._clean_audio(utterance)

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
                    return False, self._clean_audio(utterance)

                return True, None  # still waiting for more silence

            # Not in speech — discard and reset model state
            self._speech_samples = 0
            self._buffer.clear()
            self.model.reset_states()
            return False, None

    def reset(self):
        """Clear all state for a fresh utterance."""
        self._in_speech = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._continuous_speech_samples = 0
        self._buffer.clear()
        self.model.reset_states()
        # Restore base threshold (dynamic adjustment recomputes per-chunk)
        self.threshold = self._base_threshold

    @property
    def is_speaking(self) -> bool:
        return self._in_speech

    @property
    def noise_level(self) -> float:
        """Current estimated noise floor RMS."""
        return self._noise_rms
