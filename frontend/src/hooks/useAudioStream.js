import { useRef, useState, useCallback, useEffect } from "react";

const TARGET_SR = 16000;
const CHUNK_SAMPLES = 512; // ~32 ms

// ── Global FFT data for shader consumption ─────────────────
// Updated every animation frame by whichever source is active
const FFT_SIZE = 128;
window.__fftData = new Float32Array(FFT_SIZE);
window.__fftSmoothed = new Float32Array(FFT_SIZE);
window.__fftBass = 0;
window.__fftMid = 0;
window.__fftTreble = 0;
window.__fftEnergy = 0;

function _updateFFTGlobals(analyser) {
  if (!analyser) return;
  const buf = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(buf);
  const n = Math.min(buf.length, FFT_SIZE);
  let bass = 0, mid = 0, treble = 0, total = 0;
  for (let i = 0; i < n; i++) {
    const v = buf[i] / 255;
    window.__fftData[i] = v;
    window.__fftSmoothed[i] += (v - window.__fftSmoothed[i]) * 0.3;
    total += v;
    if (i < n * 0.2) bass += v;
    else if (i < n * 0.6) mid += v;
    else treble += v;
  }
  window.__fftBass = bass / (n * 0.2);
  window.__fftMid = mid / (n * 0.4);
  window.__fftTreble = treble / (n * 0.4);
  window.__fftEnergy = total / n;
}

/**
 * Captures microphone audio via Web Audio API and streams
 * 16 kHz Int16 PCM chunks to the provided `sendAudio` callback.
 *
 * Also exposes an `playPcm(arrayBuffer)` function that plays
 * incoming TTS audio without writing any files.
 */
export default function useAudioStream(sendAudio) {
  const [micActive, setMicActive] = useState(false);
  const ctxRef = useRef(null);
  const streamRef = useRef(null);
  const workletRef = useRef(null);

  // Playback
  const playCtxRef = useRef(null);
  const nextPlayTime = useRef(0);
  const playGainRef = useRef(null);

  // FFT analysers
  const micAnalyserRef = useRef(null);
  const playAnalyserRef = useRef(null);
  const fftRafRef = useRef(null);

  // FFT tick loop — reads whichever analyser is active
  useEffect(() => {
    let running = true;
    const tick = () => {
      if (!running) return;
      if (micAnalyserRef.current) _updateFFTGlobals(micAnalyserRef.current);
      else if (playAnalyserRef.current) _updateFFTGlobals(playAnalyserRef.current);
      fftRafRef.current = requestAnimationFrame(tick);
    };
    tick();
    return () => { running = false; cancelAnimationFrame(fftRafRef.current); };
  }, []);

  // ── Start mic capture ──────────────────────────────────────

  const startMic = useCallback(async () => {
    if (micActive) return;

    console.log("[Audio] Requesting microphone access...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          // sampleRate: TARGET_SR,  // Remove this - some browsers don't support it
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      console.log("[Audio] Microphone access granted");

      console.log("[Audio] Creating AudioContext...");
      // Resume context if it was suspended
      const ctx = new AudioContext({ sampleRate: TARGET_SR });
      if (ctx.state === 'suspended') {
        await ctx.resume();
      }
      console.log("[Audio] AudioContext created, state:", ctx.state, "sampleRate:", ctx.sampleRate);

    // ScriptProcessor fallback (AudioWorklet requires served file)
    const source = ctx.createMediaStreamSource(stream);
    const processor = ctx.createScriptProcessor(CHUNK_SAMPLES, 1, 1);

    processor.onaudioprocess = (e) => {
      const float32 = e.inputBuffer.getChannelData(0);
      // Convert float32 → int16
      const int16 = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i++) {
        const s = Math.max(-1, Math.min(1, float32[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      sendAudio(int16.buffer);
    };

    // Create mic analyser for FFT
    const analyser = ctx.createAnalyser();
    analyser.fftSize = FFT_SIZE * 2;
    analyser.smoothingTimeConstant = 0.75;
    source.connect(analyser);
    micAnalyserRef.current = analyser;

    source.connect(processor);
    processor.connect(ctx.destination); // required for onaudioprocess to fire

    ctxRef.current = ctx;
    streamRef.current = stream;
    workletRef.current = processor;
    setMicActive(true);
    } catch (error) {
      console.error("[Audio] Error starting microphone:", error);
    }
  }, [micActive, sendAudio]);

  // ── Stop mic capture ───────────────────────────────────────

  const stopMic = useCallback(() => {
    workletRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    ctxRef.current?.close();
    ctxRef.current = null;
    streamRef.current = null;
    workletRef.current = null;
    micAnalyserRef.current = null;
    setMicActive(false);
  }, []);

  // ── Play incoming TTS audio (PCM Int16 in ArrayBuffer) ─────

  const playPcm = useCallback((arrayBuffer) => {
    const sampleRate = window.__tts_sample_rate || 24000;

    if (!playCtxRef.current || playCtxRef.current.state === "closed") {
      // Use browser's native sample rate — avoids resampling glitches
      playCtxRef.current = new AudioContext();
      nextPlayTime.current = 0;
    }
    const ctx = playCtxRef.current;

    if (ctx.state === "suspended") ctx.resume();

    // Decode Int16 PCM -> Float32
    const int16 = new Int16Array(arrayBuffer);
    if (int16.length === 0) return;
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768;
    }

    // Create buffer at TTS sample rate — browser resamples to native rate cleanly
    const buf = ctx.createBuffer(1, float32.length, sampleRate);
    buf.getChannelData(0).set(float32);

    // Create playback chain: source -> gain -> analyser -> destination
    if (!playAnalyserRef.current || playAnalyserRef.current.context !== ctx) {
      const gain = ctx.createGain();
      gain.gain.value = 0.85; // slight headroom to prevent clipping
      const analyser = ctx.createAnalyser();
      analyser.fftSize = FFT_SIZE * 2;
      analyser.smoothingTimeConstant = 0.8;
      gain.connect(analyser);
      analyser.connect(ctx.destination);
      playGainRef.current = gain;
      playAnalyserRef.current = analyser;
    }

    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(playGainRef.current);

    // Schedule seamlessly after previous chunk (no gap, no overlap)
    const now = ctx.currentTime;
    const startAt = Math.max(now, nextPlayTime.current);
    src.start(startAt);
    nextPlayTime.current = startAt + buf.duration;
  }, []);

  // ── Stop playback (for interrupts) ─────────────────────────

  const stopPlayback = useCallback(() => {
    if (playCtxRef.current && playCtxRef.current.state !== "closed") {
      try {
        playCtxRef.current.close();
      } catch {
        // already closed or closing
      }
      playCtxRef.current = null;
      playAnalyserRef.current = null;
      playGainRef.current = null;
      nextPlayTime.current = 0;
    }
  }, []);

  return { micActive, startMic, stopMic, playPcm, stopPlayback };
}
