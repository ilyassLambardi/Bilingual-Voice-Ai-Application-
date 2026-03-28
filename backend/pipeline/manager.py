"""
Module 4: Control Flow / Scheduling — Pipeline Manager.

This module is the **central orchestrator** for the real-time
Speech-to-Speech pipeline.  It schedules and coordinates the four
processing stages:

    1. VAD  → detect speech boundaries in streaming audio
    2. ASR  → transcribe completed utterances
    3. LLM  → generate conversational response (streaming tokens)
    4. TTS  → synthesise speech (sentence-by-sentence, concurrent)

Architecture role:
    - **Module 1 (I/O)**:  ``io_handler.py`` — message parsing & sending
    - **Module 2 (Core)**:  ``vad.py``, ``asr*.py``, ``llm*.py``, ``tts*.py``
    - **Module 3 (State)**: ``session_state.py`` + ``memory.py`` — all mutable state
    - **Module 4 (This)**:  ``manager.py`` — control flow, scheduling,
      interrupt handling, accumulation timers, pipeline sequencing

The manager reads/writes session state from Module 3, dispatches
work to the processing modules (Module 2), and communicates results
back through the I/O layer (Module 1).

Supports full interruptibility: if the user speaks while the AI is
talking, the interrupt scheduler stops TTS output, preserves the
user’s speech, and restarts the pipeline.
"""

import asyncio
import json
import logging
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np

# Module 2: Processing / Logic (Core)
from .vad import VADProcessor
from .asr import ASRProcessor
from .llm import LLMProcessor, FallbackLLM
from .tts import TTSProcessor, _detect_sentence_lang

# Module 3: Data Storage / State Management
from .session_state import SessionState, InterruptState, AudioBuffer
from .memory import LongTermMemory

# Module 1: Input / Output
from .io_handler import (
    SendFn,
    build_state_message,
    build_transcript_message,
    build_partial_transcript,
    build_audio_config,
    build_audio_end,
    build_interrupt,
    build_backchannel,
    build_ghost_text,
    build_language_shift,
    build_error,
)

log = logging.getLogger("s2s.pipeline")


def _safe_print(msg: str):
    """Print that never crashes on Windows cp1252 encoding."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode(), flush=True)


def _clean_transcript(text: str) -> str:
    """Clean raw LLM token output for display as final transcript.

    Handles: trailing whitespace, missing terminal punctuation,
    partial sentences from token limit cutoff.
    """
    text = text.strip()
    if not text:
        return text
    # Strip leading markdown artifacts
    text = re.sub(r'^[-\*]\s*', '', text).strip()
    # Ensure terminal punctuation
    if text and text[-1] not in '.!?':
        # Find last clean sentence boundary
        match = list(re.finditer(r'[.!?](?:\s|$)', text))
        if match and match[-1].start() > 10:
            text = text[:match[-1].start() + 1]
        else:
            text = text.rstrip(',;: ') + '.'
    return text


_load_pool = ThreadPoolExecutor(max_workers=1)

# Hallucination filtering is handled by the shared hallucination_filter module
# in ASR (asr.py / asr_groq.py). No duplicate filtering needed here.


class PipelineManager:
    """Module 4: Control Flow / Scheduling.

    Orchestrates VAD → ASR → LLM → TTS with full interruptibility.

    Uses:
        - Module 1 (io_handler) for message building
        - Module 2 (vad, asr, llm, tts) for processing
        - Module 3 (SessionState, LongTermMemory) for state management
    """

    def __init__(self, config):
        self.config = config

        # ── Module 3: Session state (single source of truth) ──────────
        self._session = SessionState(
            accumulation_delay=3.0,
            rate_limit=25,
        )
        self.state = "idle"

        # ── Module 2: Processing models ───────────────────────────────
        self._vad: Optional[VADProcessor] = None
        self._asr: Optional[ASRProcessor] = None
        self._llm: Optional[LLMProcessor] = None
        self._tts: Optional[TTSProcessor] = None
        self._ltm: Optional[LongTermMemory] = None
        self._models_ready = False

        # ── Pipeline lock (prevents concurrent pipeline runs) ─────────
        self._pipeline_lock = asyncio.Lock()

        # ── Interrupt mechanics (delegated to SessionState) ───────────
        self._interrupt = self._session.interrupt.event
        self._generating = False          # True while LLM+TTS is running
        self._gen_task: Optional[asyncio.Task] = None
        self._tts_task: Optional[asyncio.Task] = None
        self._interrupt_speech_frames = 0
        self._interrupt_threshold = self._session.interrupt.threshold
        self._backchannel_max_frames = self._session.interrupt.backchannel_max_frames
        self._backchannel_cooldown = 0

        # ── Language shift detection (delegated to SessionState) ──────
        self._lang_history: list[str] = []
        self._lang_history_max = 5

        # ── Ghost texting (partial ASR while speaking) ─────────────────
        self._ghost_counter = 0
        self._ghost_interval = 25

        # ── Audio accumulation (delegated to SessionState) ────────────
        self._audio_buffer: list[np.ndarray] = []
        self._accumulation_timer: Optional[asyncio.TimerHandle] = None
        self._accumulation_delay = 1.5
        self._send_fn: Optional[SendFn] = None

        # ── Rate limiting (delegated to SessionState) ─────────────────
        self._request_times: list[float] = []
        self._rate_limit = 25

    # ── Async model loading (runs in thread pool — non-blocking) ──────

    async def load_models(self):
        """Load all models at startup.  Blocks briefly but only runs once
        before any connections are accepted."""
        if self._models_ready:
            return
        self._load_models_sync()

    def _load_models_sync(self):
        """Synchronous model loading — cloud or local based on config.mode."""
        cfg = self.config
        is_cloud = cfg.mode == "cloud"
        print(f"[Manager] Loading models ({cfg.mode} mode) ...", flush=True)

        # ── LLM ────────────────────────────────────────────────────
        print("[Manager] Step 1/4: LLM ...", flush=True)
        if is_cloud:
            try:
                from .llm_groq import GroqLLM
                self._llm = GroqLLM(
                    api_key=cfg.groq_api_key,
                    model=cfg.groq_llm_model,
                    system_prompt=cfg.llm_system_prompt,
                )
            except Exception as e:
                print(f"[Manager] Groq LLM failed ({e}). Trying local fallback ...", flush=True)
                try:
                    self._llm = FallbackLLM(system_prompt=cfg.llm_system_prompt)
                except Exception as e2:
                    print(f"[Manager] All LLM failed ({e2}). Echo mode.", flush=True)
                    self._llm = None
        else:
            # Local mode: GGUF → HuggingFace fallback
            try:
                self._llm = LLMProcessor(
                    model_path=cfg.llm_model_path,
                    n_ctx=cfg.llm_n_ctx,
                    n_gpu_layers=cfg.llm_n_gpu_layers,
                    system_prompt=cfg.llm_system_prompt,
                )
            except Exception as e:
                print(f"[Manager] GGUF LLM unavailable ({e}). Loading fallback ...", flush=True)
                try:
                    self._llm = FallbackLLM(system_prompt=cfg.llm_system_prompt)
                except Exception as e2:
                    print(f"[Manager] Fallback LLM also failed ({e2}). Echo mode.", flush=True)
                    self._llm = None

        # ── VAD (always local — lightweight) ───────────────────────
        print("[Manager] Step 2/4: VAD ...", flush=True)
        self._vad = VADProcessor(
            threshold=cfg.vad_threshold,
            min_speech_ms=cfg.min_speech_ms,
            min_silence_ms=cfg.min_silence_ms,
            sample_rate=cfg.sample_rate_in,
        )

        # ── ASR ────────────────────────────────────────────────────
        print("[Manager] Step 3/4: ASR ...", flush=True)
        if is_cloud:
            try:
                from .asr_groq import GroqASR
                self._asr = GroqASR(
                    api_key=cfg.groq_api_key,
                    model=cfg.groq_asr_model,
                )
            except Exception as e:
                print(f"[Manager] Groq ASR failed ({e}). Falling back to local ...", flush=True)
                self._asr = ASRProcessor(
                    model_size=cfg.asr_model,
                    device=cfg.asr_device,
                    compute_type=cfg.asr_compute_type,
                    beam_size=cfg.asr_beam_size,
                    language=cfg.asr_language,
                )
        else:
            self._asr = ASRProcessor(
                model_size=cfg.asr_model,
                device=cfg.asr_device,
                compute_type=cfg.asr_compute_type,
                beam_size=cfg.asr_beam_size,
                language=cfg.asr_language,
            )

        # ── TTS ──────────────────────────────────────────────────────
        print(f"[Manager] Step 4/4: TTS ({cfg.tts_engine}) ...", flush=True)
        if cfg.tts_engine == "edge":
            try:
                from .tts_edge import EdgeTTSProcessor
                self._tts = EdgeTTSProcessor(
                    sample_rate=cfg.tts_sample_rate,
                )
            except Exception as e:
                print(f"[Manager] EdgeTTS failed ({e}), falling back to Silero", flush=True)
                self._tts = TTSProcessor(sample_rate=cfg.tts_sample_rate)
        elif cfg.tts_engine == "xtts":
            try:
                from .tts_xtts import XTTSProcessor
                self._tts = XTTSProcessor(
                    sample_rate=cfg.tts_sample_rate,
                    device="cuda" if cfg.mode == "cloud" else cfg.asr_device,
                )
            except Exception as e:
                print(f"[Manager] XTTS failed ({e}), falling back to Silero", flush=True)
                self._tts = TTSProcessor(sample_rate=cfg.tts_sample_rate)
        else:
            self._tts = TTSProcessor(sample_rate=cfg.tts_sample_rate)

        # ── Long-Term Memory ───────────────────────────────────────
        print("[Manager] Step 5/5: LTM ...", flush=True)
        try:
            self._ltm = LongTermMemory()
        except Exception as e:
            print(f"[Manager] LTM init failed ({e}), continuing without memory.", flush=True)
            self._ltm = None

        self._models_ready = True
        print("[Manager] All models ready!", flush=True)

    # ── Main entry: process one audio chunk ───────────────────────────

    async def handle_audio_chunk(self, raw_bytes: bytes, send: SendFn):
        """Called for every binary WebSocket frame from the client.

        `raw_bytes` is Int16 PCM at 16 kHz mono.
        """
        if not self._models_ready:
            return  # silently drop audio until models are loaded

        try:
            is_speaking, utterance = self._vad.process_chunk(
                np.frombuffer(raw_bytes, dtype=np.int16)
            )

            # ── Interrupt with backchanneling ──────────────────────
            if self._generating:
                if self._backchannel_cooldown > 0:
                    self._backchannel_cooldown -= 1

                if is_speaking:
                    self._interrupt_speech_frames += 1
                    # Only interrupt after sustained speech (not a short "mhm" / "yeah")
                    if self._interrupt_speech_frames >= self._interrupt_threshold:
                        if not self._interrupt.is_set():  # prevent re-interrupt
                            await self._interrupt_generation(send)
                else:
                    # Speech just ended — was it short enough to be a backchannel?
                    if 0 < self._interrupt_speech_frames < self._backchannel_max_frames:
                        if self._backchannel_cooldown == 0:
                            print(f"[VAD] Backchannel detected ({self._interrupt_speech_frames} frames) -- not interrupting")
                            try:
                                await send(json.dumps({"type": "backchannel"}))
                            except Exception:
                                pass
                            self._backchannel_cooldown = 30  # ~1s cooldown
                    self._interrupt_speech_frames = 0

                # ── KEY FIX: accumulate utterances during interrupt ──
                # Don't drop user speech that completes while AI is still winding down.
                # This audio will be processed after the pipeline exits.
                if utterance is not None and self._interrupt.is_set():
                    self._audio_buffer.append(utterance)
                    total = sum(len(a) for a in self._audio_buffer)
                    print(f"[VAD] Interrupt speech buffered: {len(utterance)} samples (total: {total})")
                return

            if is_speaking and self.state != "listening":
                self.state = "listening"
                print(f"[VAD] Speech detected")
                await send(json.dumps({"type": "state", "state": "listening"}))
                # Cancel pending processing — user is still talking
                if self._accumulation_timer is not None:
                    self._accumulation_timer.cancel()
                    self._accumulation_timer = None

            # Ghost texting disabled — saves CPU for main ASR pipeline

            # ── Utterance complete → accumulate, then process after silence ──
            if utterance is not None:
                self._audio_buffer.append(utterance)
                total = sum(len(a) for a in self._audio_buffer)
                print(f"[VAD] Fragment {len(self._audio_buffer)}, {len(utterance)} samples (total buffered: {total})")

                # Cancel any pending timer and restart
                if self._accumulation_timer is not None:
                    self._accumulation_timer.cancel()
                self._send_fn = send  # always use latest send ref

                # Fix 4: flush immediately if we have enough fragments
                if len(self._audio_buffer) >= 5:
                    print("[VAD] Max fragments reached — flushing immediately")
                    asyncio.ensure_future(self._flush_accumulated(self._send_fn))
                else:
                    loop = asyncio.get_running_loop()
                    self._accumulation_timer = loop.call_later(
                        self._accumulation_delay,
                        lambda: asyncio.ensure_future(self._flush_accumulated(self._send_fn))
                    )

            # ── Idle recovery: VAD rejected noise / timed out ──
            # If VAD is not speaking AND returned no utterance AND
            # we're in "listening", it means noise was rejected → go idle
            elif not is_speaking and self.state == "listening":
                # Check if VAD truly reset (not just a brief silence gap)
                if not self._vad.is_speaking and not self._audio_buffer:
                    self.state = "idle"
                    print("[VAD] Noise rejected — returning to idle")
                    await send(json.dumps({"type": "state", "state": "idle"}))
        except Exception as e:
            print(f"[Manager] handle_audio_chunk error: {e}")
            traceback.print_exc()

    # ── Audio accumulation flush ─────────────────────────────────────

    async def _flush_accumulated(self, send: SendFn):
        """Called after silence timeout — process all buffered audio as one utterance."""
        print(f"[Flush] Called: buffer={len(self._audio_buffer)} fragments, generating={self._generating}, lock={self._pipeline_lock.locked()}")
        if not self._audio_buffer:
            print("[Flush] Empty buffer — nothing to process")
            return
        # Use lock to prevent concurrent pipeline runs (race condition fix)
        if self._pipeline_lock.locked() or self._generating:
            print("[VAD] Flush deferred -- pipeline still busy, retrying in 500ms")
            # DON'T clear buffer -- preserve user's speech for retry
            # Set new timer directly (no None gap to prevent race condition)
            loop = asyncio.get_running_loop()
            self._accumulation_timer = loop.call_later(
                0.5,
                lambda: asyncio.ensure_future(self._flush_accumulated(send))
            )
            return
        # Concatenate all fragments into one continuous audio
        combined = np.concatenate(self._audio_buffer)
        n_fragments = len(self._audio_buffer)
        self._audio_buffer.clear()
        self._accumulation_timer = None
        duration = len(combined) / self.config.sample_rate_in

        # Guard: reject audio that's too short (likely noise, not real speech)
        min_dur = getattr(self.config, "min_audio_duration", 0.6)
        if duration < min_dur:
            print(f"[VAD] Audio too short ({duration:.2f}s < {min_dur}s) -- discarding")
            self.state = "idle"
            await send(json.dumps({"type": "state", "state": "idle"}))
            return

        # Apply noise cancellation off the event loop (Fix 3)
        rms_before = float(np.sqrt(np.mean(combined ** 2)))
        if self._vad and rms_before > 0.001 and getattr(self.config, 'noise_cancellation', False):
            loop = asyncio.get_running_loop()
            cleaned = await loop.run_in_executor(
                _load_pool, self._vad.clean_audio, combined
            )
            rms_after = float(np.sqrt(np.mean(cleaned ** 2)))
            # Safety: if clean_audio destroyed the signal, keep original
            if rms_after > rms_before * 0.1:
                combined = cleaned
                print(f"[VAD] Noise cancel: RMS {rms_before:.4f} -> {rms_after:.4f}")
            else:
                print(f"[VAD] Noise cancel SKIPPED (would destroy signal): RMS {rms_before:.4f} -> {rms_after:.4f}")

        print(f"[VAD] Processing {n_fragments} fragment(s), {duration:.1f}s, RMS={float(np.sqrt(np.mean(combined**2))):.4f}")
        self._generating = True  # set early to prevent double-runs (P2 fix)
        self._gen_task = asyncio.create_task(
            self._run_pipeline(combined, send)
        )

    # ── Full pipeline ─────────────────────────────────────────────────

    async def _run_pipeline(self, audio: np.ndarray, send: SendFn):
        """ASR → LLM (streaming) → TTS (sentence-by-sentence)."""
        async with self._pipeline_lock:
            self._interrupt.clear()
            try:
                # 30s hard timeout prevents indefinite hangs on API failures
                await asyncio.wait_for(
                    self._run_pipeline_inner(audio, send),
                    timeout=30.0,
                )
            except asyncio.CancelledError:
                # Fast-path: task was cancelled by _interrupt_generation
                print("[Pipeline] Cancelled by interrupt — releasing pipeline")
                # Cancel orphaned TTS worker if still running
                if self._tts_task and not self._tts_task.done():
                    self._tts_task.cancel()
                    try:
                        await self._tts_task
                    except (asyncio.CancelledError, Exception):
                        pass
                self._tts_task = None
                self._generating = False
                # Schedule processing of audio buffered during interrupt
                if self._audio_buffer:
                    total = sum(len(a) for a in self._audio_buffer)
                    print(f"[Pipeline] Post-interrupt: {len(self._audio_buffer)} fragment(s), {total} samples queued")
                    self._send_fn = send
                    loop = asyncio.get_running_loop()
                    self._accumulation_timer = loop.call_later(
                        0.5,
                        lambda: asyncio.ensure_future(self._flush_accumulated(self._send_fn))
                    )
            except asyncio.TimeoutError:
                print("[Pipeline] TIMEOUT: pipeline exceeded 30s, aborting")
                self._generating = False
                self.state = "idle"
                await send(json.dumps({"type": "state", "state": "idle"}))
                await send(json.dumps({"type": "error", "message": "Response timed out. Please try again."}))
            except Exception as e:
                print(f"[Pipeline] ERROR: {e}")
                traceback.print_exc()
                self.state = "idle"
                await send(json.dumps({"type": "state", "state": "idle"}))
                await send(json.dumps({"type": "error", "message": str(e)}))
            finally:
                # ALWAYS reset generating flag — prevents permanent stuck state
                # when _run_pipeline_inner returns early (empty ASR, rate limit, etc.)
                self._generating = False

    def _check_rate_limit(self) -> bool:
        """Return True if within rate limit, False if exceeded."""
        now = time.time()
        # Remove entries older than 60 seconds
        self._request_times = [t for t in self._request_times if now - t < 60]
        if len(self._request_times) >= self._rate_limit:
            return False
        self._request_times.append(now)
        return True

    async def _run_pipeline_inner(self, audio: np.ndarray, send: SendFn):
        """Inner pipeline — errors bubble up to _run_pipeline for handling."""

        # ── Rate limit check ─────────────────────────────────────────
        if not self._check_rate_limit():
            print("[Manager] Rate limit hit -- skipping request")
            await send(json.dumps({
                "type": "transcript",
                "role": "assistant",
                "text": "Hold on, let me catch my breath! Too many requests. Wait a moment.",
                "language": "en",
                "time": 0,
            }))
            self.state = "idle"
            await send(json.dumps({"type": "state", "state": "idle"}))
            return

        # ── Stage 1: ASR ──────────────────────────────────────────────
        t0 = time.time()
        self.state = "thinking"
        await send(json.dumps({"type": "state", "state": "thinking"}))

        try:
            asr_result = await self._asr.transcribe(audio)
        except Exception as e:
            print(f"[ASR] TRANSCRIPTION FAILED: {e}")
            traceback.print_exc()
            await send(json.dumps({"type": "error", "message": f"Speech recognition failed: {e}"}))
            self.state = "idle"
            await send(json.dumps({"type": "state", "state": "idle"}))
            return

        user_text = asr_result["text"]
        lang = asr_result["language"] or "en"

        # Guard: ASR returned empty text (already filtered by hallucination_filter)
        if not user_text.strip():
            rms = float(np.sqrt(np.mean(audio ** 2)))
            print(f"[ASR] Empty transcription — skipping (audio: {len(audio)/16000:.1f}s, RMS={rms:.4f})")
            self.state = "idle"
            await send(json.dumps({"type": "state", "state": "idle"}))
            return

        await send(json.dumps({
            "type": "transcript",
            "role": "user",
            "text": user_text,
            "language": lang,
            "time": round(time.time() - t0, 3),
        }))

        t_asr = time.time() - t0
        _safe_print(f"[ASR] {t_asr:.1f}s [{lang}] {user_text}")

        # ── Language shift detection ──────────────────────────────
        prev_langs = self._lang_history[-3:]  # last 3 messages
        self._lang_history.append(lang)
        if len(self._lang_history) > self._lang_history_max:
            self._lang_history = self._lang_history[-self._lang_history_max:]

        # Detect shift: if last 3 were all one lang and now it's different
        if prev_langs and all(l != lang for l in prev_langs) and len(prev_langs) >= 2:
            print(f"[Lang] Language shift detected: {prev_langs[-1]} -> {lang}")
            try:
                await send(json.dumps({
                    "type": "language_shift",
                    "from": prev_langs[-1],
                    "to": lang,
                }))
            except Exception:
                pass

        # LTM recall: retrieve relevant memories for context (P5 fix)
        memory_context = ""
        if self._ltm and getattr(self.config, 'ltm_recall_enabled', False):
            try:
                memories = self._ltm.recall(user_text, limit=2)
                if memories:
                    memory_context = "\n".join(memories)
                    print(f"[LTM] Recalled {len(memories)} memories for context")
            except Exception as e:
                print(f"[LTM] Recall failed: {e}")

        # ── Start generating immediately (no filler delay) ────────────
        self._generating = True
        self.state = "speaking"
        await send(json.dumps({"type": "state", "state": "speaking"}))
        await send(json.dumps({
            "type": "audio_config",
            "sample_rate": self._tts.get_sample_rate(),
        }))
        t_llm_start = time.time()

        full_response = ""
        sentence_buf = ""
        sentence_idx = 0

        # Inject memory context into the user prompt if available
        llm_input = user_text
        if memory_context:
            llm_input = f"{user_text}\n\n(Context from previous conversations:\n{memory_context})"

        # ── Concurrent TTS: LLM pushes sentences, TTS worker synthesizes ──
        tts_queue: asyncio.Queue = asyncio.Queue()

        async def _tts_worker():
            """Process sentences from queue in order — runs concurrently with LLM."""
            while True:
                item = await tts_queue.get()
                if item is None:  # sentinel = done
                    break
                if self._interrupt.is_set():
                    continue  # drain remaining items
                sent_text, sent_lang, sent_idx = item
                await send(json.dumps({
                    "type": "partial_transcript",
                    "role": "assistant",
                    "text": sent_text,
                    "language": sent_lang,
                    "index": sent_idx,
                }))
                await self._speak_sentence(sent_text, sent_lang, send)

        tts_task = asyncio.create_task(_tts_worker())
        self._tts_task = tts_task

        if self._llm is not None:
            async for token in self._llm.stream(
                llm_input,
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
                lang=lang,
            ):
                if self._interrupt.is_set():
                    break

                sentence_buf += token
                full_response += token

                # Flush on sentence boundary — shorter chunks = faster TTS
                should_flush = False
                if any(sentence_buf.rstrip().endswith(p) for p in ".?!"):
                    should_flush = True
                elif len(sentence_buf) > 50 and any(sentence_buf.rstrip().endswith(p) for p in ",;:"):
                    should_flush = True  # break long clauses for natural pacing

                if should_flush:
                    sentence = sentence_buf.strip()
                    sentence_buf = ""
                    sentence_idx += 1
                    await tts_queue.put((sentence, lang, sentence_idx))

                    # Soft limit: stop after 5 sentences for voice pacing
                    if sentence_idx >= 5:
                        break

            # Flush remaining text
            if sentence_buf.strip() and not self._interrupt.is_set() and sentence_idx < 5:
                sentence = sentence_buf.strip()
                sentence_idx += 1
                await tts_queue.put((sentence, lang, sentence_idx))
        else:
            # Echo mode (no LLM available) — repeat back in detected language
            full_response = user_text
            await tts_queue.put((full_response, lang, 1))

        # Signal TTS worker to finish, then wait
        await tts_queue.put(None)
        await tts_task

        # ── Done ──────────────────────────────────────────────────────
        t_total = time.time() - t0
        t_llm_tts = time.time() - t_llm_start
        print(f"[Perf] ASR={t_asr:.1f}s  LLM+TTS={t_llm_tts:.1f}s  TOTAL={t_total:.1f}s")

        self._tts_task = None
        if self._interrupt.is_set():
            # Interrupted via flag (slow path) — release generating flag
            print("[Pipeline] Exited gracefully after interrupt (flag)")
            self._generating = False
            # Schedule processing of any audio buffered during the interrupt
            if self._audio_buffer:
                total = sum(len(a) for a in self._audio_buffer)
                print(f"[Pipeline] Post-interrupt: {len(self._audio_buffer)} fragment(s), {total} samples queued")
                self._send_fn = send
                loop = asyncio.get_running_loop()
                self._accumulation_timer = loop.call_later(
                    0.5,
                    lambda: asyncio.ensure_future(self._flush_accumulated(self._send_fn))
                )
            else:
                # No buffered audio — user may still be speaking, stay listening
                pass
        else:
            # Normal completion — send final transcript and go idle
            clean_response = _clean_transcript(full_response)
            await send(json.dumps({
                "type": "transcript",
                "role": "assistant",
                "text": clean_response,
                "language": lang,
            }))
            await send(json.dumps({"type": "audio_end"}))

            # LTM: store conversation exchange
            if self._ltm and full_response.strip():
                try:
                    self._ltm.store_conversation(user_text, full_response, lang)
                except Exception:
                    pass

            self._generating = False  # reset before state change (P7 fix)
            self.state = "idle"
            await send(json.dumps({"type": "state", "state": "idle"}))

    # ── TTS helper ────────────────────────────────────────────────────

    async def _speak_sentence(self, text: str, lang: str, send: SendFn):
        """Synthesise one sentence and stream the PCM bytes.

        Uses per-sentence language detection so the correct TTS model is
        used even when the LLM responds in a different language than ASR detected.
        """
        if self._interrupt.is_set() or not text:
            return
        try:
            # Per-sentence language detection overrides ASR hint
            detected = _detect_sentence_lang(text)
            if detected != lang:
                print(f"[TTS] Lang override: ASR={lang} -> sentence={detected} for: {text[:50]}")
            tts_lang = detected
            _safe_print(f"[TTS] Synthesizing ({tts_lang}): {text[:60]}...")
            pcm_bytes = await self._tts.synthesize(text, lang=tts_lang)
            if not pcm_bytes:
                _safe_print(f"[TTS] Skipped empty audio for: {text[:40]}")
                return
            print(f"[TTS] Generated {len(pcm_bytes)} bytes")
            if not self._interrupt.is_set():
                # Stream audio in ~100ms chunks for faster time-to-first-audio
                chunk_size = int(self._tts.get_sample_rate() * 2 * 0.1)  # 100ms of int16
                for i in range(0, len(pcm_bytes), chunk_size):
                    if self._interrupt.is_set():
                        break
                    await send(pcm_bytes[i:i + chunk_size])
        except Exception as e:
            print(f"[TTS] Error: {e}")
            traceback.print_exc()

    # ── Ghost texting (partial ASR while speaking) ──────────────────

    async def _send_ghost_text(self, send: SendFn):
        """Run ASR on current speech buffer and send partial result."""
        try:
            # Snapshot the current buffer (don't modify VAD state)
            buf_copy = [b.copy() for b in self._vad._buffer]
            if not buf_copy:
                return
            audio = np.concatenate(buf_copy)
            if len(audio) < 4000:  # too short for meaningful ASR
                return

            result = await self._asr.transcribe_fast(audio)
            ghost = result.get("text", "").strip()
            if ghost:
                await send(json.dumps({
                    "type": "ghost_text",
                    "text": ghost,
                }))
        except Exception:
            pass  # ghost texting is best-effort, never block pipeline

    # ── Interrupt logic ───────────────────────────────────────────────

    async def _interrupt_generation(self, send: SendFn):
        """User spoke while AI was talking → stop AI immediately, start listening.

        Cancels the running generation task and TTS worker for instant stop,
        then sets the interrupt flag as a backup for any code that checks it.
        """
        print("[Manager] Interrupted by user! Stopping AI speech.")
        self._interrupt.set()
        self._interrupt_speech_frames = 0

        # Cancel pending accumulation timer (user is still talking)
        if self._accumulation_timer is not None:
            self._accumulation_timer.cancel()
            self._accumulation_timer = None

        # Force-cancel running tasks for immediate stop
        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
        if self._gen_task and not self._gen_task.done():
            self._gen_task.cancel()

        # Do NOT reset VAD or clear audio buffer — the user is still speaking.
        # VAD continues tracking their speech. Completed utterances are now
        # accumulated in handle_audio_chunk (the KEY FIX above).

        # Tell frontend to stop playback immediately
        self.state = "listening"
        try:
            await send(build_audio_end())
            await send(build_state_message("listening"))
            await send(build_interrupt())
        except Exception:
            pass

    # ── Text chat (skip VAD/ASR) ─────────────────────────────────────

    async def handle_text_chat(self, text: str, send: SendFn):
        """Handle typed text input — skip VAD/ASR, go straight to LLM→TTS."""
        if not self._models_ready:
            return
        if self._generating:
            return  # already processing

        text = text.strip()
        if not text:
            return

        # Simple language detection for typed text
        lang = self._detect_text_language(text)

        # Rate limit
        if not self._check_rate_limit():
            await send(json.dumps({
                "type": "transcript", "role": "assistant",
                "text": "Too many requests. Please wait a moment.",
                "language": "en",
            }))
            return

        # Send user transcript
        await send(json.dumps({
            "type": "transcript", "role": "user",
            "text": text, "language": lang, "time": 0,
        }))

        # Run LLM → TTS
        self._interrupt.clear()
        self._generating = True
        self.state = "thinking"
        await send(json.dumps({"type": "state", "state": "thinking"}))

        try:
            await asyncio.wait_for(
                self._run_text_pipeline(text, lang, send),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            print("[Chat] TIMEOUT: text pipeline exceeded 30s, aborting")
            await send(json.dumps({"type": "error", "message": "Response timed out. Please try again."}))
        except asyncio.CancelledError:
            print("[Chat] Cancelled by interrupt")
            if self._tts_task and not self._tts_task.done():
                self._tts_task.cancel()
                try:
                    await self._tts_task
                except (asyncio.CancelledError, Exception):
                    pass
            self._tts_task = None
        except Exception as e:
            print(f"[Manager] Text chat error: {e}")
            traceback.print_exc()
        finally:
            if not self._interrupt.is_set():
                self._generating = False
                self.state = "idle"
                await send(json.dumps({"type": "state", "state": "idle"}))
            else:
                self._generating = False

    async def _run_text_pipeline(self, user_text: str, lang: str, send: SendFn):
        """LLM → TTS pipeline for typed text input (concurrent TTS)."""
        t0 = time.time()

        self.state = "speaking"
        await send(json.dumps({"type": "state", "state": "speaking"}))
        await send(json.dumps({
            "type": "audio_config",
            "sample_rate": self._tts.get_sample_rate(),
        }))

        full_response = ""
        sentence_buf = ""
        sentence_idx = 0

        # Concurrent TTS queue (same pattern as voice pipeline)
        tts_queue: asyncio.Queue = asyncio.Queue()

        async def _tts_worker():
            while True:
                item = await tts_queue.get()
                if item is None:
                    break
                if self._interrupt.is_set():
                    continue
                sent_text, sent_lang, sent_idx = item
                await send(json.dumps({
                    "type": "partial_transcript", "role": "assistant",
                    "text": sent_text, "language": sent_lang, "index": sent_idx,
                }))
                await self._speak_sentence(sent_text, sent_lang, send)

        tts_task = asyncio.create_task(_tts_worker())
        self._tts_task = tts_task

        if self._llm is not None:
            async for token in self._llm.stream(
                user_text,
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
                lang=lang,
            ):
                if self._interrupt.is_set():
                    break
                sentence_buf += token
                full_response += token

                should_flush = False
                if any(sentence_buf.rstrip().endswith(p) for p in ".?!"):
                    should_flush = True
                elif len(sentence_buf) > 50 and any(sentence_buf.rstrip().endswith(p) for p in ",;:"):
                    should_flush = True

                if should_flush:
                    sentence = sentence_buf.strip()
                    sentence_buf = ""
                    sentence_idx += 1
                    await tts_queue.put((sentence, lang, sentence_idx))
                    if sentence_idx >= 5:
                        break

            if sentence_buf.strip() and not self._interrupt.is_set() and sentence_idx < 5:
                sentence = sentence_buf.strip()
                sentence_idx += 1
                await tts_queue.put((sentence, lang, sentence_idx))
        else:
            full_response = f"Echo: {user_text}"
            await tts_queue.put((full_response, lang, 1))

        await tts_queue.put(None)
        await tts_task

        if not self._interrupt.is_set():
            clean_response = _clean_transcript(full_response)
            await send(json.dumps({
                "type": "transcript", "role": "assistant",
                "text": clean_response, "language": lang,
            }))
            await send(json.dumps({"type": "audio_end"}))

        if self._ltm and full_response.strip() and not self._interrupt.is_set():
            try:
                self._ltm.store_conversation(user_text, full_response, lang)
            except Exception:
                pass

        print(f"[Chat] LLM+TTS={time.time() - t0:.1f}s [{lang}]")

    @staticmethod
    def _detect_text_language(text: str) -> str:
        """Detect if typed text is German or English.

        English is the strong default.  Only returns 'de' when the text
        is clearly German (special chars + keyword, or >= 40% keywords).
        """
        _DE_CHARS = {"ä", "ö", "ü", "ß"}
        _DE_WORDS = {
            "ich", "bin", "ist", "und", "der", "die",
            "das", "ein", "eine", "nicht", "wie", "was", "hallo", "danke",
            "bitte", "gut", "mir", "dir", "mich", "dich", "kannst", "hast",
            "wir", "sie", "haben", "werden", "können", "möchte", "geht",
        }
        words = set(text.lower().split())
        chars = set(text.lower())
        has_de_chars = bool(chars & _DE_CHARS)
        de_word_hits = len(words & _DE_WORDS)
        # Require special chars + at least 1 keyword, or >= 40% keywords
        if has_de_chars and de_word_hits >= 1:
            return "de"
        if len(words) >= 2 and de_word_hits / len(words) >= 0.4:
            return "de"
        return "en"

    # ── Session reset ─────────────────────────────────────────────────

    async def clear(self, send: SendFn):
        # Summarize conversation into LTM before clearing
        if self._ltm and self._llm and hasattr(self._llm, '_history'):
            try:
                self._ltm.summarize_and_store(self._llm._history)
            except Exception:
                pass

        # Cancel running generation/TTS tasks to prevent orphaned work (P4 fix)
        self._interrupt.set()
        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
            try:
                await self._tts_task
            except (asyncio.CancelledError, Exception):
                pass
        self._tts_task = None
        if self._gen_task and not self._gen_task.done():
            self._gen_task.cancel()
            try:
                await self._gen_task
            except (asyncio.CancelledError, Exception):
                pass
        self._gen_task = None

        if self._llm:
            self._llm.clear_history()
        if self._vad:
            self._vad.reset()
        # Cancel any pending accumulation timer
        if self._accumulation_timer is not None:
            self._accumulation_timer.cancel()
            self._accumulation_timer = None
        # Clear all buffered audio
        self._audio_buffer.clear()
        # Reset language history
        self._lang_history.clear()
        # Reset interrupt and generation state
        self._generating = False
        self._interrupt.clear()
        self._interrupt_speech_frames = 0
        self._backchannel_cooldown = 0
        self.state = "idle"
        await send(build_state_message("idle"))
