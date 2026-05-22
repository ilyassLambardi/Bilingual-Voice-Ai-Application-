"""
Module 4 (Control Flow): Pipeline Manager — CLOUD mode.

Orchestrates VAD -> ASR -> LLM -> TTS with full interruptibility.
Uses Groq APIs for ASR + LLM and Edge TTS for speech synthesis.
VAD always runs locally (lightweight, CPU-based).

Uses:
    - io_handler for message building
    - Groq ASR, Groq LLM, Edge TTS for processing
    - Local VAD for voice activity detection
    - LongTermMemory for persistent conversational memory
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Cloud pipeline modules ─────────────────────────────────────────
from .asr_groq import GroqASR
from .llm_groq import GroqLLM
from .tts_edge import EdgeTTSProcessor
from .lang_detect import detect_sentence_lang
from .llm_common import clean_response
from .io_handler import SendFn, build_state_message, build_audio_end, build_interrupt

# VAD is always local (lightweight, CPU-based)
from ..local.vad import VADProcessor
# Fallbacks to local models if cloud APIs fail
from ..local.llm import FallbackLLM
from ..local.asr import ASRProcessor
from ..local.tts import TTSProcessor

_load_pool = ThreadPoolExecutor(max_workers=2)


class PipelineManager:
    """Cloud-mode pipeline orchestrator.

    Orchestrates VAD -> ASR -> LLM -> TTS with full interruptibility.
    ASR + LLM via Groq API, TTS via Edge TTS, VAD local.
    """

    def __init__(self, config):
        self.config = config

        self.state = "idle"

        # ── Processing models ──────────────────────────────────────────
        self._vad: Optional[VADProcessor] = None
        self._asr: Any = None
        self._llm: Any = None
        self._tts: Any = None
        self._ltm = None
        self._models_ready = False

        # ── Pipeline lock (prevents concurrent pipeline runs) ─────────
        self._pipeline_lock = asyncio.Lock()

        # ── Interrupt mechanics ──────────────────────────────────────
        self._interrupt = asyncio.Event()
        self._generating = False          # True while LLM+TTS is running
        self._gen_task: Optional[asyncio.Task] = None
        self._tts_task: Optional[asyncio.Task] = None
        self._interrupt_speech_frames = 0
        self._interrupt_threshold = 4         # ~128ms sustained speech to trigger
        self._backchannel_max_frames = 12     # ~384ms — shorter = backchannel
        self._backchannel_cooldown = 0

        # ── Language shift detection ────────────────────────────────
        self._lang_history: list[str] = []
        self._lang_history_max = 5

        # ── Audio accumulation ────────────────────────────────────
        self._audio_buffer: list[np.ndarray] = []
        self._accumulation_timer: Optional[asyncio.TimerHandle] = None
        self._accumulation_delay = 1.5
        self._send_fn: Optional[SendFn] = None

        # ── Rate limiting ─────────────────────────────────────────
        self._request_times: list[float] = []
        self._rate_limit = 25

    # ── Async model loading (runs in thread pool — non-blocking) ──────

    async def load_models(self):
        """Load all models at startup.  Blocks briefly but only runs once
        before any connections are accepted."""
        if self._models_ready:
            return
        self._load_models_sync()
        await self._warmup_tts()

    async def _warmup_tts(self):
        """Pre-warm TTS to avoid ~15-20s cold-start on first user interaction."""
        if not self._tts:
            return
        try:
            logger.info("[Manager] Warming up TTS (first call is slow) ...")
            t0 = time.time()
            await self._tts.synthesize("Hello.", lang="en")
            logger.info("[Manager] TTS warmup done in %.1fs", time.time() - t0)
        except Exception as e:
            logger.warning("[Manager] TTS warmup failed: %s", e)

    def _load_models_sync(self):
        """Synchronous model loading — cloud APIs + local VAD."""
        cfg = self.config
        logger.info("[Manager] Loading models (cloud mode) ...")

        # ── LLM (Groq API, with local fallback) ─────────────────────
        logger.info("[Manager] Step 1/5: LLM ...")
        try:
            self._llm = GroqLLM(
                api_key=cfg.groq_api_key,
                model=cfg.groq_llm_model,
                system_prompt=cfg.llm_system_prompt,
            )
        except Exception as e:
            logger.warning("[Manager] Groq LLM failed (%s). Trying local fallback ...", e)
            try:
                self._llm = FallbackLLM(system_prompt=cfg.llm_system_prompt)
            except Exception as e2:
                logger.warning("[Manager] Fallback LLM also failed (%s). Echo mode.", e2)
                self._llm = None

        # ── VAD (always local — lightweight) ───────────────────
        logger.info("[Manager] Step 2/5: VAD ...")
        self._vad = VADProcessor(
            threshold=cfg.vad_threshold,
            min_speech_ms=cfg.min_speech_ms,
            min_silence_ms=cfg.min_silence_ms,
            sample_rate=cfg.sample_rate_in,
        )

        # ── ASR (Groq Whisper API, with local fallback) ────────────
        logger.info("[Manager] Step 3/5: ASR ...")
        try:
            self._asr = GroqASR(
                api_key=cfg.groq_api_key,
                model=cfg.groq_asr_model,
            )
        except Exception as e:
            logger.warning("[Manager] Groq ASR failed (%s). Falling back to local ...", e)
            self._asr = ASRProcessor(
                model_size=cfg.asr_model,
                device=cfg.asr_device,
                compute_type=cfg.asr_compute_type,
                beam_size=cfg.asr_beam_size,
                language=cfg.asr_language,
            )

        # ── TTS (Edge TTS, with Silero fallback) ──────────────────
        logger.info("[Manager] Step 4/5: TTS (edge) ...")
        try:
            self._tts = EdgeTTSProcessor(
                sample_rate=cfg.tts_sample_rate,
            )
        except Exception as e:
            logger.warning("[Manager] EdgeTTS failed (%s), falling back to Silero", e)
            self._tts = TTSProcessor(sample_rate=cfg.tts_sample_rate)

        # ── Long-Term Memory (disabled — cloud mode has no cross-session memory)
        self._ltm = None

        self._models_ready = True
        logger.info("[Manager] All models ready!")

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
                            logger.info("[VAD] Backchannel detected (%d frames) -- not interrupting", self._interrupt_speech_frames)
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
                    logger.info("[VAD] Interrupt speech buffered: %d samples (total: %d)", len(utterance), total)
                return

            if is_speaking and self.state != "listening":
                self.state = "listening"
                logger.info("[VAD] Speech detected")
                await send(json.dumps({"type": "state", "state": "listening"}))
                # Cancel pending processing — user is still talking
                if self._accumulation_timer is not None:
                    self._accumulation_timer.cancel()
                    self._accumulation_timer = None

            # ── Utterance complete -> accumulate, then process after silence ──
            if utterance is not None:
                self._audio_buffer.append(utterance)
                total = sum(len(a) for a in self._audio_buffer)
                logger.info("[VAD] Fragment %d, %d samples (total buffered: %d)", len(self._audio_buffer), len(utterance), total)

                # Cancel any pending timer and restart
                if self._accumulation_timer is not None:
                    self._accumulation_timer.cancel()
                self._send_fn = send  # always use latest send ref
                send_fn = self._send_fn

                # Fix 4: flush immediately if we have enough fragments
                if len(self._audio_buffer) >= 5:
                    logger.info("[VAD] Max fragments reached -- flushing immediately")
                    asyncio.ensure_future(self._flush_accumulated(send_fn))
                else:
                    loop = asyncio.get_running_loop()
                    self._accumulation_timer = loop.call_later(
                        self._accumulation_delay,
                        lambda: asyncio.ensure_future(self._flush_accumulated(send_fn))
                    )

            # ── Idle recovery: VAD rejected noise / timed out ──
            # If VAD is not speaking AND returned no utterance AND
            # we're in "listening", it means noise was rejected -> go idle
            elif not is_speaking and self.state == "listening":
                # Check if VAD truly reset (not just a brief silence gap)
                if not self._vad.is_speaking and not self._audio_buffer:
                    self.state = "idle"
                    logger.info("[VAD] Noise rejected -- returning to idle")
                    await send(json.dumps({"type": "state", "state": "idle"}))
        except Exception as e:
            logger.error("[Manager] handle_audio_chunk error: %s", e, exc_info=True)

    # ── Audio accumulation flush ─────────────────────────────────────

    async def _flush_accumulated(self, send: SendFn):
        """Called after silence timeout — process all buffered audio as one utterance."""
        logger.info("[Flush] Called: buffer=%d fragments, generating=%s, lock=%s", len(self._audio_buffer), self._generating, self._pipeline_lock.locked())
        if not self._audio_buffer:
            logger.info("[Flush] Empty buffer -- nothing to process")
            return
        # Use lock to prevent concurrent pipeline runs (race condition fix)
        if self._pipeline_lock.locked() or self._generating:
            logger.info("[VAD] Flush deferred -- pipeline still busy, retrying in 500ms")
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
        min_dur = self.config.min_audio_duration
        if duration < min_dur:
            logger.info("[VAD] Audio too short (%.2fs < %.2fs) -- discarding", duration, min_dur)
            self.state = "idle"
            await send(json.dumps({"type": "error", "message": "Audio too short. Please speak a bit longer."}))
            await send(json.dumps({"type": "state", "state": "idle"}))
            return

        # Apply noise cancellation off the event loop (Fix 3)
        rms_before = float(np.sqrt(np.mean(combined ** 2)))
        if self._vad and rms_before > 0.001 and self.config.noise_cancellation:
            loop = asyncio.get_running_loop()
            cleaned = await loop.run_in_executor(
                _load_pool, self._vad.clean_audio, combined
            )
            rms_after = float(np.sqrt(np.mean(cleaned ** 2)))
            # Safety: if clean_audio destroyed too much signal, keep original
            if rms_after > rms_before * 0.35:
                combined = cleaned
                logger.info("[VAD] Noise cancel: RMS %.4f -> %.4f", rms_before, rms_after)
            else:
                logger.info("[VAD] Noise cancel SKIPPED (would destroy signal): RMS %.4f -> %.4f", rms_before, rms_after)

        logger.info("[VAD] Processing %d fragment(s), %.1fs, RMS=%.4f", n_fragments, duration, float(np.sqrt(np.mean(combined**2))))
        self._generating = True  # set early to prevent double-runs (P2 fix)
        try:
            self._gen_task = asyncio.create_task(
                self._run_pipeline(combined, send)
            )
        except Exception as e:
            logger.error("[VAD] Task creation failed: %s", e)
            self._generating = False
            self.state = "idle"
            await send(json.dumps({"type": "state", "state": "idle"}))

    # ── Full pipeline ─────────────────────────────────────────────────

    async def _run_pipeline(self, audio: np.ndarray, send: SendFn):
        """ASR -> LLM (streaming) -> TTS (sentence-by-sentence)."""
        async with self._pipeline_lock:
            self._interrupt.clear()
            try:
                # 60s hard timeout
                await asyncio.wait_for(
                    self._run_pipeline_inner(audio, send),
                    timeout=60.0,
                )
            except asyncio.CancelledError:
                # Fast-path: task was cancelled by _interrupt_generation
                logger.info("[Pipeline] Cancelled by interrupt -- releasing pipeline")
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
                    logger.info("[Pipeline] Post-interrupt: %d fragment(s), %d samples queued", len(self._audio_buffer), total)
                    self._send_fn = send
                    send_fn = self._send_fn
                    loop = asyncio.get_running_loop()
                    self._accumulation_timer = loop.call_later(
                        0.5,
                        lambda: asyncio.ensure_future(self._flush_accumulated(send_fn))
                    )
            except asyncio.TimeoutError:
                logger.warning("[Pipeline] TIMEOUT: pipeline exceeded 60s, aborting")
                self._generating = False
                self.state = "idle"
                await send(json.dumps({"type": "state", "state": "idle"}))
                await send(json.dumps({"type": "error", "message": "Response timed out. Please try again."}))
            except Exception as e:
                logger.error("[Pipeline] ERROR: %s", e, exc_info=True)
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
            logger.info("[Manager] Rate limit hit -- skipping request")
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
            # Always use dual-pass (no hint) — lets ASR detect language
            # fresh each time. Auto-hinting caused sticky language locks
            # where switching from DE back to EN was impossible.
            asr_result = await self._asr.transcribe(audio, lang_hint=None)
        except Exception as e:
            logger.error("[ASR] TRANSCRIPTION FAILED: %s", e, exc_info=True)
            await send(json.dumps({"type": "error", "message": f"Speech recognition failed: {e}"}))
            self.state = "idle"
            await send(json.dumps({"type": "state", "state": "idle"}))
            return

        user_text = asr_result["text"]
        lang = asr_result["language"] or "en"

        # Guard: ASR returned empty text (already filtered by hallucination_filter)
        if not user_text.strip():
            rms = float(np.sqrt(np.mean(audio ** 2)))
            logger.info("[ASR] Empty transcription -- skipping (audio: %.1fs, RMS=%.4f)", len(audio) / 16000, rms)
            self.state = "idle"
            await send(json.dumps({"type": "error", "message": "Couldn't understand that. Please try again."}))
            await send(json.dumps({"type": "audio_end"}))
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
        logger.info("[ASR] %.1fs [%s] %s", t_asr, lang, user_text)

        # ── Language shift detection ──────────────────────────────
        prev_langs = self._lang_history[-3:]  # last 3 messages
        self._lang_history.append(lang)
        if len(self._lang_history) > self._lang_history_max:
            self._lang_history = self._lang_history[-self._lang_history_max:]

        # Detect shift: if last 3 were all one lang and now it's different
        if prev_langs and all(l != lang for l in prev_langs) and len(prev_langs) >= 2:
            logger.info("[Lang] Language shift detected: %s -> %s", prev_langs[-1], lang)
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
        if self._ltm and self.config.ltm_recall_enabled:
            try:
                memories = self._ltm.recall(user_text, limit=2)
                if memories:
                    memory_context = "\n".join(memories)
                    logger.info("[LTM] Recalled %d memories for context", len(memories))
            except Exception as e:
                logger.warning("[LTM] Recall failed: %s", e)

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

        try:
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

                        # Soft limit: stop after 8 sentences for voice pacing
                        if sentence_idx >= 8:
                            break

                # Flush remaining text
                if sentence_buf.strip() and not self._interrupt.is_set() and sentence_idx < 8:
                    sentence = sentence_buf.strip()
                    sentence_idx += 1
                    await tts_queue.put((sentence, lang, sentence_idx))
            else:
                # Echo mode (no LLM available) — repeat back in detected language
                full_response = user_text
                await tts_queue.put((full_response, lang, 1))
        finally:
            # Always signal TTS worker to finish — prevents deadlock on exception
            await tts_queue.put(None)
        await tts_task

        # ── Done ──────────────────────────────────────────────────────
        t_total = time.time() - t0
        t_llm_tts = time.time() - t_llm_start
        logger.info("[Perf] ASR=%.1fs  LLM+TTS=%.1fs  TOTAL=%.1fs", t_asr, t_llm_tts, t_total)

        self._tts_task = None
        if self._interrupt.is_set():
            # Interrupted via flag (slow path) — release generating flag
            logger.info("[Pipeline] Exited gracefully after interrupt (flag)")
            self._generating = False
            # Schedule processing of any audio buffered during the interrupt
            if self._audio_buffer:
                total = sum(len(a) for a in self._audio_buffer)
                logger.info("[Pipeline] Post-interrupt: %d fragment(s), %d samples queued", len(self._audio_buffer), total)
                self._send_fn = send
                send_fn = self._send_fn
                loop = asyncio.get_running_loop()
                self._accumulation_timer = loop.call_later(
                    0.5,
                    lambda: asyncio.ensure_future(self._flush_accumulated(send_fn))
                )
        else:
            # Normal completion — send final transcript and go idle
            cleaned = clean_response(full_response)
            await send(json.dumps({
                "type": "transcript",
                "role": "assistant",
                "text": cleaned,
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
            detected = detect_sentence_lang(text)
            if detected != lang:
                logger.info("[TTS] Lang override: ASR=%s -> sentence=%s for: %s", lang, detected, text[:50])
            tts_lang = detected
            logger.info("[TTS] Synthesizing (%s): %s...", tts_lang, text[:60])
            pcm_bytes = await self._tts.synthesize(text, lang=tts_lang)
            if not pcm_bytes:
                logger.info("[TTS] Skipped empty audio for: %s", text[:40])
                return
            logger.info("[TTS] Generated %d bytes", len(pcm_bytes))
            if not self._interrupt.is_set():
                # Stream audio in ~100ms chunks for faster time-to-first-audio
                chunk_size = int(self._tts.get_sample_rate() * 2 * 0.1)  # 100ms of int16
                for i in range(0, len(pcm_bytes), chunk_size):
                    if self._interrupt.is_set():
                        break
                    await send(pcm_bytes[i:i + chunk_size])
        except Exception as e:
            logger.error("[TTS] Error: %s", e, exc_info=True)

    # ── Interrupt logic ───────────────────────────────────────────────

    async def _interrupt_generation(self, send: SendFn):
        """User spoke while AI was talking -> stop AI immediately, start listening.

        Cancels the running generation task and TTS worker for instant stop,
        then sets the interrupt flag as a backup for any code that checks it.
        """
        logger.info("[Manager] Interrupted by user! Stopping AI speech.")
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
        """Handle typed text input — skip VAD/ASR, go straight to LLM->TTS."""
        if not self._models_ready:
            return
        if self._generating:
            return  # already processing

        text = text.strip()
        if not text:
            return

        lang = detect_sentence_lang(text)

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

        # Run LLM -> TTS
        self._interrupt.clear()
        self._generating = True
        self.state = "thinking"
        await send(json.dumps({"type": "state", "state": "thinking"}))

        try:
            await asyncio.wait_for(
                self._run_text_pipeline(text, lang, send),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            logger.warning("[Chat] TIMEOUT: text pipeline exceeded 60s, aborting")
            await send(json.dumps({"type": "error", "message": "Response timed out. Please try again."}))
        except asyncio.CancelledError:
            logger.info("[Chat] Cancelled by interrupt")
            if self._tts_task and not self._tts_task.done():
                self._tts_task.cancel()
                try:
                    await self._tts_task
                except (asyncio.CancelledError, Exception):
                    pass
            self._tts_task = None
        except Exception as e:
            logger.error("[Manager] Text chat error: %s", e, exc_info=True)
        finally:
            self._generating = False
            await send(json.dumps({"type": "audio_end"}))
            if self._interrupt.is_set() and self._audio_buffer:
                self.state = "listening"
                await send(json.dumps({"type": "state", "state": "listening"}))
                self._interrupt.clear()
                self._send_fn = send
                send_fn = self._send_fn
                loop = asyncio.get_running_loop()
                self._accumulation_timer = loop.call_later(
                    0.5,
                    lambda: asyncio.ensure_future(self._flush_accumulated(send_fn))
                )
            else:
                self.state = "idle"
                await send(json.dumps({"type": "state", "state": "idle"}))

    async def _run_text_pipeline(self, user_text: str, lang: str, send: SendFn):
        """LLM -> TTS pipeline for typed text input (concurrent TTS)."""
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

        try:
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
                        if sentence_idx >= 8:
                            break

                if sentence_buf.strip() and not self._interrupt.is_set() and sentence_idx < 8:
                    sentence = sentence_buf.strip()
                    sentence_idx += 1
                    await tts_queue.put((sentence, lang, sentence_idx))
            else:
                full_response = f"Echo: {user_text}"
                await tts_queue.put((full_response, lang, 1))
        finally:
            await tts_queue.put(None)
        await tts_task

        if not self._interrupt.is_set():
            cleaned = clean_response(full_response)
            await send(json.dumps({
                "type": "transcript", "role": "assistant",
                "text": cleaned, "language": lang,
            }))
            # NOTE: audio_end is sent by handle_text_chat's finally block

        if self._ltm and full_response.strip() and not self._interrupt.is_set():
            try:
                self._ltm.store_conversation(user_text, full_response, lang)
            except Exception:
                pass

        logger.info("[Chat] LLM+TTS=%.1fs [%s]", time.time() - t0, lang)

    # ── Session reset ─────────────────────────────────────────────────

    async def clear(self, send: SendFn):
        # Summarize conversation into LTM before clearing
        if self._ltm and self._llm and hasattr(self._llm, '_history'):
            try:
                self._ltm.summarize_and_store(self._llm._history)
            except Exception:
                pass

        # Cancel running generation/TTS tasks to prevent orphaned work
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
