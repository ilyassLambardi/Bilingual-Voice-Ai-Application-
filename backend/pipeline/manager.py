"""
Pipeline Manager — async orchestrator for the S2S loop.

Three-stage async flow:
    1. Mic → VAD → Whisper
    2. Whisper → LLM  (streaming tokens)
    3. LLM sentence → TTS → WebSocket binary

Supports interruptibility: if the user speaks while the AI is
talking, we kill TTS output and flush the queues immediately.
"""

import asyncio
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np


def _safe_print(msg: str):
    """Print that never crashes on Windows cp1252 encoding."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode(), flush=True)

from .vad import VADProcessor
from .asr import ASRProcessor
from .llm import LLMProcessor, FallbackLLM
from .tts import TTSProcessor, _detect_sentence_lang
from .memory import LongTermMemory


# Type alias for the WebSocket send function
SendFn = Callable[[str | bytes], asyncio.coroutines]

_load_pool = ThreadPoolExecutor(max_workers=1)


class PipelineManager:
    """Coordinates VAD → ASR → LLM → TTS with full interruptibility."""

    def __init__(self, config):
        self.config = config
        self.state = "idle"

        # ── Models ────────────────────────────────────────────────────
        self._vad: Optional[VADProcessor] = None
        self._asr: Optional[ASRProcessor] = None
        self._llm: Optional[LLMProcessor] = None
        self._tts: Optional[TTSProcessor] = None
        self._ltm: Optional[LongTermMemory] = None
        self._models_ready = False

        # ── Interrupt mechanics ───────────────────────────────────────
        self._interrupt = asyncio.Event()
        self._generating = False          # True while LLM+TTS is running
        self._gen_task: Optional[asyncio.Task] = None
        self._interrupt_speech_frames = 0  # consecutive speaking frames during generation
        self._interrupt_threshold = 12    # need ~384ms sustained speech to interrupt (was 8)

        # ── Backchanneling (smart: ignore "mhm", "yeah", short affirmations) ──
        self._backchannel_max_frames = 25   # ~800ms — shorter = affirmation, not interrupt
        self._backchannel_cooldown = 0      # cooldown frames after backchannel detection

        # ── Language shift detection ─────────────────────────────────
        self._lang_history: list[str] = []  # last N detected languages
        self._lang_history_max = 5          # track last 5 messages

        # ── Ghost texting (partial ASR while speaking) ─────────────────
        self._ghost_counter = 0           # audio chunks since last ghost
        self._ghost_interval = 25         # run ghost ASR every ~25 chunks (~800ms)

        # ── Audio accumulation (merge split speech into one big utterance) ──
        self._audio_buffer: list[np.ndarray] = []   # accumulated audio fragments
        self._accumulation_timer: Optional[asyncio.TimerHandle] = None
        self._accumulation_delay = 1.8     # seconds of silence before processing
        self._send_fn: Optional[SendFn] = None  # cached for timer callback

        # ── Rate limiting (protect Groq free tier: 30 RPM) ──────────────
        self._request_times: list[float] = []  # timestamps of recent API calls
        self._rate_limit = 25                  # leave headroom below 30 RPM

    # ── Async model loading (runs in thread pool — non-blocking) ──────

    async def load_models(self):
        """Load all models at startup.  Blocks briefly but only runs once
        before any connections are accepted."""
        if self._models_ready:
            return
        self._load_models_sync()

    def _load_models_sync(self):
        """Synchronous model loading — cloud or local based on config.mode."""
        import sys
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
        if cfg.tts_engine == "xtts":
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
            chunk = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            is_speaking, utterance = self._vad.process_chunk(
                np.frombuffer(raw_bytes, dtype=np.int16)
            )

            # ── Interrupt with backchanneling ──────────────────────────
            if self._generating:
                if self._backchannel_cooldown > 0:
                    self._backchannel_cooldown -= 1

                if is_speaking:
                    self._interrupt_speech_frames += 1
                    # Only interrupt after sustained speech (not a short "mhm" / "yeah")
                    if self._interrupt_speech_frames >= self._interrupt_threshold:
                        await self._interrupt_generation(send)
                else:
                    # Speech just ended — was it short enough to be a backchannel?
                    if 0 < self._interrupt_speech_frames < self._backchannel_max_frames:
                        if self._backchannel_cooldown == 0:
                            print(f"[VAD] Backchannel detected ({self._interrupt_speech_frames} frames) — not interrupting")
                            try:
                                await send(json.dumps({"type": "backchannel"}))
                            except Exception:
                                pass
                            self._backchannel_cooldown = 30  # ~1s cooldown
                    self._interrupt_speech_frames = 0
                # CRITICAL: while generating, do NOT change state or accumulate
                # utterances — only handle interrupt/backchannel above.
                # This prevents _flush_accumulated from running concurrently
                # with the active pipeline, which causes message corruption.
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
                self._send_fn = send
                loop = asyncio.get_running_loop()
                self._accumulation_timer = loop.call_later(
                    self._accumulation_delay,
                    lambda: asyncio.ensure_future(self._flush_accumulated(send))
                )
        except Exception as e:
            print(f"[Manager] handle_audio_chunk error: {e}")
            traceback.print_exc()

    # ── Audio accumulation flush ─────────────────────────────────────

    async def _flush_accumulated(self, send: SendFn):
        """Called after silence timeout — process all buffered audio as one utterance."""
        if not self._audio_buffer:
            return
        # Safety: never start a second pipeline while one is already running
        if self._generating:
            print("[VAD] Flush skipped — pipeline already generating")
            self._audio_buffer.clear()
            self._accumulation_timer = None
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
            print(f"[VAD] Audio too short ({duration:.2f}s < {min_dur}s) — discarding")
            self.state = "idle"
            await send(json.dumps({"type": "state", "state": "idle"}))
            return

        print(f"[VAD] Processing {n_fragments} fragment(s), {duration:.1f}s of audio")
        self._gen_task = asyncio.create_task(
            self._run_pipeline(combined, send)
        )

    # ── Full pipeline ─────────────────────────────────────────────────

    async def _run_pipeline(self, audio: np.ndarray, send: SendFn):
        """ASR → LLM (streaming) → TTS (sentence-by-sentence)."""
        self._interrupt.clear()
        try:
            await self._run_pipeline_inner(audio, send)
        except Exception as e:
            print(f"[Pipeline] ERROR: {e}")
            traceback.print_exc()
            self._generating = False
            self.state = "idle"
            await send(json.dumps({"type": "state", "state": "idle"}))
            await send(json.dumps({"type": "error", "message": str(e)}))

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
            print("[Manager] Rate limit hit — skipping request")
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

        asr_result = await self._asr.transcribe(audio)
        user_text = asr_result["text"]
        lang = asr_result["language"] or "en"

        # Filter empty or hallucinated transcriptions
        _HALLUCINATIONS = {
            "", "you", "thank you", "thanks", "bye", "goodbye",
            "the end", "thanks for watching", "thank you for watching",
            "subtitles by", "amara.org", "...", ".", "huh",
            "danke", "tschüss", "untertitel", "untertitelung",
        }
        cleaned = user_text.strip().rstrip(".!?,").lower()
        if not cleaned or cleaned in _HALLUCINATIONS or len(cleaned) < 2:
            print(f"[ASR] Rejected hallucination: '{user_text.strip()}'")
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
            print(f"[Lang] Language shift detected: {prev_langs[-1]} → {lang}")
            try:
                await send(json.dumps({
                    "type": "language_shift",
                    "from": prev_langs[-1],
                    "to": lang,
                }))
            except Exception:
                pass

        # LTM recall disabled — small model gets confused by injected memories
        memory_context = ""

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

                # Flush on sentence boundary — shorter chunks = more natural TTS
                should_flush = False
                if any(sentence_buf.rstrip().endswith(p) for p in ".?!"):
                    should_flush = True
                elif len(sentence_buf) > 80 and any(sentence_buf.rstrip().endswith(p) for p in ",;:"):
                    should_flush = True  # break long clauses for natural pacing

                if should_flush:
                    sentence = sentence_buf.strip()
                    sentence_buf = ""
                    sentence_idx += 1
                    await send(json.dumps({
                        "type": "partial_transcript",
                        "role": "assistant",
                        "text": sentence,
                        "language": lang,
                        "index": sentence_idx,
                    }))
                    await self._speak_sentence(sentence, lang, send)

                    # Stop after 3 sentences — enough for substantive answers
                    if sentence_idx >= 3:
                        break

            # Flush remaining text (only if we haven't hit 3 sentences yet)
            if sentence_buf.strip() and not self._interrupt.is_set() and sentence_idx < 3:
                sentence = sentence_buf.strip()
                sentence_idx += 1
                await send(json.dumps({
                    "type": "partial_transcript",
                    "role": "assistant",
                    "text": sentence,
                    "language": lang,
                    "index": sentence_idx,
                }))
                await self._speak_sentence(sentence, lang, send)
        else:
            # Echo mode (no LLM available) — repeat back in detected language
            full_response = user_text
            await send(json.dumps({
                "type": "partial_transcript",
                "role": "assistant",
                "text": full_response,
                "language": lang,
                "index": 1,
            }))
            await self._speak_sentence(full_response, lang, send)

        # ── Done ──────────────────────────────────────────────────────
        # CRITICAL: keep _generating = True until ALL final messages are sent.
        # Setting it False earlier opens a window where a second pipeline
        # can start from accumulated noise, causing duplicate/disappearing messages.
        t_total = time.time() - t0
        t_llm_tts = time.time() - t_llm_start
        print(f"[Perf] ASR={t_asr:.1f}s  LLM+TTS={t_llm_tts:.1f}s  TOTAL={t_total:.1f}s")

        if not self._interrupt.is_set():
            await send(json.dumps({
                "type": "transcript",
                "role": "assistant",
                "text": full_response,
                "language": lang,
            }))
            await send(json.dumps({"type": "audio_end"}))

        # ── LTM: store conversation exchange ──────────────────────
        if self._ltm and full_response.strip() and not self._interrupt.is_set():
            try:
                self._ltm.store_conversation(user_text, full_response, lang)
            except Exception:
                pass

        self.state = "idle"
        await send(json.dumps({"type": "state", "state": "idle"}))
        self._generating = False  # NOW safe — all messages sent, state is idle

    # ── TTS helper ────────────────────────────────────────────────────

    async def _speak_sentence(self, text: str, lang: str, send: SendFn):
        """Synthesise one sentence and stream the PCM bytes."""
        if self._interrupt.is_set() or not text:
            return
        try:
            _safe_print(f"[TTS] Synthesizing ({lang}): {text[:60]}...")
            pcm_bytes = await self._tts.synthesize(text, lang=lang)
            if not pcm_bytes:
                _safe_print(f"[TTS] Skipped empty audio for: {text[:40]}")
                return
            print(f"[TTS] Generated {len(pcm_bytes)} bytes")
            if not self._interrupt.is_set():
                await send(pcm_bytes)   # binary frame
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
        """User spoke while AI was talking → kill everything."""
        print("[Manager] Interrupted by user!")
        self._interrupt.set()
        self._generating = False
        self._interrupt_speech_frames = 0

        if self._gen_task and not self._gen_task.done():
            self._gen_task.cancel()
            try:
                await self._gen_task
            except (asyncio.CancelledError, Exception):
                pass

        # Clear accumulated audio and cancel pending timer
        self._audio_buffer.clear()
        if self._accumulation_timer is not None:
            self._accumulation_timer.cancel()
            self._accumulation_timer = None

        self._vad.reset()
        self.state = "listening"
        await send(json.dumps({"type": "state", "state": "listening"}))
        await send(json.dumps({"type": "interrupt"}))

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
            await self._run_text_pipeline(text, lang, send)
        except Exception as e:
            print(f"[Manager] Text chat error: {e}")
            traceback.print_exc()
        finally:
            self._generating = False
            self.state = "idle"
            await send(json.dumps({"type": "state", "state": "idle"}))

    async def _run_text_pipeline(self, user_text: str, lang: str, send: SendFn):
        """LLM → TTS pipeline for typed text input."""
        import time as _time
        t0 = _time.time()

        self.state = "speaking"
        await send(json.dumps({"type": "state", "state": "speaking"}))
        await send(json.dumps({
            "type": "audio_config",
            "sample_rate": self._tts.get_sample_rate(),
        }))

        full_response = ""
        sentence_buf = ""
        sentence_idx = 0

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
                elif len(sentence_buf) > 80 and any(sentence_buf.rstrip().endswith(p) for p in ",;:"):
                    should_flush = True

                if should_flush:
                    sentence = sentence_buf.strip()
                    sentence_buf = ""
                    sentence_idx += 1
                    await send(json.dumps({
                        "type": "partial_transcript", "role": "assistant",
                        "text": sentence, "language": lang, "index": sentence_idx,
                    }))
                    await self._speak_sentence(sentence, lang, send)
                    if sentence_idx >= 3:
                        break

            if sentence_buf.strip() and not self._interrupt.is_set() and sentence_idx < 3:
                sentence = sentence_buf.strip()
                sentence_idx += 1
                await send(json.dumps({
                    "type": "partial_transcript", "role": "assistant",
                    "text": sentence, "language": lang, "index": sentence_idx,
                }))
                await self._speak_sentence(sentence, lang, send)
        else:
            full_response = f"Echo: {user_text}"
            await send(json.dumps({
                "type": "partial_transcript", "role": "assistant",
                "text": full_response, "language": lang, "index": 1,
            }))
            await self._speak_sentence(full_response, lang, send)

        if not self._interrupt.is_set():
            await send(json.dumps({
                "type": "transcript", "role": "assistant",
                "text": full_response, "language": lang,
            }))
            await send(json.dumps({"type": "audio_end"}))

        if self._ltm and full_response.strip() and not self._interrupt.is_set():
            try:
                self._ltm.store_conversation(user_text, full_response, lang)
            except Exception:
                pass

        t_total = _time.time() - t0
        print(f"[Chat] LLM+TTS={t_total:.1f}s [{lang}]")

    @staticmethod
    def _detect_text_language(text: str) -> str:
        """Simple heuristic to detect if typed text is German or English."""
        _DE_MARKERS = {
            "ä", "ö", "ü", "ß", "ich", "bin", "ist", "und", "der", "die",
            "das", "ein", "eine", "nicht", "wie", "was", "hallo", "danke",
            "bitte", "gut", "mir", "dir", "mich", "dich", "kannst", "hast",
            "wir", "sie", "haben", "werden", "können", "möchte", "geht",
        }
        words = set(text.lower().split())
        chars = set(text.lower())
        de_score = len(words & _DE_MARKERS) + len(chars & {"ä", "ö", "ü", "ß"})
        return "de" if de_score >= 2 else "en"

    # ── Session reset ─────────────────────────────────────────────────

    async def clear(self, send: SendFn):
        # Summarize conversation into LTM before clearing
        if self._ltm and self._llm and hasattr(self._llm, '_history'):
            try:
                self._ltm.summarize_and_store(self._llm._history)
            except Exception:
                pass
        if self._llm:
            self._llm.clear_history()
        if self._vad:
            self._vad.reset()
        self._generating = False
        self._interrupt.clear()
        self.state = "idle"
        await send(json.dumps({"type": "state", "state": "idle"}))
