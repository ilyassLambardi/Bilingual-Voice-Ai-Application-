"""Pipeline Manager — LOCAL mode. Orchestrates VAD → ASR → LLM → TTS."""
import asyncio
import json
import logging
import time
from typing import Any, Optional
import numpy as np

logger = logging.getLogger(__name__)
from .vad import VADProcessor
from .asr import ASRProcessor
from .llm import LLMProcessor, FallbackLLM, clean_response
from .tts import TTSProcessor
from .lang_detect import detect_sentence_lang
from .memory import LongTermMemory
from .io_handler import SendFn, build_state_message, build_audio_end, build_interrupt
try:
    from ..cloud.tts_edge import EdgeTTSProcessor
except Exception:
    EdgeTTSProcessor = None

def _jmsg(**kw) -> str:
    return json.dumps(kw)

class PipelineManager:
    def __init__(self, config):
        self.config, self.state = config, "idle"
        self._vad: Optional[VADProcessor] = None
        self._asr: Any = None
        self._llm: Any = None
        self._tts: Any = None
        self._ltm: Optional[LongTermMemory] = None
        self._models_ready = False
        self._pipeline_lock = asyncio.Lock()
        self._interrupt = asyncio.Event()
        self._generating = False
        self._gen_task: Optional[asyncio.Task] = None
        self._tts_task: Optional[asyncio.Task] = None
        self._interrupt_speech_frames = 0
        self._interrupt_threshold = 4
        self._backchannel_max_frames = 12
        self._backchannel_cooldown = 0
        self._lang_history: list[str] = []
        self._lang_history_max = 5
        self._audio_buffer: list[np.ndarray] = []
        self._accumulation_timer: Optional[asyncio.TimerHandle] = None
        self._accumulation_delay = 0.6
        self._send_fn: Optional[SendFn] = None
        self._request_times: list[float] = []
        self._rate_limit = 25

    async def load_models(self):
        if self._models_ready: return
        self._load_models_sync()
        if self._tts:
            try: await self._tts.synthesize("Hello.", lang="en")
            except Exception: pass

    def _load_models_sync(self):
        cfg = self.config
        from .paths import MODELS_DIR
        gguf = list(MODELS_DIR.glob("*.gguf")) if MODELS_DIR.exists() else []
        if gguf:
            try:
                self._llm = LLMProcessor(model_path=str(gguf[0]), system_prompt=cfg.llm_system_prompt)
                logger.info("[LLM] GGUF loaded: %s", gguf[0].name)
            except Exception as e:
                logger.warning("[LLM] GGUF failed: %s — trying Qwen fallback", e)
                try:
                    self._llm = FallbackLLM(system_prompt=cfg.llm_system_prompt)
                    logger.info("[LLM] Qwen fallback loaded")
                except Exception as e2:
                    logger.error("[LLM] All LLM backends failed: %s", e2)
                    self._llm = None
        else:
            logger.info("[LLM] No GGUF found, trying Qwen...")
            try:
                self._llm = FallbackLLM(system_prompt=cfg.llm_system_prompt)
                logger.info("[LLM] Qwen loaded")
            except Exception as e:
                logger.error("[LLM] Qwen failed: %s", e)
                self._llm = None
        self._vad = VADProcessor(threshold=cfg.vad_threshold, min_speech_ms=cfg.min_speech_ms,
                                  min_silence_ms=cfg.min_silence_ms, sample_rate=cfg.sample_rate_in)
        self._asr = ASRProcessor(model_size=cfg.asr_model, device=cfg.asr_device,
                                  compute_type=cfg.asr_compute_type, beam_size=cfg.asr_beam_size, language=cfg.asr_language)
        if cfg.tts_engine == "edge" and EdgeTTSProcessor:
            try: self._tts = EdgeTTSProcessor(sample_rate=cfg.tts_sample_rate)
            except Exception: self._tts = TTSProcessor(sample_rate=cfg.tts_sample_rate)
        else: self._tts = TTSProcessor(sample_rate=cfg.tts_sample_rate)
        try: self._ltm = LongTermMemory()
        except Exception: self._ltm = None
        self._models_ready = True
        logger.info("[Manager] All models ready!")

    async def handle_audio_chunk(self, raw_bytes: bytes, send: SendFn):
        if not self._models_ready: return
        try:
            is_speaking, utterance = self._vad.process_chunk(np.frombuffer(raw_bytes, dtype=np.int16))
            if self._generating:
                if self._backchannel_cooldown > 0: self._backchannel_cooldown -= 1
                if is_speaking:
                    self._interrupt_speech_frames += 1
                    if self._interrupt_speech_frames >= self._interrupt_threshold and not self._interrupt.is_set():
                        await self._do_interrupt(send)
                else:
                    if 0 < self._interrupt_speech_frames < self._backchannel_max_frames and self._backchannel_cooldown == 0:
                        try: await send(_jmsg(type="backchannel"))
                        except Exception: pass
                        self._backchannel_cooldown = 30
                    self._interrupt_speech_frames = 0
                if utterance is not None and self._interrupt.is_set():
                    self._audio_buffer.append(utterance)
                return
            if is_speaking and self.state != "listening":
                self.state = "listening"
                await send(build_state_message("listening"))
                if self._accumulation_timer: self._accumulation_timer.cancel(); self._accumulation_timer = None
            if utterance is not None:
                self._audio_buffer.append(utterance)
                if self._accumulation_timer: self._accumulation_timer.cancel()
                self._send_fn = send
                if len(self._audio_buffer) >= 5:
                    asyncio.ensure_future(self._flush(send))
                else:
                    loop = asyncio.get_running_loop()
                    self._accumulation_timer = loop.call_later(
                        self._accumulation_delay, lambda: asyncio.ensure_future(self._flush(send)))
            elif not is_speaking and self.state == "listening" and not self._vad.is_speaking and not self._audio_buffer:
                self.state = "idle"
                await send(build_state_message("idle"))
        except Exception as e:
            logger.error("[Manager] Audio error: %s", e, exc_info=True)

    async def _flush(self, send: SendFn):
        if not self._audio_buffer: return
        if self._pipeline_lock.locked() or self._generating:
            self._schedule_flush(send); return
        combined = np.concatenate(self._audio_buffer)
        self._audio_buffer.clear(); self._accumulation_timer = None
        dur = len(combined) / self.config.sample_rate_in
        if dur < self.config.min_audio_duration:
            logger.info("[VAD] Audio too short (%.2fs) — discarding", dur)
            self.state = "idle"
            await send(_jmsg(type="error", message="Audio too short. Please speak a bit longer."))
            await send(build_state_message("idle")); return
        logger.info("[Pipeline] Processing %.2fs audio (RMS=%.4f)", dur, float(np.sqrt(np.mean(combined.astype(np.float32)**2))))
        self._generating = True
        try: self._gen_task = asyncio.create_task(self._run_pipeline(combined, send))
        except Exception:
            self._generating = False; self.state = "idle"
            await send(build_state_message("idle"))

    async def _run_pipeline(self, audio: np.ndarray, send: SendFn):
        async with self._pipeline_lock:
            self._interrupt.clear()
            try:
                await asyncio.wait_for(self._pipeline_inner(audio, send), timeout=60.0)
            except asyncio.CancelledError:
                await self._cancel_tts(); self._generating = False
                if self._audio_buffer: self._schedule_flush(send)
            except asyncio.TimeoutError:
                self._generating = False; self.state = "idle"
                await send(build_state_message("idle"))
                await send(_jmsg(type="error", message="Response timed out."))
            except Exception as e:
                self.state = "idle"; await send(build_state_message("idle"))
                await send(_jmsg(type="error", message=str(e)))
            finally:
                self._generating = False

    def _check_rate(self) -> bool:
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        if len(self._request_times) >= self._rate_limit: return False
        self._request_times.append(now); return True

    async def _pipeline_inner(self, audio: np.ndarray, send: SendFn):
        if not self._check_rate():
            await send(_jmsg(type="transcript", role="assistant",
                text="Too many requests. Wait a moment.", language="en", time=0))
            self.state = "idle"; await send(build_state_message("idle")); return

        t0 = time.time(); self.state = "thinking"
        await send(build_state_message("thinking"))
        try: asr_result = await self._asr.transcribe(audio)
        except Exception as e:
            logger.error("[ASR] Failed: %s", e, exc_info=True)
            await send(_jmsg(type="error", message=f"ASR failed: {e}"))
            self.state = "idle"; await send(build_state_message("idle")); return

        user_text, lang = asr_result["text"], asr_result["language"] or "en"
        logger.info("[ASR] Result [%s]: '%s'", lang, user_text[:80])
        if not user_text.strip():
            logger.info("[ASR] Empty transcription — skipping")
            self.state = "idle"
            await send(_jmsg(type="error", message="Couldn't understand that. Please try again."))
            await send(build_audio_end()); await send(build_state_message("idle")); return

        await send(_jmsg(type="transcript", role="user", text=user_text,
            language=lang, time=round(time.time()-t0, 3)))

        prev = self._lang_history[-3:]
        self._lang_history.append(lang)
        self._lang_history = self._lang_history[-self._lang_history_max:]
        if prev and all(l != lang for l in prev) and len(prev) >= 2:
            try: await send(json.dumps({"type": "language_shift", "from": prev[-1], "to": lang}))
            except Exception: pass

        mem_ctx = ""
        if self._ltm and self.config.ltm_recall_enabled:
            try:
                mems = self._ltm.recall(user_text, limit=2)
                if mems: mem_ctx = "\n".join(mems)
            except Exception: pass

        self._generating = True
        llm_input = f"{user_text}\n\n(Context:\n{mem_ctx})" if mem_ctx else user_text
        logger.info("[Pipeline] LLM=%s, TTS=%s, lang=%s", type(self._llm).__name__ if self._llm else 'None', type(self._tts).__name__, lang)
        full = await self._stream_and_speak(llm_input, lang, send, fallback=user_text)

        if self._interrupt.is_set():
            self._generating = False
            if self._audio_buffer: self._schedule_flush(send)
        else:
            await send(_jmsg(type="transcript", role="assistant",
                text=clean_response(full), language=lang))
            await send(build_audio_end())
            if self._ltm and full.strip():
                try: self._ltm.store_conversation(user_text, full, lang)
                except Exception: pass
            self._generating = False; self.state = "idle"
            await send(build_state_message("idle"))

    async def _stream_and_speak(self, text: str, lang: str, send: SendFn, fallback: str = "") -> str:
        """Stream LLM → split sentences → TTS. Returns full response text."""
        self.state = "speaking"
        await send(build_state_message("speaking"))
        await send(_jmsg(type="audio_config", sample_rate=self._tts.get_sample_rate()))
        full, buf, idx = "", "", 0
        q: asyncio.Queue = asyncio.Queue()

        async def _tts_worker():
            while True:
                item = await q.get()
                if item is None: break
                if self._interrupt.is_set(): continue
                t, l, i = item
                await send(_jmsg(type="partial_transcript", role="assistant", text=t, language=l, index=i))
                await self._speak(t, l, send)

        task = asyncio.create_task(_tts_worker()); self._tts_task = task
        try:
            if self._llm:
                t_llm = time.time(); tok_count = 0
                logger.info("[LLM] Starting generation...")
                async for tok in self._llm.stream(text, max_tokens=self.config.llm_max_tokens,
                                                   temperature=self.config.llm_temperature, lang=lang):
                    if self._interrupt.is_set(): break
                    buf += tok; full += tok; tok_count += 1
                    if tok_count == 1:
                        logger.info("[LLM] First token in %.1fs", time.time() - t_llm)
                    if any(buf.rstrip().endswith(p) for p in ".?!") or (len(buf) > 50 and any(buf.rstrip().endswith(p) for p in ",;:")):
                        idx += 1; await q.put((buf.strip(), lang, idx)); buf = ""
                        logger.info("[TTS] Sentence %d queued: '%s'", idx, full[-60:])
                        if idx >= 8: break
                if buf.strip() and not self._interrupt.is_set() and idx < 8:
                    idx += 1; await q.put((buf.strip(), lang, idx))
                logger.info("[LLM] Done: %d tokens in %.1fs (%.1f tok/s)", tok_count, time.time()-t_llm,
                            tok_count / max(time.time()-t_llm, 0.01))
            else:
                full = fallback or text; await q.put((full, lang, 1))
        finally:
            await q.put(None)
        await task; self._tts_task = None
        return full

    async def _speak(self, text: str, lang: str, send: SendFn):
        if self._interrupt.is_set() or not text: return
        try:
            pcm = await self._tts.synthesize(text, lang=detect_sentence_lang(text))
            if not pcm or self._interrupt.is_set(): return
            cs = int(self._tts.get_sample_rate() * 2 * 0.1)
            for i in range(0, len(pcm), cs):
                if self._interrupt.is_set(): break
                await send(pcm[i:i+cs])
        except Exception as e:
            logger.error("[TTS] %s", e)

    async def _do_interrupt(self, send: SendFn):
        self._interrupt.set(); self._interrupt_speech_frames = 0
        if self._accumulation_timer: self._accumulation_timer.cancel(); self._accumulation_timer = None
        if self._tts_task and not self._tts_task.done(): self._tts_task.cancel()
        if self._gen_task and not self._gen_task.done(): self._gen_task.cancel()
        self.state = "listening"
        try: await send(build_audio_end()); await send(build_state_message("listening")); await send(build_interrupt())
        except Exception: pass

    async def _cancel_tts(self):
        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
            try: await self._tts_task
            except Exception: pass
        self._tts_task = None

    def _schedule_flush(self, send: SendFn):
        self._send_fn = send
        asyncio.get_running_loop().call_later(0.5, lambda: asyncio.ensure_future(self._flush(send)))

    async def handle_text_chat(self, text: str, send: SendFn):
        if not self._models_ready or self._generating: return
        text = text.strip()
        if not text: return
        lang = detect_sentence_lang(text)
        if not self._check_rate():
            await send(_jmsg(type="transcript", role="assistant",
                text="Too many requests.", language="en")); return
        await send(_jmsg(type="transcript", role="user", text=text, language=lang, time=0))
        self._interrupt.clear(); self._generating = True; self.state = "thinking"
        await send(build_state_message("thinking"))
        try:
            full = await asyncio.wait_for(
                self._stream_and_speak(text, lang, send, fallback=f"Echo: {text}"), timeout=60.0)
            if not self._interrupt.is_set():
                await send(_jmsg(type="transcript", role="assistant",
                    text=clean_response(full), language=lang))
            if self._ltm and full.strip() and not self._interrupt.is_set():
                try: self._ltm.store_conversation(text, full, lang)
                except Exception: pass
        except (asyncio.TimeoutError, asyncio.CancelledError):
            await self._cancel_tts()
        except Exception as e:
            logger.error("[Chat] %s", e)
        finally:
            self._generating = False
            await send(build_audio_end())
            if self._interrupt.is_set() and self._audio_buffer:
                self.state = "listening"
                await send(build_state_message("listening"))
                self._interrupt.clear()
                self._schedule_flush(send)
            else:
                self.state = "idle"; await send(build_state_message("idle"))

    async def clear(self, send: SendFn):
        if self._ltm and self._llm and hasattr(self._llm, '_history'):
            try: self._ltm.summarize_and_store(self._llm._history)
            except Exception: pass
        self._interrupt.set()
        await self._cancel_tts()
        if self._gen_task and not self._gen_task.done():
            self._gen_task.cancel()
            try: await self._gen_task
            except Exception: pass
        self._gen_task = None
        if self._llm: self._llm.clear_history()
        if self._vad: self._vad.reset()
        if self._accumulation_timer: self._accumulation_timer.cancel(); self._accumulation_timer = None
        self._audio_buffer.clear(); self._lang_history.clear()
        self._generating = False; self._interrupt.clear()
        self._interrupt_speech_frames = self._backchannel_cooldown = 0
        self.state = "idle"
        await send(build_state_message("idle"))
