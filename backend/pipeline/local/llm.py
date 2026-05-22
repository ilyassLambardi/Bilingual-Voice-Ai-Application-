"""LLM — llama-cpp-python (primary) or HuggingFace transformers (fallback)."""
import asyncio
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Optional
from .paths import MODELS_DIR

def clean_response(text: str) -> str:
    if not text or not text.strip():
        return "Hmm, I didn't catch that."
    text = text.strip().strip('"')
    text = re.sub(r'^[-\*]\s*', '', text).strip()
    if text and text[-1] not in '.!?':
        m = list(re.finditer(r'[.!?](?:\s|$)', text))
        text = text[:m[-1].start()+1] if m and m[-1].start() > 10 else text.rstrip(',;: ')+'.' 
    return text

def build_context_hint(lang: str, user_text: str) -> str:
    hint = ("[SPRACHE: DEUTSCH. Antworte NUR auf Deutsch. KEIN Englisch.]\n" if lang == "de"
            else "[Respond in English. No German unless explicitly asked.]\n")
    if any(t in user_text.lower() for t in ("what does","was bedeutet","was hei\u00dft","meaning of","bedeutung von")):
        hint += ("[TEACHER MODE] Explain the word naturally, give 2-3 examples in the other language, "
                 "brief cultural context. Keep it conversational.")
    return hint

logger = logging.getLogger(__name__)
_pool = ThreadPoolExecutor(max_workers=1)

class LLMProcessor:
    def __init__(self, model_path: str, n_ctx=2048, n_gpu_layers=-1, system_prompt=""):
        from llama_cpp import Llama
        self.model = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
        self.system_prompt, self._history = system_prompt, []
        logger.info("[LLM] GGUF ready")

    async def stream(self, user_text: str, max_tokens=256, temperature=0.7, lang="en") -> AsyncIterator[str]:
        msgs = self._build_messages(user_text, lang)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Optional[str]] = asyncio.Queue()
        def _gen():
            try:
                full = ""
                for chunk in self.model.create_chat_completion(
                    messages=msgs, max_tokens=max_tokens, temperature=temperature,
                    top_p=0.9, frequency_penalty=0.3, presence_penalty=0.3, stream=True):
                    tok = chunk["choices"][0].get("delta", {}).get("content")
                    if tok: full += tok; loop.call_soon_threadsafe(q.put_nowait, tok)
                self._history += [{"role":"user","content":user_text},{"role":"assistant","content":full}]
                self._history = self._history[-12:]
            except Exception as e:
                logger.error("[LLM] %s", e)
                loop.call_soon_threadsafe(q.put_nowait, "Sorry, I'm having trouble responding.")
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)
        _pool.submit(_gen)
        while (tok := await q.get()) is not None: yield tok

    def _build_messages(self, text: str, lang="en") -> list[dict]:
        msgs = [{"role":"system","content":self.system_prompt}]
        msgs += self._history[-20:]
        msgs.append({"role":"system","content":build_context_hint(lang, text)})
        msgs.append({"role":"user","content":text})
        return msgs

    def clear_history(self): self._history.clear()


class FallbackLLM:
    def __init__(self, system_prompt=""):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        local = MODELS_DIR / "Qwen2.5-1.5B-Instruct"
        self._model_name = str(local) if local.exists() else "Qwen/Qwen2.5-1.5B-Instruct"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
        if device == "cuda":
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_name, torch_dtype=torch.float16, trust_remote_code=True).to(device)
                logger.info("[LLM] Loaded float16 on CUDA")
            except Exception as e:
                logger.warning("[LLM] CUDA float16 failed (%s), falling back to CPU", e)
                device = "cpu"
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_name, torch_dtype=torch.float32, trust_remote_code=True)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name, torch_dtype=torch.float32, trust_remote_code=True)
        self._device, self.system_prompt, self._history = device, system_prompt, []
        logger.info("[LLM] %s ready on %s", self._model_name, device)

    async def stream(self, user_text: str, max_tokens=256, temperature=0.7, lang="en") -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Optional[str]] = asyncio.Queue()
        msgs = self._build_messages(user_text, lang)
        model, tok_ref, dev = self._model, self._tokenizer, self._device
        history_ref = self  # capture self for history update inside thread
        def _gen():
            full_response = ""
            try:
                import torch
                from transformers import TextIteratorStreamer
                text = tok_ref.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = tok_ref(text, return_tensors="pt").to(dev)
                streamer = TextIteratorStreamer(tok_ref, skip_prompt=True, skip_special_tokens=True)
                def _run():
                    with torch.no_grad():
                        model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature,
                                       do_sample=True, top_p=0.85, top_k=15, repetition_penalty=1.2, streamer=streamer)
                t = threading.Thread(target=_run); t.start()
                for tok in streamer:
                    full_response += tok
                    loop.call_soon_threadsafe(q.put_nowait, tok)
                t.join()
            except Exception as e: logger.error("[LLM] %s", e)
            finally:
                if full_response.strip():
                    history_ref._history += [{"role":"user","content":user_text},
                                             {"role":"assistant","content":clean_response(full_response)}]
                    history_ref._history = history_ref._history[-12:]
                loop.call_soon_threadsafe(q.put_nowait, None)
        _pool.submit(_gen)
        while (tok := await q.get()) is not None: yield tok

    def _build_messages(self, text: str, lang="en") -> list[dict]:
        msgs = [{"role":"system","content":self.system_prompt}]
        if lang == "de":
            msgs += [{"role":"user","content":"Hallo, wie geht's dir?"},
                     {"role":"assistant","content":"Hey, mir geht's gut! Was machst du so?"}]
        else:
            msgs += [{"role":"user","content":"Hey, how's it going?"},
                     {"role":"assistant","content":"Pretty good! What've you been up to?"}]
        msgs += self._history[-8:]
        # Language enforcement AFTER history — prevents prior German from bleeding over
        msgs.append({"role":"system","content":build_context_hint(lang, text)})
        msgs.append({"role":"user","content":text})
        return msgs

    def clear_history(self): self._history.clear()
