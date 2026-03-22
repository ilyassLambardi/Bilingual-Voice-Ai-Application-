"""
Module 2 (Processing/Core): LLM — llama-cpp-python (primary) or HuggingFace transformers (fallback).

Primary: GGUF model via llama-cpp-python with streaming tokens.
Fallback: DialoGPT-medium via transformers (auto-loaded when no GGUF).

Both expose the same async `stream()` interface.
"""

import asyncio
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncIterator, Optional

_pool = ThreadPoolExecutor(max_workers=1)


# ═══════════════════════════════════════════════════════════════════════════
# Primary: llama-cpp-python (GGUF)
# ═══════════════════════════════════════════════════════════════════════════

class LLMProcessor:
    """Streaming chat completion using a GGUF model via llama-cpp-python."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        system_prompt: str = "",
    ):
        print(f"[LLM] Loading {model_path} ...")
        from llama_cpp import Llama

        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.system_prompt = system_prompt
        self._history: list[dict] = []
        print("[LLM] Ready.")

    async def stream(
        self,
        user_text: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        lang: str = "en",
    ) -> AsyncIterator[str]:
        """Yield tokens as they are generated (async)."""
        messages = self._build_messages(user_text, lang)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        def _generate():
            resp = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.3,
                stream=True,
            )
            full = ""
            for chunk in resp:
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content")
                if token:
                    full += token
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            self._history.append({"role": "user", "content": user_text})
            self._history.append({"role": "assistant", "content": full})
            if len(self._history) > 8:
                self._history = self._history[-8:]

        def _generate_wrapper():
            try:
                _generate()
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        _pool.submit(_generate_wrapper)

        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

    def _build_messages(self, user_text: str, lang: str = "en") -> list[dict]:
        context_hint = (
            f"[ASR detected: {lang} — IGNORE this tag. Analyze the user's actual intent.]\n"
            "Apply MIRRORING: match the user's language. If they mix languages, "
            "respond in the dominant one. If they ask about a word's meaning, "
            "activate TEACHER MODE.\n"
        )

        # Detect teacher mode intent
        lower = user_text.lower()
        teacher_triggers = [
            "what does", "what is", "was bedeutet", "was heißt",
            "was ist", "what do you mean by", "explain the word",
            "meaning of", "bedeutung von",
        ]
        is_teacher = any(t in lower for t in teacher_triggers)
        if is_teacher:
            context_hint += (
                "[TEACHER MODE ACTIVE] The user is asking about a word. "
                "Structure your response as: "
                "1) Explain nuance/feeling in the QUESTION language. "
                "2) Give 2-3 natural example sentences in the OTHER language. "
                "3) Brief cultural context in the question language. "
                "Keep it conversational, not like a textbook."
            )

        sys_msg = f"{self.system_prompt}\n\n{context_hint}"
        msgs = [{"role": "system", "content": sys_msg}]
        msgs.extend(self._history[-20:])
        msgs.append({"role": "user", "content": user_text})
        return msgs

    @staticmethod
    def format_educational_response(word: str, meaning_en: str, examples_de: list[str]) -> str:
        """Format a Teacher Mode response for word explanation."""
        response = (
            f"The word '{word}' is fascinating. In English, it roughly means: {meaning_en}. "
            f"Here is how you'd actually use it in Germany:\n"
        )
        for i, ex in enumerate(examples_de, 1):
            response += f"{i}. {ex}\n"
        response += "Does that help you feel the nuance of the word?"
        return response

    def clear_history(self):
        self._history.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Fallback: HuggingFace transformers (DialoGPT / text-generation)
# ═══════════════════════════════════════════════════════════════════════════

_FEW_SHOT = [
    {"role": "user", "content": "Hey, how's it going?"},
    {"role": "assistant", "content": "Pretty good, yeah! What've you been up to today?"},
    {"role": "user", "content": "Hallo, wie geht's dir?"},
    {"role": "assistant", "content": "Hey, mir geht's gut! Was machst du gerade so?"},
    {"role": "user", "content": "I had a really long day at work."},
    {"role": "assistant", "content": "Oh man, those days are rough. What happened?"},
    {"role": "user", "content": "Ich bin heute total müde."},
    {"role": "assistant", "content": "Ja das kenn ich. Hast du schlecht geschlafen?"},
]

# No phrase filtering — unrestricted output


class FallbackLLM:
    """Conversational LLM using Qwen2.5-0.5B-Instruct with few-shot steering.

    Uses injected conversation examples to teach the model human-like style.
    Post-processes output to strip robotic AI phrases.
    """

    def __init__(self, system_prompt: str = ""):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Try local models/ folder first, fall back to HuggingFace
        _local_path = Path(__file__).resolve().parent.parent.parent / "models" / "Qwen2.5-1.5B-Instruct"
        if _local_path.exists():
            self._model_name = str(_local_path)
            print(f"[LLM] Loading local model: {_local_path.name} ...")
        else:
            self._model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            print(f"[LLM] Loading from HuggingFace: {self._model_name} ...")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, trust_remote_code=True
        )

        # Load model — try int8 first, fall back to FP16
        if device == "cuda":
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_name,
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                    trust_remote_code=True,
                )
                print("[LLM] Loaded with int8 quantization")
            except Exception:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                ).to(device)
                print("[LLM] Loaded with FP16")
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )

        # torch.compile for kernel fusion (PyTorch 2.0+)
        try:
            self._model = torch.compile(self._model, mode="reduce-overhead")
            print("[LLM] torch.compile enabled")
        except Exception:
            pass

        self._device = device
        self.system_prompt = system_prompt
        self._history: list[dict] = []
        print(f"[LLM] {self._model_name} ready on {self._device}.")

    async def stream(
        self,
        user_text: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        lang: str = "en",
    ) -> AsyncIterator[str]:
        """Yield tokens as they are generated (non-blocking async streaming)."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        model_ref = self._model
        tokenizer_ref = self._tokenizer
        device_ref = self._device

        # Build messages on the main thread (thread-safe access to history)
        messages = self._build_messages(user_text, lang)

        def _generate():
            try:
                import torch
                from transformers import TextIteratorStreamer

                text = tokenizer_ref.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer_ref(text, return_tensors="pt").to(device_ref)

                streamer = TextIteratorStreamer(
                    tokenizer_ref, skip_prompt=True, skip_special_tokens=True
                )

                gen_kwargs = {
                    **inputs,
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "do_sample": True,
                    "top_p": 0.85,
                    "top_k": 20,
                    "repetition_penalty": 1.2,
                    "streamer": streamer,
                }

                # Start model.generate in a sub-thread (feeds streamer)
                def _run_generate():
                    with torch.no_grad():
                        model_ref.generate(**gen_kwargs)

                gen_thread = threading.Thread(target=_run_generate)
                gen_thread.start()

                # Read tokens from streamer and push to async queue
                for token_text in streamer:
                    loop.call_soon_threadsafe(queue.put_nowait, token_text)
                gen_thread.join()
            except Exception as e:
                print(f"[LLM] Generation error: {e}")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # always send sentinel

        # Run the blocking generator in thread pool
        _pool.submit(_generate)

        full_response = ""
        while True:
            token = await queue.get()
            if token is None:
                break
            full_response += token
            yield token

        # Post-process and store in history
        clean = self._clean_response(full_response)
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": clean})
        if len(self._history) > 24:
            self._history = self._history[-24:]

    def _build_messages(self, user_text: str, lang: str = "en") -> list[dict]:
        lang_rule = (
            f"ASR detected language hint: {lang}. BUT this may be wrong if the user "
            "mentioned a foreign word inside a sentence in another language.\n"
            "CRITICAL: Respond in the language the user is ACTUALLY speaking in. "
            "If they ask in English about a German word, answer in English. "
            "If they speak a full German sentence, reply in German. "
            "Judge by the overall sentence, not individual words."
        )

        sys_msg = f"{self.system_prompt}\n{lang_rule}"
        messages = [{"role": "system", "content": sys_msg}]
        messages.extend(_FEW_SHOT)
        messages.extend(self._history[-10:])
        messages.append({"role": "user", "content": user_text})
        return messages

    @staticmethod
    def _clean_response(text: str) -> str:
        """Minimal cleanup -- preserve full response."""
        if not text:
            return "Hmm, I didn't catch that."

        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        text = re.sub(r'^[-\*]\s*', '', text).strip()

        # Ensure it ends with punctuation
        # Use sentence-boundary regex to avoid cutting at decimals/abbreviations
        if text and text[-1] not in '.!?':
            match = list(re.finditer(r'[.!?](?:\s|$)', text))
            if match and match[-1].start() > 10:
                text = text[:match[-1].start() + 1]
            else:
                text = text.rstrip(',;: ') + '.'

        return text

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences for streaming."""
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p.strip() for p in parts if p.strip()]

    def clear_history(self):
        self._history.clear()
