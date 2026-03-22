"""
Module 2 (Processing/Core): LLM provider — Groq Cloud API (Llama-3.3-70B-Versatile).

Streaming chat completions via Groq's ultra-fast inference.
Free tier: 30 RPM, ~14k tokens/day — enough for demos + testing.
Paid tier: very cheap per-token pricing for production.

Same async `stream()` interface as the local LLM providers.
"""

import asyncio
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Optional

log = logging.getLogger("s2s.llm")

_pool = ThreadPoolExecutor(max_workers=2)

# No phrase filtering — unrestricted output


class GroqLLM:
    """Streaming LLM via Groq API — fast enough for real-time voice."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        system_prompt: str = "",
    ):
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self._api_key or self._api_key.startswith("gsk_your"):
            raise ValueError(
                "GROQ_API_KEY not set. Get one free at https://console.groq.com/keys"
            )

        from groq import Groq
        self._client = Groq(api_key=self._api_key)
        self._model = model
        self.system_prompt = system_prompt
        self._history: list[dict] = []

        # Validate connection
        print(f"[LLM] Groq API -> {model}")
        print(f"[LLM] Ready.")

    async def stream(
        self,
        user_text: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
        lang: str = "en",
    ) -> AsyncIterator[str]:
        """Yield tokens as they stream from Groq API."""
        messages = self._build_messages(user_text, lang)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        def _generate():
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    stream = self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=0.9,
                        stream=True,
                    )
                    full = ""
                    for chunk in stream:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            token = delta.content
                            full += token
                            loop.call_soon_threadsafe(queue.put_nowait, token)

                    # Store in history
                    clean = self._clean_response(full)
                    self._history.append({"role": "user", "content": user_text})
                    self._history.append({"role": "assistant", "content": clean})
                    if len(self._history) > 30:
                        self._history = self._history[-30:]
                    return  # success

                except Exception as e:
                    is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                    if attempt < max_retries and is_rate_limit:
                        wait = 1.0 * (2 ** attempt)
                        log.warning(f"[LLM] Groq rate-limited, retry {attempt+1} in {wait:.0f}s...")
                        time.sleep(wait)
                    else:
                        log.error(f"[LLM] Groq API error: {e}")
                        return
            # all retries failed — sentinel sent in finally

        def _generate_wrapper():
            try:
                _generate()
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        # Fire-and-forget: thread fills queue while we drain it in real-time
        # Do NOT await — that would block until all tokens are generated
        _pool.submit(_generate_wrapper)

        # Drain queue as tokens arrive (true streaming)
        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

    def _build_messages(self, user_text: str, lang: str = "en") -> list[dict]:
        if lang == "de":
            context_hint = (
                "[The user is speaking GERMAN. Respond fully in German. "
                "Use natural German speech patterns and fillers.]\n"
            )
        else:
            context_hint = (
                "[The user is speaking ENGLISH. Respond in English. "
                "Do NOT use any German unless the user explicitly asks about German words.]\n"
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

        sys_msg = f"{self.system_prompt}\n{context_hint}"
        messages = [{"role": "system", "content": sys_msg}]
        # Include conversation history for context
        messages.extend(self._history[-20:])
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
            # Find last sentence-ending punctuation followed by space or end
            match = list(re.finditer(r'[.!?](?:\s|$)', text))
            if match and match[-1].start() > 10:
                text = text[:match[-1].start() + 1]
            else:
                text = text.rstrip(',;: ') + '.'

        return text

    def clear_history(self):
        self._history.clear()
