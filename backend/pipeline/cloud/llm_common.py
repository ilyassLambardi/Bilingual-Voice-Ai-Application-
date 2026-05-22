"""
LLM utilities — response cleanup and message building helpers.
"""

import re


def clean_response(text: str) -> str:
    """Minimal cleanup of LLM output — strip markdown artifacts, ensure punctuation."""
    if not text:
        return "Hmm, I didn't catch that."

    text = text.strip()
    if not text:
        return text

    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    text = re.sub(r'^[-\*]\s*', '', text).strip()

    if text and text[-1] not in '.!?':
        match = list(re.finditer(r'[.!?](?:\s|$)', text))
        if match and match[-1].start() > 10:
            text = text[:match[-1].start() + 1]
        else:
            text = text.rstrip(',;: ') + '.'

    return text


def build_context_hint(lang: str, user_text: str) -> str:
    """Build language context hint and teacher mode for LLM messages."""
    if lang == "de":
        context_hint = (
            "[SPRACHE: DEUTSCH. Du MUSST auf Deutsch antworten. "
            "Antworte NUR auf Deutsch. Verwende natürliche deutsche Sprache. "
            "KEIN Englisch in deiner Antwort.]\n"
        )
    else:
        context_hint = (
            "[The user is speaking ENGLISH. Respond in English. "
            "Do NOT use any German unless the user explicitly asks about German words.]\n"
        )

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

    return context_hint
