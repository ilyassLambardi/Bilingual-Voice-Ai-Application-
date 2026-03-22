"""
Advanced Hallucination Filter for ASR output.

Shared module used by both local ASR (asr.py) and Groq ASR (asr_groq.py).
Combines multiple detection layers to catch Whisper hallucinations while
preserving valid short conversational input.

Detection layers:
  1. Exact match against known hallucination phrases
  2. Substring match for partial hallucination patterns
  3. Bracket/tag detection: [Music], (Applause), ♪, etc.
  4. Character entropy: hallucinations often have abnormally low entropy
  5. N-gram repetition: "I'm going to I'm going to I'm going to"
  6. Sentence-level repetition: "Thank you so much. Thank you so much."
  7. Dominant word ratio: >60% of words are the same word
  8. Numeric-heavy: random digit noise from silence
  9. Unicode noise: musical symbols, special chars
 10. Confidence scoring: combines multiple weak signals
"""

import logging
import math
import re
from collections import Counter
from typing import Optional

log = logging.getLogger("s2s.hallucination")

# ── Known hallucination phrases (lowercased, stripped) ─────────────────
HALLUCINATION_EXACT = {
    # English Whisper noise hallucinations
    "thank you", "thanks", "thanks for watching", "thank you for watching",
    "thanks for listening", "thank you for listening",
    "subscribe", "like and subscribe", "like subscribe",
    "please subscribe", "hit the bell", "hit the like button",
    "ring the bell", "click subscribe",
    "bye", "goodbye", "bye bye", "see you", "see you next time",
    "you", "the", "the end", "end",
    "um", "uh", "hmm", "huh",
    "oh", "ah", "mhm",
    "yes", "no",  # only as standalone full transcripts, not in longer text
    "subtitles by", "amara.org", "subtitles made by",
    "subtitled by", "captions by",
    "copyright", "all rights reserved",
    "music", "applause", "laughter", "silence",
    "music playing", "music plays",
    "inaudible", "indistinct",
    "foreign", "foreign language",
    "no sound", "no audio",
    "you're welcome",
    "i don't know what to say",
    "so", "and", "but",
    "okay so", "alright",
    # German Whisper noise hallucinations
    "danke", "danke schön", "danke schon", "tschüss", "tschuss",
    "wie geht's die", "wie gehts die", "wie geht es dir",
    "wie geht's", "wie gehts", "hallo wie geht's",
    "guten tag", "auf wiedersehen", "bis bald",
    "vielen dank", "herzlich willkommen",
    "untertitel von", "untertitelung", "untertitel",
    "untertitel der amara.org-community",
    "musik", "beifall", "gelächter", "stille",
    "bis zum nächsten mal", "bis dann",
}

# Substrings that indicate hallucination even in longer text
HALLUCINATION_SUBSTRINGS = (
    "thanks for watching", "thank you for watching",
    "thanks for listening", "thank you for listening",
    "subscribe", "subtitles by", "amara.org",
    "untertitel", "copyright", "all rights reserved",
    "please like", "hit the bell", "hit the like",
    "ring the bell", "click subscribe",
    "captions by", "subtitled by",
    "translated by", "übersetzt von",
    "community beiträge", "community contributions",
)

# Regex for bracket/tag patterns: [Music], (Applause), ♪, etc.
_BRACKET_RE = re.compile(
    r'^\s*[\[\(]'           # starts with [ or (
    r'[A-Za-zÀ-ÿ\s]+'      # words inside
    r'[\]\)]\s*$'            # ends with ] or )
)
_MUSIC_SYMBOLS_RE = re.compile(r'[♪♫♬♩🎵🎶🎼]')
_SOUND_TAG_RE = re.compile(
    r'\[(?:music|applause|laughter|silence|inaudible|foreign|blank_audio)\]',
    re.IGNORECASE,
)

# Short words that are valid conversational input — NEVER filter these
VALID_SHORT = {
    "yes", "no", "ok", "okay", "hi", "hey", "why", "how",
    "what", "when", "who", "where", "help", "stop", "go",
    "ja", "nein", "gut", "naja", "ach", "doch", "klar",
    "wow", "cool", "nice", "sure", "fine", "yep", "nah",
    "hallo", "hello", "bitte", "genau", "echt", "krass",
    "oh wow", "oh nice", "oh cool", "oh no", "oh yeah",
    "na klar", "na gut", "na ja",
}

# ── Character entropy calculation ──────────────────────────────────────

def _char_entropy(text: str) -> float:
    """Shannon entropy of character distribution. Low entropy = repetitive."""
    if not text:
        return 0.0
    freq = Counter(text.lower())
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values() if c > 0)


# ── N-gram repetition detection ───────────────────────────────────────

def _ngram_repetition_score(words: list[str], n: int = 2) -> float:
    """Ratio of repeated n-grams to total n-grams. 1.0 = all identical."""
    if len(words) < n + 1:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(ngrams)


def _sentence_repetition_score(text: str) -> float:
    """Detect repeated sentences. Returns ratio of most-repeated to total."""
    # Split on sentence boundaries
    sentences = [s.strip().strip('.,!?;:').lower() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) < 2:
        return 0.0
    counts = Counter(sentences)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(sentences)


# ── Main filter function ──────────────────────────────────────────────

def filter_hallucination(
    text: str,
    no_speech_prob: Optional[float] = None,
    log_prob: Optional[float] = None,
    duration_s: Optional[float] = None,
) -> str:
    """Advanced hallucination filter. Returns cleaned text or '' if hallucinated.

    Parameters
    ----------
    text : str
        Raw ASR transcription output.
    no_speech_prob : float, optional
        Whisper's no_speech_probability for the segment (0-1).
        Higher = more likely silence/noise.
    log_prob : float, optional
        Average log probability of the transcription.
        Lower (more negative) = less confident.
    duration_s : float, optional
        Audio duration in seconds. Used for length-based heuristics.

    Returns
    -------
    str : Cleaned text, or '' if the text is a hallucination.
    """
    if not text or not text.strip():
        return ""

    raw = text.strip()
    lower = raw.lower()
    cleaned = lower.strip("., !?…-–—\"'")

    if not cleaned:
        log.info(f"[Filter] Empty after strip: '{text}'")
        return ""

    # ── Layer 1: Pure punctuation / whitespace noise ──────────────
    if all(ch in '., !?…-–—\t\n\r"\'()[]{}' for ch in lower):
        log.info(f"[Filter] Punctuation noise: '{text}'")
        return ""

    # ── Layer 2: Bracket/tag patterns: [Music], (Applause), ♪ ────
    if _BRACKET_RE.match(raw) or _SOUND_TAG_RE.search(lower):
        log.info(f"[Filter] Bracket/tag hallucination: '{text}'")
        return ""

    if _MUSIC_SYMBOLS_RE.search(raw):
        log.info(f"[Filter] Music symbol noise: '{text}'")
        return ""

    # ── Layer 3: Exact hallucination match ────────────────────────
    if cleaned in HALLUCINATION_EXACT:
        # But don't filter if it's a valid short word in context
        if cleaned not in VALID_SHORT:
            log.info(f"[Filter] Exact hallucination: '{text}'")
            return ""

    # ── Layer 4: Substring hallucination match ────────────────────
    for sub in HALLUCINATION_SUBSTRINGS:
        if sub in cleaned:
            log.info(f"[Filter] Substring hallucination '{sub}': '{text}'")
            return ""

    # ── Layer 5: Character entropy (low = repetitive noise) ───────
    words = cleaned.split()
    alpha_text = re.sub(r'[^a-zäöüß]', '', cleaned)
    if len(alpha_text) >= 6:
        entropy = _char_entropy(alpha_text)
        # Very low entropy + short text = likely noise
        # Normal English text has entropy ~3.5-4.5
        if entropy < 1.8 and len(words) <= 8:
            log.info(f"[Filter] Low entropy ({entropy:.2f}): '{text}'")
            return ""

    # ── Layer 6: N-gram repetition ────────────────────────────────
    if len(words) >= 4:
        # Bigram repetition: "I'm going to I'm going to"
        bigram_score = _ngram_repetition_score(words, n=2)
        if bigram_score > 0.5:
            log.info(f"[Filter] Bigram repetition ({bigram_score:.2f}): '{text}'")
            return ""

        # Trigram repetition: "thank you so much thank you so much"
        if len(words) >= 6:
            trigram_score = _ngram_repetition_score(words, n=3)
            if trigram_score > 0.4:
                log.info(f"[Filter] Trigram repetition ({trigram_score:.2f}): '{text}'")
                return ""

    # ── Layer 7: Sentence-level repetition ────────────────────────
    sent_score = _sentence_repetition_score(raw)
    if sent_score > 0.6:
        log.info(f"[Filter] Sentence repetition ({sent_score:.2f}): '{text}'")
        return ""

    # ── Layer 8: Dominant word ratio ──────────────────────────────
    if len(words) >= 3:
        unique = set(words)
        if len(unique) == 1:
            log.info(f"[Filter] All-same-word repetition: '{text}'")
            return ""
        word_counts = Counter(words)
        top_count = word_counts.most_common(1)[0][1]
        if top_count / len(words) > 0.6:
            log.info(f"[Filter] Dominant word ({top_count}/{len(words)}): '{text}'")
            return ""

    # ── Layer 9: Numeric-heavy (digit sequences from noise) ───────
    digit_chars = sum(1 for ch in cleaned if ch.isdigit())
    alpha_chars = sum(1 for ch in cleaned if ch.isalpha())
    if alpha_chars > 0 and digit_chars / (alpha_chars + digit_chars) > 0.6:
        log.info(f"[Filter] Numeric-heavy noise: '{text}'")
        return ""
    if alpha_chars == 0 and digit_chars > 0:
        log.info(f"[Filter] Pure numeric noise: '{text}'")
        return ""

    # ── Layer 10: Too-short single words not in valid set ─────────
    if len(words) == 1 and len(cleaned) <= 2 and cleaned not in VALID_SHORT:
        log.info(f"[Filter] Too-short unknown: '{text}'")
        return ""

    # ── Layer 11: Confidence scoring (combine weak signals) ───────
    # Each signal adds to a "suspicion" score. If total > threshold, reject.
    suspicion = 0.0

    if no_speech_prob is not None and no_speech_prob > 0.4:
        suspicion += no_speech_prob  # 0.4–1.0 adds 0.4–1.0

    if log_prob is not None and log_prob < -0.8:
        suspicion += min(abs(log_prob) * 0.3, 0.5)  # low confidence adds up to 0.5

    if duration_s is not None and duration_s < 0.5 and len(words) > 4:
        # Many words in very short audio = suspicious
        suspicion += 0.3

    if len(alpha_text) >= 6:
        entropy = _char_entropy(alpha_text)
        if entropy < 2.5:
            suspicion += (2.5 - entropy) * 0.2  # up to 0.5 for very low entropy

    # Check if any word partially matches known hallucinations
    halluc_word_hits = sum(1 for w in words if any(w in h for h in HALLUCINATION_EXACT))
    if len(words) > 0:
        suspicion += (halluc_word_hits / len(words)) * 0.3

    if suspicion > 1.0:
        log.info(f"[Filter] High suspicion ({suspicion:.2f}): '{text}'")
        return ""

    # ── Passed all filters ────────────────────────────────────────
    return raw
