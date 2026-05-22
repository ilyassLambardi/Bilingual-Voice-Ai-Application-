"""
Language detection utilities for the cloud pipeline.

Uses the ``langid`` library as the primary statistical detector,
restricted to EN/DE for fast bilingual classification.

For very short text (1-2 words) where statistical models are unreliable,
falls back to an unambiguous German keyword set that excludes
EN/DE homographs (will, die, war, hat, in, an, bin, tag, halt, gut, alt).
"""

import re

import langid

# Restrict langid to bilingual EN/DE — faster and more accurate
langid.set_languages(["en", "de"])

# German special characters — always a definitive German signal
_DE_CHARS = set("äöüß")

# Unambiguous German words for short-text validation.
# Deliberately excludes EN/DE homographs: will, die, war, hat, in, an,
# bin, tag, halt, gut, alt, so, also, was (English "was").
_UNAMBIGUOUS_DE = {
    "ich", "du", "er", "sie", "es", "wir", "ihr",
    "mein", "dein", "sein", "das", "der", "ein", "eine",
    "ist", "sind", "habe", "hast", "hatte", "haben",
    "wird", "werde", "wirst", "werden", "wurde",
    "kann", "muss", "soll", "darf", "mag",
    "und", "oder", "aber", "mit", "von", "zu", "auf",
    "nicht", "auch", "noch", "dass", "wenn", "weil", "dann",
    "sehr", "nur", "mehr", "wie", "als", "seit", "bis",
    "über", "unter", "nach", "vor", "zwischen", "durch", "für",
    "hier", "dort", "immer", "vielleicht", "eigentlich", "natürlich",
    "wirklich", "gerade", "heute", "gestern", "morgen", "jetzt",
    "kein", "keine", "keinen", "keinem", "keiner",
    "schlecht", "groß", "klein", "neu",
    "glaube", "denke", "finde", "meine", "würde", "könnte", "sollte",
    "hallo", "danke", "bitte", "guten", "abend", "nacht",
    "ja", "nein", "warum", "wo", "wer", "wann",
    "dir", "mir", "mich", "dich", "uns", "euch", "ihnen",
    "alles", "nichts", "müde", "freut", "arbeit", "zeit",
    "gar", "doch", "eben", "schon",
    "geht", "kommt", "macht", "sagt", "gibt", "nimmt",
    "weiß", "kennt", "brauche", "brauchst", "braucht",
}


def detect_lang_from_text(text: str) -> str:
    """Detect EN or DE using the langid library.

    For very short text (1-2 words), cross-validates with unambiguous
    German keywords since statistical models are less reliable there.
    English is the default when uncertain.
    """
    stripped = text.strip()
    if not stripped:
        return "en"

    # Primary: langid statistical classifier
    lang, _confidence = langid.classify(stripped)

    # Restrict to EN/DE; fall back on German special characters
    if lang not in ("en", "de"):
        lang = "de" if any(c in stripped.lower() for c in _DE_CHARS) else "en"

    # Short text (1-2 words): langid less reliable, require unambiguous signal
    words = stripped.lower().split()
    if len(words) <= 2:
        clean_words = {re.sub(r"[^\w]", "", w) for w in words} - {""}
        has_de_char = any(c in stripped.lower() for c in _DE_CHARS)
        has_de_word = bool(clean_words & _UNAMBIGUOUS_DE)
        return "de" if (has_de_char or has_de_word) else "en"

    return lang


def detect_sentence_lang(text: str) -> str:
    """Alias for detect_lang_from_text — used by TTS and manager."""
    return detect_lang_from_text(text)
