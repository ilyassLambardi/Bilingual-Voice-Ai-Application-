"""Language detection (EN/DE) — signal-based, English default.

Uses explicit German signals (ä/ö/ü/ß and unambiguous German words)
instead of statistical classifiers like langid which often misclassify
short English phrases as German.
"""
import re

_DE_CHARS = set("äöüß")
_UNAMBIGUOUS_DE = {
    "ich","du","er","wir","ihr","mein","dein","sein","das","der","ein","eine",
    "ist","sind","habe","haben","wird","kann","muss","soll","darf",
    "und","oder","aber","mit","von","zu","auf","nicht","auch","noch",
    "dass","wenn","weil","dann","sehr","nur","wie","für","über","nach",
    "hier","dort","immer","jetzt","kein","keine","danke","bitte","guten",
    "ja","nein","warum","wo","wer","wann","dir","mir","uns",
    "geht","kommt","macht","gibt","weiß","brauche",
    "schön","natürlich","vielleicht","eigentlich","trotzdem","deshalb",
    "alles","nichts","etwas","welche","dieser","diese","dieses",
}

def detect_lang_from_text(text: str) -> str:
    """Detect EN or DE. English is default; German requires explicit signals."""
    if not text.strip():
        return "en"
    lower = text.strip().lower()
    words = {re.sub(r"[^\w]", "", w) for w in lower.split()} - {""}
    if any(c in lower for c in _DE_CHARS) or words & _UNAMBIGUOUS_DE:
        return "de"
    return "en"

detect_sentence_lang = detect_lang_from_text
