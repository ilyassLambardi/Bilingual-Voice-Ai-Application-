"""ASR hallucination filter — multi-layer detection for Whisper noise."""
import logging
import math
import re
from collections import Counter
from typing import Optional

log = logging.getLogger("s2s.hallucination")

HALLUCINATION_EXACT = {
    "thanks for watching","thank you for watching","thanks for listening","thank you for listening",
    "subscribe","like and subscribe","see you next time","you","the","the end","end",
    "um","uh","mhm","oh","ah","subtitles by","amara.org","copyright","all rights reserved",
    "music","applause","laughter","silence","inaudible","foreign","so","and","but","okay so",
    "untertitel von","untertitelung","untertitel","musik","stille",
}
HALLUCINATION_SUBSTRINGS = (
    "thanks for watching","subscribe","subtitles by","amara.org",
    "untertitel","copyright","all rights reserved","captions by",
)
VALID_SHORT = {
    "yes","no","ok","okay","hi","hey","why","how","what","when","who","where","help","stop","go",
    "ja","nein","gut","doch","klar","wow","cool","nice","sure","fine","hallo","hello","bitte","genau",
    "danke","danke schön","vielen dank","tschüss","auf wiedersehen","guten tag","guten morgen",
    "wie geht's","alles klar","bye","goodbye","thanks","thank you","see you",
}
_BRACKET_RE = re.compile(r'^\s*[\[\(][A-Za-zÀ-ÿ\s]+[\]\)]\s*$')
_NOISE_RE = re.compile(r'[♪♫♬♩🎵🎶🎼]|\[(?:music|applause|laughter|silence|inaudible|foreign)\]', re.I)

def _entropy(text):
    if not text: return 0.0
    f = Counter(text.lower()); n = len(text)
    return -sum((c/n)*math.log2(c/n) for c in f.values())

def _rep(words, n=2):
    if len(words) < n+1: return 0.0
    ng = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return Counter(ng).most_common(1)[0][1]/len(ng) if ng else 0.0

def filter_hallucination(text: str, no_speech_prob: Optional[float]=None,
                         log_prob: Optional[float]=None, duration_s: Optional[float]=None) -> str:
    if not text or not text.strip(): return ""
    raw = text.strip(); lower = raw.lower()
    cleaned = lower.strip("., !?…-–—\"'")
    if not cleaned: return ""
    if all(ch in '., !?…-–—\t\n\r"\'()[]{}' for ch in lower): return ""
    if _BRACKET_RE.match(raw) or _NOISE_RE.search(raw): return ""
    if cleaned in HALLUCINATION_EXACT and cleaned not in VALID_SHORT: return ""
    if any(s in cleaned for s in HALLUCINATION_SUBSTRINGS): return ""
    words = cleaned.split()
    alpha = re.sub(r'[^a-zäöüß]', '', cleaned)
    if len(alpha) >= 6 and _entropy(alpha) < 1.8 and len(words) <= 8: return ""
    if len(words) >= 4 and _rep(words) > 0.5: return ""
    if len(words) >= 6 and _rep(words, 3) > 0.4: return ""
    # Sentence repetition
    sents = [s.strip().strip('.,!?;:').lower() for s in re.split(r'[.!?]+', raw) if s.strip()]
    if len(sents) >= 2 and Counter(sents).most_common(1)[0][1]/len(sents) > 0.6: return ""
    # Dominant word
    if len(words) >= 3 and Counter(words).most_common(1)[0][1]/len(words) > 0.6: return ""
    # Numeric noise
    digs = sum(c.isdigit() for c in cleaned); alps = sum(c.isalpha() for c in cleaned)
    if alps == 0 and digs > 0: return ""
    if alps > 0 and digs/(alps+digs) > 0.6: return ""
    if len(words) == 1 and len(cleaned) <= 2 and cleaned not in VALID_SHORT: return ""
    # Confidence scoring
    sus = 0.0
    if no_speech_prob and no_speech_prob > 0.4: sus += no_speech_prob
    if log_prob and log_prob < -0.8: sus += min(abs(log_prob)*0.3, 0.5)
    if duration_s and duration_s < 0.5 and len(words) > 4: sus += 0.3
    if sus > 1.0: return ""
    return raw
