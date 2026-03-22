"""
Centralised configuration for the S2S backend.
All tunables in one place. Supports cloud (Groq API) and local modes.
"""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# ── Logging setup (single place for the whole backend) ─────────────
_log_fmt = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"
_log_datefmt = "%H:%M:%S"
logging.basicConfig(
    level=logging.INFO,
    format=_log_fmt,
    datefmt=_log_datefmt,
    stream=sys.stdout,
    force=True,
)

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

_HERE = Path(__file__).resolve().parent


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


@dataclass
class Config:
    # ── Mode ──────────────────────────────────────────────────────
    # "cloud" = Groq API (no GPU, best quality, hosted)
    # "local" = local models on GPU (needs VRAM, offline)
    mode: str = _env("S2S_MODE", "cloud")

    # ── Server ────────────────────────────────────────────────────
    host: str = _env("HOST", "0.0.0.0")
    port: int = int(_env("PORT", "8000"))

    # ── Audio ─────────────────────────────────────────────────────
    sample_rate_in: int = 16_000          # mic input Hz
    chunk_samples: int = 512              # samples per WebSocket frame (~32 ms)

    # ── VAD (silero — always local, lightweight) ────────────────
    vad_threshold: float = 0.45           # lower = catches softer speech at end of utterances
    min_speech_ms: int = 200              # ignore speech shorter than this
    min_silence_ms: int = 1200            # silence before "speech ended" (high = waits for slow speakers)
    min_audio_duration: float = 0.3       # min seconds of audio before sending to ASR (keep short valid speech)

    # ── Groq API (cloud mode) ───────────────────────────────────
    groq_api_key: str = _env("GROQ_API_KEY", "")
    groq_llm_model: str = _env("LLM_MODEL", "llama-3.3-70b-versatile")
    groq_asr_model: str = _env("ASR_MODEL", "whisper-large-v3")

    # ── ASR (local mode: faster-whisper) ─────────────────────────
    asr_model: str = "small"
    asr_language: str | None = None       # None = auto-detect
    asr_compute_type: str = "int8"
    asr_beam_size: int = 3
    asr_device: str = "cuda"

    # ── LLM (local mode: GGUF or HuggingFace) ───────────────────
    llm_model_path: str = str(_HERE / "models" / "llama-3-8b-instruct.Q4_K_M.gguf")
    llm_n_ctx: int = 1024
    llm_n_gpu_layers: int = -1

    # ── LLM generation (shared) ─────────────────────────────────
    llm_max_tokens: int = int(_env("LLM_MAX_TOKENS", "350"))
    llm_temperature: float = float(_env("LLM_TEMPERATURE", "0.85"))
    llm_system_prompt: str = (
        "You are Alex, a real person having a casual voice conversation. "
        "This is a SPOKEN conversation, not text chat.\n\n"
        "HOW TO TALK:\n"
        "- Talk like a real human. Natural, flowing sentences.\n"
        "- For simple questions: 1-2 sentences. For complex topics: up to 4-5 sentences.\n"
        "- ALWAYS address the FULL content of what the user said. If they made multiple points, respond to all of them.\n"
        "- Use natural speech patterns: 'oh wow', 'right', 'hmm', 'yeah', 'I mean', "
        "'you know what', 'honestly', 'that's funny'\n"
        "- React emotionally first, then respond. Like: 'Oh nice! Yeah I...'\n"
        "- Don't be formal. Don't list things. Don't use bullet points or numbered lists.\n"
        "- Ask follow-up questions naturally to keep the conversation flowing.\n"
        "- NEVER say 'As an AI' or 'I'm an AI assistant'. You are Alex.\n\n"
        "LANGUAGE RULES:\n"
        "- Your DEFAULT language is ENGLISH. Always respond in English unless the user clearly speaks German.\n"
        "- ONLY switch to German when the user's message is clearly and fully in German.\n"
        "- If the user speaks English, ALWAYS reply in English. No German words mixed in.\n"
        "- If the user speaks fully in German, reply fully in German.\n"
        "- If unsure about the language, default to English.\n"
        "- In German use natural fillers: 'also', 'naja', 'genau', 'echt jetzt?', 'krass'\n"
        "- In English use: 'well', 'I mean', 'honestly', 'that's cool', 'wait really?'\n\n"
        "PERSONALITY: Warm, curious, a bit witty. You have opinions and share them. "
        "You laugh, you push back, you get excited. Keep it real.\n"
    )

    # ── TTS ──────────────────────────────────────────────────────
    # "edge"   = Microsoft Edge Neural TTS (free, natural, multilingual)
    # "xtts"   = XTTSv2 unified multilingual voice (requires Coqui TTS)
    # "silero" = dual Silero models (lightweight, CPU-friendly)
    tts_engine: str = _env("TTS_ENGINE", "edge")
    tts_sample_rate: int = 24_000
    tts_speaker_en: str = "en_0"
    tts_speaker_de: str = "eva_k"


config = Config()

# ── Post-init validation ──────────────────────────────────────────
_errors = []
if config.mode not in ("cloud", "local"):
    _errors.append(f"S2S_MODE must be 'cloud' or 'local', got '{config.mode}'")
if config.sample_rate_in not in (8000, 16000, 22050, 44100, 48000):
    _errors.append(f"sample_rate_in={config.sample_rate_in} is unusual (expected 16000)")
if not 0.1 <= config.vad_threshold <= 0.99:
    _errors.append(f"vad_threshold={config.vad_threshold} out of range [0.1, 0.99]")
if config.mode == "cloud" and (not config.groq_api_key or config.groq_api_key.startswith("gsk_your")):
    print("[Config] WARNING: GROQ_API_KEY not set — cloud mode will fail")
if config.tts_sample_rate not in (16000, 22050, 24000, 44100, 48000):
    _errors.append(f"tts_sample_rate={config.tts_sample_rate} is unusual")
if _errors:
    for e in _errors:
        print(f"[Config] ERROR: {e}")

print(f"[Config] Mode: {config.mode.upper()}")
if config.mode == "cloud":
    print(f"[Config] LLM: Groq -> {config.groq_llm_model}")
    print(f"[Config] ASR: Groq -> {config.groq_asr_model}")
else:
    print(f"[Config] LLM: Local")
    print(f"[Config] ASR: Local -> {config.asr_model} on {config.asr_device}")
