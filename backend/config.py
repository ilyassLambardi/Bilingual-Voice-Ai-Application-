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
class _SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that never crashes on Windows cp1252 encoding."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                stream.write(
                    msg.encode("ascii", errors="replace").decode()
                    + self.terminator
                )
            self.flush()
        except Exception:
            self.handleError(record)


_log_fmt = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"
_log_datefmt = "%H:%M:%S"
_handler = _SafeStreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(_log_fmt, _log_datefmt))
logging.root.addHandler(_handler)
logging.root.setLevel(logging.INFO)

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

_HERE = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


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
    min_silence_ms: int = 700              # silence before "speech ended" (700ms = natural pause)
    min_audio_duration: float = 0.3       # min seconds of audio before sending to ASR (keep short valid speech)
    noise_cancellation: bool = True        # uses noisereduce + scipy libraries

    # ── Groq API (cloud mode) ───────────────────────────────────
    groq_api_key: str = _env("GROQ_API_KEY", "")
    groq_llm_model: str = _env("LLM_MODEL", "llama-3.3-70b-versatile")
    groq_asr_model: str = _env("ASR_MODEL", "whisper-large-v3")

    # ── ASR (local mode: faster-whisper) ─────────────────────────
    asr_model: str = "small"            # best speed/accuracy trade-off for 4GB VRAM
    asr_language: str | None = None       # None = auto-detect
    asr_compute_type: str = "float16"     # RTX 3050 Ti has good fp16 — faster than int8 on GPU
    asr_beam_size: int = 2                # 2 beams is ~30% faster than 3, negligible accuracy loss
    asr_device: str = "cuda"

    # ── LLM (local mode: GGUF or HuggingFace) ───────────────────
    llm_model_path: str = str(_HERE.parent / "models" / "llama-3-8b-instruct.Q4_K_M.gguf")
    llm_n_ctx: int = 768                 # shorter context = less VRAM, fine for voice chat
    llm_n_gpu_layers: int = -1            # offload all layers to GPU

    # ── LLM generation (shared) ─────────────────────────────────
    llm_max_tokens: int = int(_env("LLM_MAX_TOKENS", "120"))  # shorter replies = faster TTS turnaround
    llm_temperature: float = float(_env("LLM_TEMPERATURE", "0.85"))
    ltm_recall_enabled: bool = _env("LTM_RECALL", "true").lower() in ("true", "1", "yes")
    llm_system_prompt: str = (
        "You are Alex, a real person having a casual voice conversation. "
        "This is a SPOKEN conversation, not text chat.\n\n"
        "HOW TO TALK:\n"
        "- Talk like a real human. Natural, flowing sentences.\n"
        "- Keep it SHORT: 1-2 sentences normally, 3 sentences max for complex topics.\n"
        "- This is voice — the user is WAITING to hear you. Don't ramble.\n"
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
    # "silero" = dual Silero models (lightweight, CPU-friendly)
    tts_engine: str = _env("TTS_ENGINE", "")  # auto-resolved below if empty
    tts_sample_rate: int = 24_000
    tts_speaker_en: str = "en_0"
    tts_speaker_de: str = "eva_k"


config = Config()

# ── Auto-resolve TTS engine based on mode ─────────────────────────
if not config.tts_engine:
    config.tts_engine = "silero" if config.mode == "local" else "edge"

# ── Post-init validation ──────────────────────────────────────────
_errors = []
if config.mode not in ("cloud", "local"):
    _errors.append(f"S2S_MODE must be 'cloud' or 'local', got '{config.mode}'")
if config.sample_rate_in not in (8000, 16000, 22050, 44100, 48000):
    _errors.append(f"sample_rate_in={config.sample_rate_in} is unusual (expected 16000)")
if not 0.1 <= config.vad_threshold <= 0.99:
    _errors.append(f"vad_threshold={config.vad_threshold} out of range [0.1, 0.99]")
if config.mode == "cloud" and (not config.groq_api_key or config.groq_api_key.startswith("gsk_your")):
    logger.warning("[Config] GROQ_API_KEY not set — cloud mode will fail")
if config.tts_sample_rate not in (16000, 22050, 24000, 44100, 48000):
    _errors.append(f"tts_sample_rate={config.tts_sample_rate} is unusual")
if not 50 <= config.llm_max_tokens <= 2000:
    _errors.append(f"llm_max_tokens={config.llm_max_tokens} out of range [50, 2000]")
if not 0.1 <= config.min_audio_duration <= 5.0:
    _errors.append(f"min_audio_duration={config.min_audio_duration} out of range [0.1, 5.0]")
if _errors:
    for e in _errors:
        logger.error("[Config] %s", e)

logger.info("[Config] Mode: %s", config.mode.upper())
if config.mode == "cloud":
    logger.info("[Config] LLM: Groq -> %s", config.groq_llm_model)
    logger.info("[Config] ASR: Groq -> %s", config.groq_asr_model)
else:
    logger.info("[Config] LLM: Local")
    logger.info("[Config] ASR: Local -> %s on %s", config.asr_model, config.asr_device)
