# ══════════════════════════════════════════════════════════
# Bilingual S2S Voice AI — Docker (HuggingFace Spaces)
# ══════════════════════════════════════════════════════════
# Cloud mode: Groq API for LLM + ASR, Silero TTS on CPU
# No GPU needed. ~2GB image.

FROM python:3.11-slim

# System deps for torch + audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ──────────────────────────────────
COPY requirements.txt .
# Install torch CPU-only (smaller image, no CUDA needed in cloud mode)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Frontend build ───────────────────────────────────────
COPY frontend/package.json frontend/package-lock.json* frontend/
RUN cd frontend && npm install --production=false

COPY frontend/ frontend/
RUN cd frontend && npm run build

# ── Backend code ─────────────────────────────────────────
COPY backend/ backend/

# ── Models (Silero VAD + TTS — small, ~120MB total) ─────
# These are downloaded on first run if not present
COPY models/silero_vad.jit models/silero_vad.jit
COPY models/v3_en.pt models/v3_en.pt
COPY models/v3_de.pt models/v3_de.pt

# ── Data directory for LTM ───────────────────────────────
RUN mkdir -p data

# ── Environment ──────────────────────────────────────────
ENV S2S_MODE=cloud
ENV HOST=0.0.0.0
ENV PORT=7860
# GROQ_API_KEY must be set as a secret in HF Spaces

EXPOSE 7860

# ── Start ────────────────────────────────────────────────
CMD ["python", "backend/main.py"]
