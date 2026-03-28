# ══════════════════════════════════════════════════════════
# Bilingual S2S Voice AI — HuggingFace Spaces (Docker)
# ══════════════════════════════════════════════════════════
# Cloud mode: Groq API for LLM + ASR, Edge Neural TTS
# No GPU needed. ~2GB image.
# HF Spaces URL: https://huggingface.co/spaces/ilyass1/bilingual-voice-ai

FROM python:3.11-slim AS base

# System deps for torch + audio + ffmpeg (for Edge TTS MP3 decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (cloud-only: skip local ASR/LLM models) ──
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-cloud.txt

# ── Frontend build (multi-stage: Node builds, then discard) ──
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --production=false
COPY frontend/ ./
RUN npm run build

# ── Final image ──────────────────────────────────────────
FROM base

# Copy built frontend
COPY --from=frontend-build /app/frontend/dist frontend/dist

# Backend code
COPY backend/ backend/

# Models (Silero VAD — small, ~10MB)
COPY models/silero_vad.jit models/silero_vad.jit

# Data directory for LTM
RUN mkdir -p data

# ── Environment ──────────────────────────────────────────
ENV S2S_MODE=cloud
ENV TTS_ENGINE=edge
ENV HOST=0.0.0.0
ENV PORT=7860
# GROQ_API_KEY must be set as a Secret in HF Space Settings

EXPOSE 7860

# ── Start ────────────────────────────────────────────────
CMD ["python", "backend/main.py"]
