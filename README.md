# Bilingual Speech-to-Speech Conversational AI

> **Graduation Thesis Project** — Real-time bilingual (English/German) voice AI with streaming pipeline, language-aware persona, and interactive 3D visualization.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

A full-stack **speech-to-speech** conversational AI system that enables real-time bilingual conversations in **English** and **German**. The system features a streaming WebSocket pipeline connecting Voice Activity Detection, Automatic Speech Recognition, Large Language Model inference, and Text-to-Speech synthesis — all orchestrated with sub-second latency.

The frontend provides a 3D fluid orb visualization built with Three.js/WebGL shaders that reacts to pipeline state and language context, alongside a modern React UI with dark/light themes.

### Demo

```
You (English):  "Hey Alex, what does Fernweh mean?"
Alex (English): "Fernweh is this beautiful German word — it means a longing for
                 faraway places, almost like reverse homesickness..."
                 • "Ich hab so Fernweh, ich muss einfach wieder reisen."
                 • "Das Fernweh packt mich jedes Mal im Winter."
```

```
You (German):   "Erzähl mir was über Machine Learning."
Alex (German):  "Also, Machine Learning ist im Grunde ein Teilgebiet der KI,
                 bei dem Algorithmen aus Daten lernen, anstatt explizit
                 programmiert zu werden..."
```

---

## Key Features

### Speech Pipeline
- **Real-time streaming** — token-by-token LLM generation with sentence-level TTS for minimal perceived latency
- **Bilingual language detection** — automatic EN/DE recognition with mid-conversation code-switching
- **Interruptible generation** — speak while the AI is talking to cut it off naturally
- **Backchannel detection** — recognizes short affirmations ("mhm", "yeah") without interrupting
- **Ghost texting** — see partial ASR transcription as you speak

### Language Intelligence
- **Polyglot Persona** — dynamic language mirroring (responds in whichever language you use)
- **Denglisch handling** — mixed EN/DE input triggers dominant-language responses with natural borrowing
- **Teacher Mode** — ask "What does [word] mean?" to get cross-lingual explanations with example sentences and cultural context
- **Intent-based detection** — overrides ASR language tags with structural sentence analysis
- **Language shift tracking** — detects when users switch languages and emits events for UI adaptation

### AI & Models
- **Dual-mode architecture** — cloud mode (Groq API: Llama 3.3 70B + Whisper Large v3) or local mode (Qwen 2.5 1.5B + faster-whisper)
- **Long-term memory** — SQLite-backed semantic memory with cross-session context recall
- **Conversation history** — sliding window with configurable depth
- **Hallucination filtering** — ASR output validation to reject noise artifacts

### Frontend & Visualization
- **3D Fluid Orb** — WebGL shader with real-time state morphing (idle → listening → thinking → speaking)
- **Language-bleed shader** — lava-lamp color blending between English (blue) and German (amber) tones
- **Glow ring & particle effects** — reactive visual feedback for speech activity
- **Voice waveform** — real-time audio visualization during listening/speaking
- **Pipeline visualizer** — live display of each pipeline stage with latency metrics
- **Dark/Light themes** — full theme support with smooth transitions
- **Keyboard shortcuts** — spacebar push-to-talk, Escape to stop, etc.
- **Chat export** — download conversation as JSON
- **Responsive design** — mobile-friendly layout

### Infrastructure
- **WebSocket streaming** — binary PCM audio + JSON control messages over a single connection
- **Auto-reconnect** — exponential backoff on connection drops
- **Docker support** — containerized deployment with Dockerfile
- **Configurable** — environment variables for all tunable parameters

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                         │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │ FluidOrb │  │ Transcript│  │  Controls │  │  AudioStream │  │
│  │ (WebGL)  │  │  Window  │  │ + ChatInput│  │  (PCM I/O)   │  │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘  └──────┬───────┘  │
│       └──────────────┴──────────────┴───────────────┘           │
│                           │ WebSocket                           │
└───────────────────────────┼─────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│                     Backend (FastAPI)                            │
│                           │                                     │
│  ┌────────────────────────▼────────────────────────────────┐    │
│  │              Pipeline Manager (Orchestrator)            │    │
│  │                                                         │    │
│  │   ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐            │    │
│  │   │ VAD │ →  │ ASR │ →  │ LLM │ →  │ TTS │ → Audio    │    │
│  │   │Silero│   │Whisper│  │Groq/ │   │Silero│   Stream   │    │
│  │   │     │    │      │   │Qwen  │   │/XTTS│            │    │
│  │   └─────┘    └─────┘   └──┬──┘    └─────┘            │    │
│  │                            │                            │    │
│  │                     ┌──────▼──────┐                     │    │
│  │                     │  LTM Memory │                     │    │
│  │                     │  (SQLite)   │                     │    │
│  │                     └─────────────┘                     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
├── backend/                        
│   ├── main.py                     # FastAPI server + WebSocket endpoint
│   ├── config.py                   # Centralized configuration (env-driven)
│   └── pipeline/
│       ├── vad.py                  # Voice Activity Detection (Silero VAD)
│       ├── asr.py                  # Local ASR (faster-whisper, bilingual)
│       ├── asr_groq.py             # Cloud ASR (Groq Whisper API)
│       ├── llm.py                  # Local LLM (Qwen2.5-1.5B-Instruct)
│       ├── llm_groq.py             # Cloud LLM (Groq Llama 3.3 70B)
│       ├── tts.py                  # TTS (Silero v3, EN + DE voices)
│       ├── tts_xtts.py             # TTS (XTTSv2, unified multilingual)
│       ├── memory.py               # Long-term memory (SQLite + semantic)
│       └── manager.py              # Pipeline orchestrator + state machine
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # Main application (state management)
│   │   ├── hooks/
│   │   │   ├── useWebSocket.js     # WebSocket connection management
│   │   │   └── useAudioStream.js   # PCM audio capture + playback
│   │   └── components/
│   │       ├── FluidOrb.jsx        # 3D orb (Three.js + custom shaders)
│   │       ├── TranscriptWindow.jsx# Chat message display
│   │       ├── SubtitleBar.jsx     # Bilingual subtitle overlay
│   │       ├── ChatInput.jsx       # Text input with send
│   │       ├── PipelineVisualizer.jsx # Live pipeline stage display
│   │       ├── VoiceWaveform.jsx   # Audio waveform visualization
│   │       ├── SettingsPanel.jsx   # Configuration UI
│   │       └── ...                 # 15+ additional UI components
│   └── package.json
├── models/                         # Model weights (gitignored, ~4 GB)
├── scripts/
│   └── download_models.py          # Automated model downloader
├── Dockerfile                      # Container deployment
├── requirements.txt                # Python dependencies
└── docs/                           # Documentation + project report
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Server** | FastAPI + Uvicorn | Async WebSocket server |
| **VAD** | Silero VAD | Speech boundary detection |
| **ASR** | faster-whisper / Groq Whisper | Bilingual speech recognition |
| **LLM** | Qwen 2.5 1.5B / Llama 3.3 70B | Conversational AI with language mirroring |
| **TTS** | Silero v3 / XTTSv2 | Bilingual speech synthesis |
| **Memory** | SQLite | Long-term conversation memory |
| **Frontend** | React 18 + Vite | Single-page application |
| **3D** | Three.js + React Three Fiber | WebGL shader-based orb visualization |
| **Styling** | TailwindCSS + Framer Motion | Responsive UI with animations |
| **Audio** | Web Audio API + AudioWorklet | Real-time PCM streaming |

---

## Quick Start

### Prerequisites

- **Python** >= 3.10
- **Node.js** >= 18
- **CUDA GPU** recommended (~4 GB VRAM) — falls back to CPU automatically

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/bilingual-voice-ai.git
cd bilingual-voice-ai

# Backend
pip install -r requirements.txt

# Frontend
cd frontend && npm install && cd ..
```

### 2. Configure

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and add your Groq API key (free tier available)
```

### 3. Run

```bash
# Option A: Production (backend serves frontend)
cd frontend && npm run build && cd ..
python backend/main.py
# → Open http://localhost:8000

# Option B: Development (hot reload)
python backend/main.py          # Terminal 1 → port 8000
cd frontend && npm run dev      # Terminal 2 → port 5173
```

### 4. Use

1. Click the **microphone button** or hold **Spacebar**
2. Speak in **English**, **German**, or switch mid-sentence
3. The AI responds in your language with natural voice
4. Ask **"What does [word] mean?"** to activate Teacher Mode

---

## Models

| Component | Model | Parameters | Mode |
|-----------|-------|-----------|------|
| VAD | Silero VAD v5 | ~2M | Always local |
| ASR | Whisper Large v3 | 1.5B | Cloud (Groq) |
| ASR | faster-whisper base | 74M | Local (int8) |
| LLM | Llama 3.3 70B Versatile | 70B | Cloud (Groq) |
| LLM | Qwen 2.5 1.5B Instruct | 1.5B | Local |
| TTS | Silero v3 (EN + DE) | ~10M each | Always local |
| TTS | XTTSv2 (unified) | ~500M | Optional |

Models are downloaded automatically on first run via `scripts/download_models.py`.

---

## Configuration

All settings are configurable via environment variables in `backend/.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `S2S_MODE` | `cloud` | `cloud` (Groq API) or `local` (GPU inference) |
| `GROQ_API_KEY` | — | API key from [console.groq.com](https://console.groq.com) |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Cloud LLM model |
| `LLM_MAX_TOKENS` | `150` | Max response tokens |
| `TTS_ENGINE` | `silero` | `silero` or `xtts` (XTTSv2) |
| `ASR_MODEL` | `whisper-large-v3` | Cloud ASR model |

---

## Docker

```bash
docker build -t voice-ai .
docker run -p 8000:8000 --env-file backend/.env voice-ai
```

---

## Technical Highlights

### Streaming Pipeline Design
The system uses a **sentence-level streaming** architecture: the LLM generates tokens continuously, and as soon as a complete sentence is detected (via punctuation heuristics), it is immediately sent to TTS for synthesis while the LLM continues generating the next sentence. This overlapping approach minimizes perceived latency.

### Concurrency & Race Condition Handling
The pipeline manager implements careful state machine logic with `_generating` guards to prevent concurrent pipeline executions from overlapping audio. Interrupt detection uses a frame-counting mechanism to distinguish intentional speech from brief noise.

### Language-Aware Shader
The FluidOrb fragment shader uses a custom `uLanguageBlend` uniform that drives a lava-lamp-style color interpolation between English (electric blue) and German (sunset amber) color spaces, modulated by dual sine waves for organic spatial variation.

### Memory System
Long-term memory uses SQLite with semantic similarity search, allowing the AI to recall context from previous conversations. Memory entries are automatically created from conversation exchanges and retrieved based on relevance to the current topic.

---

## License

This project is part of a graduation thesis. See [LICENSE](LICENSE) for details.

---

## Author

**Ilyass** — Graduation Thesis Project  
Bilingual Speech-to-Speech Conversational AI System
