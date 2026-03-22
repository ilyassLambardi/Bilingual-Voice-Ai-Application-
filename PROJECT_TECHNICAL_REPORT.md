# Bilingual Speech-to-Speech Conversational AI — Technical Report

**Author:** Ilyass  
**Project Type:** Graduation Thesis — Machine Learning Engineering  
**Date:** 2025  
**Deployment:** [https://ilyass1-starch.hf.space](https://ilyass1-starch.hf.space)

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [System Architecture](#4-system-architecture)
5. [Data Flow & Protocol](#5-data-flow--protocol)
6. [VAD Component](#6-vad-component)
7. [ASR Component](#7-asr-component)
8. [LLM Component](#8-llm-component)
9. [TTS Component](#9-tts-component)
10. [Long-Term Memory](#10-long-term-memory)
11. [Pipeline Orchestration](#11-pipeline-orchestration)
12. [Interrupt & Backchannel](#12-interrupt--backchannel-handling)
13. [Language Detection](#13-language-detection--code-switching)
14. [Concurrency Model](#14-concurrency-model)
15. [Frontend Architecture](#15-frontend-architecture)
16. [Audio I/O Subsystem](#16-audio-io-subsystem)
17. [Rate Limiting](#17-rate-limiting)
18. [Configuration](#18-configuration)
19. [Deployment](#19-deployment)
20. [Performance](#20-performance-analysis)
21. [Limitations](#21-limitations)
22. [Future Work](#22-future-work)
23. [Conclusion](#23-conclusion)
24. [Appendices](#appendices)

---

## 1. Abstract

This report presents a real-time bilingual speech-to-speech (S2S) conversational AI system supporting English and German. The system implements a four-stage streaming pipeline — Voice Activity Detection (VAD), Automatic Speech Recognition (ASR), Large Language Model (LLM) inference, and Text-to-Speech (TTS) synthesis — orchestrated over WebSocket with sub-2.5s end-to-end latency. The architecture supports mid-utterance interruption, backchannel detection, language code-switching, and long-term memory. It operates in dual mode: cloud (Groq API — Llama 3.3 70B, Whisper Large v3) and local (Qwen 2.5 1.5B, faster-whisper). The frontend provides a React UI with a real-time WebGL fluid orb driven by FFT audio analysis. Deployed as a Docker container on HuggingFace Spaces.

---

## 2. Introduction

### 2.1 Motivation

Most conversational AI remains text-only. Existing voice assistants (Siri, Alexa) have high latency, limited language support, and rigid turn-taking. Bilingual voice AI with real-time language switching is underexplored. This project builds a system that feels like talking to a real bilingual person.

### 2.2 Objectives

1. End-to-end latency under 2.5s (speech input → first audio output)
2. Seamless bilingual EN/DE with mid-conversation code-switching
3. Natural conversation mechanics: interruption, backchannel, turn-taking
4. Production-quality web frontend with real-time visual feedback
5. Accessible web deployment requiring no client installation

---

## 3. Problem Statement

### 3.1 Latency

Human turn-taking tolerates ~200–500ms silence. S2S systems must minimize cumulative latency across four sequential ML stages. **Solution:** Sentence-level streaming with concurrent LLM+TTS.

### 3.2 Bilingual Code-Switching

Users switch languages mid-conversation or mid-sentence. **Solution:** Dual-layer language detection (ASR tag + text-based keyword analysis) with per-sentence TTS language override and single multilingual voice.

### 3.3 Interrupt Handling

Users interrupt speakers naturally. **Solution:** Continuous VAD during generation with frame-counting interrupt detection and cooperative pipeline termination.

---

## 4. System Architecture

### 4.1 High-Level Diagram

```
┌──────────────────────────────────────────────────────────┐
│                Frontend (React 18 + WebGL)                │
│  FluidOrb │ Transcript │ Controls │ AudioStream (PCM I/O)│
│                    │ WebSocket (binary + JSON)            │
└────────────────────┼─────────────────────────────────────┘
                     │
┌────────────────────┼─────────────────────────────────────┐
│               Backend (FastAPI + asyncio)                  │
│  ┌─────────────────▼───────────────────────────────────┐  │
│  │         PipelineManager (State Machine)             │  │
│  │  VAD → ASR → LLM (streaming) → TTS → Audio Out     │  │
│  │  Silero   Whisper  Llama 70B    Edge Neural         │  │
│  │              │          │                            │  │
│  │         Rate Limiter   LTM (SQLite)                 │  │
│  └─────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────┘
```

### 4.2 Dual-Mode Architecture

| Aspect | Cloud Mode | Local Mode |
|--------|-----------|------------|
| **LLM** | Groq → Llama 3.3 70B | Qwen 2.5 1.5B (HuggingFace) |
| **ASR** | Groq → Whisper Large v3 | faster-whisper small (INT8) |
| **TTS** | Edge Neural TTS | Silero v3 / XTTSv2 |
| **VAD** | Silero v5 (always local) | Silero v5 (always local) |
| **GPU** | Not required | ~4 GB VRAM |
| **Sessions** | Per-session (multi-user) | Shared (single-user) |

### 4.3 Module Dependency Graph

```
config.py → main.py → pipeline/manager.py
                          ├── vad.py        (Silero VAD)
                          ├── asr.py        (local ASR)
                          ├── asr_groq.py   (cloud ASR)
                          ├── llm.py        (local LLM)
                          ├── llm_groq.py   (cloud LLM)
                          ├── tts.py        (Silero TTS)
                          ├── tts_edge.py   (Edge TTS)
                          ├── tts_xtts.py   (XTTSv2)
                          └── memory.py     (SQLite LTM)
```

---

## 5. Data Flow & Protocol

### 5.1 WebSocket Protocol

Single WebSocket at `/ws` carries binary audio + JSON control messages bidirectionally.

**Client → Server:**

| Format | Content |
|--------|---------|
| binary | Int16 PCM, 16 kHz mono, 512-sample frames (~32ms) |
| JSON | `{"type":"clear"}` — reset conversation |
| JSON | `{"type":"chat","text":"..."}` — typed input (bypasses VAD/ASR) |
| JSON | `{"type":"config",...}` — runtime overrides |

**Server → Client:**

| Format | Content |
|--------|---------|
| JSON | `state`, `transcript`, `partial_transcript`, `audio_config`, `audio_end`, `interrupt`, `backchannel`, `language_shift` |
| binary | Int16 PCM at rate from `audio_config` |

### 5.2 Pipeline Data Flow

```
Mic → Float32→Int16 → WebSocket → VAD (512 samples/chunk)
  → accumulate utterances (3.0s silence timeout)
  → ASR (hallucination filter + language detect)
  → LLM stream (sentence boundary detection → asyncio.Queue)
  → TTS worker (per-sentence synthesis, 100ms chunk streaming)
  → WebSocket binary → AudioContext playback
```

### 5.3 State Machine

```
IDLE → (speech) → LISTENING → (ASR) → THINKING → (LLM) → SPEAKING
  ↑                                                          │
  └──────────── (complete) ◄─────────────────────────────────┘
                                    (interrupt) → LISTENING
```

---

## 6. VAD Component

**File:** `pipeline/vad.py` | **Model:** Silero VAD v5 (~2.2 MB JIT)

### 6.1 Processing per 512-sample chunk (32ms)

1. **Type normalize:** Int16 → Float32 [-1, 1]
2. **Energy gate:** RMS < 0.005 → silence (skip model). Reduces CPU ~60%
3. **Silero inference:** `model(tensor, 16000) → probability`
4. **State logic:** Speech confirmed after 250ms consecutive, silence after 600ms
5. **Utterance emission:** Concatenate buffer, validate full-utterance RMS (>1.5× threshold)

### 6.2 Key Design Decisions

- **Energy gate before model:** Eliminates hallucinations from background noise
- **Full-utterance RMS check:** Catches Silero false positives on transients
- **Class-level model singleton:** `_shared_model` loaded once, shared across sessions

### 6.3 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.45 | Silero probability cutoff |
| `min_speech_ms` | 250 | Min speech to confirm start |
| `min_silence_ms` | 600 | Silence to confirm end |
| `energy_threshold` | 0.005 | RMS noise floor |

---

## 7. ASR Component

**File:** `pipeline/asr_groq.py` | **Model:** Whisper Large v3 (1.5B params, Groq API)

### 7.1 Pipeline

1. Float32 numpy → WAV bytes (16-bit PCM, `io.BytesIO` + `wave`)
2. Groq API call with `response_format="verbose_json"`, bilingual prompt hint
3. Text-based language override (§13)
4. Hallucination filtering

### 7.2 Execution Model

Sync API call wrapped in `loop.run_in_executor()` with 2-thread pool. Retry: exponential backoff (1s, 2s) on 429 errors.

### 7.3 Hallucination Filter

30+ curated patterns in `_HALLUCINATIONS` set. Filtering pipeline:
1. Exact match (lowercase, stripped)
2. Prefix match (>3 chars)
3. Repetition detection (3+ identical words)
4. Length gate (<5 chars single word → reject)
5. Duration gate (manager: <0.6s audio → discard before ASR)

### 7.4 Local Fallback

`pipeline/asr.py`: faster-whisper with CTranslate2, `small` model, INT8, beam_size=3.

---

## 8. LLM Component

**File:** `pipeline/llm_groq.py` | **Model:** Llama 3.3 70B Versatile (Groq LPU)

### 8.1 Streaming Architecture

Producer-consumer with `asyncio.Queue`:

```python
def _generate():          # runs in ThreadPoolExecutor
    for chunk in stream:
        token = chunk.choices[0].delta.content
        loop.call_soon_threadsafe(queue.put_nowait, token)

_pool.submit(_generate_wrapper)   # fire-and-forget

while True:                       # async consumer
    token = await queue.get()
    if token is None: break
    yield token
```

Bridges sync Groq SDK → async event loop via `call_soon_threadsafe`.

### 8.2 System Prompt

Persona "Alex" — warm, curious, bilingual. Key rules:
- Natural speech patterns with fillers
- Language mirroring (German in = German out)
- Never identify as AI
- Teacher Mode: cross-lingual word explanations

### 8.3 Teacher Mode Detection

```python
teacher_triggers = ["what does", "was bedeutet", "was heißt", "was ist", ...]
if any(t in lower for t in teacher_triggers):
    # Inject: explain nuance, give examples in OTHER language, cultural context
```

### 8.4 History & Cleanup

- Sliding window: last 30 messages (15 turns)
- Response cleanup: strip quotes, markdown bullets, ensure terminal punctuation, truncate at sentence boundary

### 8.5 Local Fallback

`pipeline/llm.py`: Qwen 2.5 1.5B Instruct (HuggingFace Transformers) or GGUF via llama-cpp-python.

---

## 9. TTS Component

**File:** `pipeline/tts_edge.py` | **Engine:** Microsoft Edge Neural TTS

### 9.1 Voice

`en-US-AndrewMultilingualNeural` — single voice for both EN and DE. Eliminates the "two different people" problem.

### 9.2 Synthesis Pipeline

```
text → _sanitize_text() → prosody modulation → edge_tts stream (MP3)
  → ffmpeg decode (MP3→Int16 PCM 24kHz) → volume normalize (92% peak)
  → fade in/out (10ms) → trailing silence (30-60ms) → bytes
```

### 9.3 Prosody Modulation

| Punctuation | Pitch | Rate |
|-------------|-------|------|
| `?` (question) | +2Hz | -3% |
| `!` (exclamation) | +1Hz | +5% |
| Other | +0Hz | +0% |

### 9.4 MP3→PCM via ffmpeg

```python
proc = subprocess.run(
    [_FFMPEG, "-i", "pipe:0", "-f", "s16le", "-acodec", "pcm_s16le",
     "-ar", "24000", "-ac", "1", "pipe:1"],
    input=mp3_data, capture_output=True)
return np.frombuffer(proc.stdout, dtype=np.int16)
```

ffmpeg binary from `imageio-ffmpeg` (bundled). Runs in ThreadPoolExecutor.

### 9.5 Retry & Filler Cache

- 3 retries with 0.5/1.0/2.0s backoff on timeout
- Pre-cached filler phrases ("Hmm, let me think...", "Moment mal...") for instant playback

### 9.6 Alternative Engines

| Engine | File | Pros | Cons |
|--------|------|------|------|
| Edge Neural | `tts_edge.py` | Free, natural, multilingual | Requires internet |
| Silero v3 | `tts.py` | Offline, CPU-only | Separate EN/DE models |
| XTTSv2 | `tts_xtts.py` | Voice cloning | ~500MB, GPU required |

---

## 10. Long-Term Memory

**File:** `pipeline/memory.py` | **Backend:** SQLite (WAL mode)

### 10.1 Schema

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT DEFAULT 'conversation',
    content TEXT NOT NULL,
    keywords TEXT DEFAULT '',
    language TEXT DEFAULT 'en',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    relevance_count INTEGER DEFAULT 0
);
CREATE TABLE user_prefs (key TEXT PRIMARY KEY, value TEXT, updated_at TIMESTAMP);
```

### 10.2 Keyword Extraction

Bilingual stopword filtering (110+ EN+DE stopwords). Top 20 keywords by frequency via `Counter.most_common()`.

### 10.3 Retrieval

Jaccard similarity on keyword sets. Threshold: 0.05. Retrieved memories get `relevance_count++` for future prioritization.

---

## 11. Pipeline Orchestration

**File:** `pipeline/manager.py` (~800 lines)

### 11.1 Audio Accumulation

Speech fragments buffered in `_audio_buffer[]`. 3.0s silence timeout triggers `_flush_accumulated()`:

```python
self._accumulation_timer = loop.call_later(
    self._accumulation_delay,
    lambda: asyncio.ensure_future(self._flush_accumulated(send))
)
```

All fragments concatenated → single ASR call. Min duration gate (0.6s) rejects noise.

### 11.2 Concurrent LLM + TTS

LLM tokens buffered into sentences. Each sentence pushed to `asyncio.Queue`, consumed by concurrent TTS worker:

```python
tts_queue: asyncio.Queue = asyncio.Queue()
async def _tts_worker():
    while True:
        item = await tts_queue.get()
        if item is None: break
        if self._interrupt.is_set(): continue
        await self._speak_sentence(sent_text, sent_lang, send)
tts_task = asyncio.create_task(_tts_worker())
```

**Sentence boundary heuristics:**
- `.?!` → flush immediately
- `,;:` when buffer >50 chars → flush for pacing
- Soft limit: 5 sentences max per response

### 11.3 Audio Chunk Streaming

TTS output split into 100ms chunks for progressive playback:

```python
chunk_size = int(sample_rate * 2 * 0.1)  # 100ms Int16
for i in range(0, len(pcm_bytes), chunk_size):
    if self._interrupt.is_set(): break
    await send(pcm_bytes[i:i + chunk_size])
```

---

## 12. Interrupt & Backchannel Handling

### 12.1 Frame-Counting Detection

During generation, every audio chunk is VAD-processed:

```python
if is_speaking:
    self._interrupt_speech_frames += 1
    if self._interrupt_speech_frames >= 6:   # ~192ms
        if not self._interrupt.is_set():
            await self._interrupt_generation(send)
else:
    if 0 < frames < 15:  # ~480ms = backchannel
        await send({"type": "backchannel"})
    self._interrupt_speech_frames = 0
```

### 12.2 Cooperative Termination

`_interrupt_generation` uses event flag, NOT task cancellation:

```python
async def _interrupt_generation(self, send):
    self._interrupt.set()          # all loops check this
    self.state = "listening"
    await send({"type": "audio_end"})
    await send({"type": "state", "state": "listening"})
    await send({"type": "interrupt"})
```

**Why not `task.cancel()`?** Cancelling the parent task orphans the internal `tts_task` — it continues synthesizing audio after interrupt. Cooperative flags ensure all nested tasks exit cleanly.

### 12.3 Post-Interrupt Pipeline Exit

```python
if self._interrupt.is_set():
    self._generating = False      # release, keep "listening" state
else:
    await send(transcript)
    await send(audio_end)
    self.state = "idle"
    self._generating = False
```

### 12.4 Buffer Preservation

`_flush_accumulated` defers (500ms retry) instead of clearing buffer when pipeline is busy:

```python
if self._pipeline_lock.locked() or self._generating:
    self._accumulation_timer = loop.call_later(0.5, retry)
    return  # DON'T clear buffer
```

---

## 13. Language Detection & Code-Switching

### 13.1 Dual-Layer Detection

**Layer 1:** Whisper API language tag  
**Layer 2:** Text-based German keyword detection (80+ common words + äöüß chars)

Override: if text-based disagrees with API → text-based wins. Handles Whisper misidentifying short German phrases.

### 13.2 Per-Sentence TTS Language

Each sentence individually detected before TTS, handling mixed-language LLM responses.

### 13.3 Language Shift Events

Tracks last 5 languages. Shift event when current differs from last 3 consecutive.

---

## 14. Concurrency Model

### 14.1 Race Conditions Mitigated

| Race Condition | Mitigation |
|----------------|-----------|
| Concurrent pipelines | `asyncio.Lock` |
| Orphaned TTS worker | Cooperative event flag (no cancel) |
| Buffer lost on flush | Deferred retry (500ms) |
| Double interrupt | `if not self._interrupt.is_set()` |
| Send after disconnect | try/except wrapper |

### 14.2 Thread Safety

- SQLite: WAL mode + 5s busy timeout
- ThreadPoolExecutor workers communicate via `call_soon_threadsafe()` + `asyncio.Queue`
- No shared mutable state between threads

---

## 15. Frontend Architecture

### 15.1 Stack

React 18, Vite 5, Three.js (React Three Fiber), TailwindCSS, Framer Motion, Lucide icons.

### 15.2 Components

```
App.jsx → WelcomeIntro, FluidOrb (GLSL shaders), TranscriptWindow,
          SubtitleBar, ChatInput, VoiceWaveform, PipelineVisualizer,
          SettingsPanel, ConnectionStatus, Toast
hooks/  → useWebSocket.js, useAudioStream.js
```

### 15.3 FluidOrb Shader

- Custom GLSL vertex (noise displacement) + fragment (color) shaders
- `uLanguageBlend`: 0=EN (blue) → 1=DE (amber), lava-lamp interpolation
- `uEnergy`/`uBass`/`uMid`/`uTreble`: FFT-driven
- State-reactive: idle=calm, listening=energetic, thinking=rapid, speaking=flowing
- Bloom post-processing

---

## 16. Audio I/O Subsystem

### 16.1 Capture

`ScriptProcessor` at 16 kHz mono. Float32→Int16 conversion. Echo cancellation + noise suppression + auto gain.

### 16.2 Playback

`AudioContext` with scheduled `BufferSource` nodes. Gapless: each chunk starts at `nextPlayTime`.

### 16.3 Interrupt

`AudioContext.close()` — immediately stops ALL scheduled sources.

### 16.4 FFT

128-bin analysis (bass/mid/treble split). Exposed as `window.__fft*` globals for shader consumption at 60fps.

---

## 17. Rate Limiting

Backend: 25 RPM limit (below Groq's 30). Sliding window of timestamps.

Per-provider retry: ASR (2 retries, 1-2s), LLM (2 retries, 1-2s), TTS (3 retries, 0.5-2s).

---

## 18. Configuration

**File:** `backend/config.py` — single `@dataclass` with environment variable overrides.

| Variable | Default | Description |
|----------|---------|-------------|
| `S2S_MODE` | `cloud` | `cloud` or `local` |
| `GROQ_API_KEY` | — | Required for cloud |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | LLM identifier |
| `ASR_MODEL` | `whisper-large-v3` | ASR identifier |
| `LLM_MAX_TOKENS` | `350` | Max response tokens |
| `LLM_TEMPERATURE` | `0.85` | Sampling temperature |
| `TTS_ENGINE` | `edge` | `edge`, `silero`, `xtts` |

Runtime per-session overrides via WebSocket JSON.

---

## 19. Deployment

### 19.1 Docker (Multi-Stage)

```dockerfile
FROM python:3.11-slim AS base     # Python + ffmpeg
FROM node:20-slim AS frontend     # npm build
FROM base                         # Final: backend + built frontend + silero_vad.jit
```

Image: ~2 GB. CPU-only PyTorch.

### 19.2 HuggingFace Spaces

- Docker SDK, port 7860
- `GROQ_API_KEY` as Space Secret
- Build: ~10 min
- Live: https://ilyass1-starch.hf.space

### 19.3 Deploy Script

`scripts/deploy_hf.py`: Stages filtered files → `upload_folder()` via HuggingFace Hub API.

---

## 20. Performance Analysis

### 20.1 Latency Breakdown (Cloud)

| Stage | Latency |
|-------|---------|
| VAD per chunk | <1ms |
| Accumulation timeout | 3.0s (configurable) |
| ASR (Groq Whisper) | 0.2–0.5s |
| LLM first token | 0.3–0.8s |
| TTS first sentence | 0.8–1.5s |
| **E2E (speech end → first audio)** | **1.5–2.5s** |

### 20.2 Resource Usage

| Resource | Cloud | Local |
|----------|-------|-------|
| CPU | ~10% idle, ~30% active | ~50-80% |
| RAM | ~500 MB | ~4-6 GB |
| GPU | None | ~4 GB VRAM |

---

## 21. Limitations

1. **Accumulation delay** (3.0s) dominates latency
2. **Edge TTS** requires internet; Silero fallback lower quality
3. **Groq rate limits** (30 RPM free tier)
4. **Short phrase language misidentification** by Whisper
5. **No streaming TTS** — full sentence synthesized before streaming
6. **ScriptProcessor** deprecated (AudioWorklet requires served worker file)
7. **LTM recall disabled** — 70B model confused by injected context
8. **Ephemeral HF storage** — LTM lost on restart

---

## 22. Future Work

1. Additional languages (French, Spanish, Arabic)
2. Speaker diarization (multi-user)
3. Emotion detection from voice prosody
4. Streaming TTS (ElevenLabs, Cartesia)
5. AudioWorklet migration
6. RAG with document upload
7. Voice cloning (XTTSv2)
8. On-device deployment (ONNX, TensorRT)
9. Persistent cloud database for LTM
10. Automated evaluation (WER, BLEU, MOS)

---

## 23. Conclusion

This project demonstrates that real-time, bilingual speech-to-speech conversational AI is achievable with modern cloud APIs and careful pipeline engineering. Key contributions:

1. **Sentence-level streaming** overlapping LLM generation with TTS synthesis → sub-2.5s E2E latency
2. **Cooperative interrupt handling** via event flags preventing orphaned tasks and audio leaks
3. **Dual-layer language detection** combining ASR API tags with text-based keyword analysis for robust bilingual routing
4. **Audio accumulation with deferred flush** preserving user speech during pipeline transitions
5. **Single multilingual voice** solving the dual-voice identity problem
6. **Production deployment** as accessible web application with 3D visualization

The system bridges the gap between text-based chatbots and natural human conversation.

---

## Appendices

### Appendix A — Project Structure

```
├── backend/
│   ├── main.py              # FastAPI + WebSocket server
│   ├── config.py            # Centralized configuration
│   ├── __init__.py
│   └── pipeline/
│       ├── manager.py       # Pipeline orchestrator (~800 lines)
│       ├── vad.py           # Silero VAD
│       ├── asr.py           # Local ASR (faster-whisper)
│       ├── asr_groq.py      # Cloud ASR (Groq Whisper)
│       ├── llm.py           # Local LLM (Qwen 2.5)
│       ├── llm_groq.py      # Cloud LLM (Groq Llama)
│       ├── tts.py           # Silero TTS
│       ├── tts_edge.py      # Edge Neural TTS
│       ├── tts_xtts.py      # XTTSv2
│       └── memory.py        # SQLite LTM
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── hooks/           # useWebSocket, useAudioStream
│   │   └── components/      # 15+ UI components
│   └── package.json
├── models/                  # Weights (gitignored, ~4 GB)
├── scripts/
│   ├── deploy_hf.py
│   └── download_models.py
├── Dockerfile
├── requirements.txt
└── README.md
```

### Appendix B — Dependencies

```
# Core
torch>=2.1.0, numpy>=1.24.0, python-dotenv>=1.0.0

# Server
fastapi>=0.104.0, uvicorn[standard]>=0.24.0, websockets>=12.0

# Cloud APIs
groq>=0.4.0

# Local ASR
faster-whisper>=1.0.0, openai-whisper>=20231117

# Local LLM
transformers>=4.36.0, accelerate>=0.25.0, sentencepiece>=0.1.99

# TTS/VAD
edge-tts>=7.0.0, imageio-ffmpeg>=0.5.1

# Frontend
react@18, vite@5, three.js, @react-three/fiber, tailwindcss@3, framer-motion
```

### Appendix C — Model Specifications

| Component | Model | Parameters | Quantization | Mode |
|-----------|-------|-----------|-------------|------|
| VAD | Silero VAD v5 | ~2M | JIT | Always local |
| ASR | Whisper Large v3 | 1.5B | — | Cloud (Groq) |
| ASR | faster-whisper small | 244M | INT8 | Local |
| LLM | Llama 3.3 70B | 70B | — | Cloud (Groq) |
| LLM | Qwen 2.5 1.5B | 1.5B | FP16 | Local |
| TTS | Edge Neural (Andrew) | — | — | Cloud (Microsoft) |
| TTS | Silero v3 EN + DE | ~10M each | JIT | Local |
| TTS | XTTSv2 | ~500M | FP32 | Local (optional) |
