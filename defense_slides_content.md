# Defense Presentation — 7-10 minutes
# Structure: Background → Design → Results → Live Demo Video

---

# Slide 1 — Title (30s)

## Design and Implementation of An On-Device Bilingual AI Voice Assistant System for English and German Based on Deep Learning

**Student:** Ilyass Lambardi (202239060075)  
**Supervisor:** Jianbo Wang  
**School:** Computer Science and Software Engineering, Southwest Petroleum University  
**Date:** May 2026

---

# Slide 2 — Background & Motivation (1.5 min)

## Research Background
- Voice AI market growing rapidly (Siri, Alexa, ChatGPT Voice) — but all cloud-dependent
- Bilingual users (60%+ of world population) forced to use separate monolingual systems
- No existing open-source solution for real-time bilingual speech-to-speech with code-switching

## Problems Addressed
| Problem | Impact |
|---------|--------|
| Cloud-only architecture | High latency, privacy risk, no offline use |
| Monolingual per session | Users must manually switch language settings |
| No mid-conversation switching | Unnatural for bilingual speakers |
| Full-response TTS | User waits for entire LLM output before hearing anything |

## Research Goal
Design a **streaming bilingual voice AI** with dual-mode (cloud + local), supporting natural EN/DE code-switching, with sub-2s perceived latency.

---

# Slide 3 — System Design: Overall Architecture (1 min)

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Frontend (React + WebGL)                │
│   3D Fluid Orb  |  Transcript  |  Controls  |  Audio    │
└──────────────────────────┬──────────────────────────────┘
                           │ WebSocket (PCM + JSON)
┌──────────────────────────┼──────────────────────────────┐
│                   Backend (FastAPI)                       │
│                          │                               │
│   ┌──────────────────────▼─────────────────────────┐    │
│   │         Pipeline Manager (State Machine)        │    │
│   │                                                 │    │
│   │  [VAD] → [ASR] → [LLM] → [TTS] → Audio Out    │    │
│   │  Silero   Whisper  Llama/   Edge/               │    │
│   │           Large v3 Qwen     Silero              │    │
│   │                      ↕                          │    │
│   │              [Long-Term Memory]                  │    │
│   └─────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

**Dual-mode:** Cloud (Groq API — best quality) | Local (on-device GPU — private, offline)

---

# Slide 4 — System Design: Streaming Strategy (1 min)

## Sentence-Level Streaming (Key Innovation)

Traditional approach:
```
User speaks → [wait ASR] → [wait full LLM] → [wait full TTS] → play audio
                                              Total wait: 4-8 seconds
```

Our approach:
```
User speaks → [ASR] → LLM streams tokens...
                       ├─ Sentence 1 complete → TTS → play immediately
                       ├─ Sentence 2 complete → TTS → queue
                       └─ Sentence 3 complete → TTS → queue
                       Perceived wait: ~1.5-2s (time to first sentence only)
```

## Interrupt Mechanism
- VAD monitors mic continuously during AI speech
- 4+ speech frames (~128ms) = interrupt → cancel generation + stop audio
- Short sounds (<384ms) classified as backchannel ("mhm") → ignored

---

# Slide 5 — System Design: Core Modules (1.5 min)

## Module Details

| Module | Cloud Mode | Local Mode | Function |
|--------|-----------|------------|----------|
| **VAD** | Silero v5 (local) | Silero v5 (local) | Speech boundary detection |
| **ASR** | Groq Whisper Large v3 (1.5B) | faster-whisper small (244M) | Bilingual transcription |
| **LLM** | Llama 3.3 70B via Groq | Qwen 2.5 1.5B on-device | Response generation |
| **TTS** | Edge Neural TTS | Silero v3 dual-voice | Speech synthesis |
| **Memory** | SQLite per-session | SQLite shared | Cross-session recall |

## Bilingual Language Intelligence
- **Intent-based detection** — structural analysis overrides ASR language tags
- **Language mirroring** — responds in the language the user speaks
- **Hallucination filter** — pattern matching + entropy scoring rejects noise transcripts

---

# Slide 6 — System Design: State Machine & Concurrency (1 min)

## Pipeline State Machine
```
     ┌──────────────────────────────────────┐
     │                                      │
     ▼                                      │
   [IDLE] ──speech detected──→ [LISTENING]  │
                                    │       │
                              silence 700ms  │
                                    │       │
                                    ▼       │
                              [THINKING]    │
                                    │       │
                              first audio   │
                                    │       │
                                    ▼       │
                              [SPEAKING] ───┘
                                    │
                              interrupt ──→ [IDLE]
```

## Concurrency Design
- `asyncio.Lock()` — prevents overlapping pipeline runs
- `asyncio.Event` — signals interrupt across tasks
- Per-session managers (cloud) vs shared manager (local)
- Cleanup on disconnect: cancel tasks, close SQLite, clear buffers

---

# Slide 7 — Results: Latency (1 min)

## End-to-End Latency (Cloud Mode)

| Pipeline Stage | Latency |
|----------------|---------|
| VAD (speech end detection) | ~700ms |
| ASR (Groq Whisper) | 300-500ms |
| LLM first token (Groq) | 200-400ms |
| TTS first sentence (Edge) | 400-600ms |
| **Perceived total** | **1.5-2.5s** |

## Comparison with Traditional Approach

| Metric | Traditional (full-response) | Our System (streaming) |
|--------|----------------------------|----------------------|
| Time to first audio | 4-8s | **1.5-2.5s** |
| Interrupt response | Not supported | **~128ms** |
| Language switch | Manual toggle | **Automatic** |

**~40% reduction** in perceived latency via sentence-level streaming overlap.

---

# Slide 8 — Results: Bilingual Performance (1 min)

## Language Detection & Switching

| Test Case | Accuracy |
|-----------|----------|
| Pure English input | >98% |
| Pure German input | >95% |
| Mid-sentence code-switch | >90% |
| Backchannel rejection | >95% |
| Hallucination filter (noise rejection) | >90% |

## Qualitative Results
- Natural EN↔DE switching without manual selection
- Teacher Mode correctly triggered by "What does X mean?"
- Denglisch (mixed) input handled gracefully — responds in dominant language
- Long-term memory successfully recalls topics from previous sessions

---

# Slide 9 — Limitations & Future Work (30s)

## Current Limitations
- Cloud mode needs internet; local mode trades quality for privacy
- Only EN + DE supported (adding languages requires TTS voices + prompt tuning)
- Local mode: single-user only (GPU not shareable)
- No speaker diarization (single speaker assumed)

## Future Work
- Extend to Chinese, French, Arabic
- Better local LLM (Phi-3, Gemma 2)
- Voice cloning with XTTSv2
- Mobile deployment (ONNX runtime)
- RAG integration for factual grounding

---

# Slide 10 — System Demo (1 min video)

## Live Demonstration

> **[Play 1-minute screen recording here]**

Demo shows:
1. User speaks English → AI responds in English with natural voice
2. User switches to German mid-conversation → AI mirrors to German
3. User asks "What does Fernweh mean?" → Teacher Mode activates
4. User interrupts AI mid-speech → generation stops immediately
5. 3D orb visualization reacts to pipeline state + language color

---

**Thank you! Questions?**
