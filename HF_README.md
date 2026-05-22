---
title: Bilingual Voice AI
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# Bilingual Speech-to-Speech Voice AI

Real-time bilingual (English/German) voice assistant with streaming pipeline.

## Features
- **Dual-mode**: Cloud (Groq Llama 3.3 70B + Whisper) or Local (Qwen 2.5 + faster-whisper)
- **Native code-switching**: Speak in EN or DE, the AI mirrors your language
- **Streaming**: Sentence-level TTS for sub-2s latency
- **Interruptible**: Speak to cut off the AI mid-response
- **3D Visualization**: WebGL fluid orb reacts to pipeline state

## Usage
1. Click the microphone or hold Spacebar
2. Speak in English or German
3. The AI responds in your language

## Configuration
Set `GROQ_API_KEY` in Space Settings → Secrets.
