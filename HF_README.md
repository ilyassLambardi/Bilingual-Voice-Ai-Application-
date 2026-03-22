---
title: Bilingual Voice AI
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
---

# Bilingual Speech-to-Speech Voice AI

A real-time bilingual (English + German) voice conversation AI. Speak naturally and get spoken responses.

## Features
- **Bilingual**: Speaks English and German, switches automatically
- **Real-time**: ~2s response time
- **Natural**: Debates, explains, asks follow-up questions
- **Interruptible**: Speak while the AI is talking to interrupt

## Tech Stack
- **ASR**: Whisper large-v3 via Groq API
- **LLM**: Llama-3.3-70B via Groq API  
- **TTS**: Microsoft Edge Neural TTS (English + German)
- **VAD**: Silero VAD
- **Backend**: FastAPI + WebSocket
- **Frontend**: React + Vite + TailwindCSS
