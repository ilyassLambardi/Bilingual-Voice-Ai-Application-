"""
Backend server for real-time bilingual Speech-to-Speech.

FastAPI + WebSocket server with streaming VAD → ASR → LLM → TTS pipeline.

Architecture (4 modules):
    1. Input/Output       — pipeline/io_handler.py
    2. Processing (Core)  — pipeline/vad.py, asr*.py, llm*.py, tts*.py
    3. Data Storage/State — pipeline/session_state.py, pipeline/memory.py
    4. Control Flow       — pipeline/manager.py

Entry point: main.py
"""
