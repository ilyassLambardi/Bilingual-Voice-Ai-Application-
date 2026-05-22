"""
Backend server for real-time bilingual Speech-to-Speech.

FastAPI + WebSocket server with streaming VAD -> ASR -> LLM -> TTS pipeline.

Architecture:
    pipeline/
        cloud/          Cloud API pipeline (Groq ASR/LLM, Edge TTS)
            manager.py, io_handler.py, memory.py, asr_groq.py, llm_groq.py, tts_edge.py, ...
        local/          Local on-device pipeline (VAD, Whisper, GGUF LLM, Silero TTS)
            manager.py, io_handler.py, memory.py, vad.py, asr.py, llm.py, tts.py, ...

Entry point: main.py
"""
