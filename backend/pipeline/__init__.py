"""
Real-time Speech-to-Speech pipeline — modular architecture.

Four architectural modules:

    Module 1 — Input/Output (io_handler.py)
        WebSocket message parsing, protocol definitions, outbound
        message builders, safe async send wrapper.

    Module 2 — Processing / Logic Core (vad.py, asr.py, asr_groq.py,
               llm.py, llm_groq.py, tts.py, tts_edge.py, tts_xtts.py)
        Stateless processing workers: Voice Activity Detection,
        Automatic Speech Recognition, Large Language Model inference,
        and Text-to-Speech synthesis.

    Module 3 — Data Storage / State Management (session_state.py, memory.py)
        SessionState: pipeline state machine, audio buffers, interrupt
        flags, language history, rate limiting.
        LongTermMemory: SQLite-backed persistent conversational memory
        with keyword-based retrieval.

    Module 4 — Control Flow / Scheduling (manager.py)
        PipelineManager: orchestrates VAD → ASR → LLM → TTS,
        handles interrupt scheduling, audio accumulation timers,
        concurrent TTS workers, and pipeline sequencing.

Data flow:  Mic → Module 1 → Module 4 → Module 2 → Module 4 → Module 1 → Speaker
State:      Module 4 reads/writes Module 3 throughout the pipeline
"""
