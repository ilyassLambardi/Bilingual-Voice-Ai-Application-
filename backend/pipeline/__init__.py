"""

Real-time Speech-to-Speech pipeline — modular architecture.



Directory structure:

    pipeline/

        cloud/                  Cloud API pipeline (Groq + Edge TTS)

            manager.py          Pipeline orchestrator (cloud mode)

            io_handler.py       WebSocket I/O protocol layer

            memory.py           SQLite long-term memory

            asr_groq.py         Groq Whisper API ASR

            llm_groq.py         Groq LLM chat completions

            tts_edge.py         Microsoft Edge Neural TTS

            hallucination_filter.py  ASR hallucination detection

            llm_common.py       LLM response cleanup + context hints

            tts_common.py       TTS text sanitization + audio fades

            lang_detect.py      Text-based EN/DE language detection

        local/                  Local on-device pipeline

            manager.py          Pipeline orchestrator (local mode)

            io_handler.py       WebSocket I/O protocol layer

            memory.py           SQLite long-term memory

            vad.py              Silero Voice Activity Detection

            asr.py              Local Whisper / faster-whisper ASR

            llm.py              GGUF (llama-cpp) / HuggingFace LLM

            tts.py              Silero TTS

            tts_xtts.py         Coqui XTTSv2 TTS

            hallucination_filter.py  ASR hallucination detection

            llm_common.py       LLM response cleanup + context hints

            tts_common.py       TTS text sanitization + audio fades

            lang_detect.py      Text-based EN/DE language detection

            paths.py            Central path definitions



    Each subpackage is FULLY self-contained with its own manager,

    io_handler, memory, and processing modules.



    cloud/ depends on local/ only for: VAD (always local, lightweight)

    and fallback ASR/LLM/TTS when cloud APIs fail.



Data flow:  Mic -> io_handler -> manager -> ASR -> LLM -> TTS -> io_handler -> Speaker

State:      manager reads/writes memory throughout the pipeline

"""

