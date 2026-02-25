import { motion } from "framer-motion";

const STAGES = [
  { id: "vad", label: "VAD", desc: "Voice Detection", c: "16,185,129", states: ["listening"] },
  { id: "asr", label: "ASR", desc: "Speech Recognition", c: "6,182,212", states: ["thinking"] },
  { id: "llm", label: "LLM", desc: "Language Model", c: "168,85,247", states: ["thinking", "speaking"] },
  { id: "tts", label: "TTS", desc: "Text to Speech", c: "244,114,182", states: ["speaking"] },
];

export default function PipelineVisualizer({ pipelineState, theme = "dark" }) {
  const isDark = theme === "dark";

  return (
    <div className="flex items-center" style={{ gap: 3 }}>
      {STAGES.map((stage, i) => {
        const active = stage.states.includes(pipelineState);
        const pastActive = (() => {
          const stateOrder = ["idle", "listening", "thinking", "speaking"];
          const currentIdx = stateOrder.indexOf(pipelineState);
          const stageMinIdx = Math.min(...stage.states.map(s => stateOrder.indexOf(s)));
          return currentIdx > stageMinIdx;
        })();
        const highlight = active || pastActive;

        return (
          <div key={stage.id} className="flex items-center" style={{ gap: 3 }}>
            <motion.div
              animate={{
                background: active
                  ? `rgba(${stage.c}, 0.12)`
                  : highlight
                    ? `rgba(${stage.c}, 0.06)`
                    : isDark ? "rgba(255,255,255,0.02)" : "rgba(255,255,255,0.03)",
                borderColor: active
                  ? `rgba(${stage.c}, 0.3)`
                  : highlight
                    ? `rgba(${stage.c}, 0.1)`
                    : isDark ? "rgba(255,255,255,0.04)" : "rgba(255,255,255,0.06)",
                scale: active ? 1.05 : 1,
              }}
              transition={{ duration: 0.3 }}
              className="flex flex-col items-center"
              style={{
                padding: "4px 8px",
                borderRadius: 8,
                border: "1px solid",
                minWidth: 38,
                position: "relative",
                overflow: "hidden",
              }}
            >
              {/* Active pulse */}
              {active && (
                <motion.div
                  className="absolute inset-0"
                  animate={{ opacity: [0.05, 0.12, 0.05] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                  style={{ background: `rgba(${stage.c}, 0.1)` }}
                />
              )}
              <motion.span
                animate={{
                  color: active
                    ? `rgba(${stage.c}, 0.9)`
                    : highlight
                      ? `rgba(${stage.c}, 0.5)`
                      : isDark ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.2)",
                }}
                style={{
                  fontSize: 8, fontWeight: 700, letterSpacing: "0.06em",
                  fontFamily: "'Space Grotesk', monospace",
                  position: "relative", zIndex: 1,
                }}
              >
                {stage.label}
              </motion.span>
            </motion.div>

            {/* Connector line */}
            {i < STAGES.length - 1 && (
              <motion.div
                animate={{
                  background: highlight
                    ? `rgba(${stage.c}, 0.25)`
                    : isDark ? "rgba(255,255,255,0.04)" : "rgba(255,255,255,0.06)",
                }}
                transition={{ duration: 0.3 }}
                style={{ width: 8, height: 1.5, borderRadius: 1 }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
