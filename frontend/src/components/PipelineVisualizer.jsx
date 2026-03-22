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
              {/* Active pulse + shimmer */}
              {active && (
                <>
                  <motion.div
                    className="absolute inset-0"
                    animate={{ opacity: [0.05, 0.15, 0.05] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                    style={{ background: `rgba(${stage.c}, 0.1)` }}
                  />
                  <div className="absolute inset-0" style={{
                    background: `linear-gradient(105deg, transparent 35%, rgba(${stage.c}, 0.08) 50%, transparent 65%)`,
                    animation: "shimmer-sweep 2s ease-in-out infinite",
                  }} />
                </>
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

            {/* Connector line with flow dot */}
            {i < STAGES.length - 1 && (
              <div className="relative" style={{ width: 10, height: 2 }}>
                <motion.div
                  animate={{
                    background: highlight
                      ? `linear-gradient(to right, rgba(${stage.c}, 0.3), rgba(${STAGES[i+1].c}, 0.3))`
                      : isDark ? "rgba(255,255,255,0.04)" : "rgba(255,255,255,0.06)",
                  }}
                  transition={{ duration: 0.3 }}
                  style={{ position: "absolute", inset: 0, borderRadius: 1 }}
                />
                {active && (
                  <motion.div
                    className="absolute rounded-full"
                    animate={{ left: ["-20%", "120%"] }}
                    transition={{ duration: 0.8, repeat: Infinity, ease: "linear" }}
                    style={{ top: -1, width: 4, height: 4, background: `rgba(${stage.c}, 0.6)`,
                      boxShadow: `0 0 6px rgba(${stage.c}, 0.4)` }}
                  />
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
