import { motion, AnimatePresence } from "framer-motion";
import { Mic, MicOff, Trash2, Power } from "lucide-react";

const ACCENTS = {
  idle: "#06b6d4", listening: "#f472b6", thinking: "#f59e0b", speaking: "#10b981",
};

export default function Controls({
  connected, micActive, pipelineState, onToggleMic, onClear, onConnect, theme = "dark",
}) {
  const accent = ACCENTS[pipelineState] || ACCENTS.idle;
  const isDark = theme === "dark";

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="flex items-center gap-3 px-3 py-2 rounded-2xl" style={{
        background: isDark ? "rgba(255,255,255,0.025)" : "rgba(255,255,255,0.035)",
        border: `1px solid ${micActive ? `${accent}25` : (isDark ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.07)")}`,
        backdropFilter: "blur(24px)",
        WebkitBackdropFilter: "blur(24px)",
        boxShadow: micActive ? `0 0 30px ${accent}08, 0 4px 20px rgba(0,0,0,0.15)` : "0 4px 20px rgba(0,0,0,0.1)",
        transition: "all 0.6s ease",
      }}>
        <motion.button
          whileTap={{ scale: 0.9 }}
          whileHover={{ scale: 1.08 }}
          onClick={onConnect}
          className="flex items-center justify-center rounded-xl"
          style={{
            width: 40, height: 40,
            background: connected
              ? "rgba(16,185,129,0.08)"
              : isDark ? "rgba(255,255,255,0.03)" : "rgba(255,255,255,0.04)",
            border: connected
              ? "1px solid rgba(16,185,129,0.15)"
              : isDark ? "1px solid rgba(255,255,255,0.06)" : "1px solid rgba(255,255,255,0.07)",
            color: connected ? "rgba(16,185,129,0.7)" : (isDark ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.3)"),
            transition: "all 0.4s",
            cursor: "pointer",
          }}
        >
          <Power size={16} />
        </motion.button>

        <div className="relative flex items-center justify-center" style={{ width: 64, height: 64 }}>
          {micActive && (
            <div className="absolute rounded-full" style={{
              width: 64, height: 64,
              boxShadow: `0 0 24px ${accent}40, 0 0 48px ${accent}18`,
              animation: "glow-breathe 2s ease-in-out infinite",
            }} />
          )}
          {micActive && [0, 0.6].map((d, i) => (
            <div key={i} className="absolute rounded-full" style={{
              width: 56, height: 56,
              border: `1px solid ${accent}25`,
              animation: `pulse-ring 2.2s ease-out ${d}s infinite`,
            }} />
          ))}

          <motion.button
            whileTap={{ scale: 0.88 }}
            whileHover={{ scale: 1.08, boxShadow: `0 0 24px ${accent}30` }}
            onClick={onToggleMic}
            disabled={!connected}
            className="relative rounded-full flex items-center justify-center"
            style={{
              width: 56, height: 56, zIndex: 2,
              background: micActive
                ? `linear-gradient(135deg, ${accent}18, ${accent}0c)`
                : isDark ? "rgba(255,255,255,0.03)" : "rgba(255,255,255,0.04)",
              border: micActive
                ? `1.5px solid ${accent}44`
                : isDark ? "1.5px solid rgba(255,255,255,0.07)" : "1.5px solid rgba(255,255,255,0.09)",
              color: micActive ? "#fff" : (isDark ? "rgba(255,255,255,0.35)" : "rgba(255,255,255,0.4)"),
              boxShadow: micActive ? `0 0 20px ${accent}20, inset 0 0 20px ${accent}08` : "none",
              transition: "all 0.5s ease",
              opacity: connected ? 1 : 0.2,
              cursor: connected ? "pointer" : "not-allowed",
            }}
          >
            <AnimatePresence mode="wait">
              <motion.div
                key={micActive ? "on" : "off"}
                initial={{ scale: 0.5, opacity: 0, rotate: -60 }}
                animate={{ scale: 1, opacity: 1, rotate: 0 }}
                exit={{ scale: 0.5, opacity: 0, rotate: 60 }}
                transition={{ duration: 0.18 }}
              >
                {micActive ? <Mic size={22} /> : <MicOff size={22} />}
              </motion.div>
            </AnimatePresence>
          </motion.button>
        </div>

        <motion.button
          whileTap={{ scale: 0.9 }}
          whileHover={{ scale: 1.08 }}
          onClick={onClear}
          className="flex items-center justify-center rounded-xl"
          style={{
            width: 40, height: 40,
            background: isDark ? "rgba(255,255,255,0.03)" : "rgba(255,255,255,0.04)",
            border: isDark ? "1px solid rgba(255,255,255,0.06)" : "1px solid rgba(255,255,255,0.07)",
            color: isDark ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.3)",
            cursor: "pointer",
            transition: "all 0.3s",
          }}
        >
          <Trash2 size={15} />
        </motion.button>
      </div>
    </div>
  );
}
