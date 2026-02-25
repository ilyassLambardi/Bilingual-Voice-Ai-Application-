import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

/**
 * Shows real-time response latency breakdown when the AI responds.
 * Tracks thinking → first-audio time as a single metric.
 */
export default function ResponseLatency({ pipelineState, theme = "dark" }) {
  const isDark = theme === "dark";
  const [latency, setLatency] = useState(null);
  const thinkStart = useRef(null);
  const shown = useRef(false);

  useEffect(() => {
    if (pipelineState === "thinking" && !thinkStart.current) {
      thinkStart.current = Date.now();
      shown.current = false;
    }
    if (pipelineState === "speaking" && thinkStart.current && !shown.current) {
      const ms = Date.now() - thinkStart.current;
      setLatency(ms);
      shown.current = true;
    }
    if (pipelineState === "idle") {
      thinkStart.current = null;
    }
  }, [pipelineState]);

  return (
    <AnimatePresence>
      {latency !== null && pipelineState === "speaking" && (
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.3 }}
          className="flex items-center gap-1.5"
          style={{
            fontSize: 9,
            fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
            color: latency < 2000
              ? "rgba(16,185,129,0.45)"
              : latency < 4000
                ? "rgba(245,158,11,0.45)"
                : "rgba(239,68,68,0.4)",
          }}
        >
          <svg width="9" height="9" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="13 2 3 14 12 14 11 22 21 10 12 10" />
          </svg>
          {(latency / 1000).toFixed(1)}s
        </motion.div>
      )}
    </AnimatePresence>
  );
}
