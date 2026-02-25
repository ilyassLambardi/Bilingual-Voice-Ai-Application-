import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

export default function ConnectionIndicator({ connected }) {
  const [latency, setLatency] = useState(null);
  const [quality, setQuality] = useState("good"); // good, fair, poor
  const pingRef = useRef(null);

  // Simulate latency measurement via periodic pings
  useEffect(() => {
    if (!connected) { setLatency(null); setQuality("poor"); return; }

    const measure = () => {
      const start = performance.now();
      // Use a tiny fetch to measure round-trip
      fetch("/health", { cache: "no-store" })
        .then(() => {
          const ms = Math.round(performance.now() - start);
          setLatency(ms);
          setQuality(ms < 150 ? "good" : ms < 400 ? "fair" : "poor");
        })
        .catch(() => {
          setLatency(null);
          setQuality("poor");
        });
    };

    measure();
    pingRef.current = setInterval(measure, 10000);
    return () => clearInterval(pingRef.current);
  }, [connected]);

  const colors = {
    good: { dot: "#10b981", bg: "rgba(16,185,129,0.08)", border: "rgba(16,185,129,0.15)", text: "rgba(16,185,129,0.6)" },
    fair: { dot: "#f59e0b", bg: "rgba(245,158,11,0.08)", border: "rgba(245,158,11,0.15)", text: "rgba(245,158,11,0.6)" },
    poor: { dot: "#ef4444", bg: "rgba(239,68,68,0.08)", border: "rgba(239,68,68,0.15)", text: "rgba(239,68,68,0.6)" },
  };

  const c = connected ? colors[quality] : colors.poor;
  const bars = quality === "good" ? 3 : quality === "fair" ? 2 : 1;

  return (
    <div className="flex items-center gap-2">
      {/* Signal bars */}
      <div className="flex items-end" style={{ gap: 1.5, height: 12 }}>
        {[1, 2, 3].map(i => (
          <motion.div
            key={i}
            animate={{
              background: i <= bars && connected ? c.dot : "rgba(255,255,255,0.06)",
              height: 4 + i * 3,
            }}
            transition={{ duration: 0.4 }}
            style={{
              width: 2.5, borderRadius: 1,
            }}
          />
        ))}
      </div>

      {/* Latency badge */}
      <AnimatePresence mode="wait">
        {connected && latency !== null && (
          <motion.span
            key={latency}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.2 }}
            style={{
              fontSize: 8, fontWeight: 600, letterSpacing: "0.04em",
              color: c.text,
              fontFamily: "'Space Grotesk', monospace",
            }}
          >
            {latency}ms
          </motion.span>
        )}
      </AnimatePresence>
    </div>
  );
}
