import { motion } from "framer-motion";
import { useMemo } from "react";

const COLORS = {
  idle: { primary: "6,182,212", secondary: "129,140,248" },
  listening: { primary: "244,114,182", secondary: "168,85,247" },
  thinking: { primary: "245,158,11", secondary: "239,68,68" },
  speaking: { primary: "16,185,129", secondary: "6,182,212" },
};

export default function VoiceWaveform({ state = "idle", active = false }) {
  const c = COLORS[state] || COLORS.idle;
  const barCount = 32;

  const bars = useMemo(() =>
    Array.from({ length: barCount }, (_, i) => ({
      id: i,
      baseH: 2 + Math.random() * 3,
      maxH: state === "listening" ? 12 + Math.random() * 22 : 6 + Math.random() * 16,
      dur: 0.25 + Math.random() * 0.35,
      delay: i * 0.015,
    })),
    [state]
  );

  return (
    <motion.div
      className="flex items-end justify-center"
      style={{ gap: 2, height: 36, overflow: "hidden",
        filter: active ? `drop-shadow(0 0 4px rgba(${c.primary}, 0.2))` : "none",
        transition: "filter 0.5s ease" }}
      initial={{ opacity: 0 }}
      animate={{ opacity: active ? 1 : 0.15 }}
      transition={{ duration: 0.4 }}
    >
      {bars.map((b) => (
        <motion.div
          key={b.id}
          className="rounded-full"
          style={{
            width: 2,
            background: active
              ? `linear-gradient(to top, rgba(${c.primary}, 0.5), rgba(${c.secondary}, 0.35))`
              : `rgba(${c.primary}, 0.15)`,
            transition: "background 0.5s",
          }}
          animate={
            active
              ? { height: [b.baseH, b.maxH, b.baseH] }
              : { height: b.baseH }
          }
          transition={
            active
              ? { repeat: Infinity, duration: b.dur, ease: "easeInOut", delay: b.delay }
              : { duration: 0.3 }
          }
        />
      ))}
    </motion.div>
  );
}
