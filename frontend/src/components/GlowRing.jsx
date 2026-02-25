import { motion } from "framer-motion";

const STATE_COLORS = {
  idle: { inner: "rgba(6,182,212,0.06)", outer: "rgba(6,182,212,0.02)", glow: "rgba(6,182,212,0.08)" },
  listening: { inner: "rgba(244,114,182,0.12)", outer: "rgba(244,114,182,0.04)", glow: "rgba(244,114,182,0.15)" },
  thinking: { inner: "rgba(245,158,11,0.1)", outer: "rgba(245,158,11,0.03)", glow: "rgba(245,158,11,0.12)" },
  speaking: { inner: "rgba(16,185,129,0.1)", outer: "rgba(16,185,129,0.03)", glow: "rgba(16,185,129,0.12)" },
};

export default function GlowRing({ state = "idle" }) {
  const c = STATE_COLORS[state] || STATE_COLORS.idle;
  const isActive = state !== "idle";

  return (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-none" style={{ zIndex: 0 }}>
      {/* Outer pulse ring */}
      <motion.div
        className="absolute rounded-full"
        animate={{
          width: isActive ? 240 : 200,
          height: isActive ? 240 : 200,
          background: `radial-gradient(circle, ${c.outer} 0%, transparent 70%)`,
          opacity: isActive ? 1 : 0.3,
        }}
        transition={{ duration: 1.2, ease: "easeInOut" }}
      />

      {/* Inner glow ring */}
      <motion.div
        className="absolute rounded-full"
        animate={{
          width: isActive ? 180 : 160,
          height: isActive ? 180 : 160,
          boxShadow: isActive
            ? `0 0 40px ${c.glow}, 0 0 80px ${c.outer}, inset 0 0 30px ${c.outer}`
            : `0 0 20px ${c.glow}`,
          borderColor: c.inner,
        }}
        transition={{ duration: 0.8 }}
        style={{
          border: "1px solid",
          background: "transparent",
        }}
      />

      {/* Breathing pulse (only when active) */}
      {isActive && (
        <motion.div
          className="absolute rounded-full"
          animate={{
            width: [170, 190, 170],
            height: [170, 190, 170],
            opacity: [0.15, 0.3, 0.15],
          }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          style={{
            border: `1px solid ${c.inner}`,
            background: "transparent",
          }}
        />
      )}

      {/* Rotating arc (thinking/speaking) */}
      {(state === "thinking" || state === "speaking") && (
        <motion.div
          className="absolute"
          animate={{ rotate: 360 }}
          transition={{ duration: state === "thinking" ? 2 : 4, repeat: Infinity, ease: "linear" }}
          style={{ width: 200, height: 200 }}
        >
          <div
            className="absolute rounded-full"
            style={{
              width: 4, height: 4, top: 0, left: "50%", marginLeft: -2,
              background: c.inner,
              boxShadow: `0 0 8px ${c.glow}`,
            }}
          />
        </motion.div>
      )}
    </div>
  );
}
