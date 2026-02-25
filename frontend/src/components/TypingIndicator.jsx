import { motion } from "framer-motion";

export default function TypingIndicator({ active, theme = "dark" }) {
  if (!active) return null;

  const isDark = theme === "dark";

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      transition={{ duration: 0.25 }}
      className="flex items-center"
      style={{
        gap: 10, padding: "8px 14px", borderRadius: 16,
        background: isDark ? "rgba(255,255,255,0.03)" : "rgba(255,255,255,0.05)",
        border: isDark ? "1px solid rgba(255,255,255,0.04)" : "1px solid rgba(255,255,255,0.06)",
        maxWidth: "fit-content",
      }}
    >
      <div className="flex items-center" style={{ gap: 3 }}>
        {[0, 1, 2].map(i => (
          <motion.div
            key={i}
            animate={{ y: [0, -4, 0], opacity: [0.3, 0.8, 0.3] }}
            transition={{
              duration: 0.6, repeat: Infinity,
              delay: i * 0.15, ease: "easeInOut",
            }}
            style={{
              width: 5, height: 5, borderRadius: "50%",
              background: "rgba(6,182,212,0.5)",
            }}
          />
        ))}
      </div>
      <span style={{
        fontSize: 10, fontWeight: 500, letterSpacing: "0.03em",
        color: isDark ? "rgba(255,255,255,0.2)" : "rgba(255,255,255,0.25)",
        fontFamily: "'Inter', sans-serif",
      }}>
        AI is thinking...
      </span>
    </motion.div>
  );
}
