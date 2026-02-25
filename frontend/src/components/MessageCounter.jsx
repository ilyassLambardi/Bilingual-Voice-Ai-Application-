import { motion, AnimatePresence } from "framer-motion";

/**
 * Animated message count badge — shows total messages with a pop animation on new ones.
 */
export default function MessageCounter({ count = 0, theme = "dark" }) {
  const isDark = theme === "dark";
  if (count === 0) return null;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={count}
        initial={{ scale: 1.3, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.8, opacity: 0 }}
        transition={{ type: "spring", stiffness: 500, damping: 25 }}
        className="flex items-center gap-1"
        style={{
          padding: "2px 7px",
          borderRadius: 10,
          background: isDark ? "rgba(6,182,212,0.08)" : "rgba(6,182,212,0.1)",
          border: isDark ? "1px solid rgba(6,182,212,0.1)" : "1px solid rgba(6,182,212,0.12)",
        }}
      >
        <svg width="9" height="9" viewBox="0 0 24 24" fill="none"
          stroke={isDark ? "rgba(6,182,212,0.4)" : "rgba(6,182,212,0.5)"}
          strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
        <span style={{
          fontSize: 9, fontWeight: 700,
          color: isDark ? "rgba(6,182,212,0.5)" : "rgba(6,182,212,0.6)",
          fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
        }}>{count}</span>
      </motion.div>
    </AnimatePresence>
  );
}
