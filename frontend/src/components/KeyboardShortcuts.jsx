import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";

const SHORTCUTS = [
  { keys: ["Space"], desc: "Toggle microphone", color: "244,114,182" },
  { keys: ["Enter"], desc: "Send chat message", color: "6,182,212" },
  { keys: ["?"], desc: "Show keyboard shortcuts", color: "168,85,247" },
  { keys: ["Esc"], desc: "Close panels / Stop", color: "245,158,11" },
  { keys: ["S"], desc: "Open settings", color: "16,185,129" },
  { keys: ["C"], desc: "Clear conversation", color: "251,146,60" },
];

export default function KeyboardShortcuts({ open, onClose }) {
  if (!open) return null;

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0"
            style={{ zIndex: 70, background: "rgba(0,0,0,0.5)", backdropFilter: "blur(6px)" }}
          />

          <motion.div
            initial={{ opacity: 0, scale: 0.92, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.92, y: 20 }}
            transition={{ type: "spring", stiffness: 400, damping: 28 }}
            className="fixed flex flex-col"
            style={{
              zIndex: 71,
              top: "50%", left: "50%", transform: "translate(-50%, -50%)",
              width: 360, maxHeight: "80vh",
              background: "linear-gradient(180deg, rgba(10,10,20,0.98), rgba(5,5,12,0.99))",
              border: "1px solid rgba(255,255,255,0.06)",
              borderRadius: 20,
              backdropFilter: "blur(20px)",
              boxShadow: "0 20px 60px rgba(0,0,0,0.4), 0 0 80px rgba(6,182,212,0.03)",
              overflow: "hidden",
            }}
          >
            {/* Header */}
            <div className="flex items-center justify-between" style={{ padding: "18px 20px 14px" }}>
              <div>
                <h2 style={{
                  fontSize: 15, fontWeight: 700, color: "rgba(255,255,255,0.9)",
                  letterSpacing: "0.02em",
                }}>Keyboard Shortcuts</h2>
                <p style={{ fontSize: 10, color: "rgba(255,255,255,0.25)", marginTop: 2 }}>Quick actions for power users</p>
              </div>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={onClose}
                style={{
                  width: 30, height: 30, borderRadius: 8, border: "none",
                  background: "rgba(255,255,255,0.05)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  cursor: "pointer",
                }}
              >
                <X size={14} style={{ color: "rgba(255,255,255,0.4)" }} />
              </motion.button>
            </div>

            <div style={{ height: 1, background: "rgba(255,255,255,0.04)", margin: "0 20px" }} />

            {/* Shortcuts list */}
            <div className="flex flex-col" style={{ padding: "12px 20px 20px" }}>
              {SHORTCUTS.map((s, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.05 + i * 0.04 }}
                  className="flex items-center justify-between"
                  style={{
                    padding: "10px 0",
                    borderBottom: i < SHORTCUTS.length - 1 ? "1px solid rgba(255,255,255,0.03)" : "none",
                  }}
                >
                  <span style={{
                    fontSize: 12, fontWeight: 500,
                    color: "rgba(255,255,255,0.6)",
                  }}>{s.desc}</span>
                  <div className="flex gap-1">
                    {s.keys.map((k, j) => (
                      <span key={j} style={{
                        padding: "3px 10px", borderRadius: 6,
                        fontSize: 10, fontWeight: 700, fontFamily: "'Space Grotesk', monospace",
                        color: `rgba(${s.color}, 0.8)`,
                        background: `rgba(${s.color}, 0.06)`,
                        border: `1px solid rgba(${s.color}, 0.1)`,
                        letterSpacing: "0.04em",
                      }}>{k}</span>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Footer hint */}
            <div style={{
              padding: "10px 20px 14px",
              borderTop: "1px solid rgba(255,255,255,0.03)",
            }}>
              <span style={{
                fontSize: 9, color: "rgba(255,255,255,0.12)",
                fontWeight: 500, letterSpacing: "0.06em",
              }}>Press ? again or Esc to close</span>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
