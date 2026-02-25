import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useState, useCallback, useRef } from "react";

const TOAST_DURATION = 3200;

const ICONS = {
  connected: "🟢",
  disconnected: "🔴",
  language: "🌐",
  listening: "🎤",
  thinking: "🧠",
  speaking: "🔊",
  chat: "💬",
  cleared: "🗑️",
};

export default function StatusToast({ events = [] }) {
  const [visible, setVisible] = useState([]);
  const timerRef = useRef({});
  const seenRef = useRef(new Set());

  const dismiss = useCallback((id) => {
    setVisible((prev) => prev.filter((t) => t.id !== id));
    clearTimeout(timerRef.current[id]);
    delete timerRef.current[id];
  }, []);

  useEffect(() => {
    if (!events.length) return;
    const latest = events[events.length - 1];
    if (seenRef.current.has(latest.id)) return;
    seenRef.current.add(latest.id);

    setVisible((prev) => [...prev.slice(-2), latest]);

    timerRef.current[latest.id] = setTimeout(() => dismiss(latest.id), TOAST_DURATION);

    return () => {
      clearTimeout(timerRef.current[latest.id]);
    };
  }, [events, dismiss]);

  return (
    <div className="fixed flex flex-col items-center gap-2" style={{
      top: 64, left: "50%", transform: "translateX(-50%)", zIndex: 60,
      pointerEvents: "none",
    }}>
      <AnimatePresence>
        {visible.map((t) => (
          <motion.div
            key={t.id}
            initial={{ opacity: 0, y: -20, scale: 0.9, filter: "blur(4px)" }}
            animate={{ opacity: 1, y: 0, scale: 1, filter: "blur(0px)" }}
            exit={{ opacity: 0, y: -12, scale: 0.95, filter: "blur(4px)" }}
            transition={{ type: "spring", stiffness: 400, damping: 25 }}
            className="flex items-center gap-2.5"
            style={{
              padding: "8px 18px", borderRadius: 14,
              background: "rgba(6,182,212,0.06)",
              border: "1px solid rgba(6,182,212,0.1)",
              backdropFilter: "blur(16px)",
              WebkitBackdropFilter: "blur(16px)",
              pointerEvents: "auto",
              cursor: "pointer",
              boxShadow: "0 4px 24px rgba(0,0,0,0.15)",
            }}
            onClick={() => dismiss(t.id)}
          >
            <span style={{ fontSize: 14 }}>{ICONS[t.type] || "ℹ️"}</span>
            <span style={{
              fontSize: 12, fontWeight: 500, letterSpacing: "0.02em",
              color: "rgba(255,255,255,0.7)",
              fontFamily: "'Space Grotesk', sans-serif",
            }}>{t.message}</span>
            <div style={{
              position: "absolute", bottom: 0, left: 12, right: 12, height: 2,
              borderRadius: 1, overflow: "hidden",
            }}>
              <motion.div
                initial={{ width: "100%" }}
                animate={{ width: "0%" }}
                transition={{ duration: TOAST_DURATION / 1000, ease: "linear" }}
                style={{
                  height: "100%",
                  background: "linear-gradient(90deg, rgba(6,182,212,0.4), rgba(244,114,182,0.3))",
                  borderRadius: 1,
                }}
              />
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
