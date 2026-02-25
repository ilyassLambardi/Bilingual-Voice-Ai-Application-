import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect, useRef } from "react";
import { Clock, MessageSquare, Globe, Activity } from "lucide-react";

function fmt(s) {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export default function SessionStats({ messages = [], connected, pipelineState, detectedLang, open, onToggle }) {
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef(null);

  useEffect(() => {
    if (connected && !startRef.current) startRef.current = Date.now();
    if (!connected) { startRef.current = null; setElapsed(0); }
  }, [connected]);

  useEffect(() => {
    if (!connected) return;
    const iv = setInterval(() => {
      if (startRef.current) setElapsed(Math.floor((Date.now() - startRef.current) / 1000));
    }, 1000);
    return () => clearInterval(iv);
  }, [connected]);

  const userMsgs = messages.filter((m) => m.role === "user").length;
  const aiMsgs = messages.filter((m) => m.role === "assistant").length;
  const words = messages.reduce((a, m) => a + (m.text?.split(/\s+/).length || 0), 0);
  const langs = [...new Set(messages.map((m) => m.language).filter(Boolean))];

  const stats = [
    { icon: <Clock size={13} />, label: "Session", value: fmt(elapsed), color: "6,182,212" },
    { icon: <MessageSquare size={13} />, label: "Messages", value: `${userMsgs + aiMsgs}`, color: "244,114,182" },
    { icon: <Activity size={13} />, label: "Words", value: `${words}`, color: "245,158,11" },
    { icon: <Globe size={13} />, label: "Languages", value: langs.length ? langs.map(l => l.toUpperCase()).join(" · ") : "—", color: "16,185,129" },
  ];

  return (
    <>
      {/* Toggle pill */}
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={onToggle}
        className="flex items-center gap-1.5"
        style={{
          padding: "4px 10px", borderRadius: 10,
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.06)",
          cursor: "pointer", fontSize: 10, fontWeight: 600,
          color: "rgba(6,182,212,0.5)", letterSpacing: "0.06em",
          backdropFilter: "blur(8px)",
        }}
      >
        <Activity size={11} />
        <span>{fmt(elapsed)}</span>
      </motion.button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 400, damping: 28 }}
            className="absolute flex flex-col gap-2"
            style={{
              top: 42, right: 0, width: 200, padding: 12, borderRadius: 14,
              background: "rgba(8,8,16,0.95)",
              border: "1px solid rgba(255,255,255,0.06)",
              backdropFilter: "blur(20px)",
              WebkitBackdropFilter: "blur(20px)",
              boxShadow: "0 8px 32px rgba(0,0,0,0.3)",
              zIndex: 30,
            }}
          >
            {stats.map((s, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.05 }}
                className="flex items-center justify-between"
                style={{ padding: "4px 0" }}
              >
                <div className="flex items-center gap-2">
                  <div style={{
                    width: 22, height: 22, borderRadius: 6,
                    background: `rgba(${s.color}, 0.08)`,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    color: `rgba(${s.color}, 0.6)`,
                  }}>{s.icon}</div>
                  <span style={{ fontSize: 10, color: "rgba(255,255,255,0.35)", fontWeight: 500 }}>{s.label}</span>
                </div>
                <span style={{
                  fontSize: 11, fontWeight: 700, letterSpacing: "0.03em",
                  color: `rgba(${s.color}, 0.7)`,
                  fontFamily: "'Space Grotesk', monospace",
                }}>{s.value}</span>
              </motion.div>
            ))}

            {/* Pipeline status */}
            <div style={{
              marginTop: 4, paddingTop: 6,
              borderTop: "1px solid rgba(255,255,255,0.04)",
            }}>
              <div className="flex items-center justify-between">
                <span style={{ fontSize: 9, color: "rgba(255,255,255,0.2)", fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase" }}>Status</span>
                <div className="flex items-center gap-1.5">
                  <div className="rounded-full" style={{
                    width: 5, height: 5,
                    background: connected ? "#10b981" : "rgba(255,255,255,0.1)",
                    boxShadow: connected ? "0 0 8px rgba(16,185,129,0.4)" : "none",
                  }} />
                  <span style={{
                    fontSize: 10, fontWeight: 600,
                    color: connected ? "rgba(16,185,129,0.6)" : "rgba(255,255,255,0.15)",
                  }}>{connected ? "Online" : "Offline"}</span>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
