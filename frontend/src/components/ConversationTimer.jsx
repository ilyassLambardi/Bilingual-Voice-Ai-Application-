import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";

/**
 * Live elapsed conversation timer — starts when first message arrives,
 * shows mm:ss format with a subtle pulse on the colon.
 */
export default function ConversationTimer({ messages = [], connected = false, theme = "dark" }) {
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef(null);
  const isDark = theme === "dark";

  useEffect(() => {
    if (messages.length > 0 && !startRef.current) {
      startRef.current = Date.now();
    }
    if (messages.length === 0) {
      startRef.current = null;
      setElapsed(0);
    }
  }, [messages.length]);

  useEffect(() => {
    if (!startRef.current || !connected) return;
    const iv = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startRef.current) / 1000));
    }, 1000);
    return () => clearInterval(iv);
  }, [connected, messages.length]);

  if (!connected || messages.length === 0) return null;

  const mm = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const ss = String(elapsed % 60).padStart(2, "0");

  return (
    <div className="flex items-center gap-1" style={{
      fontSize: 10, fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
      color: isDark ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.2)",
      letterSpacing: "0.05em",
    }}>
      <svg width="10" height="10" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
        style={{ opacity: 0.5 }}>
        <circle cx="12" cy="12" r="10" />
        <polyline points="12 6 12 12 16 14" />
      </svg>
      <span>{mm}</span>
      <motion.span
        animate={{ opacity: [1, 0.3, 1] }}
        transition={{ repeat: Infinity, duration: 1.0 }}
      >:</motion.span>
      <span>{ss}</span>
    </div>
  );
}
