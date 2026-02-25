import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, MessageSquare, X } from "lucide-react";

export default function ChatInput({ onSend, disabled, pipelineState, theme = "dark" }) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  const inputRef = useRef(null);
  const isDark = theme === "dark";

  useEffect(() => {
    if (open && inputRef.current) inputRef.current.focus();
  }, [open]);

  const handleSend = () => {
    const msg = text.trim();
    if (!msg || disabled) return;
    onSend(msg);
    setText("");
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
    if (e.key === "Escape") setOpen(false);
  };

  const busy = pipelineState === "thinking" || pipelineState === "speaking";

  return (
    <div className="flex items-center gap-2">
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 260, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="flex items-center overflow-hidden rounded-full"
            style={{
              background: isDark ? "rgba(255,255,255,0.04)" : "rgba(255,255,255,0.05)",
              border: isDark ? "1px solid rgba(255,255,255,0.06)" : "1px solid rgba(255,255,255,0.08)",
              backdropFilter: "blur(12px)",
              height: 40,
            }}
          >
            <input
              ref={inputRef}
              value={text}
              onChange={(e) => setText(e.target.value.slice(0, 500))}
              onKeyDown={handleKey}
              disabled={busy}
              placeholder={busy ? "AI is responding..." : "Type a message..."}
              style={{
                flex: 1, border: "none", outline: "none", background: "transparent",
                padding: "0 14px", fontSize: 13, fontFamily: "'Space Grotesk', sans-serif",
                color: isDark ? "rgba(255,255,255,0.8)" : "rgba(255,255,255,0.85)",
              }}
            />
            {/* Character counter */}
            {text.length > 0 && (
              <span style={{
                fontSize: 8, fontWeight: 600, fontFamily: "'Space Grotesk', monospace",
                color: text.length > 450 ? "rgba(239,68,68,0.5)" : "rgba(255,255,255,0.12)",
                marginRight: 4, whiteSpace: "nowrap", transition: "color 0.2s",
              }}>{text.length}/500</span>
            )}
            <motion.button
              onClick={handleSend}
              disabled={!text.trim() || busy}
              whileTap={{ scale: 0.85 }}
              style={{
                width: 32, height: 32, marginRight: 4, borderRadius: "50%",
                border: "none", cursor: text.trim() && !busy ? "pointer" : "default",
                display: "flex", alignItems: "center", justifyContent: "center",
                background: text.trim() && !busy ? "rgba(6,182,212,0.15)" : "transparent",
                transition: "all 0.2s",
              }}
            >
              <Send size={14} style={{
                color: text.trim() && !busy ? "#06b6d4" : isDark ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.18)",
                transition: "color 0.2s",
              }} />
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>

      <motion.button
        whileHover={{ scale: 1.08 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setOpen(!open)}
        style={{
          width: 40, height: 40, borderRadius: "50%",
          border: isDark ? "1px solid rgba(255,255,255,0.06)" : "1px solid rgba(255,255,255,0.08)",
          background: isDark ? "rgba(255,255,255,0.03)" : "rgba(255,255,255,0.04)",
          display: "flex", alignItems: "center", justifyContent: "center",
          cursor: "pointer", backdropFilter: "blur(8px)",
        }}
      >
        {open ? (
          <X size={16} style={{ color: isDark ? "rgba(255,255,255,0.4)" : "rgba(255,255,255,0.45)" }} />
        ) : (
          <MessageSquare size={16} style={{ color: isDark ? "rgba(255,255,255,0.4)" : "rgba(255,255,255,0.45)" }} />
        )}
      </motion.button>
    </div>
  );
}
