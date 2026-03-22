import { useRef, useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Copy, Check, ChevronDown } from "lucide-react";

function timeAgo(ts) {
  if (!ts) return "";
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 5) return "now";
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m`;
}

function MsgBubble({ msg, isDark }) {
  const isUser = msg.role === "user";
  const [copied, setCopied] = useState(false);
  const [hovered, setHovered] = useState(false);
  const [feedback, setFeedback] = useState(null); // 'up' | 'down' | null

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(msg.text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    });
  }, [msg.text]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -4 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      style={{
        display: "flex", flexDirection: "column",
        alignItems: isUser ? "flex-end" : "flex-start",
        marginBottom: 8,
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="relative" style={{
        maxWidth: "85%",
        padding: "8px 14px",
        borderRadius: isUser ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
        fontSize: 13, lineHeight: 1.65,
        background: isUser
          ? isDark ? "linear-gradient(135deg, rgba(6,182,212,0.08), rgba(129,140,248,0.04))"
                   : "linear-gradient(135deg, rgba(6,182,212,0.12), rgba(129,140,248,0.06))"
          : msg.language === "de"
            ? isDark ? "linear-gradient(135deg, rgba(245,158,11,0.04), rgba(217,119,6,0.02))"
                     : "linear-gradient(135deg, rgba(245,158,11,0.06), rgba(217,119,6,0.03))"
            : isDark ? "linear-gradient(135deg, rgba(255,255,255,0.025), rgba(129,140,248,0.01))" : "linear-gradient(135deg, rgba(255,255,255,0.03), rgba(129,140,248,0.015))",
        color: isUser
          ? isDark ? "rgba(255,255,255,0.8)" : "rgba(255,255,255,0.85)"
          : isDark ? "rgba(255,255,255,0.65)" : "rgba(255,255,255,0.7)",
        border: isUser
          ? isDark ? "1px solid rgba(6,182,212,0.1)" : "1px solid rgba(6,182,212,0.15)"
          : msg.language === "de"
            ? isDark ? "1px solid rgba(245,158,11,0.08)" : "1px solid rgba(245,158,11,0.1)"
            : isDark ? "1px solid rgba(255,255,255,0.04)" : "1px solid rgba(255,255,255,0.05)",
      }}>
        {!isUser && (
          <span style={{
            fontSize: 9, fontWeight: 700, letterSpacing: "0.08em",
            color: msg.language === "de" ? "rgba(245,158,11,0.5)" : "rgba(6,182,212,0.5)",
            marginRight: 6, transition: "color 0.5s",
          }}>ALEX</span>
        )}
        {msg.text}
        {msg._streaming && (
          <motion.span
            style={{
              display: "inline-block", width: 2, height: 14, marginLeft: 3,
              background: "rgba(6,182,212,0.6)", borderRadius: 1, verticalAlign: "middle",
            }}
            animate={{ opacity: [1, 0, 1] }}
            transition={{ repeat: Infinity, duration: 0.5 }}
          />
        )}

        {/* Action buttons on hover */}
        <AnimatePresence>
          {hovered && !msg._streaming && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              transition={{ duration: 0.12 }}
              className="flex items-center"
              style={{
                position: "absolute", top: -10,
                right: isUser ? "auto" : -10, left: isUser ? -10 : "auto",
                gap: 2,
              }}
            >
              {/* Copy */}
              <button onClick={handleCopy} style={{
                width: 22, height: 22, borderRadius: 6,
                background: "rgba(6,182,212,0.12)",
                border: "1px solid rgba(6,182,212,0.15)",
                display: "flex", alignItems: "center", justifyContent: "center",
                cursor: "pointer",
              }}>
                {copied
                  ? <Check size={10} style={{ color: "#10b981" }} />
                  : <Copy size={10} style={{ color: "rgba(6,182,212,0.7)" }} />
                }
              </button>
              {/* Thumbs up/down for AI messages */}
              {!isUser && (
                <>
                  <button onClick={() => setFeedback(f => f === "up" ? null : "up")} style={{
                    width: 22, height: 22, borderRadius: 6,
                    background: feedback === "up" ? "rgba(16,185,129,0.15)" : "rgba(255,255,255,0.04)",
                    border: feedback === "up" ? "1px solid rgba(16,185,129,0.25)" : "1px solid rgba(255,255,255,0.06)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    cursor: "pointer", transition: "all 0.2s",
                  }}>
                    <svg width="10" height="10" viewBox="0 0 24 24" fill={feedback === "up" ? "rgba(16,185,129,0.6)" : "none"}
                      stroke={feedback === "up" ? "#10b981" : "rgba(255,255,255,0.3)"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
                    </svg>
                  </button>
                  <button onClick={() => setFeedback(f => f === "down" ? null : "down")} style={{
                    width: 22, height: 22, borderRadius: 6,
                    background: feedback === "down" ? "rgba(239,68,68,0.12)" : "rgba(255,255,255,0.04)",
                    border: feedback === "down" ? "1px solid rgba(239,68,68,0.2)" : "1px solid rgba(255,255,255,0.06)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    cursor: "pointer", transition: "all 0.2s",
                  }}>
                    <svg width="10" height="10" viewBox="0 0 24 24" fill={feedback === "down" ? "rgba(239,68,68,0.5)" : "none"}
                      stroke={feedback === "down" ? "#ef4444" : "rgba(255,255,255,0.3)"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17" />
                    </svg>
                  </button>
                </>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Timestamp + language badge */}
      <div className="flex items-center gap-1.5" style={{
        marginTop: 3,
        paddingLeft: isUser ? 0 : 4,
        paddingRight: isUser ? 4 : 0,
      }}>
        {msg.language && (
          <span style={{
            fontSize: 8, fontWeight: 700, letterSpacing: "0.06em",
            color: msg.language === "de" ? "rgba(245,158,11,0.35)" : "rgba(6,182,212,0.35)",
            textTransform: "uppercase", transition: "color 0.5s",
          }}>{msg.language === "de" ? "DE" : "EN"}</span>
        )}
        <span style={{ fontSize: 9, color: "rgba(255,255,255,0.12)" }}>
          {timeAgo(msg._ts)}
        </span>
      </div>
    </motion.div>
  );
}

export default function TranscriptWindow({ messages = [], ghostText = "", isListening = false, theme = "dark" }) {
  const scrollRef = useRef(null);
  const isDark = theme === "dark";
  const [atBottom, setAtBottom] = useState(true);

  const scrollToBottom = useCallback(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, []);

  useEffect(() => {
    if (atBottom) scrollToBottom();
  }, [messages.length, messages[messages.length - 1]?.text, atBottom, scrollToBottom]);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 30);
  }, []);

  const visible = messages.slice(-12);
  const hasContent = visible.length > 0 || (ghostText && isListening);

  if (!hasContent) {
    return (
      <motion.div
        className="flex items-center justify-center py-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2, duration: 0.8 }}
      >
        <span style={{ fontSize: 12, fontWeight: 400, letterSpacing: "0.03em",
          color: isDark ? "rgba(255,255,255,0.1)" : "rgba(255,255,255,0.15)" }}>
          Tap the mic, press Space, or type a message
        </span>
      </motion.div>
    );
  }

  return (
    <div className="relative">
      <div ref={scrollRef} onScroll={handleScroll} className="rounded-xl" style={{
        maxHeight: 190, overflowY: "auto", padding: "10px 12px",
        background: isDark
          ? "linear-gradient(135deg, rgba(255,255,255,0.015), rgba(6,182,212,0.008))"
          : "linear-gradient(135deg, rgba(255,255,255,0.025), rgba(6,182,212,0.012))",
        border: isDark ? "1px solid rgba(255,255,255,0.05)" : "1px solid rgba(255,255,255,0.06)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        boxShadow: "0 4px 24px rgba(0,0,0,0.12), inset 0 1px 0 rgba(255,255,255,0.02)",
        transition: "background 0.6s, border-color 0.6s",
      }}>
        <AnimatePresence initial={false}>
          {visible.map((msg, i) => (
            <MsgBubble key={msg.id ?? i} msg={msg} isDark={isDark} />
          ))}
        </AnimatePresence>

        {ghostText && isListening && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            style={{ display: "flex", justifyContent: "flex-end", marginBottom: 8 }}
          >
            <div style={{
              maxWidth: "85%", padding: "8px 14px",
              borderRadius: "14px 14px 4px 14px",
              fontSize: 13, fontStyle: "italic",
              color: isDark ? "rgba(255,255,255,0.2)" : "rgba(255,255,255,0.25)",
              border: isDark ? "1px dashed rgba(255,255,255,0.06)" : "1px dashed rgba(255,255,255,0.08)",
            }}>
              {ghostText}
            </div>
          </motion.div>
        )}
      </div>

      {/* Scroll-to-bottom FAB */}
      <AnimatePresence>
        {!atBottom && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            onClick={scrollToBottom}
            style={{
              position: "absolute", bottom: 8, right: 8,
              width: 28, height: 28, borderRadius: 8,
              background: "rgba(6,182,212,0.12)",
              border: "1px solid rgba(6,182,212,0.15)",
              display: "flex", alignItems: "center", justifyContent: "center",
              cursor: "pointer", backdropFilter: "blur(8px)",
              boxShadow: "0 2px 12px rgba(0,0,0,0.2)",
            }}
          >
            <ChevronDown size={14} style={{ color: "rgba(6,182,212,0.7)" }} />
          </motion.button>
        )}
      </AnimatePresence>
    </div>
  );
}
