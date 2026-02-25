import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";

export default function ExportChat({ messages, theme = "dark" }) {
  const [showMenu, setShowMenu] = useState(false);
  const [copied, setCopied] = useState(false);
  const isDark = theme === "dark";

  if (!messages || messages.length === 0) return null;

  const formatMessages = (format) => {
    if (format === "text") {
      return messages.map(m => {
        const role = m.role === "user" ? "You" : "Alex";
        const lang = m.language ? ` [${m.language.toUpperCase()}]` : "";
        return `${role}${lang}: ${m.text}`;
      }).join("\n\n");
    }
    if (format === "json") {
      return JSON.stringify(messages.map(m => ({
        role: m.role,
        text: m.text,
        language: m.language || null,
        timestamp: m._ts || null,
      })), null, 2);
    }
    if (format === "markdown") {
      let md = "# Voice AI Conversation\n\n";
      md += `*Exported: ${new Date().toLocaleString()}*\n\n---\n\n`;
      messages.forEach(m => {
        const role = m.role === "user" ? "**You**" : "**Alex**";
        const lang = m.language ? ` \`${m.language.toUpperCase()}\`` : "";
        md += `${role}${lang}: ${m.text}\n\n`;
      });
      return md;
    }
    return "";
  };

  const handleExport = (format) => {
    const content = formatMessages(format);
    const ext = { text: "txt", json: "json", markdown: "md" }[format];
    const mime = { text: "text/plain", json: "application/json", markdown: "text/markdown" }[format];

    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `voice-ai-chat-${Date.now()}.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
    setShowMenu(false);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(formatMessages("text"));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    setShowMenu(false);
  };

  return (
    <div className="relative">
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setShowMenu(p => !p)}
        style={{
          width: 30, height: 30, borderRadius: 8,
          background: "transparent", border: "none", cursor: "pointer",
          display: "flex", alignItems: "center", justifyContent: "center",
          color: isDark ? "rgba(255,255,255,0.2)" : "rgba(255,255,255,0.25)",
          transition: "color 0.3s",
        }}
        title="Export conversation"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
      </motion.button>

      <AnimatePresence>
        {showMenu && (
          <motion.div
            initial={{ opacity: 0, y: -4, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            style={{
              position: "absolute", top: 36, right: 0, zIndex: 50,
              background: isDark ? "rgba(15,15,20,0.95)" : "rgba(20,25,40,0.95)",
              border: "1px solid rgba(255,255,255,0.06)",
              borderRadius: 10, padding: 4, minWidth: 140,
              backdropFilter: "blur(12px)",
              boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
            }}
          >
            {[
              { label: copied ? "Copied!" : "Copy text", action: handleCopy, icon: "M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2" },
              { label: "Save as .txt", action: () => handleExport("text"), icon: "M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z M14 2v6h6" },
              { label: "Save as .md", action: () => handleExport("markdown"), icon: "M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z M14 2v6h6 M16 13H8 M16 17H8 M10 9H8" },
              { label: "Save as .json", action: () => handleExport("json"), icon: "M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z M14 2v6h6 M8 13h2 M8 17h2" },
            ].map((item, i) => (
              <motion.button
                key={i}
                whileHover={{ background: "rgba(255,255,255,0.04)" }}
                onClick={item.action}
                style={{
                  display: "flex", alignItems: "center", gap: 8,
                  width: "100%", padding: "7px 10px", borderRadius: 6,
                  background: "transparent", border: "none", cursor: "pointer",
                  color: "rgba(255,255,255,0.5)", fontSize: 11, fontWeight: 500,
                  fontFamily: "'Inter', sans-serif",
                  textAlign: "left",
                }}
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d={item.icon} />
                </svg>
                {item.label}
              </motion.button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
