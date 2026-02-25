import { motion, AnimatePresence } from "framer-motion";
import { X, Sun, Moon, Globe, Volume2, Zap } from "lucide-react";

export default function SettingsPanel({ open, onClose, theme, onToggleTheme, language, onSetLanguage }) {
  const isDark = theme === "dark";

  if (!open) return null;

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0"
            style={{ zIndex: 50, background: "rgba(0,0,0,0.4)", backdropFilter: "blur(4px)" }}
          />

          {/* Panel */}
          <motion.div
            initial={{ opacity: 0, x: 300 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 300 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="fixed right-0 top-0 bottom-0 flex flex-col"
            style={{
              zIndex: 51, width: 320,
              background: isDark
                ? "linear-gradient(180deg, rgba(8,8,16,0.97), rgba(3,3,8,0.99))"
                : "linear-gradient(180deg, rgba(12,18,32,0.98), rgba(8,12,24,0.99))",
              borderLeft: isDark ? "1px solid rgba(255,255,255,0.05)" : "1px solid rgba(255,255,255,0.06)",
              backdropFilter: "blur(20px)",
            }}
          >
            {/* Header */}
            <div className="flex items-center justify-between" style={{ padding: "20px 20px 16px" }}>
              <h2 style={{
                fontSize: 16, fontWeight: 700,
                color: isDark ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.9)",
                letterSpacing: "0.02em",
              }}>Settings</h2>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={onClose}
                style={{
                  width: 32, height: 32, borderRadius: 8, border: "none",
                  background: isDark ? "rgba(255,255,255,0.05)" : "rgba(255,255,255,0.06)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  cursor: "pointer",
                }}
              >
                <X size={16} style={{ color: isDark ? "rgba(255,255,255,0.5)" : "rgba(255,255,255,0.5)" }} />
              </motion.button>
            </div>

            <div style={{ height: 1, background: isDark ? "rgba(255,255,255,0.04)" : "rgba(255,255,255,0.05)", margin: "0 20px" }} />

            <div className="flex flex-col gap-1" style={{ padding: "16px 20px", flex: 1, overflowY: "auto" }}>

              {/* Theme Toggle */}
              <SettingRow
                icon={isDark ? <Moon size={16} /> : <Sun size={16} />}
                label="Appearance"
                description={isDark ? "Dark mode" : "Light mode"}
                isDark={isDark}
              >
                <motion.button
                  whileHover={{ scale: 1.04 }}
                  whileTap={{ scale: 0.96 }}
                  onClick={onToggleTheme}
                  style={{
                    width: 48, height: 26, borderRadius: 13, border: "none", cursor: "pointer",
                    background: isDark
                      ? "linear-gradient(135deg, rgba(6,182,212,0.3), rgba(244,114,182,0.2))"
                      : "linear-gradient(135deg, rgba(245,158,11,0.3), rgba(244,114,182,0.2))",
                    position: "relative", transition: "background 0.3s",
                  }}
                >
                  <motion.div
                    animate={{ x: isDark ? 22 : 2 }}
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                    style={{
                      width: 22, height: 22, borderRadius: 11,
                      background: isDark ? "#06b6d4" : "#f59e0b",
                      position: "absolute", top: 2,
                      boxShadow: isDark ? "0 0 8px rgba(6,182,212,0.4)" : "0 0 8px rgba(245,158,11,0.4)",
                      display: "flex", alignItems: "center", justifyContent: "center",
                    }}
                  >
                    {isDark
                      ? <Moon size={11} style={{ color: "#030308" }} />
                      : <Sun size={11} style={{ color: "#451a03" }} />
                    }
                  </motion.div>
                </motion.button>
              </SettingRow>

              {/* Language */}
              <SettingRow
                icon={<Globe size={16} />}
                label="Language"
                description="Detection mode"
                isDark={isDark}
              >
                <div className="flex gap-1.5">
                  {[
                    { value: "auto", label: "Auto" },
                    { value: "en", label: "EN" },
                    { value: "de", label: "DE" },
                  ].map((opt) => (
                    <motion.button
                      key={opt.value}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => onSetLanguage(opt.value)}
                      style={{
                        padding: "4px 10px", borderRadius: 8, cursor: "pointer",
                        fontSize: 11, fontWeight: 600, letterSpacing: "0.04em",
                        background: language === opt.value
                          ? "rgba(6,182,212,0.15)"
                          : isDark ? "rgba(255,255,255,0.03)" : "rgba(255,255,255,0.04)",
                        color: language === opt.value
                          ? "#06b6d4"
                          : isDark ? "rgba(255,255,255,0.4)" : "rgba(255,255,255,0.45)",
                        border: language === opt.value
                          ? "1px solid rgba(6,182,212,0.2)"
                          : isDark ? "1px solid rgba(255,255,255,0.04)" : "1px solid rgba(255,255,255,0.06)",
                        transition: "all 0.2s",
                      }}
                    >{opt.label}</motion.button>
                  ))}
                </div>
              </SettingRow>

              {/* Info Cards */}
              <div style={{ marginTop: 16 }}>
                <span style={{
                  fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase",
                  color: isDark ? "rgba(255,255,255,0.2)" : "rgba(255,255,255,0.25)",
                }}>Pipeline</span>
              </div>

              <div className="flex flex-col gap-2" style={{ marginTop: 8 }}>
                {[
                  { icon: <Volume2 size={14} />, label: "VAD", desc: "Silero Voice Activity", color: "16,185,129" },
                  { icon: <Globe size={14} />, label: "ASR", desc: "Groq Whisper Large v3", color: "6,182,212" },
                  { icon: <Zap size={14} />, label: "LLM", desc: "Groq Llama 3.3 70B", color: "244,114,182" },
                  { icon: <Volume2 size={14} />, label: "TTS", desc: "Silero Multi-lang", color: "245,158,11" },
                ].map((item, i) => (
                  <div key={i} className="flex items-center gap-3" style={{
                    padding: "10px 12px", borderRadius: 10,
                    background: isDark ? "rgba(255,255,255,0.02)" : "rgba(255,255,255,0.03)",
                    border: isDark ? "1px solid rgba(255,255,255,0.03)" : "1px solid rgba(255,255,255,0.04)",
                  }}>
                    <div style={{
                      width: 28, height: 28, borderRadius: 7,
                      background: `rgba(${item.color}, 0.1)`,
                      display: "flex", alignItems: "center", justifyContent: "center",
                      color: `rgba(${item.color}, 0.7)`,
                    }}>{item.icon}</div>
                    <div>
                      <div style={{
                        fontSize: 12, fontWeight: 600,
                        color: isDark ? "rgba(255,255,255,0.7)" : "rgba(255,255,255,0.75)",
                      }}>{item.label}</div>
                      <div style={{
                        fontSize: 10,
                        color: isDark ? "rgba(255,255,255,0.3)" : "rgba(255,255,255,0.35)",
                      }}>{item.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Footer */}
            <div style={{
              padding: "12px 20px",
              borderTop: isDark ? "1px solid rgba(255,255,255,0.03)" : "1px solid rgba(255,255,255,0.04)",
            }}>
              <span style={{
                fontSize: 9, fontWeight: 500, letterSpacing: "0.1em",
                color: isDark ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.1)",
              }}>
                Bilingual Voice AI v1.0
              </span>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

function SettingRow({ icon, label, description, isDark, children }) {
  return (
    <div className="flex items-center justify-between" style={{
      padding: "14px 0",
      borderBottom: isDark ? "1px solid rgba(255,255,255,0.03)" : "1px solid rgba(255,255,255,0.04)",
    }}>
      <div className="flex items-center gap-3">
        <div style={{ color: isDark ? "rgba(255,255,255,0.4)" : "rgba(255,255,255,0.45)" }}>{icon}</div>
        <div>
          <div style={{
            fontSize: 13, fontWeight: 600,
            color: isDark ? "rgba(255,255,255,0.8)" : "rgba(255,255,255,0.85)",
          }}>{label}</div>
          <div style={{
            fontSize: 10,
            color: isDark ? "rgba(255,255,255,0.3)" : "rgba(255,255,255,0.35)",
          }}>{description}</div>
        </div>
      </div>
      {children}
    </div>
  );
}
