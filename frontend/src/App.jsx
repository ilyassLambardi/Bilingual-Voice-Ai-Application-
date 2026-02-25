import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import FluidOrb from "./components/FluidOrb";
import TranscriptWindow from "./components/TranscriptWindow";
import Controls from "./components/Controls";
import ChatInput from "./components/ChatInput";
import SubtitleBar from "./components/SubtitleBar";
import SettingsPanel from "./components/SettingsPanel";
import VoiceWaveform from "./components/VoiceWaveform";
import StatusToast from "./components/StatusToast";
import SessionStats from "./components/SessionStats";
import KeyboardShortcuts from "./components/KeyboardShortcuts";
import ConnectionIndicator from "./components/ConnectionIndicator";
import TypingIndicator from "./components/TypingIndicator";
import ExportChat from "./components/ExportChat";
import MicLevelMeter from "./components/MicLevelMeter";
import PipelineVisualizer from "./components/PipelineVisualizer";
import QuickActions from "./components/QuickActions";
import GlowRing from "./components/GlowRing";
import AudioScope from "./components/AudioScope";
import ConversationTimer from "./components/ConversationTimer";
import ResponseLatency from "./components/ResponseLatency";
import MessageCounter from "./components/MessageCounter";
import ParticleBurst from "./components/ParticleBurst";
import EnergyMeter from "./components/EnergyMeter";
import useWebSocket from "./hooks/useWebSocket";
import useAudioStream from "./hooks/useAudioStream";

let msgId = 0;

const SC = {
  idle:      { accent: "#06b6d4", rgb: "6,182,212",   g1: "rgba(6,182,212,0.12)",  g2: "rgba(14,116,144,0.08)", text: "rgba(6,182,212,0.5)" },
  listening: { accent: "#f472b6", rgb: "244,114,182",  g1: "rgba(244,114,182,0.16)", g2: "rgba(236,72,153,0.08)", text: "rgba(244,114,182,0.9)" },
  thinking:  { accent: "#f59e0b", rgb: "245,158,11",   g1: "rgba(245,158,11,0.14)", g2: "rgba(217,119,6,0.07)",  text: "rgba(245,158,11,0.9)" },
  speaking:  { accent: "#10b981", rgb: "16,185,129",   g1: "rgba(16,185,129,0.15)", g2: "rgba(5,150,105,0.08)",  text: "rgba(16,185,129,0.9)" },
};

// Alt theme: deep navy with richer aurora glow
const SC_ALT = {
  idle:      { g1: "rgba(6,182,212,0.18)",  g2: "rgba(99,102,241,0.1)" },
  listening: { g1: "rgba(244,114,182,0.2)",  g2: "rgba(168,85,247,0.1)" },
  thinking:  { g1: "rgba(245,158,11,0.16)", g2: "rgba(251,146,60,0.08)" },
  speaking:  { g1: "rgba(16,185,129,0.18)",  g2: "rgba(52,211,153,0.08)" },
};

// ── Welcome Intro ──
function WelcomeIntro({ onDone }) {
  const [stage, setStage] = useState(0);
  const [ready, setReady] = useState(false);
  const [exiting, setExiting] = useState(false);

  // Mesh dots for animated background
  const meshDots = useMemo(() => Array.from({ length: 80 }, (_, i) => ({
    id: i,
    x: Math.random() * 100, y: Math.random() * 100,
    size: Math.random() * 1.5 + 0.5,
    opacity: Math.random() * 0.15 + 0.02,
    dur: Math.random() * 30 + 20,
    delay: Math.random() * -30,
  })), []);

  useEffect(() => {
    const t = [
      setTimeout(() => setStage(1), 200),
      setTimeout(() => setStage(2), 1000),
      setTimeout(() => setStage(3), 2200),
      setTimeout(() => setStage(4), 3400),
      setTimeout(() => setStage(5), 4800),
      setTimeout(() => setReady(true), 6200),
    ];
    return () => t.forEach(clearTimeout);
  }, []);

  const handleEnter = useCallback(() => {
    if (exiting) return;
    setExiting(true);
    setTimeout(onDone, 700);
  }, [onDone, exiting]);

  // Auto-enter after ready + 2s
  useEffect(() => {
    if (!ready) return;
    const t = setTimeout(handleEnter, 2000);
    return () => clearTimeout(t);
  }, [ready, handleEnter]);

  const s = stage; // shorthand

  return (
    <motion.div
      className="fixed inset-0 flex flex-col items-center justify-center overflow-hidden"
      style={{ zIndex: 100, background: "#08080c", fontFamily: "'Sora', sans-serif" }}
      animate={exiting ? { opacity: 0, scale: 1.05, filter: "blur(12px)" } : {}}
      transition={{ duration: 0.7, ease: [0.4, 0, 0.2, 1] }}
    >
      {/* ── ANIMATED MESH BACKGROUND ── */}
      <div className="absolute inset-0 overflow-hidden">
        {meshDots.map(d => (
          <div key={d.id} className="absolute rounded-full" style={{
            left: `${d.x}%`, top: `${d.y}%`,
            width: d.size, height: d.size,
            background: `rgba(6,182,212,${d.opacity})`,
            animation: `aurora-drift ${d.dur}s ease-in-out ${d.delay}s infinite`,
          }} />
        ))}
      </div>

      {/* ── GRADIENT ORBS ── */}
      <div className="absolute" style={{
        width: 700, height: 700, top: "10%", left: "50%",
        transform: "translateX(-50%)",
        background: "conic-gradient(from 0deg at 50% 50%, rgba(6,182,212,0.06), rgba(168,85,247,0.04), rgba(244,114,182,0.05), rgba(245,158,11,0.03), rgba(16,185,129,0.04), rgba(6,182,212,0.06))",
        filter: "blur(80px)", borderRadius: "50%",
        animation: "intro-gradient-rotate 30s linear infinite",
      }} />
      <div className="absolute" style={{
        width: 400, height: 400, bottom: "5%", left: "15%",
        background: "radial-gradient(circle, rgba(99,102,241,0.05) 0%, transparent 70%)",
        filter: "blur(60px)", borderRadius: "50%",
        animation: "aurora-drift-2 20s ease-in-out infinite",
      }} />

      {/* ── NOISE TEXTURE OVERLAY ── */}
      <div className="absolute inset-0" style={{
        backgroundImage: "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E\")",
        opacity: 0.4, pointerEvents: "none",
      }} />

      {/* ── HERO SECTION ── */}
      <div className="relative flex flex-col items-center" style={{ zIndex: 2, maxWidth: 600, padding: "0 24px" }}>

        {/* Version pill */}
        {s >= 1 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            style={{
              marginBottom: 32,
              padding: "4px 14px", borderRadius: 20,
              background: "rgba(6,182,212,0.06)",
              border: "1px solid rgba(6,182,212,0.1)",
              fontSize: 10, fontWeight: 600, letterSpacing: "0.08em",
              color: "rgba(6,182,212,0.5)",
              fontFamily: "'Space Grotesk', monospace",
            }}
          >
            v1.0 &nbsp;·&nbsp; Bilingual Voice Assistant
          </motion.div>
        )}

        {/* Title */}
        {s >= 2 && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            className="flex flex-col items-center" style={{ marginBottom: 20 }}
          >
            <h1 style={{
              fontSize: 64, fontWeight: 800, letterSpacing: "-0.04em", lineHeight: 0.95,
              textAlign: "center",
              background: "linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.5) 100%)",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            }}>
              Voice AI
            </h1>
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: 80 }}
              transition={{ duration: 0.6, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
              style={{
                height: 2, marginTop: 16, borderRadius: 2,
                background: "linear-gradient(90deg, rgba(6,182,212,0.5), rgba(168,85,247,0.3))",
              }}
            />
          </motion.div>
        )}

        {/* Subtitle */}
        {s >= 3 && (
          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
            style={{
              fontSize: 16, fontWeight: 400, lineHeight: 1.7,
              color: "rgba(255,255,255,0.35)", textAlign: "center",
              fontFamily: "'Inter', sans-serif",
              maxWidth: 380, marginBottom: 40,
            }}
          >
            Speak naturally in <span style={{ color: "rgba(6,182,212,0.7)" }}>English</span> or{" "}
            <span style={{ color: "rgba(244,114,182,0.7)" }}>German</span>.
            <br />Real-time speech recognition with natural AI responses.
          </motion.p>
        )}

        {/* Feature cards — 3 columns */}
        {s >= 4 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            style={{
              display: "grid", gridTemplateColumns: "repeat(3, 1fr)",
              gap: 10, width: "100%", maxWidth: 440, marginBottom: 36,
            }}
          >
            {[
              { icon: "M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z M19 10v2a7 7 0 0 1-14 0v-2 M12 19v4 M8 23h8", label: "Voice", c: "6,182,212" },
              { icon: "M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z", label: "Chat", c: "99,102,241" },
              { icon: "M12 2L2 7l10 5 10-5-10-5z M2 17l10 5 10-5 M2 12l10 5 10-5", label: "Memory", c: "16,185,129" },
              { icon: "M13 2L3 14h9l-1 8 10-12h-9l1-8z", label: "Fast", c: "245,158,11" },
              { icon: "M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z", label: "Secure", c: "168,85,247" },
              { icon: "M11 5.882V19.24a1.76 1.76 0 0 1-3.417.592l-2.147-6.15M18 13a3 3 0 1 0 0-6 M6 6l.002-.001 M6 18H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h1", label: "TTS", c: "244,114,182" },
            ].map((f, i) => (
              <motion.div
                key={f.label}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.06, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                className="flex flex-col items-center"
                style={{
                  padding: "14px 8px 12px", borderRadius: 14,
                  background: "rgba(255,255,255,0.02)",
                  border: "1px solid rgba(255,255,255,0.04)",
                  backdropFilter: "blur(8px)",
                  gap: 8,
                }}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
                  stroke={`rgba(${f.c},0.6)`} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d={f.icon} />
                </svg>
                <span style={{
                  fontSize: 10, fontWeight: 600, letterSpacing: "0.04em",
                  color: `rgba(${f.c},0.5)`,
                }}>{f.label}</span>
              </motion.div>
            ))}
          </motion.div>
        )}

        {/* Pipeline architecture */}
        {s >= 5 && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="flex flex-col items-center" style={{ gap: 12, marginBottom: 32 }}
          >
            <span style={{
              fontSize: 9, fontWeight: 700, letterSpacing: "0.18em", textTransform: "uppercase",
              color: "rgba(255,255,255,0.1)",
            }}>Pipeline</span>
            <div className="flex items-center" style={{ gap: 3 }}>
              {[
                { l: "VAD", c: "16,185,129" },
                { l: "ASR", c: "6,182,212" },
                { l: "LLM", c: "244,114,182" },
                { l: "TTS", c: "245,158,11" },
              ].map((p, i) => (
                <motion.div key={p.l} className="flex items-center" style={{ gap: 3 }}
                  initial={{ opacity: 0, x: -6 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.08, duration: 0.3 }}
                >
                  <span style={{
                    fontSize: 9, fontWeight: 700, letterSpacing: "0.05em",
                    color: `rgba(${p.c},0.45)`,
                    padding: "3px 8px", borderRadius: 5,
                    background: `rgba(${p.c},0.05)`,
                    border: `1px solid rgba(${p.c},0.08)`,
                    fontFamily: "'Space Grotesk', monospace",
                  }}>{p.l}</span>
                  {i < 3 && (
                    <motion.div
                      initial={{ scaleX: 0 }}
                      animate={{ scaleX: 1 }}
                      transition={{ delay: 0.2 + i * 0.1, duration: 0.3 }}
                      style={{ width: 12, height: 1, background: "rgba(255,255,255,0.06)" }}
                    />
                  )}
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Enter button */}
        {ready && (
          <motion.button
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            whileHover={{ scale: 1.04, boxShadow: "0 0 30px rgba(6,182,212,0.2)" }}
            whileTap={{ scale: 0.97 }}
            onClick={handleEnter}
            style={{
              padding: "10px 36px", borderRadius: 30,
              background: "linear-gradient(135deg, rgba(6,182,212,0.15), rgba(168,85,247,0.1))",
              border: "1px solid rgba(6,182,212,0.2)",
              color: "rgba(255,255,255,0.8)",
              fontSize: 13, fontWeight: 600, letterSpacing: "0.06em",
              cursor: "pointer",
              fontFamily: "'Sora', sans-serif",
              boxShadow: "0 0 20px rgba(6,182,212,0.08)",
              transition: "box-shadow 0.3s",
            }}
          >
            Get Started
          </motion.button>
        )}

        {/* Loading indicator before ready */}
        {s >= 5 && !ready && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col items-center gap-2.5"
          >
            <div style={{
              width: 140, height: 2, borderRadius: 1,
              background: "rgba(255,255,255,0.03)", overflow: "hidden",
            }}>
              <div style={{
                width: "30%", height: "100%", borderRadius: 1,
                background: "linear-gradient(90deg, transparent, rgba(6,182,212,0.5), transparent)",
                animation: "shimmer-sweep 1.4s ease-in-out infinite",
              }} />
            </div>
            <span style={{
              fontSize: 9, fontWeight: 500, letterSpacing: "0.12em",
              textTransform: "uppercase", color: "rgba(255,255,255,0.1)",
            }}>Loading</span>
          </motion.div>
        )}
      </div>

      {/* ── FOOTER ── */}
      {s >= 3 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.6 }}
          className="absolute flex items-center gap-3"
          style={{ bottom: 24, zIndex: 2 }}
        >
          <span style={{
            fontSize: 9, fontWeight: 400, letterSpacing: "0.08em",
            color: "rgba(255,255,255,0.06)",
            fontFamily: "'Inter', sans-serif",
          }}>
            Groq &middot; Silero &middot; Llama 3.3
          </span>
        </motion.div>
      )}
    </motion.div>
  );
}

function Particles({ accent, theme }) {
  const isDark = theme === "dark";
  const dots = useMemo(() => Array.from({ length: 22 }, (_, i) => ({
    id: i, x: Math.random() * 100, s: Math.random() * 2 + 0.6,
    d: Math.random() * 22 + 12, dl: Math.random() * 24, o: Math.random() * (isDark ? 0.2 : 0.12) + 0.04,
  })), [isDark]);
  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden" style={{ zIndex: 1 }}>
      {dots.map(q => (
        <div key={q.id} className="absolute rounded-full" style={{
          left: `${q.x}%`, bottom: "-3%", width: q.s, height: q.s,
          background: `rgba(${accent}, ${q.o})`,
          animation: `float-up ${q.d}s linear ${q.dl}s infinite`,
        }} />
      ))}
    </div>
  );
}

// ══════════════════════════════════════════════════════════
let toastId = 0;

export default function App() {
  const [messages, setMessages] = useState([]);
  const [pipelineState, setPipelineState] = useState("idle");
  const [detectedLang, setDetectedLang] = useState(null);
  const [ghostText, setGhostText] = useState("");
  const [introComplete, setIntroComplete] = useState(false);
  const [theme, setTheme] = useState("dark");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [langPref, setLangPref] = useState("auto");
  const [subtitle, setSubtitle] = useState({ text: "", language: "" });
  const [toasts, setToasts] = useState([]);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [statsOpen, setStatsOpen] = useState(false);
  const [backchannelPulse, setBackchannelPulse] = useState(0);
  const audioStreamRef = useRef(null);
  const prevConnected = useRef(null);

  const isDark = theme === "dark";
  const toggleTheme = useCallback(() => setTheme(t => t === "dark" ? "light" : "dark"), []);

  const pushToast = useCallback((type, message) => {
    setToasts(prev => [...prev, { id: ++toastId, type, message }]);
  }, []);

  const handleAudio = useCallback((ab) => { audioStreamRef.current?.playPcm(ab); }, []);
  const handleTranscript = useCallback((msg) => {
    if (msg.language) setDetectedLang(msg.language);
    if (msg.role === "assistant" && (msg.type === "partial_transcript" || msg.type === "transcript")) {
      setSubtitle({ text: msg.text, language: msg.language || "en" });
    }
    setMessages((prev) => {
      const ts = Date.now();
      const last = prev[prev.length - 1];

      // Helper: find last streaming assistant message (may not be the very last element)
      const findStreamingIdx = () => {
        for (let i = prev.length - 1; i >= 0; i--) {
          if (prev[i].role === "assistant" && prev[i]._streaming) return i;
        }
        return -1;
      };

      // ── Assistant partial (streaming sentence) ──
      if (msg.type === "partial_transcript" && msg.role === "assistant") {
        const sIdx = findStreamingIdx();
        if (sIdx >= 0) {
          const u = [...prev];
          u[sIdx] = { ...prev[sIdx], text: prev[sIdx].text + " " + msg.text };
          return u;
        }
        return [...prev, { id: ++msgId, role: "assistant", text: msg.text, language: msg.language || "", _streaming: true, _ts: ts }];
      }

      // ── Assistant final transcript ──
      if (msg.type === "transcript" && msg.role === "assistant") {
        const sIdx = findStreamingIdx();
        if (sIdx >= 0) {
          const u = [...prev];
          u[sIdx] = { ...prev[sIdx], text: msg.text, _streaming: false };
          return u;
        }
        // No streaming message — update last assistant or add new
        if (last?.role === "assistant") {
          const u = [...prev]; u[u.length - 1] = { ...last, text: msg.text, _streaming: false }; return u;
        }
        return [...prev, { id: ++msgId, role: "assistant", text: msg.text, language: msg.language || "", _ts: ts }];
      }

      // ── User or other transcript ──
      // Dedup guard: skip if last message has same role + same text
      if (last?.role === msg.role && last?.text === msg.text) {
        return prev;
      }
      return [...prev, { id: ++msgId, role: msg.role, text: msg.text, language: msg.language || "", _ts: ts }];
    });
  }, []);
  const handleGhostText = useCallback((t) => setGhostText(t || ""), []);
  const handleStateChange = useCallback((s) => {
    setPipelineState(s);
    if (s !== "listening") setGhostText("");
    if (s !== "speaking") setSubtitle({ text: "", language: "" });
  }, []);
  const handleBackchannel = useCallback(() => {
    setBackchannelPulse(p => p + 1);
  }, []);

  const { connected, connect, disconnect, sendAudio, sendCommand } =
    useWebSocket({ onAudio: handleAudio, onTranscript: handleTranscript, onStateChange: handleStateChange, onGhostText: handleGhostText, onBackchannel: handleBackchannel });
  const audioStream = useAudioStream(sendAudio);
  audioStreamRef.current = audioStream;
  window.__stopPlayback = audioStream.stopPlayback;

  // Toast on connection change
  useEffect(() => {
    if (prevConnected.current === null) { prevConnected.current = connected; return; }
    if (connected && !prevConnected.current) pushToast("connected", "Connected to server");
    if (!connected && prevConnected.current) pushToast("disconnected", "Disconnected from server");
    prevConnected.current = connected;
  }, [connected, pushToast]);

  // Toast on language detection
  const prevLang = useRef(null);
  useEffect(() => {
    if (detectedLang && detectedLang !== prevLang.current) {
      pushToast("language", `Language: ${detectedLang === "de" ? "Deutsch" : "English"}`);
      prevLang.current = detectedLang;
    }
  }, [detectedLang, pushToast]);

  useEffect(() => { const t = setTimeout(() => connect(), 800); return () => clearTimeout(t); }, [connect]);

  const handleConnect = useCallback(() => {
    if (connected) { audioStream.stopMic(); disconnect(); } else { connect(); }
  }, [connected, connect, disconnect, audioStream]);
  const handleToggleMic = useCallback(() => {
    audioStream.micActive ? audioStream.stopMic() : audioStream.startMic();
  }, [audioStream]);
  const handleClear = useCallback(() => {
    setMessages([]); setDetectedLang(null); setSubtitle({ text: "", language: "" });
    sendCommand({ type: "clear" }); audioStream.stopPlayback();
    pushToast("cleared", "Conversation cleared");
  }, [sendCommand, audioStream, pushToast]);
  const handleSendChat = useCallback((text) => {
    sendCommand({ type: "chat", text });
  }, [sendCommand]);
  const handleSetLanguage = useCallback((lang) => {
    setLangPref(lang);
    sendCommand({ type: "config", language: lang });
  }, [sendCommand]);

  // Keyboard shortcuts: Space, ?, S, C, Esc
  useEffect(() => {
    const fn = (e) => {
      const tag = e.target.tagName;
      const isInput = tag === "INPUT" || tag === "TEXTAREA";

      if (e.code === "Space" && !isInput && connected) {
        e.preventDefault(); audioStream.micActive ? audioStream.stopMic() : audioStream.startMic();
      }
      if (e.key === "?" && !isInput) {
        e.preventDefault(); setShortcutsOpen(p => !p);
      }
      if (e.key === "s" && !isInput && !e.metaKey && !e.ctrlKey) {
        e.preventDefault(); setSettingsOpen(p => !p);
      }
      if (e.key === "c" && !isInput && !e.metaKey && !e.ctrlKey && connected) {
        e.preventDefault(); handleClear();
      }
      if (e.key === "Escape") {
        setSettingsOpen(false); setShortcutsOpen(false); setStatsOpen(false);
      }
    };
    window.addEventListener("keydown", fn);
    return () => window.removeEventListener("keydown", fn);
  }, [connected, audioStream, handleClear]);

  const c = SC[pipelineState] || SC.idle;
  const cAlt = SC_ALT[pipelineState] || SC_ALT.idle;
  const label = { idle: "Ready", listening: "Listening", thinking: "Processing", speaking: "Speaking" }[pipelineState] || "";
  const show = introComplete;

  // Semantic color shifting: cool blue for EN, warm amber for DE
  const langTint = detectedLang === "de" ? { bg: isDark ? "#080503" : "#140e08", div: "rgba(245,158,11,0.08)" }
    : { bg: isDark ? "#030308" : "#0c1220", div: "rgba(6,182,212,0.08)" };
  const bg = langTint.bg;
  const dividerColor = langTint.div;

  return (
    <div className="h-full w-full flex flex-col items-center select-none overflow-hidden relative"
      style={{ background: bg, transition: "background 0.6s ease" }}>

      <AnimatePresence>
        {!introComplete && <WelcomeIntro onDone={() => setIntroComplete(true)} />}
      </AnimatePresence>

      {/* ═══ AURORA BACKGROUND (language-aware) ═══ */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden" style={{ zIndex: 0 }}>
        <motion.div className="absolute rounded-full"
          animate={{ background: `radial-gradient(circle, ${isDark ? c.g1 : cAlt.g1} 0%, transparent 65%)` }}
          transition={{ duration: 2.5 }}
          style={{ width: 1000, height: 1000, top: "-25%", left: "50%", marginLeft: -500,
            filter: "blur(100px)", animation: "aurora-drift 24s ease-in-out infinite" }} />
        <motion.div className="absolute rounded-full"
          animate={{ background: `radial-gradient(circle, ${isDark ? c.g2 : cAlt.g2} 0%, transparent 65%)` }}
          transition={{ duration: 2.5 }}
          style={{ width: 700, height: 700, bottom: "-10%", left: "-10%",
            filter: "blur(110px)", animation: "aurora-drift-2 30s ease-in-out infinite" }} />
        {/* Language-tinted aurora — amber for DE, blue for EN */}
        <motion.div className="absolute rounded-full"
          animate={{
            background: detectedLang === "de"
              ? "radial-gradient(circle, rgba(245,158,11,0.06) 0%, transparent 65%)"
              : "radial-gradient(circle, rgba(99,102,241,0.04) 0%, transparent 65%)",
          }}
          transition={{ duration: 3 }}
          style={{
            width: 500, height: 500, top: "35%", right: "-12%",
            filter: "blur(80px)", animation: "aurora-drift 22s ease-in-out infinite reverse",
          }} />
        {/* Semantic accent orb — appears stronger when language is detected */}
        <motion.div className="absolute rounded-full"
          animate={{
            background: detectedLang === "de"
              ? "radial-gradient(circle, rgba(217,119,6,0.07) 0%, transparent 60%)"
              : "radial-gradient(circle, rgba(6,182,212,0.05) 0%, transparent 60%)",
            opacity: detectedLang ? 1 : 0.3,
          }}
          transition={{ duration: 2.5 }}
          style={{
            width: 600, height: 600, top: "50%", left: "50%", marginTop: -300, marginLeft: -300,
            filter: "blur(90px)", animation: "aurora-drift-2 18s ease-in-out infinite",
          }} />
      </div>

      <Particles accent={detectedLang === "de" ? "245,158,11" : c.rgb} theme={theme} />

      {/* ═══ TOP BAR ═══ */}
      <motion.header className="relative w-full max-w-2xl px-6 pt-5 pb-3 flex items-center justify-between"
        style={{ zIndex: 10 }}
        initial={{ opacity: 0, y: -20 }} animate={{ opacity: show ? 1 : 0, y: show ? 0 : -20 }}
        transition={{ duration: 0.6, delay: 0.1 }}>
        <div className="flex items-center gap-2.5">
          <div className="relative">
            <span className="block rounded-full" style={{
              width: 8, height: 8,
              background: connected ? "#10b981" : (isDark ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.08)"),
              boxShadow: connected ? "0 0 12px rgba(16,185,129,0.5)" : "none",
              transition: "all 0.6s",
            }} />
            {connected && <span className="absolute inset-0 rounded-full animate-ping"
              style={{ width: 8, height: 8, background: "#10b981", opacity: 0.2 }} />}
          </div>
          <span style={{ fontSize: 11, fontWeight: 500, transition: "color 0.5s",
            color: connected ? "rgba(16,185,129,0.6)" : (isDark ? "rgba(255,255,255,0.12)" : "rgba(255,255,255,0.18)") }}>
            {connected ? "Connected" : "Offline"}
          </span>
          <ConnectionIndicator connected={connected} />
          <ConversationTimer messages={messages} connected={connected} theme={theme} />
        </div>

        <span style={{
          fontSize: 13, fontWeight: 700, letterSpacing: "0.22em", textTransform: "uppercase",
          background: "linear-gradient(135deg, #67e8f9, #06b6d4, #22d3ee, #a5f3fc)",
          backgroundSize: "300% 300%",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          animation: "gradient-flow 5s ease infinite",
        }}>
          Voice AI
        </span>

        <div className="flex items-center gap-2 relative">
          <AnimatePresence mode="wait">
            <motion.div key={detectedLang || "x"}
              initial={{ opacity: 0, scale: 0.85 }} animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.85 }} transition={{ type: "spring", stiffness: 400, damping: 22 }}
              className="flex items-center gap-1.5">
              {detectedLang ? (
                <>
                  <span style={{ fontSize: 14 }}>{detectedLang === "de" ? "\uD83C\uDDE9\uD83C\uDDEA" : "\uD83C\uDDEC\uD83C\uDDE7"}</span>
                  <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.06em",
                    color: isDark ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.35)" }}>
                    {detectedLang === "de" ? "DE" : "EN"}
                  </span>
                </>
              ) : (
                <span style={{ fontSize: 10, color: isDark ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.12)" }}>EN · DE</span>
              )}
            </motion.div>
          </AnimatePresence>

          {/* Session stats */}
          <SessionStats
            messages={messages}
            connected={connected}
            pipelineState={pipelineState}
            detectedLang={detectedLang}
            open={statsOpen}
            onToggle={() => setStatsOpen(p => !p)}
          />

          {/* Message count + Export */}
          <MessageCounter count={messages.length} theme={theme} />
          <ExportChat messages={messages} theme={theme} />

          {/* Settings gear */}
          <motion.button
            whileHover={{ scale: 1.1, rotate: 30 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => setSettingsOpen(true)}
            style={{
              width: 30, height: 30, borderRadius: 8, border: "none", cursor: "pointer",
              background: "transparent", display: "flex", alignItems: "center", justifyContent: "center",
              color: isDark ? "rgba(255,255,255,0.2)" : "rgba(255,255,255,0.25)",
              transition: "color 0.3s",
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
            </svg>
          </motion.button>
        </div>
      </motion.header>

      <motion.div className="w-full max-w-2xl px-8 relative" style={{ zIndex: 10 }}
        initial={{ opacity: 0 }} animate={{ opacity: show ? 1 : 0 }} transition={{ delay: 0.2 }}>
        <div style={{ height: 1, background: `linear-gradient(to right, transparent, ${dividerColor}, transparent)` }} />
      </motion.div>

      {/* ═══ PIPELINE VISUALIZER ═══ */}
      <motion.div className="relative w-full flex justify-center py-2" style={{ zIndex: 10 }}
        initial={{ opacity: 0 }} animate={{ opacity: show ? 1 : 0 }} transition={{ delay: 0.25 }}>
        <PipelineVisualizer pipelineState={pipelineState} theme={theme} />
      </motion.div>

      {/* ═══ ORB ═══ */}
      <motion.div className="relative flex-1 flex flex-col items-center justify-center w-full" style={{ zIndex: 10 }}
        initial={{ opacity: 0, scale: 0.88 }} animate={{ opacity: show ? 1 : 0, scale: show ? 1 : 0.88 }}
        transition={{ duration: 0.9, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}>
        <div className="relative">
          <GlowRing state={pipelineState} />
          <FluidOrb state={pipelineState} lang={detectedLang || "en"} />
          <ParticleBurst pipelineState={pipelineState} />
          {/* Backchannel pulse — quick ring flash when user says "mhm" */}
          <AnimatePresence>
            {backchannelPulse > 0 && (
              <motion.div
                key={backchannelPulse}
                className="absolute inset-0 flex items-center justify-center pointer-events-none"
                initial={{ opacity: 0.6, scale: 0.9 }}
                animate={{ opacity: 0, scale: 1.4 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
              >
                <div className="rounded-full" style={{
                  width: 200, height: 200,
                  border: `2px solid rgba(${c.rgb}, 0.35)`,
                  boxShadow: `0 0 20px rgba(${c.rgb}, 0.15)`,
                }} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <motion.div className="mt-7 flex flex-col items-center gap-3"
          key={pipelineState} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
          <div className="flex items-center gap-2.5">
            {pipelineState === "thinking" && (
              <div style={{ width: 14, height: 14, borderRadius: "50%", border: "2px solid transparent",
                borderTopColor: c.accent, borderRightColor: `rgba(${c.rgb},0.15)`,
                animation: "spin 0.7s linear infinite" }} />
            )}
            <span style={{ fontSize: 12, fontWeight: 600, letterSpacing: "0.22em", textTransform: "uppercase",
              color: c.text, transition: "color 0.8s" }}>
              {label}
            </span>
            <ResponseLatency pipelineState={pipelineState} theme={theme} />
          </div>

          <VoiceWaveform
            state={pipelineState}
            active={pipelineState === "listening" || pipelineState === "speaking"}
          />
        </motion.div>
      </motion.div>

      {/* SubtitleBar removed */}

      {/* ═══ TYPING INDICATOR ═══ */}
      <AnimatePresence>
        {show && pipelineState === "thinking" && (
          <motion.div className="relative w-full max-w-lg px-5 shrink-0" style={{ zIndex: 10 }}
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <TypingIndicator active={true} theme={theme} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* ═══ TRANSCRIPT ═══ */}
      <motion.div className="relative w-full max-w-lg px-5 shrink-0" style={{ zIndex: 10 }}
        initial={{ opacity: 0 }} animate={{ opacity: show ? 1 : 0 }} transition={{ delay: 0.5 }}>
        <TranscriptWindow messages={messages} ghostText={ghostText} isListening={pipelineState === "listening"} theme={theme} />
      </motion.div>

      {/* ═══ CONTROLS + CHAT ═══ */}
      <motion.div className="relative shrink-0 pt-3 pb-5 flex flex-col items-center gap-3" style={{ zIndex: 10 }}
        initial={{ opacity: 0, y: 20 }} animate={{ opacity: show ? 1 : 0, y: show ? 0 : 20 }}
        transition={{ delay: 0.4, duration: 0.5 }}>
        <div className="flex items-center gap-3">
          <ChatInput
            onSend={handleSendChat}
            disabled={!connected}
            pipelineState={pipelineState}
            theme={theme}
          />
          <EnergyMeter active={audioStream.micActive || pipelineState === "speaking"} theme={theme} />
          <MicLevelMeter micActive={audioStream.micActive} theme={theme} />
          <Controls connected={connected} micActive={audioStream.micActive} pipelineState={pipelineState}
            onToggleMic={handleToggleMic} onClear={handleClear} onConnect={handleConnect} theme={theme} />
        </div>
      </motion.div>

      {/* ═══ AUDIO SCOPE ═══ */}
      {show && (pipelineState === "listening" || pipelineState === "speaking") && (
        <motion.div className="relative flex justify-center" style={{ zIndex: 10 }}
          initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 36 }}
          exit={{ opacity: 0, height: 0 }} transition={{ duration: 0.4 }}>
          <AudioScope
            active={pipelineState === "listening" || pipelineState === "speaking"}
            mode={pipelineState === "speaking" ? "bars" : "wave"}
            theme={theme}
          />
        </motion.div>
      )}

      {/* ═══ SPACEBAR HINT ═══ */}
      <AnimatePresence>
        {show && connected && !audioStream.micActive && pipelineState === "idle" && (
          <motion.div className="absolute flex items-center gap-2" style={{ bottom: 6, zIndex: 10 }}
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ delay: 4 }}>
            <kbd style={{ padding: "2px 7px", borderRadius: 4, fontSize: 9, fontFamily: "monospace",
              color: isDark ? "rgba(255,255,255,0.1)" : "rgba(255,255,255,0.12)",
              border: isDark ? "1px solid rgba(6,182,212,0.06)" : "1px solid rgba(6,182,212,0.08)",
              background: isDark ? "rgba(255,255,255,0.015)" : "rgba(255,255,255,0.02)" }}>space</kbd>
            <span style={{ fontSize: 9, color: isDark ? "rgba(255,255,255,0.05)" : "rgba(255,255,255,0.07)" }}>to talk</span>
            <span style={{ fontSize: 9, color: isDark ? "rgba(255,255,255,0.03)" : "rgba(255,255,255,0.05)", margin: "0 4px" }}>·</span>
            <kbd style={{ padding: "2px 7px", borderRadius: 4, fontSize: 9, fontFamily: "monospace",
              color: isDark ? "rgba(255,255,255,0.1)" : "rgba(255,255,255,0.12)",
              border: isDark ? "1px solid rgba(168,85,247,0.06)" : "1px solid rgba(168,85,247,0.08)",
              background: isDark ? "rgba(255,255,255,0.015)" : "rgba(255,255,255,0.02)" }}>?</kbd>
            <span style={{ fontSize: 9, color: isDark ? "rgba(255,255,255,0.05)" : "rgba(255,255,255,0.07)" }}>shortcuts</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ═══ SETTINGS PANEL ═══ */}
      <SettingsPanel
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        theme={theme}
        onToggleTheme={toggleTheme}
        language={langPref}
        onSetLanguage={handleSetLanguage}
      />

      {/* ═══ TOAST NOTIFICATIONS ═══ */}
      <StatusToast events={toasts} />

      {/* ═══ KEYBOARD SHORTCUTS MODAL ═══ */}
      <KeyboardShortcuts open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />

      {/* ═══ QUICK ACTIONS FAB ═══ */}
      {show && (
        <QuickActions
          theme={theme}
          onAction={(id) => {
            if (id === "clear") handleClear();
            if (id === "settings") setSettingsOpen(true);
            if (id === "shortcuts") setShortcutsOpen(true);
            if (id === "stats") setStatsOpen(p => !p);
            if (id === "theme") toggleTheme();
          }}
        />
      )}
    </div>
  );
}
