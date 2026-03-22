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
import WelcomeIntro from "./components/WelcomeIntro";
import VoiceRipples from "./components/VoiceRipples";
import ConversationInsights from "./components/ConversationInsights";
import SmartSuggestions from "./components/SmartSuggestions";
import useWebSocket from "./hooks/useWebSocket";
import useAudioStream from "./hooks/useAudioStream";

let msgId = 0;

const SC = {
  idle:      { accent: "#06b6d4", rgb: "6,182,212",   g1: "rgba(6,182,212,0.12)",  g2: "rgba(14,116,144,0.08)", text: "rgba(6,182,212,0.5)" },
  listening: { accent: "#f472b6", rgb: "244,114,182",  g1: "rgba(244,114,182,0.16)", g2: "rgba(236,72,153,0.08)", text: "rgba(244,114,182,0.9)" },
  thinking:  { accent: "#f59e0b", rgb: "245,158,11",   g1: "rgba(245,158,11,0.14)", g2: "rgba(217,119,6,0.07)",  text: "rgba(245,158,11,0.9)" },
  speaking:  { accent: "#10b981", rgb: "16,185,129",   g1: "rgba(16,185,129,0.15)", g2: "rgba(5,150,105,0.08)",  text: "rgba(16,185,129,0.9)" },
};

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
  const [modelsReady, setModelsReady] = useState(true);
  const [insightsOpen, setInsightsOpen] = useState(false);
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
  const label = { idle: "Ready", listening: "Listening", thinking: "Processing", speaking: "Speaking" }[pipelineState] || "";
  const show = introComplete;

  // Semantic color shifting: cool blue for EN, warm amber for DE
  const langTint = detectedLang === "de" ? { bg: isDark ? "#080503" : "#140e08", div: "rgba(245,158,11,0.08)" }
    : { bg: isDark ? "#030308" : "#0c1220", div: "rgba(6,182,212,0.08)" };
  const bg = langTint.bg;
  const dividerColor = langTint.div;

  return (
    <div className="h-full w-full flex flex-col items-center select-none overflow-hidden relative"
      style={{ background: bg, transition: "background 0.8s ease" }}>

      <AnimatePresence>
        {!introComplete && <WelcomeIntro onDone={() => setIntroComplete(true)} />}
      </AnimatePresence>

      {/* ═══ AMBIENT MESH GRADIENT (state + language reactive) ═══ */}
      <div className="ambient-aura">
        <motion.div className="blob-1"
          animate={{ background: `radial-gradient(circle, ${c.g1} 0%, transparent 65%)` }}
          transition={{ duration: 2 }} />
        <motion.div className="blob-2"
          animate={{ background: `radial-gradient(circle, ${c.g2} 0%, transparent 65%)` }}
          transition={{ duration: 2 }} />
        <motion.div className="blob-3"
          animate={{
            background: detectedLang === "de"
              ? "radial-gradient(circle, rgba(245,158,11,0.06) 0%, transparent 60%)"
              : "radial-gradient(circle, rgba(129,140,248,0.04) 0%, transparent 60%)",
          }}
          transition={{ duration: 2.5 }} />
        {/* Semantic accent bloom */}
        <motion.div className="absolute rounded-full"
          animate={{
            background: detectedLang === "de"
              ? "radial-gradient(circle, rgba(217,119,6,0.06) 0%, transparent 55%)"
              : "radial-gradient(circle, rgba(6,182,212,0.04) 0%, transparent 55%)",
            opacity: detectedLang ? 0.8 : 0.3,
          }}
          transition={{ duration: 3 }}
          style={{ width: 500, height: 500, top: "40%", left: "50%", marginTop: -250, marginLeft: -250,
            filter: "blur(80px)", animation: "mesh-drift-2 22s ease-in-out infinite" }} />
      </div>

      {/* ── Noise texture overlay for depth ── */}
      <div className="noise-overlay" />

      {/* ── Cinematic vignette ── */}
      <div className="vignette" />

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
          background: "linear-gradient(135deg, #67e8f9, #06b6d4, #818cf8, #a78bfa, #22d3ee)",
          backgroundSize: "400% 400%",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          animation: "gradient-flow 6s ease infinite",
          filter: "drop-shadow(0 0 8px rgba(6,182,212,0.15))",
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

          <SessionStats
            messages={messages}
            connected={connected}
            pipelineState={pipelineState}
            detectedLang={detectedLang}
            open={statsOpen}
            onToggle={() => setStatsOpen(p => !p)}
          />

          <ConversationInsights
            messages={messages}
            pipelineState={pipelineState}
            open={insightsOpen}
            onToggle={() => setInsightsOpen(p => !p)}
            theme={theme}
          />

          <MessageCounter count={messages.length} theme={theme} />
          <ExportChat messages={messages} theme={theme} />

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
          <VoiceRipples state={pipelineState} lang={detectedLang || "en"} micActive={audioStream.micActive} />
          <ParticleBurst pipelineState={pipelineState} />
          {/* Backchannel pulse */}
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
            {pipelineState !== "thinking" && (
              <motion.div
                animate={{ scale: [1, 1.4, 1], opacity: [0.6, 1, 0.6] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                style={{ width: 6, height: 6, borderRadius: "50%", background: c.accent,
                  boxShadow: `0 0 8px ${c.accent}66` }} />
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

      {/* ═══ SUBTITLE BAR ═══ */}
      <motion.div className="relative w-full max-w-lg px-5 shrink-0" style={{ zIndex: 10, marginBottom: 4 }}
        initial={{ opacity: 0 }} animate={{ opacity: show ? 1 : 0 }} transition={{ delay: 0.5 }}>
        <SubtitleBar
          text={subtitle.text}
          language={subtitle.language}
          isActive={pipelineState === "speaking" && !!subtitle.text}
          theme={theme}
        />
      </motion.div>

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

      {/* ═══ SMART SUGGESTIONS ═══ */}
      {show && modelsReady && (
        <motion.div className="relative w-full max-w-lg px-5 shrink-0" style={{ zIndex: 10 }}
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }}>
          <SmartSuggestions
            messages={messages}
            pipelineState={pipelineState}
            detectedLang={detectedLang}
            onSend={handleSendChat}
            theme={theme}
          />
        </motion.div>
      )}

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
