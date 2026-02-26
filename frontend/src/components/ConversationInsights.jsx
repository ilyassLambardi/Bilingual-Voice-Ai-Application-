import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BarChart3, X, Globe, Clock, MessageSquare, Zap } from "lucide-react";

/**
 * ConversationInsights — slide-out analytics panel showing real-time
 * conversation metrics: word count, language distribution, response
 * times, turn count, and a mini bar chart of language usage over time.
 */

function StatCard({ icon: Icon, label, value, sub, color, isDark }) {
  return (
    <div style={{
      padding: "10px 12px", borderRadius: 10,
      background: isDark ? "rgba(255,255,255,0.02)" : "rgba(255,255,255,0.03)",
      border: isDark ? "1px solid rgba(255,255,255,0.04)" : "1px solid rgba(255,255,255,0.06)",
    }}>
      <div className="flex items-center gap-2" style={{ marginBottom: 6 }}>
        <Icon size={12} style={{ color, opacity: 0.7 }} />
        <span style={{ fontSize: 9, fontWeight: 600, letterSpacing: "0.08em",
          textTransform: "uppercase", color: "rgba(255,255,255,0.3)" }}>
          {label}
        </span>
      </div>
      <div style={{ fontSize: 20, fontWeight: 700, color, lineHeight: 1.1 }}>
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 9, color: "rgba(255,255,255,0.2)", marginTop: 2 }}>
          {sub}
        </div>
      )}
    </div>
  );
}

function LanguageBar({ enPct, dePct, isDark }) {
  return (
    <div>
      <div className="flex items-center justify-between" style={{ marginBottom: 4 }}>
        <span style={{ fontSize: 9, color: "rgba(255,255,255,0.3)", fontWeight: 600,
          letterSpacing: "0.08em", textTransform: "uppercase" }}>
          Language Distribution
        </span>
      </div>
      <div className="flex" style={{
        height: 6, borderRadius: 3, overflow: "hidden",
        background: "rgba(255,255,255,0.04)",
      }}>
        <motion.div
          initial={{ width: "0%" }}
          animate={{ width: `${enPct}%` }}
          transition={{ duration: 0.5 }}
          style={{
            background: "linear-gradient(90deg, #06b6d4, #22d3ee)",
            borderRadius: "3px 0 0 3px",
          }}
        />
        <motion.div
          initial={{ width: "0%" }}
          animate={{ width: `${dePct}%` }}
          transition={{ duration: 0.5 }}
          style={{
            background: "linear-gradient(90deg, #f59e0b, #fbbf24)",
            borderRadius: "0 3px 3px 0",
          }}
        />
      </div>
      <div className="flex justify-between" style={{ marginTop: 4 }}>
        <span style={{ fontSize: 9, color: "#06b6d4", opacity: 0.6 }}>
          🇬🇧 English {enPct}%
        </span>
        <span style={{ fontSize: 9, color: "#f59e0b", opacity: 0.6 }}>
          Deutsch {dePct}% 🇩🇪
        </span>
      </div>
    </div>
  );
}

function MiniTimeline({ messages, isDark }) {
  // Show last 10 messages as small colored dots
  const recent = messages.slice(-10);
  if (recent.length === 0) return null;

  return (
    <div>
      <span style={{ fontSize: 9, color: "rgba(255,255,255,0.3)", fontWeight: 600,
        letterSpacing: "0.08em", textTransform: "uppercase", display: "block", marginBottom: 6 }}>
        Recent Activity
      </span>
      <div className="flex items-center gap-1.5">
        {recent.map((msg, i) => {
          const isUser = msg.role === "user";
          const isDE = msg.language === "de";
          const color = isUser
            ? (isDE ? "rgba(245,158,11,0.5)" : "rgba(6,182,212,0.5)")
            : (isDE ? "rgba(245,158,11,0.3)" : "rgba(6,182,212,0.3)");

          return (
            <motion.div
              key={msg.id ?? i}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: i * 0.03 }}
              style={{
                width: isUser ? 8 : 12,
                height: isUser ? 8 : 12,
                borderRadius: isUser ? "50%" : 3,
                background: color,
                border: `1px solid ${color}`,
              }}
              title={`${msg.role}: ${msg.text?.slice(0, 30)}...`}
            />
          );
        })}
      </div>
    </div>
  );
}

export default function ConversationInsights({ messages = [], pipelineState, open, onToggle, theme = "dark" }) {
  const isDark = theme === "dark";

  const stats = useMemo(() => {
    const userMsgs = messages.filter(m => m.role === "user");
    const aiMsgs = messages.filter(m => m.role === "assistant");
    const allText = messages.map(m => m.text || "").join(" ");
    const wordCount = allText.split(/\s+/).filter(Boolean).length;
    const enMsgs = messages.filter(m => m.language !== "de").length;
    const deMsgs = messages.filter(m => m.language === "de").length;
    const total = enMsgs + deMsgs || 1;
    const enPct = Math.round((enMsgs / total) * 100);
    const dePct = 100 - enPct;

    // Average words per message
    const avgWords = messages.length > 0
      ? Math.round(wordCount / messages.length)
      : 0;

    // Conversation duration
    const firstTs = messages[0]?._ts;
    const lastTs = messages[messages.length - 1]?._ts;
    const durationSec = firstTs && lastTs ? Math.round((lastTs - firstTs) / 1000) : 0;
    const durationStr = durationSec > 60
      ? `${Math.floor(durationSec / 60)}m ${durationSec % 60}s`
      : `${durationSec}s`;

    return {
      turns: messages.length,
      userTurns: userMsgs.length,
      aiTurns: aiMsgs.length,
      wordCount,
      avgWords,
      enPct, dePct,
      duration: durationStr,
    };
  }, [messages]);

  return (
    <>
      {/* Toggle button */}
      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        onClick={onToggle}
        style={{
          width: 30, height: 30, borderRadius: 8,
          background: open ? "rgba(6,182,212,0.1)" : "transparent",
          border: open ? "1px solid rgba(6,182,212,0.15)" : "1px solid transparent",
          display: "flex", alignItems: "center", justifyContent: "center",
          cursor: "pointer", color: isDark ? "rgba(255,255,255,0.2)" : "rgba(255,255,255,0.25)",
          transition: "all 0.3s",
        }}
      >
        <BarChart3 size={14} />
      </motion.button>

      {/* Panel */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 400, damping: 25 }}
            style={{
              position: "absolute", top: 44, right: 0,
              width: 260, padding: 16, borderRadius: 14,
              background: isDark ? "rgba(10,10,20,0.92)" : "rgba(20,20,40,0.92)",
              border: "1px solid rgba(255,255,255,0.06)",
              backdropFilter: "blur(24px)",
              WebkitBackdropFilter: "blur(24px)",
              boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
              zIndex: 50,
            }}
          >
            {/* Header */}
            <div className="flex items-center justify-between" style={{ marginBottom: 14 }}>
              <span style={{
                fontSize: 11, fontWeight: 700, letterSpacing: "0.1em",
                textTransform: "uppercase", color: "rgba(255,255,255,0.4)",
              }}>
                Conversation Insights
              </span>
              <button onClick={onToggle} style={{
                background: "none", border: "none", cursor: "pointer",
                color: "rgba(255,255,255,0.2)",
              }}>
                <X size={12} />
              </button>
            </div>

            {/* Stats grid */}
            <div className="grid grid-cols-2 gap-2" style={{ marginBottom: 14 }}>
              <StatCard icon={MessageSquare} label="Turns" value={stats.turns}
                sub={`${stats.userTurns} you · ${stats.aiTurns} AI`}
                color="#06b6d4" isDark={isDark} />
              <StatCard icon={Zap} label="Words" value={stats.wordCount}
                sub={`~${stats.avgWords} per turn`}
                color="#a78bfa" isDark={isDark} />
              <StatCard icon={Clock} label="Duration" value={stats.duration}
                color="#f59e0b" isDark={isDark} />
              <StatCard icon={Globe} label="Status" value={
                { idle: "Ready", listening: "Listening", thinking: "Thinking", speaking: "Speaking" }[pipelineState] || "—"
              } color="#10b981" isDark={isDark} />
            </div>

            {/* Language bar */}
            <div style={{ marginBottom: 14 }}>
              <LanguageBar enPct={stats.enPct} dePct={stats.dePct} isDark={isDark} />
            </div>

            {/* Mini timeline */}
            <MiniTimeline messages={messages} isDark={isDark} />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
