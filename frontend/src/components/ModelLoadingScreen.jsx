import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const STEPS = [
  { id: "connect", label: "Connecting to server", icon: "🔌" },
  { id: "vad", label: "Voice Activity Detection", icon: "🎙️" },
  { id: "asr", label: "Speech Recognition (Whisper)", icon: "📝" },
  { id: "llm", label: "Language Model", icon: "🧠" },
  { id: "tts", label: "Text-to-Speech Synthesis", icon: "🔊" },
  { id: "ready", label: "All systems ready", icon: "✨" },
];

function ShimmerBar({ progress, color }) {
  return (
    <div style={{
      width: "100%", height: 3, borderRadius: 2,
      background: "rgba(255,255,255,0.04)",
      overflow: "hidden",
    }}>
      <motion.div
        initial={{ width: "0%" }}
        animate={{ width: `${progress}%` }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        style={{
          height: "100%", borderRadius: 2,
          background: `linear-gradient(90deg, ${color}, ${color}88)`,
          boxShadow: `0 0 12px ${color}44`,
        }}
      />
    </div>
  );
}

function FloatingOrb() {
  return (
    <div className="relative" style={{ width: 80, height: 80, margin: "0 auto 32px" }}>
      <motion.div
        className="absolute inset-0 rounded-full"
        animate={{
          scale: [1, 1.15, 1],
          opacity: [0.3, 0.6, 0.3],
        }}
        transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
        style={{
          background: "radial-gradient(circle, rgba(6,182,212,0.15) 0%, transparent 70%)",
          filter: "blur(8px)",
        }}
      />
      <motion.div
        className="absolute inset-0 rounded-full"
        animate={{ rotate: 360 }}
        transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
        style={{
          background: "conic-gradient(from 0deg, rgba(6,182,212,0.2), rgba(168,85,247,0.15), rgba(245,158,11,0.1), rgba(6,182,212,0.2))",
          filter: "blur(2px)",
        }}
      />
      <motion.div
        className="absolute rounded-full"
        animate={{ scale: [0.92, 1, 0.92] }}
        transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
        style={{
          inset: 8,
          background: "radial-gradient(circle at 35% 35%, rgba(6,182,212,0.25), rgba(15,23,42,0.9) 70%)",
          border: "1px solid rgba(6,182,212,0.1)",
          boxShadow: "0 0 30px rgba(6,182,212,0.1), inset 0 0 20px rgba(6,182,212,0.05)",
        }}
      />
    </div>
  );
}

export default function ModelLoadingScreen({ connected, onReady }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [finished, setFinished] = useState(false);

  useEffect(() => {
    if (!connected) {
      setCurrentStep(0);
      return;
    }
    // Simulate model loading progress
    setCurrentStep(1);
    const timers = [
      setTimeout(() => setCurrentStep(2), 800),
      setTimeout(() => setCurrentStep(3), 1800),
      setTimeout(() => setCurrentStep(4), 3200),
      setTimeout(() => setCurrentStep(5), 4200),
      setTimeout(() => { setFinished(true); }, 5000),
    ];
    return () => timers.forEach(clearTimeout);
  }, [connected]);

  useEffect(() => {
    if (finished) {
      const t = setTimeout(() => onReady?.(), 600);
      return () => clearTimeout(t);
    }
  }, [finished, onReady]);

  const progress = Math.round((currentStep / (STEPS.length - 1)) * 100);
  const activeColor = currentStep >= 5 ? "#10b981" : "#06b6d4";

  return (
    <motion.div
      className="fixed inset-0 flex flex-col items-center justify-center"
      style={{
        background: "radial-gradient(ellipse at 50% 30%, rgba(6,182,212,0.04) 0%, #030308 70%)",
        zIndex: 100,
      }}
      exit={{ opacity: 0, scale: 1.02 }}
      transition={{ duration: 0.5 }}
    >
      <FloatingOrb />

      <motion.h1
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        style={{
          fontSize: 18, fontWeight: 700, letterSpacing: "0.15em", textTransform: "uppercase",
          background: "linear-gradient(135deg, #67e8f9, #06b6d4, #a78bfa)",
          backgroundSize: "200% 200%",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          animation: "gradient-flow 4s ease infinite",
          marginBottom: 6,
        }}
      >
        Voice AI
      </motion.h1>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        style={{
          fontSize: 11, color: "rgba(255,255,255,0.2)",
          letterSpacing: "0.08em", marginBottom: 36,
        }}
      >
        Bilingual Speech-to-Speech System
      </motion.p>

      {/* Progress bar */}
      <div style={{ width: 280, marginBottom: 28 }}>
        <ShimmerBar progress={progress} color={activeColor} />
        <div className="flex justify-between" style={{ marginTop: 6 }}>
          <span style={{ fontSize: 9, color: "rgba(255,255,255,0.15)" }}>
            Loading models...
          </span>
          <span style={{ fontSize: 9, color: activeColor, opacity: 0.6 }}>
            {progress}%
          </span>
        </div>
      </div>

      {/* Steps */}
      <div style={{ width: 280 }}>
        {STEPS.map((step, i) => {
          const isActive = i === currentStep;
          const isDone = i < currentStep;
          const isPending = i > currentStep;

          return (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, x: -10 }}
              animate={{
                opacity: isPending ? 0.2 : 1,
                x: 0,
              }}
              transition={{ delay: i * 0.08, duration: 0.3 }}
              className="flex items-center gap-3"
              style={{ padding: "5px 0" }}
            >
              {/* Status indicator */}
              <div className="relative flex items-center justify-center" style={{ width: 20, height: 20 }}>
                {isDone && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", stiffness: 500, damping: 25 }}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  </motion.div>
                )}
                {isActive && (
                  <motion.div
                    className="rounded-full"
                    animate={{ scale: [1, 1.3, 1], opacity: [0.6, 1, 0.6] }}
                    transition={{ duration: 1.2, repeat: Infinity }}
                    style={{
                      width: 8, height: 8,
                      background: activeColor,
                      boxShadow: `0 0 8px ${activeColor}66`,
                    }}
                  />
                )}
                {isPending && (
                  <div className="rounded-full" style={{
                    width: 6, height: 6,
                    background: "rgba(255,255,255,0.08)",
                  }} />
                )}
              </div>

              {/* Label */}
              <span style={{
                fontSize: 11, fontWeight: isActive ? 600 : 400,
                color: isDone ? "rgba(16,185,129,0.6)"
                  : isActive ? "rgba(255,255,255,0.7)"
                  : "rgba(255,255,255,0.15)",
                transition: "color 0.3s",
              }}>
                {step.icon} {step.label}
              </span>

              {/* Shimmer for active step */}
              {isActive && (
                <motion.div
                  animate={{ opacity: [0.3, 0.7, 0.3] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                  style={{
                    marginLeft: "auto",
                    width: 28, height: 3, borderRadius: 2,
                    background: `linear-gradient(90deg, transparent, ${activeColor}44, transparent)`,
                  }}
                />
              )}
            </motion.div>
          );
        })}
      </div>

      {/* Bottom hint */}
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.15 }}
        transition={{ delay: 1.5 }}
        style={{
          position: "absolute", bottom: 24,
          fontSize: 9, color: "rgba(255,255,255,0.3)",
          letterSpacing: "0.06em",
        }}
      >
        {!connected ? "Waiting for server connection..." : finished ? "Launching..." : "Initializing pipeline components..."}
      </motion.p>
    </motion.div>
  );
}
