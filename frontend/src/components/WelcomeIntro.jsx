import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

const STEPS = [
  { key: "VAD", label: "Voice Activity", icon: "M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z M19 10v2a7 7 0 0 1-14 0v-2", color: "#06b6d4" },
  { key: "ASR", label: "Speech Recognition", icon: "M4 7h16 M4 12h10 M4 17h14", color: "#818cf8" },
  { key: "LLM", label: "Language Model", icon: "M12 2L2 7l10 5 10-5-10-5z M2 17l10 5 10-5 M2 12l10 5 10-5", color: "#f59e0b" },
  { key: "TTS", label: "Speech Synthesis", icon: "M11 5L6 9H2v6h4l5 4V5z M15.54 8.46a5 5 0 0 1 0 7.07 M19.07 4.93a10 10 0 0 1 0 14.14", color: "#10b981" },
];

const TECH_STACK = [
  { label: "Groq Whisper", color: "#06b6d4" },
  { label: "Llama 3.3 70B", color: "#818cf8" },
  { label: "Edge Neural TTS", color: "#10b981" },
  { label: "Silero VAD", color: "#f472b6" },
  { label: "React", color: "#61dafb" },
  { label: "WebGL", color: "#f59e0b" },
];

/* ── Constellation Particles — connected by proximity lines ── */
function ConstellationField() {
  const canvasRef = useRef(null);
  const particles = useRef([]);
  const raf = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = () => canvas.width;
    const H = () => canvas.height;

    const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
    resize();
    window.addEventListener("resize", resize);

    // Init particles
    particles.current = Array.from({ length: 60 }, () => ({
      x: Math.random() * W(), y: Math.random() * H(),
      vx: (Math.random() - 0.5) * 0.3, vy: (Math.random() - 0.5) * 0.3,
      r: Math.random() * 1.5 + 0.5,
      o: Math.random() * 0.3 + 0.05,
      hue: Math.random() > 0.5 ? 185 : 260,
    }));

    const draw = () => {
      ctx.clearRect(0, 0, W(), H());
      const pts = particles.current;
      for (const p of pts) {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > W()) p.vx *= -1;
        if (p.y < 0 || p.y > H()) p.vy *= -1;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${p.hue},70%,65%,${p.o})`;
        ctx.fill();
      }
      // Draw connections
      for (let i = 0; i < pts.length; i++) {
        for (let j = i + 1; j < pts.length; j++) {
          const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            ctx.beginPath();
            ctx.moveTo(pts[i].x, pts[i].y);
            ctx.lineTo(pts[j].x, pts[j].y);
            ctx.strokeStyle = `rgba(6,182,212,${0.06 * (1 - dist / 120)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }
      raf.current = requestAnimationFrame(draw);
    };
    draw();
    return () => { cancelAnimationFrame(raf.current); window.removeEventListener("resize", resize); };
  }, []);

  return <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" style={{ opacity: 0.7 }} />;
}

/* ── Typewriter effect ── */
function TypewriterText({ text, delay = 0, speed = 35 }) {
  const [displayed, setDisplayed] = useState("");
  useEffect(() => {
    let i = 0;
    const t = setTimeout(() => {
      const iv = setInterval(() => {
        i++;
        setDisplayed(text.slice(0, i));
        if (i >= text.length) clearInterval(iv);
      }, speed);
      return () => clearInterval(iv);
    }, delay);
    return () => clearTimeout(t);
  }, [text, delay, speed]);
  return <>{displayed}<span style={{ opacity: displayed.length < text.length ? 1 : 0, animation: "blink-cursor 0.7s step-end infinite" }}>|</span></>;
}

/* ── Scanning horizontal line ── */
function ScanLine() {
  return (
    <motion.div
      className="absolute left-0 right-0 pointer-events-none"
      initial={{ top: "-2%" }}
      animate={{ top: "102%" }}
      transition={{ duration: 6, repeat: Infinity, ease: "linear", repeatDelay: 2 }}
      style={{ height: 1, background: "linear-gradient(to right, transparent, rgba(6,182,212,0.08), rgba(129,140,248,0.06), transparent)", zIndex: 2 }}
    />
  );
}

export default function WelcomeIntro({ onDone }) {
  const [phase, setPhase] = useState(0);
  const [exiting, setExiting] = useState(false);
  const [activeStep, setActiveStep] = useState(-1);
  const [dataFlowIdx, setDataFlowIdx] = useState(-1);

  useEffect(() => {
    const t = [
      setTimeout(() => setPhase(1), 600),
      setTimeout(() => setPhase(2), 1500),
      setTimeout(() => setPhase(3), 2600),
      setTimeout(() => setPhase(4), 4200),
      setTimeout(() => setPhase(5), 5600),
    ];
    return () => t.forEach(clearTimeout);
  }, []);

  // Animate pipeline steps sequentially with data flow
  useEffect(() => {
    if (phase < 3) return;
    let step = 0;
    const iv = setInterval(() => {
      setActiveStep(step);
      setDataFlowIdx(step);
      step++;
      if (step >= STEPS.length) {
        clearInterval(iv);
        setTimeout(() => { setActiveStep(-1); setDataFlowIdx(-1); }, 900);
      }
    }, 350);
    return () => clearInterval(iv);
  }, [phase]);

  const handleEnter = useCallback(() => {
    if (exiting) return;
    setExiting(true);
    setTimeout(onDone, 900);
  }, [onDone, exiting]);

  useEffect(() => {
    if (phase < 5) return;
    const t = setTimeout(handleEnter, 2600);
    return () => clearTimeout(t);
  }, [phase, handleEnter]);

  return (
    <motion.div
      className="fixed inset-0 flex flex-col items-center justify-center overflow-hidden"
      style={{
        zIndex: 100,
        background: "linear-gradient(145deg, #020208 0%, #04061a 30%, #0a0418 60%, #060212 100%)",
        cursor: phase >= 5 ? "pointer" : "default",
      }}
      onClick={phase >= 5 ? handleEnter : undefined}
      animate={exiting ? { opacity: 0, scale: 1.03, filter: "blur(8px)" } : {}}
      transition={{ duration: 0.9, ease: [0.4, 0, 0.2, 1] }}
    >
      {/* ── Constellation particle field ── */}
      <ConstellationField />

      {/* ── Aurora blobs ── */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <motion.div className="absolute rounded-full"
          animate={{ x: [0, 40, -30, 0], y: [0, -30, 20, 0], scale: [1, 1.15, 0.9, 1] }}
          transition={{ duration: 22, repeat: Infinity, ease: "easeInOut" }}
          style={{ width: 800, height: 800, top: "30%", left: "50%", marginTop: -400, marginLeft: -400,
            background: "radial-gradient(circle, rgba(6,182,212,0.07) 0%, rgba(99,102,241,0.04) 35%, transparent 65%)", filter: "blur(80px)" }} />
        <motion.div className="absolute rounded-full"
          animate={{ x: [0, -30, 20, 0], y: [0, 25, -30, 0] }}
          transition={{ duration: 28, repeat: Infinity, ease: "easeInOut" }}
          style={{ width: 600, height: 600, top: "15%", left: "20%",
            background: "radial-gradient(circle, rgba(168,85,247,0.05) 0%, rgba(244,114,182,0.03) 40%, transparent 60%)", filter: "blur(70px)" }} />
        <motion.div className="absolute rounded-full"
          animate={{ x: [0, 25, -20, 0], y: [0, -20, 25, 0] }}
          transition={{ duration: 24, repeat: Infinity, ease: "easeInOut" }}
          style={{ width: 500, height: 500, bottom: "0%", right: "5%",
            background: "radial-gradient(circle, rgba(245,158,11,0.04) 0%, rgba(239,68,68,0.02) 40%, transparent 55%)", filter: "blur(65px)" }} />
        {/* Extra violet bloom */}
        <motion.div className="absolute rounded-full"
          animate={{ scale: [1, 1.08, 1], opacity: [0.4, 0.7, 0.4] }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
          style={{ width: 350, height: 350, top: "60%", left: "65%",
            background: "radial-gradient(circle, rgba(139,92,246,0.04) 0%, transparent 55%)", filter: "blur(50px)" }} />
      </div>

      <ScanLine />

      {/* ── Hexagonal grid ── */}
      <div className="absolute inset-0 pointer-events-none" style={{
        backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='56' height='100'%3E%3Cpath d='M28 66L0 50L0 16L28 0L56 16L56 50L28 66L28 100' fill='none' stroke='rgba(6,182,212,0.018)' stroke-width='0.5'/%3E%3C/svg%3E")`,
        backgroundSize: "56px 100px",
        maskImage: "radial-gradient(ellipse at center, black 20%, transparent 65%)",
        WebkitMaskImage: "radial-gradient(ellipse at center, black 20%, transparent 65%)",
      }} />

      {/* Central content */}
      <div className="relative flex flex-col items-center" style={{ maxWidth: 600 }}>

        {/* ── Orb ── */}
        <motion.div
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 1.4, ease: [0.16, 1, 0.3, 1] }}
          className="relative flex items-center justify-center"
          style={{ width: 180, height: 180, marginBottom: 36 }}
        >
          {/* Orbital ring 1 — tilted */}
          <motion.div className="absolute" animate={{ rotate: 360 }}
            transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
            style={{ width: 180, height: 180, borderRadius: "50%",
              border: "1px solid transparent", borderTopColor: "rgba(6,182,212,0.15)", borderRightColor: "rgba(129,140,248,0.08)",
              transform: "rotateX(70deg) rotateZ(0deg)" }} />
          {/* Orbital ring 2 — counter */}
          <motion.div className="absolute" animate={{ rotate: -360 }}
            transition={{ duration: 18, repeat: Infinity, ease: "linear" }}
            style={{ width: 150, height: 150, borderRadius: "50%",
              border: "1px solid transparent", borderBottomColor: "rgba(168,85,247,0.12)", borderLeftColor: "rgba(244,114,182,0.06)" }} />
          {/* Orbital ring 3 — fast inner */}
          <motion.div className="absolute" animate={{ rotate: 360 }}
            transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
            style={{ width: 120, height: 120, borderRadius: "50%",
              border: "1px solid transparent", borderTopColor: "rgba(245,158,11,0.1)", borderLeftColor: "rgba(16,185,129,0.05)" }} />

          {/* Breathing pulse halo */}
          <motion.div className="absolute rounded-full"
            animate={{ scale: [1, 1.2, 1], opacity: [0.1, 0.3, 0.1] }}
            transition={{ duration: 3.5, repeat: Infinity, ease: "easeInOut" }}
            style={{ width: 130, height: 130,
              background: "radial-gradient(circle, rgba(6,182,212,0.1) 0%, rgba(129,140,248,0.04) 50%, transparent 70%)" }} />

          {/* Core orb — richer gradient */}
          <div className="rounded-full" style={{
            width: 92, height: 92,
            background: `
              radial-gradient(circle at 30% 25%, rgba(6,182,212,0.35) 0%, transparent 45%),
              radial-gradient(circle at 70% 75%, rgba(129,140,248,0.2) 0%, transparent 45%),
              radial-gradient(circle at 50% 50%, rgba(168,85,247,0.08) 0%, transparent 55%),
              radial-gradient(circle, rgba(8,10,30,0.97) 35%, rgba(6,182,212,0.06) 100%)
            `,
            border: "1px solid rgba(6,182,212,0.12)",
            boxShadow: "0 0 60px rgba(6,182,212,0.08), 0 0 120px rgba(129,140,248,0.04), 0 0 20px rgba(168,85,247,0.04), inset 0 0 40px rgba(6,182,212,0.05)",
          }} />

          {/* Specular highlights */}
          <motion.div className="absolute rounded-full"
            animate={{ opacity: [0.3, 0.8, 0.3] }}
            transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
            style={{ width: 4, height: 4, top: "28%", left: "36%",
              background: "rgba(6,182,212,0.8)", boxShadow: "0 0 12px rgba(6,182,212,0.6)", filter: "blur(1px)" }} />
          <motion.div className="absolute rounded-full"
            animate={{ opacity: [0.15, 0.45, 0.15] }}
            transition={{ duration: 3.5, repeat: Infinity, ease: "easeInOut", delay: 1 }}
            style={{ width: 3, height: 3, top: "40%", left: "62%",
              background: "rgba(168,85,247,0.7)", boxShadow: "0 0 8px rgba(168,85,247,0.5)", filter: "blur(1px)" }} />

          {/* Orbiting dot */}
          <motion.div className="absolute" animate={{ rotate: 360 }}
            transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
            style={{ width: 160, height: 160 }}>
            <div className="absolute rounded-full" style={{ width: 4, height: 4, top: -2, left: "50%", marginLeft: -2,
              background: "#06b6d4", boxShadow: "0 0 10px rgba(6,182,212,0.6), 0 0 20px rgba(6,182,212,0.3)" }} />
          </motion.div>
        </motion.div>

        {/* ── Badge ── */}
        {phase >= 1 && (
          <motion.div initial={{ opacity: 0, y: 16, filter: "blur(8px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            style={{ marginBottom: 14 }}>
            <span className="relative overflow-hidden" style={{
              fontSize: 10, fontWeight: 600, letterSpacing: "0.22em", textTransform: "uppercase",
              color: "rgba(6,182,212,0.55)", padding: "5px 18px", borderRadius: 20,
              background: "rgba(6,182,212,0.04)", border: "1px solid rgba(6,182,212,0.1)",
              backdropFilter: "blur(12px)", display: "inline-block",
            }}>
              Final Year Thesis
              {/* Shimmer sweep */}
              <span className="absolute inset-0" style={{
                background: "linear-gradient(105deg, transparent 40%, rgba(6,182,212,0.08) 50%, transparent 60%)",
                animation: "shimmer-sweep 3s ease-in-out infinite",
              }} />
            </span>
          </motion.div>
        )}

        {/* ── Title ── */}
        {phase >= 1 && (
          <motion.h1
            initial={{ opacity: 0, y: 24, filter: "blur(12px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 1, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
            style={{
              fontSize: 46, fontWeight: 800, letterSpacing: "-0.025em",
              textAlign: "center", lineHeight: 1.05, marginBottom: 8,
              background: "linear-gradient(135deg, rgba(255,255,255,0.97) 0%, rgba(6,182,212,0.85) 30%, rgba(129,140,248,0.8) 60%, rgba(244,114,182,0.7) 100%)",
              backgroundSize: "300% 300%",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
              animation: "gradient-flow 8s ease infinite",
            }}
          >
            Bilingual Voice AI
          </motion.h1>
        )}

        {/* ── Subtitle (typewriter) ── */}
        {phase >= 2 && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="flex flex-col items-center" style={{ marginBottom: 10 }}>
            <p style={{ fontSize: 14, fontWeight: 400, lineHeight: 1.7,
              color: "rgba(255,255,255,0.4)", textAlign: "center", maxWidth: 400,
              fontFamily: "'Space Grotesk', sans-serif" }}>
              <TypewriterText
                text="Real-time speech-to-speech conversational AI with seamless EN ↔ DE language switching"
                delay={100} speed={28}
              />
            </p>
            <motion.div className="flex items-center gap-3"
              initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.5, duration: 0.6 }}
              style={{ marginTop: 12 }}>
              <div className="flex items-center gap-1.5">
                <span style={{ fontSize: 13 }}>🇬🇧</span>
                <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.06em", color: "rgba(6,182,212,0.5)" }}>English</span>
              </div>
              <div style={{ width: 32, height: 1, background: "linear-gradient(to right, rgba(6,182,212,0.25), rgba(129,140,248,0.15), rgba(245,158,11,0.25))" }} />
              <div className="flex items-center gap-1.5">
                <span style={{ fontSize: 13 }}>🇩🇪</span>
                <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.06em", color: "rgba(245,158,11,0.5)" }}>Deutsch</span>
              </div>
            </motion.div>
          </motion.div>
        )}

        {/* ── Pipeline Architecture ── */}
        {phase >= 3 && (
          <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="flex items-center justify-center gap-1"
            style={{ marginTop: 20, marginBottom: 16 }}>
            {STEPS.map((step, i) => {
              const isActive = activeStep === i;
              const isPast = activeStep > i;
              const hasFlow = dataFlowIdx === i;
              return (
                <div key={step.key} className="flex items-center">
                  <motion.div initial={{ opacity: 0, scale: 0.6, y: 12 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    transition={{ delay: i * 0.12, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                    className="flex flex-col items-center" style={{ minWidth: 72 }}>
                    <motion.div
                      animate={{
                        boxShadow: isActive ? `0 0 24px ${step.color}40, 0 0 48px ${step.color}15` : "0 0 0px transparent",
                        borderColor: isActive ? `${step.color}55` : isPast ? `${step.color}28` : "rgba(255,255,255,0.06)",
                        background: isActive ? `${step.color}0d` : "rgba(255,255,255,0.015)",
                      }}
                      transition={{ duration: 0.35 }}
                      className="relative flex items-center justify-center rounded-xl overflow-hidden"
                      style={{ width: 44, height: 44, marginBottom: 6, border: "1px solid rgba(255,255,255,0.06)" }}>
                      <svg width="17" height="17" viewBox="0 0 24 24" fill="none"
                        stroke={isActive || isPast ? step.color : "rgba(255,255,255,0.15)"}
                        strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
                        style={{ transition: "stroke 0.3s", opacity: isActive ? 1 : isPast ? 0.7 : 0.35, position: "relative", zIndex: 1 }}>
                        <path d={step.icon} />
                      </svg>
                      {/* Active shimmer fill */}
                      {isActive && <motion.div className="absolute inset-0"
                        initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                        style={{ background: `linear-gradient(135deg, ${step.color}08, ${step.color}15, ${step.color}08)`,
                          animation: "shimmer-sweep 1.5s ease-in-out infinite" }} />}
                    </motion.div>
                    <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.06em",
                      color: isActive ? step.color : isPast ? `${step.color}90` : "rgba(255,255,255,0.2)",
                      transition: "color 0.3s", fontFamily: "'JetBrains Mono', 'Space Grotesk', monospace" }}>
                      {step.key}
                    </span>
                    <span style={{ fontSize: 7.5, fontWeight: 400, letterSpacing: "0.02em",
                      color: isActive ? "rgba(255,255,255,0.4)" : "rgba(255,255,255,0.12)",
                      transition: "color 0.3s", marginTop: 1 }}>
                      {step.label}
                    </span>
                  </motion.div>
                  {/* Connector with data flow pulse */}
                  {i < STEPS.length - 1 && (
                    <div className="relative" style={{ width: 22, height: 1, marginBottom: 22, marginLeft: 1, marginRight: 1 }}>
                      <motion.div initial={{ scaleX: 0 }} animate={{ scaleX: 1 }}
                        transition={{ delay: i * 0.12 + 0.35, duration: 0.5 }}
                        style={{ position: "absolute", inset: 0,
                          background: isPast ? `linear-gradient(to right, ${step.color}50, ${STEPS[i+1].color}50)` : "rgba(255,255,255,0.06)",
                          transition: "background 0.3s" }} />
                      {/* Flowing data dot */}
                      {hasFlow && (
                        <motion.div className="absolute rounded-full"
                          initial={{ left: 0, opacity: 0 }}
                          animate={{ left: "100%", opacity: [0, 1, 1, 0] }}
                          transition={{ duration: 0.35, ease: "easeInOut" }}
                          style={{ top: -2, width: 5, height: 5, background: step.color,
                            boxShadow: `0 0 8px ${step.color}88`, marginLeft: -2.5 }} />
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </motion.div>
        )}

        {/* ── Tech Stack Badges ── */}
        {phase >= 4 && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            transition={{ duration: 0.6 }}
            className="flex flex-wrap items-center justify-center gap-2"
            style={{ marginTop: 8, marginBottom: 16, maxWidth: 420 }}>
            {TECH_STACK.map((tech, i) => (
              <motion.span key={tech.label}
                initial={{ opacity: 0, scale: 0.7, y: 10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                transition={{ delay: i * 0.08, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
                className="relative overflow-hidden"
                style={{
                  fontSize: 9, fontWeight: 600, letterSpacing: "0.04em",
                  color: `${tech.color}aa`, padding: "3px 10px", borderRadius: 12,
                  background: `${tech.color}08`, border: `1px solid ${tech.color}18`,
                  display: "inline-block",
                }}>
                {tech.label}
                <span className="absolute inset-0" style={{
                  background: `linear-gradient(105deg, transparent 35%, ${tech.color}10 50%, transparent 65%)`,
                  animation: `shimmer-sweep ${2.5 + i * 0.3}s ease-in-out ${i * 0.2}s infinite`,
                }} />
              </motion.span>
            ))}
          </motion.div>
        )}

        {/* ── Enter Button ── */}
        {phase >= 5 && (
          <motion.button
            initial={{ opacity: 0, y: 12, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
            whileHover={{ scale: 1.06, boxShadow: "0 0 40px rgba(6,182,212,0.18), 0 0 80px rgba(129,140,248,0.08)" }}
            whileTap={{ scale: 0.96 }}
            onClick={handleEnter}
            className="relative overflow-hidden"
            style={{
              padding: "11px 36px", borderRadius: 28,
              background: "linear-gradient(135deg, rgba(6,182,212,0.12) 0%, rgba(129,140,248,0.1) 50%, rgba(168,85,247,0.08) 100%)",
              border: "1px solid rgba(6,182,212,0.18)",
              color: "rgba(255,255,255,0.75)",
              fontSize: 12, fontWeight: 600, letterSpacing: "0.14em", textTransform: "uppercase",
              cursor: "pointer", backdropFilter: "blur(16px)",
            }}
          >
            Start Conversation
            <span className="absolute inset-0" style={{
              background: "linear-gradient(105deg, transparent 30%, rgba(255,255,255,0.04) 50%, transparent 70%)",
              animation: "shimmer-sweep 2.5s ease-in-out infinite",
            }} />
          </motion.button>
        )}
      </div>

      {/* ── Footer ── */}
      <motion.div
        className="absolute bottom-0 left-0 right-0 flex flex-col items-center pb-5 gap-2"
        initial={{ opacity: 0 }} animate={{ opacity: phase >= 2 ? 1 : 0 }}
        transition={{ duration: 0.8, delay: 0.5 }}
      >
        <div style={{ width: 60, height: 1, background: "linear-gradient(to right, transparent, rgba(6,182,212,0.12), transparent)", marginBottom: 4 }} />
        <span style={{ fontSize: 10, fontWeight: 400, letterSpacing: "0.05em", color: "rgba(255,255,255,0.12)" }}>
          Designed & Developed by{" "}
          <span style={{
            fontWeight: 700,
            background: "linear-gradient(135deg, rgba(6,182,212,0.5), rgba(129,140,248,0.4))",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          }}>Ilyass</span>
        </span>
      </motion.div>
    </motion.div>
  );
}
