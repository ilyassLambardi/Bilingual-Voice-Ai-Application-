import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";

/**
 * Circular energy meter that shows real-time audio energy level.
 * Uses SVG arc that fills based on FFT energy.
 */
export default function EnergyMeter({ active = false, size = 32, theme = "dark" }) {
  const [energy, setEnergy] = useState(0);
  const rafRef = useRef(null);
  const isDark = theme === "dark";

  useEffect(() => {
    const tick = () => {
      const e = window.__fftEnergy || 0;
      setEnergy(prev => prev + (e - prev) * 0.2);
      rafRef.current = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  const r = (size - 4) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - Math.min(energy, 1));

  const color = energy > 0.6 ? "rgba(239,68,68,0.6)" :
                energy > 0.3 ? "rgba(245,158,11,0.6)" :
                "rgba(6,182,212,0.5)";

  return (
    <motion.div
      animate={{ opacity: active ? 1 : 0.25 }}
      transition={{ duration: 0.4 }}
      style={{ width: size, height: size, position: "relative" }}
    >
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Background circle */}
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none"
          stroke={isDark ? "rgba(255,255,255,0.04)" : "rgba(255,255,255,0.06)"}
          strokeWidth={2}
        />
        {/* Energy arc */}
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none"
          stroke={color}
          strokeWidth={2}
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: "stroke-dashoffset 0.1s, stroke 0.3s" }}
        />
      </svg>
      {/* Center value */}
      <div style={{
        position: "absolute", inset: 0,
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 7, fontWeight: 700,
        fontFamily: "'JetBrains Mono', monospace",
        color: isDark ? "rgba(255,255,255,0.2)" : "rgba(255,255,255,0.25)",
      }}>
        {Math.round(energy * 100)}
      </div>
    </motion.div>
  );
}
