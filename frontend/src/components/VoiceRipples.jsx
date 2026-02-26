import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect, useRef } from "react";

/**
 * VoiceRipples — concentric expanding rings around the orb that pulse
 * with the user's voice amplitude during listening, and with AI speech
 * energy during speaking. Creates a sonar/radar-like visual effect.
 */
export default function VoiceRipples({ state, lang = "en", micActive = false }) {
  const [ripples, setRipples] = useState([]);
  const idRef = useRef(0);
  const intervalRef = useRef(null);

  const isActive = state === "listening" || state === "speaking";
  const color = lang === "de" ? "245,158,11" : "6,182,212";
  const speed = state === "speaking" ? 1800 : 2200;

  useEffect(() => {
    if (isActive && micActive) {
      // Spawn ripples at regular intervals
      intervalRef.current = setInterval(() => {
        const id = ++idRef.current;
        setRipples(prev => [...prev.slice(-4), { id, ts: Date.now() }]);
      }, state === "speaking" ? 600 : 800);
    } else {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [isActive, micActive, state]);

  // Clean up old ripples
  useEffect(() => {
    if (ripples.length === 0) return;
    const timer = setTimeout(() => {
      setRipples(prev => prev.filter(r => Date.now() - r.ts < speed + 200));
    }, speed + 300);
    return () => clearTimeout(timer);
  }, [ripples, speed]);

  return (
    <div className="absolute inset-0 pointer-events-none flex items-center justify-center"
      style={{ zIndex: 0 }}>
      <AnimatePresence>
        {ripples.map((r, i) => (
          <motion.div
            key={r.id}
            className="absolute rounded-full"
            initial={{ width: 140, height: 140, opacity: 0.35, borderWidth: 1.5 }}
            animate={{ width: 320, height: 320, opacity: 0, borderWidth: 0.5 }}
            exit={{ opacity: 0 }}
            transition={{ duration: speed / 1000, ease: "easeOut" }}
            style={{
              borderStyle: "solid",
              borderColor: `rgba(${color}, 0.25)`,
              boxShadow: `0 0 8px rgba(${color}, 0.06)`,
            }}
          />
        ))}
      </AnimatePresence>

      {/* Static inner ring — always visible when active */}
      {isActive && micActive && (
        <motion.div
          className="absolute rounded-full"
          animate={{
            scale: [1, 1.04, 1],
            opacity: [0.15, 0.25, 0.15],
          }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
          style={{
            width: 150, height: 150,
            border: `1px solid rgba(${color}, 0.12)`,
          }}
        />
      )}
    </div>
  );
}
