import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

/**
 * Animated particle burst that fires on pipeline state transitions.
 * Spawns a ring of tiny dots that fly outward and fade.
 */
const PARTICLE_COUNT = 12;

function Burst({ color, id }) {
  const particles = useRef(
    Array.from({ length: PARTICLE_COUNT }, (_, i) => {
      const angle = (i / PARTICLE_COUNT) * Math.PI * 2;
      const dist = 60 + Math.random() * 40;
      return {
        x: Math.cos(angle) * dist,
        y: Math.sin(angle) * dist,
        size: 2 + Math.random() * 3,
        delay: Math.random() * 0.15,
      };
    })
  ).current;

  return (
    <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
      {particles.map((p, i) => (
        <motion.div
          key={`${id}-${i}`}
          initial={{ x: 0, y: 0, opacity: 0.8, scale: 1 }}
          animate={{ x: p.x, y: p.y, opacity: 0, scale: 0.3 }}
          transition={{ duration: 0.7, delay: p.delay, ease: "easeOut" }}
          style={{
            position: "absolute",
            width: p.size, height: p.size,
            borderRadius: "50%",
            background: color,
            boxShadow: `0 0 6px ${color}`,
          }}
        />
      ))}
    </div>
  );
}

export default function ParticleBurst({ pipelineState }) {
  const [bursts, setBursts] = useState([]);
  const prevState = useRef(pipelineState);
  const idRef = useRef(0);

  const COLOR_MAP = {
    listening: "rgba(244,114,182,0.7)",
    thinking: "rgba(245,158,11,0.6)",
    speaking: "rgba(16,185,129,0.7)",
    idle: "rgba(6,182,212,0.5)",
  };

  useEffect(() => {
    if (pipelineState !== prevState.current && pipelineState !== "idle") {
      const id = ++idRef.current;
      const color = COLOR_MAP[pipelineState] || COLOR_MAP.idle;
      setBursts(prev => [...prev, { id, color }]);
      // Clean up old bursts after animation
      setTimeout(() => {
        setBursts(prev => prev.filter(b => b.id !== id));
      }, 1200);
    }
    prevState.current = pipelineState;
  }, [pipelineState]);

  return (
    <div className="absolute inset-0 pointer-events-none overflow-visible">
      <AnimatePresence>
        {bursts.map(b => (
          <Burst key={b.id} id={b.id} color={b.color} />
        ))}
      </AnimatePresence>
    </div>
  );
}
