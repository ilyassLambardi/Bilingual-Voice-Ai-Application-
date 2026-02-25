import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

export default function MicLevelMeter({ micActive, theme = "dark" }) {
  const [level, setLevel] = useState(0);
  const analyserRef = useRef(null);
  const rafRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    if (!micActive) {
      setLevel(0);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      }
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      analyserRef.current = null;
      return;
    }

    let cancelled = false;

    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
      if (cancelled) { stream.getTracks().forEach(t => t.stop()); return; }
      streamRef.current = stream;
      const ctx = new AudioContext();
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.7;
      src.connect(analyser);
      analyserRef.current = analyser;

      const buf = new Uint8Array(analyser.frequencyBinCount);
      const tick = () => {
        if (cancelled) return;
        analyser.getByteFrequencyData(buf);
        const avg = buf.reduce((a, b) => a + b, 0) / buf.length;
        setLevel(Math.min(avg / 80, 1));
        rafRef.current = requestAnimationFrame(tick);
      };
      tick();
    }).catch(() => {});

    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      }
    };
  }, [micActive]);

  const bars = 12;
  const isDark = theme === "dark";

  return (
    <AnimatePresence>
      {micActive && (
        <motion.div
          initial={{ opacity: 0, width: 0 }}
          animate={{ opacity: 1, width: "auto" }}
          exit={{ opacity: 0, width: 0 }}
          transition={{ duration: 0.25 }}
          className="flex items-center"
          style={{ gap: 1.5, height: 20, overflow: "hidden" }}
        >
          {Array.from({ length: bars }, (_, i) => {
            const threshold = i / bars;
            const active = level > threshold;
            const color = i < bars * 0.6
              ? "rgba(16,185,129,0.7)"
              : i < bars * 0.85
                ? "rgba(245,158,11,0.7)"
                : "rgba(239,68,68,0.7)";
            return (
              <motion.div
                key={i}
                animate={{
                  height: active ? 4 + (level - threshold) * 16 : 3,
                  background: active ? color : (isDark ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.08)"),
                }}
                transition={{ duration: 0.05 }}
                style={{
                  width: 2.5,
                  borderRadius: 1,
                  minHeight: 3,
                }}
              />
            );
          })}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
