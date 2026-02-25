import { useRef, useEffect } from "react";

/**
 * Real-time audio waveform/spectrum scope.
 * Reads from window.__fftSmoothed (set by useAudioStream).
 */
export default function AudioScope({ active = false, mode = "wave", theme = "dark", width = 180, height = 36 }) {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);

  useEffect(() => {
    const cvs = canvasRef.current;
    if (!cvs) return;
    const ctx = cvs.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    cvs.width = width * dpr;
    cvs.height = height * dpr;
    ctx.scale(dpr, dpr);

    const draw = () => {
      ctx.clearRect(0, 0, width, height);
      const fft = window.__fftSmoothed;
      if (!fft || !active) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const n = Math.min(fft.length, 64);

      if (mode === "bars") {
        // Spectrum bars
        const barW = width / n;
        for (let i = 0; i < n; i++) {
          const v = fft[i] || 0;
          const h = v * height * 0.9;
          const hue = 180 + i * (60 / n); // cyan → blue gradient
          ctx.fillStyle = `hsla(${hue}, 80%, 60%, ${0.3 + v * 0.5})`;
          ctx.beginPath();
          ctx.roundRect(i * barW + 1, height - h, Math.max(barW - 2, 1), h, 1);
          ctx.fill();
        }
      } else {
        // Waveform
        ctx.beginPath();
        ctx.strokeStyle = theme === "dark" ? "rgba(6,182,212,0.4)" : "rgba(6,182,212,0.5)";
        ctx.lineWidth = 1.5;
        const step = width / n;
        for (let i = 0; i < n; i++) {
          const v = fft[i] || 0;
          const y = height / 2 - v * height * 0.4;
          if (i === 0) ctx.moveTo(0, y);
          else ctx.lineTo(i * step, y);
        }
        ctx.stroke();

        // Mirror
        ctx.beginPath();
        ctx.strokeStyle = theme === "dark" ? "rgba(244,114,182,0.2)" : "rgba(244,114,182,0.25)";
        ctx.lineWidth = 1;
        for (let i = 0; i < n; i++) {
          const v = fft[i] || 0;
          const y = height / 2 + v * height * 0.35;
          if (i === 0) ctx.moveTo(0, y);
          else ctx.lineTo(i * step, y);
        }
        ctx.stroke();
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(rafRef.current);
  }, [active, mode, theme, width, height]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width, height,
        opacity: active ? 1 : 0.2,
        transition: "opacity 0.5s",
        borderRadius: 8,
      }}
    />
  );
}
