import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const ACTIONS = [
  {
    id: "clear", label: "Clear",
    icon: "M3 6h18 M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2",
    c: "239,68,68", angle: 0,
  },
  {
    id: "shortcuts", label: "Keys",
    icon: "M18 3a3 3 0 00-3 3v12a3 3 0 003 3 3 3 0 003-3 3 3 0 00-3-3H6a3 3 0 00-3 3 3 3 0 003 3 3 3 0 003-3V6a3 3 0 00-3-3",
    c: "168,85,247", angle: 72,
  },
  {
    id: "settings", label: "Settings",
    icon: "M12 15a3 3 0 100-6 3 3 0 000 6z M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 01-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09a1.65 1.65 0 00-1-1.51 1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09a1.65 1.65 0 001.51-1 1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9c.26.604.852.997 1.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z",
    c: "6,182,212", angle: 144,
  },
  {
    id: "stats", label: "Stats",
    icon: "M18 20V10 M12 20V4 M6 20v-6",
    c: "16,185,129", angle: 216,
  },
  {
    id: "theme", label: "Theme",
    icon: "M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z",
    c: "245,158,11", angle: 288,
  },
];

export default function QuickActions({ onAction, theme = "dark" }) {
  const [open, setOpen] = useState(false);
  const isDark = theme === "dark";
  const radius = 56;

  return (
    <div className="fixed" style={{ bottom: 20, left: 20, zIndex: 50 }}>
      {/* Action items */}
      <AnimatePresence>
        {open && ACTIONS.map((action, i) => {
          const angleRad = ((action.angle - 90) * Math.PI) / 180;
          const x = Math.cos(angleRad) * radius;
          const y = Math.sin(angleRad) * radius;
          return (
            <motion.button
              key={action.id}
              initial={{ opacity: 0, x: 0, y: 0, scale: 0.3 }}
              animate={{ opacity: 1, x, y: y - radius + 10, scale: 1 }}
              exit={{ opacity: 0, x: 0, y: 0, scale: 0.3 }}
              transition={{ duration: 0.25, delay: i * 0.03, ease: [0.16, 1, 0.3, 1] }}
              whileHover={{ scale: 1.15, boxShadow: `0 0 16px rgba(${action.c},0.3)` }}
              whileTap={{ scale: 0.9 }}
              onClick={() => { onAction(action.id); setOpen(false); }}
              className="absolute flex items-center justify-center"
              style={{
                width: 34, height: 34, borderRadius: 10,
                background: `rgba(${action.c}, 0.08)`,
                border: `1px solid rgba(${action.c}, 0.15)`,
                cursor: "pointer",
                backdropFilter: "blur(8px)",
              }}
              title={action.label}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                stroke={`rgba(${action.c}, 0.7)`} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d={action.icon} />
              </svg>
            </motion.button>
          );
        })}
      </AnimatePresence>

      {/* Main FAB */}
      <motion.button
        animate={{ rotate: open ? 45 : 0 }}
        whileHover={{ scale: 1.08, boxShadow: "0 0 20px rgba(6,182,212,0.2)" }}
        whileTap={{ scale: 0.92 }}
        onClick={() => setOpen(o => !o)}
        style={{
          width: 40, height: 40, borderRadius: 12,
          background: open
            ? "rgba(6,182,212,0.12)"
            : isDark ? "rgba(255,255,255,0.03)" : "rgba(255,255,255,0.05)",
          border: open
            ? "1px solid rgba(6,182,212,0.2)"
            : isDark ? "1px solid rgba(255,255,255,0.05)" : "1px solid rgba(255,255,255,0.07)",
          cursor: "pointer",
          display: "flex", alignItems: "center", justifyContent: "center",
          backdropFilter: "blur(8px)",
          boxShadow: "0 2px 16px rgba(0,0,0,0.15)",
          transition: "background 0.3s, border-color 0.3s",
          position: "relative", zIndex: 2,
        }}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
          stroke={open ? "rgba(6,182,212,0.8)" : (isDark ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.3)")}
          strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <line x1="12" y1="5" x2="12" y2="19" />
          <line x1="5" y1="12" x2="19" y2="12" />
        </svg>
      </motion.button>
    </div>
  );
}
