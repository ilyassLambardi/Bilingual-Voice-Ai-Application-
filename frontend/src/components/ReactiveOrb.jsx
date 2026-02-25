import { motion, AnimatePresence } from "framer-motion";

const STATES = {
  idle: {
    gradient: "from-violet-600/30 via-indigo-500/20 to-blue-600/30",
    glow: "shadow-[0_0_60px_rgba(99,102,241,0.15)]",
    ring: "border-white/[0.06]",
    label: "Ready",
    coreColor: "bg-gradient-to-br from-indigo-400/30 to-violet-500/30",
  },
  listening: {
    gradient: "from-cyan-400/40 via-sky-500/30 to-blue-500/40",
    glow: "shadow-[0_0_80px_rgba(34,211,238,0.25)]",
    ring: "border-cyan-400/30",
    label: "Listening",
    coreColor: "bg-gradient-to-br from-cyan-400/40 to-sky-500/40",
  },
  thinking: {
    gradient: "from-amber-400/30 via-orange-400/20 to-yellow-500/30",
    glow: "shadow-[0_0_80px_rgba(251,191,36,0.2)]",
    ring: "border-amber-400/20",
    label: "Processing",
    coreColor: "bg-gradient-to-br from-amber-400/30 to-orange-400/30",
  },
  speaking: {
    gradient: "from-emerald-400/40 via-green-400/30 to-teal-500/40",
    glow: "shadow-[0_0_100px_rgba(52,211,153,0.25)]",
    ring: "border-emerald-400/30",
    label: "Speaking",
    coreColor: "bg-gradient-to-br from-emerald-400/40 to-teal-400/40",
  },
};

export default function ReactiveOrb({ state = "idle" }) {
  const s = STATES[state] || STATES.idle;

  return (
    <div className="relative flex items-center justify-center w-56 h-56 mx-auto select-none">
      {/* Ambient glow */}
      <motion.div
        className={`absolute w-44 h-44 rounded-full bg-gradient-to-br ${s.gradient} blur-3xl`}
        animate={{ scale: [1, 1.1, 1], opacity: [0.5, 0.8, 0.5] }}
        transition={{ repeat: Infinity, duration: 4, ease: "easeInOut" }}
      />

      {/* Outer ring */}
      <motion.div
        className={`absolute w-40 h-40 rounded-full border ${s.ring} ${
          state === "thinking" ? "animate-orbit" : ""
        }`}
        animate={{
          scale: state === "listening" ? [1, 1.06, 1] : 1,
          rotate: state === "thinking" ? 360 : 0,
        }}
        transition={{
          scale: { repeat: Infinity, duration: 1.4, ease: "easeInOut" },
          rotate: { repeat: Infinity, duration: 3, ease: "linear" },
        }}
      >
        {state === "thinking" && (
          <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full bg-amber-400/60" />
        )}
      </motion.div>

      {/* Core sphere */}
      <motion.div
        className={`relative w-28 h-28 rounded-full ${s.coreColor} ${s.glow} backdrop-blur-2xl`}
        animate={{
          scale:
            state === "speaking"
              ? [1, 1.12, 0.94, 1.08, 1]
              : state === "listening"
              ? [1, 1.04, 1]
              : [1, 1.02, 1],
        }}
        transition={{
          repeat: Infinity,
          duration: state === "speaking" ? 0.7 : state === "listening" ? 1.2 : 4,
          ease: "easeInOut",
        }}
      >
        <div className="absolute inset-0 rounded-full bg-gradient-to-t from-transparent to-white/[0.08]" />
        <div className="absolute inset-0 m-auto w-3 h-3 rounded-full bg-white/70 blur-[2px]" />
      </motion.div>

      {/* Ripple rings when speaking */}
      <AnimatePresence>
        {state === "speaking" &&
          [0, 0.6, 1.2].map((delay) => (
            <motion.div
              key={delay}
              className="absolute w-32 h-32 rounded-full border border-emerald-400/20"
              initial={{ scale: 0.9, opacity: 0.5 }}
              animate={{ scale: 2, opacity: 0 }}
              exit={{ opacity: 0 }}
              transition={{ repeat: Infinity, duration: 2, delay, ease: "easeOut" }}
            />
          ))}
      </AnimatePresence>

      {/* State label */}
      <motion.span
        className="absolute -bottom-10 text-[11px] font-medium tracking-[0.2em] uppercase text-white/30"
        key={state}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {s.label}
      </motion.span>
    </div>
  );
}
