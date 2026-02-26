import { useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";

/**
 * SmartSuggestions — contextual prompt chips that appear below the
 * transcript when idle. Suggests conversation starters, language
 * switches, and Teacher Mode prompts based on conversation context.
 */

const STARTERS = [
  { text: "Hey Alex, tell me about yourself", lang: "en" },
  { text: "What does Fernweh mean?", lang: "en", tag: "Teacher Mode" },
  { text: "Erzähl mir einen Witz", lang: "de" },
  { text: "Switch: Sprich mal Deutsch mit mir", lang: "de" },
  { text: "Was bedeutet 'serendipity'?", lang: "de", tag: "Teacher Mode" },
  { text: "What's the difference between Heimweh and Fernweh?", lang: "en", tag: "Teacher Mode" },
];

const FOLLOWUPS_EN = [
  { text: "Tell me more about that", lang: "en" },
  { text: "Can you explain that in German?", lang: "en" },
  { text: "What does that word mean?", lang: "en", tag: "Teacher Mode" },
  { text: "Give me an example", lang: "en" },
  { text: "Interesting! What else?", lang: "en" },
];

const FOLLOWUPS_DE = [
  { text: "Erzähl mir mehr darüber", lang: "de" },
  { text: "Kannst du das auf Englisch sagen?", lang: "de" },
  { text: "Was bedeutet das genau?", lang: "de", tag: "Teacher Mode" },
  { text: "Gib mir ein Beispiel", lang: "de" },
  { text: "Spannend! Was noch?", lang: "de" },
];

function pickRandom(arr, n) {
  const shuffled = [...arr].sort(() => 0.5 - Math.random());
  return shuffled.slice(0, n);
}

export default function SmartSuggestions({
  messages = [], pipelineState, detectedLang, onSend, theme = "dark",
}) {
  const isDark = theme === "dark";
  const isIdle = pipelineState === "idle";

  const suggestions = useMemo(() => {
    if (!isIdle) return [];

    // No messages yet — show starters
    if (messages.length === 0) {
      return pickRandom(STARTERS, 3);
    }

    // Has messages — show contextual follow-ups
    const pool = detectedLang === "de" ? FOLLOWUPS_DE : FOLLOWUPS_EN;
    // Mix in one from the other language to encourage switching
    const other = detectedLang === "de" ? FOLLOWUPS_EN : FOLLOWUPS_DE;
    const mixed = [...pickRandom(pool, 2), ...pickRandom(other, 1)];
    return mixed;
  }, [messages.length, isIdle, detectedLang]);

  if (!isIdle || suggestions.length === 0) return null;

  return (
    <motion.div
      className="flex flex-wrap justify-center gap-1.5"
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 6 }}
      transition={{ duration: 0.4, delay: 0.3 }}
      style={{ padding: "4px 0" }}
    >
      <AnimatePresence>
        {suggestions.map((s, i) => {
          const isDE = s.lang === "de";
          const accent = isDE ? "245,158,11" : "6,182,212";

          return (
            <motion.button
              key={s.text}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ delay: i * 0.06, duration: 0.2 }}
              whileHover={{ scale: 1.04, y: -1 }}
              whileTap={{ scale: 0.96 }}
              onClick={() => onSend?.(s.text)}
              className="flex items-center gap-1.5"
              style={{
                padding: "5px 10px",
                borderRadius: 20,
                fontSize: 10,
                fontWeight: 500,
                color: `rgba(${accent}, 0.7)`,
                background: `rgba(${accent}, 0.04)`,
                border: `1px solid rgba(${accent}, 0.1)`,
                cursor: "pointer",
                transition: "all 0.2s",
                whiteSpace: "nowrap",
              }}
            >
              {isDE ? "🇩🇪" : "🇬🇧"}
              <span>{s.text}</span>
              {s.tag && (
                <span style={{
                  fontSize: 7, fontWeight: 700, letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  padding: "1px 4px", borderRadius: 4,
                  background: `rgba(${accent}, 0.1)`,
                  color: `rgba(${accent}, 0.5)`,
                }}>
                  {s.tag}
                </span>
              )}
            </motion.button>
          );
        })}
      </AnimatePresence>
    </motion.div>
  );
}
