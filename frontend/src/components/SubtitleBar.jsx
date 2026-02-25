import { motion, AnimatePresence } from "framer-motion";
import { useMemo, useRef, useEffect } from "react";

// ── Mini translation dictionary for common cross-language words ──
const DE_TO_EN = {
  schadenfreude: "pleasure from others' pain", fernweh: "wanderlust",
  gemütlichkeit: "coziness", zeitgeist: "spirit of the time",
  wanderlust: "desire to travel", kindergarten: "kindergarten",
  angst: "anxiety/dread", weltanschauung: "worldview",
  doppelgänger: "look-alike", heimweh: "homesickness",
  wunderbar: "wonderful", genau: "exactly", eigentlich: "actually",
  natürlich: "of course", vielleicht: "maybe", trotzdem: "nevertheless",
  manchmal: "sometimes", übrigens: "by the way", jedenfalls: "anyway",
  grundsätzlich: "fundamentally", wahrscheinlich: "probably",
  selbstverständlich: "obviously", unbedingt: "absolutely",
  tatsächlich: "indeed", offensichtlich: "obviously",
};
const EN_TO_DE = {
  actually: "eigentlich", obviously: "offensichtlich",
  awesome: "großartig", mindblowing: "umwerfend",
  nevertheless: "trotzdem", basically: "grundsätzlich",
  literally: "buchstäblich", apparently: "anscheinend",
};

// Detect if a word is likely from the "other" language
const DE_MARKERS = /[äöüßÄÖÜ]/;
const DE_COMMON = new Set(Object.keys(DE_TO_EN));
const EN_COMMON = new Set(Object.keys(EN_TO_DE));

function detectForeignWord(word, primaryLang) {
  const lower = word.toLowerCase().replace(/[.,!?;:'"]/g, "");
  if (!lower || lower.length < 4) return null;
  if (primaryLang === "en") {
    // In English text, detect German words
    if (DE_MARKERS.test(lower)) return DE_TO_EN[lower] || "DE";
    if (DE_COMMON.has(lower)) return DE_TO_EN[lower];
  } else {
    // In German text, detect English words
    if (EN_COMMON.has(lower)) return EN_TO_DE[lower];
  }
  return null;
}

// Word-by-word fluid animation with ruby translation overlay
function FluidWords({ text, isDark, language }) {
  const words = useMemo(() => text.split(/\s+/).filter(Boolean), [text]);
  const prevCountRef = useRef(0);
  const primaryLang = language === "de" ? "de" : "en";

  useEffect(() => {
    prevCountRef.current = words.length;
  });

  return (
    <span style={{
      flex: 1, fontSize: 13, lineHeight: 2.0,
      color: isDark ? "rgba(255,255,255,0.78)" : "rgba(255,255,255,0.82)",
      fontFamily: "'Space Grotesk', sans-serif",
      display: "flex", flexWrap: "wrap", gap: "0 5px", alignItems: "baseline",
    }}>
      {words.map((word, i) => {
        const isNew = i >= prevCountRef.current;
        const translation = detectForeignWord(word, primaryLang);

        return (
          <motion.span
            key={`${i}-${word}`}
            initial={isNew ? { opacity: 0, filter: "blur(8px)", y: 4 } : false}
            animate={{ opacity: 1, filter: "blur(0px)", y: 0 }}
            transition={{
              duration: 0.35,
              delay: isNew ? (i - prevCountRef.current) * 0.04 : 0,
              ease: [0.16, 1, 0.3, 1],
            }}
            style={{
              display: "inline-flex", flexDirection: "column",
              alignItems: "center", position: "relative",
            }}
          >
            {/* Ruby translation annotation above foreign words */}
            {translation && (
              <motion.span
                initial={{ opacity: 0, y: 2 }}
                animate={{ opacity: 0.45, y: 0 }}
                transition={{ duration: 0.4, delay: 0.2 }}
                style={{
                  position: "absolute", top: -14,
                  fontSize: 8, fontWeight: 600,
                  letterSpacing: "0.03em",
                  color: primaryLang === "en"
                    ? "rgba(245,158,11,0.6)"  // amber for DE words
                    : "rgba(6,182,212,0.6)",   // cyan for EN words
                  whiteSpace: "nowrap",
                  fontStyle: "italic",
                  pointerEvents: "none",
                }}
              >
                {translation}
              </motion.span>
            )}
            <span style={{
              borderBottom: translation
                ? `1px dotted ${primaryLang === "en" ? "rgba(245,158,11,0.3)" : "rgba(6,182,212,0.3)"}`
                : "none",
              paddingBottom: translation ? 1 : 0,
            }}>
              {word}
            </span>
          </motion.span>
        );
      })}
      <motion.span
        style={{
          display: "inline-block", width: 2, height: 14,
          marginLeft: 2, borderRadius: 1, verticalAlign: "middle",
          background: isDark ? "rgba(6,182,212,0.6)" : "rgba(6,182,212,0.7)",
        }}
        animate={{ opacity: [1, 0.2, 1] }}
        transition={{ repeat: Infinity, duration: 0.6 }}
      />
    </span>
  );
}

export default function SubtitleBar({ text, language, isActive, theme = "dark" }) {
  const isDark = theme === "dark";
  if (!isActive || !text) return null;

  const flag = language === "de" ? "\u{1f1e9}\u{1f1ea}" : "\u{1f1ec}\u{1f1e7}";
  const langLabel = language === "de" ? "Deutsch" : "English";
  const langAccent = language === "de" ? "245,158,11" : "6,182,212";

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 12, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 8, scale: 0.97 }}
        transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        className="flex items-center justify-center"
        style={{ width: "100%", padding: "0 20px" }}
      >
        <div className="flex items-center gap-3" style={{
          maxWidth: 520, width: "100%",
          padding: "10px 18px", borderRadius: 16,
          background: isDark
            ? `linear-gradient(135deg, rgba(${langAccent},0.06), rgba(${langAccent},0.02))`
            : `linear-gradient(135deg, rgba(${langAccent},0.1), rgba(${langAccent},0.04))`,
          border: `1px solid rgba(${langAccent},0.1)`,
          backdropFilter: "blur(16px)",
          WebkitBackdropFilter: "blur(16px)",
          transition: "background 1.5s, border-color 1.5s",
        }}>
          {/* Language badge — shifts color with language */}
          <motion.div
            className="flex items-center gap-1.5 shrink-0"
            animate={{
              background: `rgba(${langAccent},0.1)`,
              borderColor: `rgba(${langAccent},0.15)`,
            }}
            transition={{ duration: 1.2 }}
            style={{
              padding: "3px 8px", borderRadius: 8,
              border: "1px solid",
            }}
          >
            <span style={{ fontSize: 12 }}>{flag}</span>
            <motion.span
              animate={{ color: `rgba(${langAccent},0.7)` }}
              transition={{ duration: 1.2 }}
              style={{
                fontSize: 9, fontWeight: 700, letterSpacing: "0.08em",
                textTransform: "uppercase",
              }}
            >{langLabel}</motion.span>
          </motion.div>

          {/* Fluid word-by-word subtitle text with ruby translations */}
          <FluidWords text={text} isDark={isDark} language={language} />
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
