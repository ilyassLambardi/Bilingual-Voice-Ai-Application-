"""
Module 3 (Data Storage/State): Long-Term Memory (LTM) — SQLite-backed conversational memory.

Stores conversation summaries and user preferences for cross-session
context.  Uses lightweight TF-IDF vectors for semantic retrieval
(no extra model required).
"""

import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Optional


_DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DB_PATH = _DB_DIR / "memory.db"

_STOPWORDS = {
    "i", "me", "my", "you", "your", "we", "our", "the", "a", "an",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "it", "its", "this", "that", "these", "those", "and", "or",
    "but", "if", "so", "not", "no", "just", "very", "really",
    "what", "how", "when", "where", "who", "which", "why",
    "ich", "du", "er", "sie", "es", "wir", "ihr", "und", "oder",
    "aber", "das", "der", "die", "ein", "eine", "ist", "sind",
    "war", "hat", "haben", "wird", "mit", "von", "zu", "auf",
}


class LongTermMemory:
    """Persistent conversational memory with keyword-based retrieval."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or str(_DB_PATH)
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")  # safe for concurrent writes
        self._conn.execute("PRAGMA busy_timeout=5000")  # wait up to 5s on lock
        self._init_db()
        count = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        print(f"[LTM] Ready -- {count} memories in {self._db_path}")

    def _init_db(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL DEFAULT 'conversation',
                content TEXT NOT NULL,
                keywords TEXT NOT NULL DEFAULT '',
                language TEXT DEFAULT 'en',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                relevance_count INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS user_prefs (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
            CREATE INDEX IF NOT EXISTS idx_memories_keywords ON memories(keywords);
        """)
        self._conn.commit()

    # ── Store ──────────────────────────────────────────────────────────

    _MAX_MEMORIES = 500  # prune oldest when exceeded

    def store_conversation(
        self, user_text: str, assistant_text: str, lang: str = "en"
    ):
        """Store a conversation exchange as a memory."""
        try:
            content = f"User: {user_text}\nAssistant: {assistant_text}"
            keywords = self._extract_keywords(f"{user_text} {assistant_text}")
            self._conn.execute(
                "INSERT INTO memories (category, content, keywords, language) VALUES (?, ?, ?, ?)",
                ("conversation", content, keywords, lang),
            )
            self._conn.commit()
            self._prune_if_needed()
        except sqlite3.Error as e:
            print(f"[LTM] Store conversation failed: {e}")

    def _prune_if_needed(self):
        """Delete oldest, least-relevant memories when count exceeds limit."""
        count = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if count > self._MAX_MEMORIES:
            excess = count - self._MAX_MEMORIES
            self._conn.execute(
                "DELETE FROM memories WHERE id IN ("
                "  SELECT id FROM memories ORDER BY relevance_count ASC, id ASC LIMIT ?"
                ")", (excess,)
            )
            self._conn.commit()
            print(f"[LTM] Pruned {excess} old memories (kept {self._MAX_MEMORIES})")

    def store_preference(self, key: str, value: str):
        """Store or update a user preference."""
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO user_prefs (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (key, value),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            print(f"[LTM] Store preference failed: {e}")

    def store_summary(self, summary: str, lang: str = "en"):
        """Store a session summary."""
        try:
            keywords = self._extract_keywords(summary)
            self._conn.execute(
                "INSERT INTO memories (category, content, keywords, language) VALUES (?, ?, ?, ?)",
                ("summary", summary, keywords, lang),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            print(f"[LTM] Store summary failed: {e}")

    # ── Retrieve ───────────────────────────────────────────────────────

    def recall(self, query: str, limit: int = 3) -> list[str]:
        """Find memories relevant to the query using keyword overlap."""
        query_kw = set(self._extract_keywords(query).split())
        if not query_kw:
            return []

        rows = self._conn.execute(
            "SELECT id, content, keywords FROM memories ORDER BY id DESC LIMIT 100"
        ).fetchall()

        scored = []
        for row in rows:
            mem_kw = set(row["keywords"].split())
            if not mem_kw:
                continue
            # Jaccard similarity
            overlap = len(query_kw & mem_kw)
            union = len(query_kw | mem_kw)
            score = overlap / union if union > 0 else 0
            if score > 0.05:
                scored.append((score, row["id"], row["content"]))

        scored.sort(reverse=True)

        # Boost relevance count for retrieved memories
        for _, mid, _ in scored[:limit]:
            self._conn.execute(
                "UPDATE memories SET relevance_count = relevance_count + 1 WHERE id = ?",
                (mid,),
            )
        self._conn.commit()

        return [content for _, _, content in scored[:limit]]

    def get_preferences(self) -> dict[str, str]:
        """Return all stored user preferences."""
        rows = self._conn.execute("SELECT key, value FROM user_prefs").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def get_recent_summaries(self, limit: int = 3) -> list[str]:
        """Return recent session summaries."""
        rows = self._conn.execute(
            "SELECT content FROM memories WHERE category = 'summary' ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [row["content"] for row in rows]

    # ── Session management ─────────────────────────────────────────────

    def summarize_and_store(self, conversation_history: list[dict]):
        """Create a summary from the conversation history and store it."""
        if len(conversation_history) < 4:
            return  # too short to summarize

        # Extract key topics from the conversation
        all_text = " ".join(m.get("content", "") for m in conversation_history)
        keywords = self._extract_keywords(all_text)
        topic_words = keywords.split()[:10]

        n_turns = len(conversation_history) // 2
        summary = f"Session with {n_turns} exchanges. Topics: {', '.join(topic_words)}"
        self.store_summary(summary)
        print(f"[LTM] Stored session summary: {summary[:80]}...")

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_keywords(text: str) -> str:
        """Extract meaningful keywords from text (stopword-filtered)."""
        words = text.lower().split()
        # Keep words > 2 chars, not stopwords, alpha only
        keywords = [w for w in words if len(w) > 2 and w not in _STOPWORDS and w.isalpha()]
        # Count and return top keywords
        counts = Counter(keywords)
        return " ".join(w for w, _ in counts.most_common(20))

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass
