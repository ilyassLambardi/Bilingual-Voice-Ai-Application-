"""Long-Term Memory — SQLite-backed conversational memory with keyword retrieval."""
import logging
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
_DB_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "memory.db"
_STOP = {
    "i","me","my","you","your","we","the","a","an","is","am","are","was","were","be",
    "have","has","had","do","does","did","will","would","could","should","to","of","in",
    "for","on","with","at","by","from","it","this","that","and","or","but","if","so","not",
    "just","very","what","how","when","where","who","which","why",
    "ich","du","er","sie","es","wir","und","oder","aber","das","der","die","ein",
    "ist","sind","war","hat","haben","mit","von","zu","auf",
}

class LongTermMemory:
    _MAX = 500

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or str(_DB_PATH)
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT DEFAULT 'conversation',
                content TEXT NOT NULL, keywords TEXT DEFAULT '', language TEXT DEFAULT 'en',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, relevance_count INTEGER DEFAULT 0);
            CREATE INDEX IF NOT EXISTS idx_kw ON memories(keywords);
        """)
        self._conn.commit()
        logger.info("[LTM] Ready — %d memories", self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def store_conversation(self, user_text: str, assistant_text: str, lang: str = "en"):
        try:
            self._conn.execute("INSERT INTO memories (category,content,keywords,language) VALUES (?,?,?,?)",
                ("conversation", f"User: {user_text}\nAssistant: {assistant_text}",
                 self._kw(f"{user_text} {assistant_text}"), lang))
            self._conn.commit(); self._prune()
        except sqlite3.Error: pass

    def store_summary(self, summary: str, lang: str = "en"):
        try:
            self._conn.execute("INSERT INTO memories (category,content,keywords,language) VALUES (?,?,?,?)",
                ("summary", summary, self._kw(summary), lang))
            self._conn.commit()
        except sqlite3.Error: pass

    def recall(self, query: str, limit: int = 3) -> list[str]:
        qkw = set(self._kw(query).split())
        if not qkw: return []
        rows = self._conn.execute("SELECT id,content,keywords FROM memories ORDER BY id DESC LIMIT 100").fetchall()
        scored = []
        for r in rows:
            mkw = set(r["keywords"].split())
            union = len(qkw | mkw)
            score = len(qkw & mkw) / union if union else 0
            if score > 0.05: scored.append((score, r["id"], r["content"]))
        scored.sort(reverse=True)
        for _, mid, _ in scored[:limit]:
            self._conn.execute("UPDATE memories SET relevance_count=relevance_count+1 WHERE id=?", (mid,))
        self._conn.commit()
        return [c for _, _, c in scored[:limit]]

    def summarize_and_store(self, history: list[dict]):
        if len(history) < 4: return
        topics = self._kw(" ".join(m.get("content","") for m in history)).split()[:10]
        self.store_summary(f"Session with {len(history)//2} exchanges. Topics: {', '.join(topics)}")

    def _prune(self):
        cnt = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if cnt > self._MAX:
            self._conn.execute("DELETE FROM memories WHERE id IN "
                "(SELECT id FROM memories ORDER BY relevance_count ASC,id ASC LIMIT ?)", (cnt-self._MAX,))
            self._conn.commit()

    @staticmethod
    def _kw(text: str) -> str:
        return " ".join(w for w, _ in Counter(
            w for w in text.lower().split() if len(w)>2 and w not in _STOP and w.isalpha()
        ).most_common(20))

    def close(self):
        try: self._conn.close()
        except Exception: pass
