"""
database/note_manager.py
SQLite-backed notes storage.
"""

import sqlite3
import os
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "smart_home.db")


class NoteManager:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    content    TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

    def save_note(self, content: str) -> int:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute("INSERT INTO notes (content, created_at) VALUES (?, ?)", (content, ts))
            logger.info(f"Note saved id={cur.lastrowid}")
            return cur.lastrowid

    def get_all_notes(self) -> list:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT id, content, created_at FROM notes ORDER BY id DESC").fetchall()
        return [{"id": r[0], "content": r[1], "created_at": r[2]} for r in rows]

    def delete_note(self, note_id: int):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM notes WHERE id=?", (note_id,))
