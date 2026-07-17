"""CHARKHA memory — SQLite-backed conversation store."""

import sqlite3


class Memory:
    def __init__(self, path: str):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS turns("
            "id INTEGER PRIMARY KEY, ts REAL, role TEXT, text TEXT)"
        )
        self.db.execute("CREATE TABLE IF NOT EXISTS facts(key TEXT PRIMARY KEY, val TEXT)")
        self.db.commit()

    def remember(self, role: str, text: str, ts: float = 0.0):
        self.db.execute("INSERT INTO turns(ts, role, text) VALUES(?,?,?)", (ts, role, text))
        self.db.commit()

    def recall(self, k: int = 6):
        rows = self.db.execute(
            "SELECT role, text FROM turns ORDER BY id DESC LIMIT ?", (k,)
        ).fetchall()
        return list(reversed(rows))

    def set_fact(self, key: str, val: str):
        self.db.execute(
            "INSERT INTO facts(key, val) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET val=excluded.val",
            (key, val),
        )
        self.db.commit()

    def get_facts(self):
        return dict(self.db.execute("SELECT key, val FROM facts").fetchall())

    def close(self):
        self.db.close()


__all__ = ["Memory"]
