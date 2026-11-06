"""SQLite persistence for nodes, experiments, rounds and studies."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, url TEXT NOT NULL UNIQUE, added_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS studies (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL,
    config TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY, study_id TEXT, name TEXT NOT NULL, status TEXT NOT NULL,
    config TEXT NOT NULL, tags TEXT, created_at REAL NOT NULL, started_at REAL, finished_at REAL,
    error TEXT, summary TEXT, progress INTEGER NOT NULL DEFAULT 0, total_rounds INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS rounds (
    experiment_id TEXT NOT NULL, round INTEGER NOT NULL, data TEXT NOT NULL,
    PRIMARY KEY (experiment_id, round)
);
CREATE INDEX IF NOT EXISTS idx_exp_study ON experiments(study_id);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class Database:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        with self._lock:
            c = sqlite3.connect(self.path, timeout=30)
            c.row_factory = sqlite3.Row
            try:
                yield c
                c.commit()
            finally:
                c.close()

    # -------------------------------------------------------------------- nodes
    def add_node(self, name: str, url: str) -> dict:
        url = url.rstrip("/")
        with self.conn() as c:
            row = c.execute("SELECT * FROM nodes WHERE url=?", (url,)).fetchone()
            if row:
                c.execute("UPDATE nodes SET name=? WHERE id=?", (name, row["id"]))
                return {**dict(row), "name": name}
            nid = new_id("node")
            c.execute("INSERT INTO nodes VALUES (?,?,?,?)", (nid, name, url, time.time()))
            return {"id": nid, "name": name, "url": url, "added_at": time.time()}

    def list_nodes(self) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM nodes ORDER BY name")]

    def get_node(self, node_id: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
            return dict(r) if r else None

    def delete_node(self, node_id: str) -> bool:
        with self.conn() as c:
            return c.execute("DELETE FROM nodes WHERE id=?", (node_id,)).rowcount > 0

    # -------------------------------------------------------------- experiments
    def create_experiment(self, name: str, config: dict, study_id: str | None = None,
                          tags: dict | None = None) -> str:
        eid = new_id("exp")
        with self.conn() as c:
            c.execute(
                "INSERT INTO experiments (id, study_id, name, status, config, tags, created_at, total_rounds) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (eid, study_id, name, "queued", json.dumps(config), json.dumps(tags or {}), time.time(),
                 int(config["n_rounds"])),
            )
        return eid

    @staticmethod
    def _exp(r: sqlite3.Row, with_config: bool = True) -> dict:
        d = dict(r)
        d["config"] = json.loads(d["config"]) if with_config else None
        d["tags"] = json.loads(d["tags"] or "{}")
        d["summary"] = json.loads(d["summary"]) if d["summary"] else None
        return d

    def get_experiment(self, eid: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM experiments WHERE id=?", (eid,)).fetchone()
            return self._exp(r) if r else None

    def list_experiments(self, study_id: str | None = None, limit: int = 500) -> list[dict]:
        with self.conn() as c:
            if study_id:
                rows = c.execute("SELECT * FROM experiments WHERE study_id=? ORDER BY created_at", (study_id,))
            else:
                rows = c.execute("SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (limit,))
            return [self._exp(r) for r in rows.fetchall()]

    def update_experiment(self, eid: str, **fields) -> None:
        if not fields:
            return
        for k in ("summary", "config", "tags"):
            if k in fields and not isinstance(fields[k], str) and fields[k] is not None:
                fields[k] = json.dumps(fields[k])
        sets = ", ".join(f"{k}=?" for k in fields)
        with self.conn() as c:
            c.execute(f"UPDATE experiments SET {sets} WHERE id=?", (*fields.values(), eid))

    def delete_experiment(self, eid: str) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM rounds WHERE experiment_id=?", (eid,))
            c.execute("DELETE FROM experiments WHERE id=?", (eid,))

    def mark_interrupted(self) -> int:
        """On startup: anything left running/queued by a previous process cannot continue."""
        with self.conn() as c:
            return c.execute(
                "UPDATE experiments SET status='failed', error='coordinator restarted', finished_at=? "
                "WHERE status IN ('running','queued')", (time.time(),)).rowcount

    # ------------------------------------------------------------------- rounds
    def save_round(self, eid: str, rnd: int, data: dict) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO rounds VALUES (?,?,?)", (eid, rnd, json.dumps(data)))
            c.execute("UPDATE experiments SET progress=? WHERE id=?", (rnd, eid))

    def get_rounds(self, eid: str) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT data FROM rounds WHERE experiment_id=? ORDER BY round", (eid,))
            return [json.loads(r["data"]) for r in rows.fetchall()]

    # ------------------------------------------------------------------ studies
    def create_study(self, name: str, kind: str, config: dict) -> str:
        sid = new_id("study")
        with self.conn() as c:
            c.execute("INSERT INTO studies VALUES (?,?,?,?,?,?)",
                      (sid, name, kind, "queued", json.dumps(config), time.time()))
        return sid

    def get_study(self, sid: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM studies WHERE id=?", (sid,)).fetchone()
            if not r:
                return None
            d = dict(r)
            d["config"] = json.loads(d["config"])
            return d

    def list_studies(self) -> list[dict]:
        with self.conn() as c:
            out = []
            for r in c.execute("SELECT * FROM studies ORDER BY created_at DESC"):
                d = dict(r)
                d["config"] = json.loads(d["config"])
                out.append(d)
            return out

    def update_study(self, sid: str, **fields) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)
        with self.conn() as c:
            c.execute(f"UPDATE studies SET {sets} WHERE id=?", (*fields.values(), sid))

    def delete_study(self, sid: str) -> None:
        with self.conn() as c:
            ids = [r["id"] for r in c.execute("SELECT id FROM experiments WHERE study_id=?", (sid,))]
            for i in ids:
                c.execute("DELETE FROM rounds WHERE experiment_id=?", (i,))
            c.execute("DELETE FROM experiments WHERE study_id=?", (sid,))
            c.execute("DELETE FROM studies WHERE id=?", (sid,))
