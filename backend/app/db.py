"""SQLite 持久化：机构定义与分析会话状态。"""
from __future__ import annotations

import json
import os
import sqlite3
import threading

_DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "linkage.db")),
)
_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    d = os.path.dirname(_DB_PATH)
    if d:
        os.makedirs(d, exist_ok=True)
    con = sqlite3.connect(_DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init():
    with _lock, _connect() as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS mechanisms(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"""
        )
        con.execute(
            """CREATE TABLE IF NOT EXISTS sessions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mechanism_id INTEGER NOT NULL REFERENCES mechanisms(id),
                state TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"""
        )


def create_mechanism(data: dict) -> int:
    with _lock, _connect() as con:
        cur = con.execute("INSERT INTO mechanisms(data) VALUES (?)",
                          (json.dumps(data, ensure_ascii=False),))
        return int(cur.lastrowid)


def create_session(mechanism_id: int, state: dict) -> int:
    with _lock, _connect() as con:
        cur = con.execute("INSERT INTO sessions(mechanism_id, state) VALUES (?, ?)",
                          (mechanism_id, json.dumps(state, ensure_ascii=False)))
        return int(cur.lastrowid)


def update_session(session_id: int, state: dict):
    with _lock, _connect() as con:
        con.execute(
            "UPDATE sessions SET state = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (json.dumps(state, ensure_ascii=False), session_id),
        )


def get_session(session_id: int):
    """返回 (机构数据, 会话状态)；不存在则 None。"""
    with _lock, _connect() as con:
        row = con.execute(
            """SELECT s.state AS state, m.data AS mech
               FROM sessions s JOIN mechanisms m ON m.id = s.mechanism_id
               WHERE s.id = ?""",
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    return json.loads(row["mech"]), json.loads(row["state"])
