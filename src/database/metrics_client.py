"""react_metrics.db — isolated write-only metrics store for react_worker outcomes.

Two write APIs:
  - async (init_metrics_db, insert_metric_event) for async callers
  - sync (insert_metric_event_sync) for react_worker.py which uses sqlite3
"""
import datetime
import os
import sqlite3

import aiosqlite


def get_metrics_db_path() -> str:
    return os.environ.get("REACT_METRICS_DB_PATH", "react_metrics.db")


_CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS metric_events (
        id INTEGER PRIMARY KEY,
        item_id INTEGER NOT NULL,
        action TEXT NOT NULL,
        outcome TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
"""


async def init_metrics_db(db: aiosqlite.Connection) -> None:
    await db.execute(_CREATE_SQL)
    await db.commit()


def init_metrics_db_sync(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_CREATE_SQL)
        conn.commit()
    finally:
        conn.close()


async def insert_metric_event(
    db: aiosqlite.Connection, item_id: int, action: str, outcome: str
) -> None:
    now = datetime.datetime.now(datetime.UTC).isoformat()
    await db.execute(
        "INSERT INTO metric_events (item_id, action, outcome, created_at) VALUES (?, ?, ?, ?)",
        (item_id, action, outcome, now),
    )
    await db.commit()


def insert_metric_event_sync(db_path: str, item_id: int, action: str, outcome: str) -> None:
    """Sync version for react_worker.py which uses sqlite3 directly."""
    now = datetime.datetime.now(datetime.UTC).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_CREATE_SQL)
        conn.execute(
            "INSERT INTO metric_events (item_id, action, outcome, created_at) VALUES (?, ?, ?, ?)",
            (item_id, action, outcome, now),
        )
        conn.commit()
    finally:
        conn.close()


async def get_metric_events(db: aiosqlite.Connection) -> list[dict]:
    db.row_factory = aiosqlite.Row
    async with db.execute("SELECT * FROM metric_events ORDER BY id") as cur:
        return [dict(row) for row in await cur.fetchall()]
