"""Phase 40 — agent_traces schema + recording helpers.

Captures per-call subprocess timing and outcome for editor / curator / finish_worker
LLM invocations. Schema is defensive — claude CLI stream-json format may shift
between releases; current pass writes one row per `generate_with_claude` call
(tool='claude_cli'), with truncated args_json/result_json. Stream-json parsing
can extend this later without schema changes.

Schema (auto-created on first call):
    CREATE TABLE agent_traces (
        id INTEGER PRIMARY KEY,
        run_id TEXT NOT NULL,
        agent TEXT NOT NULL,
        step_idx INTEGER NOT NULL,
        tool TEXT,
        args_json TEXT,
        result_json TEXT,
        duration_ms INTEGER,
        started_at TEXT NOT NULL
    )

Idempotent — `ensure_table` safe to re-run.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

KARAGANDA_TZ = timezone(timedelta(hours=5))

# Trim oversize JSON blobs so a single bad prompt can't blow up the DB.
MAX_JSON_FIELD_CHARS = 4000

# WR-04: process-local cache of (db_path) -> bool. Set after the first
# successful ensure_table call so subsequent record_step calls in the same
# process skip the full DDL chain (CREATE + ALTER + INDEX + commit). Reduces
# DDL contention on WAL when cron + manual finish_worker overlap, and cuts
# CPU cost in the hot loop. Per-path keying so multi-DB test fixtures work.
_ENSURED_PATHS: set[str] = set()


def reset_ensured_cache() -> None:
    """Test helper — clear the ensured-paths cache so a subsequent
    record_step against a freshly-created DB re-runs the DDL."""
    _ENSURED_PATHS.clear()


def now_karaganda() -> datetime:
    return datetime.now(KARAGANDA_TZ)


def ensure_table(conn: sqlite3.Connection) -> None:
    """Create agent_traces table + indexes if missing. Idempotent."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_traces (
            id INTEGER PRIMARY KEY,
            run_id TEXT NOT NULL,
            agent TEXT NOT NULL,
            step_idx INTEGER NOT NULL,
            tool TEXT,
            args_json TEXT,
            result_json TEXT,
            duration_ms INTEGER,
            started_at TEXT NOT NULL,
            memory_context_tokens INTEGER
        )
        """
    )
    # Phase 87-02: idempotent ALTER for legacy DBs that pre-date the
    # memory_context_tokens column. CREATE TABLE IF NOT EXISTS above is a
    # no-op for existing tables, so we need this ALTER for migrations.
    try:
        conn.execute(
            "ALTER TABLE agent_traces ADD COLUMN memory_context_tokens INTEGER"
        )
    except sqlite3.OperationalError:
        # Duplicate column — benign on already-migrated DBs.
        pass
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_traces_run_id "
        "ON agent_traces(run_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_traces_agent_started_at "
        "ON agent_traces(agent, started_at DESC)"
    )
    conn.commit()


def start_run(prefix: str = "run") -> str:
    """Generate a unique run identifier scoped to one logical operation."""
    ts = now_karaganda().strftime("%Y%m%dT%H%M%S")
    return f"{prefix}-{ts}-{secrets.token_hex(4)}"


def _trim(value: str) -> str:
    if value is None:
        return ""
    if len(value) <= MAX_JSON_FIELD_CHARS:
        return value
    return value[:MAX_JSON_FIELD_CHARS] + "...[truncated]"


def _json_dump(payload: Any) -> str:
    try:
        return _trim(json.dumps(payload, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return _trim(repr(payload))


def _next_step_idx(conn: sqlite3.Connection, run_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(step_idx), -1) + 1 AS next FROM agent_traces "
        "WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    return int(row[0]) if row else 0


def record_step(
    db_path: str | Path,
    *,
    run_id: str,
    agent: str,
    tool: str | None = None,
    args: Any = None,
    result: Any = None,
    duration_ms: int | None = None,
    started_at: str | None = None,
    memory_context_tokens: int | None = None,
) -> int:
    """Insert one trace row. Fail-soft: returns -1 on any sqlite error so
    instrumentation never breaks production paths.

    Phase 87-02: ``memory_context_tokens`` records the approximate token
    cost of memory-layer context prepended to a finish-iter1 prompt
    (None when WORKER_MEMORY_MODE=off or non-iter1 calls).
    """
    started_at = started_at or now_karaganda().isoformat()
    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error:
        return -1
    try:
        # WR-04: skip full DDL chain on the hot-path after first success per
        # process+db_path. ensure_table is idempotent but still runs CREATE +
        # ALTER + 2 INDEX + commit on every record_step call.
        path_key = str(db_path)
        if path_key not in _ENSURED_PATHS:
            ensure_table(conn)
            _ENSURED_PATHS.add(path_key)
        step_idx = _next_step_idx(conn, run_id)
        cur = conn.execute(
            "INSERT INTO agent_traces"
            "(run_id, agent, step_idx, tool, args_json, result_json, "
            " duration_ms, started_at, memory_context_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                agent,
                step_idx,
                tool,
                _json_dump(args) if args is not None else None,
                _json_dump(result) if result is not None else None,
                duration_ms,
                started_at,
                memory_context_tokens,
            ),
        )
        conn.commit()
        return int(cur.lastrowid or 0)
    except sqlite3.Error:
        return -1
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass


def get_traces(db_path: str | Path, run_id: str, limit: int = 500) -> list[dict]:
    """Read traces by run_id ordered by step_idx ASC. Empty list on errors / missing table."""
    p = Path(str(db_path))
    if not p.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return []
    try:
        ensure_table(conn)
        rows = conn.execute(
            "SELECT id, run_id, agent, step_idx, tool, args_json, result_json, "
            "       duration_ms, started_at "
            "FROM agent_traces WHERE run_id = ? "
            "ORDER BY step_idx ASC LIMIT ?",
            (run_id, limit),
        ).fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        for key in ("args_json", "result_json"):
            raw = d.get(key)
            if raw:
                try:
                    d[key] = json.loads(raw)
                except (TypeError, ValueError):
                    pass
        out.append(d)
    return out


def list_recent_runs(db_path: str | Path, limit: int = 50) -> list[dict]:
    """Recent distinct run_ids with their first agent + earliest started_at."""
    p = Path(str(db_path))
    if not p.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return []
    try:
        ensure_table(conn)
        rows = conn.execute(
            "SELECT run_id, MIN(started_at) AS first_at, MAX(started_at) AS last_at, "
            "       COUNT(*) AS step_count, "
            "       GROUP_CONCAT(DISTINCT agent) AS agents "
            "FROM agent_traces "
            "GROUP BY run_id "
            "ORDER BY first_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass
    return [dict(r) for r in rows]
