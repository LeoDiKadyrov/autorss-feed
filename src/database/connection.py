"""SQLite connection helpers — emit WAL/busy_timeout/synchronous PRAGMAs at open.

Phase 15 / NFR-06: every connection in the codebase MUST go through one of these
helpers so WAL mode + busy_timeout=5000 + synchronous=NORMAL are guaranteed
before any read/write. Async variant for FastAPI/pipeline; sync variant for the
worker process (Phase 17) and the briefing reader (Phase 22).

`:memory:` DBs are exempt by design — WAL is a no-op on memory-backed databases,
and existing test fixtures intentionally use raw aiosqlite.connect(':memory:').
"""
import aiosqlite
import sqlite3


async def open_db(path: str) -> aiosqlite.Connection:
    """Open aiosqlite connection with WAL/busy_timeout/synchronous PRAGMAs.

    LO-02: PRAGMAs run in auto-commit mode (no BEGIN/COMMIT needed) — SQLite
    docs guarantee per-statement atomic effect outside an explicit transaction.
    Absence of `db.commit()` after these PRAGMAs is correct, not an oversight.
    """
    db = await aiosqlite.connect(path)
    # PRAGMAs auto-commit — no explicit BEGIN/COMMIT required (LO-02).
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA busy_timeout=5000")
    await db.execute("PRAGMA synchronous=NORMAL")
    return db


def open_db_sync(path: str) -> sqlite3.Connection:
    """Sync variant for worker process (Phase 17) + briefing reader (Phase 22).

    LO-02: same auto-commit semantics as open_db — PRAGMAs apply immediately
    without a wrapping transaction.
    """
    db = sqlite3.connect(path, timeout=5.0)
    # PRAGMAs auto-commit — no explicit BEGIN/COMMIT required (LO-02).
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=5000")
    db.execute("PRAGMA synchronous=NORMAL")
    return db
