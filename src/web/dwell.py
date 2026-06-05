"""Phase 85 / DWELL-BEACON-01: dwell-time beacon ingest endpoint.

Exposes a single endpoint:

    POST /api/dwell  body={"item_id": int, "dwell_ms": int, "session_id": str}

UPSERT-MAX semantics: same (item_id, session_id) pair keeps the MAX dwell_ms
observed across the session. This matches the JS beacon contract — IntersectionObserver
re-enters as the user scrolls back, and we want the longest single attention span,
not the sum (which would double-count overlapping re-entries).

Validation (FastAPI/Pydantic auto-generates 422 on failure):
  - item_id   : int >= 1
  - dwell_ms  : int in [0, 86_400_000]  (24h upper bound — abuse cap)
  - session_id: str len [1, 128], must match _SESSION_ID_PATTERN
  - raw body  : <= 1024 bytes (413 if larger)

Attack surface (BL-01, Phase 85 review):
  Endpoint is a training-label channel — every accepted row gets blended into
  `_soft_label` by reaction_predictor.train(). Without controls, any local
  process (drive-by tab, stray fetch loop, buggy reload) can spam unbounded
  ghost-item synthetic positive labels. Hardening:
    1) session_id must match a hex/uuid-ish charset (rejects arbitrary blobs).
    2) item_id must exist in raw_posts (FK-like 404 instead of silent ghost row).
    3) Per-session rate limit: _RATE_LIMIT_MAX_PER_SESSION beacons per
       _RATE_LIMIT_WINDOW_S — 429 on exceed. JS dedup caps real traffic at
       1 beacon/item/session-page-load, so the budget is generous for honest
       use but kills runaway fetch loops fast.
  Bucket dict is pruned in-place on each write; size capped by
  _RATE_LIMIT_MAX_SESSIONS (LRU-ish, oldest dropped) to bound memory growth
  across days of uptime.

NFR-08 / module-level env binding gotcha: DB_PATH is read inside the handler via
`_get_db_path()` (NOT at module top) so test monkeypatches of
`src.web.main.DB_PATH` take effect.

Concurrency: sync sqlite3 (matches reactions.py / react_worker pattern). WAL mode
is active on curator.db (see migrate_db tail) so we don't fight aiosqlite readers.

Mounted in src/web/main.py via `app.include_router(dwell_router)`.

NOTE: adding new @router.post paths requires a cold uvicorn restart — the
`--reload` watcher misses route-topology changes (see MEMORY ref
`reference_uvicorn_reload_misses_routes.md`).
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from collections import defaultdict, deque

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

router = APIRouter()

# 1 KiB body limit. Anything bigger is either accidental (mobile autofill) or
# malicious (session_id flooding). 413 is the canonical "payload too large" code.
_MAX_BODY_BYTES = 1024

# BL-01 hardening: session_id must look like a UUID/hex token, not arbitrary
# free text. UUID v4 default is 36 chars; allow hex / dashes / underscores up
# to the pydantic 128-char cap.
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# Per-session rate-limit: 30 beacons / 60s. JS dedup caps real traffic at
# 1 beacon/item/page-load so 30 covers any honest digest page; well below
# what a runaway fetch loop would emit.
_RATE_LIMIT_WINDOW_S = 60.0
_RATE_LIMIT_MAX_PER_SESSION = 30
# Cap total tracked sessions to bound memory. Oldest entry evicted on overflow.
_RATE_LIMIT_MAX_SESSIONS = 10_000
_rate_log: "dict[str, deque[float]]" = defaultdict(deque)


def _check_rate_limit(session_id: str) -> bool:
    """Return True if request is within budget; False if it should 429."""
    now = time.monotonic()
    bucket = _rate_log[session_id]
    # Drop timestamps outside the rolling window.
    while bucket and now - bucket[0] >= _RATE_LIMIT_WINDOW_S:
        bucket.popleft()
    if len(bucket) >= _RATE_LIMIT_MAX_PER_SESSION:
        return False
    bucket.append(now)
    # LRU-ish eviction: if we blew past the session cap, drop an arbitrary
    # (first-iterated) other entry. Bound is loose by design — exact LRU is
    # not worth a dependency for an abuse-mitigation bucket.
    if len(_rate_log) > _RATE_LIMIT_MAX_SESSIONS:
        for k in list(_rate_log.keys()):
            if k != session_id:
                _rate_log.pop(k, None)
                break
    return True


def _reset_rate_limit_for_tests() -> None:
    """Test helper — clear the in-memory bucket between cases."""
    _rate_log.clear()


class DwellPayload(BaseModel):
    item_id: int = Field(ge=1)
    # 24h ceiling = 86_400_000 ms. Anything larger is impossible for a single
    # IntersectionObserver entry-then-leave cycle in a real browser session.
    dwell_ms: int = Field(ge=0, le=86_400_000)
    session_id: str = Field(min_length=1, max_length=128)


def _get_db_path() -> str:
    """Resolve curator.db path with the same fallback chain as
    `src.web.main._get_db_path`, but without forcing import of `src.web.main`
    every call.

    WR-04 (Phase 85 review): the previous function-local import created a
    partial-module ImportError trap for any consumer (test or tool) that
    imported `src.web.dwell` before `src.web.main` finished initializing.
    We now prefer the value already loaded on the `main` module (so test
    monkeypatches of `src.web.main.DB_PATH` still win) and fall back to env
    + sentinel default.
    """
    try:
        from src.web import main as _main
        # Honour monkeypatched override, but ignore the literal "curator.db"
        # sentinel — that's the unset default and we want env-vars below.
        main_path = getattr(_main, "DB_PATH", None)
        if main_path and main_path != "curator.db":
            return main_path
    except ImportError:
        pass
    return os.environ.get("DB_PATH", "curator.db")


@router.post("/api/dwell")
async def record_dwell(request: Request):
    """UPSERT a dwell-time observation. Returns {"ok": true} on success."""
    body = await request.body()
    if len(body) > _MAX_BODY_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"payload exceeds {_MAX_BODY_BYTES} bytes",
        )

    # Manual parse so the 413 check happens BEFORE Pydantic validation —
    # otherwise FastAPI would auto-422 oversized but still well-formed bodies.
    try:
        import json
        payload = DwellPayload(**json.loads(body))
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # BL-01: pattern check on session_id (reject arbitrary blobs).
    if not _SESSION_ID_PATTERN.match(payload.session_id):
        raise HTTPException(status_code=422, detail="invalid session_id")

    # BL-01: per-session rate limit. 429 instead of silent enqueue.
    if not _check_rate_limit(payload.session_id):
        raise HTTPException(status_code=429, detail="rate limited")

    db_path = _get_db_path()
    try:
        with sqlite3.connect(db_path) as conn:
            # BL-01: FK-like existence check — reject ghost item_ids with 404
            # so the training label channel cannot be poisoned with synthetic
            # IDs that have no raw_posts backing.
            row = conn.execute(
                "SELECT 1 FROM raw_posts WHERE id=?", (payload.item_id,)
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="item_id not found")
            conn.execute(
                """
                INSERT INTO post_dwell (item_id, dwell_ms, session_id)
                VALUES (?, ?, ?)
                ON CONFLICT(item_id, session_id) DO UPDATE SET
                    dwell_ms = MAX(post_dwell.dwell_ms, excluded.dwell_ms),
                    recorded_at = datetime('now')
                """,
                (payload.item_id, payload.dwell_ms, payload.session_id),
            )
            conn.commit()
    except sqlite3.Error as exc:
        logger.warning(
            "dwell insert failed item_id=%s session=%s: %s",
            payload.item_id, payload.session_id[:8], exc,
        )
        raise HTTPException(status_code=500, detail="dwell persistence failed")

    return {"ok": True}
