"""Phase 16 / REACT-02 + REACT-04 + REACT-05 + NFR-01: reactions API router.

Exposes a single endpoint:

    POST /react/{item_id}/{action}

where `action` ∈ {brainstorm, docs, link, finish, skip}. Returns 200 with
`{"status": "queued"|"skipped", "reaction_id": <int>}` on success.

Validation order (matters — keeps invalid input cheap):
  1. action allowlist  → 400 BEFORE any DB call (T-16-01)
  2. item_id existence → 404 against digest_items (T-16-02)
  3. 60s soft-window pre-check inside insert_reaction (REACT-04)
  4. partial UNIQUE collision → 409 (T-16-08, race-loser path)

Skip semantics (REACT-05): writes status='reviewed' so the worker filter
(`status='pending'`) ignores it. Worker is Phase 17.

Mounting: this plan (16-01) only defines the router. Plan 16-03 wires
`app.include_router(router)` in src/web/main.py.

Conventions:
  - No module-level env binding (NFR-08 / D-12). _get_db_path() defers to
    src.web.main._get_db_path so the monkeypatch in tests still works.
  - Parameterized SQL only (T-16-03).
  - HTTPException details mirror the existing /feedback handler shape.
"""
import datetime
import logging
import sqlite3

from fastapi import APIRouter, HTTPException

from src.database.client import insert_finish_reaction_softcapped, insert_reaction
from src.database.connection import open_db


logger = logging.getLogger(__name__)


ALLOWED_ACTIONS: tuple[str, ...] = (
    "brainstorm", "docs", "link", "finish", "debate", "skip",
)
SKIP_ACTION = "skip"
FINISH_ACTION = "finish"
SOFT_CAP_LIMIT = 5  # Phase 20 / GATE-04: max active finish-chains globally.

router = APIRouter()


def _get_db_path() -> str:
    """Defer env/path resolution to call time (NFR-08).

    Imports inside the function to avoid an import-time circular reference if
    src/web/main.py imports this module before defining its own _get_db_path.
    """
    from src.web.main import _get_db_path as _main_get_db_path
    return _main_get_db_path()


@router.post("/react/{item_id}/{action}")
async def react(item_id: int, action: str):
    """Record a user reaction (intent) for a digest item.

    See module docstring for validation order and response contract.
    """
    # 1. Allowlist (T-16-01) — runs BEFORE any DB connection is opened.
    if action not in ALLOWED_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"invalid action {action!r}; "
                f"allowed: {', '.join(ALLOWED_ACTIONS)}"
            ),
        )

    db = await open_db(_get_db_path())
    try:
        # 2. item_id existence vs digest_items cache (T-16-02).
        async with db.execute(
            "SELECT 1 FROM digest_items WHERE item_id = ?", (item_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"item_id {item_id} not found in digest_items",
            )

        # 3. Finish takes the soft-capped atomic-INSERT path (Phase 20 / GATE-04).
        # Other actions fall through to the existing insert_reaction flow below.
        if action == FINISH_ACTION:
            try:
                rid = await insert_finish_reaction_softcapped(
                    db, item_id, soft_cap=SOFT_CAP_LIMIT
                )
            except sqlite3.IntegrityError as exc:
                # Repeat-click outside the 60s window collides on the partial
                # UNIQUE index — race-loser path (mirrors P16 contract).
                logger.warning(
                    "finish race-loser IntegrityError item_id=%s: %s",
                    item_id, exc,
                )
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"reaction for item_id {item_id} action 'finish' "
                        f"already in progress (race-loser path)"
                    ),
                )
            if rid is None:
                # Helper returns None for THREE distinct reasons (CR-02
                # REVIEW-FIX P20 added the third):
                #   (a) cap hit — active count >= SOFT_CAP_LIMIT, or
                #   (b) 60s soft-window dedup — recent finish on same item, or
                #   (c) an active chain already exists on this item (NOT EXISTS
                #       guard fired; chain at iter1_drafted >60s after creation
                #       would otherwise spawn a duplicate).
                # Disambiguate in PRECEDENCE order: soft-window first (REACT-04
                # contract — return SAME rid on rapid re-click), then cap, then
                # already-active-chain.
                threshold = (
                    datetime.datetime.now(datetime.UTC)
                    - datetime.timedelta(seconds=60)
                ).isoformat()
                async with db.execute(
                    "SELECT id FROM reactions WHERE item_id=? AND action='finish' "
                    "AND created_at >= ? ORDER BY id DESC LIMIT 1",
                    (item_id, threshold),
                ) as cur:
                    hit = await cur.fetchone()
                if hit is not None:
                    # Soft-window dedup path — return same rid (200).
                    return {"status": "queued", "reaction_id": hit[0]}

                # Probe global cap.
                async with db.execute(
                    "SELECT COUNT(*) FROM reactions "
                    "WHERE action='finish' "
                    "AND status IN ('pending','processing','drafted') "
                    "AND current_iter < total_iters",
                ) as cur:
                    active = (await cur.fetchone())[0]
                if active >= SOFT_CAP_LIMIT:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "soft_cap_exceeded",
                            "active": active,
                            "limit": SOFT_CAP_LIMIT,
                        },
                    )

                # CR-02: cap not hit, soft-window missed → an active chain on
                # this item already exists at iter*_pending/iter*_drafted. Look
                # it up and return 409 chain_already_active so the client can
                # not spawn a parallel chain on the same digest item.
                async with db.execute(
                    "SELECT id FROM reactions WHERE item_id=? AND action='finish' "
                    "AND chain_state IN ('iter1_pending','iter1_drafted',"
                    "                    'iter2_pending','iter2_drafted',"
                    "                    'iter3_pending','iter3_drafted') "
                    "ORDER BY id DESC LIMIT 1",
                    (item_id,),
                ) as cur:
                    existing = await cur.fetchone()
                existing_rid = existing[0] if existing else None
                logger.warning(
                    "finish chain_already_active item_id=%s existing_rid=%s",
                    item_id, existing_rid,
                )
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "chain_already_active",
                        "item_id": item_id,
                        "existing_reaction_id": existing_rid,
                    },
                )
            return {"status": "queued", "reaction_id": rid}

        # 4. Skip semantics (REACT-05) — worker bypass via status='reviewed'.
        status = "reviewed" if action == SKIP_ACTION else "pending"

        # 5. Insert (with 60s soft-window inside the helper).
        try:
            rid = await insert_reaction(db, item_id, action, status=status)
        except sqlite3.IntegrityError as exc:
            # Partial UNIQUE collision — race-loser path (T-16-08).
            # WR-01 fix: do NOT leak `str(exc)` (column/index names) into the
            # HTTP response. Log the raw IntegrityError for diagnostics, return
            # a static client-safe message.
            logger.warning(
                "reaction race-loser IntegrityError item_id=%s action=%r: %s",
                item_id,
                action,
                exc,
            )
            raise HTTPException(
                status_code=409,
                detail=(
                    f"reaction for item_id {item_id} action {action!r} "
                    f"already in progress (race-loser path)"
                ),
            )

        # 5. Soft-window hit → fetch the existing reaction id so the client
        # gets the SAME reaction_id on a repeat click within 60s (REACT-04).
        # CR-01 fix: same Python-ISO threshold normalisation as in
        # insert_reaction — see src/database/client.py:insert_reaction for
        # rationale. SQLite datetime('now',...) emits a space separator while
        # created_at uses 'T' (Python isoformat), causing lexicographic >= to
        # widen the window to ~24h.
        if rid is None:
            threshold = (
                datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=60)
            ).isoformat()
            async with db.execute(
                "SELECT id FROM reactions WHERE item_id=? AND action=? "
                "AND created_at >= ? "
                "ORDER BY id DESC LIMIT 1",
                (item_id, action, threshold),
            ) as cur:
                hit = await cur.fetchone()
            rid = hit[0] if hit else None
    finally:
        await db.close()

    if action == SKIP_ACTION:
        return {"status": "skipped", "reaction_id": rid}
    return {"status": "queued", "reaction_id": rid}
