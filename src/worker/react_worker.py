"""Worker entrypoint (WORKER-01 + WORKER-02 + WORKER-03 + WORKER-09 + WORKER-11).

Fired by Windows Task Scheduler (AutorssReactWorker, /SC MINUTE /MO 1).
Each invocation = one tick = drain all pending reactions (up to hard cap
of 50, with a 3-consecutive-failure circuit breaker).

All env-var reads INSIDE function bodies (WORKER-12 / CI guard from plan 17-04).
"""
from __future__ import annotations

import datetime
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Final

# IN-02 (REVIEW-FIX P20): ``traceback`` is only referenced in the
# unexpected-exception branch of _dispatch (rare path). Importing it inside
# the except block keeps the cost off the happy-path import chain. Python
# caches the module after the first import, so subsequent failures incur
# only a dict lookup.

import logging

from src.database.client import claim_pending_reaction
from src.database.connection import open_db_sync
from src.database.metrics_client import insert_metric_event_sync, get_metrics_db_path
from src.rl import REWARD_BY_ACTION
from src.worker import auto_archive
from src.worker.claude_subprocess import (
    ClaudeSubprocessError,
    ClaudeSubprocessTimeout,
)
from src.worker.digest_entry import extract_digest_entry
from src.worker.logging_jsonl import log_reaction_event
from src.worker.modes import brainstorm as brainstorm_mode
from src.worker.modes import debate as debate_mode
from src.worker.modes import docs as docs_mode
from src.worker.modes import finish as finish_mode
from src.worker.modes import link as link_mode
from src.worker.vault import is_writable

logger = logging.getLogger(__name__)

_DRAIN_HARD_CAP: Final[int] = 50
_CIRCUIT_BREAKER_THRESHOLD: Final[int] = 3
_DEFAULT_DB_PATH: Final[str] = "curator.db"
_DEFAULT_VAULT_PATH: Final[str] = "vault"

# Mode dispatch — resolved BY NAME at call time so monkeypatching the
# underlying module's ``handle`` attribute in tests works. Capturing the
# function reference at import time would freeze the binding and bypass
# patches applied later.
_MODE_MODULES: dict = {
    "brainstorm": brainstorm_mode,
    "docs": docs_mode,
    "link": link_mode,
    "debate": debate_mode,  # Phase 73 / DEBATE-REACT-01
    "finish": finish_mode,  # Phase 20 / WORKER-06
}

# Phase 30 / RL-02 — reaction action → human_reward float mapping.
# CONTEXT D-RL-02: brainstorm/docs/link/finish → +1.0; skip → -1.0.
# Same value for all positive actions to avoid premature reward-weighting
# bias before Phase 31 scalarization is designed.
#
# Phase 30 / WR-03 REVIEW-FIX: canonical mapping moved to src/rl/__init__.py
# so that scripts/backfill_human_reward.py + Phase 31 aggregator can import
# the same dict without duplicating the literal (which previously required
# a parity test to guard against drift). The module-level ``_REWARD_BY_ACTION``
# alias is preserved so that existing tests + callers that reference
# ``react_worker._REWARD_BY_ACTION`` keep working.
_REWARD_BY_ACTION: Final[dict[str, float]] = REWARD_BY_ACTION


def _resolve_handler(action: str):
    """Look up ``modes.<action>.handle`` dynamically (test-patchable)."""
    mod = _MODE_MODULES.get(action)
    if mod is None:
        return None
    return getattr(mod, "handle", None)


def _fetch_digest_item(db: sqlite3.Connection, item_id: int) -> dict | None:
    """Lookup the latest digest_items row for ``item_id`` (read-only).

    Also extracts ``entry_md`` — the rendered paragraph for this item from
    ``digests.markdown_content`` (the same line the operator sees in the web
    UI). The mode handlers prepend it to the draft body so the operator can
    judge the draft against the full curated entry, not just a snippet.
    """
    cur = db.execute(
        "SELECT item_id, digest_id, post_id, channel, url, snippet, linked_project "
        "FROM digest_items WHERE item_id=? ORDER BY digest_id DESC LIMIT 1",
        (item_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    digest_md: str | None = None
    if row[1] is not None:
        try:
            c2 = db.execute(
                "SELECT markdown_content FROM digests WHERE id=?",
                (row[1],),
            )
            r2 = c2.fetchone()
            if r2 is not None:
                digest_md = r2[0]
        except sqlite3.OperationalError:
            # ``digests`` table may be absent in narrow test fixtures that
            # only seed ``digest_items`` — fall back to entry_md=None.
            digest_md = None
    return {
        "item_id": row[0],
        "digest_id": row[1],
        "post_id": row[2],
        "channel": row[3],
        "url": row[4],
        "snippet": row[5],
        "linked_project": row[6],
        "entry_md": extract_digest_entry(digest_md, row[0]),
    }


def _mark_drafted(
    db: sqlite3.Connection,
    reaction_id: int,
    draft_path: str,
    *,
    action: str | None = None,
    current_iter: int | None = None,
) -> None:
    """Mark a row as drafted.

    Phase 20 / WORKER-06: when ``action == 'finish'`` and ``current_iter`` is
    one of {1, 2, 3}, ALSO write ``chain_state = 'iter<N>_drafted'`` so the
    cross_db.try_mark_continued expected-state guard matches on /post-draft
    close (plan 20-02). For non-finish actions OR an out-of-range
    current_iter, fall through to the legacy 3-column UPDATE — chain_state
    column is left untouched (NULL on non-finish rows by schema design).

    CR-01 (REVIEW-FIX P20): the finish-action UPDATE is GUARDED on the
    expected pre-state (``status='processing'`` AND chain_state matches
    ``iter<N>_pending``). If a kill/defer raced ahead between claim and
    handler completion, ``cur.rowcount == 0`` and we MUST NOT resurrect
    the chain. The orphaned draft file is unlinked (best-effort) and the
    incident is logged. Non-finish UPDATEs keep the legacy unguarded shape
    (no chain_state machinery to corrupt).
    """
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    if action == "finish" and current_iter in (1, 2, 3):
        expected_chain_state = f"iter{current_iter}_pending"
        cur = db.execute(
            "UPDATE reactions SET status='drafted', draft_path=?, "
            "chain_state=?, updated_at=? "
            "WHERE id=? AND status='processing' AND chain_state=?",
            (draft_path, f"iter{current_iter}_drafted", now_iso,
             reaction_id, expected_chain_state),
        )
        db.commit()
        if cur.rowcount == 0:
            # Row was killed/deferred mid-handle (or otherwise advanced).
            # Cleanup the orphaned draft file and log; do NOT resurrect
            # the chain. Best-effort unlink — vault may be offline.
            try:
                os.unlink(draft_path)
            except OSError:
                pass
            log_reaction_event({
                "event": "draft_orphan_cleanup",
                "reaction_id": reaction_id,
                "reason": "chain_state changed mid-handle",
                "expected_chain_state": expected_chain_state,
            })
        return
    db.execute(
        "UPDATE reactions SET status='drafted', draft_path=?, updated_at=? "
        "WHERE id=?",
        (draft_path, now_iso, reaction_id),
    )
    db.commit()


def _mark_failed(
    db: sqlite3.Connection,
    reaction_id: int,
    error_msg: str,
) -> None:
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    db.execute(
        "UPDATE reactions SET status='failed', error_msg=?, updated_at=? "
        "WHERE id=?",
        (error_msg[:500], now_iso, reaction_id),  # T-17-03 / T-17-24 truncation
    )
    db.commit()


def _write_human_reward(
    db: sqlite3.Connection,
    item_id: int,
    action: str,
) -> None:
    """Phase 30 / RL-02: broadcast human_reward to all curation_logs rows for this post.

    Idempotent via ``human_reward IS NULL`` guard — re-running on already-populated
    rows is a no-op. NFR-03 fault-isolation: a ``sqlite3.Error`` never raises out of
    this helper — it logs a ``human_reward_fail`` event and returns. Multi-row note:
    a post can have 2+ curation_logs rows (re-curation by different backend/stage).
    All NULL rows receive the same reward.

    Unknown actions emit ``human_reward_skip`` and write nothing — the dict lookup
    is the firewall (T-30-03 mitigation).
    """
    reward = _REWARD_BY_ACTION.get(action)
    if reward is None:
        log_reaction_event({
            "event": "human_reward_skip",
            "item_id": item_id,
            "action": action,
            "reason": "action_not_in_reward_map",
        })
        return
    try:
        cur = db.execute(
            "UPDATE curation_logs SET human_reward = ? "
            "WHERE post_id = ? AND human_reward IS NULL",
            (reward, item_id),
        )
        db.commit()
        if cur.rowcount == 0:
            # Orphan OR already-populated (idempotent re-run).
            log_reaction_event({
                "event": "human_reward_noop",
                "item_id": item_id,
                "action": action,
                "reason": "no_curation_logs_or_already_populated",
            })
    except sqlite3.Error as e:
        # NFR-03: reward write failure MUST NOT block subsequent reactions.
        log_reaction_event({
            "event": "human_reward_fail",
            "item_id": item_id,
            "action": action,
            "error": f"{type(e).__name__}: {e}",
        })


def _dispatch(
    reaction_row: dict,
    digest_item: dict | None,
    vault_drafts_dir: str,
    db: sqlite3.Connection,
) -> bool:
    """Dispatch a single claimed reaction. Returns True on success, False on failure.

    All exceptions in the mode handler are caught here (NFR-03 isolation):
    one bad reaction never blocks subsequent rows in the same drain tick.
    """
    rid = reaction_row["id"]
    action = reaction_row["action"]
    start = time.monotonic()

    # No digest_item row for this item_id → soft failure (data inconsistency).
    if digest_item is None:
        _mark_failed(
            db, rid,
            f"digest_item missing for item_id={reaction_row['item_id']}",
        )
        try:
            insert_metric_event_sync(
                get_metrics_db_path(),
                item_id=reaction_row["item_id"],
                action=action,
                outcome="failed",
            )
        except Exception:
            pass
        log_reaction_event({
            "event": "draft_fail",
            "reaction_id": rid,
            "error": "digest_item_missing",
        })
        return False

    handler = _resolve_handler(action)
    if handler is None:
        _mark_failed(db, rid, f"mode {action} not yet implemented in P17")
        try:
            insert_metric_event_sync(
                get_metrics_db_path(),
                item_id=reaction_row["item_id"],
                action=action,
                outcome="failed",
            )
        except Exception:
            pass
        log_reaction_event({
            "event": "draft_fail",
            "reaction_id": rid,
            "error": f"mode_not_implemented:{action}",
        })
        return False

    try:
        log_reaction_event({
            "event": "draft_write",
            "reaction_id": rid,
            "action": action,
        })
        draft_path = handler(reaction_row, digest_item, vault_drafts_dir)
        # Phase 30 / RL-02: write human_reward to curation_logs BEFORE _mark_drafted.
        # Best-effort, never raises (NFR-03). Broadcasts to all curation_logs rows
        # for this post_id where human_reward IS NULL.
        _write_human_reward(db, reaction_row["item_id"], action)
        _mark_drafted(
            db, rid, draft_path,
            action=action,
            current_iter=reaction_row.get("current_iter"),
        )
        try:
            insert_metric_event_sync(
                get_metrics_db_path(),
                item_id=reaction_row["item_id"],
                action=action,
                outcome="drafted",
            )
        except Exception:
            pass
        duration = time.monotonic() - start
        log_reaction_event({
            "event": "draft_done",
            "reaction_id": rid,
            "path": draft_path,
            "duration_s": round(duration, 2),
        })
        return True
    except ClaudeSubprocessTimeout as e:
        _mark_failed(db, rid, f"claude timeout: {e}")
        try:
            insert_metric_event_sync(
                get_metrics_db_path(),
                item_id=reaction_row["item_id"],
                action=action,
                outcome="failed",
            )
        except Exception:
            pass
        log_reaction_event({
            "event": "draft_fail",
            "reaction_id": rid,
            "error": "claude_timeout",
        })
        return False
    except ClaudeSubprocessError as e:
        _mark_failed(db, rid, f"claude error: {e}")
        try:
            insert_metric_event_sync(
                get_metrics_db_path(),
                item_id=reaction_row["item_id"],
                action=action,
                outcome="failed",
            )
        except Exception:
            pass
        log_reaction_event({
            "event": "draft_fail",
            "reaction_id": rid,
            "error": "claude_error",
        })
        return False
    except Exception as e:  # NFR-03 isolation: catch ALL unexpected exceptions
        # IN-01: actually USE the traceback in the log event (previously
        # collected and discarded). error_msg in DB stays short (500 cap),
        # but the full short traceback lands in the JSONL log for later
        # diagnosis without blocking the drain.
        # IN-02 (REVIEW-FIX P20): import traceback here so the cost is paid
        # only on the (rare) failure path, not on every worker tick startup.
        import traceback
        tb = traceback.format_exc(limit=3)
        _mark_failed(db, rid, f"unexpected: {type(e).__name__}: {e}")
        try:
            insert_metric_event_sync(
                get_metrics_db_path(),
                item_id=reaction_row["item_id"],
                action=action,
                outcome="failed",
            )
        except Exception:
            pass
        log_reaction_event({
            "event": "draft_fail",
            "reaction_id": rid,
            "error": "unexpected",
            "exc_type": type(e).__name__,
            "traceback": tb,
        })
        return False


def run_tick(db_path: str, vault_drafts_dir: str) -> dict:
    """Drain all pending reactions in one tick (up to hard cap of 50).

    Flow (Phase 21 / GATE-06 ordering):
      1. Open DB (cheap, no vault I/O).
      2. Auto-archive sweep: stale 'drafted' rows >7d → 'archived', files
         moved to 10_Drafts/_archive/. Wrapped in try/except so a sweep
         failure CANNOT block the rest of the tick (NFR-03 isolation).
      3. Vault preflight: if drafts dir not writable, skip drain (still
         return archived count from step 2 — DB updates already committed).
      4. Loop until claim returns None or hard cap reached.
      5. Dispatch each claimed reaction; track success/failure counters.
      6. Trip circuit breaker after 3 consecutive failures (DEGRADED).
      7. Close DB; return stats dict.
    """
    db = open_db_sync(db_path)
    # WR-01: sweep returns SweepResult; track archived (DB), moved + missing
    # (filesystem) separately so log/metrics can show divergence.
    archived_count = 0
    archived_files_moved = 0
    archived_files_missing = 0
    try:
        result = auto_archive.sweep(db, vault_drafts_dir)
        archived_count = result.archived
        archived_files_moved = result.files_moved
        archived_files_missing = result.files_missing
    except Exception as e:
        # Sweep failure must NOT block drain (NFR-03 isolation).
        # IN-03: db connection lifetime across this except is non-obvious.
        # Three exit paths each close it exactly once:
        #   1. vault offline → explicit db.close() inside the next ``if``
        #   2. happy path  → drain loop's ``finally`` closes db at line 343
        #   3. unexpected raise inside drain → same ``finally`` covers it
        # Re-closing in this except would double-close on path 2/3. Letting
        # the sweep error fall through preserves the single-close guarantee.
        log_reaction_event({
            "event": "auto_archive_unexpected",
            "error": f"{type(e).__name__}: {e}",
        })

    if not is_writable(vault_drafts_dir):
        logger.warning(
            "DRAFT_OUTPUT_DIR not writable or missing: %s — worker tick skipped. "
            "Set DRAFT_OUTPUT_DIR env var to a writable directory.",
            vault_drafts_dir,
        )
        try:
            db.close()
        except Exception:
            pass
        log_reaction_event({
            "event": "tick_skip",
            "reason": "vault_offline",
            "vault_path": vault_drafts_dir,
            "archived": archived_count,
            "archived_files_moved": archived_files_moved,
            "archived_files_missing": archived_files_missing,
        })
        return {
            "drained": 0,
            "failed": 0,
            "circuit_broken": False,
            "skipped_vault_offline": True,
            "archived": archived_count,
            "archived_files_moved": archived_files_moved,
            "archived_files_missing": archived_files_missing,
        }

    drained = 0
    failed = 0
    consecutive_failures = 0
    circuit_broken = False
    iterations = 0

    try:
        while iterations < _DRAIN_HARD_CAP:
            iterations += 1
            reaction_row = claim_pending_reaction(db)
            if reaction_row is None:
                break  # queue empty
            log_reaction_event({
                "event": "claim",
                "reaction_id": reaction_row["id"],
                "item_id": reaction_row["item_id"],
                "action": reaction_row["action"],
            })
            digest_item = _fetch_digest_item(db, reaction_row["item_id"])
            ok = _dispatch(reaction_row, digest_item, vault_drafts_dir, db)
            if ok:
                drained += 1
                consecutive_failures = 0  # success resets the streak
            else:
                failed += 1
                consecutive_failures += 1
                if consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
                    circuit_broken = True
                    log_reaction_event({
                        "event": "circuit_breaker_tripped",
                        "consecutive_failures": consecutive_failures,
                    })
                    break
        log_reaction_event({
            "event": "tick_done",
            "drained": drained,
            "failed": failed,
            "circuit_broken": circuit_broken,
            "iterations": iterations,
            "archived": archived_count,
            "archived_files_moved": archived_files_moved,
            "archived_files_missing": archived_files_missing,
        })
    finally:
        try:
            db.close()
        except Exception:
            pass
    return {
        "drained": drained,
        "failed": failed,
        "circuit_broken": circuit_broken,
        "skipped_vault_offline": False,
        "archived": archived_count,
        "archived_files_moved": archived_files_moved,
        "archived_files_missing": archived_files_missing,
    }


def main() -> int:
    """Task Scheduler entrypoint. Returns 0 on clean tick, 1 on catastrophic error.

    WORKER-12: env reads MUST be inside this function body (CI guard from
    plan 17-04 enforces no module-top ``os.environ.get(`` in src/worker/).
    """
    # schtasks fires from System32 — chdir to project root so curator.db resolves.
    os.chdir(Path(__file__).resolve().parent.parent.parent)
    # Phase 107 TRIM-02: personal paths live in .env now (code defaults are generic).
    # Scheduled-task invocations don't inherit a shell env — load .env explicitly.
    from src.env_loader import load_env

    load_env()
    db_path = os.environ.get("DB_PATH", _DEFAULT_DB_PATH)
    vault_root = (
        os.environ.get("DRAFT_OUTPUT_DIR")
        or os.environ.get("OBSIDIAN_VAULT_PATH", _DEFAULT_VAULT_PATH)
    )
    vault_drafts_dir = str(Path(vault_root) / "10_Drafts")
    try:
        run_tick(db_path, vault_drafts_dir)
        return 0
    except Exception as e:
        # Catastrophic startup failure — log and exit non-zero.
        log_reaction_event({
            "event": "tick_catastrophic",
            "error": f"{type(e).__name__}: {e}",
        })
        return 1


if __name__ == "__main__":
    sys.exit(main())
