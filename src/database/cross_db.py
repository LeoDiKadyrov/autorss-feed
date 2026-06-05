"""Public cross-project entry point for reactions write-back.

second_brain companion handler imports this module (or copies the helpers
verbatim) when implementing the `/post-draft` digest-reaction close actions
(continue / defer / kill). Schema contract: see
`.planning/contracts/v1.2-second-brain.md`.

Phase 19 / REVIEW-04, NFR-06, NFR-07. Provides:

  - connect_curator(db_path)      — wraps open_db_sync + PRAGMA introspection
  - mark_archived(db, rid)        — close after continue (status='archived')
  - mark_deferred(db, rid)        — close after defer    (status='deferred')
  - mark_killed(db, rid)          — close after kill     (status='killed' + null draft_path)
  - try_mark_archived(...)        — variant returning CloseResult enum (WR-01)
  - try_mark_deferred(...)        — same; second_brain SHOULD prefer these
  - try_mark_killed(...)          — same; raise-style strict helpers stay
                                    available for autorss-internal callers

Design notes:
  - PRAGMA introspection on every connect (NFR-06 defense-in-depth, T-19-07).
    A stale aiosqlite or forgotten PRAGMA on the second_brain side WILL fail
    loudly here instead of silently corrupting concurrent worker writes.
  - Status guards are FORWARD-ONLY (CR-02). Each helper accepts only the
    pre-close states for its own target; cross-terminal flips
    (archived -> deferred, deferred -> killed) are forbidden because they
    silently corrupt audit history. Idempotent re-call on an
    already-archived/deferred row is NOT supported via cross_db -- callers
    must check current status before re-issuing.
        mark_archived: status IN ('drafted', 'reviewed')
        mark_deferred: status IN ('drafted', 'reviewed')
        mark_killed:   status NOT IN ('archived', 'killed')   -- universal
                       cleanup, but not from terminal states. Includes
                       'failed' for crashed-worker recovery (WR-06).
  - All UPDATEs use ? placeholders; reaction_id type-checked by sqlite3
    binding -> no string concatenation, no SQL injection (T-19-04).
  - CURATOR_DB_PATH env var read INSIDE connect_curator (recurring-trap
    pattern, see MEMORY: reference_module_level_env_binding).
  - Forbidden ops in this module: INSERT, DELETE, ALTER, schema mutations,
    raw markdown re-parse. second_brain MUST NOT add ad-hoc SQL outside
    these helpers (T-19-01); contract document and Phase 19-04 PR checklist
    enforce it.
"""
import contextlib
import datetime
import enum
import logging
import os
import sqlite3

from src.database.connection import open_db_sync

_log = logging.getLogger(__name__)


class CloseResult(enum.Enum):
    """Outcome enum returned by the try_mark_* family (WR-01).

    Per contract v1.2 line 135: 'second_brain SHOULD log a warning and
    surface it to the user, NOT retry blindly' when the close UPDATE
    affects 0 rows. The strict mark_* helpers raise ValueError; the
    try_mark_* variants catch that ValueError and convert it to a
    NOT_IN_ALLOWED_STATE result so callers can branch on a value rather
    than wrap every call in try/except.
    """

    CLOSED = "closed"  # rowcount == 1, status transitioned forward
    NOT_IN_ALLOWED_STATE = "not_in_allowed_state"  # rowcount == 0
    # Phase 20 / WORKER-06 + REVIEW-02: finish-chain state-machine variants.
    # CHAIN_PROGRESSED — try_mark_continued advanced iter1->iter2 or iter2->iter3.
    # CHAIN_COMPLETE   — try_mark_continued on iter3_drafted -> archived (final).
    # STATE_CONFLICT   — chain_state guard rejected the UPDATE (rowcount=0):
    #                    double-continue, continue-after-kill, continue-on-deferred,
    #                    wrong-iter arg, stale row race, non-finish action.
    CHAIN_PROGRESSED = "chain_progressed"
    CHAIN_COMPLETE = "chain_complete"
    STATE_CONFLICT = "state_conflict"


def _now_iso() -> str:
    """ISO-8601 UTC timestamp matching src/worker/react_worker.py::_mark_drafted."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def connect_curator(db_path: str | None = None) -> sqlite3.Connection:
    """Open curator.db with WAL/busy_timeout/synchronous PRAGMAs verified.

    If `db_path` is None, read `CURATOR_DB_PATH` env var (default `curator.db`).
    Env read happens INSIDE the function — never at import — so test fixtures
    can set the var per-test via monkeypatch.setenv.

    Verifies via PRAGMA introspection that:
      - journal_mode is 'wal' (case-insensitive)
      - busy_timeout is 5000
      - synchronous is 1 (NORMAL)

    Raises RuntimeError if any check fails. second_brain MUST NOT swallow
    this exception — proceeding without WAL risks corrupting concurrent
    worker writes (T-19-07).
    """
    if db_path is None:
        db_path = os.environ.get("CURATOR_DB_PATH", "curator.db")

    db = open_db_sync(db_path)

    # PRAGMA introspection — defense-in-depth (NFR-06).
    jm = db.execute("PRAGMA journal_mode").fetchone()[0]
    bt = db.execute("PRAGMA busy_timeout").fetchone()[0]
    sync = db.execute("PRAGMA synchronous").fetchone()[0]

    if str(jm).lower() != "wal":
        db.close()
        raise RuntimeError(
            f"cross_db: PRAGMA journal_mode not applied (got={jm!r}, want='wal') "
            "- second_brain MUST NOT proceed"
        )
    if bt != 5000:
        db.close()
        raise RuntimeError(
            f"cross_db: PRAGMA busy_timeout not applied (got={bt}, want=5000) "
            "- second_brain MUST NOT proceed"
        )
    if sync != 1:
        db.close()
        raise RuntimeError(
            f"cross_db: PRAGMA synchronous not applied (got={sync}, want=1 NORMAL) "
            "- second_brain MUST NOT proceed"
        )

    return db


def open_curator_readonly(db_path: str | None = None) -> sqlite3.Connection:
    """Read-only access to curator.db for briefing/reporting modules.

    Phase 22 / NFR-07. Companion to connect_curator() above (Phase 19).

    For second_brain briefing.py + any cross-repo reader that MUST NOT
    contend with worker writes. NFR-07: no write-intent lock acquisition,
    no row-level lock contention with the autorss worker tick.

    If db_path is None, reads CURATOR_DB_PATH env var (default 'curator.db').
    Env read happens INSIDE the function — never at import — so test fixtures
    can monkeypatch.setenv per-test (recurring-trap pattern, see MEMORY:
    reference_module_level_env_binding).

    Returns a sqlite3 Connection with:
      - mode=ro URI — INSERT/UPDATE/DELETE raise OperationalError
        ('attempt to write a readonly database')
      - busy_timeout=5000 — absorbs writer-commit waits without raising
        'database is locked' under contention
      - row_factory=sqlite3.Row — dict-like access for briefing
        ('row["status"]', 'row["item_id"]')

    Notes:
      - PRAGMA journal_mode is intentionally NOT re-emitted. WAL is
        per-file state set by writer connections; emitting on a ro conn
        raises 'attempt to write a readonly database' (verified by
        tests/integration/test_cross_db_companion.py::_reader_loop).
      - WAL/synchronous PRAGMA introspection is NOT done on the ro conn —
        a reader cannot fix a misconfigured DB; introspection is for
        writers (see connect_curator above).
      - db_path MUST be a plain filesystem path; URI components beyond
        mode=ro are not supported (passing 'foo?cache=shared' breaks the
        f-string concat). T-22-03 disposition: accept (developer error,
        not adversarial input — db_path comes from env var or hardcoded
        default, not network input).
      - second_brain briefing.py SHOULD import this function or copy it
        verbatim — see deferred-items.md companion PR checklist.
    """
    if db_path is None:
        db_path = os.environ.get("CURATOR_DB_PATH", "curator.db")
    uri = f"file:{db_path}?mode=ro"
    db = sqlite3.connect(uri, uri=True, timeout=5.0)
    db.execute("PRAGMA busy_timeout=5000")
    db.row_factory = sqlite3.Row
    return db


def _close_reaction(
    db: sqlite3.Connection,
    reaction_id: int,
    new_status: str,
    *,
    allowed_from: tuple[str, ...],
    new_draft_path: str | None = None,
    null_draft_path: bool = False,
) -> None:
    """Internal helper: UPDATE reactions to a terminal status.

    Status guard is FORWARD-ONLY -- caller passes the explicit set of
    pre-close states permitted for this transition (CR-02). The set is
    inlined into the SQL via parameterised IN-clause so SQL injection is
    impossible (each entry is a separate ? bind).

    draft_path handling (CR-01 / contract v1.2 lines 102-108):
      - `null_draft_path=True` (set by mark_killed only -- not part of
        public API): SET draft_path = NULL.
      - `new_draft_path=<str>` (set by mark_archived / mark_deferred when
        the caller has the post-move final path): SET draft_path = ?.
      - both unset: draft_path column is left untouched (legacy
        backwards-compat -- caller has no new path).
    `null_draft_path=True` and `new_draft_path is not None` are mutually
    exclusive; passing both raises ValueError at call time.

    Raises ValueError if rowcount == 0 (no row in `allowed_from` matched
    `reaction_id`); call sites in mark_archived/mark_deferred/mark_killed
    catch and downgrade per WR-01.
    """
    if null_draft_path and new_draft_path is not None:
        raise ValueError(
            "cross_db._close_reaction: null_draft_path and new_draft_path "
            "are mutually exclusive"
        )

    placeholders = ",".join("?" for _ in allowed_from)
    if null_draft_path:
        sql = (
            "UPDATE reactions "
            "SET status = ?, updated_at = ?, draft_path = NULL "
            f"WHERE id = ? AND status IN ({placeholders})"
        )
        params = (new_status, _now_iso(), reaction_id, *allowed_from)
    elif new_draft_path is not None:
        sql = (
            "UPDATE reactions "
            "SET status = ?, updated_at = ?, draft_path = ? "
            f"WHERE id = ? AND status IN ({placeholders})"
        )
        params = (
            new_status,
            _now_iso(),
            new_draft_path,
            reaction_id,
            *allowed_from,
        )
    else:
        sql = (
            "UPDATE reactions "
            "SET status = ?, updated_at = ? "
            f"WHERE id = ? AND status IN ({placeholders})"
        )
        params = (new_status, _now_iso(), reaction_id, *allowed_from)

    with contextlib.closing(db.execute(sql, params)) as cur:
        db.commit()
        rowcount = cur.rowcount

    if rowcount == 0:
        raise ValueError(
            f"cross_db: row {reaction_id} not in allowed pre-close status "
            f"{allowed_from!r} (rowcount=0) - target_status={new_status!r}"
        )


# Public guard sets -- exposed (without underscore) so tests + second_brain
# can introspect the contract without re-deriving it.
ARCHIVED_ALLOWED_FROM = ("drafted", "reviewed")
DEFERRED_ALLOWED_FROM = ("drafted", "reviewed")
# Kill is the universal escape hatch: any non-terminal state. NOT from
# 'archived'/'killed' (terminal flips are forbidden, CR-02). Includes
# 'failed' so users can clean up crashed-worker rows (WR-06).
KILLED_ALLOWED_FROM = (
    "pending",
    "processing",
    "drafted",
    "reviewed",
    "deferred",
    "failed",
)


def mark_archived(
    db: sqlite3.Connection,
    reaction_id: int,
    draft_path: str | None = None,
) -> None:
    """Close a digest-reaction draft after `continue` verdict.

    Forward-only: status MUST be 'drafted' or 'reviewed'. Re-calling on an
    already-archived row raises ValueError (the rowcount==0 path); callers
    that need idempotent semantics MUST pre-check status before calling
    (see WR-01 for the warning-style downgrade pattern).

    Args:
        db: connect_curator() handle.
        reaction_id: reactions.id from the draft filename.
        draft_path: optional final path after second_brain moves the file
            (e.g. '20_Evergreen/Topic.md'). Per contract v1.2 lines 102-108,
            second_brain SHOULD pass the post-move location so subsequent
            reads (briefing module, Phase 22) point at the live file. If
            None, the existing draft_path column is left untouched
            (backwards-compat for callers that never moved the file).

    UPDATEs reactions.status -> 'archived', bumps updated_at, commits.

    Raises ValueError if row not in pre-close status (rowcount==0).
    """
    _close_reaction(
        db,
        reaction_id,
        "archived",
        allowed_from=ARCHIVED_ALLOWED_FROM,
        new_draft_path=draft_path,
    )


def mark_deferred(
    db: sqlite3.Connection,
    reaction_id: int,
    draft_path: str | None = None,
) -> None:
    """Close a digest-reaction draft after `defer` verdict.

    Forward-only: status MUST be 'drafted' or 'reviewed'. Same guard
    semantics as mark_archived -- re-issuing on a deferred row raises.

    Args:
        db: connect_curator() handle.
        reaction_id: reactions.id from the draft filename.
        draft_path: optional final path after move (defer can also relocate
            the draft to a `30_Deferred/` shelf). None = leave column
            untouched.

    UPDATEs reactions.status -> 'deferred', bumps updated_at, commits.

    Raises ValueError if row not in pre-close status (rowcount==0).
    """
    _close_reaction(
        db,
        reaction_id,
        "deferred",
        allowed_from=DEFERRED_ALLOWED_FROM,
        new_draft_path=draft_path,
    )


def mark_killed(db: sqlite3.Connection, reaction_id: int) -> None:
    """Close a digest-reaction draft after `kill` verdict.

    Universal cleanup: status MUST be in any non-terminal state
    (KILLED_ALLOWED_FROM). Killing an already-archived or already-killed
    row raises -- terminal flips are forbidden because they corrupt the
    audit trail (CR-02).

    UPDATEs reactions.status -> 'killed', NULLs draft_path, bumps
    updated_at, commits.

    Raises ValueError if row in 'archived' or 'killed' status (rowcount==0).
    """
    _close_reaction(
        db,
        reaction_id,
        "killed",
        allowed_from=KILLED_ALLOWED_FROM,
        null_draft_path=True,
    )


# ---------------------------------------------------------------------------
# WR-01: try_mark_* variants -- non-raising, return CloseResult enum.
#
# second_brain SHOULD prefer these over the raise-style helpers above so
# the caller can log a warning + surface it to the user (per contract
# v1.2 line 135) without wrapping every close call in try/except.
# ---------------------------------------------------------------------------


def try_mark_archived(
    db: sqlite3.Connection,
    reaction_id: int,
    draft_path: str | None = None,
) -> CloseResult:
    """Non-raising variant of mark_archived (WR-01).

    Returns CloseResult.CLOSED on success, CloseResult.NOT_IN_ALLOWED_STATE
    on rowcount==0. Emits a logging.warning so operators see the event in
    logs even when the caller silently branches on the return value.
    """
    try:
        mark_archived(db, reaction_id, draft_path=draft_path)
    except ValueError as exc:
        _log.warning(
            "cross_db.try_mark_archived: rid=%s not in allowed pre-close "
            "status; second_brain SHOULD surface to user (contract v1.2 "
            "line 135). detail=%s",
            reaction_id,
            exc,
        )
        return CloseResult.NOT_IN_ALLOWED_STATE
    return CloseResult.CLOSED


def try_mark_deferred(
    db: sqlite3.Connection,
    reaction_id: int,
    draft_path: str | None = None,
) -> CloseResult:
    """Non-raising variant of mark_deferred (WR-01)."""
    try:
        mark_deferred(db, reaction_id, draft_path=draft_path)
    except ValueError as exc:
        _log.warning(
            "cross_db.try_mark_deferred: rid=%s not in allowed pre-close "
            "status; second_brain SHOULD surface to user. detail=%s",
            reaction_id,
            exc,
        )
        return CloseResult.NOT_IN_ALLOWED_STATE
    return CloseResult.CLOSED


def try_mark_killed(
    db: sqlite3.Connection, reaction_id: int
) -> CloseResult:
    """Non-raising variant of mark_killed (WR-01)."""
    try:
        mark_killed(db, reaction_id)
    except ValueError as exc:
        _log.warning(
            "cross_db.try_mark_killed: rid=%s not in allowed pre-close "
            "status; second_brain SHOULD surface to user. detail=%s",
            reaction_id,
            exc,
        )
        return CloseResult.NOT_IN_ALLOWED_STATE
    return CloseResult.CLOSED


# ---------------------------------------------------------------------------
# Phase 20 / WORKER-06 + REVIEW-02: finish-chain state-machine helpers.
#
# Each helper is a SINGLE atomic UPDATE with an explicit chain_state guard
# in the WHERE clause — invalid transitions (double-continue,
# continue-after-kill, continue-on-deferred, edited-stale-row races) are
# impossible at the SQL level. No SELECT-then-UPDATE pattern, no
# Python-side TOCTOU window. action='finish' guard prevents accidental
# misuse against brainstorm/docs/link rows (those use the simpler Phase-19
# state machine via mark_archived/mark_deferred/mark_killed).
# ---------------------------------------------------------------------------

# Maps current_iter -> (expected_chain_state, next_chain_state, next_status,
#                      next_current_iter, is_final).
# Iter3 final continue is the only path that flips status='archived' AND
# leaves current_iter unchanged at 3 (no iter4).
_CONTINUE_TRANSITIONS: dict[int, tuple[str, str, str, int, bool]] = {
    1: ("iter1_drafted", "iter2_pending",   "pending",  2, False),
    2: ("iter2_drafted", "iter3_pending",   "pending",  3, False),
    3: ("iter3_drafted", "iter3_continued", "archived", 3, True),
}

# Kill is valid from any non-terminal chain_state. EXCLUDES 'killed',
# 'deferred', 'iter3_continued' — those are terminal for the kill helper
# (no resurrection, no archived-flip). Includes both pending + drafted
# variants of every iter so a worker-crash recovery can kill mid-generation.
KILLED_FROM_CHAIN_STATES = (
    "iter1_pending", "iter1_drafted",
    "iter2_pending", "iter2_drafted",
    "iter3_pending", "iter3_drafted",
)

# Defer = "park the user-edited draft", only meaningful from iter*_drafted.
# Deferring an iter*_pending row would silently lose a queued generation;
# explicit narrow guard rejects that case with STATE_CONFLICT.
DEFERRED_FROM_CHAIN_STATES = (
    "iter1_drafted", "iter2_drafted", "iter3_drafted",
)


def try_mark_continued(
    db: sqlite3.Connection,
    reaction_id: int,
    current_iter: int,
) -> CloseResult:
    """Phase 20 / WORKER-06 + REVIEW-02: advance a finish-chain row.

    Atomic single-statement UPDATE — the chain_state=expected guard rejects
    all invalid transitions (double-continue, continue-after-kill,
    continue-on-deferred, edited-stale-row races) at the SQL layer.
    rowcount=0 -> STATE_CONFLICT regardless of cause; caller MUST surface
    to user (contract v1.2 line 135).

    Args:
        db: connect_curator() handle.
        reaction_id: reactions.id.
        current_iter: caller's view of the row's current iter (1, 2, or 3).
            The UPDATE additionally guards on chain_state, so a stale view
            still rejects safely (no silent skip-ahead).

    Returns:
        CloseResult.CHAIN_PROGRESSED on iter1->iter2 or iter2->iter3.
        CloseResult.CHAIN_COMPLETE   on iter3 final continue (status='archived').
        CloseResult.STATE_CONFLICT   on rowcount=0 (any reason).

    STATE_CONFLICT semantics — IMPORTANT for callers (REVIEW-FIX P20 / WR-03):
        STATE_CONFLICT is NOT a "user did something wrong" signal. The row's
        chain_state may have legitimately advanced between the caller's
        SELECT and this UPDATE (TOCTOU window — common with second_brain
        callers that read-then-call). Callers SHOULD treat STATE_CONFLICT as
        "refresh the row and retry once" — re-SELECT chain_state, recompute
        the expected current_iter, and call again with the fresh value. If
        the re-SELECT shows a terminal state (killed/deferred/iter3_continued),
        STATE_CONFLICT is final.

        Although ``current_iter`` is technically derivable from chain_state
        (the helper internally maps it via _CONTINUE_TRANSITIONS), the
        explicit parameter is preserved as a footgun-guard: an UPDATE with
        a stale current_iter view safely rejects rather than silently
        skipping ahead. See test_continue_with_wrong_iter_arg_rejected for
        the documented behavior.
    """
    if current_iter not in _CONTINUE_TRANSITIONS:
        _log.warning(
            "cross_db.try_mark_continued: invalid current_iter=%s for rid=%s",
            current_iter, reaction_id,
        )
        return CloseResult.STATE_CONFLICT
    expected, next_state, next_status, next_iter, is_final = (
        _CONTINUE_TRANSITIONS[current_iter]
    )
    sql = (
        "UPDATE reactions SET "
        "chain_state = ?, status = ?, current_iter = ?, updated_at = ? "
        "WHERE id = ? AND chain_state = ? AND action = 'finish'"
    )
    params = (
        next_state, next_status, next_iter, _now_iso(),
        reaction_id, expected,
    )
    with contextlib.closing(db.execute(sql, params)) as cur:
        db.commit()
        rowcount = cur.rowcount
    if rowcount == 0:
        _log.warning(
            "cross_db.try_mark_continued: rid=%s expected chain_state=%r "
            "(rowcount=0) -- already advanced/killed/deferred or not-finish",
            reaction_id, expected,
        )
        return CloseResult.STATE_CONFLICT
    return CloseResult.CHAIN_COMPLETE if is_final else CloseResult.CHAIN_PROGRESSED


def try_mark_iter_killed(
    db: sqlite3.Connection,
    reaction_id: int,
) -> CloseResult:
    """Phase 20: kill a finish-chain row at any non-terminal iter state.

    Atomic UPDATE with chain_state IN KILLED_FROM_CHAIN_STATES guard.
    Sets BOTH status='killed' AND chain_state='killed' AND draft_path=NULL
    in one statement. Already-killed/deferred/iter3_continued rows return
    STATE_CONFLICT (rowcount=0).

    Returns CloseResult.CLOSED on success, CloseResult.STATE_CONFLICT on
    rowcount=0.
    """
    placeholders = ",".join("?" for _ in KILLED_FROM_CHAIN_STATES)
    sql = (
        "UPDATE reactions SET "
        "status = 'killed', chain_state = 'killed', "
        "draft_path = NULL, updated_at = ? "
        f"WHERE id = ? AND action = 'finish' AND chain_state IN ({placeholders})"
    )
    params = (_now_iso(), reaction_id, *KILLED_FROM_CHAIN_STATES)
    with contextlib.closing(db.execute(sql, params)) as cur:
        db.commit()
        rowcount = cur.rowcount
    if rowcount == 0:
        _log.warning(
            "cross_db.try_mark_iter_killed: rid=%s not in non-terminal "
            "chain_state (rowcount=0)", reaction_id,
        )
        return CloseResult.STATE_CONFLICT
    return CloseResult.CLOSED


def try_mark_iter_deferred(
    db: sqlite3.Connection,
    reaction_id: int,
    draft_path: str | None = None,
) -> CloseResult:
    """Phase 20: defer a finish-chain row at iter*_drafted state.

    Atomic UPDATE; only valid from iter1_drafted / iter2_drafted /
    iter3_drafted (DEFERRED_FROM_CHAIN_STATES). draft_path optional --
    second_brain may relocate the file before deferring. Already-deferred
    or iter*_pending rows return STATE_CONFLICT.

    Returns CloseResult.CLOSED on success, CloseResult.STATE_CONFLICT on
    rowcount=0.
    """
    placeholders = ",".join("?" for _ in DEFERRED_FROM_CHAIN_STATES)
    if draft_path is not None:
        sql = (
            "UPDATE reactions SET "
            "status = 'deferred', chain_state = 'deferred', "
            "draft_path = ?, updated_at = ? "
            f"WHERE id = ? AND action = 'finish' AND chain_state IN ({placeholders})"
        )
        params = (
            draft_path, _now_iso(), reaction_id, *DEFERRED_FROM_CHAIN_STATES,
        )
    else:
        sql = (
            "UPDATE reactions SET "
            "status = 'deferred', chain_state = 'deferred', "
            "updated_at = ? "
            f"WHERE id = ? AND action = 'finish' AND chain_state IN ({placeholders})"
        )
        params = (_now_iso(), reaction_id, *DEFERRED_FROM_CHAIN_STATES)
    with contextlib.closing(db.execute(sql, params)) as cur:
        db.commit()
        rowcount = cur.rowcount
    if rowcount == 0:
        _log.warning(
            "cross_db.try_mark_iter_deferred: rid=%s not in deferrable "
            "chain_state (rowcount=0)", reaction_id,
        )
        return CloseResult.STATE_CONFLICT
    return CloseResult.CLOSED
