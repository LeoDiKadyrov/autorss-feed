"""Phase 21 Plan 02 — Public precision-gate API.

Reads `~/.claude/draft-precision.jsonl`, filters rows by subtag prefix
`digest:`, applies recency decay + Wilson lower bound (Plan 21-01), runs
the cross-action curiosity unlock pass, and exposes:

  - is_unlocked(action: str) -> bool
  - get_gate_status() -> dict[str, dict]

Locked decisions (CONTEXT G-3 + 21-01 forward note):
  - z = 1.0364 (one-sided 0.7 cumulative; Plan 21-01 Z_CONF_70).
  - Thresholds: 0.5 for brainstorm/docs/link; 0.35 for finish.
  - **Min n>=6 floor:** even if Wilson > threshold, action stays LOCKED
    when decayed n < 6. With z=1.0364, raw 4/5 → Wilson 0.57 (clears 0.5);
    the floor preserves small-N rejection that the original 95%-CI math
    would have provided. Per Plan 21-01 forward note option (b).
  - Curiosity unlock: when ≥1 action UNLOCKED, every still-LOCKED action
    gets +1.0 added to its decayed n; re-evaluation can flip it (the
    +1 sample is uncertainty credit, NOT a guaranteed success). Applied once.
  - Default on missing/empty JSONL: ALL actions LOCKED (GATE-03).
  - 60s TTL + mtime-aware cache: avoid re-parsing JSONL on every UI request.

WORKER-12 mitigation: ALL `os.environ.get(...)` calls are INSIDE function
bodies (never at module top). The CI guard `scripts/check_module_env.py`
enforces this.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from src.worker.precision_math import decay_samples, wilson_lower_bound

ACTIONS: Final[tuple[str, ...]] = ("brainstorm", "docs", "link", "finish")
THRESHOLDS: Final[dict[str, float]] = {
    "brainstorm": 0.5,
    "docs": 0.5,
    "link": 0.5,
    "finish": 0.35,
}
MIN_N_FLOOR: Final[float] = 6.0  # forward note from 21-01 — small-N rejection
_SUBTAG_PREFIX: Final[str] = "digest:"
_CACHE_TTL_S: Final[float] = 60.0
_HALF_LIFE_DAYS: Final[float] = 7.0
# WR-04: ``_HALF_LIFE_SECONDS`` previously lived here but was never read.
# decay_samples() owns the days→seconds conversion internally; this module
# only passes ``half_life_days`` through. Do not re-introduce a duplicate
# constant unless you have a use site for it.
#
# WR-03: defense-in-depth cap on JSONL line length. The file is shared with
# second_brain; a buggy peer-process write (50 MB serialized stack trace,
# multi-MB embedded HTML) would otherwise be slurped into memory before
# json.loads rejects it. Typical row is ~200 bytes; 4 KiB tolerates large
# `reason` strings while bounding worst-case allocation per line.
_MAX_LINE_BYTES: Final[int] = 4096

# Module-level mutable cache (single-process worker; no concurrency concern).
# Keys:
#   samples_by_action: dict[action, list[(decision, age_seconds)]] | None
#   loaded_at: float (time.monotonic when cache populated)
#   mtime: float (file mtime when cache populated)
#   gate_status: dict[action, dict] | None  — IN-02 memo of get_gate_status()
#       result; invalidated whenever samples_by_action is re-parsed (same
#       loaded_at + mtime as the JSONL cache).
_cache: dict = {
    "samples_by_action": None,
    "loaded_at": 0.0,
    "mtime": 0.0,
    "gate_status": None,
}


def _resolve_jsonl_path() -> Path:
    """Resolve JSONL path. Reads env INSIDE function (WORKER-12 trap)."""
    override = os.environ.get("DRAFT_PRECISION_JSONL")
    if override:
        return Path(override)
    return Path.home() / ".claude" / "draft-precision.jsonl"


def _parse_row(line: str) -> tuple[str, str, float] | None:
    """Parse one JSONL line.

    Returns (action, decision, age_seconds) or None if malformed/non-digest.
    Never raises — JSONL is user-editable; one typo must not brick the gate.
    """
    line = line.strip()
    if not line:
        return None
    try:
        row = json.loads(line)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(row, dict):
        return None
    subtag = row.get("subtag", "")
    if not isinstance(subtag, str) or not subtag.startswith(_SUBTAG_PREFIX):
        return None
    action = subtag[len(_SUBTAG_PREFIX):]
    if action not in ACTIONS:
        return None
    decision = row.get("decision", "")
    if decision not in ("continue", "defer", "kill"):
        return None
    ts_str = row.get("ts", "")
    if not isinstance(ts_str, str) or not ts_str:
        return None
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - ts).total_seconds()
    except (ValueError, TypeError):
        return None
    return action, decision, age_seconds


def _load_samples(*, force: bool = False) -> dict[str, list[tuple[str, float]]]:
    """Parse JSONL and group by action with 60s + mtime cache.

    Returns {action: [(decision, age_seconds), ...]} for all 4 ACTIONS.
    Missing JSONL → all-empty dict (GATE-03 default LOCKED).
    """
    path = _resolve_jsonl_path()
    now_s = time.monotonic()
    try:
        mtime = path.stat().st_mtime if path.exists() else 0.0
    except OSError:
        mtime = 0.0

    cached = _cache["samples_by_action"]
    fresh = (
        not force
        and cached is not None
        and (now_s - _cache["loaded_at"]) < _CACHE_TTL_S
        and mtime == _cache["mtime"]
    )
    if fresh:
        # IN-01: return a shallow copy so callers cannot mutate cache state.
        # Lists are also copied (per-action sample buffers); the inner tuples
        # are immutable so a deeper copy is unnecessary. Cost: ~hundreds of
        # tuple references — negligible against the JSONL parse we just
        # avoided.
        return {a: list(samples) for a, samples in cached.items()}

    by_action: dict[str, list[tuple[str, float]]] = {a: [] for a in ACTIONS}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                # WR-03: bound per-line memory at _MAX_LINE_BYTES. ``readline``
                # with a size limit returns at most that many chars without a
                # trailing newline; if the line is longer, the next readline
                # continues mid-line. We detect oversized lines by the absence
                # of the trailing ``\n`` and drain the rest in chunks before
                # resuming the parse loop.
                while True:
                    line = fh.readline(_MAX_LINE_BYTES + 1)
                    if not line:
                        break  # EOF
                    if len(line) > _MAX_LINE_BYTES and not line.endswith("\n"):
                        # Oversized line — skip the rest of it, log once, move on.
                        # Drain until newline or EOF (in bounded chunks to keep
                        # memory cap intact even on a multi-GB malformed line).
                        while True:
                            chunk = fh.readline(_MAX_LINE_BYTES + 1)
                            if not chunk or chunk.endswith("\n"):
                                break
                        # Defensive log: malformed JSONL is a peer-process bug,
                        # not a worker bug — keep going (GATE-03 fail-closed).
                        # No log_reaction_event import here to avoid circular
                        # dep with logging_jsonl; second_brain owns the file.
                        continue
                    parsed = _parse_row(line)
                    if parsed is None:
                        continue
                    action, decision, age_seconds = parsed
                    by_action[action].append((decision, age_seconds))
        except OSError:
            pass  # unreadable JSONL → empty samples → all LOCKED (GATE-03)

    _cache["samples_by_action"] = by_action
    _cache["loaded_at"] = now_s
    _cache["mtime"] = mtime
    # IN-02: invalidate the gate_status memo whenever the underlying samples
    # are re-parsed. Re-evaluation happens lazily on the next get_gate_status.
    _cache["gate_status"] = None
    return {a: list(samples) for a, samples in by_action.items()}


def wilson_threshold(n: float, eval_window: int | None = None) -> float:
    """Phase 82 — bounded-temperature anneal for the finish-action precision gate.

    Returns the Wilson lower-bound threshold a topic must clear for the
    ``finish`` action to unlock, as a function of the effective sample count
    ``n``. Early-life topics (low n) clear easily so exploratory drafts can
    promote; mature topics (n≥20) face the strict 0.8 ceiling.

    Curve (CONTEXT specifics, 82-01-PLAN.md):

      n   | threshold
      ----|----------
        0 |  0.500
        5 |  0.575
       10 |  0.650
       15 |  0.725
       20 |  0.800
       50 |  0.800

    Formula: ``conf(n) = 0.5 + 0.3 * min(n/20, 1.0)``

    Env override (D-03, MEMORY: module-level env binding gotcha):
      ``WILSON_ANNEAL_DISABLED=1`` → flat 0.7 for all n. Env is read INSIDE
      this function on every call, never at module import. Tests rely on
      monkeypatching env *after* import.

    Used only by the ``finish`` branch of ``_evaluate_one`` (D-02). The other
    three actions (brainstorm/docs/link) keep flat ``THRESHOLDS`` lookups,
    and the LOO eval at ``scripts/eval_routing_loo.py`` keeps flat conf=0.7.
    """
    # D-03: env read INSIDE function body (module-level binding gotcha).
    if os.environ.get("WILSON_ANNEAL_DISABLED") == "1":
        return 0.7
    base = 0.5 + 0.3 * min(n / 20.0, 1.0)
    # Phase 93: narrow-window noise floor. When the gate re-runs on only a
    # tail slice of a long transcript (CONTEXT: last 10K tokens), the smaller
    # eval window has higher variance — lift the floor to 0.6 so low-n
    # tail-only re-checks cannot dip below the anneal-curve confidence we
    # require for full-text classify.
    if eval_window is not None:
        return max(base, 0.6)
    return base


def _evaluate_one(
    samples: list[tuple[str, float]],
    action: str,
    n_bonus: float = 0.0,
) -> tuple[bool, float, float, float]:
    """Evaluate gate for one action with optional n-denominator bonus.

    Returns (unlocked, wilson, n_effective, threshold). UNLOCKED requires:
      1. n_effective >= MIN_N_FLOOR (small-N defence, 21-01 forward note)
      2. wilson_lower_bound > threshold, where threshold is:
         - ``wilson_threshold(n_eff)`` for action == "finish" (Phase 82 anneal)
         - ``THRESHOLDS[action]`` for brainstorm/docs/link (D-02: unchanged)

    IN-02 (Phase 82 review): threshold is returned as 4th element so
    ``get_gate_status`` reads a single source of truth instead of recomputing
    ``wilson_threshold(n_eff)`` — prevents future drift if the curve changes.
    """
    s, n = decay_samples(samples, half_life_days=_HALF_LIFE_DAYS)
    n_eff = n + n_bonus
    # Phase 82: anneal the finish-action threshold based on n_eff; leave the
    # other three actions on the flat THRESHOLDS dict per D-02.
    threshold = wilson_threshold(n_eff) if action == "finish" else THRESHOLDS[action]
    if n_eff <= 0:
        return False, 0.0, 0.0, threshold
    wilson = wilson_lower_bound(s, n_eff)
    # Float-tolerance on the floor: 6 fresh samples may sum to 5.99999... due
    # to recency_weight at sub-second age. Round-to-2dp matches the public n
    # we expose in get_gate_status.
    unlocked = (round(n_eff, 2) >= MIN_N_FLOOR) and (wilson > threshold)
    return unlocked, wilson, n_eff, threshold


def get_gate_status() -> dict[str, dict]:
    """Return {action: {locked, wilson, n, threshold, min_n}} for all 4 ACTIONS.

    Curiosity-unlock pass: if any action unlocked at first evaluation, re-evaluate
    every still-LOCKED action with +1.0 added to its decayed n (one-shot).
    The +1 sample is uncertainty credit (denominator only, not a guaranteed
    success) — interpretation per CONTEXT "one extra sample of curiosity".

    Always returns dict with exactly len(ACTIONS) keys; UI never sees missing keys.

    IN-02: result memoized alongside the JSONL cache. When the underlying
    samples are still fresh AND a prior gate_status exists, this short-circuits
    the 4-action evaluation loop + curiosity unlock pass — important when
    every draft write triggers ``apply_preliminary_label`` which calls
    ``is_unlocked`` (which calls back here).
    """
    samples_by_action = _load_samples()
    # IN-02: read the memo AFTER _load_samples, which clears it on a fresh
    # parse. A non-None value here means: samples are still fresh AND we
    # already evaluated this exact JSONL state.
    cached_status = _cache.get("gate_status")
    if cached_status is not None:
        # Defensive copy so callers can't mutate the cache.
        return {a: dict(v) for a, v in cached_status.items()}
    first_pass: dict[str, tuple[bool, float, float, float]] = {
        a: _evaluate_one(samples_by_action[a], a) for a in ACTIONS
    }
    any_unlocked = any(unlocked for unlocked, _, _, _ in first_pass.values())
    result: dict[str, dict] = {}
    for action in ACTIONS:
        unlocked, wilson, n_eff, eff_threshold = first_pass[action]
        if not unlocked and any_unlocked:
            # Curiosity unlock — +1 to denominator, recompute Wilson.
            unlocked, wilson, n_eff, eff_threshold = _evaluate_one(
                samples_by_action[action], action, n_bonus=1.0
            )
        # IN-02: eff_threshold returned from _evaluate_one — single source of
        # truth (annealed for finish, flat THRESHOLDS for the other three).
        result[action] = {
            "locked": not unlocked,
            "wilson": round(wilson, 4),
            "n": round(n_eff, 2),
            "threshold": eff_threshold,
            "min_n": MIN_N_FLOOR,
        }
    # IN-02: store memo. Subsequent calls within the JSONL cache TTL skip
    # the per-action loop + curiosity unlock recomputation.
    _cache["gate_status"] = result
    return {a: dict(v) for a, v in result.items()}


def is_unlocked(action: str) -> bool:
    """Public gate. Returns False (LOCKED) for unknown actions or missing JSONL."""
    if action not in ACTIONS:
        return False
    return get_gate_status()[action]["locked"] is False
