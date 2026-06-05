"""
Phase 61 / COST-CAP-01 + NFR-V23-02 — per-category daily USD cap reader.

When today's claude / claude-batch spend on a category crosses
`daily_usd_caps[category]` in `config/curator_routing.yaml`, the curator
backend router silently falls back to ``ollama`` until midnight GMT+5.

USD is estimated per call from `_USD_PER_CALL_FALLBACK` constants, overridable
via yaml `usd_per_call:` block (MR-02). No per-call usd column exists in the
schema; `cost_snapshots` is a 7-day rolling aggregate that does not bucket
per category. We count rows in `curation_logs` with
backend ∈ {claude, claude-batch} since today's GMT+5 midnight (in UTC),
joined to `sources` for category filtering, then multiply by the per-call
estimate.

Design:
  - In-process TTL cache (60s) — at most one bounded SELECT per category per
    minute. Cache eviction on day-boundary (GMT+5) is automatic via the
    cached UTC-midnight timestamp comparison.
  - Fail-soft: any DB error → `is_cap_exceeded` returns False (no fallback),
    logs WARNING; never propagates an exception to routing.
  - Counter trio mirrors `src/curator/snippet_quality.py` / `giveaway_filter.py`
    (process-global, thread-unsafe by design — curator runs single-threaded).
  - TZ: `timezone(timedelta(hours=5))` per CLAUDE.md (Asia/Karaganda is
    missing from Windows tzdata so ZoneInfo would crash on the schtasks host).

OK-log accretion: `cost_cap_falls_back=N` slot sits BETWEEN
`giveaway_rejected=` and `profile_fallback=` per CONTEXT 61 position rule.
Lockstep: `_format_ok_line` + `_LOG_OK_RE` + 4 test files updated.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# --- Constants -----------------------------------------------------------

# GMT+5 fixed offset — Asia/Karaganda not in Windows tzdata (CLAUDE.md).
GMT5 = timezone(timedelta(hours=5))

# Per-call USD estimate FALLBACK. Calibrated from observed Claude headless cost
# (~$0.015 per single-post scorer call, sonnet at ~5K in / 500 out) and the
# batch path (20 posts/call but per-CALL billing still applies).
# MR-02 (Phase 61): operator overrides this dict via yaml
# `usd_per_call:` in config/curator_routing.yaml — when Anthropic shifts pricing
# the cap reads correct USD without a code deploy. This hardcoded dict is the
# fallback when yaml is missing the block or the backend entry.
_USD_PER_CALL_FALLBACK = {
    "claude":       0.015,
    "claude-batch": 0.012,
    "ollama":       0.0,
}


def _get_usd_per_call(backend: str) -> float:
    """MR-02: return per-call USD for `backend`, preferring yaml override.

    Priority:
      1. yaml `usd_per_call:` block (operator-editable on pricing change)
      2. `_USD_PER_CALL_FALLBACK` (calibrated defaults)
      3. 0.0 (unknown backend — safe default; cap can't trigger)

    Bad yaml value (non-numeric, bool, negative) falls through to fallback.
    """
    try:
        from src.llm import routing
        cfg = routing._load() or {}
    except Exception:
        return _USD_PER_CALL_FALLBACK.get(backend, 0.0)
    yaml_block = cfg.get("usd_per_call") or {}
    if isinstance(yaml_block, dict):
        raw = yaml_block.get(backend)
        # Reject bool (int subclass) and non-numeric / negative entries.
        if not isinstance(raw, bool) and isinstance(raw, (int, float)) and raw >= 0:
            return float(raw)
    return _USD_PER_CALL_FALLBACK.get(backend, 0.0)

# In-process TTL — at most one DB query per category per minute.
_CACHE_TTL_SECONDS = 60

# Module-level state. Mirrors snippet_quality/giveaway_filter pattern.
# Cache value: (usd_today, fetched_at_utc, day_midnight_utc).
_cache: dict[str, tuple[float, datetime, datetime]] = {}
_cost_cap_falls_back: int = 0

# HR-02: once-per-process latch for the singular-typo warning. Without this,
# every is_cap_exceeded() call would re-warn (60s cache misses + every miss
# re-reads the yaml). Reset by _reset_cache() so tests can re-trigger.
_singular_typo_warned: bool = False

# MR-03: per-category latch for bool-value typo warning (e.g. `ai: true`).
# Latches per category so each typo'd cap entry warns once. Reset by
# _reset_cache() so tests can re-trigger.
_bool_warned: dict[str, bool] = {}


# --- Helpers -------------------------------------------------------------


def _now_utc() -> datetime:
    """Indirection seam — monkeypatchable in tests for day-boundary roll."""
    return datetime.now(timezone.utc)


def _midnight_gmt5_utc(now: datetime | None = None) -> str:
    """Return ISO of today's GMT+5 midnight, expressed in UTC.

    Used as the cutoff for the curation_logs SELECT and as the cache-eviction
    key (day-boundary detection compares this against the cached value).
    """
    if now is None:
        now = _now_utc().astimezone(GMT5)
    else:
        # Accept either tz-aware or naive (treat naive as GMT+5).
        if now.tzinfo is None:
            now = now.replace(tzinfo=GMT5)
        now = now.astimezone(GMT5)
    midnight_local = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight_local.astimezone(timezone.utc).isoformat()


def _db_path() -> str:
    """Read DB_PATH inside the function — module-level binding breaks tests.

    Per `reference_module_level_env_binding.md` — `DB_PATH = os.environ.get(...)`
    at module top binds once and `patch.dict(os.environ, ...)` cannot override.
    """
    return os.environ.get("DB_PATH", "curator.db")


def _query_usd_today(category: str, cutoff: str | None = None) -> float:
    """Open a bounded sqlite3 query; return USD-estimate for today's claude+batch
    calls on this category. Raises on DB error (caller fail-softs).

    `cutoff` is the ISO-UTC string for today's GMT+5 midnight. Caller passes it
    in to avoid re-deriving via `_now_utc()` (which lets test seams mock the
    clock with a single time-tick budget per public call).

    Single SELECT counts rows per backend; we multiply by per-call USD and sum.
    Filter: ``scored_at >= today_gmt5_midnight_utc``. Sub-50ms with the
    `idx_curation_logs_scored_at_backend` composite index added in MR-01
    (src/database/client.py::migrate_db). Pre-MR-01 review caught the original
    docstring claim of a "default scored_at index from Phase 30" — no such
    index actually existed; query was a full-scan.
    """
    if cutoff is None:
        cutoff = _midnight_gmt5_utc()
    con = sqlite3.connect(_db_path())
    try:
        rows = con.execute(
            "SELECT cl.backend, COUNT(*) FROM curation_logs cl "
            "JOIN raw_posts rp ON rp.id = cl.post_id "
            "JOIN sources s ON s.id = rp.source_id "
            "WHERE cl.backend IN (?, ?) "
            "  AND s.category = ? "
            "  AND cl.scored_at >= ? "
            "GROUP BY cl.backend",
            ("claude", "claude-batch", category, cutoff),
        ).fetchall()
    finally:
        con.close()
    usd = 0.0
    for backend, count in rows:
        per_call = _get_usd_per_call(backend)  # MR-02: yaml-overridable
        usd += per_call * count
    return usd


def _get_cap(category: str) -> float | None:
    """Read daily_usd_caps[category] from routing._load(). Returns None if unset.

    HR-02: if the plural key is absent but a similar typo'd key
    (singular `daily_usd_cap`) is present, log WARNING once-per-process so the
    operator notices the silently-disabled cap.
    """
    global _singular_typo_warned
    try:
        # Lazy import — avoid module-import cycle at top level.
        from src.llm import routing
        cfg = routing._load() or {}
    except Exception:
        return None

    # HR-02: detect singular-key typo. Warn once per process; non-fatal.
    if "daily_usd_caps" not in cfg and "daily_usd_cap" in cfg and not _singular_typo_warned:
        logger.warning(
            "[cost_cap] config typo detected: 'daily_usd_cap' (singular) found "
            "but the expected key is 'daily_usd_caps' (plural). Cap is DISABLED "
            "for all categories. Fix config/curator_routing.yaml to enable."
        )
        _singular_typo_warned = True

    caps = cfg.get("daily_usd_caps") or {}
    if not isinstance(caps, dict):
        return None
    raw = caps.get(category)
    # MR-03: Python's True is an int subclass — `isinstance(True, (int, float))`
    # passes and `True > 0` evaluates True, so `daily_usd_caps: {ai: true}`
    # would silently yield cap=1.0. Reject bool first; warn once per category.
    if isinstance(raw, bool):
        if not _bool_warned.get(category):
            logger.warning(
                "[cost_cap] daily_usd_caps[%s] = %r (bool) — treated as NO cap. "
                "Fix config/curator_routing.yaml to a numeric USD value.",
                category, raw,
            )
            _bool_warned[category] = True
        return None
    if isinstance(raw, (int, float)) and raw > 0:
        return float(raw)
    return None


# --- Main entry ----------------------------------------------------------


def is_cap_exceeded(category: str) -> bool:
    """Return True iff today's claude+batch spend on `category` >= cap.

    Fail-soft: any DB error → False (no fallback), WARNING logged. Cache
    eviction on day-boundary (GMT+5) is automatic. Counter is incremented
    on every True return (telemetry for OK-log accretion).
    """
    cap = _get_cap(category)
    if cap is None:
        return False  # No cap configured → no DB hit.

    now = _now_utc()
    today_midnight = _midnight_gmt5_utc(now)

    cached = _cache.get(category)
    if cached is not None:
        usd_today, fetched_at, cached_day_midnight = cached
        # Evict if day rolled OR TTL expired.
        if cached_day_midnight == today_midnight and (now - fetched_at).total_seconds() < _CACHE_TTL_SECONDS:
            # Cache hit — reuse cached value.
            if usd_today >= cap:
                _incr_cost_cap_falls_back()
                return True
            return False

    # Cache miss — query DB (fail-soft).
    try:
        usd_today = _query_usd_today(category, today_midnight)
    except Exception as exc:
        logger.warning(
            "[cost_cap] DB error category=%s err=%s — fail-soft, no cap fallback",
            category, exc,
        )
        return False

    _cache[category] = (usd_today, now, today_midnight)
    if usd_today >= cap:
        _incr_cost_cap_falls_back()
        return True
    return False


# --- Counter trio (process-global; mirrors snippet_quality / giveaway_filter) ---


def get_cost_cap_falls_back() -> int:
    return _cost_cap_falls_back


def reset_cost_cap_falls_back() -> None:
    global _cost_cap_falls_back
    _cost_cap_falls_back = 0


def _incr_cost_cap_falls_back() -> None:
    global _cost_cap_falls_back
    _cost_cap_falls_back += 1


# --- Test helper ---------------------------------------------------------


def _reset_cache() -> None:
    """Test-only: clear TTL cache AND zero the counter AND clear typo latches."""
    global _cache, _singular_typo_warned, _bool_warned
    _cache = {}
    _singular_typo_warned = False
    _bool_warned = {}
    reset_cost_cap_falls_back()
