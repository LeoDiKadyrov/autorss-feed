"""Local Conv+Transformer scorer fallback (Phase 98-02, LOCAL-SCORER-01).

Per D-08 (env gate default off) and D-09 (fallback only fires when both
primary backends — ollama AND claude — are unavailable). The operator's
CURATOR_BACKEND override is a separate, higher-priority concern handled in
src/llm/routing.py; this module only answers "is the primary chain dead AND
do we have a local weights file ready to take over?".

Public API:
  - is_local_fallback_enabled() -> bool
  - score_with_local(text) -> float | None  (returns score in [0, 100], or None)
  - should_use_local_fallback(primary_backend) -> bool
  - _reset_cache()  (test hook)

All paths are fail-soft: any internal exception is swallowed and the function
returns None / False so the caller falls back to the normal primary chain.
"""
from __future__ import annotations

import logging
import math
import os
import time as _time
from typing import Any

logger = logging.getLogger(__name__)

# Cached LocalConvScorer instance (loaded once via load_latest()). None = not
# yet attempted OR last attempt found no weights. We retry on every call when
# _model_cache is None — the cost is one Path.glob() so this is fine.
_model_cache: Any = None

# (mono_ts, ollama_alive, claude_alive). None = no probe yet performed.
_probe_cache: tuple[float, bool, bool] | None = None
_PROBE_TTL_S = 30.0

# Warn-once flag for missing weights — prevents log spam every call.
_warned_no_weights = False


def is_local_fallback_enabled() -> bool:
    """True only when LOCAL_SCORER_FALLBACK=on (case-insensitive). Default off."""
    return (os.environ.get("LOCAL_SCORER_FALLBACK", "off") or "").strip().lower() == "on"


def _ollama_alive() -> bool:
    """Cheap GET http://localhost:11434/api/tags with 1s timeout. Fail-soft."""
    try:
        import httpx  # type: ignore[import]
        r = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


def _claude_alive() -> bool:
    """CR-02 fix (Phase 98): real liveness probe — not just API-key presence.

    Per D-09 fallback fires only when BOTH primary backends are DOWN. Pre-fix
    this returned True iff ANTHROPIC_API_KEY was set, which is always true in
    production → fallback never fired on Anthropic outages.

    Strategy: check `cost_snapshots` for a recent successful Claude call
    (within CLAUDE_ALIVE_WINDOW_MIN, default 30 min). When no record exists,
    fall back to a cheap subprocess probe of `claude --version` with a tight
    timeout. Both signals fail-soft → False (treat backend as dead).
    """
    # Strategy 1: recent Claude row in cost_snapshots (cheap, no USD spend).
    try:
        import sqlite3
        from datetime import datetime, timezone, timedelta
        db_path = os.environ.get("CURATOR_DB_PATH", "curator.db")
        if os.path.exists(db_path):
            window_min = int(os.environ.get("CLAUDE_ALIVE_WINDOW_MIN", "30"))
            cutoff = (
                datetime.now(timezone.utc) - timedelta(minutes=window_min)
            ).isoformat()
            conn = sqlite3.connect(db_path)
            try:
                cur = conn.execute(
                    "SELECT 1 FROM cost_snapshots "
                    "WHERE backend LIKE 'claude%' "
                    "  AND datetime(replace(created_at, 'T', ' ')) >= "
                    "      datetime(replace(?, 'T', ' ')) "
                    "LIMIT 1",
                    (cutoff,),
                )
                if cur.fetchone() is not None:
                    return True
            finally:
                conn.close()
    except Exception:
        pass

    # Strategy 2: `claude --version` subprocess probe (fast, no API call).
    try:
        import subprocess
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, OSError, Exception):
        pass

    return False


def _probe_both() -> tuple[bool, bool]:
    """Cached (~30s) probe of (ollama_alive, claude_alive)."""
    global _probe_cache
    now = _time.monotonic()
    if _probe_cache is not None and (now - _probe_cache[0]) < _PROBE_TTL_S:
        return _probe_cache[1], _probe_cache[2]
    oa = _ollama_alive()
    ca = _claude_alive()
    _probe_cache = (now, oa, ca)
    return oa, ca


def _get_model() -> Any:
    """Lazy-load LocalConvScorer from data/local_scorer_v*.pt. None if absent.

    Cached across calls (single load per process). Fail-soft on any import or
    load error — torch may not be installed in some environments and that
    should silently degrade to None, not raise.
    """
    global _model_cache, _warned_no_weights
    if _model_cache is not None:
        return _model_cache
    try:
        from src.scorers.local_conv import load_latest
        _model_cache = load_latest()
    except Exception as e:
        logger.warning("local_scorer_fallback: load_latest raised %s", e)
        _model_cache = None
    if _model_cache is None and not _warned_no_weights:
        logger.warning(
            "local_scorer_fallback: no weights file found in data/ — "
            "fallback will be unavailable until a model is trained"
        )
        _warned_no_weights = True
    elif _model_cache is not None:
        # WR-03 fix (Phase 98): log the successful first load so operators can
        # confirm the fallback model picked up freshly-trained weights mid-
        # process. Pre-fix the warn-once was asymmetric — failures silent,
        # successes silent.
        logger.info("local_scorer_fallback: loaded local scorer weights")
    return _model_cache


def _to_percent(raw: float) -> float:
    """Clamp raw predict() output to [0, 100].

    CR-01 fix (Phase 98): the trainer fits MSE against curation_logs.relevance_score
    in [0, 100], so the head's outputs are ALREADY on the 0-100 scale. Pre-fix
    we applied sigmoid * 100 which saturated to ~100 for any raw>>0 (even
    raw=5 → sigmoid≈0.993 → 99.3) — every post passed CURATOR_THRESHOLD=75
    once fallback fired. Clamp is the right shape for the training target.

    NaN/Inf safety: NaN → 0.0; +Inf → 100.0; -Inf → 0.0.
    """
    try:
        if math.isnan(raw):
            return 0.0
        if math.isinf(raw):
            return 100.0 if raw > 0 else 0.0
        return max(0.0, min(100.0, float(raw)))
    except (TypeError, ValueError):
        return 0.0


def score_with_local(text: str) -> float | None:
    """Score `text` via the local scorer. Returns float in [0, 100] or None.

    Returns None when:
      - No weights file exists in data/
      - torch / local_conv import fails
      - predict() raises for any reason

    Never raises. Caller falls back to primary backend on None.
    """
    try:
        model = _get_model()
        if model is None:
            return None
        from src.scorers.local_conv import predict
        raw = predict(text, model)
        return _to_percent(float(raw))
    except Exception as e:
        logger.warning("local_scorer_fallback: score_with_local raised %s", e)
        return None


def should_use_local_fallback(primary_backend: str) -> bool:
    """Should the caller route to the local scorer instead of `primary_backend`?

    True only when ALL of:
      1. LOCAL_SCORER_FALLBACK=on
      2. Both ollama and claude probe dead
      3. Local weights are loadable

    Per D-09, fallback never fires when at least one primary is alive — the
    local scorer is a degradation path, not a co-equal backend. The
    `primary_backend` argument is accepted for API symmetry with future
    backend-specific logic (e.g. only-fall-back-when-claude-was-chosen) but is
    currently unused.
    """
    del primary_backend  # reserved for future selective-fallback logic
    if not is_local_fallback_enabled():
        return False
    oa, ca = _probe_both()
    if oa or ca:
        return False
    return _get_model() is not None


def _reset_cache() -> None:
    """Test-only: clear model + probe caches + warn-once flag."""
    global _model_cache, _probe_cache, _warned_no_weights
    _model_cache = None
    _probe_cache = None
    _warned_no_weights = False
