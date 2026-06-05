"""
Phase 52 / DIVERGE-03 — Shadow router.

Strategy A Claude sampling gate + hourly bucket throttle + per-run telemetry
counters consumed by run_pipeline.py OK-log emission (NFR-05).

Env vars (read at call time — NEVER module-level per reference_module_level_env_binding.md):
- SCORER_SAMPLING_RATE: float, default 0.05 (5%). Bernoulli draw threshold.
- CLAUDE_HOURLY_CAP: int, default 50. Max Claude calls per rolling 1-hour window.

Counters: process-level globals, mirror src/llm/routing.py _truncation_count
pattern. Reset at run start by run_pipeline.main(), read at OK-log emit site.
"""
from __future__ import annotations

import logging
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Asia/Karaganda fixed offset (KZ no DST; not in Windows tzdata —
# see reference_asia_karaganda_tzdata_missing.md). Retained as a public
# constant for downstream consumers — note that the bucket cutoff now
# computes in UTC (see CR-01 fix in `_claude_calls_last_hour`).
KARAGANDA_TZ = timezone(timedelta(hours=5))

# IN-01 fix (2026-05-18): module-level SystemRandom instance avoids the
# per-call allocation in `should_claude_sample`. `secrets.SystemRandom` is
# documented immune to `random.seed()`, so the test-immunity contract
# survives module-level binding (`tests/llm/test_shadow_router.py::
# test_secrets_systemrandom_immune_to_random_seed` still holds).
_RNG = secrets.SystemRandom()

# WR-04 fix (2026-05-18): dedupe WARNING log on malformed env values so
# a misconfigured `.env` doesn't produce one WARNING per post.
_bad_env_warned: set[str] = set()


class ClaudeRateLimitError(Exception):
    """Raised when Claude CLI returns a rate-limit signal in stderr OR stdout.

    Distinct from RuntimeError so the curator shadow fan-out (plan 03) can
    catch it specifically and increment claude_calls_skipped instead of the
    generic shadow_failures path.
    """


# ---------- Bucket query ----------

def _claude_calls_last_hour(db_path: str | Path) -> int:
    """COUNT rows in scorer_divergence with sample_source='strategy_a_truth_sample'
    AND scored_at >= now-1h (UTC).

    CR-01 fix (2026-05-18): cutoff is computed in UTC to match `scored_at`
    written by curator's `_commit_result` / `_shadow_fan_out`
    (both use `datetime.datetime.now(datetime.UTC).isoformat()`). SQLite
    compares ISO-8601 strings lexicographically, and lex ordering only matches
    chronological ordering when the offset suffix matches. Previously this
    helper computed cutoff in Karaganda local (+05:00) while curator wrote
    UTC (+00:00); the suffix mismatch silently disabled the CLAUDE_HOURLY_CAP
    in production (verified by tests using the same convention as the
    helper, never crossing the production write boundary).

    Fail-soft: any sqlite3 error returns 0 (treat as empty bucket — primary
    curate is the priority; over-sampling on read failure is acceptable, the
    next bucket check will catch up).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    # WR-05 fix (2026-05-18): `sqlite3.connect(path)` *creates* the file when
    # missing. If schtasks/cron/CI fires the bucket query from a working dir
    # without curator.db (partial chdir, system-account scheduled task), the
    # helper silently created an empty `curator.db` somewhere unintended and
    # subsequent migrate_db calls could target the wrong file. Treat missing
    # DB as empty bucket and use `mode=ro` URI so a wrong path can never
    # materialise a phantom file.
    p = Path(db_path)
    if not p.exists():
        return 0
    try:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        try:
            cur = conn.execute(
                "SELECT COUNT(*) FROM scorer_divergence "
                "WHERE sample_source = ? AND scored_at >= ?",
                ("strategy_a_truth_sample", cutoff),
            )
            return int(cur.fetchone()[0])
        finally:
            conn.close()
    except Exception:
        return 0


# ---------- Sampling decision ----------

def should_claude_sample(post_id: int, db_path: str | Path) -> bool:
    """Per-post Bernoulli draw with hourly bucket cap.

    Both env vars are read on EVERY call (Pattern D — no module-level binding).
    RNG is secrets.SystemRandom (non-seedable per STACK.md).

    Returns True iff:
      1. Bucket has fewer than CLAUDE_HOURLY_CAP rows in the last hour, AND
      2. random() < SCORER_SAMPLING_RATE.

    WR-04 fix (2026-05-18): malformed env values (e.g. `0.05%`, `50.5`) no
    longer propagate ValueError into curator's outer `except Exception` —
    fall back to defaults with a one-line WARNING (same `_shadow_3b_warned`
    semantics: log once per unique bad value per run). Rate is clamped to
    [0.0, 1.0] and cap to >=0 to keep the sampler well-behaved on
    nonsensical-but-parseable values like `-0.5` or `200`.
    """
    raw_rate = os.environ.get("SCORER_SAMPLING_RATE", "0.05")
    try:
        rate = float(raw_rate)
    except (TypeError, ValueError):
        if raw_rate not in _bad_env_warned:
            _bad_env_warned.add(raw_rate)
            logger.warning(
                "Invalid SCORER_SAMPLING_RATE=%r; falling back to 0.05", raw_rate
            )
        rate = 0.05
    if rate < 0.0:
        rate = 0.0
    elif rate > 1.0:
        rate = 1.0

    raw_cap = os.environ.get("CLAUDE_HOURLY_CAP", "50")
    try:
        cap = int(raw_cap)
    except (TypeError, ValueError):
        if raw_cap not in _bad_env_warned:
            _bad_env_warned.add(raw_cap)
            logger.warning(
                "Invalid CLAUDE_HOURLY_CAP=%r; falling back to 50", raw_cap
            )
        cap = 50
    if cap < 0:
        cap = 0

    if _claude_calls_last_hour(db_path) >= cap:
        return False
    return _RNG.random() < rate


# ---------- Telemetry counters (process-level; reset by run_pipeline) ----------

_divergence_logged: int = 0
_claude_sampled: int = 0
_claude_calls_skipped: int = 0


def get_divergence_logged() -> int:
    return _divergence_logged


def reset_divergence_logged() -> None:
    global _divergence_logged
    _divergence_logged = 0


def _incr_divergence_logged() -> None:
    global _divergence_logged
    _divergence_logged += 1


def get_claude_sampled() -> int:
    return _claude_sampled


def reset_claude_sampled() -> None:
    global _claude_sampled
    _claude_sampled = 0


def _incr_claude_sampled() -> None:
    global _claude_sampled
    _claude_sampled += 1


def get_claude_calls_skipped() -> int:
    return _claude_calls_skipped


def reset_claude_calls_skipped() -> None:
    global _claude_calls_skipped
    _claude_calls_skipped = 0


def _incr_claude_calls_skipped() -> None:
    global _claude_calls_skipped
    _claude_calls_skipped += 1


# WR-03 (2026-05-18): track shadow 3b model-missing / load failures so
# operators can distinguish "no curated posts" (divergence_logged=0) from
# "shadow model missing" (divergence_logged=0, shadow_3b_failures>0).
# Logs a single WARNING on the first failure per run with the model name and
# a pull hint; subsequent failures only increment the counter.
_shadow_3b_failures: int = 0
_shadow_3b_warned: set[str] = set()


def get_shadow_3b_failures() -> int:
    return _shadow_3b_failures


def reset_shadow_3b_failures() -> None:
    global _shadow_3b_failures
    _shadow_3b_failures = 0
    _shadow_3b_warned.clear()


def _note_shadow_3b_failure(model: str) -> None:
    global _shadow_3b_failures
    _shadow_3b_failures += 1
    if model not in _shadow_3b_warned:
        _shadow_3b_warned.add(model)
        logger.warning(
            "Phase 52 shadow 3b model unavailable: %s. "
            "Hint: `ollama pull %s` (from PowerShell). "
            "Subsequent failures will be counted silently in this run.",
            model, model,
        )
