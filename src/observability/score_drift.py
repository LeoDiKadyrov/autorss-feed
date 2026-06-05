"""
Score drift detector — rolling 7d mean/std on curation_logs.relevance_score
per (backend, source_category). Flags when |today_mean - prior_7d_mean| > 1σ.

Mirrors cost_snapshots.anomaly_flag pattern but for scoring quality.

Schema (auto-created on first call):
    CREATE TABLE score_drift_snapshots (
        id INTEGER PRIMARY KEY,
        scored_at TEXT NOT NULL,           -- snapshot timestamp (Asia/Karaganda ISO)
        backend TEXT NOT NULL,             -- ollama / claude / claude-batch / null
        source_category TEXT NOT NULL,     -- canonical category (ai/crypto/...)
        mu_today REAL,                     -- mean of last 24h (or NULL if N<5)
        n_today INTEGER NOT NULL,          -- sample count
        mu_prior REAL,                     -- mean of prior 7d
        sigma_prior REAL,                  -- stddev of prior 7d
        n_prior INTEGER NOT NULL,          -- prior sample count
        drift_magnitude REAL,              -- |mu_today - mu_prior| / sigma_prior (NULL if math undefined)
        flagged INTEGER NOT NULL DEFAULT 0 -- 1 if drift_magnitude > 1.0
    )

Idempotent — safe to re-run; INSERT OR IGNORE on (scored_at, backend, source_category).
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

# Asia/Karaganda fixed offset (KZ has no DST; not in Windows tzdata).
KARAGANDA_TZ = timezone(timedelta(hours=5))

# Minimum sample count for a meaningful mean.
MIN_N_TODAY = 5
MIN_N_PRIOR = 5

# Drift threshold — flag if absolute deviation > 1 sigma.
DRIFT_SIGMA_THRESHOLD = 1.0

# Recent flag window for OK log surfacing (24h).
RECENT_FLAGS_WINDOW_HOURS = 24


def now_karaganda() -> datetime:
    return datetime.now(KARAGANDA_TZ)


@dataclass(frozen=True)
class DriftRecord:
    backend: str
    source_category: str
    mu_today: float | None
    n_today: int
    mu_prior: float | None
    sigma_prior: float | None
    n_prior: int
    drift_magnitude: float | None
    flagged: bool


def ensure_table(conn: sqlite3.Connection) -> None:
    """Create score_drift_snapshots table if missing. Idempotent."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS score_drift_snapshots (
            id INTEGER PRIMARY KEY,
            scored_at TEXT NOT NULL,
            backend TEXT NOT NULL,
            source_category TEXT NOT NULL,
            mu_today REAL,
            n_today INTEGER NOT NULL,
            mu_prior REAL,
            sigma_prior REAL,
            n_prior INTEGER NOT NULL,
            drift_magnitude REAL,
            flagged INTEGER NOT NULL DEFAULT 0,
            UNIQUE(scored_at, backend, source_category)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_score_drift_scored_at "
        "ON score_drift_snapshots(scored_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_score_drift_flagged "
        "ON score_drift_snapshots(flagged) WHERE flagged = 1"
    )
    conn.commit()


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _stddev(xs: list[float]) -> float | None:
    """Sample stddev (N-1). Returns None for N<2."""
    if len(xs) < 2:
        return None
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _fetch_scores_window(
    conn: sqlite3.Connection,
    window_start: datetime,
    window_end: datetime,
) -> list[tuple[str | None, str | None, int]]:
    """
    Return [(backend, source_category, relevance_score)] for curation_logs scored within window.
    backend and source_category may be NULL — caller handles bucketing.
    """
    cur = conn.execute(
        """
        SELECT
            cl.backend AS backend,
            s.category AS source_category,
            cl.relevance_score AS relevance_score
        FROM curation_logs cl
        LEFT JOIN raw_posts rp ON rp.id = cl.post_id
        LEFT JOIN sources s    ON s.id = rp.source_id
        WHERE cl.scored_at >= ? AND cl.scored_at < ?
          AND cl.relevance_score IS NOT NULL
        """,
        (window_start.isoformat(), window_end.isoformat()),
    )
    return cur.fetchall()


def compute_drift_for_window(
    conn: sqlite3.Connection,
    ts_now: datetime | None = None,
    today_hours: int = 24,
    prior_days: int = 7,
) -> list[DriftRecord]:
    """
    Compute drift per (backend, source_category) bucket.

    today window: [ts_now - today_hours, ts_now)
    prior window: [ts_now - today_hours - prior_days, ts_now - today_hours)

    A bucket is flagged when n_today >= MIN_N_TODAY AND n_prior >= MIN_N_PRIOR
    AND sigma_prior > 0 AND |mu_today - mu_prior| / sigma_prior > DRIFT_SIGMA_THRESHOLD.
    """
    if ts_now is None:
        ts_now = now_karaganda()

    today_start = ts_now - timedelta(hours=today_hours)
    prior_start = today_start - timedelta(days=prior_days)

    today_rows = _fetch_scores_window(conn, today_start, ts_now)
    prior_rows = _fetch_scores_window(conn, prior_start, today_start)

    def bucket(rows: list[tuple[str | None, str | None, int]]) -> dict[tuple[str, str], list[float]]:
        out: dict[tuple[str, str], list[float]] = {}
        for backend, cat, score in rows:
            key = (backend or "unknown", cat or "other")
            out.setdefault(key, []).append(float(score))
        return out

    today_buckets = bucket(today_rows)
    prior_buckets = bucket(prior_rows)

    all_keys = set(today_buckets) | set(prior_buckets)
    records: list[DriftRecord] = []
    for key in sorted(all_keys):
        backend, cat = key
        t = today_buckets.get(key, [])
        p = prior_buckets.get(key, [])
        mu_t = _mean(t)
        mu_p = _mean(p)
        sigma_p = _stddev(p)

        drift_mag: float | None = None
        flagged = False
        if (
            len(t) >= MIN_N_TODAY
            and len(p) >= MIN_N_PRIOR
            and sigma_p is not None
            and sigma_p > 0
            and mu_t is not None
            and mu_p is not None
        ):
            drift_mag = abs(mu_t - mu_p) / sigma_p
            flagged = drift_mag > DRIFT_SIGMA_THRESHOLD

        records.append(
            DriftRecord(
                backend=backend,
                source_category=cat,
                mu_today=mu_t,
                n_today=len(t),
                mu_prior=mu_p,
                sigma_prior=sigma_p,
                n_prior=len(p),
                drift_magnitude=drift_mag,
                flagged=flagged,
            )
        )
    return records


def apply_drift_snapshots(
    db_path: str | Path,
    ts_now: datetime | None = None,
) -> int:
    """
    Compute drift + insert snapshot rows. Returns count of flagged rows in this snapshot.

    Idempotent — INSERT OR IGNORE on UNIQUE(scored_at, backend, source_category).
    """
    if ts_now is None:
        ts_now = now_karaganda()
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_table(conn)
        records = compute_drift_for_window(conn, ts_now=ts_now)
        scored_at = ts_now.isoformat()
        flagged_count = 0
        for r in records:
            conn.execute(
                """
                INSERT OR IGNORE INTO score_drift_snapshots
                (scored_at, backend, source_category, mu_today, n_today,
                 mu_prior, sigma_prior, n_prior, drift_magnitude, flagged)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scored_at, r.backend, r.source_category,
                    r.mu_today, r.n_today,
                    r.mu_prior, r.sigma_prior, r.n_prior,
                    r.drift_magnitude, 1 if r.flagged else 0,
                ),
            )
            if r.flagged:
                flagged_count += 1
        conn.commit()
        return flagged_count
    finally:
        conn.close()


def count_recent_flags(db_path: str | Path, hours: int = RECENT_FLAGS_WINDOW_HOURS) -> int:
    """
    Return count of flagged rows in score_drift_snapshots within last `hours` hours.

    Used by run_pipeline.py to surface `score_drift_flags=N` in OK log.
    Fail-soft: returns 0 if table missing or any error.
    """
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            ensure_table(conn)
            cutoff = (now_karaganda() - timedelta(hours=hours)).isoformat()
            cur = conn.execute(
                "SELECT COUNT(*) FROM score_drift_snapshots WHERE scored_at >= ? AND flagged = 1",
                (cutoff,),
            )
            return int(cur.fetchone()[0])
        finally:
            conn.close()
    except Exception:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute curation_logs score drift snapshot.")
    parser.add_argument("--db", default="curator.db", help="Path to curator.db (default: curator.db)")
    args = parser.parse_args(argv)
    flagged = apply_drift_snapshots(args.db)
    print(f"score_drift_snapshots inserted; flagged={flagged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
