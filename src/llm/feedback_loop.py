"""Predictive coding feedback loop.

Curator score = prior. Web UI thumb-up/down = prediction error.
Recomputes per-category weights from recent vote history and persists
them in the 'feedback_priors' table.

Weight formula per category:
    weight = 0.5 + up_rate   (range [0.5, 1.5])
    up_rate = up_count / (up_count + down_count), defaults to 0.5 when no data.

Weights are rendered as a PREDICTIVE FEEDBACK WEIGHTS block appended to the
curator context, instructing the LLM to score high-feedback categories higher
and penalised categories lower.
"""
from __future__ import annotations

import datetime
import logging

import aiosqlite

logger = logging.getLogger(__name__)


async def update_priors(db: aiosqlite.Connection, window_days: int = 30) -> dict[str, float]:
    """Recompute per-category weights from recent feedback and persist them.

    Returns the updated weights dict {category: weight}.
    """
    cutoff = (
        datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=window_days)
    ).isoformat()

    async with db.execute(
        """
        SELECT COALESCE(s.category, 'other') AS category,
               SUM(CASE WHEN pf.rating = 1  THEN 1 ELSE 0 END) AS up_count,
               SUM(CASE WHEN pf.rating = -1 THEN 1 ELSE 0 END) AS down_count
        FROM post_feedback pf
        JOIN raw_posts rp ON rp.id = pf.post_id
        JOIN sources s ON s.id = rp.source_id
        WHERE pf.rated_at >= ?
        GROUP BY category
        """,
        (cutoff,),
    ) as cur:
        rows = await cur.fetchall()

    weights: dict[str, float] = {}
    for category, up, down in rows:
        total = (up or 0) + (down or 0)
        if total == 0:
            continue
        up_rate = (up or 0) / total
        weights[category] = round(0.5 + up_rate, 3)

    if not weights:
        return weights

    now = datetime.datetime.now(datetime.UTC).isoformat()
    await db.executemany(
        "INSERT OR REPLACE INTO feedback_priors (category, weight, updated_at) VALUES (?, ?, ?)",
        [(cat, w, now) for cat, w in weights.items()],
    )
    await db.commit()
    return weights


async def get_category_weights(db: aiosqlite.Connection) -> dict[str, float]:
    """Return persisted per-category weights. Empty dict when no priors exist."""
    async with db.execute("SELECT category, weight FROM feedback_priors") as cur:
        rows = await cur.fetchall()
    return {row[0]: row[1] for row in rows}


def render_weights_block(weights: dict[str, float]) -> str:
    """Format weight deviations as a curator context block.

    Only includes categories that meaningfully deviate from neutral (1.0).
    Returns empty string when there's nothing to report.
    """
    if not weights:
        return ""
    boosted = sorted([(cat, w) for cat, w in weights.items() if w > 1.1])
    penalised = sorted([(cat, w) for cat, w in weights.items() if w < 0.9])
    if not boosted and not penalised:
        return ""

    lines = ["\n\nPREDICTIVE FEEDBACK WEIGHTS (based on your recent votes):"]
    if boosted:
        cats = ", ".join(f"{cat} ({w:.2f}x)" for cat, w in boosted)
        lines.append(f"  BOOST: {cats} — score similar content higher than baseline.")
    if penalised:
        cats = ", ".join(f"{cat} ({w:.2f}x)" for cat, w in penalised)
        lines.append(f"  REDUCE: {cats} — score similar content lower than baseline.")
    return "\n".join(lines)
