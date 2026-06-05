"""Phase 51 (TRUNC-04, HARD-GATED + KILL-PENDING-VERDICT) — chunking helpers.

For posts >50K tokens (~200K chars), split into overlapping ~30K-token windows
and aggregate per-window scores via max-vote. Logs per-chunk score variance
for diagnostics (column add deferred until KILL/PROCEED decision lands).

**KILL-PENDING-VERDICT:** This module ships as future-ready infrastructure
that is NOT wired into the batch scorer. Activation gated on Phase 48 audit
verdict + post-P49/P50 re-measurement. See `.planning/phases/51-*/51-CONTEXT.md`.

Char/token rate: 4 chars/token (consistent with Phase 42 / 43).
Default window: 120000 chars ≈ 30K tokens.
Default overlap: 8000 chars ≈ 2K tokens — boundary signal preservation.
"""
from __future__ import annotations

import math
from typing import Sequence

DEFAULT_WINDOW = 120_000  # ~30K tokens at 4 chars/token
DEFAULT_OVERLAP = 8_000   # ~2K tokens overlap
MIN_CHARS_TO_CHUNK = 200_000  # ~50K tokens threshold for activation


def chunk_overlap(
    text: str,
    window: int = DEFAULT_WINDOW,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Split text into overlapping windows.

    Returns:
    - `[]` for empty text
    - `[text]` for text shorter than window (no chunking needed)
    - `[chunk1, chunk2, ...]` with `overlap` chars of shared content between
      adjacent chunks for boundary signal preservation

    Fail-soft on invalid args: defaults applied if window<=0 or overlap<0.
    """
    if not text:
        return []
    if window <= 0:
        window = DEFAULT_WINDOW
    if overlap < 0 or overlap >= window:
        overlap = DEFAULT_OVERLAP
    if len(text) <= window:
        return [text]

    chunks: list[str] = []
    step = window - overlap
    start = 0
    while start < len(text):
        end = min(start + window, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def max_vote_aggregate(scores: Sequence[float]) -> int:
    """Max-score aggregation across chunk scores. Returns 0 for empty input."""
    if not scores:
        return 0
    return int(max(scores))


def compute_variance(scores: Sequence[float]) -> float:
    """Population variance of chunk scores. Returns 0 for empty / single."""
    n = len(scores)
    if n <= 1:
        return 0.0
    mean = sum(scores) / n
    return sum((s - mean) ** 2 for s in scores) / n


def needs_chunking(text: str, threshold: int = MIN_CHARS_TO_CHUNK) -> bool:
    """Check whether text qualifies for chunking. Used by activation gate."""
    return bool(text) and len(text) >= threshold


def estimate_chunk_count(text: str, window: int = DEFAULT_WINDOW, overlap: int = DEFAULT_OVERLAP) -> int:
    """Predict chunk count without actually splitting. Useful for cost estimation."""
    if not text:
        return 0
    if len(text) <= window:
        return 1
    if window <= 0 or overlap < 0 or overlap >= window:
        return 1
    step = window - overlap
    return math.ceil((len(text) - overlap) / step)
