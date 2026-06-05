"""
Phase 80 / PROMPT-ABLEND-01 — prompt-version α-blend router.

Provides:
- discover_versions(prompts_dir) -> list[(int, Path)] sorted asc by version int.
  Scans `<dir>/curator_v{n}.txt`; ignores `archive/` subdir and non-matching files.
- load_rollout_state(path) -> {"active": int, "rollout_started_at": iso|None}.
  Missing file → safe defaults ({"active": 1, "rollout_started_at": None}).
- alpha_for_now(state, now_dt, n_versions) -> float in [0.1, 1.0].
  α = 1.0 when rollout_started_at is None OR n_versions <= 1; else
  α = clamp(0.1 + 0.9 * elapsed_days / 7, 0.1, 1.0) (D-02).
- pick_version(post_id, versions, alpha) -> (int, Path).
  Deterministic: hash(post_id) % 1000 / 1000 < alpha → newest version, else
  previous-newest. Hash uses SHA-1 prefix for stability across Python runs
  (D-03; avoids PYTHONHASHSEED nondeterminism in builtin hash()).

Plan 02 will wire pick_version into the curator scoring call site.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Reject leading zeros (curator_v01.txt) — would collide with curator_v1.txt on int parse.
_VERSION_RE = re.compile(r"^curator_v([1-9]\d*)\.txt$")
_FLOOR = 0.1
_CEIL = 1.0
_ANNEAL_DAYS = 7.0


def discover_versions(prompts_dir: Path | str) -> list[tuple[int, Path]]:
    """Return sorted list of (version_int, path) for curator_v*.txt in prompts_dir.

    - Ignores subdirectories (including `archive/`).
    - Ignores files that don't match `curator_v{n}.txt`.
    - Returns [] when the directory is missing or empty.
    """
    p = Path(prompts_dir)
    if not p.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for entry in p.iterdir():
        if not entry.is_file():
            continue
        m = _VERSION_RE.match(entry.name)
        if not m:
            continue
        out.append((int(m.group(1)), entry))
    out.sort(key=lambda t: t[0])
    # WR-03 defence: regex above rejects leading zeros, but double-check no duplicates slip in.
    seen: dict[int, Path] = {}
    for v, pp in out:
        if v in seen:
            raise ValueError(
                f"duplicate prompt version {v}: {seen[v]} vs {pp}"
            )
        seen[v] = pp
    return out


def load_rollout_state(path: Path | str) -> dict:
    """Return rollout_state dict; missing file → safe defaults."""
    p = Path(path)
    if not p.exists():
        return {"active": 1, "rollout_started_at": None}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"active": 1, "rollout_started_at": None}
    return {
        "active": int(data.get("active", 1)),
        "rollout_started_at": data.get("rollout_started_at"),
    }


def alpha_for_now(state: dict, now_dt: datetime, n_versions: int = 2) -> float:
    """Compute current α per D-02.

    α = 1.0 when rollout_started_at is None OR n_versions <= 1
    α = clamp(0.1 + 0.9 * elapsed_days/7, 0.1, 1.0) otherwise.
    """
    if n_versions <= 1:
        return 1.0
    started_at = state.get("rollout_started_at")
    if not started_at:
        return 1.0
    try:
        started_dt = datetime.fromisoformat(started_at)
    except (TypeError, ValueError):
        return 1.0
    if started_dt.tzinfo is None:
        # WR-04: refuse to silently coerce naive timestamps to UTC.
        # An operator hand-editing rollout_state.json from local time would otherwise
        # see a multi-hour α drift. Behave as if rollout has not started.
        logger.warning(
            "rollout_state.rollout_started_at lacks tzinfo (%s) — refusing to anneal",
            started_at,
        )
        return 1.0
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    elapsed_days = (now_dt - started_dt).total_seconds() / 86400.0
    alpha = _FLOOR + 0.9 * (elapsed_days / _ANNEAL_DAYS)
    if alpha < _FLOOR:
        return _FLOOR
    if alpha > _CEIL:
        return _CEIL
    return alpha


def _stable_hash(post_id) -> int:
    """SHA-1 prefix hash — stable across Python invocations.

    Builtin hash() is randomised by PYTHONHASHSEED, which would shuffle
    α-blend assignments per process and break reproducibility. SHA-1 of
    str(post_id) gives a deterministic 32-bit int.
    """
    h = hashlib.sha1(str(post_id).encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def pick_version(
    post_id,
    versions: list[tuple[int, Path]],
    alpha: float,
) -> tuple[int, Path]:
    """Return (version_int, path) routed for post_id.

    - Empty versions list → raises ValueError (caller must seed at least v1).
    - 1 version → always returns it.
    - >=2 versions: hash(post_id) % 1000 / 1000 < alpha → newest; else
      second-newest (previous version, D-03). Only the top two versions
      are considered for routing — older versions are held for archive.
    """
    if not versions:
        raise ValueError("no prompt versions discovered")
    if len(versions) == 1:
        return versions[0]
    newest = versions[-1]
    previous = versions[-2]
    bucket = _stable_hash(post_id) % 1000 / 1000.0
    return newest if bucket < alpha else previous


__all__ = [
    "discover_versions",
    "load_rollout_state",
    "alpha_for_now",
    "pick_version",
]
