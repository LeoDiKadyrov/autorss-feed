"""Profile loader — META layer entry point.

Reads structured profile from OBSIDIAN_PROFILE_DIR (PROFILE_OBSIDIAN_PATH legacy alias). Generic fallback path: "profile".
Falls back to v1 hardcoded text on D-17 conditions:
  1. Path does not exist or is unreadable
  2. Any of 3 critical files missing (D-03)
  3. Heuristic parser raises an exception

Logs a single WARNING line per fallback trigger (D-18). Web UI badge picks up the warning
to surface "PROFILE FALLBACK" status (D-19).

Phase 14 META-02: mtime-based caching. `_PROFILE_CACHE` holds (profile_text, max_mtime).
On each `load_profile()` call we stat CRITICAL + ADVISORY files and compare max-mtime
to the cached snapshot. Cache hit → return cached string (no re-parse). Mtime advanced
or cache empty → rebuild + repopulate. Drive-unmount with cache present → return last
known-good cache (preserve operation through transient unavailability).
"""
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level default — read inside _get_profile_path() so monkeypatch works (Pitfall 3)
DEFAULT_PROFILE_PATH = Path("profile")  # Generic fallback; override via OBSIDIAN_PROFILE_DIR

# D-03: critical files for fallback decision
# Generic names; users populate with their own content.
# Override via PROFILE_CRITICAL_FILES env var (comma-separated filenames).
CRITICAL_FILES: tuple[str, ...] = tuple(
    f.strip()
    for f in os.environ.get(
        "PROFILE_CRITICAL_FILES", "fact_dossier.md,voice_profile.md,writing_guide.md"
    ).split(",")
    if f.strip()
)

# Advisory files — read if present, skip silently if missing (no fallback trigger)
# Override via PROFILE_ADVISORY_FILES env var (comma-separated filenames).
ADVISORY_FILES: tuple[str, ...] = tuple(
    f.strip()
    for f in os.environ.get(
        "PROFILE_ADVISORY_FILES", "goals.md,content_persona.md,project-graph.md"
    ).split(",")
    if f.strip()
)

# Phase 14 META-02: in-memory cache (profile_text, max_mtime_at_build).
# None until first successful build; module-level so it survives across multiple
# load_profile() calls within the same process (e.g., curator reading the profile
# multiple times in one pipeline run, D-12).
_PROFILE_CACHE: tuple[str, float] | None = None


def clear_profile_cache() -> None:
    """Reset the module-level cache. Used by tests for isolation; callers may also
    invoke this after pipeline completion or to force a rebuild on next call."""
    global _PROFILE_CACHE
    _PROFILE_CACHE = None


def _get_profile_path() -> Path:
    """Resolve profile path with priority: OBSIDIAN_PROFILE_DIR > PROFILE_OBSIDIAN_PATH > DEFAULT_PROFILE_PATH.

    Priority rules:
    - OBSIDIAN_PROFILE_DIR: new, public var (set this one).
    - PROFILE_OBSIDIAN_PATH: legacy alias from pre-phase-107 installs.
    - DEFAULT_PROFILE_PATH: generic "profile" fallback if neither env var is set.
    - If OBSIDIAN_PROFILE_DIR is set to "" (empty), falls through to PROFILE_OBSIDIAN_PATH.

    Read INSIDE the function so monkeypatch.setenv works (Pitfall 1 — module-level
    binding freezes value at import time and breaks tests).
    """
    return Path(
        os.environ.get("OBSIDIAN_PROFILE_DIR")
        or os.environ.get("PROFILE_OBSIDIAN_PATH", str(DEFAULT_PROFILE_PATH))
    )


def _v1_hardcoded_profile() -> str:
    """Return the v1.0 fallback profile text. Imports the constant from src.agents.curator
    so there is a single source of truth for the verbatim string."""
    from src.agents.curator import _V1_FALLBACK_PROFILE
    return _V1_FALLBACK_PROFILE


def _max_profile_mtime(path: Path) -> float:
    """Return max mtime across CRITICAL + ADVISORY files in `path`.

    CRITICAL files are required — stat() raises OSError if any is missing/unreadable.
    ADVISORY files are skipped silently if not present.

    Caller decides fallback strategy on OSError (D-08).
    """
    mtimes: list[float] = []
    for fname in CRITICAL_FILES:
        mtimes.append((path / fname).stat().st_mtime)
    for fname in ADVISORY_FILES:
        adv = path / fname
        if adv.exists():
            mtimes.append(adv.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


def load_profile() -> str:
    """Public API. Load and assemble profile from filesystem; fall back on D-17 conditions.

    Phase 14: mtime-based cache invalidation. The first call builds and caches; subsequent
    calls within the same process compare current max-mtime to the cached value and either
    return the cached string (D-05 hit) or rebuild (mtime advanced).

    Returns:
        Assembled profile string suitable for embedding in curator prompt.
        Never raises — fall back to v1 text on any error.
    """
    global _PROFILE_CACHE
    path = _get_profile_path()

    # D-17 case 1 / D-06 / D-07: path unreachable
    if not path.exists() or not path.is_dir():
        if _PROFILE_CACHE is not None:
            logger.warning(
                "PROFILE REFRESH: path %s unreachable, using cached profile (last known good)",
                path,
            )
            return _PROFILE_CACHE[0]
        logger.warning("PROFILE FALLBACK: path %s unreachable", path)
        return _v1_hardcoded_profile()

    # D-17 case 2 / D-09: critical-file missing — does NOT cache
    missing = [f for f in CRITICAL_FILES if not (path / f).exists()]
    if missing:
        logger.warning("PROFILE FALLBACK: critical files missing: %s", missing)
        return _v1_hardcoded_profile()

    # D-03 / D-08: cache freshness check via aggregate max mtime
    try:
        current_mtime = _max_profile_mtime(path)
    except OSError as exc:
        logger.warning("PROFILE REFRESH: stat failed (%s); using cache or fallback", exc)
        if _PROFILE_CACHE is not None:
            return _PROFILE_CACHE[0]
        return _v1_hardcoded_profile()

    # D-05: cache hit when current_mtime <= cached_mtime
    if _PROFILE_CACHE is not None and current_mtime <= _PROFILE_CACHE[1]:
        logger.debug("PROFILE REFRESH: cache hit (mtime=%s)", current_mtime)
        return _PROFILE_CACHE[0]

    # Cache miss or mtime advanced — rebuild
    try:
        files_text: dict[str, str] = {}
        for fname in CRITICAL_FILES:
            files_text[fname] = (path / fname).read_text(encoding="utf-8")
        for fname in ADVISORY_FILES:
            adv = path / fname
            if adv.exists():
                files_text[fname] = adv.read_text(encoding="utf-8")

        # Late imports to keep module-load deterministic and to allow monkeypatching
        # `src.profile.extractors.extract_high` in tests (Pitfall 3 — fallback test).
        from src.profile import extractors
        from src.profile.schema import ProfileTiers, cap_tiers, render_profile

        tiers = ProfileTiers(
            high=extractors.extract_high(files_text),
            medium=extractors.extract_medium(files_text),
            reject_phrases=extractors.extract_reject_phrases(files_text),
            raw_prose=extractors.extract_raw_prose(files_text),
        )
        capped = cap_tiers(tiers, high_cap=15, medium_cap=25)
        profile_text = render_profile(capped)

        if _PROFILE_CACHE is None:
            logger.info("PROFILE REFRESH: cache built (mtime=%s)", current_mtime)
        else:
            logger.info(
                "PROFILE REFRESH: cache rebuilt (mtime advanced from %s to %s)",
                _PROFILE_CACHE[1], current_mtime,
            )
        _PROFILE_CACHE = (profile_text, current_mtime)
        return profile_text
    except Exception:
        logger.exception("PROFILE FALLBACK: parser raised on profile assembly")
        return _v1_hardcoded_profile()
