"""Linked-project resolver (WORKER-07).

Maps `Связь: <project>` alias strings from digest items to absolute
filesystem paths under the Obsidian vault. Used by docs-mode (Plan 18-03)
to compute `target_path: 40_Projects/<linked>/<slug>.md` and by link-mode
(Plan 18-04) to prefer cross-refs to the linked project.

Pure-function module. All env reads happen INSIDE function bodies — the
Phase 17 CI guard (WORKER-12) rejects module-level `os.environ.get(`. The
aliases file is read lazily and mtime-cached: edits to
`config/project_aliases.json` hot-reload on the next call.

Path traversal note: alias VALUES must stay vault-relative (e.g.
`40_Projects/Ковчег`). The aliases file is checked into git, so values
are reviewed at commit time. We do not defensively reject `..` because
the threat model (T-18-04) treats this as user-edited config, not
externally writable input.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Final

_DEFAULT_ALIASES_PATH: Final[str] = "config/project_aliases.json"
_DEFAULT_VAULT_PATH: Final[str] = "vault"  # Generic fallback; override via OBSIDIAN_VAULT_PATH

# (cache_key, parsed_aliases_dict) — cache_key is (mtime, st_size, st_ino).
# WR-01: mtime alone is insufficient on Windows (NTFS 1-2s resolution); two
# writes within the same FS-tick can produce identical mtime but different
# content. Combining size + inode catches the common edit-twice-fast case.
# On Windows, st_ino may be 0 for some filesystems — size alone still differs
# whenever a key is added/removed, which is the realistic edit pattern.
_CACHE: tuple[tuple[float, int, int], dict[str, str]] | None = None


def _aliases_path() -> Path:
    return Path(os.environ.get("PROJECT_ALIASES_PATH", _DEFAULT_ALIASES_PATH))


def _vault_root() -> Path:
    return Path(
        os.environ.get("DRAFT_OUTPUT_DIR")
        or os.environ.get("OBSIDIAN_VAULT_PATH", _DEFAULT_VAULT_PATH)
    )


def _load_aliases() -> dict[str, str]:
    """Read aliases JSON with mtime-based caching.

    Missing file → return {} (graceful: caller treats as 'unresolved').
    Bad JSON or non-dict root → return {} (T-18-01: tampering mitigation —
    a corrupted config should not crash the worker).
    """
    global _CACHE
    path = _aliases_path()
    if not path.exists():
        # Drop stale cache so a later `path` recreation is picked up.
        _CACHE = None
        return {}
    stat = path.stat()
    # WR-01: composite cache key catches sub-second hot-edit races on Windows.
    key = (stat.st_mtime, stat.st_size, getattr(stat, "st_ino", 0) or 0)
    if _CACHE is not None and _CACHE[0] == key:
        return _CACHE[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    aliases = {str(k): str(v) for k, v in data.items()}
    _CACHE = (key, aliases)
    return aliases


def _lookup_rel(name: str) -> str | None:
    """Internal alias lookup (exact then casefold). Returns vault-relative
    path string from the aliases file, or None on miss. Pure helper used by
    both ``resolve`` and ``resolve_rel`` so the casefold-fallback semantics
    stay in one place.

    IN-01 note: on duplicate-casefold keys, dict insertion order wins.
    """
    aliases = _load_aliases()
    rel = aliases.get(name)
    if rel is None:
        target = name.casefold()
        for k, v in aliases.items():
            if k.casefold() == target:
                rel = v
                break
    return rel


def resolve(name: str | None) -> tuple[str | None, str]:
    """Map an alias to (absolute_dir_path, status).

    status ∈ {'resolved', 'unresolved'}.
      - None / empty name → (None, 'unresolved')
      - exact alias hit  → (vault / aliases[name], 'resolved')
      - case-insensitive (.casefold()) fallback hit → (vault / matched, 'resolved')
      - miss             → (None, 'unresolved')

    Never raises — bad input or bad config returns ('unresolved').

    IN-01 — duplicate-casefold precedence:
      If ``project_aliases.json`` happens to contain two keys that casefold
      to the same string (e.g. ``"Ковчег"`` and ``"КОВЧЕГ"``) and the lookup
      misses on exact match, the casefold fallback returns whichever key
      appears FIRST in dict insertion order. There is no warning and no
      precedence rule beyond insertion order. The aliases file is
      hand-curated and reviewed at commit time, so duplicates are expected
      to be caught by a human; if you need deterministic behaviour, ensure
      keys are unique under ``str.casefold()``.
    """
    if not name:
        return (None, "unresolved")
    rel = _lookup_rel(name)
    if rel is None:
        return (None, "unresolved")
    return (str(_vault_root() / rel), "resolved")


def resolve_rel(name: str | None) -> tuple[str | None, str]:
    """Map an alias to (vault_relative_path, status).

    Same semantics as ``resolve`` but returns the vault-relative path
    (e.g. ``'40_Projects/Ковчег'``) instead of the absolute filesystem path.
    Used by docs-mode (WR-02) to build ``target_path`` frontmatter values
    that stay separator-consistent across platforms — Windows backslashes
    in the vault root would mix with forward-slash slug joins otherwise.
    """
    if not name:
        return (None, "unresolved")
    rel = _lookup_rel(name)
    if rel is None:
        return (None, "unresolved")
    return (rel, "resolved")
