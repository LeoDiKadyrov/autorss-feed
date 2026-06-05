"""Vault writability preflight (WORKER-11).

The worker calls :func:`is_writable` on the vault drafts directory BEFORE
claiming any reaction row. If the Obsidian vault drive is unmounted (e.g.
``D:\\`` unavailable), the worker skips the entire tick -- no row is claimed,
no partial draft is written, a warning is logged, and the next tick retries.

Never raises. All exceptions collapse to ``False`` (T-17-05 mitigation).

WR-05: ``is_writable(path)`` no longer auto-creates a missing directory by
default. Previously a typo in ``OBSIDIAN_VAULT_PATH`` (e.g.
``/path/to/vault/wrong-name``) caused the worker to silently create
``/path/to/vault/wrong-name/10_Drafts/`` and write drafts there -- the user
wondered why their real vault had no new files. Now a missing dir returns
``False`` (preflight fail -> tick skip). Pass ``create=True`` for
project-local paths where auto-create is desired (e.g. ``logs/``).
"""
from __future__ import annotations

import os
from pathlib import Path


def is_writable(path: str, *, create: bool = False) -> bool:
    """Return True iff ``path`` is a writable directory.

    Behaviour:
    - Existing directory + ``os.W_OK`` -> True
    - Existing path that is NOT a directory (e.g. a file) -> False
    - Nonexistent path with ``create=False`` (default) -> False (WR-05).
      The caller MUST pass ``create=True`` to opt into auto-creation.
    - Nonexistent path with ``create=True``: try
      ``mkdir(parents=True, exist_ok=True)``; on success check ``os.W_OK``
      on the new dir.
    - Any error (drive offline, permission denied, malformed path) -> False

    Never raises.
    """
    try:
        if not path:
            # Empty path resolves to "." which is misleading for vault preflight.
            return False
        p = Path(path)
        if p.exists():
            if not p.is_dir():
                return False
            return os.access(str(p), os.W_OK)
        # WR-05: do NOT silently create on a misconfigured vault path.
        # Caller must explicitly opt in with create=True.
        if not create:
            return False
        try:
            p.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            return False
        return os.access(str(p), os.W_OK)
    except Exception:
        return False
