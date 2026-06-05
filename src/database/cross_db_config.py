"""Cross-repo curator.db path resolver.

3-tier resolution: CURATOR_DB_PATH env > config/post_draft.json > 'curator.db' default.

See `.planning/contracts/v1.2-second-brain.md` for the cross-repo contract
between autorss-feed and second_brain. The JSON config file at
`<repo>/config/post_draft.json` declares an absolute path to `curator.db`
so the second_brain `/post-draft` companion handler knows where to write back.

NOTE on contract divergence (autorss-side vs second_brain-side):
The contract document instructs second_brain to "fail loudly" on a missing
config file. THAT rule applies to second_brain's direct json.load of the file.
This Python resolver is for AUTORSS-side use (tests, possible future
cross_db.connect_curator() integration). Autorss already knows curator.db
location locally — loud-fail on missing file would break unrelated tests.
Therefore: missing file + no env var → silent fallback to 'curator.db'.

Schema version mismatch and malformed JSON DO raise loudly here, since they
indicate active drift / corruption rather than the file simply being absent.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

EXPECTED_SCHEMA_VERSION = "v1.2"


def _resolve_config_path() -> Path:
    """Return the canonical path to config/post_draft.json relative to repo root.

    Anchored to module location (parents[2] from src/database/cross_db_config.py
    is the repo root), NOT cwd. This makes resolution stable regardless of
    where the caller invokes Python from.

    WR-05: validate the resolved root by looking for `pyproject.toml` (a
    sentinel file that MUST exist at repo root). If a future relocation of
    this module changes its depth from the root, parents[2] would silently
    return a wrong path; the sentinel check raises RuntimeError instead of
    letting the resolver fall through to the 'curator.db' default with no
    signal.
    """
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").exists():
        raise RuntimeError(
            f"cross_db_config: repo root sentinel missing at {root} "
            "(expected pyproject.toml). Module may have been relocated "
            "without updating parents[N] depth."
        )
    return root / "config" / "post_draft.json"


def get_curator_db_path() -> str:
    """Resolve the path to curator.db using 3-tier precedence.

    Order:
      1. Env var CURATOR_DB_PATH (if set and non-empty after strip).
      2. config/post_draft.json (validated; schema_version must equal 'v1.2').
      3. Default: 'curator.db' (relative to cwd).

    Raises:
        ValueError: if config file is present but malformed JSON.
        ValueError: if config file is present but schema_version is missing
                    or != 'v1.2'.

    Returns:
        Resolved curator.db path as a string.
    """
    # Tier 1: env var
    env = os.environ.get("CURATOR_DB_PATH")
    if env and env.strip():
        return env

    # Tier 2: config file
    config_path = _resolve_config_path()
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"cross_db_config: invalid JSON at {config_path}: {exc.msg}"
            ) from exc

        got_version = data.get("schema_version")
        if got_version != EXPECTED_SCHEMA_VERSION:
            raise ValueError(
                f"cross_db_config: schema_version mismatch in {config_path}: "
                f"expected '{EXPECTED_SCHEMA_VERSION}', got {got_version!r}"
            )

        path = data.get("curator_db_path")
        if path:
            return path
        # Schema valid but key missing → fall through to default

    # Tier 3: default
    return "curator.db"
