"""Phase 107 TRIM-02: central .env loader for scheduled-task entry points.

Personal paths (OBSIDIAN_PROFILE_DIR, DRAFT_OUTPUT_DIR, PROFILE_*_FILES) moved
from code defaults into .env. Scheduled tasks and uvicorn boot don't inherit a
shell env, so entry points call ``load_env()`` explicitly.

Test isolation: pytest sets ``AUTORSS_DISABLE_DOTENV=1`` in tests/conftest.py —
without the guard, a module-level ``load_dotenv()`` (e.g. src.web.main imported
at collection time) leaks the operator's personal .env values into the entire
test session and breaks tests that rely on unset env vars.
"""

import os


def load_env() -> None:
    """Load .env into process env unless disabled for tests.

    No-op when ``AUTORSS_DISABLE_DOTENV=1`` or python-dotenv is missing.
    Never overrides already-set env vars (dotenv default).
    """
    if os.environ.get("AUTORSS_DISABLE_DOTENV") == "1":
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()
