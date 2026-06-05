"""Wave 0 smoke + import-isolation tests for the OSS entry point.

PKG-01: main_oss.py is a generic entry point (no personal modules).
PKG-02: .env.example, LICENSE, README.md are complete and correct.
PKG-03: smoke script exists (tests/test_smoke_oss.py is the scripted portion;
        scripts/smoke_oss.py is the live manual step).

Tests for PKG-01 go GREEN in Plan 01 (Task 2).
Tests for PKG-02 go GREEN in Plan 02.
Tests for PKG-03 go GREEN in Plan 03.
"""

import os
from pathlib import Path


# ── PKG-01 ─────────────────────────────────────────────────────────────────


def test_main_oss_imports_only_generic_modules():
    """main_oss.py must not import personal modules or contain personal tokens.

    Checks for:
    - Personal module imports (dashboard, brainstorm_decisions, compute_mi, compute_roi)
    - Personal content tokens (personal name in Cyrillic, paper_tldr)
    - Personal route path literals (/telemetry, /roi, /papers as @app.get decorators)
    """
    source = Path("src/web/main_oss.py").read_text(encoding="utf-8")

    # Personal module import patterns — these must not appear as import statements
    banned_imports = [
        "from src.web.dashboard",
        "from src.web.brainstorm_decisions",
        "from src.eval.mi_backend",
        "from src.eval.channel_roi",
        "import dashboard",
        "import brainstorm_decisions",
        "import compute_mi",
        "import compute_roi",
    ]
    for token in banned_imports:
        assert token not in source, (
            f"Personal import {token!r} found in src/web/main_oss.py"
        )

    # Personal content tokens that must never appear anywhere in the file
    banned_content = [
        "compute_mi_all",
        "compute_roi",
        "Арыстан",
        "paper_tldr",
    ]
    for token in banned_content:
        assert token not in source, (
            f"Personal content token {token!r} found in src/web/main_oss.py"
        )

    # Personal route decorators — these routes must not be defined
    banned_routes = [
        '@app.get("/telemetry")',
        '@app.get("/roi")',
        '@app.get("/papers")',
        '@app.get("/career")',
        '@app.get("/compare")',
    ]
    for token in banned_routes:
        assert token not in source, (
            f"Personal route {token!r} found in src/web/main_oss.py"
        )


def test_main_oss_mounts_generic_routers():
    """main_oss.py must include_router for all 5 generic routers."""
    source = Path("src/web/main_oss.py").read_text(encoding="utf-8")
    required = [
        "reactions_router",
        "sources_router",
        "chat_router",
        "threshold_router",
        "dwell_router",
    ]
    for name in required:
        assert name in source, f"Router {name!r} missing from src/web/main_oss.py"
    assert source.count("include_router") >= 5, (
        "Expected at least 5 include_router calls in src/web/main_oss.py"
    )


async def test_main_oss_feed_loads(tmp_path, monkeypatch):
    """Integration: main_oss app serves GET / with 200 without personal modules."""
    import aiosqlite
    from httpx import AsyncClient, ASGITransport
    from src.database.client import init_db

    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    # Initialise schema so GET / can query the digests table.
    async with aiosqlite.connect(db_path) as db:
        await init_db(db)

    # Import inside test so DB_PATH env is set before module-level code runs.
    from src.web.main_oss import app  # noqa: PLC0415

    monkeypatch.setattr("src.web.main_oss.DB_PATH", db_path)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get("/")
    assert resp.status_code == 200


# ── PKG-02 ─────────────────────────────────────────────────────────────────


def test_env_example_has_required_keys():
    """.env.example must document all Phase 107/108 env vars."""
    env_text = Path(".env.example").read_text(encoding="utf-8")
    required_keys = [
        "OBSIDIAN_PROFILE_DIR",
        "DRAFT_OUTPUT_DIR",
        "PROFILE_CRITICAL_FILES",
        "PROFILE_ADVISORY_FILES",
        "CURATOR_THRESHOLD",
        "CURATOR_BACKEND",
        "CURATOR_MODEL",
    ]
    for key in required_keys:
        assert key in env_text, f"Key {key!r} missing from .env.example"
    # EDITOR_BACKEND without suffix is dead code (see MEMORY.md); use per-cat vars instead.
    lines = env_text.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("EDITOR_BACKEND=") and not stripped.startswith("#"):
            raise AssertionError(
                "EDITOR_BACKEND= (no suffix) is dead code; use EDITOR_BACKEND_<CAT>= instead"
            )
    assert any("EDITOR_BACKEND_" in ln for ln in lines), (
        "EDITOR_BACKEND_<CAT> pattern missing from .env.example"
    )


def test_license_is_mit():
    """LICENSE must exist and contain MIT License with project attribution."""
    text = Path("LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "autorss_feed contributors" in text


def test_readme_has_required_sections():
    """README.md must have quickstart, mermaid diagram, as-is notice, CURATOR_BACKEND."""
    readme = Path("README.md").read_text(encoding="utf-8")
    lower = readme.lower()
    assert "quickstart" in lower or "quick start" in lower, (
        "README.md missing Quickstart section"
    )
    assert "```mermaid" in readme, "README.md missing Mermaid diagram block"
    assert "as-is" in lower or "no support" in lower or "no-support" in lower, (
        "README.md missing as-is / no-support notice"
    )
    assert "CURATOR_BACKEND" in readme, "README.md missing CURATOR_BACKEND reference"


# ── PKG-03 ─────────────────────────────────────────────────────────────────


def test_smoke_script_exists():
    """scripts/smoke_oss.py must exist (PKG-03 live smoke entry point)."""
    assert Path("scripts/smoke_oss.py").exists(), (
        "scripts/smoke_oss.py missing — will be created in Plan 03"
    )
