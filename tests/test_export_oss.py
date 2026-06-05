"""Tests for scripts/export_oss.py — REL-01 + REL-02 coverage.

Task 1 (TDD RED): scan_tree tests — 4 behaviors
Task 2 (TDD RED): export orchestration tests — 6 behaviors
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Bootstrap repo root onto sys.path so scripts/export_oss.py can be imported.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.export_oss import copy_included, run_export, scan_tree  # noqa: E402


# ─── Task 1: scan_tree tests ─────────────────────────────────────────────────


def test_scan_gate_clean(tmp_path):
    """scan_tree on a fixture tree with no markers returns empty list."""
    (tmp_path / "module.py").write_text("x = 1\nprint('hello')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Generic title\nNo personal data here.\n", encoding="utf-8")
    findings = scan_tree(tmp_path)
    assert findings == [], f"Expected 0 findings, got: {findings}"


def test_scan_gate_detects(tmp_path):
    """scan_tree detects an injected personal marker — regression guard."""
    # Plant a marker in a non-export_oss file to prove gate fires.
    planted = tmp_path / "some_module.py"
    planted.write_text("user = 'Arystan'\nprint(user)\n", encoding="utf-8")
    findings = scan_tree(tmp_path)
    assert len(findings) >= 1, "scan_tree must detect injected marker 'Arystan'"
    paths = [str(f[0]) for f in findings]
    assert any("some_module" in p for p in paths), "finding must be in the planted file"


def test_scan_opacity_fp(tmp_path):
    """CSS opacity in .html does NOT trigger scan gate (path-anchored regex)."""
    html = tmp_path / "index.html"
    html.write_text(
        "<style>.hero { opacity: 0.5; transition: opacity 0.2s; }</style>\n"
        "<div style='opacity:1'>content</div>\n",
        encoding="utf-8",
    )
    findings = scan_tree(tmp_path)
    assert findings == [], f"CSS opacity must not trigger scan; got: {findings}"


def test_scan_dobsidian_variants(tmp_path):
    """All D:\\Obsidian / D:/Obsidian / D:\\\\Obsidian variants are flagged."""
    lines = [
        r'path = r"D:\Obsidian\opacity"',
        r'path = "D:/Obsidian/notes"',
        r'path = "D:\\Obsidian\\thing"',
    ]
    for i, line in enumerate(lines):
        f = tmp_path / f"file{i}.py"
        f.write_text(line + "\n", encoding="utf-8")

    findings = scan_tree(tmp_path)
    assert len(findings) >= 3, f"Expected ≥3 findings for D:\\Obsidian variants, got: {findings}"


# ─── Task 2: export orchestration tests ──────────────────────────────────────


def _make_minimal_repo(root: Path) -> None:
    """Build a minimal fake repo tree for export orchestration tests."""
    # src/web/main_oss.py (the file to be renamed)
    main_oss = root / "src" / "web" / "main_oss.py"
    main_oss.parent.mkdir(parents=True, exist_ok=True)
    main_oss.write_text("# main_oss\napp = None\n", encoding="utf-8")

    # scripts/smoke_oss.py (must be patched post-copy)
    scripts = root / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "smoke_oss.py").write_text(
        "from src.web.main_oss import app\n", encoding="utf-8"
    )

    # config/sources.txt (must NOT ship)
    config = root / "config"
    config.mkdir(exist_ok=True)
    (config / "sources.txt").write_text("@personal_channel | ai | Personal\n", encoding="utf-8")

    # README.md with rename note + main_oss uvicorn cmd
    readme_text = (
        "# autorss-feed\n\n"
        "```bash\n"
        "python -m uvicorn src.web.main_oss:app --reload\n"
        "# Note: in this source repo the entry point is src/web/main_oss.py;\n"
        "# the export script (phase 110) renames it to src/web/main.py.\n"
        "```\n"
    )
    (root / "README.md").write_text(readme_text, encoding="utf-8")

    # LICENSE (neutral)
    (root / "LICENSE").write_text("MIT License\n", encoding="utf-8")

    # run_pipeline.py
    (root / "run_pipeline.py").write_text("# pipeline\n", encoding="utf-8")

    # pyproject.toml
    (root / "pyproject.toml").write_text("[project]\nname='autorss-feed'\n", encoding="utf-8")

    # .planning/ (must NOT ship)
    planning = root / ".planning"
    planning.mkdir(exist_ok=True)
    (planning / "STATE.md").write_text("state", encoding="utf-8")

    # .env (must NOT ship)
    (root / ".env").write_text("SECRET=abc\n", encoding="utf-8")

    # export/ALLOWLIST.yaml
    export_dir = root / "export"
    export_dir.mkdir(exist_ok=True)
    allowlist_content = (
        "included:\n"
        "  - src/web/main_oss.py\n"
        "  - scripts/smoke_oss.py\n"
        "  - README.md\n"
        "  - LICENSE\n"
        "  - run_pipeline.py\n"
        "  - pyproject.toml\n"
        "  - export/ALLOWLIST.yaml\n"
    )
    (export_dir / "ALLOWLIST.yaml").write_text(allowlist_content, encoding="utf-8")


def test_export_tree_contents(tmp_path):
    """After run_export, included files exist; excluded files are absent."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_minimal_repo(repo)
    dest = tmp_path / "export"

    run_export(repo, dest, force=False)

    # Included files present
    assert (dest / "src" / "web" / "main.py").exists(), "main.py must exist in export"
    assert (dest / "README.md").exists(), "README.md must exist"
    assert (dest / "LICENSE").exists(), "LICENSE must exist"

    # Excluded files absent
    assert not (dest / ".planning").exists(), ".planning must not ship"
    assert not (dest / ".env").exists(), ".env must not ship"
    assert not (dest / "config" / "sources.txt").exists(), "config/sources.txt must not ship"


def test_main_rename(tmp_path):
    """main_oss.py is renamed to main.py; main_oss.py absent in export."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_minimal_repo(repo)
    dest = tmp_path / "export"

    run_export(repo, dest, force=False)

    assert (dest / "src" / "web" / "main.py").exists(), "dest/src/web/main.py must exist"
    assert not (dest / "src" / "web" / "main_oss.py").exists(), "main_oss.py must be absent"


def test_smoke_import_patched(tmp_path):
    """Exported smoke_oss.py references src.web.main, not src.web.main_oss."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_minimal_repo(repo)
    dest = tmp_path / "export"

    run_export(repo, dest, force=False)

    smoke_text = (dest / "scripts" / "smoke_oss.py").read_text(encoding="utf-8")
    assert "src.web.main_oss" not in smoke_text, "smoke_oss.py must not reference main_oss after patch"
    assert "src.web.main" in smoke_text, "smoke_oss.py must reference src.web.main"


def test_sources_example(tmp_path):
    """sources.txt.example present; sources.txt absent in export."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_minimal_repo(repo)
    dest = tmp_path / "export"

    run_export(repo, dest, force=False)

    assert (dest / "config" / "sources.txt.example").exists(), "sources.txt.example must exist"
    assert not (dest / "config" / "sources.txt").exists(), "sources.txt must not ship"


def test_readme_patched(tmp_path):
    """Exported README has no rename-note line and uvicorn uses src.web.main:app."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_minimal_repo(repo)
    dest = tmp_path / "export"

    run_export(repo, dest, force=False)

    readme = (dest / "README.md").read_text(encoding="utf-8")
    assert "main_oss" not in readme, "README must not mention main_oss after patch"
    assert "src.web.main:app" in readme or "main:app" in readme, "README uvicorn cmd must use main:app"
    # Rename note lines must be gone
    assert "the export script (phase 110)" not in readme, "rename note must be stripped"


def test_git_init(tmp_path):
    """Export dir has .git/ present after run_export."""
    if not shutil.which("git"):
        pytest.skip("git not available")

    repo = tmp_path / "repo"
    repo.mkdir()
    _make_minimal_repo(repo)
    dest = tmp_path / "export"

    run_export(repo, dest, force=False)

    assert (dest / ".git").exists(), "export dir must contain .git after git init"
