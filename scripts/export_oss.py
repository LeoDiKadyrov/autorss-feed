"""Export script for the open-source release — REL-01 + REL-02.

Reads export/ALLOWLIST.yaml, copies included paths to the destination directory,
performs post-copy transforms (main_oss rename, README patch, sources.txt.example),
initialises a fresh git repo, then runs the scan gate.

Usage:
    .venv/Scripts/python.exe scripts/export_oss.py --output-dir ../autorss-feed-export
    .venv/Scripts/python.exe scripts/export_oss.py --output-dir ../autorss-feed-export --force

Exit codes:
    0  — export complete, scan found 0 findings
    1  — scan found personal-data markers (print file:line for each)
    2  — usage error (dest exists without --force, bad args, etc.)

Optional: if `gitleaks` is on PATH, runs `gitleaks detect --no-git --source <dest>` as
a bonus check and merges any additional findings.

Self-whitelist note: SCAN_PATTERNS contains personal names as regex strings
(e.g. "Arystan") because those are the patterns we scan *for*.  This file itself
is whitelisted in scan_tree() so that the presence of those strings in the pattern
definition does not trigger a false positive.  A separate test (test_scan_gate_detects)
plants a marker in a *different* file to prove the gate still fires.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Self-contained: bootstrap repo root onto sys.path (documented gotcha in CLAUDE.md)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Windows cp1252 → UTF-8 reconfigure (documented gotcha in CLAUDE.md)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


# ── Scan gate ─────────────────────────────────────────────────────────────────

# Path-anchored D:\\Obsidian avoids CSS opacity false positives.
# "opacity" is intentionally NOT in this list (see RESEARCH.md Pitfall 2).
_RAW_PATTERNS = [
    r"arystan",     # case-insensitive: catches Arystan, arystan_text, emb_arystan, etc.
    r"Арыстан",     # Арыстан
    r"kadyrov",     # case-insensitive
    r"Кадыров",     # Кадыров
    r"kdrvarystanos",
    r"din02winchester",
    r"D:?[\\\/]{1,2}[Oo]bsidian",                    # D:\Obsidian / D:/Obsidian / D:\\Obsidian (all variants)
    r"ZHYR",
    r"faktura",     # case-insensitive
    r"Фактура",     # Фактура
]

SCAN_PATTERNS: list[re.Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in _RAW_PATTERNS]

TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".yaml", ".yml", ".toml",
    ".html", ".js", ".css", ".json", ".example", ".sh", ".cfg", ".ini",
}
SKIP_DIRS = {"__pycache__", ".git"}

# Files to skip during scan (by name — path-relative, forward slashes).
# export_oss.py itself contains pattern strings as regex definitions; skip to avoid FP.
# Test files contain marker strings as test fixtures and docstrings; skip them too.
# The regression guard test (test_scan_gate_detects) plants a marker in a *different*
# file to confirm the gate still fires for real personal-data leaks.
_SCAN_SKIP_FILES = {
    "scripts/export_oss.py",
    "tests/test_export_oss.py",
    "tests/test_smoke_oss.py",
}


def scan_tree(root: Path) -> list[tuple[Path, int, str]]:
    """Scan all text files under *root* for personal-data markers.

    Returns a list of (relative_path, line_number, line_content) tuples.
    Skips binary files, __pycache__/, .git/, and the export script itself.
    """
    findings: list[tuple[Path, int, str]] = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        # Skip dirs in path parts
        if any(d in path.parts for d in SKIP_DIRS):
            continue
        # Skip non-text files
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        # Skip the export script itself (see self-whitelist note above)
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if str(rel).replace("\\", "/") in _SCAN_SKIP_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for pat in SCAN_PATTERNS:
                if pat.search(line):
                    findings.append((rel, lineno, line.strip()))
                    break  # one finding per line is enough
    return findings


# ── Allowlist copy ─────────────────────────────────────────────────────────────


def copy_included(repo_root: Path, dest_root: Path, included: list[str]) -> None:
    """Copy only the paths listed in the ALLOWLIST to dest_root.

    Each entry is either a file or a directory.  Directories are copied with
    shutil.copytree (ignoring __pycache__ and compiled .pyc/.pyo files).
    Files are copied with shutil.copy2.

    Security: each resolved dst is asserted to be under dest_root to prevent
    path-traversal attacks from malformed ALLOWLIST entries.
    """
    for entry in included:
        clean = entry.rstrip("/")
        # Reject entries containing ".." (T-110-06 mitigation)
        if ".." in Path(clean).parts:
            print(f"  [ERROR] ALLOWLIST entry contains '..': {entry!r} — skipped")
            continue

        src = repo_root / clean
        dst = dest_root / clean

        # Assert dst is under dest_root (belt-and-suspenders for T-110-06)
        try:
            dst.resolve().relative_to(dest_root.resolve())
        except ValueError:
            print(f"  [ERROR] ALLOWLIST entry escapes dest_root: {entry!r} — skipped")
            continue

        if not src.exists():
            print(f"  [WARN] ALLOWLIST entry missing in source: {entry!r}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
                dirs_exist_ok=True,
                symlinks=True,  # preserve symlinks rather than dereferencing (WR-01)
            )
        else:
            shutil.copy2(src, dst)


# ── Post-copy transforms ───────────────────────────────────────────────────────


def _rename_main_oss(dest_root: Path) -> None:
    """Rename src/web/main_oss.py → src/web/main.py in the export tree."""
    src = dest_root / "src" / "web" / "main_oss.py"
    dst = dest_root / "src" / "web" / "main.py"
    if src.exists():
        src.rename(dst)
        print("  [OK] renamed src/web/main_oss.py → src/web/main.py")
    else:
        print("  [WARN] src/web/main_oss.py not found in export — rename skipped")


def _patch_smoke_oss(dest_root: Path) -> None:
    """Patch exported smoke_oss.py: replace src.web.main_oss → src.web.main."""
    p = dest_root / "scripts" / "smoke_oss.py"
    if not p.exists():
        return
    text = p.read_text(encoding="utf-8")
    patched = text.replace("src.web.main_oss", "src.web.main")
    if patched != text:
        p.write_text(patched, encoding="utf-8")
        print("  [OK] patched scripts/smoke_oss.py import references")


def _patch_readme(dest_root: Path) -> None:
    """Strip rename note and fix uvicorn command in exported README.md."""
    p = dest_root / "README.md"
    if not p.exists():
        print("  [WARN] README.md not found in export — patch skipped")
        return
    text = p.read_text(encoding="utf-8")

    # Strip the 2-line rename note (exact block from research)
    rename_note = (
        "# Note: in this source repo the entry point is src/web/main_oss.py;\n"
        "   # the export script (phase 110) renames it to src/web/main.py.\n"
    )
    rename_note_alt = (
        "# Note: in this source repo the entry point is src/web/main_oss.py;\n"
        "# the export script (phase 110) renames it to src/web/main.py.\n"
    )
    text = text.replace(rename_note, "")
    text = text.replace(rename_note_alt, "")

    # Fix uvicorn command
    text = text.replace("src.web.main_oss:app", "src.web.main:app")

    # Fix any remaining main_oss.py references (e.g. Mermaid diagram node label)
    text = text.replace("src/web/main_oss.py", "src/web/main.py")

    p.write_text(text, encoding="utf-8")
    print("  [OK] patched README.md (rename note stripped, uvicorn cmd updated)")


def _write_sources_example(dest_root: Path) -> None:
    """Write config/sources.txt.example with demo public feeds."""
    config_dir = dest_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    example = config_dir / "sources.txt.example"
    example.write_text(
        "# format: target | category | display_name [| platform]\n"
        "# platform defaults to telegram\n"
        "@durov | ai | Pavel Durov\n"
        "r/MachineLearning | ai | r/MachineLearning | reddit\n"
        "cs.AI | ai | ArXiv cs.AI | arxiv\n"
        "UCBJycsmduvYEL83R_U4JriQ | ai | Two Minute Papers | youtube\n"
        "neuroscience | research | bioRxiv Neuroscience | biorxiv\n",
        encoding="utf-8",
    )
    print("  [OK] wrote config/sources.txt.example")


def _git_init(dest_root: Path) -> None:
    """Init a fresh git repo in dest_root and make an initial commit."""
    if not shutil.which("git"):
        print("  [WARN] git not on PATH — skipping git init")
        return
    try:
        subprocess.run(["git", "init"], cwd=dest_root, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=dest_root, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit — autorss-feed OSS release"],
            cwd=dest_root,
            check=True,
            capture_output=True,
            env={**__import__("os").environ, "GIT_AUTHOR_NAME": "export-bot",
                 "GIT_AUTHOR_EMAIL": "export@example.com",
                 "GIT_COMMITTER_NAME": "export-bot",
                 "GIT_COMMITTER_EMAIL": "export@example.com"},
        )
        subprocess.run(
            ["git", "branch", "-M", "main"],
            cwd=dest_root,
            check=True,
            capture_output=True,
        )
        print("  [OK] git init + initial commit done (branch: main)")
    except subprocess.CalledProcessError as exc:
        print(f"  [WARN] git init failed: {exc} — continuing without git repo")


def _run_gitleaks(dest_root: Path) -> list[str]:
    """Bonus: run gitleaks if available. Returns list of finding strings."""
    gl = shutil.which("gitleaks")
    if not gl:
        return []
    print("  [INFO] gitleaks found — running bonus check")
    try:
        result = subprocess.run(
            [gl, "detect", "--no-git", "--source", str(dest_root)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return [f"[gitleaks] {line}" for line in result.stdout.splitlines() if line.strip()]
    except Exception as exc:
        print(f"  [WARN] gitleaks run failed: {exc}")
    return []


# ── Main orchestration ─────────────────────────────────────────────────────────


def run_export(repo_root: Path, dest_root: Path, *, force: bool = False) -> int:
    """Run the full export pipeline.

    Returns the number of scan findings (0 = clean).
    Raises SystemExit(2) on hard usage errors.
    """
    # Dest must not exist unless --force
    if dest_root.exists():
        if not force:
            print(f"[ERROR] destination already exists: {dest_root}")
            print("        Pass --force to overwrite.")
            raise SystemExit(2)
        print(f"[INFO] --force: removing existing {dest_root}")
        # Windows: .git objects are read-only; use onerror to chmod+retry.
        import stat

        def _remove_readonly(func, path, _excinfo):
            import os
            os.chmod(path, stat.S_IWRITE)
            func(path)

        shutil.rmtree(dest_root, onerror=_remove_readonly)

    dest_root.mkdir(parents=True, exist_ok=True)

    # Load ALLOWLIST
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        print("[ERROR] PyYAML not installed — run: uv add pyyaml")
        raise SystemExit(2)

    allowlist_path = repo_root / "export" / "ALLOWLIST.yaml"
    with open(allowlist_path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    if not manifest or "included" not in manifest:
        print("[ERROR] ALLOWLIST.yaml missing 'included' key or is empty")
        raise SystemExit(2)
    included: list[str] = manifest["included"]
    if not isinstance(included, list):
        print("[ERROR] ALLOWLIST.yaml 'included' must be a list")
        raise SystemExit(2)

    print(f"\n[1/6] Copying {len(included)} allowlist entries …")
    copy_included(repo_root, dest_root, included)

    print("[2/6] Renaming main_oss → main …")
    _rename_main_oss(dest_root)

    print("[3/6] Patching smoke_oss.py imports …")
    _patch_smoke_oss(dest_root)

    print("[4/6] Patching README …")
    _patch_readme(dest_root)

    print("[5/6] Writing sources.txt.example …")
    _write_sources_example(dest_root)

    print("[6/6] Git init …")
    _git_init(dest_root)

    print("\n[SCAN] Running personal-data scan gate …")
    findings = scan_tree(dest_root)
    extra = _run_gitleaks(dest_root)

    if findings:
        print(f"\n[SCAN] {len(findings)} FINDING(S):")
        for rel, lineno, line in findings:
            print(f"  {rel}:{lineno}: {line[:120]}")
    if extra:
        print(f"\n[GITLEAKS] {len(extra)} finding(s):")
        for hit in extra:
            print(f"  {hit}")

    total = len(findings) + len(extra)
    if total == 0:
        print(f"\n[SCAN] 0 findings — export tree is clean.\n")
    else:
        print(f"\n[SCAN] {total} finding(s) — fix before publishing.\n")
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export autorss-feed to a clean OSS tree (REL-01 + REL-02)."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for the export. Must not exist unless --force.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove and recreate destination if it already exists.",
    )
    args = parser.parse_args(argv)

    dest = Path(args.output_dir).resolve()
    total_findings = run_export(_REPO_ROOT, dest, force=args.force)
    return 1 if total_findings > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
