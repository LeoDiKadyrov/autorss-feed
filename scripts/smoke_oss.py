"""OSS smoke test — PKG-03.

Validates the open-source path end-to-end:
  seed → curate-only pipeline → web entry-point (GET / + GET /chat)

Usage:
    .venv/Scripts/python.exe scripts/smoke_oss.py

Exits 0 on success, 1 on any hard failure.
Pipeline subprocess failures (e.g. Ollama absent in CI) are soft warnings —
the web-probe is the load-bearing assertion.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Self-contained imports: scripts/ runs without PYTHONPATH=. (documented gotcha) —
# bootstrap the repo root onto sys.path before any src.* import.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Windows console defaults to cp1252 — non-ASCII output raises UnicodeEncodeError
# (documented gotcha). Reconfigure stdout/stderr to UTF-8 with replacement.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


# ── helpers ────────────────────────────────────────────────────────────────


def _check(condition: bool, label: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


# ── main ───────────────────────────────────────────────────────────────────


async def _probe_web(db_path: str) -> bool:
    """Import main_oss app and probe GET / and GET /chat."""
    import httpx

    os.environ["DB_PATH"] = db_path

    # Import after env is set so DB_PATH module-level binding picks it up.
    from src.web.main import app  # noqa: PLC0415

    import aiosqlite
    from src.database.client import init_db

    async with aiosqlite.connect(db_path) as db:
        await init_db(db)

    ok = True
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp_root = await ac.get("/")
        ok &= _check(resp_root.status_code == 200, f"GET / → {resp_root.status_code}")

        resp_chat = await ac.get("/chat")
        # Empty corpus → graceful message acceptable; 200 is the smoke-pass condition.
        ok &= _check(resp_chat.status_code == 200, f"GET /chat → {resp_chat.status_code}")

    return ok


def main() -> int:
    print("=== autorss_feed OSS smoke test (PKG-03) ===\n")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        # ── Step 1: curate-only pipeline with seeded test posts ──────────────
        print("Step 1: run_pipeline.py --mode curate-only --seed-test-data")
        env = {**os.environ, "DB_PATH": db_path, "PYTHONPATH": "."}
        proc = subprocess.run(
            [
                sys.executable,
                "run_pipeline.py",
                "--mode",
                "curate-only",
                "--seed-test-data",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(
                f"  [WARN] pipeline exited {proc.returncode} "
                "(Ollama may be absent — continuing to web probe)\n"
                f"  stdout: {proc.stdout[-300:] if proc.stdout else ''}\n"
                f"  stderr: {proc.stderr[-300:] if proc.stderr else ''}"
            )
        else:
            print("  [PASS] pipeline exited 0")

        # ── Step 2: web entry-point probe ────────────────────────────────────
        print("\nStep 2: probe src.web.main app (GET / + GET /chat)")
        web_ok = asyncio.run(_probe_web(db_path))

        # ── Result ────────────────────────────────────────────────────────────
        print()
        if web_ok:
            print("=== SMOKE PASS ===")
            return 0
        else:
            print("=== SMOKE FAIL ===")
            return 1

    finally:
        try:
            Path(db_path).unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
