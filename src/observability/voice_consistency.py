"""Phase 46 (VOICE-02) — Editor Voice Consistency Check.

Compares the configured-persona editor digest output against generic-persona output
and reference voice samples from the configured profile dir.

Schema (auto-created):
    CREATE TABLE voice_consistency_snapshots (
        id INTEGER PRIMARY KEY,
        digest_id INTEGER NOT NULL,
        scored_at TEXT NOT NULL,
        digest_text TEXT,
        cosine_to_generic REAL,
        cosine_to_reference REAL,
        verdict TEXT NOT NULL,
        raw_scores_json TEXT,
        UNIQUE(digest_id, scored_at)
    )

Verdicts:
- PASS: cosine_to_generic <= 0.85 AND cosine_to_reference >= 0.6
- FAIL_TOO_GENERIC: cosine_to_generic > 0.85
- FAIL_TOO_DIFFERENT_FROM_REFERENCE: cosine_to_reference < 0.6
- skipped: Voice_Profile.md missing / Ollama unreachable / empty input

Fail-soft: never raises into caller.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

KARAGANDA_TZ = timezone(timedelta(hours=5))

REJECT_TO_GENERIC = 0.85
MIN_TO_REFERENCE = 0.6

# NOTE: VOICE_PROFILE_PATH is intentionally NOT bound at module level — see CLAUDE.md
# "Module-level env var binding" gotcha. Read inside _load_voice_profile() instead.
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def now_karaganda() -> datetime:
    return datetime.now(KARAGANDA_TZ)


def ensure_table(conn: sqlite3.Connection) -> None:
    """Create voice_consistency_snapshots table + index. Idempotent."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS voice_consistency_snapshots (
            id INTEGER PRIMARY KEY,
            digest_id INTEGER NOT NULL,
            scored_at TEXT NOT NULL,
            digest_text TEXT,
            cosine_to_generic REAL,
            cosine_to_reference REAL,
            verdict TEXT NOT NULL,
            raw_scores_json TEXT,
            UNIQUE(digest_id, scored_at)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_voice_consistency_digest_id "
        "ON voice_consistency_snapshots(digest_id)"
    )
    conn.commit()


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Manual cosine similarity. Returns 0.0 on degenerate inputs (fail-soft)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _classify_verdict(cos_generic: float, cos_reference: float) -> str:
    """Verdict from threshold comparison."""
    if cos_generic > REJECT_TO_GENERIC:
        return "FAIL_TOO_GENERIC"
    if cos_reference < MIN_TO_REFERENCE:
        return "FAIL_TOO_DIFFERENT_FROM_REFERENCE"
    return "PASS"


def _load_voice_profile(path: Path | None = None) -> str | None:
    """Read reference voice samples. Returns None if file missing.

    Path resolution order: explicit argument > VOICE_PROFILE_PATH env > "voice_profile.md".
    Read inside the function (not at module level) to allow env override in tests.
    """
    if path is None:
        path = Path(os.environ.get("VOICE_PROFILE_PATH", "voice_profile.md"))
    try:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _embed_sync(text: str, host: str = DEFAULT_OLLAMA_HOST, model: str = "bge-m3") -> list[float] | None:
    """Synchronous Ollama embed call. Returns None on any failure."""
    import urllib.request
    import urllib.error

    if not text:
        return None
    try:
        payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            f"{host}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            embedding = data.get("embedding")
            if isinstance(embedding, list) and embedding:
                return [float(x) for x in embedding]
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError, TypeError):
        pass
    return None


def score_digest(
    db_path: str,
    *,
    digest_id: int,
    digest_text: str,
    generic_text: str,
    voice_profile_path: Path | None = None,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    embed_fn=None,
    voice_profile_loader=None,
) -> dict:
    """Score a single digest. Records verdict to voice_consistency_snapshots.

    `embed_fn` injection enables testing without Ollama. `voice_profile_loader`
    is used so tests can avoid touching the real Obsidian path.

    Returns dict: {ok, verdict, cosine_to_generic, cosine_to_reference, error?}.
    """
    embed = embed_fn or _embed_sync
    loader = voice_profile_loader or (lambda: _load_voice_profile(voice_profile_path))

    if not digest_text or not generic_text:
        _record_skipped(db_path, digest_id, "empty_input")
        return {"ok": False, "verdict": "skipped", "error": "empty_input"}

    reference_text = loader()
    if not reference_text:
        _record_skipped(db_path, digest_id, "voice_profile_missing")
        return {"ok": False, "verdict": "skipped", "error": "voice_profile_missing"}

    emb_digest = embed(digest_text, host=ollama_host)
    emb_generic = embed(generic_text, host=ollama_host)
    emb_reference = embed(reference_text, host=ollama_host)
    if not emb_digest or not emb_generic or not emb_reference:
        _record_skipped(db_path, digest_id, "embed_failure")
        return {"ok": False, "verdict": "skipped", "error": "embed_failure"}

    cos_generic = _cosine_similarity(emb_digest, emb_generic)
    cos_reference = _cosine_similarity(emb_digest, emb_reference)
    verdict = _classify_verdict(cos_generic, cos_reference)

    try:
        conn = sqlite3.connect(db_path)
        ensure_table(conn)
        conn.execute(
            "INSERT OR IGNORE INTO voice_consistency_snapshots "
            "(digest_id, scored_at, digest_text, cosine_to_generic, "
            " cosine_to_reference, verdict, raw_scores_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                digest_id,
                now_karaganda().isoformat(),
                digest_text[:4000],  # truncation guard
                cos_generic,
                cos_reference,
                verdict,
                json.dumps({"cosine_to_generic": cos_generic, "cosine_to_reference": cos_reference}),
            ),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error:
        return {
            "ok": False,
            "verdict": "skipped",
            "error": "sqlite_error",
            "cosine_to_generic": cos_generic,
            "cosine_to_reference": cos_reference,
        }

    return {
        "ok": True,
        "verdict": verdict,
        "cosine_to_generic": cos_generic,
        "cosine_to_reference": cos_reference,
    }


def _record_skipped(db_path: str, digest_id: int, reason: str) -> None:
    """Best-effort skipped row write. Silent on failure."""
    try:
        conn = sqlite3.connect(db_path)
        ensure_table(conn)
        conn.execute(
            "INSERT OR IGNORE INTO voice_consistency_snapshots "
            "(digest_id, scored_at, verdict, raw_scores_json) "
            "VALUES (?, ?, ?, ?)",
            (digest_id, now_karaganda().isoformat(), "skipped", json.dumps({"reason": reason})),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error:
        pass


def count_recent_failures(db_path: str, *, hours: int = 24) -> int:
    """Count voice_consistency_snapshots with verdict starting with 'FAIL_' in last N hours."""
    try:
        cutoff = (now_karaganda() - timedelta(hours=hours)).isoformat()
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT COUNT(*) FROM voice_consistency_snapshots "
            "WHERE scored_at >= ? AND verdict LIKE 'FAIL_%'",
            (cutoff,),
        ).fetchone()
        conn.close()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


# --- CLI ---


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Voice consistency check (Phase 46)")
    parser.add_argument("--db", default="curator.db")
    parser.add_argument("--digest-id", type=int, required=False, help="Specific digest to score; otherwise score most recent")
    parser.add_argument("--generic-text", default=None, help="Generic-persona text for comparison. If omitted, a fixed placeholder is used (verdict not meaningful).")
    args = parser.parse_args()

    # Pull latest digest if no id given
    conn = sqlite3.connect(args.db)
    if args.digest_id:
        row = conn.execute("SELECT id, markdown_content FROM digests WHERE id = ?", (args.digest_id,)).fetchone()
    else:
        row = conn.execute("SELECT id, markdown_content FROM digests ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    if not row:
        print("No digest found.")
        return
    digest_id, markdown = row

    # Use supplied generic text, or a fixed placeholder for smoke-only mode.
    # Scoring against itself (identity) always yields FAIL_TOO_GENERIC — misleading.
    if args.generic_text:
        generic = args.generic_text
    else:
        generic = (
            "This is a generic placeholder digest with no personal voice. "
            "It covers recent developments in technology and research without "
            "any particular editorial style or perspective."
        )
        print(
            f"NOTE: --generic-text not supplied; using fixed placeholder. "
            f"Verdict reflects distance from placeholder, not a real generic digest. "
            f"Use --generic-text to supply actual generic-persona output."
        )

    result = score_digest(
        args.db,
        digest_id=digest_id,
        digest_text=markdown,
        generic_text=generic,
    )
    print(f"Digest {digest_id}: verdict={result.get('verdict')} "
          f"cosine_to_generic={result.get('cosine_to_generic')} "
          f"cosine_to_reference={result.get('cosine_to_reference')}")


if __name__ == "__main__":
    main()
