"""Phase 47 (VOICE-03) — Persona Leak Detector in Drafts.

Scans a draft's final text for persona-leak signals before finish_worker
promotes to `20_Evergreen`. Two-tier detector:
  1. Regex catalogue — fast, catches obvious ChatGPT-isms (EN + RU)
  2. Optional bge-m3 cosine vs known-leak corpus — catches paraphrases

Score formula:
    score = clamp(regex_hits / MAX_REGEX_HITS_CAP, 0, 1) * 0.6
          + clamp(cos_to_leak_corpus, 0, 1) * 0.4

Verdict:
- BLOCK if score > threshold (default 0.3, env PERSONA_LEAK_THRESHOLD)
- PASS otherwise
- skipped on missing draft / unexpected failure

Schema (auto-created):
    CREATE TABLE persona_leak_snapshots (
        id INTEGER PRIMARY KEY,
        draft_path TEXT NOT NULL,
        scored_at TEXT NOT NULL,
        score REAL,
        regex_hits INTEGER,
        cosine_to_leak_corpus REAL,
        verdict TEXT NOT NULL,
        raw_matches_json TEXT,
        UNIQUE(draft_path, scored_at)
    )

Fail-soft: never raises into caller.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

KARAGANDA_TZ = timezone(timedelta(hours=5))

# Tunable threshold — env override
DEFAULT_THRESHOLD = 0.3
MAX_REGEX_HITS_CAP = 5  # ≥5 hits saturates the regex tier
REGEX_WEIGHT = 0.6
EMBED_WEIGHT = 0.4

# Optional bge-m3 corpus (override via LEAK_CORPUS_PATH env var)
import os as _os
LEAK_CORPUS_PATH = Path(_os.environ.get("LEAK_CORPUS_PATH", "persona_leaks.md"))
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Leak phrase regex catalogue (case-insensitive, multi-line)
# EN + RU ChatGPT-isms. Add more as discovered in practice.
LEAK_PATTERNS = [
    r"\bas an? AI(?:\s+language\s+model)?\b",
    r"\bi(?:'m| am) (?:happy|glad|delighted)\s+to\s+help\b",
    r"\bi(?:'m| am) sorry,?\s+(?:but\s+)?i (?:cannot|can't|won't|am unable)\b",
    r"\bi (?:cannot|can't|won't|am unable)\s+(?:assist|help|provide|answer)\b",
    r"\bcertainly[!.]? here\s+(?:is|are)\b",
    r"\bof course[!.]? (?:i'd|i would|let me)\b",
    r"\bi (?:don't|do not) have (?:personal|access to real-time)\b",
    r"\bplease (?:note|be aware|keep in mind) that\b",
    r"\bit's (?:important|worth) (?:to note|noting|mentioning) that\b",
    # Russian
    r"\bкак (?:большая |)языковая модель\b",
    r"\bя (?:не могу|не имею)\s+(?:доступа|возможности|право)\b",
    r"\bя (?:рад|рада|с радостью)\s+пом(?:очь|огу)\b",
    r"\bконечно[!,]\s+вот\b",
    r"\bобратите внимание[, ]\s*что\b",
    r"\bстоит (?:отметить|упомянуть)[, ]\s*что\b",
]
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in LEAK_PATTERNS]


def now_karaganda() -> datetime:
    return datetime.now(KARAGANDA_TZ)


def ensure_table(conn: sqlite3.Connection) -> None:
    """Create persona_leak_snapshots table + index. Idempotent."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS persona_leak_snapshots (
            id INTEGER PRIMARY KEY,
            draft_path TEXT NOT NULL,
            scored_at TEXT NOT NULL,
            score REAL,
            regex_hits INTEGER,
            cosine_to_leak_corpus REAL,
            verdict TEXT NOT NULL,
            raw_matches_json TEXT,
            UNIQUE(draft_path, scored_at)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_persona_leak_draft_path "
        "ON persona_leak_snapshots(draft_path)"
    )
    conn.commit()


def _count_regex_hits(text: str) -> tuple[int, list[str]]:
    """Return (hit_count, matched_phrases). Each pattern counted once max."""
    matches: list[str] = []
    for pat in _COMPILED_PATTERNS:
        m = pat.search(text)
        if m:
            matches.append(m.group(0))
    return len(matches), matches


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Manual cosine. Mirrors P46 voice_consistency._cosine_similarity for consistency."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _load_leak_corpus(path: Path = LEAK_CORPUS_PATH) -> str | None:
    """Read leak phrase corpus. Returns None if file missing."""
    try:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _embed_sync(text: str, host: str = DEFAULT_OLLAMA_HOST, model: str = "bge-m3") -> list[float] | None:
    """Synchronous embed call. Returns None on any failure (mirrors P46)."""
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


def _compute_score(regex_hit_count: int, cosine_to_corpus: float) -> float:
    """Weighted score in [0, 1]."""
    regex_norm = min(regex_hit_count / MAX_REGEX_HITS_CAP, 1.0)
    embed_norm = max(0.0, min(cosine_to_corpus, 1.0))
    return regex_norm * REGEX_WEIGHT + embed_norm * EMBED_WEIGHT


def _get_threshold() -> float:
    """Threshold via PERSONA_LEAK_THRESHOLD env or default."""
    try:
        return float(os.environ.get("PERSONA_LEAK_THRESHOLD", str(DEFAULT_THRESHOLD)))
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD


def check_text(
    text: str,
    *,
    embed_fn=None,
    corpus_loader=None,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
) -> dict:
    """Score raw text for persona leaks. Returns dict (does NOT write DB).

    Use `check_draft` for the file + DB-write entry-point.
    """
    embed = embed_fn or _embed_sync
    loader = corpus_loader or (lambda: _load_leak_corpus())

    if not text:
        return {
            "score": 0.0,
            "regex_hits": 0,
            "matches": [],
            "cosine_to_leak_corpus": 0.0,
            "verdict": "PASS",
            "threshold": _get_threshold(),
        }

    hit_count, matches = _count_regex_hits(text)

    corpus_text = loader()
    cosine = 0.0
    if corpus_text:
        emb_text = embed(text, host=ollama_host)
        emb_corpus = embed(corpus_text, host=ollama_host)
        if emb_text and emb_corpus:
            cosine = _cosine_similarity(emb_text, emb_corpus)

    score = _compute_score(hit_count, cosine)
    threshold = _get_threshold()
    verdict = "BLOCK" if score > threshold else "PASS"

    return {
        "score": score,
        "regex_hits": hit_count,
        "matches": matches,
        "cosine_to_leak_corpus": cosine,
        "verdict": verdict,
        "threshold": threshold,
    }


def check_draft(
    db_path: str,
    draft_path: str,
    *,
    embed_fn=None,
    corpus_loader=None,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
) -> dict:
    """Score a draft file + persist snapshot. Returns same dict as check_text.

    Fail-soft: missing file → verdict='skipped'.
    """
    try:
        p = Path(draft_path)
        if not p.exists():
            _record_skipped(db_path, draft_path, "draft_missing")
            return {"verdict": "skipped", "error": "draft_missing"}
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        _record_skipped(db_path, draft_path, "read_failure")
        return {"verdict": "skipped", "error": "read_failure"}

    result = check_text(text, embed_fn=embed_fn, corpus_loader=corpus_loader, ollama_host=ollama_host)
    try:
        conn = sqlite3.connect(db_path)
        ensure_table(conn)
        conn.execute(
            "INSERT OR IGNORE INTO persona_leak_snapshots "
            "(draft_path, scored_at, score, regex_hits, cosine_to_leak_corpus, "
            " verdict, raw_matches_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                draft_path,
                now_karaganda().isoformat(),
                result["score"],
                result["regex_hits"],
                result["cosine_to_leak_corpus"],
                result["verdict"],
                json.dumps(result["matches"]),
            ),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error:
        pass
    return result


def _record_skipped(db_path: str, draft_path: str, reason: str) -> None:
    """Best-effort skipped row."""
    try:
        conn = sqlite3.connect(db_path)
        ensure_table(conn)
        conn.execute(
            "INSERT OR IGNORE INTO persona_leak_snapshots "
            "(draft_path, scored_at, verdict, raw_matches_json) "
            "VALUES (?, ?, ?, ?)",
            (draft_path, now_karaganda().isoformat(), "skipped",
             json.dumps({"reason": reason})),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error:
        pass


def count_recent_blocks(db_path: str, *, hours: int = 24) -> int:
    """Count BLOCK verdicts in last N hours."""
    try:
        cutoff = (now_karaganda() - timedelta(hours=hours)).isoformat()
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT COUNT(*) FROM persona_leak_snapshots "
            "WHERE scored_at >= ? AND verdict = 'BLOCK'",
            (cutoff,),
        ).fetchone()
        conn.close()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


# --- CLI ---


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Persona leak detector (Phase 47)")
    parser.add_argument("--db", default="curator.db")
    parser.add_argument("--draft", required=True)
    args = parser.parse_args()

    result = check_draft(args.db, args.draft)
    print(f"Draft: {args.draft}")
    print(f"Verdict: {result.get('verdict')}")
    print(f"Score: {result.get('score', 0):.3f} (threshold {result.get('threshold', DEFAULT_THRESHOLD)})")
    print(f"Regex hits: {result.get('regex_hits', 0)} {result.get('matches', [])}")
    print(f"Cosine to leak corpus: {result.get('cosine_to_leak_corpus', 0):.3f}")


if __name__ == "__main__":
    main()
