import aiosqlite
import datetime
import logging
import os
import sqlite3
from pathlib import Path
from urllib.parse import urlparse, urlunparse, urlencode, parse_qsl

import numpy as np
from simhash import Simhash

# Phase 12 / TOPIC-02: defense-in-depth — re-apply allow-list at DB layer
# (T-12-03). Imported as a constant from src.llm.ollama; pure data, no I/O,
# so the DB→llm direction is acceptable.
from src.llm.ollama import ALLOWED_TOPICS

# Phase 65 / DEDUP-BGE-01: bge-m3 embedding wrapper + dedup_rejected counter.
# Direction DB→llm acceptable for same reason as ALLOWED_TOPICS above (pure I/O
# wrapper, no schema dependency). insert_raw_post calls get_embedding before
# the existing simhash dedup block; None return → fall through to simhash.
from src.llm.embedding import get_embedding, increment_dedup_rejected

# Phase 27 / NB-01 + NB-03 + SIG-02 (D-04, D-08, D-09): write-time cognition
# tagging plumbing. COGNITION_CSV_DIR is module-level so tests can monkeypatch
# the lookup path; production reads from <repo>/data/neuroband/*.csv.
_REPO_ROOT = Path(__file__).resolve().parents[2]
COGNITION_CSV_DIR: Path = _REPO_ROOT / "data" / "neuroband"

# Module-level counters tracking write-time cognition outcomes since the last
# pipeline-run boundary (or test reset). Plan 27-06 consumes via accessor for
# the `cognition_tagged=N` OK log field (SIG-02).
_reactions_with_cognition = 0
_reactions_without_cognition = 0

_cog_logger = logging.getLogger(__name__)


def get_cognition_counters() -> tuple[int, int]:
    """Return ``(tagged, untagged)`` write-time cognition counters.

    ``tagged`` = reactions whose ±10 min NFB window had ≥3 samples per
    cog key (cog_* columns populated post-insert).
    ``untagged`` = reactions whose window was empty / sub-threshold OR whose
    cognition compute raised (logged + swallowed). Counters are process-local;
    Plan 27-06 emits + resets at the pipeline-run boundary.
    """
    return _reactions_with_cognition, _reactions_without_cognition


def reset_cognition_counters() -> None:
    """Reset both cognition counters to 0.

    Test hook + pipeline-run boundary reset (Plan 27-06 will call this after
    emitting the OK log field).
    """
    global _reactions_with_cognition, _reactions_without_cognition
    _reactions_with_cognition = 0
    _reactions_without_cognition = 0

# Phase 13 DEDUP-04/05: char-3 n-gram tokenizer + tuned Hamming threshold.
# Plan 01 (scripts/tune_simhash_threshold.py + tests/fixtures/dedup_corpus.json)
# evidentially validated F1 = 0.89 at threshold = 5 on a 50-pair labeled corpus
# (D-08's "= 3" target was an estimate; D-07 + D-13 mandate F1-driven empirical
# selection — threshold = 3 yields F1 = 0.78, below the 0.85 bar).
# CURRENT_SIMHASH_VERSION bump invalidates v1-default rows; cross-version compares
# are refused at query level (Phase 9 invariant) so existing rows degrade
# gracefully until scripts/recompute_simhash.py runs as a post-deploy step (D-17).
CURRENT_SIMHASH_VERSION = "v1.1-char-3-gram"
HAMMING_THRESHOLD = 5  # D-08 (relaxed): F1-tuned vs labeled corpus; replaces v1.0 literal `<= 6`


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    """Char-n-gram tokenizer for Simhash (D-01, D-02).

    Pre-tokenize: lowercase, collapse internal whitespace runs to single space,
    strip leading/trailing whitespace. NO punctuation strip (preserves "$10K" /
    ".com" / "@user" markers). Empty input -> []. Shorter than n -> single
    token (the normalized text) so Simhash always has at least 1 feature.

    Pure function: no I/O, fully deterministic. Body is identical to
    scripts/tune_simhash_threshold.py:_char_ngrams (Plan 01) by design — the
    F1 tuning result transfers iff the production tokenizer matches.
    """
    if not text:
        return []
    normalized = " ".join(text.lower().split())
    if not normalized:
        return []
    if len(normalized) < n:
        return [normalized]
    return [normalized[i:i + n] for i in range(len(normalized) - n + 1)]


def normalize_url(url: str | None) -> str | None:
    """
    Strip utm_* tracking params and trailing slash from path.
    Returns None if url is None or empty string.
    Conservative: only removes utm_* — does not reorder remaining params or lowercase domain.
    Malformed URLs are returned as-is (INSERT OR IGNORE handles dedup failure gracefully).
    """
    if not url:
        return None
    try:
        parsed = urlparse(url)
        # Filter out utm_* query params only
        clean_params = [
            (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if not k.startswith("utm_")
        ]
        clean_query = urlencode(clean_params)
        # Strip trailing slash from path, but preserve root "/"
        clean_path = parsed.path.rstrip("/") or "/"
        normalized = urlunparse(parsed._replace(query=clean_query, path=clean_path))
        return normalized
    except Exception:
        return url  # malformed URL — return as-is

async def init_db(db: aiosqlite.Connection):
    # Phase 11 EMAIL-04 / D-01: per-account IMAP credentials at end of column list.
    # account_pass_env stores the env-var NAME (e.g. "EMAIL_PASS_PERSONAL"), never
    # the secret value (D-02). NULL columns fall back to legacy EMAIL_* env vars
    # (D-03 backward-compat). Denormalized on sources, NOT a separate FK table (D-04).
    await db.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY, platform TEXT, target TEXT, is_active BOOLEAN,
            last_fetched_at TEXT, last_message_id INTEGER,
            category TEXT DEFAULT 'other', display_name TEXT,
            account_host TEXT, account_port INTEGER,
            account_user TEXT, account_pass_env TEXT
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS raw_posts (
            id INTEGER PRIMARY KEY, source_id INTEGER, external_id TEXT UNIQUE,
            raw_text TEXT, author TEXT, url TEXT, published_at TEXT, status TEXT,
            platform TEXT, content_type TEXT, canonical_url TEXT,
            simhash INTEGER, source_score INTEGER DEFAULT 0,
            simhash_version TEXT NOT NULL DEFAULT 'v1-default',
            extracted_body TEXT, extraction_status TEXT, extracted_at TEXT,
            embedding BLOB,
            email_class TEXT
        )
    """)
    await db.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_canonical_url "
        "ON raw_posts(canonical_url) WHERE canonical_url IS NOT NULL"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_simhash_version ON raw_posts(simhash_version)"
    )
    await db.execute("""
        CREATE TABLE IF NOT EXISTS curation_logs (
            id INTEGER PRIMARY KEY,
            post_id INTEGER REFERENCES raw_posts(id),
            relevance_score INTEGER,
            reason TEXT,
            scored_at TEXT,
            suspect_flag INTEGER NOT NULL DEFAULT 0,
            profile_hash TEXT,
            stage TEXT,
            backend TEXT,
            cohort TEXT,
            tokens_in INTEGER,
            tokens_out INTEGER
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS digests (
            id INTEGER PRIMARY KEY, created_at TEXT, markdown_content TEXT,
            backend TEXT DEFAULT 'ollama'
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS backend_preference (
            id INTEGER PRIMARY KEY,
            digest_id_a INTEGER NOT NULL,
            digest_id_b INTEGER NOT NULL,
            winner INTEGER NOT NULL CHECK(winner IN (0, 1)),
            voted_at TEXT NOT NULL
        )
    """)
    # --- post_feedback table (Phase 5 / FB-01) ---
    # D-01: locked schema; D-02: UNIQUE index on post_id supports INSERT OR REPLACE.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS post_feedback (
            id INTEGER PRIMARY KEY,
            post_id INTEGER NOT NULL,
            rating INTEGER CHECK(rating IN (-1, 1)),
            rated_at TEXT NOT NULL,
            FOREIGN KEY(post_id) REFERENCES raw_posts(id)
        )
    """)
    await db.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_post_feedback_post_id "
        "ON post_feedback(post_id)"
    )
    # --- post_topics junction table (Phase 12 / TOPIC-02) ---
    # D-01: per-post topic tags 1-to-many; UNIQUE(post_id, topic) prevents
    # duplicate tag rows on re-runs. D-02: 2 indexes for digest JOINs and
    # topic-filter queries. No CHECK constraint on topic — allow-list filter
    # lives in insert_post_topics() Python layer (D-03: tags evolve via code,
    # not schema migrations).
    await db.execute("""
        CREATE TABLE IF NOT EXISTS post_topics (
            id INTEGER PRIMARY KEY,
            post_id INTEGER NOT NULL,
            topic TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            UNIQUE(post_id, topic),
            FOREIGN KEY(post_id) REFERENCES raw_posts(id)
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_post_topics_post_id "
        "ON post_topics(post_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_post_topics_topic "
        "ON post_topics(topic)"
    )
    # Phase 15 / REACT-03: reactions table — single-row-per-chain for finish-mode.
    # G-1: item_id = raw_posts.id (already-deduped). No CHECK on status/action enums
    # (matches existing convention — Python layer enforces, schema stays string-typed).
    # status enum: pending|processing|drafted|reviewed|archived|deferred|killed|failed.
    # action enum: brainstorm|docs|link|finish|skip.
    # Phase 20 / WORKER-06 + GATE-04: chain_state TEXT appended after error_msg
    # for finish-chain state transitions (iter1_pending|iter1_drafted|iter1_continued|
    # iter2_*|iter3_*|killed|deferred|NULL-for-non-finish). Position-at-end so existing
    # INSERT statements (insert_reaction L696-700) keep working without column-list changes.
    # Phase 27 / NB-02 (D-05, D-06, D-07): 5 REAL NULL cog_* columns capture
    # per-reaction NFB cognition window means (concentration/fatigue/relaxation/
    # alpha/beta). NULL = "no overlapping NFB samples in +/-10 min window" — never
    # sentinel. Position-at-end same reason as chain_state. PRAGMA-guarded ALTER
    # path below handles legacy DBs missing these columns.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS reactions (
            id INTEGER PRIMARY KEY,
            item_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            draft_path TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            current_iter INTEGER NOT NULL DEFAULT 1,
            total_iters INTEGER NOT NULL DEFAULT 1,
            error_msg TEXT,
            chain_state TEXT,
            cog_concentration REAL,
            cog_fatigue REAL,
            cog_relaxation REAL,
            cog_alpha REAL,
            cog_beta REAL,
            entry_md_hash TEXT,
            FOREIGN KEY(item_id) REFERENCES raw_posts(id)
        )
    """)
    # REACT-04: partial UNIQUE — repeat clicks during pending/processing dedup;
    # re-click after archived/killed/etc. inserts a fresh row. Mirrors
    # idx_canonical_url precedent at line 95.
    await db.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_reactions_active "
        "ON reactions(item_id, action) "
        "WHERE status IN ('pending','processing')"
    )
    # Phase 20 / WORKER-06 + GATE-04: index on chain_state for fast COUNT/SELECT
    # WHERE chain_state=? in atomic UPDATE-with-guard transitions (plan 20-02+).
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_reactions_chain_state "
        "ON reactions(chain_state)"
    )
    # REACT-06: digest_items cache — worker reads from this instead of re-parsing
    # markdown. Rows written inside the same transaction as the digests INSERT
    # (Plan 15-03 wires the editor write path).
    # Idea 4 (Attention Schema): saliency_* columns store three-slot attention
    # reason per entry (novelty/relevance/urgency — see src/llm/saliency.py).
    await db.execute("""
        CREATE TABLE IF NOT EXISTS digest_items (
            item_id INTEGER NOT NULL,
            digest_id INTEGER NOT NULL,
            post_id INTEGER NOT NULL,
            channel TEXT,
            url TEXT,
            snippet TEXT,
            linked_project TEXT,
            saliency_novelty TEXT,
            saliency_relevance TEXT,
            saliency_urgency TEXT,
            PRIMARY KEY (item_id, digest_id),
            FOREIGN KEY(digest_id) REFERENCES digests(id),
            FOREIGN KEY(post_id) REFERENCES raw_posts(id)
        )
    """)
    # Idea 3 (Predictive coding): per-category feedback weight priors.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS feedback_priors (
            category TEXT PRIMARY KEY,
            weight REAL NOT NULL DEFAULT 1.0,
            updated_at TEXT NOT NULL
        )
    """)
    # Idea 5 (HOT meta-digest): weekly pattern analysis output.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS weekly_meta (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            week_start TEXT,
            markdown_content TEXT
        )
    """)
    # Task T1: cost_snapshots table for telemetry tracking
    # Phase 42 (999.6) adds token_breakdown_json (idempotent ALTER in init_db below).
    # Phase 92 / POS-BIAS-01 adds position_bias_delta (mirrored in migrate_db).
    await db.execute("""
        CREATE TABLE IF NOT EXISTS cost_snapshots (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            backend TEXT NOT NULL,
            calls_7d INTEGER NOT NULL,
            est_tokens_7d INTEGER NOT NULL,
            anomaly_flag INTEGER NOT NULL DEFAULT 0,
            analysis_md TEXT,
            token_breakdown_json TEXT,
            position_bias_delta REAL,
            cluster_propagated_n INTEGER DEFAULT 0,
            cluster_floor_skipped_n INTEGER DEFAULT 0
        )
    """)
    # Phase 69 / DRIFT-DET-01: continual-learning drift detector snapshots.
    # One run writes N per-post rows (is_aggregate=0) + 1 aggregate row
    # (is_aggregate=1, post_id NULL, is_flagged 0/1). Mirrors cost_snapshots
    # late-add idempotent pattern; schema also created in migrate_db below.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS drift_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_at TEXT NOT NULL,
            post_id INTEGER,
            original_score INTEGER,
            current_score INTEGER,
            delta_abs REAL,
            backend TEXT,
            is_flagged INTEGER DEFAULT 0,
            is_aggregate INTEGER DEFAULT 0,
            aggregate_mean REAL,
            aggregate_std REAL,
            prompt_hash TEXT
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_drift_snapshot_at ON drift_snapshots(snapshot_at)"
    )
    # Brainstorm-review pipeline Phase 2 (2026-05-18):
    # one row per evaluated brainstorm idea. UNIQUE(draft_id,idea_num,review_run_id)
    # prevents duplicate rows when a weekly run re-evaluates the same pending draft.
    # user_decision stays NULL until the dashboard widget records promote/drop/spike.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS brainstorm_decisions (
            id INTEGER PRIMARY KEY,
            draft_id INTEGER NOT NULL,
            idea_num INTEGER NOT NULL,
            review_run_id TEXT NOT NULL,
            project TEXT NOT NULL,
            title TEXT,
            verdict TEXT NOT NULL CHECK(verdict IN ('fit','skip','duplicate','needs-spike')),
            score INTEGER NOT NULL CHECK(score BETWEEN 0 AND 10),
            summary_30w TEXT,
            risks_json TEXT,
            shipped_overlap TEXT,
            next_step TEXT,
            raw_json TEXT NOT NULL,
            user_decision TEXT CHECK(user_decision IS NULL OR user_decision IN ('promote','drop','spike')),
            decided_at TEXT,
            decision_target_path TEXT,
            migrated_to_roadmap TEXT,
            spiked_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(draft_id, idea_num, review_run_id)
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_brainstorm_pending "
        "ON brainstorm_decisions(verdict, score) "
        "WHERE user_decision IS NULL"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_brainstorm_run "
        "ON brainstorm_decisions(review_run_id)"
    )
    # Phase 78 / ROUTE-FEAT-01: routing_features. Mirrors migrate_db block.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS routing_features (
            post_id INTEGER PRIMARY KEY,
            post_length INTEGER,
            has_canonical_url INTEGER,
            simhash_dist INTEGER,
            embed_cosine REAL,
            source_category TEXT,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Phase 85 / DWELL-SCHEMA-01: passive dwell-time beacon signal.
    # New table — NOT a column on `reactions` — to avoid JOIN inflation when
    # the same post has multiple reaction rows (see MEMORY ref
    # `reference_join_inflation_rescore_chain_state.md`). Most posts have no
    # reaction row at all, so a side table is also more space-efficient.
    # UNIQUE(item_id, session_id) backs the UPSERT-MAX semantics in the
    # POST /api/dwell handler (see src/web/dwell.py).
    await db.execute("""
        CREATE TABLE IF NOT EXISTS post_dwell (
            id INTEGER PRIMARY KEY,
            item_id INTEGER NOT NULL,
            dwell_ms INTEGER NOT NULL CHECK(dwell_ms >= 0),
            session_id TEXT NOT NULL,
            recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(item_id, session_id)
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_post_dwell_item ON post_dwell(item_id)"
    )
    # Phase 101 / CHAMPION-HIST-01: champion_history — mirrors migrate_db block.
    # Tracks every curator prompt champion swap. variant_id NOT unique. demoted_at
    # NULL marks the currently-reigning champion. Written atomically by
    # src/eval/champion_gate.py::promote_champion before the prompts file os.replace.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS champion_history (
            id INTEGER PRIMARY KEY,
            variant_id TEXT NOT NULL,
            promoted_at TEXT NOT NULL,
            demoted_at TEXT,
            elo_at_promotion REAL NOT NULL,
            n_matches_at_promotion INTEGER NOT NULL
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_champion_history_variant ON champion_history(variant_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_champion_history_promoted_at ON champion_history(promoted_at)"
    )
    # Phase 86 / KL-SIM-01: source_similarity table for JSD redundancy detection.
    # Canonical a<b ordering enforced by CHECK; UNIQUE pair index backs the
    # INSERT OR REPLACE upsert in src/eval/source_similarity.py.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS source_similarity (
            source_a_id INTEGER NOT NULL,
            source_b_id INTEGER NOT NULL,
            jsd_divergence REAL NOT NULL,
            n_a INTEGER NOT NULL,
            n_b INTEGER NOT NULL,
            computed_at TEXT NOT NULL,
            UNIQUE(source_a_id, source_b_id),
            CHECK(source_a_id < source_b_id)
        )
    """)
    await db.commit()

async def insert_raw_post(
    db: aiosqlite.Connection,
    source_id: int,
    external_id: str,
    raw_text: str,
    author: str,
    url: str | None,
    platform: str | None = None,
    content_type: str | None = None,
    published_at: str | None = None,
    source_score: int = 0,
    extracted_body: str | None = None,        # Phase 10 EXTRACT-05
    extraction_status: str | None = None,     # Phase 10 EXTRACT-05
    extracted_at: str | None = None,          # Phase 10 EXTRACT-05
    email_class: str | None = None,           # Phase 96 IMAP-TRIAGE-01
) -> int | None:
    canonical = normalize_url(url)
    ts = published_at or datetime.datetime.now(datetime.UTC).isoformat()

    # Phase 65 / DEDUP-BGE-01: bge-m3 semantic dedup BEFORE simhash fallback.
    # Compute embedding once; None = backend unreachable → skip the bge-m3
    # branch entirely and fall through to the existing simhash path below.
    # Cosine threshold default 0.92, env-overridable. 7d sliding window.
    # BR-02: track pending bge-m3 reject so we can roll back the prior-row
    # UPDATE if the subsequent INSERT OR IGNORE skips on UNIQUE conflict.
    _bge_pending_reject_id: int | None = None
    _bge_pending_reject_prev_status: str | None = None

    new_emb = get_embedding(raw_text)
    if new_emb is not None:
        # Read threshold INSIDE function (module-level env binding gotcha per
        # reference_module_level_env_binding.md).
        try:
            cos_threshold = float(os.environ.get("DEDUP_COSINE_THRESHOLD", "0.92"))
        except (TypeError, ValueError):
            cos_threshold = 0.92
        # BR-01 (P52 TZ bug recurrence): collectors emit heterogeneous
        # published_at suffixes — telethon `+00:00`, feedparser RSS `Z`, some
        # legacy rows naive. SQLite string lex-compare against an ISO cutoff
        # mis-classifies `"…Z"` vs `"…+00:00"` (`'Z' > '+'` lexically). Pull
        # candidates with embeddings and filter by parsed datetime in Python so
        # all suffixes normalise to the same UTC instant.
        cutoff_7d_dt = datetime.datetime.now(datetime.UTC) - datetime.timedelta(
            days=7
        )
        async with db.execute(
            "SELECT id, embedding, source_score, published_at FROM raw_posts "
            "WHERE embedding IS NOT NULL"
        ) as sel:
            raw_candidates = await sel.fetchall()
        emb_candidates = []
        for row_id, blob, stored_score, pub_at in raw_candidates:
            if not pub_at:
                continue
            try:
                # Python 3.11+ fromisoformat handles trailing 'Z' natively.
                pub_dt = datetime.datetime.fromisoformat(pub_at)
            except (TypeError, ValueError):
                continue
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=datetime.UTC)
            if pub_dt >= cutoff_7d_dt:
                emb_candidates.append((row_id, blob, stored_score))
        new_norm = float(np.linalg.norm(new_emb))
        if new_norm > 0:
            for row_id, blob, stored_score in emb_candidates:
                if blob is None:
                    continue
                try:
                    # BR-03: explicit little-endian read mirrors write side.
                    if len(blob) != 4096:
                        continue
                    stored = np.frombuffer(blob, dtype="<f4")
                    stored_norm = float(np.linalg.norm(stored))
                    if stored_norm == 0 or stored.shape != new_emb.shape:
                        continue
                    cos = float(
                        np.dot(new_emb, stored) / (new_norm * stored_norm)
                    )
                except (ValueError, TypeError):
                    continue
                if cos >= cos_threshold:
                    # Source-score-aware tie-break mirrors the simhash block
                    # below (D-20-ish): new wins → reject existing, then
                    # continue to INSERT new; existing wins → drop new.
                    if source_score > (stored_score or 0):
                        # BR-02: transactional new-wins. The UPDATE that flips
                        # the prior row to 'rejected' MUST roll back if the
                        # subsequent INSERT OR IGNORE skips (UNIQUE conflict
                        # on canonical_url/external_id), else we orphan-reject
                        # a curated row with no replacement. Capture the prior
                        # status so we can restore it on conflict. Counter
                        # increments in BOTH branches (new-wins + existing-
                        # wins) — both are dedup-driven rejections per OK-log.
                        async with db.execute(
                            "SELECT status FROM raw_posts WHERE id = ?",
                            (row_id,),
                        ) as st_sel:
                            st_row = await st_sel.fetchone()
                        _bge_pending_reject_prev_status = (
                            st_row[0] if st_row else None
                        )
                        _bge_pending_reject_id = row_id
                        await db.execute(
                            "UPDATE raw_posts SET status = 'rejected' WHERE id = ?",
                            (row_id,),
                        )
                        increment_dedup_rejected()
                        break
                    else:
                        increment_dedup_rejected()
                        await db.commit()
                        return None

    # D-17: Compute simhash fingerprint (64-bit unsigned int from the simhash lib).
    # SQLite INTEGER is signed 64-bit, so values >= 2**63 raise OverflowError on insert.
    # Convert to signed two's-complement representation for storage; the 64-bit mask in the
    # XOR comparison below normalises stored values back to unsigned, so dedup math is invariant.
    #
    # Defensive: simhash 2.1.2 + numpy 2.x has a known uint8 overflow on highly repetitive
    # text where a single token's weight exceeds 255 (encountered with YouTube transcript
    # truncation tests, and possible in production with chant/code transcripts). When this
    # happens, skip dedup and store NULL simhash — INSERT OR IGNORE on canonical_url and
    # external_id still protects against exact duplicates.
    try:
        # Phase 13 DEDUP-04: char-3 n-gram tokenizer (D-01) replaces the simhash
        # library's default whitespace tokenizer. Positional list[str] arg —
        # the library branches on type internally; there is NO `features=` kwarg.
        new_hash = Simhash(_char_ngrams(raw_text, 3)).value
        new_hash_signed = (
            new_hash - (1 << 64) if new_hash >= (1 << 63) else new_hash
        )
    except (OverflowError, ValueError) as exc:
        new_hash = None
        new_hash_signed = None

    # D-18, D-21: Query last-24h posts with simhash for near-duplicate detection.
    # Skip the entire near-dup pass when our own simhash failed (new_hash is None) —
    # there is nothing to compare against and INSERT OR IGNORE still handles exact dups.
    candidates = []
    if new_hash is not None:
        cutoff = (
            datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=86400)
        ).isoformat()
        # Phase 9 DEDUP-03: filter to current simhash_version only — refuse cross-version compares
        async with db.execute(
            "SELECT id, simhash, source_score, status FROM raw_posts "
            "WHERE published_at >= ? AND simhash IS NOT NULL "
            "AND simhash_version = ?",
            (cutoff, CURRENT_SIMHASH_VERSION)
        ) as sel:
            candidates = await sel.fetchall()

    for row_id, stored_hash, stored_score, stored_status in candidates:
        # D-20: Hamming distance with 64-bit mask (Pitfall 4 — SQLite returns signed int)
        dist = bin(
            (stored_hash & 0xFFFFFFFFFFFFFFFF) ^ (new_hash & 0xFFFFFFFFFFFFFFFF)
        ).count('1')
        # Phase 13 DEDUP-05: HAMMING_THRESHOLD = 5 — F1-tuned (F1 = 0.89,
        # precision = 1.0, recall = 0.80) against the labeled corpus at
        # tests/fixtures/dedup_corpus.json (Plan 01). The v1.0 threshold was 6
        # because the library default whitespace tokenizer put year-swap pairs
        # at distance 5-8 with the short-string canary; the v1.1 char-3
        # tokenizer brings them to <= 5 directly. Distinct-topic pairs in the
        # corpus stay at distance >= 18 (no false positive at any threshold
        # <= 10). D-08 estimated 3 but threshold=3 yields F1=0.78 (below the
        # 0.85 bar) — empirical evidence supersedes per D-07 + D-13.
        if dist <= HAMMING_THRESHOLD:
            if source_score > (stored_score or 0):
                # New post wins — reject the existing duplicate
                await db.execute(
                    "UPDATE raw_posts SET status = 'rejected' WHERE id = ?", (row_id,)
                )
                break  # continue to INSERT the new post below
            else:
                # Existing post wins — discard the new post silently
                await db.commit()
                return None

    # Original INSERT OR IGNORE logic — extended with Phase 10 extraction columns.
    # 16 columns = 16 placeholders = 16 values. D-26: extracted_body never feeds simhash.
    # Phase 65 / DEDUP-BGE-01: persist bge-m3 embedding alongside the row.
    # NULL when get_embedding returned None (backend unreachable) — safe by
    # design: the bge-m3 dedup SELECT filters `WHERE embedding IS NOT NULL`.
    # BR-03: explicit little-endian dtype on serialise/deserialise so BLOBs
    # survive host moves (Windows→Linux deploys, future cross-arch). 1024
    # float32 = 4096 bytes invariant; assert before write.
    if new_emb is not None:
        emb_blob = np.asarray(new_emb, dtype="<f4").tobytes()
        assert len(emb_blob) == 4096, f"bge-m3 BLOB must be 4096 bytes, got {len(emb_blob)}"
    else:
        emb_blob = None
    cursor = await db.execute(
        "INSERT OR IGNORE INTO raw_posts "
        "(source_id, external_id, raw_text, author, url, published_at, status, "
        "platform, content_type, canonical_url, simhash, source_score, simhash_version, "
        "extracted_body, extraction_status, extracted_at, embedding, email_class) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (source_id, external_id, raw_text, author, url, ts, "unprocessed",
         platform, content_type, canonical, new_hash_signed, source_score,
         CURRENT_SIMHASH_VERSION,
         extracted_body, extraction_status, extracted_at, emb_blob, email_class)
    )
    await db.commit()
    # WR-01: cursor.lastrowid is unreliable for detecting INSERT OR IGNORE skips
    # — sqlite3 does not guarantee 0 on the IGNORE branch (may retain the previous
    # insert's rowid in the same connection). cursor.rowcount IS authoritative:
    # rowcount==0 means the row was not inserted (canonical_url OR external_id
    # collision), rowcount==1 means a fresh INSERT.
    #
    # When the IGNORE path fires, we have to find the existing row. The dup may
    # be on either the external_id UNIQUE constraint OR the canonical_url unique
    # index — search by both with OR to recover the id in either case.
    if cursor.rowcount == 0:
        # BR-02: INSERT skipped on UNIQUE conflict. If the bge-m3 tie-break
        # already flipped a prior row to 'rejected', revert it now — there is
        # no replacement row, so the rejection would be data loss. Also undo
        # the counter bump.
        if _bge_pending_reject_id is not None:
            await db.execute(
                "UPDATE raw_posts SET status = ? WHERE id = ?",
                (_bge_pending_reject_prev_status or "unprocessed",
                 _bge_pending_reject_id),
            )
            await db.commit()
            # Decrement counter to match restored state. Import locally to
            # avoid a top-level cycle (embedding imports nothing from here).
            from src.llm.embedding import _DEDUP_REJECTED  # noqa: F401
            import src.llm.embedding as _emb_mod
            if _emb_mod._DEDUP_REJECTED > 0:
                _emb_mod._DEDUP_REJECTED -= 1
        async with db.execute(
            "SELECT id FROM raw_posts "
            "WHERE external_id = ? OR (canonical_url IS NOT NULL AND canonical_url = ?)",
            (external_id, canonical),
        ) as sel:
            row = await sel.fetchone()
        return row[0] if row else None
    return cursor.lastrowid

async def get_unprocessed_posts(db: aiosqlite.Connection):
    """Return unprocessed raw_posts joined with the originating source's
    ``category`` column.

    Phase 70 WR-05: curator gate paths (snippet_quality_gate /
    giveaway_regex_gate) build factor-buckets that need the post's category
    to preserve the category axis of the composite key. ``raw_posts`` itself
    has no category column — it lives on ``sources`` — so we LEFT JOIN here.
    """
    db.row_factory = aiosqlite.Row
    async with db.execute(
        "SELECT rp.*, s.category AS category "
        "FROM raw_posts rp "
        "LEFT JOIN sources s ON s.id = rp.source_id "
        "WHERE rp.status = 'unprocessed'"
    ) as cursor:
        return [dict(row) for row in await cursor.fetchall()]

async def update_post_status(db: aiosqlite.Connection, post_id: int, status: str):
    await db.execute("UPDATE raw_posts SET status = ? WHERE id = ?", (status, post_id))
    await db.commit()

async def get_curated_posts(db: aiosqlite.Connection):
    """ME-04 fix: GROUP BY rp.id eliminates duplicate rows when multiple
    curation_logs exist per post (FB-01 backfill or repeat curate runs).
    Without this, the LEFT JOIN with curation_logs returns N rows per post and
    the editor's items list builds duplicate (item_id, digest_id) keys, which
    violates the digest_items PRIMARY KEY (Phase 15 / REACT-06).

    MAX(cl.relevance_score) picks the highest score across all logs for a post
    — this matches the de-facto "winning" score callers expected before the
    JOIN ever produced duplicates.
    """
    db.row_factory = aiosqlite.Row
    async with db.execute("""
        SELECT rp.id, rp.source_id, rp.external_id, rp.raw_text, rp.author,
               rp.url, rp.published_at, rp.status,
               MAX(cl.relevance_score) AS relevance_score,
               s.target AS source_target,
               COALESCE(s.display_name, s.target) AS source_display_name,
               COALESCE(s.category, 'other') AS source_category
        FROM raw_posts rp
        LEFT JOIN curation_logs cl ON cl.post_id = rp.id
        LEFT JOIN sources s ON s.id = rp.source_id
        WHERE rp.status = 'curated'
        GROUP BY rp.id
        ORDER BY source_category, relevance_score DESC
    """) as cursor:
        return [dict(row) for row in await cursor.fetchall()]

async def insert_digest(db: aiosqlite.Connection, content: str):
    now = datetime.datetime.now(datetime.UTC).isoformat()
    await db.execute("INSERT INTO digests (created_at, markdown_content) VALUES (?, ?)", (now, content))
    await db.execute("UPDATE raw_posts SET status = 'archived' WHERE status = 'curated'")
    await db.commit()


async def insert_digest_with_items(
    db: aiosqlite.Connection,
    content: str,
    items: list[dict],
    backend: str = "ollama",
) -> int:
    """Atomic digest creation + digest_items cache population.

    Phase 15 / REACT-06: editor calls this instead of insert_digest so the
    `digest_items` cache rows are written in the same transaction as the
    parent digest INSERT and the raw_posts archive UPDATE.

    Each `items` dict must contain: item_id, post_id, channel, url, snippet,
    linked_project (linked_project may be None — Phase 18 will populate it
    from "Связь: <project>" parsing).

    Atomicity (HI-01 fix): explicit BEGIN/COMMIT with try/rollback so partial
    failure (e.g. IntegrityError on digest_items PRIMARY KEY) cannot leave a
    `digests` row without its cache rows or with raw_posts in mixed state.
    On exception: rollback + re-raise. On success: commit.

    Returns the new digest's lastrowid.
    """
    now = datetime.datetime.now(datetime.UTC).isoformat()
    await db.execute("BEGIN")
    try:
        cursor = await db.execute(
            "INSERT INTO digests (created_at, markdown_content, backend) VALUES (?, ?, ?)",
            (now, content, backend),
        )
        digest_id = cursor.lastrowid
        if items:
            await db.executemany(
                "INSERT INTO digest_items "
                "(item_id, digest_id, post_id, channel, url, snippet, linked_project, "
                " saliency_novelty, saliency_relevance, saliency_urgency) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        it["item_id"],
                        digest_id,
                        it["post_id"],
                        it.get("channel"),
                        it.get("url"),
                        it.get("snippet"),
                        it.get("linked_project"),
                        it.get("saliency_novelty"),
                        it.get("saliency_relevance"),
                        it.get("saliency_urgency"),
                    )
                    for it in items
                ],
            )
        await db.execute("UPDATE raw_posts SET status = 'archived' WHERE status = 'curated'")
        await db.commit()
        return digest_id
    except Exception:
        await db.rollback()
        raise

async def get_digests(db: aiosqlite.Connection, limit: int = 1):
    db.row_factory = aiosqlite.Row
    async with db.execute(
        "SELECT * FROM digests ORDER BY id DESC LIMIT ?", (limit,)
    ) as cursor:
        return [dict(row) for row in await cursor.fetchall()]


async def get_digest_by_id(db: aiosqlite.Connection, digest_id: int) -> dict | None:
    db.row_factory = aiosqlite.Row
    async with db.execute("SELECT * FROM digests WHERE id = ?", (digest_id,)) as cur:
        row = await cur.fetchone()
    return dict(row) if row else None


async def get_digest_nav(db: aiosqlite.Connection, digest_id: int) -> dict:
    """Return prev_id (older), next_id (newer), total, position (1=newest) for digest_id."""
    async with db.execute("SELECT COUNT(*) FROM digests") as cur:
        total = (await cur.fetchone())[0]
    async with db.execute(
        "SELECT id FROM digests WHERE id < ? ORDER BY id DESC LIMIT 1", (digest_id,)
    ) as cur:
        prev_row = await cur.fetchone()
    async with db.execute(
        "SELECT id FROM digests WHERE id > ? ORDER BY id ASC LIMIT 1", (digest_id,)
    ) as cur:
        next_row = await cur.fetchone()
    async with db.execute(
        "SELECT COUNT(*) FROM digests WHERE id > ?", (digest_id,)
    ) as cur:
        newer_count = (await cur.fetchone())[0]
    return {
        "prev_id": prev_row[0] if prev_row else None,
        "next_id": next_row[0] if next_row else None,
        "total": total,
        "position": newer_count + 1,
    }

async def get_active_sources(db: aiosqlite.Connection):
    db.row_factory = aiosqlite.Row
    async with db.execute("SELECT * FROM sources WHERE is_active = 1") as cursor:
        return [dict(row) for row in await cursor.fetchall()]

async def update_source_last_message_id(db: aiosqlite.Connection, source_id: int, message_id: int):
    await db.execute(
        "UPDATE sources SET last_message_id = ? WHERE id = ?", (message_id, source_id)
    )
    await db.commit()

async def update_source_last_cursor(db: aiosqlite.Connection, source_id: int, cursor: str):
    """Update last_cursor (platform-agnostic replacement for update_source_last_message_id)."""
    await db.execute(
        "UPDATE sources SET last_cursor = ?, last_fetched_at = ? WHERE id = ?",
        (cursor, datetime.datetime.now(datetime.UTC).isoformat(), source_id)
    )
    await db.commit()

async def insert_curation_log(
    db: aiosqlite.Connection,
    post_id: int,
    score: int,
    reason: str,
    cohort: str | None = None,
    prompt_version: str | None = None,
):
    """Insert a curation_log row.

    Phase 79-03 / LEARN-ROUTE-01: `cohort` is the optional A/B tag from
    `route_curator_ab` (None when LEARNABLE_ROUTER is off — the default).
    The column is populated only inside the env-gated harness; downstream
    Wilson LB compares cohort A (learnable router) vs cohort B (static yaml).

    Phase 80 / PROMPT-ABLEND-01: `prompt_version` records which prompt
    template served this score (e.g. 'v1', 'v2', or 'legacy' when the
    prompts/ directory was missing). Population happens at the curator
    call site via `prompt_router.pick_version`. NULL is preserved for
    pre-Phase-80 rows.
    """
    now = datetime.datetime.now(datetime.UTC).isoformat()
    await db.execute(
        "INSERT INTO curation_logs (post_id, relevance_score, reason, scored_at, cohort, prompt_version) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (post_id, score, reason, now, cohort, prompt_version)
    )
    await db.commit()

async def ensure_sources(db: aiosqlite.Connection, sources: list[dict], platform: str = "telegram"):
    """
    Upsert sources into the sources table.

    Each source dict may carry an explicit "platform" key (D-02). When present, that
    platform overrides the function-level default — this is how the Reddit collector's
    sources flow through the pipeline. The function-level `platform` kwarg is kept for
    backward compatibility with any caller that pre-dates the per-source platform field.
    """
    for s in sources:
        target = s["target"]
        platform_val = s.get("platform", platform)  # D-02: per-source override
        category = s.get("category", "other")
        display_name = s.get("display_name", target)
        await db.execute(
            "INSERT INTO sources (platform, target, is_active, category, display_name) "
            "SELECT ?, ?, ?, ?, ? WHERE NOT EXISTS "
            "(SELECT 1 FROM sources WHERE platform = ? AND target = ?)",
            (platform_val, target, True, category, display_name, platform_val, target)
        )
        await db.execute(
            "UPDATE sources SET category = ?, display_name = ? "
            "WHERE platform = ? AND target = ?",
            (category, display_name, platform_val, target)
        )
    await db.commit()

async def migrate_db(db: aiosqlite.Connection):
    # --- sources table migrations ---
    async with db.execute("PRAGMA table_info(sources)") as cursor:
        columns = [row[1] for row in await cursor.fetchall()]

    if "last_message_id" not in columns:
        await db.execute("ALTER TABLE sources ADD COLUMN last_message_id INTEGER")
    if "category" not in columns:
        await db.execute("ALTER TABLE sources ADD COLUMN category TEXT DEFAULT 'other'")
    if "display_name" not in columns:
        await db.execute("ALTER TABLE sources ADD COLUMN display_name TEXT")
    if "last_cursor" not in columns:
        await db.execute("ALTER TABLE sources ADD COLUMN last_cursor TEXT")
        # Backfill: copy existing last_message_id into last_cursor for Telegram rows.
        # Guard is INSIDE the "not in columns" check — runs only on first migration.
        await db.execute(
            "UPDATE sources SET last_cursor = CAST(last_message_id AS TEXT) "
            "WHERE last_message_id IS NOT NULL"
        )

    # Phase 11 EMAIL-04 / D-01: per-account IMAP credentials (4 columns).
    # NULL columns fall back to legacy EMAIL_* env vars (D-03 backward-compat).
    # D-02: account_pass_env stores env-var NAME, never the secret value.
    # D-08: no backfill UPDATE here — Plan 02's migration script seeds existing
    # email rows from EMAIL_USER/EMAIL_HOST/EMAIL_PORT env vars explicitly.
    if "account_host" not in columns:
        await db.execute("ALTER TABLE sources ADD COLUMN account_host TEXT")
    if "account_port" not in columns:
        await db.execute("ALTER TABLE sources ADD COLUMN account_port INTEGER")
    if "account_user" not in columns:
        await db.execute("ALTER TABLE sources ADD COLUMN account_user TEXT")
    if "account_pass_env" not in columns:
        await db.execute("ALTER TABLE sources ADD COLUMN account_pass_env TEXT")

    # Phase 62 / SOURCE-THRESH-01: per-source curator threshold override.
    # NULL -> fallback to global CURATOR_THRESHOLD env (default 75).
    # Lookup path: extended source_meta batch query in curator.process_unprocessed
    # (no per-post DB hit). Idempotency = PRAGMA table_info guard above.
    if "curator_threshold" not in columns:
        await db.execute("ALTER TABLE sources ADD COLUMN curator_threshold REAL")

    # Phase 81 / MIX-PROMPTS-01: per-source mixture-of-prompts weights.
    # NULL -> use category default at resolve time (see src/editor/prompt_mixer.py
    # default_weights_for_category). Non-NULL = JSON dict {style: weight}.
    # Idempotency = PRAGMA table_info guard above.
    if "adapter_weights" not in columns:
        await db.execute("ALTER TABLE sources ADD COLUMN adapter_weights TEXT")

    # Phase 89 / CHQ-SCORE-01: per-source curator pass-rate signal over 30d.
    # NULL = insufficient data (<10 posts in window). Recomputed daily by
    # scripts/compute_channel_quality.py (schtask 07:45 GMT+5).
    if "quality_score" not in columns:
        await db.execute("ALTER TABLE sources ADD COLUMN quality_score REAL")
    if "quality_computed_at" not in columns:
        await db.execute("ALTER TABLE sources ADD COLUMN quality_computed_at TEXT")

    # Phase 100 / AUTO-UNSUB-01: two-stage auto-unsubscribe gate.
    # warned_at: timestamp when source first matched (bottom-10 ROI ∩ quality<0.05).
    # unsubbed_at: timestamp when is_active flipped to 0 (set ≥14d after warned_at).
    # override (default 0): operator opt-out — when 1, gate skips source entirely.
    # Idempotency via PRAGMA table_info guard.
    if "auto_unsub_warned_at" not in columns:
        await db.execute("ALTER TABLE sources ADD COLUMN auto_unsub_warned_at TEXT")
    if "auto_unsubbed_at" not in columns:
        await db.execute("ALTER TABLE sources ADD COLUMN auto_unsubbed_at TEXT")
    if "auto_unsub_override" not in columns:
        await db.execute("ALTER TABLE sources ADD COLUMN auto_unsub_override INTEGER DEFAULT 0")

    # --- raw_posts table migrations ---
    async with db.execute("PRAGMA table_info(raw_posts)") as cursor:
        post_cols = [row[1] for row in await cursor.fetchall()]

    if "platform" not in post_cols:
        await db.execute("ALTER TABLE raw_posts ADD COLUMN platform TEXT")
    if "content_type" not in post_cols:
        await db.execute("ALTER TABLE raw_posts ADD COLUMN content_type TEXT")
    if "canonical_url" not in post_cols:
        # Add column FIRST, then create index — order is mandatory (Pitfall 3)
        await db.execute("ALTER TABLE raw_posts ADD COLUMN canonical_url TEXT")
        await db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_canonical_url "
            "ON raw_posts(canonical_url) WHERE canonical_url IS NOT NULL"
        )
    if "simhash" not in post_cols:
        await db.execute("ALTER TABLE raw_posts ADD COLUMN simhash INTEGER")
    if "source_score" not in post_cols:
        await db.execute("ALTER TABLE raw_posts ADD COLUMN source_score INTEGER DEFAULT 0")
    if "simhash_version" not in post_cols:
        # Phase 9 DEDUP-03: backfill defaults to 'v1-default' for existing rows
        await db.execute(
            "ALTER TABLE raw_posts ADD COLUMN simhash_version TEXT NOT NULL DEFAULT 'v1-default'"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_simhash_version ON raw_posts(simhash_version)"
        )
    # Phase 96 IMAP-TRIAGE-01: nullable 4-way label
    # {newsletter, support, personal, action}. NULL = not classified
    # (gate off or fail-soft). No CHECK constraint — operator-facing only.
    if "email_class" not in post_cols:
        await db.execute("ALTER TABLE raw_posts ADD COLUMN email_class TEXT")
    # Phase 10 EXTRACT-05: 3 new columns for article-body extraction.
    # Pattern matches Phase 9 simhash_version idempotent ALTER TABLE block.
    # D-26 invariant: extracted_body is NEVER a simhash input — purely additive.
    if "extracted_body" not in post_cols:
        await db.execute("ALTER TABLE raw_posts ADD COLUMN extracted_body TEXT")
    if "extraction_status" not in post_cols:
        await db.execute("ALTER TABLE raw_posts ADD COLUMN extraction_status TEXT")
    if "extracted_at" not in post_cols:
        await db.execute("ALTER TABLE raw_posts ADD COLUMN extracted_at TEXT")
    if "discovered_via" not in post_cols:
        await db.execute("ALTER TABLE raw_posts ADD COLUMN discovered_via INTEGER")
    # Phase 65 / DEDUP-BGE-01: bge-m3 1024-dim float32 embedding (4096 bytes).
    # Stored as BLOB via np.float32.tobytes(); read via np.frombuffer(blob, dtype=np.float32).
    # NULL when bge-m3 backend unreachable at insert (silent fail-soft to simhash).
    # Idempotent ALTER guarded by PRAGMA table_info above (mirrors P40/P62 pattern).
    if "embedding" not in post_cols:
        await db.execute("ALTER TABLE raw_posts ADD COLUMN embedding BLOB")
    # WR-02 (P65 review): bge-m3 dedup SELECT scans raw_posts on every insert.
    # Index published_at so the 7-day window candidate scan is index-bound,
    # not full-table. Idempotent. Guarded on column existence — some legacy
    # test fixtures create raw_posts without published_at and rely on
    # migrate_db being tolerant.
    if "published_at" in post_cols:
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_raw_posts_published_at "
            "ON raw_posts(published_at)"
        )

    # --- Academic: references_graph table ---
    await db.execute("""
        CREATE TABLE IF NOT EXISTS references_graph (
            id INTEGER PRIMARY KEY,
            parent_doi TEXT NOT NULL,
            child_doi TEXT NOT NULL,
            child_title TEXT,
            child_abstract TEXT,
            depth INTEGER NOT NULL DEFAULT 1,
            fetched_at TEXT NOT NULL,
            UNIQUE(parent_doi, child_doi)
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_refgraph_parent ON references_graph(parent_doi)"
    )

    # --- post_feedback table (Phase 5 / FB-01) ---
    # CREATE TABLE IF NOT EXISTS is idempotent — on legacy DBs (pre-Phase 5)
    # this creates the table; on already-migrated DBs it's a no-op. Mirrors the
    # existing canonical_url index pattern.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS post_feedback (
            id INTEGER PRIMARY KEY,
            post_id INTEGER NOT NULL,
            rating INTEGER CHECK(rating IN (-1, 1)),
            rated_at TEXT NOT NULL,
            FOREIGN KEY(post_id) REFERENCES raw_posts(id)
        )
    """)
    await db.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_post_feedback_post_id "
        "ON post_feedback(post_id)"
    )

    # --- post_topics table (Phase 12 / TOPIC-02) ---
    # CREATE TABLE IF NOT EXISTS is idempotent on legacy DBs lacking the table
    # AND on already-migrated DBs (T-12-05). Mirrors post_feedback pattern.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS post_topics (
            id INTEGER PRIMARY KEY,
            post_id INTEGER NOT NULL,
            topic TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            UNIQUE(post_id, topic),
            FOREIGN KEY(post_id) REFERENCES raw_posts(id)
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_post_topics_post_id ON post_topics(post_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_post_topics_topic ON post_topics(topic)"
    )

    # Phase 15 / REACT-03,04,06: reactions + digest_items tables. CREATE TABLE
    # IF NOT EXISTS makes this idempotent on legacy AND already-migrated DBs.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS reactions (
            id INTEGER PRIMARY KEY,
            item_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            draft_path TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            current_iter INTEGER NOT NULL DEFAULT 1,
            total_iters INTEGER NOT NULL DEFAULT 1,
            error_msg TEXT,
            FOREIGN KEY(item_id) REFERENCES raw_posts(id)
        )
    """)
    await db.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_reactions_active "
        "ON reactions(item_id, action) "
        "WHERE status IN ('pending','processing')"
    )
    await db.execute("""
        CREATE TABLE IF NOT EXISTS digest_items (
            item_id INTEGER NOT NULL,
            digest_id INTEGER NOT NULL,
            post_id INTEGER NOT NULL,
            channel TEXT,
            url TEXT,
            snippet TEXT,
            linked_project TEXT,
            PRIMARY KEY (item_id, digest_id),
            FOREIGN KEY(digest_id) REFERENCES digests(id),
            FOREIGN KEY(post_id) REFERENCES raw_posts(id)
        )
    """)

    # Phase 20 / WORKER-06 + GATE-04: chain_state enum column for finish-chain
    # state transitions. Idempotency via PRAGMA table_info guard (mirrors
    # raw_posts/sources patterns at L478-547). SQLite has NO `IF NOT EXISTS`
    # for ALTER TABLE ADD COLUMN — the PRAGMA guard IS the idempotency mechanism.
    async with db.execute("PRAGMA table_info(reactions)") as cursor:
        reaction_cols = [row[1] for row in await cursor.fetchall()]
    if "chain_state" not in reaction_cols:
        await db.execute(
            "ALTER TABLE reactions ADD COLUMN chain_state TEXT"
        )
        # Backfill existing rows. CASE handles NULL-on-non-finish + every
        # action='finish' status combination. WHERE chain_state IS NULL
        # protects already-correct values from being overwritten on partial-
        # migration replay (CR-02 forward-only spirit; cheap defense even
        # though the PRAGMA guard above prevents re-execution).
        # WR-02 (REVIEW-FIX P20): the archived backfill previously hardcoded
        # 'iter3_continued' for any archived finish row. Pre-Phase 20 finish
        # rows had total_iters=1 (single-iter docs/brainstorm-style finishes,
        # no chain semantics); flagging those as iter3_continued
        # mis-represents the historical chain in briefing surfaces. Refine:
        # only rows that genuinely completed a 3-iter chain
        # (current_iter == total_iters AND total_iters >= 2) get
        # 'iter3_continued'; legacy total_iters=1 archived rows stay NULL
        # (no historical chain claim).
        await db.execute("""
            UPDATE reactions SET chain_state = CASE
              WHEN action != 'finish' THEN NULL
              WHEN status = 'killed' THEN 'killed'
              WHEN status = 'deferred' THEN 'deferred'
              WHEN status = 'pending'    AND current_iter = 1 THEN 'iter1_pending'
              WHEN status = 'processing' AND current_iter = 1 THEN 'iter1_pending'
              WHEN status = 'drafted'    AND current_iter = 1 THEN 'iter1_drafted'
              WHEN status = 'reviewed'   AND current_iter = 1 THEN 'iter1_drafted'
              WHEN status = 'pending'    AND current_iter = 2 THEN 'iter2_pending'
              WHEN status = 'processing' AND current_iter = 2 THEN 'iter2_pending'
              WHEN status = 'drafted'    AND current_iter = 2 THEN 'iter2_drafted'
              WHEN status = 'reviewed'   AND current_iter = 2 THEN 'iter2_drafted'
              WHEN status = 'pending'    AND current_iter = 3 THEN 'iter3_pending'
              WHEN status = 'processing' AND current_iter = 3 THEN 'iter3_pending'
              WHEN status = 'drafted'    AND current_iter = 3 THEN 'iter3_drafted'
              WHEN status = 'reviewed'   AND current_iter = 3 THEN 'iter3_drafted'
              WHEN status = 'archived' AND current_iter = total_iters
                                       AND total_iters >= 2
                                       THEN 'iter3_continued'
              WHEN status = 'archived' THEN NULL
              WHEN status = 'failed' THEN NULL
              ELSE NULL
            END
            WHERE chain_state IS NULL
        """)
    # Idempotent index — survives legacy + fresh + double-migrate.
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_reactions_chain_state "
        "ON reactions(chain_state)"
    )

    # Phase 27 / NB-02 (D-05, D-06): per-reaction NFB cognition window means.
    # 5 REAL NULL columns — never sentinel; NULL = "no overlapping NFB samples
    # in +/-10 min window" (Plan 27-03 write-time + Plan 27-04 backfill set
    # non-NULL when window has >=3 samples per D-03).
    # No index (D-07): cog_* read-along with id/item_id; full-scan acceptable
    # at current reactions volume.
    # PRAGMA re-read because the chain_state ALTER above may have just modified
    # the column list; reaction_cols captured before that mutation is stale for
    # the cog_* checks on legacy DBs.
    async with db.execute("PRAGMA table_info(reactions)") as cursor:
        reaction_cols = [row[1] for row in await cursor.fetchall()]
    if "cog_concentration" not in reaction_cols:
        await db.execute("ALTER TABLE reactions ADD COLUMN cog_concentration REAL")
    if "cog_fatigue" not in reaction_cols:
        await db.execute("ALTER TABLE reactions ADD COLUMN cog_fatigue REAL")
    if "cog_relaxation" not in reaction_cols:
        await db.execute("ALTER TABLE reactions ADD COLUMN cog_relaxation REAL")
    if "cog_alpha" not in reaction_cols:
        await db.execute("ALTER TABLE reactions ADD COLUMN cog_alpha REAL")
    if "cog_beta" not in reaction_cols:
        await db.execute("ALTER TABLE reactions ADD COLUMN cog_beta REAL")
    # Phase 44 (999.22): entry_md_hash for multi-iter draft block dedup (LCD-02).
    if "entry_md_hash" not in reaction_cols:
        await db.execute("ALTER TABLE reactions ADD COLUMN entry_md_hash TEXT")

    # --- curation_logs migrations (v1.3) ---
    # CREATE TABLE IF NOT EXISTS is idempotent — on legacy DBs lacking the table
    # this creates it; on already-migrated DBs it is a no-op. Mirrors post_feedback pattern.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS curation_logs (
            id INTEGER PRIMARY KEY,
            post_id INTEGER REFERENCES raw_posts(id),
            relevance_score INTEGER,
            reason TEXT,
            scored_at TEXT
        )
    """)
    async with db.execute("PRAGMA table_info(curation_logs)") as cursor:
        curation_cols = [row[1] for row in await cursor.fetchall()]
    if "suspect_flag" not in curation_cols:
        await db.execute(
            "ALTER TABLE curation_logs ADD COLUMN suspect_flag INTEGER NOT NULL DEFAULT 0"
        )
    if "profile_hash" not in curation_cols:
        await db.execute(
            "ALTER TABLE curation_logs ADD COLUMN profile_hash TEXT"
        )
    if "stage" not in curation_cols:
        await db.execute(
            "ALTER TABLE curation_logs ADD COLUMN stage TEXT"
        )
    if "backend" not in curation_cols:
        await db.execute(
            "ALTER TABLE curation_logs ADD COLUMN backend TEXT"
        )
    # Phase 30 / RL-01: human_reward column for RL Foundations data accumulation.
    # NULL = "no human signal yet for this post"; non-NULL float = explicit reaction signal.
    # Range enforcement deferred to Phase 31 aggregation logic per CONTEXT D-RL-04.
    if "human_reward" not in curation_cols:
        await db.execute(
            "ALTER TABLE curation_logs ADD COLUMN human_reward REAL"
        )
    # Phase 70 / FACTOR-BUCKET-01: composite key for factorial drift analysis.
    # Format: <backend>:<category>:<iso_week>:<profile_sha8>. Nullable for backfill window.
    if "factor_bucket" not in curation_cols:
        await db.execute(
            "ALTER TABLE curation_logs ADD COLUMN factor_bucket TEXT"
        )
    # Phase 74 / CAUSAL-EMERGENCE-01: phi-proxy score on post->reaction->draft chain.
    # NULL = no reaction signal yet for this post (backfill leaves NULL safely).
    if "causal_emergence_score" not in curation_cols:
        await db.execute(
            "ALTER TABLE curation_logs ADD COLUMN causal_emergence_score REAL"
        )
    # Phase 79-03 / LEARN-ROUTE-01: A/B cohort tag for learnable-router rollout.
    # NULL = LEARNABLE_ROUTER off (default, vast majority of rows). 'A' = post
    # routed by LearnableRouter; 'B' = post routed by static yaml under same
    # gate. Downstream Wilson LB compare uses this to detect routing-policy
    # uplift. Idempotent ALTER. No default — NULL is the safe legacy value.
    if "cohort" not in curation_cols:
        await db.execute(
            "ALTER TABLE curation_logs ADD COLUMN cohort TEXT"
        )
    # Phase 80 / PROMPT-ABLEND-01: prompt-version tag for α-blend rollout.
    # NULL = legacy rows before Plan 02 wires curator. Non-NULL string like
    # 'v1' / 'v2' records which prompts/curator_v{n}.txt actually served the
    # post. Downstream telemetry compares v_new vs v_old quality/cost without
    # firing cost_snapshots.anomaly_flag on every prompt edit. Idempotent ALTER.
    if "prompt_version" not in curation_cols:
        await db.execute(
            "ALTER TABLE curation_logs ADD COLUMN prompt_version TEXT"
        )
    # Phase 95-01 / XFAM-JUDGE-01: cross-family re-score divergence flag.
    # NULL = not audited or in-agreement. 'family_disagree' = |primary -
    # secondary| > XFAM_DISAGREE_THRESHOLD (default 15) when re-scored by
    # an alternate Ollama family (default llama3.1:8b). DISTINCT from the
    # v1.3 `suspect_flag INTEGER NOT NULL DEFAULT 0` adversarial-payload
    # column — that is preserved untouched. Idempotent ALTER.
    if "xfam_suspect_flag" not in curation_cols:
        await db.execute(
            "ALTER TABLE curation_logs ADD COLUMN xfam_suspect_flag TEXT"
        )
    # Phase 100 / AUTO-UNSUB-01 CR-01 fix: tokens_in/tokens_out per-post token
    # accounting consumed by src/eval/channel_roi.py compute_roi SQL. Columns
    # are NULL-default on legacy rows (channel_roi already COALESCEs to 0);
    # scorer backends populate them when token counts are available. Without
    # these columns, compute_roi raises OperationalError, the fail-soft
    # auto_unsubscribe driver swallows it, and the gate becomes a permanent
    # no-op. Idempotent ALTER per CLAUDE.md migrate_db pattern.
    if "tokens_in" not in curation_cols:
        await db.execute(
            "ALTER TABLE curation_logs ADD COLUMN tokens_in INTEGER"
        )
    if "tokens_out" not in curation_cols:
        await db.execute(
            "ALTER TABLE curation_logs ADD COLUMN tokens_out INTEGER"
        )

    # Phase 61 MR-01: composite index for cost_cap._query_usd_today.
    # The cap query is `WHERE scored_at >= ? AND backend IN (?, ?) GROUP BY backend`
    # joined to raw_posts/sources. The docstring in cost_cap claimed a
    # (scored_at, backend) index existed "from Phase 30 et al." but no such
    # index was in the schema — review caught it as a full-scan as the table
    # grows. Composite leading on scored_at supports the range scan; backend
    # as the second column lets SQLite avoid an extra lookup per row for
    # the IN+GROUP BY. Idempotent CREATE INDEX IF NOT EXISTS.
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_curation_logs_scored_at_backend "
        "ON curation_logs(scored_at, backend)"
    )

    # --- post_feedback migrations (v1.3) ---
    async with db.execute("PRAGMA table_info(post_feedback)") as cursor:
        feedback_cols = [row[1] for row in await cursor.fetchall()]
    if "feedback_provenance" not in feedback_cols:
        await db.execute(
            "ALTER TABLE post_feedback ADD COLUMN feedback_provenance TEXT DEFAULT NULL"
        )

    # --- sources: Idea 6 phi_score column ---
    async with db.execute("PRAGMA table_info(sources)") as cursor:
        src_cols_v2 = [row[1] for row in await cursor.fetchall()]
    if "phi_score" not in src_cols_v2:
        await db.execute("ALTER TABLE sources ADD COLUMN phi_score REAL DEFAULT 0.0")

    # --- digest_items: Idea 4 saliency columns ---
    await db.execute("""
        CREATE TABLE IF NOT EXISTS digest_items (
            item_id INTEGER NOT NULL,
            digest_id INTEGER NOT NULL,
            post_id INTEGER NOT NULL,
            channel TEXT,
            url TEXT,
            snippet TEXT,
            linked_project TEXT,
            saliency_novelty TEXT,
            saliency_relevance TEXT,
            saliency_urgency TEXT,
            PRIMARY KEY (item_id, digest_id),
            FOREIGN KEY(digest_id) REFERENCES digests(id),
            FOREIGN KEY(post_id) REFERENCES raw_posts(id)
        )
    """)
    async with db.execute("PRAGMA table_info(digest_items)") as cursor:
        di_cols = [row[1] for row in await cursor.fetchall()]
    for col in ("saliency_novelty", "saliency_relevance", "saliency_urgency"):
        if col not in di_cols:
            await db.execute(f"ALTER TABLE digest_items ADD COLUMN {col} TEXT")

    # --- Idea 3: feedback_priors table ---
    await db.execute("""
        CREATE TABLE IF NOT EXISTS feedback_priors (
            category TEXT PRIMARY KEY,
            weight REAL NOT NULL DEFAULT 1.0,
            updated_at TEXT NOT NULL
        )
    """)

    # --- Idea 5: weekly_meta table ---
    await db.execute("""
        CREATE TABLE IF NOT EXISTS weekly_meta (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            week_start TEXT,
            markdown_content TEXT
        )
    """)

    # --- Academic Idea 3: references_graph — citation edges from OpenAlex ---
    await db.execute("""
        CREATE TABLE IF NOT EXISTS references_graph (
            id INTEGER PRIMARY KEY,
            parent_doi TEXT NOT NULL,
            child_doi TEXT NOT NULL,
            child_title TEXT,
            child_abstract TEXT,
            depth INTEGER NOT NULL DEFAULT 1,
            fetched_at TEXT NOT NULL,
            UNIQUE(parent_doi, child_doi)
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_refgraph_parent ON references_graph(parent_doi)"
    )

    # --- Idea 1: tool_mentions — AI/research tool mentions extracted from digests ---
    await db.execute("""
        CREATE TABLE IF NOT EXISTS tool_mentions (
            id INTEGER PRIMARY KEY,
            post_id INTEGER REFERENCES raw_posts(id),
            digest_id INTEGER REFERENCES digests(id),
            tool_name TEXT NOT NULL,
            context_snippet TEXT,
            stage TEXT NOT NULL DEFAULT 'general',
            extracted_at TEXT NOT NULL
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_tool_mentions_tool ON tool_mentions(tool_name)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_tool_mentions_post ON tool_mentions(post_id)"
    )

    # --- Idea 4: post_claims — falsifiable claims extracted per curated post ---
    await db.execute("""
        CREATE TABLE IF NOT EXISTS post_claims (
            id INTEGER PRIMARY KEY,
            post_id INTEGER NOT NULL REFERENCES raw_posts(id),
            claim_text TEXT NOT NULL,
            claim_rank INTEGER NOT NULL DEFAULT 0
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_post_claims_post ON post_claims(post_id)"
    )

    # --- Idea 7: backend column on digests + backend_preference table ---
    # Guard: digests table may not exist on very old legacy DBs (pre-Phase 15).
    # CREATE TABLE IF NOT EXISTS first, then check for the column.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS digests (
            id INTEGER PRIMARY KEY, created_at TEXT, markdown_content TEXT,
            backend TEXT DEFAULT 'ollama'
        )
    """)
    async with db.execute("PRAGMA table_info(digests)") as cursor:
        digest_cols = [row[1] for row in await cursor.fetchall()]
    if "backend" not in digest_cols:
        await db.execute("ALTER TABLE digests ADD COLUMN backend TEXT DEFAULT 'ollama'")
    await db.execute("""
        CREATE TABLE IF NOT EXISTS backend_preference (
            id INTEGER PRIMARY KEY,
            digest_id_a INTEGER NOT NULL,
            digest_id_b INTEGER NOT NULL,
            winner INTEGER NOT NULL CHECK(winner IN (0, 1)),
            voted_at TEXT NOT NULL
        )
    """)

    # --- digest_chunks + digest_chunks_vec tables (semantic search) ---
    await db.execute("""
        CREATE TABLE IF NOT EXISTS digest_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            digest_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL
        )
    """)
    try:
        import sqlite_vec

        def _load_for_migration():
            db._conn.enable_load_extension(True)
            sqlite_vec.load(db._conn)
            db._conn.enable_load_extension(False)

        await db._execute(_load_for_migration)
        await db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS digest_chunks_vec USING vec0(
                chunk_embedding FLOAT[1024]
            )
        """)
    except Exception:
        pass  # sqlite-vec not installed — vec search disabled, rest of pipeline unaffected

    # --- Telemetry: cost_snapshots table ---
    await db.execute("""
        CREATE TABLE IF NOT EXISTS cost_snapshots (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            backend TEXT NOT NULL,
            calls_7d INTEGER NOT NULL,
            est_tokens_7d INTEGER NOT NULL,
            anomaly_flag INTEGER NOT NULL DEFAULT 0,
            analysis_md TEXT,
            token_breakdown_json TEXT,
            position_bias_delta REAL
        )
    """)
    # Phase 42 (999.6): per-agent token split — additive ALTER for existing installs
    async with db.execute("PRAGMA table_info(cost_snapshots)") as _cur:
        _cost_cols = [r[1] for r in await _cur.fetchall()]
    if "token_breakdown_json" not in _cost_cols:
        await db.execute("ALTER TABLE cost_snapshots ADD COLUMN token_breakdown_json TEXT")
    # Phase 92 / POS-BIAS-01: position-bias probe delta (end_score - start_score).
    if "position_bias_delta" not in _cost_cols:
        await db.execute("ALTER TABLE cost_snapshots ADD COLUMN position_bias_delta REAL")
    # Phase 103 / CLUSTER-FLOOR-01: per-member cosine floor telemetry pair.
    if "cluster_propagated_n" not in _cost_cols:
        await db.execute("ALTER TABLE cost_snapshots ADD COLUMN cluster_propagated_n INTEGER DEFAULT 0")
    if "cluster_floor_skipped_n" not in _cost_cols:
        await db.execute("ALTER TABLE cost_snapshots ADD COLUMN cluster_floor_skipped_n INTEGER DEFAULT 0")

    # --- Phase 69 / DRIFT-DET-01: drift_snapshots (legacy-DB idempotent create) ---
    # Schema MUST match init_db above exactly. CREATE TABLE IF NOT EXISTS makes
    # this safe on existing DBs (no ALTER needed — fresh table).
    await db.execute("""
        CREATE TABLE IF NOT EXISTS drift_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_at TEXT NOT NULL,
            post_id INTEGER,
            original_score INTEGER,
            current_score INTEGER,
            delta_abs REAL,
            backend TEXT,
            is_flagged INTEGER DEFAULT 0,
            is_aggregate INTEGER DEFAULT 0,
            aggregate_mean REAL,
            aggregate_std REAL,
            prompt_hash TEXT
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_drift_snapshot_at ON drift_snapshots(snapshot_at)"
    )
    # P69 review WR-04: idempotent ALTER for legacy DBs created before
    # prompt_hash column existed. Distinguishes prompt-drift from model-drift.
    async with db.execute("PRAGMA table_info(drift_snapshots)") as _cur:
        _drift_cols = [r[1] for r in await _cur.fetchall()]
    if "prompt_hash" not in _drift_cols:
        await db.execute("ALTER TABLE drift_snapshots ADD COLUMN prompt_hash TEXT")

    # --- Personal Dashboard Phase 1: dashboard_checkins ---
    # Daily check-in (state/training/sleep/journal). One row per local-date
    # (UNIQUE constraint), upserted by date. neuroband_data is a nullable JSON
    # blob reserved for Phase 2 (neuroband ingest).
    await db.execute("""
        CREATE TABLE IF NOT EXISTS dashboard_checkins (
            id INTEGER PRIMARY KEY,
            date TEXT NOT NULL UNIQUE,
            energy INTEGER CHECK (energy IS NULL OR energy BETWEEN 1 AND 10),
            focus INTEGER CHECK (focus IS NULL OR focus BETWEEN 1 AND 10),
            mood INTEGER CHECK (mood IS NULL OR mood BETWEEN 1 AND 10),
            training_done INTEGER NOT NULL DEFAULT 0,
            training_label TEXT,
            training_note TEXT,
            sleep_hours REAL CHECK (sleep_hours IS NULL OR (sleep_hours >= 0 AND sleep_hours <= 16)),
            sleep_score INTEGER CHECK (sleep_score IS NULL OR sleep_score BETWEEN 1 AND 5),
            journal TEXT,
            neuroband_data TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_dashboard_checkins_date "
        "ON dashboard_checkins(date)"
    )

    # --- Brainstorm-review pipeline Phase 2 (2026-05-18): legacy-DB idempotent create ---
    # Mirrors the cost_snapshots / dashboard_checkins late-add pattern. Schema must
    # match init_db above EXACTLY.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS brainstorm_decisions (
            id INTEGER PRIMARY KEY,
            draft_id INTEGER NOT NULL,
            idea_num INTEGER NOT NULL,
            review_run_id TEXT NOT NULL,
            project TEXT NOT NULL,
            title TEXT,
            verdict TEXT NOT NULL CHECK(verdict IN ('fit','skip','duplicate','needs-spike')),
            score INTEGER NOT NULL CHECK(score BETWEEN 0 AND 10),
            summary_30w TEXT,
            risks_json TEXT,
            shipped_overlap TEXT,
            next_step TEXT,
            raw_json TEXT NOT NULL,
            user_decision TEXT CHECK(user_decision IS NULL OR user_decision IN ('promote','drop','spike')),
            decided_at TEXT,
            decision_target_path TEXT,
            migrated_to_roadmap TEXT,
            spiked_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(draft_id, idea_num, review_run_id)
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_brainstorm_pending "
        "ON brainstorm_decisions(verdict, score) "
        "WHERE user_decision IS NULL"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_brainstorm_run "
        "ON brainstorm_decisions(review_run_id)"
    )

    # Brainstorm migrator (2026-05-21): track which promote decisions have
    # been migrated into a project's ROADMAP.md. NULL = pending migration.
    # Non-NULL values:
    #   'YYYY-MM-DDThh:mm:ss[tz]' — successfully migrated (ISO timestamp)
    #   'skipped:<reason>'        — intentionally skipped (e.g. duplicate of shipped backlog)
    # Migrator (scripts/migrate_brainstorms_to_roadmap.py) is autorss_feed-scoped:
    # other projects don't have a .planning/ROADMAP.md to migrate into.
    async with db.execute("PRAGMA table_info(brainstorm_decisions)") as cursor:
        _bd_cols = [row[1] for row in await cursor.fetchall()]
    if "migrated_to_roadmap" not in _bd_cols:
        await db.execute(
            "ALTER TABLE brainstorm_decisions ADD COLUMN migrated_to_roadmap TEXT"
        )
    if "spiked_at" not in _bd_cols:
        # Stage 2: per-project spike widget marks spiked_at when user triggers
        # `/gsd-spike` for an item from BRAINSTORM_SPIKES.md.
        await db.execute(
            "ALTER TABLE brainstorm_decisions ADD COLUMN spiked_at TEXT"
        )

    # Phase 52 / DIVERGE-01 + NFR-02 + NFR-03: scorer_divergence — dual-backend
    # disagreement rollup. UNIQUE(raw_post_id, backend_a, backend_b) enforces
    # NFR-02 race protection (used by INSERT OR IGNORE in curator, plan 03).
    # is_holdout = NFR-03 rolling 20% (insert-rule: raw_post_id % 5 == 4).
    # Truncation columns (was_truncated, truncation_mode, char_count_seen)
    # prevent P48/P49 confound (see PITFALLS Pitfall 7 + curator truncation
    # audit reference).
    #
    # WR-02 fix (2026-05-18): NO inline FK declaration. aiosqlite defaults
    # `PRAGMA foreign_keys=OFF`; enabling globally in init/migrate would be
    # backward-incompatible (many pre-existing raw_posts REFERENCES rows
    # across digest_items, post_feedback, post_topics, brainstorm_decisions,
    # etc. — none of which use CASCADE — would start rejecting legacy
    # data orphaned by historical deletes). Instead: any code that deletes
    # `raw_posts` rows MUST also call `cleanup_scorer_divergence_orphans`
    # below to drop dangling rollup rows. Documented expectation, explicit
    # cleanup — no decorative CASCADE that doesn't fire in production.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS scorer_divergence (
            id INTEGER PRIMARY KEY,
            raw_post_id INTEGER NOT NULL,
            backend_a TEXT NOT NULL,
            backend_b TEXT NOT NULL,
            score_a REAL NOT NULL,
            score_b REAL NOT NULL,
            delta_abs REAL NOT NULL,
            is_disputed INTEGER NOT NULL DEFAULT 0,
            is_holdout INTEGER NOT NULL DEFAULT 0,
            sample_source TEXT NOT NULL,
            scored_at TEXT NOT NULL,
            was_truncated INTEGER NOT NULL DEFAULT 0,
            truncation_mode TEXT,
            char_count_seen INTEGER,
            UNIQUE(raw_post_id, backend_a, backend_b)
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_scorer_div_delta "
        "ON scorer_divergence(delta_abs DESC)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_scorer_div_post "
        "ON scorer_divergence(raw_post_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_scorer_div_holdout "
        "ON scorer_divergence(is_holdout)"
    )

    # Phase 78 / ROUTE-FEAT-01: routing_features — hidden-state inputs for P79
    # sigmoid gate. One row per scored post; INSERT OR REPLACE keyed on post_id.
    # Zero LLM burn — populated by extract_features() using cached embeddings.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS routing_features (
            post_id INTEGER PRIMARY KEY,
            post_length INTEGER,
            has_canonical_url INTEGER,
            simhash_dist INTEGER,
            embed_cosine REAL,
            source_category TEXT,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Phase 85 / DWELL-SCHEMA-01: passive dwell-time beacon signal (mirrors init_db).
    # New table only — no ALTERs needed. CREATE TABLE IF NOT EXISTS makes the
    # migrate idempotent on already-bootstrapped DBs. See init_db for rationale.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS post_dwell (
            id INTEGER PRIMARY KEY,
            item_id INTEGER NOT NULL,
            dwell_ms INTEGER NOT NULL CHECK(dwell_ms >= 0),
            session_id TEXT NOT NULL,
            recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(item_id, session_id)
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_post_dwell_item ON post_dwell(item_id)"
    )

    # Phase 86 / KL-SIM-01: source_similarity table (mirrors init_db).
    # CREATE TABLE IF NOT EXISTS keeps the migration idempotent.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS source_similarity (
            source_a_id INTEGER NOT NULL,
            source_b_id INTEGER NOT NULL,
            jsd_divergence REAL NOT NULL,
            n_a INTEGER NOT NULL,
            n_b INTEGER NOT NULL,
            computed_at TEXT NOT NULL,
            UNIQUE(source_a_id, source_b_id),
            CHECK(source_a_id < source_b_id)
        )
    """)

    # Phase 87-02 / WORKER-MEM-02: agent_traces.memory_context_tokens column
    # for finish_worker memory layer telemetry. Idempotent ALTER — duplicate
    # column error is benign (column already exists). Wrapped per CLAUDE.md
    # "migrate_db ALTER TABLE wrapped in try/except OperationalError".
    # agent_traces table itself is created by src/observability/traces.py
    # (ensure_table) on demand; ALTER here is a no-op when the table is
    # absent — we catch that OperationalError too.
    try:
        await db.execute(
            "ALTER TABLE agent_traces ADD COLUMN memory_context_tokens INTEGER"
        )
    except sqlite3.OperationalError:
        # WR-03: narrowed from bare Exception. OperationalError covers the
        # two benign cases (duplicate column on already-migrated DB, no such
        # table on legacy DBs where traces.ensure_table hasn't run yet).
        # Other errors (IOError, programmer typos) must propagate.
        pass

    # Phase 97 / GA-PROMPT-01: prompts_elo — Elo tournament state for curator
    # rubric variants. variant_id is the prompt identifier (slug or hash);
    # elo_score defaults to seed rating 1500.0; n_matches counts pairwise
    # comparisons accumulated; generation tags the 4-week rewrite window.
    # Idempotent via CREATE TABLE IF NOT EXISTS.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS prompts_elo (
            variant_id TEXT PRIMARY KEY,
            elo_score REAL NOT NULL DEFAULT 1500.0,
            n_matches INTEGER NOT NULL DEFAULT 0,
            generation INTEGER NOT NULL DEFAULT 0,
            parent_variant_id TEXT
        )
    """)

    # Phase 102 / GA-LINEAGE-01: add parent_variant_id to prompts_elo for
    # mutation lineage tracking. NULL = seed/root (pre-existing rows).
    # Idempotent via PRAGMA introspection.
    try:
        async with db.execute("PRAGMA table_info(prompts_elo)") as cur:
            cols = {row[1] for row in await cur.fetchall()}
        if "parent_variant_id" not in cols:
            await db.execute(
                "ALTER TABLE prompts_elo ADD COLUMN parent_variant_id TEXT"
            )
    except sqlite3.OperationalError:
        # WR-03 pattern: swallow benign duplicate-column races.
        pass

    # Phase 101 / CHAMPION-HIST-01: champion_history — tracks every curator
    # prompt champion swap. variant_id NOT unique (same variant can be
    # re-promoted later, each promotion gets a new row). demoted_at NULL
    # marks the currently-reigning champion. Atomic write paired with the
    # prompts file os.replace by promote_champion() — see src/eval/champion_gate.py.
    # Mirrored in init_db schema block.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS champion_history (
            id INTEGER PRIMARY KEY,
            variant_id TEXT NOT NULL,
            promoted_at TEXT NOT NULL,
            demoted_at TEXT,
            elo_at_promotion REAL NOT NULL,
            n_matches_at_promotion INTEGER NOT NULL
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_champion_history_variant ON champion_history(variant_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_champion_history_promoted_at ON champion_history(promoted_at)"
    )

    # Phase 104 / CANARY-01: per-feature canary run ledger.
    # One row per activation (started_at). ended_at NULL while live; verdict
    # ('keep'|'rollback'|'extended'|'skipped') written by operator via
    # POST /api/canary/{id}/decision. metrics_json stores compare() output.
    # Idempotent — CREATE IF NOT EXISTS for both table + index.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS canary_runs (
            id INTEGER PRIMARY KEY,
            feature TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            verdict TEXT,
            metrics_json TEXT
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_canary_feature_started "
        "ON canary_runs(feature, started_at)"
    )
    # BL-03: enforce "one canary at a time" per feature at the schema level
    # via partial UNIQUE index on (ended_at IS NULL). SQLite ≥3.8 supports
    # partial indexes (universal on modern Python). Idempotent.
    await db.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uniq_canary_one_active "
        "ON canary_runs(feature) WHERE ended_at IS NULL"
    )

    # Phase 106 / SELF-TUNE-01: per-category curator threshold tuner ledger.
    # One row per (category, computed_at) recommendation. applied_at NULL while
    # pending, ISO ts on operator-apply, 'dismissed' sentinel on dismiss.
    # Partial idx makes the pending-list query trivial AND excludes 'dismissed'
    # from `WHERE applied_at IS NULL` (sentinel string is not NULL). Idempotent.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS threshold_recommendations (
            id INTEGER PRIMARY KEY,
            category TEXT NOT NULL,
            current_threshold INTEGER NOT NULL,
            recommended_threshold INTEGER NOT NULL,
            mi_current REAL,
            mi_recommended REAL,
            computed_at TEXT NOT NULL,
            applied_at TEXT,
            applied_by TEXT
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_threshold_rec_pending "
        "ON threshold_recommendations(applied_at) WHERE applied_at IS NULL"
    )

    await db.commit()

    # WR-03: enable WAL so the sync sqlite3 writer in routing.persist_features
    # doesn't block on the aiosqlite curator pipeline (and vice versa).
    # journal_mode is a connection-scoped PRAGMA that returns a result row and
    # must run OUTSIDE the open transaction (after commit). Persistent setting —
    # sqlite stores journal_mode in the db file header on first WAL switch.
    # Fail-soft: errors here must not break migrate (e.g. memory db, locked).
    try:
        async with db.execute("PRAGMA journal_mode=WAL") as cur:
            await cur.fetchone()
    except Exception:
        pass


async def upsert_checkin(db: aiosqlite.Connection, data: dict) -> int:
    """Insert or replace daily check-in. Returns row id.

    `data["date"]` is required (UNIQUE key). All other fields optional —
    missing keys are stored as NULL. `training_done` accepts bool or int.
    """
    fields = (
        "date", "energy", "focus", "mood",
        "training_done", "training_label", "training_note",
        "sleep_hours", "sleep_score", "journal", "neuroband_data",
    )
    values = {k: data.get(k) for k in fields}
    if isinstance(values["training_done"], bool):
        values["training_done"] = int(values["training_done"])
    if values["training_done"] is None:
        values["training_done"] = 0

    placeholders = ", ".join(f":{f}" for f in fields)
    columns = ", ".join(fields)
    sql = f"""
        INSERT INTO dashboard_checkins ({columns}, updated_at)
        VALUES ({placeholders}, datetime('now'))
        ON CONFLICT(date) DO UPDATE SET
            energy=excluded.energy,
            focus=excluded.focus,
            mood=excluded.mood,
            training_done=excluded.training_done,
            training_label=excluded.training_label,
            training_note=excluded.training_note,
            sleep_hours=excluded.sleep_hours,
            sleep_score=excluded.sleep_score,
            journal=excluded.journal,
            neuroband_data=excluded.neuroband_data,
            updated_at=datetime('now')
    """
    cur = await db.execute(sql, values)
    await db.commit()
    # ON CONFLICT path: lastrowid may be 0 — look up by date.
    if cur.lastrowid and cur.lastrowid > 0:
        return int(cur.lastrowid)
    async with db.execute(
        "SELECT id FROM dashboard_checkins WHERE date = ?", (values["date"],)
    ) as cur2:
        row = await cur2.fetchone()
    return int(row[0])


async def get_checkin(db: aiosqlite.Connection, date: str) -> dict | None:
    """Return the check-in row for YYYY-MM-DD as a dict, or None."""
    db.row_factory = aiosqlite.Row
    async with db.execute(
        "SELECT * FROM dashboard_checkins WHERE date = ?", (date,)
    ) as cur:
        row = await cur.fetchone()
    return dict(row) if row else None


async def get_recent_checkins(db: aiosqlite.Connection, days: int) -> list[dict]:
    """Return up to `days` most recent check-ins, newest first."""
    db.row_factory = aiosqlite.Row
    async with db.execute(
        "SELECT * FROM dashboard_checkins ORDER BY date DESC LIMIT ?",
        (days,),
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def record_feedback(db: aiosqlite.Connection, post_id: int, rating: int) -> None:
    """
    Upsert a single feedback row keyed on post_id (D-02 INSERT OR REPLACE).

    Validates rating in {-1, 1} via the SQL CHECK constraint — invalid ratings
    raise sqlite3.IntegrityError at execute time. Caller is responsible for
    verifying post_id exists in raw_posts (FK is documentation-only since the
    codebase doesn't set PRAGMA foreign_keys=ON).

    Args:
      db: open aiosqlite connection
      post_id: id of the raw_posts row being rated
      rating: -1 (downvote) or 1 (upvote); 0 is rejected by CHECK
    """
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    await db.execute(
        "INSERT OR REPLACE INTO post_feedback (post_id, rating, rated_at) "
        "VALUES (?, ?, ?)",
        (post_id, rating, now_iso),
    )
    await db.commit()


async def insert_reaction(
    db: aiosqlite.Connection,
    item_id: int,
    action: str,
    status: str = "pending",
) -> int | None:
    """Phase 16 / REACT-02 + REACT-04 + REACT-05: insert a reaction row.

    Returns the new row's lastrowid on success, or None when the 60s soft-window
    pre-check finds an existing row for the same (item_id, action). The caller
    is responsible for fetching the existing id and translating None into the
    appropriate response.

    REACT-04 / T-16-08: pre-check is a best-effort optimization; the partial
    UNIQUE index `idx_reactions_active` is the authoritative second line of
    defense for the race-loser path. We DELIBERATELY do NOT catch
    sqlite3.IntegrityError here — the route handler maps that to HTTP 409.

    Skip semantics (REACT-05): caller passes status='reviewed' for skip; the
    soft-window pre-check still applies (one click per 60s regardless of action).

    Convention notes: no logging here (CONVENTIONS.md — DB layer is silent;
    caller logs). No module-level env binding (NFR-08 / D-12 trap).
    """
    # 60s soft-window pre-check.
    # CR-01 fix: pass an ISO-formatted threshold as a parameter instead of the
    # SQLite `datetime('now', '-60 seconds')` function. created_at is written
    # via Python `datetime.isoformat()` (uses 'T' separator); the SQLite
    # datetime() function emits a space separator. A lexicographic >= against
    # mismatched formats compares 'T' (ASCII 84) to ' ' (ASCII 32) at index 10
    # and the window silently widens to ~24h. Python-side normalisation aligns
    # both sides to the same ISO format.
    threshold = (
        datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=60)
    ).isoformat()
    async with db.execute(
        "SELECT id FROM reactions WHERE item_id=? AND action=? "
        "AND created_at >= ? "
        "ORDER BY id DESC LIMIT 1",
        (item_id, action, threshold),
    ) as cur:
        hit = await cur.fetchone()
    if hit is not None:
        return None

    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    cursor = await db.execute(
        "INSERT INTO reactions (item_id, action, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (item_id, action, status, now_iso, now_iso),
    )
    await db.commit()
    rid = cursor.lastrowid

    # Phase 27 / NB-01 + NB-03 (D-08, D-09): post-insert cognition tagging.
    # Best-effort — any failure leaves cog_* NULL and increments the untagged
    # counter; the reaction insert NEVER fails because of cognition. The 60s
    # soft-window dedup path above returns BEFORE this block, so dedup-skipped
    # calls never touch the counters (D-03: only "reactions inserted").
    # Lazy import keeps the DB→neuroband dependency runtime-only and avoids
    # pulling the parser at module load for callers that never react.
    global _reactions_with_cognition, _reactions_without_cognition
    try:
        from src.neuroband.cognition import compute_cognition_window
        cog = compute_cognition_window(now_iso, COGNITION_CSV_DIR)
        if cog is not None:
            await db.execute(
                "UPDATE reactions SET "
                "cog_concentration=?, cog_fatigue=?, cog_relaxation=?, "
                "cog_alpha=?, cog_beta=? WHERE id=?",
                (
                    cog["concentration"],
                    cog["fatigue"],
                    cog["relaxation"],
                    cog["alpha"],
                    cog["beta"],
                    rid,
                ),
            )
            await db.commit()
            _reactions_with_cognition += 1
        else:
            _reactions_without_cognition += 1
    except Exception:
        _cog_logger.warning(
            "cognition tagging failed for reaction %s (swallowed)",
            rid,
            exc_info=True,
        )
        _reactions_without_cognition += 1

    return rid


async def insert_finish_reaction_softcapped(
    db: aiosqlite.Connection,
    item_id: int,
    *,
    soft_cap: int = 5,
    total_iters: int = 3,
) -> int | None:
    """Phase 20 / GATE-04: atomic conditional INSERT for finish-chain.

    SQLite's `INSERT ... SELECT ... WHERE ...` is atomic at the statement
    level — no TOCTOU window between counting active chains and inserting
    a new one. The 60s soft-window dedup runs FIRST (mirrors insert_reaction);
    the partial UNIQUE index `idx_reactions_active` covers the race-loser
    path for repeat finish clicks on the same item_id (caller maps to 409).

    Active = action='finish' AND status IN ('pending','processing','drafted')
             AND current_iter < total_iters. Deferred / killed / archived
             rows are excluded by design (they no longer consume a slot).

    Returns:
        lastrowid (positive int) on successful INSERT.
        None when the WHERE clause filtered the SELECT (cap reached) OR
          when the 60s soft-window pre-check found a recent duplicate.

    Raises:
        sqlite3.IntegrityError if the partial UNIQUE index fires (race-loser
        path); caller maps to HTTP 409.
    """
    # 60s soft-window pre-check (same shape as insert_reaction — repeat-click
    # dedup). CR-01: ISO-format threshold to align with Python-written
    # created_at; SQLite datetime('now',...) emits a space separator and
    # would silently widen the window to ~24h.
    threshold = (
        datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=60)
    ).isoformat()
    async with db.execute(
        "SELECT id FROM reactions WHERE item_id=? AND action='finish' "
        "AND created_at >= ? ORDER BY id DESC LIMIT 1",
        (item_id, threshold),
    ) as cur:
        hit = await cur.fetchone()
    if hit is not None:
        return None  # caller fetches existing rid

    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    # Atomic conditional INSERT. SQLite's INSERT ... SELECT ... WHERE runs
    # the SELECT and INSERT under a single write transaction, so two
    # concurrent connections at active=4 will see exactly one succeed —
    # the other's SELECT is forced to read AFTER the first commit (busy_timeout
    # blocks until the writer releases the lock; WAL gives readers a snapshot).
    #
    # CR-02 (REVIEW-FIX P20): the partial UNIQUE index `idx_reactions_active`
    # only covers `status IN ('pending','processing')`. A row at
    # chain_state='iter1_drafted' (status='drafted') is INVISIBLE to it, so
    # a finish click >60s after the first iter drafts would slip past both
    # the soft-window pre-check and the partial UNIQUE and start a SECOND
    # active chain on the same item. The added NOT EXISTS guard rejects any
    # in-flight chain on this item_id (every iter*_pending / iter*_drafted
    # state). On rowcount=0 the caller must disambiguate cap-hit vs
    # already-active-chain (the helper alone cannot — both surface as
    # rowcount=0).
    cursor = await db.execute(
        "INSERT INTO reactions "
        "(item_id, action, status, current_iter, total_iters, "
        " chain_state, created_at, updated_at) "
        "SELECT ?, 'finish', 'pending', 1, ?, 'iter1_pending', ?, ? "
        "WHERE (SELECT COUNT(*) FROM reactions "
        "       WHERE action='finish' "
        "       AND status IN ('pending','processing','drafted') "
        "       AND current_iter < total_iters) < ? "
        "  AND NOT EXISTS (SELECT 1 FROM reactions "
        "                  WHERE item_id=? AND action='finish' "
        "                  AND chain_state IN ('iter1_pending','iter1_drafted',"
        "                                      'iter2_pending','iter2_drafted',"
        "                                      'iter3_pending','iter3_drafted'))",
        (item_id, total_iters, now_iso, now_iso, soft_cap, item_id),
    )
    await db.commit()
    # WR-01 trap: cursor.lastrowid is unreliable on the WHERE-filtered path
    # (sqlite3 may retain the previous successful insert's id within the same
    # connection — same trap documented for raw_posts at L312-318). rowcount
    # IS authoritative: 0 = WHERE filtered (cap hit OR already-active chain
    # on this item), 1 = inserted.
    if cursor.rowcount == 0:
        return None
    return cursor.lastrowid


def claim_pending_reaction(db: sqlite3.Connection) -> dict | None:
    """Phase 17 / WORKER-02: atomic conditional claim of the oldest pending row.

    Sync variant for the worker process — uses Phase 15 schema (reactions table
    with status/action columns) and Phase 15's `open_db_sync` connection.

    Atomicity: SELECT id ORDER BY id LIMIT 1, then UPDATE WHERE id=? AND
    status='pending'. If `cursor.rowcount == 0`, another worker won between
    SELECT and UPDATE — we LOOP and try the next pending row instead of
    returning None. Returning None previously conflated "queue empty" with
    "race-loss"; the caller's drain loop would then halt prematurely under
    concurrent ticks (CR-01 / regression: test_run_tick_race_loser_continues).

    Returns None ONLY when the SELECT finds zero pending rows.

    Side effects on success: status='processing', updated_at=now(UTC iso).
    Returns full row as dict; None on no-pending only.

    Convention: no module-level env binding; no logging here (DB layer is
    silent — caller logs); commits inline.
    """
    while True:
        # Step 1: peek oldest pending id.
        cur = db.execute(
            "SELECT id FROM reactions WHERE status='pending' ORDER BY id LIMIT 1"
        )
        row = cur.fetchone()
        if row is None:
            return None  # genuinely empty queue
        rid = row[0]
        # Step 2: conditional claim. UPDATE+commit are the atomic boundary;
        # rowcount tells us whether we won or lost the race.
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        cur2 = db.execute(
            "UPDATE reactions SET status='processing', updated_at=? "
            "WHERE id=? AND status='pending'",
            (now_iso, rid),
        )
        db.commit()
        if cur2.rowcount == 1:
            break  # we won — fall through to read-back
        # Race lost on this row; another worker claimed it. Loop and try the
        # next pending row — DO NOT return None here (would halt drain).

    # Step 3: read full row back so caller has item_id/action/etc.
    db.row_factory = sqlite3.Row  # WR-06: forward-compat against column reorder
    cur3 = db.execute(
        "SELECT id, item_id, action, status, draft_path, created_at, "
        "updated_at, current_iter, total_iters, error_msg "
        "FROM reactions WHERE id=?",
        (rid,),
    )
    r = cur3.fetchone()
    if r is None:
        return None
    return {
        "id": r["id"],
        "item_id": r["item_id"],
        "action": r["action"],
        "status": r["status"],
        "draft_path": r["draft_path"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
        "current_iter": r["current_iter"],
        "total_iters": r["total_iters"],
        "error_msg": r["error_msg"],
    }


async def get_all_feedback(db: aiosqlite.Connection) -> dict[int, int]:
    """Return {post_id: rating} for all rows in post_feedback.

    Used by the web UI to restore .voted/.disabled state on page load so the
    buttons reflect persisted ratings even after a server restart. The UNIQUE
    INDEX on post_id guarantees one row per post.
    """
    async with db.execute("SELECT post_id, rating FROM post_feedback") as cur:
        rows = await cur.fetchall()
    return {row[0]: row[1] for row in rows}


async def insert_post_topics(
    db: aiosqlite.Connection,
    post_id: int,
    topics: list[dict],
) -> None:
    """Phase 12 / TOPIC-02: persist 1-3 topic tags for a curated post.

    Defense-in-depth at DB layer (T-12-03 + T-12-04):
    - Drops topics where name NOT in ALLOWED_TOPICS (silent skip).
    - Clamps confidence to [0.0, 1.0]; defaults to 1.0 on missing/invalid.
    - INSERT OR IGNORE on (post_id, topic) UNIQUE — idempotent re-runs.

    D-04: confidence defaults to 1.0 when caller omits or value is non-numeric.
    Empty topics list = no-op (no DB write, no error).

    Parameterized SQL only (T-12-02 mitigation).
    """
    if not topics:
        return
    rows = []
    for t in topics:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        if not isinstance(name, str) or name not in ALLOWED_TOPICS:
            continue
        try:
            conf = float(t.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        conf = max(0.0, min(1.0, conf))
        rows.append((post_id, name, conf))
    if not rows:
        return
    await db.executemany(
        "INSERT OR IGNORE INTO post_topics (post_id, topic, confidence) "
        "VALUES (?, ?, ?)",
        rows,
    )
    await db.commit()


async def get_topics_for_posts(
    db: aiosqlite.Connection,
    post_ids: list[int],
) -> dict[int, list[dict]]:
    """Phase 12 / TOPIC-02: return ordered topics per post for digest rendering.

    Returns {post_id: [{"name": str, "confidence": float}, ...]} ordered by
    (confidence DESC, topic ASC) so:
      - The first-listed topic is the highest-confidence one (D-18 — Plan 03
        uses index [0] as the post's primary topic for sub-header grouping).
      - Topic ASC is a stable tiebreaker so test assertions are deterministic.

    Posts without any topics are absent from the dict (callers default via
    `.get(pid, [])`). Empty input list → empty dict (no SQL issued).

    Parameterized SQL only (T-12-02): the IN-clause `placeholders` is built
    from list length, not from user input.
    """
    if not post_ids:
        return {}
    placeholders = ",".join("?" * len(post_ids))
    db.row_factory = aiosqlite.Row
    async with db.execute(
        f"SELECT post_id, topic, confidence FROM post_topics "
        f"WHERE post_id IN ({placeholders}) "
        f"ORDER BY post_id ASC, confidence DESC, topic ASC",
        tuple(post_ids),
    ) as cursor:
        rows = await cursor.fetchall()
    result: dict[int, list[dict]] = {}
    for row in rows:
        pid = row["post_id"]
        result.setdefault(pid, []).append(
            {"name": row["topic"], "confidence": row["confidence"]}
        )
    return result


async def get_feedback_summary(db: aiosqlite.Connection, days: int = 7) -> dict:
    """
    Compute a 7-day rolling feedback summary for the curator's scoring context.

    Joins post_feedback -> raw_posts -> sources, groups by sources.category,
    returns the top-3 categories per direction (upvoted/downvoted) plus totals.
    Used by Wave 3 (curator.get_mcp_context) to surface recent feedback as a
    context line; computed at call time (not cached) — small table at our scale.

    Args:
      db: open aiosqlite connection
      days: rolling window length (default 7 per D-18)

    Returns:
      {
        "upvoted_categories":   [(category, count), ...],   # top 3 by count
        "downvoted_categories": [(category, count), ...],   # top 3 by count
        "total_up":   int,
        "total_down": int,
      }
      Empty feedback returns the same shape with empty lists and zero totals
      so callers can read keys without try/except.
    """
    cutoff = (
        datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    ).isoformat()

    # D-22: top-3 upvoted categories (rating = 1).
    # ORDER BY cnt DESC, category ASC — stable tiebreaker for tests.
    async with db.execute(
        "SELECT COALESCE(s.category, 'other') AS category, COUNT(*) AS cnt "
        "FROM post_feedback pf "
        "JOIN raw_posts rp ON rp.id = pf.post_id "
        "JOIN sources s ON s.id = rp.source_id "
        "WHERE pf.rating = 1 AND pf.rated_at >= ? "
        "GROUP BY category "
        "ORDER BY cnt DESC, category ASC "
        "LIMIT 3",
        (cutoff,),
    ) as cur:
        upvoted = [(row[0], row[1]) for row in await cur.fetchall()]

    # D-22: top-3 downvoted categories (rating = -1).
    async with db.execute(
        "SELECT COALESCE(s.category, 'other') AS category, COUNT(*) AS cnt "
        "FROM post_feedback pf "
        "JOIN raw_posts rp ON rp.id = pf.post_id "
        "JOIN sources s ON s.id = rp.source_id "
        "WHERE pf.rating = -1 AND pf.rated_at >= ? "
        "GROUP BY category "
        "ORDER BY cnt DESC, category ASC "
        "LIMIT 3",
        (cutoff,),
    ) as cur:
        downvoted = [(row[0], row[1]) for row in await cur.fetchall()]

    # Totals — single aggregate; SUM may return NULL on empty result.
    async with db.execute(
        "SELECT "
        "  SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END), "
        "  SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) "
        "FROM post_feedback "
        "WHERE rated_at >= ?",
        (cutoff,),
    ) as cur:
        row = await cur.fetchone()
    total_up = (row[0] or 0) if row else 0
    total_down = (row[1] or 0) if row else 0

    return {
        "upvoted_categories": upvoted,
        "downvoted_categories": downvoted,
        "total_up": int(total_up),
        "total_down": int(total_down),
    }


async def get_feedback_post_ids(db: aiosqlite.Connection, days: int = 7) -> list[int]:
    """Return post_ids from post_feedback within the rolling window."""
    cutoff = (
        datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    ).isoformat()
    async with db.execute(
        "SELECT post_id FROM post_feedback WHERE rated_at >= ? ORDER BY post_id",
        (cutoff,),
    ) as cur:
        return [row[0] for row in await cur.fetchall()]


async def mark_feedback_used(db: aiosqlite.Connection, post_ids: list[int]) -> None:
    """Mark post_feedback rows as used_in_context for wireheading tracking."""
    if not post_ids:
        return
    placeholders = ",".join("?" * len(post_ids))
    await db.execute(
        f"UPDATE post_feedback SET feedback_provenance='used_in_context' "
        f"WHERE post_id IN ({placeholders}) AND feedback_provenance IS NULL",
        tuple(post_ids),
    )
    await db.commit()


async def get_feedback_provenance_ratio(db: aiosqlite.Connection) -> float:
    """Return fraction of post_feedback rows already marked used_in_context."""
    async with db.execute("SELECT COUNT(*) FROM post_feedback") as cur:
        total = (await cur.fetchone())[0]
    if total == 0:
        return 0.0
    async with db.execute(
        "SELECT COUNT(*) FROM post_feedback WHERE feedback_provenance='used_in_context'"
    ) as cur:
        used = (await cur.fetchone())[0]
    return used / total


async def get_feedback_per_category(db: aiosqlite.Connection, days: int = 30) -> list[dict]:
    """Phase 8 CUR-03: per-category up/down counts in rolling window.

    Returns list of dicts {"category": str, "up": int, "down": int} for ALL categories
    with at least one feedback row in the window. Categories without feedback are absent.

    Uses parameterized SQL via ? placeholders. Read-only.
    """
    query = """
    SELECT
        COALESCE(s.category, 'other') AS category,
        SUM(CASE WHEN pf.rating = 1 THEN 1 ELSE 0 END) AS up_count,
        SUM(CASE WHEN pf.rating = -1 THEN 1 ELSE 0 END) AS down_count
    FROM post_feedback pf
    JOIN raw_posts rp ON rp.id = pf.post_id
    JOIN sources s ON s.id = rp.source_id
    WHERE pf.rated_at > datetime('now', ?)
    GROUP BY category
    """
    window = f"-{int(days)} days"
    async with db.execute(query, (window,)) as cur:
        rows = await cur.fetchall()
    return [
        {"category": r[0], "up": int(r[1] or 0), "down": int(r[2] or 0)}
        for r in rows
    ]


async def get_saliency_by_item(
    db: aiosqlite.Connection, digest_id: int
) -> dict[int, dict]:
    """Idea 4 (Attention Schema): return {item_id: saliency_dict} for a digest.

    Returns only rows where at least one saliency column is non-NULL.
    """
    db.row_factory = aiosqlite.Row
    async with db.execute(
        "SELECT item_id, saliency_novelty, saliency_relevance, saliency_urgency "
        "FROM digest_items WHERE digest_id = ? "
        "AND (saliency_novelty IS NOT NULL OR saliency_relevance IS NOT NULL "
        "     OR saliency_urgency IS NOT NULL)",
        (digest_id,),
    ) as cursor:
        rows = await cursor.fetchall()
    return {
        row["item_id"]: {
            "novelty": row["saliency_novelty"],
            "relevance": row["saliency_relevance"],
            "urgency": row["saliency_urgency"],
        }
        for row in rows
    }


async def get_suspect_post_ids(db: aiosqlite.Connection) -> set[int]:
    """Return set of post_ids where curation_logs.suspect_flag = 1."""
    async with db.execute(
        "SELECT DISTINCT post_id FROM curation_logs WHERE suspect_flag = 1"
    ) as cur:
        return {row[0] for row in await cur.fetchall()}


async def get_posts_with_extraction(db: aiosqlite.Connection) -> dict[int, dict]:
    """Phase 10 EXTRACT-06: return {post_id: {extracted_body, extraction_status, extracted_at}}.

    Includes only posts where extraction_status IS NOT NULL — covers all posts
    where extraction was attempted (success or any failure state).
    Used by Web UI to render badges + collapsible body panel per post.
    """
    db.row_factory = aiosqlite.Row
    async with db.execute(
        "SELECT id, extracted_body, extraction_status, extracted_at "
        "FROM raw_posts WHERE extraction_status IS NOT NULL"
    ) as cursor:
        rows = await cursor.fetchall()
    return {
        row["id"]: {
            "extracted_body": row["extracted_body"] or "",
            "extraction_status": row["extraction_status"],
            "extracted_at": row["extracted_at"],
        }
        for row in rows
    }


async def insert_backend_preference(
    db: aiosqlite.Connection,
    digest_id_a: int,
    digest_id_b: int,
    winner: int,
) -> None:
    """Record a pairwise preference vote (winner=0 means A won, 1 means B won)."""
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    await db.execute(
        "INSERT INTO backend_preference (digest_id_a, digest_id_b, winner, voted_at) "
        "VALUES (?, ?, ?, ?)",
        (digest_id_a, digest_id_b, winner, now_iso),
    )
    await db.commit()


def _wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    """Wilson score interval lower bound for a proportion. Returns 0.0 when total == 0."""
    if total == 0:
        return 0.0
    n = total
    p_hat = successes / n
    centre = p_hat + z * z / (2 * n)
    margin = z * ((p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) ** 0.5)
    denom = 1 + z * z / n
    return (centre - margin) / denom


async def get_backend_preference_stats(
    db: aiosqlite.Connection,
    digest_id_a: int,
    digest_id_b: int,
) -> dict:
    """Return vote counts + Wilson LB for the A-vs-B pairing.

    Returns:
        {"votes_a": int, "votes_b": int, "total": int,
         "wilson_lb_a": float, "wilson_lb_b": float,
         "winner": "a"|"b"|"tie"|"insufficient"}
    """
    async with db.execute(
        "SELECT winner, COUNT(*) FROM backend_preference "
        "WHERE digest_id_a=? AND digest_id_b=? GROUP BY winner",
        (digest_id_a, digest_id_b),
    ) as cur:
        rows = await cur.fetchall()
    votes = {0: 0, 1: 0}
    for winner_val, cnt in rows:
        votes[winner_val] = cnt
    total = votes[0] + votes[1]
    lb_a = _wilson_lower_bound(votes[0], total)
    lb_b = _wilson_lower_bound(votes[1], total)
    if total < 5:
        winner_label = "insufficient"
    elif lb_a >= 0.7:
        winner_label = "a"
    elif lb_b >= 0.7:
        winner_label = "b"
    else:
        winner_label = "tie"
    return {
        "votes_a": votes[0],
        "votes_b": votes[1],
        "total": total,
        "wilson_lb_a": round(lb_a, 3),
        "wilson_lb_b": round(lb_b, 3),
        "winner": winner_label,
    }


async def get_backend_winner_stats(db: aiosqlite.Connection) -> dict:
    """Cumulative cross-backend stats: aggregate all backend_preference rows,
    join to digests.backend, count wins per backend label.

    Returns:
        {"wins_claude": int, "wins_ollama": int, "wins_other": int,
         "total": int, "wilson_lb_claude": float, "wilson_lb_ollama": float,
         "winner": "claude"|"ollama"|"tie"|"insufficient"}
    """
    async with db.execute(
        """
        SELECT da.backend AS backend_a, db_.backend AS backend_b, bp.winner
        FROM backend_preference bp
        JOIN digests da ON da.id = bp.digest_id_a
        JOIN digests db_ ON db_.id = bp.digest_id_b
        """
    ) as cur:
        rows = await cur.fetchall()
    counts: dict[str, int] = {}
    for backend_a, backend_b, winner in rows:
        chosen = backend_a if winner == 0 else backend_b
        counts[chosen] = counts.get(chosen, 0) + 1
    wins_claude = counts.get("claude", 0)
    wins_ollama = counts.get("ollama", 0)
    wins_other = sum(v for k, v in counts.items() if k not in ("claude", "ollama"))
    total = wins_claude + wins_ollama + wins_other
    lb_claude = _wilson_lower_bound(wins_claude, total)
    lb_ollama = _wilson_lower_bound(wins_ollama, total)
    if total < 5:
        winner_label = "insufficient"
    elif lb_claude >= 0.7:
        winner_label = "claude"
    elif lb_ollama >= 0.7:
        winner_label = "ollama"
    else:
        winner_label = "tie"
    return {
        "wins_claude": wins_claude,
        "wins_ollama": wins_ollama,
        "wins_other": wins_other,
        "total": total,
        "wilson_lb_claude": round(lb_claude, 3),
        "wilson_lb_ollama": round(lb_ollama, 3),
        "winner": winner_label,
    }


async def get_paper_dois_for_digest(
    db: aiosqlite.Connection, digest_id: int
) -> dict[int, str]:
    """Return {item_id: external_id} for paper posts in a digest.

    Used by web UI to inject paper action buttons (TLDR, Similar, Graph)
    only on digest entries that are academic papers (arxiv/biorxiv/academic).
    """
    async with db.execute(
        "SELECT di.item_id, rp.external_id "
        "FROM digest_items di "
        "JOIN raw_posts rp ON rp.id = di.post_id "
        "WHERE di.digest_id = ? "
        "AND rp.platform IN ('arxiv', 'biorxiv', 'academic', 'local_pdf')",
        (digest_id,),
    ) as cur:
        rows = await cur.fetchall()
    return {row[0]: row[1] for row in rows if row[1]}


# ---------------------------------------------------------------------------
# Brainstorm-review pipeline CRUD (Phase 2, 2026-05-18)
# ---------------------------------------------------------------------------

async def insert_brainstorm_decision(
    db: aiosqlite.Connection,
    *,
    draft_id: int,
    idea_num: int,
    review_run_id: str,
    project: str,
    title: str | None,
    verdict: str,
    score: int,
    summary_30w: str | None,
    risks_json: str | None,
    shipped_overlap: str | None,
    next_step: str | None,
    raw_json: str,
) -> int | None:
    """Insert one evaluated brainstorm idea. Returns row id, or None if a row
    for (draft_id, idea_num, review_run_id) already exists (idempotent re-runs).

    The CHECK constraints on verdict/score reject malformed input at the DB
    layer — callers can pass subagent output through without pre-validation.
    """
    cur = await db.execute(
        """
        INSERT OR IGNORE INTO brainstorm_decisions
          (draft_id, idea_num, review_run_id, project, title, verdict, score,
           summary_30w, risks_json, shipped_overlap, next_step, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (draft_id, idea_num, review_run_id, project, title, verdict, score,
         summary_30w, risks_json, shipped_overlap, next_step, raw_json),
    )
    await db.commit()
    # INSERT OR IGNORE: rowcount==0 when the UNIQUE conflict skipped the row.
    # lastrowid persists from a previous successful insert on this connection,
    # so checking rowcount is the only reliable conflict signal.
    if cur.rowcount <= 0:
        return None
    return int(cur.lastrowid) if cur.lastrowid else None


async def get_pending_brainstorm_decisions(
    db: aiosqlite.Connection,
    score_min: int = 6,
    score_max: int | None = None,
    verdicts: tuple[str, ...] | None = None,
) -> list[dict]:
    """Return brainstorm rows awaiting a user decision, ordered by score
    descending then draft_id ascending (highest-leverage first).

    Defaults to score_min=6 because <6 is auto-discarded per the pipeline
    spec — the widget surface should never show them.
    """
    where = ["user_decision IS NULL", "score >= ?"]
    params: list = [score_min]
    if score_max is not None:
        where.append("score <= ?")
        params.append(score_max)
    if verdicts:
        placeholders = ", ".join(["?"] * len(verdicts))
        where.append(f"verdict IN ({placeholders})")
        params.extend(verdicts)
    sql = (
        "SELECT id, draft_id, idea_num, review_run_id, project, title, "
        "verdict, score, summary_30w, risks_json, shipped_overlap, next_step, "
        "raw_json, created_at "
        "FROM brainstorm_decisions WHERE "
        + " AND ".join(where)
        + " ORDER BY score DESC, draft_id ASC, idea_num ASC"
    )
    async with db.execute(sql, params) as cur:
        rows = await cur.fetchall()
    cols = ("id", "draft_id", "idea_num", "review_run_id", "project", "title",
            "verdict", "score", "summary_30w", "risks_json", "shipped_overlap",
            "next_step", "raw_json", "created_at")
    return [dict(zip(cols, r)) for r in rows]


async def set_brainstorm_user_decision(
    db: aiosqlite.Connection,
    decision_id: int,
    user_decision: str,
    target_path: str | None = None,
) -> bool:
    """Record a promote/drop/spike click. Returns True if the row was updated,
    False if already decided or the id does not exist.

    The guarded WHERE clause (`user_decision IS NULL`) prevents accidental
    overwrite on double-click — mirrors the chain_state guard precedent
    documented in src/worker/gate_state.py.
    """
    if user_decision not in {"promote", "drop", "spike"}:
        raise ValueError(f"invalid user_decision: {user_decision!r}")
    cur = await db.execute(
        "UPDATE brainstorm_decisions "
        "SET user_decision = ?, decided_at = datetime('now'), decision_target_path = ? "
        "WHERE id = ? AND user_decision IS NULL",
        (user_decision, target_path, decision_id),
    )
    await db.commit()
    return cur.rowcount > 0


async def count_brainstorm_decisions(
    db: aiosqlite.Connection,
    review_run_id: str | None = None,
) -> dict[str, int]:
    """Return {verdict: count} for one run (or all runs combined if None).

    Used by the dashboard widget header and by tests verifying weekly
    review job emitted the expected row counts.
    """
    sql = "SELECT verdict, COUNT(*) FROM brainstorm_decisions"
    params: tuple = ()
    if review_run_id is not None:
        sql += " WHERE review_run_id = ?"
        params = (review_run_id,)
    sql += " GROUP BY verdict"
    async with db.execute(sql, params) as cur:
        rows = await cur.fetchall()
    return {row[0]: int(row[1]) for row in rows}


async def cleanup_scorer_divergence_orphans(db: aiosqlite.Connection) -> int:
    """Delete `scorer_divergence` rows whose `raw_post_id` no longer exists in
    `raw_posts`. Returns the number of rows deleted.

    Phase 52 / WR-02 (2026-05-18): explicit cleanup helper to replace the
    decorative FK CASCADE that never fires under aiosqlite's default
    `PRAGMA foreign_keys=OFF`. Any caller that deletes `raw_posts` rows
    (e.g. retention pruning, manual ops, future migration scripts) should
    run this helper afterwards to keep the rollup table consistent with
    the calibration helpers in `src/eval/non_identifiability.py`.

    Idempotent; safe to call when no orphans exist (returns 0). Commits.
    """
    async with db.execute(
        "DELETE FROM scorer_divergence "
        "WHERE raw_post_id NOT IN (SELECT id FROM raw_posts)"
    ) as cur:
        deleted = cur.rowcount
    await db.commit()
    return int(deleted or 0)
