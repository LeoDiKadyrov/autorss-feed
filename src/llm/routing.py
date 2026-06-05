"""Per-category curator backend routing.

Reads config/curator_routing.yaml. CURATOR_BACKEND env var overrides all.
Falls back to 'ollama' if config is missing or a category is unlisted.

Phase 43 (999.21) adds per-backend max input token caps + truncation helper +
process-level truncation counter (reset/get/incr) — surfaced by run_pipeline
in OK log line as `truncation_count=N`.
"""
from __future__ import annotations

import os
from pathlib import Path

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "curator_routing.yaml"
)
_cached: dict | None = None
# LR-02 (Phase 61 review): cached mtime of _CONFIG_PATH. _load() re-reads when
# mtime changes — operator can edit daily_usd_caps mid-pipeline (and usd_per_call
# from MR-02) without restarting the long-running scheduled job. None = no
# successful load yet; -1.0 = file missing on last load.
_cached_mtime: float | None = None

# Phase 43: char-to-token coefficient (matches Phase 42).
_CHARS_PER_TOKEN = 4

# Default caps when yaml missing the entry. Conservative for safety.
_DEFAULT_MAX_INPUT_TOKENS = {
    "ollama": 24000,
    "claude": 180000,
    "claude-batch": 180000,
}
_FALLBACK_CAP_TOKENS = 24000  # used for unknown backends

_TRUNCATION_SUFFIX = "\n\n...[TRUNCATED]"

# Process-level counter. Reset at run start; read at run end for OK log.
_truncation_count = 0


def _load() -> dict:
    """Read curator_routing.yaml with mtime-based cache invalidation.

    LR-02 (Phase 61 review): pre-fix the cache was sticky for the lifetime of
    the process. Operator edits to daily_usd_caps / usd_per_call mid-pipeline
    were ignored until restart — a real pain for the long-running scheduled
    pipeline. Now we stat() the config file and re-read if mtime changed.
    Stat failures (missing file, permission) fall through to the cached value
    if present, else empty dict.
    """
    global _cached, _cached_mtime
    try:
        current_mtime = _CONFIG_PATH.stat().st_mtime
    except OSError:
        current_mtime = -1.0  # file missing / unreadable

    if _cached is not None and _cached_mtime == current_mtime:
        return _cached

    try:
        import yaml  # type: ignore[import]
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            _cached = yaml.safe_load(f) or {}
        _cached_mtime = current_mtime
    except Exception:
        # Parse / IO failure — keep prior cached value if any, else empty dict.
        if _cached is None:
            _cached = {}
            _cached_mtime = current_mtime
    return _cached


def _reset_cache() -> None:
    """Test-only: clear yaml cache + Phase 78 centroid/chroma module caches."""
    global _cached, _cached_mtime
    global _centroid_cache, _centroid_call_count, _last_compute_at
    global _chroma_collection_cache, _chroma_collection_attempted
    _cached = None
    _cached_mtime = None
    _centroid_cache = None
    _centroid_call_count = 0
    _last_compute_at = -(10**9)
    _chroma_collection_cache = None
    _chroma_collection_attempted = False
    # Phase 83-01: also flush spill state from the global reset hook so test
    # helpers that call _reset_cache get a fully clean slate. IN-01 (Phase 83
    # review): `_reset_spill_state` is defined later in the module but only
    # invoked at runtime (post-load), so the NameError guard was dead code.
    _reset_spill_state()
    # Phase 98-02: also flush local_scorer_fallback caches via lazy import to
    # avoid a hard module dependency (the fallback module imports routing-free
    # via `should_use_local_fallback`, but importing it at module-top here
    # would create the cycle).
    try:
        from src.llm import local_scorer_fallback as _fb
        _fb._reset_cache()
    except Exception:
        pass


def get_backend_for_category(category: str | None) -> str:
    """Return backend name for a source category.

    Priority:
      1. CURATOR_BACKEND env var (global override — if set, always wins)
      2. Phase 61 / COST-CAP-01: per-category daily USD cap — if today's
         claude spend on this category >= daily_usd_caps[cat], return
         'ollama' regardless of yaml routing. Fail-soft: cost_cap import
         or check error → fall through to yaml (counter stays 0).
      3. Per-category entry in curator_routing.yaml
      4. default: entry in curator_routing.yaml
      5. 'ollama' hard fallback
    """
    env = os.environ.get("CURATOR_BACKEND", "")
    if env:
        # Phase 61 HR-01 fix: any non-empty CURATOR_BACKEND (incl. claude-batch)
        # is an explicit operator override and must SKIP the cap-check entirely
        # — operator intent wins over budget guard. Pre-fix this branch
        # excluded 'claude-batch' so cap could silently downgrade it to ollama.
        return env

    # Phase 61 (COST-CAP-01 / NFR-V23-02): per-category daily USD cap.
    # If today's claude spend on this category exceeds daily_usd_caps[cat],
    # silently fall back to ollama. is_cap_exceeded is itself fail-soft (DB
    # error → returns False → no fallback) so this can't raise — but we wrap
    # in try/except as defence-in-depth against import error / unforeseen bug.
    try:
        from src.llm.cost_cap import is_cap_exceeded
        if is_cap_exceeded(category or "other"):
            return "ollama"
    except Exception:
        pass

    cfg = _load()
    categories: dict = cfg.get("categories", {})
    default: str = cfg.get("default", "ollama")
    return categories.get(category or "other", default)


# --- Phase 106 / SELF-TUNE-01: per-category curator threshold resolver ---


def get_threshold_for_category(category: str | None) -> int:
    """Resolve curator threshold for a category.

    Precedence:
      1. yaml.thresholds[category] (per-category override; integer-typed)
      2. yaml.thresholds['other'] (catch-all if category not listed)
      3. env CURATOR_THRESHOLD (read INSIDE this function — module-level
         binding gotcha per CLAUDE.md reference_module_level_env_binding)
      4. 75 (final fallback)

    Invalid yaml entries (non-int) silently fall back to env with a WARNING log.
    Never raises — curator hot path must never crash on malformed yaml.
    """
    env_default = int(os.environ.get("CURATOR_THRESHOLD", "75"))
    cfg = _load()
    thresholds = cfg.get("thresholds") or {}
    cat_key = category or "other"
    raw = thresholds.get(cat_key)
    if raw is None and category is not None:
        raw = thresholds.get("other")
    if raw is None:
        return env_default
    try:
        return int(raw)
    except (TypeError, ValueError):
        import logging
        logging.getLogger(__name__).warning(
            "routing: invalid threshold %r for category %s; "
            "falling back to env %d",
            raw, category, env_default,
        )
        return env_default


# --- Phase 43 (999.21): max input tokens + truncation ---


def get_max_input_tokens(backend: str) -> int:
    """Return token cap for a backend.

    Priority:
      1. yaml max_input_tokens[backend]
      2. _DEFAULT_MAX_INPUT_TOKENS[backend]
      3. _FALLBACK_CAP_TOKENS (unknown backend)
    """
    cfg = _load()
    caps = cfg.get("max_input_tokens") or {}
    raw = caps.get(backend)
    if isinstance(raw, int) and raw > 0:
        return raw
    return _DEFAULT_MAX_INPUT_TOKENS.get(backend, _FALLBACK_CAP_TOKENS)


def truncate_for_backend(text: str, backend: str) -> tuple[str, bool]:
    """Return (text, was_truncated). Char-budget = cap_tokens × 4.

    Never raises. Empty / None input returns ("", False).
    Increments module truncation counter on every truncation.
    """
    if not text:
        return "", False
    cap_tokens = get_max_input_tokens(backend)
    char_budget = cap_tokens * _CHARS_PER_TOKEN
    if len(text) <= char_budget:
        return text, False
    # Reserve room for the suffix; cut at char_budget - len(suffix)
    keep = char_budget - len(_TRUNCATION_SUFFIX)
    if keep < 0:
        keep = char_budget  # extreme small cap — drop suffix
        truncated = text[:keep]
    else:
        truncated = text[:keep] + _TRUNCATION_SUFFIX
    _incr_truncation_count()
    return truncated, True


def get_truncation_count() -> int:
    return _truncation_count


def reset_truncation_count() -> None:
    global _truncation_count
    _truncation_count = 0


def _incr_truncation_count() -> None:
    global _truncation_count
    _truncation_count += 1


# --- Phase 78 / ROUTE-FEAT-01: hidden-state feature extraction ----------------
#
# extract_features(post) → 5-tuple consumed by P79's sigmoid routing gate.
# Zero LLM burn: reuses cached bge-m3 embeddings (ChromaDB `notebook` collection)
# and raw_posts.simhash. All chroma/db lookups are graceful — any failure
# (missing post_id, chroma down, empty curated window) yields None for that
# feature without raising.

import sqlite3 as _sqlite3
import threading
from typing import Any

_DEFAULT_DB_PATH = "curator.db"
_CHROMA_PATH = ".chromadb"
_CHROMA_COLLECTION = "notebook"
_SIMHASH_WINDOW = 50  # last-N curated posts for Hamming-min lookup
_CENTROID_WINDOW = 200  # last-N curated posts for centroid mean
_CENTROID_RECOMPUTE_EVERY = 20  # recompute every Nth routing call

# Sentinel value cached when _compute_centroid fails — distinguishes "never
# computed" (None) from "computed and got None" (miss). Prevents WR-02
# thundering-herd against chroma when centroid compute persistently fails.
_CENTROID_MISS: Any = object()

_centroid_cache: Any = None  # np.ndarray | _CENTROID_MISS | None
_centroid_call_count = 0
_last_compute_at = -(10**9)  # WR-02: track last-attempt counter
_centroid_lock = threading.Lock()  # WR-01: guard mutation of cache + counter


def _get_attr(post: Any, key: str, default: Any = None) -> Any:
    """Read `key` from post supporting dict, sqlite3.Row, or arbitrary object."""
    if post is None:
        return default
    if isinstance(post, dict):
        return post.get(key, default)
    # sqlite3.Row supports __getitem__ but not .get
    try:
        return post[key]
    except (KeyError, IndexError, TypeError):
        pass
    return getattr(post, key, default)


def _hamming(a: int, b: int) -> int:
    return bin((a ^ b) & 0xFFFFFFFFFFFFFFFF).count("1")


def _simhash_min_hamming(
    simhash_val: int | None,
    db_path: str,
    post_id: int | None = None,
) -> int | None:
    """Return min Hamming distance of `simhash_val` vs last N curated simhashes.

    None if simhash_val is None, no curated window exists, or any DB failure.

    IN-01: excludes `post_id` from the candidate set so a re-extraction call
    after the post is marked curated (planned for P79 backfill) does not
    Hamming-to-self → 0. Curator pre-status-flip use is unaffected (post_id
    is not yet in the curated window).
    """
    if simhash_val is None:
        return None
    try:
        conn = _sqlite3.connect(db_path)
        try:
            if post_id is None:
                cur = conn.execute(
                    "SELECT simhash FROM raw_posts WHERE status='curated' "
                    "AND simhash IS NOT NULL ORDER BY id DESC LIMIT ?",
                    (_SIMHASH_WINDOW,),
                )
            else:
                cur = conn.execute(
                    "SELECT simhash FROM raw_posts WHERE status='curated' "
                    "AND simhash IS NOT NULL AND id != ? "
                    "ORDER BY id DESC LIMIT ?",
                    (post_id, _SIMHASH_WINDOW),
                )
            rows = [r[0] for r in cur.fetchall()]
        finally:
            conn.close()
    except Exception:
        return None
    if not rows:
        return None
    try:
        return min(_hamming(int(simhash_val), int(s)) for s in rows if s is not None)
    except Exception:
        return None


def _cosine(a: Any, b: Any) -> float | None:
    """Cosine similarity. Returns None if norms are zero or numpy unavailable."""
    try:
        import numpy as np
        av = np.asarray(a, dtype=float)
        bv = np.asarray(b, dtype=float)
        na = float(np.linalg.norm(av))
        nb = float(np.linalg.norm(bv))
        if na == 0.0 or nb == 0.0:
            return None
        return float(np.dot(av, bv) / (na * nb))
    except Exception:
        return None


_chroma_collection_cache: Any = None  # IN-02: memoized handle
_chroma_collection_attempted: bool = False


def _chroma_collection():
    """Return ChromaDB notebook collection or None if unavailable.

    IN-02: memoize the handle at module level so a single routing call no
    longer opens two PersistentClients (one in _embed_cosine_for_post, one
    in _compute_centroid). Lock-guarded for the WR-01 thread-safety contract.
    Failure is cached as None until process restart — chroma availability is
    not expected to flap mid-process; the existing fail-soft path tolerates
    permanent miss.
    """
    global _chroma_collection_cache, _chroma_collection_attempted
    with _centroid_lock:
        if _chroma_collection_attempted:
            return _chroma_collection_cache
    try:
        import chromadb  # type: ignore[import]
        client = chromadb.PersistentClient(path=_CHROMA_PATH)
        coll = client.get_collection(_CHROMA_COLLECTION)
    except Exception:
        coll = None
    with _centroid_lock:
        _chroma_collection_cache = coll
        _chroma_collection_attempted = True
    return coll


def _compute_centroid(db_path: str) -> Any:
    """Mean embedding over last N curated posts. None on any failure."""
    try:
        import numpy as np
        conn = _sqlite3.connect(db_path)
        try:
            cur = conn.execute(
                "SELECT id FROM raw_posts WHERE status='curated' "
                "ORDER BY id DESC LIMIT ?",
                (_CENTROID_WINDOW,),
            )
            ids = [str(r[0]) for r in cur.fetchall()]
        finally:
            conn.close()
        if not ids:
            return None
        coll = _chroma_collection()
        if coll is None:
            return None
        got = coll.get(ids=ids, include=["embeddings"])
        embs = got.get("embeddings") if got else None
        if not embs:
            return None
        arr = np.asarray([e for e in embs if e is not None], dtype=float)
        if arr.size == 0:
            return None
        return arr.mean(axis=0)
    except Exception:
        return None


def _get_centroid(db_path: str) -> Any:
    """Lazy centroid with periodic recompute every N calls.

    WR-01: thread-safe via _centroid_lock — async pipeline + uvicorn web
    handler can both hit this concurrently; the lock prevents lost increments
    and racing _compute_centroid calls (each opening its own chroma client).

    WR-02: tracks _last_compute_at so a persistently-failing centroid (chroma
    down, empty curated window) caches the miss sentinel and respects the
    "every 20 calls" recompute interval — pre-fix, a None cache thundered
    against chroma on every call when degraded.
    """
    global _centroid_cache, _centroid_call_count, _last_compute_at
    with _centroid_lock:
        _centroid_call_count += 1
        need = (
            _centroid_cache is None
            or (_centroid_call_count - _last_compute_at) >= _CENTROID_RECOMPUTE_EVERY
        )
        if need:
            _last_compute_at = _centroid_call_count
    if need:
        new = _compute_centroid(db_path)
        with _centroid_lock:
            _centroid_cache = new if new is not None else _CENTROID_MISS
    with _centroid_lock:
        cached = _centroid_cache
    if cached is _CENTROID_MISS:
        return None
    return cached


def _embed_cosine_for_post(post_id: int | None, db_path: str) -> float | None:
    """Cosine of this post's embedding vs centroid. Graceful None on any miss."""
    if post_id is None:
        return None
    try:
        coll = _chroma_collection()
        if coll is None:
            return None
        got = coll.get(ids=[str(post_id)], include=["embeddings"])
        embs = got.get("embeddings") if got else None
        if not embs or embs[0] is None:
            return None
        centroid = _get_centroid(db_path)
        if centroid is None:
            return None
        return _cosine(embs[0], centroid)
    except Exception:
        return None


def extract_features(
    post: Any,
    db_path: str | None = None,
) -> tuple[int, int, int | None, float | None, str | None]:
    """Return (post_length, has_canonical_url, simhash_dist, embed_cosine, source_category).

    All four numeric features fail-soft to None on lookup errors:
      - simhash_dist: None when post.simhash is None or no curated window exists
      - embed_cosine: None when chroma is unavailable or post_id is missing
        from the cached collection (P75 chroma↔post_id mismatch caveat)

    source_category is a passthrough from `post` (curator already joins
    sources.category via the unprocessed-fetch query).
    """
    db = db_path or _DEFAULT_DB_PATH
    raw_text = _get_attr(post, "raw_text", "") or ""
    canonical_url = _get_attr(post, "canonical_url")
    simhash_val = _get_attr(post, "simhash")
    post_id = _get_attr(post, "id")
    source_category = _get_attr(post, "source_category")

    post_length = len(raw_text)
    has_canonical_url = 1 if canonical_url else 0
    simhash_dist = _simhash_min_hamming(simhash_val, db, post_id=post_id)
    embed_cosine = _embed_cosine_for_post(post_id, db)

    return (post_length, has_canonical_url, simhash_dist, embed_cosine, source_category)


def persist_features(
    post_id: int,
    features: tuple[int, int, int | None, float | None, str | None],
    db_path: str | None = None,
) -> None:
    """INSERT OR REPLACE into routing_features. Logs (not silent) on DB error.

    WR-03: sync sqlite3 writer in an async pipeline can collide with aiosqlite
    holding an exclusive lock (see reference_sqlite_db_locked_concurrent.md).
    WAL mode is enabled at table-creation time in migrate_db; this writer still
    logs on any failure so dropped feature rows surface in stderr instead of
    silently degrading downstream P79 training data.
    """
    db = db_path or _DEFAULT_DB_PATH
    try:
        conn = _sqlite3.connect(db)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO routing_features "
                "(post_id, post_length, has_canonical_url, simhash_dist, "
                " embed_cosine, source_category) VALUES (?, ?, ?, ?, ?, ?)",
                (post_id, features[0], features[1], features[2], features[3], features[4]),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "persist_features failed for post_id=%s: %s", post_id, e
        )
        return


def route_curator(
    post: Any = None,
    features: tuple | None = None,  # P78: accepted but unused; P79's sigmoid gate consumes
    category: str | None = None,
) -> str:
    """Backward-compat dispatcher.

    P78 still routes by category only — the `features` kwarg is reserved for
    P79's sigmoid gate. If `category` is None and `post` is provided, read
    source_category off the post. Always delegates to get_backend_for_category.
    """
    if category is None and post is not None:
        category = _get_attr(post, "source_category")
    return get_backend_for_category(category)


# --- Phase 79-03 / LEARN-ROUTE-01: env-gated A/B harness ---------------------
#
# `route_curator_ab(post)` is the new entry point for curator wiring (P79-04+).
# Default (env unset or 'off') it is a pure pass-through to `route_curator` and
# returns cohort=None — zero behavioural drift from P78. When
# LEARNABLE_ROUTER=on, it deterministically splits posts 50/50 by md5(post_id)
# (avoiding Python's randomised hash); cohort A consults LearnableRouter and
# picks argmax over the three curator backends; cohort B uses yaml. Either
# cohort returns its tag so the caller persists it to curation_logs.cohort.
#
# Insufficient-data fallback: if no weights file exists yet, or every backend
# score is None, cohort A silently falls back to yaml — the tag is still 'A'
# so Wilson LB compare can later detect that the learnable arm degenerated to
# yaml (and didn't, e.g., land an uplift purely from chance).

import hashlib

_ROUTER_BACKENDS: tuple[str, ...] = ("ollama", "claude", "claude-batch")
_ROUTER_WEIGHTS_DIR: Any = Path(__file__).resolve().parent.parent.parent / "data"
_ROUTER_WEIGHTS_GLOB = "learnable_router_v*.json"

_learnable_router_cache: Any = None
_learnable_router_attempted: bool = False


def _reset_learnable_router_cache() -> None:
    """Test-only: clear cached router instance + attempt flag."""
    global _learnable_router_cache, _learnable_router_attempted, _learnable_router_path
    _learnable_router_cache = None
    _learnable_router_attempted = False
    _learnable_router_path = None


def _latest_router_weights_path() -> Path | None:
    """Return newest `data/learnable_router_v*.json` by version suffix, else None."""
    try:
        candidates = list(Path(_ROUTER_WEIGHTS_DIR).glob(_ROUTER_WEIGHTS_GLOB))
    except OSError:
        return None
    if not candidates:
        return None

    def _version_key(p: Path) -> int:
        # filename: learnable_router_v{n}.json — extract n; fallback to mtime ordering
        stem = p.stem  # learnable_router_v3
        try:
            return int(stem.rsplit("_v", 1)[-1])
        except (ValueError, IndexError):
            return -1

    return max(candidates, key=lambda p: (_version_key(p), p.stat().st_mtime))


_learnable_router_path: Any = None  # last weights path successfully loaded


def _get_learnable_router() -> Any:
    """Lazy-load LearnableRouter from latest weights file. None on miss.

    WR-04 (P79 review): re-stat the weights dir on each call and invalidate
    cache when a newer-versioned file appears. Pre-fix, the long-running
    scheduled pipeline + uvicorn process kept the cache value (often `None`)
    forever — the Sunday trainer's fresh weights file never picked up.

    WR-05 (P79 review): thread-safe via `_centroid_lock` (re-using the
    pattern already proven for `_get_centroid` / `_chroma_collection`).
    Concurrent uvicorn + pipeline callers no longer race the cache mutation.

    WR-06 (P79 review): log load failures at WARNING. A corrupted weights file
    or schema-version mismatch was indistinguishable from "no file yet" pre-fix.
    """
    global _learnable_router_cache, _learnable_router_attempted, _learnable_router_path
    path = _latest_router_weights_path()
    with _centroid_lock:
        # Fast path: previously attempted and weights file path unchanged.
        if _learnable_router_attempted and path == _learnable_router_path:
            return _learnable_router_cache

    if path is None:
        with _centroid_lock:
            _learnable_router_cache = None
            _learnable_router_path = None
            _learnable_router_attempted = True
        return None

    try:
        from src.llm.learnable_router import LearnableRouter
        router = LearnableRouter()
        router.load(str(path))
        with _centroid_lock:
            _learnable_router_cache = router
            _learnable_router_path = path
            _learnable_router_attempted = True
        return router
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "LearnableRouter load failed from %s: %s", path, e
        )
        with _centroid_lock:
            _learnable_router_cache = None
            _learnable_router_path = path  # remember path so we don't thrash on retry
            _learnable_router_attempted = True
        return None


def _cohort_for_post_id(post_id: Any) -> str:
    """Deterministic 50/50 split via md5 (avoids PYTHONHASHSEED randomisation)."""
    h = hashlib.md5(str(post_id).encode("utf-8")).hexdigest()
    return "A" if int(h, 16) % 2 == 0 else "B"


def _features_dict_from_post(post: Any, db_path: str | None = None) -> dict[str, Any]:
    """Build the features dict consumed by LearnableRouter.score_backend.

    CR-02 (P79 review): train/serve feature consistency. Trainer reads
    `simhash_dist` + `embed_cosine` from the `routing_features` table (P78
    populated by `persist_features`). The curator's raw_posts dict does NOT
    carry these columns at inference time — `_get_attr(post, "simhash_dist")`
    silently returned None for every production post, collapsing the 4-numeric
    feature space to (post_length, has_canonical_url, category one-hot) only.

    Resolution: when the post dict is missing the derived features (the common
    production case), look them up in `routing_features` by post_id. If the
    row is absent (extract_features hasn't been called for this post yet),
    fall through to extract_features on-demand so the inference-time view
    matches what the trainer saw. Either path can still legitimately return
    None — score_backend tolerates it.
    """
    post_length = len(_get_attr(post, "raw_text", "") or "")
    has_canonical_url = 1 if _get_attr(post, "canonical_url") else 0
    simhash_dist = _get_attr(post, "simhash_dist")
    embed_cosine = _get_attr(post, "embed_cosine")

    if simhash_dist is None and embed_cosine is None:
        post_id = _get_attr(post, "id")
        db = db_path or _DEFAULT_DB_PATH
        # Pull from routing_features table (cheap point lookup; matches trainer view).
        if post_id is not None:
            try:
                conn = _sqlite3.connect(db)
                try:
                    cur = conn.execute(
                        "SELECT simhash_dist, embed_cosine FROM routing_features "
                        "WHERE post_id = ?",
                        (post_id,),
                    )
                    row = cur.fetchone()
                finally:
                    conn.close()
                if row is not None:
                    simhash_dist = row[0] if simhash_dist is None else simhash_dist
                    embed_cosine = row[1] if embed_cosine is None else embed_cosine
            except Exception:
                pass  # fail-soft to extract_features below
        # If still missing, compute on demand (matches trainer's view exactly).
        if (simhash_dist is None and embed_cosine is None) and post_id is not None:
            try:
                _, _, sd, ec, _ = extract_features(post, db_path=db)
                if simhash_dist is None:
                    simhash_dist = sd
                if embed_cosine is None:
                    embed_cosine = ec
            except Exception:
                pass

    return {
        "post_length": post_length,
        "has_canonical_url": has_canonical_url,
        "simhash_dist": simhash_dist,
        "embed_cosine": embed_cosine,
    }


def route_curator_ab(
    post: Any,
    category: str | None = None,
) -> tuple[str, str | None]:
    """Return (backend, cohort).

    cohort is None when LEARNABLE_ROUTER is off (default) — caller treats this
    exactly like the pre-P79 yaml path. When LEARNABLE_ROUTER=on:
      - cohort 'A' tries LearnableRouter; argmax over backend scores
      - cohort 'B' uses yaml
      - cohort 'A' with no weights / all-None scores → yaml fallback (tag stays 'A')
    """
    if category is None:
        category = _get_attr(post, "source_category")

    # CR-01 (P79 review): honour CURATOR_BACKEND env override BEFORE the cohort
    # split. Pre-fix, cohort B fell through to get_backend_for_category() (which
    # honours the env) while cohort A always ran the learnable argmax — meaning
    # with `CURATOR_BACKEND=ollama` set (the standing operator override per
    # reference_curator_routing_claude_limits.md), the two arms routed on
    # different policies and the downstream Wilson LB compare measured "with vs
    # without env override" rather than "learnable vs yaml". Now: any non-empty
    # global override defeats the A/B harness entirely (cohort=None); both arms
    # see whatever the operator pinned and are parity-comparable.
    env_override = os.environ.get("CURATOR_BACKEND", "")
    if env_override:
        return env_override, None

    env = (os.environ.get("LEARNABLE_ROUTER") or "").strip().lower()
    if env != "on":
        return get_backend_for_category(category), None

    post_id = _get_attr(post, "id")
    cohort = _cohort_for_post_id(post_id)

    if cohort == "B":
        return get_backend_for_category(category), "B"

    # cohort A: consult LearnableRouter; fall back to yaml on miss
    router = _get_learnable_router()
    if router is None:
        return get_backend_for_category(category), "A"

    features = _features_dict_from_post(post)
    best_backend: str | None = None
    best_score = -1.0
    for backend in _ROUTER_BACKENDS:
        score = router.score_backend(features, backend, category)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_backend = backend

    if best_backend is None:
        return get_backend_for_category(category), "A"
    return best_backend, "A"


# --- Phase 83-01 / SPILL-ROUTE-01: substrate-conditional ollama→claude-batch spill ---
#
# `should_spill(queue_depth, recent_latencies)` returns True when the curator
# should divert an ollama-bound post to claude-batch under load. Off by default
# (SPILL_ROUTING_MODE != 'on' short-circuits to False — zero behavioural drift
# until the operator opts in). Two triggers, evaluated in order:
#   1. queue_depth > SPILL_QUEUE_THRESHOLD (default 200)
#   2. p95(recent_latencies) > SPILL_LATENCY_P95_MS (default 30000), requires
#      ≥ _SPILL_MIN_SAMPLES_FOR_P95 samples to avoid single-call false positives
# After any True result, _SPILL_COOLDOWN_SEC of lockout follows — subsequent
# calls return False regardless of inputs. T-83-02 mitigation: prevents
# oscillation between ollama and claude-batch under sustained load.
#
# Env vars are read INSIDE should_spill (CLAUDE.md module-level-binding gotcha).

import collections
import math
import time as _spill_time
from typing import Callable, Sequence as _Sequence

_SPILL_COOLDOWN_SEC = 300
_SPILL_LATENCIES_WINDOW = 50
_SPILL_MIN_SAMPLES_FOR_P95 = 20

# WR-01 (Phase 83 review): bind callable indirection so tests can patch
# `src.llm.routing._now` without mutating stdlib `time.monotonic` globally
# (which races asyncio + other importers in parallel runs). The module still
# exposes `time` as an alias for backward compatibility with legacy tests.
time = _spill_time
_now: Callable[[], float] = _spill_time.monotonic

_spill_latencies: collections.deque = collections.deque(maxlen=_SPILL_LATENCIES_WINDOW)
_last_spill_ts: float | None = None
_spill_lock = threading.Lock()


def _reset_spill_state() -> None:
    """Test-only: clear rolling deque + cooldown timestamp."""
    global _last_spill_ts
    with _spill_lock:
        _spill_latencies.clear()
        _last_spill_ts = None


def record_ollama_latency(elapsed_ms: float) -> None:
    """Thread-safe append to the rolling latency window (last 50 calls)."""
    with _spill_lock:
        _spill_latencies.append(float(elapsed_ms))


def should_spill(
    queue_depth: int,
    recent_latencies: _Sequence[float] | None = None,
    force_spill: bool = False,
) -> bool:
    """Return True when the curator should spill ollama→claude-batch.

    See module-level docstring above for trigger semantics + cooldown.

    WR-02 (Phase 83 review): `force_spill=True` bypasses the cooldown check —
    caller passes this for every post in the same pipeline run AFTER the first
    trigger fires. Pre-fix the cooldown caused the spill to fire for exactly 1
    post per 5min under sustained load (first post → cooldown set → remaining
    N-1 posts hit cooldown branch → all routed back to ollama). The arming
    timestamp is still updated on a forced spill so cross-run cooldown holds.
    """
    global _last_spill_ts

    # Env reads INSIDE the function (CLAUDE.md gotcha).
    mode = (os.environ.get("SPILL_ROUTING_MODE") or "").strip().lower()
    if mode != "on":
        return False

    # Cooldown lockout — even huge queue / slow latencies stay quiet.
    # WR-01 (Phase 83 review): use `_now` indirection (not `time.monotonic`)
    # so tests don't mutate the stdlib module.
    now = _now()
    if not force_spill:
        with _spill_lock:
            last = _last_spill_ts
        if last is not None and (now - last) < _SPILL_COOLDOWN_SEC:
            return False

    try:
        queue_threshold = int(os.environ.get("SPILL_QUEUE_THRESHOLD", "200"))
    except ValueError:
        queue_threshold = 200
    try:
        latency_threshold = int(os.environ.get("SPILL_LATENCY_P95_MS", "30000"))
    except ValueError:
        latency_threshold = 30000

    triggered = False

    # Trigger 1: queue depth (strict >).
    if queue_depth > queue_threshold:
        triggered = True

    # Trigger 2: p95 of latency window (strict >). Need a snapshot of the deque
    # if caller passed None (production curator wiring).
    if not triggered:
        if recent_latencies is None:
            with _spill_lock:
                lat_snapshot = list(_spill_latencies)
        else:
            lat_snapshot = list(recent_latencies)
        if len(lat_snapshot) >= _SPILL_MIN_SAMPLES_FOR_P95:
            ordered = sorted(lat_snapshot)
            # WR-04 (Phase 83 review): ceil-1 convention so p95 is consistent
            # across N. Pre-fix `int(N*0.95)` gave p100 for N=20 (idx 19) and
            # p94 for N=50 (idx 47). Now: N=20 → idx 18, N=50 → idx 47.
            n = len(ordered)
            idx = min(int(math.ceil(0.95 * n)) - 1, n - 1)
            if idx < 0:
                idx = 0
            p95 = ordered[idx]
            if p95 > latency_threshold:
                triggered = True

    if triggered:
        with _spill_lock:
            _last_spill_ts = now
        return True

    return False


# --- Phase 98-02 / LOCAL-SCORER-01: local Conv+Transformer fallback ----------
#
# Adds a final pre-fallback branch consulted by curator wiring (additive — does
# NOT modify get_backend_for_category). Per D-08, the env gate
# LOCAL_SCORER_FALLBACK defaults off so behaviour is byte-identical until the
# operator opts in. Per D-09, fallback only fires when BOTH primary backends
# (ollama + claude) probe dead AND a local weights file is loadable. The
# operator override (CURATOR_BACKEND) bypasses fallback entirely — explicit
# operator intent always wins over the degradation path.


def route_curator_with_local_fallback(category: str | None) -> str:
    """Return 'local' when fallback should fire, else the normal primary backend.

    Resolution order:
      1. CURATOR_BACKEND set → defer to get_backend_for_category (override wins)
      2. consult local_scorer_fallback.should_use_local_fallback(primary)
         - True  → return 'local'
         - False → return primary
      3. any exception in the fallback path → return primary (fail-soft)
    """
    if os.environ.get("CURATOR_BACKEND", ""):
        return get_backend_for_category(category)
    primary = get_backend_for_category(category)
    try:
        from src.llm.local_scorer_fallback import should_use_local_fallback
        if should_use_local_fallback(primary):
            return "local"
    except Exception:
        pass
    return primary
