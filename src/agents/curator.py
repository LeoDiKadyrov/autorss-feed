import aiosqlite
import datetime
import logging
import os
import subprocess
import time as _curator_time  # Phase 83-01: per-post ollama latency wrap
from src.database.client import (
    get_unprocessed_posts,
    update_post_status,
    insert_curation_log,
    get_feedback_summary,
    insert_post_topics,
)
from src.llm.ollama import score_content
from src.llm import claude_scorer
from src.llm.claude_batch_scorer import score_batch, BATCH_SIZE
from src.llm import shadow_router  # phase 52 — sampling + counters + ClaudeRateLimitError
from src.curator.snippet_quality import (  # phase 59 / SNIPPET-01
    is_low_quality_snippet,
    _incr_snippet_rejected,
)
from src.curator.giveaway_filter import (  # phase 60 / GIVEAWAY-01
    is_giveaway,
    _incr_giveaway_rejected,
)
from src.eval.factor_buckets import compute_bucket  # phase 70 / FACTOR-BUCKET-01
from src.llm import prompt_router  # phase 80 / PROMPT-ABLEND-01
from src.llm.routing import get_threshold_for_category  # phase 106 / SELF-TUNE-01
from src.llm.cluster_prefilter import cluster_posts, _decode_embedding  # phase 99 / CLUSTER-PRE-01 + phase 103 / CLUSTER-FLOOR-01
import numpy as np  # phase 103 / CLUSTER-FLOOR-01 — cosine gate

logger = logging.getLogger(__name__)


# Phase 96 / IMAP-TRIAGE-01 (Plan 96-02) — per-class email triage counters.
#
# Telemetry contract: counter increments for EVERY post that arrives at the
# curator with a non-NULL email_class — independent of whether that class is
# skipped (support / personal) or proceeds to the scorer (newsletter / action).
# This way the OK-log emits the full classification distribution, not just the
# skip subset, so an operator tuning IMAP_TRIAGE_MODE can see "the classifier
# is firing but mostly returning newsletter" vs "the classifier is silent".
#
# Reset at run start by reset_email_triage_counts() (mirrors P59/P60 pattern,
# called from run_pipeline.main() before any curator work). Read fail-soft at
# OK-log emit site via get_email_triage_counts().
#
# Single-threaded contract (same as snippet_quality / giveaway_filter): curator
# runs inside one asyncio event loop; increments happen between `await` points,
# so no lock needed under cooperative scheduling. If a future refactor fans
# out via to_thread/threadpool, wrap mutations in threading.Lock().
_email_triage_counts: dict[str, int] = {
    "newsletter": 0,
    "support": 0,
    "personal": 0,
    "action": 0,
}


def get_email_triage_counts() -> dict[str, int]:
    """Return a SHALLOW COPY of the per-class triage counters (read-only API)."""
    return dict(_email_triage_counts)


def reset_email_triage_counts() -> None:
    """Zero all 4 counter keys — invoked at run start by run_pipeline.main()."""
    for k in _email_triage_counts:
        _email_triage_counts[k] = 0


# Phase 99 / CLUSTER-PRE-01 (Plan 99-02) — env-gated cluster pre-filter counter.
# Increments by (cluster.size - 1) for each multi-member cluster: the medoid
# scores normally via the per-post path, the non-medoid members are written
# via _commit_result with reason='cluster_propagated medoid=<id> ...'.
# Singletons (size=1) increment by 0 — they score via the unchanged per-post
# path. Counter is process-global, reset at run start by run_pipeline.main()
# via reset_cluster_propagated_count(); read fail-soft at OK-log emit site.
# Lockstep with _LOG_OK_RE in src/web/main.py + 4 test files per
# reference_ok_log_accretion_6_file_lockstep.md.
_cluster_propagated_count: int = 0


def get_cluster_propagated_count() -> int:
    """Read-only API for OK-log emit site (run_pipeline.main())."""
    return _cluster_propagated_count


def reset_cluster_propagated_count() -> None:
    """Zero the counter — invoked at run start by run_pipeline.main()."""
    global _cluster_propagated_count
    _cluster_propagated_count = 0


def _incr_cluster_propagated(n: int = 1) -> None:
    global _cluster_propagated_count
    _cluster_propagated_count += n


# Phase 103 / CLUSTER-FLOOR-01 — per-member cosine floor skip counter.
# Increments by 1 for each cluster member that drops OUT of the propagation
# skip-set (because its cosine to the medoid was below CLUSTER_MEMBER_COSINE_FLOOR
# OR its embedding could not be decoded). Skipped members are re-routed through
# the standard per-post curator path — they are NOT rejected. Read-only API via
# get_cluster_floor_skipped_count(); reset at run start by run_pipeline.main()
# via reset_cluster_floor_skipped_count() (same lifecycle as the propagated counter).
_cluster_floor_skipped_count: int = 0


def get_cluster_floor_skipped_count() -> int:
    """Read-only API for OK-log emit site (run_pipeline.main())."""
    return _cluster_floor_skipped_count


def reset_cluster_floor_skipped_count() -> None:
    """Zero the counter — invoked at run start by run_pipeline.main()."""
    global _cluster_floor_skipped_count
    _cluster_floor_skipped_count = 0


def _incr_cluster_floor_skipped(n: int = 1) -> None:
    global _cluster_floor_skipped_count
    _cluster_floor_skipped_count += n


# WR-04 fix (Phase 99 review): track cluster_posts exceptions separately so
# operators can distinguish "no qualifying clusters" (cluster_propagated=0,
# errors=0) from "cluster_posts raised and we silently fell through"
# (errors>=1). Surfaced via get_cluster_prefilter_errors(); OK-log
# accretion deferred to a follow-up phase to avoid touching 4 test files.
_cluster_prefilter_errors: int = 0


def get_cluster_prefilter_errors() -> int:
    return _cluster_prefilter_errors


def reset_cluster_prefilter_errors() -> None:
    global _cluster_prefilter_errors
    _cluster_prefilter_errors = 0


def _incr_cluster_prefilter_errors(n: int = 1) -> None:
    global _cluster_prefilter_errors
    _cluster_prefilter_errors += n


_V1_FALLBACK_PROFILE = """[Profile not configured]
No personal profile loaded. Set OBSIDIAN_PROFILE_DIR to a directory containing
profile markdown files (see README for expected file names).
The curator will score content on general relevance and quality only.

Default scoring rubric (0-100 relevance):
- HIGH (75-100): technically substantive, original insight, actionable, data-backed.
- MEDIUM (50-74): informational but broad or derivative.
- LOW (0-49): promotional, engagement-bait, shallow, or off-topic.

Reject (score 0-30): pure promo ("sign up", "limited spots"), motivational fluff
without data, personal lifestyle updates with no transferable insight, posts under
3 sentences with no data/link/concrete claim.
"""


def get_mcp_context() -> str:
    """Phase 7 META-01: returns Obsidian-derived profile via src.profile.loader.load_profile.

    Falls back to _V1_FALLBACK_PROFILE on D-17 conditions (path missing, critical files
    missing, parser exception). The fallback is a generic stub with no personal data;
    it is owned by load_profile() which imports the constant from this module to keep
    a single source of truth.
    """
    from src.profile.loader import load_profile
    return load_profile()


async def build_curator_context(db: aiosqlite.Connection, seal_tag: str) -> str:
    """Phase 7 META-05 + Phase 8 CUR-03: per-pipeline-run seal_tag wraps the assembled
    profile (META-01 base + Phase 5 feedback line OR Phase 8 OVERLAY block).

    Layer order (D-01):
      1. META-01 base from get_mcp_context()
      2. Overlay or Phase 5 feedback line
      3. Idea 3: PREDICTIVE FEEDBACK WEIGHTS block
      4. wrap_with_seal envelope
    """
    from src.profile.sealing import wrap_with_seal
    from src.curator.overlay import compute_overlay, render_overlay_block, _get_overlay_threshold
    from src.llm.feedback_loop import update_priors, render_weights_block
    from src.database.client import get_feedback_post_ids, mark_feedback_used, get_feedback_provenance_ratio

    # v1.3 wireheading guard: skip feedback if >50% already used_in_context
    ratio = await get_feedback_provenance_ratio(db)
    if ratio > 0.5:
        logger.warning(
            "wireheading guard triggered — %.0f%% of feedback used_in_context; "
            "skipping feedback block",
            ratio * 100,
        )
        base = get_mcp_context()
        return wrap_with_seal(base, seal_tag)

    base = get_mcp_context()
    threshold = _get_overlay_threshold()
    overlay = await compute_overlay(db, days=30, threshold=threshold)

    if overlay is not None:
        block = render_overlay_block(overlay)
        profile_text = base + block

        # Track feedback provenance in overlay path too (wireheading guard D-26)
        used_ids = await get_feedback_post_ids(db, days=7)
        await mark_feedback_used(db, used_ids)

        # Idea 3 (Predictive coding): append per-category weight adjustments
        try:
            weights = await update_priors(db, window_days=30)
            weights_block = render_weights_block(weights)
            if weights_block:
                profile_text = profile_text + weights_block
        except Exception:
            logger.exception("feedback_loop.update_priors failed — skipping weights block")

        return wrap_with_seal(profile_text, seal_tag)

    summary = await get_feedback_summary(db, days=7)
    if summary["total_up"] == 0 and summary["total_down"] == 0:
        profile_text = base
    else:
        upvoted_str = ", ".join(
            f"{cat} ({n})" for cat, n in summary["upvoted_categories"]
        ) or "none"
        downvoted_str = ", ".join(
            f"{cat} ({n})" for cat, n in summary["downvoted_categories"]
        ) or "none"
        feedback_line = (
            f"\n\nFEEDBACK (last 7 days): "
            f"upvoted {summary['total_up']} posts in {upvoted_str}. "
            f"Downvoted {summary['total_down']} posts in {downvoted_str}. "
            f"Use this signal to score similar new content."
        )
        profile_text = base + feedback_line

        # Mark these post_ids as used in context (wireheading tracking)
        used_ids = await get_feedback_post_ids(db, days=7)
        await mark_feedback_used(db, used_ids)

    # Idea 3 (Predictive coding): append per-category weight adjustments
    try:
        weights = await update_priors(db, window_days=30)
        weights_block = render_weights_block(weights)
        if weights_block:
            profile_text = profile_text + weights_block
    except Exception:
        logger.exception("feedback_loop.update_priors failed — skipping weights block")

    return wrap_with_seal(profile_text, seal_tag)


async def _score_with_backend(
    context: str, text: str, ollama_host: str, model: str, backend: str
) -> dict:
    """Dispatch scoring to the requested backend."""
    if backend == "claude":
        return await claude_scorer.generate_score(context, text)
    return await score_content(ollama_host, model, context, text)


_GENERIC_MARKERS = frozenset([
    "интересно", "relevant", "useful", "good", "важно", "подходит",
    "полезно", "нужно", "ок", "ok", "хорошо",
])

_VERIFIER_PROMPT = (
    "You are evaluating the quality of a content-scoring justification.\n"
    "The curator scored a post and gave this reason:\n\n"
    "\"{reason}\"\n\n"
    "Rate the justification quality:\n"
    "1 = generic/vague (could apply to any post)\n"
    "2 = adequate (some specificity)\n"
    "3 = specific/evidenced (cites concrete data, topic, or claim from the post)\n\n"
    "Reply with ONLY the number 1, 2, or 3."
)


async def _run_verifier(reason: str, backend: str, ollama_host: str, model: str, context: str) -> int:
    prompt = _VERIFIER_PROMPT.format(reason=reason)
    try:
        result = await _score_with_backend("Rate justification quality only.", prompt, ollama_host, model, backend)
        raw = str(result.get("score", result.get("reason", "2"))).strip()
        for char in raw:
            if char in "123":
                return int(char)
        return 2
    except Exception:
        return 2


async def _check_suspect(
    score: int, reason: str, backend: str, ollama_host: str, model: str, context: str,
    category: str | None = None,
) -> bool:
    # Phase 106 / SELF-TUNE-01: per-category threshold via resolver (yaml override
    # > env CURATOR_THRESHOLD > 75 default). Read INSIDE function — no module
    # cache trap.
    threshold = get_threshold_for_category(category)
    if score < threshold:
        return False
    words = set(reason.lower().split())
    is_short = len(reason.strip()) < 60
    is_generic = bool(words & _GENERIC_MARKERS)
    if not (is_short or is_generic):
        return False
    rating = await _run_verifier(reason, backend, ollama_host, model, context)
    return rating == 1


async def _commit_result(db, post_id, score, reason, topics, stage="general",
                          profile_hash=None, backend="", ollama_host="", model="", context="",
                          threshold_override: float | None = None,
                          category: str | None = None,
                          prompt_version: str | None = None):
    # Phase 62 / SOURCE-THRESH-01 + Phase 106 / SELF-TUNE-01: precedence
    # 1. per-source `sources.curator_threshold` (operator pin per channel)
    # 2. per-category yaml.thresholds[cat] override (Phase 106 self-tuner)
    # 3. env CURATOR_THRESHOLD (resolver fallback)
    # Resolver read INSIDE function to avoid module-level binding trap.
    env_default = get_threshold_for_category(category)
    threshold = threshold_override if threshold_override is not None else env_default
    new_status = "curated" if score >= threshold else "rejected"
    await db.execute("UPDATE raw_posts SET status = ? WHERE id = ?", (new_status, post_id))
    # Phase 70 / FACTOR-BUCKET-01: composite key written on every new row.
    scored_at_dt = datetime.datetime.now(datetime.UTC)
    factor_bucket = compute_bucket(backend or "unknown", category, scored_at_dt, profile_hash)
    await db.execute(
        "INSERT INTO curation_logs (post_id, relevance_score, reason, scored_at, profile_hash, stage, backend, factor_bucket, prompt_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (post_id, score, reason, scored_at_dt.isoformat(), profile_hash, stage, backend, factor_bucket, prompt_version),
    )
    await db.commit()
    if new_status == "curated" and topics:
        try:
            await insert_post_topics(db, post_id, topics)
        except Exception:
            logger.exception("Failed to persist topics for post %s — status remains 'curated'", post_id)
    # Suspect check (non-fatal)
    try:
        if await _check_suspect(score, reason, backend, ollama_host, model, context, category=category):
            await db.execute(
                "UPDATE curation_logs SET suspect_flag=1 WHERE id = ("
                "  SELECT id FROM curation_logs WHERE post_id=? ORDER BY id DESC LIMIT 1"
                ")",
                (post_id,),
            )
            await db.commit()
    except Exception:
        logger.exception("Suspect check failed for post %s — flag skipped", post_id)
    return new_status


def _get_profile_hash(vault_path: str) -> str | None:
    """Get git HEAD hash of the Obsidian vault for reward-spec versioning."""
    if not vault_path:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", vault_path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


async def _shadow_fan_out(
    db: aiosqlite.Connection,
    post_id: int,
    primary_score: int,
    context: str,
    score_text: str,
    was_truncated: bool,
    ollama_host: str,
    profile_hash: str | None,
    category: str | None = None,
    prompt_version: str | None = None,
) -> None:
    """Phase 52 / DIVERGE-02 + DIVERGE-03 + NFR-01 + NFR-04.

    Inline shadow scoring AFTER primary `_commit_result` has committed.
    NFR-01: every failure path inside this function is caught and converted
    to a logger.exception + counter increment — NEVER raises. Primary post
    state is already committed; this function only adds rollup rows.

    Scope (H1): only invoked from the per-post path. `claude-batch` early-
    returns at L305 before reaching the per-post loop, so this helper does
    NOT run in batch mode — that's intentional and out of scope for v2.0 P52.

    Truncation column semantics (H2/M3 — locked):
    `score_text` arrives ALREADY truncated by the primary path's
    `truncate_for_backend` call. We do NOT re-truncate per backend — both the
    3b and Claude shadow scorers see the SAME `score_text`. Therefore all
    scorer_divergence rows produced for this post share IDENTICAL values
    for `was_truncated`, `truncation_mode`, `char_count_seen` — these
    columns describe the PRIMARY 7b path's view, not per-backend
    re-truncation. Downstream query helpers (plan 05) document this same
    convention so P55/P58 do not misinterpret.

    Writes (best case, Claude-sampled post):
      - curation_logs row: backend='ollama-3b', stage='shadow'
      - curation_logs row: backend='claude',   stage='shadow' (if sampled)
      - scorer_divergence row: (ollama-7b, ollama-3b)
      - scorer_divergence row: (ollama-7b, claude)        (if sampled)
      - scorer_divergence row: (ollama-3b, claude)        (if sampled)
    """
    # Capture once — all rows share these values (NFR-04 + truncation triple)
    scored_at_dt = datetime.datetime.now(datetime.UTC)
    scored_at = scored_at_dt.isoformat()
    # Phase 70 / FACTOR-BUCKET-01: pre-compute per-backend buckets for shadow rows.
    _bucket_3b = compute_bucket("ollama-3b", category, scored_at_dt, profile_hash)
    _bucket_claude = compute_bucket("claude", category, scored_at_dt, profile_hash)
    is_holdout = 1 if (post_id % 5 == 4) else 0
    trunc_mode = os.environ.get("CURATOR_TRUNCATION_MODE", "prefix")
    char_count = len(score_text)
    db_path = os.environ.get("CURATOR_DB_PATH", "curator.db")

    # --- Strategy C: unconditional 3b shadow ---
    # WR-03 fix (2026-05-18): shadow model is env-configurable so operators
    # without `qwen2.5:3b` pulled aren't silently stuck at divergence_logged=0
    # (which is also the legitimate output for "no curated posts this run").
    # Override via `SHADOW_MODEL_3B`; default keeps prior behaviour.
    shadow_model_3b = os.environ.get("SHADOW_MODEL_3B", "qwen2.5:3b")
    score_3b: int | None = None
    try:
        result_3b = await _score_with_backend(
            context, score_text, ollama_host, shadow_model_3b, "ollama"
        )
        score_3b = int(result_3b.get("score", 0))
        reason_3b = str(result_3b.get("reason", ""))[:1000]

        # curation_logs row for ollama-3b (analytics distinguisher)
        await db.execute(
            "INSERT INTO curation_logs "
            "(post_id, relevance_score, reason, scored_at, profile_hash, stage, backend, factor_bucket, prompt_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (post_id, score_3b, reason_3b, scored_at, profile_hash, "shadow", "ollama-3b", _bucket_3b, prompt_version),
        )

        # scorer_divergence row for (7b, 3b) — Strategy C bootstrap
        delta_7b_3b = abs(float(primary_score) - float(score_3b))
        await db.execute(
            "INSERT OR IGNORE INTO scorer_divergence "
            "(raw_post_id, backend_a, backend_b, score_a, score_b, delta_abs, "
            " is_disputed, is_holdout, sample_source, scored_at, "
            " was_truncated, truncation_mode, char_count_seen) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                post_id, "ollama-7b", "ollama-3b",
                float(primary_score), float(score_3b), delta_7b_3b,
                0, is_holdout, "strategy_c_bootstrap", scored_at,
                int(was_truncated), trunc_mode, char_count,
            ),
        )
        await db.commit()
        shadow_router._incr_divergence_logged()
    except Exception as e:
        # WR-06 fix (2026-05-18): rollback any pending uncommitted writes
        # before bailing. Without rollback, a partial transaction (e.g. the
        # `curation_logs` INSERT succeeded but the divergence INSERT failed
        # mid-statement) would persist as orphan rows on the NEXT post's
        # primary `_commit_result` commit, surfacing later as "extra
        # curation_logs.backend='ollama-3b' rows for the wrong post_id".
        # Rollback is itself wrapped in try/except per NFR-01: shadow path
        # must NEVER raise.
        try:
            await db.rollback()
        except Exception:
            pass
        # WR-03 fix (2026-05-18): elevate a "model not found" first-occurrence
        # to a one-line WARNING with the model name + pull hint so an operator
        # on a fresh checkout / CI box can distinguish "shadow disabled" from
        # "no curated posts yet". Subsequent failures stay at logger.exception
        # level for full traceback.
        msg = str(e).lower()
        if "not found" in msg or "model" in msg:
            try:
                shadow_router._note_shadow_3b_failure(shadow_model_3b)
            except Exception:
                pass
        logger.exception(
            "Phase 52 shadow 3b failed for post %s -- primary unaffected (NFR-01)", post_id
        )
        return  # NFR-01: do not attempt Claude sampling if 3b failed

    if score_3b is None:
        return  # defensive (unreachable — covered by try/except above)

    # --- Strategy A: 5-10% Claude sample (gated) ---
    try:
        if not shadow_router.should_claude_sample(post_id, db_path):
            return
        try:
            result_claude = await claude_scorer.generate_score(context, score_text)
        except shadow_router.ClaudeRateLimitError:
            shadow_router._incr_claude_calls_skipped()
            logger.warning(
                "Phase 52 Claude rate-limit hit for post %s -- primary unaffected", post_id
            )
            return
        score_claude = int(result_claude.get("score", 0))
        reason_claude = str(result_claude.get("reason", ""))[:1000]

        # curation_logs row for claude (analytics distinguisher)
        await db.execute(
            "INSERT INTO curation_logs "
            "(post_id, relevance_score, reason, scored_at, profile_hash, stage, backend, factor_bucket, prompt_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (post_id, score_claude, reason_claude, scored_at, profile_hash, "shadow", "claude", _bucket_claude, prompt_version),
        )

        # scorer_divergence rows: (7b, claude) and (3b, claude)
        # Note: was_truncated/truncation_mode/char_count_seen mirror the primary
        # 7b path's view (H2/M3). Claude saw the SAME already-truncated score_text.
        delta_7b_cl = abs(float(primary_score) - float(score_claude))
        delta_3b_cl = abs(float(score_3b) - float(score_claude))
        await db.execute(
            "INSERT OR IGNORE INTO scorer_divergence "
            "(raw_post_id, backend_a, backend_b, score_a, score_b, delta_abs, "
            " is_disputed, is_holdout, sample_source, scored_at, "
            " was_truncated, truncation_mode, char_count_seen) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                post_id, "ollama-7b", "claude",
                float(primary_score), float(score_claude), delta_7b_cl,
                0, is_holdout, "strategy_a_truth_sample", scored_at,
                int(was_truncated), trunc_mode, char_count,
            ),
        )
        await db.execute(
            "INSERT OR IGNORE INTO scorer_divergence "
            "(raw_post_id, backend_a, backend_b, score_a, score_b, delta_abs, "
            " is_disputed, is_holdout, sample_source, scored_at, "
            " was_truncated, truncation_mode, char_count_seen) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                post_id, "ollama-3b", "claude",
                float(score_3b), float(score_claude), delta_3b_cl,
                0, is_holdout, "strategy_a_truth_sample", scored_at,
                int(was_truncated), trunc_mode, char_count,
            ),
        )
        await db.commit()
        shadow_router._incr_claude_sampled()
    except Exception:
        # WR-06 fix (2026-05-18): rollback before bailing — see analogous
        # block in Strategy C above. Strategy A's transaction touches 1
        # curation_logs row + 2 scorer_divergence rows before commit; any
        # failure between the first INSERT and `db.commit()` would otherwise
        # carry over to the next post's primary commit.
        try:
            await db.rollback()
        except Exception:
            pass
        logger.exception(
            "Phase 52 Claude shadow failed for post %s -- primary unaffected (NFR-01)", post_id
        )


async def process_unprocessed(db: aiosqlite.Connection, ollama_host: str, model: str, seal_tag: str):
    posts = await get_unprocessed_posts(db)
    context = await build_curator_context(db, seal_tag)
    global_backend = os.environ.get("CURATOR_BACKEND", "")
    vault_path = os.environ.get("OBSIDIAN_VAULT_PATH", "")
    profile_hash = _get_profile_hash(vault_path)

    # Phase 80 / PROMPT-ABLEND-01: discover available prompt versions and load
    # rollout state once per pipeline run. Per-post pick happens in the
    # scoring loop via prompt_router.pick_version. Fallback to 'legacy' when
    # the prompts/ directory is missing or contains no curator_v*.txt files —
    # preserves pre-Phase-80 behaviour (curation_logs.prompt_version='legacy').
    _prompts_dir = os.environ.get("CURATOR_PROMPTS_DIR", "prompts")
    _rollout_path = os.environ.get(
        "CURATOR_ROLLOUT_STATE_PATH",
        os.path.join(_prompts_dir, "rollout_state.json"),
    )
    _versions = prompt_router.discover_versions(_prompts_dir)
    _rollout_state = prompt_router.load_rollout_state(_rollout_path)
    _alpha = prompt_router.alpha_for_now(
        _rollout_state, datetime.datetime.now(datetime.UTC), n_versions=len(_versions)
    )
    # Lazy per-version text cache (path -> contents). Plan 02 reads but does
    # not yet use the template downstream — backends keep their inline literal
    # until a future plan extracts the formatting layer.
    _prompt_text_cache: dict = {}

    def _pick_prompt_version_for_post(post_id) -> str:
        """Return the prompt_version string to stamp on curation_logs rows.

        Returns:
            'legacy' when no curator_v*.txt exists (preserves pre-Phase-80
            behaviour); 'v{N}' otherwise. Caches loaded prompt text per path
            even though Plan 02 does not yet feed it into the backends (this
            keeps the cache populated for the follow-up plan that swaps
            inline literals for template-from-disk).
        """
        if not _versions:
            return "legacy"
        version_int, path = prompt_router.pick_version(post_id, _versions, _alpha)
        if path not in _prompt_text_cache:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    _prompt_text_cache[path] = f.read()
            except OSError:
                logger.exception("Failed to read prompt template %s — using legacy", path)
                return "legacy"
        return f"v{version_int}"

    # Single active version for batch / gate paths that do not route per-post.
    # Uses the newest available version when ≥1 exists; falls back to 'legacy'.
    if _versions:
        _active_prompt_version = f"v{_versions[-1][0]}"
    else:
        _active_prompt_version = "legacy"

    # Pre-filter: reject too-short posts before any LLM call
    scoreable = []
    for post in posts:
        # Phase 96 / IMAP-TRIAGE-01 (Plan 96-02): pre-LLM email_class skip gate.
        # Counter increments for EVERY non-NULL email_class (full distribution
        # telemetry, NOT skip-only — see _email_triage_counts docstring at
        # module top). Skip branch fires for {support, personal}; newsletter
        # and action fall through to the regular curator path. NULL bypasses
        # entirely so pre-Phase-96 rows + IMAP_TRIAGE_MODE=off runs behave
        # identically to the pre-triage codebase.
        ec = post.get("email_class")
        if ec in ("newsletter", "support", "personal", "action"):
            # WR-04 (Phase 96 review): counter contract is "per-process
            # post-visits", NOT "per-post". A post left as 'unprocessed' by a
            # failed scorer run will be re-counted on the next pipeline tick.
            # Dedup happens via raw_posts.status flipping away from
            # 'unprocessed' on the success path.
            _email_triage_counts[ec] += 1
        if ec in ("support", "personal"):
            try:
                _et_dt = datetime.datetime.now(datetime.UTC).isoformat()
                # WR-03 fix (Phase 96, reference_file_write_before_db_update_pattern):
                # INSERT audit row FIRST, then UPDATE status. Pre-fix order
                # could leave status='rejected' with no curation_logs row if
                # the INSERT failed after the UPDATE. Retry-safe: if INSERT
                # fails, status stays 'unprocessed' and next run retries.
                await db.execute(
                    "INSERT INTO curation_logs "
                    "(post_id, relevance_score, reason, scored_at, stage, backend) "
                    "VALUES (?, 0, ?, ?, 'general', 'email_triage_gate')",
                    (post["id"], f"email_triage_skip:{ec}", _et_dt),
                )
                await db.execute(
                    "UPDATE raw_posts SET status='rejected' WHERE id=?", (post["id"],)
                )
                await db.commit()
            except Exception:
                await db.rollback()
                logger.exception(
                    "email_triage_gate insert failed for post %s — skipping",
                    post["id"],
                )
            continue
        if len(post["raw_text"].strip()) < 100:
            await update_post_status(db, post["id"], "rejected")
            continue
        # Phase 59 / SNIPPET-01: pre-LLM mojibake / non-printable gate.
        # Pure regex + ratio check (sub-ms). Mirrors <100-char filter shape:
        # marks raw_post 'rejected', writes a curation_logs row with
        # backend='snippet_quality_gate' so the rejection is grep-able, and
        # increments the process-global counter consumed by run_pipeline's
        # OK-log (snippet_rejected=N field).
        extracted = post.get("extracted_body") or ""
        text_for_gate = post["raw_text"] + (("\n\n" + extracted) if extracted else "")
        is_low, gate_reason = is_low_quality_snippet(text_for_gate)
        if is_low:
            # M-01 (Phase 59 review): wrap in try/except so a single bad
            # insert (constraint violation, lock timeout, partial schema
            # migration) cannot abort the whole pre-filter loop and strand
            # every subsequent post as `unprocessed`. Mirrors the per-post
            # path at L592-594 + the claude-batch path at L519-521.
            try:
                await db.execute(
                    "UPDATE raw_posts SET status='rejected' WHERE id=?", (post["id"],)
                )
                # Phase 70 / FACTOR-BUCKET-01: bucket with gate-name as backend slot.
                # WR-05 (Phase 70 review): thread raw_post.category through so
                # gate-rejected rows preserve the category axis of the
                # composite key (was hardcoded None → "unknown").
                _snip_dt = datetime.datetime.now(datetime.UTC)
                _snip_bucket = compute_bucket(
                    "snippet_quality_gate", post.get("category"), _snip_dt, profile_hash
                )
                await db.execute(
                    "INSERT INTO curation_logs "
                    "(post_id, relevance_score, reason, scored_at, profile_hash, stage, backend, factor_bucket, prompt_version) "
                    "VALUES (?, 0, ?, ?, ?, 'general', 'snippet_quality_gate', ?, ?)",
                    (
                        post["id"],
                        gate_reason,
                        _snip_dt.isoformat(),
                        profile_hash,
                        _snip_bucket,
                        _active_prompt_version,
                    ),
                )
                await db.commit()
                _incr_snippet_rejected()
            except Exception:
                await db.rollback()
                logger.exception(
                    "snippet_quality_gate insert failed for post %s — skipping",
                    post["id"],
                )
            continue
        # Phase 60 / GIVEAWAY-01: pre-LLM giveaway / promo / referral regex gate.
        # Pure regex + co-occurrence check (sub-ms). Mirrors snippet_quality_gate
        # shape: marks raw_post 'rejected', writes curation_logs row with
        # backend='giveaway_regex_gate' (grep-able), increments process-global
        # counter consumed by run_pipeline OK-log (giveaway_rejected=N field).
        # Reuses `text_for_gate` already built by snippet_quality_gate above.
        is_gv, gv_reason = is_giveaway(text_for_gate)
        if is_gv:
            try:
                await db.execute(
                    "UPDATE raw_posts SET status='rejected' WHERE id=?", (post["id"],)
                )
                # Phase 70 / FACTOR-BUCKET-01: bucket with gate-name as backend slot.
                # WR-05 (Phase 70 review): thread raw_post.category through so
                # gate-rejected rows preserve the category axis of the
                # composite key (was hardcoded None → "unknown").
                _gv_dt = datetime.datetime.now(datetime.UTC)
                _gv_bucket = compute_bucket(
                    "giveaway_regex_gate", post.get("category"), _gv_dt, profile_hash
                )
                await db.execute(
                    "INSERT INTO curation_logs "
                    "(post_id, relevance_score, reason, scored_at, profile_hash, stage, backend, factor_bucket, prompt_version) "
                    "VALUES (?, 0, ?, ?, ?, 'general', 'giveaway_regex_gate', ?, ?)",
                    (
                        post["id"],
                        gv_reason,
                        _gv_dt.isoformat(),
                        profile_hash,
                        _gv_bucket,
                        _active_prompt_version,
                    ),
                )
                await db.commit()
                _incr_giveaway_rejected()
            except Exception:
                await db.rollback()
                logger.exception(
                    "giveaway_regex_gate insert failed for post %s — skipping",
                    post["id"],
                )
            continue
        scoreable.append(post)

    if global_backend == "claude-batch":
        # Batch path: BATCH_SIZE posts per Claude CLI call
        from src.llm.routing import truncate_for_backend as _truncate_for_batch
        # Phase 70 / FACTOR-BUCKET-01: per-post category lookup for bucket key.
        batch_source_cat: dict[int, str] = {}
        async with db.execute("SELECT id, category FROM sources") as cur:
            async for row in cur:
                batch_source_cat[row[0]] = row[1] or "other"
        for i in range(0, len(scoreable), BATCH_SIZE):
            batch = scoreable[i: i + BATCH_SIZE]
            texts = [
                _truncate_for_batch(
                    p["raw_text"]
                    + (("\n\n" + p.get("extracted_body", "")) if p.get("extracted_body") else ""),
                    "claude-batch",
                )[0]
                for p in batch
            ]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(scoreable) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"  batch {batch_num}/{total_batches} ({len(batch)} posts)...", flush=True)
            try:
                results = await score_batch(context, texts)
                for post, result in zip(batch, results):
                    try:
                        cat_b = batch_source_cat.get(post.get("source_id"), "other")
                        # Phase 80: batch path uses a single active version
                        # per run (no per-post routing inside the batched
                        # Claude CLI call). Default to newest available;
                        # 'legacy' when prompts/ missing.
                        status = await _commit_result(
                            db, post["id"],
                            result.get("score", 0), result.get("reason", ""), [],
                            profile_hash=profile_hash,
                            backend=global_backend, ollama_host=ollama_host, model=model, context=context,
                            category=cat_b,
                            prompt_version=_active_prompt_version,
                        )
                        marker = "curated" if status == "curated" else "rejected"
                        safe = post['raw_text'][:60].encode('ascii', errors='replace').decode('ascii')
                        print(f"    [{marker:>8}] {safe}", flush=True)
                    except Exception:
                        await db.rollback()
                        logger.exception("Failed to commit score for post %s", post["id"])
            except Exception:
                logger.exception("Batch %d failed — posts remain unprocessed for retry", batch_num)
        return

    # Per-post path with per-category routing
    from src.llm.routing import (
        get_backend_for_category,
        truncate_for_backend,
        should_spill,
        record_ollama_latency,
    )
    from src.llm.free_energy import FreeEnergyTracker

    # Phase 83-01 / SPILL-ROUTE-01: queue_depth snapshot, used by should_spill
    # for every post in this run. Stable across the loop; if a post-mid-loop
    # spike happens we'll catch it on the next pipeline tick.
    queue_depth = len(scoreable)
    # WR-02 (Phase 83 review): once spill fires for the first post in this run,
    # all subsequent posts bypass the cooldown via force_spill=True. Pre-fix
    # the cooldown re-routed posts 2..N back to ollama → overload protection
    # fired for 1/N posts and the queue stayed backlogged.
    _spill_active_this_run = False

    fe_mode = FreeEnergyTracker.is_enabled()
    fe_tracker = FreeEnergyTracker() if fe_mode else None
    if fe_mode:
        # Phase 106 / SELF-TUNE-01: telemetry surface — show env-default
        # threshold (per-category overrides applied at commit time, not here).
        # Resolver with category=None falls back to yaml.thresholds['other']
        # then env CURATOR_THRESHOLD.
        threshold_actual = get_threshold_for_category(None)
        print(f"  [FREE_ENERGY] mode active, min_score={fe_tracker.min_score}, "
              f"normal threshold={threshold_actual}", flush=True)

    # Phase 62 / SOURCE-THRESH-01: extend batch query to fetch per-source
    # threshold override alongside category. Single SQL read for whole batch
    # -- zero per-post overhead. NULL curator_threshold -> env fallback in
    # _commit_result via threshold_override=None path.
    source_meta: dict[int, tuple[str, float | None]] = {}
    async with db.execute("SELECT id, category, curator_threshold FROM sources") as cur:
        async for row in cur:
            source_meta[row[0]] = (row[1] or "other", row[2])

    # Phase 99 / CLUSTER-PRE-01 (Plan 99-02): env-gated cluster pre-filter.
    # Read env INSIDE function (module-level binding gotcha). Default off →
    # behaviour byte-identical to pre-Phase-99. On → call cluster_posts on
    # scoreable, build a {medoid_id -> [non-medoid member_ids]} map plus a
    # skip-set of non-medoid members so the per-post loop scores ONLY medoids
    # (and singletons, which arrive as size-1 clusters with the post itself
    # as medoid). After each medoid scores, propagate verdict to members.
    _cluster_prefilter_on = (
        os.environ.get("CHROMA_CLUSTER_PREFILTER", "off").lower() == "on"
    )
    _cluster_members_by_medoid: dict[int, list[int]] = {}
    _cluster_skip_member_ids: set[int] = set()
    if _cluster_prefilter_on and scoreable:
        try:
            # CR-01 fix (Phase 99): defensive log when no scoreable post
            # carries an embedding — pre-fix the pre-filter silently
            # degenerated to all-singletons with no operator-visible signal.
            _emb_count = sum(1 for _p in scoreable if _p.get("embedding"))
            if _emb_count == 0:
                logger.warning(
                    "Phase 99 cluster pre-filter: zero embeddings in %d "
                    "scoreable posts — feature is no-op this run "
                    "(check raw_posts.embedding column / chroma sync)",
                    len(scoreable),
                )
            _clusters = cluster_posts(scoreable)
            # Phase 103 / CLUSTER-FLOOR-01: per-member cosine floor gate.
            # Read env INSIDE function (module-level binding gotcha). Decode
            # member embeddings once + defensive L2-normalise. Members with
            # cosine < floor (or NULL/decode-failed embedding) drop from the
            # skip-set → per-post scoring fallback. Members ≥ floor stay in
            # skip-set → inherit medoid verdict (P99 path unchanged).
            # Phase 103 review WR-01: fail-soft env parse + clamp to [0.0, 1.0].
            # Bad string → default 0.7 + warning; out-of-range value → clamp.
            _floor_raw = os.environ.get("CLUSTER_MEMBER_COSINE_FLOOR", "0.7")
            try:
                _floor = float(_floor_raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Phase 103 cluster floor: invalid CLUSTER_MEMBER_COSINE_FLOOR=%r "
                    "— defaulting to 0.7",
                    _floor_raw,
                )
                _floor = 0.7
            if _floor < 0.0:
                _floor = 0.0
            elif _floor > 1.0:
                _floor = 1.0
            _emb_by_id: dict[int, "np.ndarray | None"] = {}
            for _p in scoreable:
                _v = _decode_embedding(_p.get("embedding"))
                if _v is not None:
                    _n = float(np.linalg.norm(_v))
                    if _n > 0:
                        _v = (_v / _n).astype(np.float32)
                    else:
                        _v = None
                _emb_by_id[int(_p["id"])] = _v

            for _cr in _clusters:
                if _cr.size <= 1:
                    continue
                _medoid_emb = _emb_by_id.get(_cr.medoid_id)
                if _medoid_emb is None:
                    # No medoid embedding → cannot gate; preserve P99 blind-propagate.
                    non_medoid = [m for m in _cr.member_ids if m != _cr.medoid_id]
                    if non_medoid:
                        _cluster_members_by_medoid[_cr.medoid_id] = non_medoid
                        _cluster_skip_member_ids.update(non_medoid)
                    continue
                _kept: list[int] = []
                for _mid in _cr.member_ids:
                    if _mid == _cr.medoid_id:
                        continue
                    _mem_emb = _emb_by_id.get(_mid)
                    if _mem_emb is None:
                        _incr_cluster_floor_skipped(1)
                        continue
                    _cos = float(np.dot(_medoid_emb, _mem_emb))
                    if _cos < _floor:
                        _incr_cluster_floor_skipped(1)
                        continue
                    _kept.append(_mid)
                if _kept:
                    _cluster_members_by_medoid[_cr.medoid_id] = _kept
                    _cluster_skip_member_ids.update(_kept)
        except Exception:
            logger.exception(
                "Phase 99 cluster_posts failed — falling back to per-post path"
            )
            _cluster_members_by_medoid = {}
            _cluster_skip_member_ids = set()
            # WR-04 fix (Phase 99): explicit error counter so the silent
            # fall-through is no longer invisible in production logs.
            try:
                _incr_cluster_prefilter_errors()
            except Exception:
                pass
    # Index scoreable posts by id so we can fetch propagation targets quickly
    # without re-querying the DB (mirrors source_meta single-read pattern).
    _post_by_id: dict[int, dict] = {p["id"]: p for p in scoreable}

    for post in scoreable:
        # Phase 99: non-medoid cluster members are processed below when their
        # medoid scores — skip them here. Singletons (size==1) are NOT in
        # _cluster_skip_member_ids and proceed through the regular per-post path.
        if post["id"] in _cluster_skip_member_ids:
            continue
        try:
            cat, threshold_override = source_meta.get(
                post.get("source_id"), ("other", None)
            )
            backend = get_backend_for_category(cat)
            # Phase 83-01 / SPILL-ROUTE-01: only the ollama→claude-batch direction
            # spills (operator pinning claude/claude-batch is honoured as-is).
            spilled = False
            if backend == "ollama" and should_spill(
                queue_depth, force_spill=_spill_active_this_run
            ):
                backend = "claude-batch"
                spilled = True
                _spill_active_this_run = True
            # Phase 80: pick prompt version per post via stable hash → α-blend.
            post_prompt_version = _pick_prompt_version_for_post(post["id"])
            extracted = post.get("extracted_body") or ""
            score_text = post["raw_text"] + (("\n\n" + extracted) if extracted else "")
            # Phase 43 (999.21): truncate to backend's max input tokens (counter incr)
            score_text, _trunc = truncate_for_backend(score_text, backend)
            if spilled:
                # Phase 83-01: single-element batch through claude-batch.
                batch_results = await score_batch(context, [score_text])
                result = batch_results[0] if batch_results else {"score": 0, "reason": "spill_empty"}
                # T-83-03 mitigation: attribute the spilled call in cost_snapshots.
                # Failure here MUST NOT abort curator processing (mirrors the
                # snippet_quality_gate fail-soft pattern at L575-580).
                try:
                    _sp_dt = datetime.datetime.now(datetime.UTC).isoformat()
                    await db.execute(
                        "INSERT INTO cost_snapshots "
                        "(created_at, backend, calls_7d, est_tokens_7d, anomaly_flag, analysis_md) "
                        "VALUES (?, 'spillover', 1, ?, 0, ?)",
                        (_sp_dt, len(score_text) // 4, f"auto-spill: post_id={post['id']}"),
                    )
                    await db.commit()
                except Exception:
                    await db.rollback()
                    logger.exception(
                        "spillover cost_snapshots insert failed for post %s — continuing",
                        post["id"],
                    )
            else:
                if backend == "ollama":
                    # Phase 83-01: record latency into rolling window for the
                    # p95 spill trigger. perf_counter has best resolution for ms.
                    # WR-03 (Phase 83 review): try/finally so a hang/exception
                    # in _score_with_backend STILL records latency (sentinel
                    # large value on raise) — pre-fix, the exact failure mode
                    # the p95 trigger is designed to detect was invisible.
                    _t0 = _curator_time.perf_counter()
                    _spill_recorded = False
                    try:
                        result = await _score_with_backend(context, score_text, ollama_host, model, backend)
                    except Exception:
                        record_ollama_latency(60000.0)  # sentinel: 60s = catastrophic
                        _spill_recorded = True
                        raise
                    finally:
                        if not _spill_recorded:
                            record_ollama_latency((_curator_time.perf_counter() - _t0) * 1000.0)
                else:
                    result = await _score_with_backend(context, score_text, ollama_host, model, backend)
            score = result.get("score", 0)
            reason = result.get("reason", "")
            topics = result.get("topics", [])
            stage = result.get("research_stage", "general")

            # Idea 7 (Free Energy): override threshold decision when mode active
            if fe_mode and fe_tracker is not None:
                if fe_tracker.should_curate(score, score_text):
                    new_status = await _commit_result(db, post["id"], score, reason, topics, stage,
                                                      profile_hash=profile_hash, backend=backend,
                                                      ollama_host=ollama_host, model=model, context=context,
                                                      threshold_override=threshold_override,
                                                      category=cat,
                                                      prompt_version=post_prompt_version)
                    if new_status == "curated":
                        fe_tracker.add(score_text)
                    fe_label = f"fe/H={fe_tracker.corpus_entropy:.1f}"
                else:
                    await update_post_status(db, post["id"], "rejected")
                    new_status = "rejected"
                    fe_label = "fe/drop"
                safe = post['raw_text'][:60].encode('ascii', errors='replace').decode('ascii')
                print(f"  [{new_status:>8}/{backend[:6]}] [{fe_label}] {safe}", flush=True)
            else:
                new_status = await _commit_result(db, post["id"], score, reason, topics, stage,
                                                  profile_hash=profile_hash, backend=backend,
                                                  ollama_host=ollama_host, model=model, context=context,
                                                  threshold_override=threshold_override,
                                                  category=cat,
                                                  prompt_version=post_prompt_version)
                safe = post['raw_text'][:60].encode('ascii', errors='replace').decode('ascii')
                print(f"  [{new_status:>8}/{backend[:6]}] {safe}", flush=True)

            # Phase 52 / DIVERGE-02 + DIVERGE-03: shadow fan-out (per-post path ONLY)
            # Wrapped in its own try/except via _shadow_fan_out — cannot raise (NFR-01).
            if new_status == "curated":
                await _shadow_fan_out(
                    db,
                    post["id"],
                    score,
                    context,
                    score_text,
                    bool(_trunc),
                    ollama_host,
                    profile_hash,
                    category=cat,
                    prompt_version=post_prompt_version,
                )

            # Phase 99 / CLUSTER-PRE-01 (Plan 99-02): propagate medoid verdict
            # to non-medoid cluster members. Score + topics + stage carry over;
            # reason is rewritten to encode the propagation lineage so analysts
            # can trace which medoid drove each member's verdict. _commit_result
            # also writes to raw_posts.status, so members get the same curated/
            # rejected outcome as the medoid. Per-member try/except so a single
            # failed propagation cannot abort the whole loop.
            _propagate_targets = _cluster_members_by_medoid.get(post["id"])
            if _propagate_targets:
                _prop_reason = f"cluster_propagated medoid={post['id']} {reason}"[:1000]
                for _member_id in _propagate_targets:
                    _member_post = _post_by_id.get(_member_id)
                    if _member_post is None:
                        continue
                    _member_cat, _member_thresh = source_meta.get(
                        _member_post.get("source_id"), ("other", None)
                    )
                    try:
                        _member_status = await _commit_result(
                            db, _member_id, score, _prop_reason, topics, stage,
                            profile_hash=profile_hash, backend=backend,
                            ollama_host=ollama_host, model=model, context=context,
                            threshold_override=_member_thresh,
                            category=_member_cat,
                            prompt_version=post_prompt_version,
                        )
                        _incr_cluster_propagated(1)
                        # WR-01 fix (Phase 99): also call _shadow_fan_out on
                        # propagated members so v2.0 divergence telemetry
                        # captures ALL curated rows, not just the medoid
                        # (~10x signal recovery for the audit pipeline).
                        if _member_status == "curated":
                            try:
                                _member_text = _member_post["raw_text"] + (
                                    ("\n\n" + (_member_post.get("extracted_body") or ""))
                                    if _member_post.get("extracted_body") else ""
                                )
                                _member_text, _ = truncate_for_backend(_member_text, backend)
                                await _shadow_fan_out(
                                    db,
                                    _member_id,
                                    score,
                                    context,
                                    _member_text,
                                    bool(_trunc),
                                    ollama_host,
                                    profile_hash,
                                    category=_member_cat,
                                    prompt_version=post_prompt_version,
                                )
                            except Exception:
                                logger.exception(
                                    "Phase 99 shadow_fan_out failed for "
                                    "propagated member %s — primary unaffected",
                                    _member_id,
                                )
                    except Exception:
                        await db.rollback()
                        logger.exception(
                            "Phase 99 cluster propagation failed for member %s "
                            "(medoid=%s) — member remains unprocessed",
                            _member_id,
                            post["id"],
                        )
        except Exception:
            await db.rollback()
            logger.exception("Failed to score post %s, leaving as unprocessed for retry", post["id"])

    # Phase 103 / CLUSTER-FLOOR-01: emit one cost_snapshots row per run when
    # either counter > 0. Skipped when both are zero to avoid log spam on
    # runs where the cluster prefilter is off or produced no multi-member
    # clusters. Fail-soft — any DB error is logged but cannot abort the run.
    try:
        _prop_n = get_cluster_propagated_count()
        _skip_n = get_cluster_floor_skipped_count()
        if _prop_n > 0 or _skip_n > 0:
            try:
                _ts_iso = datetime.datetime.now(datetime.UTC).isoformat()
                await db.execute(
                    "INSERT INTO cost_snapshots "
                    "(created_at, backend, calls_7d, est_tokens_7d, anomaly_flag, "
                    "cluster_propagated_n, cluster_floor_skipped_n) "
                    "VALUES (?, 'cluster_floor', 0, 0, 0, ?, ?)",
                    (_ts_iso, _prop_n, _skip_n),
                )
                await db.commit()
            except Exception:
                await db.rollback()
                logger.exception(
                    "Phase 103 cost_snapshots insert failed — "
                    "counters still readable in-process"
                )
    except Exception:
        logger.exception("Phase 103 counter read failed — skipping snapshot")
