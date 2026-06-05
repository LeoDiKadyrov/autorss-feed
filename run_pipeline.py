"""
Full pipeline: collect from Telegram → curate → generate digest → start web server.

Prerequisites:
  1. Copy .env.example to .env and fill in TG_API_ID, TG_API_HASH
  2. Run: .venv/Scripts/python.exe scripts/auth_telegram.py
  3. Run: .venv/Scripts/python.exe scripts/add_source.py @your_channel

Usage:
  .venv/Scripts/python.exe run_pipeline.py                          # collect from Telegram (full)
  .venv/Scripts/python.exe run_pipeline.py --mode collect-only      # collect only
  .venv/Scripts/python.exe run_pipeline.py --mode curate-only       # curate only
  .venv/Scripts/python.exe run_pipeline.py --mode digest-only       # digest only
  .venv/Scripts/python.exe run_pipeline.py --mode curate-only --seed-test-data  # seed + curate
  .venv/Scripts/python.exe run_pipeline.py --no-collect             # [DEPRECATED] use --mode curate-only --seed-test-data
"""
import argparse
import asyncio
import logging as _logging
import os
import secrets
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import aiohttp
import aiosqlite
from src.env_loader import load_env
from src.database.client import (
    init_db,
    migrate_db,
    ensure_sources,
    insert_raw_post,
    get_cognition_counters,
    reset_cognition_counters,
)
from src.database.connection import open_db
from src.agents.collector import collect_all, discover_telegram_dialogs
from src.agents.curator import process_unprocessed
from src.agents.editor import create_daily_digest
from src.collectors.enricher import enrich_web_links
from src.agents.workspace import get_workspace, reset_workspace
from src.extract.http import SSRFSafeResolver
from src.telegram.client import TelegramCollectorClient
from src.worker.finish_worker import run_all_pending as run_finish_worker
from src.worker.draft_manager import list_pending as list_pending_drafts

SOURCES_FILE = Path(__file__).parent / "config" / "sources.txt"

def load_sources() -> list[dict]:
    if not SOURCES_FILE.exists():
        return []
    result = []
    for line in SOURCES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        result.append({
            "target": parts[0],
            "category": parts[1] if len(parts) > 1 else "other",
            "display_name": parts[2] if len(parts) > 2 else parts[0],
            "platform": parts[3] if len(parts) > 3 else "telegram",  # D-01: default=telegram
        })
    return result

# Phase 107 TRIM-02: guarded loader (no-op under pytest via AUTORSS_DISABLE_DOTENV)
# so personal .env values don't leak into the test session at import time.
load_env()

OLLAMA_HOST = "http://localhost:11434"
CURATOR_MODEL = "qwen2.5:7b"   # JSON scoring — qwen handles structured output well
EDITOR_MODEL = "gemma3:4b"     # Instruction following — fallback when backend=ollama
EDITOR_BACKEND = os.environ.get("EDITOR_BACKEND", "claude")  # "claude" or "ollama"


# WR-02: read DB_PATH inside the function that uses it. Module-level binding
# captured the env at import time and could not be overridden by tests using
# patch.dict(os.environ, {"DB_PATH": ...}) — see reference_module_level_env_binding
# in project memory for the recurring gotcha pattern.
def _get_db_path() -> str:
    return os.environ.get("DB_PATH", "curator.db")

# --- Phase 6 / AUTO-02: pipeline run log (D-07, D-08, D-09, D-10, D-20) ---

LOG_PATH = Path(__file__).parent / "logs" / "pipeline.log"

# D-02 / CLAUDE.md: project owner is in GMT+5 (Karaganda); schtasks fires
# tasks in local time. We stamp logs in the same TZ so the operator's
# eye-grep across schtasks /Query and the log file aligns.
_GMT_PLUS_5 = timezone(timedelta(hours=5))


def _now_iso() -> str:
    """Current time as ISO 8601 with +05:00 suffix (D-08 example uses this form)."""
    return datetime.now(_GMT_PLUS_5).isoformat(timespec="seconds")


def _append_log_line(line: str) -> None:
    """
    Append one line to logs/pipeline.log with newline. Creates the directory
    if missing.

    D-09: append-only mode ("a") — never truncates.
    D-20: silent failure — if the write raises (disk full, permission denied),
          swallow the error so the pipeline continues. The log is best-effort.
    """
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
    except Exception:
        # D-20: never propagate a logging failure into the pipeline.
        # Print to stderr so the user has SOME signal, but don't crash.
        try:
            print(f"  ! failed to write {LOG_PATH}: (silenced)", file=sys.stderr)
        except Exception:
            pass


def _launch_snapshot() -> None:
    """Fire-and-forget cost snapshot after log write. Never raises."""
    try:
        snapshot_script = Path(__file__).parent / "scripts" / "_cost_snapshot.py"
        subprocess.Popen(
            [sys.executable, str(snapshot_script)],
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    except Exception:
        pass


def _launch_seed_claude() -> None:
    """Fire-and-forget: generate 1 claude digest after successful ollama digest. Never raises."""
    try:
        seed_script = Path(__file__).parent / "scripts" / "seed_claude_digests.py"
        subprocess.Popen(
            [sys.executable, str(seed_script), "--count", "1", "--auto"],
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    except Exception:
        pass


def _format_ok_line(
    collected: int,
    curated: int,
    digest_state: str,
    *,
    profile_fallback: bool = False,
    drafts_pending: int = 0,
    tg_discovered: int = 0,
    ig_collected: int = 0,
    cognition_tagged: int = 0,
    gaming_q: int = 0,
    health_days: int = 0,
    dataset_size: int = 0,
    aggregate_imputed: int = 0,
    replay_violations: int = 0,
    replay_orphans: int = 0,
    score_drift_flags: int = 0,
    truncation_count: int = 0,
    # Phase 52 / NFR-05 — shadow telemetry accretion (3 fields).
    # Position in format string: AFTER truncation_count, BEFORE profile_fallback.
    divergence_logged: int = 0,
    claude_sampled: int = 0,
    claude_calls_skipped: int = 0,
    # Phase 59 / SNIPPET-01 — pre-LLM snippet quality gate counter.
    # Position: AFTER claude_calls_skipped, BEFORE profile_fallback per
    # CONTEXT 59 accretion. Counter from src.curator.snippet_quality.
    # Reset at run start by reset_snippet_rejected() in main(); read fail-soft
    # at OK-log emit site (try/except → 0). Lockstep with _LOG_OK_RE in
    # src/web/main.py + 3 test files per reference_ok_log_field_accretion.md.
    snippet_rejected: int = 0,
    # Phase 60 / GIVEAWAY-01 — pre-LLM giveaway/promo/referral regex gate counter.
    # Position: AFTER snippet_rejected, BEFORE profile_fallback per CONTEXT 60
    # accretion (mirrors P59 placement rule one slot to the right). Counter
    # from src.curator.giveaway_filter.get_giveaway_rejected. Reset at run
    # start by reset_giveaway_rejected() in main(); read fail-soft at OK-log
    # emit site (try/except → 0). Lockstep with _LOG_OK_RE in src/web/main.py
    # + 4 test files (incl. test_run_pipeline_offline_replay per P59
    # Rule-3 deviation lesson) per reference_ok_log_field_accretion.md.
    giveaway_rejected: int = 0,
    # Phase 61 / COST-CAP-01 — per-category daily USD cap counter.
    # Position: AFTER giveaway_rejected, BEFORE profile_fallback per CONTEXT 61
    # accretion (mirrors P60 placement rule one slot to the right). Counter
    # from src.llm.cost_cap.get_cost_cap_falls_back. Reset at run start by
    # reset_cost_cap_falls_back() in main(); read fail-soft at OK-log emit
    # site (try/except → 0). Lockstep with _LOG_OK_RE in src/web/main.py
    # + 4 test files per reference_ok_log_field_accretion.md.
    # Position invariant: snippet_rejected < giveaway_rejected <
    # cost_cap_falls_back < profile_fallback.
    #
    # LR-01 (Phase 61 review): SEMANTICS — this is a DIVERTED-CALL count,
    # not a unique cap-event count. The counter increments on EVERY True
    # return from is_cap_exceeded (cache-hit returns count too). So a single
    # cap-trip that diverts 500 posts will emit `cost_cap_falls_back=500`,
    # not `=1`. Alarm thresholds and dashboards should treat this as
    # "calls diverted to ollama due to cap" — useful for budget-impact
    # quantification, NOT for "how many distinct caps tripped today".
    # (Renaming the field would break the 6-file lockstep + web parser
    # backward-compat — semantic doc-fix only.)
    cost_cap_falls_back: int = 0,
    # Phase 64 / DIGEST-CONFORM-01 — digest spec-conformance violation counter.
    # Position: AFTER cost_cap_falls_back, BEFORE profile_fallback per CONTEXT 64
    # accretion (mirrors P61 placement rule one slot to the right). Counter
    # from src.editor.spec_conformance.get_conformance_violations. Reset at
    # run start by reset_conformance_violations() in main(); read fail-soft
    # at OK-log emit site (try/except → 0). Lockstep with _LOG_OK_RE in
    # src/web/main.py + 4 test files per reference_ok_log_field_accretion.md.
    # Position invariant: snippet_rejected < giveaway_rejected <
    # cost_cap_falls_back < conformance_violations < profile_fallback.
    # Counter increments PER VIOLATION (not per digest run) — a digest with
    # 3 malformed entries + 1 newsletter silent-drop emits =4. Block-mode
    # also flips a separate `_LAST_BLOCK_FAIL` sentinel which run_pipeline
    # reads to swap `digest=skipped` -> `digest=conformance_fail`.
    conformance_violations: int = 0,
    # Phase 65 / DEDUP-BGE-01 — bge-m3 semantic dedup reject counter.
    # Position: AFTER conformance_violations, BEFORE profile_fallback per
    # CONTEXT 65 accretion (mirrors P64 placement rule one slot to the right).
    # Counter from src.llm.embedding.get_dedup_rejected. Reset at run start
    # by reset_dedup_rejected() in main(); read fail-soft at OK-log emit
    # site (try/except → 0). Lockstep with _LOG_OK_RE in src/web/main.py
    # + 4 test files per reference_ok_log_field_accretion.md.
    # Position invariant: cost_cap_falls_back < conformance_violations <
    # dedup_rejected < profile_fallback.
    dedup_rejected: int = 0,
    # Phase 69 / DRIFT-DET-01 — continual-learning drift detector flag count.
    # Position: AFTER dedup_rejected, BEFORE profile_fallback per CONTEXT 69
    # accretion. Counter read fail-soft at emit site (SELECT COUNT(*) FROM
    # drift_snapshots WHERE is_aggregate=1 AND is_flagged=1 AND snapshot_at
    # >= last-7d). Reset NOT needed — value is derived live from a
    # rolling-7d query, not from a process-global counter. Lockstep with
    # _LOG_OK_RE in src/web/main.py + 4 test files.
    # Position invariant: cost_cap_falls_back < conformance_violations <
    # dedup_rejected < drift_flags < profile_fallback.
    drift_flags: int = 0,
    # Phase 72 / EVENT-CLUSTER-01 — release-day event-cluster fire count.
    # Position: AFTER drift_flags, BEFORE profile_fallback per CONTEXT 72
    # accretion. Counter from src.agents.editor.get_event_clusters_count
    # (module-level, reset at create_daily_digest start; fail-soft → 0).
    # Lockstep with _LOG_OK_RE in src/web/main.py + 4 test files per
    # reference_ok_log_field_accretion.md.
    # Position invariant: dedup_rejected < drift_flags < event_clusters <
    # profile_fallback.
    event_clusters: int = 0,
    # Phase 96 / IMAP-TRIAGE-01 (Plan 96-02) — per-class email_class triage
    # counters surfaced for operator telemetry. Counters fire for ALL
    # classified posts (newsletter/action proceed to scorer, support/personal
    # are skipped pre-LLM) — the OK log reports the FULL classification
    # distribution, not just the skip subset. NULL email_class bypasses
    # entirely and is invisible here, by design.
    # Position: AFTER event_clusters, BEFORE profile_fallback per CONTEXT 96
    # accretion. Counters from src.agents.curator.get_email_triage_counts
    # (module-level dict, reset at run start by reset_email_triage_counts()
    # in main(); read fail-soft at OK-log emit site).
    # Lockstep with _LOG_OK_RE in src/web/main.py + 4 test files per
    # reference_ok_log_accretion_6_file_lockstep.md.
    # Position invariant (extended): drift_flags < event_clusters <
    # email_triage_newsletter < email_triage_support < email_triage_personal
    # < email_triage_action < profile_fallback.
    email_triage_newsletter: int = 0,
    email_triage_support: int = 0,
    email_triage_personal: int = 0,
    email_triage_action: int = 0,
    # Phase 99 / CLUSTER-PRE-01 (Plan 99-02) — env-gated cluster pre-filter
    # propagation count. Counter from src.agents.curator.get_cluster_propagated_count
    # (module-level int, reset at run start by reset_cluster_propagated_count()
    # in main(); read fail-soft at OK-log emit site).
    # Position: AFTER email_triage_action, BEFORE profile_fallback per
    # CONTEXT 99 accretion. Lockstep with _LOG_OK_RE in src/web/main.py +
    # 4 test files per reference_ok_log_accretion_6_file_lockstep.md.
    # Position invariant (extended): email_triage_action < cluster_propagated
    # < profile_fallback. Default 0 means CHROMA_CLUSTER_PREFILTER=off OR
    # no multi-member clusters this run.
    cluster_propagated: int = 0,
) -> str:
    """D-08 + Phase 7 D-19 + Phase 23 PIPE-02 (TG) + Phase 26 PIPE-02 (IG) + Phase 27 SIG-02 + Phase 28 SIG-02 + Phase 29 SIG-02 + Phase 30 SIG-04 + Phase 31 SIG-04 + Phase 33 SIG-04 + Phase 33 WR-04:
    '<ISO> | OK | collected=N curated=M digest=state tg_discovered=N ig_collected=N cognition_tagged=N gaming_q=N health_days=N dataset_size=N aggregate_imputed=N replay_violations=N replay_orphans=N profile_fallback=true|false drafts_pending=N'.

    tg_discovered field added Phase 23 (PIPE-02 TG side). ig_collected
    added Phase 26 (PIPE-02 IG side) — per D-19 it lives AFTER
    tg_discovered, BEFORE profile_fallback. 0 means the Instagram
    collector did not run (INSTAGRAM_USER unset, warmup gate refused, or
    soft-block abort) — explicit 0 distinguishes 'IG skipped' from 'old
    log format predating field'.

    cognition_tagged added Phase 27 (SIG-02) — write-time cognition
    counter from src.database.client._reactions_with_cognition.
    Position: between ig_collected= and profile_fallback= per D-19
    accretion pattern. Counter reset at run start (D-17). Backfill
    script does NOT touch this counter (it logs to stderr only).

    gaming_q added Phase 28 (SIG-02) — read-only ingest counter from
    src.gaming.ingest.ingest_gaming_signal (returns
    (gaming_q, skipped_old, no_match); only gaming_q surfaces on OK line).
    Position: between cognition_tagged= and profile_fallback= per D-12
    accretion pattern. Counter is per-call (no global reset needed).

    health_days added Phase 29 (SIG-02) — read-only ingest counter from
    src.health.ingest.ingest_health_signal (returns
    (health_days, days_partial, skipped_unparseable); only health_days
    surfaces on OK line). Position: between gaming_q= and profile_fallback=
    per D-23 accretion pattern. Counter is per-call (no global reset needed).

    dataset_size added Phase 30 (RL-03 / SIG-04) — live COUNT(*) of
    curation_logs rows with human_reward IS NOT NULL. Position: between
    health_days= and profile_fallback= per D-RL-03 accretion pattern.
    Counter is per-call (no global reset needed). dataset_size is the FIRST
    OK-log field that needs a live DB query — the SELECT MUST run INSIDE
    the try/finally surrounding db.close() (see RESEARCH Pitfall 6).
    Fail-soft: any error → WARNING + dataset_size=0.

    aggregate_imputed added Phase 31 (AGG-03 / SIG-04) — total imputed channels
    across all aggregated rows in this run (sum of popcount over imputed_mask
    column). Position: between dataset_size= and profile_fallback= per CONTEXT
    AGG-03 accretion pattern. Counter is per-call (no global reset needed).
    Fail-soft: any error in aggregate_rewards() → WARNING + aggregate_imputed=0.
    Placement: aggregate_rewards() call is INSIDE the SUCCESS-path try block
    AFTER the dataset_size SELECT, BEFORE the finally→db.close() line.

    replay_violations added Phase 33 (OFFLINE-02 / SIG-04) — count of contract
    violations detected by src.rl.offline_replay.run_offline_replay during the
    per-pool replay. Position: between aggregate_imputed= and replay_orphans=
    per CONTEXT 33 accretion pattern. Counter is per-call (no global reset
    needed). Fail-soft: any error in run_offline_replay() → WARNING +
    replay_violations=0. Placement: offline_replay() call is INSIDE the
    SUCCESS-path try block AFTER aggregate_rewards(), BEFORE the
    finally→db.close() line. Non-zero value blocks Phase 35 trainer entry
    (CONTEXT 33 line 60).

    replay_orphans added Phase 33 review (WR-04) — count of env._pool entries
    skipped because no curation_logs row exists for the post_id. Surfaced so
    silent DB/CSV drift (orphan_count ≫ rows_written) cannot mask a P35 DATA
    GATE false-clear. Position: between replay_violations= and
    profile_fallback=. Fail-soft alongside replay_violations: any error in
    run_offline_replay() → WARNING + replay_orphans=0.

    divergence_logged added Phase 52 (DIVERGE-02 / NFR-05) — count of
    scorer_divergence rows logged this run via shadow fan-out. Position:
    between truncation_count= and profile_fallback= per CONTEXT 52 accretion.
    Counter is process-global (mirrors _truncation_count). Reset at run start
    by reset_divergence_logged() in main(). Read fail-soft at OK-log emit site.

    claude_sampled added Phase 52 (DIVERGE-03 / NFR-05) — count of posts that
    crossed both the Bernoulli rate AND the hourly bucket gate; one full
    Claude scorer call per increment. Position: between divergence_logged=
    and claude_calls_skipped=. Reset/read pattern identical to divergence_logged.

    claude_calls_skipped added Phase 52 (DIVERGE-03 / NFR-05) — count of times
    ClaudeRateLimitError was raised inside the Claude shadow branch (stderr or
    stdout matched a rate-limit marker). Non-zero values are operator-visible
    signal that SCORER_SAMPLING_RATE should be lowered or CLAUDE_HOURLY_CAP
    adjusted. Position: between claude_sampled= and profile_fallback=.

    profile_fallback field added Phase 7 (META-04). drafts_pending field
    added Task 5. Older log lines without these fields are still parsed
    by src/web/main.py:_LOG_OK_RE (the regex makes optional fields optional).
    """
    fallback_str = "true" if profile_fallback else "false"
    return (
        f"{_now_iso()} | OK | "
        f"collected={collected} curated={curated} digest={digest_state} "
        f"tg_discovered={tg_discovered} ig_collected={ig_collected} "
        f"cognition_tagged={cognition_tagged} gaming_q={gaming_q} "
        f"health_days={health_days} dataset_size={dataset_size} "
        f"aggregate_imputed={aggregate_imputed} "
        f"replay_violations={replay_violations} "
        f"replay_orphans={replay_orphans} "
        f"score_drift_flags={score_drift_flags} "
        f"truncation_count={truncation_count} "
        f"divergence_logged={divergence_logged} "
        f"claude_sampled={claude_sampled} "
        f"claude_calls_skipped={claude_calls_skipped} "
        f"snippet_rejected={snippet_rejected} "
        f"giveaway_rejected={giveaway_rejected} "
        f"cost_cap_falls_back={cost_cap_falls_back} "
        f"conformance_violations={conformance_violations} "
        f"dedup_rejected={dedup_rejected} "
        f"drift_flags={drift_flags} "
        f"event_clusters={event_clusters} "
        f"email_triage_newsletter={email_triage_newsletter} "
        f"email_triage_support={email_triage_support} "
        f"email_triage_personal={email_triage_personal} "
        f"email_triage_action={email_triage_action} "
        f"cluster_propagated={cluster_propagated} "
        f"profile_fallback={fallback_str} drafts_pending={drafts_pending}"
    )


def _format_fail_line(err: BaseException) -> str:
    """
    D-08: '<ISO> | FAIL | error=<one-line message>'.

    The exception's str() may contain newlines; we collapse to single-line
    by replacing whitespace runs with single spaces, then truncate to 500
    chars to keep the log file's lines bounded.
    """
    msg = " ".join(str(err).split())
    if len(msg) > 500:
        msg = msg[:497] + "..."
    return f"{_now_iso()} | FAIL | error={msg}"


@contextmanager
def _profile_fallback_capture():
    """Context manager that captures 'PROFILE FALLBACK' WARNING messages emitted by the
    src.profile.loader module during the pipeline run.

    Usage:
        with _profile_fallback_capture() as fallback_seen:
            ...do pipeline work...
        if fallback_seen["flag"]:
            ...

    fallback_seen is a dict whose 'flag' key is False initially; the embedded logging.Filter
    sets it to True the first time a record containing 'PROFILE FALLBACK' passes through
    the loader's logger. The filter is unconditionally removed on context exit (success or
    exception) so subsequent runs / tests are not polluted.
    """
    fallback_seen: dict[str, bool] = {"flag": False}

    class _Filter(_logging.Filter):
        def filter(self, record: _logging.LogRecord) -> bool:
            if "PROFILE FALLBACK" in record.getMessage():
                fallback_seen["flag"] = True
            return True

    profile_logger = _logging.getLogger("src.profile.loader")
    f = _Filter()
    profile_logger.addFilter(f)
    try:
        yield fallback_seen
    finally:
        profile_logger.removeFilter(f)


TEST_POSTS = [
    (1, "post_1", "OpenAI releases GPT-5 with reasoning capabilities surpassing PhD-level benchmarks.", "Tech Crunch", "https://example.com/1"),
    (1, "post_2", "Local bakery introduces new sourdough recipe with ancient grains.", "Food Blog", "https://example.com/2"),
    (1, "post_3", "Google DeepMind publishes multi-agent framework for autonomous coding assistants.", "DeepMind Blog", "https://example.com/3"),
    (1, "post_4", "New study: Strength training improves cognitive function and stress resilience.", "Sports Science", "https://example.com/4"),
    (1, "post_5", "Stock market hits record high on tech earnings beat.", "Bloomberg", "https://example.com/5"),
]


def parse_args(argv=None):
    """
    Parse pipeline arguments. Exposed as named function for testability.

    Args:
        argv: argument list (defaults to sys.argv[1:] when None)
    """
    parser = argparse.ArgumentParser(
        description="AutoRSS pipeline: collect -> curate -> digest"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "collect-only", "curate-only", "digest-only", "meta"],
        default="full",
        help="Pipeline stages to run (default: full). 'meta' generates weekly HOT analysis.",
    )
    # Deprecated alias — kept for backward compat with CLAUDE.md instructions and scripts
    parser.add_argument(
        "--no-collect",
        dest="no_collect",
        action="store_true",
        default=False,
        help="[DEPRECATED] Use --mode curate-only --seed-test-data instead. "
             "Kept for backward compatibility.",
    )
    parser.add_argument(
        "--seed-test-data",
        dest="seed_test_data",
        action="store_true",
        default=False,
        help="Seed TEST_POSTS into the DB before running curate/digest stages. "
             "Use for local testing without Telegram auth.",
    )
    return parser.parse_args(argv)


async def check_goodhart_categories(db: aiosqlite.Connection, log_path: str | None = None) -> None:
    """v1.3 Phase 7: alert if any category dominates curated posts (>40% over 7 days)."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).replace(tzinfo=None).isoformat()
    async with db.execute(
        "SELECT COALESCE(s.category,'other') AS cat, COUNT(*) AS cnt "
        "FROM raw_posts rp "
        "JOIN sources s ON s.id = rp.source_id "
        "WHERE rp.status='curated' AND rp.published_at >= ? "
        "GROUP BY cat",
        (cutoff,),
    ) as cur:
        rows = await cur.fetchall()

    if not rows:
        return

    total = sum(r[1] for r in rows)
    if total == 0:
        return

    threshold = 40.0
    actual_log_path = log_path or str(Path(__file__).resolve().parent / "logs" / "pipeline.log")

    for cat, cnt in rows:
        share = cnt / total * 100
        if share > threshold:
            ts = datetime.now(timezone.utc).astimezone().isoformat()
            line = (
                f"{ts} | WARN | goodhart: category '{cat}' = {share:.1f}% "
                f"of curated posts (7d, total={total})\n"
            )
            _logging.getLogger(__name__).warning(line.strip())
            Path(actual_log_path).parent.mkdir(exist_ok=True)
            with open(actual_log_path, "a", encoding="utf-8") as f:
                f.write(line)


async def main():
    args = parse_args()

    # Phase 43 (999.21): reset per-run truncation counter before any curator work.
    try:
        from src.llm.routing import reset_truncation_count
        reset_truncation_count()
    except Exception:
        pass

    # Phase 52 / NFR-05: reset shadow telemetry counters before any curator work.
    try:
        from src.llm.shadow_router import (
            reset_divergence_logged,
            reset_claude_sampled,
            reset_claude_calls_skipped,
        )
        reset_divergence_logged()
        reset_claude_sampled()
        reset_claude_calls_skipped()
    except Exception:
        pass

    # Phase 59 / SNIPPET-01: reset snippet-quality-gate counter before any
    # curator work so each run reports only its own delta (mirrors P52 reset).
    try:
        from src.curator.snippet_quality import reset_snippet_rejected
        reset_snippet_rejected()
    except Exception:
        pass

    # Phase 60 / GIVEAWAY-01: reset giveaway-regex-gate counter before any
    # curator work so each run reports only its own delta (mirrors P59).
    try:
        from src.curator.giveaway_filter import reset_giveaway_rejected
        reset_giveaway_rejected()
    except Exception:
        pass

    # Phase 61 / COST-CAP-01: reset per-category USD cap fallback counter
    # before any curator work so each run reports only its own delta
    # (mirrors P60). Counter also lives behind a TTL cache; the cache itself
    # is process-global and intentionally NOT reset here (a per-run reset
    # would defeat the 60s rate-limit purpose).
    try:
        from src.llm.cost_cap import reset_cost_cap_falls_back
        reset_cost_cap_falls_back()
    except Exception:
        pass

    # Phase 64 / DIGEST-CONFORM-01: reset spec-conformance counter + block-fail
    # sentinel before any editor work so each run reports only its own delta
    # (mirrors P61). Counter is process-global; sentinel is single-bool flipped
    # by editor.create_daily_digest when DIGEST_CONFORMANCE_MODE=block + the
    # validator surfaces ≥1 violation.
    try:
        from src.editor.spec_conformance import (
            reset_conformance_violations,
            reset_last_conformance_block_fail,
        )
        reset_conformance_violations()
        reset_last_conformance_block_fail()
    except Exception:
        pass

    # Phase 65 / DEDUP-BGE-01: reset bge-m3 dedup counter before any
    # insert_raw_post work so each run reports only its own delta (mirrors P64).
    try:
        from src.llm.embedding import reset_dedup_rejected
        reset_dedup_rejected()
    except Exception:
        pass

    # Phase 96 / IMAP-TRIAGE-01 (Plan 96-02): reset 4 per-class email_triage
    # counters before any curator work so each run reports only its own delta.
    # WR-02 fix (Phase 96 review): position is "per-process delta" — placed
    # BEFORE --mode/--no-collect resolution intentionally so multi-stage runs
    # within one process share a single counter set (vs P59/P60 which reset
    # AFTER mode resolution for per-stage isolation). Counter is process-
    # global dict, mutated in-place by process_unprocessed() for every
    # classified post.
    try:
        from src.agents.curator import reset_email_triage_counts
        reset_email_triage_counts()
    except Exception:
        pass

    # Phase 99 / CLUSTER-PRE-01 (Plan 99-02): reset cluster-propagated counter
    # before any curator work so each run reports only its own delta. Counter
    # increments inside process_unprocessed when CHROMA_CLUSTER_PREFILTER=on.
    try:
        from src.agents.curator import reset_cluster_propagated_count
        reset_cluster_propagated_count()
    except Exception:
        pass

    # Phase 103 / CLUSTER-FLOOR-01 (review BL-01): reset cluster-floor-skipped
    # counter at run start (same lifecycle as cluster_propagated above). Without
    # this, the module-level int leaks across runs in long-lived processes
    # (uvicorn) and inflates cost_snapshots.cluster_floor_skipped_n.
    try:
        from src.agents.curator import reset_cluster_floor_skipped_count
        reset_cluster_floor_skipped_count()
    except Exception:
        pass

    # Resolve effective mode: --no-collect alias overrides --mode
    if args.no_collect:
        import warnings
        warnings.warn(
            "--no-collect is deprecated. Use --mode curate-only --seed-test-data instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        mode = "curate-only"
        # Preserve backward-compat: --no-collect used to seed test data
        seed = True
    else:
        mode = args.mode
        seed = args.seed_test_data

    run_collect = mode in ("full", "collect-only")
    run_curate  = mode in ("full", "curate-only")
    run_digest  = mode in ("full", "digest-only")
    run_meta    = mode == "meta"

    # AUTO-02: counters live OUTSIDE the try so they survive into the failure
    # branch. Defaults reflect "stage did not run".
    collected_count = 0
    curated_count = 0
    digest_state = "skipped"
    # Phase 23 PIPE-02 (TG side): count from discover_telegram_dialogs.
    # Default 0 means discovery did not run (mode=off, missing config,
    # collect stage skipped, or pre-failure path).
    n_disc = 0
    # Phase 26 PIPE-02 (IG side): count of IG-platform posts inserted in
    # this run (delta of raw_posts WHERE platform='instagram' across the
    # collect stage). Default 0 = collector did not run (INSTAGRAM_USER
    # unset, warmup gate refused, or soft-block abort).
    n_ig = 0

    # Idea 2 (Global Workspace): fresh workspace per run.
    reset_workspace()
    workspace = get_workspace()

    # Phase 27 / NB-03 + SIG-02 (D-17): reset per-run cognition counter so the
    # OK log emits this run's delta only. Backfill script (Plan 27-04) logs
    # to stderr and does NOT touch this counter.
    reset_cognition_counters()

    # Phase 7 META-05 (D-15): per-run random hex tag, generated ONCE.
    seal_tag = secrets.token_hex(8)

    # Phase 7 META-04 (D-19): capture PROFILE FALLBACK warnings via context manager
    with _profile_fallback_capture() as fallback_seen:
        try:
            # NFR-06: open_db() returns a pre-awaited Connection; we manage close
            # explicitly via try/finally to avoid aiosqlite's __aenter__ re-awaiting
            # an already-started thread (which raises "threads can only be started once").
            db = await open_db(_get_db_path())
            try:
                await init_db(db)
                await migrate_db(db)
                await ensure_sources(db, load_sources())

                # Seed test data if requested (independent of --mode)
                if seed:
                    print("Seeding test posts...")
                    for post in TEST_POSTS:
                        post_id = await insert_raw_post(db, *post)
                        print(f"  Inserted post {post_id}: {post[2][:60]}...")

                # Collect stage
                if run_collect:
                    # WARNING-01 fix: snapshot IG-platform raw_post count BEFORE
                    # collect_all so we can emit the true delta of THIS run. The
                    # post-collect query that counts unprocessed IG posts conflates
                    # curator-lag (rows left over from prior runs) with this-run
                    # inserts, inflating ig_collected. Capture before/after instead.
                    async with db.execute(
                        "SELECT COUNT(*) FROM raw_posts WHERE platform = 'instagram'"
                    ) as cur:
                        _ig_before_row = await cur.fetchone()
                    ig_before = _ig_before_row[0] if _ig_before_row else 0

                    api_id_str = os.environ.get("TG_API_ID")
                    api_hash = os.environ.get("TG_API_HASH")
                    if not api_id_str or not api_hash:
                        sys.exit(
                            "Error: TG_API_ID and TG_API_HASH must be set in .env "
                            "(copy .env.example)"
                        )
                    api_id = int(api_id_str)
                    session = os.environ.get("TG_SESSION_PATH", "telegram_session")
                    tg_client = TelegramCollectorClient(api_id, api_hash, session)

                    # Phase 10: SSRF-safe HTTP session shared across the entire collect
                    # stage. Pattern 4 from 10-RESEARCH.md — pipeline-scoped session
                    # for connection pooling + global Semaphore(5) for D-10 concurrency
                    # cap across ALL collectors (Telegram / Reddit / YouTube / Email).
                    # CR-02: ttl_dns_cache=0 + use_dns_cache=False forces every
                    # redirect hop to re-resolve through SSRFSafeResolver, closing
                    # the DNS-rebinding window where a hostname's resolution could
                    # change between validation and TCP connect. Tradeoff: one
                    # extra DNS lookup per request — acceptable for the curator's
                    # low-volume, security-first profile.
                    # CR-04: tg_client.start/stop lives INSIDE the aiohttp session
                    # block AND inside try/finally so that a failure in collect_all
                    # never leaks the Telethon connection. Previously stop() lived
                    # after the `async with` body and was skipped on any exception.
                    async with aiohttp.ClientSession(
                        connector=aiohttp.TCPConnector(
                            resolver=SSRFSafeResolver(),
                            ttl_dns_cache=0,
                            use_dns_cache=False,
                        ),
                        headers={"User-Agent": "autorss-feed/1.1 (+private-curator)"},
                    ) as extract_session:
                        await tg_client.start()
                        try:
                            extract_semaphore = asyncio.Semaphore(5)  # D-10
                            # Auto-discover unread subscribed channels/groups
                            # and persist as sources before collect runs.
                            # User-confirmed scope (2026-05-13): channels +
                            # groups only, no DMs.
                            n_disc = await discover_telegram_dialogs(db, tg_client)
                            if n_disc:
                                print(f"Auto-discovered {n_disc} unread Telegram dialogs")
                            print("Running Telegram collector...")
                            await collect_all(
                                db, tg_client, extract_session, extract_semaphore
                            )
                        finally:
                            await tg_client.stop()

                    async with db.execute(
                        "SELECT COUNT(*) FROM raw_posts WHERE status = 'unprocessed'"
                    ) as cur:
                        row = await cur.fetchone()
                    collected_count = row[0] if row else 0

                    # Phase 26 PIPE-02 IG side: count of IG-platform posts
                    # INSERTED THIS RUN. WARNING-01 fix: use the before/after
                    # delta of `platform='instagram'` rows around collect_all
                    # so curator-lag from prior runs doesn't inflate the count.
                    # 0 when IG collector silent-skip / warmup-gate refusal /
                    # soft-block abort.
                    async with db.execute(
                        "SELECT COUNT(*) FROM raw_posts WHERE platform = 'instagram'"
                    ) as cur:
                        _ig_after_row = await cur.fetchone()
                    ig_after = _ig_after_row[0] if _ig_after_row else 0
                    n_ig = max(0, ig_after - ig_before)

                # Enrich stage: fetch full article bodies for posts with external URLs
                if run_collect:
                    try:
                        enriched_count = await enrich_web_links(db)
                        if enriched_count:
                            print(f"  Enriched {enriched_count} posts with full article body")
                    except Exception as _enrich_err:
                        _logging.getLogger(__name__).warning("Web enrichment failed (non-fatal): %s", _enrich_err)

                # Curate stage
                if run_curate:
                    print("\nRunning curator agent...")
                    await process_unprocessed(db, OLLAMA_HOST, CURATOR_MODEL, seal_tag=seal_tag)

                    db.row_factory = aiosqlite.Row
                    async with db.execute("SELECT id, status, raw_text FROM raw_posts") as cursor:
                        rows = await cursor.fetchall()
                    for row in rows:
                        print(f"  [{row['status']:>10}] {row['raw_text'][:60]}...")

                    async with db.execute(
                        "SELECT COUNT(*) FROM raw_posts WHERE status = 'curated'"
                    ) as cur:
                        crow = await cur.fetchone()
                    curated_count = crow[0] if crow else 0

                    # Idea 2 (Global Workspace): broadcast curator summary
                    rejected_count = sum(1 for r in rows if r["status"] == "rejected")
                    await workspace.publish("curator.done", {
                        "curated": curated_count, "rejected": rejected_count,
                    })

                    # Auto-expand paper pool via Semantic Scholar (free, ~30s)
                    try:
                        import subprocess as _sp
                        _sp.run(
                            [sys.executable, "scripts/discover_similar.py", "--limit", "3"],
                            timeout=120, check=False,
                        )
                        print("  Semantic Scholar discovery: done")
                    except Exception as _e:
                        print(f"  Semantic Scholar discovery skipped: {_e}")

                # Digest stage
                if run_digest:
                    print(f"\nRunning editor agent (backend={EDITOR_BACKEND})...")
                    await create_daily_digest(db, OLLAMA_HOST, EDITOR_MODEL, backend=EDITOR_BACKEND)

                    db.row_factory = aiosqlite.Row
                    async with db.execute(
                        "SELECT id, created_at FROM digests ORDER BY id DESC LIMIT 1"
                    ) as cursor:
                        digest = await cursor.fetchone()
                    # Phase 64 / DIGEST-CONFORM-01: if create_daily_digest
                    # returned without writing a row AND the block-fail sentinel
                    # is set, the editor refused the digest because it failed
                    # the spec-conformance gate in block mode. Surface as a
                    # dedicated state on the OK log line so the operator can
                    # distinguish "no curated posts" from "digest blocked".
                    try:
                        from src.editor.spec_conformance import (
                            get_last_conformance_block_fail,
                        )
                        _block_fail = get_last_conformance_block_fail()
                    except Exception:
                        _block_fail = False

                    if digest:
                        print(f"  Digest #{digest['id']} created at {digest['created_at']}")
                        digest_state = "created"

                        # Idea 2: broadcast digest.created
                        await workspace.publish("digest.created", {
                            "digest_id": digest["id"], "post_count": curated_count,
                        })

                        # Auto: extract tool mentions from new digest
                        try:
                            from src.analysis.tool_extractor import extract_all_digests
                            new_mentions = await extract_all_digests(db)
                            if new_mentions:
                                print(f"  Tool extractor: {new_mentions} new mentions indexed")
                        except Exception as _te:
                            print(f"  Tool extractor skipped: {_te}")

                        # Auto: rebuild notebook index (non-blocking background subprocess)
                        try:
                            import subprocess as _sp
                            _sp.Popen(
                                [sys.executable, "scripts/notebook.py", "--build"],
                                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                            )
                            print("  Notebook index rebuild started (background)")
                        except Exception as _nbe:
                            print(f"  Notebook rebuild skipped: {_nbe}")

                        # Voice digest (edge-tts, free — no API key)
                        try:
                            from src.agents.voice_digest import generate_audio
                            async with db.execute(
                                "SELECT markdown_content FROM digests WHERE id=?",
                                (digest["id"],),
                            ) as cur:
                                md_row = await cur.fetchone()
                            if md_row and md_row[0]:
                                print("  Generating audio digest...")
                                audio_path = await generate_audio(md_row[0])
                                print(f"  Audio saved: {audio_path.name}")
                        except Exception as _ve:
                            print(f"  Audio generation skipped: {_ve}")
                    else:
                        if _block_fail:
                            print("  Digest BLOCKED by spec-conformance gate")
                            digest_state = "conformance_fail"
                        else:
                            print("  No digest created (no curated posts?)")
                            digest_state = "skipped"

                # Idea 5 (HOT meta-digest): --mode meta
                if run_meta:
                    print("\nRunning HOT meta-analyst...")
                    from src.agents.meta_analyst import analyze_weekly, save_meta_digest
                    analysis = await analyze_weekly(
                        db, n_digests=7, backend=EDITOR_BACKEND,
                        ollama_host=OLLAMA_HOST, ollama_model=EDITOR_MODEL,
                    )
                    if analysis:
                        meta_id = await save_meta_digest(db, analysis)
                        print(f"  Meta-digest #{meta_id} saved.")
                        print("\n" + analysis[:500] + ("..." if len(analysis) > 500 else ""))
                    else:
                        print("  Not enough digests for meta-analysis (need >=2).")

                # v1.3: Goodhart category monitor — warn if any category dominates
                await check_goodhart_categories(db)

                # Phase 30 / RL-03 + SIG-04: count rows with non-NULL human_reward.
                # MUST be inside this try block — db is closed in `finally` below,
                # and this is the FIRST OK-log field needing live DB access (gaming
                # and health read CSV files outside this block). Fail-soft per
                # CONTEXT D-RL-03: any error logs WARNING + emits dataset_size=0.
                try:
                    async with db.execute(
                        "SELECT COUNT(*) FROM curation_logs WHERE human_reward IS NOT NULL"
                    ) as _ds_cur:
                        _ds_row = await _ds_cur.fetchone()
                    n_dataset_size = _ds_row[0] if _ds_row else 0
                except Exception as _ds_err:
                    _logging.getLogger(__name__).warning(
                        "dataset_size count failed (non-fatal): %s", _ds_err
                    )
                    n_dataset_size = 0

                # Phase 31 / AGG-03 + SIG-04: aggregate 4 reward channels into
                # data/rl/aggregated_rewards.csv. Placement: INSIDE the try block
                # AFTER the dataset_size SELECT, BEFORE the finally → db.close().
                # Fail-soft per CONTEXT empty-rows guard: any error logs WARNING +
                # emits aggregate_imputed=0. Defensive local import inside try wrap
                # is belt-and-suspenders for a fresh subsystem (mirrors Phase 28
                # D-09 + Phase 29 D-17 + Phase 30 D-RL-03 pattern).
                try:
                    from src.rl.aggregate import aggregate_rewards
                    # CR-01 fix: pass explicit db_path so DB_PATH env is honored
                    # (otherwise aggregator reads the live <project_root>/curator.db
                    # and ignores test/operator overrides). out_csv left default —
                    # operator path is fine in production; tests override explicitly.
                    _, n_aggregate_imputed = aggregate_rewards(
                        db_path=Path(_get_db_path())
                    )
                except Exception as _agg_err:
                    _logging.getLogger(__name__).warning(
                        "aggregate_rewards failed (non-fatal): %s", _agg_err
                    )
                    n_aggregate_imputed = 0

                # Phase 33 / OFFLINE-02 + SIG-04: replay env._pool through
                # run_offline_replay contract checks. Placement: INSIDE the
                # try block AFTER aggregate_rewards (so the aggregated CSV
                # the env reads is fresh), BEFORE finally→db.close() (so
                # any DB-path overrides remain in scope). Fail-soft per
                # CONTEXT 33 line 33: any error in offline_replay (incl.
                # CuratorEnv constructor failure, e.g. empty pool) logs
                # WARNING + emits replay_violations=0. Defensive local
                # import inside try wrap is belt-and-suspenders for a fresh
                # subsystem (mirrors Phase 28 D-09 + Phase 29 D-17 +
                # Phase 30 D-RL-03 + Phase 31 AGG-03 patterns).
                try:
                    from src.rl.curator_env import CuratorEnv
                    from src.rl.offline_replay import run_offline_replay
                    # WR-02 (P33 review): preserve the literal ``:memory:`` URI
                    # string when DB_PATH=":memory:" is set (test/operator
                    # override). ``Path(":memory:")`` on Windows mangles it
                    # into a relative file path with literal-colon segments
                    # → sqlite3 silently opens a junk file, the env's empty
                    # pool raises RuntimeError, fail-soft below masks it as
                    # ``replay_violations=0`` (false-clear of P35 DATA GATE).
                    _db_str = _get_db_path()
                    _db_arg = _db_str if _db_str == ":memory:" else Path(_db_str)
                    _env = CuratorEnv(db_path=_db_arg)
                    # WR-04 (P33 review): 3-tuple return — capture orphan_count
                    # so it can be surfaced on the OK log line as
                    # ``replay_orphans=N`` (P35 DATA GATE input).
                    _, n_replay_violations, n_replay_orphans = run_offline_replay(
                        _env,
                        db_path=_db_arg,
                        output_path=(
                            Path(__file__).resolve().parent
                            / "data" / "rl" / "offline_replay.csv"
                        ),
                    )
                except Exception as _replay_err:
                    _logging.getLogger(__name__).warning(
                        "offline_replay failed (non-fatal): %s", _replay_err
                    )
                    n_replay_violations = 0
                    n_replay_orphans = 0

            finally:
                await db.close()

            # Draft pipeline: auto-advance finish-type drafts
            try:
                finish_results = run_finish_worker()
                _logging.getLogger(__name__).info("finish_worker: %s", finish_results)
            except Exception as _fw_err:
                _logging.getLogger(__name__).warning("finish_worker failed: %s", _fw_err)
                finish_results = {"attempted": 0, "succeeded": 0, "failed": 0}

            # Count pending drafts for badge
            try:
                drafts_pending = len(list_pending_drafts())
            except Exception:
                drafts_pending = 0

            if not run_meta:
                # SUCCESS path — write the OK log line.
                # Phase 27 / SIG-02: read write-time cognition counter
                # (tagged, untagged); only `tagged` surfaces on the OK line.
                n_cog_tagged, _n_cog_untagged = get_cognition_counters()
                # Phase 28 / SIG-02: read-only ingest of wtp-plugin gaming state →
                # data/gaming/sessions.csv. Per D-09 fail-soft: ANY error in
                # ingest_gaming_signal logs a WARNING and sets gaming_q=0, pipeline
                # still emits OK log line. Local import + try wrap is defensive
                # belt-and-suspenders for a fresh subsystem.
                try:
                    from src.gaming.ingest import ingest_gaming_signal
                    n_gaming_q, _n_skipped_old, _n_no_match = ingest_gaming_signal()
                except Exception as _gaming_err:
                    _logging.getLogger(__name__).warning(
                        "gaming ingest failed (non-fatal): %s", _gaming_err
                    )
                    n_gaming_q = 0
                # Phase 29 / SIG-02: read-only walk of Obsidian 80_Health/ →
                # data/health/daily.csv. Per D-17 + D-18 fail-soft: ANY error in
                # ingest_health_signal logs a WARNING and sets health_days=0,
                # pipeline still emits OK log line. Local import + try wrap is
                # defensive belt-and-suspenders for a fresh subsystem (mirrors
                # Phase 28 D-09).
                try:
                    from src.health.ingest import ingest_health_signal
                    n_health_days, _n_days_partial, _n_skipped_unparseable = ingest_health_signal()
                except Exception as _health_err:
                    _logging.getLogger(__name__).warning(
                        "health ingest failed (non-fatal): %s", _health_err
                    )
                    n_health_days = 0
                # Phase 38 / OBS-02: read-only count of score_drift flagged rows in last 24h.
                # Per OBS-02 fail-soft: count_recent_flags swallows all errors → returns 0.
                # Position: between replay_orphans= and profile_fallback= per CONTEXT 38 accretion.
                try:
                    from src.observability.score_drift import count_recent_flags as _count_drift_flags
                    n_score_drift_flags = _count_drift_flags("curator.db", hours=24)
                except Exception as _drift_err:
                    _logging.getLogger(__name__).warning(
                        "score_drift count failed (non-fatal): %s", _drift_err
                    )
                    n_score_drift_flags = 0
                # Phase 43 (999.21): per-backend truncation counter — process global
                try:
                    from src.llm.routing import get_truncation_count
                    n_truncations = get_truncation_count()
                except Exception:
                    n_truncations = 0
                # Phase 52 / NFR-05: shadow telemetry counters — process globals (fail-soft).
                try:
                    from src.llm.shadow_router import (
                        get_divergence_logged,
                        get_claude_sampled,
                        get_claude_calls_skipped,
                    )
                    n_divergence = get_divergence_logged()
                    n_claude_sampled = get_claude_sampled()
                    n_claude_skipped = get_claude_calls_skipped()
                except Exception:
                    n_divergence = 0
                    n_claude_sampled = 0
                    n_claude_skipped = 0
                # Phase 59 / SNIPPET-01: pre-LLM snippet quality gate counter
                # — process global (fail-soft mirrors P52 pattern).
                try:
                    from src.curator.snippet_quality import get_snippet_rejected
                    n_snippet_rejected = get_snippet_rejected()
                except Exception:
                    n_snippet_rejected = 0
                # Phase 60 / GIVEAWAY-01: pre-LLM giveaway regex gate counter
                # — process global (fail-soft mirrors P59 pattern).
                try:
                    from src.curator.giveaway_filter import get_giveaway_rejected
                    n_giveaway_rejected = get_giveaway_rejected()
                except Exception:
                    n_giveaway_rejected = 0
                # Phase 61 / COST-CAP-01: per-category daily USD cap counter
                # — process global (fail-soft mirrors P60 pattern). Increments
                # when routing.get_backend_for_category() detects cap exceeded.
                try:
                    from src.llm.cost_cap import get_cost_cap_falls_back
                    n_cost_cap_falls_back = get_cost_cap_falls_back()
                except Exception:
                    n_cost_cap_falls_back = 0
                # Phase 64 / DIGEST-CONFORM-01: spec-conformance violation count
                # (per-violation, not per-digest) — process global (fail-soft).
                try:
                    from src.editor.spec_conformance import get_conformance_violations
                    n_conformance_violations = get_conformance_violations()
                except Exception:
                    n_conformance_violations = 0
                # Phase 65 / DEDUP-BGE-01: bge-m3 semantic dedup reject count
                # (per-insert) — process global (fail-soft mirrors P64 pattern).
                try:
                    from src.llm.embedding import get_dedup_rejected
                    n_dedup_rejected = get_dedup_rejected()
                except Exception:
                    n_dedup_rejected = 0
                # Phase 69 / DRIFT-DET-01: count flagged drift aggregate rows
                # in last 7 days (live query, fail-soft to 0).
                try:
                    import sqlite3 as _sqlite_drift
                    with _sqlite_drift.connect("curator.db") as _drift_c:
                        # CR-01 fix: wrap snapshot_at in datetime() so SQLite
                        # parses TZ-aware ISO strings (e.g. '...+00:00' or
                        # '...+05:00') and compares them in UTC against
                        # datetime('now','-7 days'). Lex-compare of the raw
                        # column against UTC SQLite-format would silently
                        # under-count (P52/P65 TZ-mismatch bug class).
                        n_drift_flags = _drift_c.execute(
                            "SELECT COUNT(*) FROM drift_snapshots "
                            "WHERE is_aggregate=1 AND is_flagged=1 "
                            "AND datetime(snapshot_at) >= datetime('now','-7 days')"
                        ).fetchone()[0]
                except Exception:
                    n_drift_flags = 0
                # Phase 72 / EVENT-CLUSTER-01: read editor module-level counter
                # (fail-soft → 0). Counter is set by create_daily_digest based
                # on cluster_recent_events output when EVENT_CLUSTER_MODE=on.
                try:
                    from src.agents.editor import get_event_clusters_count
                    n_event_clusters = get_event_clusters_count()
                except Exception:
                    n_event_clusters = 0
                # Phase 96 / IMAP-TRIAGE-01 (Plan 96-02): fail-soft read of
                # per-class triage counters. Default 0 per class on any
                # import / KeyError so the OK-log emit path never breaks on
                # the telemetry surface.
                try:
                    from src.agents.curator import get_email_triage_counts
                    _et_counts = get_email_triage_counts()
                    n_et_newsletter = int(_et_counts.get("newsletter", 0))
                    n_et_support = int(_et_counts.get("support", 0))
                    n_et_personal = int(_et_counts.get("personal", 0))
                    n_et_action = int(_et_counts.get("action", 0))
                except Exception:
                    n_et_newsletter = 0
                    n_et_support = 0
                    n_et_personal = 0
                    n_et_action = 0
                # Phase 99 / CLUSTER-PRE-01 (Plan 99-02): fail-soft read of
                # cluster-propagated counter. Default 0 on any import / read
                # error so the OK-log emit path never breaks on the telemetry
                # surface (mirrors P59/P60/P96 pattern).
                try:
                    from src.agents.curator import get_cluster_propagated_count
                    n_cluster_propagated = int(get_cluster_propagated_count())
                except Exception:
                    n_cluster_propagated = 0
                _append_log_line(
                    _format_ok_line(
                        collected_count, curated_count, digest_state,
                        profile_fallback=fallback_seen["flag"],
                        drafts_pending=drafts_pending,
                        tg_discovered=n_disc,
                        ig_collected=n_ig,
                        cognition_tagged=n_cog_tagged,
                        gaming_q=n_gaming_q,
                        health_days=n_health_days,
                        dataset_size=n_dataset_size,
                        aggregate_imputed=n_aggregate_imputed,
                        replay_violations=n_replay_violations,
                        replay_orphans=n_replay_orphans,
                        score_drift_flags=n_score_drift_flags,
                        truncation_count=n_truncations,
                        divergence_logged=n_divergence,
                        claude_sampled=n_claude_sampled,
                        claude_calls_skipped=n_claude_skipped,
                        snippet_rejected=n_snippet_rejected,
                        giveaway_rejected=n_giveaway_rejected,
                        cost_cap_falls_back=n_cost_cap_falls_back,
                        conformance_violations=n_conformance_violations,
                        dedup_rejected=n_dedup_rejected,
                        drift_flags=n_drift_flags,
                        event_clusters=n_event_clusters,
                        email_triage_newsletter=n_et_newsletter,
                        email_triage_support=n_et_support,
                        email_triage_personal=n_et_personal,
                        email_triage_action=n_et_action,
                        cluster_propagated=n_cluster_propagated,
                    )
                )
                _launch_snapshot()
                _launch_seed_claude()

            # Idea 2: print workspace stats
            ws_stats = workspace.stats()
            if ws_stats:
                stats_str = ", ".join(f"{t}={n}" for t, n in ws_stats.items())
                print(f"\n[Workspace] {stats_str}")

            print("\nDone. Start web server:")
            print("  .venv/Scripts/python.exe -m uvicorn src.web.main:app --reload")

        except SystemExit:
            raise
        except KeyboardInterrupt:
            _append_log_line(_format_fail_line(KeyboardInterrupt("interrupted by user")))
            _launch_snapshot()
            raise
        except Exception as err:
            _append_log_line(_format_fail_line(err))
            _launch_snapshot()
            print(f"\n  ! Pipeline FAILED: {err}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
