"""Phase 41 — Auto-Remediation Worker.

Reads recent failures (from agent_traces + failure_miner bucket counts) and
applies known idempotent remediations per bucket. Logs every action as an
`agent_traces` row with `agent='auto_remediation'`.

Scheduled task `AutorssFeed_auto_remediation` runs this every 30 min.

CLI:
    python -m src.worker.auto_remediation [--db curator.db] [--dry-run]

Design constraints:
- Sync sqlite3 (mirrors react_worker, score_drift, failure_miner).
- Idempotent — re-running within minutes makes no incremental changes.
- Fail-soft — each handler wrapped in try/except; outcome recorded in trace.
- Bounded — never touches more than last 6h of raw_posts for re-pooling.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.observability import failure_miner, traces

KARAGANDA_TZ = timezone(timedelta(hours=5))

# Bucket → action map. Buckets not in this map are recorded as "no-op".
ACTIONABLE_BUCKETS = {
    "TIMEOUT_CLAUDE",
    "VAULT_OFFLINE",
    "MODEL_404",
    "EDITOR_EMPTY_DIGEST",
}

DEFAULT_TIMEOUT_FLAG_VALUE = "240"  # seconds; consumed by future pipeline runs
DEFAULT_BACKEND_FLAG_VALUE = "ollama"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"

RECENT_FAILURE_WINDOW_HOURS = 24
REPOOL_WINDOW_HOURS = 6


def _default_vault_drafts() -> Path:
    """Resolve the drafts directory from env vars. All reads inside function body (WORKER-12).

    DRAFT_OUTPUT_DIR semantics: this must be the vault ROOT (e.g. /home/user/vault or C:/vault),
    NOT the drafts subdirectory. This function appends "10_Drafts" automatically.
    Do not set DRAFT_OUTPUT_DIR=<vault>/10_Drafts or you will get double-nesting.
    """
    draft_output = os.environ.get("DRAFT_OUTPUT_DIR")
    if draft_output:
        return Path(draft_output) / "10_Drafts"
    return Path(os.environ.get("VAULT_DRAFTS_PATH", "vault/10_Drafts"))


def now_karaganda() -> datetime:
    return datetime.now(KARAGANDA_TZ)


def _gather_bucket_counts(project_root: Path, since: datetime) -> dict[str, int]:
    """Reuse failure_miner parsers to count last-24h failures by bucket."""
    react_jsonl = project_root / "logs" / "react_worker.jsonl"
    pipeline_log = project_root / "logs" / "pipeline.log"
    finish_stderr = project_root / "logs" / "finish_worker.stderr"

    events: list[tuple[datetime, str]] = []
    events.extend(failure_miner.parse_react_jsonl(react_jsonl, since=since))
    events.extend(failure_miner.parse_pipeline_log(pipeline_log, since=since))
    events.extend(failure_miner.parse_finish_worker_stderr(finish_stderr, since=since))
    buckets = failure_miner.aggregate(events)
    return {b: data["count"] for b, data in buckets.items()}


def _gather_recent_trace_errors(db_path: str | Path, since: datetime) -> int:
    """Count agent_traces rows in last 24h with non-zero exit / error result."""
    p = Path(str(db_path))
    if not p.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error:
        return 0
    try:
        traces.ensure_table(conn)
        row = conn.execute(
            "SELECT COUNT(*) FROM agent_traces "
            "WHERE started_at >= ? "
            "AND (result_json LIKE '%\"exit\": \"error\"%' OR result_json LIKE '%\"error\":%')",
            (since.isoformat(),),
        ).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass


def _write_flag(logs_dir: Path, name: str, value: str, *, dry_run: bool) -> dict[str, Any]:
    target = logs_dir / name
    if dry_run:
        return {"action": f"write_flag:{name}", "ok": True, "dry_run": True, "value": value}
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(value, encoding="utf-8")
        return {"action": f"write_flag:{name}", "ok": True, "path": str(target), "value": value}
    except OSError as exc:
        return {"action": f"write_flag:{name}", "ok": False, "error": str(exc)}


def _ensure_vault_dir(*, dry_run: bool) -> dict[str, Any]:
    target = _default_vault_drafts()
    if dry_run:
        return {"action": "mkdir_vault_drafts", "ok": True, "dry_run": True, "path": str(target)}
    try:
        target.mkdir(parents=True, exist_ok=True)
        return {"action": "mkdir_vault_drafts", "ok": True, "path": str(target)}
    except OSError as exc:
        return {"action": "mkdir_vault_drafts", "ok": False, "error": str(exc)}


def _check_ollama_model(model: str = DEFAULT_OLLAMA_MODEL) -> dict[str, Any]:
    """Run `ollama list` and report whether the model is installed."""
    ollama = shutil.which("ollama")
    if not ollama:
        return {"action": "ollama_list", "ok": False, "error": "ollama not on PATH"}
    try:
        result = subprocess.run(
            [ollama, "list"], capture_output=True, text=True, timeout=10, encoding="utf-8",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"action": "ollama_list", "ok": False, "error": str(exc)}
    present = model in (result.stdout or "")
    return {
        "action": "ollama_list", "ok": True,
        "model": model, "present": present,
        "stdout_excerpt": (result.stdout or "")[:200],
    }


def _repool_recent_archived(db_path: str | Path, *, dry_run: bool) -> dict[str, Any]:
    p = Path(str(db_path))
    if not p.exists():
        return {"action": "repool_archived", "ok": False, "error": f"db missing: {db_path}"}
    cutoff = (now_karaganda() - timedelta(hours=REPOOL_WINDOW_HOURS)).isoformat()
    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error as exc:
        return {"action": "repool_archived", "ok": False, "error": str(exc)}
    try:
        candidates = conn.execute(
            "SELECT COUNT(*) FROM raw_posts rp "
            "JOIN curation_logs cl ON cl.post_id = rp.id "
            "WHERE rp.status = 'archived' AND cl.scored_at >= ?",
            (cutoff,),
        ).fetchone()[0]
        if dry_run:
            return {
                "action": "repool_archived", "ok": True, "dry_run": True,
                "candidates": int(candidates), "cutoff": cutoff,
            }
        cur = conn.execute(
            "UPDATE raw_posts SET status = 'curated' "
            "WHERE status = 'archived' AND id IN ("
            "  SELECT cl.post_id FROM curation_logs cl WHERE cl.scored_at >= ?"
            ")",
            (cutoff,),
        )
        conn.commit()
        return {
            "action": "repool_archived", "ok": True,
            "candidates": int(candidates), "updated": cur.rowcount,
            "cutoff": cutoff,
        }
    except sqlite3.Error as exc:
        return {"action": "repool_archived", "ok": False, "error": str(exc)}
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass


def _handle_bucket(
    bucket: str, count: int, *, project_root: Path, db_path: str | Path, dry_run: bool,
) -> dict[str, Any]:
    logs_dir = project_root / "logs"
    if bucket == "TIMEOUT_CLAUDE":
        return _write_flag(logs_dir, "extend_claude_timeout.flag",
                           DEFAULT_TIMEOUT_FLAG_VALUE, dry_run=dry_run)
    if bucket == "VAULT_OFFLINE":
        return _ensure_vault_dir(dry_run=dry_run)
    if bucket == "MODEL_404":
        check = _check_ollama_model()
        flag = _write_flag(logs_dir, "curator_backend.flag",
                           DEFAULT_BACKEND_FLAG_VALUE, dry_run=dry_run)
        return {"action": "fallback_ollama", "ok": flag.get("ok", False),
                "ollama_check": check, "flag": flag}
    if bucket == "EDITOR_EMPTY_DIGEST":
        return _repool_recent_archived(db_path, dry_run=dry_run)
    return {"action": "no-op", "ok": True, "bucket": bucket}


def run_once(
    db_path: str | Path = "curator.db",
    project_root: Path | None = None,
    *,
    dry_run: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Single tick: scan recent failures, apply remediations, log traces.

    Returns a summary dict; also writes one agent_traces row per actioned bucket
    plus a final summary row.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    now = now or now_karaganda()
    since = now - timedelta(hours=RECENT_FAILURE_WINDOW_HOURS)

    run_id = traces.start_run(prefix="auto_remediation")
    bucket_counts = _gather_bucket_counts(project_root, since)
    trace_errors = _gather_recent_trace_errors(db_path, since)

    actions: list[dict[str, Any]] = []
    for bucket in failure_miner.FAILURE_PATTERNS.keys():
        count = bucket_counts.get(bucket, 0)
        if count == 0:
            continue
        if bucket not in ACTIONABLE_BUCKETS:
            actions.append({"bucket": bucket, "count": count,
                            "result": {"action": "record-only", "ok": True}})
            traces.record_step(
                db_path, run_id=run_id, agent="auto_remediation",
                tool=bucket, args={"bucket": bucket, "count": count},
                result={"action": "record-only", "ok": True},
            )
            continue
        outcome = _handle_bucket(
            bucket, count, project_root=project_root, db_path=db_path, dry_run=dry_run,
        )
        actions.append({"bucket": bucket, "count": count, "result": outcome})
        traces.record_step(
            db_path, run_id=run_id, agent="auto_remediation",
            tool=bucket, args={"bucket": bucket, "count": count, "dry_run": dry_run},
            result=outcome,
        )

    summary = {
        "run_id": run_id,
        "scanned_at": now.isoformat(),
        "window_hours": RECENT_FAILURE_WINDOW_HOURS,
        "bucket_counts": bucket_counts,
        "trace_errors_24h": trace_errors,
        "actions": actions,
        "dry_run": dry_run,
    }
    traces.record_step(
        db_path, run_id=run_id, agent="auto_remediation",
        tool="summary", args={"window_hours": RECENT_FAILURE_WINDOW_HOURS},
        result=summary,
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    # Phase 107 TRIM-02: personal paths live in .env now (code defaults are generic).
    # Scheduled-task invocations don't inherit a shell env — load .env explicitly.
    from src.env_loader import load_env

    load_env()
    parser = argparse.ArgumentParser(description="Phase 41 — Auto-Remediation Worker")
    parser.add_argument("--db", default=os.environ.get("AUTORSS_DB_PATH", "curator.db"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip side effects; record traces only")
    args = parser.parse_args(argv)
    summary = run_once(args.db, dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
