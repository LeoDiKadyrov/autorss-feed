"""
Failure cluster miner — scans logs and buckets failures by regex pattern.

Reads:
- logs/react_worker.jsonl — line-delimited JSON, looks for 'error'/'stderr' fields
- logs/pipeline.log — append-only single-line FAIL/WARN/OK format
- logs/finish_worker.stderr — captured stderr from finish_worker subprocess (if exists)

Writes:
- logs/agent_failures.md — markdown weekly report (overwrites each run)

Buckets (8 total):
- TIMEOUT_CLAUDE — claude subprocess timeout
- VAULT_OFFLINE — Obsidian vault directory not reachable
- GATE_MISMATCH — chain_state inconsistency
- MODEL_404 — ollama model not pulled / claude model id unknown
- CP1252_ERROR — Windows console encoding errors with Cyrillic
- COLD_START_CRASH — task scheduler pre-import crash
- EDITOR_EMPTY_DIGEST — editor returned empty markdown
- UNKNOWN — catch-all for unmatched FAIL lines

Pain class: silent "exit 0 no stdout" failures requiring manual log greps.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

# Asia/Karaganda is not in Windows tzdata — use fixed offset (KZ has no DST).
KARAGANDA_TZ = timezone(timedelta(hours=5))

# Bucket regex patterns. Order matters — first match wins; UNKNOWN is last.
FAILURE_PATTERNS: dict[str, re.Pattern[str]] = {
    "TIMEOUT_CLAUDE": re.compile(r"(?i)\b(timeout|timed out|TimeoutError|TimeoutExpired)\b.*\bclaude\b|\bclaude\b.*\b(timeout|timed out|TimeoutError|TimeoutExpired)\b"),
    "VAULT_OFFLINE": re.compile(r"(?i)(D:\\?Obsidian|10_Drafts|vault[_-]?offline)"),
    "GATE_MISMATCH": re.compile(r"(?i)(gate[_-]?mismatch|chain_state.*(NULL|mismatch|inconsistent)|iter1_pending.*expected)"),
    "MODEL_404": re.compile(r"(?i)\b(model not found|404 .*model|KeyError.*response|model .*not pulled|unknown model)\b"),
    "CP1252_ERROR": re.compile(r"(?i)(UnicodeEncodeError|UnicodeDecodeError).*(cp1252|charmap)|('charmap' codec can't encode)"),
    "COLD_START_CRASH": re.compile(r"(?i)(-2147020576|cold[_-]?start|pre[_-]?import)"),
    "EDITOR_EMPTY_DIGEST": re.compile(r"(?i)(editor.*(empty|returned empty)|digest.*(skipped|empty markdown)|empty digest)"),
}

# Maximum sample line length in the report to keep it readable.
SAMPLE_TRUNCATE = 200
# Maximum samples per bucket in the report.
MAX_SAMPLES_PER_BUCKET = 3


def now_karaganda() -> datetime:
    """Return current time in Asia/Karaganda fixed offset."""
    return datetime.now(KARAGANDA_TZ)


def classify(line: str) -> str:
    """Return bucket name for a log line, or UNKNOWN if no pattern matches."""
    for bucket, pattern in FAILURE_PATTERNS.items():
        if pattern.search(line):
            return bucket
    return "UNKNOWN"


def _truncate(line: str, n: int = SAMPLE_TRUNCATE) -> str:
    line = line.strip()
    if len(line) <= n:
        return line
    return line[:n] + "…"


def parse_react_jsonl(path: Path, since: datetime | None = None) -> Iterable[tuple[datetime, str]]:
    """
    Yield (timestamp, message) for each error/stderr-tagged line in react_worker.jsonl.

    Lines without parseable timestamps are skipped. Lines older than `since` are skipped.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            err = rec.get("error") or rec.get("stderr") or rec.get("traceback")
            if not err:
                continue
            ts_str = rec.get("ts") or rec.get("timestamp") or rec.get("started_at")
            ts = _parse_iso(ts_str)
            if ts is None:
                continue
            if since is not None and ts < since:
                continue
            yield ts, str(err)


_PIPELINE_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z))\s*\|\s*(?P<kind>OK|FAIL|WARN)\s*\|\s*(?P<msg>.*)$"
)


def parse_pipeline_log(path: Path, since: datetime | None = None) -> Iterable[tuple[datetime, str]]:
    """
    Yield (timestamp, message) for FAIL lines from pipeline.log.

    Format: `<iso-ts> | FAIL | error=...`. WARN lines also included since they
    typically surface non-fatal observability concerns.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            m = _PIPELINE_LINE_RE.match(raw)
            if not m:
                continue
            if m.group("kind") not in ("FAIL", "WARN"):
                continue
            ts = _parse_iso(m.group("ts"))
            if ts is None:
                continue
            if since is not None and ts < since:
                continue
            yield ts, m.group("msg")


def parse_finish_worker_stderr(path: Path, since: datetime | None = None) -> Iterable[tuple[datetime, str]]:
    """
    Yield (timestamp, message) for finish_worker stderr capture.

    If file does not exist, yields nothing (fail-soft — stderr capture is best-effort).
    Format expected: `<iso-ts> <message>` per line. Lines without parseable timestamp
    are still yielded with file-mtime as fallback timestamp.
    """
    if not path.exists():
        return
    fallback_ts = datetime.fromtimestamp(path.stat().st_mtime, KARAGANDA_TZ)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.rstrip()
            if not raw:
                continue
            ts = None
            # Try to parse an ISO timestamp at the start.
            head = raw.split(None, 1)
            if head:
                ts = _parse_iso(head[0])
            if ts is None:
                ts = fallback_ts
            if since is not None and ts < since:
                continue
            yield ts, raw


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        # datetime.fromisoformat handles "Z" only in 3.11+; handle both branches.
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KARAGANDA_TZ)
        return dt
    except (ValueError, TypeError):
        return None


def aggregate(events: Iterable[tuple[datetime, str]]) -> dict[str, dict]:
    """
    Bucket events by failure pattern. Returns:
        {bucket: {"count": N, "samples": [(ts, line), ...]}}

    Samples are most-recent first, capped at MAX_SAMPLES_PER_BUCKET.
    """
    buckets: dict[str, dict] = {b: {"count": 0, "samples": []} for b in (*FAILURE_PATTERNS.keys(), "UNKNOWN")}
    for ts, line in events:
        bucket = classify(line)
        buckets[bucket]["count"] += 1
        buckets[bucket]["samples"].append((ts, line))
    # Sort samples by timestamp descending; cap.
    for b in buckets.values():
        b["samples"].sort(key=lambda t: t[0], reverse=True)
        b["samples"] = b["samples"][:MAX_SAMPLES_PER_BUCKET]
    return buckets


def render_report(
    current: dict[str, dict],
    prior: dict[str, dict],
    window_start: datetime,
    window_end: datetime,
    scan_ts: datetime,
) -> str:
    """
    Render markdown report comparing current 7d window against prior 7d window.

    Total failure count + per-bucket section with count, prior count, delta,
    and up to 3 most-recent sample lines (truncated).
    """
    total = sum(b["count"] for b in current.values())
    prior_total = sum(b["count"] for b in prior.values())
    delta = total - prior_total

    lines: list[str] = []
    lines.append(f"# Agent Failures — {window_start:%Y-%m-%d} → {window_end:%Y-%m-%d}")
    lines.append("")
    lines.append(f"**Scan timestamp:** {scan_ts.isoformat()}")
    lines.append(f"**Window:** 7 days (current) vs prior 7 days")
    lines.append(f"**Total failures (current):** {total}")
    lines.append(f"**Total failures (prior):** {prior_total}")
    delta_sign = "+" if delta > 0 else ""
    lines.append(f"**Delta (14d rolling):** {delta_sign}{delta}")
    lines.append("")
    lines.append("## Buckets")
    lines.append("")
    lines.append("| Bucket | Current 7d | Prior 7d | Delta |")
    lines.append("|-|-|-|-|")
    bucket_order = list(FAILURE_PATTERNS.keys()) + ["UNKNOWN"]
    for bucket in bucket_order:
        cur = current[bucket]["count"]
        prv = prior[bucket]["count"]
        d = cur - prv
        d_sign = "+" if d > 0 else ""
        lines.append(f"| {bucket} | {cur} | {prv} | {d_sign}{d} |")
    lines.append("")

    # Per-bucket sample sections (only buckets with count > 0).
    for bucket in bucket_order:
        cur = current[bucket]
        if cur["count"] == 0:
            continue
        lines.append(f"### {bucket}")
        lines.append("")
        lines.append(f"Count: {cur['count']} (current 7d)")
        lines.append("")
        lines.append("Recent samples:")
        for ts, sample in cur["samples"]:
            lines.append(f"- `{ts.isoformat()}` — {_truncate(sample)}")
        lines.append("")

    return "\n".join(lines) + "\n"


def run_miner(
    project_root: Path,
    out_path: Path | None = None,
    scan_ts: datetime | None = None,
) -> Path:
    """
    Run the miner end-to-end. Scans logs, aggregates two 7d windows, writes report.

    Returns the path of the written report.
    """
    if scan_ts is None:
        scan_ts = now_karaganda()
    if out_path is None:
        out_path = project_root / "logs" / "agent_failures.md"

    window_current_start = scan_ts - timedelta(days=7)
    window_prior_start = scan_ts - timedelta(days=14)

    react_jsonl = project_root / "logs" / "react_worker.jsonl"
    pipeline_log = project_root / "logs" / "pipeline.log"
    finish_stderr = project_root / "logs" / "finish_worker.stderr"

    def gather(since: datetime) -> list[tuple[datetime, str]]:
        events: list[tuple[datetime, str]] = []
        events.extend(parse_react_jsonl(react_jsonl, since=since))
        events.extend(parse_pipeline_log(pipeline_log, since=since))
        events.extend(parse_finish_worker_stderr(finish_stderr, since=since))
        return events

    all_events = gather(window_prior_start)
    current_events = [(t, l) for t, l in all_events if t >= window_current_start]
    prior_events = [(t, l) for t, l in all_events if window_prior_start <= t < window_current_start]

    current_buckets = aggregate(current_events)
    prior_buckets = aggregate(prior_events)

    report = render_report(current_buckets, prior_buckets, window_current_start, scan_ts, scan_ts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return out_path
