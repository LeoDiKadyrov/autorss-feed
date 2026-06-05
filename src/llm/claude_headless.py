import asyncio
import contextvars
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

# Phase 40: contextvar holding the active run_id + agent for tracing. Editor /
# curator / finish_worker set these via `trace_run()` context manager so the
# trace records get a shared run_id without threading kwargs through every
# call site.
_trace_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_trace_run_id", default=None
)
_trace_agent: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_trace_agent", default=None
)


class trace_run:
    """Context manager to bind a run_id + agent for the surrounding async block."""

    def __init__(self, agent: str, run_id: Optional[str] = None):
        from src.observability.traces import start_run
        self.agent = agent
        self.run_id = run_id or start_run(prefix=agent)
        self._t_id = None
        self._t_ag = None

    def __enter__(self) -> str:
        self._t_id = _trace_run_id.set(self.run_id)
        self._t_ag = _trace_agent.set(self.agent)
        return self.run_id

    def __exit__(self, exc_type, exc, tb) -> None:
        _trace_run_id.reset(self._t_id)
        _trace_agent.reset(self._t_ag)


async def generate_with_claude(prompt: str, timeout: int = 300) -> str:
    """Call claude CLI via stdin to avoid Windows 32k command-line length limit."""
    loop = asyncio.get_event_loop()

    def _run():
        return subprocess.run(
            ["claude", "-p", "--output-format", "text"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )

    run_id = _trace_run_id.get()
    agent = _trace_agent.get()
    record_trace = run_id is not None and agent is not None

    if record_trace:
        from src.observability.traces import record_step, now_karaganda
        started_at = now_karaganda().isoformat()
        db_path = os.environ.get("AUTORSS_DB_PATH", "curator.db")
    t0 = time.monotonic()

    try:
        result = await loop.run_in_executor(None, _run)
    except Exception as exc:
        if record_trace:
            duration_ms = int((time.monotonic() - t0) * 1000)
            record_step(
                db_path, run_id=run_id, agent=agent, tool="claude_cli",
                args={"prompt_len": len(prompt), "timeout": timeout},
                result={"exit": "error", "error": str(exc)[:500]},
                duration_ms=duration_ms, started_at=started_at,
            )
        raise

    duration_ms = int((time.monotonic() - t0) * 1000)
    if result.returncode != 0:
        # Claude CLI may emit JSON error envelopes to stdout (rate-limit, quota,
        # auth) while stderr stays empty — see reference_claude_cli_subprocess_exit1.
        # Capture both so the next outage is diagnosable in seconds, not hours.
        stderr_tail = (result.stderr or "")[:500]
        stdout_tail = (result.stdout or "")[:500]
        if record_trace:
            record_step(
                db_path, run_id=run_id, agent=agent, tool="claude_cli",
                args={"prompt_len": len(prompt), "timeout": timeout},
                result={
                    "exit": result.returncode,
                    "stderr": stderr_tail,
                    "stdout": stdout_tail,
                },
                duration_ms=duration_ms, started_at=started_at,
            )
        raise RuntimeError(
            f"claude exited {result.returncode}: "
            f"stderr={stderr_tail!r} stdout={stdout_tail!r}"
        )
    out = result.stdout.strip()
    if record_trace:
        record_step(
            db_path, run_id=run_id, agent=agent, tool="claude_cli",
            args={"prompt_len": len(prompt), "timeout": timeout},
            result={"exit": 0, "stdout_len": len(out), "sample": out[:300]},
            duration_ms=duration_ms, started_at=started_at,
        )
    return out


