import logging
import os
import subprocess
from datetime import date
from pathlib import Path

from src.observability.traces import record_step, start_run
from src.worker.draft_manager import _drafts_dir, read_draft, set_status
from src.worker.token_counter import estimate_tokens

logger = logging.getLogger(__name__)

# Phase 93-01 — chunk re-classification gate constants.
# Approximate chars-per-token; matches src/worker/token_counter.py baseline
# and src/llm/routing.py::_CHARS_PER_TOKEN. Used only to slice the tail of
# the transcript before the second classify call.
_CHARS_PER_TOKEN = 4
# Last N tokens to send to the tail re-classify (CONTEXT: "re-run precision
# gate on last 10K tokens only").
_TAIL_TOKEN_BUDGET = 10_000
# One score band on the 0-100 scale.
_DISAGREEMENT_BAND = 10


def _run_async_safe(coro):
    """Run a coroutine tolerating an already-running loop (WR-01 fix Phase 93).

    Pattern from scripts/position_bias_probe.py: spawn a worker thread with
    its own loop when called from a running loop, else asyncio.run directly.
    Without this, `asyncio.run()` raises RuntimeError when finish_worker is
    invoked from an async caller — silently disabling the gate.
    """
    import asyncio
    import threading
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_box: dict = {}

    def _worker():
        try:
            result_box["value"] = asyncio.run(coro)
        except BaseException as exc:
            result_box["error"] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()
    if "error" in result_box:
        raise result_box["error"]
    return result_box.get("value")


def _reclassify_score(text: str, backend: str) -> int | None:
    """Phase 93-01 — dispatch a single classify call to ``backend``.

    Returns the 0-100 relevance score or ``None`` on any failure (callers
    fail-soft). Routing follows the existing curator dispatch:
      - ``ollama`` (default fallback)
      - ``claude`` / ``claude-batch``
    On any exception (network, missing scorer, parse failure), returns None.

    Kept thin so tests can monkeypatch this single seam.

    WR-01 fix (Phase 93): uses _run_async_safe so async callers don't break
    the gate via `asyncio.run() cannot be called from a running event loop`.
    """
    try:
        # Late import — avoids circular dep with src.llm.* at module load.
        if backend == "claude":
            from src.llm.claude_scorer import score_content as _claude_score
            res = _run_async_safe(_claude_score("", text))
            return int(res.get("score", 0)) if isinstance(res, dict) else None
        # Default + claude-batch fallback to ollama (single-post path).
        from src.llm.ollama import score_content as _ollama_score
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        model = os.environ.get("CURATOR_MODEL", "qwen2.5:7b")
        res = _run_async_safe(_ollama_score(host, model, "", text))
        return int(res.get("score", 0)) if isinstance(res, dict) else None
    except Exception as exc:  # noqa: BLE001 — fail-soft seam
        logger.warning("chunk_regate: _reclassify_score failed: %s", exc)
        return None


def maybe_chunk_regate(
    *,
    reaction_id: int,
    transcript: str,
    source_backend: str,
    db_path: str,
) -> bool:
    """Phase 93-01 — chunk re-classification gate.

    Returns True if promotion should proceed, False if blocked (caller must
    NOT promote when False is returned).

    Behavior (CONTEXT 93-CONTEXT.md):
      - If CHUNK_REGATE_MODE=="off" or tokens<=CHUNK_REGATE_TOKEN_THRESHOLD
        (default 50000) -> zero-overhead fast path, no record_step, no status
        change, returns True.
      - Otherwise: compute full + tail scores using the SAME backend that
        originally scored the source post. Tail = last 10K tokens worth of
        chars. abs(full-tail) > 10 -> set_status('needs_review') and record
        agent_traces row with tool='chunk_regate' result={'blocked': True},
        returns False. Within band -> record_step with blocked=False, returns
        True (telemetry counter).
      - Fail-soft: any exception in the regate path logs a warning and
        returns True (advisory gate per v1.3 pattern). NO trace row written
        when the scorer itself fails (matches test expectation case e).

    Env (D-03 — read INSIDE function, never at module level):
      CHUNK_REGATE_MODE         on|off, default 'on'
      CHUNK_REGATE_TOKEN_THRESHOLD  int, default 50000
    """
    # D-03: env reads inside function body.
    mode = os.environ.get("CHUNK_REGATE_MODE", "on").strip().lower()
    if mode == "off":
        return True
    try:
        threshold = int(os.environ.get("CHUNK_REGATE_TOKEN_THRESHOLD", "50000"))
    except ValueError:
        threshold = 50_000

    tokens = estimate_tokens(transcript)
    if tokens <= threshold:
        return True

    # Long-transcript path — same backend for both calls per CONTEXT.
    try:
        tail_chars = _TAIL_TOKEN_BUDGET * _CHARS_PER_TOKEN
        tail_text = transcript[-tail_chars:] if len(transcript) > tail_chars else transcript

        full_score = _reclassify_score(transcript, source_backend)
        tail_score = _reclassify_score(tail_text, source_backend)
        if full_score is None or tail_score is None:
            # Scorer failed — fail-soft, but WR-02 fix (Phase 93): write a
            # trace row so operators can distinguish "gate ran and passed"
            # from "gate silently bypassed because scorer is down". Combined
            # with WR-01, a persistent outage now leaves an audit trail.
            logger.warning(
                "chunk_regate: scorer returned None (full=%s tail=%s), proceeding",
                full_score, tail_score,
            )
            try:
                run_id = start_run("chunk_regate")
                record_step(
                    db_path,
                    run_id=run_id,
                    agent="finish_worker",
                    tool="chunk_regate",
                    args={
                        "reaction_id": reaction_id,
                        "backend": source_backend,
                        "token_count": tokens,
                        "threshold": threshold,
                        "full_score": full_score,
                        "tail_score": tail_score,
                    },
                    result={"blocked": False, "scorer_failed": True},
                )
            except Exception as _trace_exc:  # noqa: BLE001
                logger.warning(
                    "chunk_regate: trace row write failed: %s", _trace_exc
                )
            return True
        delta = abs(int(full_score) - int(tail_score))
        blocked = delta > _DISAGREEMENT_BAND
        run_id = start_run("chunk_regate")
        # T-93-04: args carry scores + counts only; never transcript text.
        record_step(
            db_path,
            run_id=run_id,
            agent="finish_worker",
            tool="chunk_regate",
            args={
                "reaction_id": reaction_id,
                "backend": source_backend,
                "token_count": tokens,
                "threshold": threshold,
                "full_score": int(full_score),
                "tail_score": int(tail_score),
                "delta": delta,
            },
            result={"blocked": blocked},
        )
        if blocked:
            try:
                set_status(reaction_id, "needs_review")
            except Exception as exc:  # noqa: BLE001
                logger.warning("chunk_regate: set_status failed: %s", exc)
            return False
        return True
    except Exception as exc:  # noqa: BLE001 — fail-soft outer guard
        logger.warning("chunk_regate: unexpected failure, proceeding: %s", exc)
        return True

def find_unfinished_finish_drafts() -> list[Path]:
    results = []
    for p in _drafts_dir().glob("_PRELIMINARY_*-finish-*.md"):
        try:
            fm, _ = read_draft(p)
            if fm.get("gate_state") == "archived":
                continue
            current = int(fm.get("current_iter", 0))
            total = int(fm.get("total_iters", 0))
            if total == 0:
                logger.warning("finish_worker: %s has total_iters=0, skipping", p.name)
                continue
            if current < total:
                results.append(p)
        except Exception as exc:
            logger.warning("finish_worker: error reading %s: %s", p.name, exc)
    return results


def _call_claude(prompt: str) -> str:
    timeout = int(os.environ.get("FINISH_WORKER_TIMEOUT", "180"))
    result = subprocess.run(
        ["claude", "--output-format", "text", "--print", prompt],
        capture_output=True,
        encoding="utf-8",
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Claude exited {result.returncode}: {result.stderr[:200]}")
    return result.stdout.strip()


def _build_prompt(path: Path, fm: dict, body: str) -> str:
    current = int(fm.get("current_iter", 1))
    total = int(fm.get("total_iters", 3))
    next_iter = current + 1
    today = date.today().isoformat()
    file_content = path.read_text(encoding="utf-8")
    return (
        f"You are advancing a finish-type Obsidian reaction draft from iter {current} to iter {next_iter} of {total}.\n\n"
        f"Current file content:\n\n{file_content}\n\n"
        f"Instructions:\n"
        f"1. Read the '## Immediate Next Step' section and execute it.\n"
        f"2. Update `current_iter` in the YAML frontmatter from {current} to {next_iter}.\n"
        f"3. Append a '## Iter {next_iter} Output ({today})' section at the end with the results.\n"
        f"4. Return ONLY the complete updated markdown file. No preamble, no explanation."
    )


def run_finish_iteration(path: Path) -> bool:
    try:
        fm, body = read_draft(path)
        prompt = _build_prompt(path, fm, body)
        updated = _call_claude(prompt)
        if not updated or "current_iter" not in updated:
            logger.warning("finish_worker: Claude output missing frontmatter for %s", path.name)
            return False
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(updated, encoding="utf-8")
            tmp.replace(path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        logger.info("finish_worker: advanced iter for %s", path.name)
        return True
    except Exception as e:
        logger.error("finish_worker: failed for %s: %s", path.name, e)
        return False


ACTIVE_STATES = {"research", "execute", "research_execute"}


def find_active_drafts() -> list[tuple[Path, str]]:
    """Find any _PRELIMINARY_*.md with gate_state in ACTIVE_STATES."""
    results = []
    for p in _drafts_dir().glob("_PRELIMINARY_*.md"):
        try:
            fm, _ = read_draft(p)
            state = fm.get("gate_state", "")
            if state in ACTIVE_STATES:
                results.append((p, state))
        except Exception as exc:
            logger.warning("finish_worker: error reading %s: %s", p.name, exc)
    return results


def _build_research_prompt(path: Path) -> str:
    today = date.today().isoformat()
    file_content = path.read_text(encoding="utf-8")
    return (
        f"You are doing research for an Obsidian knowledge draft.\n\n"
        f"Current draft:\n\n{file_content}\n\n"
        f"Task:\n"
        f"1. Research the core topic/idea — find key concepts, examples, counterarguments, sources.\n"
        f"2. Add a '## Research ({today})' section at the end with structured findings.\n"
        f"3. In the YAML frontmatter, change gate_state from 'research' to 'preliminary'.\n"
        f"4. Return ONLY the complete updated markdown file. No preamble."
    )


def _build_execute_prompt(path: Path) -> str:
    today = date.today().isoformat()
    file_content = path.read_text(encoding="utf-8")
    return (
        f"You are executing the next step for an Obsidian knowledge draft.\n\n"
        f"Current draft:\n\n{file_content}\n\n"
        f"Task:\n"
        f"1. Find '## Immediate Next Step' section, or infer next logical step from the content.\n"
        f"2. Execute it: implement, write, create, or produce whatever is described.\n"
        f"3. Add an '## Execution Output ({today})' section with full results.\n"
        f"4. In the YAML frontmatter, change gate_state from 'execute' to 'preliminary'.\n"
        f"5. Return ONLY the complete updated markdown file. No preamble."
    )


def _write_claude_output(path: Path, updated: str) -> None:
    """Atomic write of Claude output."""
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(updated, encoding="utf-8")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def run_research_iteration(path: Path) -> bool:
    try:
        prompt = _build_research_prompt(path)
        updated = _call_claude(prompt)
        if not updated or "gate_state" not in updated:
            logger.warning("finish_worker: bad research output for %s", path.name)
            return False
        _write_claude_output(path, updated)
        logger.info("finish_worker: research done for %s", path.name)
        return True
    except Exception as e:
        logger.error("finish_worker: research failed for %s: %s", path.name, e)
        return False


def run_execute_iteration(path: Path) -> bool:
    try:
        prompt = _build_execute_prompt(path)
        updated = _call_claude(prompt)
        if not updated or "gate_state" not in updated:
            logger.warning("finish_worker: bad execute output for %s", path.name)
            return False
        _write_claude_output(path, updated)
        logger.info("finish_worker: execute done for %s", path.name)
        return True
    except Exception as e:
        logger.error("finish_worker: execute failed for %s: %s", path.name, e)
        return False


def run_research_then_execute(path: Path) -> bool:
    """Two-pass: research first, then execute on the updated file."""
    today = date.today().isoformat()
    file_content = path.read_text(encoding="utf-8")
    # Pass 1: research (keep gate_state as research_execute for pass 2)
    research_prompt = (
        f"You are doing research for an Obsidian knowledge draft.\n\n"
        f"Current draft:\n\n{file_content}\n\n"
        f"Task:\n"
        f"1. Research the core topic/idea — find key concepts, examples, counterarguments, sources.\n"
        f"2. Add a '## Research ({today})' section at the end with structured findings.\n"
        f"3. Keep gate_state as 'research_execute' in the YAML frontmatter (do not change it).\n"
        f"4. Return ONLY the complete updated markdown file. No preamble."
    )
    try:
        updated = _call_claude(research_prompt)
        if not updated or "gate_state" not in updated:
            logger.warning("finish_worker: bad research pass for %s", path.name)
            return False
        _write_claude_output(path, updated)
    except Exception as e:
        logger.error("finish_worker: research pass failed for %s: %s", path.name, e)
        return False

    # Pass 2: execute on updated file
    execute_prompt = (
        f"You are executing the next step for an Obsidian knowledge draft.\n\n"
        f"Current draft:\n\n{path.read_text(encoding='utf-8')}\n\n"
        f"Task:\n"
        f"1. Find '## Immediate Next Step' section, or infer next logical step.\n"
        f"2. Execute it: implement, write, create, or produce whatever is described.\n"
        f"3. Add an '## Execution Output ({today})' section with full results.\n"
        f"4. In the YAML frontmatter, change gate_state from 'research_execute' to 'preliminary'.\n"
        f"5. Return ONLY the complete updated markdown file. No preamble."
    )
    try:
        updated2 = _call_claude(execute_prompt)
        if not updated2 or "gate_state" not in updated2:
            logger.warning("finish_worker: bad execute pass for %s", path.name)
            return False
        _write_claude_output(path, updated2)
        logger.info("finish_worker: research+execute done for %s", path.name)
        return True
    except Exception as e:
        logger.error("finish_worker: execute pass failed for %s: %s", path.name, e)
        return False


def run_active_drafts() -> dict:
    drafts = find_active_drafts()
    results = {"attempted": len(drafts), "succeeded": 0, "failed": 0}
    for p, state in drafts:
        if state == "research":
            ok = run_research_iteration(p)
        elif state == "execute":
            ok = run_execute_iteration(p)
        elif state == "research_execute":
            ok = run_research_then_execute(p)
        else:
            ok = False
        if ok:
            results["succeeded"] += 1
        else:
            results["failed"] += 1
    return results


def run_all_pending() -> dict:
    """Run both finish-iter drafts and gate_state-triggered active drafts."""
    # Pass 1: finish-type iter advancement
    finish_drafts = find_unfinished_finish_drafts()
    finish_results = {"attempted": len(finish_drafts), "succeeded": 0, "failed": 0}
    for p in finish_drafts:
        if run_finish_iteration(p):
            finish_results["succeeded"] += 1
        else:
            finish_results["failed"] += 1

    # Pass 2: research/execute/research_execute gate_state triggers
    active_results = run_active_drafts()

    return {
        "attempted": finish_results["attempted"] + active_results["attempted"],
        "succeeded": finish_results["succeeded"] + active_results["succeeded"],
        "failed": finish_results["failed"] + active_results["failed"],
    }
