"""Brainstorm mode handler (WORKER-03).

Pure function of (reaction_row, digest_item, vault_drafts_dir). Caller is
responsible for vault preflight + DB status updates + log events. This
module ONLY: builds the prompt, invokes ``claude -p`` via ``run_powershell``,
writes the draft file, returns its absolute path.

Exceptions from ``run_powershell`` (``ClaudeSubprocessError`` /
``ClaudeSubprocessTimeout``) are NOT caught here — caller (``run_tick`` in
plan 17-06) catches and marks the reaction failed.

WR-04: Drafts are written atomically via a ``.tmp`` sibling + ``os.replace``.
A crash / disk-full / kill mid-write leaves either (a) the previous draft
intact (if any) or (b) no draft, never a half-written one. The post-condition
is: if ``handle()`` returns successfully, the path on disk is complete; the
caller is then safe to call ``_mark_drafted``.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any, Final

import frontmatter

from src.worker.claude_subprocess import run_powershell
from src.worker.digest_entry import prepend_digest_entry
from src.worker.gate_state import apply_preliminary_label, gate_state_value
from src.worker.prompts.brainstorm_prompt import build_brainstorm_prompt

_DEFAULT_MODEL: Final[str] = "claude-sonnet-4-6"
_DEFAULT_TIMEOUT: Final[int] = 120
_DEFAULT_VAULT_PATH: Final[str] = "vault"  # Generic fallback; override via OBSIDIAN_VAULT_PATH
_PORTFOLIO_REL_PATH: Final[str] = "90_Meta/project-graph.md"

# Brainstorm-review pipeline Stage 4: pattern feedback file. Maintained
# (manually for now, may auto-update from user decisions in future)
# alongside project-graph.md. Lists suppression rules + soft de-priorities
# distilled from the 2026-05-18 review of 630 brainstormed ideas.
_NEGATIVE_TARGETS_REL_PATH: Final[str] = "90_Meta/brainstorm_negative_targets.md"


def _vault_root() -> Path:
    return Path(
        os.environ.get("DRAFT_OUTPUT_DIR")
        or os.environ.get("OBSIDIAN_VAULT_PATH", _DEFAULT_VAULT_PATH)
    )


def _load_portfolio_md() -> str | None:
    """Read ``<vault>/90_Meta/project-graph.md`` fail-soft.

    Returns the raw text on success, ``None`` if the vault / file is missing
    or unreadable. Never raises — a vault outage must not block brainstorm.
    """
    path = _vault_root() / _PORTFOLIO_REL_PATH
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _load_negative_targets_md() -> str | None:
    """Read ``<vault>/90_Meta/brainstorm_negative_targets.md`` fail-soft.

    Returns the raw text on success, ``None`` if the file is missing —
    in which case the brainstorm prompt simply omits the negative-target
    block. A vault outage must not block brainstorm.
    """
    path = _vault_root() / _NEGATIVE_TARGETS_REL_PATH
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def handle(
    reaction_row: dict[str, Any],
    digest_item: dict[str, Any],
    vault_drafts_dir: str,
) -> str:
    """Write a brainstorm draft for ``reaction_row['id']``.

    Inputs:
      reaction_row: dict from ``claim_pending_reaction`` (Plan 17-02). Required
        keys: ``id``, ``item_id``, ``action``.
      digest_item: dict shaped like a row from ``digest_items``. Required keys:
        ``item_id``, ``channel`` (nullable), ``url`` (nullable), ``snippet``
        (nullable), ``linked_project`` (nullable).
      vault_drafts_dir: absolute path to ``<vault>/10_Drafts/``. Caller (the
        worker tick in plan 17-06) is responsible for vault preflight.

    Side effects:
      - Calls ``run_powershell`` with the built prompt. Exceptions
        (``ClaudeSubprocessError`` / ``ClaudeSubprocessTimeout``) propagate.
      - Writes a UTF-8 NO-BOM, LF-newline file at
        ``<vault_drafts_dir>/<reaction_id>-brainstorm.md``.

    Returns: absolute path of the written file as a string.
    """
    portfolio_md = _load_portfolio_md()
    negative_targets_md = _load_negative_targets_md()
    prompt = build_brainstorm_prompt(
        snippet=digest_item.get("snippet"),
        channel=digest_item.get("channel"),
        url=digest_item.get("url"),
        linked_project=digest_item.get("linked_project"),
        portfolio_md=portfolio_md,
        negative_targets_md=negative_targets_md,
    )

    # cwd=vault_root: prevents headless Claude from picking up the autorss_feed
    # CLAUDE.md (worker process cwd) and collapsing every brainstorm into
    # autorss-feed-shaped ideas (2026-05-17 bug). Vault root has no CLAUDE.md
    # → Claude relies on the inlined portfolio block in the prompt.
    vault_cwd = str(_vault_root()) if _vault_root().exists() else None

    # May raise ClaudeSubprocessError / ClaudeSubprocessTimeout — propagate.
    body = run_powershell(
        prompt,
        model=_DEFAULT_MODEL,
        timeout=_DEFAULT_TIMEOUT,
        cwd=vault_cwd,
    )
    body = prepend_digest_entry(body, digest_item.get("entry_md"))

    linked = digest_item.get("linked_project")
    _snippet = digest_item.get("snippet") or ""
    meta: dict[str, Any] = {
        "source": "digest-reaction",
        "reaction_id": reaction_row["id"],
        "item_id": reaction_row["item_id"],
        "action": "brainstorm",
        "linked_project": linked,
        "linked_project_status": "resolved" if linked else "unresolved",
        "gate_state": gate_state_value("brainstorm"),
        # IN-04: standardise on datetime.UTC (Python 3.11+) — matches react_worker.py.
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        # Source context: lets operator identify the post without DB JOIN.
        "source_url": digest_item.get("url"),
        "source_name": digest_item.get("channel"),
        "source_excerpt": _snippet[:200],
    }

    post = frontmatter.Post(content=body, **meta)
    rendered = frontmatter.dumps(post)

    # WORKER-10: PRELIMINARY label is applied here, the single choke point.
    filename = apply_preliminary_label(
        f"{reaction_row['id']}-brainstorm.md", "brainstorm"
    )
    target = Path(vault_drafts_dir) / filename
    # WR-04: write to <target>.tmp first, fsync, then os.replace. os.replace is
    # atomic on Windows + POSIX. If the process is killed or the disk fills
    # mid-write, the .tmp is incomplete but the existing target (if any) is
    # untouched; we also clean up the orphaned .tmp on exception.
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(rendered)
            if not rendered.endswith("\n"):
                f.write("\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                # fsync may fail on some FS / on win32 console handles —
                # data is still in the kernel buffer; replace below is the
                # atomicity primitive. Don't promote to error.
                pass
        os.replace(str(tmp), str(target))
    except Exception:
        # Cleanup: remove orphaned .tmp so retries don't see stale files.
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise

    return str(target.resolve())
