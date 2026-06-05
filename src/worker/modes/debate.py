"""Debate mode handler (Phase 73 / DEBATE-REACT-01).

Mirror of ``link.py`` shape. Writes an adversarial counter-position draft
to ``<vault>/10_Drafts/[_PRELIMINARY_]<rid>-debate.md``. The body contains:
  - prepended digest entry block (``## Из дайджеста``)
  - Claude-generated ``## Counter-position`` + ``## Preconditions``

Differences from link.py:
  - No anchor-project resolution (debate is project-agnostic). ``linked_project``
    flows into the prompt only as informational context; not into frontmatter
    as ``linked_project_status``.
  - Frontmatter carries ``mode: debate`` (downstream identification).
  - Filename suffix ``-debate.md``.

Atomic write: ``.tmp`` sibling + ``os.replace`` (same shape as brainstorm/docs/link).
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
from src.worker.prompts.debate_prompt import build_debate_prompt

_DEFAULT_MODEL: Final[str] = "claude-sonnet-4-6"
_DEFAULT_TIMEOUT: Final[int] = 120


def handle(
    reaction_row: dict[str, Any],
    digest_item: dict[str, Any],
    vault_drafts_dir: str,
) -> str:
    """Write a debate-mode draft for ``reaction_row['id']``.

    Inputs:
      reaction_row: claim_pending_reaction row. Required: ``id``, ``item_id``,
        ``action``.
      digest_item: digest_items row. Used keys: ``item_id``, ``channel``,
        ``url``, ``snippet``, ``linked_project``, ``entry_md``.
      vault_drafts_dir: absolute path to ``<vault>/10_Drafts/``.

    Side effects:
      - Calls ``run_powershell`` with the built prompt (exceptions propagate;
        the dispatcher in react_worker catches them — NFR-03 isolation).
      - Writes a UTF-8 NO-BOM, LF-newline file at
        ``<vault_drafts_dir>/[_PRELIMINARY_]<reaction_id>-debate.md``.

    Returns: absolute path of the written file as a string.
    """
    linked = digest_item.get("linked_project")

    prompt = build_debate_prompt(
        snippet=digest_item.get("snippet"),
        channel=digest_item.get("channel"),
        url=digest_item.get("url"),
        linked_project=linked,
    )

    body = run_powershell(prompt, model=_DEFAULT_MODEL, timeout=_DEFAULT_TIMEOUT)
    body = prepend_digest_entry(body, digest_item.get("entry_md"))

    _snippet = digest_item.get("snippet") or ""
    meta: dict[str, Any] = {
        "source": "digest-reaction",
        "reaction_id": reaction_row["id"],
        "item_id": reaction_row["item_id"],
        "action": "debate",
        "mode": "debate",
        "gate_state": gate_state_value("debate"),
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "source_url": digest_item.get("url"),
        "source_name": digest_item.get("channel"),
        "source_excerpt": _snippet[:200],
    }

    post = frontmatter.Post(content=body, **meta)
    rendered = frontmatter.dumps(post)

    filename = apply_preliminary_label(
        f"{reaction_row['id']}-debate.md", "debate"
    )
    target = Path(vault_drafts_dir) / filename
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
                pass
        os.replace(str(tmp), str(target))
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise

    return str(target.resolve())
