"""Link mode handler (WORKER-05). Cross-refs in body; no extra frontmatter.

Mirror of ``docs.py`` and ``brainstorm.py``, with one CONTEXT-locked
difference: link-mode drafts have NO extra frontmatter keys beyond the 8
base ones (no ``target_path``). The cross-references live entirely in the
body — the user reviews them via /post-draft (Phase 19) and decides
continue/defer/kill.

Sentinel handling (WR-04 doc fix):
  The prompt instructs claude to produce either ≥3 cross-refs OR the exact
  line ``"no relevant cross-refs found"`` (the SENTINEL constant in
  ``link_prompt``). This handler does NOT inspect the body — sentinel
  detection and ref-count enforcement are deliberately deferred to
  /post-draft (Phase 19) so that drafts are persisted as written and the
  user reviews them in one place. Sentinel-only outputs are a documented
  graceful path, not an error. If a future requirement needs the worker to
  branch on sentinel (e.g. for skipping vault writes), add the check in
  this module and a corresponding ``link_sentinel`` frontmatter flag — for
  Phase 18 the handler is intentionally body-agnostic.

WR-04 (other): Drafts are written atomically via ``.tmp`` sibling + ``os.replace``.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any, Final

import frontmatter

from src.worker.digest_entry import prepend_digest_entry

from src.worker.claude_subprocess import run_powershell
from src.worker.gate_state import apply_preliminary_label, gate_state_value
from src.worker.linked_project import resolve
from src.worker.prompts.link_prompt import build_link_prompt

_DEFAULT_MODEL: Final[str] = "claude-sonnet-4-6"
_DEFAULT_TIMEOUT: Final[int] = 120


def handle(
    reaction_row: dict[str, Any],
    digest_item: dict[str, Any],
    vault_drafts_dir: str,
) -> str:
    """Write a link-mode draft for ``reaction_row['id']``.

    Inputs:
      reaction_row: dict from ``claim_pending_reaction``. Required keys:
        ``id``, ``item_id``, ``action``.
      digest_item: dict shaped like a row from ``digest_items``. Required keys:
        ``item_id``, ``channel`` (nullable), ``url`` (nullable), ``snippet``
        (nullable), ``linked_project`` (nullable).
      vault_drafts_dir: absolute path to ``<vault>/10_Drafts/``.

    Side effects:
      - Calls ``run_powershell`` with the built prompt (exceptions propagate).
      - Writes a UTF-8 NO-BOM, LF-newline file at
        ``<vault_drafts_dir>/[_PRELIMINARY_]<reaction_id>-link.md``.

    Returns: absolute path of the written file as a string.

    Notes:
      - The body is written verbatim regardless of structure. We DO NOT parse
        or validate ref count. The sentinel ``"no relevant cross-refs found"``
        is the documented graceful path; <3 refs without sentinel is a
        downstream review signal, not an exception here.
    """
    linked = digest_item.get("linked_project")
    linked_path, status = resolve(linked)

    prompt = build_link_prompt(
        snippet=digest_item.get("snippet"),
        channel=digest_item.get("channel"),
        url=digest_item.get("url"),
        linked_project=linked,
        linked_project_path=linked_path,
    )

    # May raise ClaudeSubprocessError / ClaudeSubprocessTimeout — propagate.
    body = run_powershell(prompt, model=_DEFAULT_MODEL, timeout=_DEFAULT_TIMEOUT)
    body = prepend_digest_entry(body, digest_item.get("entry_md"))

    _snippet = digest_item.get("snippet") or ""
    meta: dict[str, Any] = {
        "source": "digest-reaction",
        "reaction_id": reaction_row["id"],
        "item_id": reaction_row["item_id"],
        "action": "link",
        "linked_project": linked,
        "linked_project_status": status,
        "gate_state": gate_state_value("link"),
        # IN-04: standardise on datetime.UTC (Python 3.11+) — matches react_worker.py.
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        # Source context: lets operator identify the post without DB JOIN.
        "source_url": digest_item.get("url"),
        "source_name": digest_item.get("channel"),
        "source_excerpt": _snippet[:200],
    }

    post = frontmatter.Post(content=body, **meta)
    rendered = frontmatter.dumps(post)

    # WORKER-10: PRELIMINARY label applied at the single choke point.
    filename = apply_preliminary_label(
        f"{reaction_row['id']}-link.md", "link"
    )
    target = Path(vault_drafts_dir) / filename
    # WR-04: write to <target>.tmp first, fsync, then os.replace. Same shape
    # as brainstorm.py / docs.py — atomic on Windows + POSIX.
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
