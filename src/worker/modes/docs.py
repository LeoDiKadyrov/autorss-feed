"""Docs mode handler (WORKER-04).

Mirror of ``brainstorm.py`` with one extra frontmatter key (``target_path``)
and integration with the linked-project resolver from Plan 18-01. When the
user clicks 📄 docs on a digest item with ``Связь: <project>``, this handler
writes a *merge-ready* draft destined for ``40_Projects/<linked>/<slug>.md``.

Slug parsing reads the ``## Suggested filename`` section from the LLM body.
The slug regex is conservative (``[\\w\\-./]+\\.md``) and the parsed slug is
sanitised (T-18-08): the slug is collapsed to its basename FIRST, then must
end in ``.md`` and contain no ``..`` literal — see ``_parse_target_path``.

target_path semantic (IN-05 — deliberate divergence from brainstorm/link):
  Docs ALWAYS writes the ``target_path`` key into the frontmatter. The value
  is either the resolved vault-relative path (``40_Projects/<linked>/<slug>.md``)
  or ``None`` when the linked project is unresolved or the slug failed to parse.
  Brainstorm and link OMIT the key entirely. Phase 19 ``/post-draft`` consumers
  can therefore treat key presence as a docs-mode marker; ``meta.get("target_path")``
  yields ``None`` for both ``key absent`` and ``key=None``, but a strict-schema
  validator can distinguish the two and surface unresolved-docs drafts for
  manual placement.

Pure function of (reaction_row, digest_item, vault_drafts_dir). Caller is
responsible for vault preflight + DB status updates + log events. This
module ONLY: builds the prompt, invokes ``claude -p`` via ``run_powershell``,
parses the slug, writes the draft file, returns its absolute path.

WR-04: Drafts are written atomically via ``.tmp`` sibling + ``os.replace``.
"""
from __future__ import annotations

import datetime
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Final

import frontmatter

from src.worker.digest_entry import prepend_digest_entry

from src.worker.claude_subprocess import run_powershell
from src.worker.gate_state import apply_preliminary_label, gate_state_value
from src.worker.linked_project import resolve, resolve_rel
from src.worker.prompts.docs_prompt import build_docs_prompt

_DEFAULT_MODEL: Final[str] = "claude-sonnet-4-6"
_DEFAULT_TIMEOUT: Final[int] = 120

# Match the ``## Suggested filename`` section: header line, optional blank
# line(s), then a slug ending in ``.md`` optionally wrapped in backticks.
# Character class restricts to word chars, dot, hyphen, slash — slashes are
# stripped by ``_parse_target_path`` so the final slug is filename-only.
_SLUG_RE = re.compile(
    r"##\s*Suggested filename\s*\n+\s*`?([\w\-./]+\.md)`?",
    flags=re.IGNORECASE,
)


def _parse_target_path(body: str, linked_rel: str | None) -> str | None:
    """Parse the suggested filename out of the LLM body and join with linked_rel.

    ``linked_rel`` is the VAULT-RELATIVE path (e.g. ``'40_Projects/Ковчег'``),
    not the absolute filesystem path. Returning relative paths keeps the
    ``target_path`` frontmatter value separator-consistent across platforms
    (WR-02: Windows backslashes from the draft output directory prefix would
    otherwise mix with the forward-slash slug join).

    Returns None when:
      - ``linked_rel`` is None (no resolved project to anchor the slug to)
      - the body contains no parseable ``## Suggested filename`` section
      - the parsed slug is empty after sanitisation
      - the parsed slug does not end in ``.md`` after sanitisation
      - the parsed slug still contains ``..`` after basename collapse

    Defense-in-depth (T-18-08, WR-03): the slug is collapsed to its basename
    FIRST (handling both ``/`` and ``\\`` separators), THEN validated. This
    closes the backslash-traversal gap the previous iterative ``..`` strip
    relied on the trailing basename rule to neutralise.
    """
    if not linked_rel:
        return None
    m = _SLUG_RE.search(body)
    if not m:
        return None
    raw_slug = m.group(1).strip()
    # WR-03: collapse to basename FIRST. Normalise backslashes to forward
    # slashes, then take the trailing path segment. After this step the slug
    # is guaranteed to contain no separator — any ``..`` left is a literal
    # filename component and rejected below.
    slug = Path(raw_slug.replace("\\", "/")).name
    if not slug or not slug.endswith(".md") or ".." in slug:
        return None
    # WR-02: build the relative target_path with PurePosixPath so the
    # separator stays ``/`` regardless of host OS.
    return str(PurePosixPath(linked_rel) / slug)


def handle(
    reaction_row: dict[str, Any],
    digest_item: dict[str, Any],
    vault_drafts_dir: str,
) -> str:
    """Write a docs draft for ``reaction_row['id']``.

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
        ``<vault_drafts_dir>/[_PRELIMINARY_]<reaction_id>-docs.md``.

    Returns: absolute path of the written file as a string.
    """
    linked = digest_item.get("linked_project")
    # ``linked_path``: absolute filesystem path used in the prompt (gives claude
    # a concrete pointer for grounding). ``linked_rel``: vault-relative path
    # used in frontmatter (WR-02 — separator-consistent across platforms).
    linked_path, status = resolve(linked)
    linked_rel, _ = resolve_rel(linked)

    prompt = build_docs_prompt(
        snippet=digest_item.get("snippet"),
        channel=digest_item.get("channel"),
        url=digest_item.get("url"),
        linked_project=linked,
        linked_project_path=linked_path,
    )

    # May raise ClaudeSubprocessError / ClaudeSubprocessTimeout — propagate.
    body = run_powershell(prompt, model=_DEFAULT_MODEL, timeout=_DEFAULT_TIMEOUT)

    # target_path is parsed BEFORE prepending the digest entry block so the
    # parser sees the raw model output (line layout matters for the regex).
    target_path = _parse_target_path(body, linked_rel)
    body = prepend_digest_entry(body, digest_item.get("entry_md"))

    _snippet = digest_item.get("snippet") or ""
    meta: dict[str, Any] = {
        "source": "digest-reaction",
        "reaction_id": reaction_row["id"],
        "item_id": reaction_row["item_id"],
        "action": "docs",
        "linked_project": linked,
        "linked_project_status": status,
        "gate_state": gate_state_value("docs"),
        # IN-04: standardise on datetime.UTC (Python 3.11+) — matches react_worker.py.
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "target_path": target_path,
        # Source context: lets operator identify the post without DB JOIN.
        "source_url": digest_item.get("url"),
        "source_name": digest_item.get("channel"),
        "source_excerpt": _snippet[:200],
    }

    post = frontmatter.Post(content=body, **meta)
    rendered = frontmatter.dumps(post)

    # WORKER-10: PRELIMINARY label applied at the single choke point.
    filename = apply_preliminary_label(
        f"{reaction_row['id']}-docs.md", "docs"
    )
    target = Path(vault_drafts_dir) / filename
    # WR-04: write to <target>.tmp first, fsync, then os.replace. Same shape
    # as brainstorm.py — atomic on Windows + POSIX.
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
