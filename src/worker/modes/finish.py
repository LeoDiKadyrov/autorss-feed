"""Finish-chain mode handler (WORKER-06).

Dispatches on ``reaction_row['current_iter']`` to one of three iter prompts:
  - Iter 1: open-ended brainstorm (no prior input).
  - Iter 2: docs-style — input = previous iter's DRAFT FILE current contents.
  - Iter 3: deep-research — input = previous iter's DRAFT FILE current contents.

File-as-input rule (the core correctness invariant of WORKER-06):
  Iter<N+1> reads the live contents of Iter<N>'s draft file at handler-call
  time. The user may edit the file between iters; those edits propagate
  into the next iter's prompt. NO Python-side memoization of prev iter
  text is permitted — every call is a fresh ``Path.read_text``.

DB-side state transitions (chain_state UPDATE iter${N}_pending →
iter${N}_drafted) live in cross_db.try_mark_continued (plan 20-02);
this handler is a pure file-IO + claude subprocess function (Phase 17 contract).

Same atomic-write convention as brainstorm.py / docs.py (WR-04):
  ``.tmp`` sibling + ``os.replace``. Cleanup ``.tmp`` on exception so retries
  do not see stale half-written files.

Timeout bumped to 180s (vs 120s for brainstorm/docs) because Iter 3
deep-research with web tools rounds-trips slower than inline generation.
"""
from __future__ import annotations

import datetime
import os
import re
from pathlib import Path
from typing import Any, Final

import frontmatter

from src.worker.digest_entry import prepend_digest_entry

from src.worker.claude_subprocess import _CONTROL_TRANS, run_powershell
from src.worker.gate_state import apply_preliminary_label, gate_state_value
from src.worker.logging_jsonl import log_reaction_event
from src.worker.memory import recall_context
from src.worker.token_counter import estimate_tokens
from src.worker.prompts.finish_iter1_prompt import build_finish_iter1_prompt
from src.worker.prompts.finish_iter2_prompt import build_finish_iter2_prompt
from src.worker.prompts.finish_iter3_prompt import build_finish_iter3_prompt
from src.worker.prompts.finish_with_memory_prompt import (
    build_finish_with_memory_prompt,
)


def _open_memory_db():
    """Open a sync sqlite3 connection for recall_context lookups.

    Env-overridable via ``CURATOR_DB`` (default ``curator.db``). Read INSIDE
    function so test monkeypatches take effect. Returns a Connection with
    Row factory; caller must close.
    """
    import sqlite3
    path = os.environ.get("CURATOR_DB", "curator.db")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def record_memory_trace(**kwargs: Any) -> None:
    """Phase 87-02 telemetry hook.

    No-op by default — exposed at module level so tests (and a future
    react_worker wiring) can monkeypatch it to capture
    ``memory_context_tokens`` for an iter1 call without touching the
    sqlite agent_traces table from inside ``handle``.
    """
    return None

_DEFAULT_MODEL: Final[str] = "claude-sonnet-4-6"
# Iter 3 web-tool round-trips run slower than inline brainstorm/docs (120s).
_DEFAULT_TIMEOUT: Final[int] = 180


# ---------------------------------------------------------------------------
# Phase 54 (UX-DIV-01): `## Alternatives` section parser
# ---------------------------------------------------------------------------
# `## Alternatives` header — anchored to a line of its own. MULTILINE +
# IGNORECASE so the regex tolerates LLM casing drift. Plural-only by
# WR-07 (REVIEW-FIX P54): the YAML template, all three iter prompts,
# the docstring, and 54-VERIFICATION.md all hardcode the plural form;
# accepting singular caused a silent split between regex contract and
# downstream tooling that greps for `## Alternatives` literally
# (Obsidian dataview queries, future widgets reading the body).
_ALT_SECTION_RE: Final[re.Pattern[str]] = re.compile(
    r"^##\s*Alternatives\s*$",
    re.MULTILINE | re.IGNORECASE,
)
# Stub format locked: `- **[label]**: precondition: <text>`.
# Tolerant of: leading whitespace, `-` or `*` bullet markers, optional `[ ]`
# around the label inside the bold pair, optional whitespace around the
# `precondition` token. The label must NOT contain `]` or `*` to anchor the
# bold-pair close. Multiline so each stub matches as its own line.
_ALT_ENTRY_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*[-*]\s*\*\*\[?([^\]\*\n]+?)\]?\*\*\s*:\s*precondition\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_MAX_ALTS: Final[int] = 3
_MIN_ALTS: Final[int] = 2
_LABEL_MAX: Final[int] = 80
_PRECOND_MAX: Final[int] = 240

# Control / separator chars stripped before yaml.dump (V5 Input Validation,
# threat T-54-02). LLM output is untrusted. WR-03 (REVIEW-FIX P54): reuse
# the canonical ``_CONTROL_TRANS`` set from claude_subprocess so the
# docstring claim ("mirrors ``_sanitize_prompt`` control-char set") is
# enforced by sharing the actual table — not a Phase-54-local copy that
# drifted (was missing standalone backslash-r, letting Mac classic /
# broken-CRLF carriage returns through into yaml frontmatter values).
# carriage returns through into yaml frontmatter values).
# Canonical set strips NUL, CR, VT, FF, U+2028, U+2029 (deletion). The
# subsequent ``_sanitize_alt_field`` step collapses runs of whitespace, so
# deletion vs replace-with-space is behaviour-equivalent for label/precond
# (the surrounding ASCII whitespace lets ``str.split`` tokenise either way).
# Source-hygiene rule: U+2028 / U+2029 are referenced via `` `` /
# `` `` escape sequences in the canonical set — never paste the
# literal invisible characters into source (per plan source-hygiene rule +
# CI invisible-unicode warnings).
_ALT_CONTROL_TRANS: Final[dict[int, int | None]] = _CONTROL_TRANS


def _sanitize_alt_field(value: str, max_len: int) -> str:
    """Collapse multi-line + strip control chars + cap length on label/precond.

    Mirrors ``src/worker/claude_subprocess._sanitize_prompt`` control-char set
    by reusing its ``_CONTROL_TRANS`` table (T-54-02 mitigation): strips NUL,
    CR (incl. standalone Mac classic), VT, FF, U+2028, U+2029 (\u2028 /
    \u2029 by escape sequence, never literal). Embedded LF / CRLF collapses
    to a single space (single-line stub invariant). Length cap prevents a
    runaway label / precondition value from bloating the frontmatter
    (T-54-03 DoS).
    """
    if not value:
        return ""
    out = value.replace("\r\n", " ").replace("\n", " ")
    out = out.translate(_ALT_CONTROL_TRANS)
    # Collapse runs of whitespace to a single space so labels don't carry
    # accidental double-space artefacts from the LLM.
    out = " ".join(out.split())
    return out[:max_len].strip()


def _parse_alternatives_block(body: str | None) -> list[dict[str, Any]]:
    """Parse the `## Alternatives` section from an LLM-generated draft body.

    Returns a list of ``{id, label, precondition, primary}`` dicts, sized
    ``_MIN_ALTS`` (2) to ``_MAX_ALTS`` (3) inclusive. Returns ``[]`` when:
      - body is empty / None / not a string
      - no `## Alternatives` header is found
      - fewer than ``_MIN_ALTS`` parseable entries (under-min → drop all)
      - header is found but every entry line is malformed

    Invariants enforced on the way out:
      - ``len(result)`` in ``{2, 3}`` — >3 truncates to first 3, <2 returns []
      - exactly one entry has ``primary=True``. By locked decision (D-O1 /
        Pitfall #2 in ``54-RESEARCH.md``), ``alts[0]`` is the primary
        regardless of LLM output — body section ordering (primary header
        first, alternatives second) IS the source of truth, so no separate
        primary-header regex is consulted.
      - ``label`` and ``precondition`` are sanitised (single-line, no control
        chars, length-capped) before return.
      - IDs are 1..len(result) with no gaps (re-indexed after dropping any
        malformed entries that survived the regex match but failed
        sanitisation).

    Pure function — no I/O. Never raises; on any unexpected failure returns
    []. Mirrors ``docs._parse_target_path`` defensive contract.
    """
    if not body or not isinstance(body, str):
        return []
    section = _ALT_SECTION_RE.search(body)
    if not section:
        return []
    # BL-01 (REVIEW-FIX P54): restrict entry capture to the substring AFTER
    # the `## Alternatives` header. Otherwise any matching `- **[label]**:
    # precondition: X` line ANYWHERE in the body (primary-section bullets,
    # iter3 findings, YAML-template echoes) pollutes the alternatives list.
    # Also stop at the next `^##` header so a later section's bullets cannot
    # leak either (e.g. iter3's `## Citations`).
    tail = body[section.end():]
    next_section = re.search(r"^##\s", tail, re.MULTILINE)
    if next_section:
        tail = tail[:next_section.start()]
    matches = _ALT_ENTRY_RE.findall(tail)
    if len(matches) < _MIN_ALTS:
        return []

    # WR-04 (REVIEW-FIX P54): sanitise THEN truncate, not the other way
    # round. Previously the loop iterated ``matches[:_MAX_ALTS]`` — if any of
    # the first 3 entries failed sanitisation, valid entries at positions
    # 4+ were never seen and the parser dropped the whole block. Iterate
    # ALL matches, count successful entries, and break once we hit _MAX_ALTS.
    alts: list[dict[str, Any]] = []
    for label, precond in matches:
        clean_label = _sanitize_alt_field(label, _LABEL_MAX)
        clean_precond = _sanitize_alt_field(precond, _PRECOND_MAX)
        if not clean_label or not clean_precond:
            # Malformed entry (sanitisation emptied a field) — drop.
            continue
        alts.append({
            "id": 0,  # re-numbered below after drops
            "label": clean_label,
            "precondition": clean_precond,
            "primary": False,
        })
        if len(alts) >= _MAX_ALTS:
            break
    if len(alts) < _MIN_ALTS:
        return []
    # Re-id 1..len(alts) so any dropped entries don't leave gaps.
    for i, entry in enumerate(alts, start=1):
        entry["id"] = i
    # Exactly-one-primary invariant. Per D-O1 + Pitfall #2:
    # alts[0] is the primary regardless of whether the LLM flagged it,
    # because the BODY section ordering (primary header first, alternatives
    # second) is the source of truth.
    alts[0]["primary"] = True
    return alts


def _read_prev_iter_body(draft_path: str | None, expected_iter: int) -> str:
    """Read previous iter's draft FILE current contents (allows user edits).

    Strips frontmatter via ``frontmatter.loads`` so the next-iter prompt sees
    only the markdown body.

    Raises ValueError on:
      - draft_path is None / empty (no prior iter to read)
      - file does not exist (vault offline? user moved file?)
    Both errors carry helpful messages identifying ``expected_iter`` so the
    JSONL log surfaces which iter could not proceed.

    WR-01 (REVIEW-FIX P20): the previous defensive try/except around
    ``frontmatter.loads`` was unreachable in practice — python-frontmatter
    catches its own YAML parse errors internally and returns ``{}`` metadata
    + raw content (incl. fences). That meant a draft with mangled YAML
    leaked the raw frontmatter into the next-iter prompt despite the
    "fallback" comment. Replace with an explicit pre-check:
      1. If the file does not start with ``---\\n`` AND does not start with
         ``---\\r\\n``, skip the parse and return raw text — there are no
         fences to strip.
      2. Otherwise call ``loads`` and return ``.content``. If the YAML is
         malformed but enclosed in fences, manually strip everything between
         the opening ``---`` fence and the next ``---`` line so we don't
         leak frontmatter into the prompt.
    The ``except`` is preserved as a true last-resort guard — if loads ever
    raises (theoretical: corrupted unicode, library bug), we still attempt
    a fence strip rather than echoing raw frontmatter into the prompt.
    """
    if not draft_path:
        raise ValueError(
            f"finish iter {expected_iter}: reaction_row.draft_path is "
            "empty; cannot run without prior iter draft (file-as-input rule)"
        )
    path = Path(draft_path)
    if not path.exists():
        raise ValueError(
            f"finish iter {expected_iter}: prior iter file missing at "
            f"{draft_path!r} (vault offline? user moved file? file-as-input "
            "rule cannot proceed)"
        )
    text = path.read_text(encoding="utf-8")

    # No leading fence → no frontmatter to strip; return raw body.
    if not (text.startswith("---\n") or text.startswith("---\r\n")):
        return text

    try:
        post = frontmatter.loads(text)
    except Exception:
        # python-frontmatter normally swallows its own YAML errors, but if
        # the library ever surfaces one we still must NOT leak the raw
        # frontmatter into the next-iter prompt (test_handle_iter2_strips_
        # frontmatter_from_prev contract). Fall through to manual strip.
        return _manual_strip_frontmatter(text)

    body = post.content
    # If python-frontmatter swallowed a parse error, it returns ``{}``
    # metadata plus the WHOLE original text (fences included) as content.
    # Detect that case (body still starts with ``---``) and strip manually.
    if body.startswith("---\n") or body.startswith("---\r\n"):
        return _manual_strip_frontmatter(body)
    return body


def _manual_strip_frontmatter(text: str) -> str:
    """Drop the leading ``---``-fenced YAML block. Used when frontmatter.loads
    silently fails to strip (malformed YAML with intact fences).

    Strategy: split on ``\\n---\\n`` (or CRLF) once after a leading ``---``,
    keep everything after the second fence. If no closing fence is found,
    return the original text minus the opening ``---`` line so SOMETHING
    visible reaches the prompt without leaking ``key: value`` lines.
    """
    # Normalise CRLF for the search; don't mutate original.
    norm = text.replace("\r\n", "\n")
    if not norm.startswith("---\n"):
        return text
    # Find the closing fence: a ``---`` on its own line after the opener.
    after_open = norm[4:]  # skip leading ``---\n``
    close_idx = after_open.find("\n---\n")
    if close_idx == -1:
        # No closing fence — strip just the opening line as best-effort.
        return after_open
    # +1 for the ``\n`` we matched after the YAML body, +4 for ``---\n``.
    return after_open[close_idx + 1 + 4:]


def handle(
    reaction_row: dict[str, Any],
    digest_item: dict[str, Any],
    vault_drafts_dir: str,
) -> str:
    """Write a finish-chain draft for ``reaction_row['id']`` at the current iter.

    Inputs:
      reaction_row: dict from ``claim_pending_reaction`` (Plan 17-02). Required
        keys: ``id``, ``item_id``, ``action``, ``current_iter``. Optional:
        ``total_iters`` (defaults to 3), ``draft_path`` (required for iter 2/3).
      digest_item: dict shaped like a row from ``digest_items``. Required keys:
        ``item_id``, ``channel`` (nullable), ``url`` (nullable), ``snippet``
        (nullable for iter 2/3 — they read the prev iter file instead),
        ``linked_project`` (nullable).
      vault_drafts_dir: absolute path to ``<vault>/10_Drafts/``.

    Side effects:
      - For iter 2/3: reads the previous iter's draft file at
        ``reaction_row['draft_path']`` (CURRENT contents — file-as-input rule).
      - Calls ``run_powershell`` with the built prompt (exceptions propagate).
      - Writes a UTF-8 NO-BOM, LF-newline file at
        ``<vault_drafts_dir>/[_PRELIMINARY_]<reaction_id>-finish-iter<N>.md``.

    Raises:
      ValueError: ``current_iter`` ∉ {1,2,3}; or iter 2/3 with missing/None
        draft_path or missing prior iter file.
      ClaudeSubprocessError / ClaudeSubprocessTimeout: from ``run_powershell``;
        propagate to caller (react_worker._dispatch handles via _mark_failed).

    Returns: absolute path of the written file as a string.
    """
    rid = reaction_row["id"]
    current_iter = reaction_row.get("current_iter", 1)
    total_iters = reaction_row.get("total_iters", 3)

    if current_iter not in (1, 2, 3):
        raise ValueError(
            f"finish.handle: current_iter={current_iter!r} not in {{1,2,3}} "
            f"(reaction_id={rid})"
        )

    if current_iter == 1:
        prompt = build_finish_iter1_prompt(
            snippet=digest_item.get("snippet"),
            channel=digest_item.get("channel"),
            url=digest_item.get("url"),
            linked_project=digest_item.get("linked_project"),
        )
        # Phase 87-02 (WORKER-MEM-02): canary memory-layer injection.
        # Env read INSIDE function (CLAUDE.md module-level env binding rule).
        # OFF path (default) leaves ``prompt`` byte-identical to baseline.
        # Iter1 only — non-iter1 paths never consult the memory layer.
        if os.environ.get("WORKER_MEMORY_MODE", "off") == "on":
            source_url = digest_item.get("url")
            if source_url:
                memory_ctx = None
                try:
                    db_conn = _open_memory_db()
                    try:
                        memory_ctx = recall_context(source_url, 1, db_conn)
                    finally:
                        try:
                            db_conn.close()
                        except Exception:
                            pass
                except Exception:
                    # Fail-soft: any retrieval error falls back to OFF path.
                    memory_ctx = None
                if memory_ctx and (memory_ctx.get("prior_drafts") or memory_ctx.get("neighbor_chunks")):
                    new_prompt = build_finish_with_memory_prompt(prompt, memory_ctx)
                    # Approx token cost: (len(new) - len(old)) // 4
                    memory_tokens = max(0, (len(new_prompt) - len(prompt)) // 4)
                    prompt = new_prompt
                    try:
                        record_memory_trace(
                            reaction_id=rid,
                            current_iter=1,
                            source_url=source_url,
                            memory_context_tokens=memory_tokens,
                        )
                    except Exception:
                        # Telemetry must never break the production path.
                        pass
    elif current_iter == 2:
        prev = _read_prev_iter_body(reaction_row.get("draft_path"), 2)
        prompt = build_finish_iter2_prompt(
            prev_iter_content=prev,
            channel=digest_item.get("channel"),
            url=digest_item.get("url"),
            linked_project=digest_item.get("linked_project"),
        )
    else:  # current_iter == 3
        prev = _read_prev_iter_body(reaction_row.get("draft_path"), 3)
        prompt = build_finish_iter3_prompt(
            prev_iter_content=prev,
            channel=digest_item.get("channel"),
            url=digest_item.get("url"),
            linked_project=digest_item.get("linked_project"),
        )

    # P91-01: prompt-token telemetry. Read threshold INSIDE function to dodge
    # the module-level env binding trap. Logging wrapped in try/except as an
    # extra guard — log_reaction_event is already fail-soft, but telemetry
    # MUST NEVER break the production path.
    try:
        prompt_tokens = estimate_tokens(prompt)
        threshold = int(os.environ.get("TOKEN_AUDIT_THRESHOLD", "100000"))
        event: dict[str, Any] = {
            "mode": "finish",
            "reaction_id": rid,
            "current_iter": current_iter,
            "prompt_tokens": prompt_tokens,
        }
        if prompt_tokens > threshold:
            event["flag"] = "token_cliff"
        log_reaction_event(event)
    except Exception:
        # Telemetry must never break the production path.
        pass

    # May raise ClaudeSubprocessError / ClaudeSubprocessTimeout — propagate.
    body = run_powershell(prompt, model=_DEFAULT_MODEL, timeout=_DEFAULT_TIMEOUT)
    # Phase 54 (UX-DIV-01): parse the `## Alternatives` block out of the RAW
    # LLM output BEFORE prepend_digest_entry (mirrors docs._parse_target_path
    # ordering at docs.py:141). The parser is defensive — never raises;
    # returns [] on parse failure or legacy bodies, in which case
    # `alternatives_count: 0` is written explicitly per D-O3.
    parsed_alts = _parse_alternatives_block(body)
    # Only iter 1 gets the digest entry prepended — iter 2/3 inherit/refine
    # the prev iter body, which already carries the block from iter 1.
    if current_iter == 1:
        body = prepend_digest_entry(body, digest_item.get("entry_md"))

    linked = digest_item.get("linked_project")
    _snippet = digest_item.get("snippet") or ""
    meta: dict[str, Any] = {
        "source": "digest-reaction",
        "reaction_id": rid,
        "item_id": reaction_row["item_id"],
        "action": "finish",
        "current_iter": current_iter,
        "total_iters": total_iters,
        "linked_project": linked,
        "linked_project_status": "resolved" if linked else "unresolved",
        "gate_state": gate_state_value("finish"),
        # IN-04: standardise on datetime.UTC (Python 3.11+).
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        # Source context: lets operator identify the post without DB JOIN.
        "source_url": digest_item.get("url"),
        "source_name": digest_item.get("channel"),
        "source_excerpt": _snippet[:200],
        # Phase 54 (UX-DIV-01): alternatives list + count. Per D-O3,
        # `alternatives_count` is ALWAYS present — 0 sentinel for legacy /
        # parse-failure so analytics never need COALESCE.
        "alternatives": parsed_alts,
        "alternatives_count": len(parsed_alts),
    }

    post = frontmatter.Post(content=body, **meta)
    rendered = frontmatter.dumps(post)

    # WORKER-10: PRELIMINARY label applied at the single choke point.
    filename = apply_preliminary_label(
        f"{rid}-finish-iter{current_iter}.md", "finish"
    )
    target = Path(vault_drafts_dir) / filename
    # WR-04: write to <target>.tmp first, fsync, then os.replace.
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
