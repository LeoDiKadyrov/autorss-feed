import logging
import os
import re
import shutil
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

VALID_STATES = {"preliminary", "research", "execute", "research_execute", "archived", "needs_review"}


def _drafts_dir() -> Path:
    draft_output = os.environ.get("DRAFT_OUTPUT_DIR")
    if draft_output:
        return Path(draft_output) / "10_Drafts"
    vault_root = os.environ.get("OBSIDIAN_VAULT_PATH")
    if vault_root:
        return Path(vault_root) / "10_Drafts"
    return Path(os.environ.get("VAULT_DRAFTS_PATH", "vault/10_Drafts"))


def find_draft(reaction_id: int) -> Path | None:
    if not isinstance(reaction_id, int) or reaction_id < 0:
        raise ValueError(f"reaction_id must be a non-negative int, got {reaction_id!r}")
    matches = list(_drafts_dir().glob(f"_PRELIMINARY_{reaction_id}-*.md"))
    if len(matches) > 1:
        raise RuntimeError(f"Multiple drafts for reaction_id={reaction_id}: {[p.name for p in matches]}")
    return matches[0] if matches else None


def read_draft(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
    if not m:
        return {}, text
    fm = yaml.safe_load(m.group(1)) or {}
    return fm, m.group(2)


def write_draft(path: Path, fm: dict, body: str) -> None:
    content = f"---\n{yaml.dump(fm, allow_unicode=True, default_flow_style=False)}---\n{body}"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def set_gate_state(reaction_id: int, state: str) -> bool:
    if state not in VALID_STATES:
        raise ValueError(f"Invalid state {state!r}. Valid: {VALID_STATES}")
    path = find_draft(reaction_id)
    if path is None:
        return False
    if state == "archived":
        archive_dir = _drafts_dir() / "_archive"
        archive_dir.mkdir(exist_ok=True)
        shutil.move(str(path), str(archive_dir / path.name))
        return True
    fm, body = read_draft(path)
    fm["gate_state"] = state
    write_draft(path, fm, body)
    return True


def set_status(reaction_id: int, state: str) -> bool:
    """Phase 93-01 — thin wrapper over set_gate_state for status semantics.

    Raises ValueError on unknown state (passes through from set_gate_state).
    Returns False if the draft is not found, True on successful write.
    """
    return set_gate_state(reaction_id, state)


def list_pending() -> list[dict]:
    """Return drafts with gate_state=='preliminary' (awaiting user review).

    Does NOT return research/execute/research_execute states — those are
    picked up directly by finish_worker.find_active_drafts().
    """
    results = []
    for p in _drafts_dir().glob("_PRELIMINARY_*.md"):
        try:
            fm, _ = read_draft(p)
            if fm.get("gate_state") == "preliminary":
                results.append({
                    "reaction_id": fm.get("reaction_id"),
                    "item_id": fm.get("item_id"),
                    "gate_state": fm.get("gate_state", "preliminary"),
                    "action": fm.get("action", ""),
                    "source_name": fm.get("source_name", ""),
                    "file": str(p),
                })
        except Exception as exc:
            logger.warning("list_pending: skipping %s: %s", p.name, exc)
    return results


def archive_draft(reaction_id: int) -> bool:
    path = find_draft(reaction_id)
    if path is None:
        return False
    archive_dir = _drafts_dir() / "_archive"
    archive_dir.mkdir(exist_ok=True)
    shutil.move(str(path), str(archive_dir / path.name))
    return True


