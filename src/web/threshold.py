"""Phase 106 / SELF-TUNE-01 — operator-gated threshold apply/dismiss/unset.

Endpoints:
  POST /api/threshold/{rec_id}/apply    — write yaml override + mark applied_at
  POST /api/threshold/{rec_id}/dismiss  — set applied_at='dismissed' (sentinel)
  POST /api/threshold/unset/{category}  — remove yaml override (rollback path)

YAML strategy: PyYAML for parsing + manual line-edit on text round-trip to
preserve comments + key order. ruamel.yaml NOT added as new dep per CONTEXT
line 96 (regex line-edit fallback).

Atomic write: tmp file + os.replace — partial yaml never observable on disk.

`applied_at='dismissed'` is a sentinel string (non-NULL), so the Plan 01
partial idx `WHERE applied_at IS NULL` correctly excludes it from pending.
"""
from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from src.llm import routing as routing_mod
from src.web.csrf import csrf_ok

router = APIRouter()

TZ = timezone(timedelta(hours=5))

# Module-level — tests monkeypatch.
DB_PATH: Path = Path(__file__).resolve().parents[2] / "curator.db"


def _yaml_path() -> Path:
    """Resolve the curator_routing.yaml path via routing module (tests can swap)."""
    return routing_mod._CONFIG_PATH


def _set_threshold_in_yaml_text(text: str, category: str, value: int) -> str:
    """Set thresholds[category]=value in yaml text, preserving comments/order.

    If `thresholds:` block missing, append at end with leading comment.
    If category already present, replace the value line.
    If absent, insert in alphabetical order within the block.
    """
    # Validate category name to avoid regex injection (cat is the URL param)
    if not re.fullmatch(r"[a-z_]+", category):
        raise ValueError(f"invalid category name: {category!r}")
    val_str = str(int(value))

    # Detect block presence + extent
    lines = text.splitlines(keepends=True)
    block_start = None  # line index of `thresholds:`
    block_end = None    # line index AFTER last block line (exclusive)
    for i, ln in enumerate(lines):
        if re.match(r"^thresholds\s*:\s*$", ln.rstrip("\n")):
            block_start = i
            j = i + 1
            while j < len(lines):
                s = lines[j]
                stripped = s.strip()
                if stripped == "" or stripped.startswith("#"):
                    j += 1
                    continue
                # block line is indented (2 spaces)
                if s.startswith("  ") and not s.startswith("   "):
                    j += 1
                    continue
                break
            block_end = j
            break

    if block_start is None:
        # Append block at end (ensure trailing newline first)
        suffix = ""
        if not text.endswith("\n"):
            suffix = "\n"
        block = (
            "\n# Phase 106 SELF-TUNE-01: per-category curator threshold overrides.\n"
            "# Written by POST /api/threshold/{id}/apply only. Operator-managed.\n"
            "thresholds:\n"
            f"  {category}: {val_str}\n"
        )
        return text + suffix + block

    # Block exists — does category line exist within it?
    cat_re = re.compile(rf"^(\s+){re.escape(category)}\s*:\s*.*$")
    for i in range(block_start + 1, block_end):
        m = cat_re.match(lines[i].rstrip("\n"))
        if m:
            indent = m.group(1)
            lines[i] = f"{indent}{category}: {val_str}\n"
            return "".join(lines)

    # Not present — insert in alphabetical order within block
    existing_keys: list[tuple[int, str]] = []  # (line_idx, key)
    key_re = re.compile(r"^\s+([a-z_]+)\s*:")
    for i in range(block_start + 1, block_end):
        m = key_re.match(lines[i].rstrip("\n"))
        if m:
            existing_keys.append((i, m.group(1)))

    insert_idx = block_end  # default: append at end of block
    for idx, key in existing_keys:
        if category < key:
            insert_idx = idx
            break

    new_line = f"  {category}: {val_str}\n"
    lines.insert(insert_idx, new_line)
    return "".join(lines)


def _remove_threshold_in_yaml_text(text: str, category: str) -> str:
    """Delete thresholds[category] line. If block becomes empty, drop block + preceding comments."""
    if not re.fullmatch(r"[a-z_]+", category):
        raise ValueError(f"invalid category name: {category!r}")
    lines = text.splitlines(keepends=True)
    block_start = None
    block_end = None
    for i, ln in enumerate(lines):
        if re.match(r"^thresholds\s*:\s*$", ln.rstrip("\n")):
            block_start = i
            j = i + 1
            while j < len(lines):
                s = lines[j]
                stripped = s.strip()
                if stripped == "" or stripped.startswith("#"):
                    j += 1
                    continue
                if s.startswith("  ") and not s.startswith("   "):
                    j += 1
                    continue
                break
            block_end = j
            break
    if block_start is None:
        return text  # nothing to remove

    cat_re = re.compile(rf"^\s+{re.escape(category)}\s*:\s*.*$")
    removed_any = False
    new_block: list[str] = []
    remaining_keys = 0
    key_re = re.compile(r"^\s+([a-z_]+)\s*:")
    for i in range(block_start + 1, block_end):
        if cat_re.match(lines[i].rstrip("\n")):
            removed_any = True
            continue
        new_block.append(lines[i])
        if key_re.match(lines[i].rstrip("\n")):
            remaining_keys += 1

    if not removed_any:
        return text

    if remaining_keys == 0:
        # Drop the whole block + preceding comment block (header) + leading blank line.
        # Walk backwards from block_start through contiguous comment / blank lines.
        del_start = block_start
        i = block_start - 1
        while i >= 0:
            stripped = lines[i].strip()
            if stripped.startswith("#") or stripped == "":
                del_start = i
                i -= 1
                continue
            break
        return "".join(lines[:del_start] + lines[block_end:])

    return "".join(
        lines[:block_start + 1] + new_block + lines[block_end:]
    )


def _atomic_write(target: Path, content: str) -> None:
    """Write content to target atomically via tmp + os.replace.

    WR-03 (Phase 106 review): cleanup only fires on the error path. Pre-fix
    the finally-block called `tmp.unlink()` unconditionally — on a fast FS
    where another process recreated `target` between `os.replace` and the
    finally-block, the `tmp.exists()` check could race and delete the
    renamed target. Now we set `replaced=True` after a successful replace
    and skip cleanup entirely; the error branch handles leftover tmp.
    """
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    replaced = False
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass  # best-effort on platforms that don't support fsync on text
        os.replace(str(tmp), str(target))
        replaced = True
    except Exception:
        # Best-effort cleanup of the temp file. Never raise on cleanup failure.
        if not replaced and tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def _connect_db():
    path = DB_PATH
    if not path.exists():
        raise HTTPException(status_code=503, detail="curator.db missing")
    return sqlite3.connect(str(path))


@router.post("/api/threshold/{rec_id}/apply")
def apply_recommendation(rec_id: int, request: Request) -> JSONResponse:
    """Operator-gated apply: write yaml override + mark applied_at='<iso>'.

    BL-02 (Phase 106 review): mirror Phase 104 WR-05 CSRF gate.
    WR-04 (Phase 106 review): guarded UPDATE eliminates TOCTOU between
    SELECT and UPDATE — concurrent apply requests for the same row return
    409 instead of double-writing the yaml.
    WR-02 (Phase 106 review): removed routing_mod._reset_cache() — yaml
    mtime check in _load() handles refresh automatically, and the global
    reset purged unrelated P78/P83/P98 caches.
    """
    if not csrf_ok(request):
        return JSONResponse(
            {"error": "CSRF: origin not allowed"}, status_code=403
        )
    conn = _connect_db()
    try:
        cur = conn.execute(
            "SELECT category, recommended_threshold, applied_at "
            "FROM threshold_recommendations WHERE id=?",
            (rec_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="recommendation not found")
        category, rec_threshold, applied_at = row
        if applied_at is not None:
            raise HTTPException(status_code=409, detail="already applied or dismissed")

        yaml_target = _yaml_path()
        original = yaml_target.read_text(encoding="utf-8") if yaml_target.exists() else ""
        new_text = _set_threshold_in_yaml_text(original, category, int(rec_threshold))
        # Atomic write FIRST — if it fails, DB stays NULL (all-or-nothing).
        try:
            _atomic_write(yaml_target, new_text)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"yaml write failed: {e}")
        # WR-04: guarded UPDATE — fail to 409 if another request raced us.
        now = datetime.now(TZ).isoformat()
        upd = conn.execute(
            "UPDATE threshold_recommendations "
            "SET applied_at=?, applied_by='operator' "
            "WHERE id=? AND applied_at IS NULL",
            (now, rec_id),
        )
        conn.commit()
        if upd.rowcount != 1:
            raise HTTPException(status_code=409, detail="already applied or dismissed")
        # WR-02: do NOT call routing_mod._reset_cache() — _load() already
        # re-stats the yaml on every call and picks up the new mtime
        # automatically. Reset would purge unrelated P78/P83/P98 caches.
        return JSONResponse(
            {"ok": True, "id": rec_id, "category": category,
             "recommended_threshold": int(rec_threshold), "applied_at": now}
        )
    finally:
        conn.close()


@router.post("/api/threshold/{rec_id}/dismiss")
def dismiss_recommendation(rec_id: int, request: Request) -> JSONResponse:
    """Set applied_at='dismissed' (sentinel; yaml untouched).

    BL-02: CSRF gate (mirror apply_recommendation).
    """
    if not csrf_ok(request):
        return JSONResponse(
            {"error": "CSRF: origin not allowed"}, status_code=403
        )
    conn = _connect_db()
    try:
        cur = conn.execute(
            "UPDATE threshold_recommendations "
            "SET applied_at='dismissed' WHERE id=? AND applied_at IS NULL",
            (rec_id,),
        )
        conn.commit()
        if cur.rowcount == 0:
            # Already applied/dismissed or missing
            row = conn.execute(
                "SELECT applied_at FROM threshold_recommendations WHERE id=?",
                (rec_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="recommendation not found")
            raise HTTPException(status_code=409, detail="already applied or dismissed")
        return JSONResponse({"ok": True, "id": rec_id, "applied_at": "dismissed"})
    finally:
        conn.close()


@router.post("/api/threshold/unset/{category}")
def unset_threshold(category: str, request: Request) -> JSONResponse:
    """Remove per-category yaml override. Rollback path — no DB write.

    BL-02: CSRF gate.
    WR-02: removed routing_mod._reset_cache() (yaml mtime check handles it).
    """
    if not csrf_ok(request):
        return JSONResponse(
            {"error": "CSRF: origin not allowed"}, status_code=403
        )
    if not re.fullmatch(r"[a-z_]+", category):
        raise HTTPException(status_code=400, detail="invalid category name")
    yaml_target = _yaml_path()
    if not yaml_target.exists():
        raise HTTPException(status_code=404, detail="curator_routing.yaml missing")
    original = yaml_target.read_text(encoding="utf-8")
    new_text = _remove_threshold_in_yaml_text(original, category)
    if new_text == original:
        return JSONResponse({"ok": True, "changed": False, "category": category})
    _atomic_write(yaml_target, new_text)
    return JSONResponse({"ok": True, "changed": True, "category": category})
