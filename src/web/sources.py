import json
import logging
import math
import os
from pathlib import Path

import aiosqlite
from fastapi import APIRouter, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.categories import CANONICAL_CATEGORIES
from src.editor.prompt_mixer import KNOWN_STYLES

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

_ROOT = Path(__file__).resolve().parents[2]
SOURCES_FILE = _ROOT / "config" / "sources.txt"
CANDIDATES_FILE = _ROOT / "config" / "sources_candidates.txt"
REJECTED_FILE = _ROOT / "config" / "sources_rejected.txt"

VALID_CATEGORIES = list(CANONICAL_CATEGORIES)
# Platforms available in the manual-add form (not all platforms in sources.txt)
ADDABLE_PLATFORMS = ["telegram", "reddit", "youtube"]


def _normalize(target: str) -> str:
    t = target.strip().lower().lstrip("@")
    if t.startswith("r/"):
        t = t[2:]
    return t


# WR-02: expose _normalize to Jinja so the template key matches the Python
# key used to build db_meta_by_key. The previous chained `|lower|replace`
# diverged (no .strip(), replaced all '@' and any 'r/' substring).
templates.env.filters["normalize_target"] = _normalize


def _sanitize_field(value: str) -> str:
    """Strip characters that would corrupt sources.txt line format."""
    return value.replace("|", "").replace("\n", " ").replace("\r", " ").strip()


def parse_sources(path: Path) -> list[dict]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if not parts[0]:
            continue
        result.append({
            "target": parts[0],
            "category": parts[1] if len(parts) > 1 else "other",
            "display_name": parts[2] if len(parts) > 2 else parts[0],
            "platform": parts[3] if len(parts) > 3 else "telegram",
        })
    return result


def parse_candidates(path: Path) -> list[dict]:
    if not path.exists():
        return []
    result = []
    pending_meta: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            meta: dict = {}
            for part in line.lstrip("#").split("|"):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    meta[k.strip()] = v.strip()
            if "score" in meta:
                pending_meta = meta
            continue
        parts = [p.strip() for p in line.split("|")]
        result.append({
            "handle": parts[0],
            "category": parts[1] if len(parts) > 1 else "other",
            "display_name": parts[2] if len(parts) > 2 else parts[0].lstrip("@"),
            "score": float(pending_meta.get("score", 0)),
            "posts_count": int(pending_meta.get("posts_seen", 0)),
            "russian_ratio": float(pending_meta.get("russian", 0)),
        })
        pending_meta = {}
    return result


def parse_rejected(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        _normalize(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


class ApprovePayload(BaseModel):
    handle: str
    category: str
    display_name: str


class RejectPayload(BaseModel):
    handle: str


class AddPayload(BaseModel):
    platform: str
    target: str
    category: str
    display_name: str


class AdapterWeightsPayload(BaseModel):
    # Phase 81 / MIX-PROMPTS-01: per-source mixture-of-prompts override.
    # JSON string body (e.g. '{"timeline": 0.6, "paper": 0.4}') or null to clear.
    # Server-side validates parse + style names + non-negative weights.
    value: str | None


class ThresholdPayload(BaseModel):
    # Phase 62 / SOURCE-THRESH-01: per-source curator threshold override.
    # None clears the override -> falls back to env CURATOR_THRESHOLD (default 75).
    # Pydantic rejects non-numeric / non-null values with 422 automatically.
    value: float | None


@router.post("/api/sources/approve")
async def approve_source(payload: ApprovePayload):
    if not payload.handle.strip():
        raise HTTPException(status_code=422, detail="handle must not be empty")
    handle = _sanitize_field(payload.handle)
    display_name = _sanitize_field(payload.display_name)
    known = {_normalize(s["target"]) for s in parse_sources(SOURCES_FILE)}
    if _normalize(handle) in known:
        raise HTTPException(status_code=409, detail="Already in sources.txt")
    line = f"{handle} | {payload.category} | {display_name}\n"
    with open(SOURCES_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    return {"ok": True}


@router.post("/api/sources/reject")
async def reject_source(payload: RejectPayload):
    normalized = _normalize(payload.handle)
    with open(REJECTED_FILE, "a", encoding="utf-8") as f:
        f.write(normalized + "\n")
    return {"ok": True}


@router.post("/api/sources/add")
async def add_source(payload: AddPayload):
    if payload.platform not in ADDABLE_PLATFORMS:
        raise HTTPException(status_code=422, detail=f"Invalid platform: {payload.platform}")
    if not payload.target.strip():
        raise HTTPException(status_code=422, detail="target must not be empty")
    target = _sanitize_field(payload.target)
    display_name = _sanitize_field(payload.display_name)
    known = {_normalize(s["target"]) for s in parse_sources(SOURCES_FILE)}
    if _normalize(target) in known:
        raise HTTPException(status_code=409, detail="Already in sources.txt")
    if payload.platform == "telegram":
        line = f"{target} | {payload.category} | {display_name}\n"
    else:
        line = f"{target} | {payload.category} | {display_name} | {payload.platform}\n"
    with open(SOURCES_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    return {"ok": True}


@router.patch("/api/source/{source_id}/threshold")
async def update_source_threshold(source_id: int, payload: ThresholdPayload):
    """Phase 62 / SOURCE-THRESH-01: per-source curator threshold override.

    Validates 0-100 range (Pydantic rejects non-numeric/non-null with 422),
    persists to sources.curator_threshold. NULL value clears override ->
    falls back to env CURATOR_THRESHOLD (default 75). DB path read INSIDE
    function to avoid module-level env-binding gotcha (CLAUDE.md).

    Note: uvicorn cold restart required after adding this route — auto-reload
    misses new @router.patch paths (CLAUDE.md gotcha).
    """
    if payload.value is not None and not (0.0 <= payload.value <= 100.0):
        raise HTTPException(
            status_code=422, detail="value must be in [0, 100] or null"
        )
    db_path = os.environ.get("CURATOR_DB_PATH", "curator.db")
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT id FROM sources WHERE id=?", (source_id,)
        ) as cur:
            if not await cur.fetchone():
                raise HTTPException(status_code=404, detail="source not found")
        await db.execute(
            "UPDATE sources SET curator_threshold=? WHERE id=?",
            (payload.value, source_id),
        )
        await db.commit()
    return {"ok": True, "id": source_id, "curator_threshold": payload.value}


@router.patch("/api/source/{source_id}/adapter_weights")
async def update_source_adapter_weights(
    source_id: int, payload: AdapterWeightsPayload
):
    """Phase 81 / MIX-PROMPTS-01: per-source mixture-of-prompts override.

    Accepts a JSON-string body (parsed server-side) or null to clear. Validates:
    - parses as JSON dict
    - all values numeric and ≥0
    - at least one key in KNOWN_STYLES with positive weight

    Stores the operator-supplied string verbatim (preserves formatting).
    DB path read INSIDE function (module-level env binding gotcha).

    Note: uvicorn cold restart required after adding this route — auto-reload
    misses new @router.patch paths (CLAUDE.md gotcha).
    """
    db_path = os.environ.get("CURATOR_DB_PATH", "curator.db")

    if payload.value is None:
        # Clear override.
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                "SELECT id FROM sources WHERE id=?", (source_id,)
            ) as cur:
                if not await cur.fetchone():
                    raise HTTPException(status_code=404, detail="source not found")
            await db.execute(
                "UPDATE sources SET adapter_weights=NULL WHERE id=?", (source_id,)
            )
            await db.commit()
        return {"ok": True, "id": source_id, "adapter_weights": None}

    # Parse + validate. Reject NaN / Infinity literals at parse time (Python's
    # json.loads accepts them by default as a non-standard extension; downstream
    # f"{w:.2f}" would render "inf"/"nan" and poison renormalisation). WR-01.
    def _reject_non_finite(literal):
        raise ValueError(f"non-finite literal {literal!r} not allowed")

    try:
        parsed = json.loads(payload.value, parse_constant=_reject_non_finite)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=422, detail=f"invalid JSON: {e}")
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail="weights must be a JSON object")
    has_positive_known = False
    for k, v in parsed.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise HTTPException(
                status_code=422, detail=f"weight for {k!r} must be numeric"
            )
        if not math.isfinite(v):
            raise HTTPException(
                status_code=422,
                detail=f"weight for {k!r} must be a finite number",
            )
        if v < 0:
            raise HTTPException(
                status_code=422, detail=f"weight for {k!r} must be non-negative"
            )
        if k in KNOWN_STYLES and v > 0:
            has_positive_known = True
    if not has_positive_known:
        raise HTTPException(
            status_code=422,
            detail=(
                "at least one known style with positive weight required "
                f"(known: {list(KNOWN_STYLES)})"
            ),
        )

    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT id FROM sources WHERE id=?", (source_id,)
        ) as cur:
            if not await cur.fetchone():
                raise HTTPException(status_code=404, detail="source not found")
        await db.execute(
            "UPDATE sources SET adapter_weights=? WHERE id=?",
            (payload.value, source_id),
        )
        await db.commit()
    return {"ok": True, "id": source_id, "adapter_weights": payload.value}


async def _load_redundancy_pairs() -> dict[int, dict]:
    """Phase 86 / KL-UI-01: per-source lowest-JSD partner under threshold.

    Returns `{source_id: {"partner_id": int, "partner_display": str, "jsd": float}}`
    keyed by both members of each redundant pair. Each source maps to its
    LOWEST-JSD partner only (column stays compact).

    Threshold env `SOURCE_KL_REDUNDANT_THRESHOLD` (default 0.1) read INSIDE
    this helper so operators can tune without restart-code changes (CLAUDE.md
    module-level-env-binding gotcha).

    Fail-soft: missing DB / missing source_similarity table → returns {}.
    """
    db_path = os.environ.get("CURATOR_DB_PATH", "curator.db")
    if not Path(db_path).exists():
        return {}
    threshold = float(os.environ.get("SOURCE_KL_REDUNDANT_THRESHOLD", "0.1"))
    out: dict[int, dict] = {}
    try:
        async with aiosqlite.connect(db_path) as db:
            # Build id → display_name lookup first.
            display_by_id: dict[int, str] = {}
            async with db.execute(
                "SELECT id, display_name, target FROM sources"
            ) as cur:
                async for row in cur:
                    display_by_id[row[0]] = row[1] or row[2]
            # ORDER BY jsd ASC → first hit per source_id is the lowest partner.
            # WR-04: deterministic secondary sort on (a_id, b_id) so equal-JSD
            # ties resolve consistently across re-runs (SQLite default ordering
            # is implementation-defined).
            async with db.execute(
                "SELECT source_a_id, source_b_id, jsd_divergence "
                "FROM source_similarity WHERE jsd_divergence < ? "
                "ORDER BY jsd_divergence ASC, source_a_id ASC, source_b_id ASC",
                (threshold,),
            ) as cur:
                async for a, b, jsd in cur:
                    if a not in out:
                        out[a] = {
                            "partner_id": b,
                            "partner_display": display_by_id.get(b, f"src-{b}"),
                            "jsd": jsd,
                        }
                    if b not in out:
                        out[b] = {
                            "partner_id": a,
                            "partner_display": display_by_id.get(a, f"src-{a}"),
                            "jsd": jsd,
                        }
    except Exception:
        logger.exception("Failed to load source_similarity for /sources page")
        return {}
    return out


async def _load_quality_by_source_id() -> dict[int, dict]:
    """Phase 89 / CHQ-UI-01: per-source quality_score for /sources column.

    Returns `{source_id: {"quality_score": float|None, "quality_computed_at": str|None}}`.
    Reads sources.quality_score + quality_computed_at (added in P89-01 migrate_db).

    Fail-soft: missing DB or missing column (legacy schema) → returns {}.
    DB path read INSIDE function (module-level env binding gotcha — CLAUDE.md).
    """
    db_path = os.environ.get("CURATOR_DB_PATH", "curator.db")
    if not Path(db_path).exists():
        return {}
    out: dict[int, dict] = {}
    try:
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                "SELECT id, quality_score, quality_computed_at FROM sources"
            ) as cur:
                async for row in cur:
                    out[row[0]] = {
                        "quality_score": row[1],
                        "quality_computed_at": row[2],
                    }
    except Exception:
        logger.exception("Failed to load source quality_score for /sources page")
        return {}
    return out


async def _load_auto_unsub_by_source_id() -> dict[int, dict]:
    """Phase 100-02 / AUTO-UNSUB-01: per-source auto-unsub state for /sources badges.

    Returns ``{source_id: {"warned_at": str|None, "disabled_at": str|None,
    "override": int|None}}`` for sources where either warned_at OR disabled_at
    is set (clean sources are omitted to keep the badge dict small).

    Fail-soft: missing DB or legacy schema (no auto_unsub_warned_at column)
    → returns ``{}``. DB path read INSIDE function (module-level env binding
    gotcha — CLAUDE.md).
    """
    db_path = os.environ.get("CURATOR_DB_PATH", "curator.db")
    if not Path(db_path).exists():
        return {}
    out: dict[int, dict] = {}
    try:
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                "SELECT id, auto_unsub_warned_at, auto_unsubbed_at, auto_unsub_override "
                "FROM sources"
            ) as cur:
                async for row in cur:
                    if row[1] or row[2]:  # only surface flagged sources
                        out[row[0]] = {
                            "warned_at": row[1],
                            "disabled_at": row[2],
                            "override": row[3],
                        }
    except Exception:
        logger.exception("Failed to load auto_unsub state for /sources page")
        return {}
    return out


async def _load_db_source_meta() -> dict[tuple[str, str], dict]:
    """Phase 62: load (platform, normalized_target) -> {id, curator_threshold}
    for the /sources active-table inline-edit column.

    Fail-soft: returns empty dict if DB missing / unreadable so /sources never
    returns 500 in fresh-checkout demos. Normalised key matches _normalize()
    (lowercase, leading @/r/ stripped) so file rows match DB rows even when
    operators write @Handle in sources.txt and DB stores @handle.
    """
    db_path = os.environ.get("CURATOR_DB_PATH", "curator.db")
    if not Path(db_path).exists():
        return {}
    meta: dict[tuple[str, str], dict] = {}
    try:
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                "SELECT id, platform, target, curator_threshold, adapter_weights "
                "FROM sources"
            ) as cur:
                async for row in cur:
                    key = (row[1], _normalize(row[2]))
                    meta[key] = {
                        "id": row[0],
                        "curator_threshold": row[3],
                        "adapter_weights": row[4],
                    }
    except Exception:
        logger.exception("Failed to load DB source meta for /sources page")
        return {}
    return meta


@router.get("/sources")
async def sources_page(request: Request):
    candidates = parse_candidates(CANDIDATES_FILE)
    known = {_normalize(s["target"]) for s in parse_sources(SOURCES_FILE)}
    rejected = parse_rejected(REJECTED_FILE)
    filtered = [
        c for c in candidates
        if _normalize(c["handle"]) not in known
        and _normalize(c["handle"]) not in rejected
    ]
    active = parse_sources(SOURCES_FILE)
    active_by_platform: dict[str, list] = {}
    for s in active:
        active_by_platform.setdefault(s["platform"], []).append(s)
    db_meta_by_key = await _load_db_source_meta()
    redundancy_by_source_id = await _load_redundancy_pairs()
    quality_by_source_id = await _load_quality_by_source_id()
    auto_unsub_by_source_id = await _load_auto_unsub_by_source_id()

    # Phase 89 / CHQ-UI-01: optional sort param, whitelisted (injection-safe).
    sort = request.query_params.get("sort")
    sort_dir = request.query_params.get("dir", "desc")
    if sort != "quality_score":
        sort = None
    if sort_dir not in ("asc", "desc"):
        sort_dir = "desc"

    if sort == "quality_score":
        def _sort_key(s):
            meta = db_meta_by_key.get((s["platform"], _normalize(s["target"])))
            sid = meta.get("id") if meta else None
            q = quality_by_source_id.get(sid, {}) if sid else {}
            score = q.get("quality_score")
            # NULL last in both directions; secondary key = score (negated for desc).
            if score is None:
                return (1, 0.0)
            return (0, -score if sort_dir == "desc" else score)
        for plat in list(active_by_platform.keys()):
            active_by_platform[plat] = sorted(
                active_by_platform[plat], key=_sort_key
            )

    return templates.TemplateResponse(
        request,
        "sources.html",
        {
            "candidates": filtered,
            "active_by_platform": active_by_platform,
            "categories": VALID_CATEGORIES,
            "db_meta_by_key": db_meta_by_key,
            "redundancy_by_source_id": redundancy_by_source_id,
            "quality_by_source_id": quality_by_source_id,
            "auto_unsub_by_source_id": auto_unsub_by_source_id,
            "sort": sort,
            "sort_dir": sort_dir,
        },
    )
