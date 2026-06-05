"""OSS entry point — generic routes only. Personal routes live in src/web/main.py.

This module is the public-facing FastAPI application for the autorss_feed OSS release.
It exposes the digest feed, feedback, search, draft management, and agent traces.
Personal-only routes and personal module imports are absent from this file.
"""

import asyncio
import html as _html
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Literal

import aiosqlite
import markdown
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.env_loader import load_env
from src.web.reactions import router as reactions_router
from src.web.sources import router as sources_router
from src.web.chat import router as chat_router
from src.web.threshold import router as threshold_router
from src.web.dwell import router as dwell_router
from src.worker.precision import get_gate_status
from src.worker.draft_manager import list_pending, set_gate_state
from src.database.client import (
    get_all_feedback,
    get_digest_by_id,
    get_digest_nav,
    get_digests,
    get_paper_dois_for_digest,
    get_posts_with_extraction,
    get_saliency_by_item,
    get_suspect_post_ids,
    record_feedback,
)
from src.database.connection import open_db

logger = logging.getLogger(__name__)

# Load .env at app init (no-op under pytest via AUTORSS_DISABLE_DOTENV).
load_env()

app = FastAPI()
app.include_router(reactions_router)
app.include_router(sources_router)
app.include_router(chat_router)
app.include_router(threshold_router)
app.include_router(dwell_router)

_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# WR-02: DB_PATH stays as a module-level attribute for backward compat with
# tests that monkeypatch it via `monkeypatch.setattr("src.web.main_oss.DB_PATH", ...)`.
# Functions below call `_get_db_path()` which prefers the (possibly monkeypatched)
# module attribute but otherwise reads DB_PATH env fresh.
DB_PATH = "curator.db"


def _get_db_path() -> str:
    """Return the configured DB path, deferring env reads to call time.

    Resolution order:
      1. Module-level DB_PATH if monkeypatched away from the default sentinel.
      2. DB_PATH environment variable.
      3. "curator.db" fallback.
    """
    if DB_PATH != "curator.db":
        return DB_PATH
    return os.environ.get("DB_PATH", "curator.db")


# LOG_PATH: <root>/logs/pipeline.log
LOG_PATH: Path = Path(__file__).resolve().parents[2] / "logs" / "pipeline.log"

_LOG_OK_RE = re.compile(
    r"^(?P<ts>\S+)\s*\|\s*OK\s*\|\s*"
    r"collected=(?P<collected>\d+)\s+"
    r"curated=(?P<curated>\d+)\s+"
    r"digest=(?P<digest_state>\S+)"
    r"(?:\s+tg_discovered=(?P<tg_discovered>-?\d+))?"
    r"(?:\s+ig_collected=(?P<ig_collected>\d+))?"
    r"(?:\s+cognition_tagged=(?P<cognition_tagged>\d+))?"
    r"(?:\s+gaming_q=(?P<gaming_q>\d+))?"
    r"(?:\s+health_days=(?P<health_days>\d+))?"
    r"(?:\s+dataset_size=(?P<dataset_size>\d+))?"
    r"(?:\s+aggregate_imputed=(?P<aggregate_imputed>\d+))?"
    r"(?:\s+replay_violations=(?P<replay_violations>\d+))?"
    r"(?:\s+replay_orphans=(?P<replay_orphans>\d+))?"
    r"(?:\s+score_drift_flags=(?P<score_drift_flags>\d+))?"
    r"(?:\s+truncation_count=(?P<truncation_count>\d+))?"
    r"(?:\s+divergence_logged=(?P<divergence_logged>\d+))?"
    r"(?:\s+claude_sampled=(?P<claude_sampled>\d+))?"
    r"(?:\s+claude_calls_skipped=(?P<claude_calls_skipped>\d+))?"
    r"(?:\s+snippet_rejected=(?P<snippet_rejected>\d+))?"
    r"(?:\s+giveaway_rejected=(?P<giveaway_rejected>\d+))?"
    r"(?:\s+cost_cap_falls_back=(?P<cost_cap_falls_back>\d+))?"
    r"(?:\s+conformance_violations=(?P<conformance_violations>\d+))?"
    r"(?:\s+dedup_rejected=(?P<dedup_rejected>\d+))?"
    r"(?:\s+drift_flags=(?P<drift_flags>\d+))?"
    r"(?:\s+event_clusters=(?P<event_clusters>\d+))?"
    r"(?:\s+email_triage_newsletter=(?P<email_triage_newsletter>\d+))?"
    r"(?:\s+email_triage_support=(?P<email_triage_support>\d+))?"
    r"(?:\s+email_triage_personal=(?P<email_triage_personal>\d+))?"
    r"(?:\s+email_triage_action=(?P<email_triage_action>\d+))?"
    r"(?:\s+cluster_propagated=(?P<cluster_propagated>\d+))?"
    r"(?:\s+profile_fallback=(?P<profile_fallback>true|false))?"
    r"(?:\s+drafts_pending=(?P<drafts_pending>\d+))?"
    r"\s*$"
)
_LOG_FAIL_RE = re.compile(
    r"^(?P<ts>\S+)\s*\|\s*FAIL\s*\|\s*error=(?P<error>.+)$"
)


def read_last_pipeline_run(log_path: Path) -> dict | None:
    """Read the last line of the pipeline log and parse it into a status dict.

    Returns None if log does not exist, is empty, or has a malformed last line.
    """
    try:
        if not log_path.exists():
            return None
        text = log_path.read_text(encoding="utf-8")
    except OSError:
        return None

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    last = lines[-1]
    m_ok = _LOG_OK_RE.match(last)
    if m_ok:
        try:
            return {
                "timestamp": m_ok.group("ts"),
                "status": "OK",
                "collected": int(m_ok.group("collected")),
                "curated": int(m_ok.group("curated")),
                "digest_state": m_ok.group("digest_state"),
                "tg_discovered": int(m_ok.group("tg_discovered") or 0),
                "ig_collected": int(m_ok.group("ig_collected") or 0),
                "cognition_tagged": int(m_ok.group("cognition_tagged") or 0),
                "gaming_q": int(m_ok.group("gaming_q") or 0),
                "health_days": int(m_ok.group("health_days") or 0),
                "dataset_size": int(m_ok.group("dataset_size") or 0),
                "aggregate_imputed": int(m_ok.group("aggregate_imputed") or 0),
                "replay_violations": int(m_ok.group("replay_violations") or 0),
                "replay_orphans": int(m_ok.group("replay_orphans") or 0),
                "divergence_logged": int(m_ok.group("divergence_logged") or 0),
                "claude_sampled": int(m_ok.group("claude_sampled") or 0),
                "claude_calls_skipped": int(m_ok.group("claude_calls_skipped") or 0),
                "snippet_rejected": int(m_ok.group("snippet_rejected") or 0),
                "giveaway_rejected": int(m_ok.group("giveaway_rejected") or 0),
                "cost_cap_falls_back": int(m_ok.group("cost_cap_falls_back") or 0),
                "conformance_violations": int(m_ok.group("conformance_violations") or 0),
                "dedup_rejected": int(m_ok.group("dedup_rejected") or 0),
                "drift_flags": int(m_ok.group("drift_flags") or 0),
                "event_clusters": int(m_ok.group("event_clusters") or 0),
                "email_triage_newsletter": int(m_ok.group("email_triage_newsletter") or 0),
                "email_triage_support": int(m_ok.group("email_triage_support") or 0),
                "email_triage_personal": int(m_ok.group("email_triage_personal") or 0),
                "email_triage_action": int(m_ok.group("email_triage_action") or 0),
                "cluster_propagated": int(m_ok.group("cluster_propagated") or 0),
                "profile_fallback": (m_ok.group("profile_fallback") == "true"),
                "drafts_pending": int(m_ok.group("drafts_pending") or 0),
            }
        except (ValueError, KeyError):
            return None
    m_fail = _LOG_FAIL_RE.match(last)
    if m_fail:
        return {
            "timestamp": m_fail.group("ts"),
            "status": "FAIL",
            "error": m_fail.group("error").strip(),
        }
    return None


# ── Button + badge injection helpers (copied verbatim from main.py) ─────────

_POST_ID_MARKER_RE = re.compile(r"<!--\s*post-id:\s*(\d+)\s*-->")
_REACT_ITEM_ID_MARKER_RE = re.compile(r"<!--\s*item-id:\s*(\d+)\s*-->")

_REACT_BUTTONS: tuple[tuple[str, str], ...] = (
    ("brainstorm", "\U0001F9E0"),
    ("docs",       "\U0001F4C4"),
    ("link",       "\U0001F517"),
    ("finish",     "\U0001F3C1"),
    ("debate",     "⚔️"),
    ("skip",       "❌"),
)

_SALIENCY_COLORS = {
    ("novelty", "high"):   ("badge-sal-novelty-high",   "★ novel"),
    ("novelty", "medium"): ("badge-sal-novelty-med",    "◑ novel"),
    ("relevance", "high"):   ("badge-sal-rel-high",   "● relevant"),
    ("relevance", "medium"): ("badge-sal-rel-med",    "◑ relevant"),
    ("urgency", "today"):  ("badge-sal-urgent",  "⚡ today"),
    ("urgency", "recent"): ("badge-sal-recent",  "○ recent"),
}


def _format_saliency_badges(sal: dict) -> str:
    parts = []
    for dim in ("novelty", "relevance", "urgency"):
        val = sal.get(dim)
        if val and val not in ("low", "stale"):
            key = (dim, val)
            if key in _SALIENCY_COLORS:
                css, label = _SALIENCY_COLORS[key]
                parts.append(f'<span class="badge-sal {css}">{label}</span>')
    return " ".join(parts)


def _inject_buttons(
    html: str,
    feedback_by_post: dict[int, int] | None = None,
    suspect_post_ids: set[int] | None = None,
) -> str:
    """Replace each `<!-- post-id:N -->` marker with an inline vote-button block."""
    fb = feedback_by_post or {}
    suspect = suspect_post_ids or set()

    def replace(match: re.Match) -> str:
        post_id_str = match.group(1)
        try:
            post_id_int = int(post_id_str)
        except ValueError:
            post_id_int = None
        rating = fb.get(post_id_int) if post_id_int is not None else None

        up_classes = "vote-up"
        down_classes = "vote-down"
        if rating == 1:
            up_classes += " voted"
        elif rating == -1:
            down_classes += " voted"

        suspect_badge = ""
        if post_id_int is not None and post_id_int in suspect:
            suspect_badge = (
                '<span class="badge-suspect" title="Low-quality justification detected">'
                "&#9888; suspect</span>"
            )

        return (
            f"{suspect_badge}"
            f'<span class="vote-buttons" data-post-id="{post_id_str}">'
            f'<button class="{up_classes}" type="button" aria-label="upvote" '
            f'onclick="voteFeedback({post_id_str}, 1, this)">\U0001F44D</button>'
            f'<button class="{down_classes}" type="button" aria-label="downvote" '
            f'onclick="voteFeedback({post_id_str}, -1, this)">\U0001F44E</button>'
            f"</span>"
        )

    return _POST_ID_MARKER_RE.sub(replace, html)


def _inject_react_buttons(
    html: str,
    saliency_by_item: dict[int, dict] | None = None,
    paper_dois: dict[int, str] | None = None,
) -> str:
    """Replace each `<!-- item-id:N -->` marker with a react-row div."""
    import urllib.parse

    sal_map = saliency_by_item or {}
    pdois = paper_dois or {}

    def replace(match: re.Match) -> str:
        item_id_str = match.group(1)
        buttons_html = "".join(
            f'<button class="react-btn react-btn--{action}" '
            f'type="button" data-action="{action}" '
            f'aria-label="{action}" '
            f"onclick=\"reactClick({item_id_str}, '{action}', this)\">"
            f"{emoji}</button>"
            for action, emoji in _REACT_BUTTONS
        )
        sal_html = ""
        try:
            sal = sal_map.get(int(item_id_str))
            if sal:
                badges = _format_saliency_badges(sal)
                if badges:
                    sal_html = f'<span class="sal-badges">{badges}</span>'
        except (ValueError, TypeError):
            pass

        paper_html = ""
        try:
            doi = pdois.get(int(item_id_str))
            if doi:
                graph_url = "/graph?doi=" + urllib.parse.quote(doi, safe="")
                paper_html = (
                    f'<span class="paper-btns">'
                    f'<a class="paper-btn" href="{graph_url}" target="_blank" title="Citation graph">\U0001F4CA</a>'
                    f'<button class="paper-btn" onclick="paperSimilar({item_id_str}, this)" title="Find similar papers">\U0001F50D</button>'
                    f'<button class="paper-btn" onclick="paperTldr({item_id_str}, this)" title="Generate TLDR summary">\U0001F4DD</button>'
                    f'</span>'
                    f'<div id="paper-out-{item_id_str}" class="paper-out"></div>'
                )
        except (ValueError, TypeError):
            pass

        return (
            f'<div class="react-row" data-item-id="{item_id_str}">'
            f"{buttons_html}"
            f"{sal_html}"
            f"{paper_html}"
            f"</div>"
        )

    return _REACT_ITEM_ID_MARKER_RE.sub(replace, html)


_EXTRACT_STATUS_CLASS_MAP = {
    "success": "badge-extract-success",
    "blocked": "badge-extract-blocked",
    "timeout": "badge-extract-timeout",
    "http_error": "badge-extract-http-error",
    "parse_error": "badge-extract-parse-error",
}


def _inject_extraction(html: str, extraction_by_post: dict[int, dict]) -> str:
    """Inject extraction badge + collapsible body panel at each `<!-- post-id:N -->` marker."""
    import html as html_module

    def replace(match: re.Match) -> str:
        post_id_str = match.group(1)
        try:
            post_id = int(post_id_str)
        except ValueError:
            return match.group(0)
        data = extraction_by_post.get(post_id)
        if data is None:
            return match.group(0)
        status = data.get("extraction_status", "")
        body = data.get("extracted_body") or ""
        status_class = _EXTRACT_STATUS_CLASS_MAP.get(status, "badge-extract-parse-error")
        injection = (
            f'<span class="badge-extract {status_class}">'
            f"{html_module.escape(status)}</span>"
        )
        if body:
            escaped_body = html_module.escape(body)
            injection += (
                f' <span class="extract-toggle" onclick="toggleExtract({post_id})">'
                f"show body</span>"
                f'<span id="extract-body-{post_id}" class="extract-body">'
                f"{escaped_body}</span>"
            )
        return match.group(0) + injection

    return _POST_ID_MARKER_RE.sub(replace, html)


_PARA_RE = re.compile(r"<p>(.*?)</p>", re.DOTALL)
_BARE_REACT_ROW_RE = re.compile(
    r'<div class="react-row"[^>]*>.*?</div>(?:\s*<div id="paper-out-\d+"[^>]*></div>)?',
    re.DOTALL,
)
_BARE_VOTE_BTNS_RE = re.compile(
    r'<span class="vote-buttons"[^>]*>.*?</span>',
    re.DOTALL,
)


def _hide_orphan_button_paras(html: str) -> str:
    """Hide orphan button blocks without associated entry text."""

    def replace_para(m: re.Match) -> str:
        body = m.group(1)
        has_buttons = 'class="react-row"' in body or 'class="vote-buttons"' in body
        has_text = "<strong>" in body
        if has_buttons and not has_text:
            return f'<p style="display:none">{body}</p>'
        return m.group(0)

    html = _PARA_RE.sub(replace_para, html)

    def _hide_if_orphan(pattern: re.Pattern, html_in: str) -> str:
        out = []
        last_end = 0
        for m in pattern.finditer(html_in):
            block = m.group(0)
            window = html_in[max(0, m.start() - 8000) : m.start()]
            anchors = []
            for token in (
                "<p>",
                '<p style="display:none">',
                "</p>",
                "<h2>",
                "</h2>",
                "<h3>",
                "</h3>",
            ):
                idx = window.rfind(token)
                if idx >= 0:
                    anchors.append((idx, token))
            if anchors:
                anchors.sort()
                last_idx, last_tok = anchors[-1]
                ctx = window[last_idx + len(last_tok) :]
            else:
                ctx = window
            if "<strong>" not in ctx:
                out.append(html_in[last_end : m.start()])
                out.append(f'<span style="display:none">{block}</span>')
                last_end = m.end()
        out.append(html_in[last_end:])
        return "".join(out)

    html = _hide_if_orphan(_BARE_REACT_ROW_RE, html)
    html = _hide_if_orphan(_BARE_VOTE_BTNS_RE, html)
    return html


# ── Routes ──────────────────────────────────────────────────────────────────


@app.get("/")
async def read_root(request: Request, digest_id: int = 0):
    db = await open_db(_get_db_path())
    saliency_by_item: dict[int, dict] = {}
    paper_dois: dict[int, str] = {}
    digest_nav: dict | None = None
    try:
        if digest_id:
            digest = await get_digest_by_id(db, digest_id)
            digests = [digest] if digest else []
        else:
            digests = await get_digests(db)
        extraction_by_post = await get_posts_with_extraction(db)
        feedback_by_post = await get_all_feedback(db)
        suspect_post_ids = await get_suspect_post_ids(db)
        if digests:
            current_id = digests[0]["id"]
            try:
                saliency_by_item = await get_saliency_by_item(db, current_id)
            except Exception:
                pass
            try:
                paper_dois = await get_paper_dois_for_digest(db, current_id)
            except Exception:
                pass
            try:
                digest_nav = await get_digest_nav(db, current_id)
            except Exception:
                pass
    finally:
        await db.close()

    for d in digests:
        rendered = markdown.markdown(d["markdown_content"])
        rendered = _inject_extraction(rendered, extraction_by_post)
        rendered = _inject_react_buttons(rendered, saliency_by_item, paper_dois)
        rendered = _inject_buttons(rendered, feedback_by_post, suspect_post_ids)
        rendered = _hide_orphan_button_paras(rendered)
        d["html_content"] = rendered

    last_run = read_last_pipeline_run(LOG_PATH)

    try:
        gate_status = get_gate_status()
    except Exception:
        gate_status = None

    try:
        from src.agents.voice_digest import get_audio_url_for_date

        audio_url = get_audio_url_for_date()
    except Exception:
        audio_url = None

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "digests": digests,
            "last_run": last_run,
            "extraction_by_post": extraction_by_post,
            "gate_status": gate_status,
            "audio_url": audio_url,
            "digest_nav": digest_nav,
        },
    )


class FeedbackPayload(BaseModel):
    """POST /feedback payload. rating must be -1 or 1."""

    post_id: int
    rating: Literal[-1, 1]


@app.post("/feedback")
async def submit_feedback(payload: FeedbackPayload):
    """Persist a single feedback row. Returns 200 on success, 404 if post_id missing."""
    db = await open_db(_get_db_path())
    try:
        async with db.execute(
            "SELECT 1 FROM raw_posts WHERE id = ?", (payload.post_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"post_id {payload.post_id} not found in raw_posts",
            )
        try:
            await record_feedback(db, payload.post_id, payload.rating)
        except sqlite3.IntegrityError as e:
            raise HTTPException(status_code=400, detail=str(e))
    finally:
        await db.close()
    return {"status": "ok"}


_SEARCH_NAV = (
    '<nav style="margin-bottom:16px;padding:8px 0;border-bottom:1px solid #ddd;font-size:.9rem">'
    '<a href="/" style="color:#1a73e8;text-decoration:none;margin-right:20px">&#128240; Feed</a>'
    '<a href="/search" style="color:#1a73e8;text-decoration:none;margin-right:20px">&#128269; Search</a>'
    "</nav>"
)

_SEARCH_FORM = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Search</title>
<style>body{{max-width:800px;margin:40px auto;font-family:system-ui,sans-serif;padding:0 20px;line-height:1.6}}
input[type=text]{{width:100%;padding:10px;font-size:1rem;border:1px solid #ccc;border-radius:4px;box-sizing:border-box;margin-top:8px}}
button{{margin-top:8px;padding:8px 20px;font-size:1rem;background:#1a73e8;color:#fff;border:none;border-radius:4px;cursor:pointer}}
.result{{border-bottom:1px solid #eee;padding:12px 0}}.score{{color:#888;font-size:.8rem}}.source{{font-size:.8rem;color:#555}}
a{{color:#1a73e8}}</style></head><body>
{{nav}}
<h1 style="font-size:1.3rem">Search digests</h1>
<form action="/search" method="get">
<input type="text" name="q" value="{{q}}" placeholder="Search, e.g. AI tools for literature review" autofocus>
<button type="submit">Search</button>
</form>
{{results}}
</body></html>"""


@app.get("/search")
async def search_endpoint(request: Request, q: str = "", limit: int = 10):
    q = q.strip()[:500]
    safe_q = _html.escape(q)

    if not q:
        return HTMLResponse(_SEARCH_FORM.format(nav=_SEARCH_NAV, q="", results=""))

    limit = min(limit, 50)

    try:
        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        from pathlib import Path as _Path

        _chroma_dir = _Path(__file__).resolve().parents[2] / ".chromadb"
        _client = chromadb.PersistentClient(path=str(_chroma_dir))
        _col = _client.get_or_create_collection(
            "notebook", embedding_function=DefaultEmbeddingFunction()
        )
        if _col.count() == 0:
            results_html = "<p style='color:#888;margin-top:16px'>Index is empty — run the pipeline first.</p>"
        else:
            _res = _col.query(query_texts=[q], n_results=min(limit, _col.count()))
            docs = _res["documents"][0]
            metas = _res["metadatas"][0]
            import re as _re

            _url_re = _re.compile(r"https?://[^\s)]+")
            items = []
            for doc, meta in zip(docs, metas):
                source = meta.get("file", "unknown")
                url_m = _url_re.search(doc)
                url_html = (
                    f'<a href="{url_m.group(0)}" target="_blank">{url_m.group(0)}</a>'
                    if url_m
                    else ""
                )
                items.append(
                    f'<div class="result">'
                    f'<div class="source">\U0001F4C1 {source}</div>'
                    f'<p>{doc[:300]}{"..." if len(doc) > 300 else ""}</p>'
                    f"{url_html}</div>"
                )
            results_html = (
                f"<p style='color:#888;margin-top:16px'>{len(docs)} results for «{safe_q}»</p>"
                + "".join(items)
            )
    except Exception as e:
        logger.error("ChromaDB search failed: %s", e)
        results_html = f"<p style='color:#c62828'>Search error: {_html.escape(str(e))}</p>"

    return HTMLResponse(_SEARCH_FORM.format(nav=_SEARCH_NAV, q=safe_q, results=results_html))


# ── Draft Pipeline Automation: status API ───────────────────────────────────


class DraftStatusPayload(BaseModel):
    state: Literal[
        "preliminary", "research", "execute", "research_execute", "archived"
    ]


@app.get("/api/drafts")
async def get_drafts():
    """Return pending drafts keyed by item_id for the frontend dropdown."""
    loop = asyncio.get_running_loop()
    try:
        drafts = await loop.run_in_executor(None, list_pending)
    except OSError as e:
        logger.warning("get_drafts: vault offline: %s", e)
        return {}
    result = {}
    for d in drafts:
        if d.get("item_id") is None:
            logger.warning(
                "get_drafts: draft reaction_id=%s has no item_id, skipping",
                d.get("reaction_id"),
            )
            continue
        result[str(d["item_id"])] = {
            "reaction_id": d["reaction_id"],
            "gate_state": d["gate_state"],
            "action": d["action"],
            "source_name": d["source_name"],
        }
    return result


@app.patch("/api/draft/{reaction_id}/status")
async def update_draft_status(reaction_id: int, payload: DraftStatusPayload):
    loop = asyncio.get_running_loop()
    try:
        found = await loop.run_in_executor(
            None, set_gate_state, reaction_id, payload.state
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not found:
        raise HTTPException(
            status_code=404, detail=f"Draft {reaction_id} not found"
        )
    return {"status": "ok"}


# ── Agent traces (generic observability) ────────────────────────────────────


@app.get("/api/traces")
async def list_traces(limit: int = 50):
    """List recent agent_traces grouped by run_id."""
    from src.observability.traces import list_recent_runs

    return {"runs": list_recent_runs(_get_db_path(), limit=limit)}


@app.get("/api/traces/{run_id}")
async def get_traces(run_id: str, limit: int = 500):
    """Return all trace rows for a run_id ordered by step_idx ASC."""
    from src.observability.traces import get_traces as _get_traces

    rows = _get_traces(_get_db_path(), run_id, limit)
    return {"run_id": run_id, "steps": rows, "count": len(rows)}
