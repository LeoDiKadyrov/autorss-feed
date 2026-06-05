"""
AcademicCollector — generic RSS/Atom collector for academic journals.

sources.txt format:
    https://www.nature.com/nature.rss | science | Nature | academic
    https://www.annualreviews.org/loi/statistics.rss | science | Annual Reviews Statistics | academic

Target field: full RSS feed URL.
External ID: DOI extracted from article link (regex); fallback to normalized link.
Cursor: max(published_at) ISO string — same pattern as ArxivCollector.
"""
import logging
import re

import aiosqlite
import feedparser

from src.collectors.base import BaseCollector
from src.database.client import insert_raw_post, update_source_last_cursor

logger = logging.getLogger(__name__)

MAX_ABSTRACT_CHARS = 3000
_DOI_RE = re.compile(r"10\.\d{4,}/[^\s\"'>]+")


def _extract_doi(entry) -> str | None:
    """Extract DOI from entry link, id, or doi fields."""
    candidates = [
        getattr(entry, "link", None) or "",
        getattr(entry, "id", None) or "",
    ]
    for dc in getattr(entry, "tags", []) or []:
        term = getattr(dc, "term", None) or ""
        candidates.append(term)
    for text in candidates:
        m = _DOI_RE.search(text)
        if m:
            doi = m.group(0).rstrip(".,;)")
            return doi
    return None


def _first_author(entry, fallback: str) -> str:
    authors = getattr(entry, "authors", None) or []
    if authors:
        first = authors[0]
        if isinstance(first, dict):
            return first.get("name", "") or fallback
        return getattr(first, "name", "") or fallback
    raw = getattr(entry, "author", None) or ""
    return raw.strip() or fallback


class AcademicCollector(BaseCollector):
    platform = "academic"

    async def collect(self, db: aiosqlite.Connection, source: dict) -> None:
        url = source["target"]
        cursor_str = source.get("last_cursor")

        parsed = feedparser.parse(url)
        if getattr(parsed, "bozo", False) and not getattr(parsed, "entries", None):
            logger.warning("feedparser bozo for academic source=%s — treating as empty", url)
            return

        collected_published: list[str] = []

        for entry in parsed.entries:
            published = (
                entry.get("published")
                or entry.get("updated")
                or entry.get("date")
            )
            if not published:
                continue
            if cursor_str is not None and published <= cursor_str:
                continue

            doi = _extract_doi(entry)
            link = getattr(entry, "link", None) or ""
            external_id = doi or link
            if not external_id:
                continue

            title = (getattr(entry, "title", "") or "").replace("\n", " ").strip()
            abstract = (getattr(entry, "summary", "") or "")[:MAX_ABSTRACT_CHARS]
            raw_text = title + "\n\n" + abstract if abstract else title

            if len(raw_text) < 100:
                continue

            author = _first_author(entry, fallback=source.get("display_name", url))

            await insert_raw_post(
                db,
                source["id"],
                external_id,
                raw_text,
                author,
                link or None,
                platform="academic",
                content_type="paper",
                published_at=published,
                source_score=0,
            )
            collected_published.append(published)

        if collected_published:
            await update_source_last_cursor(db, source["id"], max(collected_published))
