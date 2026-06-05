"""
ArxivCollector — fetches recent preprints from a single ArXiv category via the
public Atom API. No credentials required.

API endpoint: https://export.arxiv.org/api/query?search_query=cat:{category}
              &sortBy=submittedDate&sortOrder=descending&max_results=50

sources.txt format:
    cs.AI | ai | ArXiv cs.AI | arxiv
    cs.NE | science | ArXiv cs.NE | arxiv

Cursor: last_cursor = max(published_at) ISO string — same pattern as YouTubeCollector.
"""
import logging
import re

import aiosqlite
import feedparser

from src.collectors.base import BaseCollector
from src.database.client import insert_raw_post, update_source_last_cursor

logger = logging.getLogger(__name__)

ARXIV_API_URL = (
    "https://export.arxiv.org/api/query"
    "?search_query=cat:{category}"
    "&sortBy=submittedDate&sortOrder=descending&max_results=50"
)

MAX_ABSTRACT_CHARS = 2000
_VERSION_RE = re.compile(r"v\d+$")


def _extract_arxiv_id(entry_id: str) -> str:
    """
    Extract paper ID without version from entry.id.
    Input:  'http://arxiv.org/abs/2401.12345v1'
    Output: '2401.12345'
    """
    raw = entry_id.split("/abs/")[-1]
    return _VERSION_RE.sub("", raw)


def _first_author(entry, fallback: str) -> str:
    """Return first author name or fallback to source display_name."""
    authors = getattr(entry, "authors", None) or []
    if authors:
        first = authors[0]
        if isinstance(first, dict):
            return first.get("name", "") or fallback
        return getattr(first, "name", "") or fallback
    raw = getattr(entry, "author", None) or ""
    return raw.strip() or fallback


class ArxivCollector(BaseCollector):
    platform = "arxiv"

    async def collect(self, db: aiosqlite.Connection, source: dict) -> None:
        category = source["target"]
        cursor_str = source.get("last_cursor")
        url = ARXIV_API_URL.format(category=category)

        parsed = feedparser.parse(url)
        if getattr(parsed, "bozo", False) and not getattr(parsed, "entries", None):
            logger.warning(
                "feedparser bozo for arxiv category=%s — treating as empty feed",
                category,
            )
            return

        collected_published: list[str] = []

        for entry in parsed.entries:
            published = getattr(entry, "published", None)
            if not published:
                continue

            # Cursor skip — lexicographic ISO comparison (same timezone offset).
            if cursor_str is not None and published <= cursor_str:
                continue

            arxiv_id = _extract_arxiv_id(getattr(entry, "id", "") or "")
            if not arxiv_id:
                logger.warning(
                    "Skipping entry without parseable arxiv ID in category=%s", category
                )
                continue

            title = (getattr(entry, "title", "") or "").replace("\n", " ").strip()
            abstract = (getattr(entry, "summary", "") or "")[:MAX_ABSTRACT_CHARS]
            raw_text = title + "\n\n" + abstract

            # Pre-filter: skip if combined length < 100 chars (same as all collectors).
            if len(raw_text) < 100:
                continue

            author = _first_author(entry, fallback=source.get("display_name", category))
            paper_url = f"https://arxiv.org/abs/{arxiv_id}"

            await insert_raw_post(
                db,
                source["id"],
                arxiv_id,
                raw_text,
                author,
                paper_url,
                platform="arxiv",
                content_type="paper",
                published_at=published,
                source_score=0,
            )
            collected_published.append(published)

        if collected_published:
            await update_source_last_cursor(db, source["id"], max(collected_published))
