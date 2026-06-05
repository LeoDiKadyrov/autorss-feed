"""
BiorxivCollector — RSS collector for bioRxiv and medRxiv preprints.

sources.txt format:
    biophysics | science | bioRxiv Biophysics | biorxiv
    neuroscience | science | bioRxiv Neuroscience | biorxiv
    medrxiv:psychiatry | science | medRxiv Psychiatry | biorxiv

Target field:
    Plain string (no prefix) → bioRxiv: https://connect.biorxiv.org/biorxiv_xml.php?subject={target}
    "medrxiv:{subject}"     → medRxiv: https://connect.medrxiv.org/medrxiv_xml.php?subject={subject}

External ID: DOI extracted from article link.
Cursor: max(published_at) ISO string.
"""
import logging
import re

import aiosqlite
import feedparser

from src.collectors.base import BaseCollector
from src.database.client import insert_raw_post, update_source_last_cursor

logger = logging.getLogger(__name__)

MAX_ABSTRACT_CHARS = 2000
_DOI_RE = re.compile(r"10\.\d{4,}/[^\s\"'>]+")
_BIORXIV_BASE = "https://connect.biorxiv.org/biorxiv_xml.php?subject={subject}"
_MEDRXIV_BASE = "https://connect.medrxiv.org/medrxiv_xml.php?subject={subject}"


def _build_url(target: str) -> str:
    if target.startswith("medrxiv:"):
        subject = target[len("medrxiv:"):]
        return _MEDRXIV_BASE.format(subject=subject)
    return _BIORXIV_BASE.format(subject=target)


def _extract_doi(link: str) -> str | None:
    m = _DOI_RE.search(link)
    if m:
        doi = m.group(0).rstrip(".,;)")
        # Strip biorxiv/medrxiv version suffix: v1, v2, etc. + optional ?rss=1
        doi = re.sub(r"v\d+(\?.*)?$", "", doi)
        return doi or None
    return None


class BiorxivCollector(BaseCollector):
    platform = "biorxiv"

    async def collect(self, db: aiosqlite.Connection, source: dict) -> None:
        target = source["target"]
        cursor_str = source.get("last_cursor")
        feed_url = _build_url(target)

        parsed = feedparser.parse(feed_url)
        if getattr(parsed, "bozo", False) and not getattr(parsed, "entries", None):
            logger.warning("feedparser bozo for biorxiv target=%s — treating as empty", target)
            return

        collected_published: list[str] = []

        for entry in parsed.entries:
            published = (
                entry.get("published")
                or entry.get("updated")
                or entry.get("date")
                or entry.get("prism_publicationdate")
            )
            if not published:
                continue
            if cursor_str is not None and published <= cursor_str:
                continue

            link = getattr(entry, "link", None) or ""
            doi = _extract_doi(link)
            external_id = doi or link
            if not external_id:
                continue

            title = (getattr(entry, "title", "") or "").replace("\n", " ").strip()
            abstract = (getattr(entry, "summary", "") or "")[:MAX_ABSTRACT_CHARS]
            raw_text = title + "\n\n" + abstract if abstract else title

            if len(raw_text) < 100:
                continue

            author_raw = getattr(entry, "author", None) or ""
            author = author_raw.strip() or source.get("display_name", target)

            await insert_raw_post(
                db,
                source["id"],
                external_id,
                raw_text,
                author,
                link or None,
                platform="biorxiv",
                content_type="paper",
                published_at=published,
                source_score=0,
            )
            collected_published.append(published)

        if collected_published:
            await update_source_last_cursor(db, source["id"], max(collected_published))
