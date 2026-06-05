"""
Reddit collector — public RSS path (2026-05-15 rewrite).

Previously used asyncpraw with OAuth2 script-app credentials. Reddit's
Responsible Builder Policy (2024+) now requires ticket-approval before
issuing API credentials, breaking the auto-provisioning flow.

This rewrite uses Reddit's public RSS endpoints (no auth, no credentials):
  https://www.reddit.com/r/{name}/.rss

Trade-offs vs OAuth path:
- No score data — drop score>=50 filter.
- No selftext separation — content comes as HTML in <content> tag.
- Max ~25 posts per fetch (RSS default).
- Stricter rate limits without auth (~60 req/min); mitigated by custom User-Agent.

Required: custom User-Agent header. Reddit blocks default urllib/requests UAs.
"""
import asyncio
import datetime
import logging
import os
from html import unescape
from re import sub as re_sub

import aiohttp
import aiosqlite
import feedparser

from src.collectors.base import BaseCollector
from src.database.client import insert_raw_post, update_source_last_cursor
from src.extract import extract_url
from src.extract.triggers import is_http_url

logger = logging.getLogger(__name__)


def _strip_html(html: str) -> str:
    """Best-effort HTML→text for RSS content. Removes tags, unescapes entities."""
    if not html:
        return ""
    text = re_sub(r"<[^>]+>", " ", html)
    text = unescape(text)
    text = re_sub(r"\s+", " ", text).strip()
    return text


class RedditCollector(BaseCollector):
    platform = "reddit"

    def __init__(self, extract_session=None, extract_semaphore=None):
        # No credentials needed — public RSS. Read User-Agent at construction
        # time per D-03 module-level env binding rule (kept for consistency
        # with the old shape and for future env-driven tuning).
        self._user_agent = os.environ.get(
            "REDDIT_USER_AGENT", "personal-curator/1.0"
        )
        # Phase 10 EXTRACT-01: pipeline-scoped aiohttp session + Semaphore(5).
        self._extract_session = extract_session
        self._extract_semaphore = extract_semaphore

    async def collect(self, db: aiosqlite.Connection, source: dict) -> None:
        cursor_str = source.get("last_cursor")
        # Cursor format changed from float UTC to ISO datetime. Migrate
        # transparently: if old float-shaped cursor is encountered, parse it
        # as a UTC timestamp.
        last_iso: str | None = None
        if cursor_str:
            try:
                # New format: ISO 8601 string
                datetime.datetime.fromisoformat(cursor_str.replace("Z", "+00:00"))
                last_iso = cursor_str
            except ValueError:
                # Old format: float UTC timestamp string
                try:
                    last_iso = datetime.datetime.fromtimestamp(
                        float(cursor_str), tz=datetime.UTC
                    ).isoformat()
                except ValueError:
                    logger.warning(
                        "Invalid last_cursor %r for source %s — resetting to None",
                        cursor_str,
                        source["id"],
                    )

        # Strip "r/" prefix for URL building.
        subreddit_name = source["target"].lstrip("r/")
        rss_url = f"https://www.reddit.com/r/{subreddit_name}/.rss"

        # Fetch RSS with custom User-Agent (Reddit blocks default UAs).
        headers = {"User-Agent": self._user_agent}
        timeout = aiohttp.ClientTimeout(total=15)
        try:
            async with aiohttp.ClientSession(
                timeout=timeout, headers=headers
            ) as session:
                async with session.get(rss_url) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "Reddit RSS for %s returned HTTP %s — skipping this run",
                            subreddit_name,
                            resp.status,
                        )
                        return
                    body = await resp.text()
        except aiohttp.ClientError as e:
            logger.warning(
                "Reddit RSS fetch failed for %s: %s — skipping this run",
                subreddit_name,
                type(e).__name__,
            )
            return

        parsed = feedparser.parse(body)
        if parsed.bozo and not parsed.entries:
            logger.warning(
                "Reddit RSS for %s did not parse (bozo, 0 entries) — skipping",
                subreddit_name,
            )
            return

        collected_isos: list[str] = []
        for entry in parsed.entries:
            # Entry shape (Atom): id, title, link, updated, content[0].value, author
            published_iso = entry.get("updated") or entry.get("published")
            if not published_iso:
                continue

            # Cursor filter: skip already-seen posts.
            if last_iso is not None and published_iso <= last_iso:
                continue

            external_id = entry.get("id", "")
            # Atom id format: "/r/MachineLearning/comments/abc123/title/" or full URL.
            # Derive post id from URL path.
            link = entry.get("link", "")
            if not link:
                continue

            title = entry.get("title", "") or ""
            content_raw = ""
            if entry.get("content"):
                content_raw = entry.content[0].get("value", "")
            elif entry.get("summary"):
                content_raw = entry.summary

            # Reddit anti-bot fallback: RSS <content> contains a full HTML page
            # instead of post text. Strip tags → >100 chars of JS noise that
            # passes the length gate but has no substance. Detect and discard.
            stripped_raw = content_raw.lstrip()
            if stripped_raw.startswith("<!DOCTYPE") or stripped_raw.lower().startswith("<html"):
                content_raw = ""
            content_text = _strip_html(content_raw)
            author = entry.get("author", "") or ""

            # Determine if this is a self-post or link-post.
            # RSS content for link posts typically contains "[link]" anchor +
            # short description. Heuristic: if content text is very short and
            # there's an external link inside, treat as link post.
            extracted_body: str | None = None
            extraction_status: str | None = None
            extracted_at: str | None = None

            # raw_text: prefer title + content if content adds substance.
            if content_text and len(content_text) > 50:
                raw_text = f"{title}\n\n{content_text}"
            else:
                raw_text = title

            # If raw_text is short and we have an extract pipeline + the
            # outbound link in `link` is a Reddit comment URL (not external),
            # we can't extract usefully. RSS doesn't expose the original
            # external URL separately; skip extraction for now.
            if len(raw_text) < 100:
                # Same gate as old collector — drop short posts.
                continue

            await insert_raw_post(
                db,
                source["id"],
                external_id or link,
                raw_text,
                author,
                link,
                platform="reddit",
                content_type="post",
                published_at=published_iso,
                source_score=0,  # RSS does not expose score; 0 = neutral default
                extracted_body=extracted_body,
                extraction_status=extraction_status,
                extracted_at=extracted_at,
            )
            collected_isos.append(published_iso)

        if collected_isos:
            await update_source_last_cursor(
                db, source["id"], max(collected_isos)
            )

        # Polite delay between subreddit fetches to stay under Reddit's
        # public-IP rate limit (~60 req/min anonymous).
        await asyncio.sleep(1.0)
