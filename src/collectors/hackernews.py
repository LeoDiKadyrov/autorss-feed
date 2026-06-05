"""
HackernewsCollector — fetches recent stories from the free Algolia HN Search API.

No credentials required. Mirrors the arxiv.py / biorxiv.py shape:
cursor-based, fail-soft on HTTP errors, per-source last_cursor.

API endpoint:
    https://hn.algolia.com/api/v1/search_by_date?tags=story&hitsPerPage=50

sources.txt format:
    topstories | other | Hacker News | hackernews

Cursor:
    sources.last_cursor = max(created_at_i) as string (unix epoch).
    On subsequent runs the URL gets &numericFilters=created_at_i>{cursor}.

Score floor:
    HN_MIN_POINTS env (default 100). Read INSIDE collect() per the
    module-level env binding gotcha (reference_module_level_env_binding.md).

Rate limit:
    await asyncio.sleep(1) between paged requests (non-blocking; BL-01).

Pagination cap:
    5 pages per run (Algolia returns nbPages; we cap defensively).
"""
import asyncio
import logging
import os
from datetime import datetime, timezone

import aiosqlite
import requests

from src.collectors.base import BaseCollector
from src.database.client import insert_raw_post, update_source_last_cursor

logger = logging.getLogger(__name__)

HN_API_URL = (
    "https://hn.algolia.com/api/v1/search_by_date"
    "?tags=story&hitsPerPage=50"
)
MAX_PAGES = 5
MAX_TEXT_CHARS = 2000


class HackernewsCollector(BaseCollector):
    platform = "hackernews"

    async def collect(self, db: aiosqlite.Connection, source: dict) -> None:
        # Read env INSIDE function — module-level binding gotcha.
        try:
            min_points = int(os.environ.get("HN_MIN_POINTS", "100"))
        except ValueError:
            logger.warning(
                "HN_MIN_POINTS is not an int — defaulting to 100"
            )
            min_points = 100

        cursor_str = source.get("last_cursor")
        base_url = HN_API_URL
        if cursor_str:
            # WR-02: coerce cursor to int and URL-encode the '>' operator.
            # sources.last_cursor is operator-mutable TEXT; reject non-int values
            # rather than interpolate raw into the query string.
            try:
                cursor_int = int(cursor_str)
                base_url = (
                    f"{base_url}&numericFilters=created_at_i%3E{cursor_int}"
                )
            except (TypeError, ValueError):
                logger.warning(
                    "HN cursor not int for source id=%s: %r — ignoring cursor",
                    source.get("id"), cursor_str,
                )

        collected_created: list[int] = []

        page = 0
        nb_pages = 1
        while page < min(nb_pages, MAX_PAGES):
            if page > 0:
                await asyncio.sleep(1)  # rate-limit between paged requests (non-blocking)

            url = f"{base_url}&page={page}" if page > 0 else base_url
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                logger.warning(
                    "HN Algolia fetch failed for source id=%s target=%s page=%d: %s",
                    source.get("id"), source.get("target"), page, exc,
                )
                # WR-03: break (don't return) so successfully-collected pages'
                # cursor is still persisted in the trailing update below.
                break

            hits = data.get("hits", []) or []
            nb_pages = int(data.get("nbPages", 1) or 1)

            for hit in hits:
                points = int(hit.get("points") or 0)
                if points < min_points:
                    continue

                object_id = str(hit.get("objectID") or "")
                if not object_id:
                    continue
                # WR-01: namespace external_id to avoid global UNIQUE collisions
                # with numeric IDs from other platforms (reddit/telegram/biorxiv).
                external_id = f"hn:{object_id}"

                created_at_i = hit.get("created_at_i")
                if created_at_i is None:
                    continue
                try:
                    created_at_i = int(created_at_i)
                except (TypeError, ValueError):
                    continue

                title = (hit.get("title") or "").replace("\n", " ").strip()
                hit_url = hit.get("url") or ""
                story_text = (hit.get("story_text") or "")[:MAX_TEXT_CHARS]

                raw_text = f"{title}\n\n{hit_url}\n\n{story_text}".strip()

                # Pre-filter <100 chars (curator drops these anyway).
                if len(raw_text) < 100:
                    continue

                author = (hit.get("author") or "").strip() or \
                    source.get("display_name", "Hacker News")
                url_for_post = hit_url or \
                    f"https://news.ycombinator.com/item?id={object_id}"

                published_at = datetime.fromtimestamp(
                    created_at_i, tz=timezone.utc
                ).isoformat()

                try:
                    await insert_raw_post(
                        db,
                        source["id"],
                        external_id,
                        raw_text,
                        author,
                        url_for_post,
                        platform="hackernews",
                        content_type="article",
                        published_at=published_at,
                        source_score=points,
                    )
                except Exception:
                    logger.exception(
                        "insert_raw_post failed for HN objectID=%s — continuing",
                        object_id,
                    )
                    continue

                collected_created.append(created_at_i)

            page += 1

        if collected_created:
            await update_source_last_cursor(
                db, source["id"], str(max(collected_created))
            )
