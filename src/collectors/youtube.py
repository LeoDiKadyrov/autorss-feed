"""
YouTubeCollector — fetches latest videos from a YouTube channel via RSS,
downloads transcripts (or falls back to description), and writes to raw_posts.

No API key required. RSS endpoint: https://www.youtube.com/feeds/videos.xml?channel_id={ID}
Channel ID is the raw UCxxxx string stored as `target` in the sources table (D-01).

Patch-target contract locked in plan 03-01 (see 03-01-SUMMARY.md "Patch-Target Contracts"):
- `import feedparser` at module top — tests patch `src.collectors.youtube.feedparser.parse`.
- `from youtube_transcript_api import YouTubeTranscriptApi` at module top, then call
  `YouTubeTranscriptApi().fetch(video_id, languages=['en'])` — tests patch
  `src.collectors.youtube.YouTubeTranscriptApi.fetch` (instance method).
  youtube-transcript-api 1.2.4 dropped the older `get_transcript` classmethod;
  `fetch` is the supported instance method. The mock returns a list of dicts
  (`{"text": str, "start": float, "duration": float}`) which the implementation
  iterates; in production the real library returns a `FetchedTranscript` object
  whose snippets expose either dict-style access or `.text` attribute — both shapes
  are handled by `_text_of_chunk()` for forward compatibility.

References: D-01..D-28 in .planning/phases/03-youtube-collector/03-CONTEXT.md
"""
import datetime
import logging
from typing import Optional

import aiosqlite
import feedparser
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

from src.collectors.base import BaseCollector
from src.database.client import insert_raw_post, update_source_last_cursor
from src.extract import extract_url
from src.extract.triggers import extract_first_url

logger = logging.getLogger(__name__)

# D-03: RSS endpoint template — keep as a module constant for testability.
RSS_URL_TEMPLATE = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

# D-16: truncate raw_text body portion to 2000 chars (does NOT include the title prefix).
MAX_BODY_CHARS = 2000

# D-12: skip Shorts (duration < 60s).
MIN_DURATION_SECONDS = 60

# D-14: skip when combined title + " " + body < 100 chars.
MIN_COMBINED_CHARS = 100


def _text_of_chunk(chunk) -> str:
    """
    Extract text from a transcript chunk. The mock returns a dict {"text": ...};
    the real youtube-transcript-api 1.2.4 returns FetchedTranscriptSnippet objects
    that expose `.text` as an attribute. Support both shapes so swapping the mock
    for the real lib doesn't break.
    """
    if isinstance(chunk, dict):
        return chunk.get("text", "") or ""
    return getattr(chunk, "text", "") or ""


def _extract_duration(entry) -> Optional[int]:
    """
    Extract duration in seconds from feedparser entry. media_content may be missing
    on some feeds — return None to mean 'unknown' (caller treats unknown as 'do NOT skip').
    """
    media_content = getattr(entry, "media_content", None) or []
    if not media_content:
        return None
    first = media_content[0]
    raw = None
    if isinstance(first, dict):
        raw = first.get("duration")
    else:
        # Fallback for object-style media_content entries.
        try:
            raw = first.get("duration", None)  # type: ignore[attr-defined]
        except AttributeError:
            try:
                raw = first["duration"]
            except (KeyError, TypeError):
                raw = None
    if raw is None:
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _is_live_or_unpublished(entry) -> bool:
    """
    D-13: skip live streams or videos without a published_parsed (treats unparseable
    timestamps as 'not collectable').

    feedparser sets entry.media_status to 'public' for normal videos; live streams
    appear as 'live'. Missing media_status is treated as public (default-allow) so we
    do NOT regress on feeds that don't expose the field.
    """
    if getattr(entry, "published_parsed", None) is None:
        return True
    status = getattr(entry, "media_status", None)
    if status is not None and status != "public":
        return True
    return False


def _fetch_transcript(video_id: str) -> Optional[str]:
    """
    D-15 / D-17: language fallback chain.
      1. EN (manual or auto, picked by the library when languages=['en'])
      2. any-language auto-generated as last-ditch best-effort
    Returns None on failure — caller falls back to description.

    Per-video failures are logged at DEBUG (D-27): missing transcripts are expected at
    scale; do NOT spam WARNING. Unexpected errors get an exception log + return None.

    Uses the youtube-transcript-api 1.2.4 instance API: `YouTubeTranscriptApi().fetch(...)`.
    The 0.x classmethod `get_transcript` was removed upstream — see plan 03-01 SUMMARY
    for the full Rule 3 deviation explanation.
    """
    api = YouTubeTranscriptApi()

    # 1. EN preferred (covers manual + auto in one call when both available).
    try:
        chunks = api.fetch(video_id, languages=["en"])
        text = " ".join(_text_of_chunk(c) for c in chunks if _text_of_chunk(c))
        return text or None
    except (NoTranscriptFound, TranscriptsDisabled) as exc:
        logger.debug(
            "EN transcript missing for video_id=%s (%s) — trying any-language fallback",
            video_id, type(exc).__name__,
        )
    except Exception:
        # D-27: per-video transcript failures must never disrupt the channel-level loop.
        logger.exception("Unexpected error fetching EN transcript for %s", video_id)
        return None

    # 2. Any-language auto fallback. list() returns a TranscriptList we iterate.
    try:
        listing = api.list(video_id)
        for t in listing:
            try:
                fetched = t.fetch()
                text = " ".join(_text_of_chunk(c) for c in fetched if _text_of_chunk(c))
                if text:
                    return text
            except Exception:
                continue
    except (NoTranscriptFound, TranscriptsDisabled):
        pass
    except AttributeError:
        # Older / future API surface may not expose .list — silently degrade.
        pass
    except Exception:
        logger.exception("Error listing transcripts for %s", video_id)
    return None


def _description(entry) -> str:
    """Return the entry's description text (D-15 fallback). Empty string on miss."""
    return (
        getattr(entry, "summary", None)
        or getattr(entry, "description", None)
        or ""
    )


class YouTubeCollector(BaseCollector):
    platform = "youtube"

    def __init__(self, extract_session=None, extract_semaphore=None):
        # D-CONTEXT: no credentials — no env-var reads, no validation.
        # Phase 10 EXTRACT-02: pipeline-scoped aiohttp session + Semaphore(5).
        # Default-None preserves backward compat with existing tests that
        # construct YouTubeCollector() with no positional args.
        self._extract_session = extract_session
        self._extract_semaphore = extract_semaphore

    async def collect(self, db: aiosqlite.Connection, source: dict) -> None:
        cursor_str = source.get("last_cursor")  # D-10/D-11: ISO datetime string
        channel_id = source["target"]
        rss_url = RSS_URL_TEMPLATE.format(channel_id=channel_id)

        # D-28: channel-level errors must propagate to collect_all's per-source try/except.
        # feedparser.parse never raises (returns parsed.bozo=True on malformed feeds);
        # we treat bozo as empty rather than failing the whole channel.
        parsed = feedparser.parse(rss_url)
        if getattr(parsed, "bozo", False) and not getattr(parsed, "entries", None):
            logger.warning(
                "feedparser bozo for channel_id=%s — treating as empty feed", channel_id
            )
            return

        feed_title = getattr(getattr(parsed, "feed", None), "title", "") or channel_id
        collected_published: list[str] = []

        for entry in parsed.entries:
            # D-13: skip live / unparseable
            if _is_live_or_unpublished(entry):
                continue

            published = getattr(entry, "published", None)
            if not published:
                continue  # cannot record cursor without published

            # D-10: cursor filter — strictly greater than last_cursor (ISO string compare).
            # ISO 8601 strings with the same offset compare lexicographically as datetimes,
            # which is exactly what D-10/D-11 spec.
            if cursor_str is not None and published <= cursor_str:
                continue

            # D-12: skip Shorts. Unknown duration = do NOT skip (default-allow on missing field).
            duration = _extract_duration(entry)
            if duration is not None and duration < MIN_DURATION_SECONDS:
                continue

            # D-21: external_id from yt_videoid (preferred) or parsed from link as last resort.
            video_id = getattr(entry, "yt_videoid", None)
            if not video_id:
                # entry.link is like https://www.youtube.com/watch?v=VIDEO_ID
                link = getattr(entry, "link", "") or ""
                if "v=" in link:
                    video_id = link.split("v=", 1)[1].split("&", 1)[0]
            if not video_id:
                logger.warning(
                    "Skipping entry without video_id for channel_id=%s", channel_id
                )
                continue

            # D-15: try transcript -> [Phase 10 D-20: try URL extraction] -> description.
            body = _fetch_transcript(video_id)
            # Phase 50 (TRUNC-03): if YOUTUBE_EXCERPT_MODE=chapter and description
            # has parseable chapters, replace raw transcript with chapter excerpt.
            # Fail-soft: empty desc / no chapter structure → falls back to raw body.
            if body:
                from src.collectors.youtube_chapters import select_yt_body
                desc_for_chapters = _description(entry)
                body = select_yt_body(desc_for_chapters, body)
            extracted_body: Optional[str] = None
            extraction_status: Optional[str] = None
            extracted_at: Optional[str] = None
            if not body:
                # Phase 10 EXTRACT-02 / D-20: no transcript → try the FIRST URL in
                # the description, then prefer the extracted body over the raw
                # description text. Both extract_session AND extract_semaphore must
                # be present (graceful degradation when None — falls straight back
                # to description).
                desc = _description(entry)
                if (
                    self._extract_session is not None
                    and self._extract_semaphore is not None
                ):
                    candidate_url = extract_first_url(desc)
                    if candidate_url:
                        extracted_body, extraction_status = await extract_url(
                            self._extract_session,
                            candidate_url,
                            self._extract_semaphore,
                        )
                        extracted_at = datetime.datetime.now(
                            datetime.UTC
                        ).isoformat()
                # Prefer extracted body; else fall back to plain description text.
                body = extracted_body or desc

            # D-16: truncate body to 2000 chars BEFORE building raw_text.
            body_truncated = body[:MAX_BODY_CHARS]

            title = getattr(entry, "title", "") or ""
            combined = title + " " + body_truncated
            # D-14: skip if combined < 100 chars
            if len(combined) < MIN_COMBINED_CHARS:
                continue

            # D-20: raw_text = title + "\n\n" + truncated body
            raw_text = title + "\n\n" + body_truncated

            # D-25: author = entry.author or feed.title
            author = getattr(entry, "author", None) or feed_title

            # D-22: url = entry.link
            url = getattr(entry, "link", None)

            await insert_raw_post(
                db,
                source["id"],
                video_id,                  # D-21
                raw_text,                  # D-20
                author,                    # D-25
                url,                       # D-22
                platform="youtube",        # D-24
                content_type="video",      # D-24
                published_at=published,    # D-23
                source_score=0,            # D-19
                extracted_body=extracted_body,         # Phase 10 EXTRACT-02
                extraction_status=extraction_status,   # Phase 10 EXTRACT-02
                extracted_at=extracted_at,             # Phase 10 EXTRACT-02
            )
            collected_published.append(published)

        # D-11: persist last_cursor as max(entry.published) ISO string of collected videos.
        if collected_published:
            await update_source_last_cursor(
                db, source["id"], max(collected_published)
            )
