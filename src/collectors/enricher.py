import logging
import re
from urllib.parse import urlparse
import aiosqlite
from simhash import Simhash

from src.collectors.web_browse import fetch_article_body
from src.database.client import _char_ngrams, CURRENT_SIMHASH_VERSION

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r'https?://\S{15,}')
_SKIP_DOMAINS = {"t.me", "telegram.me", "telegram.org"}


def _extract_first_external_url(text: str) -> str | None:
    for match in _URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;:!?)")
        try:
            domain = urlparse(url).netloc.lower().removeprefix("www.")
            if domain not in _SKIP_DOMAINS:
                return url
        except Exception:
            pass
    return None


async def enrich_web_links(db: aiosqlite.Connection) -> int:
    async with db.execute(
        "SELECT id, raw_text, url FROM raw_posts WHERE status='unprocessed'"
    ) as cursor:
        posts = await cursor.fetchall()

    enriched = 0
    for post_id, raw_text, stored_url in posts:
        try:
            if not raw_text:
                continue
            url = _extract_first_external_url(raw_text)
            if not url and stored_url:
                try:
                    domain = urlparse(stored_url).netloc.lower().removeprefix("www.")
                    if domain not in _SKIP_DOMAINS:
                        url = stored_url
                except Exception:
                    pass
            if not url:
                continue
            body = await fetch_article_body(url)
            if body and len(body) > len(raw_text) + 200:
                try:
                    new_hash = Simhash(_char_ngrams(body, 3)).value
                    new_hash_signed = new_hash - (1 << 64) if new_hash >= (1 << 63) else new_hash
                except Exception:
                    new_hash_signed = None
                await db.execute(
                    "UPDATE raw_posts SET raw_text=?, simhash=?, simhash_version=? WHERE id=?",
                    (body, new_hash_signed, CURRENT_SIMHASH_VERSION, post_id),
                )
                await db.commit()
                enriched += 1
                logger.info("Enriched post %d — %d chars from %s", post_id, len(body), url)
        except Exception:
            logger.exception("enricher: failed on post %d — skipping", post_id)

    return enriched
