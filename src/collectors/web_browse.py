import logging
from typing import Optional
from urllib.parse import urlparse

import trafilatura

logger = logging.getLogger(__name__)

_TIMEOUT_MS = 15_000
_ALLOWED_SCHEMES = frozenset({"http", "https"})

try:
    from playwright.async_api import async_playwright as _async_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False
    _async_playwright = None


async def _playwright_fetch_html(url: str) -> Optional[str]:
    if not _PLAYWRIGHT_AVAILABLE:
        return None
    async with _async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url, timeout=_TIMEOUT_MS, wait_until="domcontentloaded")
            return await page.content()
        finally:
            await browser.close()


async def fetch_article_body(url: str) -> Optional[str]:
    scheme = urlparse(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        logger.warning("Blocked non-http(s) URL scheme: %s", scheme)
        return None

    if _PLAYWRIGHT_AVAILABLE:
        try:
            html = await _playwright_fetch_html(url)
            body = trafilatura.extract(html) if html else None
            if body and len(body) > 200:
                return body
        except Exception as exc:
            logger.debug("Playwright failed for %s: %s, trying trafilatura direct", url, exc)

    try:
        body = trafilatura.fetch_url(url)
        if body and len(body) > 200:
            return body
    except Exception as exc:
        logger.debug("trafilatura.fetch_url failed for %s: %s", url, exc)
    return None
