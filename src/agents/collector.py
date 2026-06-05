import asyncio
import logging
import os
from typing import Optional
import aiosqlite
from src.collectors.telegram import TelegramCollector
from src.collectors.reddit import RedditCollector
from src.collectors.youtube import YouTubeCollector
from src.collectors.email_imap import EmailCollector
from src.collectors.arxiv import ArxivCollector
from src.collectors.academic import AcademicCollector
from src.collectors.biorxiv import BiorxivCollector
from src.collectors.hackernews import HackernewsCollector
from src.collectors.instagram import InstagramCollector
from src.config.telegram_folders import load_folder_map
from src.database.client import ensure_sources, get_active_sources

logger = logging.getLogger(__name__)


async def discover_telegram_dialogs(db: aiosqlite.Connection, tg_client) -> int:
    """Auto-discover Telegram channels and persist new rows.

    Modes (env var ``TELEGRAM_AUTODISCOVER_MODE``, read INSIDE this function
    per D-12 / Pitfall 15 — never at module top):

        ``folders`` (default): read ``config/telegram_folders.txt`` via
            :func:`load_folder_map`; call ``tg_client.iter_dialogs_in_folders(...)``;
            resolve each row's category via ``folder_map[row['folder_title']]``
            with fallback to ``'other'``.

        ``unread``: legacy v1.4 behaviour — call ``tg_client.iter_unread_dialogs()``;
            all discovered rows default to ``category='other'``.

        ``off``: no-op; return 0.

    Pre-filter discipline (D-13, locks ``discover_dialogs_category_reset_bug``):
        Before passing discovered rows to :func:`ensure_sources`, drop targets
        already present in the ``sources`` table (``platform='telegram'``).
        This applies UNIFORMLY to both ``folders`` and ``unread`` modes. Without
        this filter, ``ensure_sources``'s blanket UPDATE would overwrite the
        user's manual category edits every pipeline run.

    Returns:
        Count of dialogs Telegram returned (NOT count of net-new rows
        inserted). Consumed by ``run_pipeline.py`` for the ``tg_discovered=N``
        log field (Plan 05).
    """
    # D-12: read env INSIDE the function — never at module top.
    mode = os.environ.get("TELEGRAM_AUTODISCOVER_MODE", "folders").strip().lower()
    if mode not in ("folders", "unread", "off"):
        logger.warning(
            "TELEGRAM_AUTODISCOVER_MODE=%r is not one of folders/unread/off "
            "— defaulting to 'folders'", mode,
        )
        mode = "folders"

    if mode == "off":
        return 0

    folder_map: dict[str, str] = {}
    if mode == "folders":
        try:
            folder_map = load_folder_map()
        except ValueError:
            # Fail-loud category error from the loader. Surface it, but do
            # not crash the pipeline run — log and return 0 so other
            # collectors still execute.
            logger.exception(
                "telegram_folders.txt has an invalid category — skipping "
                "Telegram folder auto-discovery this run"
            )
            return 0
        if not folder_map:
            # Loader already logged a distinct WARNING for missing-vs-empty
            # (D-04 / D-05). Nothing to do.
            return 0

    # Dispatch to the platform call.
    if mode == "folders":
        folder_method = getattr(tg_client, "iter_dialogs_in_folders", None)
        if folder_method is None:
            logger.warning(
                "tg_client lacks iter_dialogs_in_folders — folders mode "
                "unsupported; returning 0"
            )
            return 0
        try:
            discovered = await folder_method(list(folder_map.keys()))
        except Exception:
            logger.exception(
                "iter_dialogs_in_folders raised — skipping auto-discovery"
            )
            return 0

        # M-01 fix: iter_dialogs_in_folders matches folder titles case-insensitively
        # (client.py: `title.casefold() not in wanted_lower`) and stores the
        # Telegram-side original-case title in row["folder_title"]. The lookup
        # below must mirror that discipline or every row whose folder casing
        # differs between .txt and Telegram falls through to "other" — the
        # exact Goodhart-`other` bloat Pitfall 14 was meant to prevent.
        folder_map_lower = {k.casefold(): v for k, v in folder_map.items()}

        def _resolve_category(row: dict) -> str:
            return folder_map_lower.get(
                row.get("folder_title", "").casefold(), "other"
            )

        # M-03 fix: surface user-configured folders that Telegram never
        # returned, so a renamed/missing folder doesn't silently produce
        # net-zero discovery. We compare casefolded sets to align with the
        # case-insensitive matching above (M-01). Must fire even when
        # discovered == [] (all folders missing case).
        matched_lower = {
            row.get("folder_title", "").casefold() for row in discovered
        }
        missing = [
            t for t in folder_map.keys()
            if t.casefold() not in matched_lower
        ]
        if missing:
            logger.warning(
                "telegram_folders.txt lists %d folder(s) not present in "
                "Telegram: %s — rename them in the file or in Telegram "
                "to match",
                len(missing), missing,
            )
    else:
        # mode == "unread" — legacy v1.4 path.
        iter_method = getattr(tg_client, "iter_unread_dialogs", None)
        if iter_method is None:
            return 0
        try:
            discovered = await iter_method()
        except Exception:
            logger.exception(
                "iter_unread_dialogs raised — skipping auto-discovery"
            )
            return 0

        def _resolve_category(row: dict) -> str:
            return "other"

    if not discovered:
        return 0

    # D-13: pre-filter against existing (platform='telegram', target) rows
    # to lock the discover_dialogs_category_reset_bug fix. NEVER bypass.
    async with db.execute(
        "SELECT target FROM sources WHERE platform = 'telegram'"
    ) as cur:
        existing = {row[0] for row in await cur.fetchall()}

    new_sources = [
        {
            "target": d["target"],
            "category": _resolve_category(d),
            "display_name": d.get("title", d["target"]),
            "platform": "telegram",
        }
        for d in discovered
        if d["target"] not in existing
    ]
    if new_sources:
        await ensure_sources(db, new_sources)

    logger.info(
        "Auto-discovered %d Telegram dialogs (mode=%s, net_new=%d)",
        len(discovered), mode, len(new_sources),
    )
    # Return the discovered count (not net_new) — Plan 05's log field uses this.
    return len(discovered)


def _semaphore_key(source: dict) -> Optional[str]:
    """
    Build the per-account semaphore key (D-10).

    Per-source mode: lowercased `account_user@account_host`. Multiple sources for
    the same Gmail account therefore share a single 3-cap (D-12).

    Env-fallback mode (D-13): build key from EMAIL_USER@EMAIL_HOST so the legacy
    single-account profile also gets capped — prevents one global pipeline run
    from opening >3 connections to the env-default account when multiple legacy
    sources exist.

    Returns None when neither path can produce a valid key (no account_user AND
    no EMAIL_USER env var) — caller should skip such sources.
    """
    user = source.get("account_user")
    host = source.get("account_host")
    if user and host:
        return f"{user.lower()}@{host.lower()}"
    if user:
        # Per-source user without host: fall back to env host or default.
        env_host = os.environ.get("EMAIL_HOST", "imap.gmail.com").lower()
        return f"{user.lower()}@{env_host}"
    # D-13 fallback mode: single env-var account -> single shared semaphore.
    env_user = os.environ.get("EMAIL_USER", "").lower()
    env_host = os.environ.get("EMAIL_HOST", "imap.gmail.com").lower()
    if env_user:
        return f"{env_user}@{env_host}"
    return None


async def collect_all(
    db: aiosqlite.Connection,
    tg_client=None,
    extract_session=None,
    extract_semaphore=None,
) -> None:
    """
    Platform-agnostic dispatcher. Routes each active source to its collector by platform.

    Args:
        db: open aiosqlite connection
        tg_client: TelegramCollectorClient instance (optional — kept for backward compat
                   with existing tests that call collect_all(db, mock_client))
        extract_session: pipeline-scoped aiohttp.ClientSession with TCPConnector
                         configured to use SSRFSafeResolver. Phase 10 EXTRACT-01..04.
                         Default None — collectors gracefully skip extraction.
        extract_semaphore: asyncio.Semaphore(5) capping concurrent extractions across
                           ALL collectors per pipeline run (D-10). Default None —
                           collectors gracefully skip extraction.
    """
    collectors = {}
    if tg_client is not None:
        collectors["telegram"] = TelegramCollector(
            tg_client, extract_session, extract_semaphore
        )

    # Register Reddit collector unconditionally — public RSS path (2026-05-15
    # rewrite). No credentials required since Reddit's Responsible Builder
    # Policy (2024+) gates Data API access behind ticket-approval. RSS is
    # public and does not require auth. See src/collectors/reddit.py.
    try:
        collectors["reddit"] = RedditCollector(extract_session, extract_semaphore)
    except Exception:
        logger.exception(
            "Failed to construct RedditCollector — skipping Reddit, "
            "other platforms unaffected"
        )

    # D-CONTEXT: register YouTube collector unconditionally.
    # No credentials required (uses public RSS + youtube-transcript-api over HTTPS).
    # Constructor accepts the extract session/semaphore for Phase 10 EXTRACT-02.
    try:
        collectors["youtube"] = YouTubeCollector(extract_session, extract_semaphore)
    except Exception:
        logger.exception(
            "Failed to construct YouTubeCollector — skipping YouTube, "
            "other platforms unaffected"
        )

    # ArXiv: no credentials required — register unconditionally.
    try:
        collectors["arxiv"] = ArxivCollector()
    except Exception:
        logger.exception(
            "Failed to construct ArxivCollector — skipping ArXiv, "
            "other platforms unaffected"
        )

    # Academic RSS: generic journal feeds — no credentials required.
    try:
        collectors["academic"] = AcademicCollector()
    except Exception:
        logger.exception(
            "Failed to construct AcademicCollector — skipping academic RSS, "
            "other platforms unaffected"
        )

    # bioRxiv / medRxiv: no credentials required — register unconditionally.
    try:
        collectors["biorxiv"] = BiorxivCollector()
    except Exception:
        logger.exception(
            "Failed to construct BiorxivCollector — skipping bioRxiv/medRxiv, "
            "other platforms unaffected"
        )

    # Hacker News (Algolia HN Search API): no credentials required.
    try:
        collectors["hackernews"] = HackernewsCollector()
    except Exception:
        logger.exception(
            "Failed to construct HackernewsCollector — skipping HN, "
            "other platforms unaffected"
        )

    # Phase 26 D-04: Instagram is gated on INSTAGRAM_USER. If unset, the
    # collector is silently inactive (zero DB writes, zero log lines). Mirrors
    # the Reddit/Email pattern. Wrap construction in try/except so an exotic
    # init-time failure cannot break the other platforms.
    if os.environ.get("INSTAGRAM_USER"):
        try:
            collectors["instagram"] = InstagramCollector()
        except Exception:
            logger.exception(
                "Failed to construct InstagramCollector — skipping Instagram, "
                "other platforms unaffected"
            )

    # Phase 11 D-08/D-09: Email is constructed PER-SOURCE inside the dispatch
    # loop (NOT once for the platform) so each EmailCollector binds to its
    # source's per-account semaphore. The platform-level entry below is a
    # sentinel that signals "late-bind per source"; the dispatch loop branch
    # for platform == 'email' handles construction.
    #
    # Backward-compat (D-13): even without env vars, sources with
    # account_user IS NOT NULL still collect (per-source creds bypass the env
    # gate). Registration check is done inside the per-source loop below.
    if os.environ.get("EMAIL_USER") and not os.environ.get("EMAIL_PASS"):
        # Partial config (user without password) — warn loudly; mirrors the
        # Reddit half-config warning above.
        logger.warning(
            "EMAIL_USER is set but EMAIL_PASS is missing — "
            "Email legacy fallback path disabled (per-source creds still work)"
        )

    # D-09/D-10/EMAIL-07: per-account semaphore registry. Keyed by lowercased
    # `account_user@account_host` so multiple sources for the same Gmail account
    # share a single 3-cap (D-12). Built lazily inside the per-source loop below.
    _account_semaphores: dict[str, asyncio.Semaphore] = {}

    sources = await get_active_sources(db)

    # Phase 26 D-13: hard cap of 20 Instagram profiles per run. With >20 IG
    # seeds, processing them all is the exact ban-vector the cap exists to
    # prevent. Slice oldest-first by last_fetched_at (NULLs first, then ASC)
    # so rotation picks newly-added or least-recently-touched profiles first.
    # Done in Python after fetch (small dataset; ~20-40 sources total).
    IG_CAP = 20
    ig_sources = [s for s in sources if s.get("platform") == "instagram"]
    if len(ig_sources) > IG_CAP:
        # NULLs first, then ascending last_fetched_at. Use a stable key:
        # (has_fetched_at, last_fetched_at_or_empty). False sorts before True.
        ig_sources.sort(
            key=lambda s: (
                s.get("last_fetched_at") is not None,
                s.get("last_fetched_at") or "",
            )
        )
        kept_ig_ids = {id(s) for s in ig_sources[:IG_CAP]}
        logger.info(
            "IG D-13 cap: %d active IG profiles, capping to %d oldest-first",
            len(ig_sources), IG_CAP,
        )
        # Filter `sources` in place: keep all non-IG; for IG, keep only the
        # first IG_CAP by oldest-first ordering.
        sources = [
            s for s in sources
            if s.get("platform") != "instagram" or id(s) in kept_ig_ids
        ]

    for source in sources:
        platform = source.get("platform")
        if platform == "email":
            # D-09/D-10/D-12: build or reuse per-account semaphore from registry.
            sem_key = _semaphore_key(source)
            if sem_key is None:
                logger.warning(
                    "No credentials available for email source id=%s target=%s "
                    "(no per-source account_user, no EMAIL_USER env) — skipping",
                    source.get("id"), source.get("target"),
                )
                continue
            if sem_key not in _account_semaphores:
                _account_semaphores[sem_key] = asyncio.Semaphore(3)  # D-11
            account_sem = _account_semaphores[sem_key]
            try:
                collector = EmailCollector(
                    extract_session=extract_session,
                    extract_semaphore=extract_semaphore,
                    source=source,
                    account_semaphore=account_sem,
                )
            except Exception:
                # D-30 hygiene: source.id + target only; NEVER credential values.
                logger.exception(
                    "Failed to construct EmailCollector for source id=%s target=%s "
                    "— skipping (other sources unaffected)",
                    source.get("id"), source.get("target"),
                )
                continue
            try:
                await collector.collect(db, source)
            except Exception:
                logger.exception(
                    "Failed to collect from platform=email target=%s",
                    source.get("target"),
                )
            continue

        collector = collectors.get(platform)
        if collector is None:
            logger.warning(
                "No collector registered for platform=%r (source id=%s target=%s) — skipping",
                platform,
                source.get("id"),
                source.get("target"),
            )
            continue
        try:
            await collector.collect(db, source)
        except Exception:
            logger.exception(
                "Failed to collect from platform=%s target=%s",
                platform,
                source.get("target"),
            )
