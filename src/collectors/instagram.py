"""
InstagramCollector — text-only IG post collector via instaloader.

Reads up to 5 recent posts per IG handle, applies last_cursor, falls back to
vision-OCR for short captions when VISION_MODEL is set, aborts after 3 consecutive
ConnectionException (soft-block protection). Never logs INSTAGRAM_PASS.

Per Phase 26 D-01..D-21 in .planning/phases/26-instagram-collector/26-CONTEXT.md.

CRITICAL — D-03 invariant:
    Module-level env binding is FORBIDDEN. All INSTAGRAM_* / VISION_MODEL reads
    happen inside __init__ or methods. See reference_module_level_env_binding.

CRITICAL — D-17 invariant:
    INSTAGRAM_PASS literal must NEVER appear in any log/stdout/stderr line.
    Always catch instaloader exceptions by typed subclass and log type(e).__name__
    only — NEVER str(e) on exceptions that might quote credentials.
"""
from __future__ import annotations

import asyncio
import base64
import datetime
import logging
import os
import random
import re
from pathlib import Path
from typing import Optional

import aiohttp
import aiosqlite
import instaloader

from src.collectors.base import BaseCollector
from src.database.client import insert_raw_post, update_source_last_cursor

logger = logging.getLogger(__name__)


# Project root anchor — BLOCKER-02 fix. `Path(".cache")` is CWD-relative
# and silently desyncs whenever the entry point (test harness, future
# worker, schtasks-style cold start) doesn't run the `os.chdir` shim in
# run_pipeline.py. Anchor to <root>/.cache always.
# src/collectors/instagram.py: parents[0]=collectors, [1]=src, [2]=<root>.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Post-collection cap (D-09).
MAX_POSTS_PER_PROFILE = 5

# 3-strike consecutive ConnectionException abort (D-15). Tracked across all
# profiles within a single pipeline run.
SOFT_BLOCK_STRIKE_LIMIT = 3

# Min combined caption length to skip the vision-OCR fallback (D-11/D-12).
MIN_CAPTION_CHARS = 100

# Inter-profile sleep range — D-14.
SLEEP_MIN_SECONDS = 15
SLEEP_MAX_SECONDS = 30

# OCR prompt for the Ollama vision model — kept short, Russian output to match
# the curator/editor profile expectations.
_OCR_PROMPT = (
    "Describe this Instagram photo in Russian in 2-3 sentences. "
    "Focus on any text, data, numbers, or key concepts visible in the image. "
    "If it's a meme or text screenshot, transcribe its meaning."
)


async def _ocr_photo(host: str, model: str, photo_url: str) -> Optional[str]:
    """Fetch the photo bytes from `photo_url` and run Ollama vision OCR.

    Returns the OCR text on success, or None on any failure (caller skips
    the post). Never raises.

    D-18: only typed exception subclasses are caught here from instaloader/
    aiohttp — anything else propagates so we don't hide real bugs.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                photo_url, timeout=aiohttp.ClientTimeout(total=30)
            ) as r:
                if r.status != 200:
                    logger.warning(
                        "IG OCR photo fetch failed: status=%s url=%s",
                        r.status, photo_url,
                    )
                    return None
                photo_bytes = await r.read()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.warning(
            "IG OCR photo fetch error: %s", type(e).__name__
        )
        return None
    if not photo_bytes:
        return None
    b64 = base64.b64encode(photo_bytes).decode()
    payload = {
        "model": model,
        "prompt": _OCR_PROMPT,
        "images": [b64],
        "stream": False,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{host}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as r:
                if r.status == 404:
                    logger.warning(
                        "IG OCR: vision model '%s' not found in Ollama. "
                        "Run: ollama pull %s", model, model,
                    )
                    return None
                r.raise_for_status()
                data = await r.json()
                text = (data.get("response") or "").strip()
                if not text:
                    return None
                return f"[Vision/{model}] {text}"
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.warning("IG OCR Ollama call failed: %s", type(e).__name__)
        return None


class InstagramCollector(BaseCollector):
    """Text-only IG collector.

    D-03: env vars are read in __init__, NOT at module top.
    D-17: INSTAGRAM_PASS is held only as an instance attribute and never
    logged. Exception handling uses type(e).__name__ — never str(e).
    """

    platform = "instagram"

    def __init__(self):
        # D-03: read env vars at construction time (per-run binding).
        self._user = os.environ.get("INSTAGRAM_USER", "")
        # D-17: keep the secret on the instance; never log it.
        self._password = os.environ.get("INSTAGRAM_PASS", "")
        # Warmup gate (D-06).
        self._warmup_confirmed = bool(
            os.environ.get("INSTAGRAM_WARMUP_CONFIRMED", "").strip()
        )
        # Optional vision-OCR fallback (D-11). Empty string treated as unset.
        self._vision_model = os.environ.get("VISION_MODEL", "").strip() or None
        self._ollama_host = os.environ.get(
            "OLLAMA_HOST", "http://localhost:11434"
        )

        # Consecutive soft-block strike counter — survives across .collect()
        # calls within the same run / collector instance (D-15).
        self._consecutive_strikes = 0
        # Once we've aborted, all subsequent .collect() calls become no-ops.
        self._aborted = False
        # Lazily constructed instaloader.Instaloader so tests can patch
        # `src.collectors.instagram.instaloader.Instaloader` at the right
        # spot. None until first login attempt.
        self._loader: Optional[instaloader.Instaloader] = None
        # Track that we've already attempted login once per run (D-07).
        self._logged_in = False

    # ------------------------------------------------------------------ #
    # Session / login lifecycle
    # ------------------------------------------------------------------ #

    def _session_file(self) -> Path:
        """Path to the cached session file: `<root>/.cache/instaloader-session-{user}`.

        BLOCKER-02: anchored to PROJECT ROOT, NOT process CWD. `Path(".cache")`
        bare relative path silently desyncs when callers don't run the
        `os.chdir(Path(__file__).resolve().parent)` shim in run_pipeline.py
        (test harnesses, schtasks cold start, future workers, uvicorn workers).
        """
        return _PROJECT_ROOT / ".cache" / f"instaloader-session-{self._user}"

    def _build_loader(self) -> instaloader.Instaloader:
        """Construct instaloader with D-08 text-only options."""
        return instaloader.Instaloader(
            download_pictures=False,
            download_videos=False,
            download_comments=False,
            save_metadata=False,
            compress_json=False,
        )

    def _do_fresh_login(self, L: instaloader.Instaloader) -> bool:
        """Attempt fresh login + persist session. Return True on success."""
        try:
            # D-17: never log the password literal. Only the username is safe.
            L.login(self._user, self._password)
            try:
                # Ensure the cache dir exists for save.
                self._session_file().parent.mkdir(parents=True, exist_ok=True)
                L.save_session_to_file(str(self._session_file()))
            except OSError as save_err:
                logger.warning(
                    "IG: session save failed: %s", type(save_err).__name__
                )
            return True
        except instaloader.exceptions.BadCredentialsException as e:
            # D-18: log only the exception class — error message may quote
            # the password the caller passed in.
            logger.warning(
                "IG: login failed for user=%s — %s "
                "(check INSTAGRAM_USER / INSTAGRAM_PASS)",
                self._user, type(e).__name__,
            )
            return False
        except (
            instaloader.exceptions.ConnectionException,
            instaloader.exceptions.TwoFactorAuthRequiredException,
            instaloader.exceptions.InvalidArgumentException,
            instaloader.exceptions.LoginException,
        ) as e:
            logger.warning(
                "IG: login attempt error for user=%s — %s",
                self._user, type(e).__name__,
            )
            return False

    def _ensure_logged_in(self) -> bool:
        """Lazy login. Returns True if the loader is ready, False to abort run.

        D-07: load session first; on failure fall back to login + save.
        Single login attempt per run. If session is invalid
        (LoginRequiredException), delete + retry login once.
        """
        if self._logged_in and self._loader is not None:
            return True

        if not self._user:
            return False

        # D-06: warmup gate. Refuse if no session file AND no warmup flag.
        session_path = self._session_file()
        if not session_path.exists() and not self._warmup_confirmed:
            logger.warning(
                "IG: refusing to run — no session file at %s AND "
                "INSTAGRAM_WARMUP_CONFIRMED is unset. Use a DEDICATED IG "
                "account and warm it up 3-7 days on the mobile app, then "
                "set INSTAGRAM_WARMUP_CONFIRMED=1 in .env. Otherwise the "
                "first programmatic login will likely be banned.",
                session_path,
            )
            return False

        L = self._build_loader()
        # Step 1: try session file.
        if session_path.exists():
            try:
                L.load_session_from_file(self._user, str(session_path))
                self._loader = L
                self._logged_in = True
                return True
            except instaloader.exceptions.LoginRequiredException as e:
                # D-07: delete stale session and fall through to fresh login.
                logger.warning(
                    "IG: cached session invalid (%s) — deleting and "
                    "attempting fresh login", type(e).__name__,
                )
                try:
                    session_path.unlink()
                except OSError:
                    pass
            except instaloader.exceptions.ConnectionException as e:
                logger.warning(
                    "IG: session-load network error (%s) — falling back to login",
                    type(e).__name__,
                )
            except instaloader.exceptions.InstaloaderException as e:
                # Catch-all for typed instaloader exceptions only. NEVER
                # `except Exception` (D-18).
                logger.warning(
                    "IG: session-load failed (%s) — falling back to login",
                    type(e).__name__,
                )

        # Step 2: fresh login attempt (at most once per run, D-07).
        if not self._password:
            logger.warning(
                "IG: INSTAGRAM_USER=%s set without INSTAGRAM_PASS or valid "
                "session — skipping", self._user,
            )
            return False
        if self._do_fresh_login(L):
            self._loader = L
            self._logged_in = True
            return True
        return False

    # ------------------------------------------------------------------ #
    # Per-source collect
    # ------------------------------------------------------------------ #

    async def collect(self, db: aiosqlite.Connection, source: dict) -> None:
        """Fetch up to MAX_POSTS_PER_PROFILE posts for one IG handle.

        Per D-15, after SOFT_BLOCK_STRIKE_LIMIT consecutive
        ConnectionException across profiles, this collector becomes a no-op
        for the rest of the pipeline run. Cursor is NOT updated when the
        run aborts.
        """
        if self._aborted:
            # Sticky abort for the rest of the run.
            return

        if not self._ensure_logged_in():
            # Already logged the reason; do not update cursor, do not insert.
            return

        target = source["target"]
        last_cursor_str = source.get("last_cursor")
        last_cursor_dt: Optional[datetime.datetime] = None
        if last_cursor_str:
            try:
                last_cursor_dt = datetime.datetime.fromisoformat(last_cursor_str)
            except ValueError:
                logger.warning(
                    "IG: invalid last_cursor %r for source id=%s — ignoring",
                    last_cursor_str, source.get("id"),
                )

        # D-14: random sleep BETWEEN profiles. Skip on the first call.
        # The collector instance state tracks "have we collected at least one
        # profile already this run". We use _logged_in as a proxy alone is not
        # right — instead track explicitly.
        if getattr(self, "_collected_any", False):
            try:
                await asyncio.sleep(
                    random.uniform(SLEEP_MIN_SECONDS, SLEEP_MAX_SECONDS)
                )
            except asyncio.CancelledError:
                raise

        try:
            profile = instaloader.Profile.from_username(
                self._loader.context, target
            )
        except instaloader.exceptions.ProfileNotExistsException:
            # D-16: deactivate this source row, continue (no abort, no strike).
            logger.warning(
                "IG: profile @%s does not exist — setting is_active=0", target
            )
            await db.execute(
                "UPDATE sources SET is_active = 0 WHERE id = ?",
                (source.get("id"),),
            )
            await db.commit()
            return
        except instaloader.exceptions.ConnectionException:
            self._consecutive_strikes += 1
            self._collected_any = True  # so the sleep applies next time
            if self._consecutive_strikes >= SOFT_BLOCK_STRIKE_LIMIT:
                logger.warning(
                    "IG: %d consecutive ConnectionException — aborting run "
                    "(soft-block protection, cursor preserved)",
                    self._consecutive_strikes,
                )
                self._aborted = True
            return
        except instaloader.exceptions.QueryReturnedBadRequestException:
            # D-16: treat as soft-block; abort run.
            logger.warning(
                "IG: QueryReturnedBadRequestException for @%s — aborting run "
                "(soft-block protection)", target,
            )
            self._aborted = True
            return
        except instaloader.exceptions.LoginRequiredException as e:
            logger.warning(
                "IG: login required mid-run for @%s (%s) — aborting",
                target, type(e).__name__,
            )
            self._aborted = True
            return

        # WARNING-03 fix: do NOT reset the strike counter here. Reaching
        # Profile.from_username only proves a single metadata fetch worked —
        # mid-fetch ConnectionExceptions during get_posts() iteration are the
        # actual ban signal, and resetting too early lets three back-to-back
        # mid-fetch failures slip past the 3-strike abort. Reset moved to
        # AFTER the iteration completes successfully.
        self._collected_any = True

        collected_cursors: list[datetime.datetime] = []
        try:
            posts_iter = profile.get_posts()
        except instaloader.exceptions.ConnectionException:
            # WARNING-03: count as a strike — get_posts() can fail mid-iteration
            # exactly when IG is rate-limiting.
            self._consecutive_strikes += 1
            if self._consecutive_strikes >= SOFT_BLOCK_STRIKE_LIMIT:
                logger.warning(
                    "IG: %d consecutive ConnectionException (mid-fetch) — "
                    "aborting run (soft-block protection, cursor preserved)",
                    self._consecutive_strikes,
                )
                self._aborted = True
            return
        except instaloader.exceptions.InstaloaderException as e:
            logger.warning(
                "IG: get_posts() failed for @%s (%s)", target, type(e).__name__
            )
            return

        # D-09: cap at MAX_POSTS_PER_PROFILE (5).
        seen = 0
        try:
            for post in posts_iter:
                if seen >= MAX_POSTS_PER_PROFILE:
                    break
                seen += 1
                try:
                    shortcode = post.shortcode
                    caption = post.caption or ""
                    date_utc = post.date_utc
                    # Ensure date_utc has a tz (instaloader returns naive utc).
                    if date_utc.tzinfo is None:
                        date_utc = date_utc.replace(tzinfo=datetime.UTC)
                except instaloader.exceptions.InstaloaderException as e:
                    logger.warning(
                        "IG: skipping post (%s)", type(e).__name__
                    )
                    continue

                # D-10: skip already-seen posts.
                if last_cursor_dt is not None and date_utc <= last_cursor_dt:
                    continue

                raw_text = caption
                # D-11/D-12: vision-OCR fallback for short captions.
                if len(raw_text) < MIN_CAPTION_CHARS:
                    if self._vision_model and not getattr(post, "is_video", False):
                        photo_url = getattr(post, "url", None)
                        if photo_url:
                            ocr_text = await _ocr_photo(
                                self._ollama_host, self._vision_model, photo_url
                            )
                            if ocr_text:
                                raw_text = ocr_text
                            else:
                                # OCR failed — skip the post (no zombie row).
                                continue
                        else:
                            continue
                    else:
                        # No vision model OR is_video — skip silently.
                        continue

                url = f"https://instagram.com/p/{shortcode}/"
                published_at = date_utc.isoformat()
                try:
                    await insert_raw_post(
                        db,
                        source["id"],
                        shortcode,
                        raw_text,
                        target,            # author = the handle
                        url,
                        platform="instagram",
                        content_type="post",
                        published_at=published_at,
                    )
                    collected_cursors.append(date_utc)
                except instaloader.exceptions.InstaloaderException as e:
                    logger.warning(
                        "IG: insert_raw_post raised IG exception (%s) — skipping",
                        type(e).__name__,
                    )
                    continue
        except instaloader.exceptions.ConnectionException:
            # WARNING-03: mid-iteration ConnectionException is a ban signal.
            # Three of these in a row across profiles must abort the run.
            self._consecutive_strikes += 1
            if self._consecutive_strikes >= SOFT_BLOCK_STRIKE_LIMIT:
                logger.warning(
                    "IG: %d consecutive ConnectionException (mid-iteration) "
                    "— aborting run (soft-block protection, cursor preserved)",
                    self._consecutive_strikes,
                )
                self._aborted = True
            return

        # WARNING-03: ONLY reset the strike counter after we've completed
        # the full get_posts() iteration without raising ConnectionException.
        # This is the meaning of "truly successful work" — the profile metadata
        # AND its post list both came down without a soft-block kicking in.
        self._consecutive_strikes = 0

        # D-10: persist last_cursor as the latest published_at we saw.
        if collected_cursors:
            await update_source_last_cursor(
                db,
                source["id"],
                max(collected_cursors).isoformat(),
            )
