"""
EmailCollector — fetches unread newsletters from a configured IMAP folder via aioimaplib,
parses the body (text/plain preferred, html2text fallback for HTML-only emails),
and writes to raw_posts.

Auth: Gmail App Password OR generic IMAP username/password (D-02). Credentials are
read from os.environ at construction time, NEVER logged (D-30).

Module name: this file is `email_imap.py`, NOT `email.py`. Naming the file `email.py`
would shadow the Python stdlib email package and break `from email import message_from_bytes`
inside this very module. Locked in plan 04-01.

Patch-target contract locked in plan 04-01 (see 04-01-SUMMARY.md):
- `import aioimaplib` at module top — tests patch `src.collectors.email_imap.aioimaplib.IMAP4_SSL`.
- `import html2text` at module top, called as `html2text.html2text(...)` — tests patch
  `src.collectors.email_imap.html2text.html2text`.
- The dispatcher must `from src.collectors.email_imap import EmailCollector` at module top
  AND gate the registration on `os.environ.get("EMAIL_USER")` (D-08).

References: D-01..D-33 in .planning/phases/04-email-collector/04-CONTEXT.md
"""
import asyncio
import datetime
import logging
import os
import re
from typing import Optional

import aiohttp
import aiosqlite
import aioimaplib
import html2text

# Stdlib email package — these imports resolve to the stdlib because this module
# is named email_imap.py, NOT email.py. Do NOT rename.
from email import message_from_bytes, policy
from email.utils import parsedate_to_datetime

from src.collectors.base import BaseCollector
from src.database.client import insert_raw_post, update_source_last_cursor
from src.extract import extract_url
from src.extract.triggers import extract_first_url

logger = logging.getLogger(__name__)

# D-18: truncate body to 5000 chars (newsletters are richer than YouTube transcripts).
MAX_BODY_CHARS = 5000

# D-20: skip when combined subject + " " + body_truncated < 100 chars.
MIN_COMBINED_CHARS = 100

# D-04: default TLS port. Override via EMAIL_PORT env var.
DEFAULT_IMAP_PORT = 993

# D-03: default folder is INBOX. Gmail labels work as folder names.
DEFAULT_FOLDER = "INBOX"

# D-19: collapse 3+ consecutive newlines to 2.
_WHITESPACE_RE = re.compile(r"\n{3,}")

# T-11-02 / D-02: defense-in-depth — env-var names must match POSIX shell convention.
# Prevents a malformed account_pass_env value (e.g., "; rm -rf /") from being passed
# to os.environ.get() — even though os.environ.get is read-only, the principle is
# to validate at the trust boundary so the malformed string never enters lookup paths
# that may be inherited by future code (subprocess, shell-out, etc.).
_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def _resolve_account_creds(source: Optional[dict]) -> tuple[str, int, str, str, str]:
    """
    Resolve IMAP credentials with priority: per-source columns -> env-var fallback (D-14).

    Returns: (host, port, user, password, folder).

    Raises:
      ValueError: if source.account_pass_env is non-NULL and fails the POSIX env-var
                  name regex (T-11-02 mitigation).
      KeyError:   if env-fallback path is taken but EMAIL_USER / EMAIL_PASS are unset
                  (D-16: fail-on-first-fetch matches existing single-account behavior),
                  OR if per-source path is taken and the named env var is unset.

    D-30: NEVER log the resolved password value. Caller passes the returned tuple
    straight to aioimaplib.login; the password never enters logging.
    """
    # D-13/D-14: per-source non-NULL account_user wins; otherwise env fallback.
    if source and source.get("account_user"):
        user = source["account_user"]
        host = source.get("account_host") or os.environ.get("EMAIL_HOST", "imap.gmail.com")
        port_raw = source.get("account_port") or os.environ.get("EMAIL_PORT", str(DEFAULT_IMAP_PORT))
        port = int(port_raw)
        pass_env_name = source.get("account_pass_env") or "EMAIL_PASS"
        # T-11-02: validate env-var name BEFORE lookup
        if not _ENV_NAME_RE.match(pass_env_name):
            raise ValueError(
                f"Invalid account_pass_env name (must match {_ENV_NAME_RE.pattern}): {pass_env_name!r}"
            )
        password = os.environ[pass_env_name]  # KeyError on miss is desired (D-16)
        folder = os.environ.get("EMAIL_FOLDER", DEFAULT_FOLDER)
        return host, port, user, password, folder

    # Env fallback (D-13): legacy single-account profile
    host = os.environ.get("EMAIL_HOST", "imap.gmail.com")
    port = int(os.environ.get("EMAIL_PORT", str(DEFAULT_IMAP_PORT)))
    user = os.environ["EMAIL_USER"]
    password = os.environ["EMAIL_PASS"]
    folder = os.environ.get("EMAIL_FOLDER", DEFAULT_FOLDER)
    return host, port, user, password, folder


def _extract_body(msg) -> str:
    """
    D-17: text/plain preferred -> html2text(text/html) -> "[no readable body]" sentinel.

    Walks the parsed email message; takes the FIRST text/plain part found, falling back
    to html2text-converted text/html if no text/plain exists. Both branches default to
    the sentinel when neither type is present (e.g., attachment-only messages).

    Multipart/alternative emails commonly carry BOTH text/plain and text/html — we
    prefer the plain version (it's what the sender hand-curated; the HTML is auto-rendered).
    """
    body_text: Optional[str] = None
    body_html: Optional[str] = None

    for part in msg.walk():
        ctype = part.get_content_type()
        if ctype == "text/plain" and body_text is None:
            try:
                body_text = part.get_content()
            except Exception:
                # Per-part decode failures must not crash the message.
                logger.debug("text/plain part failed to decode; trying next part")
        elif ctype == "text/html" and body_html is None:
            try:
                body_html = part.get_content()
            except Exception:
                logger.debug("text/html part failed to decode; trying next part")

    if body_text:
        return body_text
    if body_html:
        try:
            return html2text.html2text(body_html)
        except Exception:
            # html2text is normally infallible on str input; defensive log + sentinel.
            logger.exception("html2text failed unexpectedly; falling back to sentinel")
            return "[no readable body]"
    return "[no readable body]"


def _normalize_body(body: str) -> str:
    """D-19 + D-18: collapse 3+ newlines to 2, strip ends, then truncate to 5000."""
    cleaned = _WHITESPACE_RE.sub("\n\n", body).strip()
    return cleaned[:MAX_BODY_CHARS]


def _parse_uids(uid_search_lines) -> list[int]:
    """
    Parse the response from aioimaplib uid_search into a list of integer UIDs.

    The exact return shape varies across aioimaplib versions — common shapes:
      [b'1 5 12']                — single line, space-separated UIDs
      [b'1 5 12', b'']           — possibly with a trailing empty line
      [b'']                       — no matches
      [None] / []                  — degenerate cases

    Defensive parser: flatten everything to a single bytes/str, decode, split on whitespace,
    keep only digit-tokens, parse to int. Returns empty list on any unparseable input.
    """
    if not uid_search_lines:
        return []
    parts: list[str] = []
    for line in uid_search_lines:
        if line is None:
            continue
        if isinstance(line, bytes):
            try:
                line = line.decode("ascii", errors="ignore")
            except Exception:
                continue
        elif not isinstance(line, str):
            line = str(line)
        parts.extend(line.split())
    uids: list[int] = []
    for tok in parts:
        if tok.isdigit():
            try:
                uids.append(int(tok))
            except ValueError:
                continue
    return uids


def _extract_message_bytes(fetch_response) -> Optional[bytes]:
    """
    Extract the RFC 5322 message bytes from an aioimaplib uid('fetch', ...) response.

    Common shapes:
      [b'(RFC822 {NNN}', b'<raw bytes>', b')']                       — list of bytes
      [(b'(RFC822 {NNN}', b'<raw bytes>'), b')']                     — list with one tuple
      [b'(RFC822 {NNN}', bytearray(b'<raw bytes>'), b')']            — bytearray middle
      None / [] / single-element-list                                  — degenerate

    Defensive parser: walk all elements, return the FIRST element that looks like
    a message body (>= 100 bytes, contains b'\\r\\n\\r\\n' header/body separator).
    Returns None on no match.
    """
    if not fetch_response:
        return None
    flattened = []
    for item in fetch_response:
        if isinstance(item, tuple):
            flattened.extend(item)
        else:
            flattened.append(item)
    for item in flattened:
        if isinstance(item, (bytes, bytearray)):
            data = bytes(item)
            # Heuristic: header/body separator + reasonable length filters out
            # the b'(RFC822 {NNN}' and b')' framing tokens.
            if len(data) >= 100 and b"\r\n\r\n" in data:
                return data
    return None


# --------------------------------------------------------------------------- #
# Phase 96 / IMAP-TRIAGE-01: pre-curator 4-way classifier (qwen2.5:3b)
# --------------------------------------------------------------------------- #

_TRIAGE_LABELS = frozenset({"newsletter", "support", "personal", "action"})
_TRIAGE_PUNCT_RE = re.compile(r"[^\w]+")


# WR-01 fix (Phase 96): one-time `ollama list` preflight cached for process
# lifetime. None = not yet probed; True/False = result. Mirrors
# scripts/cross_family_judge.py:_preflight_model_available shape so a missing
# qwen2.5:3b surfaces a single operator WARNING instead of silent four-zero
# counters forever.
_triage_model_present: bool | None = None


def _preflight_triage_model(model: str) -> bool:
    """Return True iff `ollama list` stdout mentions `model`. Cached.

    Any subprocess failure (missing binary, non-zero rc, timeout) -> False
    + cached False (we don't re-probe every email — the model isn't going to
    magically appear mid-run).
    """
    global _triage_model_present
    if _triage_model_present is not None:
        return _triage_model_present
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        present = result.returncode == 0 and model in (result.stdout or "")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        present = False
    _triage_model_present = present
    if not present:
        logger.warning(
            "IMAP_TRIAGE: model %s not available via `ollama list` — "
            "triage disabled until `ollama pull %s`",
            model, model,
        )
    return present


async def _triage_classify(
    subject: str,
    body: str,
    model: str = "qwen2.5:3b",
    host: Optional[str] = None,
) -> Optional[str]:
    """Classify an email as one of {newsletter, support, personal, action}.

    Fail-soft contract: ANY exception, non-2xx, or unknown label -> return None.
    Caller persists the result in raw_posts.email_class (NULL = unclassified).
    10s timeout. Env reads happen inside the function (no module-level binding).
    """
    ollama_host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    url = f"{ollama_host}/api/generate"
    prompt = (
        "Classify this email as exactly one of: newsletter, support, personal, action.\n"
        f"Subject: {subject}\n"
        f"Body: {body[:800]}\n"
        "Answer with one word only."
    )
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status < 200 or resp.status >= 300:
                    return None
                data = await resp.json()
        raw = (data or {}).get("response", "")
        if not isinstance(raw, str):
            return None
        # Normalise: strip whitespace + punctuation, lowercase, take first token.
        cleaned = _TRIAGE_PUNCT_RE.sub(" ", raw).strip().lower()
        first = cleaned.split()[0] if cleaned else ""
        return first if first in _TRIAGE_LABELS else None
    except Exception:
        # Fail-soft: any error (network, parse, attribute) -> None.
        logger.debug("triage_classify failed; returning None (fail-soft)", exc_info=True)
        return None


async def _maybe_classify_for_collect(subject: str, body: str) -> Optional[str]:
    """Env-gated wrapper. Reads IMAP_TRIAGE_MODE INSIDE the function (no module
    binding). Returns None when gate is off (no Ollama call made).

    WR-01 fix (Phase 96): preflight check before the first Ollama call so a
    missing qwen2.5:3b surfaces a single WARNING instead of silent fail-soft
    None forever. Subsequent calls hit the cache.
    """
    if os.environ.get("IMAP_TRIAGE_MODE", "off").lower() != "on":
        return None
    model = os.environ.get("IMAP_TRIAGE_MODEL", "qwen2.5:3b")
    if not _preflight_triage_model(model):
        return None
    return await _triage_classify(subject, body, model=model)


class EmailCollector(BaseCollector):
    platform = "email"

    def __init__(
        self,
        extract_session=None,
        extract_semaphore=None,
        source: Optional[dict] = None,
        account_semaphore: Optional[asyncio.Semaphore] = None,
    ):
        # D-13/D-14 + Phase 11: resolve creds with per-source -> env fallback.
        # D-30: NEVER log resolved password; passed straight to aioimaplib.login().
        # KeyError on missing required env is desired behaviour — the dispatcher
        # gates registration on EMAIL_USER (D-08) for legacy path; per-source
        # path raises KeyError on first fetch (D-16: fail-on-first-fetch).
        host, port, user, password, folder = _resolve_account_creds(source)
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._folder = folder
        # Phase 10 EXTRACT-04 / D-15: pipeline-scoped aiohttp session +
        # asyncio.Semaphore(5) for top-1-URL extraction from newsletter bodies.
        # Default-None preserves backward compat with existing tests that
        # construct EmailCollector() with no positional args.
        self._extract_session = extract_session
        self._extract_semaphore = extract_semaphore
        # D-09/D-11/EMAIL-07: per-account semaphore caps concurrent IMAP connections
        # at 3 (Gmail 24h-lockout prevention). When None (legacy callers pre-Phase-11),
        # no cap is applied — preserves backward compat with existing tests that
        # don't supply a semaphore.
        self._account_semaphore = account_semaphore

    async def collect(self, db: aiosqlite.Connection, source: dict) -> None:
        cursor_str = source.get("last_cursor")
        # D-12 + threat T-04-05: parse to int. String compare is wrong ('9' > '100' lex).
        try:
            last_uid: Optional[int] = int(cursor_str) if cursor_str else None
        except ValueError:
            logger.warning(
                "Invalid last_cursor %r for email source %s — resetting to None",
                cursor_str, source["id"]
            )
            last_uid = None

        # D-13/D-14: build search criterion. UNSEEN on first run; UNSEEN UID {n+1}:*
        # on subsequent runs (strictly newer than last_uid).
        if last_uid is None:
            search_criterion = "UNSEEN"
        else:
            search_criterion = f"UNSEEN UID {last_uid + 1}:*"

        collected_uids: list[int] = []

        # D-09/D-11/EMAIL-07: per-account semaphore caps concurrent IMAP connections
        # at 3. Acquire BEFORE construction so connection count itself is gated
        # (Gmail counts TCP-level connection slots, not just login slots).
        if self._account_semaphore is not None:
            async with self._account_semaphore:
                await self._do_collect_session(
                    db, source, collected_uids, search_criterion
                )
        else:
            await self._do_collect_session(
                db, source, collected_uids, search_criterion
            )

        # D-12: persist last_cursor as str(max(uid)). INTEGER max — string max would be
        # lexicographically wrong ("9" > "100" lex but 9 < 100 numerically).
        if collected_uids:
            await update_source_last_cursor(
                db, source["id"], str(max(collected_uids))
            )

    async def _do_collect_session(
        self,
        db: aiosqlite.Connection,
        source: dict,
        collected_uids: list,
        search_criterion: str,
    ) -> None:
        """
        IMAP session lifecycle (D-31): wait_hello -> login -> select -> uid_search ->
        per-UID fetch+insert -> logout. Mutates `collected_uids` in place (list passed
        by reference) so the caller can advance the cursor after the session ends.

        Extracted from `collect()` so the per-account asyncio.Semaphore can wrap the
        ENTIRE session (TCP connection + login + fetch loop + logout) — Gmail counts
        TCP slots not login slots, so the cap must gate connection establishment too.
        """
        # D-31: mandatory wait_hello_from_server before login.
        # D-04: TLS only — IMAP4_SSL on port 993.
        client = aioimaplib.IMAP4_SSL(host=self._host, port=self._port)
        try:
            await client.wait_hello_from_server()
            await client.login(self._user, self._password)
            await client.select(self._folder)

            _, search_lines = await client.uid_search(search_criterion)
            uids = _parse_uids(search_lines)

            for uid_int in uids:
                # Per-message try/except (D-32): a single malformed email never
                # disrupts the channel-level loop. EMAIL_PASS NEVER logged (D-30).
                uid_str = str(uid_int)
                try:
                    _, fetch_data = await client.uid(
                        "fetch", uid_str, "(RFC822)"
                    )
                    raw = _extract_message_bytes(fetch_data)
                    if raw is None:
                        logger.warning(
                            "Could not extract bytes for uid=%s in folder=%s — skipping",
                            uid_str, self._folder,
                        )
                        continue

                    # D-16: stdlib email parser with policy.default (handles charset).
                    msg = message_from_bytes(raw, policy=policy.default)

                    subject = (msg["Subject"] or "").strip()
                    from_header = (msg["From"] or "").strip()
                    message_id = (msg["Message-ID"] or "").strip()  # D-24: external_id
                    date_header = msg["Date"]

                    # D-26: parse Date header to ISO datetime string. Failures yield None
                    # so insert_raw_post falls back to its "now" default — acceptable for
                    # newsletters where the Date header is malformed (rare).
                    try:
                        published_at = (
                            parsedate_to_datetime(date_header).isoformat()
                            if date_header else None
                        )
                    except (TypeError, ValueError):
                        published_at = None

                    # D-17: extract body with text/plain -> html2text -> sentinel chain.
                    body_raw = _extract_body(msg)
                    # D-18 + D-19: normalise whitespace then truncate to 5000.
                    body_truncated = _normalize_body(body_raw)

                    # D-20: pre-filter on combined subject + " " + body_truncated.
                    combined = subject + " " + body_truncated
                    if len(combined) < MIN_COMBINED_CHARS:
                        # Still record the UID for cursor advance — we processed the
                        # message even if we chose not to insert it. This prevents
                        # short-but-otherwise-valid emails from blocking cursor progress.
                        collected_uids.append(uid_int)
                        continue

                    # D-23: raw_text = subject + "\n\n" + body_truncated
                    raw_text = subject + "\n\n" + body_truncated

                    # D-24..D-29: schema fields.
                    if not message_id:
                        # Without Message-ID we'd risk reinserting on every run — skip.
                        # The cursor still advances so we don't loop on this UID forever.
                        logger.warning(
                            "Message at uid=%s missing Message-ID header — skipping insert",
                            uid_str,
                        )
                        collected_uids.append(uid_int)
                        continue

                    # Phase 10 EXTRACT-04 / D-15: top-1 URL extraction from body.
                    # Both extract_session and extract_semaphore must be present —
                    # default-None on either disables extraction (graceful degradation
                    # for existing tests that don't supply a session).
                    extracted_body: Optional[str] = None
                    extraction_status: Optional[str] = None
                    extracted_at: Optional[str] = None
                    if (
                        self._extract_session is not None
                        and self._extract_semaphore is not None
                    ):
                        candidate_url = extract_first_url(body_truncated)
                        if candidate_url:
                            extracted_body, extraction_status = await extract_url(
                                self._extract_session,
                                candidate_url,
                                self._extract_semaphore,
                            )
                            extracted_at = datetime.datetime.now(
                                datetime.UTC
                            ).isoformat()

                    # Phase 96 IMAP-TRIAGE-01: env-gated 4-way classifier.
                    # Gate off (default) -> None; classifier error -> None.
                    email_class = await _maybe_classify_for_collect(
                        subject, body_truncated
                    )

                    await insert_raw_post(
                        db,
                        source["id"],
                        message_id,                  # D-24
                        raw_text,                    # D-23
                        from_header,                 # D-28: author
                        None,                        # D-25: url is always None
                        platform="email",            # D-27
                        content_type="newsletter",   # D-27
                        published_at=published_at,   # D-26
                        source_score=0,              # D-29
                        extracted_body=extracted_body,         # Phase 10 EXTRACT-04
                        extraction_status=extraction_status,   # Phase 10 EXTRACT-04
                        extracted_at=extracted_at,             # Phase 10 EXTRACT-04
                        email_class=email_class,               # Phase 96 IMAP-TRIAGE-01
                    )
                    collected_uids.append(uid_int)
                except Exception:
                    # D-30 + D-32: include uid + folder ONLY. NEVER include EMAIL_PASS,
                    # never use the credential variables in any log call.
                    logger.exception(
                        "Failed to process email uid=%s folder=%s — continuing",
                        uid_str, self._folder,
                    )

            await client.logout()
        except Exception:
            # D-33: connection errors propagate to collect_all's per-source try/except.
            # logger.exception in collect_all logs platform + target only — no creds leak.
            try:
                await client.logout()
            except Exception:
                pass
            raise
