import datetime
import logging
import os

import aiosqlite

from src.collectors.base import BaseCollector
from src.database.client import (
    insert_raw_post,
    update_source_last_cursor,
    update_source_last_message_id,
)
from src.extract import extract_url
from src.extract.triggers import extract_first_url
from src.llm.ollama_vision import describe_image, get_vision_model

logger = logging.getLogger(__name__)


class TelegramCollector(BaseCollector):
    platform = "telegram"

    def __init__(self, tg_client, extract_session=None, extract_semaphore=None):
        # Phase 10 EXTRACT-03: extract_session + extract_semaphore are pipeline-scoped
        # (created in run_pipeline.py main()). Default-None preserves backward compat
        # with existing tests that pass a single tg_client positional.
        self._client = tg_client
        self._extract_session = extract_session
        self._extract_semaphore = extract_semaphore

    async def collect(self, db: aiosqlite.Connection, source: dict) -> None:
        cursor_str = source.get("last_cursor") or (
            str(source["last_message_id"]) if source.get("last_message_id") else None
        )
        try:
            since_id = int(cursor_str) if cursor_str else None
        except ValueError:
            logger.warning(
                "Invalid last_cursor %r for source %s — resetting to None",
                cursor_str, source["id"]
            )
            since_id = None

        vision_model = get_vision_model()
        include_media = vision_model is not None
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

        messages = await self._client.get_new_messages(
            source["target"], since_id=since_id, include_media=include_media
        )

        for msg in messages:
            text = msg.get("text") or ""
            extracted_body: str | None = None
            extraction_status: str | None = None
            extracted_at: str | None = None

            # Vision: describe attached photo when text is absent or very short
            photo_bytes = msg.get("photo_bytes")
            if photo_bytes and vision_model and len(text) < 50:
                description = await describe_image(ollama_host, vision_model, photo_bytes)
                if description:
                    extracted_body = f"[Image description]: {description}"
                    extraction_status = "vision"
                    extracted_at = datetime.datetime.now(datetime.UTC).isoformat()

            # Phase 10 EXTRACT-03 / D-14: URL extraction (only when no vision result)
            if (
                extracted_body is None
                and self._extract_session is not None
                and self._extract_semaphore is not None
                and len(text) <= 200
            ):
                candidate_url = extract_first_url(text)
                if candidate_url:
                    extracted_body, extraction_status = await extract_url(
                        self._extract_session,
                        candidate_url,
                        self._extract_semaphore,
                    )
                    extracted_at = datetime.datetime.now(datetime.UTC).isoformat()

            await insert_raw_post(
                db,
                source["id"],
                str(msg["id"]),
                text,
                msg["author"],
                msg["url"],
                platform="telegram",
                content_type="post",
                published_at=msg.get("date"),
                extracted_body=extracted_body,
                extraction_status=extraction_status,
                extracted_at=extracted_at,
            )

        if messages:
            max_id = max(m["id"] for m in messages)
            await update_source_last_cursor(db, source["id"], str(max_id))
            # Also update last_message_id for backward compatibility with existing tests
            # and any code that still reads this field before migration
            await update_source_last_message_id(db, source["id"], max_id)
            # Mark the channel as read up to max_id in Telegram itself so the
            # operator's unread badge in the Telegram app clears once the
            # curator has ingested the messages. Best-effort: a FloodWait or
            # any other Telethon failure logs + returns False without raising.
            mark_read = getattr(self._client, "mark_read", None)
            if mark_read is not None:
                try:
                    await mark_read(source["target"], max_id)
                except Exception:
                    logger.exception(
                        "mark_read raised for target=%s max_id=%s — continuing",
                        source["target"], max_id,
                    )
