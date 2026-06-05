"""Phase 108 Plan 02 — /chat page router + /chat/stream SSE endpoint.

Routes:
    GET /chat          — renders chat.html template
    GET /chat/stream   — Server-Sent Events stream (token / citations / done / error)

Design constraints (per PLAN 108-02):
- Sync Ollama calls (embed / retrieve / generate) wrapped in run_in_executor
  so the event loop is never blocked (CHAT-02).
- Each SSE token is plain text; client uses createTextNode (CHAT-05 / T-108-06 XSS).
- Category query param validated against CANONICAL_CATEGORIES (T-108-07).
- DB path read inside _generate (never module-level — CLAUDE.md gotcha).
- Empty-corpus guard: count_embedded_posts == 0 -> error event, return (T-108-09).
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates

from src.categories import CANONICAL_CATEGORIES
from src.llm.embedding import get_embedding
from src.web.chat_retrieval import (
    build_rag_prompt,
    count_embedded_posts,
    retrieve_top_k,
    stream_ollama,
)

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _sse(event: str, data: str) -> str:
    """Format a Server-Sent Event frame."""
    return f"event: {event}\ndata: {data}\n\n"


@router.get("/chat")
async def chat_page(request: Request):
    """Render the /chat UI page."""
    return templates.TemplateResponse(
        request,
        "chat.html",
        {"categories": list(CANONICAL_CATEGORIES)},
    )


@router.get("/chat/stream")
async def chat_stream(
    q: str,
    category: str = "",
    from_date: str = "",
    to_date: str = "",
):
    """Stream LLM answer tokens + citations via SSE.

    Events emitted (in order):
        token       — html-escaped LLM token fragment
        citations   — JSON array [{channel, url}, ...]
        done        — empty data, stream complete
        error       — human-readable error message (ends stream early)
    """
    async def _generate():
        loop = asyncio.get_running_loop()

        # DB path read inside the generator (never at module level — CLAUDE.md)
        db_path = os.environ.get("CURATOR_DB_PATH", "curator.db")

        # T-108-09: empty-corpus guard FIRST — no embeddings means no retrieval
        try:
            n = await loop.run_in_executor(None, count_embedded_posts, db_path)
        except Exception as exc:
            yield _sse(
                "error",
                "Не удалось подключиться к базе данных. Подробности в логах.",
            )
            return

        if n == 0:
            yield _sse(
                "error",
                (
                    "Корпус постов ещё не проиндексирован. "
                    "Запустите пайплайн: .venv/Scripts/python.exe run_pipeline.py"
                ),
            )
            return

        # Embed the query (sync Ollama call — run in executor)
        query_vec = await loop.run_in_executor(None, get_embedding, q)
        if query_vec is None:
            yield _sse("error", "Не удалось обработать запрос (Ollama недоступен).")
            return

        # T-108-07: validate category param against allow-list
        safe_category = category if category in CANONICAL_CATEGORIES else ""

        # Retrieve top-K chunks
        try:
            chunks = await loop.run_in_executor(
                None,
                lambda: retrieve_top_k(
                    query_vec, db_path, 5, safe_category, from_date, to_date
                ),
            )
        except Exception as exc:
            safe_msg = str(exc).replace("\n", " ").replace("\r", " ")[:200]
            yield _sse("error", f"Ошибка при поиске постов: {safe_msg}")
            return

        if not chunks:
            yield _sse("error", "Нет подходящих постов по запросу.")
            return

        # Build injection-guarded RAG prompt
        prompt = build_rag_prompt(q, chunks)

        # Collect tokens from sync Ollama generator via run_in_executor
        # (stream_ollama is a sync blocking generator — collect into list)
        try:
            tokens: list[str] = await loop.run_in_executor(
                None, lambda: list(stream_ollama(prompt))
            )
        except Exception as exc:
            safe_msg = str(exc).replace("\n", " ").replace("\r", " ")[:200]
            yield _sse("error", f"Ошибка при генерации ответа: {safe_msg}")
            return

        # T-108-06: tokens are plain text; client uses createTextNode (no innerHTML)
        for token in tokens:
            yield _sse("token", token)

        # Citations event: channel + original post URL per chunk
        citations = [{"channel": c["channel"], "url": c["url"]} for c in chunks]
        yield _sse("citations", json.dumps(citations, ensure_ascii=False))

        yield _sse("done", "")

    return StreamingResponse(_generate(), media_type="text/event-stream")
