import json
import logging
from typing import Any, Callable, Awaitable

import aiohttp

logger = logging.getLogger(__name__)

EDITOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_full_article",
            "description": "Fetch the full article body from a URL. Use when a post preview is too short to write a quality summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Article URL to fetch"}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_related_posts",
            "description": "Find recently curated posts in the same category for cross-referencing or context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Category (ai, crypto, startup, psychology, science, fitness, fintech, other)"},
                    "days": {"type": "integer", "description": "How many days back to search"},
                },
                "required": ["category", "days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_project_context",
            "description": "Get the user's project description by slug to write a better Связь: line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "slug": {"type": "string", "description": "Project slug, e.g. 'project-a', 'project-b', 'project-c', 'project-d'"},
                },
                "required": ["slug"],
            },
        },
    },
]


async def chat_with_tools(
    host: str,
    model: str,
    system: str,
    user_msg: str,
    tools: list[dict],
    tool_dispatcher: Callable[[str, dict], Awaitable[Any]],
    max_rounds: int = 3,
) -> str:
    messages: list[dict] = [{"role": "user", "content": user_msg}]
    last_assistant_content = ""

    for _ in range(max_rounds):
        payload = {
            "model": model,
            "system": system,
            "messages": messages,
            "tools": tools,
            "stream": False,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{host}/api/chat", json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    sys_len = len(system or "")
                    msg_len = sum(len(m.get("content", "")) for m in messages)
                    logger.error(
                        "ollama /api/chat %s — body=%s sys_chars=%d msgs_chars=%d tools=%d msg_count=%d",
                        resp.status, body[:500], sys_len, msg_len, len(tools), len(messages),
                    )
                    resp.raise_for_status()
                data = await resp.json()

        message = data.get("message", {})
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            return message.get("content", "")

        last_assistant_content = message.get("content", "")
        messages.append({
            "role": "assistant",
            "content": last_assistant_content,
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            try:
                result = await tool_dispatcher(name, args)
            except Exception as e:
                result = f"[tool error: {e}]"
                logger.debug("Tool %r raised: %s", name, e)
            messages.append({"role": "tool", "content": str(result)})

    return last_assistant_content
