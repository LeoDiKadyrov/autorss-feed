"""Describe images via Ollama vision model (moondream, llava, etc.).

Requires a vision-capable model pulled in Ollama:
    ollama pull moondream   # ~1.7 GB, fast
    ollama pull llava       # ~4.7 GB, more accurate

Enable via env var:
    VISION_MODEL=moondream           # single model
    VISION_MODEL=moondream,llava     # A/B split — random 50/50 per image
"""
from __future__ import annotations

import base64
import logging
import os
import random

import aiohttp

logger = logging.getLogger(__name__)

_PROMPT = (
    "Describe this image in Russian in 2-3 sentences. "
    "Focus on any text, data, charts, key information, or concepts shown. "
    "If it's a meme or screenshot of a tweet, describe its meaning."
)


async def describe_image(host: str, model: str, image_bytes: bytes) -> str | None:
    """Send image bytes to Ollama vision model, return text description.

    Returns None on error (missing model, connection failed, etc.) — caller
    should treat None as 'no description available' and proceed without it.
    Description is prefixed with [Vision/{model}] so A/B results are trackable.
    """
    if not image_bytes:
        return None
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": model,
        "prompt": _PROMPT,
        "images": [b64],
        "stream": False,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{host}/api/generate", json=payload, timeout=aiohttp.ClientTimeout(total=60)
            ) as r:
                if r.status == 404:
                    logger.warning(
                        "Vision model '%s' not found in Ollama. "
                        "Run: ollama pull %s", model, model,
                    )
                    return None
                r.raise_for_status()
                data = await r.json()
                text = (data.get("response") or "").strip()
                if not text:
                    return None
                return f"[Vision/{model}] {text}"
    except Exception:
        logger.exception("ollama_vision describe_image failed for model=%s", model)
        return None


def get_vision_model() -> str | None:
    """Pick a vision model from VISION_MODEL env var.

    Supports comma-separated list for A/B testing — random choice per call:
        VISION_MODEL=moondream,llava  →  50/50 split
    Returns None when env var is unset or empty.
    """
    raw = os.environ.get("VISION_MODEL", "").strip()
    if not raw:
        return None
    models = [m.strip() for m in raw.split(",") if m.strip()]
    if not models:
        return None
    return random.choice(models)
