"""Generate audio digest from latest digest markdown using Edge TTS (free, no API key).

Voice: ru-RU-DmitryNeural by default. Override with VOICE_DIGEST_VOICE env var.
Output: src/web/static/audio/YYYY-MM-DD.mp3

Cleans markdown to plain text: strips HTML comments, link formatting, bold/italic.
Truncates to _MAX_CHARS to keep audio under ~6 min.
"""
from __future__ import annotations

import datetime
import os
import re
from pathlib import Path

import edge_tts

_AUDIO_DIR = Path(__file__).resolve().parent.parent / "web" / "static" / "audio"
_DEFAULT_VOICE = "ru-RU-DmitryNeural"
_MAX_CHARS = 8000  # ~5-6 min at ~140 wpm TTS speed


def _clean(md: str) -> str:
    """Convert digest markdown to plain text for TTS."""
    # Remove HTML comments: <!-- item-id:N -->, <!-- post-id:N -->
    text = re.sub(r'<!--[^>]+-->', '', md)
    # Category h2 headers: ## 🤖 Категория → "Раздел: Категория."
    text = re.sub(
        r'^##\s+\S*\s+(.*?)$',
        lambda m: f'\nРаздел: {m.group(1).strip()}.',
        text, flags=re.MULTILINE,
    )
    # Topic h3 sub-headers: ### Тема → newline only (skip label, keep flow)
    text = re.sub(r'^###.*$', '', text, flags=re.MULTILINE)
    # Channel bold wrapper: **[Channel]** ([→ Оригинал](url)): → "Channel: "
    text = re.sub(r'\*\*\[([^\]]+)\]\*\*\s*\([^)]+\)\s*:\s*', r'\1: ', text)
    # Strip italic markers around "Интересно потому что: ..."
    text = re.sub(r'\*Интересно потому что:\s*', 'Интересно потому что: ', text)
    text = re.sub(r'\*\s*$', '', text, flags=re.MULTILINE)
    # Markdown links [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Bold **text** → text
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    # Italic *text* → text
    text = re.sub(r'\*([^*\n]+)\*', r'\1', text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


async def generate_audio(digest_md: str, output_date: str | None = None) -> Path:
    """Generate mp3 from digest markdown. Returns absolute path to audio file."""
    _AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    date_str = output_date or datetime.date.today().isoformat()
    out_path = _AUDIO_DIR / f"{date_str}.mp3"

    text = _clean(digest_md)
    if not text:
        raise ValueError("Digest is empty after cleaning — nothing to speak.")

    if len(text) > _MAX_CHARS:
        # Truncate at last sentence boundary near the limit
        cut = text.rfind('. ', 0, _MAX_CHARS)
        text = (text[:cut + 1] if cut > 0 else text[:_MAX_CHARS]) + " Дайджест сокращён."

    voice = os.environ.get("VOICE_DIGEST_VOICE", _DEFAULT_VOICE)
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))
    return out_path


def get_audio_url_for_date(date: datetime.date | None = None) -> str | None:
    """Return /static/audio/YYYY-MM-DD.mp3 if file exists, else None."""
    d = date or datetime.date.today()
    path = _AUDIO_DIR / f"{d.isoformat()}.mp3"
    return f"/static/audio/{d.isoformat()}.mp3" if path.exists() else None
