"""Phase 50 (TRUNC-03): YouTube chapter-aware excerpt composition.

For long-transcript YouTube posts (>200K chars curated rate is 31.8% vs 69.8%
on <50K — likely truncation casualties), compose a chapter-aware excerpt
instead of using raw transcript prefix.

Source order:
1. Video description `M:SS Title` lines (YouTube convention)
2. Transcript heuristic: ALL-CAPS section headers in transcript itself
3. None → caller uses raw transcript

Env gate: YOUTUBE_EXCERPT_MODE=raw|chapter (default raw).

Usage from collector:
    body = select_yt_body(description, transcript)
    # body is either raw transcript (default) or composed excerpt (mode=chapter)
"""
from __future__ import annotations

import os
import re

_MIN_CHAPTERS = 3
_MAX_EXCERPT_CHARS = 2000
_CONTEXT_CHARS_PER_CHAPTER = 200

# `0:00 Intro` / `1:23:45 Deep dive` patterns (optional trailing punct)
_DESC_CHAPTER_RE = re.compile(
    r"^\s*(\d{1,2}(?::\d{2}){1,2})\s+[-•—:]?\s*(.+?)\s*$",
    re.MULTILINE,
)

# Heuristic: ALL-CAPS section headers (≥3 caps in a row, at least 5 chars total)
_ALLCAPS_HEADER_RE = re.compile(
    r"(?:^|[\n.!?]\s+)([A-Z][A-Z\s]{4,40}[A-Z])(?=[:.\n]|\s+[a-z])"
)


def _parse_description_chapters(description: str) -> list[tuple[str, str]]:
    """Extract (timestamp, title) tuples from a description block.

    YouTube convention: timestamps on their own line followed by the title.
    Skips lines that look like timestamps but follow text on the same line.
    """
    if not description:
        return []
    out: list[tuple[str, str]] = []
    for m in _DESC_CHAPTER_RE.finditer(description):
        ts = m.group(1)
        title = m.group(2).strip()
        # Skip URL-ish or noise lines
        if not title or title.startswith("http"):
            continue
        if len(title) > 200:
            continue  # likely captured a wall of text, not a chapter title
        out.append((ts, title))
    return out


def _heuristic_allcaps_headers(transcript: str) -> list[str]:
    """Heuristic fallback: find ALL-CAPS section-header-like substrings in transcript."""
    if not transcript:
        return []
    matches = _ALLCAPS_HEADER_RE.findall(transcript)
    seen = set()
    out: list[str] = []
    for m in matches:
        clean = m.strip()
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def _format_excerpt_from_chapters(
    chapter_titles: list[str],
    transcript: str,
    max_chars: int = _MAX_EXCERPT_CHARS,
    context_per_chapter: int = _CONTEXT_CHARS_PER_CHAPTER,
) -> str:
    """Compose excerpt with chapter title + brief context after each title."""
    parts: list[str] = []
    remaining = max_chars
    for i, title in enumerate(chapter_titles, 1):
        header = f"Chapter {i}: {title}"
        # Try to find title in transcript and pull short context after it
        context = ""
        if transcript:
            pos = transcript.find(title)
            if pos >= 0:
                context = transcript[pos + len(title): pos + len(title) + context_per_chapter].strip()
        chunk = header + ("\n" + context if context else "")
        if len(chunk) + 2 > remaining:
            chunk = chunk[: remaining - 2]
        parts.append(chunk)
        remaining -= len(chunk) + 2
        if remaining <= 0:
            break
    return "\n\n".join(parts)


def compose_excerpt(description: str, transcript: str) -> str | None:
    """Try description chapters first; ALL-CAPS heuristic fallback; None if neither.

    Returns the composed excerpt string OR None when no chapter structure found.
    """
    # Tier 1: description chapters
    desc_chapters = _parse_description_chapters(description or "")
    titles = [title for _, title in desc_chapters]
    if len(titles) >= _MIN_CHAPTERS:
        return _format_excerpt_from_chapters(titles, transcript or "")

    # Tier 2: ALL-CAPS heuristic in transcript
    heuristic = _heuristic_allcaps_headers(transcript or "")
    if len(heuristic) >= _MIN_CHAPTERS:
        return _format_excerpt_from_chapters(heuristic, transcript or "")

    return None


def select_yt_body(description: str, transcript: str) -> str:
    """Env-aware body selector. Returns raw transcript (default) or composed excerpt.

    Mode set via `YOUTUBE_EXCERPT_MODE` env: `raw` (default) or `chapter`.
    Fail-soft: if mode=chapter but no chapter structure found, returns raw.
    """
    mode = os.environ.get("YOUTUBE_EXCERPT_MODE", "raw").lower().strip()
    if mode == "chapter":
        excerpt = compose_excerpt(description, transcript)
        if excerpt:
            return excerpt
    # Default / fallback
    return transcript or ""
