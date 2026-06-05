"""Heuristic Markdown extractors for Phase 7 profile loading.

Per RESEARCH.md §1: heterogeneous Markdown input (writing_guide.md tables without pipes,
fact_dossier.md inline-nested bullets) breaks AST parsers. Regex extractors are robust to malformed
input — they return [] on no-match instead of raising, so D-17 fallback is triggered ONLY by
genuine I/O exceptions, not by parser pickiness.
"""
import logging
import re

logger = logging.getLogger(__name__)


# --- Module-level compiled patterns (leading underscore + _RE suffix per project convention) ---

# fact_dossier.md § 1. CORE IDENTITY → Content Pillars
# Matches both H3 form (### Content Pillars) and bold-inline form (* **Content Pillars:**)
_CONTENT_PILLARS_RE = re.compile(
    r"(?:###\s*|\*+\s*\*\*)\s*Content\s+Pillars\s*[:\*]*\s*\n?(.*?)(?=^###|^##|^\*\s*\*\*[A-ZА-Я]|\Z)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)

# voice_profile.md § 3. PHRASES → ### Phrases I NEVER Use (italicised entries inside bullets)
_NEVER_USE_SECTION_RE = re.compile(
    r"###\s*Phrases\s+I\s+NEVER\s+Use(.*?)(?=^###|^##|\Z)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
# Italicised phrases inside the section: *"phrase"* OR *phrase*
_ITALICISED_PHRASE_RE = re.compile(r'\*"?([^*"\n]{2,80}?)"?\*')

# Bullet-list items: lines starting with "- " or "* " (after stripping leading whitespace)
_BULLET_RE = re.compile(r"^\s*[-*]\s+(.+?)\s*$", re.MULTILINE)

# Goals.md current quarter goals (any H2 under "## Q" prefix)
_GOALS_SECTION_RE = re.compile(
    r"##\s*Q\d\s+\d{4}\s*\n(.*?)(?=^##|\Z)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)

# Telegram_Channel_Insights.md numbered list under "## ТОП-..." or "## N. ТОП-..."
_TG_TOP_SECTION_RE = re.compile(
    r"##\s*(?:\d+\.\s*)?ТОП[-\s]*\d+.*?\n(.*?)(?=^##|\Z)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
# Numbered list items: "1. content"
_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+(.+?)\s*$", re.MULTILINE)


def _extract_section_bullets(text: str, section_re: re.Pattern) -> list[str]:
    """Return bullet items from a Markdown section. Returns [] if section absent."""
    m = section_re.search(text)
    if not m:
        return []
    body = m.group(1)
    return [b.strip() for b in _BULLET_RE.findall(body) if b.strip()]


def _extract_section_numbered(text: str, section_re: re.Pattern) -> list[str]:
    """Return numbered-list items from a Markdown section. Returns [] if section absent."""
    m = section_re.search(text)
    if not m:
        return []
    body = m.group(1)
    return [b.strip() for b in _NUMBERED_RE.findall(body) if b.strip()]


def extract_high(files_text: dict[str, str]) -> list[str]:
    """Extract HIGH-tier topics in source-order (D-12).

    Source order:
    1. fact_dossier.md § Content Pillars
    2. goals.md (current quarter)
    3. content_persona.md (empirical HIGH, e.g. channel insights) ТОП-N (empirical HIGH)
    4. Content_Strategy_2026.md ТРИ СТАВКИ (if present)
    """
    items: list[str] = []
    fd = files_text.get("fact_dossier.md", "")
    items.extend(_extract_section_bullets(fd, _CONTENT_PILLARS_RE))
    goals = files_text.get("goals.md", "")
    items.extend(_extract_section_bullets(goals, _GOALS_SECTION_RE))
    tg_ins = files_text.get("content_persona.md", "")
    items.extend(_extract_section_numbered(tg_ins, _TG_TOP_SECTION_RE))
    cs = files_text.get("content_strategy.md", "")
    if cs:
        # Best-effort: any bullet under any "## ТРИ СТАВКИ" or similar
        bets_re = re.compile(r"##\s*(?:\d+\.\s*)?ТРИ\s+СТАВКИ.*?\n(.*?)(?=^##|\Z)", re.MULTILINE | re.DOTALL | re.IGNORECASE)
        items.extend(_extract_section_bullets(cs, bets_re))
    logger.debug("extract_high: %d items from %d files", len(items), len(files_text))
    return items


def extract_medium(files_text: dict[str, str]) -> list[str]:
    """Extract MEDIUM-tier topics — non-CONTENT_PILLARS bullets reachable via heuristic.

    Sources: recurring_frames.md (КЛЮЧЕВЫЕ ЛИЧНЫЕ ФРЕЙМВОРКИ) +
    content_persona.md fallback bullets.
    """
    items: list[str] = []
    rf = files_text.get("recurring_frames.md", "")
    if rf:
        frames_re = re.compile(
            r"##\s*КЛЮЧЕВЫЕ\s+ЛИЧНЫЕ.*?\n(.*?)(?=^##|\Z)",
            re.MULTILINE | re.DOTALL | re.IGNORECASE,
        )
        items.extend(_extract_section_bullets(rf, frames_re))
    persona = files_text.get("content_persona.md", "")
    if persona:
        # Any H2 → bullets — best-effort
        for h2_match in re.finditer(r"##\s+(.+?)\n(.*?)(?=^##|\Z)", persona, re.MULTILINE | re.DOTALL):
            items.extend(b.strip() for b in _BULLET_RE.findall(h2_match.group(2)) if b.strip())
    logger.debug("extract_medium: %d items", len(items))
    return items


def extract_reject_phrases(files_text: dict[str, str]) -> list[str]:
    """Extract reject-tier phrases from voice_profile.md § Phrases I NEVER Use.

    Format: italicised entries (*"phrase"* or *phrase*) inside the NEVER Use section.
    """
    vp = files_text.get("voice_profile.md", "")
    if not vp:
        return []
    section = _NEVER_USE_SECTION_RE.search(vp)
    if not section:
        return []
    phrases = _ITALICISED_PHRASE_RE.findall(section.group(1))
    cleaned = [p.strip() for p in phrases if p.strip()]
    logger.debug("extract_reject_phrases: %d phrases", len(cleaned))
    return cleaned


def extract_raw_prose(files_text: dict[str, str]) -> dict[str, str]:
    """Embed advisory-file content as raw prose for the profile body.

    Per D-02: fact_dossier.md full prose, goals.md, content_persona.md,
    recurring_frames.md, and any other advisory files
    are embedded as supplementary raw prose, not as scoring tiers.
    """
    # Return all advisory files that were loaded (keys are whatever filenames were in CRITICAL+ADVISORY)
    advisory = tuple(files_text.keys())
    return {name: files_text[name] for name in advisory if name in files_text}
