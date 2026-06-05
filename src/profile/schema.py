"""ProfileTiers dataclass + token-budget cap (D-12, D-14) + profile-body renderer.

D-12 source-order rule: first 15 HIGH + first 25 MEDIUM in extraction order.
D-14: NO cross-tier promotion — HIGH overflow is dropped, NOT bumped to MEDIUM.
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class ProfileTiers:
    """Structured representation of an assembled profile.

    high: top-priority topics (curator scores 75-100 if matched)
    medium: secondary topics (curator scores 50-75)
    reject_phrases: phrases that drop curator score to 0-30 (My_Voice_Profile NEVER Use list)
    raw_prose: filename → full file content for advisory files (Fact_dossier full prose,
               Goals, Telegram_*, Persona, Strategy) — embedded as supplementary context
    """
    high: List[str] = field(default_factory=list)
    medium: List[str] = field(default_factory=list)
    reject_phrases: List[str] = field(default_factory=list)
    raw_prose: Dict[str, str] = field(default_factory=dict)


def cap_tiers(
    tiers: ProfileTiers,
    high_cap: int = 15,
    medium_cap: int = 25,
) -> ProfileTiers:
    """D-12 source-order cap with D-14 no-cross-tier-promotion.

    Deduplicates within tier (case-insensitive on stripped lower form).
    Excludes HIGH items from MEDIUM (avoid duplication when same phrase appears in both source streams).
    """
    seen: set[str] = set()
    high: list[str] = []
    for item in tiers.high:
        norm = item.lower().strip()
        if norm in seen:
            continue
        seen.add(norm)
        high.append(item)
        if len(high) >= high_cap:
            break

    medium_seen = set(seen)  # exclude HIGH items from MEDIUM
    medium: list[str] = []
    for item in tiers.medium:
        norm = item.lower().strip()
        if norm in medium_seen:
            continue
        medium_seen.add(norm)
        medium.append(item)
        if len(medium) >= medium_cap:
            break

    return ProfileTiers(
        high=high,
        medium=medium,
        reject_phrases=list(tiers.reject_phrases),
        raw_prose=dict(tiers.raw_prose),
    )


def render_profile(tiers: ProfileTiers) -> str:
    """Render ProfileTiers as a profile string suitable for embedding in curator prompt.

    Format mirrors v1 hardcoded text shape so curator scoring instructions stay parallel:
    - "User: [intro from CURATOR_USER_INTRO]" (intro line, env-overridable)
    - HIGH relevance topics (score 75-100): bulleted list
    - MEDIUM relevance topics (score 50-75): bulleted list
    - REJECT phrases (score 0-30): bulleted list
    - Raw prose context (embedded under "## Additional Context")

    Set CURATOR_USER_INTRO env var to restore your personal intro line, e.g.:
      CURATOR_USER_INTRO="User: Jane Doe. Senior Engineer at Acme Corp."
    """
    parts: list[str] = []
    # Read env var INSIDE function (not module-level) to allow test monkeypatching
    parts.append(os.environ.get("CURATOR_USER_INTRO", "User: [your name and context here]."))
    parts.append("")

    if tiers.high:
        parts.append("HIGH relevance topics (score 75-100):")
        for item in tiers.high:
            parts.append(f"- {item}")
        parts.append("")

    if tiers.medium:
        parts.append("MEDIUM relevance topics (score 50-75):")
        for item in tiers.medium:
            parts.append(f"- {item}")
        parts.append("")

    if tiers.reject_phrases:
        parts.append("REJECT — score 0-30 immediately if post contains:")
        for phrase in tiers.reject_phrases:
            parts.append(f'- "{phrase}"')
        parts.append("")

    if tiers.raw_prose:
        parts.append("## Additional Context (raw prose from profile sources)")
        parts.append("")
        for filename, content in tiers.raw_prose.items():
            parts.append(f"### {filename}")
            parts.append(content.strip())
            parts.append("")

    return "\n".join(parts).strip() + "\n"
