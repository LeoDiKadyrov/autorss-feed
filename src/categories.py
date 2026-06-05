"""Canonical 10-category enum shared across editor, web/sources, and config loaders.

Order is load-bearing for `src.agents.editor` digest section rendering. The tuple
is frozen — mutation attempts raise. Per D-06, D-07, D-08.
"""

CANONICAL_CATEGORIES: tuple[str, ...] = (
    "ai",
    "crypto",
    "startup",
    "psychology",
    "science",
    "research",
    "fitness",
    "fintech",
    "newsletter",
    "other",
)
