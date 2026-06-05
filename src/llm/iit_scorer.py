"""IIT-inspired submodular post selector.

Greedy selection that minimises redundancy across curated posts.
Instead of top-N by score alone, selects the max-Phi subset:
posts that together maximise relevance coverage with minimal pairwise overlap.

Formula: marginal_gain(post) = relevance_score - lambda * 100 * max_jaccard(post, selected)
Greedy: at each step add the post with highest positive marginal gain.
"""
from __future__ import annotations

import os
import re
from collections import Counter


def _word_freq(text: str) -> Counter:
    return Counter(re.findall(r"\w+", text.lower()))


def _jaccard(a: Counter, b: Counter) -> float:
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union > 0 else 0.0


def submodular_select(
    posts: list[dict],
    max_posts: int | None = None,
    lam: float = 0.5,
) -> list[dict]:
    """Greedy submodular selection. Returns up to max_posts diverse posts.

    Args:
        posts: list of post dicts with 'id', 'raw_text', 'relevance_score'.
        max_posts: cap (default from IIT_MAX_POSTS env var, fallback 25).
        lam: redundancy penalty weight. 0 = ignore overlap; 1 = full penalty.

    Returns a list no longer than max_posts, ordered by selection rank.
    When len(posts) <= max_posts the input is returned unchanged (fast path).
    """
    if max_posts is None:
        max_posts = int(os.environ.get("IIT_MAX_POSTS", "25"))

    if len(posts) <= max_posts:
        return posts

    scored = sorted(posts, key=lambda p: p.get("relevance_score") or 0, reverse=True)

    selected: list[dict] = []
    selected_freqs: list[Counter] = []

    for post in scored:
        if len(selected) >= max_posts:
            break
        freq = _word_freq(post.get("raw_text", ""))
        if selected_freqs:
            max_sim = max(_jaccard(freq, sf) for sf in selected_freqs)
        else:
            max_sim = 0.0
        score = post.get("relevance_score") or 0
        marginal = score - lam * 100.0 * max_sim
        if marginal > 0 or not selected:
            selected.append(post)
            selected_freqs.append(freq)

    # Fill remaining slots from highest-scored unselected posts so we never
    # return fewer posts than max_posts when the penalty was too aggressive.
    if len(selected) < max_posts:
        selected_ids = {p["id"] for p in selected}
        for post in scored:
            if len(selected) >= max_posts:
                break
            if post["id"] not in selected_ids:
                selected.append(post)

    return selected
