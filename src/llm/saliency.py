"""Attention Schema Theory saliency tagger.

Three-slot attention reason per digest entry:
  novelty   — how different this post is from other posts in the digest
              (high/medium/low based on pairwise Jaccard dissimilarity)
  relevance — density of HIGH-tier curator topic keywords in the text
              (high/medium/low)
  urgency   — age of the post at digest-creation time
              (today/<6h, recent/<24h, stale/>24h)

Models WHY the brain should attend to each entry:
  novelty   → surprise-driven attention; novel inputs override habitual processing
  relevance → top-down interest alignment from known high-value topics
  urgency   → temporal salience; time-sensitive content decays rapidly
"""
from __future__ import annotations

import datetime
import re
from collections import Counter

_HIGH_KEYWORDS: set[str] = {
    # AI/LLM
    "agent", "agents", "llm", "llms", "gpt", "claude", "openai", "anthropic",
    "deepmind", "langchain", "multiagent", "rag", "embedding", "transformer",
    "diffusion", "inference", "finetuning", "finetune", "rlhf", "reasoning",
    # CS/tech
    "api", "benchmark", "latency", "throughput", "architecture",
    # fitness/sport
    "bjj", "jujitsu", "jiu", "strength", "hypertrophy", "vo2", "hrv",
    # crypto/fintech
    "defi", "l2", "rollup", "blockchain", "cbdc", "ruble", "рубль",
    # cog sci / neuro
    "cognitive", "neuroscience", "cortex", "prefrontal", "dopamine",
    "metacognition", "attention", "memory", "bias",
    # startup
    "startup", "founder", "yc", "saas", "indie", "mrr", "arr",
}


def _word_freq(text: str) -> Counter:
    return Counter(re.findall(r"\w+", text.lower()))


def _jaccard(a: Counter, b: Counter) -> float:
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union > 0 else 0.0


def compute_saliency(
    post: dict,
    other_post_texts: list[str],
) -> dict:
    """Compute three-slot saliency for a single post.

    Args:
        post: dict with 'raw_text' and 'published_at'.
        other_post_texts: raw_text of all other posts in the same digest run.
                          Used to compute intra-digest novelty.

    Returns:
        {"novelty": str, "relevance": str, "urgency": str}
    """
    text = post.get("raw_text", "")
    freq = _word_freq(text)
    words = set(freq.keys())

    # --- Novelty: average Jaccard similarity vs other posts in this digest ---
    if other_post_texts:
        other_freqs = [_word_freq(t) for t in other_post_texts[:50]]
        sims = [_jaccard(freq, of) for of in other_freqs]
        avg_sim = sum(sims) / len(sims) if sims else 0.0
    else:
        avg_sim = 0.0

    if avg_sim < 0.05:
        novelty = "high"
    elif avg_sim < 0.15:
        novelty = "medium"
    else:
        novelty = "low"

    # --- Relevance: count of HIGH_KEYWORDS present in post ---
    hits = len(words & _HIGH_KEYWORDS)
    if hits >= 3:
        relevance = "high"
    elif hits >= 1:
        relevance = "medium"
    else:
        relevance = "low"

    # --- Urgency: age of post ---
    published_at = post.get("published_at")
    urgency = "stale"
    if published_at:
        try:
            pub_dt = datetime.datetime.fromisoformat(published_at)
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=datetime.UTC)
            age_h = (datetime.datetime.now(datetime.UTC) - pub_dt).total_seconds() / 3600
            if age_h < 6:
                urgency = "today"
            elif age_h < 24:
                urgency = "recent"
        except (ValueError, OverflowError):
            pass

    return {"novelty": novelty, "relevance": relevance, "urgency": urgency}


def compute_saliency_batch(posts: list[dict]) -> dict[int, dict]:
    """Compute saliency for all posts in a batch, using intra-batch novelty.

    Returns {post_id: saliency_dict}.
    """
    texts = [p.get("raw_text", "") for p in posts]
    result: dict[int, dict] = {}
    for i, post in enumerate(posts):
        other_texts = texts[:i] + texts[i + 1:]
        result[post["id"]] = compute_saliency(post, other_texts)
    return result
