"""Free Energy curator — entropy-based information gain filter.

Replaces the fixed score threshold with a marginal information criterion:
a post passes curation when it contributes positive ΔH to the growing corpus.

Enable with:
    FREE_ENERGY_MODE=1  in .env

When enabled, a post is curated only when BOTH conditions hold:
  1. LLM score >= FREE_ENERGY_MIN_SCORE (default 60)
  2. info_gain(post_text, current_corpus) > 0

The digest grows only while new information is being added. Redundant posts
(high score, same topics as already-curated entries) are rejected.

Implementation: vocabulary entropy of the growing curated corpus.
    H(corpus) = -sum_w  p(w) * log2(p(w))
    ΔH = H(corpus + post) - H(corpus)

A post with entirely new vocabulary → large ΔH.
A post about topics already covered → small or zero ΔH.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _entropy(freq: Counter) -> float:
    total = sum(freq.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in freq.values() if c > 0)


class FreeEnergyTracker:
    """Tracks vocabulary entropy of the growing curated corpus within one pipeline run."""

    def __init__(self, min_score: int | None = None) -> None:
        self._corpus: Counter = Counter()
        if min_score is None:
            min_score = int(os.environ.get("FREE_ENERGY_MIN_SCORE", "60"))
        self.min_score = min_score

    def info_gain(self, text: str) -> float:
        """ΔH from adding this text. Positive = new information added."""
        tokens = _tokenize(text)
        if not tokens:
            return 0.0
        combined = self._corpus + Counter(tokens)
        return _entropy(combined) - _entropy(self._corpus)

    def should_curate(self, score: int, text: str) -> bool:
        """True when score >= min_score AND info_gain > 0 (or corpus is still empty)."""
        if score < self.min_score:
            return False
        if not self._corpus:
            return True
        return self.info_gain(text) > 0

    def add(self, text: str) -> None:
        self._corpus.update(_tokenize(text))

    @property
    def corpus_entropy(self) -> float:
        return _entropy(self._corpus)

    @staticmethod
    def is_enabled() -> bool:
        return os.environ.get("FREE_ENERGY_MODE", "").strip() == "1"
