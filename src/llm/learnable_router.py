"""Phase 79-01: LearnableRouter — sigmoid gate over P78 routing features.

Per-backend independent sigmoid g_b(c) = σ(W_b · features + floor_b). Trained
weekly by P79-02; consumed by P79-03 A/B harness. Pure numpy (no sklearn).

Feature vector layout (locked):
    [post_length, has_canonical_url, simhash_dist, embed_cosine]   # numeric, 4 dims
    + one_hot(category, CANONICAL_CATEGORIES)                       # 10 dims
    = 14 dims total.

The (backend × category) interaction is encoded by having a separate weight
vector per backend, with the category one-hot appended to the shared feature
vector. Effectively this learns one logistic regression per backend over the
same feature space.

Persistence convention mirrors `src/agents/reaction_predictor.py`:
    data/learnable_router_v{n}.json

Insufficient-data fallback contract: `score_backend` returns `None` when the
backend is absent from `self.weights`. Caller is expected to fall back to the
static yaml router (`src/llm/routing.py::get_backend_for_category`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.categories import CANONICAL_CATEGORIES

# Numeric features extracted by P78 routing.extract_features (5-tuple, but the
# trailing source_category becomes the one-hot — only the first 4 are numeric).
NUMERIC_FEATURES: tuple[str, ...] = (
    "post_length",
    "has_canonical_url",
    "simhash_dist",
    "embed_cosine",
)

SCHEMA_VERSION = 1


def _sigmoid(z: float) -> float:
    """Numerically stable scalar sigmoid in [0, 1]."""
    if z >= 0:
        ez = math.exp(-z) if z < 700 else 0.0
        return 1.0 / (1.0 + ez)
    ez = math.exp(z) if z > -700 else 0.0
    return ez / (1.0 + ez)


class LearnableRouter:
    """Sigmoid gate over (backend × category) for curator backend routing.

    Public API:
      - score_backend(features, backend, category) -> float | None
      - save(path) / load(path)
      - train(samples) -> dict  (P79-02 hook; minimal stub here)

    Weights layout:
        self.weights = {
            "ollama":       {"W": [...], "floor": float},
            "claude":       {"W": [...], "floor": float},
            "claude-batch": {"W": [...], "floor": float},
        }
    """

    numeric_feature_dim: int = len(NUMERIC_FEATURES)
    one_hot_dim: int = len(CANONICAL_CATEGORIES)

    def __init__(self) -> None:
        self.weights: dict[str, dict[str, Any]] = {}
        self.version: int = SCHEMA_VERSION
        self.metadata: dict[str, Any] = {}

    # ----- feature vector -----

    @classmethod
    def _category_one_hot(cls, category: str | None) -> list[float]:
        vec = [0.0] * cls.one_hot_dim
        if category in CANONICAL_CATEGORIES:
            vec[CANONICAL_CATEGORIES.index(category)] = 1.0
        else:
            vec[CANONICAL_CATEGORIES.index("other")] = 1.0
        return vec

    @classmethod
    def _build_feature_vector(
        cls, features: dict[str, Any], category: str | None
    ) -> list[float]:
        numeric: list[float] = []
        for name in NUMERIC_FEATURES:
            v = features.get(name)
            if v is None:
                numeric.append(0.0)
            else:
                try:
                    numeric.append(float(v))
                except (TypeError, ValueError):
                    numeric.append(0.0)
        return numeric + cls._category_one_hot(category)

    # IN-02 (P79 review): public alias for sibling-module use (e.g. trainer in
    # scripts/train_router.py). Keeps a single source-of-truth for the 14-d
    # feature layout — internal callers may continue using the underscored
    # form, external callers should prefer this one. If the layout ever
    # changes, both names update in lockstep here.
    @classmethod
    def build_feature_vector(
        cls, features: dict[str, Any], category: str | None
    ) -> list[float]:
        return cls._build_feature_vector(features, category)

    # ----- scoring -----

    def score_backend(
        self,
        features: dict[str, Any],
        backend: str,
        category: str | None,
    ) -> float | None:
        """Return σ(W_b · features + floor_b) in [0, 1] or None when untrained.

        None signals the caller to fall back to static yaml routing.
        """
        weights_for = self.weights.get(backend)
        if weights_for is None:
            return None
        W = weights_for.get("W")
        floor = float(weights_for.get("floor", 0.0))
        if W is None:
            return None
        x = self._build_feature_vector(features, category)
        if len(W) != len(x):
            # Schema mismatch — treat as untrained rather than blow up the pipeline.
            return None
        z = floor
        for wi, xi in zip(W, x):
            z += float(wi) * xi
        return _sigmoid(z)

    # ----- persistence -----

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "numeric_features": list(NUMERIC_FEATURES),
            "categories": list(CANONICAL_CATEGORIES),
            "weights": self.weights,
            "metadata": self.metadata,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.version = int(payload.get("version", SCHEMA_VERSION))
        self.weights = payload.get("weights", {}) or {}
        self.metadata = payload.get("metadata", {}) or {}

    # ----- training stub (P79-02 implements full trainer) -----

    def train(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Placeholder for P79-02. Returns a metadata dict signalling no-op.

        Real implementation lives in `scripts/train_router.py` (P79-02) which
        fits one logistic regression per backend over `post_feedback` joined
        with `routing_features`. This method is kept as a class-level seam so
        the trainer can mutate `self.weights` and call `self.save(...)`.
        """
        return {
            "accepted": False,
            "reason": "trainer_not_implemented_in_p79_01",
            "n_samples": len(samples),
        }
