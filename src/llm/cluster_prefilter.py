"""Cluster posts by bge-m3 embeddings (numpy k-means) for curator pre-filter.

Foundation module for Phase 99: one LLM call per cluster representative (medoid)
instead of per post. Massive cost reduction.

No sklearn dep — numpy-only fallback.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClusterResult:
    medoid_id: int
    member_ids: list[int]
    size: int


# WR-02 fix (Phase 99): expected embedding dimension. Production default is
# 1024 (bge-m3) but ONLY enforced when CHROMA_EMBEDDING_DIM is explicitly set.
# Otherwise we accept any non-empty float32 vector (preserves test fixtures
# using small dims). Per-run dim sanity is still enforced: all vectors in a
# single cluster_posts call must share the same dim — np.vstack would crash
# otherwise.
def _expected_dim() -> int | None:
    raw = os.environ.get("CHROMA_EMBEDDING_DIM")
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _decode_embedding(blob: bytes | None) -> np.ndarray | None:
    """Decode a SQLite BLOB into an np.float32 vector. None on any failure
    OR explicit dimension mismatch (WR-02 fix Phase 99)."""
    if not blob:
        return None
    try:
        # Accept memoryview / bytearray transparently.
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            return None
        v = np.frombuffer(bytes(blob), dtype=np.float32)
    except Exception:
        return None
    if v.size == 0:
        return None
    expected = _expected_dim()
    if expected is not None and v.size != expected:
        return None
    return v


def _kmeans(X: np.ndarray, k: int, seed: int = 0, max_iter: int = 50) -> np.ndarray:
    """Cosine-similarity k-means assuming L2-normalized inputs.

    Returns label vector of length len(X).
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=k, replace=False)
    centroids = X[idx].copy()
    labels = np.full(len(X), -1, dtype=int)
    for _ in range(max_iter):
        sims = X @ centroids.T
        new_labels = sims.argmax(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                c = X[mask].mean(axis=0)
                n = np.linalg.norm(c)
                if n > 0:
                    centroids[j] = c / n
    return labels


def _split_oversized(labels: np.ndarray, max_size: int) -> np.ndarray:
    """Split clusters larger than max_size into chunked sub-clusters with fresh label ids."""
    out = labels.copy()
    next_id = int(out.max()) + 1
    for cid in np.unique(labels):
        members = np.where(out == cid)[0]
        if len(members) > max_size:
            for chunk_start in range(max_size, len(members), max_size):
                chunk = members[chunk_start : chunk_start + max_size]
                out[chunk] = next_id
                next_id += 1
    return out


def cluster_posts(
    posts: list[dict], k: int | None = None, seed: int = 0
) -> list[ClusterResult]:
    """Group posts by embedding similarity; return one ClusterResult per cluster.

    - k=None → ceil(N_embedded / CHROMA_CLUSTER_MAX_SIZE).
    - Posts without embedding → singleton clusters (fail-soft).
    - Oversized clusters (> CHROMA_CLUSTER_MAX_SIZE) are split into chunks.
    - Medoid = member with highest mean cosine similarity to other members.
    - Deterministic given fixed `seed`.
    """
    max_size = int(os.environ.get("CHROMA_CLUSTER_MAX_SIZE", "10"))

    with_emb: list[tuple[int, np.ndarray]] = []
    without_emb: list[int] = []
    # WR-02 fix (Phase 99): when CHROMA_EMBEDDING_DIM is unset, lock to the
    # FIRST observed dim for this call and reject mixed-dim vectors (would
    # otherwise crash np.vstack downstream).
    _batch_dim: int | None = None
    for p in posts:
        v = _decode_embedding(p.get("embedding"))
        if v is None:
            without_emb.append(int(p["id"]))
            continue
        if _batch_dim is None:
            _batch_dim = int(v.size)
        elif int(v.size) != _batch_dim:
            without_emb.append(int(p["id"]))
            continue
        with_emb.append((int(p["id"]), v))

    results: list[ClusterResult] = []
    for pid in without_emb:
        results.append(ClusterResult(medoid_id=pid, member_ids=[pid], size=1))

    if not with_emb:
        return results

    ids = np.array([pid for pid, _ in with_emb], dtype=int)
    X = np.vstack([v for _, v in with_emb]).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.clip(norms, 1e-12, None)

    if k is None:
        k = max(1, math.ceil(len(X) / max_size))
    k = max(1, min(k, len(X)))

    if k == 1:
        labels = np.zeros(len(X), dtype=int)
    else:
        labels = _kmeans(X, k, seed=seed)

    labels = _split_oversized(labels, max_size)

    for cid in np.unique(labels):
        mask = labels == cid
        member_ids = [int(x) for x in ids[mask].tolist()]
        sub = X[mask]
        mean_sim = (sub @ sub.T).mean(axis=1)
        medoid_idx = int(mean_sim.argmax())
        medoid_id = int(member_ids[medoid_idx])
        results.append(
            ClusterResult(
                medoid_id=medoid_id,
                member_ids=member_ids,
                size=len(member_ids),
            )
        )
    return results
