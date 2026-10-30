"""
src/detectors/similarity.py
============================
Semantic-similarity cluster detector for Review Shield.

Coordinated campaigns often reuse the same template review with minor
variations.  This detector:
1. Embeds review texts with sentence-transformers (all-MiniLM-L6-v2).
2. Computes pairwise cosine similarities within a time window.
3. Identifies clusters of suspiciously similar reviews.

Falls back to TF-IDF cosine similarity when sentence-transformers
cannot be imported (Rule 3).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import DetectorResult, clamp, validate_result

# ---------------------------------------------------------------------------
# Embedding backend selection
# ---------------------------------------------------------------------------
_SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
_sbert_model = None   # lazy-load singleton


def _get_embedder():
    """Return a callable that converts a list[str] → np.ndarray of embeddings."""
    global _sbert_model
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        if _sbert_model is None:
            _sbert_model = SentenceTransformer(_SBERT_MODEL_NAME)
        return lambda texts: _sbert_model.encode(texts, show_progress_bar=False)
    except ImportError:
        warnings.warn(
            "sentence-transformers not available; falling back to TF-IDF embeddings.",
            RuntimeWarning,
            stacklevel=2,
        )
        from sklearn.feature_extraction.text import TfidfVectorizer

        def _tfidf_embed(texts: list[str]) -> np.ndarray:
            vec = TfidfVectorizer(max_features=500, sublinear_tf=True)
            return vec.fit_transform(texts).toarray()

        return _tfidf_embed


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------
DEFAULT_SIM_THRESHOLD: float = 0.82   # cosine similarity ≥ this = suspiciously similar
DEFAULT_CLUSTER_MIN_SIZE: int = 4     # need at least this many similar reviews to flag
DEFAULT_WINDOW_HOURS: int = 48        # look within this many hours for a cluster


def detect_similarity(
    df: pd.DataFrame,
    location: Optional[str] = None,
    platform: Optional[str] = None,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    cluster_min_size: int = DEFAULT_CLUSTER_MIN_SIZE,
    window_hours: int = DEFAULT_WINDOW_HOURS,
) -> DetectorResult:
    """
    Detect clusters of semantically similar reviews posted in a short window.

    Args:
        df: DataFrame with columns ``text``, ``timestamp``, ``location``,
            ``platform``, ``review_id``.
        location: Filter to this location; ``None`` means all.
        platform: Filter to this platform; ``None`` means all.
        sim_threshold: Cosine similarity above which two reviews are
            considered near-duplicates.
        cluster_min_size: Minimum cluster size to flag.
        window_hours: Time window in which to look for clusters.

    Returns:
        DetectorResult with keys:
            score (float): 0–1 risk score.
            evidence (dict): largest_cluster_size, similar_pairs,
                             backend, sample_reviews, sim_threshold.
    """
    subset = df.copy()
    if location:
        subset = subset[subset["location"] == location]
    if platform:
        subset = subset[subset["platform"] == platform]

    if len(subset) < cluster_min_size:
        return validate_result(
            {"score": 0.0, "evidence": {"reason": "too few reviews to analyse"}}
        )

    # Only keep the highest-density time windows for efficiency
    subset = subset.sort_values("timestamp").reset_index(drop=True)

    embedder = _get_embedder()
    texts = subset["text"].fillna("").tolist()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embeddings = embedder(texts)

    # Ensure 2-D array
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    sim_matrix = cosine_similarity(embeddings)

    # Find clusters: connected components above threshold
    n = len(subset)
    visited = [False] * n
    clusters: list[list[int]] = []

    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        for j in range(i + 1, n):
            if visited[j]:
                continue
            # Check temporal proximity
            dt = abs(
                (subset["timestamp"].iloc[i] - subset["timestamp"].iloc[j])
                .total_seconds()
                / 3600
            )
            if dt <= window_hours and sim_matrix[i, j] >= sim_threshold:
                cluster.append(j)
                visited[j] = True
        if len(cluster) >= cluster_min_size:
            clusters.append(cluster)

    if not clusters:
        # Still compute the overall average top similarity for a partial score
        upper = sim_matrix[np.triu_indices(n, k=1)]
        avg_top_sim = float(np.percentile(upper, 95)) if len(upper) > 0 else 0.0
        partial_score = clamp((avg_top_sim - 0.5) / 0.5)
        return validate_result(
            {
                "score": round(partial_score, 4),
                "evidence": {
                    "largest_cluster_size": 0,
                    "num_clusters": 0,
                    "top_95th_pct_similarity": round(avg_top_sim, 4),
                    "sim_threshold": sim_threshold,
                    "backend": _backend_name(),
                },
            }
        )

    largest_cluster = max(clusters, key=len)
    largest_size = len(largest_cluster)
    num_pairs = int(np.sum(sim_matrix[np.triu_indices(n, k=1)] >= sim_threshold))

    # Score: ramp from cluster_min_size (→0.4) to 3×cluster_min_size (→1.0)
    raw = (largest_size - cluster_min_size) / (2 * cluster_min_size)
    score = clamp(0.4 + raw * 0.6)

    sample_ids = subset["review_id"].iloc[largest_cluster[:5]].tolist()
    sample_texts = subset["text"].iloc[largest_cluster[:3]].tolist()

    return validate_result(
        {
            "score": round(score, 4),
            "evidence": {
                "largest_cluster_size": largest_size,
                "num_clusters": len(clusters),
                "similar_pairs_total": num_pairs,
                "cluster_sample_review_ids": sample_ids,
                "cluster_sample_texts": sample_texts,
                "sim_threshold": sim_threshold,
                "window_hours": window_hours,
                "backend": _backend_name(),
            },
        }
    )


def _backend_name() -> str:
    """Return the embedding backend currently in use."""
    try:
        import sentence_transformers  # noqa: F401
        return f"sentence-transformers ({_SBERT_MODEL_NAME})"
    except ImportError:
        return "TF-IDF (fallback)"
