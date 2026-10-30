"""tests/test_similarity.py – Unit tests for the semantic similarity detector."""

from __future__ import annotations

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.detectors.similarity import detect_similarity
from src.data_loader import generate_synthetic_reviews


@pytest.fixture(scope="module")
def synthetic_df():
    return generate_synthetic_reviews()


@pytest.fixture()
def cluster_df():
    """DataFrame with an obvious cluster of near-identical reviews."""
    base = datetime(2024, 2, 1, tzinfo=timezone.utc)
    template = "Absolutely terrible food. Rude staff. Do not come here."
    rows = []
    # 10 near-duplicate reviews within 3 hours
    for i in range(10):
        rows.append({
            "review_id": f"clust_{i}",
            "reviewer_id": f"bot_{i}",
            "location": "Westside",
            "platform": "Yelp",
            "rating": 1,
            "text": template + (" Very bad!" if i % 3 == 0 else ""),
            "timestamp": base + timedelta(minutes=i * 10),
            "account_age_days": 3,
            "is_attack": True,
        })
    # 10 organic reviews spread over days
    for i in range(10):
        rows.append({
            "review_id": f"org_{i}",
            "reviewer_id": f"usr_{i}",
            "location": "Westside",
            "platform": "Yelp",
            "rating": 4,
            "text": f"Review number {i}: decent food but service varies.",
            "timestamp": base + timedelta(days=i),
            "account_age_days": 400,
            "is_attack": False,
        })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@pytest.fixture()
def diverse_df():
    """DataFrame with highly diverse, unrelated reviews."""
    base = datetime(2024, 3, 1, tzinfo=timezone.utc)
    texts = [
        "Amazing pasta!",
        "The sushi was fresh and the service warm.",
        "Parking is a nightmare but food is worth it.",
        "Birthday dinner – perfect ambiance.",
        "Overpriced, but that's expected in this area.",
        "Kids loved the pizza.",
        "The vegan options are surprisingly good.",
        "Coffee is exceptional.",
    ]
    rows = []
    for i, t in enumerate(texts):
        rows.append({
            "review_id": f"div_{i}",
            "reviewer_id": f"usr_{i}",
            "location": "Downtown",
            "platform": "Google",
            "rating": 4,
            "text": t,
            "timestamp": base + timedelta(days=i * 3),
            "account_age_days": 200,
            "is_attack": False,
        })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


class TestDetectSimilarity:

    def test_returns_contract(self, synthetic_df):
        result = detect_similarity(synthetic_df)
        assert "score" in result
        assert "evidence" in result

    def test_score_in_range(self, synthetic_df):
        result = detect_similarity(synthetic_df)
        assert 0.0 <= result["score"] <= 1.0

    def test_cluster_detected(self, cluster_df):
        result = detect_similarity(cluster_df, location="Westside", platform="Yelp")
        assert result["score"] > 0.4, (
            f"Expected cluster to be flagged; got score {result['score']}"
        )

    def test_diverse_reviews_low_score(self, diverse_df):
        result = detect_similarity(diverse_df)
        assert result["score"] < 0.5, (
            f"Diverse reviews should score low; got {result['score']}"
        )

    def test_evidence_contains_backend(self, synthetic_df):
        result = detect_similarity(synthetic_df)
        assert "backend" in result["evidence"]

    def test_too_few_reviews(self):
        tiny_df = pd.DataFrame({
            "review_id": ["r1"],
            "text": ["good food"],
            "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
            "location": ["A"],
            "platform": ["B"],
            "rating": [5],
        })
        result = detect_similarity(tiny_df)
        assert result["score"] == 0.0
