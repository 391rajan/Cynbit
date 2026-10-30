"""tests/test_rating_drop.py – Unit tests for the rating-drop detector."""

from __future__ import annotations

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.detectors.rating_drop import detect_rating_drop
from src.data_loader import generate_synthetic_reviews


@pytest.fixture(scope="module")
def synthetic_df():
    return generate_synthetic_reviews()


@pytest.fixture()
def drop_df():
    """Dataset with a clear sudden rating drop."""
    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    rows = []
    # 20 days of high ratings
    for i in range(20):
        for _ in range(3):  # 3 reviews/day
            rows.append({
                "review_id": f"pre_{i}",
                "reviewer_id": f"usr_{i}",
                "location": "Midtown",
                "platform": "TripAdvisor",
                "rating": 5,
                "text": "Great!",
                "timestamp": base + timedelta(days=i, hours=_ * 3),
                "account_age_days": 300,
            })
    # 3 days of 1-star bombing
    bomb_start = base + timedelta(days=21)
    for i in range(30):
        rows.append({
            "review_id": f"bomb_{i}",
            "reviewer_id": f"bot_{i}",
            "location": "Midtown",
            "platform": "TripAdvisor",
            "rating": 1,
            "text": "Terrible!",
            "timestamp": bomb_start + timedelta(hours=i),
            "account_age_days": 1,
        })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@pytest.fixture()
def stable_df():
    """Dataset with stable ratings — no sudden drop."""
    base = datetime(2024, 7, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(60):
        for _ in range(2):
            rows.append({
                "review_id": f"stable_{i}_{_}",
                "reviewer_id": f"usr_{i}",
                "location": "Uptown",
                "platform": "Google",
                "rating": int(3 + (i % 3)),  # cycles 3, 4, 5
                "text": "Decent.",
                "timestamp": base + timedelta(days=i, hours=_ * 6),
                "account_age_days": 400,
            })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


class TestDetectRatingDrop:

    def test_returns_contract(self, synthetic_df):
        result = detect_rating_drop(synthetic_df)
        assert "score" in result and "evidence" in result

    def test_score_in_range(self, synthetic_df):
        result = detect_rating_drop(synthetic_df)
        assert 0.0 <= result["score"] <= 1.0

    def test_drop_gets_high_score(self, drop_df):
        result = detect_rating_drop(drop_df, location="Midtown", window_days=3, drop_threshold=0.5)
        assert result["score"] > 0.0, (
            f"Sudden rating drop should be flagged; got {result['score']}"
        )
        assert result["evidence"]["max_rolling_drop_stars"] > 0.3, (
            f"Expected a measurable drop; evidence: {result['evidence']}"
        )

    def test_stable_ratings_low_score(self, stable_df):
        result = detect_rating_drop(stable_df, location="Uptown")
        assert result["score"] < 0.5, (
            f"Stable ratings should score low; got {result['score']}"
        )

    def test_evidence_keys(self, drop_df):
        result = detect_rating_drop(drop_df)
        ev = result["evidence"]
        assert "max_rolling_drop_stars" in ev
        assert "drop_z_score" in ev
        assert "window_days" in ev

    def test_insufficient_data(self):
        tiny = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01"], utc=True),
            "rating": [4],
            "location": ["A"],
            "platform": ["B"],
        })
        result = detect_rating_drop(tiny)
        assert result["score"] == 0.0
