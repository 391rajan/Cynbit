"""tests/test_velocity.py – Unit tests for the velocity-burst detector."""

from __future__ import annotations

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.detectors.velocity import detect_velocity
from src.data_loader import generate_synthetic_reviews


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_df():
    return generate_synthetic_reviews()


@pytest.fixture()
def burst_df():
    """A DataFrame with a clear velocity burst."""
    base = datetime(2024, 1, 10, tzinfo=timezone.utc)
    rows = []
    # Organic: 1 review per day for 30 days
    for i in range(30):
        rows.append({
            "review_id": f"organic_{i}",
            "reviewer_id": f"usr_{i}",
            "location": "Downtown",
            "platform": "Google",
            "rating": 4,
            "text": "Good food, enjoyed it.",
            "timestamp": base + timedelta(days=i),
            "account_age_days": 500,
            "is_attack": False,
        })
    # Burst: 20 reviews in 2 hours
    burst_base = base + timedelta(days=31)
    for i in range(20):
        rows.append({
            "review_id": f"burst_{i}",
            "reviewer_id": f"bomb_{i}",
            "location": "Downtown",
            "platform": "Google",
            "rating": 1,
            "text": "Terrible place, avoid!",
            "timestamp": burst_base + timedelta(minutes=i * 5),
            "account_age_days": 2,
            "is_attack": True,
        })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@pytest.fixture()
def empty_df():
    return pd.DataFrame(columns=["review_id", "reviewer_id", "location", "platform",
                                  "rating", "text", "timestamp", "account_age_days"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestDetectVelocity:

    def test_returns_dict_with_score_and_evidence(self, synthetic_df):
        result = detect_velocity(synthetic_df)
        assert isinstance(result, dict)
        assert "score" in result
        assert "evidence" in result

    def test_score_in_range(self, synthetic_df):
        result = detect_velocity(synthetic_df)
        assert 0.0 <= result["score"] <= 1.0

    def test_burst_gets_high_score(self, burst_df):
        result = detect_velocity(burst_df, location="Downtown", platform="Google")
        assert result["score"] > 0.5, f"Expected high score for burst data, got {result['score']}"

    def test_evidence_has_required_keys(self, burst_df):
        result = detect_velocity(burst_df)
        evidence = result["evidence"]
        assert "peak_count" in evidence
        assert "multiplier" in evidence
        assert "window_hours" in evidence

    def test_empty_df_returns_zero(self, empty_df):
        result = detect_velocity(empty_df)
        assert result["score"] == 0.0

    def test_location_filter(self, burst_df):
        result_with_burst = detect_velocity(burst_df, location="Downtown")
        result_other = detect_velocity(burst_df, location="Midtown")
        assert result_with_burst["score"] >= result_other["score"]

    def test_score_is_float(self, synthetic_df):
        result = detect_velocity(synthetic_df)
        assert isinstance(result["score"], float)

    def test_evidence_is_dict(self, synthetic_df):
        result = detect_velocity(synthetic_df)
        assert isinstance(result["evidence"], dict)
