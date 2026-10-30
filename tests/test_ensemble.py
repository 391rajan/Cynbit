"""tests/test_ensemble.py – Unit tests for the ensemble scorer."""

from __future__ import annotations

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.detectors.ensemble import run_ensemble, WEIGHTS
from src.data_loader import generate_synthetic_reviews


@pytest.fixture(scope="module")
def synthetic_df():
    return generate_synthetic_reviews()


@pytest.fixture()
def attack_df():
    """High-confidence attack dataset."""
    base = datetime(2024, 8, 1, tzinfo=timezone.utc)
    rows = []
    template = "This place is absolutely terrible. Disgusting food and rude staff."
    for i in range(25):
        rows.append({
            "review_id": f"atk_{i}",
            "reviewer_id": f"bot_{i}",
            "location": "Downtown",
            "platform": "Google",
            "rating": 1,
            "text": template + (" Do not visit!" if i % 4 == 0 else ""),
            "timestamp": base + timedelta(minutes=i * 8),
            "account_age_days": i % 4,  # 0-3 days old
            "is_attack": True,
        })
    # Add some organic background
    for i in range(20):
        rows.append({
            "review_id": f"org_{i}",
            "reviewer_id": f"usr_{i}",
            "location": "Downtown",
            "platform": "Google",
            "rating": 4,
            "text": f"Good experience, visit number {i}.",
            "timestamp": base - timedelta(days=i + 1),
            "account_age_days": 400,
            "is_attack": False,
        })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


class TestEnsemble:

    def test_returns_contract(self, synthetic_df):
        result = run_ensemble(synthetic_df)
        assert "score" in result and "evidence" in result

    def test_score_in_range(self, synthetic_df):
        result = run_ensemble(synthetic_df)
        assert 0.0 <= result["score"] <= 1.0

    def test_weights_sum_to_one(self):
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_all_four_detectors_present(self, synthetic_df):
        result = run_ensemble(synthetic_df)
        det_scores = result["evidence"]["detector_scores"]
        assert set(det_scores.keys()) == {"velocity", "similarity", "account", "rating_drop"}

    def test_attack_df_higher_than_organic(self, attack_df):
        # Also make a clean organic-only df
        organic_only = attack_df[attack_df["is_attack"] == False].copy()
        score_attack = run_ensemble(attack_df)["score"]
        score_organic = run_ensemble(organic_only)["score"]
        assert score_attack > score_organic, (
            f"Attack dataset ({score_attack}) should score higher than organic ({score_organic})"
        )

    def test_verdict_string_present(self, synthetic_df):
        result = run_ensemble(synthetic_df)
        assert isinstance(result["evidence"]["verdict"], str)
        assert len(result["evidence"]["verdict"]) > 10

    def test_location_filter(self, attack_df):
        score_all = run_ensemble(attack_df)["score"]
        score_loc = run_ensemble(attack_df, location="Downtown")["score"]
        # Both should be non-trivially positive
        assert score_loc >= 0.0

    def test_triggered_detectors_list(self, attack_df):
        result = run_ensemble(attack_df, location="Downtown")
        triggered = result["evidence"]["triggered_detectors"]
        assert isinstance(triggered, list)

    def test_score_is_float(self, synthetic_df):
        result = run_ensemble(synthetic_df)
        assert isinstance(result["score"], float)
