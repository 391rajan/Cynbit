"""tests/test_account.py – Unit tests for the account-freshness detector."""

from __future__ import annotations

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.detectors.account import detect_account_freshness
from src.data_loader import generate_synthetic_reviews


@pytest.fixture(scope="module")
def synthetic_df():
    return generate_synthetic_reviews()


@pytest.fixture()
def fresh_accounts_df():
    """Dataset dominated by brand-new accounts with extreme 1-star ratings."""
    base = datetime(2024, 4, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(20):
        rows.append({
            "review_id": f"fresh_{i}",
            "reviewer_id": f"newbot_{i}",
            "location": "Eastside",
            "platform": "Google",
            "rating": 1,
            "text": "Terrible! Avoid!",
            "timestamp": base + timedelta(hours=i),
            "account_age_days": i % 5,  # 0-4 days old
            "is_attack": True,
        })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@pytest.fixture()
def old_accounts_df():
    """Dataset of established accounts with normal age distribution."""
    base = datetime(2024, 5, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(30):
        rows.append({
            "review_id": f"old_{i}",
            "reviewer_id": f"usr_{i}",
            "location": "Uptown",
            "platform": "Yelp",
            "rating": 4,
            "text": "Good food.",
            "timestamp": base + timedelta(days=i),
            "account_age_days": 200 + i * 10,  # well-established
            "is_attack": False,
        })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


class TestDetectAccountFreshness:

    def test_returns_contract(self, synthetic_df):
        result = detect_account_freshness(synthetic_df)
        assert "score" in result and "evidence" in result

    def test_score_in_range(self, synthetic_df):
        result = detect_account_freshness(synthetic_df)
        assert 0.0 <= result["score"] <= 1.0

    def test_fresh_accounts_get_high_score(self, fresh_accounts_df):
        result = detect_account_freshness(fresh_accounts_df)
        assert result["score"] > 0.5, (
            f"Fresh attack accounts should score high; got {result['score']}"
        )

    def test_old_accounts_get_low_score(self, old_accounts_df):
        result = detect_account_freshness(old_accounts_df)
        assert result["score"] < 0.3, (
            f"Old established accounts should score low; got {result['score']}"
        )

    def test_evidence_keys(self, synthetic_df):
        result = detect_account_freshness(synthetic_df)
        ev = result["evidence"]
        assert "young_account_pct" in ev
        assert "median_account_age_days" in ev
        assert "age_threshold_days" in ev

    def test_empty_df_returns_zero(self):
        empty = pd.DataFrame(columns=["account_age_days", "rating", "location",
                                       "platform", "timestamp"])
        result = detect_account_freshness(empty)
        assert result["score"] == 0.0

    def test_location_filter_isolates(self, fresh_accounts_df, old_accounts_df):
        combined = pd.concat([fresh_accounts_df, old_accounts_df]).reset_index(drop=True)
        score_fresh = detect_account_freshness(combined, location="Eastside")["score"]
        score_old = detect_account_freshness(combined, location="Uptown")["score"]
        assert score_fresh > score_old
