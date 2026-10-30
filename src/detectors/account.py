"""
src/detectors/account.py
========================
Account-freshness / profile-age detector for Review Shield.

Newly created accounts that post reviews immediately are a strong
signal of coordinated fake-review activity.  This detector analyses:
- Distribution of account_age_days in a given slice
- Proportion of accounts younger than the threshold
- Whether the young-account reviews cluster temporally
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils import DetectorResult, clamp, validate_result

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_AGE_THRESHOLD_DAYS: int = 14   # accounts younger than this are suspect
DEFAULT_YOUNG_RATE_THRESHOLD: float = 0.30  # flag if >30% of reviewers are new


def detect_account_freshness(
    df: pd.DataFrame,
    location: Optional[str] = None,
    platform: Optional[str] = None,
    age_threshold_days: int = DEFAULT_AGE_THRESHOLD_DAYS,
    young_rate_threshold: float = DEFAULT_YOUNG_RATE_THRESHOLD,
) -> DetectorResult:
    """
    Detect suspicious concentrations of brand-new reviewer accounts.

    Args:
        df: DataFrame with columns ``account_age_days``, ``timestamp``,
            ``location``, ``platform``, ``rating``.
        location: Filter to this location; ``None`` means all.
        platform: Filter to this platform; ``None`` means all.
        age_threshold_days: Accounts younger than this (in days) are suspect.
        young_rate_threshold: Proportion of young accounts that triggers
            a medium-risk flag.

    Returns:
        DetectorResult with keys:
            score (float): 0–1 risk score.
            evidence (dict): young_account_pct, young_account_count,
                             total_reviews, median_age_days,
                             age_threshold_days, avg_rating_young.
    """
    subset = df.copy()
    if location:
        subset = subset[subset["location"] == location]
    if platform:
        subset = subset[subset["platform"] == platform]

    if subset.empty:
        return validate_result({"score": 0.0, "evidence": {"reason": "no data"}})

    ages = subset["account_age_days"].fillna(365)
    young_mask = ages < age_threshold_days
    young_count = int(young_mask.sum())
    total = len(subset)
    young_rate = young_count / total

    median_age = float(np.median(ages))
    avg_rating_young = float(
        subset.loc[young_mask, "rating"].mean()
    ) if young_count > 0 else None

    # Score: linearly scales from 0 at young_rate_threshold to 1.0 at 100%
    raw = (young_rate - young_rate_threshold) / (1.0 - young_rate_threshold)
    score = clamp(raw)

    # Extra boost: young accounts with extreme ratings (≤1 or =5) are more suspicious
    if young_count > 0:
        extreme_rating_mask = young_mask & (
            (subset["rating"] <= 1) | (subset["rating"] == 5)
        )
        extreme_rate = extreme_rating_mask.sum() / young_count
        if extreme_rate > 0.75:
            score = clamp(score + 0.15)

    return validate_result(
        {
            "score": round(score, 4),
            "evidence": {
                "young_account_pct": round(young_rate, 4),
                "young_account_count": young_count,
                "total_reviews": total,
                "median_account_age_days": round(median_age, 1),
                "age_threshold_days": age_threshold_days,
                "avg_rating_young_accounts": (
                    round(avg_rating_young, 2) if avg_rating_young is not None else "n/a"
                ),
                "young_rate_threshold": young_rate_threshold,
            },
        }
    )
