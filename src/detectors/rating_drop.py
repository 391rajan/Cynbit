"""
src/detectors/rating_drop.py
============================
Sudden rating-drop detector for Review Shield.

A legitimate decline in a location's average rating typically happens
gradually. A sharp, sustained drop concentrated in a very short window
strongly suggests a coordinated review-bomb.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils import DetectorResult, clamp, validate_result

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_WINDOW_DAYS: int = 7       # detect drops within this many days
DEFAULT_DROP_THRESHOLD: float = 0.8   # flag if rating drops by ≥ 0.8 stars
DEFAULT_MIN_REVIEWS: int = 5          # minimum reviews in window to compute drop


def detect_rating_drop(
    df: pd.DataFrame,
    location: Optional[str] = None,
    platform: Optional[str] = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    drop_threshold: float = DEFAULT_DROP_THRESHOLD,
    min_reviews_in_window: int = DEFAULT_MIN_REVIEWS,
) -> DetectorResult:
    """
    Detect an anomalous sudden drop in average rating.

    Strategy:
    1. Compute a rolling ``window_days``-day average rating.
    2. Identify the maximum downward step between consecutive windows.
    3. Compare with the long-term standard deviation to assess how
       anomalous the drop is.

    Args:
        df: DataFrame with ``timestamp``, ``rating``, ``location``,
            ``platform`` columns.
        location: Filter to this location; ``None`` means all.
        platform: Filter to this platform; ``None`` means all.
        window_days: Rolling window in days for the rating average.
        drop_threshold: Minimum star-rating drop to consider suspicious.
        min_reviews_in_window: Window periods with fewer reviews are skipped.

    Returns:
        DetectorResult with keys:
            score (float): 0–1 risk score.
            evidence (dict): max_drop, pre_drop_avg, post_drop_avg,
                             drop_z_score, window_days, drop_date.
    """
    subset = df.copy()
    if location:
        subset = subset[subset["location"] == location]
    if platform:
        subset = subset[subset["platform"] == platform]

    if len(subset) < min_reviews_in_window * 2:
        return validate_result(
            {"score": 0.0, "evidence": {"reason": "insufficient reviews for trend analysis"}}
        )

    subset = subset.sort_values("timestamp")
    subset["date"] = subset["timestamp"].dt.tz_convert("UTC").dt.normalize()

    daily_avg = (
        subset.groupby("date")["rating"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_rating", "count": "n_reviews"})
    )

    # Rolling average over window_days
    rolling = (
        daily_avg["avg_rating"]
        .rolling(window=window_days, min_periods=1)
        .mean()
    )

    if len(rolling) < 2:
        return validate_result({"score": 0.0, "evidence": {"reason": "too few days"}})

    diffs = rolling.diff().dropna()

    if diffs.empty:
        return validate_result({"score": 0.0, "evidence": {"reason": "no diffs computed"}})

    min_diff = float(diffs.min())   # most negative = biggest drop
    max_drop = -min_diff            # positive value = drop magnitude

    drop_date_idx = diffs.idxmin()
    drop_date = drop_date_idx.isoformat() if hasattr(drop_date_idx, "isoformat") else str(drop_date_idx)

    pre_drop_avg = float(rolling.shift(1).loc[drop_date_idx]) if drop_date_idx in rolling.index else None
    post_drop_avg = float(rolling.loc[drop_date_idx])

    overall_std = float(daily_avg["avg_rating"].std())
    z_score = (max_drop / overall_std) if overall_std > 0 else 0.0

    # Score: ramp from drop_threshold (→0) to 2× drop_threshold (→1)
    raw = (max_drop - drop_threshold) / drop_threshold
    score = clamp(raw)

    # Additional z-score boost
    score = clamp(score + clamp((z_score - 1.5) / 3.0) * 0.3)

    return validate_result(
        {
            "score": round(score, 4),
            "evidence": {
                "max_rolling_drop_stars": round(max_drop, 3),
                "drop_date": drop_date,
                "pre_drop_rolling_avg": round(pre_drop_avg, 3) if pre_drop_avg else "n/a",
                "post_drop_rolling_avg": round(post_drop_avg, 3),
                "drop_z_score": round(z_score, 2),
                "window_days": window_days,
                "drop_threshold": drop_threshold,
                "total_reviews": len(subset),
            },
        }
    )
