"""
src/detectors/velocity.py
=========================
Velocity-burst detector for Review Shield.

Detects an anomalous spike in review volume within a short time window
for a given (location, platform) pair.  A sudden influx of reviews —
especially low-rated ones — is a classic review-bomb signal.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils import DetectorResult, clamp, validate_result

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_WINDOW_HOURS: int = 6
DEFAULT_BASELINE_DAYS: int = 30
DEFAULT_MULTIPLIER_THRESHOLD: float = 4.0   # spike must be ≥ 4× baseline rate


def detect_velocity(
    df: pd.DataFrame,
    location: Optional[str] = None,
    platform: Optional[str] = None,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    baseline_days: int = DEFAULT_BASELINE_DAYS,
    multiplier_threshold: float = DEFAULT_MULTIPLIER_THRESHOLD,
) -> DetectorResult:
    """
    Detect review-velocity bursts that indicate a coordinated campaign.

    The detector:
    1. Filters to the requested location/platform slice.
    2. Computes a rolling review count in ``window_hours``-hour buckets.
    3. Compares the peak bucket to the median baseline rate.
    4. Returns a score proportional to the excess multiplier.

    Args:
        df: DataFrame of reviews with a ``timestamp`` column (timezone-aware).
        location: Filter to this location; ``None`` means all locations.
        platform: Filter to this platform; ``None`` means all platforms.
        window_hours: Rolling window width in hours.
        baseline_days: Days of history used to estimate a baseline rate.
        multiplier_threshold: Burst multiples at which score reaches 1.0.

    Returns:
        DetectorResult with keys:
            score (float): 0–1 risk score.
            evidence (dict): peak_window, peak_count, baseline_rate,
                             multiplier, window_hours, flagged_start.
    """
    subset = df.copy()
    if location:
        subset = subset[subset["location"] == location]
    if platform:
        subset = subset[subset["platform"] == platform]

    if subset.empty:
        return validate_result({"score": 0.0, "evidence": {"reason": "no data for filter"}})

    subset = subset.sort_values("timestamp")
    ts = subset["timestamp"].dt.tz_convert("UTC")

    # ---- Resample to hourly buckets ------------------------------------
    hourly = (
        subset.set_index(subset["timestamp"].dt.tz_convert("UTC"))
        .resample(f"{window_hours}h")["review_id"]
        .count()
        .rename("count")
    )

    if len(hourly) < 2:
        return validate_result({"score": 0.0, "evidence": {"reason": "insufficient history"}})

    peak_idx = int(hourly.values.argmax())
    peak_count = int(hourly.iloc[peak_idx])
    peak_window_start = hourly.index[peak_idx].isoformat()

    # Baseline: median rate over all buckets except the peak
    non_peak = hourly.drop(hourly.index[peak_idx])
    baseline_rate = float(non_peak.mean()) if len(non_peak) > 0 else 1.0
    baseline_rate = max(baseline_rate, 0.5)   # avoid division by zero

    multiplier = peak_count / baseline_rate

    # Score: sigmoid-ish ramp between 1× and multiplier_threshold×
    raw_score = (multiplier - 1.0) / (multiplier_threshold - 1.0)
    score = clamp(raw_score)

    # Proportion of 1-star reviews in the peak window
    peak_bucket_ts = hourly.index[peak_idx]
    next_bucket_ts = peak_bucket_ts + pd.Timedelta(hours=window_hours)
    peak_reviews = subset[
        (subset["timestamp"].dt.tz_convert("UTC") >= peak_bucket_ts)
        & (subset["timestamp"].dt.tz_convert("UTC") < next_bucket_ts)
    ]
    low_star_pct = float(
        (peak_reviews["rating"] <= 2).sum() / max(len(peak_reviews), 1)
    )

    # Boost score if the burst is heavily negative
    if low_star_pct > 0.7:
        score = clamp(score * 1.25)

    return validate_result(
        {
            "score": round(score, 4),
            "evidence": {
                "peak_window_start": peak_window_start,
                "peak_count": peak_count,
                "baseline_rate_per_window": round(baseline_rate, 2),
                "multiplier": round(multiplier, 2),
                "low_star_pct_in_peak": round(low_star_pct, 3),
                "window_hours": window_hours,
                "total_reviews_analysed": len(subset),
            },
        }
    )
