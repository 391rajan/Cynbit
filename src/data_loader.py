"""
src/data_loader.py
==================
Synthetic review dataset generator and CSV loader for Review Shield.

The generator creates a realistic multi-location restaurant brand dataset with:
- Normal organic reviews
- Coordinated fake-positive campaigns (competitor sabotage or astroturfing)
- Review-bomb (coordinated negative) campaigns
- Ground-truth label column ``is_attack`` (bool) and ``attack_type`` (str)

All randomness uses SEED for reproducibility (Rule 6).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import SEED, make_rng

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOCATIONS: list[str] = [
    "Downtown – Main St",
    "Midtown – 5th Ave",
    "Westside – Sunset Blvd",
    "Eastside – Harbor Dr",
    "Uptown – Park Ave",
]

PLATFORMS: list[str] = ["Google", "Yelp", "TripAdvisor", "Facebook"]

NORMAL_TEMPLATES: list[str] = [
    "Really enjoyed the food here, will definitely come back!",
    "Service was a bit slow but the quality made up for it.",
    "Solid neighborhood spot. The pasta is genuinely good.",
    "Great ambience for a date night. Pricey but worth it.",
    "Average food, nothing special but not bad either.",
    "The staff was incredibly friendly. Loved the experience.",
    "Decent place. The burgers are great, the fries not so much.",
    "My go-to brunch spot. Can't recommend the eggs benedict enough.",
    "Mixed feelings – some dishes great, others disappointing.",
    "Portions are generous and prices are fair.",
    "Had a birthday dinner here, everything was perfect.",
    "A hidden gem! Discovered it by accident and now I'm hooked.",
    "The wait was long on a Friday night but expected.",
    "Coffee is 10/10. Pastries are decent.",
    "Tried the new seasonal menu – very impressive.",
]

BOMB_TEMPLATES: list[str] = [
    "Terrible experience. Rude staff and awful food. Never again.",
    "Found a hair in my food. Health code violations everywhere.",
    "Waited 45 minutes and the order was completely wrong. Disgraceful.",
    "The management is incompetent. Stay away.",
    "Worst restaurant I have ever been to in my life.",
    "Absolute garbage. The owner doesn't care about customers.",
    "Food poisoning! Avoid at all costs.",
    "Completely overpriced for what you get. A tourist trap.",
    "The place is dirty and the staff is rude.",
    "I have seen better food at a gas station.",
]

FAKE_POS_TEMPLATES: list[str] = [
    "Amazing place! Best food in the city hands down! 10/10!",
    "Phenomenal experience! Exceeded all expectations! Highly recommend!",
    "Absolutely outstanding! The best restaurant I have visited in years!",
    "Top-notch service and incredible food. A must-visit destination!",
    "Perfect in every way. Cannot praise this place enough!",
    "World-class food and atmosphere. Total game changer!",
    "I brought my entire family and everyone loved it! 5 stars!",
    "This place is flawless. Every single dish was magnificent.",
]


def _fake_reviewer_id(rng: np.random.Generator, prefix: str = "usr") -> str:
    """Generate a deterministic-looking reviewer ID."""
    n = rng.integers(100_000, 999_999)
    return f"{prefix}_{n}"


def _fake_account_age_days(rng: np.random.Generator, is_attack: bool) -> int:
    """Return an account age in days; attack accounts skew very young."""
    if is_attack:
        return int(rng.integers(0, 8))   # 0–7 days old
    return int(rng.integers(30, 1500))   # Normal: 1 month – ~4 years


def generate_synthetic_reviews(
    n_normal: int = 400,
    n_bomb_campaigns: int = 3,
    n_fake_pos_campaigns: int = 2,
    campaign_size_range: tuple[int, int] = (15, 35),
    start_date: Optional[datetime] = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic multi-location review dataset.

    Args:
        n_normal: Number of organic baseline reviews.
        n_bomb_campaigns: Number of review-bomb attack campaigns.
        n_fake_pos_campaigns: Number of fake-positive campaigns.
        campaign_size_range: Min/max reviews per campaign.
        start_date: Dataset start date (defaults to 90 days ago).
        seed: Random seed for full reproducibility.

    Returns:
        DataFrame with columns:
            review_id, reviewer_id, location, platform, rating, text,
            timestamp, account_age_days, is_attack (bool), attack_type (str).
    """
    rng = make_rng(seed)

    if start_date is None:
        start_date = datetime.now(tz=timezone.utc) - timedelta(days=90)

    rows: list[dict] = []

    # ---- Organic reviews -----------------------------------------------
    for _ in range(n_normal):
        ts = start_date + timedelta(
            seconds=int(rng.integers(0, 90 * 24 * 3600))
        )
        text = NORMAL_TEMPLATES[int(rng.integers(0, len(NORMAL_TEMPLATES)))]
        rows.append(
            {
                "reviewer_id": _fake_reviewer_id(rng),
                "location": LOCATIONS[int(rng.integers(0, len(LOCATIONS)))],
                "platform": PLATFORMS[int(rng.integers(0, len(PLATFORMS)))],
                "rating": int(rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.07, 0.15, 0.38, 0.35])),
                "text": text,
                "timestamp": ts,
                "account_age_days": _fake_account_age_days(rng, is_attack=False),
                "is_attack": False,
                "attack_type": "none",
            }
        )

    # ---- Review-bomb campaigns -----------------------------------------
    for c in range(n_bomb_campaigns):
        loc = LOCATIONS[int(rng.integers(0, len(LOCATIONS)))]
        platform = PLATFORMS[int(rng.integers(0, len(PLATFORMS)))]
        burst_start = start_date + timedelta(
            seconds=int(rng.integers(10 * 24 * 3600, 80 * 24 * 3600))
        )
        burst_size = int(rng.integers(*campaign_size_range))
        # All from very similar base text (semantic cluster)
        base_text = BOMB_TEMPLATES[c % len(BOMB_TEMPLATES)]
        for i in range(burst_size):
            # Slight paraphrasing: shuffle a word occasionally
            words = base_text.split()
            if rng.random() > 0.6 and len(words) > 3:
                idx = int(rng.integers(1, len(words) - 1))
                words[idx] = words[idx] + "!"
            text = " ".join(words)
            ts = burst_start + timedelta(seconds=int(rng.integers(0, 4 * 3600)))
            rows.append(
                {
                    "reviewer_id": _fake_reviewer_id(rng, f"bomb_c{c}"),
                    "location": loc,
                    "platform": platform,
                    "rating": int(rng.choice([1, 2], p=[0.85, 0.15])),
                    "text": text,
                    "timestamp": ts,
                    "account_age_days": _fake_account_age_days(rng, is_attack=True),
                    "is_attack": True,
                    "attack_type": "review_bomb",
                }
            )

    # ---- Fake-positive campaigns ----------------------------------------
    for c in range(n_fake_pos_campaigns):
        loc = LOCATIONS[int(rng.integers(0, len(LOCATIONS)))]
        platform = PLATFORMS[int(rng.integers(0, len(PLATFORMS)))]
        burst_start = start_date + timedelta(
            seconds=int(rng.integers(5 * 24 * 3600, 75 * 24 * 3600))
        )
        burst_size = int(rng.integers(*campaign_size_range))
        base_text = FAKE_POS_TEMPLATES[c % len(FAKE_POS_TEMPLATES)]
        for i in range(burst_size):
            words = base_text.split()
            if rng.random() > 0.5 and len(words) > 3:
                words.append(rng.choice(["Great!", "Love it!", "Amazing!"]))
            text = " ".join(words)
            ts = burst_start + timedelta(seconds=int(rng.integers(0, 6 * 3600)))
            rows.append(
                {
                    "reviewer_id": _fake_reviewer_id(rng, f"fpos_c{c}"),
                    "location": loc,
                    "platform": platform,
                    "rating": int(rng.choice([4, 5], p=[0.1, 0.9])),
                    "text": text,
                    "timestamp": ts,
                    "account_age_days": _fake_account_age_days(rng, is_attack=True),
                    "is_attack": True,
                    "attack_type": "fake_positive",
                }
            )

    df = pd.DataFrame(rows)
    df["review_id"] = [
        hashlib.md5(f"{r.reviewer_id}{r.timestamp}".encode()).hexdigest()[:12]
        for r in df.itertuples()
    ]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_reviews(uploaded_file=None) -> pd.DataFrame:
    """
    Load reviews from an uploaded CSV file or fall back to synthetic data.

    The CSV must contain columns: reviewer_id, location, platform, rating,
    text, timestamp, account_age_days. Optional: is_attack, attack_type.

    Args:
        uploaded_file: A Streamlit ``UploadedFile`` object, or ``None``.

    Returns:
        Cleaned DataFrame ready for detectors.
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if "is_attack" not in df.columns:
            df["is_attack"] = False
        if "attack_type" not in df.columns:
            df["attack_type"] = "unknown"
        if "account_age_days" not in df.columns:
            df["account_age_days"] = 365
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    return generate_synthetic_reviews()
