"""
src/detectors/ensemble.py
=========================
Weighted ensemble scorer for Review Shield.

Combines the four individual detector signals into a single campaign-level
risk score.  Weights are tuned so that:
- Velocity  (temporal burst)    – highest weight; most reliable alone
- Similarity (text clusters)    – high weight; very precise
- Account   (fresh profiles)    – medium weight; noisy in isolation
- RatingDrop (trend anomaly)    – medium weight; good confirmation signal
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.detectors.account import detect_account_freshness
from src.detectors.rating_drop import detect_rating_drop
from src.detectors.similarity import detect_similarity
from src.detectors.velocity import detect_velocity
from src.utils import DetectorResult, clamp, validate_result

# ---------------------------------------------------------------------------
# Detector weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHTS: dict[str, float] = {
    "velocity":    0.35,
    "similarity":  0.30,
    "account":     0.20,
    "rating_drop": 0.15,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


def run_ensemble(
    df: pd.DataFrame,
    location: Optional[str] = None,
    platform: Optional[str] = None,
) -> DetectorResult:
    """
    Run all four detectors and combine their scores into an ensemble result.

    Args:
        df: Full review DataFrame.
        location: Location to filter to; ``None`` means aggregate.
        platform: Platform to filter to; ``None`` means aggregate.

    Returns:
        DetectorResult with:
            score (float): 0–1 weighted ensemble risk score.
            evidence (dict): per-detector scores and their evidence, plus
                             the weights used and a human-readable verdict.
    """
    results: dict[str, DetectorResult] = {
        "velocity":    detect_velocity(df, location=location, platform=platform),
        "similarity":  detect_similarity(df, location=location, platform=platform),
        "account":     detect_account_freshness(df, location=location, platform=platform),
        "rating_drop": detect_rating_drop(df, location=location, platform=platform),
    }

    ensemble_score = clamp(
        sum(WEIGHTS[name] * res["score"] for name, res in results.items())
    )

    # Flag which detectors contributed meaningfully (score > 0.4)
    triggered = [name for name, res in results.items() if res["score"] > 0.4]

    return validate_result(
        {
            "score": round(ensemble_score, 4),
            "evidence": {
                "detector_scores": {
                    name: round(res["score"], 4) for name, res in results.items()
                },
                "detector_evidence": {
                    name: res["evidence"] for name, res in results.items()
                },
                "weights": WEIGHTS,
                "triggered_detectors": triggered,
                "verdict": _verdict(ensemble_score, triggered),
                "location": location or "all",
                "platform": platform or "all",
            },
        }
    )


def _verdict(score: float, triggered: list[str]) -> str:
    """Return a human-readable verdict string."""
    if score < 0.35:
        return "No coordinated activity detected."
    if score < 0.60:
        flagged = ", ".join(t.replace("_", " ") for t in triggered) or "one signal"
        return f"Possible coordinated activity — {flagged} flagged."
    if score < 0.80:
        flagged = ", ".join(t.replace("_", " ") for t in triggered)
        return f"Likely coordinated campaign — {flagged} all triggered."
    flagged = ", ".join(t.replace("_", " ") for t in triggered)
    return f"HIGH CONFIDENCE coordinated attack — {flagged} strongly triggered."
