"""
src/evaluate.py
===============
Evaluation module for Review Shield.

Computes review-level and campaign-level metrics against the ``is_attack``
ground-truth column:

* **Review-level**: precision, recall, F1, false-positive rate at a
  configurable ensemble score threshold.
* **Per-detector recall**: how often each detector independently flags
  reviews belonging to an actual attack.
* **Time-to-detect**: hours from the first attack review in each campaign
  to the first review flagged by the ensemble.
* **Confusion matrix** counts for transparency.

Results are saved to ``data/eval.json`` and printed to stdout.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.data_loader import LOCATIONS, generate_synthetic_reviews
from src.detectors.account import detect_account_freshness
from src.detectors.rating_drop import detect_rating_drop
from src.detectors.similarity import detect_similarity
from src.detectors.velocity import detect_velocity
from src.detectors.ensemble import WEIGHTS, run_ensemble
from src.utils import SEED, clamp, seed_everything


# ---------------------------------------------------------------------------
# Review-level scoring
# ---------------------------------------------------------------------------
def _score_reviews(
    df: pd.DataFrame,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Assign a binary ``predicted_attack`` flag to every review.

    Strategy:  The ensemble produces a **location-level** score (since
    campaigns target a location).  We propagate that score to each review
    belonging to that location and compare against the threshold.  Reviews
    in a highly-scored location whose individual signals are weak can still
    be flagged — this is intentional: in practice the triage unit would
    inspect all reviews in a flagged location.

    Additionally, reviews from very new accounts (< 14 days) in a flagged
    location receive a small score boost, matching how a human analyst
    would prioritise.

    Args:
        df: Review DataFrame with ``is_attack``, ``location``, etc.
        threshold: Ensemble score above which a location is flagged.

    Returns:
        DataFrame with added columns ``ensemble_score``, ``predicted_attack``.
    """
    scored = df.copy()
    scored["ensemble_score"] = 0.0
    scored["predicted_attack"] = False

    for loc in df["location"].unique():
        result = run_ensemble(df, location=loc)
        loc_score = result["score"]

        loc_mask = scored["location"] == loc

        # Per-review adjustment: new accounts + extreme rating get boost
        acct_boost = (scored["account_age_days"] < 14).astype(float) * 0.08
        rating_boost = (
            (scored["rating"] <= 1) | (scored["rating"] == 5)
        ).astype(float) * 0.05

        scored.loc[loc_mask, "ensemble_score"] = np.clip(
            loc_score + acct_boost[loc_mask] + rating_boost[loc_mask],
            0.0, 1.0,
        )
        scored.loc[loc_mask, "predicted_attack"] = (
            scored.loc[loc_mask, "ensemble_score"] >= threshold
        )

    return scored


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    df: pd.DataFrame,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """
    Compute all evaluation metrics.

    Args:
        df: Review DataFrame with ``is_attack`` ground-truth.
        threshold: Score threshold for flagging.

    Returns:
        Dict with keys: precision, recall, f1, false_positive_rate,
        per_detector_recall, time_to_detect, confusion_matrix,
        threshold, total_reviews, attack_reviews, organic_reviews.
    """
    scored = _score_reviews(df, threshold=threshold)

    y_true = scored["is_attack"].astype(bool)
    y_pred = scored["predicted_attack"].astype(bool)

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Per-detector recall ---------------------------------------------------
    per_detector_recall = _per_detector_recall(df)

    # Time-to-detect --------------------------------------------------------
    ttd = _time_to_detect(df, scored, threshold)

    return {
        "threshold": threshold,
        "total_reviews": len(df),
        "attack_reviews": int(y_true.sum()),
        "organic_reviews": int((~y_true).sum()),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "per_detector_recall": per_detector_recall,
        "time_to_detect": ttd,
    }


def _per_detector_recall(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute recall for each individual detector.

    For each detector, we check how many attack-containing locations
    the detector scores above 0.40 (its individual trigger threshold).

    Args:
        df: Review DataFrame with ground-truth labels.

    Returns:
        Dict mapping detector name to recall (0–1).
    """
    attack_locations = df[df["is_attack"]]["location"].unique().tolist()

    if not attack_locations:
        return {d: 0.0 for d in ["velocity", "similarity", "account", "rating_drop"]}

    detector_fns = {
        "velocity": detect_velocity,
        "similarity": detect_similarity,
        "account": detect_account_freshness,
        "rating_drop": detect_rating_drop,
    }

    result: dict[str, float] = {}
    for det_name, det_fn in detector_fns.items():
        detected = 0
        for loc in attack_locations:
            det_result = det_fn(df, location=loc)
            if det_result["score"] > 0.40:
                detected += 1
        result[det_name] = round(detected / len(attack_locations), 4)

    return result


def _time_to_detect(
    df: pd.DataFrame,
    scored: pd.DataFrame,
    threshold: float,
) -> list[dict[str, Any]]:
    """
    Compute time-to-detect for each injected campaign.

    For each (attack_type, location) group, finds the first attack review's
    timestamp and the first flagged review's timestamp.

    Args:
        df: Original review DataFrame.
        scored: DataFrame with ``predicted_attack`` and ``ensemble_score``.
        threshold: Score threshold used.

    Returns:
        List of dicts with campaign_id, attack_type, location,
        first_attack_ts, first_flagged_ts, hours_to_detect.
    """
    attack_df = df[df["is_attack"]].copy()
    if attack_df.empty:
        return []

    campaigns = (
        attack_df
        .groupby(["attack_type", "location"])
        .agg(
            first_attack_ts=("timestamp", "min"),
            attack_count=("review_id", "count"),
        )
        .reset_index()
    )

    ttd_results: list[dict[str, Any]] = []
    for _, campaign in campaigns.iterrows():
        loc = campaign["location"]
        atype = campaign["attack_type"]

        # Find the first flagged review in this location FOR THIS campaign type
        # (Uptown has both review_bomb + fake_positive; filter by attack_type)
        flagged_in_loc = scored[
            (scored["location"] == loc)
            & (scored["predicted_attack"])
            & (scored["is_attack"])
            & (scored["attack_type"] == atype)
        ].sort_values("timestamp")

        if flagged_in_loc.empty:
            hours = None
        else:
            first_flagged = flagged_in_loc["timestamp"].iloc[0]
            first_attack = campaign["first_attack_ts"]
            raw_hours = (first_flagged - first_attack).total_seconds() / 3600
            # Negative means the location was already hot before this campaign
            # (overlapping campaigns); clamp to 0 – still counts as detected.
            hours = round(max(0.0, raw_hours), 2)

        ttd_results.append({
            "campaign": f"{atype}@{loc}",
            "attack_type": atype,
            "location": loc,
            "attack_count": int(campaign["attack_count"]),
            "first_attack_ts": campaign["first_attack_ts"].isoformat(),
            "first_flagged_ts": (
                flagged_in_loc["timestamp"].iloc[0].isoformat()
                if not flagged_in_loc.empty
                else None
            ),
            "hours_to_detect": hours,
        })

    return ttd_results


# ---------------------------------------------------------------------------
# Save & print
# ---------------------------------------------------------------------------
def save_eval(
    metrics: dict[str, Any],
    output_dir: str = "data",
    filename: str = "eval.json",
) -> Path:
    """
    Save evaluation results to a JSON file.

    Args:
        metrics: Dict of metrics from ``compute_metrics``.
        output_dir: Directory to write to (created if missing).
        filename: Output file name.

    Returns:
        Path to the saved file.
    """
    out_path = Path(output_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "seed": SEED,
        "note": "Evaluated on synthetic data with ground-truth labels. "
                "Numbers reflect detector performance on this specific dataset "
                "and should not be extrapolated to production without real-world validation.",
        **metrics,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    return out_path


def print_eval_table(metrics: dict[str, Any]) -> str:
    """
    Format evaluation metrics as a human-readable table string.

    Args:
        metrics: Dict of metrics from ``compute_metrics``.

    Returns:
        Formatted table string.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  REVIEW SHIELD — EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append(f"  Threshold:             {metrics['threshold']:.2f}")
    lines.append(f"  Total reviews:         {metrics['total_reviews']}")
    lines.append(f"  Attack reviews:        {metrics['attack_reviews']}")
    lines.append(f"  Organic reviews:       {metrics['organic_reviews']}")
    lines.append("-" * 60)
    lines.append("  REVIEW-LEVEL METRICS")
    lines.append("-" * 60)
    lines.append(f"  Precision:             {metrics['precision']:.2%}")
    lines.append(f"  Recall:                {metrics['recall']:.2%}")
    lines.append(f"  F1 Score:              {metrics['f1']:.2%}")
    lines.append(f"  False Positive Rate:   {metrics['false_positive_rate']:.2%}")
    lines.append("-" * 60)
    lines.append("  CONFUSION MATRIX")
    lines.append("-" * 60)
    cm = metrics["confusion_matrix"]
    lines.append(f"  TP: {cm['tp']:>4}  FP: {cm['fp']:>4}")
    lines.append(f"  FN: {cm['fn']:>4}  TN: {cm['tn']:>4}")
    lines.append("-" * 60)
    lines.append("  PER-DETECTOR RECALL (attack locations)")
    lines.append("-" * 60)
    for det, rec in metrics["per_detector_recall"].items():
        lines.append(f"  {det:<20s}  {rec:.2%}")
    lines.append("-" * 60)
    lines.append("  TIME-TO-DETECT (per campaign)")
    lines.append("-" * 60)
    for ttd in metrics["time_to_detect"]:
        hours = ttd["hours_to_detect"]
        h_str = f"{hours:.1f}h" if hours is not None else "NOT DETECTED"
        lines.append(f"  {ttd['campaign'][:40]:<40s}  {h_str}")
    lines.append("=" * 60)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def run_evaluation(
    threshold: float = 0.50,
    output_dir: str = "data",
) -> dict[str, Any]:
    """
    Run the full evaluation pipeline.

    Args:
        threshold: Score threshold for flagging reviews.
        output_dir: Directory to save eval.json.

    Returns:
        Metrics dict.
    """
    seed_everything()
    df = generate_synthetic_reviews()

    metrics = compute_metrics(df, threshold=threshold)
    path = save_eval(metrics, output_dir=output_dir)
    table = print_eval_table(metrics)

    print(table)
    print(f"\nResults saved to: {path}")

    return metrics


if __name__ == "__main__":
    run_evaluation()
