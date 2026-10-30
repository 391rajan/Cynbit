"""
src/utils.py
============
Shared helpers used across Review Shield modules.

Includes:
- Fixed random seed constant and seeded RNG factory
- Typing aliases for detector contracts
- Small formatting helpers
"""

from __future__ import annotations

import random
from typing import Any, TypedDict

import numpy as np

# ---------------------------------------------------------------------------
# Fixed randomness seed (Rule 6)
# ---------------------------------------------------------------------------
SEED: int = 42


def make_rng(seed: int = SEED) -> np.random.Generator:
    """Return a seeded NumPy default RNG instance."""
    return np.random.default_rng(seed)


def seed_everything(seed: int = SEED) -> None:
    """Seed Python stdlib random and NumPy globally."""
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Detector contract types  (Rule 2)
# ---------------------------------------------------------------------------
class DetectorResult(TypedDict):
    """Strict return type for every detector function."""
    score: float        # Risk probability in [0, 1]
    evidence: dict[str, Any]   # Arbitrary structured evidence


def validate_result(result: DetectorResult) -> DetectorResult:
    """
    Validate that a detector result conforms to the contract.

    Args:
        result: Raw dict returned by a detector.

    Returns:
        The same dict, validated.

    Raises:
        ValueError: If the result is missing required keys or score is out of range.
    """
    if "score" not in result:
        raise ValueError("Detector result missing 'score' key")
    if "evidence" not in result:
        raise ValueError("Detector result missing 'evidence' key")
    score = result["score"]
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"Detector score {score!r} is not in [0, 1]")
    return result


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def pct(value: float, decimals: int = 1) -> str:
    """Format a 0-1 float as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))
