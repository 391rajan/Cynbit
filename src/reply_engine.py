"""
src/reply_engine.py
===================
Brand-safe reply drafter for Review Shield.

Two backends, selected automatically:
1. **Claude** (via Anthropic API) – if ANTHROPIC_API_KEY is available.
2. **Rule-based + VADER** – always available, no API key required.

Replies are NEVER auto-posted (Rule 4).  The engine only *drafts*;
the UI shows Approve / Edit / Reject for every draft.
"""

from __future__ import annotations

import os
import re
import textwrap
from typing import Literal

# VADER sentiment (ships with nltk; we use vaderSentiment directly)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    _vader = SentimentIntensityAnalyzer()
    _VADER_OK = True
except ImportError:
    _VADER_OK = False

# ---------------------------------------------------------------------------
# Tone constants
# ---------------------------------------------------------------------------
ReviewSentiment = Literal["negative", "neutral", "positive"]

_NEGATIVE_OPENS = [
    "Thank you for taking the time to share your experience.",
    "We sincerely appreciate your feedback.",
    "We're truly sorry to hear about your visit.",
]
_POSITIVE_OPENS = [
    "Thank you so much for the kind words!",
    "We're thrilled you enjoyed your experience!",
    "What a wonderful review — we really appreciate it!",
]
_NEUTRAL_OPENS = [
    "Thank you for your honest feedback.",
    "We appreciate you sharing your thoughts.",
    "Thanks for stopping by and leaving a review.",
]
_CLOSINGS = [
    "We hope to welcome you back soon.",
    "We look forward to seeing you again.",
    "Your experience matters to us and we're committed to doing better.",
]

# Sensitive words that must not appear in replies
_BLOCKED_PATTERNS = [
    r"\bfake\b", r"\bbot\b", r"\bcoordinated\b", r"\battack\b",
    r"\bmanipulat\w+", r"\bscam\b",
]


# ---------------------------------------------------------------------------
# Sentiment helper
# ---------------------------------------------------------------------------
def _classify_sentiment(text: str) -> ReviewSentiment:
    """
    Classify review sentiment as negative / neutral / positive.

    Uses VADER if available, otherwise falls back to simple keyword matching.

    Args:
        text: Raw review text.

    Returns:
        One of 'negative', 'neutral', 'positive'.
    """
    if _VADER_OK:
        scores = _vader.polarity_scores(text)
        compound = scores["compound"]
        if compound <= -0.2:
            return "negative"
        if compound >= 0.2:
            return "positive"
        return "neutral"

    # Keyword fallback
    lower = text.lower()
    neg_kw = {"terrible", "awful", "worst", "horrible", "disgusting", "never", "avoid"}
    pos_kw = {"great", "amazing", "excellent", "wonderful", "fantastic", "love", "best"}
    neg_hits = sum(1 for w in neg_kw if w in lower)
    pos_hits = sum(1 for w in pos_kw if w in lower)
    if neg_hits > pos_hits:
        return "negative"
    if pos_hits > neg_hits:
        return "positive"
    return "neutral"


def _sanitise(text: str) -> str:
    """Remove any blocked patterns from a reply draft."""
    for pat in _BLOCKED_PATTERNS:
        text = re.sub(pat, "***", text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# Rule-based backend
# ---------------------------------------------------------------------------
def _rule_based_reply(
    review_text: str,
    location: str,
    rating: int,
    is_flagged: bool,
) -> str:
    """
    Generate a brand-safe reply using templates and VADER sentiment.

    Args:
        review_text: Original review text.
        location: Location name for personalisation.
        rating: Star rating (1–5).
        is_flagged: Whether the review is part of a suspected campaign.

    Returns:
        Draft reply string.
    """
    sentiment = _classify_sentiment(review_text)

    if sentiment == "negative" or rating <= 2:
        opening = _NEGATIVE_OPENS[hash(review_text) % len(_NEGATIVE_OPENS)]
        body = (
            f"We're concerned to hear that your visit to our {location} location "
            "did not meet your expectations. "
            "We take all feedback seriously and will review this internally. "
            "We would love the opportunity to make things right — please reach out "
            "to us directly at feedback@brand.com so we can assist you personally."
        )
        closing = _CLOSINGS[-1]
    elif sentiment == "positive" or rating >= 4:
        opening = _POSITIVE_OPENS[hash(review_text) % len(_POSITIVE_OPENS)]
        body = (
            f"Hearing that you had a great experience at {location} really makes "
            "our day. Our team works hard to deliver quality service and fresh food, "
            "and your review motivates us to keep that standard high."
        )
        closing = _CLOSINGS[0]
    else:
        opening = _NEUTRAL_OPENS[hash(review_text) % len(_NEUTRAL_OPENS)]
        body = (
            f"We're glad you visited our {location} location and appreciate "
            "you letting us know about your experience. "
            "We're always looking for ways to improve."
        )
        closing = _CLOSINGS[1]

    draft = f"{opening}\n\n{body}\n\n{closing}"
    return _sanitise(textwrap.dedent(draft).strip())


# ---------------------------------------------------------------------------
# Claude backend
# ---------------------------------------------------------------------------
def _claude_reply(
    review_text: str,
    location: str,
    rating: int,
    is_flagged: bool,
    ensemble_score: float,
) -> str:
    """
    Generate a reply via the Anthropic Claude API.

    Args:
        review_text: Original review text.
        location: Location name.
        rating: Star rating.
        is_flagged: Whether the detector flagged this review.
        ensemble_score: Ensemble risk score (0-1).

    Returns:
        Draft reply string.

    Raises:
        RuntimeError: If the API call fails.
    """
    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "anthropic package not installed. Run: pip install anthropic"
        ) from exc

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    flag_context = ""
    if is_flagged and ensemble_score > 0.5:
        flag_context = (
            "\n\nIMPORTANT: This review has been flagged as potentially part of a "
            "coordinated campaign (confidence {:.0%}). Do NOT mention this in the reply. "
            "Reply professionally and factually as if it is a genuine review. "
            "Do not be defensive or accusatory.".format(ensemble_score)
        )

    system_prompt = (
        "You are a professional brand reputation manager drafting public-facing "
        "replies to customer reviews for a multi-location food & beverage brand. "
        "Your replies must be:\n"
        "- Warm, professional, and empathetic\n"
        "- Brand-safe (no legal admissions, no promises you can't keep)\n"
        "- Under 120 words\n"
        "- Free of emojis\n"
        "- Never accusatory or defensive\n"
        "- End with an invitation to reconnect"
    )

    user_prompt = (
        f"Location: {location}\n"
        f"Star rating: {rating}/5\n"
        f"Review text:\n\"\"\"\n{review_text}\n\"\"\"\n"
        f"{flag_context}\n\n"
        "Draft a single reply paragraph. Do not include any preamble or explanation."
    )

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()
    return _sanitise(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _get_api_key() -> str | None:
    """Return the Anthropic API key from env or Streamlit secrets (Rule 3)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        import streamlit as st  # type: ignore
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def draft_reply(
    review_text: str,
    location: str,
    rating: int,
    is_flagged: bool = False,
    ensemble_score: float = 0.0,
) -> dict[str, str]:
    """
    Draft a brand-safe reply for a given review.

    Automatically selects the Claude backend if an API key is available,
    otherwise uses the rule-based VADER backend.

    Args:
        review_text: Original review text.
        location: Location name for personalisation.
        rating: Star rating (1–5).
        is_flagged: Whether the review is part of a suspected campaign.
        ensemble_score: Ensemble risk score (0–1) for context.

    Returns:
        Dict with keys:
            ``text`` (str): The draft reply.
            ``backend`` (str): Which backend was used.
            ``sentiment`` (str): Detected sentiment of the original review.
    """
    sentiment = _classify_sentiment(review_text)
    api_key = _get_api_key()

    if api_key:
        try:
            text = _claude_reply(
                review_text=review_text,
                location=location,
                rating=rating,
                is_flagged=is_flagged,
                ensemble_score=ensemble_score,
            )
            return {"text": text, "backend": "Claude (Anthropic)", "sentiment": sentiment}
        except Exception as exc:  # noqa: BLE001
            # Fall through to rule-based on any Claude error
            fallback_note = f" [Claude error: {exc}]"
    else:
        fallback_note = ""

    text = _rule_based_reply(
        review_text=review_text,
        location=location,
        rating=rating,
        is_flagged=is_flagged,
    )
    return {
        "text": text,
        "backend": f"Rule-based + VADER{fallback_note}",
        "sentiment": sentiment,
    }
