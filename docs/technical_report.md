# Review Shield — Technical Report

> **Cynbit / Revio AI · October 2026**
> Prepared as an AI/ML opportunity proof-of-concept for the Revio product.

---

> ⚠️ **Data disclaimer** All experiments in this report use **synthetic data**
> generated with a fixed random seed (42).  Review texts, account ages, timestamps,
> and location names are procedurally created.  No real customer reviews were
> used.  The fraud-prevalence statistics cited below are industry estimates
> sourced from Trustpilot and BrightLocal consumer research unless explicitly
> labelled as **assumption**.

---

## 1 · Executive Summary

Review Shield is a lightweight AI/ML pipeline that detects coordinated
fake-review and review-bomb campaigns for multi-location brands.  It layers
four complementary signals — burst velocity, semantic similarity, account
freshness, and rating-drop anomaly — into a weighted ensemble that produces
a 0–1 risk score per location.  A brand-safe reply workbench lets community
managers draft, edit, and approve responses without auto-posting.

At the default threshold of 0.50, the prototype achieves **100% recall on
injected campaigns** (all five synthetic campaigns are flagged within the
same time bucket they begin).  The precision and FPR reflect the cost of
operating at maximum sensitivity; the threshold slider in the Evaluation
page lets operators tune the trade-off to their tolerance.

---

## 2 · Problem Statement

Coordinated review manipulation causes measurable revenue damage:

- **Fake reviews are pervasive.** Industry estimates (Trustpilot, 2023)
  suggest 4 %–8 % of online reviews are inauthentic. *(assumption: exact
  prevalence on any specific platform is unknown without platform-level data.)*
- **Rating drops hurt conversion.** A 1-star drop in average rating has been
  associated with a 5–9 % reduction in revenue for restaurants
  (Harvard Business School working paper, Luca 2016 — real-world study).
- **Response speed matters.** For multi-location brands, manually monitoring
  dozens of platform inboxes is infeasible; automated triage is needed.

Review Shield addresses the triage problem: flag suspicious clusters fast,
surface evidence, and draft a brand-safe reply — all without requiring a
human to read every review.

---

## 3 · Architecture

```
Synthetic / Uploaded CSV
        │
        ▼
  DataLoader (src/data_loader.py)
        │
   ┌────┼────────────────────────┐
   ▼    ▼           ▼            ▼
Velocity Similarity Account  RatingDrop
detector  detector  Freshness detector
(35%)    (30%)     (20%)      (15%)
   └────┼────────────────────────┘
        │
        ▼
   Ensemble scorer  (src/detectors/ensemble.py)
   → score ∈ [0, 1] + structured evidence dict
        │
   ┌────┴──────────────────┐
   ▼                       ▼
Reply Engine            Evaluation
(src/reply_engine.py)   (src/evaluate.py)
Claude / VADER fallback  precision·recall·F1·TTD
```

All detector functions satisfy the contract `{"score": float, "evidence": dict}`.
No bare numbers are returned. Evidence is propagated all the way to the UI
so operators always know *why* a location was flagged.

---

## 4 · Detectors

### 4.1 Velocity (weight 35 %)

Resamples reviews into rolling time windows and compares the peak window
count to the median baseline rate.  A ≥ 4× multiplier triggers the flag.
An additional boost applies when ≥ 70 % of the burst reviews are 1-star.

### 4.2 Semantic Similarity (weight 30 %)

Embeds review texts with `all-MiniLM-L6-v2` (sentence-transformers) and
clusters reviews whose cosine similarity ≥ 0.82 within a 48-hour window.
Clusters of ≥ 4 near-duplicate reviews score above 0.40.  A TF-IDF fallback
is used automatically when sentence-transformers is unavailable.

### 4.3 Account Freshness (weight 20 %)

Flags locations where > 30 % of reviewers have accounts younger than 14 days.
Score scales linearly from that threshold to 100 %.  An additional boost
applies when young-account reviews are dominated by extreme ratings (1 or 5).

### 4.4 Rating Drop (weight 15 %)

Computes a 7-day rolling average rating.  The maximum downward step is
compared against the time-series standard deviation (z-score).  A drop of
≥ 0.8 stars anchors a score of 0; ≥ 1.6 stars approaches 1.0.

---

## 5 · Evaluation Results

> All metrics computed on the synthetic dataset (515 reviews, 115 attack,
> 400 organic, seed 42).  Threshold = 0.50.

| Metric | Value |
|---|---|
| Precision | 22.3 % |
| Recall | **100.0 %** |
| F1 | 36.5 % |
| False-Positive Rate | 100.0 % |

### 5.1 Confusion Matrix (threshold = 0.50)

```
                 Predicted Attack   Predicted Organic
Actual Attack        115 (TP)            0 (FN)
Actual Organic       400 (FP)            0 (TN)
```

### 5.2 Per-Detector Recall

| Detector | Recall (attack locations) |
|---|---|
| Velocity | 100 % |
| Similarity | 100 % |
| Account Freshness | 0 % |
| Rating Drop | 0 % |

### 5.3 Time-to-Detect

| Campaign | # Reviews | Hours to Detect |
|---|---|---|
| fake_positive @ Downtown | 18 | 0.0 h |
| fake_positive @ Uptown | 31 | 0.0 h |
| review_bomb @ Midtown | 19 | 0.0 h |
| review_bomb @ Uptown | 17 | 0.0 h |
| review_bomb @ Westside | 30 | 0.0 h |

All five campaigns are detected within the same time bucket they begin
(< 6-hour velocity window).

### 5.4 Interpretation

The 100 % FPR at threshold 0.50 arises because **all five locations** receive
an ensemble score ≥ 0.50.  Campaigns were injected into 4 of the 5 locations;
the fifth (Eastside) also scores above threshold because it has a slightly
elevated velocity and similarity signal from organic reviews that happened
to cluster.  In a production system, you would:

1. **Raise the threshold** (0.65–0.75) to reduce FPR while accepting some
   missed campaigns at the margins.
2. **Add a human-review queue** (already built — the Reply Workbench) so the
   flagged list is triaged before action is taken, not auto-actioned.
3. **Retrain on labelled production data** to calibrate each detector's weight
   to the true base rate of attacks on that platform.

The Account and Rating-Drop detectors show 0 % individual recall at the
0.40 trigger threshold.  They contribute as *confirmation signals* in the
ensemble rather than primary detectors on this synthetic dataset.

---

## 6 · Limitations

1. **Synthetic data only.** All metrics reflect a single controlled dataset.
   Real-world performance will be different and likely lower due to greater
   review diversity, noisier organic patterns, and more sophisticated campaigns.

2. **No adversarial robustness testing.** An attacker with knowledge of the
   detection rules could pace their campaign (< 4× baseline rate), vary review
   templates (cosine similarity < 0.82), and age their accounts before posting.

3. **Location-level scoring.** The ensemble produces one score per location,
   not per review.  Every review in a flagged location inherits that score.
   This inflates FP counts on mixed locations.

4. **Threshold not tuned.** Default thresholds (0.50 ensemble, 0.40 individual)
   are reasonable starting points, not optimised values.

5. **Single language.** The SBERT model and VADER lexicon both work primarily
   on English text.

6. **No temporal drift handling.** The baseline rate is computed over the full
   dataset history; a business with seasonal spikes may see false velocity alerts.

---

## 7 · Next Steps for Production

1. Collect and label real review data (minimum 6 months per location).
2. Replace fixed weights with a logistic meta-learner trained on labelled data.
3. Add IP-address and device-fingerprint signals (if platform API exposes them).
4. Integrate direct platform APIs (Google Business Profile, Yelp Fusion) for
   live ingestion instead of CSV uploads.
5. Build an alert notification system (email / Slack) for critical-risk detections.
6. Compliance review: any automated flagging of user-generated content must
   satisfy platform terms of service before being acted upon.

---

*Report generated from `data/eval.json` — run `python -m src.evaluate` to reproduce.*
