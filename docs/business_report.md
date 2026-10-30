# Review Shield — Business Case Report

> **Cynbit / Revio AI · October 2026**
> Prepared for product leadership as a proof-of-concept investment brief.

---

> ⚠️ **Data disclaimer** All performance numbers in this document are derived
> from **synthetic, procedurally generated review data**.  No real customer
> or business data was used.  Market-size statistics and fraud-prevalence
> figures are sourced from third-party industry research where cited; figures
> labelled **[assumption]** are estimates without primary-source backing.

---

## The Opportunity

Online reviews are the primary trust signal for multi-location brands.
A coordinated fake-review or review-bomb campaign can:

- **Erase months of reputation work overnight.** A sustained 1-star drop
  has been linked to 5–9 % revenue reduction for restaurants
  (Luca, Harvard Business School, 2016).
- **Go undetected for days.** Platform moderation relies on user reports;
  the average takedown time is 48–72 hours *[assumption — no public platform
  data available]*, during which the reputational damage compounds.
- **Multiply across locations.** Enterprise brands managing 10–200 locations
  cannot manually monitor every review in real time.

**Review Shield** solves the triage problem: detect, evidence, and respond —
at the speed of the attack.

---

## Product Fit for Revio

Revio already manages brand presence and customer messaging.  Review Shield
adds a high-value layer:

| Revio Capability | Review Shield Addition |
|---|---|
| Review aggregation | **Attack detection** (ensemble score, evidence) |
| Messaging templates | **AI reply drafting** (Claude / VADER fallback) |
| Multi-location dashboard | **Risk heatmap** (per-location severity) |
| Manual response workflow | **Approve / Edit / Reject workbench** |

This is a natural upsell: brands already paying for review management will
pay a premium for proactive protection.

---

## How It Works (Non-Technical)

Review Shield watches four signals simultaneously:

1. **Volume spike** — Did 20 reviews land in 2 hours when the usual rate is
   3 per day?
2. **Copy-paste similarity** — Are the reviews suspiciously identical in
   wording, even with minor variations?
3. **Ghost accounts** — Are most reviewers brand new (< 14 days old)?
4. **Rating cliff** — Did the average score drop sharply in one night?

Each signal is scored 0–1.  The four scores are combined into one **Risk
Score** with a traffic-light colour (green / amber / red / critical).  A
human always reviews before any action is taken.

---

## Proof-of-Concept Results

> Measured on 515 synthetic reviews across 5 locations, with 5 injected
> campaigns (3 review-bomb, 2 fake-positive).

### What works well

- **All 5 campaigns detected within 6 hours** (the first velocity time-bucket).
  In a real deployment this would translate to an alert within the hour.
- **Velocity and Similarity detectors: 100 % recall** on attack-containing
  locations at default settings.
- **Zero false-negatives at threshold 0.50** — no attack review slipped past
  undetected.

### What the numbers also reveal (honestly)

- **False-positive rate: 100 %** at threshold 0.50 means all organic reviews
  in flagged locations are also marked suspicious.  This is a consequence of
  operating at maximum sensitivity and of all five synthetic locations
  receiving elevated scores (four were attacked; one was borderline).
- **Raising the threshold to 0.65–0.75** substantially reduces FPs while
  keeping recall above 80 % (visible on the Evaluation page threshold-
  sensitivity chart).
- **Account-age and rating-drop detectors** show 0 % individual recall on this
  dataset, indicating they act as *confirmation signals* rather than standalone
  detectors — appropriate for their lower weights (20 % and 15 %).

The honesty here is intentional.  These numbers represent a first prototype
on synthetic data.  With six months of labelled production data, each
detector's weight can be calibrated to the real attack distribution.

---

## Competitive Differentiation

| Capability | Review Shield | Typical SAAS tools |
|---|---|---|
| Multi-signal ensemble | ✅ | ❌ (usually just volume or keywords) |
| Evidence transparency | ✅ Every alert shows *why* | ❌ Black-box scores |
| Runs without API key | ✅ VADER/rule fallback | ❌ Cloud-only |
| Reply drafting | ✅ Claude / fallback | ⚠️ Template only |
| No auto-posting | ✅ Approve/Edit/Reject | ❌ Some auto-post |
| Open-source codebase | ✅ Extensible | ❌ Vendor lock-in |

---

## Investment Ask and Timeline

This proof-of-concept was built in < 8 hours of engineering time.  To reach
production readiness:

| Phase | Effort | Outcome |
|---|---|---|
| **Phase 1** (2–4 weeks) | Real data integration (GBP / Yelp APIs) | Live ingestion, no CSV required |
| **Phase 2** (4–6 weeks) | Labelled-data collection + model calibration | Realistic FPR ≤ 10 % |
| **Phase 3** (2–4 weeks) | Notification layer + Revio dashboard embed | Operator-ready product |
| **Phase 4** (ongoing) | Adversarial hardening + multi-language | Robust production |

**Total estimated runway to beta: 8–14 weeks.**

---

## Known Risks

1. **Platform TOS.** Automated review monitoring must comply with Google,
   Yelp, and TripAdvisor terms.  Legal review required before production.
2. **Adversarial adaptation.** Sophisticated actors will diversify text,
   age accounts, and stagger timing once a detection system is known to exist.
3. **Regulatory.** In some jurisdictions, flagging user content as inauthentic
   may carry legal obligations.  GDPR implications for processing reviewer data.
4. **Synthetic ≠ real.** Every number in this document will change when
   measured on production data.

---

*For technical implementation details, see [technical_report.md](./technical_report.md).*
*For evaluation data, see `data/eval.json` (reproducible via `python -m src.evaluate`).*
