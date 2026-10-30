# Review Shield

> AI/ML PoC for Cynbit's **Revio** product — detects coordinated fake/review-bomb campaigns and drafts brand-safe replies for multi-location brands.

---

## Quick Start

```bash
# 1. Install dependencies (Python 3.11)
pip install -r requirements.txt

# 2. (Optional) Set Claude API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run the app
streamlit run app.py
```

The app runs fully without an API key — it falls back to VADER + rule-based reply drafting.

---

## Project Structure

```
cynbit/
├── app.py                    # Streamlit entry-point (4 pages)
├── theme.py                  # Colour palette & Plotly template
├── requirements.txt
├── pytest.ini
├── .streamlit/config.toml
├── src/
│   ├── data_loader.py        # Synthetic dataset generator + CSV loader
│   ├── reply_engine.py       # Reply drafter (Claude / VADER fallback)
│   ├── utils.py              # Seed, DetectorResult type, helpers
│   └── detectors/
│       ├── velocity.py       # Velocity-burst detector
│       ├── similarity.py     # Semantic-similarity cluster detector
│       ├── account.py        # Account-freshness detector
│       ├── rating_drop.py    # Sudden rating-drop detector
│       └── ensemble.py       # Weighted ensemble scorer
└── tests/
    ├── test_velocity.py
    ├── test_similarity.py
    ├── test_account.py
    ├── test_rating_drop.py
    └── test_ensemble.py
```

---

## Detectors

| Detector | Signal | Weight |
|---|---|---|
| **Velocity** | Sudden review burst vs baseline rate | 35% |
| **Similarity** | Semantic near-duplicate clustering | 30% |
| **Account Freshness** | High proportion of brand-new accounts | 20% |
| **Rating Drop** | Anomalous rolling-average drop (z-score) | 15% |

Every detector returns `{"score": float, "evidence": dict}` — never a bare number.

---

## Reply Workbench

- Drafts replies using **Claude** (if `ANTHROPIC_API_KEY` is set) or **VADER + rule-based** templates.
- Every draft shows **Approve / Edit / Reject** — nothing is ever auto-posted.
- Blocked words (fake, bot, coordinated, attack, …) are auto-sanitised from all replies.

---

## Running Tests

```bash
pytest
# or with coverage
pytest --cov=src --cov-report=term-missing
```

---

## CSV Upload Format

| Column | Type | Required |
|---|---|---|
| `reviewer_id` | str | ✅ |
| `location` | str | ✅ |
| `platform` | str | ✅ |
| `rating` | int (1–5) | ✅ |
| `text` | str | ✅ |
| `timestamp` | datetime | ✅ |
| `account_age_days` | int | ✅ |
| `is_attack` | bool | optional |
| `attack_type` | str | optional |

---

## API Key Security

Keys are **never hardcoded**. The app reads `ANTHROPIC_API_KEY` from:
1. Environment variables (`export ANTHROPIC_API_KEY=...`)
2. `.streamlit/secrets.toml` (`ANTHROPIC_API_KEY = "..."`)
