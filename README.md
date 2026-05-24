# Table Tennis Serve Analysis

## Overview
This project analyzes point-level table tennis serve data to identify which serve patterns are most associated with winning points, forcing weak returns, and creating favorable rally structures. The current framing is a **contextual serve recommendation system**: can serve characteristics, match context, and opponent profile predict point outcomes and identify high-value serve strategies?

## Motivation
As a competitive table tennis player rated around 1910 Elo, I wanted to use data science to evaluate the strongest part of my game: serving. The goal is to move beyond intuition and build a repeatable framework for serve selection.

## Research Questions
1. Which serve attributes are most predictive of point outcome?
2. How do serve effects vary by opponent style and skill level?
3. Which serves successfully create the intended rally type?
4. Can a model recommend high-value serves **before the point begins**?

## Dataset
- 500 serve-level observations
- 8 matches
- 8 opponents
- One row per served point

Raw data is stored in `data/raw/table_tennis_serves.csv`.

## Analytical Tracks
### Model A: Pre-point serve recommendation (decision model)
Use only variables known before service contact.

**Feature set (pre-point only):**
- `serve_type`, `spin_type`, `spin_intensity`, `serve_length`, `placement_zone`
- `toss_height`, `contact_point`, `game_state`, `server_score`, `receiver_score`
- `opponent_skill_level`, `opponent_style`, `side`
- engineered features like `score_margin`, `is_pressure_point`, `serve_combo`

**Target:** `point_outcome` (or binary `point_win`)

### Model B: Post-point mechanism/diagnostic (explanatory model)
Use post-serve variables to explain *why* outcomes occur (not to make pre-point recommendations):
- `return_type`, `return_quality`, `rally_length`, `point_end_type`
- `rally_type_achieved`, `chop_rally_outcome`

## Methods
- Exploratory data analysis
- Feature engineering
- Baseline win-rate benchmark
- L1-regularized logistic regression (LASSO)
- Random forest (extendable to gradient boosting)
- Match-level validation (`leave-one-match-out` / grouped split)
- Reliability-aware serve recommendation scoring

## Evaluation Plan
- Accuracy
- ROC-AUC
- Precision / Recall
- Log loss
- Confusion matrix
- Calibration curve

Because observations within each match are related, evaluation should avoid naive random row splits and instead use `match_id`-aware validation.

## Reliability & Uncertainty
To avoid overclaiming from small samples:
- Apply minimum attempt thresholds (e.g., `n >= 20`)
- Add 95% confidence intervals for serve-combo win rates
- Provide a reliability label (`Low`, `Medium`, `High`) tied to sample size

## Serve Recommendation Score
A tactical score can rank candidate serves by combining predictive value and reliability:

\[
\text{Serve Score} = 0.50\,P(\text{win}) + 0.20\,P(\text{weak return}) + 0.15\,P(\text{intended rally}) + 0.15\,\text{reliability}
\]

This reframes output into actionable recommendations by context (opponent style, game state, score pressure).

## Repository Structure
```text
Table-Tennis-Serve-Analysis/
├── README.md
├── data/
│   ├── raw/
│   │   └── table_tennis_serves.csv
│   └── processed/
│       └── serves_cleaned.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_serve_recommendation.ipynb
├── src/
│   ├── clean_data.py
│   ├── features.py
│   ├── train_model.py
│   └── recommend_serves.py
├── reports/
│   └── figures/
├── requirements.txt
└── .gitignore
```

## Notebook Plan
1. Project objective
2. Dataset overview and dictionary
3. Data quality checks
4. Outcome distributions
5. Serve feature and interaction analysis
6. Opponent-style effects
7. Intended setup vs achieved rally type
8. Reliability/sample-size checks
9. Modeling implications
10. Final tactical recommendations

## Key Reporting Language
Use association language for observational data.

- Preferred: “Heavy spin is associated with higher point win rate.”
- Avoid: “Heavy spin causes higher point win rate.”

## Next Steps
- Expand dataset to 100+ matches
- Add video-linked labeling validation
- Add clean visuals under `reports/figures/` and surface 3–5 in this README
- Build a lightweight Streamlit serve recommendation dashboard
