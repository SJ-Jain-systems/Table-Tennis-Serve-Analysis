# Table Tennis Serve Analysis

A serve-level table tennis analytics project that turns match observations into a practical, context-aware serve recommendation workflow. The project combines exploratory analysis, feature engineering, supervised modeling, and reliability-aware recommendation scoring to answer a tactical question:

> Given the score, opponent profile, and available serve options, which serves are most likely to create a favorable point outcome?

The analysis is intentionally framed around **association and decision support**, not causal proof. The dataset is small and player-specific, so results should be interpreted as evidence for serve-selection tendencies rather than universal table tennis rules.

---

## Project Snapshot

| Area | Current State |
| --- | --- |
| Dataset size | 500 serve-level observations |
| Match coverage | 8 matches against 8 opponents |
| Raw data | `data/table_tennis_serves.csv` |
| Processed data | `data/processed/table_tennis_serves_features.csv` |
| Modeling target | Binary point outcome: server won vs. lost |
| Best cross-validated model in notebooks | LASSO logistic regression |
| Recommendation output | Ranked serve options with predicted value and reliability context |
| Primary validation approach | Match-aware grouped cross-validation |

---

## Motivation

As a competitive table tennis player, I wanted a structured way to evaluate the strongest part of my game: serving. Serve selection often depends on feel, memory, and post-match intuition. This project turns those intuitions into a repeatable data workflow by tracking each serve, engineering tactical context, and estimating which serve patterns are most associated with winning points or creating weaker returns.

The long-term goal is to support between-point decision-making: not by replacing tactical judgment, but by giving the player a clearer view of what has worked, against whom, and under what conditions.

---

## Research Questions

1. **Serve attributes:** Which serve characteristics are most associated with winning the point?
2. **Opponent effects:** How do serve outcomes vary by opponent style and skill level?
3. **Rally setup:** Which serves are most likely to produce the intended rally structure?
4. **Pressure context:** Do score state and late-game pressure change which serves look strongest?
5. **Recommendation:** Can a model rank serve choices using only information known before the point begins?

---

## Dataset

The raw dataset contains one row per served point. Each observation combines pre-serve decision variables, match context, opponent information, and post-serve outcome details.

### Raw Data Columns

The raw file contains 25 columns:

- **Serve characteristics:** `serve_type`, `spin_type`, `spin_intensity`, `serve_length`, `placement_zone`, `toss_height`, `contact_point`, `side`
- **Match context:** `match_id`, `game_number`, `server_score`, `receiver_score`, `game_state`
- **Opponent profile:** `opponent_id`, `opponent_skill_level`, `opponent_style`
- **Return and rally diagnostics:** `return_type`, `return_quality`, `return_placement`, `rally_length`, `point_end_type`
- **Outcome and tactical intent:** `point_outcome`, `intended_setup`, `rally_type_achieved`, `chop_rally_outcome`

### Important Modeling Constraint

The project separates variables into two groups:

1. **Pre-point variables** that are known before serving and can be used for prediction or recommendation.
2. **Post-serve variables** that describe what happened after contact and are useful for diagnosis, but must not be used to recommend a serve before the point starts.

This separation prevents data leakage. For example, `return_quality`, `rally_length`, and `point_end_type` are valuable explanatory fields, but they are unavailable at the moment a serve decision is made.

A second, subtler leakage source exists even among pre-point fields: historical aggregate features (like a serve combo's win rate) can leak label information if a point's own outcome is included in its own aggregate. See **Historical Combo Statistics** below for how this is handled.

---

## Analytical Design

### Track A: Pre-Point Serve Recommendation Model

This is the decision-support track. It uses only information available before service contact.

Representative feature groups include:

- Serve type, spin, length, placement, toss, contact point, and serving side
- Score state, score margin, game state, and pressure flags
- Opponent style and skill level
- Engineered serve combinations such as `serve_spin_combo`, `serve_length_spin_combo`, `serve_placement_combo`, and `full_serve_combo`
- Historical combination statistics such as `combo_attempts`, `combo_win_rate`, and `combo_reliability`

The target is a binary point outcome indicating whether the server won the point.

### Track B: Post-Point Mechanism Analysis

This is the explanatory track. It studies why particular serves may work by examining variables that occur after the serve:

- Return type and return quality
- Return placement
- Rally length
- Point-ending mechanism
- Whether the intended rally type was achieved
- Chop-rally outcome when applicable

These variables help interpret the serve's tactical mechanism, but they are excluded from pre-point modeling.

---

## Feature Engineering

The processed feature dataset expands the raw 25-column dataset into 52 columns. Engineered features are designed to capture tactical context that a raw row does not express directly.

### Key Engineered Feature Families

| Feature family | Examples | Purpose |
| --- | --- | --- |
| Outcome encoding | `point_won` | Converts point outcome into a binary modeling target |
| Score state | `score_margin`, `is_tied`, `is_trailing`, `is_leading` | Captures whether the server is ahead, behind, or level |
| Late-game pressure | `is_late_game`, `is_deuce_or_later`, `is_game_point_for_server`, `is_game_point_against_server`, `is_high_pressure` | Represents tactically different pressure states |
| Opponent indicators | `opponent_is_looper`, `opponent_is_chopper`, `opponent_is_attacker`, `opponent_skill_numeric` | Encodes opponent archetype and skill level |
| Serve combinations | `serve_spin_combo`, `serve_length_spin_combo`, `serve_placement_combo`, `full_serve_combo` | Models serves as combinations rather than isolated attributes |
| Spin flags | `is_heavy_spin`, `is_low_spin`, `spin_length_interaction` | Captures discrete tactical spin thresholds |
| Historical combo stats | `combo_attempts`, `combo_win_rate`, `combo_reliability` | Adds sample-size-aware historical context |

Notebook quality checks confirm that the engineered dataset contains 500 rows, 52 columns, 27 engineered features, and no nulls in the engineered columns.

### Historical Combo Statistics (and a leakage fix)

`combo_win_rate` and `combo_reliability` summarize how a specific serve combination has historically performed. The naive approach averages `point_won` across all historical uses of a combo, including the point's own row. That leaks the label into its own feature: a combo attempted once gets a `combo_win_rate` of exactly 0 or 1, identical to that single point's outcome. Grouped cross-validation by match does not catch this, because the same combo can recur across matches. The leak is row-level, not match-level.

The fix: `combo_win_rate` and `combo_reliability` are computed leave-one-out. Each point's own outcome is excluded from the aggregate describing it, with sparse combos falling back to the dataset's overall win rate. This dropped `combo_win_rate`'s correlation with the target out of the top 10 most-correlated features. It was previously the single strongest predictor by a wide margin. After the fix, the models rely on a genuinely diverse feature set instead of one near-tautological column.

---

## Modeling Approach

The modeling workflow evaluates whether pre-point features can predict point outcome better than a majority-class baseline.

### Models Evaluated

- **Majority-class baseline** for a simple reference point
- **LASSO logistic regression** for a sparse, interpretable linear model
- **Random forest classifier** for nonlinear interactions and feature importance analysis
- **Gradient boosting classifier** as an additional nonlinear benchmark

### Validation Strategy

The project uses match-aware validation rather than naive random row splits. Points from the same match are correlated because they share the same opponent, tactical adjustments, score dynamics, and session conditions. Randomly splitting rows can leak match-specific information between training and validation sets.

The notebooks therefore use grouped cross-validation (`GroupKFold` by `match_id`, 5 folds) so that validation folds better approximate performance on unseen match contexts.

### Notebook Results

| Model | Mean accuracy | Accuracy std. | Mean ROC-AUC | ROC-AUC std. |
| --- | ---: | ---: | ---: | ---: |
| Baseline (majority class) | 0.544 | — | n/a | n/a |
| LASSO logistic regression | 0.671 | 0.063 | 0.746 | 0.062 |
| Random forest | 0.644 | 0.023 | 0.696 | 0.083 |
| Gradient boosting | 0.617 | 0.051 | 0.646 | 0.069 |

LASSO is the best-performing model in the current notebook run, both by ROC-AUC and by Brier score, and is the model saved as the primary recommendation-system pipeline (`serve_win_probability_model.pkl`). All three models clear the majority-class baseline, with LASSO showing the largest and most consistent margin. Given the dataset size (500 points, 8 matches), these numbers should be read as a directional signal rather than a precise estimate — fold-to-fold std. is non-trivial relative to the gap between models.

---

## Exploratory Findings

Current exploratory analysis suggests several useful tactical patterns:

- **Heavy spin is associated with stronger outcomes.** In the EDA notebook, heavy spin showed a roughly 17 percentage-point win-rate gap over light spin.
- **Serve effectiveness varies by opponent style.** The best serve type differs across allround, blocker, defender, and looper opponents.
- **Combination-level patterns matter.** Serve type alone is often less informative than serve type combined with placement, length, and spin.
- **Reliability matters.** Some high win-rate combinations are based on small samples, so recommendation output should account for sample size rather than ranking by raw win rate alone.

These findings should be treated as player- and sample-specific associations until the dataset is expanded.

---

## Recommendation System

The recommendation workflow ranks candidate serves for a given match context. It combines the model's predicted win probability with historical combo-level performance and a sample-size reliability correction, so that the system does not overstate sparsely observed serve combinations.

The recommendation score implemented in `4_Serve_Recommendation.ipynb` is:

```text
Serve Score =
    0.70 * predicted point-win probability   (from the trained model)
  + 0.20 * historical combo win rate         (from observed match data)
  + 0.10 * combo reliability                 (capped sample-size correction)
```

The model-predicted probability carries the most weight because it is context-sensitive. It was trained jointly on serve attributes, score state, and opponent profile. The historical win rate adds serve-level signal the model may not fully capture, and the reliability term discounts combinations with few historical attempts so a 2-attempt outlier can't outrank a well-tested option.

The recommendation notebook demonstrates context-specific ranking scenarios such as a neutral point against a looper, a high-pressure deuce against an all-round opponent, and trailing against a defender, plus a single-variable score-margin sensitivity sweep.

**Caveat:** the underlying model draws on the full feature set rather than one dominant column, so predicted win probabilities span the full 0 to 1 range. Candidate serves with very low `combo_reliability` (little or no historical data) can receive extreme scores that look more confident than the evidence supports. Treat low-reliability recommendations with extra caution, or filter them out using the reliability threshold demonstrated in the notebook.

A separate limitation worth naming directly: `combo_win_rate` at inference time (notebook 4) is computed from the same historical data used to train the model, not refreshed leave-one-out per candidate. Combinations attempted frequently in the training data carry a more reliable historical signal than rarely-attempted ones, and which combinations get attempted often is itself a non-random artifact of how the player serves, not a random sample.

---

## Repository Structure

```text
Table-Tennis-Serve-Analysis/
├── README.md
├── LICENSE.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── table_tennis_serves.csv
│   └── processed/
│       └── table_tennis_serves_features.csv
├── models/
│   ├── lasso_model.pkl
│   ├── rf_model.pkl
│   ├── gb_model.pkl
│   ├── model_features.pkl
│   └── serve_win_probability_model.pkl
├── notebooks/
│   ├── 1_EDA.ipynb
│   ├── 2_Feature_Engineering.ipynb
│   ├── 3_Modeling.ipynb
│   └── 4_Serve_Recommendation.ipynb
├── src/
│   ├── clean_data.py
│   ├── features.py
│   ├── train_model.py
│   └── recommend_serves.py
└── reports/
    └── figures/
```

`src/` holds standalone script versions of the notebook pipeline (data cleaning, feature engineering, model training, and serve recommendation) for use outside a notebook environment, e.g. in a future dashboard or CLI tool.

---

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `1_EDA.ipynb` | Explores serve distributions, win rates, serve-placement combinations, rally length, spin effects, and opponent-style matchups |
| `2_Feature_Engineering.ipynb` | Builds score-state features, opponent encodings, serve-combination features, spin flags, interaction terms, and leakage-safe reliability statistics |
| `3_Modeling.ipynb` | Trains and evaluates baseline, LASSO logistic regression, random forest, and gradient boosting models with grouped validation |
| `4_Serve_Recommendation.ipynb` | Converts model predictions and reliability features into ranked serve recommendations for example match contexts |

---

## Installation

Create and activate a Python environment, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The main dependencies are pandas, NumPy, scikit-learn, matplotlib, seaborn, scipy, joblib, and Jupyter.

---

## Suggested Workflow

Run the notebooks in order:

1. `notebooks/1_EDA.ipynb`
2. `notebooks/2_Feature_Engineering.ipynb`
3. `notebooks/3_Modeling.ipynb`
4. `notebooks/4_Serve_Recommendation.ipynb`

Each notebook reads the previous notebook's output (raw CSV → processed CSV → saved model pipelines), so they're meant to be run in sequence on a fresh checkout.

---

## Interpretation Guidelines

Use careful language when reporting results:

- Prefer: **"Heavy spin is associated with a higher point win rate in this dataset."**
- Avoid: **"Heavy spin causes a higher point win rate."**

This distinction matters because the project uses observational match data. Serve selection is influenced by opponent quality, score state, tactical intent, and player confidence, so model outputs should be treated as decision-support evidence rather than causal conclusions.

---

## Limitations

- The dataset is small: 500 points across 8 matches.
- Observations come from one player's serving patterns and are not necessarily generalizable.
- Some serve combinations have low sample sizes, making raw win rates unstable.
- Opponents adapt during matches, so historical performance may not fully represent future performance.
- Model performance should be validated on more matches before being used for serious competitive planning.
- Recommendation scores for serve combinations with little or no historical data can be overconfident; see the caveat in **Recommendation System**.

---

## Future Work

- Expand the dataset to 100+ matches and more opponent archetypes.
- Add video-linked labeling for serve quality and return classification validation.
- Generate polished plots under `reports/figures/` and surface the most important visuals in this README.
- Build a one-page tactical summary for match preparation.
- Build a lightweight Streamlit dashboard for interactive serve recommendations using the `src/` scripts.
- Add a minimum-reliability filter (or confidence interval) directly into the recommendation notebook's top-N output, rather than as a separate manual filter step.

---

## License

MIT License. See `LICENSE.md` at the repository root. You're free to use, copy, modify, and distribute this code, with no warranty.
