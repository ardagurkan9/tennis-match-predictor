# Tennis Match Predictor

An end-to-end machine learning pipeline for predicting professional ATP tennis match outcomes using historical match data and pre-match betting odds.

The project focuses on reproducible, time-aware and leakage-free feature engineering. Each match is represented from both players' perspectives, and the model predicts whether the selected player wins:

```text
target_win = 1 if player wins
target_win = 0 if opponent wins
```

## Key Features

* ATP match history from 2000 through 2025
* Time-based train, validation and test splits
* Rolling recent-form and surface-form statistics
* Overall and surface-specific head-to-head features
* Elo and surface Elo ratings
* Rolling serve and return statistics
* Player workload and rest features
* Vig-free pre-match market probabilities
* Logistic Regression, Random Forest, XGBoost and LightGBM models

All chronological features use only information available before the predicted match. Current-match scores and serve statistics are excluded from model inputs.

## Pipeline Overview

`main.py` is the central entry point for both data pipelines. Its default `clean` mode runs the ingestion and cleaning check:

```text
main.py clean
└── ingest.py → clean.py → cleaned_matches.parquet
```

Its `advanced` mode delegates to the advanced dataset builder:

```text
main.py advanced → build_advanced_features.py
├── ingest.py
├── features.py (rolling serve/return history)
├── odds.py (pre-match market probabilities)
├── clean.py
├── features.py (form, H2H, workload, Elo and matchup features)
└── split.py → advanced train/validation/test files
```

Model training, baseline evaluation, ablation, and persistent reporting are
explicit separate steps:

```text
train.py    → trained model files
evaluate.py → rank-baseline metrics
ablation.py → time-based market-feature ablation reports
report.py   → model metrics and diagnostic plots
```

## Project Structure

```text
tennis-match-predictor/
├── data/
│   ├── raw/
│   │   └── odds/
│   ├── processed/
│   └── features/
│       └── advanced/
├── models/
├── reports/
├── src/
│   ├── ingest.py
│   ├── clean.py
│   ├── odds.py
│   ├── features.py
│   ├── build_advanced_features.py
│   ├── split.py
│   ├── train.py
│   ├── evaluate.py
│   ├── ablation.py
│   ├── report.py
│   ├── predict.py
│   ├── match_history.py
│   └── comparison.py
├── tests/
│   ├── test_leakage.py
│   ├── test_prediction.py
│   └── test_frontend.py
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

## Data Split

Random splitting is not used because tennis data is time-dependent.

| Purpose | Years | Used as model rows? |
|---|---|---|
| Historical feature warm-up | 2000–2014 | No |
| Train | 2015–2023 | Yes |
| Validation | 2024 | Yes |
| Retrospective test | 2025 | Yes |

All raw ATP CSV files from 2000 onward are processed chronologically. Matches from
2000–2014 initialize rolling form, serve statistics, H2H, Elo, and surface Elo, but
their rows are removed before model training. Modeling starts in 2015 because the
historical odds dataset covers 2015–2025.

## Installation

Create and activate a virtual environment, then install the dependencies.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Raw ATP CSV files belong in `data/raw/`. Historical odds workbooks belong in `data/raw/odds/`.

## Usage

Recommended advanced workflow:

```bash
python main.py advanced
python -m src.train --dataset advanced
python -m src.ablation
python -m src.report
python -m pytest
```

The commands below can also be run independently.

Run the ingestion and cleaning check (the default mode):

```bash
python main.py
# equivalent to: python main.py clean
```

Build the advanced feature dataset and its time-based splits:

```bash
python main.py advanced
```

Create base dataset splits:

```bash
python -m src.split
```

Train model candidates using either dataset:

```bash
# Base features
python -m src.train

# Advanced features; models are saved under models/advanced/
python -m src.train --dataset advanced
```

Evaluate the rank baseline using either dataset:

```bash
python -m src.evaluate
python -m src.evaluate --dataset advanced
```

Run the market-feature ablation with expanding-window cross-validation:

```bash
python -m src.ablation
```

This compares rank baseline, market baseline, market-only LightGBM, tennis-only
LightGBM, and the full model. The 2025 result is labeled as a retrospective
benchmark and is not used for experiment selection.

Run the automated leakage regression tests:

```bash
python -m pytest
```

Generate persistent model metrics and diagnostic plots:

```bash
python -m src.report
```

Predict a new matchup from historical state:

```bash
python -m src.predict \
  --player "Carlos Alcaraz" \
  --opponent "Jannik Sinner" \
  --surface Hard \
  --date 2026-01-15 \
  --tourney-level G \
  --best-of 5 \
  --round SF \
  --draw-size 128
```

Optional `--player-odds` and `--opponent-odds` values add a vig-free market
probability. If both are omitted, the model uses neutral market inputs and marks
odds as unavailable.

## Streamlit Frontend

A small Streamlit UI at `app.py` wraps `src.predict.predict_match()` with a
match-selection form, a color-coded stat comparison, player profiles, recent
form, and head-to-head history. It does not use live data or reimplement any
feature/prediction logic; it only calls the existing prediction pipeline.

Install dependencies (includes `streamlit`) and run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Outputs

The pipeline writes generated datasets, model artifacts, and reports to these
locations:

```text
data/processed/cleaned_matches.parquet
data/features/match_features.parquet
data/features/train.csv
data/features/validation.csv
data/features/test.csv
data/features/advanced/match_features.parquet
data/features/advanced/train.csv
data/features/advanced/validation.csv
data/features/advanced/test.csv
models/*.pkl
models/advanced/*.pkl
reports/ablation_results.csv
reports/ablation_metrics.json
reports/model_metrics.csv
reports/model_metrics.json
reports/confusion_matrix.png
reports/calibration_curve.png
```

## Current Status

The base and advanced modeling pipelines are complete. The saved advanced
LightGBM reaches 67.86% accuracy and 73.96% ROC-AUC on the retrospective 2025
benchmark.

The 2025 data was inspected during feature development, so it is not presented as
an untouched final test set. Ablation decisions use expanding-window validation
over 2020–2024; a future untouched year is required for a new final test.

The prediction CLI, a Streamlit graphical frontend, automated leakage tests, and
persistent evaluation reports are available. A broader odds-matching audit and
automated feature-symmetry coverage are still planned.

For complete metrics, dataset statistics, implemented improvements, technical notes and planned work, see [the progress report](reports/progress_report.md).
