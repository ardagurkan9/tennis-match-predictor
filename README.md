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

Model training and evaluation remain separate steps:

```text
train.py → trained model files
evaluate.py → rank-baseline metrics
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
│   └── predict.py
├── main.py
├── requirements.txt
└── README.md
```

## Data Split

Random splitting is not used because tennis data is time-dependent.

| Split | Years |
|---|---|
| Train | 2015–2023 |
| Validation | 2024 |
| Test | 2025 |

The advanced pipeline uses matches from 2000 onward to warm up player histories before retaining rows from 2015 onward for modeling.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Raw ATP CSV files belong in `data/raw/`. Historical odds workbooks belong in `data/raw/odds/`.

## Usage

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

## Outputs

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
```

## Current Status

The base and advanced modeling pipelines are complete. The current best advanced LightGBM result is approximately 67.86% test accuracy and 73.96% test ROC-AUC.

The prediction interface, persistent evaluation reports and automated tests are not implemented yet.

For complete metrics, dataset statistics, implemented improvements, technical notes and planned work, see [the progress report](reports/progress_report.md).
