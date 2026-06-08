# Tennis Match Predictor

This project is an end-to-end machine learning pipeline for predicting professional tennis match outcomes using historical ATP match data.

The goal is not only to train a model, but also to build a clean, reproducible, leakage-free data pipeline. Raw match data is ingested, cleaned, transformed into model-ready features, split by time, trained with multiple models, and evaluated against a simple baseline.

## Project Objective

The project predicts whether `player_1` wins a tennis match against `player_2`.

This is a binary classification problem:

```text
target = 1 if player_1 wins
target = 0 if player_2 wins
```

## Project Structure

```text
tennis-match-predictor/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── predictions/
├── models/
├── reports/
├── notebooks/
├── src/
│   ├── config.py
│   ├── ingest.py
│   ├── clean.py
│   ├── features.py
│   ├── split.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── main.py
├── requirements.txt
└── README.md
```

## Pipeline Overview

The pipeline follows this order:

```text
ingest.py → clean.py → features.py → split.py → train.py → evaluate.py
```

The full pipeline can be run with:

```bash
python main.py
```

## 1. Data Ingestion

Raw ATP match CSV files are stored in:

```text
data/raw/
```

Example raw files:

```text
atp_matches_2015.csv
atp_matches_2016.csv
atp_matches_2017.csv
...
```

The ingestion step reads all raw CSV files and combines them into a single dataset.

Raw data is never overwritten. This keeps the original dataset unchanged and reproducible.

Responsible file:

```text
src/ingest.py
```

## 2. Data Cleaning

The cleaning step prepares the raw match data for feature engineering.

Main cleaning tasks:

* Convert tournament dates to datetime format
* Normalize surface values
* Remove duplicate matches
* Remove post-match leakage columns
* Check missing values
* Keep useful pre-match player information for feature engineering

Cleaned data is saved to:

```text
data/processed/cleaned_matches.parquet
```

Responsible file:

```text
src/clean.py
```

## Leakage Columns Removed

The following columns are removed because they are only known after the match is played:

```text
score
minutes
w_ace
w_df
w_svpt
w_1stIn
w_1stWon
w_2ndWon
w_SvGms
w_bpSaved
w_bpFaced
l_ace
l_df
l_svpt
l_1stIn
l_1stWon
l_2ndWon
l_SvGms
l_bpSaved
l_bpFaced
```

These columns would cause data leakage and create unrealistic model performance.

## 3. Feature Engineering

The original ATP dataset stores players as `winner` and `loser`. This format cannot be used directly for modeling because it already reveals the result.

Therefore, the data is converted into a neutral format:

```text
player_1
player_2
target
```

The target column represents whether `player_1` won the match.

Example features:

```text
player_1_rank
player_2_rank
rank_diff
player_1_rank_points
player_2_rank_points
rank_points_diff
player_1_age
player_2_age
age_diff
player_1_height
player_2_height
height_diff
player_1_hand
player_2_hand
hand_matchup
surface
tourney_level
best_of
round
```

Feature data is saved to:

```text
data/features/match_features.parquet
```

Responsible file:

```text
src/features.py
```

## Data Leakage Prevention

Preventing data leakage is the most important part of this project.

No feature should use information that would not be available before the match starts.

All rolling player features were calculated using only matches prior to the prediction date to prevent data leakage.

For example, if a match was played on `2023-06-10`, the feature engineering step must not use any match data after `2023-06-10`.

## 4. Train / Validation / Test Split

Random split is not used because tennis match data is time-dependent.

The dataset is split by date to simulate real-world prediction:

```text
train: past years
validation: following year
test: final year
```

Example:

```text
train: 2015-2023
validation: 2024
test: 2025
```

Responsible file:

```text
src/split.py
```

## 5. Baseline Model

Before training machine learning models, a baseline prediction is calculated.

Baseline strategy:

```text
Predict the player with the better ranking as the winner.
```

This gives a simple benchmark. Any ML model should be compared against this baseline.

If the trained model does not beat the baseline, this should be reported honestly.

Responsible file:

```text
src/evaluate.py
```

Current rank-based baseline results:

```text
validation_baseline_accuracy: 0.6348
test_baseline_accuracy: 0.6368
```

## 6. Model Training

The project starts with simple and interpretable models before trying more complex models.

Initial models:

```text
Logistic Regression
Random Forest
```

Optional future models:

```text
XGBoost
LightGBM
```

The trained model is saved to:

```text
models/logistic_regression.pkl
models/random_forest.pkl
```

Responsible file:

```text
src/train.py
```

## 7. Evaluation

The model is evaluated using multiple metrics.

Reported metrics:

```text
accuracy
ROC-AUC
log loss
precision
recall
confusion matrix
baseline accuracy
```

Evaluation outputs are saved to:

```text
reports/metrics.json
reports/confusion_matrix.png
```

Responsible file:

```text
src/evaluate.py
```

## 8. Prediction

The prediction script loads the trained model and produces win probabilities for a new match.

Example future usage:

```bash
python src/predict.py --player1 "Carlos Alcaraz" --player2 "Jannik Sinner" --surface "Clay"
```

Example output:

```text
Carlos Alcaraz win probability: 58.4%
Jannik Sinner win probability: 41.6%
```

Responsible file:

```text
src/predict.py
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Place raw ATP match CSV files inside:

```text
data/raw/
```

Run the current preprocessing pipeline:

```bash
python main.py
```

## Outputs

Processed data:

```text
data/processed/cleaned_matches.parquet
```

Feature dataset:

```text
data/features/match_features.parquet
```

Train / validation / test split files:

```text
data/features/train.csv
data/features/validation.csv
data/features/test.csv
```

Trained model:

```text
models/logistic_regression.pkl
models/random_forest.pkl
```

Status: done

Evaluation reports:

```text
reports/metrics.json
reports/confusion_matrix.png
```

Status: planned

## Future Improvements

Planned improvements:

* Rolling last 5 match win rate
* Rolling last 10 match win rate
* Surface-specific win rate
* Days since last match
* Head-to-head win rate
* Elo rating
* Surface-specific Elo rating
* Command-line prediction interface
* FastAPI prediction endpoint

## Project Progress

- [x] Data Ingestion
- [x] Data Cleaning
- [x] Feature Engineering
- [x] Data Leakage Prevention Review
- [x] Train / Validation / Test Split
- [x] Baseline Model
- [x] Model Training
- [ ] Evaluation
- [ ] Prediction
