# Tennis Match Predictor — Progress Report

**Date:** 2026-06-07

---

## Completed

### 1. Data Ingestion (`src/ingest.py`)
Raw ATP match CSV files are read from `data/raw/` and combined into a single DataFrame.

### 2. Data Cleaning (`src/clean.py`)
- Tournament dates converted to datetime; `tourney_year`, `tourney_month`, `tourney_day` columns added
- Surface values normalized
- Duplicate matches removed
- Post-match leakage columns removed (`score`, `minutes`, `w_ace`, `l_ace`, etc. — 19 columns)
- Cleaned data saved to: `data/processed/cleaned_matches.parquet`

### 3. Feature Engineering (`src/features.py`)
- Each match expanded into two rows: winner perspective (`target_win=1`) and loser perspective (`target_win=0`)
- Difference features created: `rank_diff`, `rank_points_diff`, `age_diff`, `height_diff`, `seed_diff`
- Matchup features added: `hand_matchup`, `ioc_matchup`, `same_ioc`
- Categorical columns one-hot encoded: `surface`, `tourney_level`, `round`, `player_hand`, `opponent_hand`
- Feature dataset saved to: `data/features/match_features.parquet`
- Advanced rolling form feature dataset saved separately to: `data/features/advanced/match_features.parquet`
- Advanced split files saved to: `data/features/advanced/train.csv`, `validation.csv`, `test.csv`

### 3.1 Advanced Rolling Form Features (`src/features.py`)
- Added prior last 5 match win rate for player and opponent
- Added prior last 10 match win rate for player and opponent
- Added `last5_win_rate_diff` and `last10_win_rate_diff`
- Same `tourney_date` matches are not used as prior history to prevent leakage

### 4. Train / Validation / Test Split (`src/split.py`)
Year-based split applied to prevent temporal leakage:
- Train: 2015–2023
- Validation: 2024
- Test: 2025

### 5. Rank-Based Baseline (`src/evaluate.py`)
- Rule: the player with the better ATP ranking is predicted to win
- Validation accuracy: **0.6348**
- Test accuracy: **0.6368**

### 6. Logistic Regression (`src/train.py`)
- `StandardScaler → LogisticRegression(max_iter=1000)` pipeline built
- Model trained and saved to: `models/logistic_regression.pkl`
- Validation accuracy: **0.6364**
- Test accuracy: **0.6407**

### 7. Random Forest (`src/train.py`)
- `RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt")` pipeline built
- Model trained and saved to: `models/random_forest.pkl`
- Validation accuracy: **0.6343**
- Validation ROC-AUC: **0.7031**
- Test accuracy: **0.6459**
- Test ROC-AUC: **0.7051**

### 8. XGBoost (`src/train.py`)
- `XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05)` pipeline built
- Model trained and saved to: `models/xgboost.pkl`
- Validation accuracy: **0.6402**
- Validation ROC-AUC: **0.7086**
- Test accuracy: **0.6508**
- Test ROC-AUC: **0.7150**

### 9. LightGBM (`src/train.py`)
- `LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05)` pipeline built
- Model trained and saved to: `models/lightgbm.pkl`
- Validation accuracy: **0.6394**
- Validation ROC-AUC: **0.7076**
- Test accuracy: **0.6529**
- Test ROC-AUC: **0.7156**

### 10. Advanced LightGBM With Rolling Form Features
- Model trained on `data/features/advanced/train.csv`
- Model saved separately to: `models/advanced/lightgbm.pkl`
- Validation accuracy: **0.6460**
- Validation ROC-AUC: **0.7119**
- Test accuracy: **0.6613**
- Test ROC-AUC: **0.7207**

---

## Not Yet Done

### Evaluation Report
- `reports/metrics.json` not generated — logistic regression accuracy, ROC-AUC, and log loss results not yet saved
- `reports/confusion_matrix.png` not generated

### Prediction Script (`src/predict.py`)
- File exists but content not implemented
- Goal: given two player names and a surface, return win probabilities

### Advanced Features (Planned)
- Surface-specific win rate
- Head-to-head history
- Days since last match
- Elo rating and surface-specific Elo

---

## Summary

| Step | Status |
|---|---|
| Data Ingestion | Done |
| Data Cleaning | Done |
| Leakage Prevention | Done |
| Feature Engineering | Done |
| Train/Val/Test Split | Done |
| Rank-Based Baseline | Done (~63.5%) |
| Logistic Regression | Done (test accuracy: 0.6407) |
| Random Forest | Done (test accuracy: 0.6459) |
| XGBoost | Done (test accuracy: 0.6508) |
| LightGBM | Done (test accuracy: 0.6529) |
| Rolling Form Features | Done (saved separately) |
| Advanced LightGBM | Done (test accuracy: 0.6613) |
| Evaluation Report | Missing |
| Prediction Script | Missing |
| Advanced Features | Planned |
