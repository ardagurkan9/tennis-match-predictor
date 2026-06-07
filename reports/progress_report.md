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

---

## Not Yet Done

### Evaluation Report
- `reports/metrics.json` not generated — logistic regression accuracy, ROC-AUC, and log loss results not yet saved
- `reports/confusion_matrix.png` not generated

### Prediction Script (`src/predict.py`)
- File exists but content not implemented
- Goal: given two player names and a surface, return win probabilities

### Advanced Features (Planned)
- Rolling last 5/10 match win rate
- Surface-specific win rate
- Head-to-head history
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
| Logistic Regression | Done (model saved) |
| Evaluation Report | Missing |
| Prediction Script | Missing |
| Advanced Features | Planned |
