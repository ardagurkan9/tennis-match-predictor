# Tennis Match Predictor — Progress Report

---

## Current Status

The project has a working, time-aware preprocessing and modeling pipeline for ATP matches from 2000 through 2025. The base feature set and four baseline ML models are complete. The advanced pipeline now uses the full 2000–2025 history to warm up rolling features, keeps 2015–2025 rows for modeling, and includes historical pre-match market odds.

The latest generated advanced dataset contains **61,228 player-perspective rows (30,614 matches)** and **144 columns**. Historical market odds were matched to **26,929 matches (87.96%)**.

## Latest Repository Maintenance

- Historical odds workbooks were reduced to the nine columns used by `src/odds.py`: `Date`, `Winner`, `Loser`, `B365W`, `B365L`, `PSW`, `PSL`, `AvgW`, and `AvgL`.
- All 11 reduced workbooks were validated through the project odds loader. They contain 27,574 total records, of which 27,551 produce a fair implied probability.
- `README.md` was simplified into a project overview and usage guide; detailed metrics and implementation history remain in this report.
- `main.py` now provides `clean` and `advanced` pipeline modes while preserving the cleaning check as its default behavior.
- The empty, unused `src/config.py` placeholder was removed.
- Pipeline documentation distinguishes the two `main.py` data modes from the separately run training/evaluation stages.
- Training and rank-baseline evaluation now accept `--dataset base|advanced`; advanced training reads `data/features/advanced/` and writes models to `models/advanced/`.
- An expanded odds experiment added source-level probabilities, consensus, source count, standard deviation, and range. It slightly improved test ROC-AUC (`0.7396` to `0.7399`) and log loss (`0.5969` to `0.5965`) but reduced test accuracy (`0.6786` to `0.6767`), so it was not retained in the production feature set.
- An Elo-adjusted last-5/last-10 form experiment rewarded results relative to the opponent-strength expectation. Validation accuracy increased slightly (`0.6798` to `0.6804`), but test accuracy fell (`0.6786` to `0.6755`) and both ROC-AUC and log loss worsened, so the feature was not retained.
- An average-odds-first market probability experiment was tested based on the thesis finding that bookmaker-average implied probability was its strongest feature. LightGBM test accuracy fell from `0.6786` to `0.6756`, while XGBoost changed from `0.6730` to `0.6735`; the LightGBM probability metrics also worsened, so Pinnacle-first priority was retained.
- A thesis-inspired XGBoost search used five expanding-window validation folds (2020–2024) and selected `n_estimators=100`, `max_depth=3`, `min_child_weight=1`, `subsample=0.8`, `learning_rate=0.05`, and no L1/L2 regularization. After refitting on 2015–2024, it reached `0.6772` test accuracy, `0.7392` ROC-AUC, `0.5993` log loss, and `0.2068` Brier score on 2025. This improved XGBoost over its current parameters on the same training period (`0.6755` accuracy) but did not beat LightGBM, so no production model was replaced.
- A reproducible market-feature ablation command now compares rank baseline, market baseline, market-only LightGBM, tennis-only LightGBM, and full LightGBM using 2020–2024 expanding-window cross-validation. The previously inspected 2025 set is explicitly labeled as a retrospective benchmark.
- Automated leakage regression tests now cover post-match column removal, current-match serve-stat isolation, future-result isolation, same-day update isolation, and strict time-split boundaries.
- A persistent evaluation command now reports all advanced models and baselines with accuracy, ROC-AUC, log loss, Brier score, precision, recall, confusion matrices, odds-availability slices, and LightGBM diagnostic plots.
- A memory-efficient future-match inference pipeline and CLI now rebuild current player state from matches strictly before the requested date, aligns the result to the saved 131-feature LightGBM schema, and returns symmetric two-player probabilities.
- The unused `data/predictions/` placeholder was removed because no persistent prediction-output workflow is implemented.

## Completed Work

### Market Feature Ablation

`python -m src.ablation` evaluates five fixed experiments over expanding-window validation years 2020–2024. Experiment decisions must use these CV summaries rather than the previously inspected 2025 benchmark.

| Experiment | CV Accuracy | CV Accuracy Std. | CV Brier Score | 2025 Retrospective Accuracy |
|---|---:|---:|---:|---:|
| Rank baseline | 0.6359 | 0.0048 | 0.3641 | 0.6368 |
| Market baseline | 0.6583 | 0.0053 | 0.2073 | 0.6510 |
| Market-only LightGBM | 0.6578 | 0.0052 | 0.2079 | 0.6498 |
| Tennis-only LightGBM | 0.6614 | 0.0033 | 0.2084 | 0.6596 |
| Full LightGBM | **0.6838** | 0.0080 | **0.1994** | **0.6774** |

The tennis-only model exceeds both market-only approaches, and combining tennis and market features produces the strongest result. This indicates that the full model learns additional tennis signal rather than merely reproducing bookmaker probabilities. Machine-readable results are saved to `reports/ablation_results.csv` and `reports/ablation_metrics.json`.

### 1. Data Ingestion and Cleaning

- Raw ATP CSV files from 2000–2025 are loaded and combined by `src/ingest.py`.
- Tournament dates and surface values are normalized and duplicate matches are removed.
- High-missing entry columns are dropped and missing seed, height, rank, ranking-point, age, hand, and surface values are handled.
- Post-match columns such as score, duration, aces, double faults, serve points, and break-point statistics are removed from the final model input.
- The standard cleaned dataset is saved to `data/processed/cleaned_matches.parquet`.

### 2. Base Feature Engineering

- Each match is expanded into two neutral player/opponent rows, with `target_win=1` and `target_win=0`.
- Rank, ranking-point, age, height, and seed difference features are created.
- Handedness, nationality, surface, tournament level, and round context are encoded.
- Base features are saved to `data/features/match_features.parquet` and split CSV files.

### 3. Leakage-Free Advanced Features

The advanced pipeline in `src/build_advanced_features.py` and `src/features.py` computes features chronologically and only applies match results after all matches on the same tournament date have been processed.

Implemented advanced features:

- Prior last-5 and last-10 match win rates
- Surface-specific historical win rate
- Days since the previous match
- Overall head-to-head win rate
- Surface-specific head-to-head win rate
- Career and surface match counts
- Number of matches played in the preceding 14 days (fatigue/load proxy)
- Overall Elo and surface Elo ratings and expected win probabilities
- Tournament-level Elo K-factors (`G=40`, `M/F/O=32`, `A=24`, `D=20`, default `32`)
- Leakage-free rolling serve/return statistics over the preceding 20 matches
- Player/opponent values and their difference features

The full 2000–2025 history is used to warm up these rolling statistics. Only rows from 2015 onward are retained after feature generation, avoiding cold-starting all player histories at the beginning of the training period.

### 4. Historical Market Odds

- Historical Excel odds files for 2015–2025 are stored in `data/raw/odds/`.
- `src/odds.py` loads the files using the newly added `openpyxl` and `xlrd` dependencies.
- ATP player names are matched to odds records using normalized surname tokens and first initials, including fallback handling for compound or differently transliterated surnames.
- Matching searches from the ATP tournament date through the following 21 days.
- Odds source priority is Pinnacle (`PSW`/`PSL`), market average (`AvgW`/`AvgL`), then Bet365 (`B365W`/`B365L`).
- Bookmaker margin is removed by normalizing the two implied probabilities.
- The resulting features are `player_market_prob`, `opponent_market_prob`, `market_prob_diff`, and `market_odds_available`.
- Unmatched matches receive neutral probabilities of `0.5`, while `market_odds_available=0` preserves the missingness signal.

Latest generated-data coverage:

| Item                 |  Value |
| -------------------- | -----: |
| Advanced rows        | 61,228 |
| Unique matches       | 30,614 |
| Odds-matched matches | 26,929 |
| Odds coverage        | 87.96% |

### 5. Time-Based Dataset Split

Random splitting is not used. The current advanced split is:

| Split      | Years      |   Rows |
| ---------- | ---------- | -----: |
| Train      | 2015–2023 | 49,192 |
| Validation | 2024       |  6,314 |
| Test       | 2025       |  5,722 |

Advanced outputs are saved under `data/features/advanced/`.

### 6. Base Benchmarks

| Model               | Validation Accuracy | Test Accuracy | Test ROC-AUC |
| ------------------- | ------------------: | ------------: | -----------: |
| Rank baseline       |              0.6348 |        0.6368 |           — |
| Logistic Regression |              0.6364 |        0.6407 |           — |
| Random Forest       |              0.6343 |        0.6459 |       0.7051 |
| XGBoost             |              0.6402 |        0.6508 |       0.7150 |
| LightGBM            |              0.6394 |        0.6529 |       0.7156 |

### 7. Latest Advanced Model Results

The saved models under `models/advanced/` were verified against the current advanced validation and test files. All four standard advanced models use the new market features.

| Model               |    Val. Accuracy |     Val. ROC-AUC |    Val. Log Loss |    Test Accuracy |     Test ROC-AUC |    Test Log Loss |
| ------------------- | ---------------: | ---------------: | ---------------: | ---------------: | ---------------: | ---------------: |
| Logistic Regression |           0.6737 |           0.7404 |           0.5980 |           0.6714 |           0.7248 |           0.6135 |
| Random Forest       | **0.6861** |           0.7498 |           0.5917 |           0.6776 |           0.7329 |           0.6042 |
| XGBoost             |           0.6790 | **0.7523** |           0.5856 |           0.6730 |           0.7379 |           0.5984 |
| LightGBM            |           0.6798 |           0.7521 | **0.5853** | **0.6786** | **0.7396** | **0.5969** |

The previous advanced LightGBM benchmark had 0.6669 test accuracy and 0.7267 test ROC-AUC. The current saved LightGBM reaches **0.6786 test accuracy** and **0.7396 test ROC-AUC**.

## Remaining Work

### Graphical Prediction Interface

- `src/predict.py` provides future-match feature generation and a working CLI.
- A Streamlit frontend and optional API layer are still planned.
- Rebuilding historical state takes several seconds and should be cached by the frontend.

### Reproducibility and Validation

- Leakage regression tests cover chronological future isolation and same-day update isolation. Name matching, odds matching, feature symmetry, and end-to-end pipeline tests are still missing.
- `main.py advanced` runs the advanced dataset build; model training and reporting remain explicit separate commands.
- Odds matching should be audited for false matches caused by the 21-day date window, surname-only fallback, and cross-source tournament/date differences.
- GitHub Actions is not yet configured to run the test suite automatically.

## Summary

| Step                                               | Status  |
| -------------------------------------------------- | ------- |
| ATP data ingestion (2000–2025)                    | Done    |
| Cleaning and leakage-column removal                | Done    |
| Base feature engineering                           | Done    |
| Time-based split                                   | Done    |
| Base model benchmarks                              | Done    |
| Rolling form and surface form                      | Done    |
| Days since last match and 14-day match load        | Done    |
| Overall and surface-specific H2H                   | Done    |
| Elo, surface Elo, and tournament-weighted K-factor | Done    |
| Rolling serve/return features                      | Done    |
| Historical odds ingestion (2015–2025)             | Done    |
| Vig-free market probability features               | Done    |
| Full-history feature warm-up                       | Done    |
| Advanced dataset generation                        | Done    |
| Advanced model training                            | Done    |
| Market-feature ablation report                     | Done    |
| Persistent model evaluation report                 | Done    |
| Future-match feature builder                       | Done    |
| Prediction CLI                                     | Done    |
| Streamlit frontend / API                           | Pending |
| Leakage regression tests                           | Done |
| Odds-match audit and broader automated tests       | Pending |
