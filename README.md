# Tennis Match Predictor

[![CI](https://github.com/ardagurkan9/tennis-match-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/ardagurkan9/tennis-match-predictor/actions/workflows/ci.yml)

An end-to-end machine learning pipeline for predicting professional ATP tennis match outcomes using historical match data, player form, Elo ratings, match statistics, and pre-match betting odds.

The project is designed around chronological data processing and leakage-free feature engineering. Every prediction uses only information that would have been available before the match started.

## Results

The production model uses LightGBM with predictions averaged across both player perspectives.

| Dataset                        | Accuracy | ROC-AUC | Log Loss | Brier Score |
| ------------------------------ | -------: | ------: | -------: | ----------: |
| Validation — 2024              |   67.69% |  75.24% |   0.5850 |      0.2012 |
| Retrospective benchmark — 2025 |   67.56% |  73.89% |   0.5973 |      0.2066 |

The 2025 dataset was inspected during development and is therefore reported as a retrospective benchmark rather than an untouched final test set.

## Project Highlights

* Historical ATP match processing from 2000 through 2025
* Chronological train, validation, and benchmark splits
* Leakage-free rolling feature engineering
* Overall and surface-specific Elo ratings
* Recent-form and surface-form statistics
* Head-to-head and surface head-to-head features
* Rolling serve and return performance
* Player workload and rest-day features
* Vig-free market probabilities from pre-match betting odds
* Logistic Regression, Random Forest, XGBoost, and LightGBM models
* Expanding-window ablation analysis
* Automated leakage, prediction, and frontend tests
* Streamlit prediction interface
* Docker and GitHub Actions support

## Demo

The Streamlit application allows users to select two players and configure a hypothetical matchup.

The interface displays:

* Predicted win probabilities
* Player profile information
* Recent form
* Head-to-head history
* Side-by-side statistical comparisons
* Dataset freshness and confidence warnings

```bash
streamlit run app.py
```

> Add a screenshot or short GIF of the Streamlit interface here.

## How It Works

The project processes raw ATP match files chronologically, builds pre-match player histories, creates model-ready datasets, trains multiple models, and evaluates predictions on future seasons.

```text
Raw ATP matches and betting odds
                |
                v
      Ingestion and cleaning
                |
                v
 Chronological feature engineering
                |
                v
    Time-based dataset splitting
                |
                v
     Model training and evaluation
                |
                v
       CLI and Streamlit prediction
```

Each historical match is represented from both player perspectives:

```text
target_win = 1 if the selected player wins
target_win = 0 if the selected player loses
```

The final matchup probability combines both direct predictions:

```text
P(A beats B) =
    (direct P(A beats B) + 1 - direct P(B beats A)) / 2
```

This reduces sensitivity to player ordering and exposes prediction asymmetry.

## Data Split

Random train-test splitting is not used because tennis results are time-dependent.

| Purpose                    | Years     | Used as model rows? |
| -------------------------- | --------- | ------------------- |
| Historical feature warm-up | 2000–2014 | No                  |
| Training                   | 2015–2023 | Yes                 |
| Validation                 | 2024      | Yes                 |
| Retrospective benchmark    | 2025      | Yes                 |

Matches from 2000–2014 initialize rolling statistics, Elo ratings, head-to-head records, and player histories. Their rows are not used for model training.

## Leakage Prevention

All chronological features are calculated using only matches completed before the predicted match.

The model does not use:

* Current-match scores
* Current-match serve statistics
* Post-match rankings
* Future player form
* Future head-to-head results
* Random data splitting

Automated regression tests verify important temporal and prediction invariants.

## Feature Groups

### Player strength

* ATP ranking and ranking difference
* Elo rating difference
* Surface-specific Elo difference

### Recent form

* Overall recent win rate
* Surface-specific recent win rate
* Rolling match counts

### Serve and return performance

* First-serve percentage
* First-serve points won
* Second-serve points won
* Break points saved
* Return performance

### Matchup history

* Overall head-to-head record
* Surface-specific head-to-head record
* Previous meetings

### Workload and recovery

* Days since previous match
* Recent matches played
* Recent sets and games played

### Market information

* Vig-free implied win probabilities
* Auditable odds matching method
* Odds matching confidence

## Models

The training pipeline compares:

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM

LightGBM is currently used as the production model.

The ablation pipeline compares:

* Ranking baseline
* Market baseline
* Market-only LightGBM
* Tennis-feature-only LightGBM
* Full LightGBM model

Experiment selection uses expanding-window validation over historical seasons rather than the 2025 benchmark.

## Architecture

| Module                    | Responsibility                                          |
| ------------------------- | ------------------------------------------------------- |
| `ingest`                  | Load and combine historical ATP data                    |
| `clean`                   | Validate and normalize match records                    |
| `odds`                    | Match betting odds and calculate vig-free probabilities |
| `features`                | Build rolling player and matchup features               |
| `build_advanced_features` | Coordinate the advanced feature pipeline                |
| `split`                   | Create chronological datasets                           |
| `train`                   | Train and save model candidates                         |
| `evaluate`                | Evaluate ranking baselines                              |
| `ablation`                | Compare feature groups and model configurations         |
| `report`                  | Generate metrics and diagnostic plots                   |
| `predict`                 | Produce matchup predictions                             |
| `match_history`           | Retrieve player history and matchup context             |
| `comparison`              | Build player comparison data for the frontend           |

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
│   └── advanced/
├── reports/
├── scripts/
│   └── download_model.py
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
├── Dockerfile
├── Makefile
├── pyproject.toml
├── requirements.txt
├── requirements.lock
└── README.md
```

## Getting Started

### Requirements

* Python 3.11 or 3.12
* Historical ATP match CSV files
* Historical betting-odds workbooks for market features

### Installation

Clone the repository:

```bash
git clone https://github.com/ardagurkan9/tennis-match-predictor.git
cd tennis-match-predictor
```

Create a virtual environment.

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

On systems with GNU Make:

```bash
make setup
```

Place ATP match files in:

```text
data/raw/
```

Place historical odds files in:

```text
data/raw/odds/
```

## Running the Pipeline

Run the basic ingestion and cleaning pipeline:

```bash
python main.py clean
```

Build the advanced feature datasets:

```bash
python main.py advanced
```

Train the advanced models:

```bash
python -m src.train --dataset advanced
```

Run the ablation analysis:

```bash
python -m src.ablation
```

Generate reports and diagnostic plots:

```bash
python -m src.report
```

## Prediction

Predict a hypothetical matchup:

```bash
python -m src.predict \
  --player "Carlos Alcaraz" \
  --opponent "Jannik Sinner" \
  --surface Hard \
  --date 2026-01-15 \
  --tourney-level G \
  --best-of 5 \
  --round SF
```

Optional market odds can be supplied:

```bash
python -m src.predict \
  --player "Carlos Alcaraz" \
  --opponent "Jannik Sinner" \
  --surface Hard \
  --date 2026-01-15 \
  --player-odds 1.70 \
  --opponent-odds 2.20
```

When odds are omitted, the model uses neutral market inputs and marks market data as unavailable.

## Production Model

Model binaries are intentionally excluded from Git.

Download the published LightGBM release asset:

```bash
python scripts/download_model.py
```

Alternatively, rebuild the dataset and train the model locally:

```bash
python main.py advanced
python -m src.train --dataset advanced
```

The CLI and Streamlit application provide recovery instructions when the model artifact is missing.

## Testing and Quality

Run the complete test suite:

```bash
python -m pytest
```

Run static analysis:

```bash
python -m ruff check .
```

The test suite covers:

* Temporal leakage checks
* Prediction symmetry behavior
* Odds matching behavior
* Feature consistency
* Missing-model handling
* Frontend helper logic

## Docker

Build the image after downloading or training the model:

```bash
docker build -t tennis-match-predictor .
```

Run the Streamlit application:

```bash
docker run --rm -p 8501:8501 tennis-match-predictor
```

Open the application at:

```text
http://localhost:8501
```

## Generated Outputs

The pipeline creates:

```text
data/processed/cleaned_matches.parquet
data/features/advanced/train.csv
data/features/advanced/validation.csv
data/features/advanced/test.csv
models/advanced/*.pkl
reports/ablation_results.csv
reports/model_metrics.csv
reports/model_metrics.json
reports/confusion_matrix.png
reports/calibration_curve.png
```

Generated datasets and model binaries are not intended to be committed directly to the repository.

## Limitations

* Predictions use historical data rather than live rankings or injury information.
* Player form may become stale when the latest available match data is old.
* Historical betting odds require approximate matching across external datasets.
* The 2025 results are retrospective and do not represent a fully untouched final test.
* Model probabilities are estimates, not guarantees.
* A future season is required for a new untouched evaluation.

## Roadmap

* [x] Chronological ingestion and cleaning pipeline
* [x] Leakage-free advanced feature engineering
* [x] Multiple model training and comparison
* [x] Expanding-window ablation analysis
* [x] Automated leakage and prediction tests
* [x] CLI and Streamlit prediction interfaces
* [ ] Complete a statistically meaningful manual odds-matching audit
* [ ] Evaluate on a future untouched ATP season
* [ ] Add automated live data refresh
* [ ] Publish a hosted demonstration

## Further Documentation

Detailed experiment notes, dataset statistics, implementation progress, and technical decisions are available in:

```text
reports/progress_report.md
```

## License

Add the repository’s actual license here. Do not claim MIT unless a matching `LICENSE` file exists.
