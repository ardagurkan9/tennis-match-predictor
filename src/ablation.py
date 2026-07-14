"""Reproducible market-feature ablation experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from src.train import build_lightgbm_model, split_features_and_target


DATA_DIR = Path("data/features/advanced")
REPORT_DIR = Path("reports")
CSV_OUTPUT_PATH = REPORT_DIR / "ablation_results.csv"
JSON_OUTPUT_PATH = REPORT_DIR / "ablation_metrics.json"

CV_YEARS = tuple(range(2020, 2025))
HOLDOUT_YEAR = 2025

MARKET_FEATURE_COLUMNS = {
    "player_market_prob",
    "opponent_market_prob",
    "market_prob_diff",
    "market_odds_available",
}

MODEL_EXPERIMENTS = ("market_only_lightgbm", "tennis_only_lightgbm", "full_lightgbm")
ALL_EXPERIMENTS = ("rank_baseline", "market_baseline", *MODEL_EXPERIMENTS)
METRIC_NAMES = ("accuracy", "roc_auc", "log_loss", "brier_score")


def load_ablation_data(
    data_dir: str | Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load development data through 2024 and the retrospective 2025 holdout."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    validation = pd.read_csv(data_dir / "validation.csv")
    holdout = pd.read_csv(data_dir / "test.csv")
    development = pd.concat([train, validation], ignore_index=True)
    return development, holdout


def select_experiment_features(features: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """Select market, tennis, or full feature groups for an experiment."""
    market_columns = [
        column for column in features.columns if column in MARKET_FEATURE_COLUMNS
    ]

    if experiment == "market_only_lightgbm":
        return features[market_columns]
    if experiment == "tennis_only_lightgbm":
        return features.drop(columns=market_columns)
    if experiment == "full_lightgbm":
        return features
    raise ValueError(f"Unknown model experiment: {experiment}")


def calculate_probability_metrics(
    target: pd.Series,
    probability: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Calculate classification and probability-quality metrics."""
    probability = np.asarray(probability, dtype="float64")
    prediction = (probability >= 0.5).astype("int8")
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probability)),
    }


def rank_baseline_probability(data: pd.DataFrame) -> np.ndarray:
    """Return hard probabilities favoring the player with the better ATP rank."""
    return (data["player_rank"] < data["opponent_rank"]).astype("float64").to_numpy()


def market_baseline_probability(data: pd.DataFrame) -> np.ndarray:
    """Return the vig-free pre-match market probability already in the dataset."""
    return data["player_market_prob"].astype("float64").to_numpy()


def evaluate_baselines(data: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Evaluate non-ML rank and market baselines on one dataset."""
    target = data["target_win"]
    return {
        "rank_baseline": calculate_probability_metrics(
            target, rank_baseline_probability(data)
        ),
        "market_baseline": calculate_probability_metrics(
            target, market_baseline_probability(data)
        ),
    }


def evaluate_model_experiment(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    experiment: str,
) -> dict[str, float]:
    """Train one LightGBM feature ablation and evaluate it."""
    x_train, y_train = split_features_and_target(train)
    x_evaluation, y_evaluation = split_features_and_target(evaluation)
    x_train = select_experiment_features(x_train, experiment)
    x_evaluation = select_experiment_features(x_evaluation, experiment)

    model = build_lightgbm_model()
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_evaluation)[:, 1]
    return calculate_probability_metrics(y_evaluation, probability)


def run_expanding_window_cv(
    development: pd.DataFrame,
    validation_years: tuple[int, ...] = CV_YEARS,
) -> dict[str, list[dict[str, float | int]]]:
    """Evaluate every experiment on expanding time-based validation folds."""
    fold_results: dict[str, list[dict[str, float | int]]] = {
        experiment: [] for experiment in ALL_EXPERIMENTS
    }

    for validation_year in validation_years:
        fold_train = development[development["tourney_year"] < validation_year]
        fold_validation = development[
            development["tourney_year"] == validation_year
        ]
        if fold_train.empty or fold_validation.empty:
            raise ValueError(f"Missing data for validation year {validation_year}")

        baseline_metrics = evaluate_baselines(fold_validation)
        for experiment, metrics in baseline_metrics.items():
            fold_results[experiment].append(
                {"validation_year": validation_year, **metrics}
            )

        for experiment in MODEL_EXPERIMENTS:
            metrics = evaluate_model_experiment(
                fold_train,
                fold_validation,
                experiment,
            )
            fold_results[experiment].append(
                {"validation_year": validation_year, **metrics}
            )

    return fold_results


def summarize_cv_results(
    fold_results: dict[str, list[dict[str, float | int]]],
) -> dict[str, dict[str, float]]:
    """Summarize expanding-window folds with metric means and standard deviations."""
    summaries = {}
    for experiment, folds in fold_results.items():
        summary: dict[str, float] = {"fold_count": float(len(folds))}
        for metric in METRIC_NAMES:
            values = np.array([float(fold[metric]) for fold in folds])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=0))
        summaries[experiment] = summary
    return summaries


def evaluate_retrospective_holdout(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Report 2025 results separately; they must not drive model selection."""
    results = evaluate_baselines(holdout)
    for experiment in MODEL_EXPERIMENTS:
        results[experiment] = evaluate_model_experiment(
            development,
            holdout,
            experiment,
        )
    return results


def build_results_table(
    cv_summary: dict[str, dict[str, float]],
    holdout_results: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Build a flat table suitable for terminal and CSV output."""
    rows = []
    for experiment in ALL_EXPERIMENTS:
        rows.append(
            {
                "experiment": experiment,
                **cv_summary[experiment],
                **{
                    f"holdout_{metric}": value
                    for metric, value in holdout_results[experiment].items()
                },
            }
        )
    return pd.DataFrame(rows)


def save_ablation_reports(
    results_table: pd.DataFrame,
    fold_results: dict[str, list[dict[str, float | int]]],
    cv_summary: dict[str, dict[str, float]],
    holdout_results: dict[str, dict[str, float]],
    report_dir: str | Path = REPORT_DIR,
) -> tuple[Path, Path]:
    """Persist flat and structured ablation reports."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / CSV_OUTPUT_PATH.name
    json_path = report_dir / JSON_OUTPUT_PATH.name

    results_table.to_csv(csv_path, index=False)
    payload = {
        "protocol": {
            "selection_data": "expanding-window cross-validation only",
            "cv_validation_years": list(CV_YEARS),
            "retrospective_holdout_year": HOLDOUT_YEAR,
            "holdout_warning": (
                "The 2025 holdout was inspected during earlier development and is "
                "reported only as a retrospective benchmark."
            ),
        },
        "market_features": sorted(MARKET_FEATURE_COLUMNS),
        "cv_folds": fold_results,
        "cv_summary": cv_summary,
        "retrospective_holdout": holdout_results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


def main() -> None:
    """Run and persist the complete ablation comparison."""
    development, holdout = load_ablation_data()
    fold_results = run_expanding_window_cv(development)
    cv_summary = summarize_cv_results(fold_results)
    holdout_results = evaluate_retrospective_holdout(development, holdout)
    results_table = build_results_table(cv_summary, holdout_results)
    csv_path, json_path = save_ablation_reports(
        results_table,
        fold_results,
        cv_summary,
        holdout_results,
    )

    display_columns = [
        "experiment",
        "accuracy_mean",
        "accuracy_std",
        "brier_score_mean",
        "holdout_accuracy",
        "holdout_brier_score",
    ]
    print("Market feature ablation")
    print("-----------------------")
    print(results_table[display_columns].to_string(index=False, float_format="%.4f"))
    print()
    print(f"CSV report: {csv_path}")
    print(f"JSON report: {json_path}")
    print("2025 is a retrospective benchmark and must not drive model selection.")


if __name__ == "__main__":
    main()
