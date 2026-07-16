"""Generate persistent evaluation reports for trained advanced models."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
os.environ.setdefault("MPLCONFIGDIR", str(Path("reports/.matplotlib-cache")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.train import split_features_and_target


DATA_DIR = Path("data/features/advanced")
MODEL_DIR = Path("models/advanced")
REPORT_DIR = Path("reports")

MODEL_NAMES = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
)
SPLIT_PATHS = {
    "validation_2024": DATA_DIR / "validation.csv",
    "retrospective_2025": DATA_DIR / "test.csv",
}


def calculate_metrics(
    target: pd.Series,
    probability: np.ndarray | pd.Series,
) -> dict[str, float | list[list[int]]]:
    """Calculate classification, ranking, and probability-quality metrics."""
    probability = np.asarray(probability, dtype="float64")
    prediction = (probability >= 0.5).astype("int8")
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probability)),
        "expected_calibration_error": expected_calibration_error(target, probability),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(target, prediction).tolist(),
    }


def expected_calibration_error(
    target: pd.Series | np.ndarray,
    probability: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Return equal-width expected calibration error (ECE)."""
    target_array = np.asarray(target, dtype="float64")
    probability_array = np.asarray(probability, dtype="float64")
    bin_ids = np.minimum((probability_array * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if mask.any():
            error += float(mask.mean()) * abs(
                float(target_array[mask].mean())
                - float(probability_array[mask].mean())
            )
    return float(error)


def symmetric_match_probabilities(
    data: pd.DataFrame,
    direct_probability: np.ndarray | pd.Series,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply the production two-perspective formula to every match.

    Each match must have exactly the winner and loser perspective rows.  The
    returned array remains row-aligned so ordinary binary metrics stay useful,
    while every complementary pair represents one production prediction.
    """
    if "match_id" not in data:
        raise ValueError("Symmetric evaluation requires a match_id column.")

    probability = np.asarray(direct_probability, dtype="float64")
    if len(probability) != len(data):
        raise ValueError("Probability count must match the evaluation rows.")

    symmetric = np.empty(len(data), dtype="float64")
    gaps: list[float] = []
    positions = pd.Series(np.arange(len(data)), index=data.index)
    for match_id, group in data.groupby("match_id", sort=False):
        if len(group) != 2 or set(group["target_win"].astype(int)) != {0, 1}:
            raise ValueError(
                f"Match {match_id!r} must contain exactly one row per perspective."
            )
        winner_index = group.index[group["target_win"].eq(1)][0]
        loser_index = group.index[group["target_win"].eq(0)][0]
        winner_position = int(positions.loc[winner_index])
        loser_position = int(positions.loc[loser_index])
        winner_direct = float(probability[winner_position])
        loser_direct = float(probability[loser_position])
        winner_symmetric = (winner_direct + (1.0 - loser_direct)) / 2.0
        symmetric[winner_position] = winner_symmetric
        symmetric[loser_position] = 1.0 - winner_symmetric
        gaps.append(abs(winner_direct + loser_direct - 1.0))

    gap_array = np.asarray(gaps, dtype="float64")
    return symmetric, {
        "match_count": int(len(gap_array)),
        "mean": float(gap_array.mean()),
        "median": float(np.median(gap_array)),
        "p95": float(np.quantile(gap_array, 0.95)),
        "max": float(gap_array.max()),
    }


def rank_probability(data: pd.DataFrame) -> np.ndarray:
    """Return hard rank-baseline probabilities."""
    return (data["player_rank"] < data["opponent_rank"]).astype("float64").to_numpy()


def market_probability(data: pd.DataFrame) -> np.ndarray:
    """Return vig-free bookmaker probabilities."""
    return data["player_market_prob"].astype("float64").to_numpy()


def evaluate_scopes(
    data: pd.DataFrame,
    probability: np.ndarray,
) -> dict[str, dict[str, float | list[list[int]]] | None]:
    """Evaluate all rows and odds-availability subgroups."""
    masks = {
        "all": pd.Series(True, index=data.index),
        "odds_available": data["market_odds_available"].eq(1),
        "odds_missing": data["market_odds_available"].eq(0),
    }
    results = {}
    for scope, mask in masks.items():
        if not mask.any() or data.loc[mask, "target_win"].nunique() < 2:
            results[scope] = None
            continue
        results[scope] = calculate_metrics(
            data.loc[mask, "target_win"],
            probability[mask.to_numpy()],
        )
    return results


def load_models(model_dir: str | Path = MODEL_DIR) -> dict[str, object]:
    """Load every persisted advanced model."""
    model_dir = Path(model_dir)
    return {
        name: joblib.load(model_dir / f"{name}.pkl")
        for name in MODEL_NAMES
    }


def evaluate_models(
    split_paths: dict[str, Path] = SPLIT_PATHS,
    model_dir: str | Path = MODEL_DIR,
) -> tuple[dict, dict[str, tuple[pd.DataFrame, np.ndarray]]]:
    """Evaluate baselines and trained models on configured data splits."""
    models = load_models(model_dir)
    report: dict[str, dict] = {}
    lightgbm_plot_data: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}

    for split_name, path in split_paths.items():
        data = pd.read_csv(path)
        features, _ = split_features_and_target(data)
        direct_probabilities = {
            "rank_baseline": rank_probability(data),
            "market_baseline": market_probability(data),
            **{
                name: model.predict_proba(features)[:, 1]
                for name, model in models.items()
            },
        }
        probabilities = {}
        symmetry = {}
        for name, direct_probability in direct_probabilities.items():
            probability, gap_summary = symmetric_match_probabilities(
                data, direct_probability
            )
            probabilities[name] = probability
            symmetry[name] = gap_summary
        report[split_name] = {
            name: {
                "scopes": evaluate_scopes(data, probability),
                "symmetry_gap": symmetry[name],
                "evaluation_method": "production_two_perspective_average",
            }
            for name, probability in probabilities.items()
        }
        lightgbm_plot_data[split_name] = (data, probabilities["lightgbm"])

    return report, lightgbm_plot_data


def flatten_report(report: dict) -> pd.DataFrame:
    """Convert the structured report to one row per split/model/scope."""
    rows = []
    for split_name, models in report.items():
        for model_name, model_result in models.items():
            scopes = model_result["scopes"]
            for scope, metrics in scopes.items():
                if metrics is None:
                    continue
                row = {
                    "split": split_name,
                    "model": model_name,
                    "scope": scope,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key != "confusion_matrix"
                    },
                    "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                    **{
                        f"symmetry_gap_{key}": value
                        for key, value in model_result["symmetry_gap"].items()
                    },
                }
                rows.append(row)
    return pd.DataFrame(rows)


def save_confusion_matrix_plot(
    target: pd.Series,
    probability: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Save the LightGBM retrospective holdout confusion matrix."""
    matrix = confusion_matrix(target, probability >= 0.5)
    output_path = Path(output_path)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set(
        title="LightGBM confusion matrix — retrospective 2025",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        yticks=[0, 1],
    )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def save_calibration_plot(
    plot_data: dict[str, tuple[pd.DataFrame, np.ndarray]],
    output_path: str | Path,
) -> Path:
    """Save LightGBM calibration curves for validation and retrospective holdout."""
    output_path = Path(output_path)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
    for split_name, (data, probability) in plot_data.items():
        observed, predicted = calibration_curve(
            data["target_win"],
            probability,
            n_bins=10,
            strategy="quantile",
        )
        axis.plot(predicted, observed, marker="o", label=split_name)
    axis.set(
        title="LightGBM calibration",
        xlabel="Mean predicted probability",
        ylabel="Observed win rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def save_reports(
    report: dict,
    plot_data: dict[str, tuple[pd.DataFrame, np.ndarray]],
    report_dir: str | Path = REPORT_DIR,
) -> dict[str, Path]:
    """Persist model metrics and LightGBM diagnostic plots."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": report_dir / "model_metrics.csv",
        "json": report_dir / "model_metrics.json",
        "confusion_matrix": report_dir / "confusion_matrix.png",
        "calibration_curve": report_dir / "calibration_curve.png",
    }
    flatten_report(report).to_csv(paths["csv"], index=False)
    paths["json"].write_text(
        json.dumps(
            {
                "protocol": {
                    "validation": 2024,
                    "retrospective_holdout": 2025,
                    "warning": (
                        "The 2025 set was inspected during development and is not an "
                        "untouched final test set."
                    ),
                    "evaluation_method": (
                        "Each match is predicted from both perspectives and combined "
                        "with the same complementary averaging formula used in production."
                    ),
                },
                "results": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    holdout_data, holdout_probability = plot_data["retrospective_2025"]
    save_confusion_matrix_plot(
        holdout_data["target_win"],
        holdout_probability,
        paths["confusion_matrix"],
    )
    save_calibration_plot(plot_data, paths["calibration_curve"])
    return paths


def main() -> None:
    """Generate all persistent model evaluation artifacts."""
    report, plot_data = evaluate_models()
    paths = save_reports(report, plot_data)
    table = flatten_report(report)
    summary = table[table["scope"].eq("all")][
        ["split", "model", "accuracy", "roc_auc", "log_loss", "brier_score"]
    ]
    print("Advanced model evaluation")
    print("-------------------------")
    print(summary.to_string(index=False, float_format="%.4f"))
    print()
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
