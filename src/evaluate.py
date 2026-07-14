"""Model evaluation utilities."""

import argparse
from pathlib import Path

import pandas as pd


DATASET_DIRECTORIES = {
    "base": Path("data/features"),
    "advanced": Path("data/features/advanced"),
}


def rank_baseline_predict(features: pd.DataFrame) -> pd.Series:
    """Predict that the player with the better ATP rank wins."""
    return (features["player_rank"] < features["opponent_rank"]).astype("int8")


def calculate_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculate classification accuracy."""
    return float((y_true == y_pred).mean())


def evaluate_rank_baseline(features: pd.DataFrame) -> float:
    """Evaluate the rank-based baseline on a feature dataset."""
    predictions = rank_baseline_predict(features)
    return calculate_accuracy(features["target_win"], predictions)


def evaluate_rank_baseline_splits(
    validation_path: str | Path = "data/features/validation.csv",
    test_path: str | Path = "data/features/test.csv",
) -> dict[str, float]:
    """Evaluate the rank-based baseline on validation and test splits."""
    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path)

    return {
        "validation_baseline_accuracy": evaluate_rank_baseline(validation),
        "test_baseline_accuracy": evaluate_rank_baseline(test),
    }


def parse_args() -> argparse.Namespace:
    """Parse the feature dataset selected for baseline evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate the rank baseline.")
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_DIRECTORIES),
        default="base",
        help="Feature dataset to evaluate (default: base).",
    )
    return parser.parse_args()


def main() -> None:
    """Run rank-based baseline evaluation."""
    args = parse_args()
    data_dir = DATASET_DIRECTORIES[args.dataset]
    metrics = evaluate_rank_baseline_splits(
        validation_path=data_dir / "validation.csv",
        test_path=data_dir / "test.csv",
    )

    title = f"Rank-based baseline evaluation ({args.dataset})"
    print(title)
    print("-" * len(title))
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
