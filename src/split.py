"""Train, validation, and test split utilities."""

from pathlib import Path

import pandas as pd


TRAIN_START_YEAR = 2015
TRAIN_END_YEAR = 2023
VALIDATION_YEAR = 2024
TEST_YEAR = 2025


def split_features_by_year(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split feature data into train, validation, and test sets by year."""
    train = features[
        features["tourney_year"].between(TRAIN_START_YEAR, TRAIN_END_YEAR)
    ].reset_index(drop=True)
    validation = features[
        features["tourney_year"] == VALIDATION_YEAR
    ].reset_index(drop=True)
    test = features[
        features["tourney_year"] == TEST_YEAR
    ].reset_index(drop=True)

    return train, validation, test


def save_split_csvs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str | Path = "data/features",
) -> dict[str, Path]:
    """Save train, validation, and test splits as CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "train": output_dir / "train.csv",
        "validation": output_dir / "validation.csv",
        "test": output_dir / "test.csv",
    }

    train.to_csv(output_paths["train"], index=False)
    validation.to_csv(output_paths["validation"], index=False)
    test.to_csv(output_paths["test"], index=False)

    return output_paths


def create_split_csvs(
    feature_path: str | Path = "data/features/match_features.parquet",
    output_dir: str | Path = "data/features",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    """Load feature data, split it by year, and save CSV outputs."""
    features = pd.read_parquet(feature_path)
    train, validation, test = split_features_by_year(features)
    output_paths = save_split_csvs(train, validation, test, output_dir)

    return train, validation, test, output_paths


def main() -> None:
    """Create train, validation, and test CSV files."""
    train, validation, test, output_paths = create_split_csvs()

    print("Train / validation / test split")
    print("-------------------------------")
    print(f"Train years: {TRAIN_START_YEAR}-{TRAIN_END_YEAR}")
    print(f"Validation year: {VALIDATION_YEAR}")
    print(f"Test year: {TEST_YEAR}")
    print(f"Train shape: {train.shape}")
    print(f"Validation shape: {validation.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Train CSV: {output_paths['train']}")
    print(f"Validation CSV: {output_paths['validation']}")
    print(f"Test CSV: {output_paths['test']}")


if __name__ == "__main__":
    main()
