"""Build the advanced feature dataset with rolling serve statistics."""

from pathlib import Path

from src.clean import clean_matches
from src.features import (
    add_rolling_serve_features,
    create_advanced_match_features,
    save_advanced_match_features,
)
from src.ingest import load_raw_matches
from src.split import save_split_csvs, split_features_by_year


OUTPUT_DIR = Path("data/features/advanced")


def build_advanced_features() -> None:
    print("Loading raw match data...")
    raw_matches = load_raw_matches()
    raw_matches = raw_matches[raw_matches["tourney_date"] >= 20150000].reset_index(drop=True)
    print(f"  Raw matches: {raw_matches.shape}")

    print("Computing rolling serve statistics (leakage-free)...")
    raw_with_serve = add_rolling_serve_features(raw_matches)
    serve_cols = [c for c in raw_with_serve.columns if "rolling_" in c and c.startswith("winner_")]
    print(f"  Added {len(serve_cols)} serve feature columns per player")

    print("Cleaning matches...")
    cleaned = clean_matches(raw_with_serve)
    print(f"  Cleaned shape: {cleaned.shape}")

    print("Building advanced match features...")
    features = create_advanced_match_features(cleaned)
    print(f"  Feature matrix shape: {features.shape}")

    save_advanced_match_features(features)
    print(f"  Features saved to: {OUTPUT_DIR}/match_features.parquet")

    train, validation, test = split_features_by_year(features)
    save_split_csvs(train, validation, test, output_dir=OUTPUT_DIR)
    print(f"  Train: {train.shape}, Validation: {validation.shape}, Test: {test.shape}")
    print("Done.")


if __name__ == "__main__":
    build_advanced_features()
