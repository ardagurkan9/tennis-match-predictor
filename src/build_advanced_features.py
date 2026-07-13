"""Build the advanced feature dataset with rolling serve statistics."""

from pathlib import Path

from src.clean import clean_matches
from src.features import (
    add_rolling_serve_features,
    create_advanced_match_features,
    save_advanced_match_features,
)
from src.ingest import load_raw_matches
from src.odds import load_odds_files, match_odds_to_matches
from src.split import save_split_csvs, split_features_by_year


OUTPUT_DIR = Path("data/features/advanced")


def build_advanced_features() -> None:
    print("Loading raw match data...")
    raw_matches = load_raw_matches()
    print(f"  Raw matches (full history): {raw_matches.shape}")

    print("Computing rolling serve statistics (leakage-free)...")
    raw_with_serve = add_rolling_serve_features(raw_matches)
    serve_cols = [c for c in raw_with_serve.columns if "rolling_" in c and c.startswith("winner_")]
    print(f"  Added {len(serve_cols)} serve feature columns per player")

    print("Loading and matching market odds (leakage-free, pre-match)...")
    odds = load_odds_files()
    raw_with_odds = match_odds_to_matches(raw_with_serve, odds)
    match_count = int(raw_with_odds["market_odds_available"].sum())
    print(f"  Market odds matched: {match_count} / {len(raw_with_odds)} matches ({match_count / len(raw_with_odds):.2%})")

    print("Cleaning matches...")
    cleaned = clean_matches(raw_with_odds)
    print(f"  Cleaned shape: {cleaned.shape}")

    print("Building advanced match features (Elo/H2H/form computed over full history)...")
    features = create_advanced_match_features(cleaned)
    print(f"  Feature matrix shape (full history): {features.shape}")

    features = features[features["tourney_year"] >= 2015].reset_index(drop=True)
    print(f"  Feature matrix shape (2015+ only, history-warmed): {features.shape}")

    save_advanced_match_features(features)
    print(f"  Features saved to: {OUTPUT_DIR}/match_features.parquet")

    train, validation, test = split_features_by_year(features)
    save_split_csvs(train, validation, test, output_dir=OUTPUT_DIR)
    print(f"  Train: {train.shape}, Validation: {validation.shape}, Test: {test.shape}")
    print("Done.")


if __name__ == "__main__":
    build_advanced_features()
