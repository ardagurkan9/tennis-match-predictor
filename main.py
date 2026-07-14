import argparse

from src.clean import (
    POST_MATCH_LEAKAGE_COLUMNS,
    clean_matches,
    get_missing_value_summary,
    save_cleaned_matches,
)
from src.ingest import load_raw_matches


def run_cleaning_check() -> None:
    """Run the ingestion and cleaning pipeline check."""
    matches = load_raw_matches()
    cleaned_matches = clean_matches(matches)
    output_path = save_cleaned_matches(cleaned_matches)

    remaining_leakage_columns = [
        column for column in POST_MATCH_LEAKAGE_COLUMNS if column in cleaned_matches.columns
    ]

    print("ATP preprocessing check")
    print("-----------------------")
    print(f"Raw data shape: {matches.shape}")
    print(f"Cleaned data shape: {cleaned_matches.shape}")
    print(f"Cleaned data saved to: {output_path}")
    print(f"Remaining leakage columns: {remaining_leakage_columns}")
    print(f"winner_rank kept: {'winner_rank' in cleaned_matches.columns}")
    print(f"loser_rank kept: {'loser_rank' in cleaned_matches.columns}")
    print(f"entry columns dropped: {not {'winner_entry', 'loser_entry'} & set(cleaned_matches.columns)}")
    print(
        "seed indicator columns present: "
        f"{all(column in cleaned_matches.columns for column in ['winner_is_seeded', 'loser_is_seeded'])}"
    )
    print(
        "missing seed values: "
        f"{cleaned_matches[['winner_seed', 'loser_seed']].isna().sum().sum()}"
    )
    print(
        "missing height values: "
        f"{cleaned_matches[['winner_ht', 'loser_ht']].isna().sum().sum()}"
    )
    print(f"tourney_date dtype: {cleaned_matches['tourney_date'].dtype}")
    print(
        "date columns present: "
        f"{all(column in cleaned_matches.columns for column in ['tourney_year', 'tourney_month', 'tourney_day'])}"
    )
    print(f"missing converted dates: {cleaned_matches['tourney_date'].isna().sum()}")

    preview_columns = [
        "tourney_date",
        "tourney_year",
        "tourney_month",
        "tourney_day",
        "surface",
        "winner_name",
        "loser_name",
        "winner_rank",
        "loser_rank",
        "winner_seed",
        "loser_seed",
        "winner_is_seeded",
        "loser_is_seeded",
        "winner_ht",
        "loser_ht",
    ]

    print()
    print("Cleaned data preview")
    print("--------------------")
    print(cleaned_matches[preview_columns].head().to_string(index=False))

    missing_summary = get_missing_value_summary(cleaned_matches)

    print()
    print("Missing value summary")
    print("---------------------")
    print(missing_summary.head(10).to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Parse the pipeline selected from the command line."""
    parser = argparse.ArgumentParser(
        description="Run a Tennis Match Predictor data pipeline.",
    )
    parser.add_argument(
        "pipeline",
        nargs="?",
        choices=("clean", "advanced"),
        default="clean",
        help=(
            "Pipeline to run: 'clean' performs the ingestion/cleaning check; "
            "'advanced' builds advanced features and time-based splits "
            "(default: clean)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the selected project pipeline."""
    args = parse_args()

    if args.pipeline == "advanced":
        from src.build_advanced_features import build_advanced_features

        build_advanced_features()
        return

    run_cleaning_check()


if __name__ == "__main__":
    main()
