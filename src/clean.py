from pathlib import Path

import pandas as pd

POST_MATCH_LEAKAGE_COLUMNS = [
    "score",
    "minutes",
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
]

ENTRY_COLUMNS = [
    "winner_entry",
    "loser_entry",
]

SEED_COLUMNS = [
    "winner_seed",
    "loser_seed",
]

HEIGHT_COLUMNS = [
    "winner_ht",
    "loser_ht",
]

RANK_COLUMNS = [
    "winner_rank",
    "loser_rank",
]

RANK_POINTS_COLUMNS = [
    "winner_rank_points",
    "loser_rank_points",
]

AGE_COLUMNS = [
    "winner_age",
    "loser_age",
]

HAND_COLUMNS = [
    "winner_hand",
    "loser_hand",
]


def remove_post_match_leakage(matches: pd.DataFrame) -> pd.DataFrame:
    """Remove post-match leakage columns from the raw data."""
    return matches.drop(columns=POST_MATCH_LEAKAGE_COLUMNS, errors="ignore")


def convert_tournament_dates(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert tournament dates to datetime and add date-part columns."""
    cleaned = matches.copy()
    cleaned["tourney_date"] = pd.to_datetime(
        cleaned["tourney_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    cleaned["tourney_year"] = cleaned["tourney_date"].dt.year
    cleaned["tourney_month"] = cleaned["tourney_date"].dt.month
    cleaned["tourney_day"] = cleaned["tourney_date"].dt.day
    return cleaned


def normalize_surface(matches: pd.DataFrame) -> pd.DataFrame:
    """Normalize surface values before feature engineering."""
    cleaned = matches.copy()
    cleaned["surface"] = cleaned["surface"].str.strip().str.title()
    return cleaned


def remove_duplicate_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate matches based on key columns."""
    duplicate_key = [
        "tourney_date",
        "tourney_name",
        "surface",
        "winner_name",
        "loser_name",
        "score",
        "round",
    ]
    available_duplicate_key = [
        column for column in duplicate_key if column in matches.columns
    ]

    return matches.drop_duplicates(
        subset=available_duplicate_key,
        keep="first"
    ).reset_index(drop=True)


def drop_entry_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Drop entry columns with very high missing values."""
    return matches.drop(columns=ENTRY_COLUMNS, errors="ignore")


def fill_seed_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Fill missing seed values and add seeded indicator columns."""
    cleaned = matches.copy()

    for column in SEED_COLUMNS:
        if column in cleaned.columns:
            player_prefix = column.removesuffix("_seed")
            cleaned[f"{player_prefix}_is_seeded"] = cleaned[column].notna().astype("int8")
            cleaned[column] = cleaned[column].fillna(0)

    return cleaned


def fill_height_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Fill missing player heights with each height column's median."""
    cleaned = matches.copy()

    for column in HEIGHT_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    return cleaned


def fill_rank_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Fill missing player ranks with each rank column's median."""
    cleaned = matches.copy()

    for column in RANK_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    return cleaned


def fill_rank_points_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Fill missing player ranking points with 0."""
    cleaned = matches.copy()

    for column in RANK_POINTS_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(0)

    return cleaned


def fill_surface_column(matches: pd.DataFrame) -> pd.DataFrame:
    """Fill missing surface values with Unknown."""
    cleaned = matches.copy()

    if "surface" in cleaned.columns:
        cleaned["surface"] = cleaned["surface"].fillna("Unknown")

    return cleaned


def fill_age_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Fill missing player ages with each age column's median."""
    cleaned = matches.copy()

    for column in AGE_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    return cleaned


def fill_hand_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Fill missing player hand values with U."""
    cleaned = matches.copy()

    for column in HAND_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("U")

    return cleaned


def get_missing_value_summary(matches: pd.DataFrame) -> pd.DataFrame:
    """Get a summary of missing values by column."""
    return (
        matches.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .rename("missing_percent")
        .reset_index()
        .rename(columns={"index": "column"})
    )


def save_cleaned_matches(
    cleaned_matches: pd.DataFrame,
    output_path: str | Path = "data/processed/cleaned_matches.parquet",
) -> Path:
    """Save cleaned match data to a parquet file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_to_save = cleaned_matches.copy()
    if "tourney_date" in data_to_save.columns:
        data_to_save["tourney_date"] = data_to_save["tourney_date"].dt.strftime("%Y-%m-%d")

    data_to_save.to_parquet(output_path, index=False)
    return output_path


def clean_matches(matches: pd.DataFrame) -> pd.DataFrame:
    cleaned = convert_tournament_dates(matches)
    cleaned = normalize_surface(cleaned)
    cleaned = remove_duplicate_matches(cleaned)
    cleaned = remove_post_match_leakage(cleaned)
    cleaned = drop_entry_columns(cleaned)
    cleaned = fill_seed_columns(cleaned)
    cleaned = fill_height_columns(cleaned)
    cleaned = fill_rank_columns(cleaned)
    cleaned = fill_rank_points_columns(cleaned)
    cleaned = fill_surface_column(cleaned)
    cleaned = fill_age_columns(cleaned)
    cleaned = fill_hand_columns(cleaned)
    return cleaned
