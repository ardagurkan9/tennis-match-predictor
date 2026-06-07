from pathlib import Path

import pandas as pd


CONTEXT_COLUMNS = [
    "match_id",
    "tourney_id",
    "tourney_name",
    "surface",
    "draw_size",
    "tourney_level",
    "tourney_date",
    "match_num",
    "best_of",
    "round",
    "tourney_year",
    "tourney_month",
    "tourney_day",
]

PLAYER_COLUMNS = [
    "id",
    "name",
    "hand",
    "ht",
    "ioc",
    "age",
    "seed",
    "is_seeded",
    "rank",
    "rank_points",
]

DIFFERENCE_FEATURES = {
    "rank_diff": ("player_rank", "opponent_rank"),
    "rank_points_diff": ("player_rank_points", "opponent_rank_points"),
    "age_diff": ("player_age", "opponent_age"),
    "height_diff": ("player_ht", "opponent_ht"),
    "seed_diff": ("player_seed", "opponent_seed"),
}

CATEGORICAL_COLUMNS_TO_ENCODE = [
    "surface",
    "tourney_level",
    "round",
    "player_hand",
    "opponent_hand",
    "hand_matchup",
]


def add_match_id(matches: pd.DataFrame) -> pd.DataFrame:
    """Create a unique match identifier."""
    features = matches.copy()
    features["match_id"] = (
        features["tourney_id"].astype(str)
        + "_"
        + features["match_num"].astype(str)
    )
    return features


def build_player_rows(
    matches: pd.DataFrame,
    player_prefix: str,
    opponent_prefix: str,
    target_win: int,
) -> pd.DataFrame:
    """Build player/opponent rows from winner or loser perspective."""

    rows = matches[CONTEXT_COLUMNS].copy()

    for column in PLAYER_COLUMNS:
        rows[f"player_{column}"] = matches[f"{player_prefix}_{column}"]
        rows[f"opponent_{column}"] = matches[f"{opponent_prefix}_{column}"]

    rows["target_win"] = target_win
    return rows


def create_player_opponent_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Create two neutral player/opponent rows for each match."""

    matches_with_id = add_match_id(matches)

    winner_rows = build_player_rows(
        matches_with_id,
        player_prefix="winner",
        opponent_prefix="loser",
        target_win=1,
    )
    loser_rows = build_player_rows(
        matches_with_id,
        player_prefix="loser",
        opponent_prefix="winner",
        target_win=0,
    )

    return pd.concat([winner_rows, loser_rows], ignore_index=True)


def add_difference_features(match_features: pd.DataFrame) -> pd.DataFrame:
    """Create numeric difference features between player and opponent."""
    features = match_features.copy()

    for feature_name, (player_column, opponent_column) in DIFFERENCE_FEATURES.items():
        features[feature_name] = features[player_column] - features[opponent_column]

    return features


def add_matchup_features(match_features: pd.DataFrame) -> pd.DataFrame:
    """Create player/opponent matchup features."""
    features = match_features.copy()

    features["hand_matchup"] = (
        features["player_hand"].astype(str)
        + "_vs_"
        + features["opponent_hand"].astype(str)
    )
    features["ioc_matchup"] = (
        features["player_ioc"].astype(str)
        + "_vs_"
        + features["opponent_ioc"].astype(str)
    )
    features["same_ioc"] = (
        features["player_ioc"] == features["opponent_ioc"]
    ).astype("int8")

    return features


def encode_categorical_features(match_features: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode selected categorical feature columns."""

    columns_to_encode = [
        column for column in CATEGORICAL_COLUMNS_TO_ENCODE
        if column in match_features.columns
    ]

    return pd.get_dummies(
        match_features,
        columns=columns_to_encode,
        dtype="int8",
    )


def create_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Create the current match feature dataset."""
    
    features = create_player_opponent_rows(matches)
    features = add_difference_features(features)
    features = add_matchup_features(features)
    features = encode_categorical_features(features)
    return features


def save_match_features(
    match_features: pd.DataFrame,
    output_path: str | Path = "data/features/match_features.parquet",
) -> Path:
    """Save match feature data to a parquet file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    match_features.to_parquet(output_path, index=False)
    return output_path
