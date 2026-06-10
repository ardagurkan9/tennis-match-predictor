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

ADVANCED_PLAYER_COLUMNS = [
    "last5_win_rate",
    "last10_win_rate",
]

DIFFERENCE_FEATURES = {
    "rank_diff": ("player_rank", "opponent_rank"),
    "rank_points_diff": ("player_rank_points", "opponent_rank_points"),
    "age_diff": ("player_age", "opponent_age"),
    "height_diff": ("player_ht", "opponent_ht"),
    "seed_diff": ("player_seed", "opponent_seed"),
    "last5_win_rate_diff": ("player_last5_win_rate", "opponent_last5_win_rate"),
    "last10_win_rate_diff": ("player_last10_win_rate", "opponent_last10_win_rate"),
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

    optional_columns = [
        column for column in ADVANCED_PLAYER_COLUMNS
        if (
            f"{player_prefix}_{column}" in matches.columns
            and f"{opponent_prefix}_{column}" in matches.columns
        )
    ]

    for column in PLAYER_COLUMNS + optional_columns:
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
        if player_column in features.columns and opponent_column in features.columns:
            features[feature_name] = features[player_column] - features[opponent_column]

    return features


def calculate_prior_win_rate(results: list[int], window: int) -> float:
    """Calculate a player's prior rolling win rate."""
    recent_results = results[-window:]

    if not recent_results:
        return 0.5

    return float(sum(recent_results) / len(recent_results))


def add_rolling_form_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-free rolling win-rate features for each player."""
    features = matches.copy()
    features["tourney_date"] = pd.to_datetime(features["tourney_date"])
    features = features.sort_values(
        ["tourney_date", "tourney_id", "match_num"]
    ).reset_index(drop=True)

    player_results: dict[int, list[int]] = {}

    for prefix in ["winner", "loser"]:
        features[f"{prefix}_last5_win_rate"] = 0.5
        features[f"{prefix}_last10_win_rate"] = 0.5

    for _, date_matches in features.groupby("tourney_date", sort=False):
        pending_updates: list[tuple[int, int]] = []

        for row_index, match in date_matches.iterrows():
            winner_id = int(match["winner_id"])
            loser_id = int(match["loser_id"])

            winner_results = player_results.get(winner_id, [])
            loser_results = player_results.get(loser_id, [])

            features.at[row_index, "winner_last5_win_rate"] = calculate_prior_win_rate(
                winner_results,
                window=5,
            )
            features.at[row_index, "winner_last10_win_rate"] = calculate_prior_win_rate(
                winner_results,
                window=10,
            )
            features.at[row_index, "loser_last5_win_rate"] = calculate_prior_win_rate(
                loser_results,
                window=5,
            )
            features.at[row_index, "loser_last10_win_rate"] = calculate_prior_win_rate(
                loser_results,
                window=10,
            )

            pending_updates.extend([(winner_id, 1), (loser_id, 0)])

        for player_id, result in pending_updates:
            player_results.setdefault(player_id, []).append(result)

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


def create_advanced_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Create match features with leakage-free rolling form features."""
    matches_with_form = add_rolling_form_features(matches)
    return create_match_features(matches_with_form)


def save_match_features(
    match_features: pd.DataFrame,
    output_path: str | Path = "data/features/match_features.parquet",
) -> Path:
    """Save match feature data to a parquet file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    match_features.to_parquet(output_path, index=False)
    return output_path


def save_advanced_match_features(
    match_features: pd.DataFrame,
    output_path: str | Path = "data/features/advanced/match_features.parquet",
) -> Path:
    """Save advanced match feature data to a separate parquet file."""
    return save_match_features(match_features, output_path)
