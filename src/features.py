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
