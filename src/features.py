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
    "market_odds_available",
    "odds_match_confidence",
    "odds_match_method",
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
    "surface_win_rate",
    "days_since_last_match",
    "h2h_win_rate",
    "h2h_surface_win_rate",
    "matches_played",
    "surface_matches_played",
    "matches_last_14_days",
    "elo_rating",
    "elo_win_probability",
    "surface_elo_rating",
    "surface_elo_win_probability",
    "rolling_ace_rate",
    "rolling_df_rate",
    "rolling_1st_in_pct",
    "rolling_1st_won_pct",
    "rolling_2nd_won_pct",
    "rolling_bp_save_pct",
    "rolling_return_won_pct",
    "market_prob",
]

DIFFERENCE_FEATURES = {
    "rank_diff": ("player_rank", "opponent_rank"),
    "rank_points_diff": ("player_rank_points", "opponent_rank_points"),
    "age_diff": ("player_age", "opponent_age"),
    "height_diff": ("player_ht", "opponent_ht"),
    "seed_diff": ("player_seed", "opponent_seed"),
    "last5_win_rate_diff": ("player_last5_win_rate", "opponent_last5_win_rate"),
    "last10_win_rate_diff": ("player_last10_win_rate", "opponent_last10_win_rate"),
    "surface_win_rate_diff": (
        "player_surface_win_rate",
        "opponent_surface_win_rate",
    ),
    "days_since_last_match_diff": (
        "player_days_since_last_match",
        "opponent_days_since_last_match",
    ),
    "h2h_win_rate_diff": ("player_h2h_win_rate", "opponent_h2h_win_rate"),
    "h2h_surface_win_rate_diff": (
        "player_h2h_surface_win_rate",
        "opponent_h2h_surface_win_rate",
    ),
    "matches_played_diff": ("player_matches_played", "opponent_matches_played"),
    "surface_matches_played_diff": (
        "player_surface_matches_played",
        "opponent_surface_matches_played",
    ),
    "matches_last_14_days_diff": (
        "player_matches_last_14_days",
        "opponent_matches_last_14_days",
    ),
    "elo_rating_diff": ("player_elo_rating", "opponent_elo_rating"),
    "elo_win_probability_diff": (
        "player_elo_win_probability",
        "opponent_elo_win_probability",
    ),
    "surface_elo_rating_diff": (
        "player_surface_elo_rating",
        "opponent_surface_elo_rating",
    ),
    "surface_elo_win_probability_diff": (
        "player_surface_elo_win_probability",
        "opponent_surface_elo_win_probability",
    ),
    "rolling_ace_rate_diff": ("player_rolling_ace_rate", "opponent_rolling_ace_rate"),
    "rolling_df_rate_diff": ("player_rolling_df_rate", "opponent_rolling_df_rate"),
    "rolling_1st_in_pct_diff": ("player_rolling_1st_in_pct", "opponent_rolling_1st_in_pct"),
    "rolling_1st_won_pct_diff": ("player_rolling_1st_won_pct", "opponent_rolling_1st_won_pct"),
    "rolling_2nd_won_pct_diff": ("player_rolling_2nd_won_pct", "opponent_rolling_2nd_won_pct"),
    "rolling_bp_save_pct_diff": ("player_rolling_bp_save_pct", "opponent_rolling_bp_save_pct"),
    "rolling_return_won_pct_diff": ("player_rolling_return_won_pct", "opponent_rolling_return_won_pct"),
    "market_prob_diff": ("player_market_prob", "opponent_market_prob"),
}

DEFAULT_DAYS_SINCE_LAST_MATCH = 365
DEFAULT_ELO_RATING = 1500.0
ELO_K_FACTOR = 32.0

FATIGUE_WINDOW_DAYS = 14

ELO_K_FACTOR_BY_TOURNEY_LEVEL = {
    "G": 40.0,  # Grand Slam (best of 5, highest-stakes signal)
    "M": 32.0,  # Masters 1000
    "F": 32.0,  # Tour Finals
    "O": 32.0,  # Olympics
    "A": 24.0,  # Regular tour-level events
    "D": 20.0,  # Davis Cup (team context, weaker individual-form signal)
}

SERVE_STATS_WINDOW = 20

SERVE_STAT_NAMES = [
    "rolling_ace_rate",
    "rolling_df_rate",
    "rolling_1st_in_pct",
    "rolling_1st_won_pct",
    "rolling_2nd_won_pct",
    "rolling_bp_save_pct",
    "rolling_return_won_pct",
]

DEFAULT_SERVE_STATS = {
    "rolling_ace_rate": 0.060,
    "rolling_df_rate": 0.035,
    "rolling_1st_in_pct": 0.620,
    "rolling_1st_won_pct": 0.720,
    "rolling_2nd_won_pct": 0.530,
    "rolling_bp_save_pct": 0.630,
    "rolling_return_won_pct": 0.380,
}

REQUIRED_RAW_SERVE_COLUMNS = [
    "w_svpt", "w_ace", "w_df", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
    "l_svpt", "l_ace", "l_df", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced",
]

CATEGORICAL_COLUMNS_TO_ENCODE = [
    "surface",
    "tourney_level",
    "round",
    "player_hand",
    "opponent_hand",
    "hand_matchup",
]


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    """Divide two values, returning None on missing data or zero denominator."""
    if numerator is None or denominator is None:
        return None
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return None
    return float(numerator / denominator)


def _compute_player_serve_stats(
    match: pd.Series,
    serve_prefix: str,
    return_prefix: str,
) -> dict[str, float | None]:
    """Compute serve/return ratios for one player from a single raw match."""
    svpt = match.get(f"{serve_prefix}_svpt")
    ace = match.get(f"{serve_prefix}_ace")
    df_val = match.get(f"{serve_prefix}_df")
    first_in = match.get(f"{serve_prefix}_1stIn")
    first_won = match.get(f"{serve_prefix}_1stWon")
    second_won = match.get(f"{serve_prefix}_2ndWon")
    bp_saved = match.get(f"{serve_prefix}_bpSaved")
    bp_faced = match.get(f"{serve_prefix}_bpFaced")
    opp_svpt = match.get(f"{return_prefix}_svpt")
    opp_first_won = match.get(f"{return_prefix}_1stWon")
    opp_second_won = match.get(f"{return_prefix}_2ndWon")

    second_svpt: float | None = None
    if svpt is not None and not pd.isna(svpt) and first_in is not None and not pd.isna(first_in):
        second_svpt = float(svpt) - float(first_in)

    opp_pts_won_on_serve: float | None = None
    if (
        opp_svpt is not None and not pd.isna(opp_svpt) and opp_svpt > 0
        and opp_first_won is not None and not pd.isna(opp_first_won)
        and opp_second_won is not None and not pd.isna(opp_second_won)
    ):
        opp_pts_won_on_serve = float(opp_svpt) - float(opp_first_won) - float(opp_second_won)

    return {
        "rolling_ace_rate": _safe_div(ace, svpt),
        "rolling_df_rate": _safe_div(df_val, svpt),
        "rolling_1st_in_pct": _safe_div(first_in, svpt),
        "rolling_1st_won_pct": _safe_div(first_won, first_in),
        "rolling_2nd_won_pct": _safe_div(second_won, second_svpt),
        "rolling_bp_save_pct": _safe_div(bp_saved, bp_faced),
        "rolling_return_won_pct": _safe_div(opp_pts_won_on_serve, opp_svpt),
    }


def _rolling_serve_avg(
    history: list[dict[str, float | None]],
    stat: str,
    window: int,
) -> float:
    """Compute the rolling average of a serve stat from prior match history."""
    recent = [h[stat] for h in history[-window:] if h.get(stat) is not None]
    if not recent:
        return DEFAULT_SERVE_STATS[stat]
    return float(sum(recent) / len(recent))


def add_rolling_serve_features(raw_matches: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-free rolling serve/return stat features for each player.

    Must be called on raw match data (before leakage columns are removed).
    """
    if not all(col in raw_matches.columns for col in REQUIRED_RAW_SERVE_COLUMNS):
        return raw_matches

    features = raw_matches.copy()
    sort_date = pd.to_datetime(
        features["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    features = features.assign(_sort_date=sort_date).sort_values(
        ["_sort_date", "tourney_id", "match_num"]
    ).drop(columns="_sort_date").reset_index(drop=True)

    for prefix in ["winner", "loser"]:
        for stat in SERVE_STAT_NAMES:
            features[f"{prefix}_{stat}"] = DEFAULT_SERVE_STATS[stat]

    player_serve_history: dict[int, list[dict[str, float | None]]] = {}

    for _, date_matches in features.groupby("tourney_date", sort=False):
        pending: list[tuple[int, dict[str, float | None]]] = []

        for row_index, match in date_matches.iterrows():
            winner_id = int(match["winner_id"])
            loser_id = int(match["loser_id"])

            winner_history = player_serve_history.get(winner_id, [])
            loser_history = player_serve_history.get(loser_id, [])

            for stat in SERVE_STAT_NAMES:
                features.at[row_index, f"winner_{stat}"] = _rolling_serve_avg(
                    winner_history, stat, SERVE_STATS_WINDOW
                )
                features.at[row_index, f"loser_{stat}"] = _rolling_serve_avg(
                    loser_history, stat, SERVE_STATS_WINDOW
                )

            pending.append((winner_id, _compute_player_serve_stats(match, "w", "l")))
            pending.append((loser_id, _compute_player_serve_stats(match, "l", "w")))

        for player_id, stats in pending:
            player_serve_history.setdefault(player_id, []).append(stats)

    return features


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

    available_context_columns = [
        column for column in CONTEXT_COLUMNS if column in matches.columns
    ]
    rows = matches[available_context_columns].copy()

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


def calculate_days_since_last_match(
    current_date: pd.Timestamp,
    previous_date: pd.Timestamp | None,
) -> int:
    """Calculate days since a player's previous match."""
    if previous_date is None:
        return DEFAULT_DAYS_SINCE_LAST_MATCH

    return int((current_date - previous_date).days)


def calculate_elo_win_probability(player_elo: float, opponent_elo: float) -> float:
    """Calculate expected win probability from two Elo ratings."""
    return float(1 / (1 + 10 ** ((opponent_elo - player_elo) / 400)))


def get_elo_k_factor(tourney_level: str) -> float:
    """Look up the Elo K-factor for a tournament level, weighting higher-stakes events more."""
    return ELO_K_FACTOR_BY_TOURNEY_LEVEL.get(tourney_level, ELO_K_FACTOR)


def update_elo_rating(
    current_elo: float,
    expected_score: float,
    actual_score: int,
    k_factor: float = ELO_K_FACTOR,
) -> float:
    """Update an Elo rating after a match result."""
    return float(current_elo + k_factor * (actual_score - expected_score))


def count_recent_matches(
    current_date: pd.Timestamp,
    previous_dates: list[pd.Timestamp],
    window_days: int = FATIGUE_WINDOW_DAYS,
) -> int:
    """Count how many prior matches a player played within the trailing window.

    previous_dates must be chronologically sorted (ascending); iterates from the
    most recent entry backward and stops at the first date outside the window.
    """
    count = 0
    for played_date in reversed(previous_dates):
        if (current_date - played_date).days <= window_days:
            count += 1
        else:
            break
    return count


def add_rolling_form_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-free form and recency features for each player."""
    features = matches.copy()
    features["tourney_date"] = pd.to_datetime(features["tourney_date"])
    features = features.sort_values(
        ["tourney_date", "tourney_id", "match_num"]
    ).reset_index(drop=True)

    player_results: dict[int, list[int]] = {}
    player_surface_results: dict[int, dict[str, list[int]]] = {}
    player_last_match_dates: dict[int, pd.Timestamp] = {}
    player_match_dates: dict[int, list[pd.Timestamp]] = {}
    player_h2h_results: dict[tuple[int, int], list[int]] = {}
    player_h2h_surface_results: dict[tuple[int, int, str], list[int]] = {}
    player_elo_ratings: dict[int, float] = {}
    player_surface_elo_ratings: dict[int, dict[str, float]] = {}

    for prefix in ["winner", "loser"]:
        features[f"{prefix}_last5_win_rate"] = 0.5
        features[f"{prefix}_last10_win_rate"] = 0.5
        features[f"{prefix}_surface_win_rate"] = 0.5
        features[f"{prefix}_days_since_last_match"] = DEFAULT_DAYS_SINCE_LAST_MATCH
        features[f"{prefix}_h2h_win_rate"] = 0.5
        features[f"{prefix}_h2h_surface_win_rate"] = 0.5
        features[f"{prefix}_matches_played"] = 0
        features[f"{prefix}_surface_matches_played"] = 0
        features[f"{prefix}_matches_last_14_days"] = 0
        features[f"{prefix}_elo_rating"] = DEFAULT_ELO_RATING
        features[f"{prefix}_elo_win_probability"] = 0.5
        features[f"{prefix}_surface_elo_rating"] = DEFAULT_ELO_RATING
        features[f"{prefix}_surface_elo_win_probability"] = 0.5

    for match_date, date_matches in features.groupby("tourney_date", sort=False):
        pending_updates: list[
            tuple[int, int, str, int, pd.Timestamp, float, float, float, float]
        ] = []

        for row_index, match in date_matches.iterrows():
            winner_id = int(match["winner_id"])
            loser_id = int(match["loser_id"])
            surface = str(match["surface"])
            k_factor = get_elo_k_factor(str(match["tourney_level"]))

            winner_results = player_results.get(winner_id, [])
            loser_results = player_results.get(loser_id, [])
            winner_surface_results = player_surface_results.get(winner_id, {}).get(
                surface,
                [],
            )
            loser_surface_results = player_surface_results.get(loser_id, {}).get(
                surface,
                [],
            )
            winner_previous_date = player_last_match_dates.get(winner_id)
            loser_previous_date = player_last_match_dates.get(loser_id)
            winner_h2h_results = player_h2h_results.get((winner_id, loser_id), [])
            loser_h2h_results = player_h2h_results.get((loser_id, winner_id), [])
            winner_h2h_surface_results = player_h2h_surface_results.get(
                (winner_id, loser_id, surface), []
            )
            loser_h2h_surface_results = player_h2h_surface_results.get(
                (loser_id, winner_id, surface), []
            )
            winner_matches_played = len(winner_results)
            loser_matches_played = len(loser_results)
            winner_surface_matches_played = len(winner_surface_results)
            loser_surface_matches_played = len(loser_surface_results)
            winner_matches_last_14_days = count_recent_matches(
                match_date, player_match_dates.get(winner_id, [])
            )
            loser_matches_last_14_days = count_recent_matches(
                match_date, player_match_dates.get(loser_id, [])
            )
            winner_elo = player_elo_ratings.get(winner_id, DEFAULT_ELO_RATING)
            loser_elo = player_elo_ratings.get(loser_id, DEFAULT_ELO_RATING)
            winner_elo_win_probability = calculate_elo_win_probability(
                winner_elo,
                loser_elo,
            )
            loser_elo_win_probability = calculate_elo_win_probability(
                loser_elo,
                winner_elo,
            )
            winner_surface_elo = player_surface_elo_ratings.get(winner_id, {}).get(
                surface,
                DEFAULT_ELO_RATING,
            )
            loser_surface_elo = player_surface_elo_ratings.get(loser_id, {}).get(
                surface,
                DEFAULT_ELO_RATING,
            )
            winner_surface_elo_win_probability = calculate_elo_win_probability(
                winner_surface_elo,
                loser_surface_elo,
            )
            loser_surface_elo_win_probability = calculate_elo_win_probability(
                loser_surface_elo,
                winner_surface_elo,
            )

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
            features.at[row_index, "winner_surface_win_rate"] = calculate_prior_win_rate(
                winner_surface_results,
                window=len(winner_surface_results),
            )
            features.at[row_index, "loser_surface_win_rate"] = calculate_prior_win_rate(
                loser_surface_results,
                window=len(loser_surface_results),
            )
            features.at[
                row_index,
                "winner_days_since_last_match",
            ] = calculate_days_since_last_match(match_date, winner_previous_date)
            features.at[
                row_index,
                "loser_days_since_last_match",
            ] = calculate_days_since_last_match(match_date, loser_previous_date)
            features.at[row_index, "winner_h2h_win_rate"] = calculate_prior_win_rate(
                winner_h2h_results,
                window=len(winner_h2h_results),
            )
            features.at[row_index, "loser_h2h_win_rate"] = calculate_prior_win_rate(
                loser_h2h_results,
                window=len(loser_h2h_results),
            )
            features.at[
                row_index,
                "winner_h2h_surface_win_rate",
            ] = calculate_prior_win_rate(
                winner_h2h_surface_results,
                window=len(winner_h2h_surface_results),
            )
            features.at[
                row_index,
                "loser_h2h_surface_win_rate",
            ] = calculate_prior_win_rate(
                loser_h2h_surface_results,
                window=len(loser_h2h_surface_results),
            )
            features.at[row_index, "winner_matches_played"] = winner_matches_played
            features.at[row_index, "loser_matches_played"] = loser_matches_played
            features.at[
                row_index,
                "winner_surface_matches_played",
            ] = winner_surface_matches_played
            features.at[
                row_index,
                "loser_surface_matches_played",
            ] = loser_surface_matches_played
            features.at[
                row_index,
                "winner_matches_last_14_days",
            ] = winner_matches_last_14_days
            features.at[
                row_index,
                "loser_matches_last_14_days",
            ] = loser_matches_last_14_days
            features.at[row_index, "winner_elo_rating"] = winner_elo
            features.at[row_index, "loser_elo_rating"] = loser_elo
            features.at[
                row_index,
                "winner_elo_win_probability",
            ] = winner_elo_win_probability
            features.at[
                row_index,
                "loser_elo_win_probability",
            ] = loser_elo_win_probability
            features.at[row_index, "winner_surface_elo_rating"] = winner_surface_elo
            features.at[row_index, "loser_surface_elo_rating"] = loser_surface_elo
            features.at[
                row_index,
                "winner_surface_elo_win_probability",
            ] = winner_surface_elo_win_probability
            features.at[
                row_index,
                "loser_surface_elo_win_probability",
            ] = loser_surface_elo_win_probability

            pending_updates.extend(
                [
                    (
                        winner_id,
                        loser_id,
                        surface,
                        1,
                        match_date,
                        winner_elo,
                        winner_elo_win_probability,
                        winner_surface_elo,
                        winner_surface_elo_win_probability,
                        k_factor,
                    ),
                    (
                        loser_id,
                        winner_id,
                        surface,
                        0,
                        match_date,
                        loser_elo,
                        loser_elo_win_probability,
                        loser_surface_elo,
                        loser_surface_elo_win_probability,
                        k_factor,
                    ),
                ]
            )

        for (
            player_id,
            opponent_id,
            surface,
            result,
            played_date,
            prior_elo,
            expected_score,
            prior_surface_elo,
            expected_surface_score,
            k_factor,
        ) in pending_updates:
            player_results.setdefault(player_id, []).append(result)
            player_surface_results.setdefault(player_id, {}).setdefault(
                surface,
                [],
            ).append(result)
            player_last_match_dates[player_id] = played_date
            player_match_dates.setdefault(player_id, []).append(played_date)
            player_h2h_results.setdefault((player_id, opponent_id), []).append(result)
            player_h2h_surface_results.setdefault(
                (player_id, opponent_id, surface), []
            ).append(result)
            player_elo_ratings[player_id] = update_elo_rating(
                prior_elo,
                expected_score,
                result,
                k_factor,
            )
            player_surface_elo_ratings.setdefault(player_id, {})[
                surface
            ] = update_elo_rating(
                prior_surface_elo,
                expected_surface_score,
                result,
                k_factor,
            )

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
    """Create match features with leakage-free advanced form features."""
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
