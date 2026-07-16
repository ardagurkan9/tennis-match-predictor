"""Tests for the Streamlit frontend's helper modules.

Covers src.match_history (recent form / H2H lookups), src.comparison
(color-coding rules) and the same-player / partial-odds guard rails already
enforced by src.predict.
"""

import numpy as np
import pandas as pd
import pytest

from src.comparison import compare_stat, extract_one_hot_value, format_stat_value
from src.match_history import head_to_head, recent_matches
from src.predict import build_future_match_features, predict_match, split_features_and_target


def _raw_match(
    date: int,
    match_num: int,
    winner_id: int,
    winner_name: str,
    loser_id: int,
    loser_name: str,
    surface: str = "Hard",
    tourney_name: str = "Test Event",
) -> dict:
    row = {
        "tourney_id": f"{str(date)[:4]}-TEST",
        "tourney_name": tourney_name,
        "surface": surface,
        "draw_size": 32,
        "tourney_level": "A",
        "tourney_date": date,
        "match_num": match_num,
        "winner_id": winner_id,
        "winner_seed": None,
        "winner_entry": None,
        "winner_name": winner_name,
        "winner_hand": "R",
        "winner_ht": 185,
        "winner_ioc": "AAA",
        "winner_age": 25.0,
        "loser_id": loser_id,
        "loser_seed": None,
        "loser_entry": None,
        "loser_name": loser_name,
        "loser_hand": "L",
        "loser_ht": 183,
        "loser_ioc": "BBB",
        "loser_age": 26.0,
        "score": "6-4 6-4",
        "best_of": 3,
        "round": "R32",
        "minutes": 90,
        "winner_rank": 10,
        "winner_rank_points": 3000,
        "loser_rank": 20,
        "loser_rank_points": 1800,
    }
    for prefix, values in {
        "w": (60, 6, 2, 38, 28, 12, 3, 5),
        "l": (62, 3, 4, 37, 23, 10, 4, 7),
    }.items():
        (
            row[f"{prefix}_svpt"],
            row[f"{prefix}_ace"],
            row[f"{prefix}_df"],
            row[f"{prefix}_1stIn"],
            row[f"{prefix}_1stWon"],
            row[f"{prefix}_2ndWon"],
            row[f"{prefix}_bpSaved"],
            row[f"{prefix}_bpFaced"],
        ) = values
        row[f"{prefix}_SvGms"] = 10
    return row


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _raw_match(20240101, 1, 1, "Player A", 2, "Player B", surface="Hard"),
            _raw_match(20240102, 2, 2, "Player B", 3, "Player C", surface="Clay"),
            _raw_match(20240201, 3, 1, "Player A", 2, "Player B", surface="Clay"),
            _raw_match(20240301, 4, 2, "Player B", 1, "Player A", surface="Hard"),
        ]
    )


# --- match_history.recent_matches -------------------------------------------------


def test_recent_matches_only_include_matches_before_selected_date() -> None:
    history = _history()

    matches = recent_matches(history, "Player A", "2024-02-15", limit=5)

    dates = [match["date"] for match in matches]
    assert all(date < pd.Timestamp("2024-02-15") for date in dates)
    assert dates == sorted(dates, reverse=True)
    # Only the two matches before 2024-02-15 involving Player A.
    assert len(matches) == 2
    assert matches[0]["result"] == "W"
    assert matches[0]["opponent"] == "Player B"


def test_recent_matches_respects_limit() -> None:
    history = _history()

    matches = recent_matches(history, "Player A", "2024-12-31", limit=1)

    assert len(matches) == 1


# --- match_history.head_to_head ---------------------------------------------------


def test_h2h_only_includes_matches_before_selected_date() -> None:
    history = _history()

    matches = head_to_head(history, "Player A", "Player B", "2024-02-15")

    assert len(matches) == 2
    assert all(match["date"] < pd.Timestamp("2024-02-15") for match in matches)


def test_h2h_returns_empty_for_players_who_never_met() -> None:
    history = _history()

    matches = head_to_head(history, "Player A", "Player C", "2024-12-31")

    assert matches == []


# --- src.comparison.compare_stat ---------------------------------------------------


def test_atp_rank_lower_value_is_advantage() -> None:
    assert compare_stat(5, 20, "lower") == ("advantage", "disadvantage")
    assert compare_stat(20, 5, "lower") == ("disadvantage", "advantage")


def test_double_fault_rate_lower_value_is_advantage() -> None:
    assert compare_stat(0.02, 0.05, "lower") == ("advantage", "disadvantage")


def test_win_rate_and_elo_higher_value_is_advantage() -> None:
    assert compare_stat(0.65, 0.40, "higher") == ("advantage", "disadvantage")
    assert compare_stat(1700, 1500, "higher") == ("advantage", "disadvantage")


def test_missing_value_is_reported_as_missing_not_colored() -> None:
    assert compare_stat(None, 5, "higher") == ("missing", "missing")
    assert compare_stat(5, np.nan, "lower") == ("missing", "missing")


def test_equal_values_are_neutral() -> None:
    assert compare_stat(1500, 1500, "higher") == ("equal", "equal")


def test_neutral_direction_never_colors_as_advantage() -> None:
    assert compare_stat(10, 2, "neutral") == ("neutral", "neutral")


def test_format_stat_value_shows_dash_for_missing() -> None:
    assert format_stat_value(None) == "—"
    assert format_stat_value(np.nan) == "—"
    assert format_stat_value(0.523, as_percentage=True) == "%52.3"


def test_extract_one_hot_value_recovers_hand_from_dummy_columns() -> None:
    features = {"player_hand_R": 1, "opponent_hand_L": 1, "player_rank": 5}

    assert extract_one_hot_value(features, "player_hand") == "R"
    assert extract_one_hot_value(features, "opponent_hand") == "L"
    assert extract_one_hot_value(features, "player_missing") is None


# --- guard rails already enforced by src.predict, exercised for the frontend ------


def test_same_player_selection_raises_error() -> None:
    with pytest.raises(ValueError, match="must be different"):
        build_future_match_features(
            raw_matches=_history(),
            player="Player A",
            opponent="Player A",
            surface="Hard",
            match_date="2024-04-01",
        )


class _FixedProbabilityModel:
    """A fake model that returns a fixed win probability regardless of input."""

    def __init__(self, feature_names: list[str], win_probability: float) -> None:
        self.feature_names_in_ = feature_names
        self._win_probability = win_probability

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        rows = len(features)
        return np.tile([1 - self._win_probability, self._win_probability], (rows, 1))


def test_prediction_probabilities_always_sum_to_one() -> None:
    history = _history()
    player_row, _, _ = build_future_match_features(
        raw_matches=history,
        player="Player A",
        opponent="Player B",
        surface="Hard",
        match_date="2024-04-01",
    )
    feature_names = list(split_features_and_target(player_row)[0].columns)
    model = _FixedProbabilityModel(feature_names, win_probability=0.6)

    result = predict_match(
        raw_matches=history,
        player="Player A",
        opponent="Player B",
        surface="Hard",
        match_date="2024-04-01",
        model=model,
    )

    assert result.player_probability + result.opponent_probability == pytest.approx(1.0)
