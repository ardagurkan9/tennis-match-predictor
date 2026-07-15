"""Tests for leakage-free future-match inference utilities."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.odds import fair_probability_from_decimal_odds
from src.predict import (
    _market_probabilities,
    build_future_match_features,
    filter_history_before_date,
)


def _raw_match(
    date: int,
    match_num: int,
    winner_id: int,
    winner_name: str,
    loser_id: int,
    loser_name: str,
) -> dict:
    row = {
        "tourney_id": f"{str(date)[:4]}-TEST",
        "tourney_name": "Test Event",
        "surface": "Hard",
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
        "loser_hand": "R",
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
            _raw_match(20240101, 1, 1, "Player A", 2, "Player B"),
            _raw_match(20240102, 2, 2, "Player B", 3, "Player C"),
        ]
    )


def test_decimal_odds_are_normalized_without_vig() -> None:
    player, opponent = fair_probability_from_decimal_odds(1.80, 2.20)

    assert player + opponent == pytest.approx(1.0)
    assert player == pytest.approx(0.55)


def test_missing_odds_are_neutral_and_partial_odds_are_rejected() -> None:
    assert _market_probabilities(None, None) == (0.5, 0.5, 0)

    with pytest.raises(ValueError, match="both decimal odds"):
        _market_probabilities(1.8, None)


def test_history_filter_is_strictly_before_prediction_date() -> None:
    history = _history()

    filtered = filter_history_before_date(history, "2024-01-02")

    assert filtered["tourney_date"].tolist() == [20240101]


def test_future_features_use_historical_form_and_h2h() -> None:
    player_row, opponent_row, cutoff = build_future_match_features(
        raw_matches=_history(),
        player="Player A",
        opponent="Player B",
        surface="Hard",
        match_date="2024-01-03",
    )

    assert cutoff == pd.Timestamp("2024-01-02")
    assert player_row.loc[0, "player_last5_win_rate"] == 1.0
    assert opponent_row.loc[0, "player_last5_win_rate"] == 0.5
    assert player_row.loc[0, "player_h2h_win_rate"] == 1.0
    assert opponent_row.loc[0, "player_h2h_win_rate"] == 0.0
    assert player_row.loc[0, "player_market_prob"] == 0.5
    assert opponent_row.loc[0, "player_market_prob"] == 0.5


def test_appending_a_future_result_does_not_change_inference_features() -> None:
    history = _history()
    history_with_future = pd.concat(
        [
            history,
            pd.DataFrame(
                [_raw_match(20250101, 3, 2, "Player B", 1, "Player A")]
            ),
        ],
        ignore_index=True,
    )

    original_player, original_opponent, _ = build_future_match_features(
        history, "Player A", "Player B", "Hard", "2024-01-03"
    )
    future_player, future_opponent, _ = build_future_match_features(
        history_with_future, "Player A", "Player B", "Hard", "2024-01-03"
    )

    assert_frame_equal(original_player, future_player)
    assert_frame_equal(original_opponent, future_opponent)
