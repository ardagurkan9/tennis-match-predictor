"""Regression tests for the project's leakage-prevention guarantees."""

import pandas as pd
from pandas.testing import assert_frame_equal

from src.clean import POST_MATCH_LEAKAGE_COLUMNS, remove_post_match_leakage
from src.features import (
    DEFAULT_ELO_RATING,
    DEFAULT_SERVE_STATS,
    SERVE_STAT_NAMES,
    add_rolling_form_features,
    add_rolling_serve_features,
)
from src.split import split_features_by_year


def make_form_matches(rows: list[dict]) -> pd.DataFrame:
    """Create the minimum cleaned match frame required by rolling form features."""
    defaults = {
        "surface": "Hard",
        "tourney_level": "A",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def make_raw_serve_match(**overrides: float) -> pd.DataFrame:
    """Create one raw match containing every serve-stat input column."""
    row = {
        "tourney_date": 20240101,
        "tourney_id": "2024-TEST",
        "match_num": 1,
        "winner_id": 1,
        "loser_id": 2,
        "w_svpt": 60,
        "w_ace": 6,
        "w_df": 2,
        "w_1stIn": 36,
        "w_1stWon": 27,
        "w_2ndWon": 12,
        "w_bpSaved": 3,
        "w_bpFaced": 4,
        "l_svpt": 60,
        "l_ace": 3,
        "l_df": 4,
        "l_1stIn": 35,
        "l_1stWon": 21,
        "l_2ndWon": 10,
        "l_bpSaved": 4,
        "l_bpFaced": 7,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_post_match_columns_are_removed_before_modeling() -> None:
    match = pd.DataFrame(
        [
            {
                "winner_name": "Player A",
                "loser_name": "Player B",
                **{column: 1 for column in POST_MATCH_LEAKAGE_COLUMNS},
            }
        ]
    )

    cleaned = remove_post_match_leakage(match)

    assert not set(POST_MATCH_LEAKAGE_COLUMNS) & set(cleaned.columns)
    assert {"winner_name", "loser_name"}.issubset(cleaned.columns)


def test_current_match_serve_stats_do_not_enter_its_rolling_features() -> None:
    ordinary_match = make_raw_serve_match()
    extreme_match = make_raw_serve_match(
        w_ace=55,
        w_df=0,
        w_1stIn=59,
        w_1stWon=59,
        l_ace=0,
        l_df=30,
    )

    ordinary_features = add_rolling_serve_features(ordinary_match)
    extreme_features = add_rolling_serve_features(extreme_match)

    rolling_columns = [
        f"{prefix}_{stat}"
        for prefix in ("winner", "loser")
        for stat in SERVE_STAT_NAMES
    ]
    assert_frame_equal(
        ordinary_features[rolling_columns],
        extreme_features[rolling_columns],
    )
    for prefix in ("winner", "loser"):
        for stat, default in DEFAULT_SERVE_STATS.items():
            assert ordinary_features.loc[0, f"{prefix}_{stat}"] == default


def test_changing_a_future_result_does_not_change_past_form_features() -> None:
    history = [
        {
            "tourney_date": "2024-01-01",
            "tourney_id": "2024-A",
            "match_num": 1,
            "winner_id": 1,
            "loser_id": 2,
        },
        {
            "tourney_date": "2024-01-02",
            "tourney_id": "2024-B",
            "match_num": 1,
            "winner_id": 3,
            "loser_id": 1,
        },
    ]
    player_one_wins_future = make_form_matches(
        history
        + [
            {
                "tourney_date": "2024-02-01",
                "tourney_id": "2024-C",
                "match_num": 1,
                "winner_id": 1,
                "loser_id": 4,
            }
        ]
    )
    player_one_loses_future = make_form_matches(
        history
        + [
            {
                "tourney_date": "2024-02-01",
                "tourney_id": "2024-C",
                "match_num": 1,
                "winner_id": 4,
                "loser_id": 1,
            }
        ]
    )

    first_result = add_rolling_form_features(player_one_wins_future).iloc[:2]
    second_result = add_rolling_form_features(player_one_loses_future).iloc[:2]
    generated_columns = [
        column
        for column in first_result.columns
        if column.startswith("winner_") or column.startswith("loser_")
    ]

    assert_frame_equal(
        first_result[generated_columns].reset_index(drop=True),
        second_result[generated_columns].reset_index(drop=True),
    )


def test_matches_on_the_same_date_do_not_update_each_others_form() -> None:
    same_day_matches = make_form_matches(
        [
            {
                "tourney_date": "2024-01-01",
                "tourney_id": "2024-A",
                "match_num": 1,
                "winner_id": 1,
                "loser_id": 2,
            },
            {
                "tourney_date": "2024-01-01",
                "tourney_id": "2024-A",
                "match_num": 2,
                "winner_id": 1,
                "loser_id": 3,
            },
        ]
    )

    features = add_rolling_form_features(same_day_matches)

    assert features.loc[0, "winner_last5_win_rate"] == 0.5
    assert features.loc[1, "winner_last5_win_rate"] == 0.5
    assert features.loc[0, "winner_matches_played"] == 0
    assert features.loc[1, "winner_matches_played"] == 0
    assert features.loc[0, "winner_elo_rating"] == DEFAULT_ELO_RATING
    assert features.loc[1, "winner_elo_rating"] == DEFAULT_ELO_RATING


def test_time_split_never_mixes_years() -> None:
    features = pd.DataFrame(
        {
            "tourney_year": [2014, 2015, 2023, 2024, 2025, 2026],
            "target_win": [0, 1, 0, 1, 0, 1],
        }
    )

    train, validation, test = split_features_by_year(features)

    assert set(train["tourney_year"]) == {2015, 2023}
    assert set(validation["tourney_year"]) == {2024}
    assert set(test["tourney_year"]) == {2025}
    assert set(train["tourney_year"]).isdisjoint(validation["tourney_year"])
    assert set(train["tourney_year"]).isdisjoint(test["tourney_year"])
    assert set(validation["tourney_year"]).isdisjoint(test["tourney_year"])
