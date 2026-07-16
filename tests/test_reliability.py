"""Regression tests for production evaluation and matching reliability."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import DIFFERENCE_FEATURES, add_difference_features
from src.odds import (
    MIN_MATCH_CONFIDENCE,
    _find_odds_match_with_confidence,
    _name_tokens,
    match_odds_to_matches,
)
from src.predict import load_model
from src.report import expected_calibration_error, symmetric_match_probabilities


def _odds_candidate(winner: str, loser: str, date: str = "2025-01-02") -> dict:
    winner_parts = winner.split()
    loser_parts = loser.split()
    return {
        "Date": pd.Timestamp(date),
        "Tournament": "Example Open",
        "Winner": winner,
        "Loser": loser,
        "winner_surname_tokens": _name_tokens(" ".join(winner_parts[:-1])),
        "winner_initial": winner_parts[-1].rstrip(".").lower(),
        "loser_surname_tokens": _name_tokens(" ".join(loser_parts[:-1])),
        "loser_initial": loser_parts[-1].rstrip(".").lower(),
        "fair_winner_prob": 0.6,
        "fair_loser_prob": 0.4,
    }


def test_production_symmetry_formula_returns_complementary_rows() -> None:
    data = pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m1", "m2"],
            "target_win": [1, 1, 0, 0],
        }
    )
    probability, gaps = symmetric_match_probabilities(
        data, np.array([0.70, 0.40, 0.50, 0.30])
    )

    assert probability.tolist() == pytest.approx([0.60, 0.55, 0.40, 0.45])
    assert probability[0] + probability[2] == pytest.approx(1.0)
    assert probability[1] + probability[3] == pytest.approx(1.0)
    assert gaps["mean"] == pytest.approx(0.25)
    assert gaps.keys() == {"match_count", "mean", "median", "p95", "max"}


def test_symmetric_evaluation_rejects_incomplete_match_pairs() -> None:
    data = pd.DataFrame({"match_id": ["m1"], "target_win": [1]})

    with pytest.raises(ValueError, match="one row per perspective"):
        symmetric_match_probabilities(data, np.array([0.7]))


def test_expected_calibration_error_is_zero_for_perfect_bins() -> None:
    assert expected_calibration_error([0, 1], [0.0, 1.0]) == pytest.approx(0.0)


def test_all_difference_features_flip_sign_when_players_swap() -> None:
    row = {}
    for offset, (_, (player_column, opponent_column)) in enumerate(
        DIFFERENCE_FEATURES.items(), start=1
    ):
        row[player_column] = float(offset + 2)
        row[opponent_column] = float(offset)
    swapped = {
        **{
            key.replace("player_", "opponent_", 1): value
            for key, value in row.items()
            if key.startswith("player_")
        },
        **{
            key.replace("opponent_", "player_", 1): value
            for key, value in row.items()
            if key.startswith("opponent_")
        },
    }

    original_features = add_difference_features(pd.DataFrame([row]))
    swapped_features = add_difference_features(pd.DataFrame([swapped]))
    for difference_name in DIFFERENCE_FEATURES:
        assert swapped_features.loc[0, difference_name] == pytest.approx(
            -original_features.loc[0, difference_name]
        )


def test_surname_only_odds_match_is_below_model_threshold() -> None:
    candidates = pd.DataFrame([_odds_candidate("Smith Z.", "Jones Y.")])

    found, confidence, method = _find_odds_match_with_confidence(
        "Alice Smith", "Bob Jones", candidates
    )

    assert found is not None
    assert method == "surname_only"
    assert confidence < MIN_MATCH_CONFIDENCE


def test_strict_odds_match_records_confidence_and_uses_narrow_window() -> None:
    raw = pd.DataFrame(
        [
            {
                "tourney_id": "2025-X",
                "tourney_name": "Example Open",
                "tourney_date": 20250101,
                "winner_name": "Alice Smith",
                "loser_name": "Bob Jones",
            }
        ]
    )
    odds = pd.DataFrame([_odds_candidate("Smith A.", "Jones B.")])

    matched = match_odds_to_matches(raw, odds)

    assert matched.loc[0, "market_odds_available"] == 1
    assert matched.loc[0, "odds_match_confidence"] >= MIN_MATCH_CONFIDENCE
    assert "tournament" in matched.loc[0, "odds_match_method"]


def test_missing_model_error_contains_recovery_commands(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_model.py"):
        load_model(tmp_path / "missing.pkl")
