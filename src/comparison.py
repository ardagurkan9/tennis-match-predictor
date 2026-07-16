"""Pure helpers for the frontend's player-vs-opponent stat comparisons.

These only format and compare values that already exist in a PredictionResult;
they never compute or estimate a statistic themselves.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

ComparisonStatus = Literal["advantage", "disadvantage", "equal", "missing", "neutral"]

Direction = Literal["higher", "lower", "neutral"]


def _is_missing(value: object) -> bool:
    return value is None or pd.isna(value)


def compare_stat(
    player_value: float | None,
    opponent_value: float | None,
    direction: Direction,
) -> tuple[ComparisonStatus, ComparisonStatus]:
    """Classify a player/opponent value pair for color-coded display.

    direction="higher": a larger value is an advantage (e.g. win rate, Elo).
    direction="lower": a smaller value is an advantage (e.g. ATP rank, double faults).
    direction="neutral": never colored as an advantage/disadvantage (e.g. match counts).
    """
    if direction == "neutral":
        return "neutral", "neutral"
    if direction not in ("higher", "lower"):
        raise ValueError(f"Unknown direction: {direction}")

    if _is_missing(player_value) or _is_missing(opponent_value):
        return "missing", "missing"
    if player_value == opponent_value:
        return "equal", "equal"

    player_is_better = (
        player_value > opponent_value if direction == "higher" else player_value < opponent_value
    )
    return ("advantage", "disadvantage") if player_is_better else ("disadvantage", "advantage")


def extract_one_hot_value(features: dict[str, object], prefix: str) -> str | None:
    """Recover an original categorical value from its one-hot encoded columns.

    predict_match's feature rows one-hot encode columns such as player_hand
    (e.g. player_hand_R=1); a single-row prediction always has exactly one
    active dummy per encoded category, so this finds and strips that prefix.
    """
    marker = f"{prefix}_"
    for key, value in features.items():
        if key.startswith(marker) and value == 1:
            return key[len(marker) :]
    return None


def format_stat_value(value: object, as_percentage: bool = False) -> str:
    """Format a feature value for display, or '—' if missing."""
    if _is_missing(value):
        return "—"
    if as_percentage:
        return f"%{float(value) * 100:.1f}"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value)}"
        return f"{value:.1f}"
    return str(value)
