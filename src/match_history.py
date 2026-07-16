"""Raw match-history lookups for the frontend: recent form and head-to-head.

Built on top of src.predict's existing leakage-free date filtering so the
frontend never re-implements match-history logic on its own.
"""

from __future__ import annotations

import pandas as pd

from src.predict import _raw_match_dates, filter_history_before_date


def recent_matches(
    raw_matches: pd.DataFrame,
    player_name: str,
    before_date: str | pd.Timestamp,
    limit: int = 5,
) -> list[dict[str, object]]:
    """Return a player's most recent matches strictly before before_date.

    Ordered most recent first. Each entry has date, opponent, surface,
    tourney_name and result ("W" or "L").
    """
    history = filter_history_before_date(raw_matches, before_date)
    appearances = history.loc[
        history["winner_name"].eq(player_name) | history["loser_name"].eq(player_name)
    ].copy()
    if appearances.empty:
        return []

    appearances["_date"] = _raw_match_dates(appearances)
    appearances = appearances.sort_values(["_date", "match_num"])

    rows = []
    for _, match in appearances.iterrows():
        won = match["winner_name"] == player_name
        rows.append(
            {
                "date": match["_date"],
                "opponent": match["loser_name"] if won else match["winner_name"],
                "surface": match["surface"],
                "tourney_name": match.get("tourney_name"),
                "result": "W" if won else "L",
            }
        )
    return list(reversed(rows[-limit:]))


def head_to_head(
    raw_matches: pd.DataFrame,
    player_name: str,
    opponent_name: str,
    before_date: str | pd.Timestamp,
) -> list[dict[str, object]]:
    """Return every match between two players strictly before before_date.

    Ordered most recent first. Each entry has date, surface, winner and
    tourney_name. Returns an empty list if the two players never met.
    """
    history = filter_history_before_date(raw_matches, before_date)
    is_player_vs_opponent = history["winner_name"].eq(player_name) & history[
        "loser_name"
    ].eq(opponent_name)
    is_opponent_vs_player = history["winner_name"].eq(opponent_name) & history[
        "loser_name"
    ].eq(player_name)
    matches = history.loc[is_player_vs_opponent | is_opponent_vs_player].copy()
    if matches.empty:
        return []

    matches["_date"] = _raw_match_dates(matches)
    matches = matches.sort_values(["_date", "match_num"])

    rows = [
        {
            "date": match["_date"],
            "surface": match["surface"],
            "winner": match["winner_name"],
            "tourney_name": match.get("tourney_name"),
        }
        for _, match in matches.iterrows()
    ]
    return list(reversed(rows))
