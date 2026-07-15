"""Build leakage-free future-match features and generate win probabilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import (
    DEFAULT_DAYS_SINCE_LAST_MATCH,
    DEFAULT_ELO_RATING,
    SERVE_STAT_NAMES,
    _compute_player_serve_stats,
    _rolling_serve_avg,
    add_difference_features,
    add_matchup_features,
    calculate_days_since_last_match,
    calculate_elo_win_probability,
    calculate_prior_win_rate,
    count_recent_matches,
    encode_categorical_features,
    get_elo_k_factor,
    update_elo_rating,
)
from src.ingest import load_raw_matches
from src.odds import fair_probability_from_decimal_odds
from src.train import split_features_and_target


DEFAULT_MODEL_PATH = Path("models/advanced/lightgbm.pkl")
FUTURE_TOURNEY_ID_PREFIX = "FUTURE"
FUTURE_MATCH_NUM = 999_999

PLAYER_SNAPSHOT_COLUMNS = (
    "id",
    "name",
    "hand",
    "ht",
    "ioc",
    "age",
    "rank",
    "rank_points",
)


@dataclass
class PredictionResult:
    """A symmetric two-player match prediction and its display-ready feature rows."""

    player: str
    opponent: str
    player_probability: float
    opponent_probability: float
    direct_player_probability: float
    direct_opponent_probability: float
    symmetry_gap: float
    player_features: dict[str, object]
    opponent_features: dict[str, object]
    data_cutoff: pd.Timestamp


def _raw_match_dates(matches: pd.DataFrame) -> pd.Series:
    """Parse the raw integer ATP tournament date column."""
    return pd.to_datetime(
        matches["tourney_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )


def filter_history_before_date(
    raw_matches: pd.DataFrame,
    match_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return only matches strictly before the requested prediction date."""
    match_date = pd.Timestamp(match_date).normalize()
    dates = _raw_match_dates(raw_matches)
    history = raw_matches.loc[dates < match_date].copy()
    if history.empty:
        raise ValueError(f"No historical matches are available before {match_date.date()}.")
    return history.reset_index(drop=True)


def list_available_players(raw_matches: pd.DataFrame) -> list[str]:
    """List every non-empty player name in a raw match history."""
    names = pd.concat(
        [raw_matches["winner_name"], raw_matches["loser_name"]],
        ignore_index=True,
    ).dropna()
    return sorted(name for name in names.astype(str).unique() if name.strip())


def _player_appearances(raw_matches: pd.DataFrame, player_name: str) -> pd.DataFrame:
    """Normalize a player's winner/loser appearances into one chronological table."""
    frames = []
    for prefix in ("winner", "loser"):
        appearances = raw_matches[raw_matches[f"{prefix}_name"].eq(player_name)].copy()
        if appearances.empty:
            continue
        normalized = pd.DataFrame(
            {
                column: appearances[f"{prefix}_{column}"].to_numpy()
                for column in PLAYER_SNAPSHOT_COLUMNS
            }
        )
        normalized["tourney_date"] = _raw_match_dates(appearances).to_numpy()
        normalized["match_num"] = appearances["match_num"].to_numpy()
        frames.append(normalized)

    if not frames:
        raise ValueError(f"Player not found before the selected date: {player_name}")

    return pd.concat(frames, ignore_index=True).sort_values(
        ["tourney_date", "match_num"]
    )


def latest_player_snapshot(
    raw_matches: pd.DataFrame,
    player_name: str,
    match_date: str | pd.Timestamp,
) -> dict[str, object]:
    """Get the latest known pre-match profile and age it to the requested date."""
    appearances = _player_appearances(raw_matches, player_name)
    snapshot = appearances.iloc[-1].to_dict()
    match_date = pd.Timestamp(match_date).normalize()
    snapshot_date = pd.Timestamp(snapshot["tourney_date"])
    if pd.notna(snapshot.get("age")):
        elapsed_years = max(0, (match_date - snapshot_date).days) / 365.25
        snapshot["age"] = float(snapshot["age"]) + elapsed_years
    return snapshot


def _market_probabilities(
    player_odds: float | None,
    opponent_odds: float | None,
) -> tuple[float, float, int]:
    """Build market inputs, using neutral values only when both odds are absent."""
    if player_odds is None and opponent_odds is None:
        return 0.5, 0.5, 0
    if player_odds is None or opponent_odds is None:
        raise ValueError("Provide both decimal odds or leave both empty.")
    player_probability, opponent_probability = fair_probability_from_decimal_odds(
        player_odds,
        opponent_odds,
    )
    return player_probability, opponent_probability, 1


def build_future_raw_match(
    history: pd.DataFrame,
    player: str,
    opponent: str,
    surface: str,
    match_date: str | pd.Timestamp,
    tourney_level: str = "A",
    best_of: int = 3,
    round_name: str = "R32",
    draw_size: int = 32,
    player_odds: float | None = None,
    opponent_odds: float | None = None,
) -> pd.DataFrame:
    """Create a placeholder raw row whose result cannot affect its own features."""
    if player == opponent:
        raise ValueError("Player and opponent must be different.")
    surface = surface.strip().title()
    if surface not in {"Hard", "Clay", "Grass", "Carpet"}:
        raise ValueError(f"Unsupported surface: {surface}")
    if best_of not in {3, 5}:
        raise ValueError("best_of must be 3 or 5.")

    match_date = pd.Timestamp(match_date).normalize()
    player_snapshot = latest_player_snapshot(history, player, match_date)
    opponent_snapshot = latest_player_snapshot(history, opponent, match_date)
    player_market_prob, opponent_market_prob, odds_available = _market_probabilities(
        player_odds,
        opponent_odds,
    )

    row = {column: np.nan for column in history.columns}
    row.update(
        {
            "tourney_id": f"{FUTURE_TOURNEY_ID_PREFIX}-{match_date:%Y%m%d}",
            "tourney_name": "Future Match",
            "surface": surface,
            "draw_size": draw_size,
            "tourney_level": tourney_level,
            "tourney_date": int(match_date.strftime("%Y%m%d")),
            "match_num": FUTURE_MATCH_NUM,
            "best_of": best_of,
            "round": round_name,
            "winner_seed": np.nan,
            "loser_seed": np.nan,
            "winner_entry": np.nan,
            "loser_entry": np.nan,
            "score": np.nan,
            "winner_market_prob": player_market_prob,
            "loser_market_prob": opponent_market_prob,
            "market_odds_available": odds_available,
        }
    )
    for prefix, snapshot in (
        ("winner", player_snapshot),
        ("loser", opponent_snapshot),
    ):
        for column in PLAYER_SNAPSHOT_COLUMNS:
            row[f"{prefix}_{column}"] = snapshot[column]

    return pd.DataFrame([row])


def build_future_match_features(
    raw_matches: pd.DataFrame,
    player: str,
    opponent: str,
    surface: str,
    match_date: str | pd.Timestamp,
    tourney_level: str = "A",
    best_of: int = 3,
    round_name: str = "R32",
    draw_size: int = 32,
    player_odds: float | None = None,
    opponent_odds: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Build both future perspectives from a memory-efficient historical state."""
    match_date = pd.Timestamp(match_date).normalize()
    history = filter_history_before_date(raw_matches, match_date)
    if player == opponent:
        raise ValueError("Player and opponent must be different.")
    surface = surface.strip().title()
    if surface not in {"Hard", "Clay", "Grass", "Carpet"}:
        raise ValueError(f"Unsupported surface: {surface}")
    if best_of not in {3, 5}:
        raise ValueError("best_of must be 3 or 5.")

    player_snapshot = latest_player_snapshot(history, player, match_date)
    opponent_snapshot = latest_player_snapshot(history, opponent, match_date)
    player_id = int(player_snapshot["id"])
    opponent_id = int(opponent_snapshot["id"])
    market_player, market_opponent, odds_available = _market_probabilities(
        player_odds, opponent_odds
    )
    state = _build_historical_state(history)
    player_stats = _future_player_stats(
        state, player_id, opponent_id, surface, match_date
    )
    opponent_stats = _future_player_stats(
        state, opponent_id, player_id, surface, match_date
    )
    player_stats["market_prob"] = market_player
    opponent_stats["market_prob"] = market_opponent

    common = {
        "match_id": f"{FUTURE_TOURNEY_ID_PREFIX}-{match_date:%Y%m%d}_{FUTURE_MATCH_NUM}",
        "tourney_id": f"{FUTURE_TOURNEY_ID_PREFIX}-{match_date:%Y%m%d}",
        "tourney_name": "Future Match",
        "surface": surface,
        "draw_size": draw_size,
        "tourney_level": tourney_level,
        "tourney_date": match_date,
        "match_num": FUTURE_MATCH_NUM,
        "best_of": best_of,
        "round": round_name,
        "tourney_year": match_date.year,
        "tourney_month": match_date.month,
        "tourney_day": match_date.day,
        "market_odds_available": odds_available,
    }
    player_row = _build_perspective_row(
        common, player_snapshot, opponent_snapshot, player_stats, opponent_stats, 1
    )
    opponent_row = _build_perspective_row(
        common, opponent_snapshot, player_snapshot, opponent_stats, player_stats, 0
    )
    data_cutoff = _raw_match_dates(history).max()
    return player_row, opponent_row, data_cutoff


def _build_historical_state(history: pd.DataFrame) -> dict[str, dict]:
    """Process match history once without materializing a full feature matrix."""
    matches = history.copy()
    matches["_date"] = _raw_match_dates(matches)
    matches = matches.sort_values(["_date", "tourney_id", "match_num"])
    state: dict[str, dict] = {
        "results": {},
        "surface_results": {},
        "last_dates": {},
        "match_dates": {},
        "h2h": {},
        "h2h_surface": {},
        "elo": {},
        "surface_elo": {},
        "serve": {},
    }

    for played_date, date_matches in matches.groupby("_date", sort=False):
        pending = []
        for _, match in date_matches.iterrows():
            winner_id = int(match["winner_id"])
            loser_id = int(match["loser_id"])
            surface = str(match["surface"]).strip().title()
            winner_elo = state["elo"].get(winner_id, DEFAULT_ELO_RATING)
            loser_elo = state["elo"].get(loser_id, DEFAULT_ELO_RATING)
            winner_expected = calculate_elo_win_probability(winner_elo, loser_elo)
            loser_expected = 1 - winner_expected
            winner_surface_elo = state["surface_elo"].get(winner_id, {}).get(
                surface, DEFAULT_ELO_RATING
            )
            loser_surface_elo = state["surface_elo"].get(loser_id, {}).get(
                surface, DEFAULT_ELO_RATING
            )
            winner_surface_expected = calculate_elo_win_probability(
                winner_surface_elo, loser_surface_elo
            )
            loser_surface_expected = 1 - winner_surface_expected
            k_factor = get_elo_k_factor(str(match["tourney_level"]))
            winner_serve = _compute_player_serve_stats(match, "w", "l")
            loser_serve = _compute_player_serve_stats(match, "l", "w")
            pending.extend(
                [
                    (
                        winner_id, loser_id, surface, 1, winner_elo,
                        winner_expected, winner_surface_elo,
                        winner_surface_expected, k_factor, winner_serve,
                    ),
                    (
                        loser_id, winner_id, surface, 0, loser_elo,
                        loser_expected, loser_surface_elo,
                        loser_surface_expected, k_factor, loser_serve,
                    ),
                ]
            )

        for (
            player_id, other_id, surface, result, prior_elo, expected,
            prior_surface_elo, surface_expected, k_factor, serve_stats,
        ) in pending:
            state["results"].setdefault(player_id, []).append(result)
            state["surface_results"].setdefault(player_id, {}).setdefault(
                surface, []
            ).append(result)
            state["last_dates"][player_id] = played_date
            state["match_dates"].setdefault(player_id, []).append(played_date)
            state["h2h"].setdefault((player_id, other_id), []).append(result)
            state["h2h_surface"].setdefault(
                (player_id, other_id, surface), []
            ).append(result)
            state["elo"][player_id] = update_elo_rating(
                prior_elo, expected, result, k_factor
            )
            state["surface_elo"].setdefault(player_id, {})[surface] = (
                update_elo_rating(
                    prior_surface_elo, surface_expected, result, k_factor
                )
            )
            state["serve"].setdefault(player_id, []).append(serve_stats)
    return state


def _future_player_stats(
    state: dict[str, dict],
    player_id: int,
    opponent_id: int,
    surface: str,
    match_date: pd.Timestamp,
) -> dict[str, float | int]:
    """Calculate one player's advanced pre-match values from final state."""
    results = state["results"].get(player_id, [])
    surface_results = state["surface_results"].get(player_id, {}).get(surface, [])
    h2h = state["h2h"].get((player_id, opponent_id), [])
    h2h_surface = state["h2h_surface"].get(
        (player_id, opponent_id, surface), []
    )
    player_elo = state["elo"].get(player_id, DEFAULT_ELO_RATING)
    opponent_elo = state["elo"].get(opponent_id, DEFAULT_ELO_RATING)
    player_surface_elo = state["surface_elo"].get(player_id, {}).get(
        surface, DEFAULT_ELO_RATING
    )
    opponent_surface_elo = state["surface_elo"].get(opponent_id, {}).get(
        surface, DEFAULT_ELO_RATING
    )
    serve_history = state["serve"].get(player_id, [])
    stats: dict[str, float | int] = {
        "last5_win_rate": calculate_prior_win_rate(results, 5),
        "last10_win_rate": calculate_prior_win_rate(results, 10),
        "surface_win_rate": calculate_prior_win_rate(
            surface_results, len(surface_results)
        ),
        "days_since_last_match": calculate_days_since_last_match(
            match_date, state["last_dates"].get(player_id)
        ),
        "h2h_win_rate": calculate_prior_win_rate(h2h, len(h2h)),
        "h2h_surface_win_rate": calculate_prior_win_rate(
            h2h_surface, len(h2h_surface)
        ),
        "matches_played": len(results),
        "surface_matches_played": len(surface_results),
        "matches_last_14_days": count_recent_matches(
            match_date, state["match_dates"].get(player_id, [])
        ),
        "elo_rating": player_elo,
        "elo_win_probability": calculate_elo_win_probability(
            player_elo, opponent_elo
        ),
        "surface_elo_rating": player_surface_elo,
        "surface_elo_win_probability": calculate_elo_win_probability(
            player_surface_elo, opponent_surface_elo
        ),
    }
    for stat in SERVE_STAT_NAMES:
        stats[stat] = _rolling_serve_avg(serve_history, stat, 20)
    return stats


def _build_perspective_row(
    common: dict[str, object],
    player_snapshot: dict[str, object],
    opponent_snapshot: dict[str, object],
    player_stats: dict[str, object],
    opponent_stats: dict[str, object],
    target_win: int,
) -> pd.DataFrame:
    """Create and encode one model-compatible player/opponent perspective."""
    row = dict(common)
    for prefix, snapshot, stats in (
        ("player", player_snapshot, player_stats),
        ("opponent", opponent_snapshot, opponent_stats),
    ):
        for column in PLAYER_SNAPSHOT_COLUMNS:
            row[f"{prefix}_{column}"] = snapshot[column]
        row[f"{prefix}_seed"] = 0
        row[f"{prefix}_is_seeded"] = 0
        for column, value in stats.items():
            row[f"{prefix}_{column}"] = value
    row["target_win"] = target_win
    features = pd.DataFrame([row])
    features = add_difference_features(features)
    features = add_matchup_features(features)
    return encode_categorical_features(features)


def align_model_features(
    feature_row: pd.DataFrame,
    model: object,
) -> pd.DataFrame:
    """Drop metadata and align one row to the trained model's exact schema."""
    model_features, _ = split_features_and_target(feature_row)
    expected_columns = list(model.feature_names_in_)
    missing_columns = set(expected_columns) - set(model_features.columns)
    unexpected_columns = set(model_features.columns) - set(expected_columns)

    # Missing one-hot categories are valid and must be zero. Missing numeric
    # columns indicate a broken inference pipeline and must not be silently hidden.
    allowed_missing = {
        column
        for column in missing_columns
        if any(
            column.startswith(f"{prefix}_")
            for prefix in (
                "surface",
                "tourney_level",
                "round",
                "player_hand",
                "opponent_hand",
                "hand_matchup",
            )
        )
    }
    invalid_missing = missing_columns - allowed_missing
    if invalid_missing:
        raise RuntimeError(
            "Inference is missing required numeric model features: "
            f"{sorted(invalid_missing)}"
        )

    if unexpected_columns:
        model_features = model_features.drop(columns=sorted(unexpected_columns))
    return model_features.reindex(columns=expected_columns, fill_value=0)


def predict_match(
    raw_matches: pd.DataFrame,
    player: str,
    opponent: str,
    surface: str,
    match_date: str | pd.Timestamp,
    tourney_level: str = "A",
    best_of: int = 3,
    round_name: str = "R32",
    draw_size: int = 32,
    player_odds: float | None = None,
    opponent_odds: float | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> PredictionResult:
    """Predict a future match from both perspectives and enforce a symmetric result."""
    model = joblib.load(model_path)
    player_row, opponent_row, data_cutoff = build_future_match_features(
        raw_matches=raw_matches,
        player=player,
        opponent=opponent,
        surface=surface,
        match_date=match_date,
        tourney_level=tourney_level,
        best_of=best_of,
        round_name=round_name,
        draw_size=draw_size,
        player_odds=player_odds,
        opponent_odds=opponent_odds,
    )
    player_model_features = align_model_features(player_row, model)
    opponent_model_features = align_model_features(opponent_row, model)
    direct_player_probability = float(
        model.predict_proba(player_model_features)[0, 1]
    )
    direct_opponent_probability = float(
        model.predict_proba(opponent_model_features)[0, 1]
    )

    mirrored_player_probability = 1 - direct_opponent_probability
    player_probability = (
        direct_player_probability + mirrored_player_probability
    ) / 2
    opponent_probability = 1 - player_probability
    symmetry_gap = abs(direct_player_probability - mirrored_player_probability)

    return PredictionResult(
        player=player,
        opponent=opponent,
        player_probability=player_probability,
        opponent_probability=opponent_probability,
        direct_player_probability=direct_player_probability,
        direct_opponent_probability=direct_opponent_probability,
        symmetry_gap=symmetry_gap,
        player_features=player_row.iloc[0].to_dict(),
        opponent_features=opponent_row.iloc[0].to_dict(),
        data_cutoff=data_cutoff,
    )


def parse_args() -> argparse.Namespace:
    """Parse a future matchup from the command line."""
    parser = argparse.ArgumentParser(description="Predict an ATP tennis matchup.")
    parser.add_argument("--player", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--surface", choices=("Hard", "Clay", "Grass", "Carpet"), required=True)
    parser.add_argument("--date", required=True, help="Match date in YYYY-MM-DD format.")
    parser.add_argument("--tourney-level", default="A")
    parser.add_argument("--best-of", type=int, choices=(3, 5), default=3)
    parser.add_argument("--round", dest="round_name", default="R32")
    parser.add_argument("--draw-size", type=int, default=32)
    parser.add_argument("--player-odds", type=float)
    parser.add_argument("--opponent-odds", type=float)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    """Run a future matchup prediction from raw project history."""
    args = parse_args()
    raw_matches = load_raw_matches()
    result = predict_match(
        raw_matches=raw_matches,
        player=args.player,
        opponent=args.opponent,
        surface=args.surface,
        match_date=args.date,
        tourney_level=args.tourney_level,
        best_of=args.best_of,
        round_name=args.round_name,
        draw_size=args.draw_size,
        player_odds=args.player_odds,
        opponent_odds=args.opponent_odds,
        model_path=args.model,
    )
    predicted_winner = (
        result.player
        if result.player_probability >= result.opponent_probability
        else result.opponent
    )
    print("Future match prediction")
    print("-----------------------")
    print(f"{result.player}: {result.player_probability:.2%}")
    print(f"{result.opponent}: {result.opponent_probability:.2%}")
    print(f"Predicted winner: {predicted_winner}")
    print(f"Historical data cutoff: {result.data_cutoff.date()}")
    print(f"Perspective symmetry gap: {result.symmetry_gap:.4f}")


if __name__ == "__main__":
    main()
