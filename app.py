"""Streamlit frontend for the ATP future-match prediction pipeline.

This app is a thin presentation layer only: all prediction, feature-building
and market-odds math is delegated to src.predict / src.odds. No prediction
algorithm or feature formula is reimplemented here.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from src.comparison import compare_stat, extract_one_hot_value, format_stat_value
from src.ingest import load_raw_matches
from src.match_history import head_to_head, recent_matches
from src.predict import (
    DEFAULT_MODEL_PATH,
    PredictionResult,
    latest_match_date,
    list_available_players,
    load_model,
    predict_match,
)

st.set_page_config(page_title="Tennis Match Predictor", page_icon="\U0001F3BE", layout="wide")

SURFACES = ["Hard", "Clay", "Grass", "Carpet"]

TOURNEY_LEVELS = {
    "G": "Grand Slam",
    "M": "Masters 1000",
    "A": "ATP Tour (250/500)",
    "F": "Tour Finals",
    "O": "Olympics",
    "D": "Davis Cup",
}

ROUNDS = {
    "R128": "Round of 128 (R128)",
    "R64": "Round of 64 (R64)",
    "R32": "Round of 32 (R32)",
    "R16": "Round of 16 (R16)",
    "QF": "Quarterfinal (QF)",
    "SF": "Semifinal (SF)",
    "F": "Final (F)",
}

ADVANTAGE_COLOR = "var(--stat-success)"
DISADVANTAGE_COLOR = "var(--stat-danger)"
MUTED_COLOR = "var(--stat-muted)"

_GLOBAL_STYLE = """
<style>
:root {
    --stat-success: #16A34A;
    --stat-danger: #DC2626;
    --stat-muted: #64748B;
}
@media (prefers-color-scheme: dark) {
    :root {
        --stat-success: #4ADE80;
        --stat-danger: #F87171;
        --stat-muted: #94A3B8;
    }
}
.stat-table td, .stat-table th {
    font-variant-numeric: tabular-nums;
}
.stat-table tr:nth-child(even) {
    background: rgba(128, 128, 128, 0.06);
}
</style>
"""

COMPARISON_ROWS: list[tuple[str, str, str, bool]] = [
    ("ATP ranking", "rank", "lower", False),
    ("ATP ranking points", "rank_points", "higher", False),
    ("Last 5 matches win rate", "last5_win_rate", "higher", True),
    ("Last 10 matches win rate", "last10_win_rate", "higher", True),
    ("Win rate on selected surface", "surface_win_rate", "higher", True),
    ("Matches played on selected surface", "surface_matches_played", "neutral", False),
    ("Overall Elo", "elo_rating", "higher", False),
    ("Surface Elo", "surface_elo_rating", "higher", False),
    ("Overall H2H win rate", "h2h_win_rate", "higher", True),
    ("Surface H2H win rate", "h2h_surface_win_rate", "higher", True),
]

PROFILE_ROWS: list[tuple[str, str | None]] = [
    ("Age", "age"),
    ("Height (cm)", "ht"),
    ("Playing hand", None),
    ("Country", "ioc"),
    ("Days since last match", "days_since_last_match"),
    ("Matches in last 14 days", "matches_last_14_days"),
    ("Total career matches", "matches_played"),
]

SERVE_RETURN_ROWS: list[tuple[str, str, str, bool]] = [
    ("Ace rate", "rolling_ace_rate", "higher", True),
    ("Double-fault rate", "rolling_df_rate", "lower", True),
    ("First serve percentage", "rolling_1st_in_pct", "neutral", True),
    ("First serve points won rate", "rolling_1st_won_pct", "higher", True),
    ("Second serve points won rate", "rolling_2nd_won_pct", "higher", True),
    ("Return points won rate", "rolling_return_won_pct", "higher", True),
    ("Break points saved rate", "rolling_bp_save_pct", "higher", True),
]


@st.cache_resource(show_spinner="Loading historical ATP match data...")
def get_raw_matches() -> pd.DataFrame:
    return load_raw_matches()


@st.cache_data(show_spinner=False)
def get_player_list(_raw_matches: pd.DataFrame) -> list[str]:
    return list_available_players(_raw_matches)


@st.cache_resource(show_spinner="Loading model...")
def get_model(model_path: str) -> object:
    return load_model(model_path)


@st.cache_data(show_spinner=False)
def cached_predict(
    _raw_matches: pd.DataFrame,
    _model: object,
    player: str,
    opponent: str,
    surface: str,
    match_date: str,
    tourney_level: str,
    best_of: int,
    round_name: str,
    player_odds: float | None,
    opponent_odds: float | None,
) -> PredictionResult:
    return predict_match(
        raw_matches=_raw_matches,
        player=player,
        opponent=opponent,
        surface=surface,
        match_date=match_date,
        tourney_level=tourney_level,
        best_of=best_of,
        round_name=round_name,
        player_odds=player_odds,
        opponent_odds=opponent_odds,
        model=_model,
    )


def _status_color(status: str) -> str:
    return {
        "advantage": ADVANTAGE_COLOR,
        "disadvantage": DISADVANTAGE_COLOR,
        "equal": MUTED_COLOR,
        "missing": MUTED_COLOR,
        "neutral": "inherit",
    }.get(status, "inherit")


def _status_span(text: str, status: str) -> str:
    color = _status_color(status)
    weight = "700" if status in ("advantage", "disadvantage") else "500"
    return f"<span style='color:{color}; font-weight:{weight};'>{html.escape(text)}</span>"


def _comparison_table_html(
    result: PredictionResult,
    rows: list[tuple[str, str, str, bool]],
) -> str:
    header = (
        "<tr>"
        f"<th style='text-align:left; padding:8px;'>{html.escape(result.player)}</th>"
        "<th style='text-align:center; padding:8px;'>Statistic</th>"
        f"<th style='text-align:right; padding:8px;'>{html.escape(result.opponent)}</th>"
        "</tr>"
    )
    body = []
    for label, key, direction, is_pct in rows:
        player_value = result.player_features.get(f"player_{key}")
        opponent_value = result.opponent_features.get(f"player_{key}")
        player_status, opponent_status = compare_stat(player_value, opponent_value, direction)
        body.append(
            "<tr>"
            f"<td style='text-align:left; padding:8px;'>"
            f"{_status_span(format_stat_value(player_value, is_pct), player_status)}</td>"
            f"<td style='text-align:center; padding:8px; color:{MUTED_COLOR};'>{html.escape(label)}</td>"
            f"<td style='text-align:right; padding:8px;'>"
            f"{_status_span(format_stat_value(opponent_value, is_pct), opponent_status)}</td>"
            "</tr>"
        )
    return (
        "<table class='stat-table' style='width:100%; border-collapse:collapse;'>"
        + header
        + "".join(body)
        + "</table>"
    )


def render_prediction_header(result: PredictionResult) -> None:
    st.subheader("Model Prediction")

    player_pct = result.player_probability * 100
    opponent_pct = result.opponent_probability * 100
    player_is_winner = result.player_probability >= result.opponent_probability
    predicted_winner = result.player if player_is_winner else result.opponent

    col_a, col_b = st.columns(2)
    for col, name, pct, is_winner in (
        (col_a, result.player, player_pct, player_is_winner),
        (col_b, result.opponent, opponent_pct, not player_is_winner),
    ):
        with col:
            with st.container(border=True):
                st.markdown(f"**{html.escape(name)}**")
                st.markdown(f"### %{pct:.2f}")
                st.progress(min(max(pct / 100, 0.0), 1.0))
                if is_winner:
                    st.markdown(":blue-badge[Predicted winner]")

    st.markdown(f"**Predicted winner:** {html.escape(predicted_winner)}")

    if result.symmetry_gap > 0.10:
        st.warning(
            "The model produced inconsistent results from the two player "
            f"perspectives (symmetry gap: {result.symmetry_gap:.2f}). This "
            "prediction may be low-confidence."
        )

    st.warning("Live data is not used. The prediction is based only on historical match data.")


def render_stat_comparison(result: PredictionResult) -> None:
    st.subheader("Key Stat Comparison")
    st.caption(
        "Green and red only compare the two players' statistics against each other. "
        "They are not the model's exact decision rationale."
    )
    st.markdown(_comparison_table_html(result, COMPARISON_ROWS), unsafe_allow_html=True)


def render_profiles(result: PredictionResult) -> None:
    st.subheader("Player Profiles")
    col_a, col_b = st.columns(2)
    for col, name, features in (
        (col_a, result.player, result.player_features),
        (col_b, result.opponent, result.opponent_features),
    ):
        with col:
            st.markdown(f"**{html.escape(name)}**")
            hand = extract_one_hot_value(features, "player_hand") or "No data"
            lines = []
            for label, key in PROFILE_ROWS:
                value_text = hand if key is None else format_stat_value(features.get(f"player_{key}"))
                lines.append(f"- {label}: {value_text}")
            st.markdown("\n".join(lines))


def render_recent_form(raw_matches: pd.DataFrame, context: dict) -> None:
    st.subheader("Last 5 Matches")
    col_a, col_b = st.columns(2)
    for col, name in ((col_a, context["player"]), (col_b, context["opponent"])):
        with col:
            st.markdown(f"**{html.escape(name)}**")
            matches = recent_matches(raw_matches, name, context["match_date"], limit=5)
            if not matches:
                st.caption("No recorded matches found before this date.")
                continue
            for match in matches:
                won = match["result"] == "W"
                badge = ":green-badge[Won]" if won else ":red-badge[Lost]"
                st.markdown(
                    f"{badge} {match['date'].date().isoformat()} · "
                    f"{html.escape(str(match['surface']))} · vs {html.escape(str(match['opponent']))}"
                )


def render_h2h(raw_matches: pd.DataFrame, context: dict) -> None:
    st.subheader("Head-to-Head History")
    matches = head_to_head(raw_matches, context["player"], context["opponent"], context["match_date"])
    if not matches:
        st.info("These two players have never met before.")
        return

    player_name = context["player"]
    opponent_name = context["opponent"]
    player_wins = sum(1 for match in matches if match["winner"] == player_name)
    opponent_wins = len(matches) - player_wins

    surface_matches = [match for match in matches if match["surface"] == context["surface"]]
    surface_player_wins = sum(1 for match in surface_matches if match["winner"] == player_name)
    surface_opponent_wins = len(surface_matches) - surface_player_wins

    st.markdown(
        f"**Overall H2H:** {html.escape(player_name)} {player_wins} - {opponent_wins} "
        f"{html.escape(opponent_name)}"
    )
    st.markdown(
        f"**H2H on {html.escape(context['surface'])}:** {html.escape(player_name)} "
        f"{surface_player_wins} - {surface_opponent_wins} {html.escape(opponent_name)}"
    )

    for match in matches:
        st.markdown(
            f"- {match['date'].date().isoformat()} · {html.escape(str(match['surface']))} · "
            f"Winner: {html.escape(str(match['winner']))} · {html.escape(str(match['tourney_name']))}"
        )


def render_serve_return(result: PredictionResult) -> None:
    with st.expander("Detailed serve and return statistics", expanded=False):
        st.markdown(_comparison_table_html(result, SERVE_RETURN_ROWS), unsafe_allow_html=True)


def render_technical_footer(result: PredictionResult, dataset_cutoff: pd.Timestamp) -> None:
    with st.expander("Advanced technical details"):
        st.markdown(f"- Symmetry gap: {result.symmetry_gap:.4f}")
        st.markdown(f"- Player 1 direct model probability: %{result.direct_player_probability * 100:.2f}")
        st.markdown(f"- Player 2 direct model probability: %{result.direct_opponent_probability * 100:.2f}")
        st.markdown(f"- Historical data cutoff used: {result.data_cutoff.date().isoformat()}")

    st.divider()
    st.caption("This prediction is a statistical model output and does not guarantee an exact result.")
    st.caption("Live ATP data is not used.")
    st.caption(f"Last historical match date used: {dataset_cutoff.date().isoformat()}")
    st.caption(
        "2025 results were inspected during model development and are presented as a "
        "retrospective benchmark, not an untouched final test set."
    )


def _friendly_error_message(error: Exception) -> str:
    message = str(error)
    if isinstance(error, FileNotFoundError):
        return f"Model file not found: {message}"
    if isinstance(error, RuntimeError):
        return "A feature/model column mismatch occurred. Please check the model file."
    if isinstance(error, ValueError):
        if "No historical matches" in message:
            return "Not enough historical data is available before the selected date."
        if "Player not found" in message:
            player = message.split(":")[-1].strip()
            return f"Player not found before the selected date: {player}"
        if "greater than 1" in message:
            return "Invalid decimal odds. Odds must be greater than 1."
        if "both decimal odds" in message:
            return "Decimal odds were entered for only one player. Enter both odds or leave both empty."
        return message
    return "An unexpected error occurred during prediction."


def main() -> None:
    st.markdown(_GLOBAL_STYLE, unsafe_allow_html=True)
    st.title("Tennis Match Predictor")
    st.caption(
        "A statistical match prediction tool built on historical ATP data and the "
        "advanced LightGBM model."
    )
    st.warning("Live data is not used. Predictions are based only on historical match data.")

    if not DEFAULT_MODEL_PATH.exists():
        st.error(
            f"Required model file is missing: {DEFAULT_MODEL_PATH}. "
            "Download it before predicting."
        )
        st.code("python scripts/download_model.py", language="bash")
        st.caption(
            "Alternatively train it locally with: "
            "python -m src.train --dataset advanced"
        )
        st.stop()

    raw_matches = get_raw_matches()
    players = get_player_list(raw_matches)
    dataset_cutoff = latest_match_date(raw_matches)
    default_match_date = (dataset_cutoff + pd.Timedelta(days=1)).date()

    st.caption(f"Latest recorded match date in the dataset: {dataset_cutoff.date().isoformat()}")
    age_days = (pd.Timestamp.today().normalize() - dataset_cutoff.normalize()).days
    if age_days > 180:
        st.warning(
            f"The historical dataset is {age_days} days old. Predictions are "
            "low-confidence because current rankings, form and injuries are unavailable."
        )

    with st.form("match_form"):
        col1, col2 = st.columns(2)
        with col1:
            player = st.selectbox("Player 1", options=players, index=None, placeholder="Select a player")
        with col2:
            opponent = st.selectbox("Player 2", options=players, index=None, placeholder="Select a player")

        col3, col4, col5 = st.columns(3)
        with col3:
            surface = st.selectbox("Surface", options=SURFACES)
        with col4:
            match_date = st.date_input("Match date", value=default_match_date)
        with col5:
            best_of = st.selectbox("Best of", options=[3, 5])

        col6, col7 = st.columns(2)
        with col6:
            tourney_level_label = st.selectbox("Tournament level", options=list(TOURNEY_LEVELS.values()))
            tourney_level = next(code for code, label in TOURNEY_LEVELS.items() if label == tourney_level_label)
        with col7:
            round_label = st.selectbox("Round", options=list(ROUNDS.values()), index=2)
            round_name = next(code for code, label in ROUNDS.items() if label == round_label)

        col9, col10 = st.columns(2)
        with col9:
            player_odds = st.number_input(
                "Player 1 decimal odds (optional)", min_value=1.01, value=None, step=0.01, format="%.2f"
            )
        with col10:
            opponent_odds = st.number_input(
                "Player 2 decimal odds (optional)", min_value=1.01, value=None, step=0.01, format="%.2f"
            )
        st.caption("If both odds are left empty, the model uses a neutral market probability (50%/50%).")

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        if not player or not opponent:
            st.session_state["prediction_error"] = "Please select both players."
            st.session_state.pop("prediction_result", None)
        elif player == opponent:
            st.session_state["prediction_error"] = "The same player cannot be selected twice. Please choose a different Player 2."
            st.session_state.pop("prediction_result", None)
        elif (player_odds is None) != (opponent_odds is None):
            st.session_state["prediction_error"] = (
                "Decimal odds were entered for only one player. Enter both odds or leave both empty."
            )
            st.session_state.pop("prediction_result", None)
        else:
            try:
                with st.spinner("Calculating prediction..."):
                    model = get_model(str(DEFAULT_MODEL_PATH))
                    result = cached_predict(
                        raw_matches,
                        model,
                        player,
                        opponent,
                        surface,
                        match_date.isoformat(),
                        tourney_level,
                        int(best_of),
                        round_name,
                        player_odds,
                        opponent_odds,
                    )
                st.session_state["prediction_result"] = result
                st.session_state["prediction_context"] = {
                    "player": player,
                    "opponent": opponent,
                    "surface": surface,
                    "match_date": pd.Timestamp(match_date),
                }
                st.session_state.pop("prediction_error", None)
            except Exception as error:  # noqa: BLE001 - surfaced as a friendly message, never a traceback
                st.session_state["prediction_error"] = _friendly_error_message(error)
                st.session_state.pop("prediction_result", None)

    if st.session_state.get("prediction_error"):
        st.error(st.session_state["prediction_error"])

    result: PredictionResult | None = st.session_state.get("prediction_result")
    context = st.session_state.get("prediction_context")
    if result is not None and context is not None:
        render_prediction_header(result)
        st.divider()
        render_stat_comparison(result)
        st.divider()
        render_profiles(result)
        st.divider()
        render_recent_form(raw_matches, context)
        st.divider()
        render_h2h(raw_matches, context)
        st.divider()
        render_serve_return(result)
        render_technical_footer(result, dataset_cutoff)
    else:
        st.info("Fill out the form above and click 'Predict' to generate a prediction.")


if __name__ == "__main__":
    main()
