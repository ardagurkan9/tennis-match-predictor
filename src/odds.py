"""Match historical tennis-data.co.uk betting odds onto raw ATP matches.

Produces a leakage-free, pre-match "market implied probability" feature: betting
odds are set before a match starts, so using them as a feature is not leakage,
unlike in-match statistics.
"""
import re
from pathlib import Path

import pandas as pd

ODDS_COLUMN_PRIORITY = [
    ("PSW", "PSL"),
    ("AvgW", "AvgL"),
    ("B365W", "B365L"),
]

MATCH_WINDOW_DAYS = 7
MIN_MATCH_CONFIDENCE = 0.80


def fair_probability_from_decimal_odds(
    player_odds: float,
    opponent_odds: float,
) -> tuple[float, float]:
    """Convert a decimal-odds pair into vig-free probabilities."""
    if player_odds <= 1 or opponent_odds <= 1:
        raise ValueError("Decimal odds must be greater than 1.")

    player_implied = 1 / player_odds
    opponent_implied = 1 / opponent_odds
    overround = player_implied + opponent_implied
    return player_implied / overround, opponent_implied / overround


def _normalize_name_key(name: str) -> str:
    """Lowercase and strip everything but letters, for cross-source name matching."""
    return re.sub(r"[^a-z]", "", str(name).lower())


def _normalize_tournament_key(name: str) -> str:
    """Normalize tournament labels while dropping source-specific boilerplate."""
    tokens = _name_tokens(name)
    ignored = {"atp", "wta", "open", "championships", "tennis"}
    return "".join(token for token in tokens if token not in ignored)


def _split_odds_player(name: str) -> tuple[str, str]:
    """Split a 'Surname Initial.' odds-file name into (surname, initial)."""
    tokens = str(name).strip().split()
    if len(tokens) < 2:
        return str(name), ""
    initial = tokens[-1].rstrip(".")
    surname = " ".join(tokens[:-1])
    return surname, initial


def _name_tokens(name: str) -> list[str]:
    """Tokenize a name into normalized (lowercase, alpha-only) parts, splitting on hyphens too."""
    raw = str(name).replace("-", " ").strip().split()
    return [token for t in raw if (token := _normalize_name_key(t))]


def _contains_subsequence(haystack: list[str], needle: list[str]) -> bool:
    """Check whether needle appears as a contiguous run inside haystack."""
    if not needle:
        return False
    n, m = len(haystack), len(needle)
    if m > n:
        return False
    return any(haystack[i:i + m] == needle for i in range(n - m + 1))


def _surname_matches(name_tokens: list[str], surname_tokens: list[str]) -> bool:
    """Check whether a (possibly compound) surname corresponds to a full name.

    Two directions are needed because sources disagree on which parts of a compound
    surname to keep: either the surname phrase appears intact somewhere in the full
    name (e.g. odds "Bautista" inside "Roberto Bautista Agut"), or the full name's
    last token is one part of a longer surname the other source recorded in full
    (e.g. our "Albert Ramos" vs odds "Ramos-Vinolas").
    """
    if not surname_tokens or not name_tokens:
        return False
    if _contains_subsequence(name_tokens, surname_tokens):
        return True
    last_token = name_tokens[-1]
    return len(last_token) >= 3 and last_token in surname_tokens


def _full_name_tokens_and_initial(name: str) -> tuple[list[str], str]:
    """Build (name tokens, first-name initial) for a full player name."""
    tokens = _name_tokens(name)
    initial = tokens[0][0] if tokens else ""
    return tokens, initial


def compute_fair_implied_probability(row: pd.Series) -> tuple[float | None, float | None]:
    """Compute vig-free implied win probabilities from the best-available odds pair."""
    for winner_col, loser_col in ODDS_COLUMN_PRIORITY:
        odds_w = row.get(winner_col)
        odds_l = row.get(loser_col)
        if pd.isna(odds_w) or pd.isna(odds_l) or odds_w <= 1 or odds_l <= 1:
            continue

        return fair_probability_from_decimal_odds(odds_w, odds_l)

    return None, None


def load_odds_files(odds_dir: str | Path = "data/raw/odds") -> pd.DataFrame:
    """Load and concatenate all tennis-data.co.uk-format odds Excel files."""
    odds_dir = Path(odds_dir)
    files = sorted(odds_dir.glob("*.xls*"))

    frames = [pd.read_excel(path) for path in files]
    odds = pd.concat(frames, ignore_index=True)
    odds["Date"] = pd.to_datetime(odds["Date"])

    winner_surname, winner_initial = zip(*odds["Winner"].map(_split_odds_player))
    loser_surname, loser_initial = zip(*odds["Loser"].map(_split_odds_player))

    odds = odds.assign(
        winner_surname_tokens=[_name_tokens(s) for s in winner_surname],
        winner_initial=[i.lower() for i in winner_initial],
        loser_surname_tokens=[_name_tokens(s) for s in loser_surname],
        loser_initial=[i.lower() for i in loser_initial],
    )

    fair_probs = odds.apply(compute_fair_implied_probability, axis=1, result_type="expand")
    odds[["fair_winner_prob", "fair_loser_prob"]] = fair_probs

    return odds


def _initial_matches(odds_initial: str, our_initial: str) -> bool:
    """Compare an odds-file initial (may be multi-letter, e.g. 'Y.H.') to our first initial."""
    normalized = _normalize_name_key(odds_initial)
    return bool(normalized) and normalized[0] == our_initial


def _find_odds_match(
    winner_name: str,
    loser_name: str,
    candidates: pd.DataFrame,
) -> pd.Series | None:
    """Find the odds row (if any) whose winner/loser surnames match the given full names.

    Tries a strict pass (surname subsequence + first-initial match) first, since that's
    fast and unambiguous. Some players are transliterated differently between sources
    (e.g. Sackmann's "Alexandr Dolgopolov" vs odds-file "Dolgopolov O." for Oleksandr),
    and some have compound surnames where one source drops part of it (e.g. "Roberto
    Bautista Agut" vs odds-file "Bautista R."), so a surname-only fallback is used when
    it uniquely identifies a single candidate.
    """
    winner_tokens, winner_first_initial = _full_name_tokens_and_initial(winner_name)
    loser_tokens, loser_first_initial = _full_name_tokens_and_initial(loser_name)

    strict = candidates[
        candidates["winner_initial"].map(lambda i: _initial_matches(i, winner_first_initial))
        & candidates["loser_initial"].map(lambda i: _initial_matches(i, loser_first_initial))
    ]
    for _, candidate in strict.iterrows():
        if _surname_matches(
            winner_tokens, candidate["winner_surname_tokens"]
        ) and _surname_matches(loser_tokens, candidate["loser_surname_tokens"]):
            return candidate

    surname_hits = [
        candidate
        for _, candidate in candidates.iterrows()
        if _surname_matches(winner_tokens, candidate["winner_surname_tokens"])
        and _surname_matches(loser_tokens, candidate["loser_surname_tokens"])
    ]
    if len(surname_hits) == 1:
        return surname_hits[0]

    return None


def _find_odds_match_with_confidence(
    winner_name: str,
    loser_name: str,
    candidates: pd.DataFrame,
) -> tuple[pd.Series | None, float, str]:
    """Return an unambiguous name match plus an auditable confidence score."""
    winner_tokens, winner_initial = _full_name_tokens_and_initial(winner_name)
    loser_tokens, loser_initial = _full_name_tokens_and_initial(loser_name)

    strict_hits = []
    surname_hits = []
    for _, candidate in candidates.iterrows():
        surnames_match = _surname_matches(
            winner_tokens, candidate["winner_surname_tokens"]
        ) and _surname_matches(loser_tokens, candidate["loser_surname_tokens"])
        if not surnames_match:
            continue
        surname_hits.append(candidate)
        if _initial_matches(candidate["winner_initial"], winner_initial) and (
            _initial_matches(candidate["loser_initial"], loser_initial)
        ):
            strict_hits.append(candidate)

    if len(strict_hits) == 1:
        return strict_hits[0], 0.90, "surname_and_initial"
    if len(surname_hits) == 1:
        # Retain the candidate for auditing, but keep it below the model-use
        # threshold because same-surname players can create false positives.
        return surname_hits[0], 0.55, "surname_only"
    if strict_hits or surname_hits:
        return None, 0.0, "ambiguous"
    return None, 0.0, "unmatched"


def match_odds_to_matches(
    raw_matches: pd.DataFrame,
    odds: pd.DataFrame,
    window_days: int = MATCH_WINDOW_DAYS,
) -> pd.DataFrame:
    """Attach leakage-free market-probability columns to raw ATP match data."""
    matches = raw_matches.copy()
    match_dates = pd.to_datetime(
        matches["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    matches = matches.assign(_match_date=match_dates)

    winner_prob = pd.Series(index=matches.index, dtype="float64")
    loser_prob = pd.Series(index=matches.index, dtype="float64")
    matched_flag = pd.Series(False, index=matches.index)
    match_confidence = pd.Series(0.0, index=matches.index, dtype="float64")
    match_method = pd.Series("unmatched", index=matches.index, dtype="object")

    odds_dates = odds["Date"]

    for _, group in matches.groupby("tourney_id", sort=False):
        tourney_date = group["_match_date"].iloc[0]
        if pd.isna(tourney_date):
            continue

        window_start = tourney_date
        window_end = tourney_date + pd.Timedelta(days=window_days)
        candidates = odds[(odds_dates >= window_start) & (odds_dates <= window_end)]

        if candidates.empty:
            continue

        for row_index, match_row in group.iterrows():
            match_candidates = candidates
            if "Tournament" in candidates.columns and pd.notna(
                match_row.get("tourney_name")
            ):
                tournament_key = _normalize_tournament_key(match_row["tourney_name"])
                if tournament_key:
                    tournament_mask = candidates["Tournament"].map(
                        _normalize_tournament_key
                    ).eq(tournament_key)
                    if tournament_mask.any():
                        match_candidates = candidates[tournament_mask]

            found, confidence, method = _find_odds_match_with_confidence(
                match_row["winner_name"], match_row["loser_name"], candidates
            )
            if match_candidates is not candidates:
                found, confidence, method = _find_odds_match_with_confidence(
                    match_row["winner_name"],
                    match_row["loser_name"],
                    match_candidates,
                )
                if found is not None:
                    confidence = min(1.0, confidence + 0.08)
                    method += "+tournament"
            match_confidence.at[row_index] = confidence
            match_method.at[row_index] = method
            if (
                found is not None
                and confidence >= MIN_MATCH_CONFIDENCE
                and pd.notna(found["fair_winner_prob"])
            ):
                winner_prob.at[row_index] = found["fair_winner_prob"]
                loser_prob.at[row_index] = found["fair_loser_prob"]
                matched_flag.at[row_index] = True

    matches["winner_market_prob"] = winner_prob.fillna(0.5)
    matches["loser_market_prob"] = loser_prob.fillna(0.5)
    matches["market_odds_available"] = matched_flag.astype("int8")
    matches["odds_match_confidence"] = match_confidence
    matches["odds_match_method"] = match_method

    return matches.drop(columns=["_match_date"])
