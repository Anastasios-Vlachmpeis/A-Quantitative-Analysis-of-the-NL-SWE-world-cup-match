"""Pre-kickoff Elo for home and away, shared with the Elo goal model.

Each team's rating uses only matches strictly before that kickoff. Post-match
updates follow the same rules as nlvswe.ratings.elo.
"""

from __future__ import annotations

import pandas as pd

from nlvswe.ratings.elo import (
    EloParams,
    expected_score,
    margin_multiplier,
)

CORPUS_INTERNATIONAL = "international"


def _completed_intl(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches[(matches["status"] == "completed") & (matches["corpus"] == CORPUS_INTERNATIONAL)].copy()
    return m.sort_values(["date_utc", "match_id"], kind="mergesort")


def compute_elo_pre_table(
    matches: pd.DataFrame,
    *,
    params: EloParams | None = None,
) -> pd.DataFrame:
    """Return match_id, home_elo_pre, away_elo_pre for every completed intl match."""
    params = params or EloParams()
    ratings: dict[str, float] = {}
    rows: list[dict] = []

    for _, m in _completed_intl(matches).iterrows():
        home_id = m["home_team_id"]
        away_id = m["away_team_id"]
        rh = ratings.get(home_id, params.initial)
        ra = ratings.get(away_id, params.initial)
        rows.append(
            {
                "match_id": m["match_id"],
                "home_elo_pre": float(rh),
                "away_elo_pre": float(ra),
            }
        )
        _apply_elo_update(m, ratings, params)

    return pd.DataFrame(rows)


def elo_pre_at_kickoff(
    match_row: pd.Series,
    history: pd.DataFrame,
    *,
    params: EloParams | None = None,
) -> tuple[float, float]:
    """Pre-kickoff Elo for home/away using only history with date_utc < kickoff."""
    params = params or EloParams()
    kickoff = pd.Timestamp(match_row["date_utc"]).tz_convert("UTC")
    prior = history[
        (history["status"] == "completed")
        & (history["corpus"] == CORPUS_INTERNATIONAL)
        & (history["date_utc"] < kickoff)
    ].sort_values(["date_utc", "match_id"], kind="mergesort")

    ratings: dict[str, float] = {}
    for _, m in prior.iterrows():
        _apply_elo_update(m, ratings, params)

    home_id = match_row["home_team_id"]
    away_id = match_row["away_team_id"]
    return (
        ratings.get(home_id, params.initial),
        ratings.get(away_id, params.initial),
    )


def _apply_elo_update(m: pd.Series, ratings: dict[str, float], params: EloParams) -> None:
    home_id = m["home_team_id"]
    away_id = m["away_team_id"]
    home_goals = int(m["home_goals"])
    away_goals = int(m["away_goals"])
    neutral = bool(m["neutral"])

    rh = ratings.setdefault(home_id, params.initial)
    ra = ratings.setdefault(away_id, params.initial)
    home_eff = rh + (0.0 if neutral else params.home_advantage)

    if home_goals > away_goals:
        score_home, score_away = 1.0, 0.0
    elif home_goals < away_goals:
        score_home, score_away = 0.0, 1.0
    else:
        score_home = score_away = 0.5

    exp_home = expected_score(home_eff, ra)
    mult = margin_multiplier(abs(home_goals - away_goals), params)
    k = params.k * mult

    ratings[home_id] = rh + k * (score_home - exp_home)
    ratings[away_id] = ra + k * (score_away - (1.0 - exp_home))
