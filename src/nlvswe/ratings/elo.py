"""Chronological Elo over international results (used when cleaning and featurizing).

Defaults: 1500 for unseen teams, K=20, +100 home advantage in expectation,
ln-margin multiplier on updates. Each row is dated at kickoff so you can't
accidentally leak the score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

INITIAL_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 100.0
GOAL_MARGIN_MULTIPLIER = 2.0 / 3.0


@dataclass(frozen=True)
class EloParams:
    initial: float = INITIAL_ELO
    k: float = K_FACTOR
    home_advantage: float = HOME_ADVANTAGE
    goal_margin_multiplier: float = GOAL_MARGIN_MULTIPLIER


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def margin_multiplier(goal_diff: int, params: EloParams) -> float:
    if goal_diff <= 1:
        return 1.0
    return GOAL_MARGIN_MULTIPLIER * math.log(goal_diff + 1)


def compute_rolling_elo(
    matches: pd.DataFrame,
    *,
    params: EloParams | None = None,
) -> pd.DataFrame:
    """Return long-format ratings dataframe (source='elo') from completed matches."""
    params = params or EloParams()
    required = {"date_utc", "home_team_id", "away_team_id", "home_goals", "away_goals", "neutral", "status"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches missing columns for Elo: {missing}")

    hist = matches[matches["status"] == "completed"].copy()
    hist = hist.sort_values(["date_utc", "match_id"], kind="mergesort")

    ratings: dict[str, float] = {}
    rows: list[dict] = []

    def get_rating(team_id: str) -> float:
        return ratings.setdefault(team_id, params.initial)

    for _, m in hist.iterrows():
        home_id = m["home_team_id"]
        away_id = m["away_team_id"]
        home_goals = int(m["home_goals"])
        away_goals = int(m["away_goals"])
        neutral = bool(m["neutral"])

        rh = get_rating(home_id)
        ra = get_rating(away_id)
        home_eff = rh + (0.0 if neutral else params.home_advantage)

        if home_goals > away_goals:
            score_home, score_away = 1.0, 0.0
        elif home_goals < away_goals:
            score_home, score_away = 0.0, 1.0
        else:
            score_home = score_away = 0.5

        exp_home = expected_score(home_eff, ra)
        exp_away = 1.0 - exp_home
        mult = margin_multiplier(abs(home_goals - away_goals), params)
        k = params.k * mult

        rh_new = rh + k * (score_home - exp_home)
        ra_new = ra + k * (score_away - exp_away)
        ratings[home_id] = rh_new
        ratings[away_id] = ra_new

        rating_date = pd.Timestamp(m["date_utc"])
        if rating_date.tzinfo is None:
            rating_date = rating_date.tz_localize("UTC")
        else:
            rating_date = rating_date.tz_convert("UTC")

        rows.append(
            {
                "team_id": home_id,
                "rating_date": rating_date,
                "source": "elo",
                "value": float(rh_new),
                "rank": None,
            }
        )
        rows.append(
            {
                "team_id": away_id,
                "rating_date": rating_date,
                "source": "elo",
                "value": float(ra_new),
                "rank": None,
            }
        )

    return pd.DataFrame(rows)
