"""Pandera schemas for canonical and feature tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandera as pa
from pandera import Check

if TYPE_CHECKING:
    import pandas as pd

SCHEMA_REGISTRY: dict[str, pa.DataFrameSchema] = {}

CORPUS_CHECK = Check.isin(["international", "club"])

TEAMS_SCHEMA = pa.DataFrameSchema(
    {
        "team_id": pa.Column(str, unique=True, nullable=False),
        "canonical_name": pa.Column(str, nullable=False),
        "country_code": pa.Column(str, nullable=True, required=False),
        "confederation": pa.Column(str, nullable=True, required=False),
        "aliases": pa.Column(object, nullable=False),
    },
    strict=True,
    name="teams",
)

VENUES_SCHEMA = pa.DataFrameSchema(
    {
        "venue_id": pa.Column(str, unique=True, nullable=False),
        "name": pa.Column(str, nullable=False),
        "city": pa.Column(str, nullable=True, required=False),
        "country_code": pa.Column(str, nullable=True, required=False),
        "lat": pa.Column(float, nullable=True, required=False),
        "lon": pa.Column(float, nullable=True, required=False),
        "altitude_m": pa.Column(float, nullable=True, required=False),
        "capacity": pa.Column("Int64", nullable=True, required=False),
    },
    strict=True,
    name="venues",
)

MATCHES_SCHEMA = pa.DataFrameSchema(
    {
        "match_id": pa.Column(str, unique=True, nullable=False),
        "date_utc": pa.Column("datetime64[ns, UTC]", nullable=False),
        "corpus": pa.Column(str, CORPUS_CHECK, nullable=False),
        "competition": pa.Column(str, nullable=False),
        "season": pa.Column(str, nullable=True, required=False),
        "stage": pa.Column(str, nullable=True, required=False),
        "home_team_id": pa.Column(str, nullable=False),
        "away_team_id": pa.Column(str, nullable=False),
        "venue_id": pa.Column(str, nullable=True, required=False),
        "neutral": pa.Column(bool, nullable=False),
        "home_goals": pa.Column("Int64", Check.ge(0), nullable=True, required=False),
        "away_goals": pa.Column("Int64", Check.ge(0), nullable=True, required=False),
        "status": pa.Column(str, Check.isin(["completed", "scheduled"]), nullable=False),
        "went_to_shootout": pa.Column("boolean", nullable=True, required=False),
    },
    strict=True,
    name="matches",
)

RATINGS_SCHEMA = pa.DataFrameSchema(
    {
        "team_id": pa.Column(str, nullable=False),
        "rating_date": pa.Column("datetime64[ns, UTC]", nullable=False),
        "source": pa.Column(str, Check.isin(["fifa", "elo"]), nullable=False),
        "corpus": pa.Column(str, CORPUS_CHECK, nullable=False),
        "value": pa.Column(float, nullable=False),
        "rank": pa.Column("Int64", nullable=True, required=False),
    },
    strict=True,
    name="ratings",
)

TEAM_MATCH_STATS_SCHEMA = pa.DataFrameSchema(
    {
        "match_id": pa.Column(str, nullable=False),
        "team_id": pa.Column(str, nullable=False),
        "xg": pa.Column(float, nullable=True, required=False),
        "shots": pa.Column("Int64", nullable=True, required=False),
        "possession": pa.Column(float, nullable=True, required=False),
    },
    strict=True,
    name="team_match_stats",
)

ODDS_SCHEMA = pa.DataFrameSchema(
    {
        "match_id": pa.Column(str, nullable=False),
        "bookmaker": pa.Column(str, nullable=False),
        "market": pa.Column(str, Check.isin(["1x2", "totals", "ah"]), nullable=False),
        "corpus": pa.Column(str, CORPUS_CHECK, nullable=False),
        "selection": pa.Column(str, nullable=False),
        "decimal_odds": pa.Column(float, Check.gt(1.0), nullable=False),
        "captured_at": pa.Column("datetime64[ns, UTC]", nullable=False),
        "is_closing": pa.Column(bool, nullable=False),
    },
    strict=True,
    name="odds",
)

CONDITIONS_SCHEMA = pa.DataFrameSchema(
    {
        "match_id": pa.Column(str, nullable=False),
        "kickoff_local": pa.Column("datetime64[ns, UTC]", nullable=True, required=False),
        "temp_c": pa.Column(float, nullable=True, required=False),
        "humidity": pa.Column(float, nullable=True, required=False),
        "weather": pa.Column(str, nullable=True, required=False),
        "altitude_m": pa.Column(float, nullable=True, required=False),
    },
    strict=True,
    name="conditions",
)

FEATURES_SCHEMA = pa.DataFrameSchema(
    {
        "match_id": pa.Column(str, unique=True, nullable=False),
        "date_utc": pa.Column("datetime64[ns, UTC]", nullable=False),
        "corpus": pa.Column(str, CORPUS_CHECK, nullable=False),
        "home_team_id": pa.Column(str, nullable=False),
        "away_team_id": pa.Column(str, nullable=False),
        "neutral": pa.Column(bool, nullable=False),
        "is_home": pa.Column(bool, nullable=False),
        "is_knockout": pa.Column(bool, nullable=False),
        "home_elo_pre": pa.Column(float, nullable=False),
        "away_elo_pre": pa.Column(float, nullable=False),
        "elo_diff": pa.Column(float, nullable=False),
        "home_fifa_points_pre": pa.Column(float, nullable=True, required=False),
        "away_fifa_points_pre": pa.Column(float, nullable=True, required=False),
        "home_fifa_rank_pre": pa.Column("Int64", nullable=True, required=False),
        "away_fifa_rank_pre": pa.Column("Int64", nullable=True, required=False),
        "fifa_points_diff": pa.Column(float, nullable=True, required=False),
        "fifa_rank_diff": pa.Column(float, nullable=True, required=False),
        "fifa_available": pa.Column(bool, nullable=False),
        "home_form_ppg_5": pa.Column(float, nullable=True, required=False),
        "away_form_ppg_5": pa.Column(float, nullable=True, required=False),
        "form_ppg_diff_5": pa.Column(float, nullable=True, required=False),
        "home_form_gd_5": pa.Column(float, nullable=True, required=False),
        "away_form_gd_5": pa.Column(float, nullable=True, required=False),
        "form_gd_diff_5": pa.Column(float, nullable=True, required=False),
        "form_5_available": pa.Column(bool, nullable=False),
        "home_form_ppg_10": pa.Column(float, nullable=True, required=False),
        "away_form_ppg_10": pa.Column(float, nullable=True, required=False),
        "form_ppg_diff_10": pa.Column(float, nullable=True, required=False),
        "home_form_gd_10": pa.Column(float, nullable=True, required=False),
        "away_form_gd_10": pa.Column(float, nullable=True, required=False),
        "form_gd_diff_10": pa.Column(float, nullable=True, required=False),
        "form_10_available": pa.Column(bool, nullable=False),
        "home_days_since_last": pa.Column(float, nullable=True, required=False),
        "away_days_since_last": pa.Column(float, nullable=True, required=False),
        "rest_diff": pa.Column(float, nullable=True, required=False),
        "home_matches_30d": pa.Column("Int64", nullable=False),
        "away_matches_30d": pa.Column("Int64", nullable=False),
        "congestion_diff": pa.Column("Int64", nullable=False),
        "altitude_m": pa.Column(float, nullable=True, required=False),
        "temp_c": pa.Column(float, nullable=True, required=False),
        "humidity": pa.Column(float, nullable=True, required=False),
        "heat_stress": pa.Column(float, nullable=True, required=False),
        "home_travel_km": pa.Column(float, nullable=True, required=False),
        "away_travel_km": pa.Column(float, nullable=True, required=False),
        "travel_diff_km": pa.Column(float, nullable=True, required=False),
        "conditions_available": pa.Column(bool, nullable=False),
        "result_1x2": pa.Column(str, Check.isin(["home", "draw", "away"]), nullable=True, required=False),
        "home_goals": pa.Column("Int64", Check.ge(0), nullable=True, required=False),
        "away_goals": pa.Column("Int64", Check.ge(0), nullable=True, required=False),
        "total_goals": pa.Column("Int64", Check.ge(0), nullable=True, required=False),
    },
    strict=True,
    name="features",
)

PREDICTIONS_SCHEMA = pa.DataFrameSchema(
    {
        "match_id": pa.Column(str, nullable=False),
        "model": pa.Column(str, nullable=False),
        "date_utc": pa.Column("datetime64[ns, UTC]", nullable=False),
        "p_home": pa.Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "p_draw": pa.Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "p_away": pa.Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "outcome": pa.Column(str, Check.isin(["home", "draw", "away"]), nullable=False),
        "has_scoreline": pa.Column(bool, nullable=False),
        "scoreline_flat": pa.Column(object, nullable=True, required=False),
    },
    strict=True,
    name="predictions",
)

MARKET_PROBS_SCHEMA = pa.DataFrameSchema(
    {
        "match_id": pa.Column(str, nullable=False),
        "model": pa.Column(str, nullable=False),
        "market": pa.Column(str, nullable=False),
        "selection": pa.Column(str, nullable=False),
        "model_prob": pa.Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "method": pa.Column(str, Check.isin(["analytic", "mc", "derived"]), nullable=False),
    },
    strict=True,
    name="market_probs",
)

EVAL_SCORES_SCHEMA = pa.DataFrameSchema(
    {
        "model": pa.Column(str, nullable=False),
        "metric": pa.Column(str, Check.isin(["rps", "log_loss", "brier"]), nullable=False),
        "mean": pa.Column(float, nullable=False),
        "ci_low": pa.Column(float, nullable=False),
        "ci_high": pa.Column(float, nullable=False),
        "n": pa.Column("Int64", Check.ge(1), nullable=False),
    },
    strict=True,
    name="eval_scores",
)

LEADERBOARD_SCHEMA = pa.DataFrameSchema(
    {
        "model": pa.Column(str, nullable=False),
        "metric": pa.Column(str, Check.isin(["rps", "log_loss", "brier"]), nullable=False),
        "mean": pa.Column(float, nullable=False),
        "ci_low": pa.Column(float, nullable=False),
        "ci_high": pa.Column(float, nullable=False),
        "n": pa.Column("Int64", Check.ge(1), nullable=False),
        "accuracy": pa.Column(float, Check.in_range(0.0, 1.0), nullable=True, required=False),
        "rank": pa.Column("Int64", Check.ge(1), nullable=True, required=False),
    },
    strict=True,
    name="leaderboard",
)

PAIRWISE_SCHEMA = pa.DataFrameSchema(
    {
        "model": pa.Column(str, nullable=False),
        "reference": pa.Column(str, nullable=False),
        "metric": pa.Column(str, Check.isin(["rps", "log_loss", "brier"]), nullable=False),
        "mean_diff": pa.Column(float, nullable=False),
        "ci_low": pa.Column(float, nullable=False),
        "ci_high": pa.Column(float, nullable=False),
        "n": pa.Column("Int64", Check.ge(1), nullable=False),
        "beats_reference": pa.Column(bool, nullable=False),
    },
    strict=True,
    name="pairwise_vs_market",
)

BETS_SCHEMA = pa.DataFrameSchema(
    {
        "match_id": pa.Column(str, nullable=False),
        "date_utc": pa.Column("datetime64[ns, UTC]", nullable=False),
        "model": pa.Column(str, nullable=False),
        "corpus": pa.Column(str, CORPUS_CHECK, nullable=False),
        "bookmaker": pa.Column(str, nullable=False),
        "market": pa.Column(str, nullable=False),
        "selection": pa.Column(str, nullable=False),
        "model_prob": pa.Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "book_prob_devig": pa.Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "edge": pa.Column(float, nullable=False),
        "ev": pa.Column(float, nullable=False),
        "odds_taken": pa.Column(float, Check.gt(1.0), nullable=False),
        "odds_closing": pa.Column(float, Check.gt(1.0), nullable=True, required=False),
        "clv_odds_ratio": pa.Column(float, nullable=True, required=False),
        "clv_prob_diff": pa.Column(float, nullable=True, required=False),
        "stake_method": pa.Column(str, Check.isin(["kelly", "flat"]), nullable=False),
        "stake": pa.Column(float, Check.ge(0.0), nullable=False),
        "result": pa.Column(
            str, Check.isin(["win", "lose", "push", "half_win", "half_lose"]), nullable=False
        ),
        "pnl": pa.Column(float, nullable=False),
    },
    strict=True,
    name="bets",
)

BANKROLL_SIM_SCHEMA = pa.DataFrameSchema(
    {
        "model": pa.Column(str, nullable=False),
        "path_type": pa.Column(str, Check.isin(["realized", "mc_terminal"]), nullable=False),
        "step": pa.Column("Int64", Check.ge(0), nullable=False),
        "date_utc": pa.Column("datetime64[ns, UTC]", nullable=True, required=False),
        "bankroll": pa.Column(float, nullable=False),
        "drawdown": pa.Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "mc_run": pa.Column("Int64", nullable=True, required=False),
    },
    strict=True,
    name="bankroll_sim",
)

LIVE_PREDICTIONS_SCHEMA = pa.DataFrameSchema(
    {
        "match_id": pa.Column(str, nullable=False),
        "model": pa.Column(str, nullable=False),
        "date_utc": pa.Column("datetime64[ns, UTC]", nullable=False),
        "p_home": pa.Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "p_draw": pa.Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "p_away": pa.Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "has_scoreline": pa.Column(bool, nullable=False),
        "scoreline_flat": pa.Column(object, nullable=True, required=False),
    },
    strict=True,
    name="live_predictions",
)

BET_SLIP_SCHEMA = pa.DataFrameSchema(
    {
        "match_id": pa.Column(str, nullable=False),
        "model": pa.Column(str, nullable=False),
        "captured_at": pa.Column("datetime64[ns, UTC]", nullable=False),
        "kickoff_utc": pa.Column("datetime64[ns, UTC]", nullable=False),
        "bookmaker": pa.Column(str, nullable=True, required=False),
        "market": pa.Column(str, nullable=True, required=False),
        "selection": pa.Column(str, nullable=True, required=False),
        "model_prob": pa.Column(float, Check.in_range(0.0, 1.0), nullable=True, required=False),
        "book_prob_devig": pa.Column(float, Check.in_range(0.0, 1.0), nullable=True, required=False),
        "edge": pa.Column(float, nullable=True, required=False),
        "ev": pa.Column(float, nullable=True, required=False),
        "odds_taken": pa.Column(float, nullable=True, required=False),
        "stake_method": pa.Column(str, Check.isin(["kelly", "flat"]), nullable=False),
        "stake": pa.Column(float, Check.ge(0.0), nullable=False),
        "status": pa.Column(str, Check.isin(["bet", "no_bet"]), nullable=False),
        "rationale": pa.Column(str, nullable=False),
    },
    strict=True,
    name="bet_slip",
)

CANONICAL_SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "teams": TEAMS_SCHEMA,
    "venues": VENUES_SCHEMA,
    "matches": MATCHES_SCHEMA,
    "ratings": RATINGS_SCHEMA,
    "team_match_stats": TEAM_MATCH_STATS_SCHEMA,
    "odds": ODDS_SCHEMA,
    "conditions": CONDITIONS_SCHEMA,
    "features": FEATURES_SCHEMA,
    "predictions": PREDICTIONS_SCHEMA,
    "market_probs": MARKET_PROBS_SCHEMA,
    "eval_scores": EVAL_SCORES_SCHEMA,
    "leaderboard": LEADERBOARD_SCHEMA,
    "pairwise_vs_market": PAIRWISE_SCHEMA,
    "bets": BETS_SCHEMA,
    "bankroll_sim": BANKROLL_SIM_SCHEMA,
    "live_predictions": LIVE_PREDICTIONS_SCHEMA,
    "bet_slip": BET_SLIP_SCHEMA,
}

for _name, _schema in CANONICAL_SCHEMAS.items():
    SCHEMA_REGISTRY[_name] = _schema


def register_schema(name: str, schema: pa.DataFrameSchema) -> None:
    """Register a named schema for validation."""
    SCHEMA_REGISTRY[name] = schema


def validate_table(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
    """Validate df against a registered schema; raise if unknown or invalid."""
    if schema_name not in SCHEMA_REGISTRY:
        raise KeyError(f"Unknown schema: {schema_name!r}. Registered: {list(SCHEMA_REGISTRY)}")
    return SCHEMA_REGISTRY[schema_name].validate(df)


def validate_business_rules(
    matches: pd.DataFrame,
    odds: pd.DataFrame,
    ratings: pd.DataFrame | None = None,
    *,
    expected_intl_match_count: int | None = None,
) -> list[str]:
    """Return list of business-rule violations (empty = pass)."""
    import re

    issues: list[str] = []
    valid_corpus = {"international", "club"}
    club_comp_re = re.compile(r"^[A-Z0-9]+_\d{4}$")

    for table_name, df in (
        ("matches", matches),
        ("odds", odds),
        ("ratings", ratings if ratings is not None else pd.DataFrame()),
    ):
        if df.empty:
            continue
        if "corpus" not in df.columns:
            issues.append(f"{table_name}: missing corpus column")
            continue
        if df["corpus"].isna().any():
            issues.append(f"{table_name}: corpus has null values")
        invalid = set(df["corpus"].unique()) - valid_corpus
        if invalid:
            issues.append(f"{table_name}: invalid corpus values {invalid}")

    if not matches.empty and "corpus" in matches.columns:
        intl = matches[matches["corpus"] == "international"]
        club_in_intl = intl["competition"].astype(str).str.match(club_comp_re, na=False)
        if club_in_intl.any():
            issues.append(
                f"matches: {int(club_in_intl.sum())} international corpus rows with club competition codes"
            )
        if expected_intl_match_count is not None:
            intl_completed = intl[intl["status"] == "completed"]
            if len(intl_completed) != expected_intl_match_count:
                issues.append(
                    f"matches: international completed count {len(intl_completed)} "
                    f"!= raw results source {expected_intl_match_count}"
                )

        self_play = matches[matches["home_team_id"] == matches["away_team_id"]]
        if len(self_play):
            issues.append(f"matches: {len(self_play)} rows with home_team_id == away_team_id")

        dupes = matches["match_id"].duplicated()
        if dupes.any():
            issues.append(f"matches: {dupes.sum()} duplicate match_id values")

        completed = matches[matches["status"] == "completed"]
        bad_scores = completed[
            completed["home_goals"].isna()
            | completed["away_goals"].isna()
            | (completed["home_goals"] < 0)
            | (completed["away_goals"] < 0)
        ]
        if len(bad_scores):
            issues.append(f"matches: {len(bad_scores)} completed matches with invalid scores")

    if not odds.empty:
        bad_odds = odds[odds["decimal_odds"] <= 1.0]
        if len(bad_odds):
            issues.append(f"odds: {len(bad_odds)} rows with decimal_odds <= 1.0")

        one_x_two = odds[odds["market"] == "1x2"]
        if not one_x_two.empty:
            for (mid, bk), grp in one_x_two.groupby(["match_id", "bookmaker"]):
                closing = grp[grp["is_closing"]]
                if closing.empty:
                    continue
                implied = (1.0 / closing["decimal_odds"]).sum()
                if implied < 0.95 or implied > 1.25:
                    issues.append(
                        f"odds: 1x2 overround {implied:.3f} out of band for match={mid} book={bk}"
                    )
                    break

    return issues
