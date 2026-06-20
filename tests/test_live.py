"""Pre-match live pipeline tests."""

from __future__ import annotations

import pandas as pd
import pytest

from nlvswe.betting.slip import build_live_bet_slip
from nlvswe.betting.strategy import expected_value
from nlvswe.config import get_config, load_config
from nlvswe.features.build import build_features, features_for_match
from nlvswe.schemas import validate_table


def _tiny_matches():
    rows = [
        {
            "match_id": "m3",
            "date_utc": pd.Timestamp("2024-03-01", tz="UTC"),
            "corpus": "international",
            "competition": "Friendly",
            "season": "2024",
            "stage": None,
            "home_team_id": "netherlands",
            "away_team_id": "sweden",
            "venue_id": None,
            "neutral": False,
            "home_goals": 3,
            "away_goals": 0,
            "status": "completed",
            "went_to_shootout": False,
        },
        {
            "match_id": "m4",
            "date_utc": pd.Timestamp("2026-06-20T17:00:00", tz="UTC"),
            "corpus": "international",
            "competition": "FIFA World Cup",
            "season": "2026",
            "stage": "group",
            "home_team_id": "netherlands",
            "away_team_id": "sweden",
            "venue_id": None,
            "neutral": True,
            "home_goals": pd.NA,
            "away_goals": pd.NA,
            "status": "scheduled",
            "went_to_shootout": pd.NA,
        },
    ]
    df = pd.DataFrame(rows)
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True).astype("datetime64[ns, UTC]")
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    return df


def test_live_feature_parity_with_batch():
    matches = _tiny_matches()
    ratings = pd.DataFrame(
        columns=["team_id", "rating_date", "source", "corpus", "value", "rank"]
    )
    conditions = pd.DataFrame(columns=["match_id", "kickoff_local", "temp_c", "humidity", "weather", "altitude_m"])
    venues = pd.DataFrame(columns=["venue_id", "name", "city", "country_code", "lat", "lon", "altitude_m", "capacity"])
    batch = build_features(matches, ratings, conditions, venues, form_windows=[5, 10])
    target = matches[matches["match_id"] == "m4"].iloc[0]
    single = features_for_match(target, matches, ratings, conditions, venues, form_windows=[5, 10])
    batch_row = batch[batch["match_id"] == "m4"].iloc[0]
    for col in batch.columns:
        if col in {"result_1x2", "home_goals", "away_goals", "total_goals"}:
            continue
        bv, sv = batch_row[col], single[col]
        if pd.isna(bv) and pd.isna(sv):
            continue
        assert bv == sv, f"mismatch on {col}"


def test_bet_slip_schema_and_stake_cap():
    get_config.cache_clear()
    cfg = load_config()
    kickoff = pd.Timestamp("2026-06-20T17:00:00", tz="UTC")
    market_probs = pd.DataFrame(
        [
            {"market": "1x2", "selection": "home", "model_prob": 0.55, "method": "derived"},
            {"market": "1x2", "selection": "draw", "model_prob": 0.25, "method": "derived"},
            {"market": "1x2", "selection": "away", "model_prob": 0.20, "method": "derived"},
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "match_id": "m4",
                "bookmaker": "bet365",
                "market": "1x2",
                "selection": "home",
                "decimal_odds": 2.2,
                "captured_at": pd.Timestamp("2026-06-20T16:00:00", tz="UTC"),
                "is_closing": False,
                "corpus": "international",
            },
            {
                "match_id": "m4",
                "bookmaker": "bet365",
                "market": "1x2",
                "selection": "draw",
                "decimal_odds": 3.4,
                "captured_at": pd.Timestamp("2026-06-20T16:00:00", tz="UTC"),
                "is_closing": False,
                "corpus": "international",
            },
            {
                "match_id": "m4",
                "bookmaker": "bet365",
                "market": "1x2",
                "selection": "away",
                "decimal_odds": 3.5,
                "captured_at": pd.Timestamp("2026-06-20T16:00:00", tz="UTC"),
                "is_closing": False,
                "corpus": "international",
            },
        ]
    )
    slip = build_live_bet_slip(
        market_probs, odds, cfg, match_id="m4", model="poisson", kickoff=kickoff
    )
    slip = validate_table(slip, "bet_slip")
    bets = slip[slip["status"] == "bet"]
    if not bets.empty:
        assert (bets["stake"] <= cfg.betting.bankroll).all()
        for _, row in bets.iterrows():
            ev = expected_value(row["model_prob"], row["odds_taken"])
            assert ev == pytest.approx(row["ev"])
            assert ev > cfg.betting.min_edge


def test_no_edge_yields_no_bet_not_error():
    get_config.cache_clear()
    cfg = load_config()
    kickoff = pd.Timestamp("2026-06-20T17:00:00", tz="UTC")
    market_probs = pd.DataFrame(
        [{"market": "1x2", "selection": "home", "model_prob": 0.40, "method": "derived"}]
    )
    odds = pd.DataFrame(
        [
            {
                "match_id": "m4",
                "bookmaker": "bet365",
                "market": "1x2",
                "selection": "home",
                "decimal_odds": 2.0,
                "captured_at": pd.Timestamp("2026-06-20T16:00:00", tz="UTC"),
                "is_closing": False,
                "corpus": "international",
            },
        ]
    )
    slip = build_live_bet_slip(
        market_probs, odds, cfg, match_id="m4", model="poisson", kickoff=kickoff
    )
    assert len(slip) == 1
    assert slip.iloc[0]["status"] == "no_bet"
    assert slip.iloc[0]["stake"] == 0.0
