"""Tests that features never peek past kickoff."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nlvswe.features.build import build_features, features_for_match
from nlvswe.features.ratings import elo_pre_at_kickoff
from nlvswe.schemas import validate_table

CORPUS = "international"


def _tiny_matches() -> pd.DataFrame:
    """Three completed intl matches + one target scheduled."""
    rows = [
        {
            "match_id": "m1",
            "date_utc": pd.Timestamp("2024-01-01", tz="UTC"),
            "corpus": CORPUS,
            "competition": "Friendly",
            "season": "2024",
            "stage": None,
            "home_team_id": "netherlands",
            "away_team_id": "belgium",
            "venue_id": None,
            "neutral": False,
            "home_goals": 2,
            "away_goals": 1,
            "status": "completed",
            "went_to_shootout": False,
        },
        {
            "match_id": "m2",
            "date_utc": pd.Timestamp("2024-02-01", tz="UTC"),
            "corpus": CORPUS,
            "competition": "Friendly",
            "season": "2024",
            "stage": None,
            "home_team_id": "sweden",
            "away_team_id": "netherlands",
            "venue_id": None,
            "neutral": True,
            "home_goals": 1,
            "away_goals": 1,
            "status": "completed",
            "went_to_shootout": False,
        },
        {
            "match_id": "m3",
            "date_utc": pd.Timestamp("2024-03-01", tz="UTC"),
            "corpus": CORPUS,
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
            "date_utc": pd.Timestamp("2026-06-20", tz="UTC"),
            "corpus": CORPUS,
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
    df["went_to_shootout"] = df["went_to_shootout"].astype("boolean")
    return df


def _tiny_ratings() -> pd.DataFrame:
    rows = [
        {
            "team_id": "netherlands",
            "rating_date": pd.Timestamp("2023-12-01", tz="UTC"),
            "source": "fifa",
            "corpus": CORPUS,
            "value": 1800.0,
            "rank": 5,
        },
        {
            "team_id": "sweden",
            "rating_date": pd.Timestamp("2023-12-01", tz="UTC"),
            "source": "fifa",
            "corpus": CORPUS,
            "value": 1700.0,
            "rank": 20,
        },
    ]
    df = pd.DataFrame(rows)
    df["rating_date"] = pd.to_datetime(df["rating_date"], utc=True).astype("datetime64[ns, UTC]")
    df["rank"] = df["rank"].astype("Int64")
    return df


@pytest.fixture
def tiny_tables():
    matches = _tiny_matches()
    ratings = _tiny_ratings()
    conditions = pd.DataFrame(columns=["match_id", "kickoff_local", "temp_c", "humidity", "weather", "altitude_m"])
    venues = pd.DataFrame(columns=["venue_id", "name", "city", "country_code", "lat", "lon", "altitude_m", "capacity"])
    return matches, ratings, conditions, venues


def test_leakage_score_flip_unchanged_pre_match_features(tiny_tables):
    matches, ratings, conditions, venues = tiny_tables
    target = matches[matches["match_id"] == "m3"].iloc[0]
    feat_before = features_for_match(target, matches, ratings, conditions, venues, form_windows=[5, 10])

    altered = matches.copy()
    altered.loc[altered["match_id"] == "m3", "home_goals"] = 0
    altered.loc[altered["match_id"] == "m3", "away_goals"] = 3
    feat_after = features_for_match(target, altered, ratings, conditions, venues, form_windows=[5, 10])

    pre_cols = [
        "home_elo_pre",
        "away_elo_pre",
        "elo_diff",
        "home_fifa_points_pre",
        "form_ppg_diff_5",
        "home_days_since_last",
    ]
    for col in pre_cols:
        assert feat_before[col] == feat_after[col], f"{col} changed after score flip (leakage)"


def test_fifa_pre_strictly_before_kickoff(tiny_tables):
    matches, ratings, conditions, venues = tiny_tables
    target = matches[matches["match_id"] == "m2"].iloc[0]
    feat = features_for_match(target, matches, ratings, conditions, venues, form_windows=[5, 10])
    assert feat["home_fifa_points_pre"] == 1700.0
    assert feat["away_fifa_points_pre"] == 1800.0


def test_features_for_match_equals_batch_row(tiny_tables):
    matches, ratings, conditions, venues = tiny_tables
    batch = build_features(matches, ratings, conditions, venues, form_windows=[5, 10])
    target = matches[matches["match_id"] == "m4"].iloc[0]
    single = features_for_match(target, matches, ratings, conditions, venues, form_windows=[5, 10])
    batch_row = batch[batch["match_id"] == "m4"].iloc[0]
    for col in batch.columns:
        bv, sv = batch_row[col], single[col]
        if pd.isna(bv) and pd.isna(sv):
            continue
        assert bv == sv, f"mismatch on {col}: batch={bv} single={sv}"


def test_non_nullable_features_present(tiny_tables):
    matches, ratings, conditions, venues = tiny_tables
    df = build_features(matches, ratings, conditions, venues, form_windows=[5, 10])
    for col in ("home_elo_pre", "away_elo_pre", "elo_diff", "neutral", "is_home"):
        assert df[col].notna().all()
    scheduled = df[df["match_id"] == "m4"].iloc[0]
    assert scheduled["is_home"] is False or scheduled["is_home"] == 0
    assert bool(scheduled["neutral"]) is True


@pytest.mark.skipif(
    not Path("data/processed/features.parquet").exists(),
    reason="requires feature build",
)
def test_production_features_pass_schema():
    df = pd.read_parquet("data/processed/features.parquet")
    validate_table(df, "features")
