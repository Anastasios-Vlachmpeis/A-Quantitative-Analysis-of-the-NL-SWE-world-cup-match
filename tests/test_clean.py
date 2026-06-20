"""Interim table cleaning tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaError

from nlvswe.data.clean import CORPUS_CLUB, CORPUS_INTERNATIONAL, flag_is_closing, make_match_id
from nlvswe.data.entities import EntityResolver, UnresolvedEntityError
from nlvswe.schemas import ODDS_SCHEMA, validate_table


@pytest.fixture
def resolver():
    return EntityResolver.from_config()


def test_alias_resolver_known(resolver):
    assert resolver.resolve_team("Holland") == "netherlands"
    assert resolver.resolve_team("Netherlands") == "netherlands"
    assert resolver.resolve_team("Sweden") == "sweden"
    assert resolver.resolve_team("Czechia") == "czech_republic"


def test_alias_unknown_strict_raises(resolver):
    with pytest.raises(UnresolvedEntityError, match="XYZZY"):
        resolver.resolve_team("XYZZY_UNKNOWN_TEAM", strict=True)


def test_match_id_deterministic():
    ts = pd.Timestamp("2026-06-15", tz="UTC")
    a = make_match_id(ts, "netherlands", "sweden", "FIFA World Cup")
    b = make_match_id(ts, "netherlands", "sweden", "FIFA World Cup")
    assert a == b
    assert len(a) == 16


def test_is_closing_logic():
    kickoff = pd.Timestamp("2026-06-15T18:00:00", tz="UTC")
    match_id = "m1"
    odds = pd.DataFrame(
        [
            {
                "match_id": match_id,
                "bookmaker": "pinnacle",
                "market": "1x2",
                "selection": "home",
                "decimal_odds": 2.0,
                "captured_at": pd.Timestamp("2026-06-15T10:00:00", tz="UTC"),
                "is_closing": False,
                "corpus": CORPUS_INTERNATIONAL,
            },
            {
                "match_id": match_id,
                "bookmaker": "pinnacle",
                "market": "1x2",
                "selection": "home",
                "decimal_odds": 1.95,
                "captured_at": pd.Timestamp("2026-06-15T16:00:00", tz="UTC"),
                "is_closing": False,
                "corpus": CORPUS_INTERNATIONAL,
            },
            {
                "match_id": match_id,
                "bookmaker": "pinnacle",
                "market": "1x2",
                "selection": "home",
                "decimal_odds": 1.90,
                "captured_at": pd.Timestamp("2026-06-15T19:00:00", tz="UTC"),
                "is_closing": False,
                "corpus": CORPUS_INTERNATIONAL,
            },
        ]
    )
    matches = pd.DataFrame({"match_id": [match_id], "date_utc": [kickoff]})
    out = flag_is_closing(odds, matches)
    closing = out[out["is_closing"]]
    assert len(closing) == 1
    assert closing.iloc[0]["decimal_odds"] == 1.95
    assert not out[out["captured_at"] > kickoff]["is_closing"].any()


def test_pandera_rejects_bad_odds():
    bad = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "bookmaker": "bk",
                "market": "1x2",
                "corpus": CORPUS_CLUB,
                "selection": "home",
                "decimal_odds": 0.5,
                "captured_at": pd.Timestamp("2026-01-01", tz="UTC"),
                "is_closing": True,
            }
        ]
    )
    with pytest.raises(SchemaError):
        ODDS_SCHEMA.validate(bad)


def test_pandera_rejects_negative_goals():
    from nlvswe.schemas import MATCHES_SCHEMA

    bad = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date_utc": pd.Timestamp("2026-01-01", tz="UTC"),
                "corpus": CORPUS_INTERNATIONAL,
                "competition": "Friendly",
                "season": "2026",
                "stage": None,
                "home_team_id": "a",
                "away_team_id": "b",
                "venue_id": None,
                "neutral": False,
                "home_goals": -1,
                "away_goals": 0,
                "status": "completed",
                "went_to_shootout": False,
            }
        ]
    )
    with pytest.raises(SchemaError):
        MATCHES_SCHEMA.validate(bad)


@pytest.mark.skipif(
    not Path("data/interim/matches.parquet").exists(),
    reason="requires clean run",
)
def test_target_match_in_canonical():
    matches = pd.read_parquet("data/interim/matches.parquet")
    target = matches[
        (
            (matches["home_team_id"] == "netherlands") & (matches["away_team_id"] == "sweden")
        )
        | (
            (matches["home_team_id"] == "sweden") & (matches["away_team_id"] == "netherlands")
        )
    ]
    assert len(target) >= 1
    assert (target["status"] == "scheduled").any() or (target["competition"].str.contains("World Cup")).any()
    assert (target["corpus"] == CORPUS_INTERNATIONAL).all()


@pytest.mark.skipif(
    not Path("data/interim/matches.parquet").exists(),
    reason="requires clean run",
)
def test_corpus_discriminator():
    import re

    matches = pd.read_parquet("data/interim/matches.parquet")
    odds = pd.read_parquet("data/interim/odds.parquet")
    ratings = pd.read_parquet("data/interim/ratings.parquet")
    club_comp_re = re.compile(r"^[A-Z0-9]+_\d{4}$")

    for name, df in (("matches", matches), ("odds", odds), ("ratings", ratings)):
        assert "corpus" in df.columns, f"{name} missing corpus"
        assert df["corpus"].notna().all(), f"{name} has null corpus"
        assert set(df["corpus"].unique()).issubset({CORPUS_INTERNATIONAL, CORPUS_CLUB})

    intl = matches[matches["corpus"] == CORPUS_INTERNATIONAL]
    assert not intl["competition"].astype(str).str.match(club_comp_re, na=False).any()
    assert (odds["corpus"] == CORPUS_CLUB).any()
    assert (ratings["corpus"] == CORPUS_INTERNATIONAL).all()
