"""Raw download and manifest tests."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from nlvswe.config import get_config, load_config
from nlvswe.data import acquire
from nlvswe.data.acquire import (
    MissingSecretError,
    fetch_elo,
    fetch_fifa_ranking,
    fetch_odds_history,
    fetch_odds_live,
    fetch_results,
    fetch_venues,
    fetch_weather,
    fetch_xg,
)
from nlvswe.io import read_raw_manifest

FIXTURES = Path(__file__).parent / "fixtures"
FAKE_API_KEY = "TEST_SECRET_DO_NOT_PERSIST_xyz789"


@pytest.fixture(autouse=True)
def _clear_config_cache():
    get_config.cache_clear()
    yield
    get_config.cache_clear()


@pytest.fixture
def raw_root(tmp_path, monkeypatch):
    """Redirect all raw writes to a temp directory."""
    monkeypatch.setattr(acquire, "PROJECT_ROOT", tmp_path)
    (tmp_path / "config").mkdir()
    template = FIXTURES.parent.parent / "config" / "wc2026_fixtures.csv"
    if template.exists():
        (tmp_path / "config" / "wc2026_fixtures.csv").write_text(
            template.read_text(encoding="utf-8"), encoding="utf-8"
        )
    else:
        (tmp_path / "config" / "wc2026_fixtures.csv").write_text(
            "match_id,home_team,away_team,competition,venue_name,city,country,latitude,longitude,altitude_m,kickoff_utc,kickoff_local\n"
            "wc2026-ned-swe,Netherlands,Sweden,FIFA World Cup,MetLife,NY,USA,40.81,-74.07,3,,\n",
            encoding="utf-8",
        )
    cfg = load_config()
    return tmp_path, cfg


def _manifest_for(path: Path) -> dict:
    return read_raw_manifest(path)


def test_fetch_results_writes_manifest(raw_root, monkeypatch):
    tmp_path, cfg = raw_root

    def fake_download(url, **kwargs):
        if url.endswith("results.csv"):
            return (FIXTURES / "results_sample.csv").read_bytes()
        if url.endswith("goalscorers.csv"):
            return b"date,home_team,away_team,team,scorer,own_goal,penalty\n"
        if url.endswith("shootouts.csv"):
            return b"date,home_team,away_team,winner,first_shooter\n"
        raise AssertionError(url)

    monkeypatch.setattr(acquire, "download_bytes", fake_download)
    paths = fetch_results(cfg)
    assert len(paths) == 3
    manifest = _manifest_for(paths[0])
    assert manifest["source"] == "intl_results"
    assert manifest["url"].endswith("results.csv")
    assert manifest["row_count"] == 1
    assert "license" in manifest
    assert "git_commit" in manifest


def test_fetch_fifa_ranking(raw_root, monkeypatch):
    tmp_path, cfg = raw_root
    monkeypatch.setattr(
        acquire,
        "download_bytes",
        lambda url, **kw: (FIXTURES / "fifa_ranking_sample.csv").read_bytes(),
    )
    path = fetch_fifa_ranking(cfg)
    manifest = _manifest_for(path)
    assert manifest["source"] == "fifa_ranking"
    df = pd.read_csv(path)
    assert {"team", "total_points", "date"}.issubset(df.columns)


def test_fetch_elo(raw_root, monkeypatch):
    tmp_path, cfg = raw_root
    monkeypatch.setattr(
        acquire,
        "download_bytes",
        lambda url, **kw: (FIXTURES / "elo_world_sample.tsv").read_bytes(),
    )
    path = fetch_elo(cfg)
    manifest = _manifest_for(path)
    assert manifest["source"] == "elo_world_snapshot"


def test_fetch_xg_skipped_when_disabled(raw_root):
    tmp_path, cfg = raw_root
    path = fetch_xg(cfg)
    assert path is not None
    payload = json.loads(path.read_text())
    assert payload["status"] == "skipped"


def test_fetch_odds_history(raw_root, monkeypatch):
    tmp_path, cfg = raw_root
    sample = (FIXTURES / "football_data_sample.csv").read_bytes()

    def fake_download(url, **kwargs):
        if url.endswith(".csv"):
            return sample
        raise AssertionError(url)

    monkeypatch.setattr(acquire, "download_bytes", fake_download)
    paths = fetch_odds_history(cfg)
    assert paths
    manifest = _manifest_for(paths[0])
    assert manifest["source"] == "club_odds_football_data"
    df = pd.read_csv(paths[0])
    assert "PSH" in df.columns


def test_fetch_venues(raw_root, monkeypatch):
    tmp_path, cfg = raw_root

    class FakeResp:
        def json(self):
            return {"elevation": [42.0]}

    monkeypatch.setattr(acquire, "get", lambda url, **kw: FakeResp())
    path = fetch_venues(cfg)
    manifest = _manifest_for(path)
    assert manifest["source"] == "wc2026_fixtures"
    df = pd.read_csv(path)
    assert "latitude" in df.columns
    assert "Netherlands" in df["home_team"].values


def test_fetch_weather_pending_without_kickoff(raw_root):
    tmp_path, cfg = raw_root
    fetch_venues(cfg)
    paths = fetch_weather(cfg)
    assert paths
    manifest = _manifest_for(paths[0])
    assert manifest["source"] == "venue_weather"


def test_odds_live_missing_key_raises(raw_root, monkeypatch):
    tmp_path, cfg = raw_root
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with pytest.raises(MissingSecretError, match="ODDS_API_KEY"):
        fetch_odds_live(cfg)


def test_odds_live_timestamped_snapshot(raw_root, monkeypatch):
    tmp_path, cfg = raw_root
    monkeypatch.setenv("ODDS_API_KEY", FAKE_API_KEY)

    class FakeResp:
        def json(self):
            return json.loads((FIXTURES / "odds_live_sample.json").read_text())

    monkeypatch.setattr(acquire, "get", lambda url, **kw: FakeResp())

    captured = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
    path = fetch_odds_live(cfg, captured_at=captured)
    assert path.name == "20260619T120000Z.json"
    manifest = _manifest_for(path)
    assert manifest["captured_at"] == "2026-06-19T12:00:00+00:00"
    assert manifest["source"] == "odds_live"
    assert FAKE_API_KEY not in path.read_text()
    assert FAKE_API_KEY not in json.dumps(manifest)


def test_no_secret_in_manifests_or_raw(raw_root, monkeypatch):
    """Grep-style check: env secret must never appear on disk."""
    tmp_path, cfg = raw_root
    monkeypatch.setenv("ODDS_API_KEY", FAKE_API_KEY)

    def fake_download(url, **kwargs):
        if "results.csv" in url:
            return (FIXTURES / "results_sample.csv").read_bytes()
        return b"col\n"

    monkeypatch.setattr(acquire, "download_bytes", fake_download)

    class FakeResp:
        def json(self):
            return []

    monkeypatch.setattr(acquire, "get", lambda url, **kw: FakeResp())

    fetch_results(cfg)
    fetch_odds_live(cfg)

    for path in tmp_path.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert FAKE_API_KEY not in text, f"Secret leaked in {path}"


def test_schema_spot_checks_on_fixtures():
    results = pd.read_csv(FIXTURES / "results_sample.csv")
    for col in ("date", "home_team", "away_team", "home_score", "away_score", "neutral", "tournament"):
        assert col in results.columns

    fifa = pd.read_csv(FIXTURES / "fifa_ranking_sample.csv")
    assert "team" in fifa.columns and "date" in fifa.columns and "total_points" in fifa.columns

    odds = pd.read_csv(FIXTURES / "football_data_sample.csv")
    assert "PSH" in odds.columns

    venues = pd.read_csv(FIXTURES.parent.parent / "config" / "wc2026_fixtures.csv")
    assert {"latitude", "longitude"}.issubset(venues.columns)
