"""Tests for parquet IO, config, and the prediction contract."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from nlvswe.config import AppConfig, config_hash, get_config, load_config
from nlvswe.io import read_table, write_table
from nlvswe.models.base import BaseModel, MatchPrediction, validate_prediction
from nlvswe.repro import set_seeds


@pytest.fixture(autouse=True)
def _clear_config_cache():
    get_config.cache_clear()
    yield
    get_config.cache_clear()


def test_config_loads():
    cfg = load_config()
    assert cfg.seed == 20260619
    assert cfg.target_match.home == "Netherlands"
    assert cfg.target_match.away == "Sweden"
    assert cfg.scope in ("history", "live")


def test_config_rejects_unknown_key():
    cfg = load_config()
    raw = cfg.model_dump(mode="json")
    raw["unexpected_key"] = True
    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw)


def test_config_hash_stable():
    cfg1 = load_config()
    cfg2 = load_config()
    assert config_hash(cfg1) == config_hash(cfg2)


def test_write_read_table_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("nlvswe.io._DATA_ROOT", tmp_path)
    df = pd.DataFrame({"id": [2, 1], "x": [10.0, 20.0]})
    write_table(df, "test_table", "processed", sort_by="id", plan="01")
    out = read_table("test_table", "processed")
    expected = df.sort_values("id", kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(out, expected)

    manifest_path = tmp_path / "processed" / "test_table.manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_plan"] == "01"
    assert manifest["row_count"] == 2
    assert "git_commit" in manifest
    assert "config_hash" in manifest
    assert "created_at" in manifest


def test_set_seeds_deterministic():
    set_seeds(42)
    a = np.random.rand(3)
    set_seeds(42)
    b = np.random.rand(3)
    np.testing.assert_array_equal(a, b)


def test_basemodel_abstract():
    with pytest.raises(TypeError):
        BaseModel()  # type: ignore[abstract]


def test_basemodel_subclass_and_validate():
    class TrivialModel(BaseModel):
        name = "trivial"

        def fit(self, train: pd.DataFrame) -> None:
            pass

        def predict(self, match: pd.Series) -> MatchPrediction:
            return MatchPrediction(
                match_id=str(match.get("id", "x")),
                scoreline=None,
                probs_1x2={"home": 0.4, "draw": 0.3, "away": 0.3},
            )

    model = TrivialModel()
    pred = model.predict(pd.Series({"id": "m1"}))
    validate_prediction(pred)

    bad = MatchPrediction(
        match_id="m1",
        scoreline=None,
        probs_1x2={"home": 0.5, "draw": 0.3, "away": 0.3},
    )
    with pytest.raises(ValueError, match="sum"):
        validate_prediction(bad)
