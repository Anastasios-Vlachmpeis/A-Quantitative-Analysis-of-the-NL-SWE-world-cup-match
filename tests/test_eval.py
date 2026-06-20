"""Walk-forward eval harness tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nlvswe.eval.backtest import walk_forward_backtest
from nlvswe.eval.calibration import expected_calibration_error, plot_reliability
from nlvswe.eval.market import MarketModel, devig
from nlvswe.eval.report import build_report
from nlvswe.eval.scoring import brier_1x2, log_loss_1x2, ranked_probability_score, score_predictions
from nlvswe.models.base import BaseModel, MatchPrediction, validate_prediction
from nlvswe.models.constant import ConstantModel


def _probs(h: float, d: float, a: float) -> dict[str, float]:
    return {"home": h, "draw": d, "away": a}


def test_rps_logloss_brier_hand_computed():
    p1 = _probs(0.5, 0.3, 0.2)
    assert ranked_probability_score(p1, "home") == pytest.approx(0.145)
    assert log_loss_1x2(p1, "home") == pytest.approx(-math.log(0.5))
    assert brier_1x2(p1, "home") == pytest.approx(0.38)

    p2 = _probs(0.6, 0.2, 0.2)
    assert ranked_probability_score(p2, "away") == pytest.approx(0.5)
    assert log_loss_1x2(p2, "away") == pytest.approx(-math.log(0.2))
    assert brier_1x2(p2, "away") == pytest.approx(1.04)

    preds = pd.DataFrame(
        [
            {"model": "m", "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2, "outcome": "home"},
            {"model": "m", "p_home": 0.6, "p_draw": 0.2, "p_away": 0.2, "outcome": "away"},
        ]
    )
    scores = score_predictions(preds, bootstrap_samples=500, seed=0)
    rps_row = scores[(scores["metric"] == "rps")].iloc[0]
    assert rps_row["mean"] == pytest.approx(0.3225)
    assert rps_row["n"] == 2


def test_devig_multiplicative_and_shin():
    odds = np.array([2.0, 3.5, 4.0])
    mult = devig(odds, method="multiplicative")
    assert mult.sum() == pytest.approx(1.0)
    assert np.all(mult >= 0)
    implied = 1.0 / odds
    assert mult[0] == pytest.approx(implied[0] / implied.sum())

    shin = devig(odds, method="shin")
    assert shin.sum() == pytest.approx(1.0)
    assert np.all(shin >= 0)
    # Shin shifts probability toward favourites vs multiplicative when overround > 1.
    heavy = np.array([1.5, 4.0, 6.0])
    assert devig(heavy, "shin")[0] > devig(heavy, "multiplicative")[0]


def _tiny_features(n: int = 5) -> pd.DataFrame:
    rows = []
    outcomes = ["home", "draw", "away", "home", "away"]
    for i in range(n):
        rows.append(
            {
                "match_id": f"m{i}",
                "date_utc": pd.Timestamp(f"2024-0{i+1}-01", tz="UTC"),
                "corpus": "international",
                "result_1x2": outcomes[i],
            }
        )
    df = pd.DataFrame(rows)
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True).astype("datetime64[ns, UTC]")
    return df


def _tiny_odds(match_ids: list[str]) -> pd.DataFrame:
    rows = []
    for mid in match_ids:
        for sel, price in [("home", 2.0), ("draw", 3.5), ("away", 4.0)]:
            rows.append(
                {
                    "match_id": mid,
                    "bookmaker": "test_bk",
                    "market": "1x2",
                    "selection": sel,
                    "decimal_odds": price,
                    "captured_at": pd.Timestamp("2024-01-01", tz="UTC"),
                    "is_closing": True,
                    "corpus": "international",
                }
            )
    df = pd.DataFrame(rows)
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True).astype("datetime64[ns, UTC]")
    return df


def test_market_model_same_harness_as_constant():
    features = _tiny_features()
    odds = _tiny_odds(["m3", "m4"])
    market = MarketModel(odds, devig_method="multiplicative", bookmaker="test_bk")
    const = ConstantModel()

    m_pred = market.predict(features.iloc[3])
    c_pred = const.predict(features.iloc[3])
    validate_prediction(m_pred)
    validate_prediction(c_pred)

    for pred, outcome in ((m_pred, "away"), (c_pred, "away")):
        assert ranked_probability_score(pred.probs_1x2, outcome) >= 0
        assert log_loss_1x2(pred.probs_1x2, outcome) >= 0


def test_calibration_perfect_vs_skewed():
    n = 1000
    rng = np.random.default_rng(0)
    outcomes = pd.Series(rng.choice(["home", "draw", "away"], size=n, p=[0.5, 0.25, 0.25]))
    y_idx = outcomes.map({"home": 0, "draw": 1, "away": 2}).to_numpy()
    perfect = np.zeros((n, 3))
    perfect[np.arange(n), y_idx] = 1.0
    perfect_ece = expected_calibration_error(perfect, outcomes, bins=10)
    assert perfect_ece == pytest.approx(0.0, abs=1e-9)

    skewed = np.tile([0.9, 0.05, 0.05], (n, 1))
    skewed_ece = expected_calibration_error(skewed, outcomes, bins=10)
    assert skewed_ece > 0.2


def test_walk_forward_no_test_leakage_in_train():
    class SpyModel(BaseModel):
        name = "spy"
        train_sets: list[set[str]]

        def __init__(self) -> None:
            self.train_sets = []

        def fit(self, train: pd.DataFrame) -> None:
            self.train_sets.append(set(train["match_id"]))

        def predict(self, match: pd.Series) -> MatchPrediction:
            return MatchPrediction(
                match_id=str(match["match_id"]),
                scoreline=None,
                probs_1x2={"home": 0.4, "draw": 0.3, "away": 0.3},
            )

    features = _tiny_features()
    spy = SpyModel()
    preds = walk_forward_backtest(spy, features, min_history_matches=2, refit_every=1)
    assert len(preds) == 3

    scored_ids = ["m2", "m3", "m4"]
    for test_id, train_ids in zip(scored_ids, spy.train_sets, strict=True):
        assert test_id not in train_ids


def test_calibration_plot_renders(tmp_path, monkeypatch):
    monkeypatch.setattr("nlvswe.io._FIGURES_ROOT", tmp_path)
    preds = pd.DataFrame(
        {
            "model": ["constant"] * 50,
            "p_home": np.linspace(0.2, 0.8, 50),
            "p_draw": np.full(50, 0.2),
            "p_away": np.linspace(0.6, 0.0, 50),
            "outcome": ["home"] * 25 + ["away"] * 25,
        }
    )
    path = plot_reliability(preds, model="constant", bins=5)
    assert path is not None
    assert path.exists()


def test_build_report_scores(tmp_path, monkeypatch):
    monkeypatch.setattr("nlvswe.io._DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr("nlvswe.io._FIGURES_ROOT", tmp_path / "figures")
    preds = pd.DataFrame(
        [
            {
                "model": "constant",
                "match_id": "m1",
                "date_utc": pd.Timestamp("2024-01-01", tz="UTC"),
                "p_home": 0.5,
                "p_draw": 0.3,
                "p_away": 0.2,
                "outcome": "home",
                "has_scoreline": False,
            },
            {
                "model": "market",
                "match_id": "m2",
                "date_utc": pd.Timestamp("2024-02-01", tz="UTC"),
                "p_home": 0.45,
                "p_draw": 0.30,
                "p_away": 0.25,
                "outcome": "draw",
                "has_scoreline": False,
            },
        ]
    )
    scores, ece = build_report(preds, write_calibration=False)
    assert len(scores) == 6
    assert "constant" in ece
