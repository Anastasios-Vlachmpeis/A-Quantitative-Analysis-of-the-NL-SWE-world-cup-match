"""Model ladder fit/predict tests."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from nlvswe.models._goals_common import (
    dc_tau,
    fit_poisson_strengths,
    independent_poisson_matrix,
    probs_1x2_from_scoreline,
)
from nlvswe.models.baseline import BaselineModel
from nlvswe.models.bayesian import BayesianHierModel
from nlvswe.models.bivariate_poisson import BivariatePoissonModel
from nlvswe.models.data import assert_international_only
from nlvswe.models.dixon_coles import DixonColesModel
from nlvswe.models.elo import EloModel
from nlvswe.models.ml import MLModel
from nlvswe.models.poisson import PoissonModel
from nlvswe.models.base import validate_prediction


CORPUS = "international"
TEAMS = ["alpha", "beta", "gamma", "delta"]


def _outcome(hg: int, ag: int) -> str:
    if hg > ag:
        return "home"
    if hg < ag:
        return "away"
    return "draw"


def _synthetic_features(n: int = 60, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        home, away = rng.choice(TEAMS, size=2, replace=False)
        neutral = bool(rng.random() < 0.25)
        hg = int(rng.poisson(1.4 if not neutral else 1.2))
        ag = int(rng.poisson(1.1))
        elo_diff = float(rng.normal(50, 120))
        rows.append(
            {
                "match_id": f"s{i:03d}",
                "date_utc": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=i * 7),
                "corpus": CORPUS,
                "competition": "Friendly",
                "home_team_id": home,
                "away_team_id": away,
                "neutral": neutral,
                "is_home": not neutral,
                "is_knockout": False,
                "home_elo_pre": 1500.0 + elo_diff / 2,
                "away_elo_pre": 1500.0 - elo_diff / 2,
                "elo_diff": elo_diff,
                "form_ppg_diff_5": float(rng.normal(0, 0.5)),
                "form_ppg_diff_10": float(rng.normal(0, 0.5)),
                "form_gd_diff_5": float(rng.normal(0, 0.3)),
                "fifa_points_diff": float(rng.normal(0, 50)),
                "rest_diff": float(rng.normal(0, 3)),
                "home_goals": hg,
                "away_goals": ag,
                "result_1x2": _outcome(hg, ag),
            }
        )
    df = pd.DataFrame(rows)
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True).astype("datetime64[ns, UTC]")
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    return df


@pytest.fixture
def train_df():
    return _synthetic_features(80, seed=42)


@pytest.fixture
def predict_row(train_df):
    return train_df.iloc[-1].copy()


@pytest.mark.parametrize(
    "model_cls,kwargs",
    [
        (BaselineModel, {}),
        (EloModel, {"seed": 0}),
        (PoissonModel, {"max_goals": 5}),
        (DixonColesModel, {"max_goals": 5}),
        (BivariatePoissonModel, {"max_goals": 5}),
        (MLModel, {"seed": 0, "max_iter": 30}),
    ],
)
def test_model_prediction_valid(model_cls, kwargs, train_df, predict_row):
    model = model_cls(**kwargs)
    model.fit(train_df)
    pred = model.predict(predict_row)
    validate_prediction(pred)
    if pred.scoreline is not None:
        implied = probs_1x2_from_scoreline(pred.scoreline)
        for k in ("home", "draw", "away"):
            assert pred.probs_1x2[k] == pytest.approx(implied[k], abs=1e-6)


def test_poisson_recovers_relative_strength_on_synthetic():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(120):
        home, away = ("strong", "weak") if i % 2 == 0 else ("weak", "strong")
        hg = int(rng.poisson(2.0 if home == "strong" else 0.8))
        ag = int(rng.poisson(0.8 if away == "strong" else 2.0))
        rows.append(
            {
                "match_id": f"p{i}",
                "date_utc": pd.Timestamp("2021-01-01", tz="UTC") + pd.Timedelta(days=i),
                "corpus": CORPUS,
                "home_team_id": home,
                "away_team_id": away,
                "neutral": True,
                "home_goals": hg,
                "away_goals": ag,
                "result_1x2": _outcome(hg, ag),
            }
        )
    train = pd.DataFrame(rows)
    strengths = fit_poisson_strengths(train, half_life_days=365)
    assert strengths.attack["strong"] > strengths.attack["weak"]


def test_dixon_coles_tau_only_low_scores():
    assert dc_tau(0, 0, 1.2, 1.0, -0.1) != 1.0
    assert dc_tau(2, 3, 1.2, 1.0, -0.1) == 1.0


def test_dixon_coles_rho_in_valid_range(train_df):
    model = DixonColesModel(max_goals=5)
    model.fit(train_df)
    assert -0.25 <= model._rho <= 0.1


def test_deterministic_predictions(train_df, predict_row):
    m1 = EloModel(seed=99)
    m2 = EloModel(seed=99)
    m1.fit(train_df)
    m2.fit(train_df)
    p1 = m1.predict(predict_row)
    p2 = m2.predict(predict_row)
    assert p1.probs_1x2 == p2.probs_1x2


def test_international_corpus_guard():
    bad = pd.DataFrame({"corpus": ["club", "international"]})
    with pytest.raises(ValueError, match="corpus"):
        assert_international_only(bad)


def test_bayesian_runs_on_tiny_dataset_within_budget():
    train = _synthetic_features(35, seed=7)
    row = train.iloc[-1]
    model = BayesianHierModel(max_goals=4, draws=40, tune=40, chains=1, seed=123)
    start = time.perf_counter()
    model.fit(train)
    pred = model.predict(row)
    elapsed = time.perf_counter() - start
    validate_prediction(pred)
    assert elapsed < 120.0


def test_bayesian_deterministic_with_same_seed():
    train = _synthetic_features(35, seed=8)
    row = train.iloc[-1]
    a = BayesianHierModel(max_goals=4, draws=30, tune=30, chains=1, seed=555)
    b = BayesianHierModel(max_goals=4, draws=30, tune=30, chains=1, seed=555)
    a.fit(train)
    b.fit(train)
    pa = a.predict(row)
    pb = b.predict(row)
    assert pa.probs_1x2["home"] == pytest.approx(pb.probs_1x2["home"])
    assert pa.probs_1x2["draw"] == pytest.approx(pb.probs_1x2["draw"])


def test_scoreline_matrix_normalization():
    mat = independent_poisson_matrix(1.3, 1.1, max_goals=5)
    assert mat.shape == (6, 6)
    assert mat.sum() == pytest.approx(1.0)
