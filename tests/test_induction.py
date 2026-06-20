"""Probability induction tests (Plan 07)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nlvswe.betting.induction import (
    MarketConfig,
    asian_handicap_probs,
    check_coherence,
    markets_from_1x2,
    markets_from_matrix,
    markets_from_samples,
    simulate_scorelines,
)


def _tiny_matrix() -> np.ndarray:
    mat = np.array(
        [
            [0.10, 0.10, 0.05],
            [0.10, 0.20, 0.10],
            [0.05, 0.10, 0.20],
        ],
        dtype=float,
    )
    return mat / mat.sum()


def _prob(rows, market, selection, method="analytic") -> float:
    for row in rows:
        if row["market"] == market and row["selection"] == selection and row["method"] == method:
            return row["model_prob"]
    raise KeyError((market, selection, method))


def test_hand_computed_1x2_ou_btts():
    mat = _tiny_matrix()
    rows = markets_from_matrix(
        mat,
        market_cfg=MarketConfig(
            totals_lines=(1.5, 2.5),
            ah_lines=(-0.5, 0.0),
            correct_score_top_n=3,
        ),
    )

    p_home = sum(mat[i, j] for i in range(3) for j in range(3) if i > j)
    p_draw = float(np.trace(mat))
    p_away = sum(mat[i, j] for i in range(3) for j in range(3) if i < j)
    assert _prob(rows, "1x2", "home") == pytest.approx(p_home)
    assert _prob(rows, "1x2", "draw") == pytest.approx(p_draw)
    assert _prob(rows, "1x2", "away") == pytest.approx(p_away)

    over_15 = sum(mat[i, j] for i in range(3) for j in range(3) if i + j > 1.5)
    assert _prob(rows, "totals_1.5", "over") == pytest.approx(over_15)
    assert _prob(rows, "totals_1.5", "under") == pytest.approx(1.0 - over_15)

    btts_yes = sum(mat[i, j] for i in range(1, 3) for j in range(1, 3))
    assert _prob(rows, "btts", "yes") == pytest.approx(btts_yes)


def test_analytic_vs_mc_within_standard_errors():
    mat = _tiny_matrix()
    cfg = MarketConfig(totals_lines=(2.5,), ah_lines=(0.0,), correct_score_top_n=5)
    analytic = markets_from_matrix(mat, market_cfg=cfg)
    n = 80_000
    rng = np.random.default_rng(0)
    mc = markets_from_samples(mat, n, rng, market_cfg=cfg)
    se = math.sqrt(0.25 / n)
    for market, selection in (("1x2", "home"), ("1x2", "draw"), ("btts", "yes"), ("totals_2.5", "over")):
        a = _prob(analytic, market, selection)
        m = _prob(mc, market, selection, method="mc")
        assert abs(a - m) <= 3 * se + 0.01


def test_coherence_identities():
    mat = _tiny_matrix()
    cfg = MarketConfig(
        totals_lines=(0.5, 1.5, 2.5),
        ah_lines=(-1.0, -0.5, 0.0, 0.5, 1.0),
        correct_score_top_n=5,
    )
    rows = markets_from_matrix(mat, market_cfg=cfg)
    assert check_coherence(rows) == []


def test_asian_handicap_push_on_level_line():
    mat = np.zeros((4, 4))
    mat[1, 1] = 1.0  # 1-1 draw only
    ah = asian_handicap_probs(mat, 0.0)
    assert ah["push"] == pytest.approx(1.0)
    assert ah["home"] == pytest.approx(0.0)
    assert ah["away"] == pytest.approx(0.0)


def test_asian_handicap_half_line_no_push():
    mat = np.zeros((3, 3))
    mat[1, 1] = 1.0
    ah = asian_handicap_probs(mat, -0.5)
    assert ah.get("push", 0.0) == pytest.approx(0.0)
    assert ah["away"] == pytest.approx(1.0)


def test_1x2_only_markets_limited():
    rows = markets_from_1x2(0.5, 0.25, 0.25)
    markets = {r["market"] for r in rows}
    assert markets == {"1x2", "double_chance"}
    assert check_coherence(rows) == []


def test_simulate_scorelines_reproducible():
    mat = _tiny_matrix()
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    s1 = simulate_scorelines(mat, 1000, rng1)
    s2 = simulate_scorelines(mat, 1000, rng2)
    np.testing.assert_array_equal(s1, s2)
