"""+EV backtest, Kelly sizing, and CLV tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nlvswe.betting.bankroll import monte_carlo_bankroll
from nlvswe.betting.clv import clv_odds_ratio, clv_prob_diff
from nlvswe.betting.strategy import (
    devigged_book_probs,
    expected_value,
    kelly_fraction,
    settle_bet,
    stake_amount,
)
from nlvswe.eval.market import devig


def test_ev_and_kelly_exact():
    p, o = 0.55, 2.0
    assert expected_value(p, o) == pytest.approx(0.55 * 2 - 1)
    # f* = (0.55*1 - 0.45)/1 = 0.10
    assert kelly_fraction(p, o) == pytest.approx(0.10)
    stake = stake_amount(1000, p, o, kelly_frac=0.25, max_stake_fraction=0.05, method="kelly")
    assert stake == pytest.approx(25.0)


def test_devig_parity_with_market_module():
    odds = np.array([2.0, 3.5, 4.0])
    p = devig(odds, method="multiplicative")
    rows = pd.DataFrame(
        [
            {"selection": "home", "decimal_odds": 2.0, "market": "1x2"},
            {"selection": "draw", "decimal_odds": 3.5, "market": "1x2"},
            {"selection": "away", "decimal_odds": 4.0, "market": "1x2"},
        ]
    )
    book = devigged_book_probs(rows, market="1x2", devig_method="multiplicative")
    assert book is not None
    assert book["home"] == pytest.approx(float(p[0]))
    assert sum(book.values()) == pytest.approx(1.0)


def test_settlement_win_lose_push():
    res, pnl = settle_bet(
        market="1x2", selection="home", home_goals=2, away_goals=1,
        decimal_odds=2.5, stake=10.0,
    )
    assert res == "win"
    assert pnl == pytest.approx(15.0)

    res, pnl = settle_bet(
        market="totals", selection="over_2.0", home_goals=1, away_goals=1,
        decimal_odds=2.0, stake=10.0,
    )
    assert res == "push"
    assert pnl == pytest.approx(0.0)

    res, pnl = settle_bet(
        market="ah_0.0", selection="home", home_goals=1, away_goals=1,
        decimal_odds=1.95, stake=10.0,
    )
    assert res == "push"
    assert pnl == pytest.approx(0.0)


def test_clv_sign_and_magnitude():
    # Beat the close: taken 2.10 vs close 2.00
    assert clv_odds_ratio(2.10, 2.00) == pytest.approx(0.05)
    diff = clv_prob_diff(0.55, 2.00, devig_method="multiplicative")
    assert diff > 0


def test_bankroll_mc_reproducible_and_ror_bounds():
    bets = pd.DataFrame(
        [
            {
                "model": "t",
                "date_utc": pd.Timestamp("2024-01-01", tz="UTC"),
                "match_id": "m1",
                "bookmaker": "b",
                "market": "1x2",
                "selection": "home",
                "model_prob": 0.55,
                "stake": 10.0,
                "odds_taken": 2.0,
                "result": "win",
                "pnl": 10.0,
            },
            {
                "model": "t",
                "date_utc": pd.Timestamp("2024-01-02", tz="UTC"),
                "match_id": "m2",
                "bookmaker": "b",
                "market": "1x2",
                "selection": "away",
                "model_prob": 0.40,
                "stake": 10.0,
                "odds_taken": 2.5,
                "result": "lose",
                "pnl": -10.0,
            },
        ]
    )
    _, s1 = monte_carlo_bankroll(bets, initial_bankroll=1000, n_sims=500, seed=42)
    _, s2 = monte_carlo_bankroll(bets, initial_bankroll=1000, n_sims=500, seed=42)
    assert s1["mean_final_bankroll"] == pytest.approx(s2["mean_final_bankroll"])
    assert 0.0 <= s1["risk_of_ruin"] <= 1.0
