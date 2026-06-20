"""EV gate, Kelly sizing, settlement for backtest and live slips."""

from __future__ import annotations

import math
import re
from typing import Literal

import numpy as np
import pandas as pd

from nlvswe.betting.induction import _ah_home_outcome
from nlvswe.eval.market import devig
from nlvswe.models.base import OUTCOMES_1X2

StakeMethod = Literal["kelly", "flat"]
BetResult = Literal["win", "lose", "push", "half_win", "half_lose"]


def expected_value(model_prob: float, decimal_odds: float) -> float:
    """EV per unit staked: p * o - 1."""
    return float(model_prob * decimal_odds - 1.0)


def kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """Full Kelly fraction for a decimal-odds bet; negative means no bet."""
    if decimal_odds <= 1.0:
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    return float((model_prob * b - q) / b)


def stake_amount(
    bankroll: float,
    model_prob: float,
    decimal_odds: float,
    *,
    kelly_frac: float,
    max_stake_fraction: float,
    method: StakeMethod = "kelly",
    flat_stake: float = 10.0,
) -> float:
    """Return stake in currency units (0 if no bet)."""
    if method == "flat":
        return min(float(flat_stake), bankroll * max_stake_fraction, bankroll)
    raw = kelly_fraction(model_prob, decimal_odds)
    if raw <= 0:
        return 0.0
    frac = min(kelly_frac * raw, max_stake_fraction)
    return max(0.0, bankroll * frac)


def _result_1x2(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if home_goals < away_goals:
        return "away"
    return "draw"


def _parse_totals_selection(selection: str) -> tuple[str, float] | None:
    m = re.match(r"^(over|under)_([0-9.]+)$", str(selection).lower())
    if not m:
        return None
    return m.group(1), float(m.group(2))


def settle_bet(
    *,
    market: str,
    selection: str,
    home_goals: int,
    away_goals: int,
    decimal_odds: float,
    stake: float,
) -> tuple[BetResult, float]:
    """Return (result, pnl) for a settled bet."""
    total = home_goals + away_goals
    sel = str(selection).lower()
    market_l = str(market).lower()

    if market_l == "1x2":
        outcome = _result_1x2(home_goals, away_goals)
        if sel == outcome:
            return "win", stake * (decimal_odds - 1.0)
        return "lose", -stake

    if market_l == "double_chance":
        outcome = _result_1x2(home_goals, away_goals)
        win = (
            (sel == "1x" and outcome in {"home", "draw"})
            or (sel == "12" and outcome in {"home", "away"})
            or (sel == "x2" and outcome in {"draw", "away"})
        )
        if win:
            return "win", stake * (decimal_odds - 1.0)
        return "lose", -stake

    if market_l.startswith("totals") or market_l == "totals":
        line = None
        side = sel
        if market_l.startswith("totals_"):
            line = float(market_l.split("_", 1)[1])
        parsed = _parse_totals_selection(sel)
        if parsed is not None:
            side, line = parsed
        if line is None:
            return "lose", -stake
        diff = total - line
        if side == "over":
            if diff > 0:
                return "win", stake * (decimal_odds - 1.0)
            if math.isclose(diff, 0.0):
                return "push", 0.0
            return "lose", -stake
        if diff < 0:
            return "win", stake * (decimal_odds - 1.0)
        if math.isclose(diff, 0.0):
            return "push", 0.0
        return "lose", -stake

    if market_l.startswith("ah_"):
        line_str = market_l[3:].replace("p", "+").replace("m", "-")
        line = float(line_str)
        ah = _ah_home_outcome(home_goals, away_goals, line)
        if selection in {"home", "win"}:
            if ah == "win":
                return "win", stake * (decimal_odds - 1.0)
            if ah == "push":
                return "push", 0.0
            return "lose", -stake
        if selection in {"away", "lose"}:
            if ah == "lose":
                return "win", stake * (decimal_odds - 1.0)
            if ah == "push":
                return "push", 0.0
            return "lose", -stake

    return "lose", -stake


def devigged_book_probs(
    odds_rows: pd.DataFrame,
    *,
    market: str,
    devig_method: str,
) -> dict[str, float] | None:
    """De-vig a single book's quote for a market."""
    market_l = str(market).lower()
    sub = odds_rows[odds_rows["market"].astype(str).str.lower() == market_l]
    if sub.empty and market_l.startswith("totals"):
        sub = odds_rows[odds_rows["market"].astype(str).str.lower() == "totals"]
    if sub.empty:
        return None

    if market_l == "1x2":
        prices = {}
        for sel in OUTCOMES_1X2:
            rows = sub[sub["selection"].astype(str).str.lower() == sel]
            if rows.empty:
                return None
            prices[sel] = float(rows.iloc[-1]["decimal_odds"])
        probs = devig(np.array([prices[k] for k in OUTCOMES_1X2]), method=devig_method)  # type: ignore[arg-type]
        return {k: float(probs[i]) for i, k in enumerate(OUTCOMES_1X2)}

    if market_l == "totals" or market_l.startswith("totals"):
        pairs: dict[str, float] = {}
        for _, row in sub.iterrows():
            pairs[str(row["selection"]).lower()] = float(row["decimal_odds"])
        if len(pairs) < 2:
            return None
        odds_vec = np.array(list(pairs.values()), dtype=float)
        probs = devig(odds_vec, method=devig_method)  # type: ignore[arg-type]
        return dict(zip(pairs.keys(), probs.astype(float), strict=True))

    return None


def consensus_opening_probs(
    odds: pd.DataFrame,
    match_id: str,
    market: str,
    *,
    devig_method: str,
) -> dict[str, float] | None:
    """Median de-vigged opening probabilities across books for one market."""
    sub = odds[
        (odds["match_id"].astype(str) == str(match_id))
        & (odds["is_closing"] == False)  # noqa: E712
    ]
    if sub.empty:
        return None

    market_l = str(market).lower()
    if market_l == "1x2":
        sub_m = sub[sub["market"].astype(str).str.lower() == "1x2"]
    elif market_l.startswith("totals"):
        sub_m = sub[sub["market"].astype(str).str.lower() == "totals"]
    else:
        sub_m = sub[sub["market"].astype(str).str.lower() == market_l]
    if sub_m.empty:
        return None

    book_maps: list[dict[str, float]] = []
    for _, group in sub_m.groupby("bookmaker", sort=True):
        latest = (
            group.sort_values(["selection", "captured_at"], kind="mergesort")
            .groupby("selection", sort=False)
            .tail(1)
        )
        probs = devigged_book_probs(latest, market=market_l if market_l != "totals" else "totals", devig_method=devig_method)
        if probs is not None:
            book_maps.append(probs)
    if not book_maps:
        return None

    keys = sorted(book_maps[0].keys())
    arr = np.array([[m[k] for k in keys] for m in book_maps], dtype=float)
    med = np.median(arr, axis=0)
    med = med / med.sum() if med.sum() > 0 else med
    return {k: float(med[i]) for i, k in enumerate(keys)}


def vectorized_mult_devig_1x2(
    prematch: pd.DataFrame,
) -> pd.DataFrame:
    """Add book_prob column via multiplicative de-vig (vectorized)."""
    if prematch.empty:
        return prematch.copy()
    latest = (
        prematch.sort_values(["match_id", "bookmaker", "selection", "captured_at"], kind="mergesort")
        .groupby(["match_id", "bookmaker", "selection"], sort=False)
        .tail(1)
    )
    wide = latest.pivot(index=["match_id", "bookmaker"], columns="selection", values="decimal_odds")
    for col in OUTCOMES_1X2:
        if col not in wide.columns:
            return pd.DataFrame()
    odds_mat = wide[list(OUTCOMES_1X2)].astype(float)
    implied = 1.0 / odds_mat
    probs = implied.div(implied.sum(axis=1), axis=0)
    long = probs.reset_index().melt(
        id_vars=["match_id", "bookmaker"], var_name="selection", value_name="book_prob"
    )
    long["selection"] = long["selection"].astype(str).str.lower()
    out = latest.merge(long, on=["match_id", "bookmaker", "selection"], how="inner")
    return out


def bulk_opening_consensus_1x2(
    odds: pd.DataFrame,
    *,
    devig_method: str,
) -> pd.DataFrame:
    """Median de-vigged opening 1X2 probabilities per match."""
    prematch = odds[(~odds["is_closing"]) & (odds["market"].astype(str).str.lower() == "1x2")].copy()
    if prematch.empty:
        return pd.DataFrame(columns=["match_id", "home", "draw", "away"])

    if devig_method == "multiplicative":
        enriched = vectorized_mult_devig_1x2(prematch)
        if enriched.empty:
            return pd.DataFrame(columns=["match_id", "home", "draw", "away"])
        med = enriched.groupby(["match_id", "selection"])["book_prob"].median().unstack()
        med = med.rename(columns=str)
        for col in OUTCOMES_1X2:
            if col not in med.columns:
                med[col] = np.nan
        med = med[list(OUTCOMES_1X2)].dropna(how="any")
        total = med.sum(axis=1)
        med = med.div(total, axis=0)
        return med.reset_index()

    book_probs: list[dict] = []
    for (match_id, bookmaker), group in prematch.groupby(["match_id", "bookmaker"], sort=False):
        latest = (
            group.sort_values(["selection", "captured_at"], kind="mergesort")
            .groupby("selection", sort=False)
            .tail(1)
        )
        probs = devigged_book_probs(latest, market="1x2", devig_method=devig_method)
        if probs is None:
            continue
        row = {"match_id": str(match_id), "bookmaker": bookmaker}
        row.update(probs)
        book_probs.append(row)

    if not book_probs:
        return pd.DataFrame(columns=["match_id", "home", "draw", "away"])

    long = pd.DataFrame(book_probs)
    consensus = long.groupby("match_id")[["home", "draw", "away"]].median().reset_index()
    total = consensus[["home", "draw", "away"]].sum(axis=1)
    consensus[["home", "draw", "away"]] = consensus[["home", "draw", "away"]].div(total, axis=0)
    return consensus


def closing_odds_for_selection(
    odds: pd.DataFrame,
    match_id: str,
    market: str,
    selection: str,
    *,
    bookmaker: str | None = None,
) -> float | None:
    """Closing decimal odds for a selection (for CLV)."""
    sub = odds[
        (odds["match_id"].astype(str) == str(match_id))
        & (odds["is_closing"] == True)  # noqa: E712
    ]
    market_l = str(market).lower()
    if market_l.startswith("totals_"):
        sub = sub[sub["market"].astype(str).str.lower() == "totals"]
    else:
        sub = sub[sub["market"].astype(str).str.lower() == market_l]
    if bookmaker is not None:
        book_sub = sub[sub["bookmaker"] == bookmaker]
        if not book_sub.empty:
            sub = book_sub
    sel = str(selection).lower()
    rows = sub[sub["selection"].astype(str).str.lower() == sel]
    if rows.empty:
        return None
    return float(rows.sort_values("captured_at").iloc[-1]["decimal_odds"])
