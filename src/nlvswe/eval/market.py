"""Treat de-vigged closing 1X2 odds as a model (the benchmark to beat)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from nlvswe.models.base import BaseModel, MatchPrediction, OUTCOMES_1X2, validate_prediction

DevigMethod = Literal["multiplicative", "shin"]


def devig(odds: np.ndarray | list[float], method: DevigMethod = "multiplicative") -> np.ndarray:
    """Remove overround from decimal odds; return probabilities summing to 1."""
    arr = np.asarray(odds, dtype=float)
    if np.any(arr <= 1.0):
        raise ValueError("decimal odds must be > 1")
    if method == "multiplicative":
        return _devig_multiplicative(arr)
    if method == "shin":
        return _devig_shin(arr)
    raise ValueError(f"unknown devig method: {method!r}")


def _devig_multiplicative(odds: np.ndarray) -> np.ndarray:
    implied = 1.0 / odds
    return implied / implied.sum()


def _devig_shin(odds: np.ndarray) -> np.ndarray:
    """Shin (1991/1992) insider-trading model via root-finding on z."""
    implied = 1.0 / odds  # raw, sum > 1 when overround present

    def _sum_probs(z: float) -> float:
        if z >= 1.0:
            return -1.0
        terms = (np.sqrt(z * z + 4.0 * (1.0 - z) * implied * implied) - z) / (2.0 * (1.0 - z))
        return float(terms.sum())

    z = brentq(lambda z: _sum_probs(z) - 1.0, 0.0, 0.999999)
    probs = (np.sqrt(z * z + 4.0 * (1.0 - z) * implied * implied) - z) / (2.0 * (1.0 - z))
    probs = np.clip(probs, 0.0, 1.0)
    return probs / probs.sum()


def _match_market_probs(
    odds: pd.DataFrame,
    match_id: str,
    *,
    devig_method: DevigMethod,
    bookmaker: str | None,
) -> np.ndarray | None:
    """De-vigged 1X2 consensus for one match.

    Named book's close if I have it; otherwise median across every book with a
    complete quote (so one missing Pinnacle line doesn't blank the whole row).
    """
    sub = odds[(odds["match_id"] == match_id) & (odds["market"] == "1x2")]
    if sub.empty:
        return None
    if sub["is_closing"].any():
        sub = sub[sub["is_closing"]]
    if bookmaker is not None:
        book_sub = sub[sub["bookmaker"] == bookmaker]
        if not book_sub.empty:
            sub = book_sub  # use the book you asked for
        # else: median across all books with a full quote

    book_probs: list[np.ndarray] = []
    for _, group in sub.groupby("bookmaker", sort=True):
        latest = (
            group.sort_values(["selection", "captured_at"], kind="mergesort")
            .groupby("selection", sort=False)
            .tail(1)
        )
        prices = {
            str(r["selection"]).lower(): float(r["decimal_odds"]) for _, r in latest.iterrows()
        }
        if not all(k in prices for k in OUTCOMES_1X2):
            continue
        odds_vec = np.array([prices[k] for k in OUTCOMES_1X2], dtype=float)
        book_probs.append(devig(odds_vec, method=devig_method))

    if not book_probs:
        return None
    arr = np.vstack(book_probs)
    if arr.shape[0] == 1:
        return arr[0]
    med = np.median(arr, axis=0)
    return med / med.sum()


class MarketModel(BaseModel):
    """De-vigged closing 1X2 odds packaged as a MatchPrediction.

    If you name a bookmaker I use their close when it's there; otherwise I fall
    back to the median across every book with a full 1X2 quote for that match.
    """

    name = "market"

    def __init__(
        self,
        odds: pd.DataFrame,
        *,
        devig_method: DevigMethod = "multiplicative",
        bookmaker: str | None = None,
    ) -> None:
        self.odds = odds
        self.devig_method = devig_method
        self.bookmaker = bookmaker

    def fit(self, train: pd.DataFrame) -> None:
        """Market probabilities come straight from odds; nothing to fit."""

    def predict(self, match: pd.Series) -> MatchPrediction:
        match_id = str(match["match_id"])
        probs = _match_market_probs(
            self.odds,
            match_id,
            devig_method=self.devig_method,
            bookmaker=self.bookmaker,
        )
        if probs is None:
            raise ValueError(f"no closing 1x2 odds for match {match_id!r}")
        pred = MatchPrediction(
            match_id=match_id,
            scoreline=None,
            probs_1x2={k: float(probs[i]) for i, k in enumerate(OUTCOMES_1X2)},
        )
        validate_prediction(pred)
        return pred
