"""Closing-line value: did I beat the market's final price?"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nlvswe.eval.market import devig


def clv_odds_ratio(odds_taken: float, odds_closing: float) -> float:
    """Relative CLV from odds: taken/closing - 1. Positive = beat the close."""
    if odds_closing <= 0:
        return float("nan")
    return float(odds_taken / odds_closing - 1.0)


def clv_prob_diff(
    model_prob: float,
    closing_odds: float,
    *,
    devig_method: str = "multiplicative",
    all_closing_odds: list[float] | None = None,
) -> float:
    """CLV as model prob minus de-vigged closing implied prob for the selection."""
    if all_closing_odds is not None and len(all_closing_odds) >= 2:
        probs = devig(np.asarray(all_closing_odds, dtype=float), method=devig_method)  # type: ignore[arg-type]
        idx = all_closing_odds.index(closing_odds) if closing_odds in all_closing_odds else 0
        closing_prob = float(probs[idx])
    else:
        closing_prob = 1.0 / closing_odds if closing_odds > 1.0 else float("nan")
    return float(model_prob - closing_prob)


def aggregate_clv(bets: pd.DataFrame) -> dict[str, float]:
    """Summary CLV statistics for a bet ledger."""
    if bets.empty:
        return {
            "mean_clv_odds_ratio": float("nan"),
            "median_clv_odds_ratio": float("nan"),
            "mean_clv_prob_diff": float("nan"),
            "pct_positive_clv": float("nan"),
            "n_with_clv": 0,
        }
    clv = bets["clv_odds_ratio"].dropna()
    cpd = bets["clv_prob_diff"].dropna()
    return {
        "mean_clv_odds_ratio": float(clv.mean()) if not clv.empty else float("nan"),
        "median_clv_odds_ratio": float(clv.median()) if not clv.empty else float("nan"),
        "mean_clv_prob_diff": float(cpd.mean()) if not cpd.empty else float("nan"),
        "pct_positive_clv": float((clv > 0).mean()) if not clv.empty else float("nan"),
        "n_with_clv": int(clv.shape[0]),
    }
