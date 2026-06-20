"""Reliability bins and calibration error for 1X2 probabilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nlvswe.io import save_figure
from nlvswe.models.base import OUTCOMES_1X2
from nlvswe.plotting.theme import apply_theme, style_axes, style_figure


def _outcome_one_hot(outcomes: pd.Series) -> np.ndarray:
    idx = outcomes.map({k: i for i, k in enumerate(OUTCOMES_1X2)}).to_numpy()
    y = np.zeros((len(outcomes), len(OUTCOMES_1X2)), dtype=float)
    y[np.arange(len(outcomes)), idx] = 1.0
    return y


def reliability_curve(
    probs: np.ndarray,
    outcomes: pd.Series,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    """Per-class reliability bins: mean predicted vs observed frequency."""
    y = _outcome_one_hot(outcomes)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict] = []
    for class_idx, label in enumerate(OUTCOMES_1X2):
        p = probs[:, class_idx]
        for b in range(bins):
            lo, hi = edges[b], edges[b + 1]
            if b < bins - 1:
                mask = (p >= lo) & (p < hi)
            else:
                mask = (p >= lo) & (p <= hi)
            if not mask.any():
                continue
            rows.append(
                {
                    "class": label,
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "bin_mid": 0.5 * (lo + hi),
                    "mean_pred": float(p[mask].mean()),
                    "obs_freq": float(y[mask, class_idx].mean()),
                    "count": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def expected_calibration_error(
    probs: np.ndarray,
    outcomes: pd.Series,
    *,
    bins: int = 10,
) -> float:
    """Weighted mean absolute calibration error across classes and bins."""
    curves = reliability_curve(probs, outcomes, bins=bins)
    if curves.empty:
        return float("nan")
    err = (curves["mean_pred"] - curves["obs_freq"]).abs() * curves["count"]
    return float(err.sum() / curves["count"].sum())


def plot_reliability(
    preds: pd.DataFrame,
    *,
    model: str,
    bins: int = 10,
    out_name: str | None = None,
) -> Path | None:
    """Plot reliability curves per class; save to reports/figures/."""
    subset = preds[preds["model"] == model]
    if subset.empty:
        return None

    apply_theme()
    probs = subset[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    curves = reliability_curve(probs, subset["outcome"], bins=bins)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"home": "#00B0FF", "draw": "#64748B", "away": "#FF5252"}
    for label in OUTCOMES_1X2:
        sub = curves[curves["class"] == label]
        if sub.empty:
            continue
        ax.plot(sub["mean_pred"], sub["obs_freq"], "o-", label=label, color=colors[label])

    ax.plot([0, 1], [0, 1], "--", color="#64748B", linewidth=1, label="perfect")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability — {model}")
    ax.legend()
    style_figure(fig)
    style_axes(ax)

    name = out_name or f"calibration_{model}"
    path = save_figure(fig, name)
    plt.close(fig)
    return path
