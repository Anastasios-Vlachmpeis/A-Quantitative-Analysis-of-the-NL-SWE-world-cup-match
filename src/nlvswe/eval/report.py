"""Write eval scores to disk and save calibration figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nlvswe.config import AppConfig, get_config
from nlvswe.eval.calibration import expected_calibration_error, plot_reliability
from nlvswe.eval.scoring import score_predictions
from nlvswe.io import write_table
from nlvswe.logging import get_logger
from nlvswe.schemas import validate_table

logger = get_logger(__name__)

PLAN = "05"
PROCESSED = "processed"


def build_report(
    preds: pd.DataFrame,
    *,
    cfg: AppConfig | None = None,
    write_scores: bool = True,
    write_calibration: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Score predictions and optionally persist scores + calibration plots."""
    cfg = cfg or get_config()
    scores = score_predictions(
        preds,
        bootstrap_samples=cfg.eval.bootstrap_samples,
        seed=cfg.seed,
    )

    ece_by_model: dict[str, float] = {}
    for model in sorted(preds["model"].unique()):
        sub = preds[preds["model"] == model]
        probs = sub[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
        ece_by_model[model] = expected_calibration_error(
            probs,
            sub["outcome"],
            bins=cfg.eval.calibration_bins,
        )
        if write_calibration:
            plot_reliability(sub, model=model, bins=cfg.eval.calibration_bins)

    if write_scores and not scores.empty:
        scores = validate_table(scores, "eval_scores")
        write_table(
            scores,
            "eval_scores",
            PROCESSED,
            sort_by=["model", "metric"],
            plan=PLAN,
            schema_name="eval_scores",
        )

    logger.info("Eval report: %d models, ECE=%s", len(ece_by_model), ece_by_model)
    return scores, ece_by_model
