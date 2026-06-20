"""Walk-forward scoring, calibration checks, and the market benchmark."""

from nlvswe.eval.calibration import expected_calibration_error, plot_reliability, reliability_curve
from nlvswe.eval.market import MarketModel, devig
from nlvswe.eval.report import build_report
from nlvswe.eval.scoring import (
    brier_1x2,
    log_loss_1x2,
    ranked_probability_score,
    score_predictions,
)

__all__ = [
    "MarketModel",
    "brier_1x2",
    "build_report",
    "devig",
    "expected_calibration_error",
    "log_loss_1x2",
    "plot_reliability",
    "ranked_probability_score",
    "reliability_curve",
    "score_predictions",
]
