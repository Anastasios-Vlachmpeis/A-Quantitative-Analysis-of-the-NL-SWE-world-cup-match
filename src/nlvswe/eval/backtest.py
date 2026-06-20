"""Walk-forward backtest: each match is predicted using only prior history."""

from __future__ import annotations

import argparse
from typing import Callable, Literal

import pandas as pd

from nlvswe.config import AppConfig, get_config, load_config
from nlvswe.eval.market import MarketModel
from nlvswe.io import read_table, write_table
from nlvswe.logging import get_logger
from nlvswe.models.base import BaseModel, MatchPrediction
from nlvswe.models.constant import ConstantModel
from nlvswe.repro import set_seeds
from nlvswe.schemas import validate_table

logger = get_logger(__name__)

PLAN = "05"
PROCESSED = "processed"
INTERIM = "interim"

CvWindow = Literal["expanding", "rolling"]


def prediction_to_row(pred: MatchPrediction, *, model: str, match: pd.Series) -> dict:
    row = {
        "match_id": pred.match_id,
        "model": model,
        "date_utc": pd.Timestamp(match["date_utc"]).tz_convert("UTC"),
        "p_home": pred.probs_1x2["home"],
        "p_draw": pred.probs_1x2["draw"],
        "p_away": pred.probs_1x2["away"],
        "outcome": match.get("result_1x2"),
        "has_scoreline": pred.scoreline is not None,
    }
    if pred.scoreline is not None:
        row["scoreline_flat"] = pred.scoreline.ravel().tolist()
    return row


def walk_forward_backtest(
    model: BaseModel,
    features: pd.DataFrame,
    *,
    min_history_matches: int = 100,
    cv_window: CvWindow = "expanding",
    rolling_matches: int | None = None,
    refit_every: int = 1,
    corpus: str = "international",
) -> pd.DataFrame:
    """Walk-forward predictions: train strictly on prior matches only."""
    subset = features[features["corpus"] == corpus].sort_values(
        ["date_utc", "match_id"], kind="mergesort"
    )
    scored = subset[subset["result_1x2"].notna()].reset_index(drop=True)
    rows: list[dict] = []
    fitted = False
    matches_since_fit = refit_every

    for i in range(len(scored)):
        test = scored.iloc[i]
        train = scored.iloc[:i]
        if len(train) < min_history_matches:
            continue
        if cv_window == "rolling" and rolling_matches is not None:
            train = train.tail(rolling_matches)

        if not fitted or matches_since_fit >= refit_every:
            model.fit(train)
            fitted = True
            matches_since_fit = 0

        try:
            pred = model.predict(test)
        except ValueError:
            matches_since_fit += 1
            continue
        rows.append(prediction_to_row(pred, model=model.name, match=test))
        matches_since_fit += 1

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["date_utc"] = pd.to_datetime(out["date_utc"], utc=True).astype("datetime64[ns, UTC]")
    out["outcome"] = out["outcome"].astype("string")
    out["has_scoreline"] = out["has_scoreline"].astype("bool")
    return out.sort_values(["date_utc", "match_id"], kind="mergesort").reset_index(drop=True)


def load_backtest_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = read_table("features", PROCESSED)
    try:
        odds = read_table("odds", INTERIM)
    except FileNotFoundError:
        odds = pd.DataFrame()
    return features, odds


ModelFactory = Callable[[AppConfig, pd.DataFrame], BaseModel]


def _factory_constant(cfg: AppConfig, _odds: pd.DataFrame) -> BaseModel:
    return ConstantModel(use_base_rates=True)


def _factory_market(cfg: AppConfig, odds: pd.DataFrame) -> BaseModel:
    bookmaker = cfg.betting.bookmakers[0] if cfg.betting.bookmakers else None
    return MarketModel(
        odds,
        devig_method=cfg.betting.devig_method,
        bookmaker=bookmaker,
    )


MODEL_FACTORIES: dict[str, ModelFactory] = {
    "constant": _factory_constant,
    "market": _factory_market,
}


def run_backtest(model_name: str, cfg: AppConfig | None = None) -> pd.DataFrame:
    cfg = cfg or get_config()
    if model_name not in MODEL_FACTORIES:
        raise KeyError(f"Unknown model {model_name!r}. Available: {sorted(MODEL_FACTORIES)}")

    features, odds = load_backtest_data()
    model = MODEL_FACTORIES[model_name](cfg, odds)
    preds = walk_forward_backtest(
        model,
        features,
        min_history_matches=cfg.eval.min_history_matches,
        cv_window=cfg.eval.cv_window,
        refit_every=cfg.model.refit_every,
    )
    if preds.empty:
        logger.warning("No predictions produced for model %s", model_name)
        return preds

    preds = validate_table(preds, "predictions")
    artifact = f"predictions_{model_name}"
    write_table(
        preds,
        artifact,
        PROCESSED,
        sort_by=["date_utc", "match_id"],
        plan=PLAN,
        schema_name="predictions",
    )
    logger.info("Walk-forward backtest %s: %d predictions", model_name, len(preds))
    return preds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest for one model")
    parser.add_argument("--model", required=True, help=f"Model name: {', '.join(sorted(MODEL_FACTORIES))}")
    args = parser.parse_args(argv)

    get_config.cache_clear()
    cfg = load_config()
    set_seeds(cfg.seed)
    run_backtest(args.model, cfg)


if __name__ == "__main__":
    main()
