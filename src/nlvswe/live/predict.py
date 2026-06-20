"""One-shot pre-match pipeline for NL vs SWE: features, prediction, slip, optional git freeze."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nlvswe.betting.induction import induce_row_markets, simulate_scorelines
from nlvswe.betting.slip import build_live_bet_slip
from nlvswe.config import AppConfig, get_config, load_config, project_root
from nlvswe.eval.backtest import prediction_to_row
from nlvswe.eval.compare import align_predictions, common_match_ids
from nlvswe.features.build import _prepare_features, build_features, features_for_match
from nlvswe.io import read_table, save_figure, write_table
from nlvswe.logging import get_logger
from nlvswe.models.base import MatchPrediction, validate_prediction
from nlvswe.models.data import load_model_features
from nlvswe.models.ensemble import (
    ENSEMBLE_WEIGHTED,
    ensemble_members_available,
    fit_inverse_rps_weights,
    weighted_probs,
)
from nlvswe.models.registry import create_model
from nlvswe.plotting.theme import apply_theme, style_axes, style_figure
from nlvswe.repro import git_commit, set_seeds
from nlvswe.schemas import validate_table

logger = get_logger(__name__)

PLAN = "10"
LIVE_SUBDIR = "processed/live"
INTERIM = "interim"
PROCESSED = "processed"
REPORTS = "reports"
CORPUS_INTERNATIONAL = "international"


def selected_model_name() -> str:
    path = project_root() / REPORTS / "MODEL_SELECTION.md"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("**Selected model:**"):
                return line.split("`")[1]
    return ENSEMBLE_WEIGHTED


def kickoff_utc(cfg: AppConfig) -> pd.Timestamp:
    ts = pd.Timestamp(cfg.target_match.kickoff_utc)
    return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")


def resolve_target_match(matches: pd.DataFrame, cfg: AppConfig) -> pd.Series:
    home = cfg.target_match.home.lower().replace(" ", "_")
    away = cfg.target_match.away.lower().replace(" ", "_")
    mask = (matches["home_team_id"] == home) & (matches["away_team_id"] == away)
    sub = matches[mask].sort_values("date_utc", kind="mergesort")
    if sub.empty:
        raise ValueError(f"Target match not found in matches table: {cfg.target_match.home} vs {cfg.target_match.away}")
    scheduled = sub[sub["status"].astype(str) == "scheduled"]
    return scheduled.iloc[-1] if not scheduled.empty else sub.iloc[-1]


def refresh_live_inputs(cfg: AppConfig) -> None:
    """Refresh FIFA, results, and live odds snapshots before I lock the pre-match call."""
    try:
        from nlvswe.data import acquire, clean
    except ImportError as exc:
        logger.warning("Data refresh skipped (acquire/clean unavailable): %s", exc)
        return

    logger.info("Refreshing results and FIFA ranking raw data")
    acquire.main(["--source", "results"])
    acquire.main(["--source", "fifa_rank"])
    try:
        acquire.main(["--source", "odds_live"])
    except Exception as exc:
        logger.warning("Live odds snapshot not refreshed: %s", exc)

    logger.info("Rebuilding interim matches, ratings, odds")
    clean.main(["--table", "matches"])
    clean.main(["--table", "ratings"])
    clean.main(["--table", "odds"])


def build_live_features_row(cfg: AppConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Same features_for_match path as the backtest, then sanity-check against the batch table."""
    matches = read_table("matches", INTERIM)
    ratings = read_table("ratings", INTERIM)
    conditions = read_table("conditions", INTERIM)
    venues = read_table("venues", INTERIM)
    target = resolve_target_match(matches, cfg)
    kickoff = kickoff_utc(cfg)

    intl = matches[matches["corpus"] == CORPUS_INTERNATIONAL].sort_values(
        ["date_utc", "match_id"], kind="mergesort"
    )
    feat = features_for_match(
        target,
        intl,
        ratings,
        conditions,
        venues,
        form_windows=list(cfg.model.form_windows),
    )
    feat["date_utc"] = kickoff

    batch = build_features(
        intl,
        ratings,
        conditions,
        venues,
        form_windows=list(cfg.model.form_windows),
        corpus=CORPUS_INTERNATIONAL,
    )
    batch_row = batch[batch["match_id"].astype(str) == str(target["match_id"])]
    if not batch_row.empty:
        br = batch_row.iloc[0]
        skip = {"result_1x2", "home_goals", "away_goals", "total_goals"}
        for col in batch.columns:
            if col in skip:
                continue
            bv, lv = br[col], feat.get(col)
            if pd.isna(bv) and pd.isna(lv):
                continue
            if bv != lv:
                logger.warning("Live/batch feature mismatch on %s: batch=%s live=%s", col, bv, lv)

    df = _prepare_features(pd.DataFrame([feat]))
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True).astype("datetime64[ns, UTC]")
    return validate_table(df, "features"), target


def _load_member_prediction_frames() -> dict[str, pd.DataFrame]:
    root = project_root() / "data" / PROCESSED
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(root.glob("predictions_*.parquet")):
        name = path.stem.replace("predictions_", "")
        if name.startswith("ensemble"):
            continue
        try:
            frames[name] = read_table(f"predictions_{name}", PROCESSED)
        except FileNotFoundError:
            pass
    return frames


def predict_ensemble_weighted(
    cfg: AppConfig,
    target_row: pd.Series,
    train: pd.DataFrame,
    odds: pd.DataFrame,
) -> MatchPrediction:
    """Fit each member on pre-kickoff history, blend with inverse-RPS weights from walk-forward preds."""
    member_frames = _load_member_prediction_frames()
    members = ensemble_members_available(member_frames)
    if len(members) < 2:
        raise ValueError("Need at least two member models for ensemble_weighted live prediction")

    kickoff = kickoff_utc(cfg)
    match_ids = common_match_ids({m: member_frames[m] for m in members})
    stacked = align_predictions({m: member_frames[m] for m in members}, match_ids)
    hist = stacked[pd.to_datetime(stacked["date_utc"], utc=True) < kickoff]
    weights = fit_inverse_rps_weights(hist, members)

    parts: list[dict[str, float]] = []
    scoreline = None
    for name in members:
        model = create_model(name, cfg, odds)
        model.fit(train)
        member_pred = model.predict(target_row)
        parts.append(member_pred.probs_1x2)
        if scoreline is None and member_pred.scoreline is not None:
            scoreline = member_pred.scoreline

    probs = weighted_probs(parts, [weights[m] for m in members])

    pred = MatchPrediction(
        match_id=str(target_row["match_id"]),
        scoreline=scoreline,
        probs_1x2=probs,
    )
    validate_prediction(pred)
    return pred


def predict_live(
    model_name: str,
    cfg: AppConfig,
    target_row: pd.Series,
    train: pd.DataFrame,
    odds: pd.DataFrame,
) -> MatchPrediction:
    if model_name == ENSEMBLE_WEIGHTED:
        return predict_ensemble_weighted(cfg, target_row, train, odds)
    model = create_model(model_name, cfg, odds)
    model.fit(train)
    pred = model.predict(target_row)
    validate_prediction(pred)
    return pred


def induce_live_markets(pred: MatchPrediction, model_name: str) -> pd.DataFrame:
    rows = induce_row_markets(
        match_id=pred.match_id,
        model=model_name,
        p_home=pred.probs_1x2["home"],
        p_draw=pred.probs_1x2["draw"],
        p_away=pred.probs_1x2["away"],
        scoreline=pred.scoreline,
    )
    df = pd.DataFrame(rows)
    df["model_prob"] = df["model_prob"].astype("float64")
    return validate_table(df, "market_probs")


def plot_live_figures(pred: MatchPrediction, model_name: str, cfg: AppConfig) -> None:
    if pred.scoreline is None:
        logger.warning("No scoreline matrix; skipping live scoreline/sim figures")
        return
    apply_theme()
    mat = np.asarray(pred.scoreline, dtype=float)
    side = min(8, mat.shape[0])
    sub = mat[:side, :side]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sub, cmap="Blues", origin="lower")
    fig.colorbar(im, ax=ax, label="P(score)")
    ax.set_xlabel("Away goals")
    ax.set_ylabel("Home goals")
    ax.set_title(f"Live scoreline — {model_name}")
    style_figure(fig)
    style_axes(ax)
    save_figure(fig, "live_scoreline")
    plt.close(fig)

    rng = np.random.default_rng(cfg.seed)
    sims = simulate_scorelines(mat, min(cfg.model.mc_samples, 5000), rng)
    home, away = sims[:, 0], sims[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(home + away, bins=20, color="#00B0FF", edgecolor="#1E293B")
    axes[0].set_title("Simulated total goals")
    axes[1].hist(home - away, bins=20, color="#00E676", edgecolor="#1E293B")
    axes[1].set_title("Simulated goal difference")
    fig.suptitle(f"Live Monte Carlo — {model_name}")
    style_figure(fig)
    for ax in axes:
        style_axes(ax)
    save_figure(fig, "live_sim_scores")
    plt.close(fig)


def plot_live_edges(bet_slip: pd.DataFrame) -> None:
    bets = bet_slip[bet_slip["status"] == "bet"]
    if bets.empty:
        return
    apply_theme()
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(bets))))
    labels = [f"{r.market}:{r.selection}@{r.bookmaker}" for r in bets.itertuples()]
    y = np.arange(len(bets))
    ax.barh(y, bets["ev"], color="#00E676", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(get_config().betting.min_edge, color="#64748B", linestyle="--", linewidth=1)
    ax.set_xlabel("Expected value (EV)")
    ax.set_title("Live +EV selections")
    style_figure(fig)
    style_axes(ax)
    save_figure(fig, "live_edges")
    plt.close(fig)


def write_prediction_md(
    *,
    cfg: AppConfig,
    model_name: str,
    features: pd.DataFrame,
    pred: MatchPrediction,
    market_probs: pd.DataFrame,
    bet_slip: pd.DataFrame,
    commit_hash: str,
    path: Path,
) -> None:
    kickoff = kickoff_utc(cfg)
    tm = cfg.target_match
    lines = [
        "# Pre-match prediction — Netherlands vs Sweden",
        "",
        f"**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}",
        f"**Git commit:** `{commit_hash}`",
        f"**Kickoff (UTC):** {kickoff.isoformat()}",
        f"**Model:** `{model_name}`",
        "",
        "## 1X2 probabilities",
        "",
        f"| Home ({tm.home}) | Draw | Away ({tm.away}) |",
        "|---|---|---|",
        f"| {pred.probs_1x2['home']:.1%} | {pred.probs_1x2['draw']:.1%} | {pred.probs_1x2['away']:.1%} |",
        "",
        "## Key features (pre-kickoff)",
        "",
    ]
    row = features.iloc[0]
    for col in ("home_elo_pre", "away_elo_pre", "elo_diff", "fifa_points_diff", "form_ppg_diff_5", "neutral"):
        if col in row.index:
            val = row[col]
            lines.append(f"- **{col}:** {val}")

    lines.extend(["", "## Bet slip", ""])
    bets = bet_slip[bet_slip["status"] == "bet"]
    if bets.empty:
        reason = bet_slip.iloc[0]["rationale"]
        lines.append(f"**No bet.** {reason}")
    else:
        lines.append("| Book | Market | Selection | Odds | Model p | De-vig p | Edge | EV | Stake |")
        lines.append("|------|--------|-----------|------|---------|----------|------|-----|-------|")
        for _, b in bets.iterrows():
            lines.append(
                f"| {b['bookmaker']} | {b['market']} | {b['selection']} | {b['odds_taken']:.2f} | "
                f"{b['model_prob']:.3f} | {b['book_prob_devig']:.3f} | {b['edge']:.3f} | "
                f"{b['ev']:.3f} | ${b['stake']:.2f} |"
            )
        lines.extend(["", "### Rationale", ""])
        for _, b in bets.iterrows():
            lines.append(f"- {b['rationale']}")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Predictions use only information available before kickoff.",
            "- Closing odds captured after this run will be used for CLV in post-match analysis.",
            "- A positive EV bet is a recommendation on a mock bankroll, not financial advice.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_prematch_freeze(cfg: AppConfig, *, message: str | None = None) -> str:
    """Commit live artifacts and tag prematch-freeze (only run this before kickoff)."""
    root = project_root()
    msg = message or f"Pre-match freeze: {cfg.target_match.home} vs {cfg.target_match.away}"
    paths = [
        "data/processed/live",
        "reports/PREDICTION.md",
        "reports/figures/live_scoreline.png",
        "reports/figures/live_sim_scores.png",
        "reports/figures/live_edges.png",
    ]
    for rel in paths:
        p = root / rel
        if p.exists():
            subprocess.run(["git", "add", rel], cwd=root, check=True, timeout=30)
    subprocess.run(["git", "commit", "-m", msg], cwd=root, check=True, timeout=60)
    commit = git_commit()
    tag = "prematch-freeze"
    subprocess.run(["git", "tag", "-f", tag, commit], cwd=root, check=True, timeout=30)
    logger.info("Created tag %s at %s", tag, commit)
    return commit


def run_live_predict(
    cfg: AppConfig | None = None,
    *,
    model_name: str | None = None,
    refresh: bool = False,
    freeze: bool = False,
) -> dict:
    cfg = cfg or get_config()
    if cfg.scope != "live":
        logger.warning("config scope=%s (expected live for production freeze)", cfg.scope)

    model_name = model_name or selected_model_name()
    kickoff = kickoff_utc(cfg)
    now = pd.Timestamp(datetime.now(timezone.utc))
    if now >= kickoff:
        logger.warning("Current time is at or after kickoff — run is for dry-run/recovery only")

    if refresh:
        refresh_live_inputs(cfg)

    features, target = build_live_features_row(cfg)
    write_table(features, "features", LIVE_SUBDIR, sort_by=["match_id"], plan=PLAN, schema_name="features")

    train = load_model_features()
    train = train[train["result_1x2"].notna()]
    train = train[pd.to_datetime(train["date_utc"], utc=True) < kickoff].reset_index(drop=True)

    try:
        odds = read_table("odds", INTERIM)
    except FileNotFoundError:
        odds = pd.DataFrame()

    target_feat = features.iloc[0]
    pred = predict_live(model_name, cfg, target_feat, train, odds)
    pred_row = prediction_to_row(pred, model=model_name, match=target_feat)
    preds_df = pd.DataFrame([pred_row]).drop(columns=["outcome"], errors="ignore")
    preds_df["date_utc"] = pd.to_datetime(preds_df["date_utc"], utc=True).astype("datetime64[ns, UTC]")
    if "has_scoreline" in preds_df.columns:
        preds_df["has_scoreline"] = preds_df["has_scoreline"].astype("bool")
    preds_df = validate_table(preds_df, "live_predictions")
    write_table(preds_df, "predictions", LIVE_SUBDIR, sort_by=["match_id"], plan=PLAN, schema_name="live_predictions")

    market_probs = induce_live_markets(pred, model_name)
    write_table(
        market_probs,
        "market_probs",
        LIVE_SUBDIR,
        sort_by=["market", "selection"],
        plan=PLAN,
        schema_name="market_probs",
    )

    bet_slip = build_live_bet_slip(
        market_probs,
        odds,
        cfg,
        match_id=str(target["match_id"]),
        model=model_name,
        kickoff=kickoff,
    )
    bet_slip = validate_table(bet_slip, "bet_slip")
    write_table(bet_slip, "bet_slip", LIVE_SUBDIR, sort_by=["status", "ev"], plan=PLAN, schema_name="bet_slip")

    plot_live_figures(pred, model_name, cfg)
    plot_live_edges(bet_slip)

    commit_hash = git_commit()
    write_prediction_md(
        cfg=cfg,
        model_name=model_name,
        features=features,
        pred=pred,
        market_probs=market_probs,
        bet_slip=bet_slip,
        commit_hash=commit_hash,
        path=project_root() / REPORTS / "PREDICTION.md",
    )

    if freeze:
        commit_hash = create_prematch_freeze(cfg)
        write_prediction_md(
            cfg=cfg,
            model_name=model_name,
            features=features,
            pred=pred,
            market_probs=market_probs,
            bet_slip=bet_slip,
            commit_hash=commit_hash,
            path=project_root() / REPORTS / "PREDICTION.md",
        )

    logger.info("Live prediction complete; model=%s bets=%d", model_name, int((bet_slip["status"] == "bet").sum()))
    return {
        "model": model_name,
        "match_id": str(target["match_id"]),
        "commit_hash": commit_hash,
        "n_bets": int((bet_slip["status"] == "bet").sum()),
        "kickoff_utc": kickoff.isoformat(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pre-match prediction for NL vs SWE, optional git freeze")
    parser.add_argument("--model", default=None, help="Model (default: read from MODEL_SELECTION.md)")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch raw data and rebuild interim tables")
    parser.add_argument("--freeze", action="store_true", help="Commit artifacts and tag prematch-freeze")
    args = parser.parse_args(argv)

    get_config.cache_clear()
    cfg = load_config()
    set_seeds(cfg.seed)
    result = run_live_predict(cfg, model_name=args.model, refresh=args.refresh, freeze=args.freeze)
    print(f"Live predict done: model={result['model']} commit={result['commit_hash']} bets={result['n_bets']}")


if __name__ == "__main__":
    main()
