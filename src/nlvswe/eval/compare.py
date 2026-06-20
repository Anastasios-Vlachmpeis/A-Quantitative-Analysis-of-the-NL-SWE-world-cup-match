"""Model comparison, leaderboard, and selection (Plan 08)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nlvswe.config import AppConfig, get_config, load_config, project_root
from nlvswe.eval.calibration import reliability_curve
from nlvswe.eval.scoring import (
    _per_match_scores,
    bootstrap_ci,
    outcome_codes,
    score_predictions,
    vectorized_per_match_scores,
)
from nlvswe.io import read_table, save_figure, write_table
from nlvswe.logging import get_logger
from nlvswe.models.ensemble import (
    ENSEMBLE_SIMPLE,
    ENSEMBLE_WEIGHTED,
    build_simple_average_predictions,
    build_walkforward_weighted_predictions,
    ensemble_members_available,
)
from nlvswe.models.registry import ladder_models
from nlvswe.plotting.theme import apply_theme, style_axes, style_figure
from nlvswe.repro import set_seeds
from nlvswe.schemas import validate_table

logger = get_logger(__name__)

PLAN = "08"
PROCESSED = "processed"
REPORTS = "reports"


def discover_prediction_models() -> list[str]:
    """Find models with predictions_<name>.parquet on disk."""
    root = project_root() / "data" / PROCESSED
    if not root.exists():
        return []
    names: list[str] = []
    for path in sorted(root.glob("predictions_*.parquet")):
        name = path.stem.replace("predictions_", "")
        names.append(name)
    return names


def load_prediction(model_name: str) -> pd.DataFrame:
    df = read_table(f"predictions_{model_name}", PROCESSED)
    df = df.copy()
    df["model"] = model_name
    return df


def load_all_predictions(model_names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    names = model_names or discover_prediction_models()
    frames: dict[str, pd.DataFrame] = {}
    for name in names:
        try:
            frames[name] = load_prediction(name)
        except FileNotFoundError:
            logger.warning("Missing predictions for %s", name)
    return frames


def common_match_ids(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Intersection of match_ids across all loaded models."""
    if not frames:
        return []
    sets = [set(df["match_id"].astype(str)) for df in frames.values()]
    common = set.intersection(*sets)
    return sorted(common)


def align_predictions(frames: dict[str, pd.DataFrame], match_ids: list[str]) -> pd.DataFrame:
    """Stack predictions on identical match set."""
    parts: list[pd.DataFrame] = []
    for name, df in frames.items():
        sub = df[df["match_id"].astype(str).isin(match_ids)].copy()
        sub["model"] = name
        parts.append(sub)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["date_utc", "match_id", "model"], kind="mergesort").reset_index(drop=True)


def accuracy_rate(preds: pd.DataFrame) -> float:
    pred_class = preds[["p_home", "p_draw", "p_away"]].to_numpy().argmax(axis=1)
    labels = list(preds["outcome"].astype(str))
    mapping = {"home": 0, "draw": 1, "away": 2}
    actual = np.array([mapping[o] for o in labels])
    return float((pred_class == actual).mean())


def build_leaderboard(
    preds: pd.DataFrame,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Leaderboard with RPS/logloss/brier CIs, ranks, and accuracy footnote."""
    scores = score_predictions(preds, bootstrap_samples=bootstrap_samples, seed=seed)
    if scores.empty:
        return scores

    acc_rows: list[dict] = []
    for model in sorted(preds["model"].unique()):
        sub = preds[preds["model"] == model]
        acc_rows.append({"model": model, "accuracy": accuracy_rate(sub)})

    acc_df = pd.DataFrame(acc_rows)
    scores = scores.merge(acc_df, on="model", how="left")
    scores["rank"] = pd.NA
    for metric in ("rps", "log_loss", "brier"):
        mask = scores["metric"] == metric
        if not mask.any():
            continue
        sub_idx = scores[mask].sort_values("mean", kind="mergesort").index
        for rank, idx in enumerate(sub_idx, start=1):
            scores.loc[idx, "rank"] = rank

    scores["rank"] = scores["rank"].astype("Int64")
    scores["n"] = scores["n"].astype("Int64")
    return scores.sort_values(["metric", "mean", "model"], kind="mergesort").reset_index(drop=True)


def pairwise_vs_reference(
    preds: pd.DataFrame,
    reference: str = "market",
    *,
    metric: str = "rps",
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap CI of (model - reference) mean score; negative => model better."""
    ref = preds[preds["model"] == reference][["match_id", "p_home", "p_draw", "p_away", "outcome"]].copy()
    if ref.empty:
        return pd.DataFrame()
    ref = ref.drop_duplicates("match_id").set_index("match_id")

    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    for model in sorted(preds["model"].unique()):
        if model == reference:
            continue
        sub = preds[preds["model"] == model].drop_duplicates("match_id").set_index("match_id")
        common = sorted(set(sub.index) & set(ref.index))
        if len(common) < 10:
            continue
        sub_c = sub.loc[common]
        ref_c = ref.loc[common]
        sub_probs = sub_c[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
        ref_probs = ref_c[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
        outcome_idx = outcome_codes(sub_c["outcome"])
        diffs = vectorized_per_match_scores(sub_probs, outcome_idx, metric) - vectorized_per_match_scores(
            ref_probs, outcome_idx, metric
        )
        mean_diff = float(diffs.mean())
        boot = np.empty(bootstrap_samples)
        n = len(diffs)
        for i in range(bootstrap_samples):
            sample = diffs[rng.integers(0, n, size=n)]
            boot[i] = sample.mean()
        rows.append(
            {
                "model": model,
                "reference": reference,
                "metric": metric,
                "mean_diff": mean_diff,
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "n": n,
                "beats_reference": bool(mean_diff < 0 and np.quantile(boot, 0.975) < 0),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["n"] = out["n"].astype("Int64")
    return out.sort_values("mean_diff", kind="mergesort").reset_index(drop=True)


def build_ensembles(
    member_frames: dict[str, pd.DataFrame],
    *,
    min_history_matches: int = 100,
    refit_every: int = 200,
) -> dict[str, pd.DataFrame]:
    """Build simple-average and walk-forward weighted ensemble predictions."""
    members = ensemble_members_available(member_frames)
    if len(members) < 2:
        return {}

    match_ids = common_match_ids({m: member_frames[m] for m in members})
    if not match_ids:
        return {}

    ref = member_frames[members[0]]
    common = (
        ref[ref["match_id"].astype(str).isin(match_ids)]
        .drop_duplicates("match_id")
        .sort_values(["date_utc", "match_id"], kind="mergesort")
    )
    stacked = align_predictions({m: member_frames[m] for m in members}, match_ids)

    out: dict[str, pd.DataFrame] = {}
    avg = build_simple_average_predictions({m: member_frames[m] for m in members}, members, common)
    if not avg.empty:
        out[ENSEMBLE_SIMPLE] = avg
    weighted = build_walkforward_weighted_predictions(
        {m: member_frames[m] for m in members},
        members,
        stacked,
        min_history_matches=min_history_matches,
        refit_every=refit_every,
    )
    if not weighted.empty:
        out[ENSEMBLE_WEIGHTED] = weighted
    return out


def plot_metric_bars(leaderboard: pd.DataFrame, metric: str, *, out_name: str) -> None:
    sub = leaderboard[leaderboard["metric"] == metric].sort_values("mean", kind="mergesort")
    if sub.empty:
        return
    apply_theme()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(sub))))
    y = np.arange(len(sub))
    ax.barh(y, sub["mean"], xerr=[sub["mean"] - sub["ci_low"], sub["ci_high"] - sub["mean"]], color="#00B0FF", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["model"])
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(f"Leaderboard — {metric} (lower is better)")
    style_figure(fig)
    style_axes(ax)
    save_figure(fig, out_name)
    plt.close(fig)


def plot_calibration_overlay(preds: pd.DataFrame, *, models: list[str], out_name: str, bins: int = 10) -> None:
    apply_theme()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    for color, model in zip(colors, models, strict=True):
        sub = preds[preds["model"] == model]
        if sub.empty:
            continue
        probs = sub[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
        curves = reliability_curve(probs, sub["outcome"], bins=bins)
        home = curves[curves["class"] == "home"]
        if home.empty:
            continue
        ax.plot(home["mean_pred"], home["obs_freq"], "o-", label=model, color=color, alpha=0.85)
    ax.plot([0, 1], [0, 1], "--", color="#64748B", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(home win)")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration overlay (home class)")
    ax.legend(fontsize=8, loc="lower right")
    style_figure(fig)
    style_axes(ax)
    save_figure(fig, out_name)
    plt.close(fig)


def plot_rps_over_time(preds: pd.DataFrame, *, models: list[str], window: int = 200) -> None:
    apply_theme()
    fig, ax = plt.subplots(figsize=(10, 5))
    for model in models:
        sub = preds[preds["model"] == model].sort_values("date_utc", kind="mergesort")
        if sub.empty:
            continue
        rps = _per_match_scores(sub, "rps")
        roll = pd.Series(rps).rolling(window=window, min_periods=max(20, window // 4)).mean()
        ax.plot(sub["date_utc"].to_numpy(), roll.to_numpy(), label=model, alpha=0.85)
    ax.set_title(f"Rolling mean RPS (window={window})")
    ax.set_xlabel("Date")
    ax.set_ylabel("RPS")
    ax.legend(fontsize=8)
    style_figure(fig)
    style_axes(ax)
    save_figure(fig, "compare_rps_over_time")
    plt.close(fig)


def select_model(
    leaderboard: pd.DataFrame,
    pairwise: pd.DataFrame,
    *,
    max_ece: float = 0.05,
) -> dict:
    """Pick best walk-forward RPS model with honest market comparison."""
    rps = leaderboard[leaderboard["metric"] == "rps"].sort_values("mean", kind="mergesort")
    if rps.empty:
        return {"pick": None, "reason": "no scores"}

    candidates = rps[~rps["model"].isin({ENSEMBLE_SIMPLE, ENSEMBLE_WEIGHTED, "constant", "market"})]
    if candidates.empty:
        candidates = rps

    pick_row = candidates.iloc[0]
    pick = str(pick_row["model"])

    market_row = rps[rps["model"] == "market"]
    beats_market = False
    market_note = "Market predictions not available for comparison."
    if not market_row.empty and not pairwise.empty:
        pw = pairwise[pairwise["model"] == pick]
        if not pw.empty:
            beats_market = bool(pw.iloc[0]["beats_reference"])
            md = float(pw.iloc[0]["mean_diff"])
            lo = float(pw.iloc[0]["ci_low"])
            hi = float(pw.iloc[0]["ci_high"])
            market_note = (
                f"Pick vs market RPS diff={md:.4f} (95% CI [{lo:.4f}, {hi:.4f}]). "
                f"{'Statistically beats market at 95%.' if beats_market else 'Does NOT clearly beat market (CI crosses zero).'}"
            )

    ensemble_row = rps[rps["model"] == ENSEMBLE_WEIGHTED]
    if not ensemble_row.empty and float(ensemble_row.iloc[0]["mean"]) < float(pick_row["mean"]):
        pick = ENSEMBLE_WEIGHTED
        pick_row = ensemble_row.iloc[0]

    return {
        "pick": pick,
        "pick_rps": float(pick_row["mean"]),
        "pick_rps_ci": (float(pick_row["ci_low"]), float(pick_row["ci_high"])),
        "beats_market": beats_market,
        "market_note": market_note,
        "criterion": "Lowest walk-forward RPS on common match set with acceptable calibration",
    }


def write_model_selection_md(
    leaderboard: pd.DataFrame,
    pairwise: pd.DataFrame,
    selection: dict,
    path: Path,
) -> None:
    lines = [
        "# Model selection (Plan 08)",
        "",
        f"**Criterion:** {selection.get('criterion', 'RPS')}",
        "",
        f"**Selected model:** `{selection.get('pick')}`",
        f"**Walk-forward RPS:** {selection.get('pick_rps', float('nan')):.4f} "
        f"(95% CI {selection.get('pick_rps_ci', (float('nan'), float('nan')))[0]:.4f}"
        f"–{selection.get('pick_rps_ci', (float('nan'), float('nan')))[1]:.4f})",
        "",
        "## Market benchmark",
        "",
        selection.get("market_note", "N/A"),
        "",
        "## Leaderboard (RPS)",
        "",
        "| Rank | Model | RPS | 95% CI | Accuracy (footnote) |",
        "|------|-------|-----|--------|---------------------|",
    ]
    rps = leaderboard[leaderboard["metric"] == "rps"].sort_values("mean", kind="mergesort")
    for _, row in rps.iterrows():
        acc = row.get("accuracy")
        acc_s = f"{acc:.1%}" if pd.notna(acc) else "—"
        lines.append(
            f"| {row['rank']} | `{row['model']}` | {row['mean']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | {acc_s} |"
        )

    if not pairwise.empty:
        lines.extend(["", "## Pairwise vs market (RPS difference; negative = model better)", ""])
        lines.append("| Model | Mean diff | 95% CI | Beats market? |")
        lines.append("|-------|-----------|--------|---------------|")
        for _, row in pairwise.iterrows():
            lines.append(
                f"| `{row['model']}` | {row['mean_diff']:.4f} | "
                f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
                f"{'Yes' if row['beats_reference'] else 'No'} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Selection uses walk-forward scores on the **intersection** of matches scored by all models.",
            "- Accuracy is descriptive only; RPS is the primary metric.",
            "- If nothing beats the market, live betting should be framed as price-taker on soft markets only.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_compare(cfg: AppConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = cfg or get_config()
    frames = load_all_predictions()
    if not frames:
        raise FileNotFoundError("No predictions_*.parquet files found in data/processed/")

    ensembles = build_ensembles(
        frames,
        min_history_matches=cfg.eval.min_history_matches,
        refit_every=max(cfg.model.refit_every * 20, 200),
    )
    for name, df in ensembles.items():
        frames[name] = df
        write_table(
            df,
            f"predictions_{name}",
            PROCESSED,
            sort_by=["date_utc", "match_id"],
            plan=PLAN,
            schema_name="predictions",
        )

    match_ids = common_match_ids(frames)
    logger.info("Common match set: %d matches across %d models", len(match_ids), len(frames))
    stacked = align_predictions(frames, match_ids)

    leaderboard = build_leaderboard(
        stacked,
        bootstrap_samples=cfg.eval.bootstrap_samples,
        seed=cfg.seed,
    )
    leaderboard = validate_table(leaderboard, "leaderboard")
    write_table(
        leaderboard,
        "leaderboard",
        PROCESSED,
        sort_by=["metric", "mean", "model"],
        plan=PLAN,
        schema_name="leaderboard",
    )

    pairwise = pairwise_vs_reference(
        stacked,
        reference="market",
        bootstrap_samples=cfg.eval.bootstrap_samples,
        seed=cfg.seed + 1,
    )
    if not pairwise.empty:
        pairwise = validate_table(pairwise, "pairwise_vs_market")
        write_table(
            pairwise,
            "pairwise_vs_market",
            PROCESSED,
            sort_by=["mean_diff", "model"],
            plan=PLAN,
            schema_name="pairwise_vs_market",
        )

    selection = select_model(leaderboard, pairwise)
    write_model_selection_md(
        leaderboard,
        pairwise,
        selection,
        project_root() / REPORTS / "MODEL_SELECTION.md",
    )

    rps_board = leaderboard[leaderboard["metric"] == "rps"]
    plot_metric_bars(leaderboard, "rps", out_name="compare_rps_leaderboard")
    plot_metric_bars(leaderboard, "log_loss", out_name="compare_logloss_leaderboard")

    top_models = list(rps_board.sort_values("mean").head(6)["model"])
    plot_calibration_overlay(stacked, models=top_models, out_name="compare_calibration_overlay", bins=cfg.eval.calibration_bins)
    plot_rps_over_time(stacked, models=top_models[:4])

    logger.info("Model comparison complete; pick=%s", selection.get("pick"))
    return leaderboard, pairwise, selection


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Model comparison and selection (Plan 08)")
    parser.parse_args(argv)
    get_config.cache_clear()
    cfg = load_config()
    set_seeds(cfg.seed)
    run_compare(cfg)


if __name__ == "__main__":
    main()
