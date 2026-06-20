"""Run the betting backtest on club odds (machinery validation, not intl profit claims)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nlvswe.betting.bankroll import monte_carlo_bankroll, roi_with_ci, simulate_realized_path
from nlvswe.betting.clv import aggregate_clv, clv_odds_ratio, clv_prob_diff
from nlvswe.betting.induction import markets_from_1x2
from nlvswe.betting.strategy import (
    bulk_opening_consensus_1x2,
    devigged_book_probs,
    expected_value,
    settle_bet,
    stake_amount,
    vectorized_mult_devig_1x2,
)
from nlvswe.config import AppConfig, get_config, load_config, project_root
from nlvswe.io import read_table, save_figure, write_table
from nlvswe.logging import get_logger
from nlvswe.models.ensemble import ENSEMBLE_SIMPLE, ENSEMBLE_WEIGHTED
from nlvswe.plotting.theme import apply_theme, style_axes, style_figure
from nlvswe.repro import set_seeds
from nlvswe.schemas import validate_table

logger = get_logger(__name__)

PLAN = "09"
PROCESSED = "processed"
INTERIM = "interim"
REPORTS = "reports"

CORPUS_CLUB = "club"
CORPUS_INTL = "international"


def discover_strategy_models() -> list[str]:
    root = project_root() / "data" / PROCESSED
    names: list[str] = []
    if root.exists():
        for path in sorted(root.glob("market_probs_*.parquet")):
            names.append(path.stem.replace("market_probs_", ""))
    if "market" not in names:
        names.append("market")
    return sorted(set(names))


def _match_results(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    scored = df[df["home_goals"].notna() & df["away_goals"].notna()].copy()
    hg = scored["home_goals"].astype(int)
    ag = scored["away_goals"].astype(int)
    scored["result_1x2"] = np.select(
        [hg > ag, hg < ag],
        ["home", "away"],
        default="draw",
    )
    scored["result_1x2"] = scored["result_1x2"].astype(str)
    return scored


def _static_1x2_from_predictions(model: str) -> dict[str, float] | None:
    try:
        preds = read_table(f"predictions_{model}", PROCESSED)
    except FileNotFoundError:
        return None
    if preds.empty:
        return None
    ph = float(preds["p_home"].mean())
    pd_ = float(preds["p_draw"].mean())
    pa = float(preds["p_away"].mean())
    total = ph + pd_ + pa
    if total <= 0:
        return None
    return {"home": ph / total, "draw": pd_ / total, "away": pa / total}


def _consensus_to_prob_rows(model: str, consensus: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in consensus.iterrows():
        mid = str(row["match_id"])
        for mrow in markets_from_1x2(float(row["home"]), float(row["draw"]), float(row["away"])):
            if mrow["market"] != "1x2":
                continue
            rows.append(
                {
                    "match_id": mid,
                    "model": model,
                    "market": "1x2",
                    "selection": mrow["selection"],
                    "model_prob": mrow["model_prob"],
                    "method": "derived",
                }
            )
    return pd.DataFrame(rows)


def build_club_model_probs(
    model: str,
    odds: pd.DataFrame,
    matches: pd.DataFrame,
    cfg: AppConfig,
) -> pd.DataFrame:
    """Build 1X2 model probability rows for club corpus backtest."""
    club_odds = odds[odds["corpus"] == CORPUS_CLUB]
    club_ids = set(matches[matches["corpus"] == CORPUS_CLUB]["match_id"].astype(str))

    if model == "market":
        consensus = bulk_opening_consensus_1x2(club_odds, devig_method=cfg.betting.devig_method)
        consensus = consensus[consensus["match_id"].isin(club_ids)]
        return _consensus_to_prob_rows(model, consensus)

    try:
        stored = read_table(f"market_probs_{model}", PROCESSED)
        hit = stored[
            (stored["match_id"].astype(str).isin(club_ids)) & (stored["market"].astype(str) == "1x2")
        ]
        if not hit.empty:
            return hit.copy()
    except FileNotFoundError:
        pass

    static = _static_1x2_from_predictions(model)
    if static is None:
        logger.warning("No club model probs for %s; skipping", model)
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "match_id": "*",
                "model": model,
                "market": "1x2",
                "selection": sel,
                "model_prob": static[sel],
                "method": "derived",
            }
            for sel in ("home", "draw", "away")
        ]
    )


def _model_prob_lookup(model_probs: pd.DataFrame) -> dict[str, float]:
    """Static 1X2 lookup when probabilities do not vary by match."""
    sub = model_probs[model_probs["market"].astype(str) == "1x2"]
    if sub.empty:
        return {}
    if set(sub["match_id"].astype(str)) == {"*"}:
        row = sub.set_index("selection")["model_prob"]
        return {str(k).lower(): float(v) for k, v in row.items()}
    by_match = sub.groupby("match_id")["model_prob"].apply(lambda s: tuple(round(x, 8) for x in s.tolist()))
    if by_match.nunique() == 1:
        row = sub.drop_duplicates("selection").set_index("selection")["model_prob"]
        return {str(k).lower(): float(v) for k, v in row.items()}
    return {}


def build_bet_candidates(
    model_probs: pd.DataFrame,
    odds: pd.DataFrame,
    matches: pd.DataFrame,
    cfg: AppConfig,
    *,
    model: str,
    corpus: str = CORPUS_CLUB,
    stake_method: str = "kelly",
) -> pd.DataFrame:
    """Find +EV prematch 1X2 bets and size stakes chronologically."""
    if model_probs.empty:
        return pd.DataFrame()

    model_probs = model_probs[model_probs["market"].astype(str) == "1x2"].copy()
    odds_c = odds[(odds["corpus"] == corpus) & (~odds["is_closing"])].copy()
    odds_c = odds_c[odds_c["market"].astype(str).str.lower() == "1x2"]
    match_info = _match_results(matches)
    match_info = match_info[match_info["corpus"] == corpus][
        ["match_id", "date_utc", "home_goals", "away_goals"]
    ].drop_duplicates("match_id")
    allowed_books = {b.lower() for b in cfg.betting.bookmakers}

    if cfg.betting.devig_method == "multiplicative":
        prematch = vectorized_mult_devig_1x2(odds_c)
    else:
        prematch = (
            odds_c.sort_values(["match_id", "bookmaker", "selection", "captured_at"], kind="mergesort")
            .groupby(["match_id", "bookmaker", "selection"], sort=False)
            .tail(1)
            .reset_index(drop=True)
        )
        devig_parts: list[pd.DataFrame] = []
        for (_, _), grp in prematch.groupby(["match_id", "bookmaker"], sort=False):
            probs = devigged_book_probs(grp, market="1x2", devig_method=cfg.betting.devig_method)
            if probs is None:
                continue
            g = grp.copy()
            g["book_prob"] = g["selection"].astype(str).str.lower().map(probs)
            devig_parts.append(g)
        prematch = pd.concat(devig_parts, ignore_index=True) if devig_parts else pd.DataFrame()

    prematch = prematch[prematch["bookmaker"].str.lower().isin(allowed_books)]
    if prematch.empty:
        return pd.DataFrame()

    static_lookup = _model_prob_lookup(model_probs)
    if static_lookup:
        prematch = prematch.copy()
        prematch["mp"] = prematch["selection"].astype(str).str.lower().map(static_lookup)
    else:
        mp = model_probs.rename(columns={"selection": "sel", "model_prob": "mp"})
        prematch = prematch.copy()
        prematch["_sel"] = prematch["selection"].astype(str).str.lower()
        mp = mp.copy()
        mp["_sel"] = mp["sel"].astype(str).str.lower()
        prematch = prematch.merge(mp, left_on=["match_id", "_sel"], right_on=["match_id", "_sel"], how="inner")
        prematch = prematch.drop(columns=["sel", "_sel"], errors="ignore")

    prematch = prematch.merge(match_info, on="match_id", how="inner")
    prematch["ev"] = prematch["mp"] * prematch["decimal_odds"] - 1.0
    prematch["edge"] = prematch["mp"] - prematch["book_prob"]
    joined = prematch[prematch["ev"] > cfg.betting.min_edge].sort_values(
        ["date_utc", "match_id", "bookmaker", "selection"], kind="mergesort"
    )

    closing_odds = odds[(odds["corpus"] == corpus) & (odds["is_closing"]) & (odds["market"].astype(str) == "1x2")]
    closing_latest = (
        closing_odds.sort_values(["match_id", "bookmaker", "selection", "captured_at"], kind="mergesort")
        .groupby(["match_id", "bookmaker", "selection"], sort=False)
        .tail(1)
        .rename(columns={"decimal_odds": "odds_closing"})
    )
    # Pre-index closing odds so CLV lookup stays O(1). Scanning the full ~141k-row
    # table per bet was the backtest bottleneck (O(bets x closing rows)).
    closing_map: dict[tuple[str, str, str], float] = {
        (str(r.match_id), str(r.bookmaker), str(r.selection).lower()): float(r.odds_closing)
        for r in closing_latest.itertuples(index=False)
    }

    bankroll = float(cfg.betting.bankroll)
    flat_stake = bankroll * min(cfg.betting.max_stake_fraction, 0.01)
    match_exposure: dict[str, float] = {}
    max_match_exposure = bankroll * cfg.betting.max_stake_fraction * 3

    bet_rows: list[dict] = []
    for _, row in joined.iterrows():
        mid = str(row["match_id"])
        exposure = match_exposure.get(mid, 0.0)
        stake = stake_amount(
            bankroll,
            float(row["mp"]),
            float(row["decimal_odds"]),
            kelly_frac=cfg.betting.kelly_fraction,
            max_stake_fraction=cfg.betting.max_stake_fraction,
            method=stake_method,  # type: ignore[arg-type]
            flat_stake=flat_stake,
        )
        if stake <= 0 or exposure + stake > max_match_exposure:
            continue

        closing = closing_map.get((mid, str(row["bookmaker"]), str(row["selection"]).lower()))
        clv_or = clv_odds_ratio(float(row["decimal_odds"]), closing) if closing else float("nan")
        clv_pd = (
            clv_prob_diff(float(row["mp"]), closing, devig_method=cfg.betting.devig_method)
            if closing
            else float("nan")
        )

        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        result, pnl = settle_bet(
            market="1x2",
            selection=str(row["selection"]),
            home_goals=hg,
            away_goals=ag,
            decimal_odds=float(row["decimal_odds"]),
            stake=stake,
        )
        bankroll += pnl
        match_exposure[mid] = exposure + stake
        bet_rows.append(
            {
                "match_id": mid,
                "date_utc": row["date_utc"],
                "model": model,
                "corpus": corpus,
                "bookmaker": row["bookmaker"],
                "market": "1x2",
                "selection": str(row["selection"]),
                "model_prob": float(row["mp"]),
                "book_prob_devig": float(row["book_prob"]),
                "edge": float(row["edge"]),
                "ev": float(row["ev"]),
                "odds_taken": float(row["decimal_odds"]),
                "odds_closing": closing,
                "clv_odds_ratio": clv_or,
                "clv_prob_diff": clv_pd,
                "stake_method": stake_method,
                "stake": stake,
                "result": result,
                "pnl": pnl,
            }
        )

    if not bet_rows:
        return pd.DataFrame()
    out = pd.DataFrame(bet_rows)
    out["date_utc"] = pd.to_datetime(out["date_utc"], utc=True).astype("datetime64[ns, UTC]")
    return out.reset_index(drop=True)


def run_strategy_backtest(
    model: str,
    *,
    cfg: AppConfig | None = None,
    corpus: str = CORPUS_CLUB,
    stake_method: str = "kelly",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = cfg or get_config()
    odds = read_table("odds", INTERIM)
    matches = read_table("matches", INTERIM)

    if corpus == CORPUS_CLUB:
        model_probs = build_club_model_probs(model, odds, matches, cfg)
    else:
        try:
            model_probs = read_table(f"market_probs_{model}", PROCESSED)
            model_probs = model_probs[model_probs["market"].astype(str) == "1x2"]
        except FileNotFoundError:
            model_probs = pd.DataFrame()

    bets = build_bet_candidates(
        model_probs, odds, matches, cfg, model=model, corpus=corpus, stake_method=stake_method
    )

    if not bets.empty:
        bets = validate_table(bets, "bets")
        write_table(
            bets,
            f"bets_{model}" if stake_method == "kelly" else f"bets_{model}_flat",
            PROCESSED,
            sort_by=["date_utc", "match_id", "bookmaker"],
            plan=PLAN,
            schema_name="bets",
        )

    realized = simulate_realized_path(bets, initial_bankroll=cfg.betting.bankroll)
    mc_sims = min(cfg.model.mc_samples, 2000)
    mc_term, mc_summary = monte_carlo_bankroll(
        bets,
        initial_bankroll=cfg.betting.bankroll,
        n_sims=mc_sims,
        seed=cfg.seed + 17,
    )
    bankroll_sim = pd.concat([realized, mc_term], ignore_index=True)
    if not bankroll_sim.empty:
        bankroll_sim = validate_table(bankroll_sim, "bankroll_sim")
        write_table(
            bankroll_sim,
            f"bankroll_sim_{model}" if stake_method == "kelly" else f"bankroll_sim_{model}_flat",
            PROCESSED,
            sort_by=["path_type", "step"],
            plan=PLAN,
            schema_name="bankroll_sim",
        )

    roi, roi_lo, roi_hi = roi_with_ci(bets, bootstrap_samples=cfg.eval.bootstrap_samples, seed=cfg.seed)
    clv_stats = aggregate_clv(bets)
    summary = {
        "model": model,
        "corpus": corpus,
        "stake_method": stake_method,
        "n_bets": int(len(bets)),
        "total_staked": float(bets["stake"].sum()) if not bets.empty else 0.0,
        "total_pnl": float(bets["pnl"].sum()) if not bets.empty else 0.0,
        "roi": roi,
        "roi_ci_low": roi_lo,
        "roi_ci_high": roi_hi,
        "hit_rate": float((bets["result"] == "win").mean()) if not bets.empty else float("nan"),
        **clv_stats,
        **mc_summary,
    }
    return bets, bankroll_sim, summary


def plot_equity_curves(summaries: list[dict], bankroll_frames: dict[str, pd.DataFrame]) -> None:
    apply_theme()
    fig, ax = plt.subplots(figsize=(10, 5))
    for s in summaries:
        model = s["model"]
        df = bankroll_frames.get(model)
        if df is None or df.empty:
            continue
        path = df[df["path_type"] == "realized"]
        if path.empty:
            continue
        ax.plot(path["step"], path["bankroll"], label=model, alpha=0.85)
    ax.set_xlabel("Bet number")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("Realized equity curves (Kelly)")
    ax.legend(fontsize=8)
    style_figure(fig)
    style_axes(ax)
    save_figure(fig, "strategy_equity_curves")
    plt.close(fig)


def plot_bankroll_distribution(summaries: list[dict], bankroll_frames: dict[str, pd.DataFrame]) -> None:
    apply_theme()
    fig, ax = plt.subplots(figsize=(8, 5))
    labels: list[str] = []
    data: list[np.ndarray] = []
    for s in summaries:
        model = s["model"]
        df = bankroll_frames.get(model)
        if df is None:
            continue
        term = df[df["path_type"] == "mc_terminal"]["bankroll"].to_numpy(dtype=float)
        if term.size == 0:
            continue
        labels.append(model)
        data.append(term)
    if data:
        ax.boxplot(data, tick_labels=labels, vert=True)
        ax.set_ylabel("Final bankroll ($)")
        ax.set_title("Monte Carlo final bankroll distribution")
        style_figure(fig)
        style_axes(ax)
        save_figure(fig, "strategy_bankroll_distribution")
    plt.close(fig)


def plot_clv_distribution(bets_by_model: dict[str, pd.DataFrame]) -> None:
    apply_theme()
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, bets in bets_by_model.items():
        if bets.empty or bets["clv_odds_ratio"].isna().all():
            continue
        ax.hist(bets["clv_odds_ratio"].dropna(), bins=40, alpha=0.5, label=model)
    ax.axvline(0, color="#64748B", linestyle="--", linewidth=1)
    ax.set_xlabel("CLV (odds ratio)")
    ax.set_ylabel("Count")
    ax.set_title("CLV distribution per bet")
    ax.legend(fontsize=8)
    style_figure(fig)
    style_axes(ax)
    save_figure(fig, "strategy_clv_distribution")
    plt.close(fig)


def plot_roi_bars(summaries: list[dict]) -> None:
    apply_theme()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(summaries))))
    labels = [s["model"] for s in summaries]
    y = np.arange(len(summaries))
    means = [s.get("roi", float("nan")) for s in summaries]
    xerr = [
        [m - s.get("roi_ci_low", m) if not np.isnan(m) else 0, s.get("roi_ci_high", m) - m if not np.isnan(m) else 0]
        for m, s in zip(means, summaries, strict=True)
    ]
    ax.barh(y, means, xerr=np.array(xerr).T, color="#00B0FF", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#64748B", linewidth=1)
    ax.set_xlabel("ROI")
    ax.set_title("ROI with 95% bootstrap CI (secondary to CLV)")
    style_figure(fig)
    style_axes(ax)
    save_figure(fig, "strategy_roi_bars")
    plt.close(fig)


def write_strategy_md(summaries: list[dict], path: Path) -> None:
    pick = "ensemble_weighted"
    sel_path = project_root() / REPORTS / "MODEL_SELECTION.md"
    if sel_path.exists():
        for line in sel_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("**Selected model:**"):
                pick = line.split("`")[1]
                break

    lines = [
        "# Betting strategy backtest",
        "",
        "Headline KPI: **Closing Line Value (CLV)**. ROI is reported with uncertainty but is",
        "variance-dominated over short horizons — do not treat a positive backtest ROI as proof",
        "of long-run profitability.",
        "",
        "## Corpus",
        "",
        "The primary backtest runs on **club-league closing-odds history** to validate EV gating,",
        "Kelly sizing, bankroll simulation, and CLV machinery at scale. International model",
        "evaluation (walk-forward RPS) is separate; there is no end-to-end international odds",
        "history for a full profitability curve. Non-market models use international-calibrated",
        "static 1X2 marginals on club matches unless per-match `market_probs` overlap exists.",
        "",
        f"**Selected model for live showcase:** `{pick}`",
        "",
        "## Parameters (fixed a priori)",
        "",
        "- Minimum edge (EV): from config `betting.min_edge`",
        "- Kelly fraction: from config `betting.kelly_fraction`",
        "- Max stake fraction: from config `betting.max_stake_fraction`",
        "",
        "## Results summary",
        "",
        "| Model | Bets | Total staked | ROI [95% CI] | Hit rate | Mean CLV | % CLV>0 | Risk of ruin |",
        "|-------|------|--------------|--------------|----------|----------|---------|--------------|",
    ]
    for s in summaries:
        roi = s.get("roi", float("nan"))
        roi_lo = s.get("roi_ci_low", float("nan"))
        roi_hi = s.get("roi_ci_high", float("nan"))
        lines.append(
            f"| `{s['model']}` | {s.get('n_bets', 0)} | ${s.get('total_staked', 0):,.0f} | "
            f"{roi:.2%} [{roi_lo:.2%}, {roi_hi:.2%}] | {s.get('hit_rate', float('nan')):.1%} | "
            f"{s.get('mean_clv_odds_ratio', float('nan')):.3f} | "
            f"{s.get('pct_positive_clv', float('nan')):.1%} | "
            f"{s.get('risk_of_ruin', float('nan')):.1%} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **CLV** measures whether taken odds beat the closing line — the best short-horizon",
            "  skill signal. Positive mean CLV with narrow CI is encouraging; negative CLV means",
            "  the strategy is paying worse than the market close on average.",
            "- **ROI** can swing wildly over thousands of bets; use the bootstrap CI and Monte Carlo",
            "  risk-of-ruin alongside CLV, not instead of it.",
            "- Strategy parameters were **not** tuned to maximize backtest ROI.",
            "- If nothing beats the market on CLV, frame the live bet as price-taker on softer",
            "  markets only.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def priority_models() -> list[str]:
    """Models to backtest: selection + ensembles + ladder with market_probs."""
    preferred = [ENSEMBLE_WEIGHTED, ENSEMBLE_SIMPLE, "poisson", "dixon_coles", "elo", "baseline", "market"]
    available = set(discover_strategy_models())
    out = [m for m in preferred if m in available]
    for m in discover_strategy_models():
        if m not in out:
            out.append(m)
    return out


def run_all(cfg: AppConfig | None = None, models: list[str] | None = None) -> list[dict]:
    cfg = cfg or get_config()
    names = models or priority_models()
    summaries: list[dict] = []
    bankroll_frames: dict[str, pd.DataFrame] = {}
    bets_by_model: dict[str, pd.DataFrame] = {}

    for model in names:
        logger.info("Strategy backtest: %s", model)
        bets, bankroll, summary = run_strategy_backtest(model, cfg=cfg, corpus=CORPUS_CLUB, stake_method="kelly")
        summaries.append(summary)
        bankroll_frames[model] = bankroll
        bets_by_model[model] = bets
        if cfg.betting.flat_stake_control and model in {ENSEMBLE_WEIGHTED, "poisson", "market"}:
            run_strategy_backtest(model, cfg=cfg, corpus=CORPUS_CLUB, stake_method="flat")

    plot_equity_curves(summaries, bankroll_frames)
    plot_bankroll_distribution(summaries, bankroll_frames)
    plot_clv_distribution(bets_by_model)
    plot_roi_bars(summaries)
    write_strategy_md(summaries, project_root() / REPORTS / "STRATEGY.md")
    return summaries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Club-corpus +EV backtest (machinery check, not a profit claim)")
    parser.add_argument("--model", required=True, help="Model name or 'all'")
    args = parser.parse_args(argv)

    get_config.cache_clear()
    cfg = load_config()
    set_seeds(cfg.seed)

    if args.model == "all":
        run_all(cfg)
    else:
        run_strategy_backtest(args.model, cfg=cfg)
        if cfg.betting.flat_stake_control:
            run_strategy_backtest(args.model, cfg=cfg, stake_method="flat")


if __name__ == "__main__":
    main()
