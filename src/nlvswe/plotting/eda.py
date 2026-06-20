"""EDA figure builders for raw interim and processed feature tables."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nlvswe.config import get_config, load_config
from nlvswe.io import read_table, save_figure
from nlvswe.logging import get_logger
from nlvswe.plotting.theme import THEME, apply_theme, diverging_cmap, style_figure
from nlvswe.repro import set_seeds

logger = get_logger(__name__)

INTERIM = "interim"


def run_raw_eda(
    matches: pd.DataFrame,
    ratings: pd.DataFrame,
    odds: pd.DataFrame,
) -> None:
    """Write EDA figures for canonical interim tables."""
    apply_theme()
    if matches.empty:
        return

    intl = matches[
        matches["competition"].str.contains(
            "FIFA|World|Friendly|UEFA|Copa|Euro", case=False, regex=True
        )
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    style_figure(fig)

    if not intl.empty:
        yearly = intl[intl["status"] == "completed"].copy()
        yearly["year"] = yearly["date_utc"].dt.year
        yearly.groupby("year").size().plot(kind="bar", ax=axes[0, 0], color=THEME.alpha)
        axes[0, 0].set_title("International matches per year")
        axes[0, 0].set_xlabel("Year")
        style_figure(fig)

        goals = yearly["home_goals"].astype(float) + yearly["away_goals"].astype(float)
        sns.histplot(goals, bins=30, ax=axes[0, 1], color=THEME.alpha_alt, edgecolor=THEME.grid)
        axes[0, 1].set_title("Goals per match distribution")
        style_figure(fig)

        def result(row: pd.Series) -> str:
            if row["home_goals"] > row["away_goals"]:
                return "home"
            if row["home_goals"] < row["away_goals"]:
                return "away"
            return "draw"

        yearly["result"] = yearly.apply(result, axis=1)
        neutral = yearly[yearly["neutral"]]
        home_venue = yearly[~yearly["neutral"]]
        pd.DataFrame(
            {
                "neutral": neutral["result"].value_counts(normalize=True),
                "home_venue": home_venue["result"].value_counts(normalize=True),
            }
        ).plot(kind="bar", ax=axes[1, 0], color=[THEME.alpha, THEME.bearish])
        axes[1, 0].set_title("Win rates: neutral vs home venue")
        axes[1, 0].tick_params(axis="x", rotation=0)
        style_figure(fig)

    if not ratings.empty:
        fifa = ratings[ratings["source"] == "fifa"]
        if not fifa.empty:
            cov = fifa.groupby(fifa["rating_date"].dt.year)["team_id"].nunique()
            cov.plot(ax=axes[1, 1], color=THEME.alpha_alt, linewidth=2)
            axes[1, 1].set_title("FIFA ratings coverage (teams/year)")
            style_figure(fig)

    plt.tight_layout()
    save_figure(fig, "eda_raw_overview")
    plt.close(fig)

    if not odds.empty:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        style_figure(fig2)
        closing = odds[(odds["market"] == "1x2") & (odds["is_closing"])]
        if not closing.empty:
            overrounds = closing.groupby(["match_id", "bookmaker"])["decimal_odds"].apply(
                lambda s: (1.0 / s).sum()
            )
            sns.histplot(overrounds, bins=40, ax=ax2, color=THEME.alpha, edgecolor=THEME.grid)
            ax2.set_title("1X2 closing overround distribution")
            ax2.set_xlabel("Sum of implied probabilities")
            style_figure(fig2)
        save_figure(fig2, "eda_raw_odds_overround")
        plt.close(fig2)


def run_feature_eda(df: pd.DataFrame) -> None:
    """Write EDA figures for feature/outcome relationships."""
    apply_theme()
    if df.empty:
        return
    completed = df[df["result_1x2"].notna()].copy()
    upcoming = df[df["result_1x2"].isna()].sort_values("date_utc")
    target = upcoming.iloc[-1] if not upcoming.empty else None
    target_diff = (
        float(target["elo_diff"])
        if target is not None and pd.notna(target.get("elo_diff"))
        else None
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    style_figure(fig)
    miss = df.isna().mean().sort_values(ascending=True)
    miss = miss[miss > 0]
    if not miss.empty:
        miss.plot(kind="barh", ax=ax, color=THEME.alpha)
        ax.set_title("Feature missingness")
        ax.set_xlabel("Fraction null")
        style_figure(fig)
        save_figure(fig, "eda_feat_missingness")
    plt.close(fig)

    if completed.empty:
        return

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    style_figure(fig2)
    completed["home_win"] = (completed["result_1x2"] == "home").astype(int)
    valid = completed[completed["elo_diff"].notna()]
    if not valid.empty:
        valid = valid.copy()
        valid["elo_bin"] = pd.qcut(valid["elo_diff"], q=10, duplicates="drop")
        rate = valid.groupby("elo_bin", observed=True)["home_win"].mean()
        rate.plot(kind="bar", ax=ax2, color=THEME.alpha_alt)
        ax2.set_title("Empirical P(home win) by elo_diff decile")
        ax2.set_ylabel("P(home win)")
        ax2.tick_params(axis="x", rotation=45)
        # Mark which decile the target fixture's elo_diff falls into.
        if target_diff is not None and len(rate):
            pos = next(
                (i for i, iv in enumerate(rate.index) if target_diff in iv),
                len(rate) - 1 if target_diff > rate.index[-1].right else 0,
            )
            if 0 <= pos < len(ax2.patches):
                ax2.patches[pos].set_color(THEME.bearish)
                yval = float(rate.iloc[pos])
                ax2.annotate(
                    f"target match ({target_diff:+.0f})",
                    xy=(pos, yval),
                    xytext=(pos, min(0.95, yval + 0.12)),
                    ha="center",
                    color=THEME.text,
                    fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=THEME.text),
                )
        style_figure(fig2)
        save_figure(fig2, "eda_feat_elo_diff_vs_home_win")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    style_figure(fig3)
    rates = completed.groupby("neutral")["result_1x2"].value_counts(normalize=True).unstack(
        fill_value=0
    )
    col_order = [c for c in ("home", "draw", "away") if c in rates.columns]
    rates = rates[col_order]
    label_map = {"home": "Home win", "draw": "Draw", "away": "Away win"}
    color_map = {"home": THEME.alpha_alt, "draw": THEME.bearish, "away": THEME.alpha}
    rates.plot(kind="bar", ax=ax3, color=[color_map[c] for c in col_order])
    ax3.set_title("1X2 base rates by venue type")
    ax3.set_xlabel("Venue type")
    ax3.set_xticklabels(
        ["Neutral venue" if bool(v) else "Home/away venue" for v in rates.index],
        rotation=0,
    )
    ax3.legend([label_map[c] for c in col_order], title="Result")
    ax3.text(
        0.5,
        -0.33,
        "Neutral venue = first-listed team (no host determined)",
        transform=ax3.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        style="italic",
        color=THEME.text,
    )
    style_figure(fig3)
    save_figure(fig3, "eda_feat_base_rates")
    plt.close(fig3)

    # Two-team Elo trajectory (the target fixture's sides).
    if target is not None:
        home_id = target["home_team_id"]
        away_id = target["away_team_id"]

        def _elo_series(team_id: str) -> pd.DataFrame:
            h = df[df["home_team_id"] == team_id][["date_utc", "home_elo_pre"]].rename(
                columns={"home_elo_pre": "elo"}
            )
            a = df[df["away_team_id"] == team_id][["date_utc", "away_elo_pre"]].rename(
                columns={"away_elo_pre": "elo"}
            )
            return pd.concat([h, a]).dropna(subset=["elo"]).sort_values("date_utc")

        fig5, ax5 = plt.subplots(figsize=(10, 5))
        style_figure(fig5)
        home_s = _elo_series(home_id)
        away_s = _elo_series(away_id)
        ax5.plot(
            home_s["date_utc"], home_s["elo"], color=THEME.alpha, linewidth=2,
            label=str(home_id).title(),
        )
        ax5.plot(
            away_s["date_utc"], away_s["elo"], color=THEME.bearish, linewidth=2,
            label=str(away_id).title(),
        )
        ax5.set_title(f"Elo trajectory: {str(home_id).title()} vs {str(away_id).title()}")
        ax5.set_ylabel("Elo (pre-match)")
        ax5.set_xlabel("Year")
        ax5.legend()
        style_figure(fig5)
        save_figure(fig5, "eda_feat_elo_trajectory")
        plt.close(fig5)

    num_cols = completed.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c not in ("home_goals", "away_goals", "total_goals")]
    if len(num_cols) >= 2:
        corr = completed[num_cols].corr()
        fig4, ax4 = plt.subplots(figsize=(12, 10))
        style_figure(fig4)
        sns.heatmap(
            corr,
            ax=ax4,
            cmap=diverging_cmap(),
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={"label": "correlation"},
        )
        ax4.set_title("Feature correlation heatmap")
        style_figure(fig4)
        plt.tight_layout()
        save_figure(fig4, "eda_feat_correlation")
        plt.close(fig4)


def run_all_eda() -> None:
    """Regenerate all raw + feature EDA figures from on-disk artifacts."""
    matches = read_table("matches", INTERIM)
    ratings = read_table("ratings", INTERIM)
    try:
        odds = read_table("odds", INTERIM)
    except FileNotFoundError:
        odds = pd.DataFrame()
    features = read_table("features", "processed")
    run_raw_eda(matches, ratings, odds)
    run_feature_eda(features)
    logger.info("Regenerated all EDA figures")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Regenerate interim EDA figures")
    parser.parse_args(argv)
    get_config.cache_clear()
    cfg = load_config()
    set_seeds(cfg.seed)
    run_all_eda()


if __name__ == "__main__":
    main()
