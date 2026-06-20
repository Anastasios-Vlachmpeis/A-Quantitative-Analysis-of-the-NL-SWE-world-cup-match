"""Scoreline-to-market probability induction (Plan 07)."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nlvswe.config import AppConfig, get_config, load_config
from nlvswe.io import read_table, save_figure, write_table
from nlvswe.logging import get_logger
from nlvswe.models.data import load_model_features
from nlvswe.models.registry import create_model, ladder_models
from nlvswe.plotting.theme import apply_theme, style_axes, style_figure
from nlvswe.repro import set_seeds
from nlvswe.schemas import validate_table

logger = get_logger(__name__)

PLAN = "07"
PROCESSED = "processed"
INTERIM = "interim"
Method = Literal["analytic", "mc", "derived"]


@dataclass(frozen=True)
class MarketConfig:
    totals_lines: tuple[float, ...]
    ah_lines: tuple[float, ...]
    correct_score_top_n: int


def default_market_config(cfg: AppConfig | None = None) -> MarketConfig:
    cfg = cfg or get_config()
    m = cfg.betting.markets
    return MarketConfig(
        totals_lines=tuple(m.totals_lines),
        ah_lines=tuple(m.ah_lines),
        correct_score_top_n=m.correct_score_top_n,
    )


def _grid_size(matrix: np.ndarray) -> int:
    mat = np.asarray(matrix, dtype=float)
    return mat.shape[0]


def simulate_scorelines(
    matrix: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample (home_goals, away_goals) pairs from a scoreline matrix."""
    mat = np.asarray(matrix, dtype=float)
    mat = mat / mat.sum()
    g = mat.shape[0]
    idx = rng.choice(g * g, size=n, p=mat.ravel())
    home = idx // g
    away = idx % g
    return np.column_stack([home, away])


def _ah_home_outcome(home: int, away: int, line: float) -> str:
    """Home-side Asian handicap result: win | push | lose."""
    diff = home + line - away
    if math.isclose(line % 0.5, 0.0) and not math.isclose(line % 1.0, 0.0):
        return "win" if diff > 0 else "lose"
    if diff > 0:
        return "win"
    if math.isclose(diff, 0.0):
        return "push"
    return "lose"


def asian_handicap_probs(matrix: np.ndarray, line: float) -> dict[str, float]:
    """Home-side AH probabilities including push when applicable."""
    mat = np.asarray(matrix, dtype=float)
    mat = mat / mat.sum()
    g = mat.shape[0]
    counts = {"home": 0.0, "away": 0.0, "push": 0.0}
    for i in range(g):
        for j in range(g):
            p = mat[i, j]
            if p <= 0:
                continue
            outcome = _ah_home_outcome(i, j, line)
            if outcome == "win":
                counts["home"] += p
            elif outcome == "push":
                counts["push"] += p
            else:
                counts["away"] += p
    return counts


def markets_from_matrix(
    matrix: np.ndarray,
    *,
    market_cfg: MarketConfig | None = None,
) -> list[dict]:
    """Analytic market probabilities from a scoreline matrix."""
    cfg = market_cfg or default_market_config()
    mat = np.asarray(matrix, dtype=float)
    mat = mat / mat.sum()
    g = mat.shape[0]
    rows: list[dict] = []

    p_home = float(np.tril(mat, k=-1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, k=1).sum())
    rows.extend(
        [
            {"market": "1x2", "selection": "home", "model_prob": p_home, "method": "analytic"},
            {"market": "1x2", "selection": "draw", "model_prob": p_draw, "method": "analytic"},
            {"market": "1x2", "selection": "away", "model_prob": p_away, "method": "analytic"},
            {"market": "double_chance", "selection": "1x", "model_prob": p_home + p_draw, "method": "analytic"},
            {"market": "double_chance", "selection": "12", "model_prob": p_home + p_away, "method": "analytic"},
            {"market": "double_chance", "selection": "x2", "model_prob": p_draw + p_away, "method": "analytic"},
        ]
    )

    for line in cfg.totals_lines:
        over = 0.0
        for i in range(g):
            for j in range(g):
                if i + j > line:
                    over += mat[i, j]
        under = 1.0 - over
        rows.append({"market": f"totals_{line}", "selection": "over", "model_prob": float(over), "method": "analytic"})
        rows.append({"market": f"totals_{line}", "selection": "under", "model_prob": float(under), "method": "analytic"})

    btts_yes = float(sum(mat[i, j] for i in range(1, g) for j in range(1, g)))
    rows.append({"market": "btts", "selection": "yes", "model_prob": btts_yes, "method": "analytic"})
    rows.append({"market": "btts", "selection": "no", "model_prob": 1.0 - btts_yes, "method": "analytic"})

    for line in cfg.ah_lines:
        ah = asian_handicap_probs(mat, line)
        market = f"ah_{line:+.1f}".replace("+", "p").replace("-", "m")
        for sel, prob in ah.items():
            if prob > 0 or sel == "push":
                rows.append(
                    {
                        "market": market,
                        "selection": sel,
                        "model_prob": float(prob),
                        "method": "analytic",
                    }
                )

    flat: list[tuple[int, int, float]] = []
    for i in range(g):
        for j in range(g):
            flat.append((i, j, float(mat[i, j])))
    flat.sort(key=lambda x: x[2], reverse=True)
    top_n = cfg.correct_score_top_n
    top_mass = sum(p for _, _, p in flat[:top_n])
    for i, j, p in flat[:top_n]:
        rows.append(
            {
                "market": "correct_score",
                "selection": f"{i}-{j}",
                "model_prob": p,
                "method": "analytic",
            }
        )
    rows.append(
        {
            "market": "correct_score",
            "selection": "other",
            "model_prob": max(0.0, 1.0 - top_mass),
            "method": "analytic",
        }
    )
    return rows


def markets_from_1x2(
    p_home: float,
    p_draw: float,
    p_away: float,
) -> list[dict]:
    """Markets available when only 1X2 probabilities exist."""
    total = p_home + p_draw + p_away
    if total <= 0:
        p_home = p_draw = p_away = 1.0 / 3.0
    else:
        p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total
    return [
        {"market": "1x2", "selection": "home", "model_prob": p_home, "method": "derived"},
        {"market": "1x2", "selection": "draw", "model_prob": p_draw, "method": "derived"},
        {"market": "1x2", "selection": "away", "model_prob": p_away, "method": "derived"},
        {"market": "double_chance", "selection": "1x", "model_prob": p_home + p_draw, "method": "derived"},
        {"market": "double_chance", "selection": "12", "model_prob": p_home + p_away, "method": "derived"},
        {"market": "double_chance", "selection": "x2", "model_prob": p_draw + p_away, "method": "derived"},
    ]


def markets_from_samples(
    matrix: np.ndarray,
    n: int,
    rng: np.random.Generator,
    *,
    market_cfg: MarketConfig | None = None,
) -> list[dict]:
    """Monte Carlo market probability estimates from scoreline samples."""
    cfg = market_cfg or default_market_config()
    samples = simulate_scorelines(matrix, n, rng)
    hg, ag = samples[:, 0], samples[:, 1]

    rows: list[dict] = []
    p_home = float(np.mean(hg > ag))
    p_draw = float(np.mean(hg == ag))
    p_away = float(np.mean(hg < ag))
    rows.extend(
        [
            {"market": "1x2", "selection": "home", "model_prob": p_home, "method": "mc"},
            {"market": "1x2", "selection": "draw", "model_prob": p_draw, "method": "mc"},
            {"market": "1x2", "selection": "away", "model_prob": p_away, "method": "mc"},
            {"market": "double_chance", "selection": "1x", "model_prob": p_home + p_draw, "method": "mc"},
            {"market": "double_chance", "selection": "12", "model_prob": p_home + p_away, "method": "mc"},
            {"market": "double_chance", "selection": "x2", "model_prob": p_draw + p_away, "method": "mc"},
        ]
    )

    totals = hg + ag
    for line in cfg.totals_lines:
        over = float(np.mean(totals > line))
        rows.append({"market": f"totals_{line}", "selection": "over", "model_prob": over, "method": "mc"})
        rows.append({"market": f"totals_{line}", "selection": "under", "model_prob": 1.0 - over, "method": "mc"})

    btts_yes = float(np.mean((hg >= 1) & (ag >= 1)))
    rows.append({"market": "btts", "selection": "yes", "model_prob": btts_yes, "method": "mc"})
    rows.append({"market": "btts", "selection": "no", "model_prob": 1.0 - btts_yes, "method": "mc"})

    for line in cfg.ah_lines:
        outcomes = np.array([_ah_home_outcome(int(h), int(a), line) for h, a in samples])
        market = f"ah_{line:+.1f}".replace("+", "p").replace("-", "m")
        p_win = float(np.mean(outcomes == "win"))
        p_push = float(np.mean(outcomes == "push"))
        p_lose = float(np.mean(outcomes == "lose"))
        rows.append({"market": market, "selection": "home", "model_prob": p_win, "method": "mc"})
        if p_push > 0 or line in (0.0, -1.0, 1.0):
            rows.append({"market": market, "selection": "push", "model_prob": p_push, "method": "mc"})
        rows.append({"market": market, "selection": "away", "model_prob": p_lose, "method": "mc"})

    score_labels, counts = np.unique([f"{h}-{a}" for h, a in samples], return_counts=True)
    order = np.argsort(-counts)
    top_n = cfg.correct_score_top_n
    top_labels = score_labels[order][:top_n]
    top_mass = counts[order][:top_n].sum() / n
    for label in top_labels:
        rows.append(
            {
                "market": "correct_score",
                "selection": label,
                "model_prob": float(np.mean([f"{h}-{a}" == label for h, a in samples])),
                "method": "mc",
            }
        )
    rows.append(
        {
            "market": "correct_score",
            "selection": "other",
            "model_prob": float(max(0.0, 1.0 - top_mass)),
            "method": "mc",
        }
    )
    return rows


def check_coherence(rows: list[dict], *, tol: float = 1e-6) -> list[str]:
    """Return list of coherence violations (empty = pass). Checked per match."""
    issues: list[str] = []
    df = pd.DataFrame(rows)
    if df.empty:
        return issues
    if "match_id" not in df.columns:
        df = df.copy()
        df["match_id"] = "single"

    for match_id, mdf in df.groupby("match_id"):
        for (market, method), grp in mdf.groupby(["market", "method"]):
            total = float(grp["model_prob"].sum())
            if market == "1x2":
                if len(grp) != 3:
                    continue
                if not math.isclose(total, 1.0, abs_tol=tol):
                    issues.append(f"{match_id}/{market}/{method}: sum={total:.6f}")
            elif market.startswith("totals_"):
                if len(grp) != 2:
                    continue
                if not math.isclose(total, 1.0, abs_tol=tol):
                    issues.append(f"{match_id}/{market}/{method}: over+under={total:.6f}")
            elif market == "btts":
                if len(grp) != 2:
                    continue
                if not math.isclose(total, 1.0, abs_tol=tol):
                    issues.append(f"{match_id}/{market}/{method}: yes+no={total:.6f}")
            elif market.startswith("ah_") and not math.isclose(total, 1.0, abs_tol=1e-4):
                issues.append(f"{match_id}/{market}/{method}: outcomes sum={total:.6f}")
            elif market == "correct_score" and not math.isclose(total, 1.0, abs_tol=tol):
                issues.append(f"{match_id}/{market}/{method}: buckets sum={total:.6f}")

        dc = mdf[mdf["market"] == "double_chance"]
        if not dc.empty:
            method = str(dc["method"].iloc[0])
            x12 = mdf[(mdf["market"] == "1x2") & (mdf["method"] == method)]
            if len(x12) == 3:
                ph = float(x12.loc[x12["selection"] == "home", "model_prob"].iloc[0])
                pd_ = float(x12.loc[x12["selection"] == "draw", "model_prob"].iloc[0])
                pa = float(x12.loc[x12["selection"] == "away", "model_prob"].iloc[0])
                for sel, expected in (("1x", ph + pd_), ("12", ph + pa), ("x2", pd_ + pa)):
                    got = float(dc.loc[dc["selection"] == sel, "model_prob"].iloc[0])
                    if not math.isclose(got, expected, abs_tol=tol):
                        issues.append(f"{match_id}/double_chance {sel}: {got} != {expected}")
    return issues


def scoreline_from_flat(flat: list[float] | np.ndarray, *, side: int) -> np.ndarray:
    """Deserialize flattened scoreline vector to square matrix."""
    arr = np.asarray(flat, dtype=float)
    side = int(side)
    if arr.size != side * side:
        raise ValueError(f"scoreline_flat length {arr.size} != {side}x{side}")
    mat = arr.reshape(side, side)
    return mat / mat.sum()


def induce_row_markets(
    *,
    match_id: str,
    model: str,
    p_home: float,
    p_draw: float,
    p_away: float,
    scoreline: np.ndarray | None,
) -> list[dict]:
    """Build market rows for one match."""
    if scoreline is not None:
        analytic = markets_from_matrix(scoreline)
        for row in analytic:
            row.update({"match_id": match_id, "model": model})
        return analytic
    derived = markets_from_1x2(p_home, p_draw, p_away)
    for row in derived:
        row.update({"match_id": match_id, "model": model})
    return derived


def _load_predictions(model_name: str) -> pd.DataFrame:
    return read_table(f"predictions_{model_name}", PROCESSED)


def _load_odds():
    try:
        return read_table("odds", INTERIM)
    except FileNotFoundError:
        return pd.DataFrame()


def _target_match_row(features: pd.DataFrame, cfg: AppConfig) -> pd.Series | None:
    tm = cfg.target_match
    home_key = tm.home.lower().replace(" ", "_")
    away_key = tm.away.lower().replace(" ", "_")
    mask = (features["home_team_id"] == home_key) & (features["away_team_id"] == away_key)
    sub = features[mask].sort_values("date_utc")
    return sub.iloc[-1] if not sub.empty else None


def _scoreline_for_target(model_name: str, cfg: AppConfig, features: pd.DataFrame) -> np.ndarray | None:
    """Fit model on completed history and predict target match scoreline."""
    target = _target_match_row(features, cfg)
    if target is None:
        return None
    train = features[features["result_1x2"].notna()]
    odds = _load_odds()
    model = create_model(model_name, cfg, odds)
    model.fit(train)
    pred = model.predict(target)
    return pred.scoreline


def plot_target_scoreline_heatmap(matrix: np.ndarray, *, model: str) -> None:
    apply_theme()
    mat = np.asarray(matrix, dtype=float)
    # Display 0..7 goals/side; higher cells carry ~0 mass and just add whitespace.
    view = min(7, mat.shape[0] - 1)
    sub = mat[: view + 1, : view + 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(sub, annot=False, cmap="Blues", ax=ax, cbar_kws={"label": "P(score)"})
    ax.set_xlabel("Away goals")
    ax.set_ylabel("Home goals")
    ax.set_title(f"Scoreline distribution — {model} (target match)")
    style_figure(fig)
    style_axes(ax)
    save_figure(fig, f"induction_scoreline_{model}")
    plt.close(fig)


def plot_simulated_scores(samples: np.ndarray, *, model: str) -> None:
    apply_theme()
    home, away = samples[:, 0], samples[:, 1]
    totals = home + away
    gd = home - away
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Total goals: bars centered on integers (so the mode reads at its true value).
    t_max = int(totals.max())
    axes[0].hist(totals, bins=np.arange(-0.5, t_max + 1.5, 1.0), color="#00B0FF", edgecolor="#1E293B")
    axes[0].set_title("Simulated total goals")
    axes[0].set_xlabel("Total goals")
    axes[0].set_xticks(range(0, t_max + 1))

    # Goal difference is an integer — integer-aligned bins + integer ticks (no 2.5).
    g_lo, g_hi = int(gd.min()), int(gd.max())
    axes[1].hist(gd, bins=np.arange(g_lo - 0.5, g_hi + 1.5, 1.0), color="#00E676", edgecolor="#1E293B")
    axes[1].set_title("Simulated goal difference (home − away)")
    axes[1].set_xlabel("Goal difference (home − away)")
    axes[1].set_xticks(range(g_lo, g_hi + 1))

    fig.suptitle(f"Monte Carlo score simulation — {model}")
    style_figure(fig)
    for ax in axes:
        style_axes(ax)
    save_figure(fig, f"induction_sim_scores_{model}")
    plt.close(fig)


def run_induction(
    model_name: str,
    *,
    cfg: AppConfig | None = None,
    mc_samples: int | None = None,
    write_figures: bool = True,
) -> pd.DataFrame:
    cfg = cfg or get_config()
    mc_samples = mc_samples or cfg.model.mc_samples

    try:
        preds = _load_predictions(model_name)
    except FileNotFoundError:
        logger.warning("No predictions for model %s; skipping", model_name)
        return pd.DataFrame()

    all_rows: list[dict] = []
    for _, row in preds.iterrows():
        scoreline = None
        if bool(row.get("has_scoreline", False)):
            flat = row.get("scoreline_flat") if "scoreline_flat" in row.index else None
            if flat is not None and not (isinstance(flat, float) and np.isnan(flat)):
                flat_list = list(flat) if hasattr(flat, "__iter__") and not isinstance(flat, str) else flat
                side = int(round(math.sqrt(len(flat_list))))
                scoreline = scoreline_from_flat(flat_list, side=side)
        all_rows.extend(
            induce_row_markets(
                match_id=str(row["match_id"]),
                model=model_name,
                p_home=float(row["p_home"]),
                p_draw=float(row["p_draw"]),
                p_away=float(row["p_away"]),
                scoreline=scoreline,
            )
        )

    if not all_rows:
        return pd.DataFrame()

    sample_issues = check_coherence(all_rows[: min(len(all_rows), 600)])
    if sample_issues:
        logger.warning("Coherence issues (sample): %s", sample_issues[:3])

    df = pd.DataFrame(all_rows)
    df["model_prob"] = df["model_prob"].astype("float64")
    df["match_id"] = df["match_id"].astype("string")
    df["model"] = df["model"].astype("string")
    df["market"] = df["market"].astype("string")
    df["selection"] = df["selection"].astype("string")
    df["method"] = df["method"].astype("string")

    if write_figures:
        features = load_model_features()
        matrix = _scoreline_for_target(model_name, cfg, features)
        if matrix is not None:
            rng = np.random.default_rng(cfg.seed)
            plot_target_scoreline_heatmap(matrix, model=model_name)
            sims = simulate_scorelines(matrix, mc_samples, rng)
            plot_simulated_scores(sims, model=model_name)

    df = validate_table(df, "market_probs")
    write_table(
        df,
        f"market_probs_{model_name}",
        PROCESSED,
        sort_by=["match_id", "market", "selection"],
        plan=PLAN,
        schema_name="market_probs",
    )
    logger.info("Induction %s: %d market rows", model_name, len(df))
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Induce market probabilities from scorelines (Plan 07)")
    parser.add_argument("--model", required=True, help="Model name or 'all'")
    parser.add_argument("--mc-samples", type=int, default=None, help="MC samples for target-match visuals")
    args = parser.parse_args(argv)

    get_config.cache_clear()
    cfg = load_config()
    set_seeds(cfg.seed)

    names = ladder_models(include_market=True, include_constant=False) if args.model == "all" else [args.model]
    for name in names:
        run_induction(name, cfg=cfg, mc_samples=args.mc_samples)


if __name__ == "__main__":
    main()
