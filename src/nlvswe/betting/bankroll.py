"""Monte Carlo bankroll paths and ROI confidence intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nlvswe.eval.scoring import bootstrap_ci


def simulate_realized_path(
    bets: pd.DataFrame,
    *,
    initial_bankroll: float,
) -> pd.DataFrame:
    """Chronological bankroll path using settled bet PnL."""
    if bets.empty:
        return pd.DataFrame(
            columns=["model", "path_type", "step", "date_utc", "bankroll", "drawdown", "mc_run"]
        )

    ordered = bets.sort_values(["date_utc", "match_id", "bookmaker", "market", "selection"], kind="mergesort")
    bankroll = float(initial_bankroll)
    peak = bankroll
    rows: list[dict] = []
    for step, (_, bet) in enumerate(ordered.iterrows(), start=1):
        bankroll += float(bet["pnl"])
        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak if peak > 0 else 0.0
        rows.append(
            {
                "model": bet["model"],
                "path_type": "realized",
                "step": step,
                "date_utc": bet["date_utc"],
                "bankroll": bankroll,
                "drawdown": dd,
                "mc_run": pd.NA,
            }
        )
    out = pd.DataFrame(rows)
    out["date_utc"] = pd.to_datetime(out["date_utc"], utc=True).astype("datetime64[ns, UTC]")
    out["mc_run"] = out["mc_run"].astype("Int64")
    return out


def monte_carlo_bankroll(
    bets: pd.DataFrame,
    *,
    initial_bankroll: float,
    n_sims: int,
    seed: int,
    ruin_threshold: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Resimulate bet outcomes from model_prob; return terminal distribution stats."""
    if bets.empty:
        return pd.DataFrame(), {
            "mean_final_bankroll": float("nan"),
            "median_final_bankroll": float("nan"),
            "mean_max_drawdown": float("nan"),
            "risk_of_ruin": float("nan"),
        }

    ordered = bets.sort_values(["date_utc", "match_id", "bookmaker", "market", "selection"], kind="mergesort")
    stakes = ordered["stake"].to_numpy(dtype=float)
    odds = ordered["odds_taken"].to_numpy(dtype=float)
    win_probs = ordered["model_prob"].to_numpy(dtype=float)
    push_mask = ordered["result"].astype(str).eq("push").to_numpy()
    # Use model_prob for Bernoulli resimulation; pushes replay as push
    rng = np.random.default_rng(seed)

    ruin_floor = initial_bankroll * ruin_threshold
    n_bets = len(ordered)
    finals = np.empty(n_sims, dtype=float)
    max_dds = np.empty(n_sims, dtype=float)
    ruined = np.zeros(n_sims, dtype=bool)

    # Per-bet PnL if the bet wins / loses (pushes settle at 0 either way). Stakes
    # are fixed (sized off the initial bankroll in build_bet_candidates), so the
    # whole resimulation vectorizes. Approximation: ignores the rare running-bankroll
    # stake cap and post-ruin stop (negligible under fractional Kelly); risk_of_ruin
    # is still computed exactly from each path's running minimum.
    win_pnl = np.where(push_mask, 0.0, stakes * (odds - 1.0))
    lose_pnl = np.where(push_mask, 0.0, -stakes)

    # Chunk over sims so the (chunk x n_bets) random matrix stays within ~160MB.
    chunk = max(1, min(n_sims, 20_000_000 // max(n_bets, 1)))
    done = 0
    while done < n_sims:
        b = min(chunk, n_sims - done)
        wins = rng.random((b, n_bets)) < win_probs[None, :]
        pnl = np.where(wins, win_pnl[None, :], lose_pnl[None, :])
        paths = initial_bankroll + np.cumsum(pnl, axis=1)
        full = np.concatenate([np.full((b, 1), initial_bankroll), paths], axis=1)
        peak = np.maximum.accumulate(full, axis=1)
        dd = np.where(peak > 0, (peak - full) / peak, 0.0)
        finals[done:done + b] = paths[:, -1]
        max_dds[done:done + b] = dd.max(axis=1)
        ruined[done:done + b] = full.min(axis=1) <= ruin_floor
        done += b

    summary = {
        "mean_final_bankroll": float(finals.mean()),
        "median_final_bankroll": float(np.median(finals)),
        "mean_max_drawdown": float(max_dds.mean()),
        "risk_of_ruin": float(ruined.mean()),
    }

    # Terminal bankrolls only (plotting every MC path would be silly).
    sample_idx = np.linspace(0, n_sims - 1, num=min(500, n_sims), dtype=int)
    rows = [
        {
            "model": ordered.iloc[0]["model"],
            "path_type": "mc_terminal",
            "step": int(i),
            "date_utc": pd.NaT,
            "bankroll": float(finals[i]),
            "drawdown": float(max_dds[i]),
            "mc_run": int(i),
        }
        for i in sample_idx
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True)
        df["mc_run"] = df["mc_run"].astype("Int64")
    return df, summary


def roi_with_ci(bets: pd.DataFrame, *, bootstrap_samples: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """ROI = total_pnl / total_staked with bootstrap CI on per-bet returns."""
    if bets.empty or bets["stake"].sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    returns = bets["pnl"].to_numpy(dtype=float) / np.maximum(bets["stake"].to_numpy(dtype=float), 1e-9)
    total_staked = float(bets["stake"].sum())
    roi = float(bets["pnl"].sum() / total_staked)
    _, lo, hi = bootstrap_ci(returns, n_samples=bootstrap_samples, seed=seed)
    # Scale CI to ROI space approximately
    mean_ret = float(returns.mean())
    if mean_ret != 0:
        scale = roi / mean_ret if mean_ret else 1.0
        return roi, float(lo * scale), float(hi * scale)
    return roi, float("nan"), float("nan")
