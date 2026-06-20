"""Turn model probs into a single-match bet slip (or a no-bet row with a reason)."""

from __future__ import annotations

import pandas as pd

from nlvswe.betting.strategy import (
    devigged_book_probs,
    expected_value,
    stake_amount,
    vectorized_mult_devig_1x2,
)
from nlvswe.config import AppConfig

CORPUS_INTERNATIONAL = "international"


def _latest_prematch_odds(odds: pd.DataFrame, match_id: str, kickoff: pd.Timestamp) -> pd.DataFrame:
    """Latest pre-kickoff quote per bookmaker/market/selection.

    I skip is_closing on purpose. The live bet uses whatever was quoted last before
    kickoff, which is often tagged is_closing=True (because it literally is). CLV
    gets the true closing snapshot elsewhere.
    """
    sub = odds[
        (odds["match_id"].astype(str) == str(match_id))
        & (odds["corpus"] == CORPUS_INTERNATIONAL)
        & (pd.to_datetime(odds["captured_at"], utc=True) < kickoff)
    ].copy()
    if sub.empty:
        return sub
    return (
        sub.sort_values(["bookmaker", "market", "selection", "captured_at"], kind="mergesort")
        .groupby(["bookmaker", "market", "selection"], sort=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _prob_lookup(market_probs: pd.DataFrame) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for _, row in market_probs.iterrows():
        lookup[(str(row["market"]).lower(), str(row["selection"]).lower())] = float(row["model_prob"])
    return lookup


def _normalize_market(market: str, selection: str) -> tuple[str, str]:
    market_l = str(market).lower()
    sel = str(selection).lower()
    if market_l == "h2h":
        market_l = "1x2"
    return market_l, sel


def build_live_bet_slip(
    market_probs: pd.DataFrame,
    odds: pd.DataFrame,
    cfg: AppConfig,
    *,
    match_id: str,
    model: str,
    kickoff: pd.Timestamp,
    captured_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Evaluate +EV lines for one match; returns bet rows or an honest no-bet row."""
    bankroll = float(cfg.betting.bankroll)
    allowed = {b.lower() for b in cfg.betting.bookmakers}
    kickoff = pd.Timestamp(kickoff).tz_convert("UTC")
    prob_map = _prob_lookup(market_probs)
    prematch = _latest_prematch_odds(odds, match_id, kickoff)
    prematch = prematch[prematch["bookmaker"].str.lower().isin(allowed)]

    if prematch.empty:
        return _no_bet_row(
            match_id,
            model,
            kickoff,
            captured_at,
            rationale="No pre-kickoff odds available for configured bookmakers.",
        )

    one_x_two = prematch[prematch["market"].astype(str).str.lower().isin({"1x2", "h2h"})].copy()
    if cfg.betting.devig_method == "multiplicative" and not one_x_two.empty:
        enriched = vectorized_mult_devig_1x2(one_x_two)
        other = prematch[~prematch.index.isin(one_x_two.index)]
        if not other.empty:
            enriched_parts = [enriched]
            for (_, _), grp in other.groupby(["match_id", "bookmaker"], sort=False):
                market_name = str(grp.iloc[0]["market"]).lower()
                probs = devigged_book_probs(grp, market=market_name, devig_method=cfg.betting.devig_method)
                if probs is None:
                    continue
                g = grp.copy()
                g["book_prob"] = g["selection"].astype(str).str.lower().map(probs)
                enriched_parts.append(g)
            enriched = pd.concat(enriched_parts, ignore_index=True)
    else:
        enriched_parts: list[pd.DataFrame] = []
        for (_, _), grp in prematch.groupby(["match_id", "bookmaker"], sort=False):
            market_name = str(grp.iloc[0]["market"]).lower()
            probs = devigged_book_probs(grp, market=market_name, devig_method=cfg.betting.devig_method)
            if probs is None:
                continue
            g = grp.copy()
            g["book_prob"] = g["selection"].astype(str).str.lower().map(probs)
            enriched_parts.append(g)
        enriched = pd.concat(enriched_parts, ignore_index=True) if enriched_parts else pd.DataFrame()

    if enriched.empty:
        return _no_bet_row(
            match_id,
            model,
            kickoff,
            captured_at,
            rationale="Could not de-vig bookmaker quotes.",
        )

    rows: list[dict] = []
    total_exposure = 0.0
    max_exposure = bankroll * cfg.betting.max_stake_fraction * 3
    candidates: list[dict] = []

    for _, quote in enriched.iterrows():
        market_l, sel = _normalize_market(quote["market"], quote["selection"])
        model_prob = prob_map.get((market_l, sel))
        if model_prob is None and market_l.startswith("totals"):
            model_prob = prob_map.get((f"totals_{sel.split('_')[-1]}", sel.split("_")[0]))
        if model_prob is None:
            continue

        odds_taken = float(quote["decimal_odds"])
        book_prob = float(quote["book_prob"])
        ev = expected_value(model_prob, odds_taken)
        edge = model_prob - book_prob
        if ev <= cfg.betting.min_edge:
            continue
        candidates.append({**quote.to_dict(), "market_l": market_l, "sel": sel, "model_prob": model_prob, "ev": ev, "edge": edge})

    candidates.sort(key=lambda r: r["ev"], reverse=True)

    for quote in candidates:

        stake = stake_amount(
            bankroll,
            float(quote["model_prob"]),
            float(quote["decimal_odds"]),
            kelly_frac=cfg.betting.kelly_fraction,
            max_stake_fraction=cfg.betting.max_stake_fraction,
            method="kelly",
        )
        if stake <= 0 or total_exposure + stake > max_exposure:
            continue

        cap = pd.Timestamp(quote["captured_at"]).tz_convert("UTC")
        ev = float(quote["ev"])
        model_prob = float(quote["model_prob"])
        book_prob = float(quote["book_prob"])
        rationale = (
            f"EV={ev:.3f} (min {cfg.betting.min_edge}); model p={model_prob:.3f} vs "
            f"de-vig book p={book_prob:.3f}; Kelly stake on ${bankroll:,.0f} bankroll."
        )
        rows.append(
            {
                "match_id": str(match_id),
                "model": model,
                "captured_at": cap,
                "kickoff_utc": kickoff,
                "bookmaker": quote["bookmaker"],
                "market": quote["market_l"],
                "selection": quote["sel"],
                "model_prob": model_prob,
                "book_prob_devig": book_prob,
                "edge": float(quote["edge"]),
                "ev": ev,
                "odds_taken": float(quote["decimal_odds"]),
                "stake_method": "kelly",
                "stake": stake,
                "status": "bet",
                "rationale": rationale,
            }
        )
        total_exposure += stake

    if not rows:
        return _no_bet_row(
            match_id,
            model,
            kickoff,
            captured_at or pd.Timestamp(enriched["captured_at"].max()).tz_convert("UTC"),
            rationale=(
                f"No selection cleared min_edge={cfg.betting.min_edge} after de-vig "
                f"({len(enriched)} quotes checked)."
            ),
        )

    out = pd.DataFrame(rows)
    return _finalize_bet_slip(out)


def _finalize_bet_slip(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["captured_at"] = pd.to_datetime(out["captured_at"], utc=True).astype("datetime64[ns, UTC]")
    out["kickoff_utc"] = pd.to_datetime(out["kickoff_utc"], utc=True).astype("datetime64[ns, UTC]")
    for col in ("model_prob", "book_prob_devig", "edge", "ev", "odds_taken", "stake"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in ("bookmaker", "market", "selection", "stake_method", "status", "rationale", "model", "match_id"):
        if col in out.columns:
            out[col] = out[col].astype("string")
    return out.sort_values(["ev", "stake"], ascending=False, kind="mergesort", na_position="last").reset_index(
        drop=True
    )


def _no_bet_row(
    match_id: str,
    model: str,
    kickoff: pd.Timestamp,
    captured_at: pd.Timestamp | None,
    *,
    rationale: str,
) -> pd.DataFrame:
    cap = captured_at if captured_at is not None else kickoff
    kickoff = pd.Timestamp(kickoff).tz_convert("UTC")
    cap = pd.Timestamp(cap).tz_convert("UTC")
    df = pd.DataFrame(
        [
            {
                "match_id": str(match_id),
                "model": model,
                "captured_at": cap,
                "kickoff_utc": kickoff,
                "bookmaker": None,
                "market": None,
                "selection": None,
                "model_prob": None,
                "book_prob_devig": None,
                "edge": None,
                "ev": None,
                "odds_taken": None,
                "stake_method": "kelly",
                "stake": 0.0,
                "status": "no_bet",
                "rationale": rationale,
            }
        ]
    )
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True).astype("datetime64[ns, UTC]")
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True).astype("datetime64[ns, UTC]")
    return _finalize_bet_slip(df)
