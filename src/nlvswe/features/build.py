"""Build the feature table without using anything that happens after kickoff."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from nlvswe.config import AppConfig, get_config, load_config, project_root
from collections import defaultdict, deque

from nlvswe.features.conditions import conditions_for_match, is_knockout_stage
from nlvswe.features.ratings import (
    CORPUS_INTERNATIONAL,
    _apply_elo_update,
    elo_pre_at_kickoff,
)
from nlvswe.ratings.elo import EloParams
from nlvswe.io import read_table, write_table
from nlvswe.plotting.eda import run_feature_eda
from nlvswe.logging import get_logger
from nlvswe.repro import set_seeds
from nlvswe.schemas import validate_table

logger = get_logger(__name__)

PLAN = "04"
PROCESSED = "processed"
INTERIM = "interim"


def _team_points(home_goals: int, away_goals: int, *, for_home: bool) -> float:
    if for_home:
        if home_goals > away_goals:
            return 3.0
        if home_goals == away_goals:
            return 1.0
        return 0.0
    if away_goals > home_goals:
        return 3.0
    if home_goals == away_goals:
        return 1.0
    return 0.0


def _team_goals(home_goals: int, away_goals: int, *, for_home: bool) -> tuple[int, int]:
    if for_home:
        return home_goals, away_goals
    return away_goals, home_goals


def _team_history_rows(history: pd.DataFrame, team_id: str, before: pd.Timestamp) -> pd.DataFrame:
    """Completed intl matches for team strictly before kickoff."""
    h = history[
        (history["status"] == "completed")
        & (history["corpus"] == CORPUS_INTERNATIONAL)
        & (history["date_utc"] < before)
        & ((history["home_team_id"] == team_id) | (history["away_team_id"] == team_id))
    ]
    return h.sort_values(["date_utc", "match_id"], kind="mergesort")


def _form_stats(history: pd.DataFrame, team_id: str, before: pd.Timestamp, k: int) -> dict[str, float | None]:
    rows = _team_history_rows(history, team_id, before)
    if rows.empty:
        return {"ppg": None, "gd_pg": None, "gf_pg": None, "ga_pg": None, "n": 0}

    tail = rows.tail(k)
    pts, gd, gf, ga = [], [], [], []
    for _, m in tail.iterrows():
        is_home = m["home_team_id"] == team_id
        hg, ag = int(m["home_goals"]), int(m["away_goals"])
        pts.append(_team_points(hg, ag, for_home=is_home))
        g_for, g_against = _team_goals(hg, ag, for_home=is_home)
        gf.append(g_for)
        ga.append(g_against)
        gd.append(g_for - g_against)

    n = len(tail)
    return {
        "ppg": sum(pts) / n,
        "gd_pg": sum(gd) / n,
        "gf_pg": sum(gf) / n,
        "ga_pg": sum(ga) / n,
        "n": n,
    }


def _rest_stats(history: pd.DataFrame, team_id: str, before: pd.Timestamp) -> dict[str, float | int | None]:
    rows = _team_history_rows(history, team_id, before)
    if rows.empty:
        return {"days_since_last": None, "matches_30d": 0}
    last = rows.iloc[-1]["date_utc"]
    days = (before - pd.Timestamp(last)).total_seconds() / 86400.0
    window_start = before - pd.Timedelta(days=30)
    matches_30d = int((rows["date_utc"] >= window_start).sum())
    return {"days_since_last": float(days), "matches_30d": matches_30d}


def _fifa_pre(
    ratings: pd.DataFrame,
    team_id: str,
    before: pd.Timestamp,
) -> dict[str, float | int | None | bool]:
    fifa = ratings[
        (ratings["source"] == "fifa")
        & (ratings["corpus"] == CORPUS_INTERNATIONAL)
        & (ratings["team_id"] == team_id)
        & (ratings["rating_date"] < before)
    ]
    if fifa.empty:
        return {"points": None, "rank": None, "available": False}
    row = fifa.sort_values("rating_date", kind="mergesort").iloc[-1]
    return {
        "points": float(row["value"]),
        "rank": int(row["rank"]) if pd.notna(row.get("rank")) else None,
        "available": True,
    }


def _targets(match_row: pd.Series) -> dict[str, Any]:
    if match_row["status"] != "completed" or pd.isna(match_row.get("home_goals")):
        return {
            "result_1x2": None,
            "home_goals": pd.NA,
            "away_goals": pd.NA,
            "total_goals": pd.NA,
        }
    hg, ag = int(match_row["home_goals"]), int(match_row["away_goals"])
    if hg > ag:
        result = "home"
    elif hg < ag:
        result = "away"
    else:
        result = "draw"
    return {
        "result_1x2": result,
        "home_goals": hg,
        "away_goals": ag,
        "total_goals": hg + ag,
    }


def features_for_match(
    match_row: pd.Series,
    history_view: pd.DataFrame,
    ratings: pd.DataFrame,
    conditions: pd.DataFrame,
    venues: pd.DataFrame,
    *,
    form_windows: list[int],
) -> dict[str, Any]:
    """One match's features from history strictly before its kickoff."""
    kickoff = pd.Timestamp(match_row["date_utc"]).tz_convert("UTC")
    neutral = bool(match_row["neutral"])
    is_home = not neutral
    home_id = match_row["home_team_id"]
    away_id = match_row["away_team_id"]

    home_elo, away_elo = elo_pre_at_kickoff(match_row, history_view)

    feat: dict[str, Any] = {
        "match_id": match_row["match_id"],
        "date_utc": kickoff,
        "corpus": match_row["corpus"],
        "home_team_id": home_id,
        "away_team_id": away_id,
        "neutral": neutral,
        "is_home": is_home,
        "is_knockout": is_knockout_stage(match_row.get("stage"), str(match_row["competition"])),
        "home_elo_pre": float(home_elo),
        "away_elo_pre": float(away_elo),
        "elo_diff": float(home_elo - away_elo),
    }

    hf = _fifa_pre(ratings, home_id, kickoff)
    af = _fifa_pre(ratings, away_id, kickoff)
    feat["home_fifa_points_pre"] = hf["points"]
    feat["away_fifa_points_pre"] = af["points"]
    feat["home_fifa_rank_pre"] = hf["rank"]
    feat["away_fifa_rank_pre"] = af["rank"]
    feat["fifa_points_diff"] = (
        float(hf["points"] - af["points"]) if hf["points"] is not None and af["points"] is not None else None
    )
    feat["fifa_rank_diff"] = (
        float(af["rank"] - hf["rank"]) if hf["rank"] is not None and af["rank"] is not None else None
    )
    feat["fifa_available"] = bool(hf["available"] and af["available"])

    for k in form_windows:
        hform = _form_stats(history_view, home_id, kickoff, k)
        aform = _form_stats(history_view, away_id, kickoff, k)
        feat[f"home_form_ppg_{k}"] = hform["ppg"]
        feat[f"away_form_ppg_{k}"] = aform["ppg"]
        feat[f"form_ppg_diff_{k}"] = (
            float(hform["ppg"] - aform["ppg"])
            if hform["ppg"] is not None and aform["ppg"] is not None
            else None
        )
        feat[f"home_form_gd_{k}"] = hform["gd_pg"]
        feat[f"away_form_gd_{k}"] = aform["gd_pg"]
        feat[f"form_gd_diff_{k}"] = (
            float(hform["gd_pg"] - aform["gd_pg"])
            if hform["gd_pg"] is not None and aform["gd_pg"] is not None
            else None
        )
        feat[f"form_{k}_available"] = bool(hform["n"] > 0 and aform["n"] > 0)

    hrest = _rest_stats(history_view, home_id, kickoff)
    arest = _rest_stats(history_view, away_id, kickoff)
    feat["home_days_since_last"] = hrest["days_since_last"]
    feat["away_days_since_last"] = arest["days_since_last"]
    feat["rest_diff"] = (
        float(arest["days_since_last"] - hrest["days_since_last"])
        if hrest["days_since_last"] is not None and arest["days_since_last"] is not None
        else None
    )
    feat["home_matches_30d"] = int(hrest["matches_30d"])
    feat["away_matches_30d"] = int(arest["matches_30d"])
    feat["congestion_diff"] = int(arest["matches_30d"]) - int(hrest["matches_30d"])

    cond = conditions_for_match(
        match_row["match_id"], home_id, away_id, venues, conditions, match_row
    )
    feat.update(
        {
            "altitude_m": cond["altitude_m"],
            "temp_c": cond["temp_c"],
            "humidity": cond["humidity"],
            "heat_stress": cond["heat_stress"],
            "home_travel_km": cond["home_travel_km"],
            "away_travel_km": cond["away_travel_km"],
            "travel_diff_km": (
                float(cond["home_travel_km"] - cond["away_travel_km"])
                if cond["home_travel_km"] is not None and cond["away_travel_km"] is not None
                else None
            ),
            "conditions_available": cond["conditions_available"],
        }
    )

    feat.update(_targets(match_row))
    return feat


def _index_fifa_ratings(ratings: pd.DataFrame) -> dict[str, pd.DataFrame]:
    fifa = ratings[(ratings["source"] == "fifa") & (ratings["corpus"] == CORPUS_INTERNATIONAL)]
    return {
        str(team): grp.sort_values("rating_date", kind="mergesort")
        for team, grp in fifa.groupby("team_id")
    }


def _fifa_pre_indexed(fifa_idx: dict[str, pd.DataFrame], team_id: str, before: pd.Timestamp) -> dict:
    grp = fifa_idx.get(team_id)
    if grp is None or grp.empty:
        return {"points": None, "rank": None, "available": False}
    sub = grp[grp["rating_date"] < before]
    if sub.empty:
        return {"points": None, "rank": None, "available": False}
    row = sub.iloc[-1]
    return {
        "points": float(row["value"]),
        "rank": int(row["rank"]) if pd.notna(row.get("rank")) else None,
        "available": True,
    }


def _form_from_records(records: deque, k: int) -> dict:
    if not records:
        return {"ppg": None, "gd_pg": None, "gf_pg": None, "ga_pg": None, "n": 0}
    tail = list(records)[-k:]
    n = len(tail)
    pts = sum(r[1] for r in tail)
    gd = sum(r[2] for r in tail)
    gf = sum(r[3] for r in tail)
    ga = sum(r[4] for r in tail)
    return {"ppg": pts / n, "gd_pg": gd / n, "gf_pg": gf / n, "ga_pg": ga / n, "n": n}


def _append_form_features(
    feat: dict,
    home_rec: deque,
    away_rec: deque,
    form_windows: list[int],
) -> None:
    for k in form_windows:
        hform = _form_from_records(home_rec, k)
        aform = _form_from_records(away_rec, k)
        feat[f"home_form_ppg_{k}"] = hform["ppg"]
        feat[f"away_form_ppg_{k}"] = aform["ppg"]
        feat[f"form_ppg_diff_{k}"] = (
            float(hform["ppg"] - aform["ppg"])
            if hform["ppg"] is not None and aform["ppg"] is not None
            else None
        )
        feat[f"home_form_gd_{k}"] = hform["gd_pg"]
        feat[f"away_form_gd_{k}"] = aform["gd_pg"]
        feat[f"form_gd_diff_{k}"] = (
            float(hform["gd_pg"] - aform["gd_pg"])
            if hform["gd_pg"] is not None and aform["gd_pg"] is not None
            else None
        )
        feat[f"form_{k}_available"] = bool(hform["n"] > 0 and aform["n"] > 0)


def build_features(
    matches: pd.DataFrame,
    ratings: pd.DataFrame,
    conditions: pd.DataFrame,
    venues: pd.DataFrame,
    *,
    form_windows: list[int] | None = None,
    corpus: str = CORPUS_INTERNATIONAL,
) -> pd.DataFrame:
    """Batch feature table in chronological order (O(n), no post-kickoff leakage)."""
    form_windows = form_windows or [5, 10]
    max_k = max(form_windows)
    subset = matches[matches["corpus"] == corpus].sort_values(
        ["date_utc", "match_id"], kind="mergesort"
    )
    fifa_idx = _index_fifa_ratings(ratings)
    elo_state: dict[str, float] = {}
    team_records: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_k))
    team_last: dict[str, pd.Timestamp] = {}
    team_recent_dates: dict[str, deque] = defaultdict(deque)
    params = EloParams()
    rows: list[dict] = []

    for _, m in subset.iterrows():
        kickoff = pd.Timestamp(m["date_utc"]).tz_convert("UTC")
        home_id, away_id = m["home_team_id"], m["away_team_id"]
        neutral = bool(m["neutral"])

        home_elo = elo_state.get(home_id, params.initial)
        away_elo = elo_state.get(away_id, params.initial)

        feat: dict[str, Any] = {
            "match_id": m["match_id"],
            "date_utc": kickoff,
            "corpus": m["corpus"],
            "home_team_id": home_id,
            "away_team_id": away_id,
            "neutral": neutral,
            "is_home": not neutral,
            "is_knockout": is_knockout_stage(m.get("stage"), str(m["competition"])),
            "home_elo_pre": float(home_elo),
            "away_elo_pre": float(away_elo),
            "elo_diff": float(home_elo - away_elo),
        }

        hf = _fifa_pre_indexed(fifa_idx, home_id, kickoff)
        af = _fifa_pre_indexed(fifa_idx, away_id, kickoff)
        feat["home_fifa_points_pre"] = hf["points"]
        feat["away_fifa_points_pre"] = af["points"]
        feat["home_fifa_rank_pre"] = hf["rank"]
        feat["away_fifa_rank_pre"] = af["rank"]
        feat["fifa_points_diff"] = (
            float(hf["points"] - af["points"]) if hf["points"] is not None and af["points"] is not None else None
        )
        feat["fifa_rank_diff"] = (
            float(af["rank"] - hf["rank"]) if hf["rank"] is not None and af["rank"] is not None else None
        )
        feat["fifa_available"] = bool(hf["available"] and af["available"])

        _append_form_features(feat, team_records[home_id], team_records[away_id], form_windows)

        def _rest(team_id: str) -> tuple[float | None, int]:
            last = team_last.get(team_id)
            if last is None:
                return None, 0
            days = (kickoff - last).total_seconds() / 86400.0
            cutoff = kickoff - pd.Timedelta(days=30)
            dates = team_recent_dates[team_id]
            while dates and dates[0] < cutoff:
                dates.popleft()
            return float(days), len(dates)

        h_days, h_30 = _rest(home_id)
        a_days, a_30 = _rest(away_id)
        feat["home_days_since_last"] = h_days
        feat["away_days_since_last"] = a_days
        feat["rest_diff"] = float(a_days - h_days) if h_days is not None and a_days is not None else None
        feat["home_matches_30d"] = h_30
        feat["away_matches_30d"] = a_30
        feat["congestion_diff"] = a_30 - h_30

        cond = conditions_for_match(m["match_id"], home_id, away_id, venues, conditions, m)
        feat.update(
            {
                "altitude_m": cond["altitude_m"],
                "temp_c": cond["temp_c"],
                "humidity": cond["humidity"],
                "heat_stress": cond["heat_stress"],
                "home_travel_km": cond["home_travel_km"],
                "away_travel_km": cond["away_travel_km"],
                "travel_diff_km": (
                    float(cond["home_travel_km"] - cond["away_travel_km"])
                    if cond["home_travel_km"] is not None and cond["away_travel_km"] is not None
                    else None
                ),
                "conditions_available": cond["conditions_available"],
            }
        )
        feat.update(_targets(m))
        rows.append(feat)

        if m["status"] == "completed" and pd.notna(m.get("home_goals")):
            _apply_elo_update(m, elo_state, params)
            hg, ag = int(m["home_goals"]), int(m["away_goals"])
            for team_id, is_home in ((home_id, True), (away_id, False)):
                pts = _team_points(hg, ag, for_home=is_home)
                gf, ga = _team_goals(hg, ag, for_home=is_home)
                team_records[team_id].append((kickoff, pts, gf - ga, gf, ga))
                team_last[team_id] = kickoff
                team_recent_dates[team_id].append(kickoff)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True).astype("datetime64[ns, UTC]")
    for col in ("home_goals", "away_goals", "total_goals"):
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    for col in ("home_fifa_rank_pre", "away_fifa_rank_pre"):
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df.sort_values(["date_utc", "match_id"], kind="mergesort").reset_index(drop=True)


def write_features_md(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Feature definitions (Plan 04)",
        "",
        "Point-in-time features: each value uses only data strictly before `date_utc`.",
        "",
        "| Feature | Description | Availability (% non-null) |",
        "|---------|-------------|---------------------------|",
    ]
    descriptions = {
        "match_id": "Unique match identifier",
        "date_utc": "Kickoff timestamp (UTC)",
        "corpus": "Data corpus discriminator",
        "home_elo_pre": "Self-computed Elo before kickoff (home)",
        "away_elo_pre": "Self-computed Elo before kickoff (away)",
        "elo_diff": "home_elo_pre - away_elo_pre",
        "home_fifa_points_pre": "Latest FIFA points before kickoff",
        "fifa_points_diff": "home - away FIFA points",
        "form_ppg_diff_5": "Points-per-game form diff (last 5)",
        "form_ppg_diff_10": "Points-per-game form diff (last 10)",
        "rest_diff": "away_days_since_last - home_days_since_last",
        "heat_stress": "Composite heat/altitude stress index",
        "travel_diff_km": "home_travel_km - away_travel_km",
        "result_1x2": "Target: home/draw/away (not a feature)",
    }
    skip = {"home_team_id", "away_team_id"}
    for col in df.columns:
        if col in skip:
            continue
        pct = 100.0 * df[col].notna().mean()
        desc = descriptions.get(col, "See Plan 04 feature set")
        lines.append(f"| `{col}` | {desc} | {pct:.1f}% |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_interim() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches = read_table("matches", INTERIM)
    ratings = read_table("ratings", INTERIM)
    conditions = read_table("conditions", INTERIM)
    try:
        venues = read_table("venues", INTERIM)
    except FileNotFoundError:
        venues = pd.DataFrame()
    return matches, ratings, conditions, venues


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    str_cols = ["match_id", "corpus", "home_team_id", "away_team_id", "result_1x2"]
    for col in str_cols:
        if col in out.columns:
            out[col] = out[col].astype("string")
    int_cols = (
        "home_matches_30d",
        "away_matches_30d",
        "congestion_diff",
        "home_goals",
        "away_goals",
        "total_goals",
        "home_fifa_rank_pre",
        "away_fifa_rank_pre",
    )
    for col in int_cols:
        if col in out.columns:
            out[col] = out[col].astype("Int64")
    float_cols = (
        "home_elo_pre",
        "away_elo_pre",
        "elo_diff",
        "home_fifa_points_pre",
        "away_fifa_points_pre",
        "fifa_points_diff",
        "fifa_rank_diff",
        "home_form_ppg_5",
        "away_form_ppg_5",
        "form_ppg_diff_5",
        "home_form_gd_5",
        "away_form_gd_5",
        "form_gd_diff_5",
        "home_form_ppg_10",
        "away_form_ppg_10",
        "form_ppg_diff_10",
        "home_form_gd_10",
        "away_form_gd_10",
        "form_gd_diff_10",
        "home_days_since_last",
        "away_days_since_last",
        "rest_diff",
        "altitude_m",
        "temp_c",
        "humidity",
        "heat_stress",
        "home_travel_km",
        "away_travel_km",
        "travel_diff_km",
    )
    for col in float_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    bool_cols = (
        "neutral",
        "is_home",
        "is_knockout",
        "fifa_available",
        "form_5_available",
        "form_10_available",
        "conditions_available",
    )
    for col in bool_cols:
        if col in out.columns:
            out[col] = out[col].astype("bool")
    return out


def run_build(cfg: AppConfig | None = None) -> pd.DataFrame:
    cfg = cfg or get_config()
    matches, ratings, conditions, venues = load_interim()
    df = build_features(
        matches,
        ratings,
        conditions,
        venues,
        form_windows=cfg.model.form_windows,
    )
    df = _prepare_features(df)
    validate_table(df, "features")
    write_table(
        df,
        "features",
        PROCESSED,
        sort_by=["date_utc", "match_id"],
        plan=PLAN,
        schema_name="features",
    )
    write_features_md(df, project_root() / "data" / PROCESSED / "FEATURES.md")
    run_feature_eda(df)
    logger.info("Built features: %d rows", len(df))
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the leakage-safe feature table")
    parser.parse_args(argv)
    get_config.cache_clear()
    cfg = load_config()
    set_seeds(cfg.seed)
    run_build(cfg)


if __name__ == "__main__":
    main()
