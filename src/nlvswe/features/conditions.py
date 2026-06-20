"""Venue, weather, and travel distance proxies for the feature table."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

# Approximate country centroids (lat, lon) for travel distance proxy.
TEAM_CENTROIDS: dict[str, tuple[float, float]] = {
    "netherlands": (52.37, 4.90),
    "sweden": (59.33, 18.07),
    "germany": (52.52, 13.41),
    "france": (48.86, 2.35),
    "england": (51.51, -0.13),
    "spain": (40.42, -3.70),
    "italy": (41.90, 12.50),
    "brazil": (-15.79, -47.88),
    "argentina": (-34.60, -58.38),
    "usa": (38.91, -77.04),
    "mexico": (19.43, -99.13),
    "portugal": (38.72, -9.14),
    "belgium": (50.85, 4.35),
    "croatia": (45.81, 15.98),
    "denmark": (55.68, 12.57),
    "poland": (52.23, 21.01),
    "ukraine": (50.45, 30.52),
    "turkey": (39.93, 32.86),
    "japan": (35.68, 139.76),
    "south_korea": (37.57, 126.98),
    "australia": (-35.28, 149.13),
    "morocco": (33.97, -6.85),
    "senegal": (14.69, -17.44),
    "nigeria": (9.08, 7.49),
    "cameroon": (3.87, 11.52),
    "ghana": (5.60, -0.19),
    "ivory_coast": (5.36, -4.01),
    "egypt": (30.04, 31.24),
    "saudi_arabia": (24.71, 46.67),
    "iran": (35.69, 51.39),
    "qatar": (25.29, 51.53),
    "canada": (45.42, -75.70),
    "switzerland": (46.95, 7.46),
    "austria": (48.21, 16.37),
    "serbia": (44.79, 20.45),
    "wales": (51.48, -3.18),
    "scotland": (55.95, -3.19),
    "uruguay": (-34.90, -56.16),
    "colombia": (4.71, -74.07),
    "chile": (-33.45, -70.67),
    "ecuador": (-0.18, -78.47),
    "costa_rica": (9.93, -84.08),
    "panama": (8.98, -79.52),
    "honduras": (14.07, -87.19),
    "jamaica": (18.00, -76.79),
    "paraguay": (-25.26, -57.58),
    "peru": (-12.05, -77.04),
    "tunisia": (36.81, 10.18),
    "algeria": (36.75, 3.06),
    "south_africa": (-25.75, 28.19),
    "china": (39.90, 116.40),
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def heat_stress(temp_c: float | None, humidity: float | None, altitude_m: float | None) -> float | None:
    """Simple composite: higher temp, humidity, altitude -> higher stress."""
    if temp_c is None and humidity is None and altitude_m is None:
        return None
    t = float(temp_c) if temp_c is not None else 20.0
    h = float(humidity) if humidity is not None else 50.0
    a = float(altitude_m) if altitude_m is not None else 0.0
    # Normalised heuristic (not physiological); for relative ranking only.
    return max(0.0, (t - 15.0) * 0.05 + (h - 40.0) * 0.01 + a / 500.0)


def travel_km(team_id: str, venue_lat: float | None, venue_lon: float | None) -> float | None:
    if venue_lat is None or venue_lon is None:
        return None
    centroid = TEAM_CENTROIDS.get(team_id)
    if centroid is None:
        return None
    return haversine_km(centroid[0], centroid[1], float(venue_lat), float(venue_lon))


def is_knockout_stage(stage: Any, competition: str) -> bool:
    text = f"{stage} {competition}".lower()
    keywords = ("knockout", "quarter", "semi", "final", "round of", "last 16", "last 32")
    return any(k in text for k in keywords)


def conditions_for_match(
    match_id: str,
    home_team_id: str,
    away_team_id: str,
    venues: pd.DataFrame,
    conditions: pd.DataFrame,
    match_row: pd.Series,
) -> dict[str, float | None | bool]:
    """Extract nullable condition features for one match."""
    out: dict[str, float | None | bool] = {
        "altitude_m": None,
        "temp_c": None,
        "humidity": None,
        "heat_stress": None,
        "home_travel_km": None,
        "away_travel_km": None,
        "conditions_available": False,
    }
    cond = conditions[conditions["match_id"] == match_id]
    if not cond.empty:
        c = cond.iloc[0]
        out["altitude_m"] = float(c["altitude_m"]) if pd.notna(c.get("altitude_m")) else None
        out["temp_c"] = float(c["temp_c"]) if pd.notna(c.get("temp_c")) else None
        out["humidity"] = float(c["humidity"]) if pd.notna(c.get("humidity")) else None
        if any(out[k] is not None for k in ("altitude_m", "temp_c", "humidity")):
            out["conditions_available"] = True

    venue_lat = venue_lon = None
    venue_id = match_row.get("venue_id")
    if pd.notna(venue_id) and not venues.empty:
        v = venues[venues["venue_id"] == venue_id]
        if not v.empty:
            if pd.notna(v.iloc[0].get("lat")):
                venue_lat = float(v.iloc[0]["lat"])
            if pd.notna(v.iloc[0].get("lon")):
                venue_lon = float(v.iloc[0]["lon"])
            if out["altitude_m"] is None and pd.notna(v.iloc[0].get("altitude_m")):
                out["altitude_m"] = float(v.iloc[0]["altitude_m"])

    out["heat_stress"] = heat_stress(out["temp_c"], out["humidity"], out["altitude_m"])
    out["home_travel_km"] = travel_km(home_team_id, venue_lat, venue_lon)
    out["away_travel_km"] = travel_km(away_team_id, venue_lat, venue_lon)
    return out
