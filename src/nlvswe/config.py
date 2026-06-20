"""Load and validate config.yaml (pydantic)."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TargetMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    home: str
    away: str
    competition: str
    season: int
    kickoff_utc: str | None = None


class DataCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_eval: str
    betting_validation: str
    internationals_betting_sanity: bool


class IntlEndpoints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results_base: str
    fifa_ranking: str
    elo_world: str


class IntlSources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: str
    fifa_rank: str
    elo: str
    xg: str
    endpoints: IntlEndpoints


class FootballDataEndpoints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    leagues: list[str]
    first_season_year: int


class ClubOddsSources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    football_data: str
    endpoints: FootballDataEndpoints


class OddsLiveEndpoints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str


class OddsLiveSources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    sport_key: str
    api_key_env: str
    snapshot_dir: str
    endpoints: OddsLiveEndpoints


class VenuesEndpoints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    open_meteo_archive: str
    open_meteo_elevation: str


class VenuesSources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixtures: str
    weather: str
    fixtures_template: str
    endpoints: VenuesEndpoints


class DataSources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intl: IntlSources
    club_odds: ClubOddsSources
    odds_live: OddsLiveSources
    venues: VenuesSources


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    history_start: str
    corpus: DataCorpus
    sources: DataSources
    include_xg: bool


class BayesianConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draws: int
    tune: int
    chains: int
    target_accept: float


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_goals: int = 10
    time_decay_half_life_days: int = 365
    form_windows: list[int] = Field(default_factory=lambda: [5, 10])
    refit_every: int = 10
    mc_samples: int = 10000
    bayesian: BayesianConfig


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cv_window: Literal["expanding", "rolling"]
    min_history_matches: int
    bootstrap_samples: int
    calibration_bins: int


class BettingMarkets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    totals_lines: list[float]
    ah_lines: list[float]
    correct_score_top_n: int


class BettingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bankroll: float = 1000.0
    kelly_fraction: float = 0.25
    max_stake_fraction: float = 0.05
    min_edge: float = 0.02
    devig_method: Literal["multiplicative", "shin"] = "multiplicative"
    flat_stake_control: bool = True
    bookmakers: list[str]
    markets: BettingMarkets


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    scope: Literal["history", "live"]
    target_match: TargetMatch
    data: DataConfig
    model: ModelConfig
    eval: EvalConfig
    betting: BettingConfig


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _project_root() -> Path:
    return project_root()


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate config.yaml."""
    if path is None:
        path = _project_root() / "config" / "config.yaml"
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return cached config (default path)."""
    return load_config()


def config_hash(cfg: AppConfig) -> str:
    """Stable sha256 of the normalized config dict."""
    normalized = cfg.model_dump(mode="json")
    payload = yaml.dump(normalized, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
