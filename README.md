# NL vs SWE — Quantitative Analysis

Quantitative research project predicting the **FIFA World Cup 2026** match
**Netherlands vs Sweden**, pricing betting markets against bookmaker odds, and
recommending a staking strategy on a mock $1,000 bankroll.

## Framing

- The single match is **N=1** — it cannot validate a model. Statistical evidence
  comes from a **walk-forward backtest** over historical matches.
- The live match is a **pre-registered showcase**: predictions and bets are frozen
  (git-tagged `prematch-freeze`) before kickoff.
- **Closing Line Value (CLV)** is the headline KPI for betting, not one-bet profit.

## Setup

Requires **Python 3.11+**.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -e .
```

## Quick check (Plan 01)

```bash
python -m nlvswe._dummy
pytest tests/test_foundation.py
```

The dummy run writes `data/processed/dummy.parquet` and a manifest sidecar,
proving the artifact pipeline works.

## Data acquisition (Plan 02)

Pull raw sources into `data/raw/` (gitignored). Each file gets a manifest sidecar.

```bash
# Individual sources
python -m nlvswe.data.acquire --source results
python -m nlvswe.data.acquire --source odds_history

# All sources (requires ODDS_API_KEY env var for live odds)
set ODDS_API_KEY=your_key_here   # Windows; never commit this
python -m nlvswe.data.acquire --source all

pytest tests/test_acquire.py
```

See `data/raw/SOURCES.md` for coverage notes from the validation spike.

## Cleaning (Plan 03)

Build canonical interim tables from raw data:

```bash
python -m nlvswe.data.clean --table all
pytest tests/test_clean.py
```

Outputs land in `data/interim/` with manifests; see `data/interim/VALIDATION.md`.

## Feature engineering (Plan 04)

Build point-in-time, leakage-safe features for international matches:

```bash
python -m nlvswe.features.build
pytest tests/test_features.py
```

Outputs: `data/processed/features.parquet`, `data/processed/FEATURES.md`, `reports/figures/eda_feat_*`.

## Evaluation harness (Plan 05)

Shared scoring, calibration, walk-forward CV, and market benchmark:

```bash
python -m nlvswe.eval.backtest --model constant
pytest tests/test_eval.py
```

Outputs: `data/processed/predictions_<model>.parquet`, `data/processed/eval_scores.parquet`, `reports/figures/calibration_*.png`.

## Models (Plan 06)

International-only model ladder with walk-forward backtest:

```bash
python -m nlvswe.models.run --model baseline
python -m nlvswe.models.run --model all
pytest tests/test_models.py
```

See `models/MODELS.md` for assumptions and limitations. Outputs: `data/processed/predictions_<model>.parquet`.

## Probability induction (Plan 07)

Turn scoreline distributions into betting market probabilities:

```bash
python -m nlvswe.betting.induction --model poisson
python -m nlvswe.betting.induction --model all
pytest tests/test_induction.py
```

Outputs: `data/processed/market_probs_<model>.parquet`, `reports/figures/induction_*`.

## Layout

```
config/config.yaml     # single source of truth
src/nlvswe/            # all logic lives here
data/raw|interim|processed/   # artifacts (gitignored except .gitkeep)
plans/                 # build plans (read 00-overview first)
```

See `plans/README.txt` for the full build order (Phases 01–11).
