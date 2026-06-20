# A Quantitative Analysis of Netherlands vs Sweden + Betting Strategy

A quantitative-research project that predicts the **FIFA World Cup 2026** match
**Netherlands vs Sweden**, prices betting markets against bookmaker odds, and
recommends a staking strategy on a **mock $1,000 bankroll** — built and frozen
*before kickoff* as a pre-registered, market-anchored exercise.

---

## The thesis (read this first)

A single match is **N = 1**: its result can neither confirm nor refute a model.
So the project is deliberately built so that:

1. **The evidence is the backtest, not the bet.** Models are judged by a
   **walk-forward** backtest over ~19,700 international matches using proper
   scoring rules (RPS / log-loss / Brier) with bootstrap confidence intervals.
2. **The live match is a pre-registered showcase.** Predictions and bets are
   frozen and git-tagged (`prematch-freeze`) before kickoff — no hindsight.
3. **The market is the benchmark.** The sharp, de-vigged bookmaker line is the
   thing to beat. **Closing Line Value (CLV)** — not one-match profit — is the
   honest measure of skill.

The point of the whole pipeline is to *earn the right* to trust our probability
over the bookmaker's on the one bet we actually place.

---

## Headline results

**Model quality (walk-forward RPS, lower is better; 19,500-match common set):**

| Model | RPS | 95% CI |
|---|---|---|
| ensemble_weighted | **0.179** | [0.177, 0.181] |
| ensemble_avg | 0.180 | [0.178, 0.181] |
| poisson | 0.181 | [0.178, 0.183] |
| dixon_coles | 0.181 | [0.178, 0.183] |
| elo | 0.186 | [0.184, 0.188] |
| baseline (base rates) | 0.226 | [0.224, 0.227] |

The goal models and ensembles **decisively beat the base-rate baseline** and
edge out Elo — but the **top four are statistically tied** (overlapping CIs), so
we frame them as a cluster, not a clear winner.

**Did we beat the market?** On internationals it's untestable — there is no
historical international closing-odds data — so "beat the market" is a **live,
N=1 question** answered only by the NL–SWE bet's CLV. (Honest by construction.)

**A bias we found and quantified.** Comparing the model to the live market
exposed a real flaw: the goal models **over-predict goals** — model
P(over 2.5) = **71%** vs the sharp consensus **~56%**. We traced it to pooled
Poisson models over-applying the "favourite scores a lot" pattern, restricted the
live bet to **1X2 only**, and documented the rest. Catching your own model's bias
against the market is the project's most important result.

---

## The bet (pre-registered)

See **[`reports/BETS.md`](reports/BETS.md)** and **`reports/PREDICTION.md`**.

- **Value bet — Netherlands to win** @ 1.74: model 62.7% vs market 55.6% →
  **+9.1% EV**, ¼-Kelly stake $31. (+127 Elo edge, neutral venue.)
- **Speculative — Over 2.5** @ 1.70: the model loves it (+21% claimed EV) but our
  own analysis flags it as a likely **−EV goal-volume artifact**; placed small and
  clearly labelled, *not* as a value bet.

Why not bet all three outcomes? You mathematically **cannot** be +EV on all of
1/X/2 at one bookmaker (your probs sum to 1; their vigged probs sum to >1). Value
comes from betting *only* the side the market under-prices — here, the home win.

---

## How it works (pipeline)

Each phase reads versioned artifacts from disk and writes new ones (with manifest
sidecars for reproducibility). Build order and detail live in `plans/`.

| Phase | What it does | Key output |
|---|---|---|
| 01 Foundation | config, IO+manifests, seeding, schemas | package skeleton |
| 02 Acquisition | results, FIFA/Elo, **club + live odds**, venues | `data/raw/*` |
| 03 Cleaning | canonical tables, entity resolution, `corpus` split | `data/interim/*` |
| 04 Features | point-in-time, **leakage-safe** features (self-computed Elo) | `features.parquet` |
| 05 Harness | RPS/log-loss/Brier, calibration, walk-forward CV, de-vig | `eval/*` |
| 06 Models | ladder: baseline→Elo→Poisson→Dixon-Coles→bivariate→Bayesian→ML | `predictions_*` |
| 07 Induction | scoreline → all markets (analytic + Monte Carlo) | `market_probs_*` |
| 08 Comparison | leaderboard + ensembles + honest selection | `MODEL_SELECTION.md` |
| 09 Strategy | EV / fractional Kelly / bankroll MC / CLV (club corpus) | `bets_*`, `bankroll_sim_*` |
| 10 Live | the NL–SWE call + bet slip, frozen pre-kickoff | `data/processed/live/*` |
| 11 Post-match | settle, CLV, P&L, report | `reports/*` |

**Hybrid corpus decision:** goal models are fit/evaluated on **internationals
only**; the EV/Kelly/CLV *machinery* is validated on a large **club-league** odds
sample (the two are never pooled). This is why model quality and betting mechanics
are reported on separate corpora — stated openly as a limitation.

---

## Limitations (stated honestly)

- **N = 1** live match — the bet illustrates, the backtest is the evidence.
- **Goal-volume bias** — goal models over-predict totals; live bet limited to 1X2.
- **No historical international odds** — "beat the market" is only the live N=1 test.
- **FIFA ratings end ~Sep 2024** — stale for the live match; the goal models rely
  on self-computed Elo (current) instead, and FIFA is treated cautiously.
- **Backtest approximations** — walk-forward refits every N matches for tractability
  (disclosed; immaterial over a 20-year history).

---

## Setup & reproduction

Requires **Python 3.11+** (3.12 recommended).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on *nix)
pip install -e .
```

Run the pipeline in order (each step has tests):

```bash
python -m nlvswe.data.acquire --source all     # needs ODDS_API_KEY in env / .env
python -m nlvswe.data.clean --table all
python -m nlvswe.features.build
python -m nlvswe.models.run --model all        # --refit-every 100 for speed
python -m nlvswe.betting.induction --model all
python -m nlvswe.eval.compare
python -m nlvswe.live.predict                  # add --refresh --freeze before kickoff
pytest                                          # full test suite
```

**The freeze:** set `scope: live` in `config/config.yaml`, then
`python -m nlvswe.live.predict --refresh --freeze` creates the `prematch-freeze`
git tag; the commit hash is recorded in `reports/PREDICTION.md`. Nothing that feeds
the bet may change after the tag — post-match work is append-only.

---

## Layout

```
config/config.yaml            # single source of truth (seed, corpus, markets, bankroll)
src/nlvswe/                    # all logic (data, features, eval, models, betting, live)
data/raw|interim|processed/   # artifacts (gitignored; manifests track provenance)
reports/                      # PREDICTION.md, BETS.md, MODEL_SELECTION.md, figures/
plans/                        # phase-by-phase build specs (read 00-overview first)
tests/                        # pytest per phase
```

See `plans/README.txt` for the full build order, and `reports/` for the writeups.
