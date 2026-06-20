# Netherlands vs Sweden: a pre-registered model and an honest post-mortem

I built a statistical model to predict the FIFA World Cup 2026 match Netherlands vs Sweden, priced the bookmaker's markets against it, placed a small mock bet where my numbers disagreed with the market, and froze the whole thing in before kickoff.

**The full writeup is in [`reports/REPORT.md`](reports/REPORT.md).** This is where the story lives, including the model bias I caught and the one metric I built everything around and then failed to measure.

## The short version

The match finished 5-1 to Netherlands and both my bets won, which is the least informative fact here, because one match cannot tell you whether a model is any good. The real work is a walk-forward backtest over roughly 19,700 international matches, scored with the bookmaker as the benchmark.

What came out of it:

- The match-result model is well-calibrated and beats a base-rate floor on twenty years of data (RPS 0.18 against the floor's 0.23).
- It sits in a statistical tie with the other goal models (overlapping confidence intervals), so I don't claim a single "best" one.
- It has a real, measured flaw: it over-predicts goals (it priced over 2.5 at 71% when the sharp market said about 56%), which traces back to the Poisson models over-applying lopsided-qualifier scorelines to a tight match. I caught it by pricing against the market, then cut totals from the live bet.
- "Did I beat the market?" only has a live answer, because there is no historical international odds to test against, and that answer is one match wide.

## Headline numbers

Walk-forward Ranked Probability Score (lower is better; 19,500-match common set):

| Model | RPS | 95% CI |
|---|---|---|
| ensemble_weighted | 0.179 | [0.177, 0.181] |
| ensemble_avg | 0.180 | [0.178, 0.181] |
| poisson | 0.181 | [0.178, 0.183] |
| dixon_coles | 0.181 | [0.178, 0.183] |
| elo | 0.186 | [0.184, 0.188] |
| baseline (base rates) | 0.226 | [0.224, 0.227] |

## The bet (frozen before kickoff)

The pre-registered call is in [`reports/BETS.md`](reports/BETS.md) and [`reports/PREDICTION.md`](reports/PREDICTION.md). Netherlands to win at 1.74 was the real one (my 62.7% against the market's 55.6%, a +9.1% edge, sized with quarter-Kelly), alongside a small over 2.5 that I flagged as a probable loser and placed anyway to watch the bias play out. Both won. The frozen state is the git tag `prematch-freeze` at commit `7273af0`, and the post-match work is append-only on top of it.

## How it's built

Each phase reads versioned artifacts off disk and writes new ones, every artifact carries a manifest sidecar (git commit and config hash), so the whole thing is reproducible and re-runnable a phase at a time.

| Phase | What it does | Key output |
|---|---|---|
| 01 Foundation | config, IO + manifests, seeding, schemas | package skeleton |
| 02 Acquisition | results, FIFA/Elo, club + live odds, venues | `data/raw/*` |
| 03 Cleaning | canonical tables, entity resolution, `corpus` split | `data/interim/*` |
| 04 Features | point-in-time, leakage-safe features (self-computed Elo) | `features.parquet` |
| 05 Harness | scoring, calibration, walk-forward CV, de-vig | `eval/*` |
| 06 Models | the ladder, baseline up to a Bayesian hierarchical model | `predictions_*` |
| 07 Induction | scoreline into every market (analytic + Monte Carlo) | `market_probs_*` |
| 08 Comparison | leaderboard, ensembles, the honest selection | `MODEL_SELECTION.md` |
| 09 Strategy | expected value, fractional Kelly, bankroll Monte Carlo, CLV | `bets_*` |
| 10 Live | the NL vs SWE call + bet slip, frozen pre-kickoff | `data/processed/live/*` |
| 11 Post-match | settle, CLV, P&L, report | `reports/*` |

One split worth knowing: the goal models are trained and scored only on internationals, while the betting math (staking and closing-line value) is checked on a club-league odds sample, because usable international closing-odds history barely exists. The two never mix, which is honest but means model quality and betting mechanics are measured on different data.

## Setup and running it

Needs Python 3.11+ (3.12 is the comfortable choice).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on *nix)
pip install -e .
```

Run the phases in order, each has its own tests:

```bash
python -m nlvswe.data.acquire --source all     # ODDS_API_KEY in env or .env
python -m nlvswe.data.clean --table all
python -m nlvswe.features.build
python -m nlvswe.models.run --model all        # --refit-every 100 for speed
python -m nlvswe.betting.induction --model all
python -m nlvswe.eval.compare
python -m nlvswe.live.predict                  # --refresh --freeze before kickoff
pytest                                          # full suite
```

To freeze for a live match, set `scope: live` in `config/config.yaml` and run
`python -m nlvswe.live.predict --refresh --freeze`, which writes the live artifacts and tags the commit `prematch-freeze`. Nothing that feeds the bet is allowed to change after that tag.

## Layout

```
config/config.yaml            # single source of truth (seed, corpus, markets, bankroll)
src/nlvswe/                   # all logic (data, features, eval, models, betting, live)
data/raw|interim|processed/   # artifacts (gitignored; manifests track provenance)
reports/                      # REPORT.md, PREDICTION.md, BETS.md, MODEL_SELECTION.md, figures/
plans/                        # phase-by-phase build specs (start at 00-overview)
tests/                        # pytest, one set per phase
```

The per-phase build specs are in `plans/`, and the writeups (start with [`reports/REPORT.md`](reports/REPORT.md)) are in `reports/`.
