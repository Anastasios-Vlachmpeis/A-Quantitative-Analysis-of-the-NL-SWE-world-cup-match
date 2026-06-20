# Bet Ticket — Netherlands vs Sweden (FIFA World Cup 2026)

**Kickoff:** 2026-06-20 17:00 UTC · **Mock bankroll:** $1,000 · **Model:** `ensemble_weighted` (1X2) / `poisson` scoreline
**Staking:** fractional Kelly (¼) for value bets; de-vig method: multiplicative

> Pre-registered before kickoff. This file records the *decisions and rationale*;
> the machine-generated slip is `data/processed/live/bet_slip.parquet` and
> `reports/PREDICTION.md`. CLV and P&L are settled after the match.

---

## Bet 1 — Netherlands to win (1X2 home) · **VALUE BET** ✅

| Field | Value |
|---|---|
| Selection | Netherlands to win (1X2: home) |
| Bookmaker / odds | Pinnacle **1.74** (best available; William Hill 1.73) |
| Model probability | **62.7%** |
| Market probability (de-vigged) | **55.6%** |
| Expected value | **+9.1%** |
| Stake (¼ Kelly) | **$31** |

**Rationale.** This is the one selection where our model genuinely disagrees with
the sharp market in our favour. Netherlands carry a **+127 Elo edge** over Sweden,
and at a **neutral** venue (no home advantage for either side) the model puts them
at 62.7% vs the market's de-vigged 55.6% — a +9.1% EV bet. It cleared our value
machinery (EV > 2% min-edge) and is the only 1X2 outcome that is +EV (draw and away
are both negative: your probability is *lower* than the book's, so betting them
would destroy value).

**Honest caveat.** Our model is documented to be **over-confident on favourites**
(a side-effect of the goal-volume bias below), so the true edge may be smaller than
9% — the market's ~56% could be closer to the truth. Fractional Kelly already sizes
this conservatively. This is exactly what a pre-registered N=1 bet tests: we'll know
from the result and from **CLV** (did 1.74 beat the closing line?).

---

## Bet 2 — Over 2.5 goals · **SPECULATIVE (NOT a value bet)** ⚠️

| Field | Value |
|---|---|
| Selection | Over 2.5 total goals |
| Bookmaker / odds | William Hill **1.70** |
| Model probability | 71.4% |
| Market probability (de-vigged) | 54.7% |
| Model-claimed EV | +21.4% |
| **Our verdict** | **Likely −EV — bias artifact** |
| Stake (flat, discretionary) | **$20** (2% of bankroll) |

**Why this is flagged, not trusted.** The model screams +21% EV here, but our own
market comparison shows the goal models **systematically over-predict goals**:
model P(over 2.5) = **71%** vs the 15-book sharp consensus of **~56%** (~+0.8 goals).
That gap is a **calibration artifact**, not edge — pooled Poisson models over-apply
the "favourite scores a lot" pattern (learned from lopsided qualifiers) to a tight
tournament match. By our value criterion this bet would be **excluded** (and our
config disables totals markets for exactly this reason).

**Why it's here anyway.** Placed as a small, clearly-labelled **speculative** stake
at user discretion — a live test of the raw scoreline model's goal view. It is *not*
part of the value strategy and should be read in the report as "the model's biased
call, placed deliberately so we can see the bias play out." If prioritising pure
discipline, **skip this bet.**

---

## Summary

| Bet | Type | Stake | Odds | If it wins |
|---|---|---|---|---|
| Netherlands to win | Value (+9.1% EV) | $31 | 1.74 | +$54.0 |
| Over 2.5 goals | Speculative (likely −EV) | $20 | 1.70 | +$14.0 |
| **Total exposure** | | **$51** (5.1% of bankroll) | | |

**Post-match (settle in Plan 11):** record result, P&L, and **CLV** for each bet
(odds taken vs closing line). Headline metric is CLV, not one-match profit — N=1.
