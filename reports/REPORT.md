# Netherlands vs Sweden: a model, a bet, and it's bias

Final score: Netherlands 5 - Sweden 1. Both of my bets won!

The result is worth mentioning, but it isn't what this project is really about. A single match can't tell you much about whether a model is good, so most of this report focuses on the process, the validation, and one mistake I caught before kickoff.

> Written after the match, on top of the frozen state (supposed to be commit `7273af0`, was actually commit `b399a33`). Nothing that produced the prediction or the bets changed once the result was known. This file only adds the outcome and the post-mortem.

## What I did

I wanted to predict this match with a transparent statistical model, price the bookmaker's markets against it, place a small mock bet wherever my numbers disagreed with the market, and freeze everything before kickoff so that I didn't quietly "rewrite history" after the match was over. Bets might or might not have been placed, for *"research purposes"* only.

Obviously, one match proves nothing. Bad models win bets and good models lose them all the time (gamblers love to make single-event-based assumptions). The model is evaluated on a walk-forward backtest covering approximately 19,700 international matches. My live bet is one additional observation.

## The data I used

International results back to 2006 (the martj42 dataset), FIFA rankings, a self-computed Elo, plus bookmaker odds from two sources (live World Cup prices from the-odds-api, and a long club-league history from football-data that I used to test the betting math). Cleaning turned that into a small set of canonical tables under one strict rule: a feature for a match may only use information that existed *before* kickoff. I verified that this holds, by flipping a match's result and confirming it's pre-match features don't change.

One choice worth naming: the goal models are trained and scored only on internationals, while the betting machinery (sizing the stakes and measuring closing-line value) is validated on the club sample, because usable international closing-odds history barely exists. The two are never mixed, which is honest but carries a cost I come back to at the end.

## How I judged the models

To score the predictions, I built a benchmark using the Ranked Probability Score alongside log-loss, with bootstrap confidence intervals because point estimates are not that reliable on small samples. Predictions are always walk-forward, meaning each match is predicted by a model trained only on matches before it, and I ran the de-vigged bookmaker line through the exact same scoring, essentially treating the market like one more "competitor".

## What the models scored

| Model | Walk-forward RPS | 95% CI |
|---|---|---|
| ensemble_weighted | 0.179 | [0.177, 0.181] |
| ensemble_avg | 0.180 | [0.178, 0.181] |
| poisson | 0.181 | [0.178, 0.183] |
| dixon_coles | 0.181 | [0.178, 0.183] |
| elo | 0.186 | [0.184, 0.188] |
| baseline (base rates) | 0.226 | [0.224, 0.227] |

The structured models and the ensembles clearly beat the base-rate floor and edge past plain Elo, which is nice, but the top four are within each other's confidence intervals, so I consider them tied. *Calling the ensemble "best" by 0.002 of RPS is just dumb.*

Did anything beat the market? On internationals I can't really say, as there is no historical international odds to score against. That question only got an answer on this one match. 

![Walk-forward RPS leaderboard](figures/compare_rps_leaderboard.png)
![Calibration overlay across models](figures/compare_calibration_overlay.png)

## From a scoreline to a bet

A goal model gives a full grid of scoreline probabilities, which you sum into whatever market you care about. I computed each market two ways (exact summation, and a Monte Carlo over 10,000 simulated matches) and checked whether they agree within sampling error (they do). The picture for Netherlands vs Sweden: Netherlands clear favourites, most likely scoreline around 2-1, with an expected total near 3.7 goals.

![Scoreline distribution](figures/induction_scoreline_poisson.png)

## The bias I caught

This has nothing to do with winning the bet.

Lined up against the market, my match-result numbers were close (62.7% on Netherlands against the market's 55.6%), but the totals were not. My model put over 2.5 goals at 71%. The sharp consensus across 15 books was about 56%. This 16-point gap initially felt like a discovery, but considering how liquid this market is I quickly realized I'm probably the one who is wrong.

After digging into it, the issue makes sense. The Poisson model learns from a lot of qualifiers where a lot of strong teams score four, five, or six goals on the regular against weaker opponents. Those matches push the attacking strengths upward. When this same model sees a tournament game between two strong sides, it can carry some of that "optimism" forward and overestimate goals.

Once I saw that, I disabled the totals markets for the live bet and kept only the match result, where the "disagreement with the market" is small enough to be reasonable.

Again, *thinking reasonably*, when your number is miles from a sharp line, the sharp line is usually right.

## The bet, and what happened

| Bet | Type | Model p | Market p | Odds | Stake | Result | P&L |
|---|---|---|---|---|---|---|---|
| Netherlands win | value (+9.1% EV) | 0.627 | 0.556 | 1.74 | $31 | won | +$22.94 |
| Over 2.5 goals | speculative (flagged -EV) | 0.714 | 0.547 | 1.70 | $20 | won | +$14.00 |
| Total | | | | | $51 | | +$36.94 |

Netherlands to win at 1.74 was the real bet: my 62.7% against the market's 55.6% is a +9.1% edge, sized with quarter-Kelly to about $31. The over 2.5 at 1.70 I placed deliberately, small and labelled speculative, precisely because I had already flagged it as a probable loser. I wanted to watch the bias play out instead of only writing about it.

Both won. Six goals cleared 2.5 with room to spare, and the total came to about +$37 on the thousand, which is pleasant (and proves nothing).

The Over 2.5 bet winning does not change my view of the model.

Before kickoff I already flagged the totals model as unreliable because of the gap with the market. A single winning bet is not even close to overturn my conclusion. If anything, this is exactly why I wanted the predictions frozen in advance. It is easy to become convinced by a result after the fact.

![Live bet edges vs the market](figures/live_edges.png)

## A metric I built everything around, and couldn't measure

Over the span of the project I was sure that closing-line value (did my price beat the final pre-kickoff line?) is the only honest short-term read on skill, since one bet's profit isn't necessarily supposed to be there. Then I captured exactly one odds snapshot before kickoff, which means the price I "took" and the "closing" line are basically the same number, so my CLV is roughly zero.. and says nothing.

That's a miss, with the fix is being the boring process of polling the odds repeatedly in the hours before kickoff to just keep the last one. Still, I'm leaving it in.

## What I trust and what I don't

- The match-result model is calibrated and beats the baseline across twenty years, while its live call pointed the same way as the market. I would run it again.
- The totals model I would not trust, not without fixing the goal-volume bias first, either through a competition-type parameter or by shrinking the team strengths so favourites stop running up imaginary scores.

This match remains an illustration rather than evidence. The backtest is the actual evidence, and the validation process earned its keep by catching several early bugs, including a goal-strength estimator that initially predicted almost every match would end in a draw, and a backtest that ran for hours before I vectorized the slow parts.

## Running it yourself

The build order and the per-phase tests are in the [README](../README.md). Nothing that fed the bet was altered after the "freeze", and this report is just a write up on top of it.