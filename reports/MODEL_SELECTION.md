# Model selection

**Criterion:** Lowest walk-forward RPS on common match set with acceptable calibration

**Selected model:** `ensemble_weighted`
**Walk-forward RPS:** 0.1791 (95% CI 0.1774–0.1808)

## Market benchmark

Market predictions not available for comparison.

## Leaderboard (RPS)

| Rank | Model | RPS | 95% CI | Accuracy (footnote) |
|------|-------|-----|--------|---------------------|
| 1 | `ensemble_weighted` | 0.1791 | [0.1774, 0.1808] | 58.8% |
| 2 | `ensemble_avg` | 0.1798 | [0.1781, 0.1814] | 58.8% |
| 3 | `poisson` | 0.1805 | [0.1782, 0.1827] | 57.9% |
| 4 | `dixon_coles` | 0.1806 | [0.1783, 0.1828] | 57.8% |
| 5 | `elo` | 0.1860 | [0.1839, 0.1882] | 57.0% |
| 6 | `baseline` | 0.2258 | [0.2244, 0.2271] | 48.0% |
| 7 | `constant` | 0.2275 | [0.2262, 0.2288] | 48.0% |

## Notes

- Selection uses walk-forward scores on the **intersection** of matches scored by all models.
- Accuracy is descriptive only; RPS is the primary metric.
- If nothing beats the market, live betting should be framed as price-taker on soft markets only.
