# APVA / SpyderTrader Cross-Era Validation Report Handoff

## Scope

This report is descriptive diagnostics over frozen fixed-rule candidates only.
No candidate rules, thresholds, or topology definitions were changed.

## Included Datasets

- `tables/apva_forward_signed_return_dataset_es_nq_062021_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_062023_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_092020_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_092022_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_092024_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_122022_generated.csv`
- `tables/apva_forward_signed_return_dataset_generated.csv`
- `tables/apva_forward_signed_return_dataset_v1.csv`

## Reference Count Check

- `RRCCC` Reference pooled count: `333`
- `CCRRR` Reference pooled count: `156`
- `PriorSlope_DominantPressureValue_Q3` Reference pooled count: `815`

These counts are written from `extended_scorecard.csv` and verified against
`robustness_rankings.csv` by the full-pipeline consistency check.

## Pooled Validation Status

| Mode | Candidate | Count | ES | NQ | Mean | PF | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Reference | `CCRRR` | 156 | 32 | 124 | 0.239110 | 1.438511 | pass |
| Reference | `PriorSlope_DominantPressureValue_Q3` | 815 | 280 | 535 | 0.286783 | 1.555612 | pass |
| Reference | `RRCCC` | 333 | 81 | 252 | 0.185956 | 1.344206 | fail |
| Spacing_10 | `CCRRR` | 135 | 31 | 104 | 0.223728 | 1.414723 | pass |
| Spacing_10 | `PriorSlope_DominantPressureValue_Q3` | 632 | 226 | 406 | 0.318444 | 1.627908 | pass |
| Spacing_10 | `RRCCC` | 276 | 74 | 202 | 0.171973 | 1.315524 | fail |
| Spacing_20 | `CCRRR` | 118 | 30 | 88 | 0.229636 | 1.411817 | fail |
| Spacing_20 | `PriorSlope_DominantPressureValue_Q3` | 570 | 206 | 364 | 0.316494 | 1.632377 | pass |
| Spacing_20 | `RRCCC` | 239 | 69 | 170 | 0.118599 | 1.213078 | fail |

## Cross-Era Robustness Ranking

| Rank | Candidate | Score | Spacing Survival | Regime Survival | Concentration Penalty |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `PriorSlope_DominantPressureValue_Q3` | 2.567260 | 1.000000 | 1.000000 | 0.366073 |
| 2 | `CCRRR` | 1.335173 | 0.666667 | 0.714286 | 0.661164 |
| 3 | `RRCCC` | 0.451991 | 0.000000 | 0.571429 | 0.719437 |

## Interpretation Boundary

Passing rows are provisional fixed-rule validation results. The principal
remaining need is new paired ES/NQ observations outside the represented
eras, with particular value in expanding sparse ES support for `CCRRR`.
