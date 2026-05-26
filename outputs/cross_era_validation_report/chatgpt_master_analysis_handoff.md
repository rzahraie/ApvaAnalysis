# APVA / SpyderTrader Cross-Era Validation Report Handoff

## Scope

This report is descriptive diagnostics over frozen fixed-rule candidates only.
No candidate rules, thresholds, or topology definitions were changed.

## Included Datasets

- `tables/apva_forward_signed_return_dataset_es_nq_062021_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_092020_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_092022_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_092024_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_122022_generated.csv`
- `tables/apva_forward_signed_return_dataset_generated.csv`
- `tables/apva_forward_signed_return_dataset_v1.csv`

## Reference Count Check

- `RRCCC` Reference pooled count: `329`
- `CCRRR` Reference pooled count: `153`
- `PriorSlope_DominantPressureValue_Q3` Reference pooled count: `786`

These counts are written from `extended_scorecard.csv` and verified against
`robustness_rankings.csv` by the full-pipeline consistency check.

## Pooled Validation Status

| Mode | Candidate | Count | ES | NQ | Mean | PF | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Reference | `CCRRR` | 153 | 32 | 121 | 0.259343 | 1.483105 | pass |
| Reference | `PriorSlope_DominantPressureValue_Q3` | 786 | 270 | 516 | 0.283492 | 1.546585 | pass |
| Reference | `RRCCC` | 329 | 82 | 247 | 0.169848 | 1.309488 | fail |
| Spacing_10 | `CCRRR` | 132 | 31 | 101 | 0.246829 | 1.466127 | pass |
| Spacing_10 | `PriorSlope_DominantPressureValue_Q3` | 603 | 215 | 388 | 0.301114 | 1.587281 | pass |
| Spacing_10 | `RRCCC` | 272 | 75 | 197 | 0.152284 | 1.274159 | fail |
| Spacing_20 | `CCRRR` | 115 | 30 | 85 | 0.256307 | 1.468832 | fail |
| Spacing_20 | `PriorSlope_DominantPressureValue_Q3` | 541 | 194 | 347 | 0.289879 | 1.572659 | pass |
| Spacing_20 | `RRCCC` | 235 | 70 | 165 | 0.094903 | 1.166829 | fail |

## Cross-Era Robustness Ranking

| Rank | Candidate | Score | Spacing Survival | Regime Survival | Concentration Penalty |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `PriorSlope_DominantPressureValue_Q3` | 2.539091 | 1.000000 | 1.000000 | 0.383986 |
| 2 | `CCRRR` | 1.487722 | 0.666667 | 0.833333 | 0.678945 |
| 3 | `RRCCC` | 0.268562 | 0.000000 | 0.500000 | 0.769899 |

## Interpretation Boundary

Passing rows are provisional fixed-rule validation results. The principal
remaining need is new paired ES/NQ observations outside the represented
eras, with particular value in expanding sparse ES support for `CCRRR`.
