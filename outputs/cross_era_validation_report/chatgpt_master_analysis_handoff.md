# APVA / SpyderTrader Cross-Era Validation Report Handoff

## Scope

This report is descriptive diagnostics over frozen fixed-rule candidates only.
No candidate rules, thresholds, or topology definitions were changed.

## Included Datasets

- `tables/apva_forward_signed_return_dataset_es_nq_062019_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_062021_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_062023_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_092020_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_092022_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_092024_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_122017_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_122022_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_122023_generated.csv`
- `tables/apva_forward_signed_return_dataset_generated.csv`
- `tables/apva_forward_signed_return_dataset_v1.csv`

## Reference Count Check

- `RRCCC` Reference pooled count: `419`
- `CCRRR` Reference pooled count: `189`
- `PriorSlope_DominantPressureValue_Q3` Reference pooled count: `1157`

These counts are written from `extended_scorecard.csv` and verified against
`robustness_rankings.csv` by the full-pipeline consistency check.

## Pooled Validation Status

| Mode | Candidate | Count | ES | NQ | Mean | PF | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Reference | `CCRRR` | 189 | 45 | 144 | 0.198542 | 1.354244 | fail |
| Reference | `PriorSlope_DominantPressureValue_Q3` | 1157 | 407 | 750 | 0.169018 | 1.307396 | fail |
| Reference | `RRCCC` | 419 | 111 | 308 | 0.118575 | 1.209552 | fail |
| Spacing_10 | `CCRRR` | 168 | 44 | 124 | 0.181110 | 1.324749 | fail |
| Spacing_10 | `PriorSlope_DominantPressureValue_Q3` | 938 | 339 | 599 | 0.175393 | 1.320296 | pass |
| Spacing_10 | `RRCCC` | 362 | 104 | 258 | 0.097305 | 1.169657 | fail |
| Spacing_20 | `CCRRR` | 150 | 43 | 107 | 0.179887 | 1.311353 | fail |
| Spacing_20 | `PriorSlope_DominantPressureValue_Q3` | 853 | 310 | 543 | 0.171161 | 1.313799 | fail |
| Spacing_20 | `RRCCC` | 324 | 99 | 225 | 0.055880 | 1.096192 | fail |

## Cross-Era Robustness Ranking

| Rank | Candidate | Score | Spacing Survival | Regime Survival | Concentration Penalty |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `PriorSlope_DominantPressureValue_Q3` | 1.407035 | 0.333333 | 0.666667 | 0.354870 |
| 2 | `CCRRR` | 0.632851 | 0.000000 | 0.666667 | 0.560132 |
| 3 | `RRCCC` | 0.342838 | 0.000000 | 0.444444 | 0.625416 |

## Interpretation Boundary

Passing rows are provisional fixed-rule validation results. The principal
remaining need is new paired ES/NQ observations outside the represented
eras, with particular value in expanding sparse ES support for `CCRRR`.
