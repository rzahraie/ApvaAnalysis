# APVA / SpyderTrader Cross-Era Validation Report Handoff

## Scope

This report is descriptive diagnostics over frozen fixed-rule candidates only.
No candidate rules, thresholds, or topology definitions were changed.

## Included Datasets

- `tables/apva_forward_signed_return_dataset_es_nq_032024_generated.csv`
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

- `RRCCC` Reference pooled count: `440`
- `CCRRR` Reference pooled count: `193`
- `PriorSlope_DominantPressureValue_Q3` Reference pooled count: `1240`

These counts are written from `extended_scorecard.csv` and verified against
`robustness_rankings.csv` by the full-pipeline consistency check.

## Pooled Validation Status

| Mode | Candidate | Count | ES | NQ | Mean | PF | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Reference | `CCRRR` | 193 | 48 | 145 | 0.173831 | 1.299739 | fail |
| Reference | `PriorSlope_DominantPressureValue_Q3` | 1240 | 451 | 789 | 0.161440 | 1.289863 | fail |
| Reference | `RRCCC` | 440 | 122 | 318 | 0.127682 | 1.226439 | fail |
| Spacing_10 | `CCRRR` | 172 | 47 | 125 | 0.153787 | 1.265329 | fail |
| Spacing_10 | `PriorSlope_DominantPressureValue_Q3` | 1009 | 375 | 634 | 0.169571 | 1.305049 | pass |
| Spacing_10 | `RRCCC` | 383 | 115 | 268 | 0.108933 | 1.190829 | fail |
| Spacing_20 | `CCRRR` | 154 | 46 | 108 | 0.149401 | 1.248294 | fail |
| Spacing_20 | `PriorSlope_DominantPressureValue_Q3` | 920 | 344 | 576 | 0.162748 | 1.292626 | fail |
| Spacing_20 | `RRCCC` | 344 | 109 | 235 | 0.077332 | 1.134879 | fail |

## Cross-Era Robustness Ranking

| Rank | Candidate | Score | Spacing Survival | Regime Survival | Concentration Penalty |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `PriorSlope_DominantPressureValue_Q3` | 1.390949 | 0.333333 | 0.666667 | 0.348181 |
| 2 | `CCRRR` | 0.525624 | 0.000000 | 0.555556 | 0.601360 |
| 3 | `RRCCC` | 0.420346 | 0.000000 | 0.444444 | 0.589316 |

## Interpretation Boundary

Passing rows are provisional fixed-rule validation results. The principal
remaining need is new paired ES/NQ observations outside the represented
eras, with particular value in expanding sparse ES support for `CCRRR`.
