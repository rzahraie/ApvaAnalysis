# APVA / SpyderTrader Cross-Era Validation Report Handoff

## Scope

This report is descriptive diagnostics over frozen fixed-rule candidates only.
No candidate rules, thresholds, or topology definitions were changed.

## Included Datasets

- `tables/apva_forward_signed_return_dataset_es_nq_032024_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_062016_generated.csv`
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

- `RRCCC` Reference pooled count: `454`
- `CCRRR` Reference pooled count: `199`
- `PriorSlope_DominantPressureValue_Q3` Reference pooled count: `1324`

These counts are written from `extended_scorecard.csv` and verified against
`robustness_rankings.csv` by the full-pipeline consistency check.

## Pooled Validation Status

| Mode | Candidate | Count | ES | NQ | Mean | PF | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Reference | `CCRRR` | 199 | 52 | 147 | 0.158314 | 1.275095 | fail |
| Reference | `PriorSlope_DominantPressureValue_Q3` | 1324 | 495 | 829 | 0.149432 | 1.266001 | fail |
| Reference | `RRCCC` | 454 | 128 | 326 | 0.118730 | 1.207694 | fail |
| Spacing_10 | `CCRRR` | 178 | 51 | 127 | 0.137115 | 1.238610 | fail |
| Spacing_10 | `PriorSlope_DominantPressureValue_Q3` | 1080 | 414 | 666 | 0.160168 | 1.285525 | fail |
| Spacing_10 | `RRCCC` | 397 | 121 | 276 | 0.099357 | 1.171452 | fail |
| Spacing_20 | `CCRRR` | 160 | 50 | 110 | 0.131019 | 1.220067 | fail |
| Spacing_20 | `PriorSlope_DominantPressureValue_Q3` | 982 | 378 | 604 | 0.155509 | 1.276711 | fail |
| Spacing_20 | `RRCCC` | 358 | 115 | 243 | 0.067949 | 1.116580 | fail |

## Cross-Era Robustness Ranking

| Rank | Candidate | Score | Spacing Survival | Regime Survival | Concentration Penalty |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `PriorSlope_DominantPressureValue_Q3` | 0.935099 | 0.000000 | 0.600000 | 0.344901 |
| 2 | `CCRRR` | 0.530857 | 0.000000 | 0.500000 | 0.577838 |
| 3 | `RRCCC` | 0.382366 | 0.000000 | 0.400000 | 0.577634 |

## Interpretation Boundary

Passing rows are provisional fixed-rule validation results. The principal
remaining need is new paired ES/NQ observations outside the represented
eras, with particular value in expanding sparse ES support for `CCRRR`.
