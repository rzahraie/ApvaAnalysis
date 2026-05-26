# APVA Full Pipeline Run Handoff

## Scope

This run rebuilds and validates frozen APVA candidates from raw NT8 state logs.
No candidate definitions or validation thresholds were changed.

## Raw Discovery

- Raw root: `data/Validation`
- Selected raw build input files after validation/fallback handling: `10`
- ES/NQ regimes processed: `062021, 092020, 092022, 092024, 122022`

| Regime | ES File | NQ File | Pair Status | Built | Readiness | Fixed Validation Included |
| --- | --- | --- | --- | --- | --- | --- |
| 062021 | `data/Validation/ES/xApvaV01StateLog_ES062021.csv` | `data/Validation/NQ/xApvaV01StateLog_NQ062021.csv` | PAIRED | True | READY | True |
| 092020 | `data/Validation/ES/xApvaV01StateLog_ES092020.csv` | `data/Validation/NQ/xApvaV01StateLog_NQ092020.csv` | PAIRED | True | READY | True |
| 092022 | `data/Validation/ES/xApvaV01StateLog_ES092022.csv` | `data/Validation/NQ/xApvaV01StateLog_NQ092022.csv` | PAIRED | True | READY | True |
| 092024 | `data/Validation/ES/xApvaV01StateLog_ES092024_CLEAN.csv` | `data/Validation/NQ/xApvaV01StateLog_NQ092024.csv` | PAIRED | True | READY | True |
| 122022 | `data/Validation/ES/xApvaV01StateLog_ES122022.csv` | `data/Validation/NQ/xApvaV01StateLog_NQ122022.csv` | PAIRED | True | READY | True |

## Excluded Or Skipped Inputs

- `data/Validation/ES/xApvaV01StateLog_ES.csv`: state-log filename does not exactly match <instrument><six-digit-regime>.csv.
- `data/Validation/ES/xApvaV01StateLog_ES092024.csv`: exact dated file failed embedded repeated-header validation.
- `data/Validation/NQ/xApvaV01StateLog_NQ.csv`: state-log filename does not exactly match <instrument><six-digit-regime>.csv.

## Readiness Candidate Coverage By Built Regime

These coverage counts describe candidate rows found in each built dataset.
Fixed validation below excludes any row without an evaluable policy outcome.

| Regime | Candidate | Count | ES | NQ |
| --- | --- | ---: | ---: | ---: |
| 062021 | `Base_ES_NQ` | 388 | 216 | 172 |
| 062021 | `CCRRR` | 6 | 2 | 4 |
| 062021 | `PriorSlope_DominantPressureValue_Q3` | 66 | 37 | 29 |
| 062021 | `RRCCC` | 14 | 9 | 5 |
| 092020 | `Base_ES_NQ` | 350 | 190 | 160 |
| 092020 | `CCRRR` | 3 | 3 | 0 |
| 092020 | `PriorSlope_DominantPressureValue_Q3` | 61 | 33 | 28 |
| 092020 | `RRCCC` | 12 | 5 | 7 |
| 092022 | `Base_ES_NQ` | 368 | 203 | 165 |
| 092022 | `CCRRR` | 4 | 2 | 2 |
| 092022 | `PriorSlope_DominantPressureValue_Q3` | 70 | 32 | 38 |
| 092022 | `RRCCC` | 20 | 11 | 9 |
| 092024 | `Base_ES_NQ` | 407 | 211 | 196 |
| 092024 | `CCRRR` | 5 | 3 | 2 |
| 092024 | `PriorSlope_DominantPressureValue_Q3` | 82 | 40 | 42 |
| 092024 | `RRCCC` | 9 | 7 | 2 |
| 122022 | `Base_ES_NQ` | 364 | 199 | 165 |
| 122022 | `CCRRR` | 4 | 2 | 2 |
| 122022 | `PriorSlope_DominantPressureValue_Q3` | 69 | 33 | 36 |
| 122022 | `RRCCC` | 16 | 3 | 13 |

## Pooled Fixed Validation

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

| Rank | Candidate | Score | Concentration Penalty |
| ---: | --- | ---: | ---: |
| 1 | `PriorSlope_DominantPressureValue_Q3` | 2.539091 | 0.383986 |
| 2 | `CCRRR` | 1.487722 | 0.678945 |
| 3 | `RRCCC` | 0.268562 | 0.769899 |

## Consistency Checks

- `RRCCC` Reference pooled count: extended validation `329`, cross-era ranking `329`, handoff `329`: `PASS`.
- `CCRRR` Reference pooled count: extended validation `153`, cross-era ranking `153`, handoff `153`: `PASS`.
- `PriorSlope_DominantPressureValue_Q3` Reference pooled count: extended validation `786`, cross-era ranking `786`, handoff `786`: `PASS`.
- Fixed-validation evaluated-entry checks: `PASS` (no missing policy outcomes and pooled ES/NQ counts reconcile to evaluated counts).

## Next Data Collection Target

Collect a new paired ES/NQ `xApvaV01StateLog` regime outside the periods
already listed above, prioritizing enough ES observations to increase
fixed-rule `CCRRR` support without changing its definition. Place the two
raw files under `data/Validation/ES/` and `data/Validation/NQ/`, then run:

```powershell
python scripts/apva_run_full_pipeline.py --raw-root data/validation
```
