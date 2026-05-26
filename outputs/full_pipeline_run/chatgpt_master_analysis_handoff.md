# APVA Full Pipeline Run Handoff

## Scope

This run rebuilds and validates frozen APVA candidates from raw NT8 state logs.
No candidate definitions or validation thresholds were changed.

## Raw Discovery

- Raw root: `data/Validation`
- Selected raw build input files after validation/fallback handling: `18`
- ES/NQ regimes processed: `062019, 062021, 062023, 092020, 092022, 092024, 122017, 122022, 122023`

| Regime | ES File | NQ File | Pair Status | Built | Readiness | Fixed Validation Included |
| --- | --- | --- | --- | --- | --- | --- |
| 062019 | `data/Validation/Temp/xApvaV01StateLog_ES062019.csv` | `data/Validation/Temp/xApvaV01StateLog_NQ062019.csv` | PAIRED | True | READY | True |
| 062021 | `data/Validation/ES/xApvaV01StateLog_ES062021.csv` | `data/Validation/NQ/xApvaV01StateLog_NQ062021.csv` | PAIRED | True | READY | True |
| 062023 | `data/Validation/ES/xApvaV01StateLog_ES062023.csv` | `data/Validation/NQ/xApvaV01StateLog_NQ062023.csv` | PAIRED | True | READY | True |
| 092020 | `data/Validation/ES/xApvaV01StateLog_ES092020.csv` | `data/Validation/NQ/xApvaV01StateLog_NQ092020.csv` | PAIRED | True | READY | True |
| 092022 | `data/Validation/ES/xApvaV01StateLog_ES092022.csv` | `data/Validation/NQ/xApvaV01StateLog_NQ092022.csv` | PAIRED | True | READY | True |
| 092024 | `data/Validation/ES/xApvaV01StateLog_ES092024.csv` | `data/Validation/NQ/xApvaV01StateLog_NQ092024.csv` | PAIRED | True | NOT_READY | True |
| 122017 | `data/Validation/ES/xApvaV01StateLog_ES122017.csv` | `data/Validation/NQ/xApvaV01StateLog_NQ122017.csv` | PAIRED | True | READY | True |
| 122022 | `data/Validation/ES/xApvaV01StateLog_ES122022.csv` | `data/Validation/NQ/xApvaV01StateLog_NQ122022.csv` | PAIRED | True | READY | True |
| 122023 | `data/Validation/Temp/xApvaV01StateLog_ES122023.csv` | `data/Validation/Temp/xApvaV01StateLog_NQ122023.csv` | PAIRED | True | READY | True |

## Excluded Or Skipped Inputs

- `data/Validation/ES/xApvaV01StateLog_ES.csv`: state-log filename does not exactly match <instrument><six-digit-regime>.csv.
- `data/Validation/ES/xApvaV01StateLog_ES092024_CLEAN.csv`: state-log filename does not exactly match <instrument><six-digit-regime>.csv.
- `data/Validation/NQ/xApvaV01StateLog_NQ.csv`: state-log filename does not exactly match <instrument><six-digit-regime>.csv.

## Readiness Candidate Coverage By Built Regime

These coverage counts describe candidate rows found in each built dataset.
Fixed validation below excludes any row without an evaluable policy outcome.

| Regime | Candidate | Count | ES | NQ |
| --- | --- | ---: | ---: | ---: |
| 062019 | `Base_ES_NQ` | 351 | 172 | 179 |
| 062019 | `CCRRR` | 7 | 3 | 4 |
| 062019 | `PriorSlope_DominantPressureValue_Q3` | 81 | 45 | 36 |
| 062019 | `RRCCC` | 17 | 9 | 8 |
| 062021 | `Base_ES_NQ` | 388 | 216 | 172 |
| 062021 | `CCRRR` | 6 | 2 | 4 |
| 062021 | `PriorSlope_DominantPressureValue_Q3` | 66 | 37 | 29 |
| 062021 | `RRCCC` | 14 | 9 | 5 |
| 062023 | `Base_ES_NQ` | 213 | 111 | 102 |
| 062023 | `CCRRR` | 3 | 0 | 3 |
| 062023 | `PriorSlope_DominantPressureValue_Q3` | 36 | 17 | 19 |
| 062023 | `RRCCC` | 9 | 4 | 5 |
| 092020 | `Base_ES_NQ` | 350 | 190 | 160 |
| 092020 | `CCRRR` | 3 | 3 | 0 |
| 092020 | `PriorSlope_DominantPressureValue_Q3` | 61 | 33 | 28 |
| 092020 | `RRCCC` | 12 | 5 | 7 |
| 092022 | `Base_ES_NQ` | 368 | 203 | 165 |
| 092022 | `CCRRR` | 4 | 2 | 2 |
| 092022 | `PriorSlope_DominantPressureValue_Q3` | 70 | 32 | 38 |
| 092022 | `RRCCC` | 20 | 11 | 9 |
| 092024 | `Base_ES_NQ` | 461 | 265 | 196 |
| 092024 | `CCRRR` | 5 | 3 | 2 |
| 092024 | `PriorSlope_DominantPressureValue_Q3` | 75 | 33 | 42 |
| 092024 | `RRCCC` | 4 | 2 | 2 |
| 122017 | `Base_ES_NQ` | 935 | 250 | 685 |
| 122017 | `CCRRR` | 19 | 6 | 13 |
| 122017 | `PriorSlope_DominantPressureValue_Q3` | 180 | 43 | 137 |
| 122017 | `RRCCC` | 47 | 11 | 36 |
| 122022 | `Base_ES_NQ` | 364 | 199 | 165 |
| 122022 | `CCRRR` | 4 | 2 | 2 |
| 122022 | `PriorSlope_DominantPressureValue_Q3` | 69 | 33 | 36 |
| 122022 | `RRCCC` | 16 | 3 | 13 |
| 122023 | `Base_ES_NQ` | 421 | 218 | 203 |
| 122023 | `CCRRR` | 7 | 4 | 3 |
| 122023 | `PriorSlope_DominantPressureValue_Q3` | 81 | 39 | 42 |
| 122023 | `RRCCC` | 22 | 10 | 12 |

## Pooled Fixed Validation

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

| Rank | Candidate | Score | Concentration Penalty |
| ---: | --- | ---: | ---: |
| 1 | `PriorSlope_DominantPressureValue_Q3` | 1.407035 | 0.354870 |
| 2 | `CCRRR` | 0.632851 | 0.560132 |
| 3 | `RRCCC` | 0.342838 | 0.625416 |

## Consistency Checks

- `RRCCC` Reference pooled count: extended validation `419`, cross-era ranking `419`, handoff `419`: `PASS`.
- `CCRRR` Reference pooled count: extended validation `189`, cross-era ranking `189`, handoff `189`: `PASS`.
- `PriorSlope_DominantPressureValue_Q3` Reference pooled count: extended validation `1157`, cross-era ranking `1157`, handoff `1157`: `PASS`.
- Fixed-validation evaluated-entry checks: `PASS` (no missing policy outcomes and pooled ES/NQ counts reconcile to evaluated counts).

## Next Data Collection Target

Collect a new paired ES/NQ `xApvaV01StateLog` regime outside the periods
already listed above, prioritizing enough ES observations to increase
fixed-rule `CCRRR` support without changing its definition. Place the two
raw files under `data/Validation/ES/` and `data/Validation/NQ/`, then run:

```powershell
python scripts/apva_run_full_pipeline.py --raw-root data/validation
```
