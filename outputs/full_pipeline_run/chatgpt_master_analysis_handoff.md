# APVA Full Pipeline Run Handoff

## Scope

This run rebuilds and validates frozen APVA candidates from raw NT8 state logs.
No candidate definitions or validation thresholds were changed.

## Raw Discovery

- Raw root: `data/Validation`
- Selected raw build input files after validation/fallback handling: `12`
- ES/NQ regimes processed: `062021, 062023, 092020, 092022, 092024, 122022`

| Regime | ES File | NQ File | Pair Status | Built | Readiness | Fixed Validation Included |
| --- | --- | --- | --- | --- | --- | --- |
| 062021 | `data/Validation/ES/xApvaV01StateLog_ES062021.csv` | `data/Validation/NQ/xApvaV01StateLog_NQ062021.csv` | PAIRED | True | READY | True |
| 062023 | `data/Validation/ES/xApvaV01StateLog_ES062023.csv` | `data/Validation/NQ/xApvaV01StateLog_NQ062023.csv` | PAIRED | True | READY | True |
| 092020 | `data/Validation/ES/xApvaV01StateLog_ES092020.csv` | `data/Validation/NQ/xApvaV01StateLog_NQ092020.csv` | PAIRED | True | READY | True |
| 092022 | `data/Validation/ES/xApvaV01StateLog_ES092022.csv` | `data/Validation/NQ/xApvaV01StateLog_NQ092022.csv` | PAIRED | True | READY | True |
| 092024 | `data/Validation/ES/xApvaV01StateLog_ES092024.csv` | `data/Validation/NQ/xApvaV01StateLog_NQ092024.csv` | PAIRED | True | NOT_READY | True |
| 122022 | `data/Validation/ES/xApvaV01StateLog_ES122022.csv` | `data/Validation/NQ/xApvaV01StateLog_NQ122022.csv` | PAIRED | True | READY | True |

## Excluded Or Skipped Inputs

- `data/Validation/ES/xApvaV01StateLog_ES.csv`: state-log filename does not exactly match <instrument><six-digit-regime>.csv.
- `data/Validation/ES/xApvaV01StateLog_ES092024_CLEAN.csv`: state-log filename does not exactly match <instrument><six-digit-regime>.csv.
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
| 122022 | `Base_ES_NQ` | 364 | 199 | 165 |
| 122022 | `CCRRR` | 4 | 2 | 2 |
| 122022 | `PriorSlope_DominantPressureValue_Q3` | 69 | 33 | 36 |
| 122022 | `RRCCC` | 16 | 3 | 13 |

## Pooled Fixed Validation

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

| Rank | Candidate | Score | Concentration Penalty |
| ---: | --- | ---: | ---: |
| 1 | `PriorSlope_DominantPressureValue_Q3` | 2.567260 | 0.366073 |
| 2 | `CCRRR` | 1.335173 | 0.661164 |
| 3 | `RRCCC` | 0.451991 | 0.719437 |

## Consistency Checks

- `RRCCC` Reference pooled count: extended validation `333`, cross-era ranking `333`, handoff `333`: `PASS`.
- `CCRRR` Reference pooled count: extended validation `156`, cross-era ranking `156`, handoff `156`: `PASS`.
- `PriorSlope_DominantPressureValue_Q3` Reference pooled count: extended validation `815`, cross-era ranking `815`, handoff `815`: `PASS`.
- Fixed-validation evaluated-entry checks: `PASS` (no missing policy outcomes and pooled ES/NQ counts reconcile to evaluated counts).

## Next Data Collection Target

Collect a new paired ES/NQ `xApvaV01StateLog` regime outside the periods
already listed above, prioritizing enough ES observations to increase
fixed-rule `CCRRR` support without changing its definition. Place the two
raw files under `data/Validation/ES/` and `data/Validation/NQ/`, then run:

```powershell
python scripts/apva_run_full_pipeline.py --raw-root data/validation
```
