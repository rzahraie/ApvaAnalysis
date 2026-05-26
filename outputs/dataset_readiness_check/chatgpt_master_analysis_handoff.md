# APVA / SpyderTrader Dataset Readiness Handoff

## Prompt For ChatGPT Master Analysis

Interpret this task as validation-infrastructure work only. It implements a
readiness check for new row-level APVA datasets using frozen candidate rules.
It does not search for candidates, tune thresholds, or add independent
validation evidence.

## Repository And Artifacts

- Repository: `https://github.com/rzahraie/ApvaAnalysis`
- Readiness script: `scripts/apva_dataset_readiness_check.py`
- New-data guide: `docs/APVA_NEW_DATASET_REQUIREMENTS.md`
- Reference dataset: `tables/apva_forward_signed_return_dataset_v1.csv`
- Output directory: `outputs/dataset_readiness_check/`

Generated outputs:

- `dataset_readiness_summary.csv`
- `dataset_readiness_candidate_counts.csv`
- `dataset_readiness_overlap.csv`
- `dataset_readiness_scorecard.csv`

## Command Run

```powershell
python scripts/apva_dataset_readiness_check.py --inputs "tables/*.csv" "outputs/**/*.csv"
```

The tool inventories output CSVs for traceability but never treats files under
`outputs/` as independent validation datasets.

## Frozen Rules Checked

Universe:

- `ES`
- `NQ`

Base entry:

- `DominantPressure == CompressionPressure`
- `RollingDirectionalPresence == 0`
- `RollingEntropy` between `0.88` and `1.30`, inclusive
- `HorizonBars == 5`

Frozen non-base candidates:

- `RRCCC`: `RotationalPressure > RotationalPressure > CompressionPressure > CompressionPressure > CompressionPressure`
- `CCRRR`: `CompressionPressure > CompressionPressure > RotationalPressure > RotationalPressure > RotationalPressure`
- `PriorSlope_DominantPressureValue_Q3`: frozen interval
  `0.004718046888521732 <= PriorSlope_DominantPressureValue <= 0.05505235072964343`

The slope candidate is checked only when `DominantPressureValue` is present.

## Leakage Constraints

The readiness script includes a runtime guard preventing these outcome fields
from being used as predictors or gates:

- `SignedReturn`
- `RawReturn`
- `NormalizedReturn`
- `SignedNormalizedReturn`
- `FutureClose`
- `DirectionalMFE`
- `DirectionalMAE`
- `DirectionalNormalizedMFE`
- `DirectionalNormalizedMAE`
- `DirectionalHit`

`SignedNormalizedReturn` and `DirectionalNormalizedMAE` are required for later
fixed-rule outcome evaluation, not used for readiness candidate selection.

## Readiness Result

| Metric | Result |
| --- | ---: |
| CSV files checked | 292 |
| Ready independent datasets | 0 |
| Schema-incompatible files | 257 |
| Non-independent or partial-overlap files | 6 |
| Derived output files excluded | 171 |
| Unreadable output artifacts inventoried | 1 |

No new file is currently available for independent fixed-rule validation.

## Compatible Table Datasets

Four table files satisfy the required row-level schema and contain frozen
candidate rows. Each has complete overlap with the reference and therefore is
`NOT_READY` as a new validation dataset.

| Dataset | Rows | Base | CCRRR | RRCCC | PriorSlope Q3 | Overlap With Reference | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `tables/apva_forward_signed_return_dataset_v1.csv` | 55,806 | 1,672 | 129 | 256 | 397 | 1.000000 | Non-independent reference |
| `tables/apva_nonoverlap_forward_signed_return_dataset_v1.csv` | 55,806 | 1,672 | 129 | 256 | 397 | 1.000000 | Non-independent copy |
| `tables/apva_walk_forward_signed_return_dataset_v1.csv` | 55,806 | 1,672 | 129 | 256 | 397 | 1.000000 | Non-independent copy |
| `tables/apva_file_level_walk_forward_dataset_v1.csv` | 55,806 | 1,672 | 129 | 256 | 397 | 1.000000 | Non-independent copy |

All four have zero duplicate full observation keys when the expected
`HorizonBars` dimension is included. Multiple horizons for one source bar are
not treated as erroneous duplicates.

Two additional table artifacts share more than 95% of comparable row keys
with the reference but do not satisfy the required validation schema:

- `tables/apva_forward_return_dataset_v1.csv`
- `tables/apva_cross_instrument_pressure_timeline_v2.csv`

They are not validation-ready and add no independent test sample.

## Derived And Incompatible Files

Files under `outputs/` are included in the inventory for visibility and are
explicitly excluded as derived analysis artifacts. Some selected-entry output
files carry the required columns, but they are not new row-level source data.

Most other CSV files under `tables/` and `outputs/` are summaries or analysis
artifacts missing one or more required row-level forward-return fields.
One empty output artifact could not be parsed and is recorded as
`FAIL_READ_ERROR`; it is not a candidate dataset.

## Readiness Decision

Files ready for validation:

- None.

Files with nonzero frozen candidate coverage in both ES and NQ:

- The four schema-compatible table datasets listed above, all of which are
  row-identical with the reference for readiness purposes.

Therefore this task does not change the prior provisional interpretation of
`CCRRR`; it provides the ingestion gate needed before another fixed-rule test
can legitimately be run.

## Exact Next Commands For A New Dataset

After exporting a genuinely new row-level ES/NQ dataset, for example
`tables/apva_new_es_nq_forward_signed_return_dataset.csv`, first run:

```powershell
python scripts/apva_dataset_readiness_check.py --inputs "tables/apva_new_es_nq_forward_signed_return_dataset.csv"
```

If it reports `READY`, keep the new file in `tables/` and run the frozen
extended validation:

```powershell
python scripts/apva_fixed_candidate_extended_validation.py
```

The fixed validator inventories compatible table datasets and will include an
independent new row-level file without changing candidate definitions or
thresholds.
