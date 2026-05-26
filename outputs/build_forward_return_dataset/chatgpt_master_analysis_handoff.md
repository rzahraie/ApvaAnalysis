# APVA / SpyderTrader Canonical Dataset Build Handoff

## Prompt For ChatGPT Master Analysis

Interpret this work as source-pipeline reconstruction and validation
readiness only. It does not discover candidates or tune any rule or
threshold.

## Implemented Artifacts

- Script: `scripts/apva_build_forward_return_dataset.py`
- Documentation: `docs/APVA_NEW_DATASET_REQUIREMENTS.md`
- Generated canonical-format dataset:
  `tables/apva_forward_signed_return_dataset_generated.csv`
- Reports: `outputs/build_forward_return_dataset/`

Report files:

- `build_dataset_summary.csv`
- `build_dataset_column_report.csv`
- `build_dataset_candidate_counts.csv`
- `build_dataset_compare_report.csv`
- `build_dataset_source_inventory.csv`
- `readiness_generated/` readiness reports
- `canonical_reconstruction/` historical reconstruction comparison reports

## Pipeline Recovered From Notebook

The transformation logic was recovered from
`apva_phase01_exploration.ipynb`:

- Cell 35 creates `apva_cross_instrument_pressure_timeline_v2.csv`.
- Cell 40 creates `apva_forward_signed_return_dataset_v1.csv`.

Recovered raw-to-pressure process:

1. Read each raw `xApvaV01*StateLog*.csv` state export.
2. Build four-state paths from `MacroState`.
3. Resolve each path through `tables/apva_online_archetype_lookup.csv`,
   using the notebook's nearest-vector soft-resolution fallback for unknown
   quadruplets.
4. Compute rolling window-10 `RollingEntropy`,
   `RollingDirectionalPresence`, and pressure values.
5. Select the maximum pressure as `DominantPressure`; retain its value as
   `DominantPressureValue`.

Recovered pressure-to-outcome process:

1. Merge pressure rows with OHLC and `ActiveDirection`.
2. Convert direction to sign (`Up` = `1`, `Down` = `-1`).
3. Compute `ATR14` on the pressure-timeline rows within `Instrument` / `File`.
4. Compute forward outcomes at requested horizons.
5. Normalize return, MFE, and MAE by the entry row's `ATR14`.

The raw columns `DominanceScore`, `DegradationScore`, `BalanceScore`,
`TransitionScore`, and `AmbiguityScore` are not used to calculate
`DominantPressure` in the recovered notebook path.

## Historical Reproduction Check

Historical source files tested:

- `data/Validation/ES/xApvaV01StateLog_ES.csv`
- `data/Validation/NQ/xApvaV01StateLog_NQ.csv`
- `data/Validation/6E/xApvaV01StateLog_6E.csv`

Command:

```powershell
python scripts/apva_build_forward_return_dataset.py `
  --inputs "data/Validation/ES/xApvaV01StateLog_ES.csv" "data/Validation/NQ/xApvaV01StateLog_NQ.csv" "data/Validation/6E/xApvaV01StateLog_6E.csv" `
  --out "outputs/build_forward_return_dataset/canonical_reconstruction/apva_forward_signed_return_dataset_rebuilt.csv" `
  --outdir "outputs/build_forward_return_dataset/canonical_reconstruction" `
  --horizons 5 10 20 `
  --compare-to "tables/apva_forward_signed_return_dataset_v1.csv"
```

Result:

- Output rows match canonical: `55,806`.
- Output columns and row keys match canonical.
- Signed normalized outcomes match on shared rows to numerical tolerance.
- Pressure/archetype fields do not match completely.
- Therefore reconstructed output is **not** byte/value-equivalent to the
  committed canonical validation dataset.

Diagnosis:

- The committed intermediate
  `tables/apva_cross_instrument_pressure_timeline_v2.csv` agrees with the
  committed canonical forward dataset.
- Rebuilding from the currently present historical raw logs produces
  different `MacroState` quadruplets for substantial portions of `NQ` and
  `6E`, while `ES` agrees.
- The raw historical files currently in the repository are therefore not
  fully provenance-equivalent inputs for the committed canonical pressure
  timeline.

This unresolved provenance mismatch must not be silently interpreted as an
exact regeneration of the prior canonical sample.

## New Raw Export Assessment

Raw candidate exports found:

- `data/Validation/ES/xApvaV01StateLog_ES092024.csv`
- `data/Validation/NQ/xApvaV01StateLog_NQ092024.csv`

These raw input exports were present as untracked workspace files during this
run and are not part of the pipeline-code commit. The generated NQ dataset is
therefore an available validation artifact, but its raw-source provenance
must be preserved separately or added deliberately in a future data commit.

The ES file is currently blocked:

- It contains an embedded repeated header at CSV row `4`.
- The builder rejects it explicitly rather than silently removing source
  rows.
- A clean ES re-export is required before building a combined ES/NQ
  extension.

The NQ file built successfully as a standalone new dataset:

| Metric | Result |
| --- | ---: |
| Generated rows | 3,264 |
| Instrument | NQ only |
| Horizon | 5 |
| Row-key overlap with canonical dataset | 0 |
| Base candidate rows | 196 |
| `RRCCC` rows | 2 |
| `CCRRR` rows | 2 |
| `PriorSlope_DominantPressureValue_Q3` rows | 42 |

Generated dataset:

- `tables/apva_forward_signed_return_dataset_generated.csv`

## Readiness Result

Command:

```powershell
python scripts/apva_dataset_readiness_check.py `
  --inputs "tables/apva_forward_signed_return_dataset_generated.csv" `
  --outdir "outputs/build_forward_return_dataset/readiness_generated"
```

Result:

- Readiness status: `READY`
- Ready independent dataset count: `1`
- Overlap with reference: `0`
- Both ES and NQ represented: `No`; NQ only

Interpretation:

- A genuinely independent canonical-format NQ dataset is now available for
  ingestion checks.
- It is not sufficient for a meaningful combined ES/NQ fixed-candidate
  validation because ES is absent and `CCRRR`/`RRCCC` each have only two
  qualifying rows.
- No new claim about any provisional candidate is supported by this NQ-only
  readiness result.

## Exact Next Step

Obtain a clean ES 2024 raw state export, then build a combined ES/NQ
canonical dataset:

```powershell
python scripts/apva_build_forward_return_dataset.py `
  --inputs "data/Validation/ES/xApvaV01StateLog_ES092024_CLEAN.csv" "data/Validation/NQ/xApvaV01StateLog_NQ092024.csv" `
  --out "tables/apva_forward_signed_return_dataset_es_nq_092024_generated.csv" `
  --horizons 5 `
  --compare-to "tables/apva_forward_signed_return_dataset_v1.csv"
```

Then check readiness:

```powershell
python scripts/apva_dataset_readiness_check.py `
  --inputs "tables/apva_forward_signed_return_dataset_es_nq_092024_generated.csv" `
  --outdir "outputs/dataset_readiness_es_nq_092024"
```

Run fixed-rule extended validation only after the new dataset is clean,
independent, and has adequate frozen-candidate coverage.
