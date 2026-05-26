# APVA New Dataset Requirements

Use this checklist when exporting genuinely new row-level APVA data for
fixed-rule validation. New data should add contracts or time ranges, rather
than repeat selected rows or summaries from previous analyses.

## Required Schema

Each row-level forward-return CSV must include:

| Column | Purpose |
| --- | --- |
| `Instrument` | Contract identifier; fixed validation currently uses `ES` and `NQ`. |
| `File` | Source stream or file identifier used to isolate history. |
| `BarIndex` | Chronological bar index within source history. |
| `HorizonBars` | Forward outcome horizon; current fixed rules use `5`. |
| `DominantPressure` | Ex-ante pressure classification. |
| `RollingDirectionalPresence` | Ex-ante entry-state feature. |
| `RollingEntropy` | Ex-ante entry-state feature. |
| `SignedNormalizedReturn` | Outcome value only. |
| `DirectionalNormalizedMAE` | Outcome simulation input for the fixed stop only. |

Useful optional columns:

- `Time`: recommended for date-range and ISO-week validation reporting.
- `DominantPressureValue`: required to evaluate the frozen
  `PriorSlope_DominantPressureValue_Q3` candidate.

Rows must be chronological within each `Instrument` / `File` history so prior
pressure paths can be constructed safely.

## Fixed Entry And Candidate Rules

Base entry state:

- `DominantPressure == CompressionPressure`
- `RollingDirectionalPresence == 0`
- `RollingEntropy` in `[0.88, 1.30]`
- `HorizonBars == 5`

Frozen candidates currently validated:

- `RRCCC`: `RotationalPressure > RotationalPressure > CompressionPressure > CompressionPressure > CompressionPressure`
- `CCRRR`: `CompressionPressure > CompressionPressure > RotationalPressure > RotationalPressure > RotationalPressure`
- `PriorSlope_DominantPressureValue_Q3`: frozen interval
  `[0.004718046888521732, 0.05505235072964343]`

Do not tune these rules or recompute the slope interval during validation.

## Outcome Fields And Leakage

Safe outcome use:

- `SignedNormalizedReturn` may be used to evaluate the result after entries
  are identified.
- `DirectionalNormalizedMAE` may be used only for the fixed policy simulation:
  if it is at most `-2.0`, policy outcome is `-2.0`.

Do not use these fields as predictors or gates:

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

## Export Guidance

Export a row-level dataset containing the full continuous source history
needed to construct prior states. Do not export only rows that already satisfy
a candidate gate.

For a genuinely new validation dataset:

- Add a new ES and/or NQ contract or an unseen time range.
- Preserve stable `Instrument`, `File`, and chronological `BarIndex` values.
- Include outcome columns for every row at the tested horizon.
- Include `Time` whenever available.
- Avoid duplicate `Instrument` / `File` / `BarIndex` rows.

Do not present prior analysis summaries, selected-entry outputs, or annotated
copies of the existing primary dataset as independent validation data.

## Recommended Future Explicit Volume Fields

The current NT8 state-log and canonical dataset exports do not include a raw
volume column. To permit direct volume/participation diagnostics for frozen
candidates, future row-level raw exports should add these optional fields:

- `Volume`: actual bar volume.
- `VolumeSMA` or a clearly specified rolling volume mean and window.
- `RelativeVolume`: bar volume relative to the documented rolling mean.
- `VolumeZScore`: standardized volume with documented lookback.
- `BarDirection`: consistent signed or categorical bar direction definition.
- `SignedVolume`: volume signed by the documented direction convention.
- `DirectionalVolumeImbalance`: documented buy/sell or directional imbalance
  proxy, if available.
- `UpDownVolume` proxy: only if the platform/export source supplies it or the
  calculation is fully documented.
- ATR-normalized volume/participation measures: only if implemented with a
  named formula and stable lookback.

These columns are optional explanatory diagnostics. They must not change a
frozen candidate definition or validation threshold without a separate,
explicitly scoped research decision. Preserve `Instrument`, `File`,
`BarIndex`, and `Time` keys so volume fields can be joined to row-level
forward outcomes without ambiguity.

## How To Build Canonical Forward-Return Dataset From Raw NT8 xApvaV01 Exports

The canonical build path has been recovered from
`apva_phase01_exploration.ipynb`, cells 35 and 40, and extracted into:

```text
scripts/apva_build_forward_return_dataset.py
```

Required raw NT8 state-log fields:

- `Instrument`
- `BarIndex`
- `Time`
- `Open`, `High`, `Low`, `Close`
- `MacroState`
- `ActiveDirection`

The raw score fields such as `DominanceScore`, `DegradationScore`,
`BalanceScore`, `TransitionScore`, and `AmbiguityScore` are not the direct
source of `DominantPressure` in the recovered pipeline. The recovered process:

1. Builds rolling `MacroState` quadruplets.
2. Resolves each quadruplet using `tables/apva_online_archetype_lookup.csv`,
   including the notebook's soft-resolution fallback for unknown paths.
3. Computes `RollingEntropy`, `RollingDirectionalPresence`, the four pressure
   values, `DominantPressure`, and `DominantPressureValue` using the recovered
   v2 pressure formulas.
4. Computes forward outcomes from OHLC and `ActiveDirection`.
5. Normalizes returns and excursions with the `ATR14` method used in the
   notebook.

For a new ES/NQ export placed beneath `data/Validation/`, run:

```powershell
python scripts/apva_build_forward_return_dataset.py `
  --inputs "data/Validation/ES/xApvaV01StateLog_ES_NEW.csv" "data/Validation/NQ/xApvaV01StateLog_NQ_NEW.csv" `
  --out "tables/apva_forward_signed_return_dataset_generated.csv" `
  --horizons 5
```

To reproduce-check the historical canonical build using the historical raw
files and all original horizons:

```powershell
python scripts/apva_build_forward_return_dataset.py `
  --inputs "data/Validation/ES/xApvaV01StateLog_ES.csv" "data/Validation/NQ/xApvaV01StateLog_NQ.csv" "data/Validation/6E/xApvaV01StateLog_6E.csv" `
  --out "tables/apva_forward_signed_return_dataset_generated.csv" `
  --horizons 5 10 20 `
  --compare-to "tables/apva_forward_signed_return_dataset_v1.csv"
```

Use `--session-filter RTH` or `--session-filter ETH` only when filtering is
intended before feature construction. The canonical notebook run used all
rows supplied in each source state log.

## Run Readiness Check

For new files saved under `tables/`:

```powershell
python scripts/apva_dataset_readiness_check.py --inputs "tables/*.csv" "outputs/**/*.csv"
```

To check a specific incoming file:

```powershell
python scripts/apva_dataset_readiness_check.py --inputs "tables/apva_new_es_nq_forward_rows.csv"
```

If a file intentionally extends the reference while sharing historical rows,
mark it explicitly for review:

```powershell
python scripts/apva_dataset_readiness_check.py --inputs "tables/apva_extension.csv" --intended-extension "tables/apva_extension.csv"
```

An override does not prove independence; its overlap metrics must still be
reviewed before interpreting validation results.

## Run Fixed Validation

Once an independent new dataset is marked `READY` and is stored in a scanned
data location, rerun the fixed-candidate extended validation:

```powershell
python scripts/apva_fixed_candidate_extended_validation.py
```

Review `outputs/fixed_candidate_extended_validation/extended_dataset_inventory.csv`
to confirm the new dataset is included as independent before interpreting
pooled summaries.
