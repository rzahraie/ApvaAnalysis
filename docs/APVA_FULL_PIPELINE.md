# APVA Full Pipeline

Use the full pipeline runner to build validation datasets from raw NT8 state
logs and refresh the frozen-candidate reports in one ordered execution.

## Raw Export Location

Place paired ES and NQ raw state-log exports under either mixed-case form of
the validation directory:

```text
data/Validation/ES/
data/Validation/NQ/
```

or:

```text
data/validation/ES/
data/validation/NQ/
```

The runner resolves these paths case-insensitively.

## Required Naming Convention

Use exact paired filenames, where `<REGIME>` is a six-digit period token such
as `092022` or `122022`:

```text
xApvaV01StateLog_ES<REGIME>.csv
xApvaV01StateLog_NQ<REGIME>.csv
```

Examples:

```text
data/Validation/ES/xApvaV01StateLog_ES122022.csv
data/Validation/NQ/xApvaV01StateLog_NQ122022.csv
```

Variant names such as `_CLEAN` are inventoried but are not initially selected
for automatic pairing. The runner first tests the exact dated filename. If
that file is rejected specifically for embedded repeated CSV header rows and
one matching `_CLEAN` companion exists, the runner records the rejection and
uses that clean companion for the affected instrument. Other build failures
stop the pipeline.

## Run The Full Pipeline

Refresh all discovered paired dated regimes:

```powershell
python scripts/apva_run_full_pipeline.py --raw-root data/validation
```

Limit building and readiness execution to a particular period:

```powershell
python scripts/apva_run_full_pipeline.py --raw-root data/validation --regimes 122022
```

Inspect planned pairing and commands without executing:

```powershell
python scripts/apva_run_full_pipeline.py --raw-root data/validation --dry-run
```

Single-instrument evaluation is not automatic. Use
`--allow-single-instrument` only when that limitation is deliberate and
should be recorded in the run status.

## Execution Order

The runner executes these existing fixed-rule stages in order:

1. Discover and pair exact ES/NQ raw state-log files by regime token.
2. Build each canonical row-level dataset with
   `scripts/apva_build_forward_return_dataset.py`.
3. Run `scripts/apva_dataset_readiness_check.py` for each built regime.
4. Run `scripts/apva_fixed_candidate_extended_validation.py`.
5. Run `scripts/apva_cross_era_validation_report.py`.
6. Verify that built datasets were included in fixed validation and that
   cross-era Reference counts agree with fixed-validation Reference counts.
   The consistency check also verifies that summarized fixed-validation
   entries have evaluable policy outcomes and reconciled ES/NQ counts.
7. Write a final machine-readable consistency check and master-analysis
   handoff.

Normal executions refresh requested paired-regime build and report outputs.
The accepted `--force` flag is a run annotation for scripted callers; it does
not alter candidate rules, thresholds, or the default refresh behavior.

## Key Outputs

Per-regime canonical data:

```text
tables/apva_forward_signed_return_dataset_es_nq_<REGIME>_generated.csv
```

Per-regime readiness reports:

```text
outputs/dataset_readiness_es_nq_<REGIME>/
```

Consolidated fixed validation:

```text
outputs/fixed_candidate_extended_validation/
```

Cross-era diagnostics and stale-output guard:

```text
outputs/cross_era_validation_report/
outputs/cross_era_validation_report/cross_era_consistency_check.csv
```

Pipeline audit and final handoff:

```text
outputs/full_pipeline_run/full_pipeline_inventory.csv
outputs/full_pipeline_run/full_pipeline_regime_status.csv
outputs/full_pipeline_run/full_pipeline_step_log.csv
outputs/full_pipeline_run/full_pipeline_consistency_check.csv
outputs/full_pipeline_run/chatgpt_master_analysis_handoff.md
```

Build diagnostics for each paired regime are retained beneath
`outputs/full_pipeline_run/build_es_nq_<REGIME>/`.

## Interpretation Rules

The runner uses the existing frozen candidate definitions and validation
thresholds. A successful run means the artifacts were generated in order and
their reported counts agree; it does not prove a candidate.

Review:

- `full_pipeline_regime_status.csv` to confirm each intended ES/NQ pair was
  built, readiness-checked, and included in fixed validation.
- `full_pipeline_consistency_check.csv` to confirm cross-era reports are not
  stale relative to fixed validation.
- `chatgpt_master_analysis_handoff.md` for the final descriptive summary.

## Do Not Do These

- Do not manually run downstream scripts out of order when interpreting the
  current report.
- Do not edit candidate definitions or validation thresholds during a fixed
  validation refresh.
- Do not treat stale narrative handoffs as current evidence.
- Do not include selected-entry outputs, summaries, or prior report tables as
  independent validation datasets.
