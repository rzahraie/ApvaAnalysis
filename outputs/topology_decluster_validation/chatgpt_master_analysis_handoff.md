# APVA / SpyderTrader Frozen Topology De-Cluster Validation Handoff

## Completed Validation

Implemented `scripts/apva_topology_decluster_validation.py` and generated the
requested outputs in `outputs/topology_decluster_validation/`.

The script reuses the frozen candidate definitions from the committed
walk-forward validator, applies only the prescribed de-clustering methods,
compares profit factor against method-matched `Base_ES_NQ`, and retains the
leakage guard.

Commit:

- `28ea8aa` - `Add frozen topology decluster validation`

## Passing Combinations

| Decluster Method | Passing Candidates |
| --- | --- |
| Reference | `RRCCC`, `CCRRR` |
| Spacing 5 | `CCRRR`, `PriorSlope_DominantPressureValue_Q3` |
| Spacing 10 | `CCRRR` |
| Spacing 20 | None |
| First per instrument/file/week | None |

## Key Findings

### CCRRR At Spacing 10

`CCRRR` still passes after minimum bar spacing `10`:

| Metric | Value |
| --- | ---: |
| Count | 108 |
| ES count | 19 |
| NQ count | 89 |
| Mean | 0.2140 |
| Median | 0.0957 |
| Profit factor | 1.4059 |
| Method-matched base profit factor | 1.1628 |
| Positive block fraction | 0.750 |
| Maximum single-block contribution | 0.3283 |

### CCRRR At Spacing 20

`CCRRR` does not pass after minimum bar spacing `20`:

- Count remains above threshold at `91`.
- Aggregate return metrics remain positive.
- It fails on block concentration: `0.5957 > 0.40`.

### RRCCC

`RRCCC` does not recover under any actual spacing de-clustering method:

- It passes only the no-declustering reference under the new `<= 0.40`
  contribution limit.
- At spacing `5`, `10`, and `20`, its maximum block contribution remains too
  high.

### PriorSlope_DominantPressureValue_Q3

`PriorSlope_DominantPressureValue_Q3` recovers only at spacing `5`:

| Metric | Value |
| --- | ---: |
| Count | 317 |
| ES count | 75 |
| NQ count | 242 |
| Median | 0.1133 |
| Profit factor | 1.4536 |
| Method-matched base profit factor | 1.0991 |
| Maximum single-block contribution | 0.3585 |

It fails spacing `10` and `20` on block concentration.

## Small-Sample Notes

The one-entry-per-`Instrument`/`File`/week method leaves only `15`
observations per candidate, below the required `75`, so those descriptive
results should not be relied upon.

`CCRRR` at spacing `10` passes formally, but its ES representation is thin at
`19` observations and remains an important limitation.

## Verification

Verification included:

- Compilation of the new script.
- Full execution against the primary dataset.
- Inspection of candidate, block, instrument, and scorecard outputs.
- Direct retained-gap checks confirming minimum gaps of `5`, `10`, and `20`
  bars under the respective spacing methods.

## Output Files

Directory: `outputs/topology_decluster_validation/`

- `decluster_entries.csv`
- `decluster_candidate_summary.csv`
- `decluster_block_summary.csv`
- `decluster_instrument_summary.csv`
- `decluster_scorecard.csv`

## Conservative Interpretation

`CCRRR` is the only frozen candidate that remains passing through spacing `10`
under the prescribed de-clustering screen. This supports continued fixed-rule
validation of `CCRRR`, while its small ES sample and failure at spacing `20`
remain material limitations.

`PriorSlope_DominantPressureValue_Q3` is still of interest because it passes
at spacing `5` with a larger sample, but it does not remain stable as spacing
increases. `RRCCC` does not survive spacing-based dependence reduction under
the specified contribution threshold.

These are candidate-validation results, not evidence sufficient to call a
production trading edge.
