# APVA / SpyderTrader Fixed-Candidate Extended Validation Handoff

## Prompt For ChatGPT Master Analysis

Interpret this completed fixed-rule validation conservatively. This run was
intended to evaluate the frozen APVA candidates across additional compatible
row-level datasets already present in the repository. It did not search for
new candidates or tune thresholds.

## Repository And Run Artifacts

- Repository: `https://github.com/rzahraie/ApvaAnalysis`
- Script: `scripts/apva_fixed_candidate_extended_validation.py`
- Output directory: `outputs/fixed_candidate_extended_validation/`
- Primary source dataset: `tables/apva_forward_signed_return_dataset_v1.csv`

Outputs:

- `extended_dataset_inventory.csv`
- `extended_entries.csv`
- `extended_candidate_summary.csv`
- `extended_dataset_candidate_summary.csv`
- `extended_block_summary.csv`
- `extended_instrument_summary.csv`
- `extended_scorecard.csv`

## Fixed Candidate Rules

The following rules were evaluated without modification:

1. `Base_ES_NQ`
   - Broad ES/NQ base compression-entry state.

2. `RRCCC`
   - `RotationalPressure > RotationalPressure > CompressionPressure > CompressionPressure > CompressionPressure`

3. `CCRRR`
   - `CompressionPressure > CompressionPressure > RotationalPressure > RotationalPressure > RotationalPressure`

4. `PriorSlope_DominantPressureValue_Q3`
   - Frozen interval:
     `0.004718046888521732 <= PriorSlope_DominantPressureValue <= 0.05505235072964343`
   - This interval was applied directly and was not recomputed.

## Base Entry And Policy

Universe:

- `ES`
- `NQ`

Base entry:

- `DominantPressure == CompressionPressure`
- `RollingDirectionalPresence == 0`
- `RollingEntropy` between `0.88` and `1.30`, inclusive
- `HorizonBars == 5`

Outcome policy:

- `NormalizedPolicyOutcome = SignedNormalizedReturn`
- If `DirectionalNormalizedMAE <= -2.0`, policy outcome is set to `-2.0`
- `DirectionalNormalizedMAE` is outcome simulation only, not a predictor

Validation modes:

- `Reference`: no de-clustering
- `Spacing_10`: minimum 10 bars between retained candidate entries within
  `Instrument` / `File` / `Candidate`
- `Spacing_20`: minimum 20 bars between retained candidate entries within
  `Instrument` / `File` / `Candidate`

Time blocks use chronological ISO weeks, consistent with prior fixed-rule
walk-forward validation.

## Leakage Constraints

The script retains a runtime guard against using these prohibited predictor or
gate fields:

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

## Dataset Inventory Result

The script inventoried `315` CSV files across `tables/`, `outputs/`, `data/`,
and the repository root.

Schema-compatible row-level files found: `34`.

Included independent compatible datasets: `1`.

Included:

| Dataset | Reason |
| --- | --- |
| `tables/apva_forward_signed_return_dataset_v1.csv` | Canonical primary row-level forward-return dataset |

Excluded schema-compatible table datasets:

| Dataset | Reason |
| --- | --- |
| `tables/apva_nonoverlap_forward_signed_return_dataset_v1.csv` | Row-identical derivative of the primary dataset; no additional observations |
| `tables/apva_walk_forward_signed_return_dataset_v1.csv` | Row-identical derivative of the primary dataset; no additional observations |
| `tables/apva_file_level_walk_forward_dataset_v1.csv` | Row-identical derivative of the primary dataset; no additional observations |

All three excluded table files contain the same `55,806` row keys as the
primary file and the same observed date range. They add annotations or
validation labels, not new contracts or time periods.

Excluded schema-compatible output files:

- `30` files under `outputs/` matched the required columns but were excluded
  because they are prior analysis outputs or selected-entry subsets, not
  independent additional validation datasets.

## Central Limitation

Only one independent compatible row-level dataset exists in the repository.
Therefore:

- The pooled results are numerically identical to the primary-dataset results.
- This run does **not** provide cross-dataset, new-contract, or new-time-range
  replication.
- It validates the dataset inventory and preserves the fixed-rule results, but
  it cannot establish pooled-data robustness beyond the original sample.

## Pass/Fail Criteria

For each non-base candidate and validation mode, pass requires:

- Count `>= 75`
- ES count `> 0` and NQ count `> 0`
- Median `> 0`
- Profit factor greater than the method-matched `Base_ES_NQ` profit factor
- Positive ISO-week block fraction `> 0.60`
- Maximum single-block contribution fraction `<= 0.40`
- ES mean `>= 0`
- NQ mean `>= 0`

`Base_ES_NQ` is reference-only.

## Pooled Summary Results

Because one independent dataset was included, the pooled summary is the same
as the dataset-specific summary.

| Mode | Candidate | Count | ES | NQ | Mean | Median | PF | Positive Blocks | Max Block Contribution | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Reference | `RRCCC` | 256 | 47 | 209 | 0.259110 | 0.069656 | 1.479950 | 0.625 | 0.394062 | Pass |
| Reference | `CCRRR` | 129 | 20 | 109 | 0.234170 | 0.168980 | 1.436742 | 0.750 | 0.307046 | Pass |
| Reference | `PriorSlope_DominantPressureValue_Q3` | 397 | 95 | 302 | 0.215520 | 0.113269 | 1.411545 | 0.625 | 0.427763 | Fail |
| Spacing 10 | `RRCCC` | 199 | 40 | 159 | 0.260671 | 0.093541 | 1.477060 | 0.625 | 0.404435 | Fail |
| Spacing 10 | `CCRRR` | 108 | 19 | 89 | 0.213981 | 0.095670 | 1.405931 | 0.750 | 0.328331 | Pass |
| Spacing 10 | `PriorSlope_DominantPressureValue_Q3` | 257 | 61 | 196 | 0.275424 | 0.081633 | 1.546629 | 0.625 | 0.436370 | Fail |
| Spacing 20 | `RRCCC` | 163 | 36 | 127 | 0.204612 | 0.044633 | 1.365167 | 0.625 | 0.580929 | Fail |
| Spacing 20 | `CCRRR` | 91 | 18 | 73 | 0.219822 | 0.120172 | 1.400861 | 0.750 | 0.595708 | Fail |
| Spacing 20 | `PriorSlope_DominantPressureValue_Q3` | 223 | 54 | 169 | 0.281366 | 0.113269 | 1.574407 | 0.625 | 0.418711 | Fail |

## Headline Interpretation

### CCRRR

`CCRRR` survives:

- Reference validation
- Spacing `10` dependence reduction

It does not survive spacing `20`, because maximum single-block contribution
increases to `0.595708`, above the fixed `0.40` threshold.

Its spacing `10` pass retains:

- Count: `108`
- ES count: `19`
- NQ count: `89`
- Median: `0.095670`
- PF: `1.405931`, above method base PF `1.162846`
- Positive block fraction: `0.750`
- Maximum single-block contribution: `0.328331`

This is consistent with `CCRRR` remaining the leading provisional candidate
after moderate dependence reduction, while its thin ES representation remains
a substantial limitation.

### RRCCC

`RRCCC` passes only the reference mode. It does not recover under either
requested spacing reduction:

- At spacing `10`, it narrowly fails block concentration:
  `0.404435 > 0.40`.
- At spacing `20`, block concentration worsens to `0.580929`.

### PriorSlope_DominantPressureValue_Q3

The frozen slope candidate does not pass any requested mode in this extended
run:

- It fails reference mode on maximum block contribution: `0.427763`.
- It fails spacing `10` on maximum block contribution: `0.436370`.
- It fails spacing `20` on maximum block contribution: `0.418711`.

Its aggregate means and profit factors remain positive, but its returns do not
meet the prespecified block-concentration criterion.

## Robustness Versus Dataset Specificity

There is no evidence here of cross-dataset robustness because no independent
additional compatible row-level dataset was available.

The results are best characterized as:

> Fixed-rule validation results within the existing primary ES/NQ dataset,
> with de-clustering sensitivity already measured, but without independent
> dataset replication.

## Verification

The new script was compiled and executed successfully.

Additional checks confirmed:

- Frozen candidate rules and thresholds were imported from the prior
  fixed-rule validators rather than rediscovered.
- The required leakage guard is active.
- Retained entry spacing is at least `10` bars for `Spacing_10`.
- Retained entry spacing is at least `20` bars for `Spacing_20`.
- Row-identical table derivatives were excluded rather than pooled as if they
  were independent evidence.

## Requested Master Analysis Questions

Please evaluate:

1. Given that only one independent compatible dataset exists, should the
   conclusion remain limited to within-sample fixed-rule validation rather than
   extended robustness?
2. Does `CCRRR` remain the best provisional candidate after surviving spacing
   `10`, despite only `19` ES observations in that mode?
3. Should the near miss for `RRCCC` at spacing `10`
   (`0.404435` versus a `0.40` limit) be treated as substantively different
   from failure, or only recorded as sensitivity to the fixed threshold?
4. Does the persistent block-concentration failure of
   `PriorSlope_DominantPressureValue_Q3` outweigh its larger sample and higher
   aggregate profit factor under spacing modes?
5. What new, independent data collection or contract/time-period expansion is
   required before any candidate can be tested for genuine replication?

Use neutral language: `candidate`, `validation`, `provisional`, and
`fixed-rule`. Do not call any result an edge.
