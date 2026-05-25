# APVA / SpyderTrader Frozen Topology Walk-Forward Validation Handoff

## Prompt For ChatGPT Master Analysis

You are reviewing a completed frozen-candidate, time-block validation run for
the APVA / SpyderTrader automation project. Analyze the outputs conservatively.
This was a validation step, not a candidate-discovery step, and no result should
be described as a production trading edge without further independent testing.

## Repository, Data, And Commit

- Repository: `https://github.com/rzahraie/ApvaAnalysis`
- Primary dataset: `tables/apva_forward_signed_return_dataset_v1.csv`
- New script: `scripts/apva_topology_walkforward_validation.py`
- Shared utility module: `scripts/apva_analysis_utils.py`
- New output directory: `outputs/topology_walkforward_validation/`
- Commit: `f0a18ca` (`Add frozen topology walk-forward validation`)

## Research State Entering This Run

The surviving robust claim before this validation was:

> The broad ES/NQ index-futures APVA compression-entry topology appears weakly
> positive in normalized units. Stronger sub-effects remain provisional until
> they survive stability and out-of-sample validation.

Relevant prior findings:

1. Raw point-scale results did not transfer reliably across instruments.
2. Normalized testing supported work on ES and NQ, while 6E behaved negatively.
3. The active hypothesis is **IndexFutures APVA**, not universal APVA.
4. Leakage auditing prohibited return, future-close, excursion, and hit fields
   as entry predictors.
5. Earlier `PriorLast_RollingDirectionalPresence Q3` results were weakened by
   a quantile issue: tied identical values had been separable only by row rank,
   which is not an executable live gate.
6. A later topology-family validation produced three relevant frozen
   candidates for this run:
   - `RRCCC`
   - `CCRRR`
   - `PriorSlope_DominantPressureValue Q3`

## Leakage Rules

The following columns must **not** be used as predictors or gates:

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

Permitted outcome use in this run:

- `SignedNormalizedReturn` is the evaluated return.
- `DirectionalNormalizedMAE` is used only to apply the fixed disaster-stop
  policy.

The validation script includes a runtime check that no prohibited outcome
fields are configured as candidate predictors.

## Universe, Base Entry, And Outcome Policy

Universe:

- `ES`
- `NQ`

Base entry state:

- `DominantPressure == CompressionPressure`
- `RollingDirectionalPresence == 0`
- `RollingEntropy` between `0.88` and `1.30`, inclusive
- `HorizonBars == 5`

Policy:

- `NormalizedPolicyOutcome = SignedNormalizedReturn`
- If `DirectionalNormalizedMAE <= -2.0`, set
  `NormalizedPolicyOutcome = -2.0`
- Disaster stop: `2.0` normalized units

Prior paths are constructed explicitly within each `Instrument` / `File`
history. No future pressure state is used as a candidate gate.

## Fixed Validation Design

This run intentionally did **not** search for new candidates, refit thresholds,
or rank newly generated alternatives.

Time blocks:

- Chronological ISO-week blocks
- Eight blocks: `2026-W14` through `2026-W21`

Frozen candidates evaluated:

1. `Base_ES_NQ`
   - Broad base compression-entry topology only.

2. `RRCCC`
   - Prior pressure sequence:
     `RotationalPressure > RotationalPressure > CompressionPressure > CompressionPressure > CompressionPressure`

3. `CCRRR`
   - Prior pressure sequence:
     `CompressionPressure > CompressionPressure > RotationalPressure > RotationalPressure > RotationalPressure`

4. `PriorSlope_DominantPressureValue_Q3`
   - Frozen interval from the prior topology-family output:
     `0.004718046888521732 <= PriorSlope_DominantPressureValue <= 0.05505235072964343`
   - This boundary was applied directly; it was **not** recomputed on the
     walk-forward sample.

## Validation Criteria

A candidate passes only if all of the following are true:

- Total count `>= 100`
- Both ES and NQ are represented
- Median normalized policy outcome `> 0`
- Profit factor `>` base ES/NQ profit factor
- Positive block fraction `> 0.60`
- Maximum single block contribution fraction `<= 0.35`
- No catastrophic negative aggregate instrument result

Operational definition used for the last criterion:

- If the aggregate mean outcome for either ES or NQ is below zero, the
  candidate fails.

Maximum single block contribution definition:

- Maximum chronological block policy sum divided by the candidate's total
  positive net policy sum, when total sum is positive.

## Output Files To Analyze

Directory: `outputs/topology_walkforward_validation/`

- `walkforward_entries.csv`
- `walkforward_candidate_summary.csv`
- `walkforward_block_summary.csv`
- `walkforward_instrument_summary.csv`
- `walkforward_scorecard.csv`

Useful prior comparison files:

- `outputs/index_topology_family_validation/topology_family_scorecard.csv`
- `outputs/index_topology_family_validation/topology_family_1d_summary.csv`
- `outputs/index_topology_family_validation/topology_family_2d_summary.csv`
- `outputs/index_futures_robustness/index_futures_scorecard.csv`
- `outputs/feature_leakage_audit/leakage_audit_scorecard.csv`

## Aggregate Candidate Results

From `walkforward_candidate_summary.csv`:

| Candidate | Count | ES | NQ | Mean | Median | t-stat | PF | Win Rate | Positive Block Fraction | Instrument-Week Positive Fraction | Max Block Contribution | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Base_ES_NQ | 1,672 | 375 | 1,297 | 0.104294 | 0.000000 | 2.363646 | 1.185991 | 0.498804 | 0.625 | 0.533333 | 0.453342 | Reference only |
| RRCCC | 256 | 47 | 209 | 0.259110 | 0.069656 | 1.913053 | 1.479950 | 0.519531 | 0.625 | 0.533333 | 0.394062 | Fail |
| CCRRR | 129 | 20 | 109 | 0.234170 | 0.168980 | 1.409083 | 1.436742 | 0.534884 | 0.750 | 0.666667 | 0.307046 | Pass |
| PriorSlope_DominantPressureValue_Q3 | 397 | 95 | 302 | 0.215520 | 0.113269 | 2.219082 | 1.411545 | 0.523929 | 0.625 | 0.600000 | 0.427763 | Fail |

## Pass/Fail Details

### CCRRR

`CCRRR` is the only frozen candidate passing every configured criterion:

- Count: `129`, above the `100` threshold
- ES/NQ represented: `20` ES and `109` NQ
- Median: `0.168980`, positive
- PF: `1.436742`, above base PF `1.185991`
- Positive block fraction: `0.750`, above `0.60`
- Maximum block contribution: `0.307046`, below `0.35`
- ES aggregate mean: `0.834496`, positive
- NQ aggregate mean: `0.124019`, positive

Interpretation limit:

- It passes this prespecified block validation, but its ES count is only `20`
  and its overall total count is `129`. This supports continued validation of
  `CCRRR` as a candidate, not a strong generalization claim.

### RRCCC

`RRCCC` remains attractive on aggregate return statistics:

- Count: `256`
- Mean: `0.259110`
- Median: `0.069656`
- PF: `1.479950`

However, it fails the validation because:

- Maximum single block contribution fraction: `0.394062`, above `0.35`

It passes the other configured requirements, including both instruments being
positive in aggregate.

Interpretation limit:

- `RRCCC` has stronger aggregate mean and PF than `CCRRR`, but its performance
  is more block-concentrated under this test.

### Frozen PriorSlope_DominantPressureValue Q3

This candidate has the largest filtered sample among non-base candidates:

- Count: `397`
- ES: `95`
- NQ: `302`
- Mean: `0.215520`
- Median: `0.113269`
- PF: `1.411545`
- t-stat: `2.219082`

However, it fails because:

- Maximum single block contribution fraction: `0.427763`, above `0.35`

Interpretation limit:

- Its larger and more balanced sample is noteworthy, but the concentration
  failure prevents it from passing the frozen block-validation screen.

## Instrument-Level Results

From `walkforward_instrument_summary.csv`:

| Candidate | ES Count | ES Mean | NQ Count | NQ Mean | Negative Instrument Aggregate? |
| --- | ---: | ---: | ---: | ---: | --- |
| Base_ES_NQ | 375 | 0.231611 | 1,297 | 0.067483 | No |
| RRCCC | 47 | 0.740185 | 209 | 0.150925 | No |
| CCRRR | 20 | 0.834496 | 109 | 0.124019 | No |
| PriorSlope_DominantPressureValue_Q3 | 95 | 0.501594 | 302 | 0.125530 | No |

All candidate variants are positive in ES and NQ on aggregate, but the ES
counts for the grammar candidates remain small.

## Block-Level Notes

All candidates have observations in each of the eight weekly blocks.

`CCRRR` positive blocks:

- Positive: `2026-W14`, `W15`, `W16`, `W18`, `W19`, `W21`
- Negative: `2026-W17`, `W20`
- Positive block fraction: `6 / 8 = 0.750`

`RRCCC` positive blocks:

- Positive: `2026-W15`, `W16`, `W18`, `W19`, `W21`
- Negative: `2026-W14`, `W17`, `W20`
- Positive block fraction: `5 / 8 = 0.625`

Frozen prior-slope Q3 positive blocks:

- Positive: `2026-W15`, `W16`, `W18`, `W20`, `W21`
- Negative: `2026-W14`, `W17`, `W19`
- Positive block fraction: `5 / 8 = 0.625`

## Conservative Current Interpretation

The result does not yet justify replacing the prior robust claim with a broad
claim of established structure. A defensible interim reading is:

> The broad ES/NQ APVA compression-entry topology remains weakly positive.
> Among four frozen rules evaluated across chronological week blocks, `CCRRR`
> is the only candidate satisfying the prespecified validation screen, while
> `RRCCC` and the frozen pressure-value-slope candidate fail due to block
> concentration. `CCRRR` warrants further fixed-rule validation, with caution
> because its ES sample remains small.

## Requested Master Analysis

Please interpret the output files and answer the following:

1. Does the fact that `CCRRR` passes while `RRCCC` fails justify elevating
   `CCRRR` as the leading frozen candidate, or is eight weekly blocks too
   limited for that distinction?
2. How should aggregate-return superiority in `RRCCC` be weighed against its
   block-concentration failure?
3. Does the frozen prior-slope candidate's larger sample and higher t-stat
   deserve additional validation despite its concentration failure?
4. Is the maximum single block contribution criterion appropriately computed
   and appropriately strict for these samples?
5. Does `CCRRR`'s small ES representation (`20` observations) remain a
   material limitation despite both instrument means being positive?
6. Are there remaining dependence, overlapping-horizon, multiple-testing,
   conditioning, leakage, or temporal-partition concerns?
7. What is the single best next **fixed-rule** validation experiment for
   `CCRRR`, `RRCCC`, and the frozen slope candidate, including:
   - exact candidate rules,
   - data split or resampling method,
   - prespecified pass/fail criteria,
   - neutral script name,
   - neutral output filenames?

## Required Analysis Style

- Keep language neutral: use `candidate`, `validation`, `block`, and
  `provisional`.
- Do not call a result an edge unless independent testing justifies it.
- Treat the pass result as conditional on the prespecified weekly block test.
- Explicitly discuss small ES sample size, block count, and block
  concentration.
- Use exact output filenames and metrics when drawing conclusions.
