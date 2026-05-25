# APVA / SpyderTrader Index Topology Family Validation Handoff

## Role For ChatGPT

You are reviewing the next completed research step in the APVA / SpyderTrader
automation project. Interpret the new validation outputs conservatively. The
goal is to decide which description of the surviving ES/NQ normalized behavior
is best supported, not to declare a production trading edge.

## Repository And Dataset

- Repository: `https://github.com/rzahraie/ApvaAnalysis`
- Primary dataset: `tables/apva_forward_signed_return_dataset_v1.csv`
- New experiment script: `scripts/apva_index_topology_family_validation.py`
- Shared helper module: `scripts/apva_analysis_utils.py`
- Commit: `1cd4a3e` (`Add index topology family validation`)

## Research Position Before This Experiment

The robust surviving claim before this run was:

> The broad ES/NQ index-futures APVA topology appears weakly positive in
> normalized units, while stronger-looking sub-effects have generally been
> instrument-specific, small-sample, unstable, or affected by leakage risk.

Prior conclusions:

1. Raw ES-only results did not transfer meaningfully in raw points because
   instrument point scales differ.
2. Normalized transfer showed ES and NQ positive, while 6E was negative.
3. The active research branch is therefore **IndexFutures APVA**, not
   universal APVA.
4. Outcome and future-derived columns must never be entry predictors.
5. Safe entry gating did not materially improve the broad topology, except
   `Instrument == ES`, which is descriptive rather than structural.
6. Compression-persistence work initially reported
   `PriorLast_RollingDirectionalPresence` `Q3` as a strong candidate.
7. Low directional-presence persistence (`D1 -> D1 -> D1`) subsequently proved
   ES-only: `53` ES observations and `0` NQ observations.

## Leakage And Interpretation Constraints

Do **not** use any of these as predictors or entry gates:

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

They may be used only for outcome evaluation or policy simulation.

Safe ex-ante candidate fields include:

- `RollingEntropy`
- `RollingDirectionalPresence`
- `RollingVelocity`
- `DominantPressureValue`
- `DominantPressure`
- `ResolvedArchetype`
- `MacroState`
- `Instrument`
- `File`
- `Time`
- `BarIndex`
- `HorizonBars`

The new script includes a runtime guard against placing prohibited outcome
fields in the predictor configuration.

## New Experiment Purpose

Determine whether the surviving ES/NQ behavior is best described by:

1. Exact pressure grammar.
2. Compression persistence.
3. Prior directional-presence level.
4. Interaction between compression persistence and prior
   directional-presence level.

## Universe, Entry State, And Policy

Universe:

- `ES`
- `NQ`

Entry state:

- `DominantPressure == CompressionPressure`
- `RollingDirectionalPresence == 0`
- `RollingEntropy` between `0.88` and `1.30`, inclusive
- `HorizonBars == 5`

Policy outcome:

- `NormalizedPolicyOutcome = SignedNormalizedReturn`
- If `DirectionalNormalizedMAE <= -2.0`, outcome is capped at `-2.0`
- `DirectionalNormalizedMAE` is used only for this policy simulation

The base population is the broad leakage-safe ES/NQ compression-entry state,
not a prefiltered exact grammar subset.

## Candidate Features Tested

One-dimensional features:

- `PriorCompressionCount`
- `PriorCompressionFraction`
- `PriorCompressionStreak`
- `PriorRotationCount`
- `PriorRotationFraction`
- `PriorLast_RollingDirectionalPresence`
- `PriorMean_RollingDirectionalPresence`
- `PriorSlope_RollingDirectionalPresence`
- `PriorLast_RollingEntropy`
- `PriorSlope_RollingEntropy`
- `PriorLast_DominantPressureValue`
- `PriorSlope_DominantPressureValue`
- `PressureTriplet`
- `PriorPressureSeq`

Two-dimensional interactions:

- `PriorCompressionStreak x PriorLast_RollingDirectionalPresence quartile`
- `PriorCompressionFraction x PriorLast_RollingDirectionalPresence quartile`
- `PressureTriplet x PriorLast_RollingDirectionalPresence quartile`

Important implementation clarification:

- `PressureTriplet` is leakage-safe and defined as:
  `prior[-2] > prior[-1] > entry`
- It does **not** use the state after entry.
- Prior paths are built explicitly within each `Instrument` / `File` group.
- The prior bug involving rolling calculations on string
  `DominantPressure` remains removed.

## Important Validation Correction

The shared quantile binning now preserves identical feature values rather than
ranking equal values by row order and splitting them across quartile labels.

This materially affects interpretation of the earlier
`PriorLast_RollingDirectionalPresence Q3` result: if many values are exactly
the same, a gate cannot trade one arbitrary rank-split subgroup of identical
values. The new validation therefore does not treat that earlier Q3 result as
an executable structural gate by itself.

## Candidate Qualification Criteria

A candidate is marked eligible only if it satisfies:

- Minimum count `>= 50`; preferred count `>= 100`
- Both ES and NQ represented
- Median improves relative to base
- Profit factor improves relative to base
- Instrument positive-fraction stability does not collapse
- Weekly positive-fraction stability does not collapse
- Instrument-week positive-fraction stability does not collapse
- No leaky predictor fields are used

## New Output Files

Directory: `outputs/index_topology_family_validation/`

- `topology_family_entries.csv`
- `topology_family_1d_summary.csv`
- `topology_family_2d_summary.csv`
- `topology_family_by_instrument.csv`
- `topology_family_by_week.csv`
- `topology_family_scorecard.csv`

Related prior outputs for comparison:

- `outputs/index_futures_robustness/index_futures_scorecard.csv`
- `outputs/safe_regime_gating/safe_regime_gating_scorecard.csv`
- `outputs/feature_leakage_audit/leakage_audit_scorecard.csv`
- `outputs/compression_persistence_analysis/compression_persistence_scorecard.csv`
- `outputs/state_transition_hazard/transition_hazard_scorecard.csv`
- `outputs/low_directional_presence_persistence/low_dir_presence_scorecard.csv`

## Base Population Result

From `topology_family_scorecard.csv`:

| Metric | Value |
| --- | ---: |
| Count | 1,672 |
| Mean normalized policy outcome | 0.104294 |
| Median | 0.000000 |
| t-stat | 2.363646 |
| Profit factor | 1.185991 |
| Win rate | 0.498804 |
| Instrument positive fraction | 1.000000 |
| Week positive fraction | 0.625000 |
| Instrument-week positive fraction | 0.533333 |

Interpret this as the broad ES/NQ baseline to beat.

## Best Eligible 1D Candidate

Feature:

- `PriorPressureSeq`

Gate:

- `CompressionPressure > CompressionPressure > RotationalPressure > RotationalPressure > RotationalPressure`

Results:

| Metric | Value |
| --- | ---: |
| Count | 129 |
| ES count | 20 |
| NQ count | 109 |
| Mean | 0.234170 |
| Median | 0.168980 |
| t-stat | 1.409083 |
| Profit factor | 1.436742 |
| Win rate | 0.534884 |
| Instrument positive fraction | 1.000000 |
| Week positive fraction | 0.750000 |
| Instrument-week positive fraction | 0.666667 |

Initial interpretation constraint:

- This satisfies the configured screening criteria and has preferred sample
  size, but it is still only a candidate. It requires additional stability,
  out-of-sample, and dependence-aware testing before being treated as a robust
  family definition.

## Best Eligible 2D Candidate

Features:

- `PriorCompressionFraction x PriorLast_RollingDirectionalPresence`

Gate:

- `0.4 x Q1`

Results:

| Metric | Value |
| --- | ---: |
| Count | 131 |
| ES count | 22 |
| NQ count | 109 |
| Mean | 0.224558 |
| Median | 0.168980 |
| t-stat | 1.369502 |
| Profit factor | 1.418497 |
| Win rate | 0.534351 |
| Instrument positive fraction | 1.000000 |
| Week positive fraction | 0.750000 |
| Instrument-week positive fraction | 0.666667 |

Initial interpretation constraint:

- The 2D result is close to, but not clearly superior to, the best 1D pressure
  sequence result. Avoid assuming that added interaction complexity is
  justified unless its distinct observations or later validation establish
  incremental value.

## Relevant Comparison Observations

From the new 1D summary:

- The prior exact `RRCCC` grammar
  (`RotationalPressure > RotationalPressure > CompressionPressure >
  CompressionPressure > CompressionPressure`) remains positive:
  `n=256`, mean `0.259110`, median `0.069656`, PF `1.479950`,
  ES `47`, NQ `209`.
- Its mean and PF are higher than the selected best 1D candidate, but the
  selected candidate ranks ahead under the validation ordering because its
  median and stability measures are stronger.
- `PriorSlope_DominantPressureValue Q3` also qualifies in this screening:
  `n=397`, mean `0.215520`, median `0.113269`, PF `1.411545`,
  ES `95`, NQ `302`.

These alternatives should be discussed as competing candidate descriptions,
not discarded without examining robustness and overlap.

## Requested Master Analysis

Please analyze the supplied outputs and answer:

1. Does this run materially change the surviving robust claim, or should the
   conclusion remain that the broad ES/NQ topology is weakly positive while
   stronger sub-effects are provisional?
2. Is the best current description more likely:
   exact pressure grammar, broad compression persistence, directional-presence
   context, or an interaction model?
3. Does the best eligible 1D sequence genuinely improve on the existing
   `RRCCC` candidate once count, median, PF, instrument imbalance, weekly
   behavior, and overlap are considered?
4. Does the 2D interaction add enough incremental information to justify its
   extra complexity, or is it substantially restating the 1D grammar result?
5. How should the tie-preserving quantile correction revise interpretation of
   the earlier `PriorLast_RollingDirectionalPresence Q3` result?
6. Are there any remaining leakage, future-state, conditioning, multiple
   comparisons, sample-dependence, or instrument-concentration concerns in
   these outputs?
7. What is the single best next validation experiment to run, with explicit
   acceptance criteria and neutral output naming?

## Required Interpretation Style

- Be conservative and explicit about uncertainty.
- Do not call a candidate an edge unless it survives appropriate robustness
  tests.
- Separate descriptive improvements from generalizable structure.
- Penalize instrument imbalance, small samples, and multiple comparisons.
- Treat medians, profit factor, weekly stability, and ES/NQ representation as
  required context alongside means.
- Use exact output filenames and metrics when making claims.
