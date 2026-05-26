# APVA Frozen Candidate Failure Attribution Handoff

## Scope

This is a diagnostic attribution report over the three frozen candidates only. It does not change candidate rules, thresholds, filters, or validation logic.

## Headline

- `PriorSlope_DominantPressureValue_Q3` remains positive in pooled Reference (mean `0.169`, PF `1.307`) but fails the frozen Reference test because `PositiveBlockFraction` fails.
- It passes `Spacing_10` and fails `Spacing_20` because `PositiveBlockFraction` fails at the stricter spacing.
- `CCRRR` fails all modes. Reference failure criteria: `PositiveBlockFraction`.
- `RRCCC` fails all modes. Reference failure criteria: `Median|PositiveBlockFraction|MaxSingleBlockContribution`.

## What Broke PriorSlope_Q3?

The Reference failure is not caused by a negative pooled mean: total contribution is `195.554` across `1157` rows. It is a breadth failure: only `0.593` of blocks are positive, below the frozen `> 0.6` requirement.

`Spacing_10` raises positive block fraction to `0.604` and passes. `Spacing_20` lowers it to `0.582` and fails. Because each spacing mode is independently selected, the stricter mode both drops and reselects observations; its selected sample loses block breadth without turning the pooled mean negative.

## Four Negative Reference Regime/Instrument Cells

| Regime | Instrument | Count | Mean | Median | PF | Sum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2019 | ES | 45 | -0.235 | -0.321 | 0.651 | -10.576 |
| 2017 | ES | 43 | -0.175 | -0.197 | 0.694 | -7.542 |
| 2023 | ES | 56 | -0.068 | -0.117 | 0.890 | -3.824 |
| 2023 | NQ | 61 | -0.041 | -0.148 | 0.938 | -2.490 |

Thirteen of `17` Reference regime/instrument cells remain positive; degradation is concentrated in the four cells above.

## Dataset Attribution

| Dataset | Volume Enabled | Count | Mean | PF | Sum |
| --- | --- | ---: | ---: | ---: | ---: |
| 122023 | True | 81 | -0.268 | 0.606 | -21.712 |
| 062019 | True | 81 | -0.121 | 0.794 | -9.835 |
| 122017 | True | 180 | -0.037 | 0.941 | -6.628 |
| 122022 | False | 68 | 0.092 | 1.129 | 6.274 |
| LegacyGenerated | False | 42 | 0.179 | 1.385 | 7.503 |
| 062023 | True | 36 | 0.428 | 1.781 | 15.398 |
| 092024 | False | 75 | 0.208 | 1.523 | 15.588 |
| 062021 | False | 66 | 0.279 | 1.608 | 18.438 |
| 092020 | False | 61 | 0.564 | 2.337 | 34.410 |
| 092022 | False | 70 | 0.722 | 2.297 | 50.555 |
| CanonicalOriginal | False | 397 | 0.216 | 1.412 | 85.562 |

`122017` is not the principal negative dataset: its Reference sum is `-6.628`, while the most negative dataset is `122023` at `-21.712`.

## Volume-Enabled Versus Legacy Coverage

| Group | Count | Mean | Median | PF | Sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| VolumeEnabled | 378 | -0.060 | -0.090 | 0.903 | -22.777 |
| LegacyOrNoExplicitVolume | 779 | 0.280 | 0.177 | 1.545 | 218.330 |

The explicit-volume group is weaker than the legacy/no-explicit-volume group in the current Reference sample, but the weakness is not uniform: `062023` contributes positively while `062019`, `122017`, and especially `122023` contribute negatively. The honest attribution is broader regime diversity revealing fragility, not volume fields themselves causing degradation.

## Blocks And Concentration

PriorSlope Reference has `37` non-positive blocks out of `91`. No single instrument is net-negative in the pooled Reference aggregate; the failure is not an ES-only or NQ-only collapse. It is a distributed breadth problem with concentrated negative dataset/cell pockets.

## Candidate Disposition

`CCRRR` and `RRCCC` do not survive as broad frozen candidates in this expanded validation: each fails every requested validation mode. This report does not replace them or modify them.

PriorSlope_Q3 retains a descriptive narrower observation worth validating later: performance is positive in most regime/instrument cells and survives `Spacing_10`, while failing in a small set of specific cells and at `Spacing_20`. That is a follow-up validation question, not a rule change.

## Next Honest Research Step

Preserve the frozen candidates and collect/validate additional comparable regimes with explicit volume exports, then repeat this attribution prospectively. The immediate evidence supports documenting where the frozen result is unstable, not revising the candidate.

## Output Files

- `candidate_regime_attribution.csv`
- `candidate_instrument_attribution.csv`
- `candidate_regime_instrument_attribution.csv`
- `candidate_block_attribution.csv`
- `candidate_dataset_attribution.csv`
- `candidate_spacing_degradation.csv`
- `volume_enabled_vs_legacy_attribution.csv`
- `prior_slope_q3_failure_cells.csv`
- `failure_attribution_scorecard.csv`
