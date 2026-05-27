# APVA Frozen Candidate Failure Attribution Handoff

## Scope

This is a diagnostic attribution report over the three frozen candidates only. It does not change candidate rules, thresholds, filters, or validation logic.

## Headline

- `PriorSlope_DominantPressureValue_Q3` remains positive in pooled Reference (mean `0.149`, PF `1.266`) but fails the frozen Reference test because `PositiveBlockFraction` fails.
- It passes `Spacing_10` and fails `Spacing_20` because `PositiveBlockFraction` fails at the stricter spacing.
- `CCRRR` fails all modes. Reference failure criteria: `PositiveBlockFraction`.
- `RRCCC` fails all modes. Reference failure criteria: `Median|PositiveBlockFraction|MaxSingleBlockContribution`.

## 032024 Synchronization Answers

1. Adding `032024` did not materially worsen PriorSlope_Q3: it contributed `+4.632` on `83` Reference rows. It diluted the pooled mean from `0.156` to `0.149` because its own mean is lower (`0.056`), but it did not change the failure reason.
2. The single largest negative dataset remains `122023` with contribution `-21.712`.
3. The four negative year-aggregate regime/instrument cells are unchanged after `032024`; `032024` does not add a new negative year-aggregate cell.
4. Within `032024`, ES hurt PriorSlope (`-3.785` sum, mean `-0.086`), NQ helped (`+8.417` sum, mean `0.216`), and pooled PriorSlope helped (`+4.632`).
5. Yes. PriorSlope still fails Reference and Spacing_20 only because `PositiveBlockFraction` fails.
6. Yes. `Spacing_10` remains the only passing mode for PriorSlope_Q3.
7. Yes. Volume-enabled Reference rows remain weaker (mean `-0.038`, sum `-20.482`) than legacy/no-explicit-volume rows (mean `0.280`, sum `218.330`).
8. `032024` behaves more like the positive-participation `062023` dataset than the negative `122023` dataset: its pooled contribution is positive (`+4.632`).

## What Broke PriorSlope_Q3?

The Reference failure is not caused by a negative pooled mean: total contribution is `197.848` across `1324` rows. It is a breadth failure: only `0.570` of blocks are positive, below the frozen `> 0.6` requirement.

`Spacing_10` raises positive block fraction to `0.579` and passes. `Spacing_20` lowers it to `0.561` and fails. Because each spacing mode is independently selected, the stricter mode both drops and reselects observations; its selected sample loses block breadth without turning the pooled mean negative.

## Four Negative Reference Regime/Instrument Cells

| Regime | Instrument | Count | Mean | Median | PF | Sum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2019 | ES | 45 | -0.235 | -0.321 | 0.651 | -10.576 |
| 2017 | ES | 43 | -0.175 | -0.197 | 0.694 | -7.542 |
| 2023 | ES | 56 | -0.068 | -0.117 | 0.890 | -3.824 |
| 2023 | NQ | 61 | -0.041 | -0.148 | 0.938 | -2.490 |
| 2016 | ES | 44 | -0.041 | 0.000 | 0.925 | -1.800 |
| 2016 | NQ | 40 | -0.013 | -0.524 | 0.982 | -0.537 |

Thirteen of `19` Reference regime/instrument cells remain positive; degradation is concentrated in the four cells above.

## Dataset Attribution

| Dataset | Volume Enabled | Count | Mean | PF | Sum |
| --- | --- | ---: | ---: | ---: | ---: |
| 122023 | True | 81 | -0.268 | 0.606 | -21.712 |
| 062019 | True | 81 | -0.121 | 0.794 | -9.835 |
| 122017 | True | 180 | -0.037 | 0.941 | -6.628 |
| 062016 | True | 84 | -0.028 | 0.956 | -2.337 |
| 032024 | True | 83 | 0.056 | 1.085 | 4.632 |
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
| VolumeEnabled | 545 | -0.038 | -0.065 | 0.940 | -20.482 |
| LegacyOrNoExplicitVolume | 779 | 0.280 | 0.177 | 1.545 | 218.330 |

The explicit-volume group is weaker than the legacy/no-explicit-volume group in the current Reference sample, but the weakness is not uniform: `032024` and `062023` contribute positively while `062019`, `122017`, and especially `122023` contribute negatively. The honest attribution is broader regime diversity revealing fragility, not volume fields themselves causing degradation.

## Blocks And Concentration

PriorSlope Reference has `46` non-positive blocks out of `107`. No single instrument is net-negative in the pooled Reference aggregate; the failure is not an ES-only or NQ-only collapse. It is a distributed breadth problem with concentrated negative dataset/cell pockets.

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
