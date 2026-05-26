# APVA / SpyderTrader Cross-Era Validation Report Handoff

## Prompt For ChatGPT Master Analysis

Interpret this report as descriptive diagnostics over already frozen validation
candidates. No candidate search, threshold tuning, new grammar, machine
learning, or nonlinear interaction modeling was performed.

## Fix Applied

Multiple datasets now share a regime label: `092022` and `122022` both map
to `2022 | tightening / bear trend`. The chart failure was repaired by
aggregating repeated `Regime` / `Candidate` Reference rows before pivoting:

- `Count` is summed.
- `Sum` is summed where provided and `Mean` is computed from total `Sum` and
  total `Count` (equivalent to a count-weighted mean).
- `ProfitFactor` is weighted by `Count`, because gross-win/gross-loss
  aggregates are not supplied in the report input summaries.

The same era-level aggregation is used for regime-stability diagnostics.
`regime_candidate_summary.csv` remains dataset-level so the two 2022 source
files remain separately auditable.

## Inputs And Outputs

Input directory:

- `outputs/fixed_candidate_extended_validation/`

Included source datasets:

- `tables/apva_forward_signed_return_dataset_es_nq_092020_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_092022_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_092024_generated.csv`
- `tables/apva_forward_signed_return_dataset_es_nq_122022_generated.csv`
- `tables/apva_forward_signed_return_dataset_generated.csv`
- `tables/apva_forward_signed_return_dataset_v1.csv`

Report outputs:

- `regime_candidate_summary.csv`
- `regime_block_summary.csv`
- `candidate_stability_summary.csv`
- `concentration_diagnostics.csv`
- `robustness_rankings.csv`
- `cross_era_validation_scorecard.csv`
- `charts/candidate_mean_by_regime.svg`
- `charts/candidate_pf_by_regime.svg`
- `charts/contribution_share_by_regime.svg`
- `charts/spacing_survival_heatmap.svg`

## Frozen Candidates Only

- `RRCCC`
- `CCRRR`
- `PriorSlope_DominantPressureValue_Q3`

Validation modes inherited unchanged:

- `Reference`
- `Spacing_10`
- `Spacing_20`

## Regime Parsing

The report parses date/year tags from filenames and supplies descriptive labels
for observed eras. The six included datasets resolve to five era labels
because both 2022 datasets belong to the same named era.

| Source Dataset Pattern | Regime Label |
| --- | --- |
| `092020` | `2020 | COVID / crisis` |
| `092022`, `122022` | `2022 | tightening / bear trend` |
| `092024` | `2024 | modern mixed regime` |
| `_v1` | `canonical | original regime` |
| `generated` without year | `generated | legacy generated regime` |

Unknown future year tags remain `<year> | observed regime` rather than being
silently assigned to an existing narrative category.

## Pooled Validation Status

| Candidate | Reference | Spacing 10 | Spacing 20 |
| --- | --- | --- | --- |
| `RRCCC` | Fail | Fail | Fail |
| `CCRRR` | Pass | Pass | Pass |
| `PriorSlope_DominantPressureValue_Q3` | Pass | Pass | Pass |

Pooled metrics:

| Mode | Candidate | Count | ES | NQ | Mean | PF | Positive Block Fraction | Max Block Contribution |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | `CCRRR` | 147 | 30 | 117 | 0.286299 | 1.540333 | 0.695652 | 0.220388 |
| Reference | `PriorSlope_DominantPressureValue_Q3` | 720 | 233 | 488 | 0.283870 | 1.541638 | 0.690909 | 0.179073 |
| Reference | `RRCCC` | 315 | 73 | 242 | 0.192702 | 1.356176 | 0.512821 | 0.430618 |
| Spacing 10 | `CCRRR` | 126 | 29 | 97 | 0.277683 | 1.532923 | 0.695652 | 0.216867 |
| Spacing 10 | `PriorSlope_DominantPressureValue_Q3` | 543 | 182 | 362 | 0.295241 | 1.567647 | 0.690909 | 0.217588 |
| Spacing 10 | `RRCCC` | 258 | 66 | 192 | 0.179234 | 1.328068 | 0.512821 | 0.453683 |
| Spacing 20 | `CCRRR` | 109 | 28 | 81 | 0.292494 | 1.543832 | 0.695652 | 0.373769 |
| Spacing 20 | `PriorSlope_DominantPressureValue_Q3` | 486 | 163 | 324 | 0.289174 | 1.565544 | 0.618182 | 0.248208 |
| Spacing 20 | `RRCCC` | 222 | 62 | 160 | 0.124868 | 1.224367 | 0.512821 | 0.698935 |

The inherited pooled extended-validation summary reports one more ES/NQ
instrument row than total rows for `PriorSlope_Q3` in each mode. This report
preserves those upstream metrics; the discrepancy should be audited in the
extended validator separately from this chart fix.

## Transparent Ranking Formula

The report ranking is descriptive and fixed:

```text
robustness_score =
    spacing_survival_score
  + regime_survival_score
  + instrument_sign_consistency
  - concentration_penalty
```

Where:

- `spacing_survival_score` is the fraction of the three pooled spacing modes
  passed.
- `regime_survival_score` is the fraction of aggregated Reference regimes with
  positive mean outcome.
- `instrument_sign_consistency` is the dominant sign fraction across
  Reference regime/instrument summaries.
- `concentration_penalty` is the maximum absolute net contribution share from
  one Reference regime.

This score is not optimized. Spacing modes reuse observations and are not
additive samples.

## Robustness Ranking

| Rank | Candidate | Score | Spacing Survival | Positive Regime Fraction | Concentration Penalty |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `PriorSlope_DominantPressureValue_Q3` | 2.490465 | 1.000000 | 1.000000 | 0.418626 |
| 2 | `CCRRR` | 1.982232 | 1.000000 | 1.000000 | 0.717768 |
| 3 | `RRCCC` | 0.420830 | 0.000000 | 0.600000 | 0.815534 |

## Required Interpretive Answers

### Which Candidate Is Most Regime-Stable?

`PriorSlope_DominantPressureValue_Q3` ranks first. It passes all three pooled
modes, remains positive across all five aggregated era labels, and has lower
era concentration than `CCRRR`.

### Which Candidate Is Most Concentration-Sensitive?

`RRCCC` is most concentration-sensitive: `81.55%` of its absolute Reference
net contribution comes from the canonical original regime, and it fails all
three pooled validation modes.

### Does CCRRR Remain Strongest After 2022 Inclusion?

`CCRRR` still passes `Reference`, `Spacing_10`, and `Spacing_20`, but it does
not lead this concentration-aware descriptive ranking. The added `122022`
sample offsets much of the favorable `092022` CCRRR result; the combined 2022
CCRRR contribution is still slightly positive, but it is based on only eight
Reference rows.

### Does PriorSlope Become Stronger Than CCRRR Under Broader Regime Diversity?

Under the declared robustness ranking, yes. `PriorSlope_Q3` has much larger
sample support and materially lower regime concentration. Their pooled
Reference mean and PF are very close; this finding concerns diversification
and support, not a changed rule.

### Is RRCCC Fundamentally Episodic?

The diagnostics remain consistent with episodic behavior, without proving
that characterization: `RRCCC` fails every pooled mode and is dominated by
its canonical-regime contribution.

### Are Pooled Results Dominated By One Regime?

- `PriorSlope_Q3` is comparatively diversified: canonical accounts for
  `41.86%` of absolute Reference net contribution.
- `CCRRR` is materially concentrated: canonical accounts for `71.78%`.
- `RRCCC` is strongly concentrated: canonical accounts for `81.55%`.

### Which Spacing Mode Appears Most Reliable?

No spacing mode separates the two surviving candidates: `CCRRR` and
`PriorSlope_Q3` pass all three modes. `Spacing_20` is the strictest included
dependence-reduction view, and both pass it; `RRCCC` does not.

### Is ES Still Underrepresented?

Yes, particularly for `CCRRR`:

- `CCRRR` Reference: ES `30` of `147` rows (`20.4%`).
- `CCRRR` Spacing 20: ES `28` of `109` rows (`25.7%`).
- `PriorSlope_Q3` Reference: ES `233` of `720` rows (`32.4%`).

### Biggest Remaining Statistical Weakness

`CCRRR` remains sparse outside the canonical dataset and concentrated in that
era. The two 2022 sources also disagree directionally for CCRRR while jointly
providing only eight Reference occurrences. More fixed-rule observations,
especially ES occurrences outside the canonical period, remain necessary.

## Conservative Conclusion

With the additional 2022 dataset incorporated, `CCRRR` remains a passing
provisional fixed candidate under all three validation modes, while
`PriorSlope_DominantPressureValue_Q3` ranks first on the declared descriptive
robustness measure because it is less concentrated and far better supported.
`RRCCC` remains the most concentration-sensitive candidate. No candidate rule
or validation threshold was changed.
