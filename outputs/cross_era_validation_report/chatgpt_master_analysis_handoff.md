# APVA / SpyderTrader Cross-Era Validation Report Handoff

## Prompt For ChatGPT Master Analysis

Interpret this output as a descriptive diagnostic layer over already frozen
validation candidates. No candidate search, threshold tuning, new grammar,
machine learning, or nonlinear interaction modeling was performed.

## Inputs And Outputs

Input directory:

- `outputs/fixed_candidate_extended_validation/`

Report script:

- `scripts/apva_cross_era_validation_report.py`

Generated outputs:

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

Validation modes inherited unchanged from extended validation:

- `Reference`
- `Spacing_10`
- `Spacing_20`

## Regime Parsing

The report parses year tags generally from dataset filenames and assigns
descriptive labels for observed named eras:

| Source Dataset Pattern | Regime Label |
| --- | --- |
| `092020` | `2020 | COVID / crisis` |
| `092022` | `2022 | tightening / bear trend` |
| `092024` | `2024 | modern mixed regime` |
| `_v1` | `canonical | original regime` |
| `generated` without year | `generated | legacy generated regime` |

Unknown future year tags are retained as `<year> | observed regime` rather
than silently classified into an existing narrative label.

## Pooled Validation Status

| Candidate | Reference | Spacing 10 | Spacing 20 |
| --- | --- | --- | --- |
| `RRCCC` | Fail | Fail | Fail |
| `CCRRR` | Pass | Pass | Pass |
| `PriorSlope_DominantPressureValue_Q3` | Pass | Pass | Pass |

Pooled metrics:

| Mode | Candidate | Count | ES | NQ | Mean | PF | Positive Block Fraction | Max Block Contribution |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | `CCRRR` | 143 | 28 | 115 | 0.323743 | 1.643983 | 0.800000 | 0.200350 |
| Reference | `PriorSlope_DominantPressureValue_Q3` | 652 | 200 | 452 | 0.303854 | 1.602780 | 0.723404 | 0.184744 |
| Reference | `RRCCC` | 299 | 70 | 229 | 0.225569 | 1.422987 | 0.580645 | 0.387559 |
| Spacing 10 | `CCRRR` | 122 | 27 | 95 | 0.321289 | 1.657088 | 0.800000 | 0.193578 |
| Spacing 10 | `PriorSlope_DominantPressureValue_Q3` | 481 | 152 | 329 | 0.314969 | 1.635319 | 0.723404 | 0.230249 |
| Spacing 20 | `CCRRR` | 105 | 26 | 79 | 0.343725 | 1.685825 | 0.800000 | 0.330177 |
| Spacing 20 | `PriorSlope_DominantPressureValue_Q3` | 431 | 136 | 295 | 0.305349 | 1.624936 | 0.638298 | 0.265056 |

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
- `regime_survival_score` is the fraction of Reference regime summaries with
  positive mean outcome.
- `instrument_sign_consistency` is the dominant sign fraction across
  Reference regime/instrument summaries.
- `concentration_penalty` is the maximum absolute net contribution share from
  one Reference regime.

This is not an optimized score.
Spacing-mode contribution shares are comparative configuration diagnostics;
the spacing modes reuse observations and are not additive samples.

## Robustness Ranking

| Rank | Candidate | Score | Spacing Survival | Positive Regime Fraction | Concentration Penalty |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `PriorSlope_DominantPressureValue_Q3` | 2.568116 | 1.000000 | 1.000000 | 0.431884 |
| 2 | `CCRRR` | 2.222494 | 1.000000 | 1.000000 | 0.652506 |
| 3 | `RRCCC` | 0.727499 | 0.000000 | 0.800000 | 0.850279 |

## Candidate Interpretation

### Most Regime-Stable Candidate

`PriorSlope_DominantPressureValue_Q3` is most regime-stable under this
descriptive ranking:

- It passes Reference, Spacing 10, and Spacing 20 pooled validation.
- It has a positive mean in all five regime labels.
- Its maximum Reference regime contribution share is `43.19%`, lower than
  `CCRRR` at `65.25%`.
- It has substantially more observations than `CCRRR`.

### Does CCRRR Remain Strongest After 2022 Inclusion?

`CCRRR` remains a leading passing fixed candidate and has the higher pooled
mean and profit factor across each spacing mode. However, it no longer ranks
first when regime concentration and sample breadth are included in the
descriptive robustness score; `PriorSlope_Q3` ranks first.

### Does PriorSlope Become Stronger Than CCRRR Under Broader Regime Diversity?

On cross-era diversification diagnostics, yes: `PriorSlope_Q3` has lower
regime concentration, lower regime-to-regime mean variance, and much larger
sample support. On pooled raw mean and PF, `CCRRR` remains slightly higher.
These are distinct descriptive findings rather than a changed candidate rule.

### Is RRCCC Fundamentally Episodic?

The evidence is consistent with episodic behavior, but does not prove a
fundamental characterization:

- `RRCCC` fails every pooled spacing mode.
- `85.03%` of its absolute Reference net contribution is from the canonical
  original regime.
- Its 2020 regime mean is negative.
- Its Spacing 20 block concentration fails the prior validation requirement.

### Are Pooled Results Dominated By One Regime?

- `PriorSlope_Q3`: no single regime exceeds half of absolute Reference net
  contribution; canonical accounts for `43.19%`, 2022 for `25.52%`, and
  2020 for `17.37%`.
- `CCRRR`: materially concentrated; canonical accounts for `65.25%`.
- `RRCCC`: strongly concentrated; canonical accounts for `85.03%`.

### Which Spacing Mode Appears Most Reliable?

No mode separates the two surviving candidates: both `CCRRR` and
`PriorSlope_Q3` pass all three pooled modes. `Spacing_20` is the most
dependence-reduced stress view and both still pass it; `RRCCC` does not.

### Is ES Still Underrepresented?

Yes, especially for `CCRRR`:

- `CCRRR` Reference: ES `28` of `143` rows (`19.6%`).
- `CCRRR` Spacing 20: ES `26` of `105` rows (`24.8%`).
- `PriorSlope_Q3` Reference: ES `200` of `652` rows (`30.7%`).

### Biggest Remaining Statistical Weakness

The largest weakness is sparse regime-level support for `CCRRR` outside the
canonical original dataset. Its Reference counts in the new named regimes are
only `3` in 2020, `4` in 2022, and `5` in 2024, with another `2` in the
legacy generated regime. Its pooled pass therefore remains provisional and
depends heavily on the original regime despite surviving spacing reduction.

## Conservative Conclusion

After 2022 inclusion and regime decomposition:

- `CCRRR` still passes all pooled validation modes and remains strong on
  pooled mean/PF.
- `PriorSlope_DominantPressureValue_Q3` also passes all pooled modes and is
  better diversified across regimes under the declared diagnostic ranking.
- `RRCCC` remains the most concentration-sensitive candidate.

The next evidence need is additional fixed-rule observations, especially ES
rows and non-canonical `CCRRR` occurrences, not new candidate development.
