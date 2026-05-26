# PriorSlope Q3 Diagnostic Handoff

## Scope

This report describes the frozen `PriorSlope_DominantPressureValue_Q3`
candidate only. No candidate rules or validation thresholds were changed.

Frozen rule:

```text
0.004718046888521732 <= PriorSlope_DominantPressureValue <= 0.05505235072964343
```

## Core Results

| Mode | Count | Mean | Median | PF | Stop Rate | Top 5% Win Contribution Share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | 786 | 0.283 | 0.173 | 1.547 | 0.167 | 0.391 |
| Spacing_10 | 603 | 0.301 | 0.161 | 1.587 | 0.169 | 0.392 |
| Spacing_20 | 541 | 0.290 | 0.135 | 1.573 | 0.168 | 0.415 |

## Regime Stability

The Reference contribution is distributed across regimes rather than being majority-dominated by one regime. The largest absolute Reference regime share is `0.384` from `canonical | original regime`. `6` of `6` aggregated regimes have positive means.

| Reference Regime | Count | Mean | PF | Absolute Contribution Share |
| --- | ---: | ---: | ---: | ---: |
| canonical \| original regime | 397 | 0.216 | 1.412 | 0.384 |
| 2022 \| tightening / bear trend | 138 | 0.412 | 1.648 | 0.255 |
| 2020 \| COVID / crisis | 61 | 0.564 | 2.337 | 0.154 |
| 2024 \| modern mixed regime | 82 | 0.245 | 1.550 | 0.090 |
| 2021 \| 2021 observed regime | 66 | 0.279 | 1.608 | 0.083 |
| generated \| legacy generated regime | 42 | 0.179 | 1.385 | 0.034 |

## Instrument Behavior

Both instruments are positive in aggregate, with ES contributing the higher mean outcome. ES mean is `0.524` on `270` Reference entries; NQ mean is `0.158` on `516` entries.

## Spacing 20

`Spacing_20` retains `541` entries with mean `0.290`, median `0.135`, PF `1.573`, and stop rate `0.168`. Its largest block absolute contribution share is `0.121`. This supports survival under dependence reduction as a distributed diagnostic result, not a new rule.

## Grammar Overlap

Most Reference entries do not overlap either frozen pressure grammar: `704` of `786` Reference entries are in neither `CCRRR` nor `RRCCC`; combined grammar overlap is `82` entries (`0.104`).
The non-overlap subset remains positive with mean `0.243`, median `0.162`, and PF `1.460`, so PriorSlope Q3 does not simply restate either fixed grammar candidate.

| Reference Overlap Class | Count | Mean | Median | PF | Contribution Share |
| --- | ---: | ---: | ---: | ---: | ---: |
| `BaseOnly_NeitherGrammar` | 704 | 0.243 | 0.162 | 1.460 | 0.769 |
| `Overlap_CCRRR` | 14 | 1.076 | 0.299 | 3.362 | 0.068 |
| `Overlap_RRCCC` | 68 | 0.535 | 0.255 | 2.274 | 0.163 |

## Win Distribution And Stops

In Reference mode the positive median outcome (`0.173`) and largest single-block absolute contribution share (`0.106`) indicate that the result is not explained only by a few block clusters. The top 5% of entries still account for `0.391` of positive contribution, so larger wins materially amplify an otherwise positive central distribution.

Stop rates remain similar from Reference (`0.167`) to Spacing_20 (`0.168`).

## Next Fixed Validation Target

Collect and run the unchanged frozen rule on additional paired ES/NQ raw-state-log regimes, with particular value in increasing ES coverage outside the already represented dates. No threshold or candidate change is indicated by this diagnostic report.
