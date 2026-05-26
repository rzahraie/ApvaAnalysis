# Volume And Participation Diagnostic Handoff

## Scope

This is a descriptive report for the frozen `PriorSlope_DominantPressureValue_Q3` candidate.
No candidate rule or validation threshold was changed.

## Is Volume Explicitly Present?

Explicit volume fields are present in the inventoried source files.

Accordingly, this report cannot test direct traded-volume behavior. It evaluates available state-participation proxies only. Textual event labels such as `PeakVolume`, where present, are not a numeric volume measurement.

## Available Participation Proxies

- Numeric proxies: `DominantPressureValue`, `PriorSlope_DominantPressureValue`, `RollingEntropy`, `RollingDirectionalPresence`, `SponsorConfidence`, `DominanceScore`, `DegradationScore`, `BalanceScore`, `TransitionScore`, `AmbiguityScore`.
- Categorical proxies: `MacroState`, `SponsorState`, `SequencePhase`.

## Pressure-State Interpretation

The frozen rule itself selects positive prior pressure-value slopes, so PriorSlope Q3 corresponds to rising pressure-state participation by construction. In Reference mode its slope spans `0.004718` to `0.055052` with median `0.034936`.

Among available continuous proxies, `PriorSlope_DominantPressureValue` has the largest absolute Spearman association with PriorSlope membership within Reference base entries (`0.227`), as expected because it defines the frozen candidate. Excluding that definitional field, the largest membership association is `BalanceScore` at only `-0.060`.

## Entropy And State Context

PriorSlope Q3 Reference entries have RollingEntropy mean `1.163` and median `1.295`; Reference base entries outside PriorSlope Q3 have entropy mean `1.138`. The difference (`0.025`) is modestly higher for PriorSlope Q3. Because the frozen base state already bounds entropy, this report treats entropy as context rather than an independent explanation.

The most common MacroState is `Unresolved` with `762` of `786` Reference entries (`0.969`). The most common SponsorState is `Unresolved` (`0.491` share), and the most common SequencePhase is `B2B` (`0.358` share).

The candidate is strongly situated in `Unresolved` MacroState, but no specific SponsorState or SequencePhase dominates a majority beyond the stated shares. No categorical state should be treated as a replacement rule from this diagnostic report.

## ES, NQ, And Regimes

Both instruments remain positive in Reference mode: ES mean `0.524` on `270` entries and NQ mean `0.158` on `516` entries.

Across Reference regime/instrument cells, `11` of `11` have positive means. See `prior_slope_q3_by_regime_instrument.csv` for the sparse-cell detail.

## Participation-State Versus Outcome Artifact

PriorSlope Q3 is defined only from an ex-ante pressure-state slope and is not constructed from return or excursion outcomes. Within Reference PriorSlope entries, the strongest numeric-proxy association with outcome is `DominantPressureValue` with Spearman `0.052`. The available data support describing the candidate as a pressure-state participation proxy, but do not establish a direct volume phenomenon because volume is absent.

## Required Future NT8 Export Fields For Direct Volume Analysis

Add these row-level fields to future raw exports while keeping stable `Instrument`, `File`, `BarIndex`, and `Time` keys:

- `Volume`
- `VolumeSMA` or another clearly specified rolling volume mean
- `RelativeVolume`
- `VolumeZScore`
- `BarDirection`
- `SignedVolume`
- `DirectionalVolumeImbalance`
- `UpDownVolume` proxy, if available
- ATR-normalized volume or participation measures, if implemented with documented formulas

## Next Fixed Validation Target

Export a new paired ES/NQ regime with the unchanged state fields plus explicit volume columns above, then rerun the existing full pipeline and this diagnostic report. The purpose is direct participation measurement under the frozen candidate, not candidate modification.
