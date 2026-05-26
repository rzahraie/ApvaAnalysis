# Volume And Participation Diagnostic Handoff

## Scope

This is a descriptive report for the frozen `PriorSlope_DominantPressureValue_Q3` candidate.
No candidate rule or validation threshold was changed.

## Is Volume Explicitly Present?

Explicit volume fields are present in the inventoried source files.

Direct volume fields are available for a subset of evaluated rows: `461` Reference PriorSlope entries from regimes `2017, 2019, 2023, 2024` and instruments `ES, NQ`. Volume results below apply only to this volume-available subset; all-data state results continue to use every available era.

## Available Participation Proxies

- Numeric proxies: `DominantPressureValue`, `PriorSlope_DominantPressureValue`, `RollingEntropy`, `RollingDirectionalPresence`, `SponsorConfidence`, `DominanceScore`, `DegradationScore`, `BalanceScore`, `TransitionScore`, `AmbiguityScore`.
- Categorical proxies: `MacroState`, `SponsorState`, `SequencePhase`.
- Direct volume fields: `Volume`, `VolumeSMA`, `RelativeVolume`, `VolumeZScore`, `BarDirection`, `SignedVolume`, `UpVolume`, `DownVolume`, `FlatVolume`, `UpDownVolumeDelta`.
- Spyder proxy fields: `SpyderDominantVolume`, `SpyderNonDominantVolume`, `SpyderDominantVolumeShare`, `SpyderNonDominantVolumeShare`, `SpyderNonDominantColor`, `SpyderSplitMethod`.

## Pressure-State Interpretation

The frozen rule itself selects positive prior pressure-value slopes, so PriorSlope Q3 corresponds to rising pressure-state participation by construction. In Reference mode its slope spans `0.004718` to `0.055052` with median `0.029380`.

Among available continuous proxies, `PriorSlope_DominantPressureValue` has the largest absolute Spearman association with PriorSlope membership within Reference base entries (`0.246`), as expected because it defines the frozen candidate. Excluding that definitional field, the largest membership association is `RollingEntropy` at only `0.152`.

## Entropy And State Context

PriorSlope Q3 Reference entries have RollingEntropy mean `1.197` and median `1.295`; Reference base entries outside PriorSlope Q3 have entropy mean `1.135`. The difference (`0.061`) is modestly higher for PriorSlope Q3. Because the frozen base state already bounds entropy, this report treats entropy as context rather than an independent explanation.

The most common MacroState is `Unresolved` with `1213` of `1240` Reference entries (`0.978`). The most common SponsorState is `Unresolved` (`0.527` share), and the most common SequencePhase is `B2B` (`0.363` share).

The candidate is strongly situated in `Unresolved` MacroState, but no specific SponsorState or SequencePhase dominates a majority beyond the stated shares. No categorical state should be treated as a replacement rule from this diagnostic report.

## ES, NQ, And Regimes

Both instruments remain positive in Reference mode: ES mean `0.247` on `451` entries and NQ mean `0.113` on `789` entries.

Across Reference regime/instrument cells, `13` of `17` have positive means. See `prior_slope_q3_by_regime_instrument.csv` for the sparse-cell detail.

## Volume-Available Subset

Within Reference base entries where explicit volume is available, PriorSlope Q3 has mean `RelativeVolume` `1.036` versus `1.037` outside PriorSlope Q3; it is therefore `lower` in this subset.

Mean `VolumeZScore` is `-0.102` for PriorSlope Q3 versus `-0.060` outside it; it is `lower` in this subset.

`SignedVolume` has Spearman `0.018` with membership and `0.046` with outcome. `UpDownVolumeDelta` has Spearman `0.018` with membership and `0.046` with outcome. These describe direction-sensitive volume behavior; they are not candidate conditions.

`SpyderDominantVolumeShare` has Spearman `0.019` with membership and `0.027` with outcome. `SpyderNonDominantVolumeShare` has Spearman `-0.019` with membership and `-0.027` with outcome. These shares are price-geometry weighted split-volume proxies, not bid/ask volume.

Instrument and regime splits are provided in `prior_slope_q3_volume_summary.csv`, `prior_slope_q3_spyder_split_summary.csv`, and `prior_slope_q3_volume_correlations.csv`. Within the available regime: ES RelativeVolume delta `0.114`; ES VolumeZScore delta `0.064`; NQ RelativeVolume delta `-0.081`; NQ VolumeZScore delta `-0.116`. Because explicit volume currently appears only in one regime, cross-regime consistency cannot yet be established; ES/NQ comparisons within that available regime are descriptive only.

## Participation-State Versus Outcome Artifact

PriorSlope Q3 is defined only from an ex-ante pressure-state slope and is not constructed from return or excursion outcomes. Within Reference PriorSlope entries, the strongest numeric-proxy association with outcome is `DegradationScore` with Spearman `-0.054` across all-data state proxies. The new volume tables provide direct-volume diagnostics only for the available subset and do not establish a new rule or threshold.

## Coverage Boundary

Continue exporting the explicit volume and Spyder split fields for additional paired ES/NQ regimes, then rerun the unchanged pipeline. The purpose is to expand direct-volume coverage under the frozen candidate, not to tune or replace it.
