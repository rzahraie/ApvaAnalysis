# APVA Auction Archetype Report

## Cluster 0: Degradation / Exhaustion Collapse

- MotifCount: 24
- TotalOccurrences: 48
- MeanNextRunLength: 1.16
- MeanImmediateFailurePct: 38.83
- MeanDirectionalContinuationPct: 0.00
- MeanUnresolvedCount: 0.83
- MeanDirectionalCount: 0.62
- MeanDegradingCount: 2.08
- MeanBalanceCount: 0.46

Top motifs:

- `Degrading->Degrading->Degrading->Degrading` count=9
- `Unresolved->Unresolved->Unresolved->Degrading` count=5
- `Unresolved->Directional->Directional->Degrading` count=3
- `Directional->Directional->Directional->Degrading` count=3
- `Directional->Degrading->Degrading->Degrading` count=3


## Cluster 1: Fragile Rotational Churn

- MotifCount: 16
- TotalOccurrences: 37
- MeanNextRunLength: 1.38
- MeanImmediateFailurePct: 56.25
- MeanDirectionalContinuationPct: 0.00
- MeanUnresolvedCount: 1.06
- MeanDirectionalCount: 0.19
- MeanDegradingCount: 0.81
- MeanBalanceCount: 1.94

Top motifs:

- `Unresolved->Unresolved->Unresolved->Balance` count=10
- `Unresolved->Unresolved->Balance->Balance` count=5
- `Degrading->Unresolved->Unresolved->Balance` count=3
- `Balance->Balance->Balance->Balance` count=3
- `Unresolved->Balance->Unresolved->Balance` count=2


## Cluster 2: Persistent Non-Directional Compression

- MotifCount: 32
- TotalOccurrences: 96
- MeanNextRunLength: 3.71
- MeanImmediateFailurePct: 12.22
- MeanDirectionalContinuationPct: 0.00
- MeanUnresolvedCount: 1.94
- MeanDirectionalCount: 0.34
- MeanDegradingCount: 0.84
- MeanBalanceCount: 0.50

Top motifs:

- `Balance->Unresolved->Unresolved->Unresolved` count=14
- `Unresolved->Unresolved->Balance->Unresolved` count=10
- `Unresolved->Balance->Unresolved->Unresolved` count=10
- `Degrading->Unresolved->Unresolved->Unresolved` count=7
- `Unresolved->Degrading->Unresolved->Unresolved` count=5


## Cluster 3: Directional Continuation / Sponsor Release

- MotifCount: 12
- TotalOccurrences: 30
- MeanNextRunLength: 1.20
- MeanImmediateFailurePct: 22.92
- MeanDirectionalContinuationPct: 27.08
- MeanUnresolvedCount: 1.00
- MeanDirectionalCount: 2.17
- MeanDegradingCount: 0.83
- MeanBalanceCount: 0.00

Top motifs:

- `Directional->Directional->Directional->Directional` count=8
- `Unresolved->Unresolved->Unresolved->Directional` count=6
- `Unresolved->Unresolved->Directional->Directional` count=5
- `Unresolved->Directional->Directional->Directional` count=3
- `Unresolved->Unresolved->Degrading->Directional` count=1


## Cluster 4: Pure Unresolved Compression Equilibrium

- MotifCount: 1
- TotalOccurrences: 186
- MeanNextRunLength: 0.00
- MeanImmediateFailurePct: 0.00
- MeanDirectionalContinuationPct: 0.00
- MeanUnresolvedCount: 4.00
- MeanDirectionalCount: 0.00
- MeanDegradingCount: 0.00
- MeanBalanceCount: 0.00

Top motifs:

- `Unresolved->Unresolved->Unresolved->Unresolved` count=186

