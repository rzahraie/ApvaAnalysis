#!/usr/bin/env python3
"""APVA Information Decay Study v0.1.

Measure how quickly structural forecast information decays in the observed
StructuralState + AgeBucket transition graph. Forward outcomes are diagnostics
only.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable

from APVA_MinimalEngine_52 import age_zone, directional_return, instrument_columns, load_results, safe_mean
from APVA_ProcessGraph_54 import node_for, node_text
from APVA_StructuralForecast_56 import (
    Forecast,
    brier_score,
    correlation,
    distribution_entropy,
    graph,
    outcome,
    propagate,
    ranked_distribution,
)
from APVA_StructuralLifeCycle_44 import STRUCTURAL_STATES, ensure_dir, fmt, pct

HORIZONS = (1, 2, 3, 5, 8, 10)
Node = tuple[str, str]


@dataclass
class Decay:
    node: Hashable
    count: int
    accuracy: dict[int, float]
    top3: dict[int, float]
    entropy: dict[int, float]
    initial_accuracy: float
    threshold: float
    half_life: float
    memory_strength: float
    entropy_slope: float
    entropy_growth: float
    entropy_growth_rate: float
    memory_class: str


@dataclass
class GroupMemory:
    name: str
    count: int
    half_life: float
    memory_strength: float
    entropy_growth: float


@dataclass
class Outcome:
    count: int
    mean_dr: float
    median_dr: float
    continuation: float
    failure: float
    flat: float
    skew: float


@dataclass
class StudyResult:
    instrument: str
    source_paths: list
    bars: list
    nodes: list[Node]
    graph: dict[Node, dict[Node, float]]
    forecasts: list[Forecast]
    decay: dict[Node, Decay]
    state_memory: dict[str, GroupMemory]
    zone_memory: dict[str, GroupMemory]
    outcomes: dict[Node, Outcome]
    memory_top1_correlation: float
    memory_top3_correlation: float
    memory_entropy_correlation: float


def mean(values: Iterable[float]) -> float:
    return safe_mean(list(values))


def build_forecasts(stream: list[Hashable], transitions: dict[Hashable, dict[Hashable, float]]) -> list[Forecast]:
    universe = set(stream)
    rows = []
    for horizon in HORIZONS:
        for index in range(len(stream) - horizon):
            current = stream[index]
            actual = stream[index + horizon]
            distribution = propagate(current, transitions, horizon)
            ranked = ranked_distribution(distribution)
            first = ranked[0][1] if ranked else 0.0
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            choices = [item for item, _ in ranked]
            rows.append(Forecast(
                index, horizon, current, actual, distribution, ranked, first,
                first - second, distribution_entropy(distribution),
                brier_score(distribution, actual, universe),
                actual in choices[:1], actual in choices[:2], actual in choices[:3],
            ))
    return rows


def trapezoid(values: dict[int, float]) -> float:
    total = 0.0
    for left, right in zip(HORIZONS, HORIZONS[1:]):
        total += (right - left) * (values[left] + values[right]) / 2
    return total / (HORIZONS[-1] - HORIZONS[0])


def half_life(accuracy: dict[int, float]) -> tuple[float, float]:
    threshold = accuracy[1] / 2
    for horizon in HORIZONS:
        if accuracy[horizon] <= threshold:
            return float(horizon), threshold
    return float(HORIZONS[-1] + 1), threshold


def half_life_text(value: float) -> str:
    return "10+" if value > HORIZONS[-1] else str(int(value))


def memory_class(value: float) -> str:
    if value >= 8:
        return "LongMemory"
    if value >= 4:
        return "MediumMemory"
    if value >= 1:
        return "ShortMemory"
    return "InstantDecay"


def build_decay(stream: list[Hashable], forecasts: list[Forecast]) -> dict[Hashable, Decay]:
    rows = {}
    for node in sorted(set(stream), key=str):
        selected = [row for row in forecasts if row.current == node]
        accuracy = {horizon: mean(row.top1 for row in selected if row.horizon == horizon) for horizon in HORIZONS}
        top3 = {horizon: mean(row.top3 for row in selected if row.horizon == horizon) for horizon in HORIZONS}
        entropy = {horizon: mean(row.entropy for row in selected if row.horizon == horizon) for horizon in HORIZONS}
        life, threshold = half_life(accuracy)
        slope = (entropy[HORIZONS[-1]] - entropy[HORIZONS[0]]) / (HORIZONS[-1] - HORIZONS[0])
        growth = entropy[HORIZONS[-1]] - entropy[HORIZONS[0]]
        growth_rate = growth / entropy[HORIZONS[0]] if entropy[HORIZONS[0]] else 0.0
        rows[node] = Decay(
            node, sum(row.horizon == 1 for row in selected), accuracy, top3, entropy,
            accuracy[1], threshold, life, trapezoid(accuracy), slope, growth,
            growth_rate, memory_class(life),
        )
    return rows


def group_memory(name: str, rows: Iterable[Decay]) -> GroupMemory:
    rows = list(rows)
    return GroupMemory(
        name, sum(row.count for row in rows), mean(row.half_life for row in rows),
        mean(row.memory_strength for row in rows), mean(row.entropy_growth for row in rows),
    )


def build_state_memory(decay: dict[Node, Decay]) -> dict[str, GroupMemory]:
    return {
        state: group_memory(state, (row for node, row in decay.items() if node[0] == state))
        for state in STRUCTURAL_STATES
        if any(node[0] == state for node in decay)
    }


def build_zone_memory(decay: dict[Node, Decay]) -> dict[str, GroupMemory]:
    return {
        zone: group_memory(zone, (row for node, row in decay.items() if age_zone(node[1]) == zone))
        for zone in ("Young", "Middle", "Late")
        if any(age_zone(node[1]) == zone for node in decay)
    }


def node_outcomes(bars: list, nodes: list[Node]) -> dict[Node, Outcome]:
    values: dict[Node, list[float]] = defaultdict(list)
    for index, node in enumerate(nodes):
        value = directional_return(bars, index, 5)
        if value is not None:
            values[node].append(value)
    return {node: Outcome(*outcome(samples).__dict__.values()) for node, samples in values.items()}


def correlations(decay: dict[Node, Decay]) -> tuple[float, float, float]:
    rows = list(decay.values())
    return (
        correlation((row.memory_strength for row in rows), (row.accuracy[1] for row in rows)),
        correlation((row.memory_strength for row in rows), (row.top3[1] for row in rows)),
        correlation((row.memory_strength for row in rows), (row.entropy_growth for row in rows)),
    )


def study(result) -> StudyResult:
    bars = result.bars
    nodes = [node_for(bar) for bar in bars]
    transitions = graph(nodes)
    forecasts = build_forecasts(nodes, transitions)
    decay = build_decay(nodes, forecasts)
    top1, top3, entropy = correlations(decay)
    return StudyResult(
        result.instrument, result.source_paths, bars, nodes, transitions, forecasts,
        decay, build_state_memory(decay), build_zone_memory(decay),
        node_outcomes(bars, nodes), top1, top3, entropy,
    )


def recommendation(results: list[StudyResult]) -> tuple[str, str]:
    life = mean(row.half_life for result in results for row in result.decay.values())
    strength = mean(row.memory_strength for result in results for row in result.decay.values())
    entropy = mean(row.entropy_growth for result in results for row in result.decay.values())
    if life >= 5:
        label = "Strong Memory Model"
    elif life >= 3:
        label = "Moderate Memory Model"
    else:
        label = "Weak Memory Model"
    reason = (
        f"MeanHalfLife={fmt(life, 2)} bars; MeanMemoryStrength={fmt(strength)}; "
        f"MeanEntropyGrowth={fmt(entropy)}; CrossInstrumentReplication={len(results)} instruments."
    )
    return label, reason


def decay_curve_header() -> str:
    return "StateAgeNode | Count | " + " | ".join(f"Top1_t+{horizon}" for horizon in HORIZONS)


def entropy_header() -> str:
    return "StateAgeNode | " + " | ".join(f"Entropy_t+{horizon}" for horizon in HORIZONS) + " | EntropySlope | EntropyGrowth | EntropyGrowthRate"


def write_per_instrument(result: StudyResult, out_root: Path) -> None:
    path = out_root / result.instrument / f"InformationDecay_{result.instrument}.txt"
    ensure_dir(path.parent)
    lines = [
        "APVA Information Decay Study v0.1", "=" * 104, "Diagnostics",
        f"Instrument: {result.instrument}",
        "Input path(s): " + ", ".join(str(item) for item in result.source_paths),
        f"Total rows: {len(result.bars)}", f"Node count: {len(result.decay)}",
        f"Forecast count: {len(result.forecasts)}",
        "", "1. Forecast Decay Curves", decay_curve_header(),
    ]
    for node, row in sorted(result.decay.items()):
        lines.append(f"{node_text(node)} | {row.count} | " + " | ".join(pct(row.accuracy[horizon]) for horizon in HORIZONS))
    lines += ["", "2. Information Half-Life", "StateAgeNode | InitialAccuracy | HalfLifeThreshold | InformationHalfLife"]
    for node, row in sorted(result.decay.items(), key=lambda item: (-item[1].half_life, item[0])):
        lines.append(f"{node_text(node)} | {pct(row.initial_accuracy)} | {pct(row.threshold)} | {half_life_text(row.half_life)} bars")
    lines += ["", "3. Structural Memory Strength", "StateAgeNode | MemoryStrength"]
    for node, row in sorted(result.decay.items(), key=lambda item: (-item[1].memory_strength, item[0])):
        lines.append(f"{node_text(node)} | {fmt(row.memory_strength)}")
    lines += ["", "4. Entropy Growth", entropy_header()]
    for node, row in sorted(result.decay.items()):
        lines.append(f"{node_text(node)} | " + " | ".join(fmt(row.entropy[horizon]) for horizon in HORIZONS) + f" | {fmt(row.entropy_slope)} | {fmt(row.entropy_growth)} | {fmt(row.entropy_growth_rate)}")
    lines += ["", "5. Memory Classes", "StateAgeNode | InformationHalfLife | MemoryClass"]
    for node, row in sorted(result.decay.items()):
        lines.append(f"{node_text(node)} | {half_life_text(row.half_life)} bars | {row.memory_class}")
    lines += ["", "6. State-Level Memory", "StructuralState | MeanHalfLife | MeanMemoryStrength | MeanEntropyGrowth"]
    for state, row in sorted(result.state_memory.items()):
        lines.append(f"{state} | {fmt(row.half_life, 2)} | {fmt(row.memory_strength)} | {fmt(row.entropy_growth)}")
    lines += ["", "7. Age Effects", "StructuralState | AgeBucket | InformationHalfLife | MemoryStrength | EntropyGrowth"]
    for node, row in sorted(result.decay.items()):
        lines.append(f"{node[0]} | {node[1]} | {half_life_text(row.half_life)} | {fmt(row.memory_strength)} | {fmt(row.entropy_growth)}")
    lines += ["", "8. Young/Middle/Late Memory", "Zone | MeanHalfLife | MeanMemoryStrength | MeanEntropyGrowth"]
    for zone, row in result.zone_memory.items():
        lines.append(f"{zone} | {fmt(row.half_life, 2)} | {fmt(row.memory_strength)} | {fmt(row.entropy_growth)}")
    lines += ["", "9. Cross-Instrument Comparison", "Instrument-only diagnostic. See aggregate report for cross-instrument replication."]
    lines += [
        "", "10. Memory vs Forecast Accuracy",
        f"Correlation MemoryStrength vs Top1Accuracy: {fmt(result.memory_top1_correlation)}",
        f"Correlation MemoryStrength vs Top3Accuracy: {fmt(result.memory_top3_correlation)}",
        "", "11. Memory vs Entropy",
        f"Correlation MemoryStrength vs EntropyGrowth: {fmt(result.memory_entropy_correlation)}",
        "", "12. Outcome Diagnostics",
        "Diagnostic only. Forward outcomes are not used in memory calculations.",
        "StateAgeNode | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew",
    ]
    for node, row in sorted(result.outcomes.items()):
        lines.append(f"{node_text(node)} | {row.count} | {fmt(row.mean_dr)} | {fmt(row.median_dr)} | {pct(row.continuation)} | {pct(row.failure)} | {pct(row.flat)} | {pct(row.skew)}")
    label, reason = recommendation([result])
    lines += [
        "", "13. Information Decay Recommendation", f"Instrument-only diagnostic: {label}", reason,
        "The aggregate report applies the cross-instrument summary.",
        "", "14. Low-DoF Audit", "Variables used:", "StructuralState", "AgeBucket",
        "TransitionProbability", "", "No Context", "No Arbitration", "No Persistence",
        "No Phase", "No Optimization", "No Fitting", "No Machine Learning",
        "No Forward Returns used in memory modeling",
        "", "15. Mechanical Research Notes",
        "- Memory is measured from fixed-horizon State+Age transition forecasts.",
        "- InformationHalfLife is the first fixed horizon at or below half of t+1 Top1 accuracy.",
        "- MemoryStrength is normalized trapezoidal area under the fixed accuracy decay curve.",
        "- Entropy growth is measured from propagated structural forecast distributions.",
        "- Outcome diagnostics remain separate from memory modeling.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_nodes(results: list[StudyResult]) -> dict[Node, list[Decay]]:
    nodes = sorted({node for result in results for node in result.decay})
    return {node: [result.decay[node] for result in results if node in result.decay] for node in nodes}


def aggregate_state_memory(results: list[StudyResult]) -> dict[str, GroupMemory]:
    return {
        state: group_memory(state, (row for result in results for node, row in result.decay.items() if node[0] == state))
        for state in STRUCTURAL_STATES
        if any(node[0] == state for result in results for node in result.decay)
    }


def aggregate_zone_memory(results: list[StudyResult]) -> dict[str, GroupMemory]:
    return {
        zone: group_memory(zone, (row for result in results for node, row in result.decay.items() if age_zone(node[1]) == zone))
        for zone in ("Young", "Middle", "Late")
    }


def aggregate_correlations(results: list[StudyResult]) -> tuple[float, float, float]:
    rows = [row for result in results for row in result.decay.values()]
    return (
        correlation((row.memory_strength for row in rows), (row.accuracy[1] for row in rows)),
        correlation((row.memory_strength for row in rows), (row.top3[1] for row in rows)),
        correlation((row.memory_strength for row in rows), (row.entropy_growth for row in rows)),
    )


def aggregate_outcomes(results: list[StudyResult], instruments: list[str]) -> list[str]:
    by_instrument = {result.instrument: result for result in results}
    nodes = sorted({node for result in results for node in result.outcomes})
    lines = ["", "Aggregate Outcome Table", "StateAgeNode | " + " | ".join(f"Count_{instrument} | Skew_{instrument} | MeanDR_{instrument}" for instrument in instruments) + " | ValidInstrumentCount | MeanSkew | MeanDR"]
    for node in nodes:
        values = [by_instrument[instrument].outcomes.get(node) for instrument in instruments]
        valid = [row for row in values if row and row.count]
        cells = [value for row in values for value in ((str(row.count), pct(row.skew), fmt(row.mean_dr)) if row else ("0", "N/A", "N/A"))]
        lines.append(f"{node_text(node)} | " + " | ".join(cells) + f" | {len(valid)} | {pct(mean(row.skew for row in valid))} | {fmt(mean(row.mean_dr for row in valid))}")
    return lines


def write_aggregate(results: list[StudyResult], out_root: Path) -> None:
    path = out_root / "InformationDecay" / "InformationDecay_All.txt"
    ensure_dir(path.parent)
    instruments = instrument_columns(results)
    by_instrument = {result.instrument: result for result in results}
    nodes = aggregate_nodes(results)
    lines = ["APVA Information Decay Study v0.1 - Aggregate", "=" * 108, "Instruments: " + ", ".join(instruments)]
    lines += ["", "Aggregate Half-Life Table", "StateAgeNode | " + " | ".join(f"HalfLife_{instrument}" for instrument in instruments) + " | MeanHalfLife | ReplicationCount | ReplicationPercent"]
    for node, rows in nodes.items():
        values = [by_instrument[instrument].decay.get(node) for instrument in instruments]
        lines.append(f"{node_text(node)} | " + " | ".join(half_life_text(row.half_life) if row else "N/A" for row in values) + f" | {fmt(mean(row.half_life for row in rows), 2)} | {len(rows)} | {pct(len(rows) / len(instruments))}")
    lines += ["", "Aggregate Memory Strength Table", "StateAgeNode | " + " | ".join(f"MemoryStrength_{instrument}" for instrument in instruments) + " | MeanMemoryStrength"]
    for node, rows in nodes.items():
        values = [by_instrument[instrument].decay.get(node) for instrument in instruments]
        lines.append(f"{node_text(node)} | " + " | ".join(fmt(row.memory_strength) if row else "N/A" for row in values) + f" | {fmt(mean(row.memory_strength for row in rows))}")
    lines += ["", "Aggregate Entropy Growth Table", "StateAgeNode | " + " | ".join(f"EntropySlope_{instrument}" for instrument in instruments) + " | MeanEntropySlope"]
    for node, rows in nodes.items():
        values = [by_instrument[instrument].decay.get(node) for instrument in instruments]
        lines.append(f"{node_text(node)} | " + " | ".join(fmt(row.entropy_slope) if row else "N/A" for row in values) + f" | {fmt(mean(row.entropy_slope for row in rows))}")
    states = aggregate_state_memory(results)
    lines += ["", "Aggregate State Memory Table", "State | MeanHalfLife | MeanMemoryStrength | MeanEntropyGrowth | ReplicationCount"]
    for state, row in states.items():
        replication = sum(state in result.state_memory for result in results)
        lines.append(f"{state} | {fmt(row.half_life, 2)} | {fmt(row.memory_strength)} | {fmt(row.entropy_growth)} | {replication}")
    lines += ["", "Aggregate Age Effect Table", "State | AgeBucket | MeanHalfLife | MeanMemoryStrength | MeanEntropyGrowth"]
    for node, rows in nodes.items():
        lines.append(f"{node[0]} | {node[1]} | {fmt(mean(row.half_life for row in rows), 2)} | {fmt(mean(row.memory_strength for row in rows))} | {fmt(mean(row.entropy_growth for row in rows))}")
    zones = aggregate_zone_memory(results)
    lines += ["", "Aggregate Young/Middle/Late Table", "Zone | MeanHalfLife | MeanMemoryStrength | MeanEntropyGrowth"]
    for zone, row in zones.items():
        lines.append(f"{zone} | {fmt(row.half_life, 2)} | {fmt(row.memory_strength)} | {fmt(row.entropy_growth)}")
    top1, top3, entropy = aggregate_correlations(results)
    lines += [
        "", "Aggregate Correlation Table", "Metric | Correlation",
        f"MemoryStrength vs Top1Accuracy | {fmt(top1)}",
        f"MemoryStrength vs Top3Accuracy | {fmt(top3)}",
        f"MemoryStrength vs EntropyGrowth | {fmt(entropy)}",
    ]
    lines += aggregate_outcomes(results, instruments)
    label, reason = recommendation(results)
    lines += [
        "", "Aggregate Information Decay Recommendation", f"Classification: {label}",
        f"Reason: {reason}", "", "Aggregate Rankings", "", "1. Longest-memory nodes",
    ]
    for node, rows in sorted(nodes.items(), key=lambda item: (-mean(row.half_life for row in item[1]), item[0]))[:20]:
        lines.append(f"{node_text(node)} | MeanHalfLife={fmt(mean(row.half_life for row in rows), 2)} | ReplicationCount={len(rows)}")
    lines += ["", "2. Shortest-memory nodes"]
    for node, rows in sorted(nodes.items(), key=lambda item: (mean(row.half_life for row in item[1]), item[0]))[:20]:
        lines.append(f"{node_text(node)} | MeanHalfLife={fmt(mean(row.half_life for row in rows), 2)} | ReplicationCount={len(rows)}")
    lines += ["", "3. Strongest-memory states"]
    for state, row in sorted(states.items(), key=lambda item: -item[1].memory_strength):
        lines.append(f"{state} | MeanMemoryStrength={fmt(row.memory_strength)} | MeanHalfLife={fmt(row.half_life, 2)}")
    lines += ["", "4. Weakest-memory states"]
    for state, row in sorted(states.items(), key=lambda item: item[1].memory_strength):
        lines.append(f"{state} | MeanMemoryStrength={fmt(row.memory_strength)} | MeanHalfLife={fmt(row.half_life, 2)}")
    lines += ["", "5. Fastest entropy growth"]
    for node, rows in sorted(nodes.items(), key=lambda item: -mean(row.entropy_growth for row in item[1]))[:20]:
        lines.append(f"{node_text(node)} | MeanEntropyGrowth={fmt(mean(row.entropy_growth for row in rows))}")
    lines += ["", "6. Slowest entropy growth"]
    for node, rows in sorted(nodes.items(), key=lambda item: mean(row.entropy_growth for row in item[1]))[:20]:
        lines.append(f"{node_text(node)} | MeanEntropyGrowth={fmt(mean(row.entropy_growth for row in rows))}")
    lines += ["", "7. Strongest age effects"]
    for state in STRUCTURAL_STATES:
        rows = [row for node, values in nodes.items() if node[0] == state for row in values]
        if rows:
            spread = max(row.memory_strength for row in rows) - min(row.memory_strength for row in rows)
            lines.append(f"{state} | MemoryStrengthSpread={fmt(spread)}")
    lines += ["", "8. Best Young/Middle/Late model"]
    for zone, row in sorted(zones.items(), key=lambda item: -item[1].memory_strength):
        lines.append(f"{zone} | MeanMemoryStrength={fmt(row.memory_strength)} | MeanHalfLife={fmt(row.half_life, 2)}")
    lines += ["", "9. Most replicated memory patterns"]
    for node, rows in sorted(nodes.items(), key=lambda item: (-len(item[1]), item[0]))[:30]:
        lines.append(f"{node_text(node)} | ReplicationCount={len(rows)} | MeanMemoryStrength={fmt(mean(row.memory_strength for row in rows))}")
    lines += [
        "", "10. Recommended APVA memory model", f"State+Age+TransitionMemory: {reason}",
        "", "Low-DoF Audit", "Variables used:", "StructuralState", "AgeBucket",
        "TransitionProbability", "", "No Context", "No Arbitration", "No Persistence",
        "No Phase", "No Optimization", "No Fitting", "No Machine Learning",
        "No Forward Returns used in memory modeling",
        "", "Research Notes",
        "- How long does structural information survive? See the half-life tables.",
        "- Which states have long or short memory? See state summaries and rankings.",
        "- Does age alter memory? See age-effect rows and strength spreads.",
        "- Do long-memory states forecast better? See memory-strength correlations.",
        "- Does entropy growth explain forecast deterioration? See entropy growth and correlation.",
        "- Can APVA be reduced to State + Age + Transition Memory? This report tests that representation mechanically.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_invariants(results: list[StudyResult]) -> None:
    for result in results:
        for destinations in result.graph.values():
            if abs(sum(destinations.values()) - 1.0) > 1e-12:
                raise RuntimeError(f"{result.instrument}: transition probabilities do not sum to one.")
        if len(result.forecasts) != sum(len(result.bars) - horizon for horizon in HORIZONS):
            raise RuntimeError(f"{result.instrument}: forecast count mismatch.")
        for row in result.forecasts:
            if abs(sum(row.distribution.values()) - 1.0) > 1e-12:
                raise RuntimeError(f"{result.instrument}: propagated mass does not sum to one.")
            if not (row.top1 <= row.top2 <= row.top3):
                raise RuntimeError(f"{result.instrument}: Top-N accuracy invariant failed.")
        for row in result.decay.values():
            if not 0.0 <= row.memory_strength <= 1.0:
                raise RuntimeError(f"{result.instrument}: normalized memory strength escaped [0, 1].")
            if row.half_life not in (*HORIZONS, HORIZONS[-1] + 1):
                raise RuntimeError(f"{result.instrument}: half-life escaped fixed horizons.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Evidence CSV files or directories")
    parser.add_argument("--out-root", default="Evidence/Output", help="Output root directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [study(result) for result in load_results(args.inputs)]
    if not results:
        raise SystemExit("No APVA evidence CSV files found.")
    validate_invariants(results)
    out_root = Path(args.out_root)
    for result in results:
        write_per_instrument(result, out_root)
    write_aggregate(results, out_root)
    print(f"Wrote {len(results)} per-instrument report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
