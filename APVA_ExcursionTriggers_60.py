#!/usr/bin/env python3
"""APVA Excursion Trigger Study v0.1.

Measure structural conditions preceding departure from fixed Study 58 Core
nodes. Forward outcomes are diagnostics only.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from APVA_InformationDecay_57 import Outcome, study as decay_study
from APVA_MinimalEngine_52 import instrument_columns, load_results, safe_mean
from APVA_NodeImportance_58 import NodeRow, aggregate_rows, local_rows, normalize, score_rows
from APVA_ProcessGraph_54 import Node, node_for, node_text
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

HORIZONS = (1, 2, 3, 5)
DEPTHS = (2, 3, 4, 5)
AGE_BUCKETS = ("1", "2", "3", "4", "5", "6-10", "11-20", "21+")
AGE_VALUE = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0, "6-10": 8.0, "11-20": 15.5, "21+": 21.0}


@dataclass
class TriggerRow:
    node: Node
    count: int
    probabilities: dict[int, float]
    bars_to_excursion: list[int]
    mean_bars: float
    median_bars: float
    minimum_bars: int
    maximum_bars: int
    memory_strength: float
    half_life: float
    entropy_growth: float
    importance: float
    replication_count: int
    replication_percent: float
    hazard_slope: float = 0.0
    stability: float = 0.0
    instability: float = 0.0


@dataclass
class SequenceRow:
    nodes: tuple[Node, ...]
    depth: int
    count: int
    probability: float


@dataclass
class TriggerResult:
    instrument: str
    source_paths: list
    bars: list
    nodes: list[Node]
    categories: list[str]
    node_rows: dict[Node, NodeRow]
    core_rows: dict[Node, TriggerRow]
    hazard: dict[str, dict[str, float]]
    hazard_slope: dict[str, float]
    hazard_correlation: dict[str, float]
    sequences: dict[tuple[Node, ...], SequenceRow]
    memory_probability_correlation: float
    half_life_probability_correlation: float
    entropy_probability_correlation: float
    entropy_time_correlation: float
    outcomes: dict[Node, Outcome]


def mean(values: Iterable[float]) -> float:
    return safe_mean(list(values))


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def correlation(left: Iterable[float], right: Iterable[float]) -> float:
    pairs = list(zip(left, right))
    if len(pairs) < 2:
        return 0.0
    xs, ys = zip(*pairs)
    x_mean, y_mean = mean(xs), mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    denominator = math.sqrt(sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys))
    return numerator / denominator if denominator else 0.0


def linear_slope(points: Iterable[tuple[float, float]]) -> float:
    points = list(points)
    if len(points) < 2:
        return 0.0
    xs, ys = zip(*points)
    x_mean, y_mean = mean(xs), mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator if denominator else 0.0


def leaves_within(categories: list[str], index: int, horizon: int) -> bool:
    return any(category != "Core Node" for category in categories[index + 1:index + horizon + 1])


def first_leave(categories: list[str], index: int) -> int | None:
    for future in range(index + 1, len(categories)):
        if categories[future] != "Core Node":
            return future - index
    return None


def build_hazard(nodes: list[Node], categories: list[str]) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, float]]:
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    leaves: dict[str, Counter[str]] = defaultdict(Counter)
    for index, (node, category) in enumerate(zip(nodes[:-1], categories[:-1])):
        if category != "Core Node":
            continue
        state, age = node
        totals[state][age] += 1
        if categories[index + 1] != "Core Node":
            leaves[state][age] += 1
    curves = {
        state: {age: leaves[state][age] / totals[state][age] if totals[state][age] else 0.0 for age in AGE_BUCKETS}
        for state in sorted(totals)
    }
    slopes, correlations = {}, {}
    for state, curve in curves.items():
        points = [(AGE_VALUE[age], curve[age]) for age in AGE_BUCKETS if totals[state][age]]
        slopes[state] = linear_slope(points)
        correlations[state] = correlation((x for x, _ in points), (y for _, y in points))
    return curves, slopes, correlations


def build_sequences(nodes: list[Node], categories: list[str]) -> dict[tuple[Node, ...], SequenceRow]:
    counts: Counter[tuple[Node, ...]] = Counter()
    totals: Counter[int] = Counter()
    for index in range(len(categories) - 1):
        if categories[index] != "Core Node" or categories[index + 1] == "Core Node":
            continue
        for depth in DEPTHS:
            if index + 1 < depth:
                continue
            sequence = tuple(nodes[index - depth + 1:index + 1])
            counts[sequence] += 1
            totals[depth] += 1
    return {nodes: SequenceRow(nodes, len(nodes), count, count / totals[len(nodes)]) for nodes, count in counts.items()}


def build_rows(nodes: list[Node], categories: list[str], node_rows: dict[Node, NodeRow], hazard_slope: dict[str, float]) -> dict[Node, TriggerRow]:
    core_nodes = sorted(node for node, row in node_rows.items() if row.category == "Core Node")
    occurrences: dict[Node, list[int]] = {node: [] for node in core_nodes}
    for index, (node, category) in enumerate(zip(nodes, categories)):
        if category == "Core Node":
            occurrences[node].append(index)
    rows = {}
    for node in core_nodes:
        indexes = occurrences[node]
        probabilities = {}
        for horizon in HORIZONS:
            eligible = [index for index in indexes if index + horizon < len(categories)]
            probabilities[horizon] = mean(leaves_within(categories, index, horizon) for index in eligible)
        times = [value for index in indexes if (value := first_leave(categories, index)) is not None]
        source = node_rows[node]
        rows[node] = TriggerRow(
            node, len(indexes), probabilities, times, mean(times), median(times),
            min(times) if times else 0, max(times) if times else 0,
            source.memory_strength, source.half_life, source.entropy_growth,
            source.importance, source.replication_count, source.replication_percent,
            hazard_slope.get(node[0], 0.0),
        )
    score_rows_fixed(rows)
    return rows


def score_rows_fixed(rows: dict[Node, TriggerRow]) -> None:
    inverse = normalize({node: 1.0 - row.probabilities[5] for node, row in rows.items()})
    bars = normalize({node: row.mean_bars for node, row in rows.items()})
    memory = normalize({node: row.memory_strength for node, row in rows.items()})
    life = normalize({node: row.half_life for node, row in rows.items()})
    replication = normalize({node: row.replication_percent for node, row in rows.items()})
    probability = normalize({node: row.probabilities[5] for node, row in rows.items()})
    entropy = normalize({node: row.entropy_growth for node, row in rows.items()})
    slope = normalize({node: row.hazard_slope for node, row in rows.items()})
    for node, row in rows.items():
        row.stability = mean((inverse[node], bars[node], memory[node], life[node], replication[node]))
        row.instability = mean((probability[node], entropy[node], slope[node]))


def build_result(loaded, node_rows: dict[Node, NodeRow], decay) -> TriggerResult:
    nodes = [node_for(bar) for bar in loaded.bars]
    categories = [node_rows[node].category for node in nodes]
    hazard, slopes, hazard_correlations = build_hazard(nodes, categories)
    rows = build_rows(nodes, categories, node_rows, slopes)
    values = list(rows.values())
    return TriggerResult(
        loaded.instrument, loaded.source_paths, loaded.bars, nodes, categories, node_rows, rows,
        hazard, slopes, hazard_correlations, build_sequences(nodes, categories),
        correlation((row.memory_strength for row in values), (row.probabilities[5] for row in values)),
        correlation((row.half_life for row in values), (row.probabilities[5] for row in values)),
        correlation((row.entropy_growth for row in values), (row.probabilities[5] for row in values)),
        correlation((row.entropy_growth for row in values), (row.mean_bars for row in values)),
        {node: decay.outcomes[node] for node in rows if node in decay.outcomes},
    )


def sequence_text(nodes: tuple[Node, ...]) -> str:
    return " -> ".join(node_text(node) for node in nodes)


def row_text(row: TriggerRow) -> str:
    probabilities = " | ".join(pct(row.probabilities[horizon]) for horizon in HORIZONS)
    return f"{node_text(row.node)} | {row.count} | {probabilities} | {fmt(row.mean_bars, 2)} | {fmt(row.median_bars, 2)} | {row.minimum_bars} | {row.maximum_bars}"


def recommendation(result: TriggerResult) -> list[str]:
    stable = max(result.core_rows.values(), key=lambda row: (row.stability, row.node))
    unstable = max(result.core_rows.values(), key=lambda row: (row.instability, row.node))
    top_sequence = max(result.sequences.values(), key=lambda row: (row.count, row.nodes), default=None)
    return [
        f"MostStableCoreNode: {node_text(stable.node)} | StabilityScore={fmt(stable.stability)}",
        f"MostLikelyToEscape: {node_text(unstable.node)} | InstabilityScore={fmt(unstable.instability)}",
        f"AgeEffect: MeanHazardSlope={fmt(mean(result.hazard_slope.values()))}",
        f"MemoryEffect: CorrelationMemoryStrengthVsExcursionProbability={fmt(result.memory_probability_correlation)}",
        f"EntropyEffect: CorrelationEntropyGrowthVsExcursionProbability={fmt(result.entropy_probability_correlation)}",
        f"PreExcursionSequence: {sequence_text(top_sequence.nodes) if top_sequence else 'N/A'}",
    ]


def append_audit(lines: list[str], numbered: bool = False) -> None:
    lines += ["", ("15. Low-DoF Audit" if numbered else "14. Low-DoF Audit"), "Variables used:", "StructuralState", "AgeBucket", "", "Derived metrics:", "MemoryStrength", "HalfLife", "ImportanceScore", "AttentionCategory", "", "No Context", "No Arbitration", "No Persistence", "No Phase", "No Optimization", "No Fitting", "No Machine Learning"]


def write_per_instrument(result: TriggerResult, out_root: Path) -> None:
    path = out_root / result.instrument / f"ExcursionTriggers_{result.instrument}.txt"
    ensure_dir(path.parent)
    rows = list(result.core_rows.values())
    lines = ["APVA Excursion Trigger Study v0.1", "=" * 108, "Diagnostics", f"Instrument: {result.instrument}", "Input path(s): " + ", ".join(str(item) for item in result.source_paths), f"Total rows: {len(result.bars)}", f"Core node count: {len(rows)}"]
    lines += ["", "1. Core Node Inventory", "CoreNode | Count | ReplicationCount | ReplicationPercent"]
    lines += [f"{node_text(row.node)} | {row.count} | {row.replication_count} | {pct(row.replication_percent)}" for row in sorted(rows, key=lambda row: row.node)]
    lines += ["", "2. Excursion Definition", "Excursion: CurrentNodeCategory=Core and NodeCategory!=Core within H bars.", "Horizons: 1, 2, 3, 5"]
    lines += ["", "3. Excursion Probability", "CoreNode | Count | P<=1 | P<=2 | P<=3 | P<=5 | MeanBars | MedianBars | MinBars | MaxBars"]
    lines += [row_text(row) for row in sorted(rows, key=lambda row: (-row.probabilities[5], row.node))]
    lines += ["", "4. Expected Time To Excursion"]
    lines += [row_text(row) for row in sorted(rows, key=lambda row: (-row.mean_bars, row.node))]
    lines += ["", "5. Hazard Rate", "State | Age1 | Age2 | Age3 | Age4 | Age5 | Age6-10 | Age11-20 | Age21+"]
    lines += [f"{state} | " + " | ".join(pct(curve[age]) for age in AGE_BUCKETS) for state, curve in result.hazard.items()]
    lines += ["", "6. Age Effect", "State | HazardSlope | CorrelationAgeBucketVsExcursionProbability"]
    lines += [f"{state} | {fmt(result.hazard_slope[state])} | {fmt(result.hazard_correlation[state])}" for state in sorted(result.hazard)]
    lines += ["", "7. Pre-Excursion Sequences", "Sequence | Depth | Count | Probability"]
    lines += [f"{sequence_text(row.nodes)} | {row.depth} | {row.count} | {pct(row.probability)}" for row in sorted(result.sequences.values(), key=lambda row: (-row.count, row.nodes))[:40]]
    lines += ["", "8. Memory Effect", f"MemoryStrength vs ExcursionProbability: {fmt(result.memory_probability_correlation)}", f"HalfLife vs ExcursionProbability: {fmt(result.half_life_probability_correlation)}"]
    lines += ["", "9. Entropy Effect", f"EntropyGrowth vs ExcursionProbability: {fmt(result.entropy_probability_correlation)}", f"EntropyGrowth vs BarsToExcursion: {fmt(result.entropy_time_correlation)}"]
    lines += ["", "10. Cross-Instrument Replication", "See aggregate report."]
    lines += ["", "11. Core Stability Ranking", "CoreNode | StabilityScore"]
    lines += [f"{node_text(row.node)} | {fmt(row.stability)}" for row in sorted(rows, key=lambda row: (-row.stability, row.node))]
    lines += ["", "12. Instability Ranking", "CoreNode | InstabilityScore"]
    lines += [f"{node_text(row.node)} | {fmt(row.instability)}" for row in sorted(rows, key=lambda row: (-row.instability, row.node))]
    lines += ["", "13. Outcome Diagnostics", "Diagnostic only. Outcomes are not used in trigger calculations.", "CoreNode | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew"]
    for node in sorted(result.outcomes):
        row = result.outcomes[node]
        lines.append(f"{node_text(node)} | {row.count} | {fmt(row.mean_dr)} | {fmt(row.median_dr)} | {pct(row.continuation)} | {pct(row.failure)} | {pct(row.flat)} | {fmt(row.skew)}")
    lines += ["", "14. Excursion Trigger Recommendation"] + recommendation(result)
    append_audit(lines, True)
    lines += ["", "16. Mechanical Research Notes", "- Trigger construction uses structural StateAge nodes and inherited Core membership only.", "- Forward outcomes are diagnostic only.", "- No trigger threshold is optimized."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_replication(results: list[TriggerResult], rows: dict[Node, NodeRow]) -> list[str]:
    instruments = instrument_columns(results)
    by_instrument = {result.instrument: result for result in results}
    lines = ["", "9. Cross-Instrument Replication", "CoreNode | " + " | ".join(f"P<=5_{instrument} | BarsToExcursion_{instrument}" for instrument in instruments) + " | ReplicationCount"]
    for node in sorted(node for node, row in rows.items() if row.category == "Core Node"):
        values = [by_instrument[instrument].core_rows.get(node) for instrument in instruments]
        cells = [value for row in values for value in ((pct(row.probabilities[5]), fmt(row.mean_bars, 2)) if row else ("N/A", "N/A"))]
        lines.append(f"{node_text(node)} | " + " | ".join(cells) + f" | {sum(row is not None for row in values)}")
    return lines


def write_aggregate(results: list[TriggerResult], rows: dict[Node, NodeRow], out_root: Path) -> None:
    path = out_root / "ExcursionTriggers" / "ExcursionTriggers_All.txt"
    ensure_dir(path.parent)
    instruments = instrument_columns(results)
    core_rows = [row for row in rows.values() if row.category == "Core Node"]
    combined = build_combined_core_rows(results, rows)
    lines = ["APVA Excursion Trigger Study v0.1 - Aggregate", "=" * 108, "Instruments: " + ", ".join(instruments)]
    lines += ["", "1. Core Node Inventory", "CoreNode | Count | ReplicationCount | ReplicationPercent"]
    lines += [f"{node_text(row.node)} | {row.count} | {row.replication_count} | {pct(row.replication_percent)}" for row in sorted(core_rows, key=lambda row: row.node)]
    lines += ["", "2. Excursion Probability Ranking", "CoreNode | P<=1 | P<=2 | P<=3 | P<=5"]
    lines += [f"{node_text(row.node)} | " + " | ".join(pct(row.probabilities[horizon]) for horizon in HORIZONS) for row in sorted(combined.values(), key=lambda row: (-row.probabilities[5], row.node))]
    lines += ["", "3. Bars To Excursion Ranking", "CoreNode | MeanBars | MedianBars | MinBars | MaxBars"]
    lines += [f"{node_text(row.node)} | {fmt(row.mean_bars, 2)} | {fmt(row.median_bars, 2)} | {row.minimum_bars} | {row.maximum_bars}" for row in sorted(combined.values(), key=lambda row: (-row.mean_bars, row.node))]
    lines += ["", "4. Hazard Curves", "Instrument | State | Age1 | Age2 | Age3 | Age4 | Age5 | Age6-10 | Age11-20 | Age21+"]
    for result in results:
        lines += [f"{result.instrument} | {state} | " + " | ".join(pct(curve[age]) for age in AGE_BUCKETS) for state, curve in result.hazard.items()]
    lines += ["", "5. Age Effects", "Instrument | State | HazardSlope | Correlation"]
    for result in results:
        lines += [f"{result.instrument} | {state} | {fmt(result.hazard_slope[state])} | {fmt(result.hazard_correlation[state])}" for state in sorted(result.hazard)]
    lines += ["", "6. Pre-Excursion Sequences", "Sequence | " + " | ".join(f"Count_{instrument}" for instrument in instruments) + " | ReplicationCount | MeanProbability"]
    lines += aggregate_sequence_lines(results, instruments)
    lines += ["", "7. Memory Effects", "Instrument | MemoryStrengthVsExcursionProbability | HalfLifeVsExcursionProbability"]
    lines += [f"{result.instrument} | {fmt(result.memory_probability_correlation)} | {fmt(result.half_life_probability_correlation)}" for result in results]
    lines += ["", "8. Entropy Effects", "Instrument | EntropyGrowthVsExcursionProbability | EntropyGrowthVsBarsToExcursion"]
    lines += [f"{result.instrument} | {fmt(result.entropy_probability_correlation)} | {fmt(result.entropy_time_correlation)}" for result in results]
    lines += aggregate_replication(results, rows)
    lines += ["", "10. Stability Ranking", "CoreNode | StabilityScore"]
    lines += [f"{node_text(row.node)} | {fmt(row.stability)}" for row in sorted(combined.values(), key=lambda row: (-row.stability, row.node))]
    lines += ["", "11. Instability Ranking", "CoreNode | InstabilityScore"]
    lines += [f"{node_text(row.node)} | {fmt(row.instability)}" for row in sorted(combined.values(), key=lambda row: (-row.instability, row.node))]
    lines += ["", "12. Outcome Diagnostics", "Diagnostic only. Outcomes are not used in trigger calculations.", "Instrument | CoreNode | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew"]
    for result in results:
        for node, row in sorted(result.outcomes.items()):
            lines.append(f"{result.instrument} | {node_text(node)} | {row.count} | {fmt(row.mean_dr)} | {fmt(row.median_dr)} | {pct(row.continuation)} | {pct(row.failure)} | {pct(row.flat)} | {fmt(row.skew)}")
    lines += ["", "13. Recommendation"]
    lines += aggregate_recommendation(results, combined)
    append_audit(lines)
    lines += ["", "Research Notes", "- What causes departures from Core? See excursion probabilities, hazard curves, and recurring sequences.", "- Are excursions predictable? See replicated sequence and instability rankings.", "- Does age influence escape probability? See fixed age-bucket hazard slopes.", "- Does memory suppress excursions? See fixed correlation diagnostics.", "- Do recurring pre-excursion patterns exist? See replicated sequences.", "- State+Age+Flow+ExcursionRisk is evaluated without adding APVA variables."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_combined_core_rows(results: list[TriggerResult], rows: dict[Node, NodeRow]) -> dict[Node, TriggerRow]:
    combined = {}
    for node, source in rows.items():
        if source.category != "Core Node":
            continue
        values = [result.core_rows[node] for result in results if node in result.core_rows]
        count = sum(row.count for row in values)
        probabilities = {horizon: sum(row.probabilities[horizon] * row.count for row in values) / count if count else 0.0 for horizon in HORIZONS}
        times = [value for row in values for value in row.bars_to_excursion]
        combined[node] = TriggerRow(node, count, probabilities, times, mean(times), median(times), min(times) if times else 0, max(times) if times else 0, source.memory_strength, source.half_life, source.entropy_growth, source.importance, source.replication_count, source.replication_percent, mean(row.hazard_slope for row in values))
    score_rows_fixed(combined)
    return combined


def aggregate_sequence_lines(results: list[TriggerResult], instruments: list[str]) -> list[str]:
    sequences = sorted({sequence for result in results for sequence in result.sequences})
    lines = []
    for sequence in sequences:
        values = [result.sequences.get(sequence) for result in results]
        if not any(values):
            continue
        counts = [str(row.count) if row else "0" for row in values]
        valid = [row for row in values if row]
        lines.append((sum(row.count for row in valid), f"{sequence_text(sequence)} | " + " | ".join(counts) + f" | {len(valid)} | {pct(mean(row.probability for row in valid))}"))
    return [line for _, line in sorted(lines, key=lambda item: (-item[0], item[1]))[:60]]


def aggregate_recommendation(results: list[TriggerResult], combined: dict[Node, TriggerRow]) -> list[str]:
    stable = max(combined.values(), key=lambda row: (row.stability, row.node))
    unstable = max(combined.values(), key=lambda row: (row.instability, row.node))
    return [
        f"MostStableCoreNode: {node_text(stable.node)} | StabilityScore={fmt(stable.stability)}",
        f"MostLikelyToEscape: {node_text(unstable.node)} | InstabilityScore={fmt(unstable.instability)}",
        f"AgeEffect: MeanHazardSlope={fmt(mean(value for result in results for value in result.hazard_slope.values()))}",
        f"MemoryEffect: MeanCorrelation={fmt(mean(result.memory_probability_correlation for result in results))}",
        f"EntropyEffect: MeanCorrelation={fmt(mean(result.entropy_probability_correlation for result in results))}",
        f"CrossInstrumentReplication: {len(results)} instruments evaluated.",
    ]


def validate_invariants(results: list[TriggerResult]) -> None:
    for result in results:
        if len(result.nodes) != len(result.categories) or len(result.nodes) != len(result.bars):
            raise RuntimeError(f"{result.instrument}: stream length mismatch.")
        if any(category == "Core Node" and node not in result.core_rows for node, category in zip(result.nodes, result.categories)):
            raise RuntimeError(f"{result.instrument}: Core occurrence missing trigger row.")
        for row in result.core_rows.values():
            probabilities = [row.probabilities[horizon] for horizon in HORIZONS]
            if probabilities != sorted(probabilities):
                raise RuntimeError(f"{result.instrument} {node_text(row.node)}: excursion probabilities are not monotonic.")
            if not all(0.0 <= value <= 1.0 for value in probabilities):
                raise RuntimeError(f"{result.instrument} {node_text(row.node)}: excursion probability escaped [0, 1].")
            if not 0.0 <= row.stability <= 1.0 or not 0.0 <= row.instability <= 1.0:
                raise RuntimeError(f"{result.instrument} {node_text(row.node)}: score escaped [0, 1].")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Evidence CSV files or directories")
    parser.add_argument("--out-root", default="Evidence/Output", help="Output root directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_results(args.inputs)
    if not loaded:
        raise SystemExit("No APVA evidence CSV files found.")
    decay_results = [decay_study(result) for result in loaded]
    aggregate = aggregate_rows(decay_results)
    score_rows(aggregate)
    aggregate_results = [build_result(result, aggregate, decay) for result, decay in zip(loaded, decay_results)]
    validate_invariants(aggregate_results)
    out_root = Path(args.out_root)
    for result, decay in zip(loaded, decay_results):
        local = local_rows(decay)
        score_rows(local)
        write_per_instrument(build_result(result, local, decay), out_root)
    write_aggregate(aggregate_results, aggregate, out_root)
    print(f"Wrote {len(aggregate_results)} per-instrument report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
