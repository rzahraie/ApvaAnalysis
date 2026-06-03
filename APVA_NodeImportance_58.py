#!/usr/bin/env python3
"""APVA Node Importance Study v0.1.

Rank established StructuralState + AgeBucket nodes using fixed, equal-weight
structural forecast and memory metrics. Forward outcomes are diagnostics only.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from APVA_InformationDecay_57 import Decay, Outcome, StudyResult as DecayResult, study as decay_study, validate_invariants as validate_decay
from APVA_MinimalEngine_52 import age_zone, instrument_columns, load_results, safe_mean
from APVA_ProcessGraph_54 import node_text
from APVA_StructuralLifeCycle_44 import STRUCTURAL_STATES, ensure_dir, fmt, pct

Node = tuple[str, str]
COVERAGE_TARGETS = (0.50, 0.75, 0.90)


@dataclass
class NodeRow:
    node: Node
    count: int
    replication_count: int
    replication_percent: float
    top1: float
    top3: float
    confidence: float
    memory_strength: float
    half_life: float
    entropy_growth: float
    entropy_slope: float
    importance: float = 0.0
    reliability: float = 0.0
    sample_class: str = ""
    category: str = ""
    reason: str = ""


@dataclass
class GroupRow:
    name: str
    count: int
    importance: float
    reliability: float
    memory_strength: float
    half_life: float
    replication: float


@dataclass
class Coverage:
    target: float
    node_count: int
    coverage: float
    reduction: float


@dataclass
class ImportanceResult:
    instrument: str
    source_paths: list
    bars: list
    decay_result: DecayResult
    rows: dict[Node, NodeRow]
    states: dict[str, GroupRow]
    zones: dict[str, GroupRow]
    coverage: dict[float, Coverage]


def mean(values: Iterable[float]) -> float:
    return safe_mean(list(values))


def weighted_mean(rows: Iterable[tuple[float, int]]) -> float:
    rows = list(rows)
    total = sum(weight for _, weight in rows)
    return sum(value * weight for value, weight in rows) / total if total else 0.0


def normalize(values: dict[Node, float]) -> dict[Node, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if math.isclose(low, high):
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def percentile(values: Iterable[float], probability: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * probability
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return values[left]
    fraction = position - left
    return values[left] * (1.0 - fraction) + values[right] * fraction


def sample_class(count: int) -> str:
    if count >= 1000:
        return "LargeSample"
    if count >= 250:
        return "MediumSample"
    if count >= 50:
        return "SmallSample"
    return "Insufficient"


def confidence_by_node(result: DecayResult) -> dict[Node, float]:
    return {
        node: mean(row.confidence for row in result.forecasts if row.horizon == 1 and row.current == node)
        for node in result.decay
    }


def local_rows(result: DecayResult) -> dict[Node, NodeRow]:
    confidence = confidence_by_node(result)
    return {
        node: NodeRow(
            node, row.count, 1, 1.0, row.accuracy[1], row.top3[1],
            confidence[node], row.memory_strength, row.half_life,
            row.entropy_growth, row.entropy_slope,
        )
        for node, row in result.decay.items()
    }


def aggregate_rows(results: list[DecayResult]) -> dict[Node, NodeRow]:
    nodes = sorted({node for result in results for node in result.decay})
    confidences = {result.instrument: confidence_by_node(result) for result in results}
    rows = {}
    for node in nodes:
        values = [(result, result.decay[node]) for result in results if node in result.decay]
        count = sum(row.count for _, row in values)
        rows[node] = NodeRow(
            node, count, len(values), len(values) / len(results),
            weighted_mean((row.accuracy[1], row.count) for _, row in values),
            weighted_mean((row.top3[1], row.count) for _, row in values),
            weighted_mean((confidences[result.instrument][node], row.count) for result, row in values),
            weighted_mean((row.memory_strength, row.count) for _, row in values),
            weighted_mean((row.half_life, row.count) for _, row in values),
            weighted_mean((row.entropy_growth, row.count) for _, row in values),
            weighted_mean((row.entropy_slope, row.count) for _, row in values),
        )
    return rows


def score_rows(rows: dict[Node, NodeRow]) -> None:
    count = normalize({node: row.count for node, row in rows.items()})
    top1 = normalize({node: row.top1 for node, row in rows.items()})
    memory = normalize({node: row.memory_strength for node, row in rows.items()})
    life = normalize({node: row.half_life for node, row in rows.items()})
    replication = normalize({node: row.replication_percent for node, row in rows.items()})
    confidence = normalize({node: row.confidence for node, row in rows.items()})
    for node, row in rows.items():
        row.importance = mean((count[node], top1[node], memory[node], life[node], replication[node]))
        row.reliability = mean((count[node], replication[node], confidence[node]))
        row.sample_class = sample_class(row.count)
    importance_80 = percentile((row.importance for row in rows.values()), 0.80)
    reliability_80 = percentile((row.reliability for row in rows.values()), 0.80)
    importance_50 = percentile((row.importance for row in rows.values()), 0.50)
    top1_median = percentile((row.top1 for row in rows.values()), 0.50)
    entropy_median = percentile((row.entropy_growth for row in rows.values()), 0.50)
    for row in rows.values():
        if row.count < 50:
            row.category = "Insufficient Node"
            row.reason = "Count below 50."
        elif row.importance >= importance_80 and row.reliability >= reliability_80:
            row.category = "Core Node"
            row.reason = "ImportanceScore and ReliabilityScore at or above 80th percentile."
        elif row.importance >= importance_50:
            row.category = "Secondary Node"
            row.reason = "ImportanceScore at or above 50th percentile."
        elif row.top1 < top1_median and row.entropy_growth > entropy_median and row.replication_count >= 2:
            row.category = "Transition Node"
            row.reason = "Below-median Top1Accuracy, above-median EntropyGrowth, replicated in at least 2 instruments."
        else:
            row.category = "Weak Node"
            row.reason = "Does not meet fixed Core, Secondary, Transition, or Insufficient criteria."


def group_row(name: str, rows: Iterable[NodeRow]) -> GroupRow:
    rows = list(rows)
    return GroupRow(
        name, sum(row.count for row in rows), mean(row.importance for row in rows),
        mean(row.reliability for row in rows), mean(row.memory_strength for row in rows),
        mean(row.half_life for row in rows), mean(row.replication_percent for row in rows),
    )


def state_rows(rows: dict[Node, NodeRow]) -> dict[str, GroupRow]:
    return {
        state: group_row(state, (row for node, row in rows.items() if node[0] == state))
        for state in STRUCTURAL_STATES
        if any(node[0] == state for node in rows)
    }


def zone_rows(rows: dict[Node, NodeRow]) -> dict[str, GroupRow]:
    return {
        zone: group_row(zone, (row for node, row in rows.items() if age_zone(node[1]) == zone))
        for zone in ("Young", "Middle", "Late")
        if any(age_zone(node[1]) == zone for node in rows)
    }


def coverage_rows(rows: dict[Node, NodeRow]) -> dict[float, Coverage]:
    ordered = sorted(rows.values(), key=lambda row: (-row.importance, row.node))
    total = sum(row.importance for row in ordered)
    output = {}
    for target in COVERAGE_TARGETS:
        retained = 0.0
        node_count = 0
        for row in ordered:
            retained += row.importance
            node_count += 1
            if total and retained / total >= target:
                break
        coverage = retained / total if total else 0.0
        output[target] = Coverage(target, node_count, coverage, 1.0 - node_count / len(ordered) if ordered else 0.0)
    return output


def build_result(result, rows: dict[Node, NodeRow] | None = None) -> ImportanceResult:
    decay = decay_study(result)
    rows = rows or local_rows(decay)
    score_rows(rows)
    return ImportanceResult(
        decay.instrument, decay.source_paths, decay.bars, decay, rows,
        state_rows(rows), zone_rows(rows), coverage_rows(rows),
    )


def result_from_decay(decay: DecayResult, rows: dict[Node, NodeRow]) -> ImportanceResult:
    score_rows(rows)
    return ImportanceResult(
        decay.instrument, decay.source_paths, decay.bars, decay, rows,
        state_rows(rows), zone_rows(rows), coverage_rows(rows),
    )


def render_node_table(lines: list[str], title: str, rows: Iterable[NodeRow]) -> None:
    lines += ["", title, "StateAgeNode | Count | ReplicationCount | ReplicationPercent | Top1Accuracy | Top3Accuracy | ForecastConfidence | MemoryStrength | InformationHalfLife | EntropyGrowth | EntropySlope | ImportanceScore | ReliabilityScore | SampleClass | AttentionCategory"]
    for row in rows:
        lines.append(f"{node_text(row.node)} | {row.count} | {row.replication_count} | {pct(row.replication_percent)} | {pct(row.top1)} | {pct(row.top3)} | {pct(row.confidence)} | {fmt(row.memory_strength)} | {fmt(row.half_life, 2)} | {fmt(row.entropy_growth)} | {fmt(row.entropy_slope)} | {fmt(row.importance)} | {fmt(row.reliability)} | {row.sample_class} | {row.category}")


def reduction_lines(coverage: dict[float, Coverage]) -> list[str]:
    lines = ["CoverageTarget | NodeCount | CoveragePercent | ReductionPercent"]
    for target in COVERAGE_TARGETS:
        row = coverage[target]
        lines.append(f"{pct(target)} | {row.node_count} | {pct(row.coverage)} | {pct(row.reduction)}")
    return lines


def write_per_instrument(result: ImportanceResult, out_root: Path) -> None:
    path = out_root / result.instrument / f"NodeImportance_{result.instrument}.txt"
    ensure_dir(path.parent)
    rows = list(result.rows.values())
    lines = [
        "APVA Node Importance Study v0.1", "=" * 104, "Diagnostics",
        f"Instrument: {result.instrument}",
        "Input path(s): " + ", ".join(str(item) for item in result.source_paths),
        f"Total rows: {len(result.bars)}", f"Node count: {len(rows)}",
    ]
    render_node_table(lines, "1. Node Inventory", sorted(rows, key=lambda row: row.node))
    lines += ["", "2. Forecastability Ranking"]
    for title, ordered in (
        ("Highest Top1Accuracy", sorted(rows, key=lambda row: (-row.top1, row.node))),
        ("Highest Top3Accuracy", sorted(rows, key=lambda row: (-row.top3, row.node))),
        ("Highest ForecastConfidence", sorted(rows, key=lambda row: (-row.confidence, row.node))),
    ):
        lines += [title, "StateAgeNode | Value"]
        metric = {"Highest Top1Accuracy": "top1", "Highest Top3Accuracy": "top3", "Highest ForecastConfidence": "confidence"}[title]
        lines += [f"{node_text(row.node)} | {pct(getattr(row, metric))}" for row in ordered]
    lines += ["", "3. Memory Ranking", "Longest HalfLife", "StateAgeNode | InformationHalfLife"]
    lines += [f"{node_text(row.node)} | {fmt(row.half_life, 2)}" for row in sorted(rows, key=lambda row: (-row.half_life, row.node))]
    lines += ["Strongest MemoryStrength", "StateAgeNode | MemoryStrength"]
    lines += [f"{node_text(row.node)} | {fmt(row.memory_strength)}" for row in sorted(rows, key=lambda row: (-row.memory_strength, row.node))]
    lines += ["", "4. Entropy Ranking", "StateAgeNode | EntropyGrowth | EntropySlope"]
    lines += [f"{node_text(row.node)} | {fmt(row.entropy_growth)} | {fmt(row.entropy_slope)}" for row in sorted(rows, key=lambda row: (row.entropy_growth, row.node))]
    lines += ["", "5. Replication Ranking", "StateAgeNode | ReplicationCount | ReplicationPercent"]
    lines += [f"{node_text(row.node)} | {row.replication_count} | {pct(row.replication_percent)}" for row in sorted(rows, key=lambda row: (-row.replication_count, row.node))]
    lines += ["", "6. Sample Sufficiency", "StateAgeNode | Count | SampleClass"]
    lines += [f"{node_text(row.node)} | {row.count} | {row.sample_class}" for row in sorted(rows, key=lambda row: (-row.count, row.node))]
    lines += ["", "7. Importance Score", "StateAgeNode | ImportanceScore"]
    lines += [f"{node_text(row.node)} | {fmt(row.importance)}" for row in sorted(rows, key=lambda row: (-row.importance, row.node))]
    lines += ["", "8. Reliability Score", "StateAgeNode | ReliabilityScore"]
    lines += [f"{node_text(row.node)} | {fmt(row.reliability)}" for row in sorted(rows, key=lambda row: (-row.reliability, row.node))]
    lines += ["", "9. Attention Categories", "StateAgeNode | AttentionCategory | Reason"]
    lines += [f"{node_text(row.node)} | {row.category} | {row.reason}" for row in sorted(rows, key=lambda row: (row.category, row.node))]
    lines += ["", "10. State-Level Importance", "State | MeanImportance | MeanReliability | MeanMemoryStrength | MeanHalfLife | MeanReplication"]
    lines += [f"{name} | {fmt(row.importance)} | {fmt(row.reliability)} | {fmt(row.memory_strength)} | {fmt(row.half_life, 2)} | {pct(row.replication)}" for name, row in sorted(result.states.items())]
    lines += ["", "11. Young/Middle/Late Importance", "Zone | MeanImportance | MeanReliability | MeanMemoryStrength | MeanHalfLife"]
    lines += [f"{name} | {fmt(row.importance)} | {fmt(row.reliability)} | {fmt(row.memory_strength)} | {fmt(row.half_life, 2)}" for name, row in result.zones.items()]
    lines += ["", "12. Node Reduction Test"] + reduction_lines(result.coverage)
    lines += ["", "13. Outcome Diagnostics", "Diagnostic only. Forward outcomes are not used in importance scoring.", "StateAgeNode | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew"]
    for node, outcome in sorted(result.decay_result.outcomes.items()):
        lines.append(f"{node_text(node)} | {outcome.count} | {fmt(outcome.mean_dr)} | {fmt(outcome.median_dr)} | {pct(outcome.continuation)} | {pct(outcome.failure)} | {pct(outcome.flat)} | {pct(outcome.skew)}")
    core, extended = result.coverage[0.75], result.coverage[0.90]
    lines += [
        "", "14. Recommendation", f"CoreUniverseSize: {core.node_count}",
        f"ExtendedUniverseSize: {extended.node_count}",
        f"ReductionPercent: {pct(core.reduction)} for 75% importance coverage.",
        "", "15. Low-DoF Audit", "Variables used:", "StructuralState", "AgeBucket",
        "", "Derived metrics:", "Count", "Top1Accuracy", "MemoryStrength", "HalfLife",
        "ReplicationPercent", "", "No Context", "No Arbitration", "No Persistence",
        "No Phase", "No Optimization", "No Fitting", "No Machine Learning",
        "No Forward Returns used in node ranking",
        "", "16. Mechanical Research Notes",
        "- ImportanceScore is an equal-weight mean of five min-max normalized structural metrics.",
        "- ReliabilityScore is an equal-weight mean of three min-max normalized structural metrics.",
        "- Attention categories use the fixed percentile and count rules only.",
        "- Outcome diagnostics remain separate from node ranking.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_group_rows(rows: dict[Node, NodeRow]) -> tuple[dict[str, GroupRow], dict[str, GroupRow]]:
    return state_rows(rows), zone_rows(rows)


def outcome_lines(results: list[DecayResult], instruments: list[str]) -> list[str]:
    by_instrument = {result.instrument: result for result in results}
    nodes = sorted({node for result in results for node in result.outcomes})
    lines = ["", "Aggregate Outcome Table", "Node | " + " | ".join(f"Count_{instrument} | MeanDR_{instrument}" for instrument in instruments) + " | ValidInstrumentCount"]
    for node in nodes:
        values = [by_instrument[instrument].outcomes.get(node) for instrument in instruments]
        valid = [row for row in values if row and row.count]
        cells = [value for row in values for value in ((str(row.count), fmt(row.mean_dr)) if row else ("0", "N/A"))]
        lines.append(f"{node_text(node)} | " + " | ".join(cells) + f" | {len(valid)}")
    return lines


def write_aggregate(results: list[DecayResult], rows: dict[Node, NodeRow], out_root: Path) -> None:
    path = out_root / "NodeImportance" / "NodeImportance_All.txt"
    ensure_dir(path.parent)
    instruments = instrument_columns(results)
    by_instrument = {result.instrument: result for result in results}
    states, zones = aggregate_group_rows(rows)
    coverage = coverage_rows(rows)
    lines = ["APVA Node Importance Study v0.1 - Aggregate", "=" * 108, "Instruments: " + ", ".join(instruments)]
    lines += ["", "Aggregate Node Table", "Node | " + " | ".join(f"Count_{instrument}" for instrument in instruments) + " | ReplicationCount | ReplicationPercent | Top1Accuracy | Top3Accuracy | ForecastConfidence | MemoryStrength | HalfLife | EntropyGrowth | ImportanceScore | ReliabilityScore | SampleClass | AttentionCategory"]
    for node, row in sorted(rows.items(), key=lambda item: (-item[1].importance, item[0])):
        counts = [str(by_instrument[instrument].decay[node].count) if node in by_instrument[instrument].decay else "0" for instrument in instruments]
        lines.append(f"{node_text(node)} | " + " | ".join(counts) + f" | {row.replication_count} | {pct(row.replication_percent)} | {pct(row.top1)} | {pct(row.top3)} | {pct(row.confidence)} | {fmt(row.memory_strength)} | {fmt(row.half_life, 2)} | {fmt(row.entropy_growth)} | {fmt(row.importance)} | {fmt(row.reliability)} | {row.sample_class} | {row.category}")
    for title, category, header, renderer in (
        ("Aggregate Core Node Table", "Core Node", "Node | ImportanceScore | ReliabilityScore | Count | ReplicationCount | Reason", lambda row: f"{node_text(row.node)} | {fmt(row.importance)} | {fmt(row.reliability)} | {row.count} | {row.replication_count} | {row.reason}"),
        ("Aggregate Secondary Node Table", "Secondary Node", "Node | ImportanceScore | ReliabilityScore", lambda row: f"{node_text(row.node)} | {fmt(row.importance)} | {fmt(row.reliability)}"),
        ("Aggregate Transition Node Table", "Transition Node", "Node | EntropyGrowth | Top1Accuracy | ReplicationCount", lambda row: f"{node_text(row.node)} | {fmt(row.entropy_growth)} | {pct(row.top1)} | {row.replication_count}"),
        ("Aggregate Insufficient Node Table", "Insufficient Node", "Node | Count | ReplicationCount", lambda row: f"{node_text(row.node)} | {row.count} | {row.replication_count}"),
    ):
        lines += ["", title, header]
        selected = sorted((row for row in rows.values() if row.category == category), key=lambda row: (-row.importance, row.node))
        lines += [renderer(row) for row in selected] or ["None"]
    lines += ["", "Aggregate State Table", "State | MeanImportance | MeanReliability | MeanMemoryStrength | MeanHalfLife | MeanReplication"]
    lines += [f"{name} | {fmt(row.importance)} | {fmt(row.reliability)} | {fmt(row.memory_strength)} | {fmt(row.half_life, 2)} | {pct(row.replication)}" for name, row in sorted(states.items(), key=lambda item: -item[1].importance)]
    lines += ["", "Aggregate Young/Middle/Late Table", "Zone | MeanImportance | MeanReliability | MeanMemoryStrength | MeanHalfLife"]
    lines += [f"{name} | {fmt(row.importance)} | {fmt(row.reliability)} | {fmt(row.memory_strength)} | {fmt(row.half_life, 2)}" for name, row in sorted(zones.items(), key=lambda item: -item[1].importance)]
    lines += ["", "Aggregate Reduction Table"] + reduction_lines(coverage)
    lines += outcome_lines(results, instruments)
    core, extended = coverage[0.75], coverage[0.90]
    most_important = sorted(rows.values(), key=lambda row: (-row.importance, row.node))[:10]
    most_reliable = sorted(rows.values(), key=lambda row: (-row.reliability, row.node))[:10]
    transitional = sorted((row for row in rows.values() if row.category == "Transition Node"), key=lambda row: (-row.entropy_growth, row.node))
    lines += [
        "", "Aggregate Recommendation", f"CoreUniverseSize: {core.node_count}",
        f"ExtendedUniverseSize: {extended.node_count}",
        f"ReductionPercent: {pct(core.reduction)} for 75% importance coverage.",
        "MostImportantNodes: " + ", ".join(node_text(row.node) for row in most_important),
        "MostReliableNodes: " + ", ".join(node_text(row.node) for row in most_reliable),
        "MostTransitionalNodes: " + (", ".join(node_text(row.node) for row in transitional[:10]) or "None"),
        "", "Aggregate Rankings", "", "1. Most important nodes",
    ]
    lines += [f"{node_text(row.node)} | ImportanceScore={fmt(row.importance)}" for row in most_important]
    lines += ["", "2. Most reliable nodes"]
    lines += [f"{node_text(row.node)} | ReliabilityScore={fmt(row.reliability)}" for row in most_reliable]
    lines += ["", "3. Longest-memory nodes"]
    lines += [f"{node_text(row.node)} | HalfLife={fmt(row.half_life, 2)}" for row in sorted(rows.values(), key=lambda row: (-row.half_life, row.node))[:20]]
    lines += ["", "4. Most forecastable nodes"]
    lines += [f"{node_text(row.node)} | Top1Accuracy={pct(row.top1)}" for row in sorted(rows.values(), key=lambda row: (-row.top1, row.node))[:20]]
    lines += ["", "5. Most replicated nodes"]
    lines += [f"{node_text(row.node)} | ReplicationCount={row.replication_count} | ReplicationPercent={pct(row.replication_percent)}" for row in sorted(rows.values(), key=lambda row: (-row.replication_count, -row.importance, row.node))[:30]]
    lines += ["", "6. Most important states"]
    lines += [f"{name} | MeanImportance={fmt(row.importance)}" for name, row in sorted(states.items(), key=lambda item: -item[1].importance)]
    lines += ["", "7. Most important age groups"]
    lines += [f"{name} | MeanImportance={fmt(row.importance)}" for name, row in sorted(zones.items(), key=lambda item: -item[1].importance)]
    lines += ["", "8. Strongest transition nodes"]
    lines += [f"{node_text(row.node)} | EntropyGrowth={fmt(row.entropy_growth)} | Top1Accuracy={pct(row.top1)}" for row in transitional] or ["None"]
    lines += ["", "9. Nodes to ignore"]
    weak = sorted((row for row in rows.values() if row.category in {"Weak Node", "Insufficient Node"}), key=lambda row: (row.category, row.importance, row.node))
    lines += [f"{node_text(row.node)} | {row.category} | ImportanceScore={fmt(row.importance)}" for row in weak] or ["None"]
    lines += [
        "", "10. Recommended APVA active universe",
        f"CoreUniverse: retain {core.node_count} nodes for {pct(core.coverage)} importance coverage.",
        f"ExtendedUniverse: retain {extended.node_count} nodes for {pct(extended.coverage)} importance coverage.",
        "", "Low-DoF Audit", "Variables used:", "StructuralState", "AgeBucket",
        "", "Derived metrics:", "Count", "Top1Accuracy", "MemoryStrength", "HalfLife",
        "ReplicationPercent", "", "No Context", "No Arbitration", "No Persistence",
        "No Phase", "No Optimization", "No Fitting", "No Machine Learning",
        "No Forward Returns used in node ranking",
        "", "Research Notes",
        "- Which nodes matter and deserve attention? See fixed scores and categories.",
        "- Which nodes are reliable? See ReliabilityScore and replication.",
        "- Can the active APVA universe be reduced? See cumulative importance coverage.",
        "- How many nodes explain most useful structure? See 50%, 75%, and 90% reduction rows.",
        "- Can APVA be reduced to State + Age + a small active node universe? This report tests that mechanically.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_invariants(decay_results: list[DecayResult], rows: dict[Node, NodeRow]) -> None:
    validate_decay(decay_results)
    for row in rows.values():
        if not 0.0 <= row.importance <= 1.0:
            raise RuntimeError(f"{node_text(row.node)}: ImportanceScore escaped [0, 1].")
        if not 0.0 <= row.reliability <= 1.0:
            raise RuntimeError(f"{node_text(row.node)}: ReliabilityScore escaped [0, 1].")
        if row.sample_class != sample_class(row.count):
            raise RuntimeError(f"{node_text(row.node)}: sample class mismatch.")
    coverage = coverage_rows(rows)
    counts = [coverage[target].node_count for target in COVERAGE_TARGETS]
    values = [coverage[target].coverage for target in COVERAGE_TARGETS]
    if counts != sorted(counts) or values != sorted(values):
        raise RuntimeError("Cumulative importance coverage is not monotonic.")
    if any(coverage[target].coverage + 1e-12 < target for target in COVERAGE_TARGETS):
        raise RuntimeError("Cumulative importance coverage did not reach target.")


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
    rows = aggregate_rows(decay_results)
    score_rows(rows)
    validate_invariants(decay_results, rows)
    out_root = Path(args.out_root)
    for result in loaded:
        write_per_instrument(build_result(result), out_root)
    write_aggregate(decay_results, rows, out_root)
    print(f"Wrote {len(decay_results)} per-instrument report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
