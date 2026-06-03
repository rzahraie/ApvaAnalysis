#!/usr/bin/env python3
"""APVA Node Necessity Study v0.1.

Run deterministic structural ablations on the fixed StateAge node universe.
Forward price outcomes are diagnostics only.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable

from APVA_BranchForecast_65 import build_stream
from APVA_InformationContribution_63 import thresholds
from APVA_InformationDecay_57 import study as decay_study
from APVA_MemoryForecast_62 import TARGETS, distribution_entropy, target
from APVA_MinimalEngine_52 import directional_return, instrument_columns, load_results, safe_mean
from APVA_NodeImportance_58 import aggregate_rows, local_rows, score_rows
from APVA_ProcessGraph_54 import Node, node_for, node_text
from APVA_StateCompression_69 import node_profiles, pairwise_similarity
from APVA_StructuralForecast_56 import Outcome, outcome
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

JUNCTION_NODES = {
    "RecoveryResolution_Age1",
    "ReassertionProcessing_Age1",
    "DestructiveRotation_Age1",
    "MixedStructure_Age1",
    "ConstructiveEmergence_Age1",
    "ExhaustionPersistence_Age1",
    "CompressionProcessing_Age1",
}
PIPELINE_NODES = {
    "NeutralProcessing_Age3",
    "NeutralProcessing_Age4",
    "NeutralProcessing_Age5",
    "NeutralProcessing_Age11-20",
}


@dataclass
class Metrics:
    memory_top1: float
    memory_top2: float
    branch_top1: float
    branch_top2: float
    branch_top3: float
    memory_brier: float
    branch_brier: float
    entropy: float
    unique_keys: int
    sparse_rate: float


@dataclass
class NodeImpact:
    node: Node
    count: int
    nearest_neighbor: Node | None
    removal_metrics: Metrics
    replacement_metrics: Metrics
    memory_top1_change: float
    memory_top2_change: float
    branch_top1_change: float
    branch_top2_change: float
    branch_top3_change: float
    memory_brier_change: float
    branch_brier_change: float
    entropy_change: float
    sparse_key_change: float
    memory_contribution: float
    branch_contribution: float
    necessity_score: float
    classification: str
    removal_damage: float
    replacement_damage: float
    substitutability_score: float
    replication_count: int


@dataclass
class MinimalSet:
    remaining_nodes: list[Node]
    removed_nodes: list[Node]
    memory_top1_loss: float
    branch_top1_loss: float
    brier_worsening: float


@dataclass
class Result:
    instrument: str
    source_paths: list
    bars: list
    rows: list
    nodes: list[Node]
    baseline: Metrics
    impacts: dict[Node, NodeImpact]
    minimal: MinimalSet
    outcomes: dict[Node, Outcome]


def mean(values: Iterable[float]) -> float:
    return safe_mean(list(values))


def node_from_text(value: str) -> Node:
    state, age = value.rsplit("_Age", 1)
    return state, age


def distributions(keys: list[Hashable], targets: list[str], universe: tuple[str, ...]) -> dict[Hashable, dict[str, float]]:
    counts: dict[Hashable, Counter[str]] = defaultdict(Counter)
    for key, value in zip(keys, targets):
        counts[key][value] += 1
    return {
        key: {item: counter[item] / sum(counter.values()) for item in universe}
        for key, counter in counts.items()
    }


def forecast_metrics(keys: list[Hashable], targets: list[str]) -> tuple[float, float, float, float, float, int, float]:
    if not keys or not targets:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0
    universe = tuple(sorted(set(targets)))
    table = distributions(keys, targets, universe)
    counts = Counter(keys)
    top1s, top2s, top3s, briers, entropies = [], [], [], [], []
    for key, actual in zip(keys, targets):
        dist = table[key]
        choices = [item for item, _ in sorted(dist.items(), key=lambda item: (-item[1], item[0]))]
        top1s.append(float(actual in choices[:1]))
        top2s.append(float(actual in choices[:2]))
        top3s.append(float(actual in choices[:3]))
        briers.append(sum((dist.get(item, 0.0) - float(item == actual)) ** 2 for item in universe))
        entropies.append(distribution_entropy(dist))
    sparse = mean(count < 50 for count in counts.values())
    return mean(top1s), mean(top2s), mean(top3s), mean(briers), mean(entropies), len(counts), sparse


def memory_targets(rows: list) -> list[str]:
    return [row.target for row in rows]


def baseline_metrics(rows: list) -> Metrics:
    mem_keys = [node_text(row.node) for row in rows]
    mem_targets = memory_targets(rows)
    branch_keys = [node_text(row.node) for row in rows]
    branch_targets = [row.next_node for row in rows]
    mt1, mt2, _, mbrier, mentropy, mkeys, msparse = forecast_metrics(mem_keys, mem_targets)
    bt1, bt2, bt3, bbrier, bentropy, bkeys, bsparse = forecast_metrics(branch_keys, branch_targets)
    return Metrics(mt1, mt2, bt1, bt2, bt3, mbrier, bbrier, mean((mentropy, bentropy)),
                   max(mkeys, bkeys), mean((msparse, bsparse)))


def remove_node_rows(rows: list, removed: Node) -> list:
    return [row for row in rows if row.node != removed and node_from_text(row.next_node) != removed]


def replacement_rows(rows: list, removed: Node, replacement: Node | None) -> tuple[list[Hashable], list[str], list[Hashable], list[str]]:
    if replacement is None:
        kept = remove_node_rows(rows, removed)
        return ([node_text(row.node) for row in kept], memory_targets(kept),
                [node_text(row.node) for row in kept], [row.next_node for row in kept])
    replacement_text = node_text(replacement)
    mem_keys, mem_targets = [], []
    for row in rows:
        key = replacement_text if row.node == removed else node_text(row.node)
        mem_keys.append(key)
        mem_targets.append(row.target)
    branch_keys, branch_targets = [], []
    for row in rows:
        key = replacement_text if row.node == removed else node_text(row.node)
        target_node = replacement_text if node_from_text(row.next_node) == removed else row.next_node
        branch_keys.append(key)
        branch_targets.append(target_node)
    return mem_keys, mem_targets, branch_keys, branch_targets


def rewired_keys_targets(rows: list, removed: Node) -> tuple[list[Hashable], list[str], list[Hashable], list[str]]:
    kept = remove_node_rows(rows, removed)
    mem_keys = [node_text(row.node) for row in kept]
    mem_targets = memory_targets(kept)
    branch_keys = [node_text(row.node) for row in kept]
    branch_targets = [row.next_node for row in kept]
    return mem_keys, mem_targets, branch_keys, branch_targets


def metrics_from_parts(mem_keys: list[Hashable], mem_targets: list[str],
                       branch_keys: list[Hashable], branch_targets: list[str]) -> Metrics:
    mt1, mt2, _, mbrier, mentropy, mkeys, msparse = forecast_metrics(mem_keys, mem_targets)
    bt1, bt2, bt3, bbrier, bentropy, bkeys, bsparse = forecast_metrics(branch_keys, branch_targets)
    return Metrics(mt1, mt2, bt1, bt2, bt3, mbrier, bbrier, mean((mentropy, bentropy)),
                   max(mkeys, bkeys), mean((msparse, bsparse)))


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def classify(score: float) -> str:
    if score >= 0.75:
        return "CriticalNode"
    if score >= 0.50:
        return "ImportantNode"
    if score >= 0.25:
        return "UsefulNode"
    return "DisposableNode"


def nearest_neighbors(rows: list, nodes: list[Node], node_rows: dict[Node, object]) -> dict[Node, Node | None]:
    profiles = node_profiles(nodes, rows, node_rows)
    similarities = pairwise_similarity(profiles)
    best: dict[Node, tuple[float, Node]] = {}
    for row in similarities:
        for node, other in ((row.node_a, row.node_b), (row.node_b, row.node_a)):
            current = best.get(node)
            if current is None or row.total_distance < current[0]:
                best[node] = (row.total_distance, other)
    return {node: best[node][1] if node in best else None for node in set(nodes)}


def outcome_rows(bars: list, nodes: list[Node]) -> dict[Node, Outcome]:
    grouped: dict[Node, list[float]] = defaultdict(list)
    for index, node in enumerate(nodes):
        value = directional_return(bars, index, 5)
        if value is not None:
            grouped[node].append(value)
    return {node: outcome(values) for node, values in grouped.items()}


def build_impacts(rows: list, nodes: list[Node], node_rows: dict[Node, object], baseline: Metrics) -> dict[Node, NodeImpact]:
    unique_nodes = sorted(set(row.node for row in rows), key=node_text)
    counts = Counter(row.node for row in rows)
    neighbors = nearest_neighbors(rows, nodes, node_rows)
    raw_rows = []
    for node in unique_nodes:
        mem_keys, mem_targets, branch_keys, branch_targets = rewired_keys_targets(rows, node)
        removal = metrics_from_parts(mem_keys, mem_targets, branch_keys, branch_targets)
        rmem_keys, rmem_targets, rbranch_keys, rbranch_targets = replacement_rows(rows, node, neighbors.get(node))
        replacement = metrics_from_parts(rmem_keys, rmem_targets, rbranch_keys, rbranch_targets)
        raw_rows.append((node, counts[node], neighbors.get(node), removal, replacement))

    memory_losses = [baseline.memory_top1 - row[3].memory_top1 for row in raw_rows]
    branch_losses = [baseline.branch_top1 - row[3].branch_top1 for row in raw_rows]
    memory_briers = [row[3].memory_brier - baseline.memory_brier for row in raw_rows]
    branch_briers = [row[3].branch_brier - baseline.branch_brier for row in raw_rows]
    entropy_increases = [row[3].entropy - baseline.entropy for row in raw_rows]
    normalized_columns = list(zip(
        normalize([max(0.0, value) for value in memory_losses]),
        normalize([max(0.0, value) for value in branch_losses]),
        normalize([max(0.0, value) for value in memory_briers]),
        normalize([max(0.0, value) for value in branch_briers]),
        normalize([max(0.0, value) for value in entropy_increases]),
    ))
    impacts = {}
    for index, (node, count, neighbor, removal, replacement) in enumerate(raw_rows):
        memory_loss = baseline.memory_top1 - removal.memory_top1
        branch_loss = baseline.branch_top1 - removal.branch_top1
        memory_brier = removal.memory_brier - baseline.memory_brier
        branch_brier = removal.branch_brier - baseline.branch_brier
        score = mean(normalized_columns[index])
        removal_damage = mean((max(0.0, memory_loss), max(0.0, branch_loss),
                               max(0.0, memory_brier), max(0.0, branch_brier)))
        replacement_damage = mean((max(0.0, baseline.memory_top1 - replacement.memory_top1),
                                   max(0.0, baseline.branch_top1 - replacement.branch_top1),
                                   max(0.0, replacement.memory_brier - baseline.memory_brier),
                                   max(0.0, replacement.branch_brier - baseline.branch_brier)))
        impacts[node] = NodeImpact(
            node, count, neighbor, removal, replacement,
            removal.memory_top1 - baseline.memory_top1,
            removal.memory_top2 - baseline.memory_top2,
            removal.branch_top1 - baseline.branch_top1,
            removal.branch_top2 - baseline.branch_top2,
            removal.branch_top3 - baseline.branch_top3,
            removal.memory_brier - baseline.memory_brier,
            removal.branch_brier - baseline.branch_brier,
            removal.entropy - baseline.entropy,
            removal.sparse_rate - baseline.sparse_rate,
            memory_loss + memory_brier,
            branch_loss + branch_brier,
            score,
            classify(score),
            removal_damage,
            replacement_damage,
            removal_damage - replacement_damage,
            1,
        )
    return impacts


def minimal_set(rows: list, impacts: dict[Node, NodeImpact], baseline: Metrics) -> MinimalSet:
    removed: list[Node] = []
    remaining = set(row.node for row in rows)
    last_metrics = baseline
    for impact in sorted(impacts.values(), key=lambda row: (row.necessity_score, row.count, node_text(row.node))):
        candidate_removed = removed + [impact.node]
        candidate_rows = [row for row in rows if row.node not in candidate_removed and node_from_text(row.next_node) not in candidate_removed]
        if len(candidate_rows) < 2:
            break
        metrics = baseline_metrics(candidate_rows)
        memory_loss = baseline.memory_top1 - metrics.memory_top1
        branch_loss = baseline.branch_top1 - metrics.branch_top1
        brier_worsening = max(metrics.memory_brier - baseline.memory_brier, metrics.branch_brier - baseline.branch_brier)
        if memory_loss > 0.02 or branch_loss > 0.02 or brier_worsening > 0.05:
            break
        removed.append(impact.node)
        remaining.discard(impact.node)
        last_metrics = metrics
    return MinimalSet(
        sorted(remaining, key=node_text), removed,
        baseline.memory_top1 - last_metrics.memory_top1,
        baseline.branch_top1 - last_metrics.branch_top1,
        max(last_metrics.memory_brier - baseline.memory_brier, last_metrics.branch_brier - baseline.branch_brier),
    )


def build_result(instrument: str, source_paths: list, bars: list, rows: list, nodes: list[Node],
                 node_rows: dict[Node, object]) -> Result:
    baseline = baseline_metrics(rows)
    impacts = build_impacts(rows, nodes, node_rows, baseline)
    minimal = minimal_set(rows, impacts, baseline)
    return Result(instrument, source_paths, bars, rows, nodes, baseline, impacts, minimal, outcome_rows(bars, nodes))


def refresh_replication(aggregate: Result, instrument_results: list[Result]) -> None:
    for node, impact in aggregate.impacts.items():
        impact.replication_count = sum(node in result.impacts for result in instrument_results)


def aggregate_by_state(impacts: Iterable[NodeImpact]) -> dict[str, tuple[float, float, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for impact in impacts:
        grouped[impact.node[0]].append(impact.necessity_score)
    return {key: (mean(values), statistics.median(values), max(values)) for key, values in grouped.items()}


def aggregate_by_age(impacts: Iterable[NodeImpact]) -> dict[str, tuple[float, float, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for impact in impacts:
        grouped[impact.node[1]].append(impact.necessity_score)
    return {key: (mean(values), statistics.median(values), max(values)) for key, values in grouped.items()}


def impact_line(row: NodeImpact) -> str:
    return (f"{node_text(row.node)} | {fmt(row.necessity_score)} | {fmt(row.memory_contribution)} | "
            f"{fmt(row.branch_contribution)} | {row.classification} | {row.replication_count}")


def removal_line(row: NodeImpact) -> str:
    return (f"{node_text(row.node)} | {pct(row.memory_top1_change)} | {pct(row.branch_top1_change)} | "
            f"{fmt(row.memory_brier_change)} | {fmt(row.branch_brier_change)} | "
            f"{fmt(row.entropy_change)}")


def replacement_line(row: NodeImpact) -> str:
    neighbor = node_text(row.nearest_neighbor) if row.nearest_neighbor else "None"
    return (f"{node_text(row.node)} | {neighbor} | {fmt(row.removal_damage)} | "
            f"{fmt(row.replacement_damage)} | {fmt(row.substitutability_score)}")


def baseline_lines(result: Result) -> list[str]:
    row = result.baseline
    return [
        "",
        "1. Baseline Snapshot",
        "Metric | Value",
        f"MemoryTop1Accuracy | {pct(row.memory_top1)}",
        f"MemoryTop2Accuracy | {pct(row.memory_top2)}",
        f"BranchTop1Accuracy | {pct(row.branch_top1)}",
        f"BranchTop2Accuracy | {pct(row.branch_top2)}",
        f"BranchTop3Accuracy | {pct(row.branch_top3)}",
        f"MemoryBrierScore | {fmt(row.memory_brier)}",
        f"BranchBrierScore | {fmt(row.branch_brier)}",
        f"ForecastEntropy | {fmt(row.entropy)}",
        f"UniqueKeys | {row.unique_keys}",
        f"SparseKeyRate | {pct(row.sparse_rate)}",
    ]


def append_common(lines: list[str], result: Result, limit: int = 80) -> None:
    impacts = sorted(result.impacts.values(), key=lambda row: (-row.necessity_score, node_text(row.node)))
    lines += baseline_lines(result)
    lines += ["", "2. Single Node Removal", "Procedure: Incoming -> RemovedNode -> Outgoing is rewired by dropping observed transitions through the removed node. No transitions are invented."]
    lines += ["", "3. Removal Impact", "Node | MemoryTop1Change | BranchTop1Change | MemoryBrierChange | BranchBrierChange | EntropyChange"]
    lines += [removal_line(row) for row in impacts[:limit]]
    lines += ["", "4. Necessity Score", "Node | NecessityScore | MemoryContribution | BranchContribution | Classification | ReplicationCount"]
    lines += [impact_line(row) for row in impacts[:limit]]
    lines += ["", "5. Critical Node Classification", "Classification | Count"]
    for name, count in sorted(Counter(row.classification for row in result.impacts.values()).items()):
        lines.append(f"{name} | {count}")
    lines += ["", "6. Node Family Necessity", "StructuralState | MeanNecessity | MedianNecessity | MaxNecessity"]
    for state, values in sorted(aggregate_by_state(result.impacts.values()).items(), key=lambda item: (-item[1][0], item[0])):
        lines.append(f"{state} | {fmt(values[0])} | {fmt(values[1])} | {fmt(values[2])}")
    lines += ["", "7. Age Necessity", "AgeBucket | MeanNecessity | MedianNecessity | MaxNecessity"]
    for age, values in sorted(aggregate_by_age(result.impacts.values()).items()):
        lines.append(f"Age{age} | {fmt(values[0])} | {fmt(values[1])} | {fmt(values[2])}")
    lines += ["", "8. Memory Dependence", "Node | MemoryContribution"]
    lines += [f"{node_text(row.node)} | {fmt(row.memory_contribution)}" for row in sorted(result.impacts.values(), key=lambda row: (-row.memory_contribution, node_text(row.node)))[:limit]]
    lines += ["", "9. Branch Dependence", "Node | BranchContribution"]
    lines += [f"{node_text(row.node)} | {fmt(row.branch_contribution)}" for row in sorted(result.impacts.values(), key=lambda row: (-row.branch_contribution, node_text(row.node)))[:limit]]
    lines += ["", "10. Junction Necessity", "Node | NecessityScore | MemoryContribution | BranchContribution"]
    lines += [f"{node_text(row.node)} | {fmt(row.necessity_score)} | {fmt(row.memory_contribution)} | {fmt(row.branch_contribution)}" for row in impacts if node_text(row.node) in JUNCTION_NODES]
    lines += ["", "11. Pipeline Necessity", "Node | NecessityScore | MemoryContribution | BranchContribution"]
    lines += [f"{node_text(row.node)} | {fmt(row.necessity_score)} | {fmt(row.memory_contribution)} | {fmt(row.branch_contribution)}" for row in impacts if node_text(row.node) in PIPELINE_NODES]
    lines += ["", "12. Node Replacement Audit", "Node | NearestNeighbor | RemovalDamage | ReplacementDamage | SubstitutabilityScore"]
    lines += [replacement_line(row) for row in sorted(result.impacts.values(), key=lambda row: (-row.substitutability_score, node_text(row.node)))[:limit]]
    lines += ["", "13. Substitutability", "High score means replacement damages less than removal; low score means the node behaves more uniquely."]
    lines += [replacement_line(row) for row in sorted(result.impacts.values(), key=lambda row: (row.substitutability_score, node_text(row.node)))[:20]]
    lines += ["", "14. Cross-Instrument Replication", "Node | NecessityScore | ReplicationCount"]
    lines += [f"{node_text(row.node)} | {fmt(row.necessity_score)} | {row.replication_count}" for row in impacts[:limit]]
    lines += ["", "15. Minimal APVA Set", "NodeCount", str(len(result.minimal.remaining_nodes)), "", "RemainingNodes"]
    lines += [node_text(node) for node in result.minimal.remaining_nodes]
    lines += ["", f"MemoryTop1Loss | {pct(result.minimal.memory_top1_loss)}",
              f"BranchTop1Loss | {pct(result.minimal.branch_top1_loss)}",
              f"BrierWorsening | {fmt(result.minimal.brier_worsening)}"]
    lines += ["", "16. Outcome Diagnostics", "Node | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5"]
    for node, out in sorted(result.outcomes.items(), key=lambda item: (-item[1].count, node_text(item[0])))[:limit]:
        lines.append(f"{node_text(node)} | {out.count} | {fmt(out.mean_dr)} | {fmt(out.median_dr)} | {pct(out.continuation)} | {pct(out.failure)} | {pct(out.flat)}")
    lines += ["", "17. Recommendation"]
    lines += recommendation_lines(result)
    append_audit(lines)
    lines += ["", "19. Mechanical Research Notes", "This study removes, rewires, or substitutes StateAge nodes using only structural observations. It does not use forward returns in necessity scoring."]


def recommendation_lines(result: Result) -> list[str]:
    counts = Counter(row.classification for row in result.impacts.values())
    critical = counts.get("CriticalNode", 0)
    important = counts.get("ImportantNode", 0)
    minimal = len(result.minimal.remaining_nodes)
    if critical >= 5:
        classification = "HighNecessityCore"
    elif critical + important >= 10:
        classification = "ModerateNecessityCore"
    else:
        classification = "MinimalAPVASet"
    return [
        f"Classification: {classification}",
        f"CriticalNodes: {critical}",
        f"ImportantNodes: {important}",
        f"UsefulNodes: {counts.get('UsefulNode', 0)}",
        f"DisposableNodes: {counts.get('DisposableNode', 0)}",
        f"MinimalNodeCount: {minimal}",
        "Reason: Mechanical ablation compares removal and nearest-neighbor substitution against the unmodified StateAge memory and branch forecasts.",
    ]


def append_audit(lines: list[str]) -> None:
    lines += [
        "",
        "18. Low-DoF Audit",
        "Variables used:",
        "StateAgeNode",
        "StructuralState",
        "AgeBucket",
        "NextNode",
        "MemoryStrength",
        "BranchProbability",
        "BranchEntropy",
        "ForecastAccuracy",
        "BrierScore",
        "",
        "No Context",
        "No Arbitration",
        "No Persistence",
        "No Phase",
        "No Optimization",
        "No Fitting",
        "No Machine Learning",
        "No Forward Returns used in necessity scoring",
    ]


def write_per_instrument(result: Result, out_root: Path) -> None:
    path = out_root / result.instrument / f"NodeNecessity_{result.instrument}.txt"
    ensure_dir(path.parent)
    lines = [
        "APVA Node Necessity Study v0.1",
        "=" * 108,
        f"Instrument: {result.instrument}",
        "Input path(s): " + ", ".join(str(path) for path in result.source_paths),
        f"Total rows: {len(result.bars)}",
        f"Original node count: {len(result.impacts)}",
    ]
    append_common(lines, result)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_outcome_lines(results: list[Result], instruments: list[str]) -> list[str]:
    by_inst = {result.instrument: result for result in results}
    nodes = sorted({node for result in results for node in result.outcomes}, key=node_text)
    lines = ["", "Aggregate Outcome Table", "Node | " + " | ".join(f"Count_{name} | MeanDR_{name}" for name in instruments) + " | ValidInstrumentCount"]
    for node in nodes:
        parts = [node_text(node)]
        valid = 0
        for name in instruments:
            out = by_inst[name].outcomes.get(node)
            if out:
                valid += 1
                parts += [str(out.count), fmt(out.mean_dr)]
            else:
                parts += ["0", "0.0000"]
        parts.append(str(valid))
        lines.append(" | ".join(parts))
    return lines


def write_aggregate(result: Result, instrument_results: list[Result], out_root: Path) -> None:
    path = out_root / "NodeNecessity" / "NodeNecessity_All.txt"
    ensure_dir(path.parent)
    instruments = instrument_columns(instrument_results)
    by_inst = {item.instrument: item for item in instrument_results}
    impacts = sorted(result.impacts.values(), key=lambda row: (-row.necessity_score, node_text(row.node)))
    lines = ["APVA Node Necessity Study v0.1 - Aggregate", "=" * 108, "Instruments: " + ", ".join(instruments)]
    lines += ["", "Aggregate Necessity Table", "Node | NecessityScore | MemoryContribution | BranchContribution | Classification | ReplicationCount"]
    lines += [impact_line(row) for row in impacts]
    lines += ["", "Aggregate Removal Impact Table", "Node | MemoryTop1Change | BranchTop1Change | MemoryBrierChange | BranchBrierChange | EntropyChange"]
    lines += [removal_line(row) for row in impacts]
    lines += ["", "Aggregate Replacement Table", "Node | NearestNeighbor | RemovalDamage | ReplacementDamage | SubstitutabilityScore"]
    lines += [replacement_line(row) for row in sorted(result.impacts.values(), key=lambda row: (-row.substitutability_score, node_text(row.node)))]
    lines += ["", "Aggregate Family Necessity Table", "StructuralState | MeanNecessity | MedianNecessity | MaxNecessity"]
    for state, values in sorted(aggregate_by_state(result.impacts.values()).items(), key=lambda item: (-item[1][0], item[0])):
        lines.append(f"{state} | {fmt(values[0])} | {fmt(values[1])} | {fmt(values[2])}")
    lines += ["", "Aggregate Age Necessity Table", "AgeBucket | MeanNecessity | MedianNecessity | MaxNecessity"]
    for age, values in sorted(aggregate_by_age(result.impacts.values()).items()):
        lines.append(f"Age{age} | {fmt(values[0])} | {fmt(values[1])} | {fmt(values[2])}")
    lines += ["", "Aggregate Junction Necessity Table", "Node | NecessityScore | MemoryContribution | BranchContribution"]
    lines += [f"{node_text(row.node)} | {fmt(row.necessity_score)} | {fmt(row.memory_contribution)} | {fmt(row.branch_contribution)}" for row in impacts if node_text(row.node) in JUNCTION_NODES]
    lines += ["", "Aggregate Pipeline Necessity Table", "Node | NecessityScore | MemoryContribution | BranchContribution"]
    lines += [f"{node_text(row.node)} | {fmt(row.necessity_score)} | {fmt(row.memory_contribution)} | {fmt(row.branch_contribution)}" for row in impacts if node_text(row.node) in PIPELINE_NODES]
    lines += ["", "Aggregate Replication Table", "Node | " + " | ".join(f"Necessity_{name}" for name in instruments) + " | ReplicationCount"]
    for row in impacts:
        values = []
        replication = 0
        for name in instruments:
            item = by_inst[name].impacts.get(row.node)
            if item:
                replication += 1
                values.append(fmt(item.necessity_score))
            else:
                values.append("NA")
        lines.append(f"{node_text(row.node)} | " + " | ".join(values) + f" | {replication}")
    lines += ["", "Aggregate Minimal APVA Set", f"NodeCount | {len(result.minimal.remaining_nodes)}",
              "RemainingNodes | " + ", ".join(node_text(node) for node in result.minimal.remaining_nodes),
              f"MemoryTop1Loss | {pct(result.minimal.memory_top1_loss)}",
              f"BranchTop1Loss | {pct(result.minimal.branch_top1_loss)}",
              f"BrierWorsening | {fmt(result.minimal.brier_worsening)}"]
    lines += aggregate_outcome_lines(instrument_results, instruments)
    lines += ["", "Aggregate Recommendation"]
    lines += recommendation_lines(result)
    lines += rankings(result)
    lines += ["", "RESEARCH NOTES", "Mechanical only.", "Which APVA nodes are indispensable?", "Which nodes can disappear?", "Which nodes can be replaced?", "How small can APVA become before accuracy suffers?", "Is APVA complexity concentrated in a few nodes?", "Can a minimal APVA core be identified?"]
    append_audit(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rankings(result: Result) -> list[str]:
    impacts = list(result.impacts.values())
    lines = ["", "AGGREGATE RANKINGS"]
    lines += ["", "1. Most necessary nodes"] + [impact_line(row) for row in sorted(impacts, key=lambda row: (-row.necessity_score, node_text(row.node)))[:10]]
    lines += ["", "2. Least necessary nodes"] + [impact_line(row) for row in sorted(impacts, key=lambda row: (row.necessity_score, node_text(row.node)))[:10]]
    lines += ["", "3. Most important junctions"] + [impact_line(row) for row in sorted((row for row in impacts if node_text(row.node) in JUNCTION_NODES), key=lambda row: (-row.necessity_score, node_text(row.node)))[:10]]
    lines += ["", "4. Most important pipelines"] + [impact_line(row) for row in sorted((row for row in impacts if node_text(row.node) in PIPELINE_NODES), key=lambda row: (-row.necessity_score, node_text(row.node)))[:10]]
    lines += ["", "5. Most substitutable nodes"] + [replacement_line(row) for row in sorted(impacts, key=lambda row: (-row.substitutability_score, node_text(row.node)))[:10]]
    lines += ["", "6. Least substitutable nodes"] + [replacement_line(row) for row in sorted(impacts, key=lambda row: (row.substitutability_score, node_text(row.node)))[:10]]
    lines += ["", "7. Highest memory contributors"] + [f"{node_text(row.node)} | {fmt(row.memory_contribution)}" for row in sorted(impacts, key=lambda row: (-row.memory_contribution, node_text(row.node)))[:10]]
    lines += ["", "8. Highest branch contributors"] + [f"{node_text(row.node)} | {fmt(row.branch_contribution)}" for row in sorted(impacts, key=lambda row: (-row.branch_contribution, node_text(row.node)))[:10]]
    lines += ["", "9. Most replicated critical nodes"] + [impact_line(row) for row in sorted((row for row in impacts if row.classification == "CriticalNode"), key=lambda row: (-row.replication_count, -row.necessity_score, node_text(row.node)))[:10]]
    lines += ["", "10. Recommended APVA core"] + [node_text(node) for node in result.minimal.remaining_nodes[:30]]
    return lines


def validate(result: Result) -> None:
    if not result.rows:
        raise RuntimeError(f"{result.instrument}: no structural rows.")
    if not result.impacts:
        raise RuntimeError(f"{result.instrument}: no node impacts.")
    if not (0 <= result.baseline.memory_top1 <= 1 and 0 <= result.baseline.branch_top1 <= 1):
        raise RuntimeError(f"{result.instrument}: baseline accuracy out of bounds.")
    for row in result.impacts.values():
        if not (0 <= row.necessity_score <= 1):
            raise RuntimeError(f"{result.instrument}: necessity score out of bounds for {node_text(row.node)}.")


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
    decay = [decay_study(row) for row in loaded]
    aggregate_node_rows = aggregate_rows(decay)
    score_rows(aggregate_node_rows)
    low, high = thresholds(loaded, aggregate_node_rows)

    instrument_results = []
    aggregate_bars, aggregate_paths, aggregate_nodes, aggregate_rows_stream = [], [], [], []
    for loaded_row, decay_row in zip(loaded, decay):
        local_node_rows = local_rows(decay_row)
        score_rows(local_node_rows)
        local_stream = build_stream(loaded_row, local_node_rows, (low, high), (0.0, 0.0), (0.0, 0.0))
        local_nodes = [node_for(bar) for bar in loaded_row.bars]
        result = build_result(loaded_row.instrument, loaded_row.source_paths, loaded_row.bars, local_stream, local_nodes, local_node_rows)
        validate(result)
        instrument_results.append(result)

        offset = len(aggregate_bars)
        aggregate_rows_stream.extend(build_stream(loaded_row, aggregate_node_rows, (low, high), (0.0, 0.0), (0.0, 0.0), offset))
        aggregate_nodes.extend(local_nodes)
        aggregate_bars.extend(loaded_row.bars)
        aggregate_paths.extend(loaded_row.source_paths)

    aggregate = build_result("Aggregate", aggregate_paths, aggregate_bars, aggregate_rows_stream, aggregate_nodes, aggregate_node_rows)
    refresh_replication(aggregate, instrument_results)
    validate(aggregate)

    out_root = Path(args.out_root)
    for result in instrument_results:
        write_per_instrument(result, out_root)
    write_aggregate(aggregate, instrument_results, out_root)
    print(f"Wrote {len(instrument_results)} per-instrument report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
