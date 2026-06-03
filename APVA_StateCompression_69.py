#!/usr/bin/env python3
"""APVA State Compression Study v0.1.

Evaluate deterministic compression of the fixed StateAge node universe.
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
from APVA_MemoryDynamics_61 import build_result as build_dynamics
from APVA_MemoryForecast_62 import TARGETS, distribution_entropy, target
from APVA_MinimalEngine_52 import directional_return, instrument_columns, load_results, safe_mean
from APVA_NodeImportance_58 import aggregate_rows, local_rows, score_rows
from APVA_ProcessGraph_54 import Node, node_for, node_text
from APVA_StructuralForecast_56 import Outcome, outcome
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

AGE_ORDER = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6-10": 6, "11-20": 7, "21+": 8}
SCHEMES = (
    "SchemeA_NoCompression",
    "SchemeB_StrongOnly",
    "SchemeC_StrongModerate",
    "SchemeD_AgeAdjacentStrong",
    "SchemeE_LowCountAbsorption",
)


@dataclass
class NodeProfile:
    node: Node
    count: int
    next_distribution: dict[str, float]
    memory_distribution: dict[str, float]
    dominant_next: str
    dominant_probability: float
    branch_entropy: float
    memory_strength: float
    replication_count: int


@dataclass
class SimilarityRow:
    node_a: Node
    node_b: Node
    total_distance: float
    next_distance: float
    memory_distance: float
    entropy_distance: float
    dominant_distance: float
    memory_strength_distance: float
    similarity_class: str
    merge_type: str


@dataclass
class SchemeMetrics:
    scheme: str
    original_keys: int
    compressed_keys: int
    key_reduction: float
    sparse_rate: float
    memory_top1: float
    memory_top2: float
    memory_brier: float
    memory_calibration: float
    memory_entropy: float
    branch_top1: float
    branch_top2: float
    branch_top3: float
    branch_brier: float
    branch_calibration: float
    branch_entropy: float
    memory_top1_loss: float = 0.0
    branch_top1_loss: float = 0.0
    memory_brier_worsening: float = 0.0
    branch_brier_worsening: float = 0.0
    entropy_increase: float = 0.0
    information_loss: float = 0.0
    efficiency: float = 0.0


@dataclass
class GroupRow:
    group_id: str
    members: tuple[Node, ...]
    total_count: int
    dominant_state: str
    dominant_age: str
    mean_memory: float
    dominant_next: str
    branch_entropy: float
    replication_count: int


@dataclass
class UnsafeMerge:
    node_a: Node
    node_b: Node
    reason: str


@dataclass
class LowCountRow:
    node: Node
    count: int
    nearest: Node
    distance: float
    recommended: bool


@dataclass
class Result:
    instrument: str
    source_paths: list
    bars: list
    nodes: list[Node]
    branch_rows: list
    profiles: dict[Node, NodeProfile]
    similarities: list[SimilarityRow]
    schemes: dict[str, SchemeMetrics]
    groups: dict[str, list[GroupRow]]
    unsafe: list[UnsafeMerge]
    low_count: list[LowCountRow]
    outcomes: dict[str, Outcome]


class UnionFind:
    def __init__(self, nodes: Iterable[Node]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: Node) -> Node:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: Node, right: Node) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        keep, move = sorted((root_left, root_right), key=node_text)
        self.parent[move] = keep

    def groups(self) -> dict[Node, tuple[Node, ...]]:
        grouped: dict[Node, list[Node]] = defaultdict(list)
        for node in self.parent:
            grouped[self.find(node)].append(node)
        return {root: tuple(sorted(values, key=node_text)) for root, values in grouped.items()}


def mean(values: Iterable[float]) -> float:
    return safe_mean(list(values))


def tv_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def normalize_scalar(value: float, span: float) -> float:
    return abs(value) / span if span else 0.0


def classify_similarity(distance: float) -> str:
    if distance <= 0.10:
        return "StrongMergeCandidate"
    if distance <= 0.20:
        return "ModerateMergeCandidate"
    return "NoMergeCandidate"


def adjacent(left: Node, right: Node) -> bool:
    return left[0] == right[0] and abs(AGE_ORDER.get(left[1], 100) - AGE_ORDER.get(right[1], -100)) == 1


def merge_type(left: Node, right: Node) -> str:
    if adjacent(left, right):
        return "AgeAdjacent"
    if left[0] == right[0]:
        return "SameStateNonadjacent"
    return "CrossState"


def distributions(keys: list[Hashable], targets: list[str], universe: tuple[str, ...]) -> dict[Hashable, dict[str, float]]:
    counts: dict[Hashable, Counter[str]] = defaultdict(Counter)
    for key, value in zip(keys, targets):
        counts[key][value] += 1
    return {
        key: {item: counter[item] / sum(counter.values()) for item in universe}
        for key, counter in counts.items()
    }


def forecast_metrics(keys: list[Hashable], targets: list[str], topn: int = 2) -> tuple[float, float, float, float, float, float, int, float]:
    if not keys or not targets:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
    universe = tuple(sorted(set(targets)))
    table = distributions(keys, targets, universe)
    counts = Counter(keys)
    top1s, top2s, top3s, briers, calibrations, entropies = [], [], [], [], [], []
    for key, actual in zip(keys, targets):
        dist = table[key]
        ranked = sorted(dist.items(), key=lambda item: (-item[1], item[0]))
        choices = [item for item, _ in ranked]
        probability = ranked[0][1] if ranked else 0.0
        top1s.append(float(actual in choices[:1]))
        top2s.append(float(actual in choices[:2]))
        top3s.append(float(actual in choices[:3]))
        briers.append(sum((dist.get(item, 0.0) - float(item == actual)) ** 2 for item in universe))
        calibrations.append(abs(probability - float(actual in choices[:1])))
        entropies.append(distribution_entropy(dist))
    sparse = mean(count < 50 for count in counts.values())
    return mean(top1s), mean(top2s), mean(top3s), mean(briers), mean(calibrations), mean(entropies), len(counts), sparse


def node_profiles(nodes: list[Node], branch_rows: list, node_rows: dict[Node, object], replication_fallback: int = 1) -> dict[Node, NodeProfile]:
    next_targets = [row.next_node for row in branch_rows]
    current_nodes = [row.node for row in branch_rows]
    next_universe = tuple(sorted(set(next_targets)))
    next_distributions = distributions(current_nodes, next_targets, next_universe)
    memory_targets = [target(node_rows[nodes[index + 1]].memory_strength - node_rows[nodes[index]].memory_strength) for index in range(len(nodes) - 1)]
    memory_universe = TARGETS
    memory_distributions = distributions(nodes[:-1], memory_targets, memory_universe)
    counts = Counter(current_nodes)
    profiles = {}
    for node in sorted(set(current_nodes), key=node_text):
        next_dist = next_distributions.get(node, {})
        ranked = sorted(next_dist.items(), key=lambda item: (-item[1], item[0]))
        profiles[node] = NodeProfile(
            node, counts[node], next_dist, memory_distributions.get(node, {item: 0.0 for item in TARGETS}),
            ranked[0][0] if ranked else "N/A", ranked[0][1] if ranked else 0.0,
            distribution_entropy(next_dist), node_rows[node].memory_strength,
            getattr(node_rows[node], "replication_count", replication_fallback),
        )
    return profiles


def pairwise_similarity(profiles: dict[Node, NodeProfile]) -> list[SimilarityRow]:
    rows = []
    values = list(profiles.values())
    entropy_span = max((row.branch_entropy for row in values), default=0.0) - min((row.branch_entropy for row in values), default=0.0)
    dominant_span = max((row.dominant_probability for row in values), default=0.0) - min((row.dominant_probability for row in values), default=0.0)
    memory_span = max((row.memory_strength for row in values), default=0.0) - min((row.memory_strength for row in values), default=0.0)
    for index, left in enumerate(values):
        for right in values[index + 1:]:
            next_distance = tv_distance(left.next_distribution, right.next_distribution)
            memory_distance = tv_distance(left.memory_distribution, right.memory_distribution)
            entropy_distance = normalize_scalar(left.branch_entropy - right.branch_entropy, entropy_span)
            dominant_distance = normalize_scalar(left.dominant_probability - right.dominant_probability, dominant_span)
            memory_distance_scalar = normalize_scalar(left.memory_strength - right.memory_strength, memory_span)
            total = mean((next_distance, memory_distance, entropy_distance, dominant_distance, memory_distance_scalar))
            rows.append(SimilarityRow(
                left.node, right.node, total, next_distance, memory_distance, entropy_distance,
                dominant_distance, memory_distance_scalar, classify_similarity(total), merge_type(left.node, right.node),
            ))
    return rows


def build_mapping(nodes: set[Node], profiles: dict[Node, NodeProfile], similarities: list[SimilarityRow], scheme: str) -> dict[Node, str]:
    uf = UnionFind(nodes)
    if scheme == "SchemeB_StrongOnly":
        selected = [row for row in similarities if row.total_distance <= 0.10]
    elif scheme == "SchemeC_StrongModerate":
        selected = [row for row in similarities if row.total_distance <= 0.20]
    elif scheme == "SchemeD_AgeAdjacentStrong":
        selected = [row for row in similarities if row.total_distance <= 0.10 and row.merge_type == "AgeAdjacent"]
    else:
        selected = []
    for row in selected:
        uf.union(row.node_a, row.node_b)
    if scheme == "SchemeE_LowCountAbsorption":
        high = [node for node, row in profiles.items() if row.count >= 50]
        for node, row in profiles.items():
            if row.count >= 50 or not high:
                continue
            nearest = min((other for other in high if other != node), key=lambda other: profile_distance(row, profiles[other]))
            uf.union(node, nearest)
    groups = uf.groups()
    mapping = {}
    for root, members in groups.items():
        label = "G_" + "__".join(node_text(node) for node in members)
        for member in members:
            mapping[member] = label
    return mapping


def profile_distance(left: NodeProfile, right: NodeProfile) -> float:
    return mean((
        tv_distance(left.next_distribution, right.next_distribution),
        tv_distance(left.memory_distribution, right.memory_distribution),
        abs(left.branch_entropy - right.branch_entropy),
        abs(left.dominant_probability - right.dominant_probability),
        abs(left.memory_strength - right.memory_strength),
    ))


def scheme_metrics(scheme: str, mapping: dict[Node, str], nodes: list[Node], branch_rows: list, profiles: dict[Node, NodeProfile],
                   baseline: SchemeMetrics | None = None) -> SchemeMetrics:
    original_keys = len(set(nodes))
    compressed_keys = len(set(mapping.values()))
    key_reduction = 1 - compressed_keys / original_keys if original_keys else 0.0
    memory_keys = [mapping[node] for node in nodes[:-1]]
    memory_targets = [target(profiles[nodes[index + 1]].memory_strength - profiles[nodes[index]].memory_strength) for index in range(len(nodes) - 1)]
    memory = forecast_metrics(memory_keys, memory_targets, topn=2)
    branch_keys = [mapping[row.node] for row in branch_rows]
    branch_targets = [mapping.get(node_from_text(row.next_node), row.next_node) for row in branch_rows]
    branch = forecast_metrics(branch_keys, branch_targets, topn=3)
    sparse = mean((memory[7], branch[7]))
    metrics = SchemeMetrics(
        scheme, original_keys, compressed_keys, key_reduction, sparse,
        memory[0], memory[1], memory[3], memory[4], memory[5],
        branch[0], branch[1], branch[2], branch[3], branch[4], branch[5],
    )
    if baseline:
        metrics.memory_top1_loss = baseline.memory_top1 - metrics.memory_top1
        metrics.branch_top1_loss = baseline.branch_top1 - metrics.branch_top1
        metrics.memory_brier_worsening = metrics.memory_brier - baseline.memory_brier
        metrics.branch_brier_worsening = metrics.branch_brier - baseline.branch_brier
        metrics.entropy_increase = mean((metrics.memory_entropy - baseline.memory_entropy, metrics.branch_entropy - baseline.branch_entropy))
        loss_parts = [max(0.0, metrics.memory_top1_loss), max(0.0, metrics.branch_top1_loss),
                      max(0.0, metrics.memory_brier_worsening), max(0.0, metrics.branch_brier_worsening)]
        metrics.information_loss = mean(loss_parts)
        metrics.efficiency = metrics.key_reduction / (1 + metrics.information_loss)
    return metrics


def node_from_text(value: str) -> Node:
    state, age = value.rsplit("_Age", 1)
    return state, age


def group_inventory(mapping: dict[Node, str], profiles: dict[Node, NodeProfile]) -> list[GroupRow]:
    grouped: dict[str, list[NodeProfile]] = defaultdict(list)
    for node, label in mapping.items():
        if node in profiles:
            grouped[label].append(profiles[node])
    rows = []
    for label, members in grouped.items():
        count = sum(row.count for row in members)
        state = Counter({row.node[0]: row.count for row in members}).most_common(1)[0][0]
        age = Counter({row.node[1]: row.count for row in members}).most_common(1)[0][0]
        next_counts = Counter()
        for row in members:
            for target_node, probability in row.next_distribution.items():
                next_counts[target_node] += probability * row.count
        dominant = next_counts.most_common(1)[0][0] if next_counts else "N/A"
        probabilities = [value / sum(next_counts.values()) for value in next_counts.values()] if next_counts else []
        rows.append(GroupRow(
            label, tuple(sorted((row.node for row in members), key=node_text)), count, state, age,
            mean(row.memory_strength for row in members), dominant, -sum(p * math.log(p) for p in probabilities if p > 0),
            max(row.replication_count for row in members),
        ))
    return sorted(rows, key=lambda row: (-row.total_count, row.group_id))


def unsafe_merges(similarities: list[SimilarityRow]) -> list[UnsafeMerge]:
    rows = []
    for row in similarities:
        if row.memory_distance <= 0.10 and row.next_distance > 0.30:
            rows.append(UnsafeMerge(row.node_a, row.node_b, "Similar Memory but different BranchDistribution"))
        elif row.next_distance <= 0.10 and row.memory_distance > 0.30:
            rows.append(UnsafeMerge(row.node_a, row.node_b, "Similar BranchDistribution but different Memory"))
    return rows


def low_count_rows(profiles: dict[Node, NodeProfile]) -> list[LowCountRow]:
    high = [node for node, row in profiles.items() if row.count >= 50]
    rows = []
    for node, row in profiles.items():
        if row.count >= 50 or not high:
            continue
        nearest = min((other for other in high if other != node), key=lambda other: profile_distance(row, profiles[other]))
        distance = profile_distance(row, profiles[nearest])
        rows.append(LowCountRow(node, row.count, nearest, distance, distance <= 0.20))
    return sorted(rows, key=lambda row: (row.distance, row.node))


def outcome_rows(bars: list, nodes: list[Node], mapping: dict[Node, str]) -> dict[str, Outcome]:
    values: dict[str, list[float]] = defaultdict(list)
    for index, node in enumerate(nodes):
        value = directional_return(bars, index, 5)
        if value is not None:
            values[mapping[node]].append(value)
    return {key: outcome(samples) for key, samples in values.items()}


def build_result(instrument: str, source_paths: list, bars: list, nodes: list[Node], branch_rows: list,
                 node_rows: dict[Node, object]) -> Result:
    profiles = node_profiles(nodes, branch_rows, node_rows)
    similarities = pairwise_similarity(profiles)
    node_set = set(profiles)
    scheme_rows = {}
    group_rows = {}
    baseline = None
    mappings = {}
    for scheme in SCHEMES:
        mapping = build_mapping(node_set, profiles, similarities, scheme)
        mappings[scheme] = mapping
        current = scheme_metrics(scheme, mapping, nodes, branch_rows, profiles, baseline)
        if scheme == "SchemeA_NoCompression":
            baseline = current
        else:
            current = scheme_metrics(scheme, mapping, nodes, branch_rows, profiles, baseline)
        scheme_rows[scheme] = current
        group_rows[scheme] = group_inventory(mapping, profiles)
    return Result(
        instrument, source_paths, bars, nodes, branch_rows, profiles, similarities, scheme_rows, group_rows,
        unsafe_merges(similarities), low_count_rows(profiles), outcome_rows(bars, nodes, mappings["SchemeC_StrongModerate"]),
    )


def profile_line(row: NodeProfile) -> str:
    return f"{node_text(row.node)} | {row.count} | {row.dominant_next} | {pct(row.dominant_probability)} | {fmt(row.branch_entropy)} | {fmt(row.memory_strength)} | {row.replication_count}"


def similarity_line(row: SimilarityRow) -> str:
    return f"{node_text(row.node_a)} | {node_text(row.node_b)} | {fmt(row.total_distance)} | {row.similarity_class} | {row.merge_type}"


def scheme_line(row: SchemeMetrics) -> str:
    return f"{row.scheme} | {row.original_keys} | {row.compressed_keys} | {pct(row.key_reduction)} | {pct(row.sparse_rate)}"


def memory_line(row: SchemeMetrics) -> str:
    return f"{row.scheme} | {pct(row.memory_top1)} | {pct(row.memory_top2)} | {fmt(row.memory_brier)} | {pct(row.memory_calibration)} | {fmt(row.memory_entropy)} | {pct(row.memory_top1_loss)} | {fmt(row.memory_brier_worsening)}"


def branch_line(row: SchemeMetrics) -> str:
    return f"{row.scheme} | {pct(row.branch_top1)} | {pct(row.branch_top2)} | {pct(row.branch_top3)} | {fmt(row.branch_brier)} | {pct(row.branch_calibration)} | {fmt(row.branch_entropy)} | {pct(row.branch_top1_loss)} | {fmt(row.branch_brier_worsening)}"


def recommendation(result: Result) -> list[str]:
    candidates = [row for name, row in result.schemes.items() if name != "SchemeA_NoCompression"]
    conservative = [row for row in candidates if row.scheme in ("SchemeB_StrongOnly", "SchemeD_AgeAdjacentStrong") and row.key_reduction >= 0.20 and max(row.memory_top1_loss, row.branch_top1_loss) <= 0.02]
    moderate = [row for row in candidates if row.scheme == "SchemeC_StrongModerate" and row.key_reduction >= 0.30 and max(row.memory_top1_loss, row.branch_top1_loss) <= 0.05]
    if moderate:
        selected = max(moderate, key=lambda row: row.efficiency)
        label = "ModerateCompression"
    elif conservative:
        selected = max(conservative, key=lambda row: row.efficiency)
        label = "ConservativeCompression"
    elif any(row.key_reduction > 0 for row in candidates):
        selected = max(candidates, key=lambda row: row.efficiency)
        label = "AggressiveCompressionRejected"
    else:
        selected = result.schemes["SchemeA_NoCompression"]
        label = "NoCompression"
    return [
        f"Classification: {label}",
        f"RecommendedScheme: {selected.scheme}",
        f"OriginalNodeCount: {selected.original_keys}",
        f"CompressedNodeCount: {selected.compressed_keys}",
        f"KeyReductionPercent: {pct(selected.key_reduction)}",
        f"InformationLoss: {fmt(selected.information_loss)}",
        f"Reason: MemoryTop1Loss={pct(selected.memory_top1_loss)}; BranchTop1Loss={pct(selected.branch_top1_loss)}; Efficiency={fmt(selected.efficiency)}.",
    ]


def append_common(lines: list[str], result: Result, limit: int = 80) -> None:
    lines += ["", "1. Node Behavior Profile", "Node | Count | DominantNextNode | DominantBranchProbability | BranchEntropy | MemoryStrength | ReplicationCount"]
    lines += [profile_line(row) for row in sorted(result.profiles.values(), key=lambda row: (-row.count, row.node))]
    lines += ["", "2. Pairwise Node Similarity", "NodeA | NodeB | TotalDistance | SimilarityClass | MergeType"]
    lines += [similarity_line(row) for row in sorted(result.similarities, key=lambda row: (row.total_distance, row.node_a, row.node_b))[:limit]]
    lines += ["", "3. Candidate Merge Pairs", "NodeA | NodeB | TotalDistance | SimilarityClass | CountA | CountB | ReplicationA | ReplicationB"]
    for row in sorted((row for row in result.similarities if row.similarity_class != "NoMergeCandidate"), key=lambda row: (row.total_distance, row.node_a))[:limit]:
        left, right = result.profiles[row.node_a], result.profiles[row.node_b]
        lines.append(f"{node_text(row.node_a)} | {node_text(row.node_b)} | {fmt(row.total_distance)} | {row.similarity_class} | {left.count} | {right.count} | {left.replication_count} | {right.replication_count}")
    for number, title, predicate in (
        (4, "Age-Adjacent Merge Audit", lambda row: row.merge_type == "AgeAdjacent"),
        (5, "Same-State Nonadjacent Merge Audit", lambda row: row.merge_type == "SameStateNonadjacent"),
        (6, "Cross-State Merge Audit", lambda row: row.merge_type == "CrossState" and row.similarity_class == "StrongMergeCandidate"),
    ):
        lines += ["", f"{number}. {title}", "NodeA | NodeB | Distance | MergeCandidateClass"]
        lines += [f"{node_text(row.node_a)} | {node_text(row.node_b)} | {fmt(row.total_distance)} | {row.similarity_class}"
                  for row in sorted((row for row in result.similarities if predicate(row)), key=lambda row: row.total_distance)[:limit]]
    lines += ["", "7. Compression Schemes", "Scheme | OriginalKeyCount | CompressedKeyCount | KeyReductionPercent | SparseKeyRate"]
    lines += [scheme_line(row) for row in result.schemes.values()]
    lines += ["", "8. Compressed Memory Forecast Test", "Scheme | Top1Accuracy | Top2Accuracy | BrierScore | CalibrationError | ForecastEntropy | Top1Loss | BrierWorsening"]
    lines += [memory_line(row) for row in result.schemes.values()]
    lines += ["", "9. Compressed Branch Forecast Test", "Scheme | Top1Accuracy | Top2Accuracy | Top3Accuracy | BrierScore | CalibrationError | ForecastEntropy | Top1Loss | BrierWorsening"]
    lines += [branch_line(row) for row in result.schemes.values()]
    lines += ["", "10. Information Loss", "Scheme | MemoryTop1Loss | BranchTop1Loss | MemoryBrierWorsening | BranchBrierWorsening | EntropyIncrease | InformationLoss"]
    lines += [f"{row.scheme} | {pct(row.memory_top1_loss)} | {pct(row.branch_top1_loss)} | {fmt(row.memory_brier_worsening)} | {fmt(row.branch_brier_worsening)} | {fmt(row.entropy_increase)} | {fmt(row.information_loss)}" for row in result.schemes.values()]
    lines += ["", "11. Compression Efficiency", "Scheme | KeyReductionPercent | InformationLoss | CompressionEfficiency"]
    lines += [f"{row.scheme} | {pct(row.key_reduction)} | {fmt(row.information_loss)} | {fmt(row.efficiency)}" for row in sorted(result.schemes.values(), key=lambda row: -row.efficiency)]
    lines += ["", "12. Node Group Inventory", "GroupID | Members | TotalCount | DominantState | DominantAge | MeanMemoryStrength | DominantNextNode | BranchEntropy"]
    for row in result.groups["SchemeC_StrongModerate"][:limit]:
        lines.append(f"{row.group_id} | {', '.join(node_text(node) for node in row.members)} | {row.total_count} | {row.dominant_state} | {row.dominant_age} | {fmt(row.mean_memory)} | {row.dominant_next} | {fmt(row.branch_entropy)}")
    lines += ["", "13. Unsafe Merges", "NodeA | NodeB | FailureReason"]
    lines += [f"{node_text(row.node_a)} | {node_text(row.node_b)} | {row.reason}" for row in result.unsafe[:limit]]
    lines += ["", "14. Low-Count Node Handling", "Node | Count | NearestNeighbor | Distance | MergeRecommended"]
    lines += [f"{node_text(row.node)} | {row.count} | {node_text(row.nearest)} | {fmt(row.distance)} | {row.recommended}" for row in result.low_count[:limit]]
    lines += ["", "15. Cross-Instrument Replication", "See aggregate report."]
    lines += ["", "16. Outcome Diagnostics", "Diagnostic only. Outcomes are not used in compression.", "GroupID | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew"]
    for group, row in sorted(result.outcomes.items(), key=lambda item: (-item[1].count, item[0]))[:limit]:
        lines.append(f"{group} | {row.count} | {fmt(row.mean_dr)} | {fmt(row.median_dr)} | {pct(row.continuation)} | {pct(row.failure)} | {pct(row.flat)} | {fmt(row.skew)}")


def append_audit(lines: list[str]) -> None:
    lines += ["", "18. Low-DoF Audit", "Variables used:", "StateAgeNode", "StructuralState", "AgeBucket", "NextNode", "MemoryStrength", "BranchProbability", "BranchEntropy", "DominantNextNode", "", "No Context", "No Arbitration", "No Persistence", "No Phase", "No Optimization", "No Fitting", "No Machine Learning", "No Forward Returns used in compression construction"]


def write_per_instrument(result: Result, out_root: Path) -> None:
    path = out_root / result.instrument / f"StateCompression_{result.instrument}.txt"
    ensure_dir(path.parent)
    lines = ["APVA State Compression Study v0.1", "=" * 108, f"Instrument: {result.instrument}", "Input path(s): " + ", ".join(str(path) for path in result.source_paths), f"Total rows: {len(result.bars)}", f"Original node count: {len(result.profiles)}"]
    append_common(lines, result)
    lines += ["", "17. Recommendation"] + recommendation(result)
    append_audit(lines)
    lines += ["", "19. Mechanical Research Notes", "- Compression schemes are deterministic and fixed.", "- Forward outcomes are diagnostic only.", "- No new states or discretionary families are introduced."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_outcome_lines(results: list[Result], instruments: list[str]) -> list[str]:
    by_name = {row.instrument: row for row in results}
    groups = sorted({group for row in results for group in row.outcomes})
    lines = ["", "Aggregate Outcome Table", "GroupID | Members | " + " | ".join(f"Count_{name} | MeanDR_{name}" for name in instruments) + " | ValidInstrumentCount"]
    for group in groups[:150]:
        cells, valid = [], 0
        members = group[2:].replace("__", ", ") if group.startswith("G_") else group
        for name in instruments:
            row = by_name[name].outcomes.get(group)
            cells.append(f"{row.count if row else 0} | {fmt(row.mean_dr) if row else 'N/A'}")
            valid += int(row is not None)
        lines.append(f"{group} | {members} | " + " | ".join(cells) + f" | {valid}")
    return lines


def write_aggregate(result: Result, instrument_results: list[Result], out_root: Path) -> None:
    path = out_root / "StateCompression" / "StateCompression_All.txt"
    ensure_dir(path.parent)
    instruments = instrument_columns(instrument_results)
    lines = ["APVA State Compression Study v0.1 - Aggregate", "=" * 108, "Instruments: " + ", ".join(instruments)]
    lines += ["", "Aggregate Node Behavior Table", "Node | Count | DominantNextNode | DominantBranchProbability | BranchEntropy | MemoryStrength | ReplicationCount"]
    lines += [profile_line(row) for row in sorted(result.profiles.values(), key=lambda row: (-row.count, row.node))]
    lines += ["", "Aggregate Similarity Table", "NodeA | NodeB | TotalDistance | SimilarityClass"]
    lines += [f"{node_text(row.node_a)} | {node_text(row.node_b)} | {fmt(row.total_distance)} | {row.similarity_class}" for row in sorted(result.similarities, key=lambda row: row.total_distance)[:250]]
    lines += ["", "Aggregate Merge Candidate Table", "NodeA | NodeB | MergeType | TotalDistance | SimilarityClass | ReplicationCount"]
    for row in sorted((row for row in result.similarities if row.similarity_class != "NoMergeCandidate"), key=lambda row: row.total_distance)[:250]:
        replication = min(result.profiles[row.node_a].replication_count, result.profiles[row.node_b].replication_count)
        lines.append(f"{node_text(row.node_a)} | {node_text(row.node_b)} | {row.merge_type} | {fmt(row.total_distance)} | {row.similarity_class} | {replication}")
    lines += ["", "Aggregate Compression Scheme Table", "Scheme | OriginalKeyCount | CompressedKeyCount | KeyReductionPercent | SparseKeyRate"]
    lines += [scheme_line(row) for row in result.schemes.values()]
    lines += ["", "Aggregate Memory Forecast Compression Table", "Scheme | Top1Accuracy | Top2Accuracy | BrierScore | CalibrationError | ForecastEntropy | Top1Loss | BrierWorsening"]
    lines += [memory_line(row) for row in result.schemes.values()]
    lines += ["", "Aggregate Branch Forecast Compression Table", "Scheme | Top1Accuracy | Top2Accuracy | Top3Accuracy | BrierScore | CalibrationError | ForecastEntropy | Top1Loss | BrierWorsening"]
    lines += [branch_line(row) for row in result.schemes.values()]
    lines += ["", "Aggregate Information Loss Table", "Scheme | MemoryTop1Loss | BranchTop1Loss | MemoryBrierWorsening | BranchBrierWorsening | EntropyIncrease | InformationLoss"]
    lines += [f"{row.scheme} | {pct(row.memory_top1_loss)} | {pct(row.branch_top1_loss)} | {fmt(row.memory_brier_worsening)} | {fmt(row.branch_brier_worsening)} | {fmt(row.entropy_increase)} | {fmt(row.information_loss)}" for row in result.schemes.values()]
    lines += ["", "Aggregate Compression Efficiency Table", "Scheme | KeyReductionPercent | InformationLoss | CompressionEfficiency"]
    lines += [f"{row.scheme} | {pct(row.key_reduction)} | {fmt(row.information_loss)} | {fmt(row.efficiency)}" for row in sorted(result.schemes.values(), key=lambda row: -row.efficiency)]
    lines += ["", "Aggregate Group Inventory", "GroupID | Members | TotalCount | DominantNextNode | BranchEntropy | MeanMemoryStrength | ReplicationCount"]
    for row in result.groups["SchemeC_StrongModerate"][:150]:
        lines.append(f"{row.group_id} | {', '.join(node_text(node) for node in row.members)} | {row.total_count} | {row.dominant_next} | {fmt(row.branch_entropy)} | {fmt(row.mean_memory)} | {row.replication_count}")
    lines += ["", "Aggregate Unsafe Merge Table", "NodeA | NodeB | FailureReason"]
    lines += [f"{node_text(row.node_a)} | {node_text(row.node_b)} | {row.reason}" for row in result.unsafe[:150]]
    lines += ["", "Aggregate Low-Count Node Table", "Node | Count | NearestNeighbor | Distance | MergeRecommended"]
    lines += [f"{node_text(row.node)} | {row.count} | {node_text(row.nearest)} | {fmt(row.distance)} | {row.recommended}" for row in result.low_count[:150]]
    lines += aggregate_outcome_lines(instrument_results, instruments)
    lines += ["", "Aggregate Recommendation"] + recommendation(result)
    lines += rankings(result)
    lines += ["", "RESEARCH NOTES", "- StateAge compression is evaluated through fixed mechanical schemes.", "- Age fragmentation, cross-state redundancy, and rare-node absorption are audited separately.", "- Compression is accepted only by fixed information-loss thresholds."]
    append_audit(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rankings(result: Result) -> list[str]:
    lines = ["", "AGGREGATE RANKINGS"]
    lines += ["", "1. Best merge candidates"] + [similarity_line(row) for row in sorted(result.similarities, key=lambda row: row.total_distance)[:10]]
    lines += ["", "2. Best age-adjacent merges"] + [similarity_line(row) for row in sorted((row for row in result.similarities if row.merge_type == "AgeAdjacent"), key=lambda row: row.total_distance)[:10]]
    lines += ["", "3. Best cross-state merges"] + [similarity_line(row) for row in sorted((row for row in result.similarities if row.merge_type == "CrossState"), key=lambda row: row.total_distance)[:10]]
    lines += ["", "4. Most unsafe merges"] + [f"{node_text(row.node_a)} | {node_text(row.node_b)} | {row.reason}" for row in result.unsafe[:10]]
    lines += ["", "5. Best compression schemes"] + [f"{row.scheme} | Efficiency={fmt(row.efficiency)}" for row in sorted(result.schemes.values(), key=lambda row: -row.efficiency)]
    lines += ["", "6. Worst compression schemes"] + [f"{row.scheme} | InformationLoss={fmt(row.information_loss)}" for row in sorted(result.schemes.values(), key=lambda row: -row.information_loss)]
    state_counts = Counter(row.node_a[0] for row in result.similarities if row.similarity_class != "NoMergeCandidate")
    lines += ["", "7. Most compressible states"] + [f"{state} | CandidatePairs={count}" for state, count in state_counts.most_common(10)]
    lines += ["", "8. Least compressible states"] + [f"{state} | CandidatePairs={count}" for state, count in sorted(state_counts.items(), key=lambda item: (item[1], item[0]))[:10]]
    lines += ["", "9. Best low-count absorption candidates"] + [f"{node_text(row.node)} -> {node_text(row.nearest)} | {fmt(row.distance)}" for row in result.low_count[:10]]
    lines += ["", "10. Recommended compressed APVA representation"] + recommendation(result)
    return lines


def validate(result: Result) -> None:
    if not result.profiles:
        raise RuntimeError(f"{result.instrument}: no node profiles.")
    if "SchemeA_NoCompression" not in result.schemes:
        raise RuntimeError(f"{result.instrument}: missing baseline scheme.")
    for row in result.schemes.values():
        if row.compressed_keys > row.original_keys:
            raise RuntimeError(f"{result.instrument}: compression increased keys.")
        if not 0 <= row.key_reduction <= 1:
            raise RuntimeError(f"{result.instrument}: invalid key reduction.")


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
    aggregate_nodes = aggregate_rows(decay)
    score_rows(aggregate_nodes)
    low, high = thresholds(loaded, aggregate_nodes)
    aggregate_bars, aggregate_paths, aggregate_node_stream, aggregate_branch_rows = [], [], [], []
    instrument_results = []
    for loaded_row, decay_row in zip(loaded, decay):
        offset = len(aggregate_bars)
        aggregate_segment = build_stream(loaded_row, aggregate_nodes, (low, high), (0.0, 0.0), (0.0, 0.0), offset)
        aggregate_branch_rows.extend(aggregate_segment)
        aggregate_node_stream.extend(node_for(bar) for bar in loaded_row.bars)
        aggregate_bars.extend(loaded_row.bars)
        aggregate_paths.extend(loaded_row.source_paths)

        local = local_rows(decay_row)
        score_rows(local)
        dynamics = build_dynamics(loaded_row, local, low, high)
        local_segment = build_stream(loaded_row, local, (low, high), (0.0, 0.0), (0.0, 0.0))
        result = build_result(loaded_row.instrument, loaded_row.source_paths, loaded_row.bars, dynamics.nodes, local_segment, local)
        validate(result)
        instrument_results.append(result)

    aggregate_result = build_result("Aggregate", aggregate_paths, aggregate_bars, aggregate_node_stream, aggregate_branch_rows, aggregate_nodes)
    validate(aggregate_result)
    out_root = Path(args.out_root)
    for result in instrument_results:
        write_per_instrument(result, out_root)
    write_aggregate(aggregate_result, instrument_results, out_root)
    print(f"Wrote {len(instrument_results)} per-instrument report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
