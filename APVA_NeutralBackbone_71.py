#!/usr/bin/env python3
"""APVA Neutral Backbone Study v0.1.

Audit whether NeutralProcessing StateAge nodes form the structural backbone.
Forward price outcomes are diagnostics only.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable

from APVA_BranchForecast_65 import build_stream
from APVA_InformationContribution_63 import thresholds
from APVA_InformationDecay_57 import study as decay_study
from APVA_MinimalEngine_52 import directional_return, instrument_columns, load_results, safe_mean
from APVA_NodeImportance_58 import aggregate_rows, local_rows, score_rows
from APVA_NodeNecessity_70 import Metrics, baseline_metrics, build_result as necessity_result, refresh_replication
from APVA_NodeNecessity_70 import validate as validate_necessity
from APVA_ProcessGraph_54 import Node, node_for, node_text
from APVA_StructuralForecast_56 import Outcome, outcome
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

NEUTRAL_NODES = [
    ("NeutralProcessing", "1"),
    ("NeutralProcessing", "2"),
    ("NeutralProcessing", "3"),
    ("NeutralProcessing", "4"),
    ("NeutralProcessing", "5"),
    ("NeutralProcessing", "6-10"),
    ("NeutralProcessing", "11-20"),
    ("NeutralProcessing", "21+"),
]
NEUTRAL_CHAIN = list(zip(NEUTRAL_NODES, NEUTRAL_NODES[1:]))
RETURN_FAMILIES = (
    "Compression", "Recovery", "Mixed", "Reassertion", "Decay",
    "Exhaustion", "Constructive", "Destructive",
)


@dataclass
class NeutralInventory:
    node: Node
    count: int
    occupancy: float
    replication_count: int
    necessity: float
    branch_entropy: float
    memory_strength: float
    memory_contribution: float
    branch_contribution: float


@dataclass
class TransitionRow:
    source: str
    target: str
    count: int
    probability: float
    replication_count: int = 1


@dataclass
class ReturnRow:
    family: str
    within_1: float
    within_2: float
    within_3: float
    within_5: float


@dataclass
class ResidenceRow:
    node: Node
    mean_length: float
    median_length: float
    max_length: int


@dataclass
class MotifRow:
    motif: tuple[str, str, str]
    count: int
    probability: float
    contribution: float
    replication_count: int = 1


@dataclass
class HubRow:
    node: Node
    incoming: int
    outgoing: int
    branch_factor: int


@dataclass
class ExcursionSummary:
    count: int
    mean_length: float
    median_length: float
    max_length: int
    return_rate: float


@dataclass
class BackboneTest:
    full: Metrics
    backbone: Metrics
    original_nodes: int
    backbone_nodes: int
    reduction: float
    forecast_loss: float


@dataclass
class Result:
    instrument: str
    source_paths: list
    bars: list
    rows: list
    nodes: list[Node]
    necessity: object
    inventory: dict[Node, NeutralInventory]
    occupancy: float
    entries: dict[tuple[str, str], TransitionRow]
    exits: dict[tuple[str, str], TransitionRow]
    progression: dict[str, TransitionRow]
    returns: dict[str, ReturnRow]
    residence: dict[Node, ResidenceRow]
    motifs: dict[tuple[str, str, str], MotifRow]
    hubs: dict[Node, HubRow]
    excursions: ExcursionSummary
    backbone: BackboneTest
    outcomes: dict[Node, Outcome]


def mean(values: Iterable[float]) -> float:
    return safe_mean(list(values))


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def is_neutral_node(node: Node | str) -> bool:
    if isinstance(node, tuple):
        return node[0] == "NeutralProcessing"
    return node.startswith("NeutralProcessing_Age")


def family_for_state(state: str) -> str:
    for family in RETURN_FAMILIES:
        if state.startswith(family):
            return family
    return state.replace("Processing", "").replace("Structure", "")


def node_from_text(value: str) -> Node:
    state, age = value.rsplit("_Age", 1)
    return state, age


def distribution_metrics(keys: list[Hashable], targets: list[str]) -> tuple[float, float, float, float, int, float]:
    if not keys or not targets:
        return 0.0, 0.0, 0.0, 0.0, 0, 0.0
    universe = tuple(sorted(set(targets)))
    grouped: dict[Hashable, Counter[str]] = defaultdict(Counter)
    for key, target in zip(keys, targets):
        grouped[key][target] += 1
    counts = Counter(keys)
    top1s, top2s, briers, entropies = [], [], [], []
    for key, actual in zip(keys, targets):
        total = sum(grouped[key].values())
        dist = {item: grouped[key][item] / total for item in universe}
        ranked = sorted(dist.items(), key=lambda item: (-item[1], item[0]))
        choices = [item for item, _ in ranked]
        top1s.append(float(actual in choices[:1]))
        top2s.append(float(actual in choices[:2]))
        briers.append(sum((dist.get(item, 0.0) - float(item == actual)) ** 2 for item in universe))
        entropies.append(-sum(value * __import__("math").log(value) for value in dist.values() if value > 0))
    return mean(top1s), mean(top2s), mean(briers), mean(entropies), len(counts), mean(count < 50 for count in counts.values())


def entropy_by_node(rows: list) -> dict[Node, float]:
    grouped: dict[Node, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[row.node][row.next_node] += 1
    output = {}
    for node, counts in grouped.items():
        total = sum(counts.values())
        probs = [count / total for count in counts.values()] if total else []
        output[node] = -sum(value * __import__("math").log(value) for value in probs if value > 0)
    return output


def neutral_inventory(rows: list, necessity, node_rows: dict[Node, object]) -> dict[Node, NeutralInventory]:
    counts = Counter(row.node for row in rows)
    entropies = entropy_by_node(rows)
    total = len(rows)
    output = {}
    for node in NEUTRAL_NODES:
        impact = necessity.impacts.get(node)
        node_row = node_rows.get(node)
        output[node] = NeutralInventory(
            node=node,
            count=counts.get(node, 0),
            occupancy=counts.get(node, 0) / total if total else 0.0,
            replication_count=impact.replication_count if impact else 0,
            necessity=impact.necessity_score if impact else 0.0,
            branch_entropy=entropies.get(node, 0.0),
            memory_strength=getattr(node_row, "memory_strength", 0.0) if node_row else 0.0,
            memory_contribution=impact.memory_contribution if impact else 0.0,
            branch_contribution=impact.branch_contribution if impact else 0.0,
        )
    return output


def transition_table(rows: list, mode: str) -> dict[tuple[str, str], TransitionRow]:
    totals: Counter[str] = Counter()
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        source = node_text(row.node)
        target = row.next_node
        source_neutral = is_neutral_node(row.node)
        target_neutral = is_neutral_node(target)
        if mode == "entry" and not source_neutral and target_neutral:
            totals[source] += 1
            counts[(source, target)] += 1
        elif mode == "exit" and source_neutral and not target_neutral:
            totals[source] += 1
            counts[(source, target)] += 1
    return {
        key: TransitionRow(key[0], key[1], count, count / totals[key[0]] if totals[key[0]] else 0.0)
        for key, count in counts.items()
    }


def progression_table(rows: list) -> dict[str, TransitionRow]:
    counts = Counter((row.node, node_from_text(row.next_node)) for row in rows)
    totals = Counter(row.node for row in rows)
    output = {}
    for source, target in NEUTRAL_CHAIN:
        key = f"{node_text(source)} -> {node_text(target)}"
        count = counts.get((source, target), 0)
        output[key] = TransitionRow(node_text(source), node_text(target), count, count / totals[source] if totals[source] else 0.0)
    return output


def return_table(nodes: list[Node]) -> dict[str, ReturnRow]:
    by_family = {family: {1: [], 2: [], 3: [], 5: []} for family in RETURN_FAMILIES}
    for index, node in enumerate(nodes):
        if is_neutral_node(node):
            continue
        family = family_for_state(node[0])
        if family not in by_family:
            continue
        for horizon in (1, 2, 3, 5):
            hit = any(index + step < len(nodes) and is_neutral_node(nodes[index + step]) for step in range(1, horizon + 1))
            by_family[family][horizon].append(float(hit))
    return {
        family: ReturnRow(family, mean(values[1]), mean(values[2]), mean(values[3]), mean(values[5]))
        for family, values in by_family.items()
    }


def residence_table(nodes: list[Node]) -> dict[Node, ResidenceRow]:
    runs: dict[Node, list[int]] = defaultdict(list)
    index = 0
    while index < len(nodes):
        node = nodes[index]
        length = 1
        while index + length < len(nodes) and nodes[index + length] == node:
            length += 1
        if is_neutral_node(node):
            runs[node].append(length)
        index += length
    return {node: ResidenceRow(node, mean(lengths), median(lengths), max(lengths) if lengths else 0) for node, lengths in runs.items()}


def motif_table(rows: list, neutral_only: bool = True) -> dict[tuple[str, str, str], MotifRow]:
    counts: Counter[tuple[str, str, str]] = Counter()
    totals: Counter[tuple[str, str]] = Counter()
    for row in rows:
        key = (row.previous, node_text(row.node), row.next_node)
        if neutral_only and not any(is_neutral_node(value) for value in key):
            continue
        counts[key] += 1
        totals[(key[0], key[1])] += 1
    output = {}
    for key, count in counts.items():
        total = totals[(key[0], key[1])]
        output[key] = MotifRow(key, count, count / total if total else 0.0, 0.0)
    return output


def attach_motif_contribution(rows: list, motifs: dict[tuple[str, str, str], MotifRow]) -> None:
    if not rows:
        return
    total = len(rows)
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[(row.previous, node_text(row.node))][row.next_node] += 1
    baseline_correct = sum(max(counter.values()) for counter in grouped.values())
    baseline_top1 = baseline_correct / total if total else 0.0
    for key, motif in motifs.items():
        prefix = (key[0], key[1])
        counter = Counter(grouped[prefix])
        before = max(counter.values()) if counter else 0
        counter[key[2]] -= motif.count
        if counter[key[2]] <= 0:
            del counter[key[2]]
        after = max(counter.values()) if counter else 0
        motif.contribution = baseline_top1 - ((baseline_correct - before + after) / total if total else 0.0)


def hub_table(rows: list) -> dict[Node, HubRow]:
    incoming: dict[Node, set[str]] = defaultdict(set)
    outgoing: dict[Node, set[str]] = defaultdict(set)
    for row in rows:
        if is_neutral_node(row.node):
            incoming[row.node].add(row.previous)
            outgoing[row.node].add(row.next_node)
    return {
        node: HubRow(node, len(incoming[node]), len(outgoing[node]), len(incoming[node]) * len(outgoing[node]))
        for node in sorted(set(incoming) | set(outgoing), key=node_text)
    }


def excursion_summary(nodes: list[Node]) -> ExcursionSummary:
    lengths, returned = [], 0
    index = 0
    while index < len(nodes) - 1:
        if is_neutral_node(nodes[index]) and not is_neutral_node(nodes[index + 1]):
            start = index + 1
            end = start
            while end < len(nodes) and not is_neutral_node(nodes[end]):
                end += 1
            length = end - start
            lengths.append(length)
            if end < len(nodes) and is_neutral_node(nodes[end]):
                returned += 1
            index = max(end, start)
        else:
            index += 1
    return ExcursionSummary(len(lengths), mean(lengths), median(lengths), max(lengths) if lengths else 0, returned / len(lengths) if lengths else 0.0)


def backbone_label(node: Node) -> str:
    if is_neutral_node(node):
        return node_text(node)
    return f"Excursion_{family_for_state(node[0])}"


def backbone_target(value: str) -> str:
    node = node_from_text(value)
    return backbone_label(node)


def backbone_test(rows: list, necessity) -> BackboneTest:
    full = baseline_metrics(rows)
    mem_keys = [backbone_label(row.node) for row in rows]
    mem_targets = [row.target for row in rows]
    branch_keys = [backbone_label(row.node) for row in rows]
    branch_targets = [backbone_target(row.next_node) for row in rows]
    mt1, mt2, mbrier, mentropy, mkeys, msparse = distribution_metrics(mem_keys, mem_targets)
    bt1, bt2, bbrier, bentropy, bkeys, bsparse = distribution_metrics(branch_keys, branch_targets)
    backbone = Metrics(mt1, mt2, bt1, bt2, 0.0, mbrier, bbrier, mean((mentropy, bentropy)), max(mkeys, bkeys), mean((msparse, bsparse)))
    original = len(necessity.impacts)
    backbone_nodes = len(set(branch_keys))
    reduction = 1 - backbone_nodes / original if original else 0.0
    forecast_loss = mean((max(0.0, full.memory_top1 - backbone.memory_top1), max(0.0, full.branch_top1 - backbone.branch_top1)))
    return BackboneTest(full, backbone, original, backbone_nodes, reduction, forecast_loss)


def outcome_rows(bars: list, nodes: list[Node]) -> dict[Node, Outcome]:
    grouped: dict[Node, list[float]] = defaultdict(list)
    for index, node in enumerate(nodes):
        if not is_neutral_node(node):
            continue
        value = directional_return(bars, index, 5)
        if value is not None:
            grouped[node].append(value)
    return {node: outcome(values) for node, values in grouped.items()}


def build_result(instrument: str, source_paths: list, bars: list, rows: list, nodes: list[Node],
                 node_rows: dict[Node, object]) -> Result:
    necessity = necessity_result(instrument, source_paths, bars, rows, nodes, node_rows)
    validate_necessity(necessity)
    motifs = motif_table(rows)
    attach_motif_contribution(rows, motifs)
    return Result(
        instrument, source_paths, bars, rows, nodes, necessity,
        neutral_inventory(rows, necessity, node_rows),
        mean(is_neutral_node(row.node) for row in rows),
        transition_table(rows, "entry"),
        transition_table(rows, "exit"),
        progression_table(rows),
        return_table([row.node for row in rows]),
        residence_table([row.node for row in rows]),
        motifs,
        hub_table(rows),
        excursion_summary([row.node for row in rows]),
        backbone_test(rows, necessity),
        outcome_rows(bars, nodes),
    )


def attach_replication(aggregate: Result, instruments: list[Result]) -> None:
    refresh_replication(aggregate.necessity, [result.necessity for result in instruments])
    for node, row in aggregate.inventory.items():
        row.replication_count = sum(node in result.inventory and result.inventory[node].count > 0 for result in instruments)
        impact = aggregate.necessity.impacts.get(node)
        if impact:
            row.necessity = impact.necessity_score
    for table_name in ("entries", "exits", "progression", "motifs"):
        table = getattr(aggregate, table_name)
        for key, row in table.items():
            row.replication_count = sum(key in getattr(result, table_name) for result in instruments)


def inv_line(row: NeutralInventory) -> str:
    return (f"{node_text(row.node)} | {row.count} | {pct(row.occupancy)} | {row.replication_count} | "
            f"{fmt(row.necessity)} | {fmt(row.branch_entropy)} | {fmt(row.memory_strength)}")


def trans_line(row: TransitionRow) -> str:
    return f"{row.source} | {row.target} | {row.count} | {pct(row.probability)} | {row.replication_count}"


def motif_text(key: tuple[str, str, str]) -> str:
    return f"{key[0]} -> {key[1]} -> {key[2]}"


def append_common(lines: list[str], result: Result, limit: int = 60) -> None:
    lines += ["", "1. Neutral Inventory", "Node | Count | PercentOfAllRows | ReplicationCount | NecessityScore | BranchEntropy | MemoryStrength"]
    lines += [inv_line(row) for row in result.inventory.values()]
    lines += ["", "2. Occupancy", f"NeutralOccupancy | {pct(result.occupancy)}"]
    lines += ["", "3. Entry Analysis", "NonNeutralNode | NeutralTarget | Count | Probability | ReplicationCount"]
    lines += [trans_line(row) for row in sorted(result.entries.values(), key=lambda row: (-row.count, row.source))[:limit]]
    lines += ["", "4. Exit Analysis", "NeutralNode | NonNeutralTarget | Count | Probability | ReplicationCount"]
    lines += [trans_line(row) for row in sorted(result.exits.values(), key=lambda row: (-row.count, row.source))[:limit]]
    lines += ["", "5. Progression Chain", "Transition | Count | Probability | ReplicationCount"]
    lines += [f"{key} | {row.count} | {pct(row.probability)} | {row.replication_count}" for key, row in result.progression.items()]
    lines += ["", "6. Return Analysis", "Family | ReturnWithin1 | ReturnWithin2 | ReturnWithin3 | ReturnWithin5"]
    lines += [f"{row.family} | {pct(row.within_1)} | {pct(row.within_2)} | {pct(row.within_3)} | {pct(row.within_5)}" for row in result.returns.values()]
    lines += ["", "7. Residence Time", "NeutralNode | MeanResidence | MedianResidence | MaxResidence"]
    lines += [f"{node_text(row.node)} | {fmt(row.mean_length)} | {fmt(row.median_length)} | {row.max_length}" for row in result.residence.values()]
    lines += ["", "8. Necessity", "Node | NecessityScore | MemoryContribution | BranchContribution"]
    lines += [f"{node_text(row.node)} | {fmt(row.necessity)} | {fmt(row.memory_contribution)} | {fmt(row.branch_contribution)}" for row in result.inventory.values()]
    lines += ["", "9. Memory Contribution", "Node | MemoryTop1Contribution | MemoryBrierContribution"]
    for node, impact in sorted(result.necessity.impacts.items(), key=lambda item: (-item[1].memory_contribution, node_text(item[0]))):
        if is_neutral_node(node):
            lines.append(f"{node_text(node)} | {pct(max(0.0, -impact.memory_top1_change))} | {fmt(max(0.0, impact.memory_brier_change))}")
    lines += ["", "10. Branch Contribution", "Node | BranchTop1Contribution | BranchBrierContribution"]
    for node, impact in sorted(result.necessity.impacts.items(), key=lambda item: (-item[1].branch_contribution, node_text(item[0]))):
        if is_neutral_node(node):
            lines.append(f"{node_text(node)} | {pct(max(0.0, -impact.branch_top1_change))} | {fmt(max(0.0, impact.branch_brier_change))}")
    lines += ["", "11. Motifs", "Motif | Frequency | Replication | Contribution"]
    lines += [f"{motif_text(key)} | {row.count} | {row.replication_count} | {fmt(row.contribution)}" for key, row in sorted(result.motifs.items(), key=lambda item: (-item[1].count, item[0]))[:limit]]
    lines += ["", "12. Hub Analysis", "NeutralNode | IncomingNodeCount | OutgoingNodeCount | BranchFactor"]
    lines += [f"{node_text(row.node)} | {row.incoming} | {row.outgoing} | {row.branch_factor}" for row in sorted(result.hubs.values(), key=lambda row: (-row.branch_factor, node_text(row.node)))]
    lines += ["", "13. Excursions", f"MeanLength | {fmt(result.excursions.mean_length)}", f"MedianLength | {fmt(result.excursions.median_length)}", f"MaxLength | {result.excursions.max_length}", f"ReturnRate | {pct(result.excursions.return_rate)}"]
    lines += ["", "14. Backbone Test", "Model | Top1 | Top2 | Brier | Entropy"]
    lines += [f"FullAPVAGraph_Memory | {pct(result.backbone.full.memory_top1)} | {pct(result.backbone.full.memory_top2)} | {fmt(result.backbone.full.memory_brier)} | {fmt(result.backbone.full.entropy)}"]
    lines += [f"NeutralBackbone_Memory | {pct(result.backbone.backbone.memory_top1)} | {pct(result.backbone.backbone.memory_top2)} | {fmt(result.backbone.backbone.memory_brier)} | {fmt(result.backbone.backbone.entropy)}"]
    lines += [f"FullAPVAGraph_Branch | {pct(result.backbone.full.branch_top1)} | {pct(result.backbone.full.branch_top2)} | {fmt(result.backbone.full.branch_brier)} | {fmt(result.backbone.full.entropy)}"]
    lines += [f"NeutralBackbone_Branch | {pct(result.backbone.backbone.branch_top1)} | {pct(result.backbone.backbone.branch_top2)} | {fmt(result.backbone.backbone.branch_brier)} | {fmt(result.backbone.backbone.entropy)}"]
    lines += ["", "15. Compression Ratio", f"OriginalNodeCount | {result.backbone.original_nodes}", f"BackboneNodeCount | {result.backbone.backbone_nodes}", f"ReductionPercent | {pct(result.backbone.reduction)}", f"ForecastLoss | {pct(result.backbone.forecast_loss)}"]
    lines += ["", "16. Replication", "Node | Count | ReplicationCount"]
    lines += [f"{node_text(row.node)} | {row.count} | {row.replication_count}" for row in result.inventory.values()]
    lines += ["", "17. Outcome Diagnostics", "NeutralNode | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5"]
    for node, out in sorted(result.outcomes.items(), key=lambda item: (-item[1].count, node_text(item[0]))):
        lines.append(f"{node_text(node)} | {out.count} | {fmt(out.mean_dr)} | {fmt(out.median_dr)} | {pct(out.continuation)} | {pct(out.failure)} | {pct(out.flat)}")
    lines += ["", "18. Recommendation"] + recommendation(result)
    append_audit(lines)
    lines += ["", "20. Mechanical Research Notes", "Neutral backbone scoring uses structural occupancy, necessity, transition, motif, return, and compressed graph retention only."]


def recommendation(result: Result) -> list[str]:
    neutral_necessity = mean(row.necessity for row in result.inventory.values())
    forecast_retention = 1 - result.backbone.forecast_loss
    if result.occupancy >= 0.60 and forecast_retention >= 0.90 and result.excursions.return_rate >= 0.70:
        label = "BackboneDominant"
    elif result.occupancy >= 0.45 and forecast_retention >= 0.85:
        label = "StrongBackbone"
    elif result.occupancy >= 0.30 and forecast_retention >= 0.75:
        label = "ModerateBackbone"
    else:
        label = "WeakBackbone"
    return [
        f"Classification: {label}",
        f"NeutralOccupancy: {pct(result.occupancy)}",
        f"NeutralNecessity: {fmt(neutral_necessity)}",
        f"NeutralContribution: {fmt(mean(max(0.0, row.memory_contribution) + max(0.0, row.branch_contribution) for row in result.inventory.values()))}",
        f"BackboneCompressionRatio: {pct(result.backbone.reduction)}",
        f"ForecastRetention: {pct(forecast_retention)}",
        "Reason: Classification uses fixed descriptive thresholds on occupancy, return behavior, and forecast retention after neutral-backbone compression.",
    ]


def append_audit(lines: list[str]) -> None:
    lines += [
        "",
        "19. Low-DoF Audit",
        "Variables used:",
        "StateAgeNode",
        "StructuralState",
        "AgeBucket",
        "PreviousNode",
        "CurrentNode",
        "NextNode",
        "BranchProbability",
        "BranchEntropy",
        "NecessityScore",
        "MemoryStrength",
        "",
        "No Context",
        "No Arbitration",
        "No Persistence",
        "No Phase",
        "No Optimization",
        "No Fitting",
        "No Machine Learning",
        "No Forward Returns used in backbone scoring",
    ]


def write_per_instrument(result: Result, out_root: Path) -> None:
    path = out_root / result.instrument / f"NeutralBackbone_{result.instrument}.txt"
    ensure_dir(path.parent)
    lines = [
        "APVA Neutral Backbone Study v0.1",
        "=" * 108,
        f"Instrument: {result.instrument}",
        "Input path(s): " + ", ".join(str(path) for path in result.source_paths),
        f"Total rows: {len(result.bars)}",
    ]
    append_common(lines, result)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_outcome_lines(results: list[Result], instruments: list[str]) -> list[str]:
    by_name = {result.instrument: result for result in results}
    lines = ["", "Aggregate Outcome Table", "Node | " + " | ".join(f"Count_{name} | MeanDR_{name}" for name in instruments) + " | ValidInstrumentCount"]
    for node in NEUTRAL_NODES:
        cells, valid = [node_text(node)], 0
        for name in instruments:
            row = by_name[name].outcomes.get(node)
            if row:
                valid += 1
                cells += [str(row.count), fmt(row.mean_dr)]
            else:
                cells += ["0", "0.0000"]
        cells.append(str(valid))
        lines.append(" | ".join(cells))
    return lines


def write_aggregate(result: Result, instrument_results: list[Result], out_root: Path) -> None:
    attach_replication(result, instrument_results)
    path = out_root / "NeutralBackbone" / "NeutralBackbone_All.txt"
    ensure_dir(path.parent)
    instruments = instrument_columns(instrument_results)
    by_name = {row.instrument: row for row in instrument_results}
    lines = ["APVA Neutral Backbone Study v0.1 - Aggregate", "=" * 108, "Instruments: " + ", ".join(instruments)]
    lines += ["", "Aggregate Neutral Inventory", "Node | Count | Occupancy | Necessity | BranchEntropy | MemoryStrength"]
    lines += [f"{node_text(row.node)} | {row.count} | {pct(row.occupancy)} | {fmt(row.necessity)} | {fmt(row.branch_entropy)} | {fmt(row.memory_strength)}" for row in result.inventory.values()]
    lines += ["", "Aggregate Entry Table", "NonNeutralNode | NeutralTarget | Count | Probability | ReplicationCount"]
    lines += [trans_line(row) for row in sorted(result.entries.values(), key=lambda row: (-row.count, row.source))[:150]]
    lines += ["", "Aggregate Exit Table", "NeutralNode | NonNeutralTarget | Count | Probability | ReplicationCount"]
    lines += [trans_line(row) for row in sorted(result.exits.values(), key=lambda row: (-row.count, row.source))[:150]]
    lines += ["", "Aggregate Progression Table", "Transition | Count | Probability | ReplicationCount"]
    lines += [f"{key} | {row.count} | {pct(row.probability)} | {row.replication_count}" for key, row in result.progression.items()]
    lines += ["", "Aggregate Return Table", "Family | ReturnWithin1 | ReturnWithin2 | ReturnWithin3 | ReturnWithin5"]
    lines += [f"{row.family} | {pct(row.within_1)} | {pct(row.within_2)} | {pct(row.within_3)} | {pct(row.within_5)}" for row in result.returns.values()]
    lines += ["", "Aggregate Residence Table", "NeutralNode | MeanResidence | MedianResidence | MaxResidence"]
    lines += [f"{node_text(row.node)} | {fmt(row.mean_length)} | {fmt(row.median_length)} | {row.max_length}" for row in result.residence.values()]
    lines += ["", "Aggregate Necessity Table", "Node | NecessityScore | MemoryContribution | BranchContribution"]
    lines += [f"{node_text(row.node)} | {fmt(row.necessity)} | {fmt(row.memory_contribution)} | {fmt(row.branch_contribution)}" for row in result.inventory.values()]
    lines += ["", "Aggregate Motif Table", "Motif | Frequency | Replication | Contribution"]
    lines += [f"{motif_text(key)} | {row.count} | {row.replication_count} | {fmt(row.contribution)}" for key, row in sorted(result.motifs.items(), key=lambda item: (-item[1].count, item[0]))[:150]]
    lines += ["", "Aggregate Hub Table", "NeutralNode | IncomingCount | OutgoingCount | BranchFactor"]
    lines += [f"{node_text(row.node)} | {row.incoming} | {row.outgoing} | {row.branch_factor}" for row in sorted(result.hubs.values(), key=lambda row: (-row.branch_factor, node_text(row.node)))]
    lines += ["", "Aggregate Excursion Table", f"MeanLength | {fmt(result.excursions.mean_length)}", f"MedianLength | {fmt(result.excursions.median_length)}", f"MaxLength | {result.excursions.max_length}", f"ReturnRate | {pct(result.excursions.return_rate)}"]
    lines += ["", "Aggregate Backbone Test", "Model | Top1 | Top2 | Brier | Entropy"]
    lines += [f"FullAPVAGraph_Memory | {pct(result.backbone.full.memory_top1)} | {pct(result.backbone.full.memory_top2)} | {fmt(result.backbone.full.memory_brier)} | {fmt(result.backbone.full.entropy)}",
              f"NeutralBackbone_Memory | {pct(result.backbone.backbone.memory_top1)} | {pct(result.backbone.backbone.memory_top2)} | {fmt(result.backbone.backbone.memory_brier)} | {fmt(result.backbone.backbone.entropy)}",
              f"FullAPVAGraph_Branch | {pct(result.backbone.full.branch_top1)} | {pct(result.backbone.full.branch_top2)} | {fmt(result.backbone.full.branch_brier)} | {fmt(result.backbone.full.entropy)}",
              f"NeutralBackbone_Branch | {pct(result.backbone.backbone.branch_top1)} | {pct(result.backbone.backbone.branch_top2)} | {fmt(result.backbone.backbone.branch_brier)} | {fmt(result.backbone.backbone.entropy)}"]
    lines += ["", "Aggregate Compression Table", f"OriginalNodes | {result.backbone.original_nodes}", f"BackboneNodes | {result.backbone.backbone_nodes}", f"ReductionPercent | {pct(result.backbone.reduction)}", f"ForecastLoss | {pct(result.backbone.forecast_loss)}"]
    lines += ["", "Aggregate Replication Table", "Node | " + " | ".join(f"Count_{name}" for name in instruments) + " | ReplicationCount"]
    for node in NEUTRAL_NODES:
        counts = [by_name[name].inventory[node].count for name in instruments]
        replication = sum(count > 0 for count in counts)
        lines.append(f"{node_text(node)} | " + " | ".join(str(count) for count in counts) + f" | {replication}")
    lines += aggregate_outcome_lines(instrument_results, instruments)
    lines += ["", "Aggregate Recommendation"] + recommendation(result)
    lines += rankings(result)
    lines += ["", "RESEARCH NOTES", "Mechanical only.", "Is Neutral the structural backbone of APVA?", "Do excursions revolve around Neutral?", "Can APVA be simplified around a Neutral lifecycle?", "Does the Neutral backbone explain most of the graph's predictive power?"]
    append_audit(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rankings(result: Result) -> list[str]:
    inv = list(result.inventory.values())
    lines = ["", "AGGREGATE RANKINGS"]
    lines += ["", "1. Most occupied Neutral nodes"] + [f"{node_text(row.node)} | {row.count} | {pct(row.occupancy)}" for row in sorted(inv, key=lambda row: (-row.count, node_text(row.node)))]
    lines += ["", "2. Most necessary Neutral nodes"] + [f"{node_text(row.node)} | {fmt(row.necessity)}" for row in sorted(inv, key=lambda row: (-row.necessity, node_text(row.node)))]
    lines += ["", "3. Strongest Neutral motifs"] + [f"{motif_text(key)} | {row.count} | {fmt(row.contribution)}" for key, row in sorted(result.motifs.items(), key=lambda item: (-item[1].contribution, -item[1].count))[:10]]
    lines += ["", "4. Largest Neutral feeders"] + [trans_line(row) for row in sorted(result.entries.values(), key=lambda row: (-row.count, row.source))[:10]]
    lines += ["", "5. Largest Neutral exits"] + [trans_line(row) for row in sorted(result.exits.values(), key=lambda row: (-row.count, row.source))[:10]]
    lines += ["", "6. Highest return-to-Neutral families"] + [f"{row.family} | {pct(row.within_5)}" for row in sorted(result.returns.values(), key=lambda row: (-row.within_5, row.family))]
    lines += ["", "7. Longest Neutral residence nodes"] + [f"{node_text(row.node)} | {fmt(row.mean_length)}" for row in sorted(result.residence.values(), key=lambda row: (-row.mean_length, node_text(row.node)))]
    lines += ["", "8. Strongest Neutral hubs"] + [f"{node_text(row.node)} | {row.branch_factor}" for row in sorted(result.hubs.values(), key=lambda row: (-row.branch_factor, node_text(row.node)))]
    lines += ["", "9. Highest Neutral contributions"] + [f"{node_text(row.node)} | {fmt(row.memory_contribution + row.branch_contribution)}" for row in sorted(inv, key=lambda row: (-(row.memory_contribution + row.branch_contribution), node_text(row.node)))]
    lines += ["", "10. Recommended APVA backbone representation"] + recommendation(result)
    return lines


def validate(result: Result) -> None:
    if not result.rows:
        raise RuntimeError(f"{result.instrument}: no stream rows.")
    if not result.inventory:
        raise RuntimeError(f"{result.instrument}: missing neutral inventory.")
    if not 0 <= result.occupancy <= 1:
        raise RuntimeError(f"{result.instrument}: invalid occupancy.")
    if result.backbone.backbone_nodes > result.backbone.original_nodes:
        raise RuntimeError(f"{result.instrument}: backbone node count exceeds original.")


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
    memory_thresholds = thresholds(loaded, aggregate_node_rows)

    instrument_results = []
    aggregate_bars, aggregate_paths, aggregate_nodes, aggregate_stream, aggregate_segments = [], [], [], [], []
    for loaded_row, decay_row in zip(loaded, decay):
        local_node_rows = local_rows(decay_row)
        score_rows(local_node_rows)
        stream = build_stream(loaded_row, local_node_rows, memory_thresholds, (0.0, 0.0), (0.0, 0.0))
        nodes = [node_for(bar) for bar in loaded_row.bars]
        result = build_result(loaded_row.instrument, loaded_row.source_paths, loaded_row.bars, stream, nodes, local_node_rows)
        validate(result)
        instrument_results.append(result)

        offset = len(aggregate_bars)
        segment = build_stream(loaded_row, aggregate_node_rows, memory_thresholds, (0.0, 0.0), (0.0, 0.0), offset)
        aggregate_stream.extend(segment)
        aggregate_segments.append(segment)
        aggregate_nodes.extend(nodes)
        aggregate_bars.extend(loaded_row.bars)
        aggregate_paths.extend(loaded_row.source_paths)

    aggregate = build_result("Aggregate", aggregate_paths, aggregate_bars, aggregate_stream, aggregate_nodes, aggregate_node_rows)
    validate(aggregate)
    out_root = Path(args.out_root)
    for result in instrument_results:
        write_per_instrument(result, out_root)
    write_aggregate(aggregate, instrument_results, out_root)
    print(f"Wrote {len(instrument_results)} per-instrument report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
