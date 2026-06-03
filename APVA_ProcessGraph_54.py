#!/usr/bin/env python3
"""APVA Process Graph Study v0.1.

Build a recurring structural process graph from the minimal APVA engine:
StructuralState + AgeBucket. Forward outcomes are diagnostics only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from APVA_MinimalEngine_52 import (
    MIN_COUNT,
    age_zone,
    directional_return,
    entropy,
    instrument_columns,
    load_results,
    normalized_entropy,
    outcome,
    safe_mean,
)
from APVA_StateTransitionModel_53 import destinations
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

DEPTHS = (2, 3, 4, 5)
MASS_LEVELS = (0.90, 0.95, 0.99)


Node = tuple[str, str]


def node_for(bar) -> Node:
    return bar.state, bar.age_bucket


def zone_node(node: Node) -> Node:
    return node[0], age_zone(node[1])


def node_text(node: Node) -> str:
    return f"{node[0]}_Age{node[1]}"


def zone_text(node: Node) -> str:
    return f"{node[0]}_{node[1]}"


def path_text(nodes: tuple[Node, ...], zone: bool = False) -> str:
    render = zone_text if zone else node_text
    return " -> ".join(render(node) for node in nodes)


def replication_label(count: int) -> str:
    return {1: "InstrumentSpecific", 2: "PartiallyReplicated", 3: "FullyReplicated"}.get(count, "Absent")


@dataclass
class Edge:
    source: Node
    destination: Node
    count: int
    probability: float


@dataclass
class NodeStats:
    node: Node
    count: int
    destinations: Counter[Node]
    dominant_destination: Node
    dominant_probability: float
    second_destination: Node | None
    second_probability: float
    confidence: float
    persistence: float
    pressure: float
    entropy: float
    normalized_entropy: float
    inbound_probability: float
    outbound_probability: float
    attractor_score: float
    inbound_count: int
    outbound_count: int
    hub_score: int
    terminal_score: float


@dataclass
class Chain:
    nodes: tuple[Node, ...]
    depth: int
    probability: float
    support: int


@dataclass
class Cycle:
    nodes: tuple[Node, ...]
    length: int
    probability: float
    support: int
    classification: str


@dataclass
class Compression:
    mass: float
    node_count: int
    edge_count: int
    compression_ratio: float


@dataclass
class GraphRecommendation:
    graph: str
    reason: str
    probability_mass_retention: float
    entropy_increase: float


@dataclass
class StudyResult:
    instrument: str
    source_paths: list[Path]
    bars: list
    nodes: Counter[Node]
    edges: dict[tuple[Node, Node], Edge]
    stats: dict[Node, NodeStats]
    chains: dict[tuple[Node, ...], Chain]
    cycles: dict[tuple[Node, ...], Cycle]
    zone_stats: dict[Node, NodeStats]
    zone_chains: dict[tuple[Node, ...], Chain]
    compressions: dict[float, Compression]
    node_outcomes: dict[Node, object]
    chain_outcomes: dict[tuple[Node, ...], object]
    cycle_outcomes: dict[tuple[Node, ...], object]
    recommendation: GraphRecommendation


def adjacent_groups(nodes: list[Node]) -> dict[Node, Counter[Node]]:
    groups: dict[Node, Counter[Node]] = defaultdict(Counter)
    for source, destination in zip(nodes, nodes[1:]):
        groups[source][destination] += 1
    return groups


def graph_edges(groups: dict[Node, Counter[Node]]) -> dict[tuple[Node, Node], Edge]:
    rows = {}
    for source, destinations_by_node in groups.items():
        total = sum(destinations_by_node.values())
        for destination, count in destinations_by_node.items():
            rows[(source, destination)] = Edge(source, destination, count, count / total)
    return rows


def weighted_probability(edges: dict[tuple[Node, Node], Edge], destination: Node) -> float:
    return sum(edge.probability for edge in edges.values() if edge.destination == destination)


def node_stats(nodes: list[Node]) -> tuple[Counter[Node], dict[tuple[Node, Node], Edge], dict[Node, NodeStats]]:
    counts = Counter(nodes)
    groups = adjacent_groups(nodes)
    edges = graph_edges(groups)
    inbound_sources: dict[Node, set[Node]] = defaultdict(set)
    for edge in edges.values():
        inbound_sources[edge.destination].add(edge.source)
    rows = {}
    for node, count in counts.items():
        outgoing = groups.get(node, Counter())
        ranked = sorted(outgoing.items(), key=lambda item: (-item[1], item[0]))
        total = sum(outgoing.values())
        dominant_destination, dominant_count = ranked[0] if ranked else (node, 0)
        second_destination, second_count = ranked[1] if len(ranked) > 1 else (None, 0)
        dominant_probability = dominant_count / total if total else 0.0
        second_probability = second_count / total if total else 0.0
        persistence = outgoing[node] / total if total else 0.0
        inbound_probability = weighted_probability(edges, node)
        outbound_probability = sum(edge.probability for edge in edges.values() if edge.source == node)
        pressure = 1.0 - persistence
        transition_entropy = entropy(outgoing.values())
        inbound_count = len(inbound_sources[node])
        outbound_count = len(outgoing)
        rows[node] = NodeStats(
            node, count, outgoing, dominant_destination, dominant_probability,
            second_destination, second_probability, dominant_probability - second_probability,
            persistence, pressure, transition_entropy, normalized_entropy(outgoing.values()),
            inbound_probability, outbound_probability, inbound_probability - outbound_probability,
            inbound_count, outbound_count, inbound_count + outbound_count,
            pressure * transition_entropy,
        )
    return counts, edges, rows


def dominant_chains(stats: dict[Node, NodeStats]) -> dict[tuple[Node, ...], Chain]:
    rows = {}
    for start in stats:
        current = start
        nodes = [start]
        probability = 1.0
        for depth in range(1, max(DEPTHS) + 1):
            row = stats.get(current)
            if not row or not row.destinations:
                break
            current = row.dominant_destination
            probability *= row.dominant_probability
            nodes.append(current)
            if depth in DEPTHS:
                key = tuple(nodes)
                rows[key] = Chain(key, depth, probability, 0)
    return rows


def canonical_cycle(nodes: tuple[Node, ...]) -> tuple[Node, ...]:
    rotations = [nodes[index:] + nodes[:index] for index in range(len(nodes))]
    return min(rotations)


def observed_cycles(nodes: list[Node], stats: dict[Node, NodeStats]) -> dict[tuple[Node, ...], Cycle]:
    cycles = {}
    candidates = set()
    for start_index, start in enumerate(nodes):
        path = [start]
        visited = {start}
        for current in nodes[start_index + 1:]:
            if current == start:
                if len(path) >= 2:
                    candidates.add(canonical_cycle(tuple(path)))
                break
            if current in visited:
                break
            visited.add(current)
            path.append(current)
    for key in candidates:
        support, _ = rolling_support(nodes, key, True)
        if support < 2:
            continue
        probability = 1.0
        for index, node in enumerate(key):
            destination = key[(index + 1) % len(key)]
            probability *= stats[node].destinations[destination] / sum(stats[node].destinations.values())
        length = len(key)
        label = "ShortCycle" if length == 2 else "MediumCycle" if length <= 5 else "LongCycle"
        cycles[key] = Cycle(key, length, probability, support, label)
    return cycles


def rolling_support(nodes: list[Node], sequence: tuple[Node, ...], cycle: bool = False) -> tuple[int, list[int]]:
    targets = [sequence]
    if cycle:
        targets = [
            sequence[index:] + sequence[:index] + (sequence[index],)
            for index in range(len(sequence))
        ]
    indexes = sorted({
        index
        for target in targets
        for index in range(len(nodes) - len(target) + 1)
        if tuple(nodes[index:index + len(target)]) == target
    })
    return len(indexes), indexes


def with_support(nodes: list[Node], chains: dict[tuple[Node, ...], Chain], cycles: dict[tuple[Node, ...], Cycle]):
    for key, chain in chains.items():
        chain.support, _ = rolling_support(nodes, key)
    for key, cycle in cycles.items():
        cycle.support, _ = rolling_support(nodes, key, True)


def compression(edges: dict[tuple[Node, Node], Edge]) -> dict[float, Compression]:
    ordered = sorted(edges.values(), key=lambda edge: (-edge.count, edge.source, edge.destination))
    total = sum(edge.count for edge in ordered)
    rows = {}
    for mass in MASS_LEVELS:
        retained = 0
        selected = []
        for edge in ordered:
            selected.append(edge)
            retained += edge.count
            if total and retained / total >= mass:
                break
        nodes = {node for edge in selected for node in (edge.source, edge.destination)}
        rows[mass] = Compression(mass, len(nodes), len(selected), len(selected) / len(edges) if edges else 0.0)
    return rows


def weighted_entropy(stats: dict[Node, NodeStats]) -> float:
    total = sum(row.count for row in stats.values())
    return sum(row.count * row.entropy for row in stats.values()) / total if total else 0.0


def weighted_dominant(stats: dict[Node, NodeStats]) -> float:
    total = sum(row.count for row in stats.values())
    return sum(row.count * row.dominant_probability for row in stats.values()) / total if total else 0.0


def recommend(stats: dict[Node, NodeStats], zone_stats: dict[Node, NodeStats], valid_instruments: int = 1) -> GraphRecommendation:
    full_entropy = weighted_entropy(stats)
    collapsed_entropy = weighted_entropy(zone_stats)
    entropy_increase = (collapsed_entropy - full_entropy) / full_entropy if full_entropy else 0.0
    full_mass = weighted_dominant(stats)
    collapsed_mass = weighted_dominant(zone_stats)
    retention = min(1.0, collapsed_mass / full_mass) if full_mass else 0.0
    graph = "YoungMiddleLateGraph" if retention >= 0.90 and entropy_increase <= 0.10 and valid_instruments >= 2 else "FullAgeBucketGraph"
    reason = f"ProbabilityMassRetention={pct(retention)}; MeanEntropyIncrease={pct(entropy_increase)}; ValidInstrumentCount={valid_instruments}."
    return GraphRecommendation(graph, reason, retention, entropy_increase)


def outcome_groups(bars: list, sequences: dict[tuple[Node, ...], object], cycle: bool = False) -> dict[tuple[Node, ...], object]:
    nodes = [node_for(bar) for bar in bars]
    rows = {}
    for key in sequences:
        _, indexes = rolling_support(nodes, key, cycle)
        values = [value for index in indexes if (value := directional_return(bars, index)) is not None]
        rows[key] = outcome(values)
    return rows


def study(result) -> StudyResult:
    bars = result.bars
    nodes = [node_for(bar) for bar in bars]
    counts, edges, stats = node_stats(nodes)
    chains = dominant_chains(stats)
    cycles = observed_cycles(nodes, stats)
    with_support(nodes, chains, cycles)
    zone_nodes = [zone_node(node) for node in nodes]
    _, _, zone_stats = node_stats(zone_nodes)
    zone_chains = dominant_chains(zone_stats)
    with_support(zone_nodes, zone_chains, {})
    node_values: dict[Node, list[float]] = defaultdict(list)
    for index, node in enumerate(nodes):
        value = directional_return(bars, index)
        if value is not None:
            node_values[node].append(value)
    return StudyResult(
        result.instrument, result.source_paths, bars, counts, edges, stats, chains, cycles,
        zone_stats, zone_chains, compression(edges),
        {key: outcome(values) for key, values in node_values.items()},
        outcome_groups(bars, chains), outcome_groups(bars, cycles, True),
        recommend(stats, zone_stats),
    )


def write_per_instrument(result: StudyResult, out_root: Path) -> None:
    path = out_root / result.instrument / f"ProcessGraph_{result.instrument}.txt"
    ensure_dir(path.parent)
    lines = [
        "APVA Process Graph Study v0.1",
        "=" * 92,
        "Diagnostics",
        f"Instrument: {result.instrument}",
        "Input path(s): " + ", ".join(str(item) for item in result.source_paths),
        f"Total rows: {len(result.bars)}",
        f"Node count: {len(result.nodes)}",
        f"Edge count: {len(result.edges)}",
        "",
        "1. Directed Graph Summary",
        "SourceNode | DestinationNode | TransitionCount | TransitionProbability",
    ]
    for edge in sorted(result.edges.values(), key=lambda item: (item.source, -item.probability, item.destination)):
        lines.append(f"{node_text(edge.source)} | {node_text(edge.destination)} | {edge.count} | {pct(edge.probability)}")
    lines += ["", "2. Dominant Edges", "Node | Count | DominantDestination | DominantProbability | SecondDestination | SecondProbability | TransitionConfidence"]
    for row in sorted(result.stats.values(), key=lambda item: item.node):
        second = node_text(row.second_destination) if row.second_destination else "N/A"
        lines.append(f"{node_text(row.node)} | {row.count} | {node_text(row.dominant_destination)} | {pct(row.dominant_probability)} | {second} | {pct(row.second_probability)} | {pct(row.confidence)}")
    lines += ["", "3. Process Chains", "Depth | Chain | ChainProbability | ChainSupport"]
    for chain in sorted(result.chains.values(), key=lambda item: (item.depth, -item.probability, item.nodes)):
        lines.append(f"{chain.depth} | {path_text(chain.nodes)} | {pct(chain.probability)} | {chain.support}")
    lines += ["", "4. Structural Cycles", "Cycle | CycleLength | Classification | CycleProbability | CycleSupport"]
    for cycle in sorted(result.cycles.values(), key=lambda item: (-item.probability, item.nodes)):
        lines.append(f"{path_text(cycle.nodes + (cycle.nodes[0],))} | {cycle.length} | {cycle.classification} | {pct(cycle.probability)} | {cycle.support}")
    if not result.cycles:
        lines.append("None")
    lines += ["", "5. Terminal Structures", "Node | PersistenceProbability | TransitionPressure | Entropy | TerminalScore"]
    for row in sorted(result.stats.values(), key=lambda item: item.terminal_score, reverse=True):
        lines.append(f"{node_text(row.node)} | {pct(row.persistence)} | {pct(row.pressure)} | {fmt(row.entropy)} | {fmt(row.terminal_score)}")
    lines += ["", "6. Structural Attractors", "Node | InboundProbability | OutboundProbability | NetFlow | AttractorScore"]
    for row in sorted(result.stats.values(), key=lambda item: item.attractor_score, reverse=True):
        lines.append(f"{node_text(row.node)} | {fmt(row.inbound_probability)} | {fmt(row.outbound_probability)} | {fmt(row.attractor_score)} | {fmt(row.attractor_score)}")
    lines += ["", "7. Structural Hubs", "Node | DistinctInboundCount | DistinctOutboundCount | TotalDegree | HubScore"]
    for row in sorted(result.stats.values(), key=lambda item: (-item.hub_score, item.node)):
        lines.append(f"{node_text(row.node)} | {row.inbound_count} | {row.outbound_count} | {row.hub_score} | {row.hub_score}")
    lines += ["", "8. Young / Middle / Late Graph", "StateZoneNode | DominantDestination | DominantProbability | PersistenceProbability | Entropy | TransitionConfidence"]
    for row in sorted(result.zone_stats.values(), key=lambda item: item.node):
        lines.append(f"{zone_text(row.node)} | {zone_text(row.dominant_destination)} | {pct(row.dominant_probability)} | {pct(row.persistence)} | {fmt(row.entropy)} | {pct(row.confidence)}")
    lines += ["", "Young / Middle / Late Chains", "Depth | Chain | ChainProbability | ChainSupport"]
    for chain in sorted(result.zone_chains.values(), key=lambda item: (item.depth, -item.probability, item.nodes)):
        lines.append(f"{chain.depth} | {path_text(chain.nodes, True)} | {pct(chain.probability)} | {chain.support}")
    lines += ["", "9. Graph Compression", "ProbabilityMass | NodeCount | EdgeCount | CompressionRatio"]
    for mass in MASS_LEVELS:
        row = result.compressions[mass]
        lines.append(f"{pct(row.mass)} | {row.node_count} | {row.edge_count} | {pct(row.compression_ratio)}")
    lines += ["", "10. Outcome Diagnostics", "Diagnostic only. Forward outcomes are not used in graph construction.", "ItemType | Item | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew"]
    for key, row in sorted(result.node_outcomes.items()):
        lines.append(f"Node | {node_text(key)} | {row.count} | {fmt(row.mean_dr)} | {fmt(row.median_dr)} | {pct(row.continuation)} | {pct(row.failure)} | {pct(row.flat)} | {pct(row.skew)}")
    for key, row in sorted(result.chain_outcomes.items()):
        lines.append(f"Chain | {path_text(key)} | {row.count} | {fmt(row.mean_dr)} | {fmt(row.median_dr)} | {pct(row.continuation)} | {pct(row.failure)} | {pct(row.flat)} | {pct(row.skew)}")
    for key, row in sorted(result.cycle_outcomes.items()):
        lines.append(f"Cycle | {path_text(key + (key[0],))} | {row.count} | {fmt(row.mean_dr)} | {fmt(row.median_dr)} | {pct(row.continuation)} | {pct(row.failure)} | {pct(row.flat)} | {pct(row.skew)}")
    lines += [
        "", "11. Graph Recommendation",
        f"Instrument-only diagnostic: {result.recommendation.graph}",
        result.recommendation.reason,
        "The aggregate report applies the required cross-instrument rule.",
        "", "12. Low-DoF Audit",
        "Variables used:",
        "StructuralState",
        "AgeBucket",
        "",
        "No Context",
        "No Arbitration",
        "No Persistence",
        "No Phase",
        "No Optimization",
        "No Fitting",
        "No Machine Learning",
        "No Forward Returns used in graph construction",
        "", "13. Mechanical Research Notes",
        "- Every node is StructuralState + AgeBucket.",
        "- Every edge is an observed adjacent node transition.",
        "- Dominant chains and recurring observed simple cycles are constructed without forward price outcomes.",
        "- Outcome diagnostics remain separate from graph construction and recommendation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_report(results: list[StudyResult], out_root: Path) -> None:
    path = out_root / "ProcessGraph" / "ProcessGraph_All.txt"
    ensure_dir(path.parent)
    instruments = instrument_columns(results)
    by_instrument = {result.instrument: result for result in results}
    lines = ["APVA Process Graph Study v0.1 - Aggregate", "=" * 108, "Instruments: " + ", ".join(instruments), ""]
    node_keys = sorted({node for result in results for node in result.nodes})
    edge_keys = sorted({key for result in results for key in result.edges})
    chain_keys = sorted({key for result in results for key in result.chains})
    cycle_keys = sorted({key for result in results for key in result.cycles})

    lines += ["Aggregate Node Table", "Node | " + " | ".join(f"Count_{i}" for i in instruments) + " | ReplicationCount | ReplicationPercent | Classification"]
    for node in node_keys:
        values = [by_instrument[i].nodes[node] for i in instruments]
        replicated = sum(value > 0 for value in values)
        lines.append(f"{node_text(node)} | " + " | ".join(map(str, values)) + f" | {replicated} | {pct(replicated / len(instruments))} | {replication_label(replicated)}")

    lines += ["", "Aggregate Edge Table", "SourceNode | DestinationNode | " + " | ".join(f"Prob_{i}" for i in instruments) + " | ReplicationCount | MeanProbability | Classification"]
    for key in edge_keys:
        values = [by_instrument[i].edges.get(key) for i in instruments]
        valid = [row for row in values if row]
        lines.append(f"{node_text(key[0])} | {node_text(key[1])} | " + " | ".join(pct(row.probability) if row else "N/A" for row in values) + f" | {len(valid)} | {pct(safe_mean(row.probability for row in valid))} | {replication_label(len(valid))}")

    lines += ["", "Aggregate Chain Table", "Depth | Chain | " + " | ".join(f"Probability_{i}" for i in instruments) + " | ReplicationCount | MeanProbability | Classification"]
    for key in chain_keys:
        values = [by_instrument[i].chains.get(key) for i in instruments]
        valid = [row for row in values if row and row.support > 0]
        lines.append(f"{len(key) - 1} | {path_text(key)} | " + " | ".join(pct(row.probability) if row else "N/A" for row in values) + f" | {len(valid)} | {pct(safe_mean(row.probability for row in valid))} | {replication_label(len(valid))}")

    lines += ["", "Aggregate Cycle Table", "Cycle | Length | " + " | ".join(f"Prob_{i}" for i in instruments) + " | ReplicationCount | MeanProbability | Classification"]
    for key in cycle_keys:
        values = [by_instrument[i].cycles.get(key) for i in instruments]
        valid = [row for row in values if row and row.support > 0]
        lines.append(f"{path_text(key + (key[0],))} | {len(key)} | " + " | ".join(pct(row.probability) if row else "N/A" for row in values) + f" | {len(valid)} | {pct(safe_mean(row.probability for row in valid))} | {replication_label(len(valid))}")
    if not cycle_keys:
        lines.append("None")

    lines += ["", "Aggregate Attractor Table", "Node | " + " | ".join(f"Inbound_{i} | Outbound_{i}" for i in instruments) + " | MeanAttractorScore"]
    attractors = []
    for node in node_keys:
        values = [by_instrument[i].stats.get(node) for i in instruments]
        valid = [row for row in values if row]
        attractors.append((node, valid))
        cells = [value for row in values for value in ((fmt(row.inbound_probability), fmt(row.outbound_probability)) if row else ("N/A", "N/A"))]
        lines.append(f"{node_text(node)} | " + " | ".join(cells) + f" | {fmt(safe_mean(row.attractor_score for row in valid))}")

    lines += ["", "Aggregate Hub Table", "Node | " + " | ".join(f"InboundCount_{i} | OutboundCount_{i} | HubScore_{i}" for i in instruments) + " | ReplicationCount | MeanHubScore"]
    hubs = []
    for node in node_keys:
        values = [by_instrument[i].stats.get(node) for i in instruments]
        valid = [row for row in values if row]
        hubs.append((node, valid))
        cells = [value for row in values for value in ((str(row.inbound_count), str(row.outbound_count), str(row.hub_score)) if row else ("N/A", "N/A", "N/A"))]
        lines.append(f"{node_text(node)} | " + " | ".join(cells) + f" | {len(valid)} | {fmt(safe_mean(row.hub_score for row in valid))}")

    lines += ["", "Aggregate Compression Table", "ProbabilityMass | " + " | ".join(f"NodeCount_{i} | EdgeCount_{i} | CompressionRatio_{i}" for i in instruments) + " | MeanNodeCount | MeanEdgeCount | MeanCompressionRatio"]
    for mass in MASS_LEVELS:
        values = [by_instrument[i].compressions[mass] for i in instruments]
        cells = [value for row in values for value in (str(row.node_count), str(row.edge_count), pct(row.compression_ratio))]
        lines.append(f"{pct(mass)} | " + " | ".join(cells) + f" | {fmt(safe_mean(row.node_count for row in values), 2)} | {fmt(safe_mean(row.edge_count for row in values), 2)} | {pct(safe_mean(row.compression_ratio for row in values))}")

    lines += ["", "Aggregate Outcome Table", "ItemType | NodeOrChain | " + " | ".join(f"Count_{i} | Skew_{i} | MeanDR_{i}" for i in instruments) + " | ValidInstrumentCount | MeanSkew | MeanDR"]
    outcome_rows = []
    for item_type, keys, getter, renderer in (
        ("Node", node_keys, lambda result, key: result.node_outcomes.get(key), node_text),
        ("Chain", chain_keys, lambda result, key: result.chain_outcomes.get(key), path_text),
        ("Cycle", cycle_keys, lambda result, key: result.cycle_outcomes.get(key), lambda key: path_text(key + (key[0],))),
    ):
        for key in keys:
            values = [getter(by_instrument[i], key) for i in instruments]
            valid = [row for row in values if row and row.count >= MIN_COUNT]
            outcome_rows.append((item_type, key, valid, renderer))
            cells = [value for row in values for value in ((str(row.count), pct(row.skew), fmt(row.mean_dr)) if row else ("0", "N/A", "N/A"))]
            lines.append(f"{item_type} | {renderer(key)} | " + " | ".join(cells) + f" | {len(valid)} | {pct(safe_mean(row.skew for row in valid))} | {fmt(safe_mean(row.mean_dr for row in valid))}")

    aggregate_recommendation = recommend(
        {node: row for result in results for node, row in result.stats.items()},
        {node: row for result in results for node, row in result.zone_stats.items()},
        len(results),
    )
    retention = safe_mean(result.recommendation.probability_mass_retention for result in results)
    entropy_increase = safe_mean(result.recommendation.entropy_increase for result in results)
    graph = "YoungMiddleLateGraph" if retention >= 0.90 and entropy_increase <= 0.10 and len(results) >= 2 else "FullAgeBucketGraph"
    reason = f"ProbabilityMassRetention={pct(retention)}; MeanEntropyIncrease={pct(entropy_increase)}; ValidInstrumentCount={len(results)}."
    lines += [
        "", "Aggregate Graph Recommendation",
        f"RecommendedGraph: {graph}",
        f"Reason: {reason}",
        f"Replication: {len(results)} instruments",
        f"Compression: 90% mass requires mean {fmt(safe_mean(result.compressions[0.90].compression_ratio for result in results) * 100, 2)}% of edges",
        f"EntropyRetention: {pct(1.0 - entropy_increase)}",
        f"ProbabilityMassRetention: {pct(retention)}",
        "", "Aggregate Rankings",
        "", "1. Most replicated nodes",
    ]
    for node in sorted(node_keys, key=lambda item: (-sum(by_instrument[i].nodes[item] > 0 for i in instruments), item)):
        replicated = sum(by_instrument[i].nodes[node] > 0 for i in instruments)
        lines.append(f"{node_text(node)} | ReplicationCount={replicated} | {replication_label(replicated)}")
    lines += ["", "2. Most replicated edges"]
    for key in sorted(edge_keys, key=lambda item: (-sum(item in result.edges for result in results), item))[:30]:
        replicated = sum(key in result.edges for result in results)
        lines.append(f"{node_text(key[0])} -> {node_text(key[1])} | ReplicationCount={replicated}")
    lines += ["", "3. Most replicated chains"]
    for key in sorted(chain_keys, key=lambda item: (-sum(item in result.chains and result.chains[item].support > 0 for result in results), item))[:30]:
        replicated = sum(key in result.chains and result.chains[key].support > 0 for result in results)
        lines.append(f"{path_text(key)} | ReplicationCount={replicated}")
    lines += ["", "4. Most replicated cycles"]
    for key in sorted(cycle_keys, key=lambda item: (-sum(item in result.cycles and result.cycles[item].support > 0 for result in results), item)):
        replicated = sum(key in result.cycles and result.cycles[key].support > 0 for result in results)
        lines.append(f"{path_text(key + (key[0],))} | ReplicationCount={replicated}")
    if not cycle_keys:
        lines.append("None")
    lines += ["", "5. Strongest attractors"]
    for node, valid in sorted(attractors, key=lambda item: safe_mean(row.attractor_score for row in item[1]), reverse=True)[:20]:
        lines.append(f"{node_text(node)} | MeanAttractorScore={fmt(safe_mean(row.attractor_score for row in valid))}")
    lines += ["", "6. Strongest hubs"]
    for node, valid in sorted(hubs, key=lambda item: safe_mean(row.hub_score for row in item[1]), reverse=True)[:20]:
        lines.append(f"{node_text(node)} | MeanHubScore={fmt(safe_mean(row.hub_score for row in valid))}")
    lines += ["", "7. Highest confidence transitions"]
    stats_rows = [(node, [result.stats[node] for result in results if node in result.stats]) for node in node_keys]
    for node, valid in sorted(stats_rows, key=lambda item: safe_mean(row.confidence for row in item[1]), reverse=True)[:20]:
        lines.append(f"{node_text(node)} | MeanConfidence={pct(safe_mean(row.confidence for row in valid))}")
    lines += ["", "8. Highest terminal scores"]
    for node, valid in sorted(stats_rows, key=lambda item: safe_mean(row.terminal_score for row in item[1]), reverse=True)[:20]:
        lines.append(f"{node_text(node)} | MeanTerminalScore={fmt(safe_mean(row.terminal_score for row in valid))}")
    lines += ["", "9. Best compressed graph"]
    for mass in MASS_LEVELS:
        lines.append(f"{pct(mass)} mass | MeanCompressionRatio={pct(safe_mean(result.compressions[mass].compression_ratio for result in results))}")
    lines += ["", "10. Recommended APVA process graph", f"{graph}: {reason}"]

    lines += [
        "", "Low-DoF Audit",
        "Variables used:",
        "StructuralState",
        "AgeBucket",
        "",
        "No Context",
        "No Arbitration",
        "No Persistence",
        "No Phase",
        "No Optimization",
        "No Fitting",
        "No Machine Learning",
        "No Forward Returns used in graph construction",
        "", "Mechanical Research Notes",
        "- APVA is represented as an observed StateAgeKey directed graph.",
        "- Dominant chains and recurring observed simple cycles summarize graph structure.",
        "- Compression reports the minimum observed edge subset needed to retain fixed probability-mass thresholds.",
        "- Outcome diagnostics remain separate from graph construction and recommendation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    out_root = Path(args.out_root)
    for result in results:
        write_per_instrument(result, out_root)
    aggregate_report(results, out_root)
    print(f"Wrote {len(results)} per-instrument report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
