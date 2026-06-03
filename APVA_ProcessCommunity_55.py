#!/usr/bin/env python3
"""APVA Process Community Study v0.1.

Discover topology-derived process communities in the minimal APVA State+Age
graph. Forward outcomes are diagnostics only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from APVA_MinimalEngine_52 import MIN_COUNT, directional_return, instrument_columns, load_results, outcome, safe_mean
from APVA_ProcessGraph_54 import Edge, dominant_chains, node_for, node_stats, node_text, path_text, rolling_support
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

Node = tuple[str, str]
CommunityID = str


def community_text(value: CommunityID) -> str:
    return value


def community_path(values: tuple[CommunityID, ...]) -> str:
    return " -> ".join(values)


def replication_label(count: int) -> str:
    return {1: "InstrumentSpecific", 2: "PartiallyReplicated", 3: "FullyReplicated"}.get(count, "Absent")


@dataclass
class Connectivity:
    node: Node
    inbound_degree: int
    outbound_degree: int
    weighted_inbound: float
    weighted_outbound: float
    total_degree: int
    reciprocal_degree: int


@dataclass
class Region:
    community_id: CommunityID
    nodes: tuple[Node, ...]
    edge_count: int
    internal_mass: float
    external_mass: float
    internal_ratio: float
    inbound_mass: float
    outbound_mass: float
    attractor_score: float
    transition_pressure: float
    internal_entropy: float
    exit_probability: float
    classification: str
    candidate: bool


@dataclass
class Basin:
    node: Node
    path: tuple[Node, ...]
    terminal: Node
    probability: float
    termination: str


@dataclass
class Compression:
    original_nodes: int
    compressed_nodes: int
    original_edges: int
    compressed_edges: int
    node_ratio: float
    edge_ratio: float


@dataclass
class Retention:
    probability_mass: float
    entropy: float
    dominant_path: float
    replication: float


@dataclass
class StudyResult:
    instrument: str
    source_paths: list[Path]
    bars: list
    nodes: Counter[Node]
    stats: dict
    connectivity: dict[Node, Connectivity]
    sccs: list[tuple[Node, ...]]
    regions: dict[CommunityID, Region]
    node_community: dict[Node, CommunityID]
    basins: dict[Node, Basin]
    compressed_nodes: Counter[CommunityID]
    compressed_edges: dict
    compressed_stats: dict
    chains: dict
    compression: Compression
    retention: Retention
    community_outcomes: dict
    chain_outcomes: dict


def stable_id(nodes: tuple[Node, ...]) -> CommunityID:
    return "Community[" + ",".join(node_text(node) for node in sorted(nodes)) + "]"


def tarjan(nodes: set[Node], adjacency: dict[Node, set[Node]]) -> list[tuple[Node, ...]]:
    index = 0
    indexes: dict[Node, int] = {}
    lowlinks: dict[Node, int] = {}
    stack: list[Node] = []
    active: set[Node] = set()
    components: list[tuple[Node, ...]] = []

    def visit(node: Node) -> None:
        nonlocal index
        indexes[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        active.add(node)
        for destination in adjacency.get(node, set()):
            if destination not in indexes:
                visit(destination)
                lowlinks[node] = min(lowlinks[node], lowlinks[destination])
            elif destination in active:
                lowlinks[node] = min(lowlinks[node], indexes[destination])
        if lowlinks[node] == indexes[node]:
            component = []
            while True:
                member = stack.pop()
                active.remove(member)
                component.append(member)
                if member == node:
                    break
            components.append(tuple(sorted(component)))

    for node in sorted(nodes):
        if node not in indexes:
            visit(node)
    return sorted(components)


def connectivity(stats: dict, edges: dict) -> dict[Node, Connectivity]:
    inbound: dict[Node, set[Node]] = defaultdict(set)
    for edge in edges.values():
        inbound[edge.destination].add(edge.source)
    rows = {}
    for node, row in stats.items():
        outgoing = set(row.destinations)
        rows[node] = Connectivity(
            node, len(inbound[node]), len(outgoing), row.inbound_probability,
            row.outbound_probability, len(inbound[node]) + len(outgoing),
            len(inbound[node] & outgoing),
        )
    return rows


def region_metrics(component: tuple[Node, ...], stats: dict, edges: dict) -> Region:
    members = set(component)
    internal = [edge for edge in edges.values() if edge.source in members and edge.destination in members]
    outbound = [edge for edge in edges.values() if edge.source in members and edge.destination not in members]
    inbound = [edge for edge in edges.values() if edge.source not in members and edge.destination in members]
    internal_mass = sum(edge.probability for edge in internal)
    outbound_mass = sum(edge.probability for edge in outbound)
    inbound_mass = sum(edge.probability for edge in inbound)
    denominator = internal_mass + outbound_mass
    internal_ratio = internal_mass / denominator if denominator else 0.0
    pressure = outbound_mass / denominator if denominator else 0.0
    internal_entropy = safe_mean(stats[node].entropy for node in component)
    label = "StableRegion" if internal_ratio >= 0.70 else "TransitionRegion" if internal_ratio < 0.40 else "MixedRegion"
    candidate = len(component) > 1 and internal_ratio >= 0.70
    return Region(
        stable_id(component), component, len(internal), internal_mass, outbound_mass,
        internal_ratio, inbound_mass, outbound_mass, inbound_mass - outbound_mass,
        pressure, internal_entropy, pressure, label, candidate,
    )


def community_map(sccs: list[tuple[Node, ...]], stats: dict, edges: dict) -> tuple[dict[CommunityID, Region], dict[Node, CommunityID]]:
    regions = {}
    mapping = {}
    for component in sccs:
        region = region_metrics(component, stats, edges)
        if not region.candidate and len(component) > 1:
            for node in component:
                singleton = region_metrics((node,), stats, edges)
                regions[singleton.community_id] = singleton
                mapping[node] = singleton.community_id
            continue
        regions[region.community_id] = region
        for node in component:
            mapping[node] = region.community_id
    return regions, mapping


def flow_basins(stats: dict) -> dict[Node, Basin]:
    rows = {}
    for start in stats:
        current = start
        path = [start]
        seen = {start}
        probability = 1.0
        termination = "Depth10"
        for _ in range(10):
            row = stats.get(current)
            if not row or not row.destinations:
                termination = "Termination"
                break
            current = row.dominant_destination
            probability *= row.dominant_probability
            path.append(current)
            if current in seen:
                termination = "Cycle"
                break
            seen.add(current)
        rows[start] = Basin(start, tuple(path), current, probability, termination)
    return rows


def compressed_graph(node_stream: list[Node], mapping: dict[Node, CommunityID]):
    stream = [mapping[node] for node in node_stream]
    counts, edges, stats = node_stats(stream)
    chains = dominant_chains(stats)
    for key, chain in chains.items():
        chain.support, _ = rolling_support(stream, key)
    return stream, counts, edges, stats, chains


def weighted_entropy(stats: dict) -> float:
    total = sum(row.count for row in stats.values())
    return sum(row.count * row.entropy for row in stats.values()) / total if total else 0.0


def retention(stats: dict, compressed_stats: dict, mapping: dict[Node, CommunityID], regions: dict[CommunityID, Region], nodes: Counter[Node]) -> Retention:
    original_entropy = weighted_entropy(stats)
    compressed_entropy = weighted_entropy(compressed_stats)
    entropy_retention = max(0.0, 1.0 - abs(compressed_entropy - original_entropy) / original_entropy) if original_entropy else 1.0
    total = sum(nodes.values())
    dominant_path = sum(
        nodes[node]
        for node, row in stats.items()
        if compressed_stats[mapping[node]].dominant_destination == mapping[row.dominant_destination]
    ) / total if total else 0.0
    internal = sum(region.internal_mass for region in regions.values())
    external = sum(region.external_mass for region in regions.values())
    probability_mass = internal / (internal + external) if internal + external else 0.0
    replicated = sum(nodes[node] for region in regions.values() if region.candidate for node in region.nodes)
    return Retention(probability_mass, entropy_retention, dominant_path, replicated / total if total else 0.0)


def community_outcomes(bars: list, mapping: dict[Node, CommunityID], chains: dict) -> tuple[dict, dict]:
    stream = [mapping[node_for(bar)] for bar in bars]
    values: dict[CommunityID, list[float]] = defaultdict(list)
    for index, community in enumerate(stream):
        value = directional_return(bars, index)
        if value is not None:
            values[community].append(value)
    chain_rows = {}
    for key in chains:
        _, indexes = rolling_support(stream, key)
        samples = [value for index in indexes if (value := directional_return(bars, index)) is not None]
        chain_rows[key] = outcome(samples)
    return {key: outcome(samples) for key, samples in values.items()}, chain_rows


def study(result) -> StudyResult:
    bars = result.bars
    stream = [node_for(bar) for bar in bars]
    nodes, edges, stats = node_stats(stream)
    adjacency = {node: set(row.destinations) for node, row in stats.items()}
    sccs = tarjan(set(nodes), adjacency)
    regions, mapping = community_map(sccs, stats, edges)
    community_stream, compressed_nodes, compressed_edges, compressed_stats, chains = compressed_graph(stream, mapping)
    compression = Compression(
        len(nodes), len(compressed_nodes), len(edges), len(compressed_edges),
        len(compressed_nodes) / len(nodes) if nodes else 0.0,
        len(compressed_edges) / len(edges) if edges else 0.0,
    )
    retained = retention(stats, compressed_stats, mapping, regions, nodes)
    community_values, chain_values = community_outcomes(bars, mapping, chains)
    return StudyResult(
        result.instrument, result.source_paths, bars, nodes, stats, connectivity(stats, edges),
        sccs, regions, mapping, flow_basins(stats), compressed_nodes, compressed_edges,
        compressed_stats, chains, compression, retained, community_values, chain_values,
    )


def community_survives(results: list[StudyResult]) -> tuple[str, str]:
    probability = safe_mean(result.retention.probability_mass for result in results)
    entropy_value = safe_mean(result.retention.entropy for result in results)
    valid = len(results)
    model = "CommunityModel" if probability >= 0.90 and entropy_value >= 0.90 and valid >= 2 else "OriginalGraph"
    return model, f"ProbabilityMassRetention={pct(probability)}; EntropyRetention={pct(entropy_value)}; ValidInstrumentCount={valid}."


def write_per_instrument(result: StudyResult, out_root: Path) -> None:
    path = out_root / result.instrument / f"ProcessCommunity_{result.instrument}.txt"
    ensure_dir(path.parent)
    lines = [
        "APVA Process Community Study v0.1",
        "=" * 96,
        "Diagnostics",
        f"Instrument: {result.instrument}",
        "Input path(s): " + ", ".join(str(item) for item in result.source_paths),
        f"Total rows: {len(result.bars)}",
        f"Node count: {len(result.nodes)}",
        f"Edge count: {sum(len(row.destinations) for row in result.stats.values())}",
        "",
        "1. Connectivity Analysis",
        "Node | InboundDegree | OutboundDegree | WeightedInboundDegree | WeightedOutboundDegree | TotalDegree | ReciprocalDegree",
    ]
    for row in sorted(result.connectivity.values(), key=lambda item: (-item.total_degree, item.node)):
        lines.append(f"{node_text(row.node)} | {row.inbound_degree} | {row.outbound_degree} | {fmt(row.weighted_inbound)} | {fmt(row.weighted_outbound)} | {row.total_degree} | {row.reciprocal_degree}")
    lines += ["", "2. Strongly Connected Regions", "CommunityID | NodeCount | EdgeCount | InternalProbabilityMass | ExternalProbabilityMass | InternalRatio"]
    edge_views = graph_edge_map(result.stats)
    for component in result.sccs:
        row = region_metrics(component, result.stats, edge_views)
        lines.append(f"{row.community_id} | {len(component)} | {row.edge_count} | {fmt(row.internal_mass)} | {fmt(row.external_mass)} | {pct(row.internal_ratio)}")
    lines += ["", "3. Community Candidates", "CommunityID | NodeCount | InternalProbabilityMass | ExternalProbabilityMass | InternalRatio | Classification"]
    for row in sorted(result.regions.values(), key=lambda item: (-item.internal_ratio, item.community_id)):
        lines.append(f"{row.community_id} | {len(row.nodes)} | {fmt(row.internal_mass)} | {fmt(row.external_mass)} | {pct(row.internal_ratio)} | {row.classification}")
    lines += ["", "4. Flow Basins", "Node | DominantFlowPath | TerminalNode | PathProbability | Termination"]
    for row in sorted(result.basins.values(), key=lambda item: item.node):
        lines.append(f"{node_text(row.node)} | {path_text(row.path)} | {node_text(row.terminal)} | {pct(row.probability)} | {row.termination}")
    lines += ["", "5. Attractor Regions", "CommunityID | TotalInboundMass | TotalOutboundMass | NetFlow | AttractorScore"]
    for row in sorted(result.regions.values(), key=lambda item: item.attractor_score, reverse=True):
        lines.append(f"{row.community_id} | {fmt(row.inbound_mass)} | {fmt(row.outbound_mass)} | {fmt(row.attractor_score)} | {fmt(row.attractor_score)}")
    lines += ["", "6. Transition Regions", "CommunityID | TransitionPressure | InternalEntropy | ExitProbability | Classification"]
    for row in sorted(result.regions.values(), key=lambda item: item.internal_ratio):
        lines.append(f"{row.community_id} | {pct(row.transition_pressure)} | {fmt(row.internal_entropy)} | {pct(row.exit_probability)} | {row.classification}")
    row = result.compression
    lines += [
        "", "7. Graph Compression",
        "OriginalNodes | CompressedNodes | OriginalEdges | CompressedEdges | NodeCompressionRatio | EdgeCompressionRatio",
        f"{row.original_nodes} | {row.compressed_nodes} | {row.original_edges} | {row.compressed_edges} | {pct(row.node_ratio)} | {pct(row.edge_ratio)}",
        "", "8. Information Retention",
        "ProbabilityMassRetention | EntropyRetention | DominantPathRetention | ReplicationRetention",
        f"{pct(result.retention.probability_mass)} | {pct(result.retention.entropy)} | {pct(result.retention.dominant_path)} | {pct(result.retention.replication)}",
        "", "9. Community Chains",
        "Depth | Chain | ChainProbability | ChainSupport",
    ]
    for chain in sorted(result.chains.values(), key=lambda item: (item.depth, -item.probability, item.nodes)):
        lines.append(f"{chain.depth} | {community_path(chain.nodes)} | {pct(chain.probability)} | {chain.support}")
    lines += ["", "10. Outcome Diagnostics", "Diagnostic only. Forward outcomes are not used in community construction.", "ItemType | Item | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew"]
    for key, value in sorted(result.community_outcomes.items()):
        lines.append(f"Community | {key} | {value.count} | {fmt(value.mean_dr)} | {fmt(value.median_dr)} | {pct(value.continuation)} | {pct(value.failure)} | {pct(value.flat)} | {pct(value.skew)}")
    for key, value in sorted(result.chain_outcomes.items()):
        lines.append(f"CommunityChain | {community_path(key)} | {value.count} | {fmt(value.mean_dr)} | {fmt(value.median_dr)} | {pct(value.continuation)} | {pct(value.failure)} | {pct(value.flat)} | {pct(value.skew)}")
    model, reason = community_survives([result])
    lines += [
        "", "11. Emergent Macro-Process Test",
        f"Instrument-only diagnostic: {model}",
        reason,
        "The aggregate report applies the required cross-instrument rule.",
        "", "12. Low-DoF Audit",
        "Variables used:",
        "StructuralState",
        "AgeBucket",
        "TransitionProbability",
        "",
        "No Context",
        "No Arbitration",
        "No Persistence",
        "No Phase",
        "No Optimization",
        "No Fitting",
        "No Machine Learning",
        "No Forward Returns used in community construction",
        "", "13. Mechanical Research Notes",
        "- SCC decomposition uses a standard Tarjan graph algorithm.",
        "- Multi-node SCCs become community candidates only when InternalRatio >= 70%.",
        "- Nodes outside candidate communities remain singleton regions.",
        "- Outcome diagnostics remain separate from community construction and recommendation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def graph_edge_map(stats: dict) -> dict:
    rows = {}
    for source, row in stats.items():
        total = sum(row.destinations.values())
        for destination, count in row.destinations.items():
            rows[(source, destination)] = Edge(source, destination, count, count / total)
    return rows


def aggregate_report(results: list[StudyResult], out_root: Path) -> None:
    path = out_root / "ProcessCommunity" / "ProcessCommunity_All.txt"
    ensure_dir(path.parent)
    instruments = instrument_columns(results)
    by_instrument = {result.instrument: result for result in results}
    community_ids = sorted({key for result in results for key in result.regions})
    chain_ids = sorted({key for result in results for key in result.chains})
    lines = ["APVA Process Community Study v0.1 - Aggregate", "=" * 112, "Instruments: " + ", ".join(instruments), ""]

    lines += ["Aggregate Community Table", "CommunityID | " + " | ".join(f"NodeCount_{i}" for i in instruments) + " | ReplicationCount | ReplicationPercent | MeanInternalRatio | Classification"]
    for key in community_ids:
        values = [by_instrument[i].regions.get(key) for i in instruments]
        valid = [row for row in values if row]
        lines.append(f"{key} | " + " | ".join(str(len(row.nodes)) if row else "0" for row in values) + f" | {len(valid)} | {pct(len(valid) / len(instruments))} | {pct(safe_mean(row.internal_ratio for row in valid))} | {replication_label(len(valid))}")

    lines += ["", "Aggregate Attractor Table", "CommunityID | " + " | ".join(f"InboundMass_{i} | OutboundMass_{i}" for i in instruments) + " | MeanAttractorScore"]
    attractors = []
    for key in community_ids:
        values = [by_instrument[i].regions.get(key) for i in instruments]
        valid = [row for row in values if row]
        attractors.append((key, valid))
        cells = [value for row in values for value in ((fmt(row.inbound_mass), fmt(row.outbound_mass)) if row else ("N/A", "N/A"))]
        lines.append(f"{key} | " + " | ".join(cells) + f" | {fmt(safe_mean(row.attractor_score for row in valid))}")

    lines += ["", "Aggregate Compression Table", "Instrument | OriginalNodes | CompressedNodes | OriginalEdges | CompressedEdges | NodeCompressionRatio | EdgeCompressionRatio"]
    for instrument in instruments:
        row = by_instrument[instrument].compression
        lines.append(f"{instrument} | {row.original_nodes} | {row.compressed_nodes} | {row.original_edges} | {row.compressed_edges} | {pct(row.node_ratio)} | {pct(row.edge_ratio)}")

    lines += ["", "Aggregate Retention Table", "Instrument | ProbabilityMassRetention | EntropyRetention | DominantPathRetention | ReplicationRetention"]
    for instrument in instruments:
        row = by_instrument[instrument].retention
        lines.append(f"{instrument} | {pct(row.probability_mass)} | {pct(row.entropy)} | {pct(row.dominant_path)} | {pct(row.replication)}")

    lines += ["", "Aggregate Community Chain Table", "Chain | " + " | ".join(f"Probability_{i}" for i in instruments) + " | ReplicationCount | MeanProbability"]
    for key in chain_ids:
        values = [by_instrument[i].chains.get(key) for i in instruments]
        valid = [row for row in values if row and row.support > 0]
        lines.append(f"{community_path(key)} | " + " | ".join(pct(row.probability) if row else "N/A" for row in values) + f" | {len(valid)} | {pct(safe_mean(row.probability for row in valid))}")

    lines += ["", "Aggregate Outcome Table", "Community | " + " | ".join(f"Count_{i} | Skew_{i} | MeanDR_{i}" for i in instruments) + " | ValidInstrumentCount | MeanSkew | MeanDR"]
    for key in community_ids:
        values = [by_instrument[i].community_outcomes.get(key) for i in instruments]
        valid = [row for row in values if row and row.count >= MIN_COUNT]
        cells = [value for row in values for value in ((str(row.count), pct(row.skew), fmt(row.mean_dr)) if row else ("0", "N/A", "N/A"))]
        lines.append(f"{key} | " + " | ".join(cells) + f" | {len(valid)} | {pct(safe_mean(row.skew for row in valid))} | {fmt(safe_mean(row.mean_dr for row in valid))}")

    model, reason = community_survives(results)
    lines += [
        "", "Aggregate Macro-Process Recommendation",
        f"Recommendation: {model}",
        f"Reason: {reason}",
        f"Replication: {len(results)} instruments",
        f"Compression: MeanNodeCompressionRatio={pct(safe_mean(result.compression.node_ratio for result in results))}; MeanEdgeCompressionRatio={pct(safe_mean(result.compression.edge_ratio for result in results))}",
        "", "Aggregate Rankings",
        "", "1. Strongest communities",
    ]
    region_rows = [(key, [result.regions[key] for result in results if key in result.regions]) for key in community_ids]
    for key, valid in sorted(region_rows, key=lambda item: safe_mean(row.internal_ratio for row in item[1]), reverse=True)[:20]:
        lines.append(f"{key} | MeanInternalRatio={pct(safe_mean(row.internal_ratio for row in valid))} | ReplicationCount={len(valid)}")
    lines += ["", "2. Strongest attractors"]
    for key, valid in sorted(attractors, key=lambda item: safe_mean(row.attractor_score for row in item[1]), reverse=True)[:20]:
        lines.append(f"{key} | MeanAttractorScore={fmt(safe_mean(row.attractor_score for row in valid))}")
    lines += ["", "3. Strongest transition regions"]
    for key, valid in sorted(region_rows, key=lambda item: safe_mean(row.transition_pressure for row in item[1]), reverse=True)[:20]:
        lines.append(f"{key} | MeanTransitionPressure={pct(safe_mean(row.transition_pressure for row in valid))}")
    lines += ["", "4. Highest information retention"]
    for result in sorted(results, key=lambda item: item.retention.entropy, reverse=True):
        lines.append(f"{result.instrument} | ProbabilityMassRetention={pct(result.retention.probability_mass)} | EntropyRetention={pct(result.retention.entropy)} | DominantPathRetention={pct(result.retention.dominant_path)}")
    lines += ["", "5. Highest compression ratio"]
    for result in sorted(results, key=lambda item: item.compression.node_ratio):
        lines.append(f"{result.instrument} | NodeCompressionRatio={pct(result.compression.node_ratio)} | EdgeCompressionRatio={pct(result.compression.edge_ratio)}")
    lines += ["", "6. Most replicated communities"]
    for key, valid in sorted(region_rows, key=lambda item: (-len(item[1]), item[0]))[:30]:
        lines.append(f"{key} | ReplicationCount={len(valid)} | {replication_label(len(valid))}")
    lines += ["", "7. Most replicated community chains"]
    for key in sorted(chain_ids, key=lambda item: (-sum(item in result.chains and result.chains[item].support > 0 for result in results), item))[:30]:
        lines.append(f"{community_path(key)} | ReplicationCount={sum(key in result.chains and result.chains[key].support > 0 for result in results)}")
    lines += ["", "8. Best attractor regions"]
    for key, valid in sorted(attractors, key=lambda item: safe_mean(row.attractor_score for row in item[1]), reverse=True)[:20]:
        lines.append(f"{key} | MeanAttractorScore={fmt(safe_mean(row.attractor_score for row in valid))}")
    lines += ["", "9. Best compressed process graph"]
    for result in sorted(results, key=lambda item: item.compression.node_ratio):
        lines.append(f"{result.instrument} | CompressedNodes={result.compression.compressed_nodes}/{result.compression.original_nodes} | EntropyRetention={pct(result.retention.entropy)}")
    lines += ["", "10. Recommended APVA macro-process model", f"{model}: {reason}"]

    lines += [
        "", "Low-DoF Audit",
        "Variables used:",
        "StructuralState",
        "AgeBucket",
        "TransitionProbability",
        "",
        "No Context",
        "No Arbitration",
        "No Persistence",
        "No Phase",
        "No Optimization",
        "No Fitting",
        "No Machine Learning",
        "No Forward Returns used in community construction",
        "", "Mechanical Research Notes",
        "- Communities are discovered from SCC topology and a fixed internal-ratio criterion.",
        "- No process region is manually merged or named.",
        "- Compression preserves all nodes by retaining non-candidate nodes as singleton regions.",
        "- Outcome diagnostics remain separate from community construction and recommendation.",
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
