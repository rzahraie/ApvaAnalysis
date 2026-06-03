#!/usr/bin/env python3
"""APVA Node Topology Study v0.1.

Classify fixed StateAge nodes by branch topology role using Study 65 branch
forecast diagnostics. Forward price outcomes are diagnostics only.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from APVA_BranchForecast_65 import (
    HORIZONS,
    build_result as build_branch_result,
    build_stream,
    mean,
    model_line,
)
from APVA_InformationContribution_63 import thresholds
from APVA_InformationDecay_57 import study as decay_study
from APVA_MinimalEngine_52 import directional_return, instrument_columns, load_results
from APVA_NodeImportance_58 import NodeRow, aggregate_rows, local_rows, score_rows
from APVA_ProcessGraph_54 import Node, node_for, node_text
from APVA_StructuralForecast_56 import Outcome, outcome
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

ROLES = ("Pipeline", "Junction", "Diffuse", "Ambiguous")


@dataclass
class TopologyNode:
    node: str
    state: str
    age: str
    count: int
    replication_count: int
    replication_percent: float
    outgoing_count: int
    dominant_next: str
    dominant_probability: float
    second_probability: float
    concentration: float
    entropy: float
    normalized_entropy: float
    previous_gain: float
    contribution: float
    role: str


@dataclass
class GroupSummary:
    name: str
    counts: dict[str, int]
    dominant_role: str


@dataclass
class FlowRow:
    source: str
    destination: str
    count: int
    probability: float


@dataclass
class ResidenceRow:
    role: str
    mean_duration: float
    median_duration: float
    max_duration: int


@dataclass
class ContributionSummary:
    role: str
    total: float
    mean_score: float
    percent: float


@dataclass
class StabilitySummary:
    role: str
    dominant_probability: float
    entropy: float
    previous_gain: float
    concentration: float


@dataclass
class Result:
    instrument: str
    source_paths: list
    bars: list
    branch_result: object
    nodes: dict[str, TopologyNode]
    state_summary: dict[str, GroupSummary]
    age_summary: dict[str, GroupSummary]
    flows: dict[tuple[str, str], FlowRow]
    residence: dict[str, ResidenceRow]
    contribution: dict[str, ContributionSummary]
    stability: dict[str, StabilitySummary]
    outcomes: dict[str, Outcome]


def role_for(row: TopologyNode) -> str:
    junction = row.previous_gain >= 0.05 and row.replication_count >= 2
    pipeline = (
        row.dominant_probability >= 0.70
        and row.concentration >= 0.40
        and row.previous_gain < 0.05
        and row.replication_count >= 2
    )
    diffuse = (
        row.normalized_entropy >= 0.70
        and row.dominant_probability < 0.50
        and row.previous_gain < 0.05
        and row.replication_count >= 2
    )
    if junction:
        return "Junction"
    if pipeline:
        return "Pipeline"
    if diffuse:
        return "Diffuse"
    return "Ambiguous"


def node_from_text(value: str) -> Node:
    state, age = value.rsplit("_", 1)
    return state, age


def build_nodes(branch_result, metadata: dict[Node, NodeRow]) -> dict[str, TopologyNode]:
    output = {}
    by_node_branches: dict[str, list] = defaultdict(list)
    for branch in branch_result.branch_rows.values():
        by_node_branches[branch.current].append(branch)
    for node_text_value, entropy in branch_result.entropy_rows.items():
        node = node_from_text(node_text_value)
        meta = metadata.get(node)
        branches = sorted(by_node_branches[node_text_value], key=lambda item: (-item.probability, item.next_node))
        dominant = branches[0] if branches else None
        second = branches[1].probability if len(branches) > 1 else 0.0
        gain = branch_result.gains["Model2_CurrentPreviousNode"].get(node_text_value)
        contribution = branch_result.contributions.get(node_text_value)
        row = TopologyNode(
            node_text_value,
            node[0],
            node[1],
            sum(branch.count for branch in branches),
            meta.replication_count if meta else 1,
            meta.replication_percent if meta else 1.0,
            len(branches),
            dominant.next_node if dominant else "NONE",
            dominant.probability if dominant else 0.0,
            second,
            entropy.concentration,
            entropy.entropy,
            entropy.normalized_entropy,
            gain.gain if gain else 0.0,
            contribution.score if contribution else 0.0,
            "",
        )
        row.role = role_for(row)
        output[node_text_value] = row
    return output


def group_summary(rows: Iterable[TopologyNode], name: str) -> GroupSummary:
    counts = {role: 0 for role in ROLES}
    for row in rows:
        counts[row.role] += 1
    dominant = max(ROLES, key=lambda role: (counts[role], role)) if sum(counts.values()) else "Ambiguous"
    return GroupSummary(name, counts, dominant)


def build_group_summaries(nodes: dict[str, TopologyNode], attr: str) -> dict[str, GroupSummary]:
    values: dict[str, list[TopologyNode]] = defaultdict(list)
    for row in nodes.values():
        values[getattr(row, attr)].append(row)
    return {name: group_summary(rows, name) for name, rows in sorted(values.items())}


def flow_matrix(segments: list[list], nodes: dict[str, TopologyNode]) -> dict[tuple[str, str], FlowRow]:
    counts = Counter()
    totals = Counter()
    for segment in segments:
        for left, right in zip(segment, segment[1:]):
            source = nodes[node_text(left.node)].role
            destination = nodes[node_text(right.node)].role
            counts[(source, destination)] += 1
            totals[source] += 1
    return {
        (source, destination): FlowRow(
            source,
            destination,
            counts[(source, destination)],
            counts[(source, destination)] / totals[source] if totals[source] else 0.0,
        )
        for source in ROLES
        for destination in ROLES
    }


def residence(segments: list[list], nodes: dict[str, TopologyNode]) -> dict[str, ResidenceRow]:
    durations: dict[str, list[int]] = defaultdict(list)
    for segment in segments:
        roles = [nodes[node_text(row.node)].role for row in segment]
        index = 0
        while index < len(roles):
            end = index + 1
            while end < len(roles) and roles[end] == roles[index]:
                end += 1
            durations[roles[index]].append(end - index)
            index = end
    return {
        role: ResidenceRow(
            role,
            mean(values),
            statistics.median(values) if values else 0.0,
            max(values, default=0),
        )
        for role in ROLES
        for values in [durations.get(role, [])]
    }


def contribution_summary(nodes: dict[str, TopologyNode]) -> dict[str, ContributionSummary]:
    total = sum(max(row.contribution, 0.0) for row in nodes.values())
    output = {}
    for role in ROLES:
        rows = [row for row in nodes.values() if row.role == role]
        role_total = sum(max(row.contribution, 0.0) for row in rows)
        output[role] = ContributionSummary(role, role_total, mean(row.contribution for row in rows), role_total / total if total else 0.0)
    return output


def stability_summary(nodes: dict[str, TopologyNode]) -> dict[str, StabilitySummary]:
    output = {}
    for role in ROLES:
        rows = [row for row in nodes.values() if row.role == role]
        output[role] = StabilitySummary(
            role,
            mean(row.dominant_probability for row in rows),
            mean(row.entropy for row in rows),
            mean(row.previous_gain for row in rows),
            mean(row.concentration for row in rows),
        )
    return output


def outcome_by_role(bars: list, branch_rows: list, nodes: dict[str, TopologyNode]) -> dict[str, Outcome]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in branch_rows:
        value = directional_return(bars, row.index, 5)
        if value is not None:
            values[nodes[node_text(row.node)].role].append(value)
    return {role: outcome(samples) for role, samples in values.items()}


def build_result(instrument: str, source_paths: list, bars: list, branch_result, metadata: dict[Node, NodeRow]) -> Result:
    nodes = build_nodes(branch_result, metadata)
    return Result(
        instrument,
        source_paths,
        bars,
        branch_result,
        nodes,
        build_group_summaries(nodes, "state"),
        build_group_summaries(nodes, "age"),
        flow_matrix(branch_result.segments, nodes),
        residence(branch_result.segments, nodes),
        contribution_summary(nodes),
        stability_summary(nodes),
        outcome_by_role(bars, branch_result.rows, nodes),
    )


def role_counts(nodes: dict[str, TopologyNode]) -> dict[str, int]:
    counter = Counter(row.role for row in nodes.values())
    return {role: counter[role] for role in ROLES}


def recommendation(result: Result) -> tuple[str, str]:
    counts = role_counts(result.nodes)
    total = sum(counts.values())
    previous_gain = result.branch_result.models["Model2_CurrentPreviousNode"].metrics.top1 - result.branch_result.models["Model1_CurrentNode"].metrics.top1
    if counts["Junction"] > 0 and previous_gain > 0:
        label = "MixedOrderTopology"
    elif total and counts["Pipeline"] / total >= 0.80:
        label = "FirstOrderStateMachine"
    elif total and counts["Diffuse"] >= counts["Pipeline"] + counts["Junction"]:
        label = "DiffuseUnresolved"
    else:
        label = "MixedOrderTopology" if counts["Junction"] else "DiffuseUnresolved"
    reason = (
        f"PipelineNodes={counts['Pipeline']}; JunctionNodes={counts['Junction']}; "
        f"DiffuseNodes={counts['Diffuse']}; AmbiguousNodes={counts['Ambiguous']}; "
        f"AggregatePreviousNodeGain={pct(previous_gain)}."
    )
    return label, reason


def top_lines(rows: Iterable, key, line, limit: int = 12) -> list[str]:
    return [line(row) for row in sorted(rows, key=key)[:limit]]


def node_line(row: TopologyNode) -> str:
    return (
        f"{row.node} | {row.count} | {row.replication_count} | {pct(row.replication_percent)} | "
        f"{row.state} | {row.age}"
    )


def branch_profile_line(row: TopologyNode) -> str:
    return (
        f"{row.node} | {row.outgoing_count} | {row.dominant_next} | {pct(row.dominant_probability)} | "
        f"{pct(row.second_probability)} | {pct(row.concentration)}"
    )


def topology_line(row: TopologyNode) -> str:
    return (
        f"{row.node} | {row.count} | {row.dominant_next} | {pct(row.dominant_probability)} | "
        f"{fmt(row.entropy)} | {pct(row.concentration)} | {pct(row.previous_gain)} | "
        f"{row.role} | {row.replication_count}"
    )


def write_per_instrument(result: Result, out_root: Path) -> None:
    path = out_root / result.instrument / f"NodeTopology_{result.instrument}.txt"
    ensure_dir(path.parent)
    lines = [
        "APVA Node Topology Study v0.1",
        "=" * 108,
        "Diagnostics",
        f"Instrument: {result.instrument}",
        "Input path(s): " + ", ".join(str(item) for item in result.source_paths),
        f"Total rows: {len(result.bars)}",
        f"Node count: {len(result.nodes)}",
        f"Branch count: {len(result.branch_result.branch_rows)}",
        "",
        "1. Node Inventory",
        "Node | Count | ReplicationCount | ReplicationPercent | StructuralState | AgeBucket",
    ]
    lines += [node_line(row) for row in sorted(result.nodes.values(), key=lambda row: (-row.count, row.node))]
    lines += ["", "2. Branch Profile", "Node | OutgoingBranchCount | DominantNextNode | DominantBranchProbability | SecondBranchProbability | BranchConcentration"]
    lines += [branch_profile_line(row) for row in sorted(result.nodes.values(), key=lambda row: (-row.count, row.node))]
    lines += ["", "3. Branch Entropy", "Node | BranchEntropy | NormalizedBranchEntropy"]
    lines += top_lines(result.nodes.values(), lambda row: (row.entropy, row.node), lambda row: f"{row.node} | {fmt(row.entropy)} | {pct(row.normalized_entropy)}", 40)
    lines += ["", "Highest entropy nodes"]
    lines += top_lines(result.nodes.values(), lambda row: (-row.entropy, row.node), lambda row: f"{row.node} | {fmt(row.entropy)} | {pct(row.normalized_entropy)}", 20)
    lines += ["", "4. PreviousNode Gain", "Node | Model1Accuracy | Model2Accuracy | PreviousNodeGain"]
    for row in sorted(result.branch_result.gains["Model2_CurrentPreviousNode"].values(), key=lambda item: (-item.gain, item.node)):
        lines.append(f"{row.node} | {pct(row.model1_accuracy)} | {pct(row.model_accuracy)} | {pct(row.gain)}")
    for number, role in ((5, "Pipeline"), (6, "Junction"), (7, "Diffuse"), (8, "Ambiguous")):
        lines += ["", f"{number}. {role} Classification", "Node | Count | DominantBranchProbability | BranchConcentration | PreviousNodeGain | BranchEntropy | ReplicationCount"]
        rows = [row for row in result.nodes.values() if row.role == role]
        lines += [f"{row.node} | {row.count} | {pct(row.dominant_probability)} | {pct(row.concentration)} | {pct(row.previous_gain)} | {fmt(row.entropy)} | {row.replication_count}" for row in sorted(rows, key=lambda row: (-row.count, row.node))]
    lines += ["", "9. Topology Role Assignment", "Node | TopologyRole"] + [f"{row.node} | {row.role}" for row in sorted(result.nodes.values(), key=lambda row: row.node)]
    lines += ["", "10. Topology by State", "State | PipelineCount | JunctionCount | DiffuseCount | AmbiguousCount | DominantTopologyRole"]
    lines += [group_line(row) for row in result.state_summary.values()]
    lines += ["", "11. Topology by Age", "AgeBucket | PipelineCount | JunctionCount | DiffuseCount | AmbiguousCount | DominantTopologyRole"]
    lines += [group_line(row) for row in result.age_summary.values()]
    lines += ["", "12. Topology Flow Matrix", "SourceRole | DestinationRole | Count | Probability"]
    for key in sorted(result.flows):
        row = result.flows[key]
        lines.append(f"{row.source} | {row.destination} | {row.count} | {pct(row.probability)}")
    lines += ["", "13. Topology Residence Time", "Role | MeanResidenceTime | MedianResidenceTime | MaxResidenceTime"]
    lines += [residence_line(row) for row in result.residence.values()]
    lines += ["", "14. Topology Contribution", "Role | TotalContributionScore | MeanContributionScore | ContributionPercent"]
    lines += [contribution_line(row) for row in result.contribution.values()]
    lines += ["", "15. Topology Stability", "Role | MeanDominantBranchProbability | MeanBranchEntropy | MeanPreviousNodeGain | MeanBranchConcentration"]
    lines += [stability_line(row) for row in result.stability.values()]
    lines += ["", "16. Cross-Instrument Replication", "See aggregate report."]
    lines += ["", "17. Outcome Diagnostics", "TopologyRole | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew"]
    for role in ROLES:
        row = result.outcomes.get(role)
        lines.append(f"{role} | {row.count if row else 0} | {fmt(row.mean_dr) if row else 'N/A'} | {fmt(row.median_dr) if row else 'N/A'} | {pct(row.continuation) if row else 'N/A'} | {pct(row.failure) if row else 'N/A'} | {pct(row.flat) if row else 'N/A'} | {fmt(row.skew) if row else 'N/A'}")
    label, reason = recommendation(result)
    lines += ["", "18. Recommendation", f"Classification: {label}", f"Reason: {reason}"]
    append_audit(lines)
    lines += ["", "20. Mechanical Research Notes", "- Node roles are assigned from fixed branch topology thresholds.", "- Junction has priority over Pipeline, Diffuse, and Ambiguous.", "- Outcomes remain diagnostic only."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def group_line(row: GroupSummary) -> str:
    return f"{row.name} | {row.counts['Pipeline']} | {row.counts['Junction']} | {row.counts['Diffuse']} | {row.counts['Ambiguous']} | {row.dominant_role}"


def residence_line(row: ResidenceRow) -> str:
    return f"{row.role} | {fmt(row.mean_duration, 2)} | {fmt(row.median_duration, 2)} | {row.max_duration}"


def contribution_line(row: ContributionSummary) -> str:
    return f"{row.role} | {fmt(row.total)} | {fmt(row.mean_score)} | {pct(row.percent)}"


def stability_line(row: StabilitySummary) -> str:
    return f"{row.role} | {pct(row.dominant_probability)} | {fmt(row.entropy)} | {pct(row.previous_gain)} | {pct(row.concentration)}"


def aggregate_flow_lines(results: list[Result], instruments: list[str]) -> list[str]:
    by_name = {row.instrument: row for row in results}
    lines = ["", "Aggregate Topology Flow Matrix", "SourceRole | DestinationRole | " + " | ".join(f"Count_{name} | Prob_{name}" for name in instruments) + " | ReplicationCount | MeanProbability"]
    for source in ROLES:
        for destination in ROLES:
            values = [by_name[name].flows[(source, destination)] for name in instruments]
            valid = [row for row in values if row.count]
            cells = [value for row in values for value in (str(row.count), pct(row.probability))]
            lines.append(f"{source} | {destination} | " + " | ".join(cells) + f" | {len(valid)} | {pct(mean(row.probability for row in valid))}")
    return lines


def aggregate_outcome_lines(results: list[Result], instruments: list[str]) -> list[str]:
    by_name = {row.instrument: row for row in results}
    lines = ["", "Aggregate Outcome Table", "TopologyRole | " + " | ".join(f"Count_{name} | MeanDR_{name}" for name in instruments) + " | ValidInstrumentCount"]
    for role in ROLES:
        values = [by_name[name].outcomes.get(role) for name in instruments]
        cells = [value for row in values for value in ((str(row.count), fmt(row.mean_dr)) if row else ("0", "N/A"))]
        lines.append(f"{role} | " + " | ".join(cells) + f" | {sum(row is not None for row in values)}")
    return lines


def write_aggregate(result: Result, instrument_results: list[Result], out_root: Path) -> None:
    path = out_root / "NodeTopology" / "NodeTopology_All.txt"
    ensure_dir(path.parent)
    instruments = instrument_columns(instrument_results)
    by_name = {row.instrument: row for row in instrument_results}
    lines = ["APVA Node Topology Study v0.1 - Aggregate", "=" * 108, "Instruments: " + ", ".join(instruments)]
    lines += ["", "Aggregate Node Topology Table", "Node | Count | DominantNextNode | DominantBranchProbability | BranchEntropy | BranchConcentration | PreviousNodeGain | TopologyRole | ReplicationCount"]
    lines += [topology_line(row) for row in sorted(result.nodes.values(), key=lambda row: (-row.count, row.node))]
    lines += ["", "Aggregate Role Summary Table", "Role | NodeCount | TotalCount | MeanDominantBranchProbability | MeanBranchEntropy | MeanPreviousNodeGain | MeanBranchConcentration"]
    for role in ROLES:
        rows = [row for row in result.nodes.values() if row.role == role]
        lines.append(f"{role} | {len(rows)} | {sum(row.count for row in rows)} | {pct(mean(row.dominant_probability for row in rows))} | {fmt(mean(row.entropy for row in rows))} | {pct(mean(row.previous_gain for row in rows))} | {pct(mean(row.concentration for row in rows))}")
    lines += ["", "Aggregate State Topology Table", "State | PipelineCount | JunctionCount | DiffuseCount | AmbiguousCount | DominantTopologyRole"]
    lines += [group_line(row) for row in result.state_summary.values()]
    lines += ["", "Aggregate Age Topology Table", "AgeBucket | PipelineCount | JunctionCount | DiffuseCount | AmbiguousCount | DominantTopologyRole"]
    lines += [group_line(row) for row in result.age_summary.values()]
    lines += aggregate_flow_lines(instrument_results, instruments)
    lines += ["", "Aggregate Residence Table", "Role | MeanResidenceTime | MedianResidenceTime | MaxResidenceTime"]
    lines += [residence_line(row) for row in result.residence.values()]
    lines += ["", "Aggregate Contribution Table", "Role | TotalContributionScore | MeanContributionScore | ContributionPercent"]
    lines += [contribution_line(row) for row in result.contribution.values()]
    lines += ["", "Aggregate Replication Table", "Node | " + " | ".join(f"Role_{name}" for name in instruments) + " | ReplicationCount"]
    for node in sorted(result.nodes):
        roles = [by_name[name].nodes.get(node).role if node in by_name[name].nodes else "Missing" for name in instruments]
        lines.append(f"{node} | " + " | ".join(roles) + f" | {sum(role != 'Missing' for role in roles)}")
    lines += aggregate_outcome_lines(instrument_results, instruments)
    label, reason = recommendation(result)
    counts = role_counts(result.nodes)
    previous_gain = result.branch_result.models["Model2_CurrentPreviousNode"].metrics.top1 - result.branch_result.models["Model1_CurrentNode"].metrics.top1
    lines += [
        "",
        "Aggregate Recommendation",
        f"Classification: {label}",
        f"Reason: {reason}",
        f"PipelineNodes: {counts['Pipeline']}",
        f"JunctionNodes: {counts['Junction']}",
        f"DiffuseNodes: {counts['Diffuse']}",
        f"AmbiguousNodes: {counts['Ambiguous']}",
        f"PreviousNodeJustification: {pct(previous_gain)} aggregate branch Top1 gain",
        f"ReplicationAssessment: {len(instruments)} instrument(s)",
    ]
    lines += rankings(result)
    lines += ["", "RESEARCH NOTES", "- StateAge nodes show role differences when branch entropy and PreviousNode gain are compared.", "- Pipeline nodes use CurrentNode -> NextNode logic.", "- Junction nodes justify PreviousNode + CurrentNode -> NextNode logic.", "- Diffuse nodes remain unresolved branch uncertainty."]
    append_audit(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rankings(result: Result) -> list[str]:
    lines = ["", "AGGREGATE RANKINGS"]
    for number, title, role, key in (
        (1, "Strongest Pipeline nodes", "Pipeline", lambda row: (-row.concentration, row.node)),
        (2, "Strongest Junction nodes", "Junction", lambda row: (-row.previous_gain, row.node)),
        (3, "Strongest Diffuse nodes", "Diffuse", lambda row: (-row.normalized_entropy, row.node)),
        (4, "Most ambiguous nodes", "Ambiguous", lambda row: (-row.count, row.node)),
    ):
        rows = [row for row in result.nodes.values() if row.role == role]
        lines += ["", f"{number}. {title}"] + top_lines(rows, key, lambda row: f"{row.node} | {row.role} | Count={row.count} | Gain={pct(row.previous_gain)} | Concentration={pct(row.concentration)} | Entropy={fmt(row.entropy)}")
    lines += ["", "5. Highest branch concentration"] + top_lines(result.nodes.values(), lambda row: (-row.concentration, row.node), lambda row: f"{row.node} | {pct(row.concentration)}")
    lines += ["", "6. Highest branch entropy"] + top_lines(result.nodes.values(), lambda row: (-row.entropy, row.node), lambda row: f"{row.node} | {fmt(row.entropy)}")
    lines += ["", "7. Highest PreviousNode gain"] + top_lines(result.nodes.values(), lambda row: (-row.previous_gain, row.node), lambda row: f"{row.node} | {pct(row.previous_gain)}")
    lines += ["", "8. Most replicated topology roles"] + top_lines(result.nodes.values(), lambda row: (-row.replication_count, row.node), lambda row: f"{row.node} | {row.role} | {row.replication_count}")
    lines += ["", "9. Most important topology roles"] + top_lines(result.contribution.values(), lambda row: (-row.total, row.role), lambda row: f"{row.role} | {fmt(row.total)}")
    lines += ["", "10. Recommended APVA topology model", recommendation(result)[0]]
    return lines


def append_audit(lines: list[str]) -> None:
    lines += [
        "",
        "19. Low-DoF Audit",
        "Variables used:",
        "CurrentNode",
        "PreviousNode",
        "NextNode",
        "StructuralState",
        "AgeBucket",
        "BranchProbability",
        "BranchEntropy",
        "PreviousNodeGain",
        "DominantBranchProbability",
        "BranchConcentration",
        "",
        "No Context",
        "No Arbitration",
        "No Persistence",
        "No Phase",
        "No Optimization",
        "No Fitting",
        "No Machine Learning",
        "No Forward Returns used in topology classification",
    ]


def validate(result: Result) -> None:
    if not result.nodes:
        raise RuntimeError(f"{result.instrument}: no topology nodes.")
    if set(role_counts(result.nodes)) != set(ROLES):
        raise RuntimeError(f"{result.instrument}: role count keys mismatch.")
    if any(row.role not in ROLES for row in result.nodes.values()):
        raise RuntimeError(f"{result.instrument}: unknown role.")
    for source in ROLES:
        total = sum(result.flows[(source, destination)].probability for destination in ROLES)
        if total and abs(total - 1.0) > 1e-9:
            raise RuntimeError(f"{result.instrument}: flow probabilities do not sum to 1 for {source}.")


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
    memory_thresholds = thresholds(loaded, aggregate_nodes)
    confidence_values = [aggregate_nodes[node_for(bar)].confidence for row in loaded for bar in row.bars]
    entropy_values = [aggregate_nodes[node_for(bar)].entropy_growth for row in loaded for bar in row.bars]
    confidence_thresholds = statistics.quantiles(confidence_values, n=3)[0], statistics.quantiles(confidence_values, n=3)[1]
    entropy_thresholds = statistics.quantiles(entropy_values, n=3)[0], statistics.quantiles(entropy_values, n=3)[1]

    aggregate_stream, aggregate_segments, aggregate_bars, aggregate_paths = [], [], [], []
    instrument_results = []
    for loaded_row, decay_row in zip(loaded, decay):
        offset = len(aggregate_bars)
        aggregate_segment = build_stream(loaded_row, aggregate_nodes, memory_thresholds, confidence_thresholds, entropy_thresholds, offset)
        aggregate_stream.extend(aggregate_segment)
        aggregate_segments.append(aggregate_segment)
        aggregate_bars.extend(loaded_row.bars)
        aggregate_paths.extend(loaded_row.source_paths)

        local = local_rows(decay_row)
        score_rows(local)
        local_segment = build_stream(loaded_row, local, memory_thresholds, confidence_thresholds, entropy_thresholds)
        branch = build_branch_result(loaded_row.instrument, loaded_row.source_paths, loaded_row.bars, local_segment, [local_segment])
        result = build_result(loaded_row.instrument, loaded_row.source_paths, loaded_row.bars, branch, aggregate_nodes)
        validate(result)
        instrument_results.append(result)

    aggregate_branch = build_branch_result("Aggregate", aggregate_paths, aggregate_bars, aggregate_stream, aggregate_segments)
    aggregate_result = build_result("Aggregate", aggregate_paths, aggregate_bars, aggregate_branch, aggregate_nodes)
    validate(aggregate_result)
    out_root = Path(args.out_root)
    for result in instrument_results:
        write_per_instrument(result, out_root)
    write_aggregate(aggregate_result, instrument_results, out_root)
    print(f"Wrote {len(instrument_results)} per-instrument report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
