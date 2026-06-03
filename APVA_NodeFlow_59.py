#!/usr/bin/env python3
"""APVA Node Flow Study v0.1.

Measure flow among the fixed Study 58 StateAge attention categories. Forward
outcomes are diagnostics only.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from APVA_InformationDecay_57 import study as decay_study
from APVA_MinimalEngine_52 import directional_return, instrument_columns, load_results, normalized_entropy, safe_mean
from APVA_NodeImportance_58 import NodeRow, aggregate_rows, local_rows, score_rows, validate_invariants as validate_importance
from APVA_ProcessGraph_54 import node_for, node_text
from APVA_StructuralForecast_56 import Outcome, outcome
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

Node = tuple[str, str]
CATEGORIES = ("Core Node", "Secondary Node", "Transition Node", "Weak Node", "Insufficient Node")
PUBLIC_CATEGORIES = ("Core Node", "Secondary Node", "Transition Node", "Weak Node")
PATH_DEPTHS = (2, 3, 4, 5)


@dataclass
class Flow:
    source: str
    destination: str
    count: int
    probability: float


@dataclass
class Balance:
    category: str
    inbound: int
    outbound: int
    net: int


@dataclass
class Excursion:
    count: int
    probability: float
    durations: list[int]
    mean_duration: float
    median_duration: float
    maximum_duration: int


@dataclass
class Return:
    count: int
    probability: float
    durations: list[int]
    mean_duration: float
    median_duration: float


@dataclass
class Residence:
    category: str
    durations: list[int]
    mean_duration: float
    median_duration: float
    maximum_duration: int
    distribution: Counter[int]


@dataclass
class Persistence:
    category: str
    stay: float
    leave: float
    ratio: float


@dataclass
class Entropy:
    category: str
    entropy: float
    normalized: float


@dataclass
class Pathway:
    path: tuple[str, ...]
    depth: int
    count: int
    probability: float


@dataclass
class Dominance:
    core: float
    secondary: float
    transition: float
    weak: float
    insufficient: float
    ratio: float


@dataclass
class FlowResult:
    instrument: str
    source_paths: list
    bars: list
    nodes: list[Node]
    node_rows: dict[Node, NodeRow]
    categories: list[str]
    node_flows: dict[tuple[Node, Node], int]
    category_flows: dict[tuple[str, str], Flow]
    balances: dict[str, Balance]
    excursion: Excursion
    returns: Return
    pathways: dict[tuple[str, ...], Pathway]
    residences: dict[str, Residence]
    persistence: dict[str, Persistence]
    entropy: dict[str, Entropy]
    dominance: Dominance
    outcomes: dict[str, Outcome]


def mean(values: Iterable[float]) -> float:
    return safe_mean(list(values))


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def category_name(value: str) -> str:
    return value.replace(" Node", "")


def replication_label(count: int) -> str:
    return {1: "InstrumentSpecific", 2: "PartiallyReplicated", 3: "FullyReplicated"}.get(count, "Absent")


def node_flow(nodes: list[Node]) -> dict[tuple[Node, Node], int]:
    return Counter(zip(nodes, nodes[1:]))


def category_flow(categories: list[str]) -> dict[tuple[str, str], Flow]:
    counts = Counter(zip(categories, categories[1:]))
    totals = Counter(source for source, _ in zip(categories, categories[1:]))
    return {
        (source, destination): Flow(source, destination, counts[(source, destination)], counts[(source, destination)] / totals[source] if totals[source] else 0.0)
        for source in CATEGORIES
        for destination in CATEGORIES
    }


def flow_balance(flows: dict[tuple[str, str], Flow]) -> dict[str, Balance]:
    return {
        category: Balance(
            category,
            sum(row.count for row in flows.values() if row.destination == category),
            sum(row.count for row in flows.values() if row.source == category),
            sum(row.count for row in flows.values() if row.destination == category) - sum(row.count for row in flows.values() if row.source == category),
        )
        for category in CATEGORIES
    }


def excursion(categories: list[str]) -> Excursion:
    starts = [index for index in range(1, len(categories)) if categories[index - 1] == "Core Node" and categories[index] in {"Secondary Node", "Transition Node"}]
    core_opportunities = sum(category == "Core Node" for category in categories[:-1])
    durations = []
    for start in starts:
        end = next((index for index in range(start + 1, len(categories)) if categories[index] == "Core Node"), len(categories))
        durations.append(end - start)
    return Excursion(
        len(starts), len(starts) / core_opportunities if core_opportunities else 0.0,
        durations, mean(durations), median(durations), max(durations, default=0),
    )


def returns(categories: list[str]) -> Return:
    starts = [index for index in range(len(categories)) if categories[index] in {"Secondary Node", "Transition Node"} and (index == 0 or categories[index - 1] not in {"Secondary Node", "Transition Node"})]
    durations = []
    returned = 0
    for start in starts:
        end = next((index for index in range(start + 1, len(categories)) if categories[index] == "Core Node"), None)
        if end is not None:
            returned += 1
            durations.append(end - start)
    return Return(returned, returned / len(starts) if starts else 0.0, durations, mean(durations), median(durations))


def pathways(categories: list[str]) -> dict[tuple[str, ...], Pathway]:
    rows = {}
    for depth in PATH_DEPTHS:
        counts = Counter(tuple(categories[index:index + depth]) for index in range(len(categories) - depth + 1))
        total = sum(counts.values())
        rows.update({path: Pathway(path, depth, count, count / total if total else 0.0) for path, count in counts.items()})
    return rows


def residence(categories: list[str]) -> dict[str, Residence]:
    durations: dict[str, list[int]] = defaultdict(list)
    index = 0
    while index < len(categories):
        end = index + 1
        while end < len(categories) and categories[end] == categories[index]:
            end += 1
        durations[categories[index]].append(end - index)
        index = end
    return {
        category: Residence(category, values, mean(values), median(values), max(values, default=0), Counter(values))
        for category in CATEGORIES
        for values in [durations.get(category, [])]
    }


def flow_persistence(flows: dict[tuple[str, str], Flow]) -> dict[str, Persistence]:
    return {
        category: Persistence(
            category, flows[(category, category)].probability,
            1.0 - flows[(category, category)].probability,
            flows[(category, category)].probability / (1.0 - flows[(category, category)].probability) if flows[(category, category)].probability < 1.0 else math.inf,
        )
        for category in CATEGORIES
    }


def flow_entropy(flows: dict[tuple[str, str], Flow]) -> dict[str, Entropy]:
    return {
        category: Entropy(
            category,
            -sum(row.probability * math.log(row.probability) for row in flows.values() if row.source == category and row.probability > 0),
            normalized_entropy(row.count for row in flows.values() if row.source == category),
        )
        for category in CATEGORIES
    }


def dominance(categories: list[str]) -> Dominance:
    count = Counter(categories)
    total = len(categories)
    share = lambda category: count[category] / total if total else 0.0
    core = share("Core Node")
    return Dominance(
        core, share("Secondary Node"), share("Transition Node"),
        share("Weak Node"), share("Insufficient Node"),
        core / (1.0 - core) if core < 1.0 else math.inf,
    )


def outcomes(bars: list, categories: list[str]) -> dict[str, Outcome]:
    values: dict[str, list[float]] = defaultdict(list)
    for index, category in enumerate(categories):
        value = directional_return(bars, index, 5)
        if value is not None:
            values[category].append(value)
    return {category: outcome(samples) for category, samples in values.items()}


def build_result(loaded, node_rows: dict[Node, NodeRow]) -> FlowResult:
    bars = loaded.bars
    nodes = [node_for(bar) for bar in bars]
    categories = [node_rows[node].category for node in nodes]
    flows = category_flow(categories)
    return FlowResult(
        loaded.instrument, loaded.source_paths, bars, nodes, node_rows, categories,
        node_flow(nodes), flows, flow_balance(flows), excursion(categories),
        returns(categories), pathways(categories), residence(categories),
        flow_persistence(flows), flow_entropy(flows), dominance(categories),
        outcomes(bars, categories),
    )


def recommendation(results: list[FlowResult]) -> tuple[str, str]:
    core = mean(result.dominance.core for result in results)
    if core >= 0.60:
        label = "Strong Core Dominance"
    elif core >= 0.40:
        label = "Moderate Core Dominance"
    else:
        label = "Weak Core Dominance"
    reason = (
        f"MeanPercentTimeInCore={pct(core)}; MeanExcursionProbability={pct(mean(result.excursion.probability for result in results))}; "
        f"MeanReturnProbability={pct(mean(result.returns.probability for result in results))}; ValidInstrumentCount={len(results)}."
    )
    return label, reason


def path_text(path: tuple[str, ...]) -> str:
    return " -> ".join(category_name(category) for category in path)


def distribution_text(distribution: Counter[int]) -> str:
    return ", ".join(f"{duration}:{count}" for duration, count in sorted(distribution.items())) or "None"


def write_per_instrument(result: FlowResult, out_root: Path) -> None:
    path = out_root / result.instrument / f"NodeFlow_{result.instrument}.txt"
    ensure_dir(path.parent)
    lines = [
        "APVA Node Flow Study v0.1", "=" * 104, "Diagnostics",
        f"Instrument: {result.instrument}",
        "Input path(s): " + ", ".join(str(item) for item in result.source_paths),
        f"Total rows: {len(result.bars)}", f"Node count: {len(result.node_rows)}",
        f"Category count: {len(set(result.categories))}",
        "", "1. Node Category Assignment", "StateAgeNode | NodeCategory | ImportanceScore | ReliabilityScore",
    ]
    for node, row in sorted(result.node_rows.items()):
        lines.append(f"{node_text(node)} | {row.category} | {fmt(row.importance)} | {fmt(row.reliability)}")
    lines += ["", "2. Node-to-Node Flow Matrix", "SourceNode | DestinationNode | TransitionCount | TransitionProbability"]
    totals = Counter(source for source, _ in zip(result.nodes, result.nodes[1:]))
    for source in sorted(result.node_rows):
        for destination in sorted(result.node_rows):
            count = result.node_flows.get((source, destination), 0)
            lines.append(f"{node_text(source)} | {node_text(destination)} | {count} | {pct(count / totals[source] if totals[source] else 0.0)}")
    lines += ["", "3. Category Flow Matrix", "SourceCategory | DestinationCategory | Count | Probability"]
    for key in sorted(result.category_flows):
        row = result.category_flows[key]
        lines.append(f"{row.source} | {row.destination} | {row.count} | {pct(row.probability)}")
    lines += ["", "4. Flow Balance", "Category | InboundTransitions | OutboundTransitions | NetFlow"]
    for row in result.balances.values():
        lines.append(f"{row.category} | {row.inbound} | {row.outbound} | {row.net}")
    row = result.excursion
    lines += ["", "5. Excursion Analysis", "ExcursionCount | ExcursionProbability | MeanExcursionDuration | MedianExcursionDuration | MaximumExcursionDuration", f"{row.count} | {pct(row.probability)} | {fmt(row.mean_duration, 2)} | {fmt(row.median_duration, 2)} | {row.maximum_duration}"]
    row = result.returns
    lines += ["", "6. Return Analysis", "ReturnCount | ReturnProbability | MeanReturnDuration | MedianReturnDuration", f"{row.count} | {pct(row.probability)} | {fmt(row.mean_duration, 2)} | {fmt(row.median_duration, 2)}"]
    lines += ["", "7. Transition Pathways", "Depth | Path | PathCount | PathProbability"]
    for row in sorted(result.pathways.values(), key=lambda row: (row.depth, -row.count, row.path)):
        lines.append(f"{row.depth} | {path_text(row.path)} | {row.count} | {pct(row.probability)}")
    lines += ["", "8. Residence Time", "Category | MeanResidenceTime | MedianResidenceTime | MaximumResidenceTime | ResidenceDistribution"]
    for row in result.residences.values():
        lines.append(f"{row.category} | {fmt(row.mean_duration, 2)} | {fmt(row.median_duration, 2)} | {row.maximum_duration} | {distribution_text(row.distribution)}")
    lines += ["", "9. Flow Persistence", "Category | StayProbability | LeaveProbability | PersistenceRatio"]
    for row in result.persistence.values():
        lines.append(f"{row.category} | {pct(row.stay)} | {pct(row.leave)} | {fmt(row.ratio)}")
    lines += ["", "10. Flow Entropy", "Category | TransitionEntropy | NormalizedEntropy"]
    for row in result.entropy.values():
        lines.append(f"{row.category} | {fmt(row.entropy)} | {fmt(row.normalized)}")
    lines += ["", "11. Cross-Instrument Replication", "Instrument-only diagnostic. See aggregate report for category-flow replication."]
    row = result.dominance
    lines += ["", "12. Core Dominance Test", "PercentTimeInCore | PercentTimeInSecondary | PercentTimeInTransition | PercentTimeInWeak | PercentTimeInInsufficient | CoreDominanceRatio", f"{pct(row.core)} | {pct(row.secondary)} | {pct(row.transition)} | {pct(row.weak)} | {pct(row.insufficient)} | {fmt(row.ratio)}"]
    lines += ["", "13. Outcome Diagnostics", "Diagnostic only. Forward outcomes are not used in flow calculations.", "Category | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew"]
    for category, row in sorted(result.outcomes.items()):
        lines.append(f"{category} | {row.count} | {fmt(row.mean_dr)} | {fmt(row.median_dr)} | {pct(row.continuation)} | {pct(row.failure)} | {pct(row.flat)} | {pct(row.skew)}")
    label, reason = recommendation([result])
    lines += [
        "", "14. Recommendation", f"Instrument-only diagnostic: {label}", reason,
        "The aggregate report applies the cross-instrument summary.",
        "", "15. Low-DoF Audit", "Variables used:", "StructuralState", "AgeBucket",
        "", "Derived metrics:", "ImportanceScore", "ReliabilityScore", "AttentionCategory",
        "", "No Context", "No Arbitration", "No Persistence", "No Phase",
        "No Optimization", "No Fitting", "No Machine Learning",
        "No Forward Returns used in flow construction",
        "", "16. Mechanical Research Notes",
        "- Node categories are reused exactly from Study 58.",
        "- Flow construction uses adjacent StateAge category transitions only.",
        "- Excursions start when Core enters Secondary or Transition.",
        "- Returns measure Secondary or Transition episodes that reach Core.",
        "- Outcome diagnostics remain separate from flow construction.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_flow(results: list[FlowResult], source: str, destination: str) -> tuple[int, float, int]:
    rows = [result.category_flows[(source, destination)] for result in results]
    valid = [row for row in rows if row.count]
    return sum(row.count for row in rows), mean(row.probability for row in rows), len(valid)


def write_aggregate(results: list[FlowResult], out_root: Path) -> None:
    path = out_root / "NodeFlow" / "NodeFlow_All.txt"
    ensure_dir(path.parent)
    instruments = instrument_columns(results)
    by_instrument = {result.instrument: result for result in results}
    lines = ["APVA Node Flow Study v0.1 - Aggregate", "=" * 108, "Instruments: " + ", ".join(instruments)]
    lines += ["", "Aggregate Category Flow Table", "SourceCategory | DestinationCategory | " + " | ".join(f"Count_{instrument} | Prob_{instrument}" for instrument in instruments) + " | ReplicationCount | ReplicationPercent | MeanProbability | Classification"]
    flow_rows = []
    for source in CATEGORIES:
        for destination in CATEGORIES:
            values = [by_instrument[instrument].category_flows[(source, destination)] for instrument in instruments]
            valid = [row for row in values if row.count]
            flow_rows.append((source, destination, values, valid))
            cells = [value for row in values for value in (str(row.count), pct(row.probability))]
            lines.append(f"{source} | {destination} | " + " | ".join(cells) + f" | {len(valid)} | {pct(len(valid) / len(instruments))} | {pct(mean(row.probability for row in values))} | {replication_label(len(valid))}")
    lines += ["", "Aggregate Flow Balance Table", "Category | Inbound | Outbound | NetFlow"]
    for category in CATEGORIES:
        inbound = sum(result.balances[category].inbound for result in results)
        outbound = sum(result.balances[category].outbound for result in results)
        lines.append(f"{category} | {inbound} | {outbound} | {inbound - outbound}")
    lines += ["", "Aggregate Excursion Table", "Instrument | ExcursionCount | ExcursionProbability | MeanExcursionDuration | MedianExcursionDuration | MaximumExcursionDuration"]
    for instrument in instruments:
        row = by_instrument[instrument].excursion
        lines.append(f"{instrument} | {row.count} | {pct(row.probability)} | {fmt(row.mean_duration, 2)} | {fmt(row.median_duration, 2)} | {row.maximum_duration}")
    lines += ["", "Aggregate Return Table", "Instrument | ReturnCount | ReturnProbability | MeanReturnDuration | MedianReturnDuration"]
    for instrument in instruments:
        row = by_instrument[instrument].returns
        lines.append(f"{instrument} | {row.count} | {pct(row.probability)} | {fmt(row.mean_duration, 2)} | {fmt(row.median_duration, 2)}")
    lines += ["", "Aggregate Residence Table", "Category | MeanResidenceTime | MedianResidenceTime | MaximumResidenceTime"]
    for category in CATEGORIES:
        values = [duration for result in results for duration in result.residences[category].durations]
        lines.append(f"{category} | {fmt(mean(values), 2)} | {fmt(median(values), 2)} | {max(values, default=0)}")
    lines += ["", "Aggregate Persistence Table", "Category | StayProbability | LeaveProbability | PersistenceRatio"]
    for category in CATEGORIES:
        stay = mean(result.persistence[category].stay for result in results)
        leave = 1.0 - stay
        lines.append(f"{category} | {pct(stay)} | {pct(leave)} | {fmt(stay / leave if leave else math.inf)}")
    lines += ["", "Aggregate Entropy Table", "Category | TransitionEntropy | NormalizedEntropy"]
    for category in CATEGORIES:
        lines.append(f"{category} | {fmt(mean(result.entropy[category].entropy for result in results))} | {fmt(mean(result.entropy[category].normalized for result in results))}")
    lines += ["", "Aggregate Core Dominance Table", "Instrument | PercentTimeInCore | PercentTimeInSecondary | PercentTimeInTransition | PercentTimeInWeak | PercentTimeInInsufficient | CoreDominanceRatio"]
    for instrument in instruments:
        row = by_instrument[instrument].dominance
        lines.append(f"{instrument} | {pct(row.core)} | {pct(row.secondary)} | {pct(row.transition)} | {pct(row.weak)} | {pct(row.insufficient)} | {fmt(row.ratio)}")
    paths = sorted({path for result in results for path in result.pathways})
    lines += ["", "Aggregate Pathway Table", "Path | " + " | ".join(f"Probability_{instrument}" for instrument in instruments) + " | ReplicationCount | MeanProbability"]
    pathway_rows = []
    for path_key in paths:
        values = [by_instrument[instrument].pathways.get(path_key) for instrument in instruments]
        valid = [row for row in values if row and row.count]
        pathway_rows.append((path_key, valid))
        lines.append(f"{path_text(path_key)} | " + " | ".join(pct(row.probability) if row else "N/A" for row in values) + f" | {len(valid)} | {pct(mean(row.probability for row in valid))}")
    lines += ["", "Aggregate Outcome Table", "Category | " + " | ".join(f"Count_{instrument} | MeanDR_{instrument}" for instrument in instruments) + " | ValidInstrumentCount"]
    for category in CATEGORIES:
        values = [by_instrument[instrument].outcomes.get(category) for instrument in instruments]
        valid = [row for row in values if row and row.count]
        cells = [value for row in values for value in ((str(row.count), fmt(row.mean_dr)) if row else ("0", "N/A"))]
        lines.append(f"{category} | " + " | ".join(cells) + f" | {len(valid)}")
    label, reason = recommendation(results)
    common_paths = sorted(pathway_rows, key=lambda item: (-mean(row.probability for row in item[1]), item[0]))[:10]
    lines += [
        "", "Aggregate Recommendation", f"CoreDominanceClassification: {label}",
        f"Reason: {reason}",
        f"ExcursionBehavior: MeanProbability={pct(mean(result.excursion.probability for result in results))}; MeanDuration={fmt(mean(result.excursion.mean_duration for result in results), 2)} bars.",
        f"ReturnBehavior: MeanProbability={pct(mean(result.returns.probability for result in results))}; MeanDuration={fmt(mean(result.returns.mean_duration for result in results), 2)} bars.",
        "MostCommonFlowPaths: " + ", ".join(path_text(path_key) for path_key, _ in common_paths),
        f"ReplicationAssessment: {len(results)} instruments evaluated.",
        "", "Aggregate Rankings", "", "1. Most common category flows",
    ]
    for source, destination, values, valid in sorted(flow_rows, key=lambda item: (-mean(row.probability for row in item[2]), item[0], item[1]))[:20]:
        lines.append(f"{source} -> {destination} | MeanProbability={pct(mean(row.probability for row in values))} | ReplicationCount={len(valid)}")
    lines += ["", "2. Strongest Core -> Secondary flows"]
    lines.append(f"Core Node -> Secondary Node | MeanProbability={pct(mean(result.category_flows[('Core Node', 'Secondary Node')].probability for result in results))}")
    lines += ["", "3. Strongest Secondary -> Core flows"]
    lines.append(f"Secondary Node -> Core Node | MeanProbability={pct(mean(result.category_flows[('Secondary Node', 'Core Node')].probability for result in results))}")
    lines += ["", "4. Longest excursions"]
    for result in sorted(results, key=lambda row: -row.excursion.mean_duration):
        lines.append(f"{result.instrument} | MeanExcursionDuration={fmt(result.excursion.mean_duration, 2)}")
    lines += ["", "5. Fastest returns"]
    for result in sorted(results, key=lambda row: row.returns.mean_duration):
        lines.append(f"{result.instrument} | MeanReturnDuration={fmt(result.returns.mean_duration, 2)}")
    lines += ["", "6. Most persistent categories"]
    for category in sorted(CATEGORIES, key=lambda item: -mean(result.persistence[item].stay for result in results)):
        lines.append(f"{category} | MeanStayProbability={pct(mean(result.persistence[category].stay for result in results))}")
    lines += ["", "7. Highest entropy categories"]
    for category in sorted(CATEGORIES, key=lambda item: -mean(result.entropy[item].entropy for result in results)):
        lines.append(f"{category} | MeanTransitionEntropy={fmt(mean(result.entropy[category].entropy for result in results))}")
    lines += ["", "8. Lowest entropy categories"]
    for category in sorted(CATEGORIES, key=lambda item: mean(result.entropy[item].entropy for result in results)):
        lines.append(f"{category} | MeanTransitionEntropy={fmt(mean(result.entropy[category].entropy for result in results))}")
    lines += ["", "9. Strongest replicated pathways"]
    for path_key, valid in sorted(pathway_rows, key=lambda item: (-len(item[1]), -mean(row.probability for row in item[1]), item[0]))[:20]:
        lines.append(f"{path_text(path_key)} | ReplicationCount={len(valid)} | MeanProbability={pct(mean(row.probability for row in valid))}")
    lines += [
        "", "10. Recommended APVA flow model", f"State+Age+NodeCategories+FlowDynamics: {reason}",
        "", "Low-DoF Audit", "Variables used:", "StructuralState", "AgeBucket",
        "", "Derived metrics:", "ImportanceScore", "ReliabilityScore", "AttentionCategory",
        "", "No Context", "No Arbitration", "No Persistence", "No Phase",
        "No Optimization", "No Fitting", "No Machine Learning",
        "No Forward Returns used in flow construction",
        "", "Research Notes",
        "- Is APVA fundamentally Core-dominated? See the fixed classification.",
        "- Are excursions rare or common, and do they return rapidly? See excursion and return tables.",
        "- What are the dominant pathways? See category flow and pathway rankings.",
        "- Does behavior replicate? See cross-instrument category-flow probabilities.",
        "- Can APVA be reduced to State + Age + Node Categories + Flow Dynamics? This report tests that mechanically.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_invariants(results: list[FlowResult]) -> None:
    for result in results:
        if len(result.categories) != len(result.bars):
            raise RuntimeError(f"{result.instrument}: category stream length mismatch.")
        if sum(result.node_flows.values()) != max(0, len(result.nodes) - 1):
            raise RuntimeError(f"{result.instrument}: node transition count mismatch.")
        if sum(row.count for row in result.category_flows.values()) != max(0, len(result.categories) - 1):
            raise RuntimeError(f"{result.instrument}: category transition count mismatch.")
        for category in CATEGORIES:
            outgoing = [row for row in result.category_flows.values() if row.source == category]
            if sum(row.count for row in outgoing) and abs(sum(row.probability for row in outgoing) - 1.0) > 1e-12:
                raise RuntimeError(f"{result.instrument}: category probabilities do not sum to one for {category}.")
        if sum(result.balances[category].net for category in CATEGORIES) != 0:
            raise RuntimeError(f"{result.instrument}: flow balance does not net to zero.")
        if abs(sum((result.dominance.core, result.dominance.secondary, result.dominance.transition, result.dominance.weak, result.dominance.insufficient)) - 1.0) > 1e-12:
            raise RuntimeError(f"{result.instrument}: category time shares do not sum to one.")


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
    validate_importance(decay_results, aggregate)
    aggregate_results = [build_result(result, aggregate) for result in loaded]
    validate_invariants(aggregate_results)
    out_root = Path(args.out_root)
    for result, decay in zip(loaded, decay_results):
        local = local_rows(decay)
        score_rows(local)
        write_per_instrument(build_result(result, local), out_root)
    write_aggregate(aggregate_results, out_root)
    print(f"Wrote {len(aggregate_results)} per-instrument report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
