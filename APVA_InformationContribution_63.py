#!/usr/bin/env python3
"""APVA Information Contribution Study v0.1.

Audit which fixed StateAge nodes contribute most to the Study 62 memory
forecast. Forward price outcomes are diagnostics only.
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from APVA_InformationDecay_57 import study as decay_study
from APVA_MemoryDynamics_61 import build_result as build_dynamics
from APVA_MemoryForecast_62 import TARGETS, Metrics, forecast_rows, metrics, target
from APVA_MinimalEngine_52 import directional_return, instrument_columns, load_results, safe_mean
from APVA_NodeImportance_58 import NodeRow, aggregate_rows, local_rows, normalize, percentile, score_rows
from APVA_ProcessGraph_54 import Node, node_for, node_text
from APVA_StructuralForecast_56 import Outcome, outcome
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

AGE_BUCKETS = ("1", "2", "3", "4", "5", "6-10", "11-20", "21+")
COVERAGE_TARGETS = (0.50, 0.75, 0.90)


@dataclass
class Removal:
    node: Node
    count: int
    metrics: Metrics
    top1_loss: float
    top2_loss: float
    brier_worsening: float
    calibration_worsening: float
    entropy_increase: float
    coverage_loss: float
    score: float = 0.0
    classification: str = ""


@dataclass
class ClusterRemoval:
    node: Node
    removed: tuple[Node, ...]
    top1_loss: float
    brier_worsening: float
    coverage_loss: float


@dataclass
class Group:
    name: str
    total: float
    average: float
    percent: float
    count: int


@dataclass
class Coverage:
    target: float
    node_count: int
    coverage: float
    reduction: float


@dataclass
class Result:
    instrument: str
    source_paths: list
    bars: list
    nodes: list[Node]
    targets: list[str]
    node_rows: dict[Node, NodeRow]
    baseline: Metrics
    inventory: dict[Node, dict[str, float]]
    removals: dict[Node, Removal]
    clusters: dict[Node, ClusterRemoval]
    states: dict[str, Group]
    ages: dict[str, Group]
    categories: dict[str, Group]
    coverage: dict[float, Coverage]
    outcomes: dict[Node, Outcome]


def mean(values: Iterable[float]) -> float:
    return safe_mean(list(values))


def variance(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.pvariance(values) if len(values) > 1 else 0.0


def stateage_metrics(nodes: list[Node], targets: list[str]) -> Metrics:
    return metrics(forecast_rows(nodes, targets, TARGETS)[0])


def loss_row(node: Node, count: int, baseline: Metrics, degraded: Metrics) -> Removal:
    return Removal(
        node, count, degraded, baseline.top1 - degraded.top1, baseline.top2 - degraded.top2,
        degraded.brier - baseline.brier, degraded.calibration - baseline.calibration,
        degraded.entropy - baseline.entropy, count / baseline.count if baseline.count else 0.0,
    )


def classify(row: Removal) -> str:
    if abs(row.top1_loss) < 0.001 and abs(row.brier_worsening) < 0.001:
        return "NeutralNode"
    if row.top1_loss < 0 or row.brier_worsening < 0:
        return "HarmfulNode"
    return "HelpfulNode"


def score_removals(rows: dict[Node, Removal]) -> None:
    top1 = normalize({node: row.top1_loss for node, row in rows.items()})
    top2 = normalize({node: row.top2_loss for node, row in rows.items()})
    brier = normalize({node: row.brier_worsening for node, row in rows.items()})
    coverage = normalize({node: row.coverage_loss for node, row in rows.items()})
    for node, row in rows.items():
        row.score = mean((top1[node], top2[node], brier[node], coverage[node]))
        row.classification = classify(row)


def leave_out(nodes: list[Node], targets: list[str], removed: set[Node]) -> tuple[Metrics, int]:
    selected = [(node, value) for node, value in zip(nodes, targets) if node not in removed]
    return stateage_metrics([node for node, _ in selected], [value for _, value in selected]), len(nodes) - len(selected)


def neighbors(node: Node, available: set[Node]) -> tuple[Node, ...]:
    state, age = node
    index = AGE_BUCKETS.index(age)
    ages = AGE_BUCKETS[max(0, index - 1):index + 2]
    return tuple((state, item) for item in ages if (state, item) in available)


def group_rows(removals: dict[Node, Removal], key) -> dict[str, Group]:
    grouped: dict[str, list[Removal]] = defaultdict(list)
    for node, row in removals.items():
        grouped[key(node)].append(row)
    total = sum(row.score for row in removals.values())
    return {
        name: Group(name, sum(row.score for row in rows), mean(row.score for row in rows), sum(row.score for row in rows) / total if total else 0.0, len(rows))
        for name, rows in grouped.items()
    }


def coverage_rows(removals: dict[Node, Removal]) -> dict[float, Coverage]:
    rows = sorted(removals.values(), key=lambda row: (-row.score, row.node))
    total = sum(row.score for row in rows)
    output = {}
    for target_value in COVERAGE_TARGETS:
        cumulative = 0.0
        count = 0
        for row in rows:
            cumulative += row.score
            count += 1
            if total == 0 or cumulative / total >= target_value:
                break
        output[target_value] = Coverage(target_value, count, cumulative / total if total else 0.0, 1.0 - count / len(rows) if rows else 0.0)
    return output


def build_result(instrument: str, source_paths: list, bars: list, nodes: list[Node], targets: list[str], node_rows: dict[Node, NodeRow]) -> Result:
    current = nodes[:-1]
    baseline = stateage_metrics(current, targets)
    counts = {node: current.count(node) for node in sorted(set(current))}
    inventory = {}
    for node, count in counts.items():
        distribution = forecast_rows([item for item in current if item == node], [value for item, value in zip(current, targets) if item == node], TARGETS)[1][node]
        ranked = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
        inventory[node] = {"count": count, "class": ranked[0][0], "probability": ranked[0][1], "confidence": ranked[0][1] - ranked[1][1]}
    removals = {}
    for node, count in counts.items():
        degraded, removed_count = leave_out(current, targets, {node})
        removals[node] = loss_row(node, removed_count, baseline, degraded)
    score_removals(removals)
    clusters = {}
    available = set(current)
    for node in counts:
        removed = neighbors(node, available)
        degraded, removed_count = leave_out(current, targets, set(removed))
        row = loss_row(node, removed_count, baseline, degraded)
        clusters[node] = ClusterRemoval(node, removed, row.top1_loss, row.brier_worsening, row.coverage_loss)
    outcomes: dict[Node, list[float]] = defaultdict(list)
    for index, node in enumerate(nodes):
        value = directional_return(bars, index, 5)
        if value is not None:
            outcomes[node].append(value)
    category = lambda node: node_rows[node].category if node in node_rows else "Insufficient Node"
    return Result(
        instrument, source_paths, bars, nodes, targets, node_rows, baseline, inventory, removals, clusters,
        group_rows(removals, lambda node: node[0]), group_rows(removals, lambda node: node[1]),
        group_rows(removals, category), coverage_rows(removals),
        {node: outcome(values) for node, values in outcomes.items()},
    )


def combined_result(loaded, node_rows: dict[Node, NodeRow]) -> Result:
    bars, nodes, targets, paths = [], [], [], []
    for row in loaded:
        local_nodes = [node_for(bar) for bar in row.bars]
        bars.extend(row.bars)
        nodes.extend(local_nodes)
        targets.extend(target(node_rows[local_nodes[index + 1]].memory_strength - node_rows[local_nodes[index]].memory_strength) for index in range(len(local_nodes) - 1))
        paths.extend(row.source_paths)
    # A sentinel prevents the final bar of one instrument from being used as a current row.
    current_nodes = []
    for row in loaded:
        local = [node_for(bar) for bar in row.bars]
        current_nodes.extend(local[:-1])
    result = build_result("Aggregate", paths, bars, current_nodes + [current_nodes[-1]], targets, node_rows)
    return result


def thresholds(loaded, node_rows: dict[Node, NodeRow]) -> tuple[float, float]:
    values = [node_rows[node_for(bar)].memory_strength for row in loaded for bar in row.bars]
    return percentile(values, 0.25), percentile(values, 0.75)


def instrument_result(loaded, rows: dict[Node, NodeRow], low: float, high: float) -> Result:
    dynamics = build_dynamics(loaded, rows, low, high)
    targets = [target(dynamics.memory[index + 1] - dynamics.memory[index]) for index in range(len(dynamics.memory) - 1)]
    return build_result(loaded.instrument, loaded.source_paths, loaded.bars, dynamics.nodes, targets, rows)


def inventory_line(node: Node, result: Result) -> str:
    row, source = result.inventory[node], result.node_rows[node]
    return f"{node_text(node)} | {row['count']} | {source.replication_count} | {pct(source.replication_percent)} | {row['class']} | {pct(row['probability'])} | {pct(row['confidence'])} | {fmt(source.memory_strength)} | {fmt(source.half_life, 2)}"


def removal_line(row: Removal) -> str:
    return f"{node_text(row.node)} | {row.count} | {pct(row.top1_loss)} | {pct(row.top2_loss)} | {fmt(row.brier_worsening)} | {pct(row.calibration_worsening)} | {fmt(row.entropy_increase)} | {pct(row.coverage_loss)} | {fmt(row.score)} | {row.classification}"


def recommendation(result: Result) -> list[str]:
    compact, extended = result.coverage[0.75], result.coverage[0.90]
    ordered = sorted(result.removals.values(), key=lambda row: (-row.score, row.node))
    return [
        f"CompactModelSize: {compact.node_count}",
        f"ExtendedModelSize: {extended.node_count}",
        "RecommendedCoreNodes: " + ", ".join(node_text(row.node) for row in ordered[:compact.node_count]),
        "HarmfulNodes: " + ", ".join(node_text(row.node) for row in ordered if row.classification == "HarmfulNode"),
        "NeutralNodes: " + ", ".join(node_text(row.node) for row in ordered if row.classification == "NeutralNode"),
    ]


def append_audit(lines: list[str], heading: str) -> None:
    lines += ["", heading, "Variables used:", "StructuralState", "AgeBucket", "StateAgeNode", "MemoryStrength", "", "No Context", "No Arbitration", "No Persistence", "No Phase", "No Optimization", "No Fitting", "No Machine Learning", "No Forward Returns used in contribution scoring"]


def report_lines(result: Result, prefix: str = "") -> list[str]:
    lines = [prefix + "Baseline StateAge Model", "Metric | Value", f"Top1Accuracy | {pct(result.baseline.top1)}", f"Top2Accuracy | {pct(result.baseline.top2)}", f"BrierScore | {fmt(result.baseline.brier)}", f"CalibrationError | {pct(result.baseline.calibration)}", f"ForecastEntropy | {fmt(result.baseline.entropy)}", "Coverage | 100.00%"]
    lines += ["", prefix + "Node Inventory", "Node | Count | ReplicationCount | ReplicationPercent | ForecastClass | ForecastProbability | ForecastConfidence | MemoryStrength | HalfLife"] + [inventory_line(node, result) for node in sorted(result.inventory)]
    lines += ["", prefix + "Leave-One-Node-Out Analysis", "Node | Count | Top1Loss | Top2Loss | BrierWorsening | CalibrationWorsening | EntropyIncrease | CoverageLoss | ContributionScore | Classification"] + [removal_line(row) for row in sorted(result.removals.values(), key=lambda row: (-row.score, row.node))]
    lines += ["", prefix + "Contribution Score", "Equal-weight normalized average of Top1Loss, Top2Loss, BrierWorsening, and CoverageLoss."]
    lines += ["", prefix + "Negative Contribution Detection", "HelpfulNode: removal degrades performance. HarmfulNode: removal improves performance. NeutralNode: fixed tolerance rule."]
    lines += ["", prefix + "Cumulative Contribution Curve", "CoverageTarget | NodeCount | CoveragePercent | ReductionPercent"] + [f"{pct(row.target)} | {row.node_count} | {pct(row.coverage)} | {pct(row.reduction)}" for row in result.coverage.values()]
    for title, rows in (("State-Level Contribution", result.states), ("Age-Level Contribution", result.ages), ("Core vs Non-Core Contribution", result.categories)):
        lines += ["", prefix + title, "Group | TotalContributionScore | MeanContributionScore | ContributionPercent | NodeCount"] + [f"{row.name} | {fmt(row.total)} | {fmt(row.average)} | {pct(row.percent)} | {row.count}" for row in sorted(rows.values(), key=lambda row: (-row.total, row.name))]
    lines += ["", prefix + "Information Efficiency", "Node | ContributionScore | Count | ContributionPerObservation"] + [f"{node_text(row.node)} | {fmt(row.score)} | {row.count} | {fmt(row.score / row.count if row.count else 0.0, 8)}" for row in sorted(result.removals.values(), key=lambda row: (-(row.score / row.count if row.count else 0.0), row.node))]
    lines += ["", prefix + "Redundancy Test", "Node | ClusterRemoved | ClusterTop1Loss | ClusterBrierWorsening | ClusterCoverageLoss"] + [f"{node_text(row.node)} | {', '.join(node_text(node) for node in row.removed)} | {pct(row.top1_loss)} | {fmt(row.brier_worsening)} | {pct(row.coverage_loss)}" for row in sorted(result.clusters.values(), key=lambda row: (-row.top1_loss, row.node))]
    return lines


def write_per_instrument(result: Result, out_root: Path) -> None:
    path = out_root / result.instrument / f"InformationContribution_{result.instrument}.txt"
    ensure_dir(path.parent)
    lines = ["APVA Information Contribution Study v0.1", "=" * 108, "Diagnostics", f"Instrument: {result.instrument}", "Input path(s): " + ", ".join(str(item) for item in result.source_paths), f"Total rows: {len(result.bars)}", f"Node count: {len(result.inventory)}"]
    sections = report_lines(result)
    section_numbers = {"Baseline StateAge Model": 1, "Node Inventory": 2, "Leave-One-Node-Out Analysis": 3, "Contribution Score": 4, "Negative Contribution Detection": 5, "Cumulative Contribution Curve": 6, "State-Level Contribution": 7, "Age-Level Contribution": 8, "Core vs Non-Core Contribution": 9, "Information Efficiency": 11, "Redundancy Test": 12}
    for line in sections:
        if line == "Information Efficiency":
            lines += ["", "10. Robustness by Instrument", "See aggregate report."]
        if line in section_numbers:
            lines += ["", f"{section_numbers[line]}. {line}"]
        else:
            lines.append(line)
    lines += ["", "13. Outcome Diagnostics", "Diagnostic only. Outcomes are not used in contribution scoring.", "Node | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew"]
    lines += [f"{node_text(node)} | {row.count} | {fmt(row.mean_dr)} | {fmt(row.median_dr)} | {pct(row.continuation)} | {pct(row.failure)} | {pct(row.flat)} | {fmt(row.skew)}" for node, row in sorted(result.outcomes.items())]
    lines += ["", "14. Recommendation"] + recommendation(result)
    append_audit(lines, "15. Low-DoF Audit")
    lines += ["", "16. Mechanical Research Notes", "- Contribution scoring uses fixed Study 62 StateAge lookup removals.", "- Neighbor-age removals are diagnostic redundancy checks.", "- Forward outcomes are diagnostic only."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aggregate(result: Result, instrument_results: list[Result], out_root: Path) -> None:
    path = out_root / "InformationContribution" / "InformationContribution_All.txt"
    ensure_dir(path.parent)
    instruments = instrument_columns(instrument_results)
    by_name = {row.instrument: row for row in instrument_results}
    lines = ["APVA Information Contribution Study v0.1 - Aggregate", "=" * 108, "Instruments: " + ", ".join(instruments)]
    lines += ["", "Aggregate Baseline Table", "Metric | " + " | ".join(instruments) + " | Aggregate"]
    for label, getter in (("Top1Accuracy", lambda row: pct(row.baseline.top1)), ("Top2Accuracy", lambda row: pct(row.baseline.top2)), ("BrierScore", lambda row: fmt(row.baseline.brier)), ("CalibrationError", lambda row: pct(row.baseline.calibration)), ("ForecastEntropy", lambda row: fmt(row.baseline.entropy)), ("Coverage", lambda row: "100.00%")):
        lines.append(f"{label} | " + " | ".join(getter(by_name[item]) for item in instruments) + f" | {getter(result)}")
    lines += ["", "Aggregate Node Contribution Table", "Node | Count | Top1Loss | Top2Loss | BrierWorsening | CalibrationWorsening | EntropyIncrease | CoverageLoss | ContributionScore | Classification | ReplicationCount"]
    lines += [removal_line(row) + f" | {result.node_rows[row.node].replication_count}" for row in sorted(result.removals.values(), key=lambda row: (-row.score, row.node))]
    lines += aggregate_group_table("Aggregate State Contribution Table", result.states, "State")
    lines += aggregate_group_table("Aggregate Age Contribution Table", result.ages, "AgeBucket")
    lines += aggregate_group_table("Aggregate Category Contribution Table", result.categories, "Category")
    lines += ["", "Aggregate Cumulative Contribution Table", "CoverageTarget | NodeCount | CoveragePercent | ReductionPercent"] + [f"{pct(row.target)} | {row.node_count} | {pct(row.coverage)} | {pct(row.reduction)}" for row in result.coverage.values()]
    lines += ["", "Aggregate Instrument Robustness Table", "Node | " + " | ".join(f"Contribution_{item}" for item in instruments) + " | MeanContribution | ContributionVariance | ReplicationCount"]
    for node in sorted(result.removals):
        values = [by_name[item].removals[node].score if node in by_name[item].removals else 0.0 for item in instruments]
        lines.append(f"{node_text(node)} | " + " | ".join(fmt(value) for value in values) + f" | {fmt(mean(values))} | {fmt(variance(values))} | {sum(node in by_name[item].removals for item in instruments)}")
    lines += ["", "Aggregate Information Efficiency Table", "Node | ContributionScore | Count | ContributionPerObservation"] + [f"{node_text(row.node)} | {fmt(row.score)} | {row.count} | {fmt(row.score / row.count if row.count else 0.0, 8)}" for row in sorted(result.removals.values(), key=lambda row: (-(row.score / row.count if row.count else 0.0), row.node))]
    lines += ["", "Aggregate Redundancy Table", "Node | ClusterRemoved | ClusterTop1Loss | ClusterBrierWorsening | ClusterCoverageLoss"] + [f"{node_text(row.node)} | {', '.join(node_text(node) for node in row.removed)} | {pct(row.top1_loss)} | {fmt(row.brier_worsening)} | {pct(row.coverage_loss)}" for row in sorted(result.clusters.values(), key=lambda row: (-row.top1_loss, row.node))]
    lines += aggregate_outcomes(instrument_results, instruments)
    lines += ["", "Aggregate Recommendation"] + recommendation(result) + [f"ReplicationAssessment: {len(instrument_results)} instruments evaluated."]
    lines += rankings(result)
    append_audit(lines, "Low-DoF Audit")
    lines += ["", "Research Notes", "- Which StateAge nodes carry forecast information? See ranked fixed removals.", "- Is forecast power broad or concentrated? See cumulative contribution.", "- Can weak nodes be removed? See NeutralNode and HarmfulNode labels.", "- Does power replicate? See instrument robustness table.", "- Compact StateAge reduction is reported mechanically, not optimized."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_group_table(title: str, rows: dict[str, Group], label: str) -> list[str]:
    return ["", title, f"{label} | TotalContributionScore | MeanContributionScore | ContributionPercent | NodeCount"] + [f"{row.name} | {fmt(row.total)} | {fmt(row.average)} | {pct(row.percent)} | {row.count}" for row in sorted(rows.values(), key=lambda row: (-row.total, row.name))]


def aggregate_outcomes(results: list[Result], instruments: list[str]) -> list[str]:
    nodes = sorted({node for row in results for node in row.outcomes})
    by_name = {row.instrument: row for row in results}
    lines = ["", "Aggregate Outcome Table", "Node | " + " | ".join(f"Count_{item} | MeanDR_{item}" for item in instruments) + " | ValidInstrumentCount"]
    for node in nodes:
        values = [by_name[item].outcomes.get(node) for item in instruments]
        lines.append(f"{node_text(node)} | " + " | ".join(value for row in values for value in ((str(row.count), fmt(row.mean_dr)) if row else ("0", "N/A"))) + f" | {sum(row is not None for row in values)}")
    return lines


def rankings(result: Result) -> list[str]:
    ordered = sorted(result.removals.values(), key=lambda row: (-row.score, row.node))
    efficiency = sorted(ordered, key=lambda row: (-(row.score / row.count if row.count else 0.0), row.node))
    clusters = sorted(result.clusters.values(), key=lambda row: (-row.top1_loss, row.node))
    lines = ["", "Aggregate Rankings", "", "1. Highest contribution nodes"] + [f"{node_text(row.node)} | {fmt(row.score)}" for row in ordered[:10]]
    lines += ["", "2. Lowest contribution nodes"] + [f"{node_text(row.node)} | {fmt(row.score)}" for row in reversed(ordered[-10:])]
    lines += ["", "3. Harmful nodes"] + [node_text(row.node) for row in ordered if row.classification == "HarmfulNode"]
    lines += ["", "4. Neutral nodes"] + [node_text(row.node) for row in ordered if row.classification == "NeutralNode"]
    lines += ["", "5. Highest contribution states"] + [f"{row.name} | {fmt(row.total)}" for row in sorted(result.states.values(), key=lambda row: (-row.total, row.name))]
    lines += ["", "6. Highest contribution age buckets"] + [f"{row.name} | {fmt(row.total)}" for row in sorted(result.ages.values(), key=lambda row: (-row.total, row.name))]
    lines += ["", "7. Highest contribution categories"] + [f"{row.name} | {fmt(row.total)}" for row in sorted(result.categories.values(), key=lambda row: (-row.total, row.name))]
    lines += ["", "8. Most efficient rare nodes"] + [f"{node_text(row.node)} | {fmt(row.score / row.count if row.count else 0.0, 8)}" for row in efficiency[:10]]
    lines += ["", "9. Most redundant node clusters"] + [f"{node_text(row.node)} | {pct(row.top1_loss)}" for row in clusters[:10]]
    lines += ["", "10. Recommended compact StateAge model", f"{result.coverage[0.75].node_count} nodes for {pct(result.coverage[0.75].coverage)} contribution coverage."]
    return lines


def validate_invariants(result: Result) -> None:
    if result.baseline.count != len(result.targets):
        raise RuntimeError(f"{result.instrument}: baseline count mismatch.")
    if set(result.inventory) != set(result.removals):
        raise RuntimeError(f"{result.instrument}: inventory/removal mismatch.")
    if any(not 0.0 <= row.score <= 1.0 for row in result.removals.values()):
        raise RuntimeError(f"{result.instrument}: contribution score escaped [0, 1].")
    counts = [result.coverage[target].node_count for target in COVERAGE_TARGETS]
    values = [result.coverage[target].coverage for target in COVERAGE_TARGETS]
    if counts != sorted(counts) or values != sorted(values):
        raise RuntimeError(f"{result.instrument}: contribution coverage is not monotonic.")


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
    aggregate_rows_scored = aggregate_rows(decay)
    score_rows(aggregate_rows_scored)
    low, high = thresholds(loaded, aggregate_rows_scored)
    instrument_results = []
    for loaded_row, decay_row in zip(loaded, decay):
        local = local_rows(decay_row)
        score_rows(local)
        result = instrument_result(loaded_row, local, low, high)
        validate_invariants(result)
        instrument_results.append(result)
    aggregate = combined_result(loaded, aggregate_rows_scored)
    validate_invariants(aggregate)
    out_root = Path(args.out_root)
    for result in instrument_results:
        write_per_instrument(result, out_root)
    write_aggregate(aggregate, instrument_results, out_root)
    print(f"Wrote {len(instrument_results)} per-instrument report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
