#!/usr/bin/env python3
"""APVA Harmful Node Heterogeneity Study v0.1.

Test whether Study 63 harmful StateAge nodes are low-information nodes or
heterogeneous mixtures under allowed structural context only.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable

from APVA_InformationContribution_63 import combined_result, instrument_result, thresholds
from APVA_InformationDecay_57 import study as decay_study
from APVA_MemoryDynamics_61 import build_result as build_dynamics
from APVA_MemoryForecast_62 import TARGETS, bucket, forecast_rows, metrics, target
from APVA_MinimalEngine_52 import directional_return, instrument_columns, load_results, safe_mean
from APVA_NodeImportance_58 import NodeRow, aggregate_rows, local_rows, score_rows
from APVA_ProcessGraph_54 import Node, node_for, node_text
from APVA_StructuralForecast_56 import Outcome, outcome
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

DIMENSIONS = ("PreviousNode", "NextNode", "MemoryClass", "ConfidenceBucket", "EntropyBucket", "NodeCategory")
DIMENSION_TITLES = {
    "PreviousNode": "Previous Node Split",
    "NextNode": "Next Node Split",
    "MemoryClass": "Memory Class Split",
    "ConfidenceBucket": "Forecast Confidence Split",
    "EntropyBucket": "Entropy Split",
    "NodeCategory": "Node Category Split",
}


@dataclass
class StreamRow:
    index: int
    node: Node
    target: str
    previous: str
    next_node: str
    memory_class: str
    confidence_bucket: str
    entropy_bucket: str
    category: str
    confidence: float
    entropy: float


@dataclass
class SplitRow:
    node: Node
    dimension: str
    count: int
    original_keys: int
    new_keys: int
    sparse_rate: float
    average_count: float
    baseline_top1: float
    split_top1: float
    top1_gain: float
    top2_gain: float
    brier_improvement: float
    calibration_improvement: float
    entropy_reduction: float
    heterogeneity: float
    information_gain: float = 0.0
    net_value: float = 0.0


@dataclass
class Result:
    instrument: str
    source_paths: list
    bars: list
    rows: list[StreamRow]
    harmful: dict[Node, object]
    node_rows: dict[Node, NodeRow]
    splits: dict[tuple[Node, str], SplitRow]
    outcomes: dict[tuple[Node, str, str], Outcome]


def mean(values: Iterable[float]) -> float:
    return safe_mean(list(values))


def variance(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.pvariance(values) if len(values) > 1 else 0.0


def normalize(values: dict[Hashable, float]) -> dict[Hashable, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high == low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def dim_value(row: StreamRow, dimension: str) -> str:
    return {
        "PreviousNode": row.previous,
        "NextNode": row.next_node,
        "MemoryClass": row.memory_class,
        "ConfidenceBucket": row.confidence_bucket,
        "EntropyBucket": row.entropy_bucket,
        "NodeCategory": row.category,
    }[dimension]


def group_metrics(rows: list[StreamRow], key_func) -> tuple[object, object, Counter]:
    keys = [key_func(row) for row in rows]
    targets = [row.target for row in rows]
    forecasts, _, counts = forecast_rows(keys, targets, TARGETS)
    return metrics(forecasts), forecasts, counts


def heterogeneity(rows: list[StreamRow], dimension: str) -> float:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[dim_value(row, dimension)].append(row.target)
    if len(groups) <= 1:
        return 0.0
    rates = []
    within = []
    total = sum(len(values) for values in groups.values())
    for values in groups.values():
        counts = Counter(values)
        p = max(counts.values()) / len(values)
        rates.extend([p] * len(values))
        within.append((len(values) / total) * p * (1.0 - p))
    return variance(rates) / (sum(within) + 1e-12)


def score_splits(splits: dict[tuple[Node, str], SplitRow]) -> None:
    top1 = normalize({key: row.top1_gain for key, row in splits.items()})
    top2 = normalize({key: row.top2_gain for key, row in splits.items()})
    brier = normalize({key: row.brier_improvement for key, row in splits.items()})
    cal = normalize({key: row.calibration_improvement for key, row in splits.items()})
    ent = normalize({key: row.entropy_reduction for key, row in splits.items()})
    sparse = normalize({key: row.sparse_rate for key, row in splits.items()})
    keys = normalize({key: row.new_keys - row.original_keys for key, row in splits.items()})
    for key, row in splits.items():
        row.information_gain = mean((top1[key], top2[key], brier[key], cal[key], ent[key]))
        row.net_value = row.information_gain - mean((sparse[key], keys[key]))


def build_stream(loaded, rows: dict[Node, NodeRow], memory_thresholds: tuple[float, float], confidence_thresholds: tuple[float, float], entropy_thresholds: tuple[float, float], index_offset: int = 0) -> list[StreamRow]:
    dynamics = build_dynamics(loaded, rows, *memory_thresholds)
    stream = []
    for index in range(len(dynamics.nodes) - 1):
        node = dynamics.nodes[index]
        previous = node_text(dynamics.nodes[index - 1]) if index else "START"
        next_node = node_text(dynamics.nodes[index + 1])
        stream.append(StreamRow(
            index + index_offset, node, target(dynamics.memory[index + 1] - dynamics.memory[index]),
            previous, next_node, dynamics.memory_classes[index],
            bucket(dynamics.confidence[index], *confidence_thresholds),
            bucket(dynamics.entropy[index], *entropy_thresholds),
            dynamics.categories[index], dynamics.confidence[index], dynamics.entropy[index],
        ))
    return stream


def build_result(name: str, source_paths: list, bars: list, stream: list[StreamRow], contribution, node_rows: dict[Node, NodeRow]) -> Result:
    harmful = {node: row for node, row in contribution.removals.items() if row.classification == "HarmfulNode"}
    splits = {}
    for node in sorted(harmful):
        rows = [row for row in stream if row.node == node]
        if not rows:
            continue
        baseline, _, _ = group_metrics(rows, lambda row: row.node)
        for dimension in DIMENSIONS:
            split, _, counts = group_metrics(rows, lambda row, d=dimension: (row.node, dim_value(row, d)))
            sparse = mean(count < 50 for count in counts.values())
            splits[(node, dimension)] = SplitRow(
                node, dimension, len(rows), 1, len(counts), sparse, mean(counts.values()),
                baseline.top1, split.top1, split.top1 - baseline.top1, split.top2 - baseline.top2,
                baseline.brier - split.brier, baseline.calibration - split.calibration,
                baseline.entropy - split.entropy, heterogeneity(rows, dimension),
            )
    score_splits(splits)
    values: dict[tuple[Node, str, str], list[float]] = defaultdict(list)
    for row in stream:
        if row.node not in harmful:
            continue
        value = directional_return(bars, row.index, 5)
        if value is not None:
            for dimension in DIMENSIONS:
                values[(row.node, dimension, dim_value(row, dimension))].append(value)
    return Result(name, source_paths, bars, stream, harmful, node_rows, splits, {key: outcome(samples) for key, samples in values.items()})


def split_lines(result: Result, dimension: str) -> list[str]:
    title = DIMENSION_TITLES.get(dimension, f"{dimension} Split")
    lines = [title, "Node | Subgroup | Count | Top1Accuracy | BrierScore | ForecastConfidence | Entropy"]
    for node in sorted(result.harmful):
        rows = [row for row in result.rows if row.node == node]
        groups: dict[str, list[StreamRow]] = defaultdict(list)
        for row in rows:
            groups[dim_value(row, dimension)].append(row)
        for subgroup, members in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
            m, forecasts, _ = group_metrics(members, lambda row, value=subgroup: value)
            lines.append(f"{node_text(node)} | {subgroup} | {len(members)} | {pct(m.top1)} | {fmt(m.brier)} | {pct(mean(row.confidence for row in forecasts))} | {fmt(m.entropy)}")
    return lines


def useful_action(rows: list[SplitRow]) -> tuple[str, str]:
    candidates = [row for row in rows if row.net_value > 0 and row.count >= 50]
    if not candidates:
        return "No Refinement", "No split had positive NetValueScore with sufficient sample."
    best = max(candidates, key=lambda row: (row.net_value, row.dimension))
    return f"Split By {best.dimension}", f"NetValueScore={fmt(best.net_value)}; Count={best.count}; SparseKeyRate={pct(best.sparse_rate)}."


def conclusion(rows: list[SplitRow], replication: int) -> str:
    best = max((row.net_value for row in rows), default=0.0)
    if best > 0.25 and replication >= 2:
        return "RefinementJustified"
    if best > 0.25:
        return "StrongHeterogeneity"
    if best > 0.10:
        return "ModerateHeterogeneity"
    if best > 0.0:
        return "WeakHeterogeneity"
    return "RefinementNotJustified"


def write_per_instrument(result: Result, out_root: Path) -> None:
    path = out_root / result.instrument / f"HarmfulNodeHeterogeneity_{result.instrument}.txt"
    ensure_dir(path.parent)
    lines = ["APVA Harmful Node Heterogeneity Study v0.1", "=" * 108, "Diagnostics", f"Instrument: {result.instrument}", "Input path(s): " + ", ".join(str(item) for item in result.source_paths), f"Total rows: {len(result.bars)}", f"Harmful node count: {len(result.harmful)}"]
    lines += ["", "1. Harmful Node Inventory", "Node | Count | ReplicationCount | ContributionScore | Top1Loss | BrierWorsening"]
    for node, row in sorted(result.harmful.items()):
        lines.append(f"{node_text(node)} | {row.count} | {result.node_rows[node].replication_count} | {fmt(row.score)} | {pct(row.top1_loss)} | {fmt(row.brier_worsening)}")
    lines += ["", "2. Baseline Harmfulness", "Node | BaselineTop1Loss | BaselineBrierWorsening | BaselineContribution"]
    lines += [f"{node_text(node)} | {pct(row.top1_loss)} | {fmt(row.brier_worsening)} | {fmt(row.score)}" for node, row in sorted(result.harmful.items())]
    for number, dimension in enumerate(DIMENSIONS, start=3):
        lines += ["", f"{number}. {DIMENSION_TITLES.get(dimension, dimension + ' Split')}"] + split_lines(result, dimension)[1:]
    lines += ["", "9. Single-Factor Heterogeneity Score", "Node | Dimension | HeterogeneityScore"] + [f"{node_text(row.node)} | {row.dimension} | {fmt(row.heterogeneity)}" for row in sorted(result.splits.values(), key=lambda row: (-row.heterogeneity, row.node, row.dimension))]
    lines += ["", "10. Useful Split Test", "Node | Dimension | Top1Gain | Top2Gain | BrierImprovement | CalibrationImprovement | ForecastEntropy"] + [f"{node_text(row.node)} | {row.dimension} | {pct(row.top1_gain)} | {pct(row.top2_gain)} | {fmt(row.brier_improvement)} | {pct(row.calibration_improvement)} | {fmt(row.entropy_reduction)}" for row in sorted(result.splits.values(), key=lambda row: (-row.top1_gain, row.node))]
    lines += ["", "11. Fragmentation Cost", "Node | Dimension | OriginalKeys | NewKeys | SparseKeyRate | AverageCountPerKey"] + [f"{node_text(row.node)} | {row.dimension} | {row.original_keys} | {row.new_keys} | {pct(row.sparse_rate)} | {fmt(row.average_count, 2)}" for row in sorted(result.splits.values(), key=lambda row: (-row.new_keys, row.node))]
    lines += ["", "12. Information Gain", "Node | Dimension | InformationGainScore"] + [f"{node_text(row.node)} | {row.dimension} | {fmt(row.information_gain)}" for row in sorted(result.splits.values(), key=lambda row: (-row.information_gain, row.node))]
    lines += ["", "13. Net Value Score", "Node | Dimension | NetValueScore"] + [f"{node_text(row.node)} | {row.dimension} | {fmt(row.net_value)}" for row in sorted(result.splits.values(), key=lambda row: (-row.net_value, row.node))]
    lines += ["", "14. Cross-Instrument Replication", "See aggregate report."]
    lines += ["", "15. Candidate Node Refinements", "Node | RecommendedAction | Reason"]
    for node in sorted(result.harmful):
        action, reason = useful_action([row for row in result.splits.values() if row.node == node])
        lines.append(f"{node_text(node)} | {action} | {reason}")
    lines += ["", "16. Outcome Diagnostics", "Diagnostic only. Outcomes are not used in refinement decisions.", "Node | Dimension | Subgroup | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew"]
    for (node, dimension, subgroup), row in sorted(result.outcomes.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        lines.append(f"{node_text(node)} | {dimension} | {subgroup} | {row.count} | {fmt(row.mean_dr)} | {fmt(row.median_dr)} | {pct(row.continuation)} | {pct(row.failure)} | {pct(row.flat)} | {fmt(row.skew)}")
    lines += ["", "17. Recommendation", f"Conclusion: {conclusion(list(result.splits.values()), 1)}"]
    append_audit(lines)
    lines += ["", "19. Mechanical Research Notes", "- Splits are diagnostics, not new APVA states.", "- Split scores use memory-forecast targets only.", "- Forward outcomes are diagnostic only."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_audit(lines: list[str]) -> None:
    lines += ["", "18. Low-DoF Audit", "Variables used:", "StructuralState", "AgeBucket", "StateAgeNode", "NodeCategory", "MemoryStrength", "ForecastConfidence", "EntropyGrowth", "", "No Context", "No Arbitration", "No Persistence", "No Phase", "No Optimization", "No Fitting", "No Machine Learning", "No Forward Returns used in refinement scoring"]


def aggregate_tables(result: Result, instrument_results: list[Result], instruments: list[str]) -> list[str]:
    by_name = {row.instrument: row for row in instrument_results}
    lines = ["", "Aggregate Harmful Node Table", "Node | Count | ReplicationCount | ContributionScore | BaselineTop1Loss | BaselineBrierWorsening"]
    for node, row in sorted(result.harmful.items()):
        lines.append(f"{node_text(node)} | {row.count} | {result.node_rows[node].replication_count} | {fmt(row.score)} | {pct(row.top1_loss)} | {fmt(row.brier_worsening)}")
    lines += ["", "Aggregate Heterogeneity Table", "Node | Dimension | HeterogeneityScore | ReplicationCount"]
    for row in sorted(result.splits.values(), key=lambda row: (-row.heterogeneity, row.node, row.dimension)):
        rep = sum((row.node, row.dimension) in item.splits for item in instrument_results)
        lines.append(f"{node_text(row.node)} | {row.dimension} | {fmt(row.heterogeneity)} | {rep}")
    lines += ["", "Aggregate Useful Split Table", "Node | Dimension | Top1Gain | Top2Gain | BrierImprovement | CalibrationImprovement | EntropyReduction"]
    lines += [f"{node_text(row.node)} | {row.dimension} | {pct(row.top1_gain)} | {pct(row.top2_gain)} | {fmt(row.brier_improvement)} | {pct(row.calibration_improvement)} | {fmt(row.entropy_reduction)}" for row in sorted(result.splits.values(), key=lambda row: (-row.top1_gain, row.node))]
    lines += ["", "Aggregate Fragmentation Table", "Node | Dimension | OriginalKeys | NewKeys | SparseKeyRateIncrease | AverageCountPerKey"]
    lines += [f"{node_text(row.node)} | {row.dimension} | {row.original_keys} | {row.new_keys} | {pct(row.sparse_rate)} | {fmt(row.average_count, 2)}" for row in sorted(result.splits.values(), key=lambda row: (-row.new_keys, row.node))]
    lines += ["", "Aggregate Information Gain Table", "Node | Dimension | InformationGainScore"] + [f"{node_text(row.node)} | {row.dimension} | {fmt(row.information_gain)}" for row in sorted(result.splits.values(), key=lambda row: (-row.information_gain, row.node))]
    lines += ["", "Aggregate Net Value Table", "Node | Dimension | NetValueScore"] + [f"{node_text(row.node)} | {row.dimension} | {fmt(row.net_value)}" for row in sorted(result.splits.values(), key=lambda row: (-row.net_value, row.node))]
    lines += ["", "Aggregate Replication Table", "Node | Dimension | " + " | ".join(f"Score_{name}" for name in instruments) + " | ReplicationCount"]
    for key, row in sorted(result.splits.items(), key=lambda item: (-item[1].net_value, item[0])):
        scores = [by_name[name].splits[key].net_value if key in by_name[name].splits else 0.0 for name in instruments]
        lines.append(f"{node_text(key[0])} | {key[1]} | " + " | ".join(fmt(score) for score in scores) + f" | {sum(key in by_name[name].splits for name in instruments)}")
    lines += ["", "Aggregate Recommendation Table", "Node | RecommendedAction | Reason | NetValueScore | ReplicationCount"]
    for node in sorted(result.harmful):
        rows = [row for row in result.splits.values() if row.node == node]
        action, reason = useful_action(rows)
        best = max(rows, key=lambda row: row.net_value, default=None)
        rep = sum(best and (best.node, best.dimension) in item.splits for item in instrument_results)
        lines.append(f"{node_text(node)} | {action} | {reason} | {fmt(best.net_value if best else 0.0)} | {rep}")
    lines += aggregate_outcomes(instrument_results, instruments)
    return lines


def aggregate_outcomes(results: list[Result], instruments: list[str]) -> list[str]:
    keys = sorted({key for row in results for key in row.outcomes})
    by_name = {row.instrument: row for row in results}
    lines = ["", "Aggregate Outcome Table", "Node | Subgroup | " + " | ".join(f"Count_{name} | MeanDR_{name}" for name in instruments) + " | ValidInstrumentCount"]
    for node, dimension, subgroup in keys:
        values = [by_name[name].outcomes.get((node, dimension, subgroup)) for name in instruments]
        lines.append(f"{node_text(node)} | {dimension}={subgroup} | " + " | ".join(value for row in values for value in ((str(row.count), fmt(row.mean_dr)) if row else ("0", "N/A"))) + f" | {sum(row is not None for row in values)}")
    return lines


def rankings(result: Result) -> list[str]:
    rows = list(result.splits.values())
    lines = ["", "AGGREGATE RANKINGS", "", "1. Most heterogeneous harmful nodes"] + [f"{node_text(row.node)} | {row.dimension} | {fmt(row.heterogeneity)}" for row in sorted(rows, key=lambda row: (-row.heterogeneity, row.node))[:10]]
    lines += ["", "2. Least heterogeneous harmful nodes"] + [f"{node_text(row.node)} | {row.dimension} | {fmt(row.heterogeneity)}" for row in sorted(rows, key=lambda row: (row.heterogeneity, row.node))[:10]]
    for number, title, dimension in ((3, "Best PreviousNode refinements", "PreviousNode"), (4, "Best NextNode refinements", "NextNode"), (5, "Best MemoryClass refinements", "MemoryClass"), (6, "Best Confidence refinements", "ConfidenceBucket"), (7, "Best Entropy refinements", "EntropyBucket")):
        lines += ["", f"{number}. {title}"] + [f"{node_text(row.node)} | {fmt(row.net_value)}" for row in sorted((row for row in rows if row.dimension == dimension), key=lambda row: (-row.net_value, row.node))[:10]]
    lines += ["", "8. Highest NetValue refinements"] + [f"{node_text(row.node)} | {row.dimension} | {fmt(row.net_value)}" for row in sorted(rows, key=lambda row: (-row.net_value, row.node))[:10]]
    lines += ["", "9. Refinements not worth complexity"] + [f"{node_text(row.node)} | {row.dimension} | {fmt(row.net_value)}" for row in sorted(rows, key=lambda row: (row.net_value, row.node))[:10]]
    lines += ["", "10. Recommended APVA node refinements"] + [f"{node_text(node)} | {useful_action([row for row in rows if row.node == node])[0]}" for node in sorted(result.harmful)]
    return lines


def write_aggregate(result: Result, instrument_results: list[Result], out_root: Path) -> None:
    path = out_root / "HarmfulNodeHeterogeneity" / "HarmfulNodeHeterogeneity_All.txt"
    ensure_dir(path.parent)
    instruments = instrument_columns(instrument_results)
    best_rep = max((sum(key in item.splits and item.splits[key].net_value > 0 for item in instrument_results) for key in result.splits), default=0)
    lines = ["APVA Harmful Node Heterogeneity Study v0.1 - Aggregate", "=" * 108, "Instruments: " + ", ".join(instruments)]
    lines += aggregate_tables(result, instrument_results, instruments)
    lines += rankings(result)
    lines += ["", "Recommendation", f"Conclusion: {conclusion(list(result.splits.values()), best_rep)}", f"BestReplicationCount: {best_rep}", "Mechanical conclusion only; no split is promoted to a new APVA state."]
    append_audit(lines)
    lines += ["", "RESEARCH NOTES", "- Are harmful nodes truly harmful? See baseline harmfulness.", "- Are they mixtures? See heterogeneity and split-gain tables.", "- Does information gain justify complexity? See NetValueScore.", "- Candidate refinements remain diagnostic only."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(result: Result) -> None:
    if not result.harmful:
        raise RuntimeError(f"{result.instrument}: no harmful nodes found.")
    if any(row.new_keys < row.original_keys for row in result.splits.values()):
        raise RuntimeError(f"{result.instrument}: split key count shrank.")
    if any(not -1.0 <= row.net_value <= 1.0 for row in result.splits.values()):
        raise RuntimeError(f"{result.instrument}: NetValueScore escaped expected range.")


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
    contribution_aggregate = combined_result(loaded, aggregate_nodes)
    aggregate_stream = []
    aggregate_bars = []
    aggregate_paths = []
    for row in loaded:
        offset = len(aggregate_bars)
        aggregate_stream.extend(build_stream(row, aggregate_nodes, memory_thresholds, confidence_thresholds, entropy_thresholds, offset))
        aggregate_bars.extend(row.bars)
        aggregate_paths.extend(row.source_paths)
    aggregate_result = build_result("Aggregate", aggregate_paths, aggregate_bars, aggregate_stream, contribution_aggregate, aggregate_nodes)
    validate(aggregate_result)
    instrument_results = []
    for loaded_row, decay_row in zip(loaded, decay):
        local = local_rows(decay_row)
        score_rows(local)
        contribution = instrument_result(loaded_row, local, *memory_thresholds)
        stream = build_stream(loaded_row, local, memory_thresholds, confidence_thresholds, entropy_thresholds)
        result = build_result(loaded_row.instrument, loaded_row.source_paths, loaded_row.bars, stream, contribution, local)
        validate(result)
        instrument_results.append(result)
    out_root = Path(args.out_root)
    for result in instrument_results:
        write_per_instrument(result, out_root)
    write_aggregate(aggregate_result, instrument_results, out_root)
    print(f"Wrote {len(instrument_results)} per-instrument report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
