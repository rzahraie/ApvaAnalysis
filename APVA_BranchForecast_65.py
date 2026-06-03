#!/usr/bin/env python3
"""APVA Branch Forecast Study v0.1.

Forecast next StateAge node selection from fixed APVA structural information.
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

from APVA_HarmfulNodeHeterogeneity_64 import StreamRow, build_stream
from APVA_InformationContribution_63 import thresholds
from APVA_InformationDecay_57 import study as decay_study
from APVA_MemoryForecast_62 import bucket
from APVA_MinimalEngine_52 import directional_return, instrument_columns, load_results, safe_mean
from APVA_NodeImportance_58 import NodeRow, aggregate_rows, local_rows, score_rows
from APVA_ProcessGraph_54 import Node, node_for, node_text
from APVA_StructuralForecast_56 import Outcome, outcome
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

HORIZONS = (1, 2, 3, 5)
MODELS = (
    "Baseline_MostCommonNextNode",
    "Model1_CurrentNode",
    "Model2_CurrentPreviousNode",
    "Model3_CurrentMemoryClass",
    "Model4_CurrentConfidenceBucket",
    "Model5_CurrentEntropyBucket",
    "Model6_Combined",
)


@dataclass
class Forecast:
    actual: str
    distribution: dict[str, float]
    ranked: list[tuple[str, float]]
    probability: float
    confidence: float
    entropy: float
    brier: float
    top1: bool
    top2: bool
    top3: bool


@dataclass
class Metrics:
    count: int
    top1: float
    top2: float
    top3: float
    brier: float
    calibration: float
    entropy: float


@dataclass
class ModelResult:
    name: str
    forecasts: list[Forecast]
    unique_keys: int
    sparse_rate: float
    metrics: Metrics


@dataclass
class BranchRow:
    current: str
    next_node: str
    count: int
    probability: float


@dataclass
class NodeValue:
    node: str
    model1_accuracy: float
    model_accuracy: float
    gain: float


@dataclass
class EntropyRow:
    node: str
    entropy: float
    normalized_entropy: float
    dominant_probability: float
    concentration: float


@dataclass
class ContributionRow:
    node: str
    score: float
    count: int


@dataclass
class Result:
    instrument: str
    source_paths: list
    bars: list
    rows: list[StreamRow]
    segments: list[list[StreamRow]]
    branch_rows: dict[tuple[str, str], BranchRow]
    models: dict[str, ModelResult]
    entropy_rows: dict[str, EntropyRow]
    gains: dict[str, dict[str, NodeValue]]
    half_life: dict[int, Metrics]
    contributions: dict[str, ContributionRow]
    outcomes: dict[str, Outcome]


def mean(values: Iterable[float]) -> float:
    return safe_mean(list(values))


def distribution_entropy(distribution: dict[str, float]) -> float:
    return -sum(value * math.log(value) for value in distribution.values() if value > 0)


def distributions(keys: list[Hashable], targets: list[str], universe: tuple[str, ...]) -> dict[Hashable, dict[str, float]]:
    counts: dict[Hashable, Counter[str]] = defaultdict(Counter)
    for key, target in zip(keys, targets):
        counts[key][target] += 1
    return {
        key: {item: counter[item] / sum(counter.values()) for item in universe}
        for key, counter in counts.items()
    }


def calibration_rows(forecasts: Iterable[Forecast]) -> list[tuple[str, int, float, float, float]]:
    forecasts = list(forecasts)
    rows = []
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        selected = [
            row for row in forecasts
            if low <= row.probability <= high and (index == 9 or row.probability < high)
        ]
        probability = mean(row.probability for row in selected)
        observed = mean(row.top1 for row in selected)
        rows.append((f"{low:.1f}-{high:.1f}", len(selected), probability, observed, abs(probability - observed)))
    return rows


def expected_calibration_error(forecasts: Iterable[Forecast]) -> float:
    forecasts = list(forecasts)
    if not forecasts:
        return 0.0
    return sum(count * error for _, count, _, _, error in calibration_rows(forecasts)) / len(forecasts)


def build_forecast(actual: str, distribution: dict[str, float], universe: tuple[str, ...]) -> Forecast:
    ranked = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
    choices = [item for item, _ in ranked]
    probability = ranked[0][1] if ranked else 0.0
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    return Forecast(
        actual, distribution, ranked, probability, probability - second,
        distribution_entropy(distribution),
        sum((distribution.get(item, 0.0) - (1.0 if item == actual else 0.0)) ** 2 for item in universe),
        bool(choices) and actual == choices[0], actual in choices[:2], actual in choices[:3],
    )


def model_result(name: str, keys: list[Hashable], targets: list[str], universe: tuple[str, ...]) -> ModelResult:
    table = distributions(keys, targets, universe)
    counts = Counter(keys)
    forecasts = [build_forecast(target, table[key], universe) for key, target in zip(keys, targets)]
    return ModelResult(
        name, forecasts, len(counts), mean(count < 50 for count in counts.values()),
        Metrics(
            len(forecasts), mean(row.top1 for row in forecasts), mean(row.top2 for row in forecasts),
            mean(row.top3 for row in forecasts), mean(row.brier for row in forecasts),
            expected_calibration_error(forecasts), mean(row.entropy for row in forecasts),
        ),
    )


def model_keys(rows: list[StreamRow]) -> dict[str, list[Hashable]]:
    return {
        "Baseline_MostCommonNextNode": ["Baseline" for _ in rows],
        "Model1_CurrentNode": [row.node for row in rows],
        "Model2_CurrentPreviousNode": [(row.node, row.previous) for row in rows],
        "Model3_CurrentMemoryClass": [(row.node, row.memory_class) for row in rows],
        "Model4_CurrentConfidenceBucket": [(row.node, row.confidence_bucket) for row in rows],
        "Model5_CurrentEntropyBucket": [(row.node, row.entropy_bucket) for row in rows],
        "Model6_Combined": [
            (row.node, row.previous, row.memory_class, row.confidence_bucket, row.entropy_bucket)
            for row in rows
        ],
    }


def branch_rows(rows: list[StreamRow]) -> dict[tuple[str, str], BranchRow]:
    totals = Counter(node_text(row.node) for row in rows)
    counts = Counter((node_text(row.node), row.next_node) for row in rows)
    return {
        key: BranchRow(key[0], key[1], count, count / totals[key[0]] if totals[key[0]] else 0.0)
        for key, count in counts.items()
    }


def entropy_rows(rows: list[StreamRow]) -> dict[str, EntropyRow]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[node_text(row.node)][row.next_node] += 1
    output = {}
    for node, counter in grouped.items():
        total = sum(counter.values())
        probs = sorted((value / total for value in counter.values()), reverse=True)
        entropy = -sum(prob * math.log(prob) for prob in probs if prob > 0)
        maximum = math.log(len(probs)) if len(probs) > 1 else 0.0
        dominant = probs[0] if probs else 0.0
        second = probs[1] if len(probs) > 1 else 0.0
        output[node] = EntropyRow(node, entropy, entropy / maximum if maximum else 0.0, dominant, dominant - second)
    return output


def node_accuracy(rows: list[StreamRow], model: ModelResult) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row, forecast in zip(rows, model.forecasts):
        values[node_text(row.node)].append(float(forecast.top1))
    return {node: mean(samples) for node, samples in values.items()}


def gains(rows: list[StreamRow], models: dict[str, ModelResult]) -> dict[str, dict[str, NodeValue]]:
    base = node_accuracy(rows, models["Model1_CurrentNode"])
    output = {}
    for name in ("Model2_CurrentPreviousNode", "Model3_CurrentMemoryClass", "Model4_CurrentConfidenceBucket", "Model5_CurrentEntropyBucket"):
        current = node_accuracy(rows, models[name])
        output[name] = {
            node: NodeValue(node, base.get(node, 0.0), current.get(node, 0.0), current.get(node, 0.0) - base.get(node, 0.0))
            for node in sorted(base)
        }
    return output


def half_life(segments: list[list[StreamRow]]) -> dict[int, Metrics]:
    output = {}
    for horizon in HORIZONS:
        keys, targets = [], []
        for segment in segments:
            for index in range(len(segment) - horizon):
                keys.append(segment[index].node)
                targets.append(node_text(segment[index + horizon].node))
        universe = tuple(sorted(set(targets)))
        output[horizon] = model_result(f"Horizon_{horizon}", keys, targets, universe).metrics if universe else Metrics(0, 0, 0, 0, 0, 0, 0)
    return output


def contributions(rows: list[StreamRow], model: ModelResult) -> dict[str, ContributionRow]:
    baseline = model.metrics.top1
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[node_text(row.node)].append(index)
    output = {}
    for node, indexes in grouped.items():
        remove = set(indexes)
        kept = [forecast for index, forecast in enumerate(model.forecasts) if index not in remove]
        score = baseline - mean(forecast.top1 for forecast in kept)
        output[node] = ContributionRow(node, max(score, 0.0), len(indexes))
    return output


def outcome_rows(bars: list, rows: list[StreamRow]) -> dict[str, Outcome]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = directional_return(bars, row.index, 5)
        if value is not None:
            grouped[f"{node_text(row.node)}->{row.next_node}"].append(value)
    return {key: outcome(values) for key, values in grouped.items()}


def build_result(instrument: str, source_paths: list, bars: list, rows: list[StreamRow], segments: list[list[StreamRow]]) -> Result:
    targets = [row.next_node for row in rows]
    universe = tuple(sorted(set(targets)))
    keys = model_keys(rows)
    models = {name: model_result(name, keys[name], targets, universe) for name in MODELS}
    return Result(
        instrument, source_paths, bars, rows, segments, branch_rows(rows), models, entropy_rows(rows),
        gains(rows, models), half_life(segments), contributions(rows, models["Model1_CurrentNode"]),
        outcome_rows(bars, rows),
    )


def top_lines(rows: Iterable, key, line, limit: int = 10) -> list[str]:
    return [line(row) for row in sorted(rows, key=key)[:limit]]


def model_line(name: str, model: ModelResult) -> str:
    metric = model.metrics
    return (
        f"{name} | {metric.count} | {pct(metric.top1)} | {pct(metric.top2)} | {pct(metric.top3)} | "
        f"{fmt(metric.brier)} | {pct(metric.calibration)} | {fmt(metric.entropy)} | "
        f"{model.unique_keys} | {pct(model.sparse_rate)}"
    )


def model_section(result: Result, number: int, title: str, name: str) -> list[str]:
    return ["", f"{number}. {title}", "Model | Count | Top1Accuracy | Top2Accuracy | Top3Accuracy | BrierScore | CalibrationError | ForecastEntropy | UniqueKeyCount | SparseKeyRate", model_line(name, result.models[name])]


def write_per_instrument(result: Result, out_root: Path) -> None:
    path = out_root / result.instrument / f"BranchForecast_{result.instrument}.txt"
    ensure_dir(path.parent)
    lines = [
        "APVA Branch Forecast Study v0.1",
        "=" * 108,
        "Diagnostics",
        f"Instrument: {result.instrument}",
        "Input path(s): " + ", ".join(str(item) for item in result.source_paths),
        f"Total rows: {len(result.bars)}",
        f"Branch count: {len(result.branch_rows)}",
        "",
        "1. Branch Universe",
        "CurrentNode | NextNode | Count | Probability",
    ]
    for row in sorted(result.branch_rows.values(), key=lambda item: (-item.count, item.current, item.next_node))[:60]:
        lines.append(f"{row.current} | {row.next_node} | {row.count} | {pct(row.probability)}")
    lines += model_section(result, 2, "Baseline Forecast", "Baseline_MostCommonNextNode")
    lines += model_section(result, 3, "CurrentNode Model", "Model1_CurrentNode")
    lines += model_section(result, 4, "PreviousNode Model", "Model2_CurrentPreviousNode")
    lines += model_section(result, 5, "MemoryClass Model", "Model3_CurrentMemoryClass")
    lines += model_section(result, 6, "Confidence Model", "Model4_CurrentConfidenceBucket")
    lines += model_section(result, 7, "Entropy Model", "Model5_CurrentEntropyBucket")
    lines += model_section(result, 8, "Combined Model", "Model6_Combined")
    lines += ["", "9. Branch Entropy", "CurrentNode | BranchEntropy | NormalizedBranchEntropy"]
    lines += top_lines(result.entropy_rows.values(), lambda row: (row.entropy, row.node), lambda row: f"{row.node} | {fmt(row.entropy)} | {pct(row.normalized_entropy)}", 40)
    lines += ["", "10. Branch Stability", "CurrentNode | DominantBranchProbability | SecondGap"]
    lines += top_lines(result.entropy_rows.values(), lambda row: (-row.concentration, row.node), lambda row: f"{row.node} | {pct(row.dominant_probability)} | {pct(row.concentration)}", 40)
    for number, title, name in (
        (11, "PreviousNode Value", "Model2_CurrentPreviousNode"),
        (12, "Memory Value", "Model3_CurrentMemoryClass"),
        (13, "Confidence Value", "Model4_CurrentConfidenceBucket"),
        (14, "Entropy Value", "Model5_CurrentEntropyBucket"),
    ):
        lines += ["", f"{number}. {title}", "CurrentNode | Model1Accuracy | ModelAccuracy | AccuracyGain"]
        lines += top_lines(result.gains[name].values(), lambda row: (-row.gain, row.node), lambda row: f"{row.node} | {pct(row.model1_accuracy)} | {pct(row.model_accuracy)} | {pct(row.gain)}", 40)
    lines += ["", "15. Branch Half-Life", "Horizon | Top1Accuracy | Top2Accuracy | Top3Accuracy | BrierScore | ForecastEntropy"]
    for horizon in HORIZONS:
        metric = result.half_life[horizon]
        lines.append(f"t+{horizon} | {pct(metric.top1)} | {pct(metric.top2)} | {pct(metric.top3)} | {fmt(metric.brier)} | {fmt(metric.entropy)}")
    lines += ["", "16. Cross-Instrument Replication", "See aggregate report."]
    lines += ["", "17. Branch Information Contribution", "CurrentNode | ContributionScore | Count"]
    lines += top_lines(result.contributions.values(), lambda row: (-row.score, row.node), lambda row: f"{row.node} | {fmt(row.score)} | {row.count}", 40)
    lines += ["", "18. Outcome Diagnostics", "Branch | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew"]
    for branch, item in sorted(result.outcomes.items(), key=lambda pair: (-pair[1].count, pair[0]))[:80]:
        lines.append(f"{branch} | {item.count} | {fmt(item.mean_dr)} | {fmt(item.median_dr)} | {pct(item.continuation)} | {pct(item.failure)} | {pct(item.flat)} | {fmt(item.skew)}")
    lines += ["", "19. Recommendation"] + recommendation_lines(result, [result])
    append_audit(lines)
    lines += ["", "21. Mechanical Research Notes", "- Branch forecasting uses fixed StateAge nodes only.", "- PreviousNode, memory, confidence, and entropy are evaluated as descriptive keys.", "- Forward returns are diagnostic only and excluded from branch forecasting."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def replication_count(key, instrument_results: list[Result], source) -> int:
    return sum(key in source(result) for result in instrument_results)


def aggregate_tables(result: Result, instrument_results: list[Result], instruments: list[str]) -> list[str]:
    by_inst = {item.instrument: item for item in instrument_results}
    lines = ["", "Aggregate Model Comparison Table", "Model | " + " | ".join(f"Top1Accuracy_{item}" for item in instruments) + " | MeanTop1Accuracy | Top2Accuracy | Top3Accuracy | MeanBrierScore | MeanCalibrationError | MeanForecastEntropy | UniqueKeyCount | SparseKeyRate"]
    for model in MODELS:
        top1s = [by_inst[item].models[model].metrics.top1 for item in instruments]
        metrics = result.models[model].metrics
        lines.append(f"{model} | " + " | ".join(pct(value) for value in top1s) + f" | {pct(mean(top1s))} | {pct(metrics.top2)} | {pct(metrics.top3)} | {fmt(metrics.brier)} | {pct(metrics.calibration)} | {fmt(metrics.entropy)} | {result.models[model].unique_keys} | {pct(result.models[model].sparse_rate)}")
    lines += ["", "Aggregate Branch Table", "CurrentNode | NextNode | Count | " + " | ".join(f"Probability_{item}" for item in instruments) + " | ReplicationCount"]
    for key, row in sorted(result.branch_rows.items(), key=lambda pair: (-pair[1].count, pair[0]))[:120]:
        probs = [by_inst[item].branch_rows.get(key, BranchRow(key[0], key[1], 0, 0.0)).probability for item in instruments]
        lines.append(f"{key[0]} | {key[1]} | {row.count} | " + " | ".join(pct(value) for value in probs) + f" | {sum(value > 0 for value in probs)}")
    lines += ["", "Aggregate Entropy Table", "CurrentNode | BranchEntropy | NormalizedBranchEntropy | DominantBranchProbability | BranchConcentration"]
    for row in sorted(result.entropy_rows.values(), key=lambda item: (-item.entropy, item.node))[:80]:
        lines.append(f"{row.node} | {fmt(row.entropy)} | {pct(row.normalized_entropy)} | {pct(row.dominant_probability)} | {pct(row.concentration)}")
    for title, model in (
        ("Aggregate PreviousNode Gain Table", "Model2_CurrentPreviousNode"),
        ("Aggregate Memory Gain Table", "Model3_CurrentMemoryClass"),
        ("Aggregate Confidence Gain Table", "Model4_CurrentConfidenceBucket"),
        ("Aggregate Entropy Gain Table", "Model5_CurrentEntropyBucket"),
    ):
        lines += ["", title, "CurrentNode | Model1Accuracy | ModelAccuracy | AccuracyGain"]
        for row in sorted(result.gains[model].values(), key=lambda item: (-item.gain, item.node))[:80]:
            lines.append(f"{row.node} | {pct(row.model1_accuracy)} | {pct(row.model_accuracy)} | {pct(row.gain)}")
    lines += ["", "Aggregate Branch Half-Life Table", "Horizon | Top1Accuracy | Top2Accuracy | Top3Accuracy | BrierScore | ForecastEntropy"]
    for horizon in HORIZONS:
        metric = result.half_life[horizon]
        lines.append(f"t+{horizon} | {pct(metric.top1)} | {pct(metric.top2)} | {pct(metric.top3)} | {fmt(metric.brier)} | {fmt(metric.entropy)}")
    lines += ["", "Aggregate Branch Contribution Table", "CurrentNode | ContributionScore | ReplicationCount"]
    for node, row in sorted(result.contributions.items(), key=lambda pair: (-pair[1].score, pair[0]))[:80]:
        rep = sum(node in item.contributions and item.contributions[node].score > 0 for item in instrument_results)
        lines.append(f"{node} | {fmt(row.score)} | {rep}")
    lines += ["", "Aggregate Outcome Table", "Branch | " + " | ".join(f"Count_{item} | MeanDR_{item}" for item in instruments) + " | ValidInstrumentCount"]
    for branch in sorted(result.outcomes, key=lambda key: (-result.outcomes[key].count, key))[:120]:
        values = []
        valid = 0
        for instrument in instruments:
            item = by_inst[instrument].outcomes.get(branch)
            if item:
                valid += 1
                values.append(f"{item.count} | {fmt(item.mean_dr)}")
            else:
                values.append("0 | N/A")
        lines.append(f"{branch} | " + " | ".join(values) + f" | {valid}")
    return lines


def best_model(result: Result) -> str:
    return max(MODELS, key=lambda name: (result.models[name].metrics.top1, -result.models[name].metrics.brier))


def recommendation_lines(result: Result, instrument_results: list[Result]) -> list[str]:
    model = best_model(result)
    metric = result.models[model].metrics
    valid = sum(item.models[model].metrics.count > 0 for item in instrument_results)
    if metric.top1 >= 0.60 and metric.top3 >= 0.85 and metric.calibration <= 0.05 and valid >= 2:
        classification = "Strong Branch Forecast"
    elif metric.top1 >= 0.45 and metric.top3 >= 0.70 and metric.calibration <= 0.10 and valid >= 2:
        classification = "Moderate Branch Forecast"
    else:
        classification = "Weak Branch Forecast"
    prev_gain = result.models["Model2_CurrentPreviousNode"].metrics.top1 - result.models["Model1_CurrentNode"].metrics.top1
    mem_gain = result.models["Model3_CurrentMemoryClass"].metrics.top1 - result.models["Model1_CurrentNode"].metrics.top1
    conf_gain = result.models["Model4_CurrentConfidenceBucket"].metrics.top1 - result.models["Model1_CurrentNode"].metrics.top1
    ent_gain = result.models["Model5_CurrentEntropyBucket"].metrics.top1 - result.models["Model1_CurrentNode"].metrics.top1
    return [
        f"Classification: {classification}",
        f"BestModel: {model}",
        f"Accuracy: Top1={pct(metric.top1)}, Top3={pct(metric.top3)}",
        f"Calibration: {pct(metric.calibration)}",
        f"BranchPredictability: entropy={fmt(metric.entropy)}, brier={fmt(metric.brier)}",
        f"PreviousNodeValue: {pct(prev_gain)}",
        f"MemoryValue: {pct(mem_gain)}",
        f"ConfidenceValue: {pct(conf_gain)}",
        f"EntropyValue: {pct(ent_gain)}",
        f"ReplicationAssessment: ValidInstrumentCount={valid}",
    ]


def rankings(result: Result) -> list[str]:
    lines = ["", "AGGREGATE RANKINGS"]
    lines += ["", "1. Most predictable nodes"] + top_lines(result.entropy_rows.values(), lambda row: (-row.concentration, row.node), lambda row: f"{row.node} | {pct(row.concentration)}")
    lines += ["", "2. Least predictable nodes"] + top_lines(result.entropy_rows.values(), lambda row: (row.concentration, row.node), lambda row: f"{row.node} | {pct(row.concentration)}")
    lines += ["", "3. Highest entropy nodes"] + top_lines(result.entropy_rows.values(), lambda row: (-row.entropy, row.node), lambda row: f"{row.node} | {fmt(row.entropy)}")
    lines += ["", "4. Lowest entropy nodes"] + top_lines(result.entropy_rows.values(), lambda row: (row.entropy, row.node), lambda row: f"{row.node} | {fmt(row.entropy)}")
    for number, title, model in (
        (5, "Highest PreviousNode gains", "Model2_CurrentPreviousNode"),
        (6, "Highest Memory gains", "Model3_CurrentMemoryClass"),
        (7, "Highest Confidence gains", "Model4_CurrentConfidenceBucket"),
        (8, "Highest Entropy gains", "Model5_CurrentEntropyBucket"),
    ):
        lines += ["", f"{number}. {title}"] + top_lines(result.gains[model].values(), lambda row: (-row.gain, row.node), lambda row: f"{row.node} | {pct(row.gain)}")
    lines += ["", "9. Highest branch-information nodes"] + top_lines(result.contributions.values(), lambda row: (-row.score, row.node), lambda row: f"{row.node} | {fmt(row.score)}")
    lines += ["", "10. Recommended APVA branch forecast model", best_model(result)]
    return lines


def write_aggregate(result: Result, instrument_results: list[Result], out_root: Path) -> None:
    path = out_root / "BranchForecast" / "BranchForecast_All.txt"
    ensure_dir(path.parent)
    instruments = instrument_columns(instrument_results)
    lines = ["APVA Branch Forecast Study v0.1 - Aggregate", "=" * 108, "Instruments: " + ", ".join(instruments)]
    lines += aggregate_tables(result, instrument_results, instruments)
    lines += ["", "Aggregate Recommendation"] + recommendation_lines(result, instrument_results)
    lines += rankings(result)
    lines += ["", "RESEARCH NOTES", "- Can APVA forecast branch selection? See model comparison and recommendation.", "- Is branch uncertainty dominant? See branch entropy and stability tables.", "- PreviousNode tests second-order Markov value mechanically.", "- No context, arbitration, persistence, phase, fitting, optimization, or outcomes are used in branch forecasting."]
    append_audit(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_audit(lines: list[str]) -> None:
    lines += [
        "",
        "20. Low-DoF Audit",
        "Variables used:",
        "CurrentNode",
        "PreviousNode",
        "StructuralState",
        "AgeBucket",
        "NodeCategory",
        "MemoryStrength",
        "ForecastConfidence",
        "EntropyGrowth",
        "",
        "No Context",
        "No Arbitration",
        "No Persistence",
        "No Phase",
        "No Optimization",
        "No Fitting",
        "No Machine Learning",
        "No Forward Returns used in branch forecasting",
    ]


def validate(result: Result) -> None:
    if not result.rows:
        raise RuntimeError(f"{result.instrument}: no forecast rows.")
    for model in MODELS:
        if result.models[model].metrics.count != len(result.rows):
            raise RuntimeError(f"{result.instrument}: model count mismatch for {model}.")
        if result.models[model].metrics.top2 + 1e-12 < result.models[model].metrics.top1:
            raise RuntimeError(f"{result.instrument}: Top2 below Top1 for {model}.")
        if result.models[model].metrics.top3 + 1e-12 < result.models[model].metrics.top2:
            raise RuntimeError(f"{result.instrument}: Top3 below Top2 for {model}.")
    if any(row.probability < 0 or row.probability > 1 for row in result.branch_rows.values()):
        raise RuntimeError(f"{result.instrument}: branch probability outside [0, 1].")


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
        segment = build_stream(loaded_row, aggregate_nodes, memory_thresholds, confidence_thresholds, entropy_thresholds, offset)
        aggregate_stream.extend(segment)
        aggregate_segments.append(segment)
        aggregate_bars.extend(loaded_row.bars)
        aggregate_paths.extend(loaded_row.source_paths)

        local = local_rows(decay_row)
        score_rows(local)
        local_segment = build_stream(loaded_row, local, memory_thresholds, confidence_thresholds, entropy_thresholds)
        result = build_result(loaded_row.instrument, loaded_row.source_paths, loaded_row.bars, local_segment, [local_segment])
        validate(result)
        instrument_results.append(result)

    aggregate_result = build_result("Aggregate", aggregate_paths, aggregate_bars, aggregate_stream, aggregate_segments)
    validate(aggregate_result)
    out_root = Path(args.out_root)
    for result in instrument_results:
        write_per_instrument(result, out_root)
    write_aggregate(aggregate_result, instrument_results, out_root)
    print(f"Wrote {len(instrument_results)} per-instrument report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
