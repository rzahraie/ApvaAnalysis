#!/usr/bin/env python3
"""APVA Memory Forecast Study v0.1.

Forecast APVA memory dynamics from fixed structural and derived graph
properties. Forward price outcomes are diagnostics only.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Hashable, Iterable

from APVA_InformationDecay_57 import study as decay_study
from APVA_MemoryDynamics_61 import CLASSES, Result as DynamicsResult, build_result as build_dynamics
from APVA_MinimalEngine_52 import directional_return, instrument_columns, load_results, safe_mean
from APVA_NodeImportance_58 import aggregate_rows, local_rows, percentile, score_rows
from APVA_ProcessGraph_54 import Node, node_for, node_text
from APVA_StructuralForecast_56 import Outcome, outcome
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

TARGETS = ("MemoryIncrease", "MemoryDecrease", "MemoryStable")
HORIZONS = (1, 2, 3, 5)
MODELS = (
    "Model1_State",
    "Model2_StateAge",
    "Model3_StateAgeCategory",
    "Model4_StateAgeCategoryMemoryClass",
    "Model5_StateAgeCategoryMemoryClassConfidenceBucket",
    "Model6_StateAgeCategoryMemoryClassConfidenceBucketEntropyBucket",
)


@dataclass
class Forecast:
    index: int
    actual: str
    distribution: dict[str, float]
    ranked: list[tuple[str, float]]
    probability: float
    confidence: float
    entropy: float
    brier: float
    top1: bool
    top2: bool


@dataclass
class Metrics:
    count: int
    top1: float
    top2: float
    entropy: float
    calibration: float
    brier: float
    entropy_accuracy_correlation: float


@dataclass
class ModelResult:
    name: str
    forecasts: list[Forecast]
    unique_keys: int
    sparse_rate: float
    metrics: Metrics


@dataclass
class Result:
    instrument: str
    source_paths: list
    bars: list
    dynamics: DynamicsResult
    targets: list[str]
    absolute_delta: list[float]
    baseline: dict[str, float]
    node_distributions: dict[Node, dict[str, float]]
    state_distributions: dict[str, dict[str, float]]
    category_distributions: dict[str, dict[str, float]]
    class_distributions: dict[str, dict[str, float]]
    class_forecasts: list[Forecast]
    models: dict[str, ModelResult]
    half_life: dict[int, Metrics]
    creation: dict[Node, tuple[int, float]]
    destruction: dict[Node, tuple[int, float]]
    persistence: dict[Node, tuple[int, float]]
    outcomes: dict[str, Outcome]


def mean(values: Iterable[float]) -> float:
    return safe_mean(list(values))


def target(delta: float) -> str:
    if delta > 0:
        return "MemoryIncrease"
    if delta < 0:
        return "MemoryDecrease"
    return "MemoryStable"


def bucket(value: float, low: float, high: float) -> str:
    if value <= low:
        return "Low"
    if value >= high:
        return "High"
    return "Medium"


def distribution_entropy(distribution: dict[str, float]) -> float:
    return -sum(value * math.log(value) for value in distribution.values() if value > 0)


def correlation(left: Iterable[float], right: Iterable[float]) -> float:
    pairs = list(zip(left, right))
    if len(pairs) < 2:
        return 0.0
    xs, ys = zip(*pairs)
    xm, ym = mean(xs), mean(ys)
    denominator = math.sqrt(sum((x - xm) ** 2 for x in xs) * sum((y - ym) ** 2 for y in ys))
    return sum((x - xm) * (y - ym) for x, y in pairs) / denominator if denominator else 0.0


def distributions(keys: list[Hashable], targets: list[str], universe: tuple[str, ...]) -> dict[Hashable, dict[str, float]]:
    counts: dict[Hashable, Counter[str]] = defaultdict(Counter)
    for key, value in zip(keys, targets):
        counts[key][value] += 1
    return {
        key: {item: counter[item] / sum(counter.values()) for item in universe}
        for key, counter in counts.items()
    }


def build_forecast(index: int, actual: str, distribution: dict[str, float], universe: tuple[str, ...]) -> Forecast:
    ranked = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
    probability = ranked[0][1] if ranked else 0.0
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    choices = [item for item, _ in ranked]
    return Forecast(
        index, actual, distribution, ranked, probability, probability - second,
        distribution_entropy(distribution),
        sum((distribution.get(item, 0.0) - (1.0 if item == actual else 0.0)) ** 2 for item in universe),
        bool(choices) and actual == choices[0], actual in choices[:2],
    )


def forecast_rows(keys: list[Hashable], targets: list[str], universe: tuple[str, ...]) -> tuple[list[Forecast], dict[Hashable, dict[str, float]], Counter[Hashable]]:
    table = distributions(keys, targets, universe)
    counts = Counter(keys)
    return [build_forecast(index, actual, table[key], universe) for index, (key, actual) in enumerate(zip(keys, targets))], table, counts


def expected_calibration_error(rows: Iterable[Forecast]) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    return sum(count * error for _, count, _, _, error in calibration_rows(rows)) / len(rows)


def metrics(rows: Iterable[Forecast]) -> Metrics:
    rows = list(rows)
    return Metrics(
        len(rows), mean(row.top1 for row in rows), mean(row.top2 for row in rows),
        mean(row.entropy for row in rows), expected_calibration_error(rows),
        mean(row.brier for row in rows), correlation((row.entropy for row in rows), (float(row.top1) for row in rows)),
    )


def model_keys(dynamics: DynamicsResult, confidence_thresholds: tuple[float, float], entropy_thresholds: tuple[float, float]) -> dict[str, list[Hashable]]:
    confidence = [bucket(value, *confidence_thresholds) for value in dynamics.confidence[:-1]]
    entropy = [bucket(value, *entropy_thresholds) for value in dynamics.entropy[:-1]]
    state = [node[0] for node in dynamics.nodes[:-1]]
    age = [node[1] for node in dynamics.nodes[:-1]]
    category = dynamics.categories[:-1]
    memory_class = dynamics.memory_classes[:-1]
    return {
        "Model1_State": state,
        "Model2_StateAge": list(zip(state, age)),
        "Model3_StateAgeCategory": list(zip(state, age, category)),
        "Model4_StateAgeCategoryMemoryClass": list(zip(state, age, category, memory_class)),
        "Model5_StateAgeCategoryMemoryClassConfidenceBucket": list(zip(state, age, category, memory_class, confidence)),
        "Model6_StateAgeCategoryMemoryClassConfidenceBucketEntropyBucket": list(zip(state, age, category, memory_class, confidence, entropy)),
    }


def class_horizon_metrics(dynamics: DynamicsResult, horizon: int) -> Metrics:
    keys = dynamics.memory_classes[:-horizon]
    actual = dynamics.memory_classes[horizon:]
    rows, _, _ = forecast_rows(keys, actual, CLASSES)
    return metrics(rows)


def binary_node_rates(dynamics: DynamicsResult) -> tuple[dict[Node, tuple[int, float]], dict[Node, tuple[int, float]], dict[Node, tuple[int, float]]]:
    creation: dict[Node, list[bool]] = defaultdict(list)
    destruction: dict[Node, list[bool]] = defaultdict(list)
    persistence: dict[Node, list[bool]] = defaultdict(list)
    for index, node in enumerate(dynamics.nodes[:-1]):
        current, future = dynamics.memory_classes[index:index + 2]
        if current == "LowMemory":
            creation[node].append(future in ("MediumMemory", "HighMemory"))
        if current == "HighMemory":
            destruction[node].append(future in ("MediumMemory", "LowMemory"))
            persistence[node].append(future == "HighMemory")
    summarize = lambda rows: {node: (len(values), mean(values)) for node, values in rows.items()}
    return summarize(creation), summarize(destruction), summarize(persistence)


def build_result(loaded, dynamics: DynamicsResult, confidence_thresholds: tuple[float, float], entropy_thresholds: tuple[float, float]) -> Result:
    targets = [target(dynamics.memory[index + 1] - dynamics.memory[index]) for index in range(len(dynamics.memory) - 1)]
    absolute = [abs(dynamics.memory[index + 1] - dynamics.memory[index]) for index in range(len(dynamics.memory) - 1)]
    baseline = {item: targets.count(item) / len(targets) if targets else 0.0 for item in TARGETS}
    node = distributions(dynamics.nodes[:-1], targets, TARGETS)
    state = distributions([item[0] for item in dynamics.nodes[:-1]], targets, TARGETS)
    category = distributions(dynamics.categories[:-1], targets, TARGETS)
    class_forecasts, class_table, _ = forecast_rows(dynamics.memory_classes[:-1], dynamics.memory_classes[1:], CLASSES)
    models = {}
    for name, keys in model_keys(dynamics, confidence_thresholds, entropy_thresholds).items():
        forecasts, _, counts = forecast_rows(keys, targets, TARGETS)
        models[name] = ModelResult(name, forecasts, len(counts), mean(count < 50 for count in counts.values()), metrics(forecasts))
    creation, destruction, persistence = binary_node_rates(dynamics)
    values: dict[str, list[float]] = defaultdict(list)
    for index, forecast_class in enumerate(targets):
        value = directional_return(loaded.bars, index, 5)
        if value is not None:
            values[forecast_class].append(value)
    return Result(
        loaded.instrument, loaded.source_paths, loaded.bars, dynamics, targets, absolute, baseline,
        node, state, category, class_table, class_forecasts, models,
        {horizon: class_horizon_metrics(dynamics, horizon) for horizon in HORIZONS},
        creation, destruction, persistence, {item: outcome(samples) for item, samples in values.items()},
    )


def distribution_text(row: dict[str, float], universe: tuple[str, ...]) -> str:
    return " | ".join(pct(row.get(item, 0.0)) for item in universe)


def calibration_rows(forecasts: Iterable[Forecast]) -> list[tuple[str, int, float, float, float]]:
    forecasts = list(forecasts)
    rows = []
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        selected = [row for row in forecasts if low <= row.probability <= high if index == 9 or row.probability < high]
        probability, observed = mean(row.probability for row in selected), mean(row.top1 for row in selected)
        rows.append((f"{low:.1f}-{high:.1f}", len(selected), probability, observed, abs(probability - observed)))
    return rows


def best_model(results: list[Result]) -> tuple[str, Metrics, float]:
    rows = []
    for name in MODELS:
        combined = [forecast for result in results for forecast in result.models[name].forecasts]
        sparse = mean(result.models[name].sparse_rate for result in results)
        rows.append((name, metrics(combined), sparse))
    return max(rows, key=lambda item: (item[1].top1, item[1].top2, -item[1].calibration, -item[2], -MODELS.index(item[0])))


def recommendation(results: list[Result]) -> tuple[str, list[str]]:
    name, row, sparse = best_model(results)
    valid = len(results)
    if row.top1 >= 0.60 and row.top2 >= 0.80 and row.calibration <= 0.05 and valid >= 2:
        label = "Strong Memory Forecast"
    elif row.top1 >= 0.45 and row.top2 >= 0.70 and row.calibration <= 0.10 and valid >= 2:
        label = "Moderate Memory Forecast"
    else:
        label = "Weak Memory Forecast"
    return label, [
        f"BestModel: {name}",
        f"Reason: Top1Accuracy={pct(row.top1)}; Top2Accuracy={pct(row.top2)}; MeanCalibrationError={pct(row.calibration)}; ValidInstrumentCount={valid}.",
        f"Accuracy: Top1={pct(row.top1)}; Top2={pct(row.top2)}.",
        f"Calibration: {pct(row.calibration)}.",
        f"Brier: {fmt(row.brier)}.",
        f"SparsePenalty: SparseKeyRate={pct(sparse)}.",
        f"CrossInstrumentReplication: {valid} instruments evaluated.",
    ]


def append_audit(lines: list[str], heading: str) -> None:
    lines += ["", heading, "Variables used:", "StructuralState", "AgeBucket", "NodeCategory", "MemoryStrength", "HalfLife", "ForecastConfidence", "EntropyGrowth", "", "No Context", "No Arbitration", "No Persistence", "No Phase", "No Optimization", "No Fitting", "No Machine Learning", "No Forward Returns used in memory forecasting"]


def write_per_instrument(result: Result, out_root: Path) -> None:
    path = out_root / result.instrument / f"MemoryForecast_{result.instrument}.txt"
    ensure_dir(path.parent)
    label, reasons = recommendation([result])
    lines = ["APVA Memory Forecast Study v0.1", "=" * 108, "Diagnostics", f"Instrument: {result.instrument}", "Input path(s): " + ", ".join(str(item) for item in result.source_paths), f"Total rows: {len(result.bars)}", f"Node count: {len(set(result.dynamics.nodes))}"]
    lines += ["", "1. Memory Stream Reconstruction", "Bar | StateAgeNode | NodeCategory | MemoryStrength | HalfLife | ForecastConfidence | EntropyGrowth | AttentionCategory"]
    lines += [f"{index} | {node_text(node)} | {category} | {fmt(memory)} | {fmt(life, 2)} | {fmt(confidence)} | {fmt(entropy)} | {category}" for index, (node, category, memory, life, confidence, entropy) in enumerate(zip(result.dynamics.nodes, result.dynamics.categories, result.dynamics.memory, result.dynamics.half_life, result.dynamics.confidence, result.dynamics.entropy))]
    lines += ["", "2. Memory Forecast Target", f"Forecast rows: {len(result.targets)}", f"MeanAbsoluteMemoryDelta: {fmt(mean(result.absolute_delta))}"]
    lines += ["", "3. Baseline Distribution"] + [f"P({item}): {pct(result.baseline[item])}" for item in TARGETS]
    for number, title, table, renderer in (
        (4, "Node-Based Memory Forecast", result.node_distributions, node_text),
        (5, "State-Level Memory Forecast", result.state_distributions, str),
        (6, "Category-Level Memory Forecast", result.category_distributions, str),
    ):
        lines += ["", f"{number}. {title}", "Key | PIncrease | PDecrease | PStable | ForecastClass | ForecastProbability | SecondClass | SecondProbability | ForecastConfidence"]
        for key, row in sorted(table.items(), key=lambda item: str(item[0])):
            ranked = sorted(row.items(), key=lambda item: (-item[1], item[0]))
            lines.append(f"{renderer(key)} | {distribution_text(row, TARGETS)} | {ranked[0][0]} | {pct(ranked[0][1])} | {ranked[1][0]} | {pct(ranked[1][1])} | {pct(ranked[0][1] - ranked[1][1])}")
    lines += ["", "7. Memory-Class Forecast", "CurrentMemoryClass | PNextLow | PNextMedium | PNextHigh"]
    lines += [f"{item} | {distribution_text(result.class_distributions.get(item, {}), CLASSES)}" for item in CLASSES]
    class_metrics = metrics(result.class_forecasts)
    lines += [f"Top1Accuracy: {pct(class_metrics.top1)}", f"Top2Accuracy: {pct(class_metrics.top2)}", f"CalibrationError: {pct(class_metrics.calibration)}", f"TransitionEntropy: {fmt(class_metrics.entropy)}"]
    lines += ["", "8. Combined Forecast Models", "Model | UniqueKeyCount | SparseKeyRate | Top1 | Top2 | Entropy | CalibrationError | Brier"]
    lines += [f"{name} | {row.unique_keys} | {pct(row.sparse_rate)} | {pct(row.metrics.top1)} | {pct(row.metrics.top2)} | {fmt(row.metrics.entropy)} | {pct(row.metrics.calibration)} | {fmt(row.metrics.brier)}" for name, row in result.models.items()]
    lines += ["", "9. Calibration Test", "ProbabilityBin | Count | ForecastProbability | ObservedFrequency | CalibrationError"]
    lines += [f"{label} | {count} | {pct(probability)} | {pct(observed)} | {pct(error)}" for label, count, probability, observed, error in calibration_rows(result.models[best_model([result])[0]].forecasts)]
    lines += ["", "10. Brier Score"] + [f"{name}: {fmt(row.metrics.brier)}" for name, row in result.models.items()]
    lines += ["", "11. Forecast Entropy"] + [f"{name}: Entropy={fmt(row.metrics.entropy)} | CorrelationEntropyAccuracy={fmt(row.metrics.entropy_accuracy_correlation)}" for name, row in result.models.items()]
    lines += ["", "12. Memory Forecast Half-Life", "Horizon | Top1 | Top2 | Brier | Entropy"] + [f"t+{horizon} | {pct(row.top1)} | {pct(row.top2)} | {fmt(row.brier)} | {fmt(row.entropy)}" for horizon, row in result.half_life.items()]
    for number, title, rows in ((13, "Memory Creation Forecast", result.creation), (14, "Memory Destruction Forecast", result.destruction), (15, "Memory Persistence Forecast", result.persistence)):
        lines += ["", f"{number}. {title}", "Node | Count | Probability"] + [f"{node_text(node)} | {count} | {pct(probability)}" for node, (count, probability) in sorted(rows.items(), key=lambda item: (-item[1][1], item[0]))]
    lines += ["", "16. Cross-Instrument Replication", "See aggregate report."]
    lines += ["", "17. Outcome Diagnostics", "Diagnostic only. Outcomes are not used in memory forecasting.", "ForecastClass | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew"]
    lines += [f"{item} | {row.count} | {fmt(row.mean_dr)} | {fmt(row.median_dr)} | {pct(row.continuation)} | {pct(row.failure)} | {pct(row.flat)} | {fmt(row.skew)}" for item, row in sorted(result.outcomes.items())]
    lines += ["", "18. Memory Forecast Recommendation", f"Classification: {label}"] + reasons
    append_audit(lines, "19. Low-DoF Audit")
    lines += ["", "20. Mechanical Research Notes", "- Forecasts use historical conditional distributions only.", "- Aggregate percentile buckets are fixed before instrument reports.", "- Sparse keys are measured, not removed.", "- Forward outcomes are diagnostic only."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aggregate(results: list[Result], out_root: Path) -> None:
    path = out_root / "MemoryForecast" / "MemoryForecast_All.txt"
    ensure_dir(path.parent)
    instruments = instrument_columns(results)
    by_name = {row.instrument: row for row in results}
    label, reasons = recommendation(results)
    best = best_model(results)[0]
    lines = ["APVA Memory Forecast Study v0.1 - Aggregate", "=" * 108, "Instruments: " + ", ".join(instruments)]
    lines += ["", "Aggregate Baseline Table", "Instrument | PIncrease | PDecrease | PStable"] + [f"{row.instrument} | {distribution_text(row.baseline, TARGETS)}" for row in results]
    lines += aggregate_node_table(results, instruments)
    lines += ["", "Aggregate Model Comparison Table", "Model | UniqueKeyCount | SparseKeyRate | " + " | ".join(f"Top1_{item}" for item in instruments) + " | MeanTop1 | " + " | ".join(f"Top2_{item}" for item in instruments) + " | MeanTop2 | MeanCalibrationError | MeanBrierScore | MeanForecastEntropy"]
    for name in MODELS:
        rows = [by_name[item].models[name] for item in instruments]
        lines.append(f"{name} | {sum(row.unique_keys for row in rows)} | {pct(mean(row.sparse_rate for row in rows))} | " + " | ".join(pct(row.metrics.top1) for row in rows) + f" | {pct(mean(row.metrics.top1 for row in rows))} | " + " | ".join(pct(row.metrics.top2) for row in rows) + f" | {pct(mean(row.metrics.top2 for row in rows))} | {pct(mean(row.metrics.calibration for row in rows))} | {fmt(mean(row.metrics.brier for row in rows))} | {fmt(mean(row.metrics.entropy for row in rows))}")
    combined = [row for result in results for row in result.models[best].forecasts]
    lines += ["", "Aggregate Calibration Table", "ProbabilityBin | Count | ForecastProbability | ObservedFrequency | CalibrationError"] + [f"{item} | {count} | {pct(probability)} | {pct(observed)} | {pct(error)}" for item, count, probability, observed, error in calibration_rows(combined)]
    lines += ["", "Aggregate Brier Table", "Model | " + " | ".join(f"Brier_{item}" for item in instruments) + " | MeanBrier"] + [f"{name} | " + " | ".join(fmt(by_name[item].models[name].metrics.brier) for item in instruments) + f" | {fmt(mean(by_name[item].models[name].metrics.brier for item in instruments))}" for name in MODELS]
    lines += entropy_accuracy_lines(results, instruments, best)
    lines += ["", "Aggregate Forecast Half-Life Table", "Horizon | Top1Accuracy | Top2Accuracy | BrierScore | ForecastEntropy"]
    for horizon in HORIZONS:
        rows = [result.half_life[horizon] for result in results]
        lines.append(f"t+{horizon} | {pct(mean(row.top1 for row in rows))} | {pct(mean(row.top2 for row in rows))} | {fmt(mean(row.brier for row in rows))} | {fmt(mean(row.entropy for row in rows))}")
    lines += aggregate_binary_table(results, instruments, "creation", "Aggregate Creation Forecast Table", "PCreate")
    lines += aggregate_binary_table(results, instruments, "destruction", "Aggregate Destruction Forecast Table", "PDestroy")
    lines += aggregate_binary_table(results, instruments, "persistence", "Aggregate Persistence Forecast Table", "PPersist")
    lines += ["", "Aggregate Outcome Table", "ForecastClass | " + " | ".join(f"Count_{item} | MeanDR_{item}" for item in instruments) + " | ValidInstrumentCount"]
    for forecast_class in TARGETS:
        values = [by_name[item].outcomes.get(forecast_class) for item in instruments]
        lines.append(f"{forecast_class} | " + " | ".join(value for row in values for value in ((str(row.count), fmt(row.mean_dr)) if row else ("0", "N/A"))) + f" | {sum(row is not None for row in values)}")
    lines += ["", "Aggregate Recommendation", f"Classification: {label}"] + reasons
    lines += rankings(results)
    append_audit(lines, "Low-DoF Audit")
    lines += ["", "Research Notes", "- Can APVA forecast memory dynamics? See fixed lookup model accuracy.", "- Is Memory forecastable better than StructuralState? Compare Model1 and Model4.", "- Does ForecastConfidence improve forecasting? Compare Model4 and Model5.", "- Does EntropyGrowth improve forecasting? Compare Model5 and Model6.", "- Fragmentation is reported through SparseKeyRate.", "- State+Memory+Flow is evaluated without adding APVA states."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_node_table(results: list[Result], instruments: list[str]) -> list[str]:
    nodes = sorted({node for result in results for node in result.node_distributions})
    by_name = {row.instrument: row for row in results}
    lines = ["", "Aggregate Node Forecast Table", "Node | " + " | ".join(f"PIncrease_{item} | PDecrease_{item} | PStable_{item}" for item in instruments) + " | ForecastClass | ForecastProbability | ForecastConfidence | ReplicationCount"]
    for node in nodes:
        values = [by_name[item].node_distributions.get(node) for item in instruments]
        combined = {target_name: mean(row[target_name] for row in values if row) for target_name in TARGETS}
        ranked = sorted(combined.items(), key=lambda item: (-item[1], item[0]))
        lines.append(f"{node_text(node)} | " + " | ".join(distribution_text(row or {}, TARGETS) for row in values) + f" | {ranked[0][0]} | {pct(ranked[0][1])} | {pct(ranked[0][1] - ranked[1][1])} | {sum(row is not None for row in values)}")
    return lines


def entropy_accuracy_lines(results: list[Result], instruments: list[str], model: str) -> list[str]:
    by_name = {row.instrument: row for row in results}
    combined = [forecast for row in results for forecast in row.models[model].forecasts]
    low, high = percentile((row.entropy for row in combined), 1 / 3), percentile((row.entropy for row in combined), 2 / 3)
    labels = ("Low", "Medium", "High")
    overall = correlation((row.entropy for row in combined), (float(row.top1) for row in combined))
    lines = ["", "Aggregate Entropy Accuracy Table", "EntropyBin | " + " | ".join(f"Accuracy_{item}" for item in instruments) + " | MeanAccuracy | CorrelationEntropyAccuracy"]
    for label in labels:
        values = []
        for instrument in instruments:
            rows = [row for row in by_name[instrument].models[model].forecasts if bucket(row.entropy, low, high) == label]
            values.append(mean(row.top1 for row in rows))
        lines.append(f"{label} | " + " | ".join(pct(value) for value in values) + f" | {pct(mean(values))} | {fmt(overall)}")
    return lines


def aggregate_binary_table(results: list[Result], instruments: list[str], field: str, title: str, probability_name: str) -> list[str]:
    nodes = sorted({node for result in results for node in getattr(result, field)})
    by_name = {row.instrument: row for row in results}
    lines = ["", title, "Node | " + " | ".join(f"Count_{item} | {probability_name}_{item}" for item in instruments) + " | Count | MeanProbability | ReplicationCount"]
    for node in nodes:
        values = [by_name[item].__getattribute__(field).get(node) for item in instruments]
        valid = [row for row in values if row]
        lines.append(f"{node_text(node)} | " + " | ".join(f"{row[0]} | {pct(row[1])}" if row else "0 | N/A" for row in values) + f" | {sum(row[0] for row in valid)} | {pct(mean(row[1] for row in valid))} | {len(valid)}")
    return lines


def rankings(results: list[Result]) -> list[str]:
    name, _, _ = best_model(results)
    model_rows = [(item, mean(row.models[item].metrics.top1 for row in results)) for item in MODELS]
    combined = lambda field: [(node, sum(count for count, _ in values), mean(probability for _, probability in values)) for node in sorted({node for row in results for node in getattr(row, field)}) for values in [[getattr(row, field)[node] for row in results if node in getattr(row, field)]]]
    creation, destruction, persistence = combined("creation"), combined("destruction"), combined("persistence")
    lines = ["", "Aggregate Rankings", "", "1. Best memory forecast models"] + [f"{item} | {pct(value)}" for item, value in sorted(model_rows, key=lambda row: (-row[1], row[0]))]
    lines += ["", "2. Worst memory forecast models"] + [f"{item} | {pct(value)}" for item, value in sorted(model_rows, key=lambda row: (row[1], row[0]))]
    for number, title, rows in ((3, "Strongest memory-increase nodes", creation), (4, "Strongest memory-decrease nodes", destruction), (5, "Strongest memory-persistence nodes", persistence)):
        lines += ["", f"{number}. {title}"] + [f"{node_text(node)} | Count={count} | Probability={pct(probability)}" for node, count, probability in sorted(rows, key=lambda row: (-row[2], -row[1], row[0]))[:10]]
    lines += ["", "6. Best calibrated models"] + [f"{item} | {pct(mean(row.models[item].metrics.calibration for row in results))}" for item in sorted(MODELS, key=lambda item: mean(row.models[item].metrics.calibration for row in results))]
    lines += ["", "7. Lowest Brier models"] + [f"{item} | {fmt(mean(row.models[item].metrics.brier for row in results))}" for item in sorted(MODELS, key=lambda item: mean(row.models[item].metrics.brier for row in results))]
    lines += ["", "8. Longest memory-forecast horizon"] + [f"t+{horizon} | {pct(mean(row.half_life[horizon].top1 for row in results))}" for horizon in HORIZONS]
    lines += ["", "9. Most replicated memory forecast patterns", "See Aggregate Node Forecast Table replication counts."]
    lines += ["", "10. Recommended APVA memory forecast model", name]
    return lines


def validate_invariants(results: list[Result]) -> None:
    for result in results:
        if len(result.targets) != max(0, len(result.bars) - 1):
            raise RuntimeError(f"{result.instrument}: target stream length mismatch.")
        for name, model in result.models.items():
            if model.metrics.count != len(result.targets):
                raise RuntimeError(f"{result.instrument} {name}: forecast count mismatch.")
            if not 0.0 <= model.metrics.top1 <= model.metrics.top2 <= 1.0:
                raise RuntimeError(f"{result.instrument} {name}: accuracy escaped bounds.")
            if any(abs(sum(row.distribution.values()) - 1.0) > 1e-12 for row in model.forecasts):
                raise RuntimeError(f"{result.instrument} {name}: probability mass mismatch.")


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
    aggregate = aggregate_rows(decay)
    score_rows(aggregate)
    memory_values = [aggregate[node_for(bar)].memory_strength for row in loaded for bar in row.bars]
    confidence_values = [aggregate[node_for(bar)].confidence for row in loaded for bar in row.bars]
    entropy_values = [aggregate[node_for(bar)].entropy_growth for row in loaded for bar in row.bars]
    memory_thresholds = percentile(memory_values, 0.25), percentile(memory_values, 0.75)
    confidence_thresholds = percentile(confidence_values, 1 / 3), percentile(confidence_values, 2 / 3)
    entropy_thresholds = percentile(entropy_values, 1 / 3), percentile(entropy_values, 2 / 3)
    aggregate_dynamics = [build_dynamics(row, aggregate, *memory_thresholds) for row in loaded]
    results = [build_result(row, dynamics, confidence_thresholds, entropy_thresholds) for row, dynamics in zip(loaded, aggregate_dynamics)]
    validate_invariants(results)
    out_root = Path(args.out_root)
    for loaded_row, decay_row in zip(loaded, decay):
        local = local_rows(decay_row)
        score_rows(local)
        dynamics = build_dynamics(loaded_row, local, *memory_thresholds)
        write_per_instrument(build_result(loaded_row, dynamics, confidence_thresholds, entropy_thresholds), out_root)
    write_aggregate(results, out_root)
    print(f"Wrote {len(results)} per-instrument report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
