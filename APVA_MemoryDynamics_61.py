#!/usr/bin/env python3
"""APVA Memory Dynamics Study v0.1.

Reconstruct how fixed Study 57 graph memory is created, maintained, and
destroyed. Forward outcomes are diagnostics only.
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
from APVA_MinimalEngine_52 import directional_return, instrument_columns, load_results, safe_mean
from APVA_NodeImportance_58 import NodeRow, aggregate_rows, local_rows, percentile, score_rows
from APVA_ProcessGraph_54 import Node, node_for, node_text
from APVA_StructuralForecast_56 import Outcome, outcome
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

CLASSES = ("LowMemory", "MediumMemory", "HighMemory")
RECOVERY_HORIZONS = (0, 1, 2, 3, 5)
AGE_VALUE = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0, "6-10": 8.0, "11-20": 15.5, "21+": 21.0}


@dataclass
class Event:
    previous_node: Node
    current_node: Node
    previous_category: str
    current_category: str


@dataclass
class Run:
    memory_class: str
    start: int
    end: int
    values: list[float]


@dataclass
class CoreRun:
    length: int
    start_memory: float
    end_memory: float
    delta: float
    maximum: float
    average: float


@dataclass
class Excursion:
    start_index: int
    before: float
    start: float
    end: float
    after_return: float
    return_index: int


@dataclass
class StateRow:
    state: str
    count: int
    mean_memory: float
    high_rate: float
    low_rate: float
    creation_rate: float
    destruction_rate: float
    persistence_rate: float


@dataclass
class Result:
    instrument: str
    source_paths: list
    bars: list
    nodes: list[Node]
    categories: list[str]
    memory: list[float]
    half_life: list[float]
    entropy: list[float]
    confidence: list[float]
    memory_classes: list[str]
    deltas: list[float]
    creations: list[Event]
    destructions: list[Event]
    runs: list[Run]
    core_runs: list[CoreRun]
    excursions: list[Excursion]
    recovery: dict[int, list[float]]
    flow: dict[tuple[str, str], int]
    states: dict[str, StateRow]
    outcomes: dict[str, Outcome]
    age_memory: dict[str, float]
    memory_confidence: float
    memory_entropy: float
    core_length_delta: float
    core_length_end: float


def mean(values: Iterable[float]) -> float:
    return safe_mean(list(values))


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def correlation(left: Iterable[float], right: Iterable[float]) -> float:
    pairs = list(zip(left, right))
    if len(pairs) < 2:
        return 0.0
    xs, ys = zip(*pairs)
    xm, ym = mean(xs), mean(ys)
    denominator = math.sqrt(sum((x - xm) ** 2 for x in xs) * sum((y - ym) ** 2 for y in ys))
    return sum((x - xm) * (y - ym) for x, y in pairs) / denominator if denominator else 0.0


def classify(value: float, low: float, high: float) -> str:
    if value <= low:
        return "LowMemory"
    if value >= high:
        return "HighMemory"
    return "MediumMemory"


def contiguous_runs(values: list[str]) -> list[Run]:
    if not values:
        return []
    rows, start = [], 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            rows.append(Run(values[start], start, index - 1, []))
            start = index
    return rows


def event_rows(nodes: list[Node], categories: list[str], classes: list[str]) -> tuple[list[Event], list[Event]]:
    creations, destructions = [], []
    for index in range(1, len(classes)):
        event = Event(nodes[index - 1], nodes[index], categories[index - 1], categories[index])
        if classes[index - 1] == "LowMemory" and classes[index] in ("MediumMemory", "HighMemory"):
            creations.append(event)
        if classes[index - 1] == "HighMemory" and classes[index] in ("MediumMemory", "LowMemory"):
            destructions.append(event)
    return creations, destructions


def core_run_rows(categories: list[str], memory: list[float]) -> list[CoreRun]:
    rows, index = [], 0
    while index < len(categories):
        if categories[index] != "Core Node":
            index += 1
            continue
        end = index
        while end + 1 < len(categories) and categories[end + 1] == "Core Node":
            end += 1
        values = memory[index:end + 1]
        rows.append(CoreRun(len(values), values[0], values[-1], values[-1] - values[0], max(values), mean(values)))
        index = end + 1
    return rows


def excursion_rows(categories: list[str], memory: list[float]) -> list[Excursion]:
    rows = []
    for index in range(1, len(categories)):
        if categories[index - 1] != "Core Node" or categories[index] == "Core Node":
            continue
        end = index
        while end + 1 < len(categories) and categories[end + 1] != "Core Node":
            end += 1
        return_index = end + 1 if end + 1 < len(categories) else end
        rows.append(Excursion(index, memory[index - 1], memory[index], memory[end], memory[return_index], return_index))
    return rows


def state_rows(nodes: list[Node], classes: list[str], creations: list[Event], destructions: list[Event]) -> dict[str, StateRow]:
    states = sorted({node[0] for node in nodes})
    rows = {}
    for state in states:
        indexes = [index for index, node in enumerate(nodes) if node[0] == state]
        previous_high = sum(classes[index - 1] == "HighMemory" for index in indexes if index)
        persistent = sum(classes[index - 1] == classes[index] == "HighMemory" for index in indexes if index)
        rows[state] = StateRow(
            state, len(indexes), 0.0,
            mean(classes[index] == "HighMemory" for index in indexes),
            mean(classes[index] == "LowMemory" for index in indexes),
            sum(event.current_node[0] == state for event in creations) / len(indexes) if indexes else 0.0,
            sum(event.current_node[0] == state for event in destructions) / len(indexes) if indexes else 0.0,
            persistent / previous_high if previous_high else 0.0,
        )
    return rows


def build_result(loaded, rows: dict[Node, NodeRow], low: float, high: float) -> Result:
    bars = loaded.bars
    nodes = [node_for(bar) for bar in bars]
    categories = [rows[node].category for node in nodes]
    memory = [rows[node].memory_strength for node in nodes]
    half_life = [rows[node].half_life for node in nodes]
    entropy = [rows[node].entropy_growth for node in nodes]
    confidence = [rows[node].confidence for node in nodes]
    classes = [classify(value, low, high) for value in memory]
    deltas = [memory[index] - memory[index - 1] for index in range(1, len(memory))]
    creations, destructions = event_rows(nodes, categories, classes)
    runs = contiguous_runs(classes)
    for run in runs:
        run.values = memory[run.start:run.end + 1]
    core_runs = core_run_rows(categories, memory)
    excursions = excursion_rows(categories, memory)
    recovery = {horizon: [memory[row.return_index + horizon] for row in excursions if row.return_index + horizon < len(memory)] for horizon in RECOVERY_HORIZONS}
    flow = Counter(zip(classes[:-1], classes[1:]))
    states = state_rows(nodes, classes, creations, destructions)
    for state, row in states.items():
        row.mean_memory = mean(value for node, value in zip(nodes, memory) if node[0] == state)
    age_memory = {state: correlation((AGE_VALUE.get(node[1], 0.0) for node in nodes if node[0] == state), (value for node, value in zip(nodes, memory) if node[0] == state)) for state in states}
    values: dict[str, list[float]] = defaultdict(list)
    for index, memory_class in enumerate(classes):
        value = directional_return(bars, index, 5)
        if value is not None:
            values[memory_class].append(value)
    return Result(
        loaded.instrument, loaded.source_paths, bars, nodes, categories, memory, half_life, entropy, confidence,
        classes, deltas, creations, destructions, runs, core_runs, excursions, recovery, dict(flow), states,
        {key: outcome(samples) for key, samples in values.items()}, age_memory,
        correlation(memory, confidence), correlation(memory, entropy),
        correlation((row.length for row in core_runs), (row.delta for row in core_runs)),
        correlation((row.length for row in core_runs), (row.end_memory for row in core_runs)),
    )


def transition_text(event: Event) -> str:
    return f"{node_text(event.previous_node)} -> {node_text(event.current_node)}"


def event_counts(events: list[Event]) -> Counter[str]:
    return Counter(transition_text(event) for event in events)


def persistence(result: Result, memory_class: str) -> tuple[float, float, float, int]:
    eligible = sum(result.memory_classes[index - 1] == memory_class for index in range(1, len(result.memory_classes)))
    stays = sum(result.memory_classes[index - 1] == result.memory_classes[index] == memory_class for index in range(1, len(result.memory_classes)))
    lengths = [run.end - run.start + 1 for run in result.runs if run.memory_class == memory_class]
    return stays / eligible if eligible else 0.0, mean(lengths), median(lengths), max(lengths, default=0)


def flow_probability(result: Result, source: str, destination: str) -> float:
    denominator = sum(count for (left, _), count in result.flow.items() if left == source)
    return result.flow.get((source, destination), 0) / denominator if denominator else 0.0


def recovery_probability(result: Result, horizon: int) -> float:
    return mean(value > row.end for row, value in zip(result.excursions, result.recovery[horizon]))


def recovery_half_life(result: Result) -> str:
    target = mean(row.before - row.end for row in result.excursions) / 2.0
    for horizon in RECOVERY_HORIZONS:
        if mean(value - row.end for row, value in zip(result.excursions, result.recovery[horizon])) >= target:
            return str(horizon)
    return "5+"


def grouped_correlations(result: Result, values: list[float]) -> list[str]:
    lines = ["Group | MemoryStrengthCorrelation"]
    for state in sorted({node[0] for node in result.nodes}):
        indexes = [index for index, node in enumerate(result.nodes) if node[0] == state]
        lines.append(f"State={state} | {fmt(correlation((result.memory[index] for index in indexes), (values[index] for index in indexes)))}")
    for category in sorted(set(result.categories)):
        indexes = [index for index, value in enumerate(result.categories) if value == category]
        lines.append(f"NodeCategory={category} | {fmt(correlation((result.memory[index] for index in indexes), (values[index] for index in indexes)))}")
    return lines


def recommendation(results: list[Result]) -> tuple[str, list[str]]:
    age = mean(value for result in results for value in result.age_memory.values())
    core = mean(result.core_length_end for result in results)
    drop = mean(row.end - row.before for result in results for row in result.excursions)
    recover = mean(row.after_return - row.end for result in results for row in result.excursions)
    if abs(age) >= 0.50:
        label = "MemoryEmergentFromAge"
    elif core >= 0.30:
        label = "MemoryEmergentFromCoreResidence"
    elif drop < 0:
        label = "MemoryDestroyedByExcursion"
    elif abs(age) < 0.20 and abs(core) < 0.20:
        label = "MemoryIndependent"
    else:
        label = "MemoryWeakOrUnclear"
    return label, [
        f"Reason: MeanAgeMemoryCorrelation={fmt(age)}; MeanCoreRunLengthEndMemoryCorrelation={fmt(core)}; MeanExcursionMemoryDrop={fmt(drop)}; MeanReturnRecovery={fmt(recover)}.",
        f"MemoryCreationMechanism: LowMemory -> MediumMemory or HighMemory events={sum(len(result.creations) for result in results)}.",
        f"MemoryDestructionMechanism: HighMemory -> MediumMemory or LowMemory events={sum(len(result.destructions) for result in results)}.",
        f"CoreResidenceEffect: CorrelationRunLengthEndMemory={fmt(core)}.",
        f"ExcursionEffect: MeanMemoryDrop={fmt(drop)}; MeanRecovery={fmt(recover)}.",
        f"AgeProxyAssessment: MeanCorrelationAgeMemory={fmt(age)}.",
    ]


def append_audit(lines: list[str], heading: str) -> None:
    lines += ["", heading, "Variables used:", "StructuralState", "AgeBucket", "NodeCategory", "", "Derived metrics:", "MemoryStrength", "HalfLife", "EntropyGrowth", "ForecastConfidence", "", "No Context", "No Arbitration", "No Persistence", "No Phase", "No Optimization", "No Fitting", "No Machine Learning", "No Forward Returns used in memory dynamics construction"]


def write_per_instrument(result: Result, out_root: Path) -> None:
    path = out_root / result.instrument / f"MemoryDynamics_{result.instrument}.txt"
    ensure_dir(path.parent)
    label, reasons = recommendation([result])
    lines = ["APVA Memory Dynamics Study v0.1", "=" * 108, "Diagnostics", f"Instrument: {result.instrument}", "Input path(s): " + ", ".join(str(item) for item in result.source_paths), f"Total rows: {len(result.bars)}", f"Node count: {len(set(result.nodes))}"]
    lines += ["", "1. Memory Stream Reconstruction", "Bar | StateAgeNode | NodeCategory | MemoryStrength | HalfLife | EntropyGrowth | ForecastConfidence"]
    lines += [f"{index} | {node_text(node)} | {category} | {fmt(memory)} | {fmt(life, 2)} | {fmt(entropy)} | {fmt(confidence)}" for index, (node, category, memory, life, entropy, confidence) in enumerate(zip(result.nodes, result.categories, result.memory, result.half_life, result.entropy, result.confidence))]
    lines += ["", "2. Memory Delta", f"MeanMemoryDelta: {fmt(mean(result.deltas))}", f"MedianMemoryDelta: {fmt(median(result.deltas))}", f"PositiveDeltaRate: {pct(mean(value > 0 for value in result.deltas))}", f"NegativeDeltaRate: {pct(mean(value < 0 for value in result.deltas))}", f"FlatDeltaRate: {pct(mean(value == 0 for value in result.deltas))}"]
    lines += ["", "3. Memory Classes", "MemoryClass | Count | Percent"] + [f"{item} | {result.memory_classes.count(item)} | {pct(result.memory_classes.count(item) / len(result.memory_classes))}" for item in CLASSES]
    for number, title, events in ((4, "Memory Creation", result.creations), (5, "Memory Destruction", result.destructions)):
        lines += ["", f"{number}. {title}", f"Count: {len(events)}", f"Rate: {pct(len(events) / max(1, len(result.bars) - 1))}", "Transition | Count"]
        lines += [f"{item} | {count}" for item, count in event_counts(events).most_common(20)]
        lines += ["StateTransition | Count"] + [f"{item} | {count}" for item, count in Counter(f"{event.previous_node[0]} -> {event.current_node[0]}" for event in events).most_common(10)]
        lines += ["CategoryTransition | Count"] + [f"{item} | {count}" for item, count in Counter(f"{event.previous_category} -> {event.current_category}" for event in events).most_common(10)]
    lines += ["", "6. Memory Persistence", "MemoryClass | PersistenceRate | MeanRunLength | MedianRunLength | MaxRunLength"]
    lines += [f"{item} | {pct(values[0])} | {fmt(values[1], 2)} | {fmt(values[2], 2)} | {values[3]}" for item in CLASSES for values in [persistence(result, item)]]
    lines += ["", "7. Core Residence and Memory", f"Correlation CoreRunLength vs MemoryDelta: {fmt(result.core_length_delta)}", f"Correlation CoreRunLength vs EndMemory: {fmt(result.core_length_end)}"]
    lines += ["", "8. Excursions and Memory", f"MeanMemoryDropDuringExcursion: {fmt(mean(row.end - row.before for row in result.excursions))}", f"MeanMemoryRecoveryAfterReturn: {fmt(mean(row.after_return - row.end for row in result.excursions))}", f"MedianMemoryRecoveryBars: {fmt(median(row.return_index - row.start_index for row in result.excursions), 2)}"]
    lines += ["", "9. Return-to-Core Memory Recovery", f"RecoveryHalfLife: {recovery_half_life(result)} bars", "BarsAfterReturn | MeanMemory | RecoveryProbability"] + [f"{horizon} | {fmt(mean(result.recovery[horizon]))} | {pct(recovery_probability(result, horizon))}" for horizon in RECOVERY_HORIZONS]
    lines += ["", "10. Age vs Memory", "State | CorrelationAgeMemory"] + [f"{state} | {fmt(value)}" for state, value in sorted(result.age_memory.items())]
    lines += ["", "11. Memory vs Forecast Confidence", f"Instrument correlation: {fmt(result.memory_confidence)}"] + grouped_correlations(result, result.confidence)
    lines += ["", "12. Memory vs Entropy Growth", f"Instrument correlation: {fmt(result.memory_entropy)}"] + grouped_correlations(result, result.entropy)
    lines += ["", "13. Memory-Regime Flow Matrix", "Source | Destination | Count | Probability"] + [f"{source} | {destination} | {result.flow.get((source, destination), 0)} | {pct(flow_probability(result, source, destination))}" for source in CLASSES for destination in CLASSES]
    lines += ["", "14. Memory-State Interaction", "State | MeanMemory | HighRate | LowRate | CreationRate | DestructionRate | PersistenceRate"] + [f"{row.state} | {fmt(row.mean_memory)} | {pct(row.high_rate)} | {pct(row.low_rate)} | {pct(row.creation_rate)} | {pct(row.destruction_rate)} | {pct(row.persistence_rate)}" for row in result.states.values()]
    lines += ["", "15. Outcome Diagnostics", "Diagnostic only. Outcomes are not used in memory dynamics calculations.", "MemoryClass | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew"]
    lines += [f"{item} | {row.count} | {fmt(row.mean_dr)} | {fmt(row.median_dr)} | {pct(row.continuation)} | {pct(row.failure)} | {pct(row.flat)} | {fmt(row.skew)}" for item, row in sorted(result.outcomes.items())]
    lines += ["", "16. Memory Dynamics Recommendation", f"Classification: {label}"] + reasons
    append_audit(lines, "17. Low-DoF Audit")
    lines += ["", "18. Mechanical Research Notes", "- Memory stream values are inherited exactly from Study 57 StateAge nodes.", "- Aggregate percentile class thresholds are fixed before instrument reports.", "- Forward outcomes are diagnostic only."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aggregate(results: list[Result], low: float, high: float, out_root: Path) -> None:
    path = out_root / "MemoryDynamics" / "MemoryDynamics_All.txt"
    ensure_dir(path.parent)
    instruments = instrument_columns(results)
    by_name = {result.instrument: result for result in results}
    label, reasons = recommendation(results)
    lines = ["APVA Memory Dynamics Study v0.1 - Aggregate", "=" * 108, "Instruments: " + ", ".join(instruments), f"AggregateLowMemoryThreshold: {fmt(low)}", f"AggregateHighMemoryThreshold: {fmt(high)}"]
    lines += ["", "Aggregate Memory Delta Table", "Instrument | MeanMemoryDelta | PositiveDeltaRate | NegativeDeltaRate | FlatDeltaRate"]
    lines += [f"{row.instrument} | {fmt(mean(row.deltas))} | {pct(mean(value > 0 for value in row.deltas))} | {pct(mean(value < 0 for value in row.deltas))} | {pct(mean(value == 0 for value in row.deltas))}" for row in results]
    lines += ["", "Aggregate Memory Class Table", "MemoryClass | " + " | ".join(f"Count_{item}" for item in instruments) + " | ValidInstrumentCount | MeanPercent"]
    lines += [f"{memory_class} | " + " | ".join(str(by_name[item].memory_classes.count(memory_class)) for item in instruments) + f" | {sum(by_name[item].memory_classes.count(memory_class) > 0 for item in instruments)} | {pct(mean(by_name[item].memory_classes.count(memory_class) / len(by_name[item].memory_classes) for item in instruments))}" for memory_class in CLASSES]
    lines += aggregate_event_table(results, instruments, True)
    lines += aggregate_event_table(results, instruments, False)
    lines += ["", "Aggregate Memory Persistence Table", "MemoryClass | " + " | ".join(f"PersistenceRate_{item}" for item in instruments) + " | MeanPersistenceRate | MeanRunLength"]
    lines += [f"{memory_class} | " + " | ".join(pct(persistence(by_name[item], memory_class)[0]) for item in instruments) + f" | {pct(mean(persistence(row, memory_class)[0] for row in results))} | {fmt(mean(persistence(row, memory_class)[1] for row in results), 2)}" for memory_class in CLASSES]
    lines += ["", "Aggregate Core Residence Memory Table", "CoreRunLengthBucket | MeanStartMemory | MeanEndMemory | MeanMemoryDelta | CorrelationRunLengthEndMemory"]
    lines += core_bucket_lines(results)
    lines += ["", "Aggregate Excursion Memory Table", "Instrument | MeanMemoryBeforeExcursion | MeanMemoryAtExcursionStart | MeanMemoryAtExcursionEnd | MeanMemoryAfterReturn | MeanMemoryDrop | MeanRecovery"]
    lines += [f"{row.instrument} | {fmt(mean(item.before for item in row.excursions))} | {fmt(mean(item.start for item in row.excursions))} | {fmt(mean(item.end for item in row.excursions))} | {fmt(mean(item.after_return for item in row.excursions))} | {fmt(mean(item.end - item.before for item in row.excursions))} | {fmt(mean(item.after_return - item.end for item in row.excursions))}" for row in results]
    lines += ["", "Aggregate Return Recovery Table", "BarsAfterReturn | MeanMemory | RecoveryProbability"] + [f"{horizon} | {fmt(mean(value for row in results for value in row.recovery[horizon]))} | {pct(mean(recovery_probability(row, horizon) for row in results))}" for horizon in RECOVERY_HORIZONS]
    lines += aggregate_age_table(results, instruments)
    lines += ["", "Aggregate Memory Correlation Table", "Metric | Correlation", f"MemoryStrength vs ForecastConfidence | {fmt(mean(row.memory_confidence for row in results))}", f"MemoryStrength vs EntropyGrowth | {fmt(mean(row.memory_entropy for row in results))}", f"AgeBucket vs MemoryStrength | {fmt(mean(value for row in results for value in row.age_memory.values()))}", f"CoreRunLength vs EndMemory | {fmt(mean(row.core_length_end for row in results))}"]
    lines += ["", "Aggregate Memory-Regime Flow Matrix", "SourceMemoryClass | DestinationMemoryClass | " + " | ".join(f"Count_{item} | Prob_{item}" for item in instruments) + " | ReplicationCount | MeanProbability"]
    lines += [f"{source} | {destination} | " + " | ".join(f"{by_name[item].flow.get((source, destination), 0)} | {pct(flow_probability(by_name[item], source, destination))}" for item in instruments) + f" | {sum(by_name[item].flow.get((source, destination), 0) > 0 for item in instruments)} | {pct(mean(flow_probability(row, source, destination) for row in results))}" for source in CLASSES for destination in CLASSES]
    lines += aggregate_state_table(results)
    lines += ["", "Aggregate Outcome Table", "MemoryClass | " + " | ".join(f"Count_{item} | MeanDR_{item}" for item in instruments) + " | ValidInstrumentCount"]
    for memory_class in CLASSES:
        values = [by_name[item].outcomes.get(memory_class) for item in instruments]
        lines.append(f"{memory_class} | " + " | ".join(value for row in values for value in ((str(row.count), fmt(row.mean_dr)) if row else ("0", "N/A"))) + f" | {sum(row is not None for row in values)}")
    lines += ["", "Aggregate Recommendation", f"Classification: {label}"] + reasons
    lines += rankings(results)
    append_audit(lines, "Low-DoF Audit")
    lines += ["", "Research Notes", "- How is APVA memory created? See fixed LowMemory exits.", "- How is APVA memory destroyed? See fixed HighMemory exits.", "- Does Core residence create memory? See Core-run correlations.", "- Do excursions destroy memory? See excursion memory table.", "- Does returning to Core restore memory? See recovery curve.", "- Age and Memory are compared mechanically without adding variables."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_event_table(results: list[Result], instruments: list[str], creation: bool) -> list[str]:
    title, field = ("Aggregate Memory Creation Table", "creations") if creation else ("Aggregate Memory Destruction Table", "destructions")
    counters = {row.instrument: event_counts(getattr(row, field)) for row in results}
    keys = sorted({key for counter in counters.values() for key in counter})
    lines = ["", title, "Transition | " + " | ".join(f"Count_{item}" for item in instruments) + " | ReplicationCount | MeanRate"]
    for key in sorted(keys, key=lambda item: (-sum(counters[name][item] for name in instruments), item))[:60]:
        counts = [counters[name][key] for name in instruments]
        lines.append(f"{key} | " + " | ".join(map(str, counts)) + f" | {sum(value > 0 for value in counts)} | {pct(mean(value / max(1, len(next(row for row in results if row.instrument == name).bars) - 1) for name, value in zip(instruments, counts)))}")
    return lines


def core_bucket_lines(results: list[Result]) -> list[str]:
    buckets = (("1-2", lambda n: n <= 2), ("3-5", lambda n: 3 <= n <= 5), ("6+", lambda n: n >= 6))
    lines = []
    for name, predicate in buckets:
        rows = [item for result in results for item in result.core_runs if predicate(item.length)]
        lines.append(f"{name} | {fmt(mean(row.start_memory for row in rows))} | {fmt(mean(row.end_memory for row in rows))} | {fmt(mean(row.delta for row in rows))} | {fmt(correlation((row.length for row in rows), (row.end_memory for row in rows)))}")
    return lines


def aggregate_age_table(results: list[Result], instruments: list[str]) -> list[str]:
    states = sorted({state for row in results for state in row.age_memory})
    by_name = {row.instrument: row for row in results}
    lines = ["", "Aggregate Age vs Memory Table", "State | " + " | ".join(f"CorrelationAgeMemory_{item}" for item in instruments) + " | MeanCorrelation"]
    for state in states:
        values = [by_name[item].age_memory.get(state) for item in instruments]
        lines.append(f"{state} | " + " | ".join(fmt(value) if value is not None else "N/A" for value in values) + f" | {fmt(mean(value for value in values if value is not None))}")
    return lines


def aggregate_state_table(results: list[Result]) -> list[str]:
    states = sorted({state for row in results for state in row.states})
    lines = ["", "Aggregate Memory-State Interaction Table", "State | MeanMemoryStrength | HighMemoryRate | LowMemoryRate | CreationRate | DestructionRate | PersistenceRate"]
    for state in states:
        values = [row.states[state] for row in results if state in row.states]
        lines.append(f"{state} | {fmt(mean(row.mean_memory for row in values))} | {pct(mean(row.high_rate for row in values))} | {pct(mean(row.low_rate for row in values))} | {pct(mean(row.creation_rate for row in values))} | {pct(mean(row.destruction_rate for row in values))} | {pct(mean(row.persistence_rate for row in values))}")
    return lines


def rankings(results: list[Result]) -> list[str]:
    creations = Counter()
    destructions = Counter()
    for row in results:
        creations.update(event_counts(row.creations))
        destructions.update(event_counts(row.destructions))
    states = sorted({state for row in results for state in row.states})
    state_persistence = {state: mean(row.states[state].persistence_rate for row in results if state in row.states) for state in states}
    age = {state: mean(row.age_memory[state] for row in results if state in row.age_memory) for state in states}
    lines = ["", "Aggregate Rankings", "", "1. Strongest memory-creating transitions"] + [f"{key} | {value}" for key, value in creations.most_common(10)]
    lines += ["", "2. Strongest memory-destroying transitions"] + [f"{key} | {value}" for key, value in destructions.most_common(10)]
    lines += ["", "3. Most memory-persistent states"] + [f"{state} | {pct(value)}" for state, value in sorted(state_persistence.items(), key=lambda item: (-item[1], item[0]))]
    lines += ["", "4. Least memory-persistent states"] + [f"{state} | {pct(value)}" for state, value in sorted(state_persistence.items(), key=lambda item: (item[1], item[0]))]
    lines += ["", "5. States where age best predicts memory"] + [f"{state} | {fmt(value)}" for state, value in sorted(age.items(), key=lambda item: (-abs(item[1]), item[0]))]
    lines += ["", "6. States where age least predicts memory"] + [f"{state} | {fmt(value)}" for state, value in sorted(age.items(), key=lambda item: (abs(item[1]), item[0]))]
    lines += ["", "7. Best Core residence memory effects", f"Mean CoreRunLength vs EndMemory correlation: {fmt(mean(row.core_length_end for row in results))}"]
    lines += ["", "8. Strongest excursion memory drops"] + [f"{row.instrument} | {fmt(mean(item.end - item.before for item in row.excursions))}" for row in sorted(results, key=lambda row: mean(item.end - item.before for item in row.excursions))]
    lines += ["", "9. Fastest return-to-Core memory recovery"] + [f"{row.instrument} | {fmt(mean(item.after_return - item.end for item in row.excursions))}" for row in sorted(results, key=lambda row: -mean(item.after_return - item.end for item in row.excursions))]
    lines += ["", "10. Recommended APVA memory dynamics model", recommendation(results)[0]]
    return lines


def validate_invariants(results: list[Result]) -> None:
    for row in results:
        size = len(row.bars)
        if any(len(values) != size for values in (row.nodes, row.categories, row.memory, row.half_life, row.entropy, row.confidence, row.memory_classes)):
            raise RuntimeError(f"{row.instrument}: memory stream length mismatch.")
        if len(row.deltas) != max(0, size - 1):
            raise RuntimeError(f"{row.instrument}: memory delta length mismatch.")
        if sum(row.flow.values()) != max(0, size - 1):
            raise RuntimeError(f"{row.instrument}: memory flow mass mismatch.")
        for source in CLASSES:
            probabilities = [flow_probability(row, source, destination) for destination in CLASSES]
            if sum(row.flow.get((source, destination), 0) for destination in CLASSES) and abs(sum(probabilities) - 1.0) > 1e-12:
                raise RuntimeError(f"{row.instrument}: memory flow probability mismatch for {source}.")


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
    weighted = [aggregate[node].memory_strength for row in loaded for node in [node_for(bar) for bar in row.bars]]
    low, high = percentile(weighted, 0.25), percentile(weighted, 0.75)
    aggregate_results = [build_result(row, aggregate, low, high) for row in loaded]
    validate_invariants(aggregate_results)
    out_root = Path(args.out_root)
    for loaded_row, decay_row in zip(loaded, decay):
        local = local_rows(decay_row)
        score_rows(local)
        write_per_instrument(build_result(loaded_row, local, low, high), out_root)
    write_aggregate(aggregate_results, low, high, out_root)
    print(f"Wrote {len(aggregate_results)} per-instrument report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
