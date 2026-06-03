#!/usr/bin/env python3
"""APVA Excursion Lifecycle Study v0.1.

Study 78 asks what happens after a Neutral_Age6-10 excursion destination forms.

Research only. No trading, optimization, fitting, machine learning, or forward
returns in lifecycle construction.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from APVA_DestinationRobustness_77 import build_contexts
from APVA_ExcursionDestinations_76 import DESTINATION_FAMILIES
from APVA_InformationContribution_63 import thresholds
from APVA_InformationDecay_57 import study as decay_study
from APVA_MinimalEngine_52 import load_results
from APVA_NeutralFailureModes_75 import directional_forward, mean, median, safe_div
from APVA_NodeImportance_58 import aggregate_rows, local_rows, score_rows
from APVA_ProcessGraph_54 import Node, node_for, node_text
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

MAX_HORIZON = 20
RETURN_HORIZONS = (1, 2, 3, 5, 10, 20)
STATE_HORIZONS = (1, 2, 3, 5, 10)
TRAJECTORY_HORIZONS = (0, 1, 2, 3, 5)
OUTCOME_HORIZONS = (1, 3, 5, 10, 20)


@dataclass
class Excursion:
    instrument: str
    start_index: int
    end_index: int | None
    destination: str
    initial_node: Node
    nodes: list[Node]
    returned: bool

    @property
    def length(self) -> int:
        return len(self.nodes)


@dataclass
class InventoryRow:
    destination: str
    count: int
    instrument_counts: Counter[str]
    mean_length: float
    median_length: float
    max_length: int
    return_rate: float
    replication_count: int


@dataclass
class PathRow:
    destination: str
    path: str
    count: int
    probability: float
    replication_count: int


@dataclass
class ReturnRow:
    destination: str
    rates: dict[int, float] = field(default_factory=dict)


@dataclass
class StateEvolutionRow:
    destination: str
    horizon: int
    top_state: str
    top_state_probability: float
    top_node: str
    top_node_probability: float


@dataclass
class MarketEvolutionRow:
    destination: str
    horizon: int
    range_relative: float
    true_range_relative: float
    body_relative: float
    volume_rolling: float
    volume_previous: float
    efficiency: float
    close_location: float


@dataclass
class TrajectoryRow:
    destination: str
    values: dict[int, float]
    classification: str


@dataclass
class RecoveryExhaustionRow:
    metric: str
    recovery: str
    exhaustion: str
    difference: str


@dataclass
class SpecialRow:
    section: str
    metric: str
    value: str
    interpretation: str


@dataclass
class SparseAuditRow:
    destination: str
    count: int
    replication_count: int
    classification: str


@dataclass
class StabilityRow:
    destination: str
    path_entropy: float
    return_entropy: float
    market_variance: float
    stability: str


@dataclass
class OutcomeRow:
    destination: str
    dr: dict[int, float] = field(default_factory=dict)
    continuation: dict[int, float] = field(default_factory=dict)
    failure: dict[int, float] = field(default_factory=dict)
    flat: dict[int, float] = field(default_factory=dict)


@dataclass
class ClassificationRow:
    destination: str
    classification: str
    reason: str


@dataclass
class ReplicationRow:
    destination: str
    classes: dict[str, str] = field(default_factory=dict)
    replication_count: int = 0


@dataclass
class Study77Row:
    destination: str
    signature: str
    finding: str
    assessment: str


@dataclass
class Recommendation:
    classification: str
    distinct: str
    diffuse: str
    transient: str
    persistent: str
    transition_basins: str
    next_step: str
    reason: str


@dataclass
class Result:
    instrument: str
    source_paths: list
    excursions: list[Excursion]
    inventory: dict[str, InventoryRow]
    paths: list[PathRow]
    returns: dict[str, ReturnRow]
    state_evolution: list[StateEvolutionRow]
    market_evolution: list[MarketEvolutionRow]
    participation: dict[str, TrajectoryRow]
    ranges: dict[str, TrajectoryRow]
    efficiency: dict[str, TrajectoryRow]
    recovery_exhaustion: list[RecoveryExhaustionRow]
    special: list[SpecialRow]
    sparse: list[SparseAuditRow]
    stability: dict[str, StabilityRow]
    outcomes: dict[str, OutcomeRow]
    classifications: dict[str, ClassificationRow]
    replication: list[ReplicationRow]
    study77: list[Study77Row]
    recommendation: Recommendation


def is_neutral(node: Node) -> bool:
    return node[0] == "NeutralProcessing"


def efficiency_ratio(bar) -> float | None:
    if bar.body is None or bar.bar_range is None or bar.bar_range == 0:
        return None
    return bar.body / max(bar.bar_range, 1e-12)


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        probability = count / total
        if probability > 0:
            value -= probability * math.log2(probability)
    maximum = math.log2(len(counter)) if len(counter) > 1 else 1.0
    return safe_div(value, maximum) or 0.0


def stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def build_excursions(contexts: list) -> list[Excursion]:
    excursions = []
    for context in contexts:
        for event in context.events:
            start = event.index
            path_nodes = []
            end_index = None
            returned = False
            for offset in range(0, MAX_HORIZON + 1):
                index = start + offset
                if index >= len(context.nodes):
                    break
                node = context.nodes[index]
                if offset > 0 and is_neutral(node):
                    end_index = index
                    returned = True
                    break
                path_nodes.append(node)
            excursions.append(Excursion(
                context.instrument,
                start,
                end_index,
                event.destination,
                context.nodes[start],
                path_nodes,
                returned,
            ))
    return excursions


def grouped(excursions: list[Excursion]) -> dict[str, list[Excursion]]:
    output = {destination: [] for destination in DESTINATION_FAMILIES}
    for excursion in excursions:
        output[excursion.destination].append(excursion)
    return output


def inventory_rows(excursions: list[Excursion]) -> dict[str, InventoryRow]:
    by_destination = grouped(excursions)
    rows = {}
    for destination, rows_for_destination in by_destination.items():
        lengths = [excursion.length for excursion in rows_for_destination]
        counts = Counter(excursion.instrument for excursion in rows_for_destination)
        rows[destination] = InventoryRow(
            destination,
            len(rows_for_destination),
            counts,
            mean(lengths),
            median(lengths),
            max(lengths) if lengths else 0,
            mean([1.0 if excursion.returned else 0.0 for excursion in rows_for_destination]),
            sum(1 for value in counts.values() if value > 0),
        )
    return rows


def path_rows(excursions: list[Excursion]) -> list[PathRow]:
    by_destination = grouped(excursions)
    output = []
    for destination, rows_for_destination in by_destination.items():
        counts = Counter()
        instruments = defaultdict(set)
        for excursion in rows_for_destination:
            path = " -> ".join(node_text(node) for node in excursion.nodes[:5])
            counts[path] += 1
            instruments[path].add(excursion.instrument)
        total = sum(counts.values())
        for path, count in counts.most_common(12):
            output.append(PathRow(destination, path, count, safe_div(count, total) or 0.0, len(instruments[path])))
    return output


def return_rows(excursions: list[Excursion]) -> dict[str, ReturnRow]:
    output = {}
    for destination, rows_for_destination in grouped(excursions).items():
        rates = {}
        for horizon in RETURN_HORIZONS:
            rates[horizon] = mean([
                1.0 if excursion.returned and excursion.end_index is not None and excursion.end_index - excursion.start_index <= horizon else 0.0
                for excursion in rows_for_destination
            ])
        output[destination] = ReturnRow(destination, rates)
    return output


def top_probability(counter: Counter) -> tuple[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return "N/A", 0.0
    value, count = counter.most_common(1)[0]
    return value, safe_div(count, total) or 0.0


def state_evolution_rows(excursions: list[Excursion]) -> list[StateEvolutionRow]:
    output = []
    for destination, rows_for_destination in grouped(excursions).items():
        for horizon in STATE_HORIZONS:
            states = Counter()
            nodes = Counter()
            for excursion in rows_for_destination:
                if horizon < len(excursion.nodes):
                    node = excursion.nodes[horizon]
                    states[node[0]] += 1
                    nodes[node_text(node)] += 1
            top_state, state_prob = top_probability(states)
            top_node, node_prob = top_probability(nodes)
            output.append(StateEvolutionRow(destination, horizon, top_state, state_prob, top_node, node_prob))
    return output


def market_metric(context_lookup: dict[str, object], excursion: Excursion, horizon: int, metric: str) -> float | None:
    context = context_lookup[excursion.instrument]
    index = excursion.start_index + horizon
    if index >= len(context.bars):
        return None
    bar = context.bars[index]
    if metric == "range":
        return bar.range_relative_to_previous
    if metric == "true_range":
        return bar.true_range_relative_to_previous
    if metric == "body":
        return bar.body_relative_to_previous
    if metric == "volume_rolling":
        return bar.volume_relative_to_rolling_mean
    if metric == "volume_previous":
        return bar.volume_relative_to_previous
    if metric == "efficiency":
        return efficiency_ratio(bar)
    if metric == "close_location":
        return bar.close_location
    return None


def market_evolution_rows(excursions: list[Excursion], contexts: list) -> list[MarketEvolutionRow]:
    lookup = {context.instrument: context for context in contexts}
    output = []
    for destination, rows_for_destination in grouped(excursions).items():
        for horizon in STATE_HORIZONS:
            output.append(MarketEvolutionRow(
                destination,
                horizon,
                mean(market_metric(lookup, excursion, horizon, "range") for excursion in rows_for_destination),
                mean(market_metric(lookup, excursion, horizon, "true_range") for excursion in rows_for_destination),
                mean(market_metric(lookup, excursion, horizon, "body") for excursion in rows_for_destination),
                mean(market_metric(lookup, excursion, horizon, "volume_rolling") for excursion in rows_for_destination),
                mean(market_metric(lookup, excursion, horizon, "volume_previous") for excursion in rows_for_destination),
                mean(market_metric(lookup, excursion, horizon, "efficiency") for excursion in rows_for_destination),
                mean(market_metric(lookup, excursion, horizon, "close_location") for excursion in rows_for_destination),
            ))
    return output


def trajectory_rows(excursions: list[Excursion], contexts: list, metric: str, kind: str) -> dict[str, TrajectoryRow]:
    lookup = {context.instrument: context for context in contexts}
    output = {}
    for destination, rows_for_destination in grouped(excursions).items():
        values = {
            horizon: mean(market_metric(lookup, excursion, horizon, metric) for excursion in rows_for_destination)
            for horizon in TRAJECTORY_HORIZONS
        }
        elevated = sum(1 for horizon in TRAJECTORY_HORIZONS if values.get(horizon, 0.0) > 1.0)
        if kind == "volume":
            if elevated >= 2:
                label = "ParticipationContinuation"
            elif values.get(2, 0.0) < 1.0:
                label = "ParticipationDecay"
            else:
                label = "ParticipationMixed"
        elif kind == "range":
            if elevated >= 2:
                label = "RangeContinuation"
            elif values.get(2, 0.0) < 1.0:
                label = "RangeCollapse"
            else:
                label = "RangeMixed"
        else:
            label = "EfficiencyHigh" if values.get(0, 0.0) >= 0.5 and values.get(2, 0.0) >= 0.5 else "EfficiencyDecay"
        output[destination] = TrajectoryRow(destination, values, label)
    return output


def transition_rates(excursions: list[Excursion], destination: str) -> dict[str, float]:
    rows = grouped(excursions).get(destination, [])
    targets = Counter()
    for excursion in rows:
        seen = {node[0] for node in excursion.nodes[1:] if node[0] != destination and node[0] != "NeutralProcessing"}
        for state in seen:
            targets[state] += 1
    total = len(rows)
    return {state: safe_div(count, total) or 0.0 for state, count in targets.items()}


def recovery_exhaustion_rows(inventory: dict[str, InventoryRow], returns: dict[str, ReturnRow],
                             participation: dict[str, TrajectoryRow], ranges: dict[str, TrajectoryRow],
                             efficiency: dict[str, TrajectoryRow], state_rows: list[StateEvolutionRow]) -> list[RecoveryExhaustionRow]:
    rec = "RecoveryResolution"
    exh = "ExhaustionPersistence"
    top_state = {(row.destination, row.horizon): row.top_state for row in state_rows}
    metrics = {
        "ReturnToNeutralRate": (inventory[rec].return_rate, inventory[exh].return_rate),
        "MeanLength": (inventory[rec].mean_length, inventory[exh].mean_length),
        "Range_t2": (ranges[rec].values.get(2, 0.0), ranges[exh].values.get(2, 0.0)),
        "Volume_t2": (participation[rec].values.get(2, 0.0), participation[exh].values.get(2, 0.0)),
        "Efficiency_t2": (efficiency[rec].values.get(2, 0.0), efficiency[exh].values.get(2, 0.0)),
        "TopState_t3": (top_state.get((rec, 3), "N/A"), top_state.get((exh, 3), "N/A")),
    }
    output = []
    for metric, (left, right) in metrics.items():
        if isinstance(left, str) or isinstance(right, str):
            diff = "Different" if left != right else "Same"
            output.append(RecoveryExhaustionRow(metric, str(left), str(right), diff))
        else:
            output.append(RecoveryExhaustionRow(metric, fmt(left), fmt(right), fmt(left - right)))
    return output


def special_rows(excursions: list[Excursion], inventory: dict[str, InventoryRow], returns: dict[str, ReturnRow],
                 participation: dict[str, TrajectoryRow], ranges: dict[str, TrajectoryRow],
                 efficiency: dict[str, TrajectoryRow]) -> list[SpecialRow]:
    rows = []
    for destination, section in (
        ("ReassertionProcessing", "10. Reassertion Lifecycle Table"),
        ("MixedStructure", "11. MixedStructure Lifecycle Table"),
        ("CompressionProcessing", "12. Compression Lifecycle Table"),
    ):
        transitions = transition_rates(excursions, destination)
        top_transition = max(transitions.items(), key=lambda item: item[1], default=("None", 0.0))
        values = [
            ("Count", str(inventory[destination].count), "Sample size."),
            ("MeanLength", fmt(inventory[destination].mean_length), "Average non-neutral lifecycle length."),
            ("ReturnWithin3", pct(returns[destination].rates.get(3, 0.0)), "Fast return pressure."),
            ("VolumeClass", participation[destination].classification, "Participation continuation or decay."),
            ("RangeClass", ranges[destination].classification, "Range continuation or collapse."),
            ("EfficiencyClass", efficiency[destination].classification, "Directional efficiency trajectory."),
            ("TopNonNeutralTransition", f"{top_transition[0]} {pct(top_transition[1])}", "Most common non-neutral continuation."),
        ]
        rows.extend(SpecialRow(section, metric, value, interpretation) for metric, value, interpretation in values)
    return rows


def sparse_rows(inventory: dict[str, InventoryRow]) -> list[SparseAuditRow]:
    output = []
    for destination in ("ConstructiveEmergence", "DestructiveRotation", "DecayToNeutral"):
        row = inventory[destination]
        classification = "SparseUnresolved" if row.count < 25 or row.replication_count < 2 else "SufficientForAudit"
        output.append(SparseAuditRow(destination, row.count, row.replication_count, classification))
    return output


def stability_rows(excursions: list[Excursion], market_rows: list[MarketEvolutionRow], returns: dict[str, ReturnRow],
                   inventory: dict[str, InventoryRow]) -> dict[str, StabilityRow]:
    output = {}
    market_values = defaultdict(list)
    for row in market_rows:
        market_values[row.destination].extend([
            row.range_relative,
            row.true_range_relative,
            row.body_relative,
            row.volume_rolling,
            row.efficiency,
        ])
    for destination, rows_for_destination in grouped(excursions).items():
        path_counter = Counter(" -> ".join(node_text(node) for node in excursion.nodes[:3]) for excursion in rows_for_destination)
        return_counter = Counter("Returned" if excursion.returned else "NotReturned" for excursion in rows_for_destination)
        path_ent = entropy(path_counter)
        return_ent = entropy(return_counter)
        variance = stdev([value for value in market_values[destination] if value is not None])
        stable = path_ent < 0.60 and inventory[destination].replication_count >= 2
        output[destination] = StabilityRow(
            destination,
            path_ent,
            return_ent,
            variance,
            "StableLifecycle" if stable else "DiffuseLifecycle",
        )
    return output


def outcome_rows(excursions: list[Excursion], contexts: list) -> dict[str, OutcomeRow]:
    lookup = {context.instrument: context for context in contexts}
    output = {}
    for destination, rows_for_destination in grouped(excursions).items():
        row = OutcomeRow(destination)
        for horizon in OUTCOME_HORIZONS:
            values = []
            for excursion in rows_for_destination:
                value = directional_forward(lookup[excursion.instrument].bars, excursion.start_index, horizon)
                if value is not None:
                    values.append(value)
            row.dr[horizon] = mean(values)
            row.continuation[horizon] = mean(1.0 if value > 0 else 0.0 for value in values)
            row.failure[horizon] = mean(1.0 if value < 0 else 0.0 for value in values)
            row.flat[horizon] = mean(1.0 if value == 0 else 0.0 for value in values)
        output[destination] = row
    return output


def classify_lifecycles(excursions: list[Excursion], inventory: dict[str, InventoryRow], returns: dict[str, ReturnRow],
                        participation: dict[str, TrajectoryRow], ranges: dict[str, TrajectoryRow]) -> dict[str, ClassificationRow]:
    output = {}
    for destination in DESTINATION_FAMILIES:
        row = inventory[destination]
        if row.count < 25 or row.replication_count < 2:
            label = "SparseUnresolved"
            reason = "Sample insufficient."
        elif returns[destination].rates.get(3, 0.0) >= 0.60:
            label = "TransientReturn"
            reason = "ReturnWithin3 >= 60%."
        elif row.mean_length >= 5 and returns[destination].rates.get(5, 0.0) < 0.50:
            label = "PersistentExcursion"
            reason = "MeanLength >= 5 and ReturnWithin5 < 50%."
        elif participation[destination].classification == "ParticipationContinuation" and ranges[destination].classification == "RangeContinuation":
            label = "ContinuationExcursion"
            reason = "Range and volume stay elevated."
        elif participation[destination].classification == "ParticipationDecay" and ranges[destination].classification == "RangeCollapse":
            label = "CollapseExcursion"
            reason = "Range and volume collapse quickly."
        else:
            transitions = transition_rates(excursions, destination)
            non_neutral_flow = sum(transitions.values())
            if non_neutral_flow >= 0.50:
                label = "TransitionBasin"
                reason = "Often transitions to another non-neutral family before return."
            else:
                label = "DiffuseLifecycle"
                reason = "No fixed lifecycle rule dominated."
        output[destination] = ClassificationRow(destination, label, reason)
    return output


def replication_rows(instrument_results: list[Result] | None, classifications: dict[str, ClassificationRow]) -> list[ReplicationRow]:
    if not instrument_results:
        return [ReplicationRow(destination, {}, 0) for destination in DESTINATION_FAMILIES]
    output = []
    for destination in DESTINATION_FAMILIES:
        classes = {result.instrument: result.classifications[destination].classification for result in instrument_results}
        dominant = Counter(classes.values()).most_common(1)[0][0] if classes else ""
        output.append(ReplicationRow(destination, classes, sum(1 for value in classes.values() if value == dominant)))
    return output


def study77_rows(classifications: dict[str, ClassificationRow], stability: dict[str, StabilityRow],
                 inventory: dict[str, InventoryRow]) -> list[Study77Row]:
    signatures = {
        "CompressionProcessing": "Robust volume expansion signature",
        "MixedStructure": "Robust mixed range/volume signature",
        "ReassertionProcessing": "Robust low-participation signature",
        "RecoveryResolution": "Robust expansion/efficiency signature",
        "ExhaustionPersistence": "Robust expansion/efficiency signature",
        "ConstructiveEmergence": "Unvalidated sparse signature",
        "DestructiveRotation": "Unvalidated sparse signature",
        "DecayToNeutral": "Unvalidated sparse signature",
    }
    rows = []
    for destination in DESTINATION_FAMILIES:
        finding = classifications[destination].classification
        if inventory[destination].count < 25:
            assessment = "Study77Confirmed"
        elif stability[destination].stability == "StableLifecycle":
            assessment = "Study77Confirmed"
        elif finding in {"DiffuseLifecycle", "SparseUnresolved"}:
            assessment = "Study77Weakened"
        else:
            assessment = "Study77Refined"
        rows.append(Study77Row(destination, signatures[destination], finding, assessment))
    return rows


def make_recommendation(classifications: dict[str, ClassificationRow], stability: dict[str, StabilityRow],
                        inventory: dict[str, InventoryRow]) -> Recommendation:
    distinct = [d for d, row in classifications.items() if row.classification not in {"DiffuseLifecycle", "SparseUnresolved"}]
    diffuse = [d for d, row in classifications.items() if row.classification == "DiffuseLifecycle" and inventory[d].count >= 25]
    transient = [d for d, row in classifications.items() if row.classification == "TransientReturn"]
    persistent = [d for d, row in classifications.items() if row.classification == "PersistentExcursion"]
    basins = [d for d, row in classifications.items() if row.classification == "TransitionBasin"]
    if len(distinct) >= 5:
        classification = "StrongLifecycleSeparation"
        next_step = "ProceedToRealTimeStateMachine"
    elif len(distinct) >= 3:
        classification = "PartialLifecycleSeparation"
        next_step = "StudyReturnToNeutral"
    elif distinct:
        classification = "WeakLifecycleSeparation"
        next_step = "StudyReturnToNeutral"
    else:
        classification = "NoDistinctLifecycle"
        next_step = "CollectMoreData"
    reason = "Classification uses fixed return, length, trajectory, transition, sample, and replication thresholds."
    return Recommendation(
        classification,
        ", ".join(distinct) or "None",
        ", ".join(diffuse) or "None",
        ", ".join(transient) or "None",
        ", ".join(persistent) or "None",
        ", ".join(basins) or "None",
        next_step,
        reason,
    )


def build_result(instrument: str, source_paths: list, contexts: list, instrument_results: list[Result] | None = None) -> Result:
    excursions = build_excursions(contexts)
    inventory = inventory_rows(excursions)
    paths = path_rows(excursions)
    returns = return_rows(excursions)
    state_rows = state_evolution_rows(excursions)
    market_rows = market_evolution_rows(excursions, contexts)
    participation = trajectory_rows(excursions, contexts, "volume_rolling", "volume")
    ranges = trajectory_rows(excursions, contexts, "range", "range")
    efficiency = trajectory_rows(excursions, contexts, "efficiency", "efficiency")
    recovery_exhaustion = recovery_exhaustion_rows(inventory, returns, participation, ranges, efficiency, state_rows)
    special = special_rows(excursions, inventory, returns, participation, ranges, efficiency)
    sparse = sparse_rows(inventory)
    stability = stability_rows(excursions, market_rows, returns, inventory)
    outcomes = outcome_rows(excursions, contexts)
    classifications = classify_lifecycles(excursions, inventory, returns, participation, ranges)
    replication = replication_rows(instrument_results, classifications)
    study77 = study77_rows(classifications, stability, inventory)
    recommendation = make_recommendation(classifications, stability, inventory)
    return Result(
        instrument,
        source_paths,
        excursions,
        inventory,
        paths,
        returns,
        state_rows,
        market_rows,
        participation,
        ranges,
        efficiency,
        recovery_exhaustion,
        special,
        sparse,
        stability,
        outcomes,
        classifications,
        replication,
        study77,
        recommendation,
    )


def append_common(lines: list[str], result: Result) -> None:
    lines += ["", "1. Lifecycle Inventory", "DestinationFamily | Count | Count_6E | Count_CL | Count_NQ | MeanLength | MedianLength | MaxLength | ReturnToNeutralRate | ReplicationCount"]
    for row in result.inventory.values():
        lines.append(f"{row.destination} | {row.count} | {row.instrument_counts.get('6E', 0)} | {row.instrument_counts.get('CL', 0)} | {row.instrument_counts.get('NQ', 0)} | {fmt(row.mean_length)} | {fmt(row.median_length)} | {row.max_length} | {pct(row.return_rate)} | {row.replication_count}")

    lines += ["", "2. Common Path Table", "DestinationFamily | Path | Count | Probability | ReplicationCount"]
    for row in result.paths[:120]:
        lines.append(f"{row.destination} | {row.path} | {row.count} | {pct(row.probability)} | {row.replication_count}")

    lines += ["", "3. Return Timing Table", "DestinationFamily | ReturnWithin1 | ReturnWithin2 | ReturnWithin3 | ReturnWithin5 | ReturnWithin10 | ReturnWithin20"]
    for row in result.returns.values():
        lines.append(f"{row.destination} | {pct(row.rates.get(1, 0.0))} | {pct(row.rates.get(2, 0.0))} | {pct(row.rates.get(3, 0.0))} | {pct(row.rates.get(5, 0.0))} | {pct(row.rates.get(10, 0.0))} | {pct(row.rates.get(20, 0.0))}")

    lines += ["", "4. State Evolution Table", "DestinationFamily | Horizon | TopState | TopStateProbability | TopNode | TopNodeProbability"]
    for row in result.state_evolution:
        lines.append(f"{row.destination} | t+{row.horizon} | {row.top_state} | {pct(row.top_state_probability)} | {row.top_node} | {pct(row.top_node_probability)}")

    lines += ["", "5. Market Evolution Table", "DestinationFamily | Horizon | RangeRelativeToPrevious | TrueRangeRelativeToPrevious | BodyRelativeToPrevious | VolumeRelativeToRollingMean | VolumeRelativeToPrevious | EfficiencyRatio | CloseLocation"]
    for row in result.market_evolution:
        lines.append(f"{row.destination} | t+{row.horizon} | {fmt(row.range_relative)} | {fmt(row.true_range_relative)} | {fmt(row.body_relative)} | {fmt(row.volume_rolling)} | {fmt(row.volume_previous)} | {fmt(row.efficiency)} | {fmt(row.close_location)}")

    lines += ["", "6. Participation Trajectory Table", "DestinationFamily | Vol_t0 | Vol_t1 | Vol_t2 | Vol_t3 | Vol_t5 | ParticipationClass"]
    for row in result.participation.values():
        lines.append(f"{row.destination} | {fmt(row.values.get(0, 0.0))} | {fmt(row.values.get(1, 0.0))} | {fmt(row.values.get(2, 0.0))} | {fmt(row.values.get(3, 0.0))} | {fmt(row.values.get(5, 0.0))} | {row.classification}")

    lines += ["", "7. Range Trajectory Table", "DestinationFamily | Range_t0 | Range_t1 | Range_t2 | Range_t3 | Range_t5 | RangeClass"]
    for row in result.ranges.values():
        lines.append(f"{row.destination} | {fmt(row.values.get(0, 0.0))} | {fmt(row.values.get(1, 0.0))} | {fmt(row.values.get(2, 0.0))} | {fmt(row.values.get(3, 0.0))} | {fmt(row.values.get(5, 0.0))} | {row.classification}")

    lines += ["", "8. Efficiency Trajectory Table", "DestinationFamily | Eff_t0 | Eff_t1 | Eff_t2 | Eff_t3 | Eff_t5"]
    for row in result.efficiency.values():
        lines.append(f"{row.destination} | {fmt(row.values.get(0, 0.0))} | {fmt(row.values.get(1, 0.0))} | {fmt(row.values.get(2, 0.0))} | {fmt(row.values.get(3, 0.0))} | {fmt(row.values.get(5, 0.0))}")

    lines += ["", "9. Recovery vs Exhaustion Table", "Metric | RecoveryResolution | ExhaustionPersistence | Difference"]
    for row in result.recovery_exhaustion:
        lines.append(f"{row.metric} | {row.recovery} | {row.exhaustion} | {row.difference}")

    for section in ("10. Reassertion Lifecycle Table", "11. MixedStructure Lifecycle Table", "12. Compression Lifecycle Table"):
        lines += ["", section, "Metric | Value | Interpretation"]
        for row in [item for item in result.special if item.section == section]:
            lines.append(f"{row.metric} | {row.value} | {row.interpretation}")

    lines += ["", "13. Sparse Destination Audit", "DestinationFamily | Count | ReplicationCount | Classification"]
    for row in result.sparse:
        lines.append(f"{row.destination} | {row.count} | {row.replication_count} | {row.classification}")

    lines += ["", "14. Lifecycle Stability Table", "DestinationFamily | PathEntropy | ReturnEntropy | MarketTrajectoryVariance | LifecycleStability"]
    for row in result.stability.values():
        lines.append(f"{row.destination} | {fmt(row.path_entropy)} | {fmt(row.return_entropy)} | {fmt(row.market_variance)} | {row.stability}")

    lines += ["", "15. Outcome Diagnostics Table", "DestinationFamily | DRFwd1 | DRFwd3 | DRFwd5 | DRFwd10 | DRFwd20 | ContinuationRate5 | ContinuationRate10 | FailureRate5 | FlatRate5"]
    for row in result.outcomes.values():
        lines.append(f"{row.destination} | {fmt(row.dr.get(1, 0.0))} | {fmt(row.dr.get(3, 0.0))} | {fmt(row.dr.get(5, 0.0))} | {fmt(row.dr.get(10, 0.0))} | {fmt(row.dr.get(20, 0.0))} | {pct(row.continuation.get(5, 0.0))} | {pct(row.continuation.get(10, 0.0))} | {pct(row.failure.get(5, 0.0))} | {pct(row.flat.get(5, 0.0))}")

    lines += ["", "16. Lifecycle Classification Table", "DestinationFamily | LifecycleClassification | Reason"]
    for row in result.classifications.values():
        lines.append(f"{row.destination} | {row.classification} | {row.reason}")

    lines += ["", "17. Cross-Instrument Replication Table", "DestinationFamily | Class_6E | Class_CL | Class_NQ | ReplicationCount"]
    for row in result.replication:
        lines.append(f"{row.destination} | {row.classes.get('6E', 'N/A')} | {row.classes.get('CL', 'N/A')} | {row.classes.get('NQ', 'N/A')} | {row.replication_count}")

    lines += ["", "18. Study77 Comparison Table", "DestinationFamily | Study77Signature | Study78LifecycleFinding | Assessment"]
    for row in result.study77:
        lines.append(f"{row.destination} | {row.signature} | {row.finding} | {row.assessment}")

    rec = result.recommendation
    lines += [
        "",
        "19. Recommendation",
        f"Classification: {rec.classification}",
        f"DistinctLifecycleFamilies: {rec.distinct}",
        f"DiffuseLifecycleFamilies: {rec.diffuse}",
        f"TransientFamilies: {rec.transient}",
        f"PersistentFamilies: {rec.persistent}",
        f"TransitionBasins: {rec.transition_basins}",
        f"RecommendedNextStep: {rec.next_step}",
        f"Reason: {rec.reason}",
        "",
        "20. Low-DoF Audit",
        "Uses only existing APVA states and OHLCV-derived variables.",
        "No new APVA states.",
        "No new APVA families.",
        "No context.",
        "No arbitration.",
        "No persistence.",
        "No phase.",
        "No optimization.",
        "No fitting.",
        "No machine learning.",
        "No trading logic.",
        "No forward returns used in lifecycle construction.",
    ]


def rankings(result: Result) -> list[str]:
    inventory = list(result.inventory.values())
    longest = sorted(inventory, key=lambda row: row.mean_length, reverse=True)
    shortest = sorted([row for row in inventory if row.count > 0], key=lambda row: row.mean_length)
    fastest = sorted(inventory, key=lambda row: result.returns[row.destination].rates.get(3, 0.0), reverse=True)
    slowest = sorted(inventory, key=lambda row: result.returns[row.destination].rates.get(20, 0.0))
    stable = sorted(result.stability.values(), key=lambda row: row.path_entropy)
    diffuse = sorted(result.stability.values(), key=lambda row: row.path_entropy, reverse=True)
    participation_high = sorted(result.participation.values(), key=lambda row: row.values.get(2, 0.0), reverse=True)
    participation_low = sorted(result.participation.values(), key=lambda row: row.values.get(2, 0.0))
    range_high = sorted(result.ranges.values(), key=lambda row: row.values.get(2, 0.0), reverse=True)
    return [
        "",
        "RANKINGS",
        "1. Longest destination lifecycles: " + "; ".join(f"{row.destination}={fmt(row.mean_length)}" for row in longest[:8]),
        "2. Shortest destination lifecycles: " + "; ".join(f"{row.destination}={fmt(row.mean_length)}" for row in shortest[:8]),
        "3. Fastest return-to-neutral destinations: " + "; ".join(f"{row.destination}={pct(result.returns[row.destination].rates.get(3, 0.0))}" for row in fastest[:8]),
        "4. Slowest return-to-neutral destinations: " + "; ".join(f"{row.destination}={pct(result.returns[row.destination].rates.get(20, 0.0))}" for row in slowest[:8]),
        "5. Most stable lifecycle families: " + "; ".join(f"{row.destination}={fmt(row.path_entropy)}" for row in stable[:8]),
        "6. Most diffuse lifecycle families: " + "; ".join(f"{row.destination}={fmt(row.path_entropy)}" for row in diffuse[:8]),
        "7. Strongest participation continuation: " + "; ".join(f"{row.destination}={fmt(row.values.get(2, 0.0))}" for row in participation_high[:8]),
        "8. Strongest participation decay: " + "; ".join(f"{row.destination}={fmt(row.values.get(2, 0.0))}" for row in participation_low[:8]),
        "9. Strongest range continuation: " + "; ".join(f"{row.destination}={fmt(row.values.get(2, 0.0))}" for row in range_high[:8]),
        f"10. Recommended APVA lifecycle model: {result.recommendation.classification}",
    ]


def write_report(result: Result, out_root: Path, aggregate: bool = False) -> None:
    if aggregate:
        path = out_root / "ExcursionLifecycle78" / "ExcursionLifecycle78_All.txt"
        title = "APVA Excursion Lifecycle Study v0.1 - Aggregate"
    else:
        path = out_root / result.instrument / f"ExcursionLifecycle78_{result.instrument}.txt"
        title = f"APVA Excursion Lifecycle Study v0.1 - {result.instrument}"
    ensure_dir(path.parent)
    lines = [
        title,
        "=" * 96,
        f"Instrument: {result.instrument}",
        f"Input path(s): {', '.join(str(path) for path in result.source_paths)}",
        f"Excursion count: {len(result.excursions)}",
    ]
    append_common(lines, result)
    lines += rankings(result)
    lines += [
        "",
        "RESEARCH NOTES",
        "Mechanical only.",
        "Questions: Once a destination forms, what happens next? Do Recovery and Exhaustion diverge? Is Mixed a destination or transition basin? Can APVA represent Neutral -> Destination -> Lifecycle -> ReturnToNeutral?",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(result: Result) -> None:
    if not result.inventory or set(result.inventory) != set(DESTINATION_FAMILIES):
        raise RuntimeError(f"{result.instrument}: lifecycle inventory incomplete.")
    if not result.returns:
        raise RuntimeError(f"{result.instrument}: return timing missing.")
    if not result.classifications:
        raise RuntimeError(f"{result.instrument}: classifications missing.")
    if not result.recommendation.classification:
        raise RuntimeError(f"{result.instrument}: recommendation missing.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Evidence CSV files or directories")
    parser.add_argument("--out-root", default="Evidence/Output", help="Output root directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_results(args.inputs)
    decay = [decay_study(row) for row in loaded]
    aggregate_node_rows = aggregate_rows(decay)
    score_rows(aggregate_node_rows)
    aggregate_thresholds = thresholds(loaded, aggregate_node_rows)
    local_node_rows = []
    for decay_row in decay:
        rows = local_rows(decay_row)
        score_rows(rows)
        local_node_rows.append(rows)
    contexts = build_contexts(loaded, local_node_rows, aggregate_thresholds)

    out_root = Path(args.out_root)
    instrument_results = []
    for context in contexts:
        result = build_result(context.instrument, context.source_paths, [context])
        validate(result)
        instrument_results.append(result)
        write_report(result, out_root, aggregate=False)
    aggregate_result = build_result("Aggregate", [path for context in contexts for path in context.source_paths], contexts, instrument_results)
    validate(aggregate_result)
    write_report(aggregate_result, out_root, aggregate=True)
    print(f"Wrote {len(instrument_results)} per-instrument Study78 report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
