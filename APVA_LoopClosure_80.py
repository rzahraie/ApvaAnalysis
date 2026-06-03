#!/usr/bin/env python3
"""APVA Loop Closure Study v0.1.

Study 80 asks whether APVA can be represented as a closed attractor loop:
Neutral formation -> lifecycle -> failure -> destination -> excursion ->
return / formation of Neutral.

Research only. No trading, optimization, fitting, machine learning, or forward
returns in loop construction or classification.
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
from APVA_ExcursionLifecycle_78 import efficiency_ratio
from APVA_InformationContribution_63 import thresholds
from APVA_InformationDecay_57 import study as decay_study
from APVA_MinimalEngine_52 import load_results
from APVA_NeutralFailureModes_75 import directional_forward, mean, median, safe_div
from APVA_NodeImportance_58 import aggregate_rows, local_rows, score_rows
from APVA_ProcessGraph_54 import Node, node_for, node_text
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

NEUTRAL_START = ("NeutralProcessing", "1")
MAX_LOOP_HORIZON = 50
OUTCOME_HORIZONS = (1, 3, 5, 10, 20)
PHASE_ORDER = (
    "FormationPhase",
    "LifecyclePhase",
    "LateNeutralPhase",
    "FailurePhase",
    "DestinationPhase",
    "ExcursionPhase",
    "ReturnPhase",
    "NextFormationPhase",
)
PHASE_MACHINE = (
    "NeutralFormation",
    "NeutralMaturation",
    "LateNeutral",
    "NeutralFailure",
    "DestinationSelection",
    "Excursion",
    "ReturnToNeutral",
)
AGE_ORDER = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6-10": 6, "11-20": 7, "21+": 8}


@dataclass
class Loop:
    instrument: str
    loop_id: str
    start: int
    end: int
    status: str
    nodes: list[Node]
    first_non_neutral_offset: int | None = None
    return_offset: int | None = None

    @property
    def duration(self) -> int:
        return self.end - self.start

    @property
    def complete(self) -> bool:
        return self.status == "Complete"


@dataclass
class InventoryRow:
    instrument: str
    complete: int
    incomplete_no_excursion: int
    incomplete_no_return: int
    completion_rate: float
    mean_duration: float
    median_duration: float
    max_duration: int


@dataclass
class PhaseDurationRow:
    phase: str
    mean_duration: float
    median_duration: float
    max_duration: int
    percent_of_loop: float


@dataclass
class CanonicalPathRow:
    path: str
    count: int
    probability: float
    replication_count: int
    mean_duration: float


@dataclass
class MaturationRow:
    max_age: str
    count: int
    probability: float
    top_destination: str


@dataclass
class FailureNodeRow:
    failure_node: str
    count: int
    probability: float
    top_destination: str
    top_return_family: str


@dataclass
class DestinationContextRow:
    max_age: str
    failure_node: str
    destination: str
    count: int
    probability: float


@dataclass
class ExcursionSequenceRow:
    sequence: str
    count: int
    probability: float
    replication_count: int
    mean_return_time: float


@dataclass
class ReturnFamilyRow:
    return_family: str
    count: int
    probability: float
    top_destination: str
    top_sequence: str


@dataclass
class MarketSignatureRow:
    phase: str
    range_relative: float
    true_range_relative: float
    body_relative: float
    volume_rolling: float
    volume_previous: float
    efficiency: float
    close_location: float


@dataclass
class DurationClassRow:
    duration_class: str
    count: int
    probability: float
    top_destination: str
    top_return_family: str
    mean_duration: float


@dataclass
class StabilityRow:
    path_entropy: float
    destination_entropy: float
    return_entropy: float
    duration_variance: float
    stability_class: str


@dataclass
class FailurePointRow:
    failure_type: str
    count: int
    last_observed_state: str
    max_neutral_age: str
    market_profile: str


@dataclass
class InstrumentReplicationRow:
    instrument: str
    completion_rate: float
    top_path: str
    top_destination: str
    top_return_family: str
    mean_duration: float
    stability_class: str


@dataclass
class PhaseTransitionRow:
    from_phase: str
    to_phase: str
    count: int
    probability: float
    replication_count: int


@dataclass
class RetentionRow:
    model: str
    top1: float
    top2: float
    brier: float
    entropy: float


@dataclass
class OutcomeRow:
    canonical_class: str
    dr: dict[int, float] = field(default_factory=dict)
    continuation5: float = 0.0
    continuation10: float = 0.0
    failure5: float = 0.0
    flat5: float = 0.0


@dataclass
class Recommendation:
    classification: str
    neutral_attractor: str
    routing_families: str
    high_energy_families: str
    reset_families: str
    next_step: str
    reason: str


@dataclass
class Result:
    instrument: str
    source_paths: list
    contexts: list
    loops: list[Loop]
    inventory: list[InventoryRow]
    phase_durations: list[PhaseDurationRow]
    canonical_paths: list[CanonicalPathRow]
    maturation: list[MaturationRow]
    failure_nodes: list[FailureNodeRow]
    destination_context: list[DestinationContextRow]
    excursion_sequences: list[ExcursionSequenceRow]
    return_families: list[ReturnFamilyRow]
    market_signature: list[MarketSignatureRow]
    duration_classes: list[DurationClassRow]
    stability: StabilityRow
    failure_points: list[FailurePointRow]
    instrument_replication: list[InstrumentReplicationRow]
    phase_transitions: list[PhaseTransitionRow]
    retention: list[RetentionRow]
    outcomes: list[OutcomeRow]
    recommendation: Recommendation


def is_neutral(node: Node) -> bool:
    return node[0] == "NeutralProcessing"


def is_non_neutral(node: Node) -> bool:
    return node[0] in DESTINATION_FAMILIES


def age_rank(node: Node) -> int:
    return AGE_ORDER.get(node[1], 0) if is_neutral(node) else 0


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


def top_counter(counter: Counter) -> tuple[str, int, float]:
    total = sum(counter.values())
    if total <= 0:
        return "N/A", 0, 0.0
    key, count = counter.most_common(1)[0]
    return str(key), count, safe_div(count, total) or 0.0


def market_value(context, index: int, metric: str) -> float | None:
    if index < 0 or index >= len(context.bars):
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


def build_loops_for_context(context) -> list[Loop]:
    loops = []
    index = 0
    serial = 1
    while index < len(context.nodes):
        if context.nodes[index] != NEUTRAL_START:
            index += 1
            continue
        end_limit = min(len(context.nodes) - 1, index + MAX_LOOP_HORIZON)
        saw_later_neutral = False
        first_non_neutral = None
        return_index = None
        for scan in range(index + 1, end_limit + 1):
            node = context.nodes[scan]
            if is_neutral(node) and age_rank(node) > 1:
                saw_later_neutral = True
            if saw_later_neutral and first_non_neutral is None and is_non_neutral(node):
                first_non_neutral = scan
            if first_non_neutral is not None and node == NEUTRAL_START:
                return_index = scan
                break
        if first_non_neutral is None:
            end = end_limit
            status = "IncompleteNoExcursion"
        elif return_index is None:
            end = end_limit
            status = "IncompleteNoReturn"
        else:
            end = return_index
            status = "Complete"
        loops.append(Loop(
            context.instrument,
            f"{context.instrument}_{serial}",
            index,
            end,
            status,
            context.nodes[index:end + 1],
            first_non_neutral - index if first_non_neutral is not None else None,
            return_index - index if return_index is not None else None,
        ))
        serial += 1
        index = end + 1 if status == "Complete" else index + 1
    return loops


def build_loops(contexts: list) -> list[Loop]:
    loops = []
    for context in contexts:
        loops.extend(build_loops_for_context(context))
    return loops


def context_lookup(contexts: list) -> dict[str, object]:
    return {context.instrument: context for context in contexts}


def complete_loops(loops: list[Loop]) -> list[Loop]:
    return [loop for loop in loops if loop.complete]


def max_neutral_age(loop: Loop) -> str:
    neutral_nodes = [node for node in loop.nodes if is_neutral(node)]
    if not neutral_nodes:
        return "N/A"
    node = max(neutral_nodes, key=age_rank)
    return node[1]


def failure_node(loop: Loop) -> Node | None:
    if loop.first_non_neutral_offset is None:
        return None
    index = loop.first_non_neutral_offset - 1
    return loop.nodes[index] if 0 <= index < len(loop.nodes) else None


def destination_family(loop: Loop) -> str:
    if loop.first_non_neutral_offset is None:
        return "None"
    return loop.nodes[loop.first_non_neutral_offset][0]


def excursion_nodes(loop: Loop) -> list[Node]:
    if loop.first_non_neutral_offset is None:
        return []
    end = loop.return_offset if loop.return_offset is not None else len(loop.nodes)
    return [node for node in loop.nodes[loop.first_non_neutral_offset:end] if is_non_neutral(node)]


def compressed_sequence(nodes: list[Node]) -> list[str]:
    output = []
    last = None
    for node in nodes:
        family = node[0]
        if family != last:
            output.append(family)
            last = family
    return output


def return_family(loop: Loop) -> str:
    nodes = excursion_nodes(loop)
    return nodes[-1][0] if nodes else "None"


def canonical_path(loop: Loop) -> str:
    sequence = compressed_sequence(excursion_nodes(loop))
    middle = " -> ".join(sequence) if sequence else "None"
    return f"NeutralStart -> Age{max_neutral_age(loop)} -> {destination_family(loop)} -> {middle} -> {return_family(loop)} -> NeutralStart"


def phase_for_offset(loop: Loop, offset: int) -> str:
    node = loop.nodes[offset]
    if offset == 0:
        return "FormationPhase"
    if offset == len(loop.nodes) - 1 and node == NEUTRAL_START and loop.complete:
        return "NextFormationPhase"
    if loop.first_non_neutral_offset is not None and offset == loop.first_non_neutral_offset - 1:
        return "FailurePhase"
    if loop.first_non_neutral_offset is not None and offset == loop.first_non_neutral_offset:
        return "DestinationPhase"
    if loop.return_offset is not None and offset == loop.return_offset - 1 and is_non_neutral(node):
        return "ReturnPhase"
    if is_non_neutral(node):
        return "ExcursionPhase"
    if is_neutral(node) and node[1] in {"2", "3", "4"}:
        return "LifecyclePhase"
    if is_neutral(node):
        return "LateNeutralPhase"
    return "ExcursionPhase"


def machine_phase(phase: str) -> str:
    return {
        "FormationPhase": "NeutralFormation",
        "LifecyclePhase": "NeutralMaturation",
        "LateNeutralPhase": "LateNeutral",
        "FailurePhase": "NeutralFailure",
        "DestinationPhase": "DestinationSelection",
        "ExcursionPhase": "Excursion",
        "ReturnPhase": "ReturnToNeutral",
        "NextFormationPhase": "NeutralFormation",
    }.get(phase, "Excursion")


def loop_phases(loop: Loop) -> list[str]:
    return [phase_for_offset(loop, offset) for offset in range(len(loop.nodes))]


def inventory_rows(loops: list[Loop]) -> list[InventoryRow]:
    by_instrument = defaultdict(list)
    for loop in loops:
        by_instrument[loop.instrument].append(loop)
    by_instrument["Aggregate"] = loops
    rows = []
    for instrument, rows_for_instrument in by_instrument.items():
        complete = [loop for loop in rows_for_instrument if loop.status == "Complete"]
        no_excursion = sum(1 for loop in rows_for_instrument if loop.status == "IncompleteNoExcursion")
        no_return = sum(1 for loop in rows_for_instrument if loop.status == "IncompleteNoReturn")
        durations = [loop.duration for loop in complete]
        rows.append(InventoryRow(
            instrument,
            len(complete),
            no_excursion,
            no_return,
            safe_div(len(complete), len(rows_for_instrument)) or 0.0,
            mean(durations),
            median(durations),
            max(durations) if durations else 0,
        ))
    return rows


def phase_duration_rows(loops: list[Loop]) -> list[PhaseDurationRow]:
    counts = defaultdict(list)
    total_nodes = 0
    for loop in complete_loops(loops):
        phase_counts = Counter(loop_phases(loop))
        total_nodes += sum(phase_counts.values())
        for phase in PHASE_ORDER:
            counts[phase].append(phase_counts.get(phase, 0))
    output = []
    for phase in PHASE_ORDER:
        values = counts[phase]
        output.append(PhaseDurationRow(
            phase,
            mean(values),
            median(values),
            max(values) if values else 0,
            safe_div(sum(values), total_nodes) or 0.0,
        ))
    return output


def canonical_rows(loops: list[Loop]) -> list[CanonicalPathRow]:
    counts = Counter()
    instruments = defaultdict(set)
    durations = defaultdict(list)
    complete = complete_loops(loops)
    for loop in complete:
        path = canonical_path(loop)
        counts[path] += 1
        instruments[path].add(loop.instrument)
        durations[path].append(loop.duration)
    total = sum(counts.values())
    return [
        CanonicalPathRow(path, count, safe_div(count, total) or 0.0, len(instruments[path]), mean(durations[path]))
        for path, count in counts.most_common(80)
    ]


def maturation_rows(loops: list[Loop]) -> list[MaturationRow]:
    counts = Counter()
    destinations = defaultdict(Counter)
    complete = complete_loops(loops)
    for loop in complete:
        age = max_neutral_age(loop)
        counts[age] += 1
        destinations[age][destination_family(loop)] += 1
    total = sum(counts.values())
    rows = []
    for age, count in counts.most_common():
        rows.append(MaturationRow(age, count, safe_div(count, total) or 0.0, top_counter(destinations[age])[0]))
    return rows


def failure_node_rows(loops: list[Loop]) -> list[FailureNodeRow]:
    counts = Counter()
    destinations = defaultdict(Counter)
    returns = defaultdict(Counter)
    complete = complete_loops(loops)
    for loop in complete:
        node = failure_node(loop)
        text = node_text(node) if node else "None"
        counts[text] += 1
        destinations[text][destination_family(loop)] += 1
        returns[text][return_family(loop)] += 1
    total = sum(counts.values())
    return [
        FailureNodeRow(text, count, safe_div(count, total) or 0.0, top_counter(destinations[text])[0], top_counter(returns[text])[0])
        for text, count in counts.most_common()
    ]


def destination_context_rows(loops: list[Loop]) -> list[DestinationContextRow]:
    counts = Counter()
    totals = Counter()
    for loop in complete_loops(loops):
        key = (max_neutral_age(loop), node_text(failure_node(loop)) if failure_node(loop) else "None", destination_family(loop))
        base = key[:2]
        counts[key] += 1
        totals[base] += 1
    return [
        DestinationContextRow(age, failure, destination, count, safe_div(count, totals[(age, failure)]) or 0.0)
        for (age, failure, destination), count in counts.most_common(160)
    ]


def excursion_sequence_rows(loops: list[Loop]) -> list[ExcursionSequenceRow]:
    counts = Counter()
    instruments = defaultdict(set)
    return_times = defaultdict(list)
    for loop in complete_loops(loops):
        sequence = " -> ".join(compressed_sequence(excursion_nodes(loop))) or "None"
        counts[sequence] += 1
        instruments[sequence].add(loop.instrument)
        if loop.first_non_neutral_offset is not None and loop.return_offset is not None:
            return_times[sequence].append(loop.return_offset - loop.first_non_neutral_offset)
    total = sum(counts.values())
    return [
        ExcursionSequenceRow(sequence, count, safe_div(count, total) or 0.0, len(instruments[sequence]), mean(return_times[sequence]))
        for sequence, count in counts.most_common(120)
    ]


def return_family_rows(loops: list[Loop]) -> list[ReturnFamilyRow]:
    counts = Counter()
    destinations = defaultdict(Counter)
    sequences = defaultdict(Counter)
    for loop in complete_loops(loops):
        family = return_family(loop)
        sequence = " -> ".join(compressed_sequence(excursion_nodes(loop))) or "None"
        counts[family] += 1
        destinations[family][destination_family(loop)] += 1
        sequences[family][sequence] += 1
    total = sum(counts.values())
    return [
        ReturnFamilyRow(family, count, safe_div(count, total) or 0.0, top_counter(destinations[family])[0], top_counter(sequences[family])[0])
        for family, count in counts.most_common()
    ]


def market_signature_rows(loops: list[Loop], contexts: list) -> list[MarketSignatureRow]:
    lookup = context_lookup(contexts)
    samples = {phase: defaultdict(list) for phase in PHASE_ORDER}
    for loop in complete_loops(loops):
        context = lookup[loop.instrument]
        for offset, phase in enumerate(loop_phases(loop)):
            index = loop.start + offset
            for metric in ("range", "true_range", "body", "volume_rolling", "volume_previous", "efficiency", "close_location"):
                value = market_value(context, index, metric)
                if value is not None:
                    samples[phase][metric].append(float(value))
    return [
        MarketSignatureRow(
            phase,
            mean(samples[phase]["range"]),
            mean(samples[phase]["true_range"]),
            mean(samples[phase]["body"]),
            mean(samples[phase]["volume_rolling"]),
            mean(samples[phase]["volume_previous"]),
            mean(samples[phase]["efficiency"]),
            mean(samples[phase]["close_location"]),
        )
        for phase in PHASE_ORDER
    ]


def duration_class(loop: Loop) -> str:
    if loop.duration <= 5:
        return "ShortLoop"
    if loop.duration <= 15:
        return "NormalLoop"
    return "LongLoop"


def duration_class_rows(loops: list[Loop]) -> list[DurationClassRow]:
    counts = Counter()
    destinations = defaultdict(Counter)
    returns = defaultdict(Counter)
    durations = defaultdict(list)
    for loop in complete_loops(loops):
        label = duration_class(loop)
        counts[label] += 1
        destinations[label][destination_family(loop)] += 1
        returns[label][return_family(loop)] += 1
        durations[label].append(loop.duration)
    total = sum(counts.values())
    return [
        DurationClassRow(label, counts[label], safe_div(counts[label], total) or 0.0, top_counter(destinations[label])[0], top_counter(returns[label])[0], mean(durations[label]))
        for label in ("ShortLoop", "NormalLoop", "LongLoop")
    ]


def stability_row(loops: list[Loop]) -> StabilityRow:
    complete = complete_loops(loops)
    path_ent = entropy(Counter(canonical_path(loop) for loop in complete))
    dest_ent = entropy(Counter(destination_family(loop) for loop in complete))
    return_ent = entropy(Counter(return_family(loop) for loop in complete))
    durations = [loop.duration for loop in complete]
    variance = statistics.pvariance(durations) if len(durations) > 1 else 0.0
    completion_rate = safe_div(len(complete), len(loops)) or 0.0
    if completion_rate < 0.80:
        label = "WeakLoop"
    elif max(path_ent, dest_ent, return_ent) <= 0.50:
        label = "StableClosedLoop"
    elif max(path_ent, dest_ent, return_ent) > 0.50:
        label = "DiffuseClosedLoop"
    else:
        label = "StableClosedLoop"
    return StabilityRow(path_ent, dest_ent, return_ent, variance, label)


def failure_point_rows(loops: list[Loop], contexts: list) -> list[FailurePointRow]:
    lookup = context_lookup(contexts)
    output = []
    for status in ("IncompleteNoExcursion", "IncompleteNoReturn"):
        rows = [loop for loop in loops if loop.status == status]
        last_states = Counter(loop.nodes[-1][0] if loop.nodes else "None" for loop in rows)
        max_ages = Counter(max_neutral_age(loop) for loop in rows)
        market = []
        for loop in rows:
            context = lookup[loop.instrument]
            market.extend([
                market_value(context, loop.end, "range"),
                market_value(context, loop.end, "volume_rolling"),
                market_value(context, loop.end, "efficiency"),
            ])
        profile = f"Range/Volume/Eff mean={fmt(mean(value for value in market if value is not None))}"
        output.append(FailurePointRow(status, len(rows), top_counter(last_states)[0], top_counter(max_ages)[0], profile))
    return output


def instrument_replication_rows(instrument_results: list[Result]) -> list[InstrumentReplicationRow]:
    rows = []
    for result in instrument_results:
        inventory = next((row for row in result.inventory if row.instrument == result.instrument), result.inventory[0])
        top_path = result.canonical_paths[0].path if result.canonical_paths else "None"
        top_dest = top_counter(Counter(destination_family(loop) for loop in complete_loops(result.loops)))[0]
        top_return = top_counter(Counter(return_family(loop) for loop in complete_loops(result.loops)))[0]
        rows.append(InstrumentReplicationRow(result.instrument, inventory.completion_rate, top_path, top_dest, top_return, inventory.mean_duration, result.stability.stability_class))
    return rows


def phase_transition_rows(loops: list[Loop], instrument_results: list[Result] | None = None) -> list[PhaseTransitionRow]:
    counts = Counter()
    instruments = defaultdict(set)
    for loop in complete_loops(loops):
        phases = [machine_phase(phase) for phase in loop_phases(loop)]
        for left, right in zip(phases, phases[1:]):
            if left == right:
                continue
            counts[(left, right)] += 1
            instruments[(left, right)].add(loop.instrument)
    outgoing = Counter()
    for (left, _right), count in counts.items():
        outgoing[left] += count
    return [
        PhaseTransitionRow(left, right, count, safe_div(count, outgoing[left]) or 0.0, len(instruments[(left, right)]))
        for (left, right), count in counts.most_common()
    ]


def probability_distribution(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: count / total for key, count in counter.items()}


def transition_retention_rows(loops: list[Loop]) -> list[RetentionRow]:
    complete = complete_loops(loops)
    transitions = []
    for loop in complete:
        phases = [machine_phase(phase) for phase in loop_phases(loop)]
        for offset in range(len(loop.nodes) - 1):
            transitions.append((node_text(loop.nodes[offset]), phases[offset], phases[offset + 1]))
    rows = []
    for model_name, key_index in (("FullStateAge", 0), ("MinimalLoopPhase", 1)):
        grouped = defaultdict(Counter)
        for transition in transitions:
            grouped[transition[key_index]][transition[2]] += 1
        correct1 = correct2 = brier_total = entropy_total = total = 0
        for transition in transitions:
            distribution = probability_distribution(grouped[transition[key_index]])
            ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
            actual = transition[2]
            if ranked and ranked[0][0] == actual:
                correct1 += 1
            if actual in [key for key, _value in ranked[:2]]:
                correct2 += 1
            phases = set(PHASE_MACHINE)
            brier_total += sum(((1.0 if phase == actual else 0.0) - distribution.get(phase, 0.0)) ** 2 for phase in phases) / len(phases)
            entropy_total += entropy(Counter({phase: int(prob * 1000000) for phase, prob in distribution.items()}))
            total += 1
        rows.append(RetentionRow(
            model_name,
            safe_div(correct1, total) or 0.0,
            safe_div(correct2, total) or 0.0,
            safe_div(brier_total, total) or 0.0,
            safe_div(entropy_total, total) or 0.0,
        ))
    return rows


def outcome_rows(loops: list[Loop], contexts: list) -> list[OutcomeRow]:
    lookup = context_lookup(contexts)
    grouped = defaultdict(list)
    for loop in complete_loops(loops):
        grouped[canonical_path(loop)].append(loop)
    rows = []
    for path, rows_for_path in Counter({key: len(value) for key, value in grouped.items()}).most_common(40):
        loops_for_path = grouped[path]
        row = OutcomeRow(path)
        for horizon in OUTCOME_HORIZONS:
            values = []
            for loop in loops_for_path:
                if loop.first_non_neutral_offset is None:
                    continue
                context = lookup[loop.instrument]
                value = directional_forward(context.bars, loop.start + loop.first_non_neutral_offset, horizon)
                if value is not None:
                    values.append(value)
            row.dr[horizon] = mean(values)
            if horizon == 5:
                row.continuation5 = mean(1.0 if value > 0 else 0.0 for value in values)
                row.failure5 = mean(1.0 if value < 0 else 0.0 for value in values)
                row.flat5 = mean(1.0 if value == 0 else 0.0 for value in values)
            if horizon == 10:
                row.continuation10 = mean(1.0 if value > 0 else 0.0 for value in values)
        rows.append(row)
    return rows


def make_recommendation(loops: list[Loop], stability: StabilityRow, return_rows: list[ReturnFamilyRow],
                        destination_rows: list[MaturationRow]) -> Recommendation:
    complete = complete_loops(loops)
    completion = safe_div(len(complete), len(loops)) or 0.0
    top_returns = [row.return_family for row in return_rows[:3]]
    top_destinations = top_counter(Counter(destination_family(loop) for loop in complete))[0]
    if completion >= 0.80 and stability.stability_class == "StableClosedLoop":
        classification = "AttractorLoopConfirmed"
        next_step = "ProceedToRealTimeStateMachine"
    elif completion >= 0.80:
        classification = "DiffuseClosedLoop"
        next_step = "ProceedToSignalMinimalism"
    elif completion >= 0.50:
        classification = "WeakClosedLoop"
        next_step = "CollectMoreData"
    else:
        classification = "NoClosedLoop"
        next_step = "RejectLoopModel"
    neutral_attractor = f"Neutral return completion rate {pct(completion)}."
    routing = ", ".join(family for family in ("CompressionProcessing", "MixedStructure", "DecayToNeutral") if family in top_returns) or ", ".join(top_returns[:2])
    high_energy = ", ".join(family for family in ("RecoveryResolution", "ExhaustionPersistence") if family == top_destinations or family in top_returns) or "RecoveryResolution, ExhaustionPersistence"
    reset = "ReassertionProcessing" if top_destinations == "ReassertionProcessing" or "ReassertionProcessing" in top_returns else "ReassertionProcessing candidate"
    reason = "Loop class uses fixed completion and entropy thresholds; no returns or trading outcomes enter construction."
    return Recommendation(classification, neutral_attractor, routing, high_energy, reset, next_step, reason)


def build_result(instrument: str, source_paths: list, contexts: list, instrument_results: list[Result] | None = None) -> Result:
    loops = build_loops(contexts)
    inventory = inventory_rows(loops)
    phases = phase_duration_rows(loops)
    canonical = canonical_rows(loops)
    maturation = maturation_rows(loops)
    failures = failure_node_rows(loops)
    destination_context = destination_context_rows(loops)
    sequences = excursion_sequence_rows(loops)
    returns = return_family_rows(loops)
    market = market_signature_rows(loops, contexts)
    duration_classes = duration_class_rows(loops)
    stability = stability_row(loops)
    failure_points = failure_point_rows(loops, contexts)
    instrument_replication = instrument_replication_rows(instrument_results) if instrument_results else []
    transitions = phase_transition_rows(loops, instrument_results)
    retention = transition_retention_rows(loops)
    outcomes = outcome_rows(loops, contexts)
    recommendation = make_recommendation(loops, stability, returns, maturation)
    return Result(
        instrument,
        source_paths,
        contexts,
        loops,
        inventory,
        phases,
        canonical,
        maturation,
        failures,
        destination_context,
        sequences,
        returns,
        market,
        duration_classes,
        stability,
        failure_points,
        instrument_replication,
        transitions,
        retention,
        outcomes,
        recommendation,
    )


def append_common(lines: list[str], result: Result) -> None:
    lines += ["", "1. Loop Inventory", "Instrument | CompleteLoopCount | IncompleteNoExcursionCount | IncompleteNoReturnCount | CompletionRate | MeanLoopDuration | MedianLoopDuration | MaxLoopDuration"]
    for row in result.inventory:
        lines.append(f"{row.instrument} | {row.complete} | {row.incomplete_no_excursion} | {row.incomplete_no_return} | {pct(row.completion_rate)} | {fmt(row.mean_duration)} | {fmt(row.median_duration)} | {row.max_duration}")

    lines += ["", "2. Phase Duration Table", "LoopPhase | MeanDuration | MedianDuration | MaxDuration | PercentOfLoop"]
    for row in result.phase_durations:
        lines.append(f"{row.phase} | {fmt(row.mean_duration)} | {fmt(row.median_duration)} | {row.max_duration} | {pct(row.percent_of_loop)}")

    lines += ["", "3. Canonical Loop Path Table", "CanonicalPath | Count | Probability | ReplicationCount | MeanDuration"]
    for row in result.canonical_paths[:80]:
        lines.append(f"{row.path} | {row.count} | {pct(row.probability)} | {row.replication_count} | {fmt(row.mean_duration)}")

    lines += ["", "4. Neutral Maturation Table", "MaxNeutralAgeReached | Count | Probability | TopDestinationFamily"]
    for row in result.maturation:
        lines.append(f"{row.max_age} | {row.count} | {pct(row.probability)} | {row.top_destination}")

    lines += ["", "5. Failure Node Table", "FailureNode | Count | Probability | TopDestinationFamily | TopReturnFamily"]
    for row in result.failure_nodes:
        lines.append(f"{row.failure_node} | {row.count} | {pct(row.probability)} | {row.top_destination} | {row.top_return_family}")

    lines += ["", "6. Destination In Loop Context Table", "MaxNeutralAgeReached | FailureNode | DestinationFamily | Count | Probability"]
    for row in result.destination_context[:160]:
        lines.append(f"{row.max_age} | {row.failure_node} | {row.destination} | {row.count} | {pct(row.probability)}")

    lines += ["", "7. Excursion Sequence Table", "ExcursionSequence | Count | Probability | ReplicationCount | MeanReturnTime"]
    for row in result.excursion_sequences[:120]:
        lines.append(f"{row.sequence} | {row.count} | {pct(row.probability)} | {row.replication_count} | {fmt(row.mean_return_time)}")

    lines += ["", "8. Return Family Table", "ReturnFamily | Count | Probability | TopDestinationFamily | TopExcursionSequence"]
    for row in result.return_families:
        lines.append(f"{row.return_family} | {row.count} | {pct(row.probability)} | {row.top_destination} | {row.top_sequence}")

    lines += ["", "9. Loop Market Signature Table", "LoopPhase | RangeRelativeToPrevious | TrueRangeRelativeToPrevious | BodyRelativeToPrevious | VolumeRelativeToRollingMean | VolumeRelativeToPrevious | EfficiencyRatio | CloseLocation"]
    for row in result.market_signature:
        lines.append(f"{row.phase} | {fmt(row.range_relative)} | {fmt(row.true_range_relative)} | {fmt(row.body_relative)} | {fmt(row.volume_rolling)} | {fmt(row.volume_previous)} | {fmt(row.efficiency)} | {fmt(row.close_location)}")

    lines += ["", "10. Loop Energy Curve Table", "LoopPhase | RangeEnergy | VolumeEnergy | EfficiencyEnergy"]
    for row in result.market_signature:
        lines.append(f"{row.phase} | {fmt(row.range_relative)} | {fmt(row.volume_rolling)} | {fmt(row.efficiency)}")

    lines += ["", "11. Loop Duration Class Table", "DurationClass | Count | Probability | TopDestination | TopReturnFamily | MeanDuration"]
    for row in result.duration_classes:
        lines.append(f"{row.duration_class} | {row.count} | {pct(row.probability)} | {row.top_destination} | {row.top_return_family} | {fmt(row.mean_duration)}")

    s = result.stability
    lines += ["", "12. Loop Stability Table", "PathEntropy | DestinationEntropy | ReturnEntropy | DurationVariance | LoopStabilityClass"]
    lines.append(f"{fmt(s.path_entropy)} | {fmt(s.destination_entropy)} | {fmt(s.return_entropy)} | {fmt(s.duration_variance)} | {s.stability_class}")

    lines += ["", "13. Loop Failure Point Table", "FailureType | Count | LastObservedState | MaxNeutralAgeReached | MarketProfileSummary"]
    for row in result.failure_points:
        lines.append(f"{row.failure_type} | {row.count} | {row.last_observed_state} | {row.max_neutral_age} | {row.market_profile}")

    lines += ["", "14. Cross-Instrument Replication Table", "Instrument | CompletionRate | TopCanonicalPath | TopDestination | TopReturnFamily | MeanDuration | LoopStabilityClass"]
    for row in result.instrument_replication:
        lines.append(f"{row.instrument} | {pct(row.completion_rate)} | {row.top_path} | {row.top_destination} | {row.top_return_family} | {fmt(row.mean_duration)} | {row.stability_class}")

    lines += ["", "15. Minimal Loop State Machine Table", "FromPhase | ToPhase | Count | Probability | ReplicationCount"]
    for row in result.phase_transitions:
        lines.append(f"{row.from_phase} | {row.to_phase} | {row.count} | {pct(row.probability)} | {row.replication_count}")

    lines += ["", "16. Information Retention Table", "Model | Top1Accuracy | Top2Accuracy | BrierScore | Entropy"]
    for row in result.retention:
        lines.append(f"{row.model} | {pct(row.top1)} | {pct(row.top2)} | {fmt(row.brier)} | {fmt(row.entropy)}")

    lines += ["", "17. Outcome Diagnostics Table", "CanonicalLoopClass | DRFwd1 | DRFwd3 | DRFwd5 | DRFwd10 | DRFwd20 | ContinuationRate5 | ContinuationRate10 | FailureRate5 | FlatRate5"]
    for row in result.outcomes[:80]:
        lines.append(f"{row.canonical_class} | {fmt(row.dr.get(1, 0.0))} | {fmt(row.dr.get(3, 0.0))} | {fmt(row.dr.get(5, 0.0))} | {fmt(row.dr.get(10, 0.0))} | {fmt(row.dr.get(20, 0.0))} | {pct(row.continuation5)} | {pct(row.continuation10)} | {pct(row.failure5)} | {pct(row.flat5)}")

    rec = result.recommendation
    lines += [
        "",
        "18. Recommendation",
        f"Classification: {rec.classification}",
        f"NeutralAttractorAssessment: {rec.neutral_attractor}",
        f"RoutingFamilies: {rec.routing_families}",
        f"HighEnergyFamilies: {rec.high_energy_families}",
        f"ResetFamilies: {rec.reset_families}",
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
        "No phase from discretionary rules.",
        "No optimization.",
        "No fitting.",
        "No machine learning.",
        "No trading logic.",
        "No forward returns used in loop construction or classification.",
    ]


def rankings(result: Result) -> list[str]:
    top_dest = Counter(destination_family(loop) for loop in complete_loops(result.loops))
    top_return = Counter(return_family(loop) for loop in complete_loops(result.loops))
    energy = sorted(result.market_signature, key=lambda row: row.range_relative + row.volume_rolling + row.efficiency, reverse=True)
    return [
        "",
        "RANKINGS",
        "1. Most common canonical loops: " + "; ".join(f"{row.path}={row.count}" for row in result.canonical_paths[:5]),
        "2. Most replicated canonical loops: " + "; ".join(f"{row.path}=R{row.replication_count}" for row in sorted(result.canonical_paths, key=lambda item: item.replication_count, reverse=True)[:5]),
        "3. Most common failure nodes: " + "; ".join(f"{row.failure_node}={row.count}" for row in result.failure_nodes[:8]),
        "4. Most common destinations: " + "; ".join(f"{key}={value}" for key, value in top_dest.most_common(8)),
        "5. Most common return families: " + "; ".join(f"{key}={value}" for key, value in top_return.most_common(8)),
        "6. Shortest loop classes: " + "; ".join(f"{row.duration_class}={fmt(row.mean_duration)}" for row in sorted(result.duration_classes, key=lambda item: item.mean_duration)),
        "7. Longest loop classes: " + "; ".join(f"{row.duration_class}={fmt(row.mean_duration)}" for row in sorted(result.duration_classes, key=lambda item: item.mean_duration, reverse=True)),
        "8. Strongest loop energy phases: " + "; ".join(f"{row.phase}={fmt(row.range_relative + row.volume_rolling + row.efficiency)}" for row in energy[:5]),
        "9. Best minimal state-machine transitions: " + "; ".join(f"{row.from_phase}->{row.to_phase}={pct(row.probability)}" for row in result.phase_transitions[:8]),
        f"10. Recommended APVA closed-loop model: {result.recommendation.classification}",
    ]


def write_report(result: Result, out_root: Path, aggregate: bool = False) -> None:
    if aggregate:
        path = out_root / "LoopClosure80" / "LoopClosure80_All.txt"
        title = "APVA Loop Closure Study v0.1 - Aggregate"
    else:
        path = out_root / result.instrument / f"LoopClosure80_{result.instrument}.txt"
        title = f"APVA Loop Closure Study v0.1 - {result.instrument}"
    ensure_dir(path.parent)
    lines = [
        title,
        "=" * 96,
        f"Instrument: {result.instrument}",
        f"Input path(s): {', '.join(str(path) for path in result.source_paths)}",
        f"Loop candidates: {len(result.loops)}",
    ]
    append_common(lines, result)
    lines += rankings(result)
    lines += [
        "",
        "RESEARCH NOTES",
        "Mechanical only.",
        "Questions: Is APVA a closed attractor loop? Is Neutral the system attractor? Can the full APVA process be represented as a compact phase machine?",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(result: Result) -> None:
    if not result.loops:
        raise RuntimeError(f"{result.instrument}: no loop candidates.")
    if not result.inventory:
        raise RuntimeError(f"{result.instrument}: loop inventory missing.")
    if not result.phase_transitions:
        raise RuntimeError(f"{result.instrument}: phase transition table missing.")
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
        write_report(result, out_root)
    aggregate_result = build_result("Aggregate", [path for context in contexts for path in context.source_paths], contexts, instrument_results)
    validate(aggregate_result)
    write_report(aggregate_result, out_root, aggregate=True)
    print(f"Wrote {len(instrument_results)} per-instrument Study80 report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
