#!/usr/bin/env python3
"""APVA Neutral Failure Modes Study v0.1.

Study 75 asks what changes before Neutral equilibrium deteriorates.

Research only. No trading, optimization, fitting, machine learning, or forward
returns in failure-mode construction.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from APVA_BranchForecast_65 import build_stream
from APVA_InformationContribution_63 import thresholds
from APVA_InformationDecay_57 import study as decay_study
from APVA_MinimalEngine_52 import instrument_columns, load_results, safe_mean
from APVA_NeutralBackbone_71 import NEUTRAL_NODES, is_neutral_node
from APVA_NodeImportance_58 import aggregate_rows, local_rows, score_rows
from APVA_ProcessGraph_54 import Node, node_for, node_text
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

EPSILON = 1e-12
LOOKBACKS = (1, 2, 3, 5)
OUTCOME_HORIZONS = (1, 3, 5, 10)
MATERIAL_EFFECT = 0.25

AGE4 = ("NeutralProcessing", "4")
AGE5 = ("NeutralProcessing", "5")
AGE6_10 = ("NeutralProcessing", "6-10")
AGE3 = ("NeutralProcessing", "3")

EVENT_TYPES = (
    "A_Age4_to_Age5",
    "B_Age5_to_Age6-10",
    "C_Age6-10_to_Excursion",
)

SIGNALS = (
    "RangeRelativeToPrevious",
    "BodyRelativeToPrevious",
    "TrueRangeRelativeToPrevious",
    "VolumeRelativeToPrevious",
    "VolumeRelativeToRollingMean",
    "VolumeRelativeToSessionMean",
    "EfficiencyRatio",
    "BranchEntropy",
    "MemoryStrength",
    "PolarityFlipFlag",
)


@dataclass
class Event:
    event_type: str
    index: int
    instrument: str


@dataclass
class InventoryRow:
    event_type: str
    count: int
    instrument_counts: Counter[str]
    percent_of_neutral_events: float
    replication_count: int


@dataclass
class ShiftRow:
    event_type: str
    signal: str
    horizon: int
    failure_mean: float
    failure_median: float
    control_mean: float
    control_median: float
    delta_mean: float
    delta_median: float
    effect_size: float
    replication_count: int = 0


@dataclass
class LeadRow:
    event_type: str
    signal: str
    earliest_horizon: str
    max_effect_size: float
    replication_count: int = 0


@dataclass
class ModeRow:
    event_type: str
    failure_mode: str
    count: int
    percent: float
    mean_effect_size: float
    median_effect_size: float
    replication_count: int = 0
    robustness: str = "WeakFailureMode"


@dataclass
class EventSummary:
    event_type: str
    dominant_mode: str
    secondary_mode: str
    top_signals: str
    earliest_signal: str
    earliest_lead_time: str
    replication_count: int


@dataclass
class PolarityRow:
    event_type: str
    flip_rate: float
    control_flip_rate: float
    delta: float
    same_rate: float
    black_to_red_rate: float
    red_to_black_rate: float
    replication_count: int = 0


@dataclass
class DivergenceRow:
    event_type: str
    range_up_volume_up: float
    range_up_volume_down: float
    range_down_volume_up: float
    range_down_volume_down: float


@dataclass
class EfficiencyRow:
    event_type: str
    mean_efficiency: float
    control_efficiency: float
    efficiency_delta: float
    close_extreme_rate: float
    control_close_extreme_rate: float


@dataclass
class EntropyMemoryRow:
    event_type: str
    memory_slope: float
    entropy_slope: float
    memory_weakens_first: bool
    entropy_rises_first: bool


@dataclass
class OutcomeRow:
    failure_mode: str
    dr: dict[int, float] = field(default_factory=dict)
    continuation: dict[int, float] = field(default_factory=dict)
    failure: dict[int, float] = field(default_factory=dict)
    flat: dict[int, float] = field(default_factory=dict)


@dataclass
class Recommendation:
    classification: str
    dominant_process: str
    reliable_signals: str
    rejected_signals: str
    study74_assessment: str
    reason: str


@dataclass
class Result:
    instrument: str
    source_paths: list
    bars: list
    nodes: list[Node]
    stream_rows: list
    event_inventory: dict[str, InventoryRow]
    shifts: list[ShiftRow]
    leads: list[LeadRow]
    modes: list[ModeRow]
    summaries: list[EventSummary]
    polarity: dict[str, PolarityRow]
    divergence: dict[str, DivergenceRow]
    efficiency: dict[str, EfficiencyRow]
    entropy_memory: dict[str, EntropyMemoryRow]
    replication: dict[tuple[str, str], dict[str, float]]
    outcomes: dict[str, OutcomeRow]
    recommendation: Recommendation


def mean(values: Iterable[float | bool | None]) -> float:
    return safe_mean([float(value) for value in values if value is not None])


def median(values: Iterable[float | None]) -> float:
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else 0.0


def stdev(values: Iterable[float | None]) -> float:
    values = [value for value in values if value is not None]
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def pooled_stdev(left: list[float], right: list[float]) -> float:
    return math.sqrt((stdev(left) ** 2 + stdev(right) ** 2) / 2.0)


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def efficiency_ratio(bar) -> float | None:
    if bar.body is None or bar.bar_range is None:
        return None
    return bar.body / max(bar.bar_range, EPSILON)


def polarity_flip(bars: list, index: int) -> float | None:
    if index <= 0:
        return None
    return float((bars[index].volume_polarity or "Neutral") != (bars[index - 1].volume_polarity or "Neutral"))


def directional_forward(bars: list, index: int, horizon: int) -> float | None:
    if index + horizon >= len(bars):
        return None
    c0 = bars[index].close
    c1 = bars[index + horizon].close
    if c0 is None or c1 is None:
        return None
    polarity = (bars[index].volume_polarity or "").strip()
    if polarity == "Black":
        return c1 - c0
    if polarity == "Red":
        return c0 - c1
    return None


def branch_entropy(stream_rows: list) -> dict[Node, float]:
    grouped: dict[Node, Counter[str]] = defaultdict(Counter)
    for row in stream_rows:
        grouped[row.node][row.next_node] += 1
    output = {}
    for node, counts in grouped.items():
        total = sum(counts.values())
        probs = [count / total for count in counts.values()] if total else []
        output[node] = -sum(prob * math.log(prob) for prob in probs if prob > 0)
    return output


def node_metrics(node_rows: dict[Node, object], stream_rows: list) -> tuple[dict[Node, float], dict[Node, float]]:
    entropy = branch_entropy(stream_rows)
    memory = {node: getattr(row, "memory_strength", 0.0) for node, row in node_rows.items()}
    return memory, entropy


def signal_value(signal: str, bars: list, nodes: list[Node], index: int,
                 memory: dict[Node, float], entropy: dict[Node, float]) -> float | None:
    if index < 0 or index >= len(bars):
        return None
    bar = bars[index]
    node = nodes[index]
    mapping = {
        "RangeRelativeToPrevious": bar.range_relative_to_previous,
        "BodyRelativeToPrevious": bar.body_relative_to_previous,
        "TrueRangeRelativeToPrevious": bar.true_range_relative_to_previous,
        "VolumeRelativeToPrevious": bar.volume_relative_to_previous,
        "VolumeRelativeToRollingMean": bar.volume_relative_to_rolling_mean,
        "VolumeRelativeToSessionMean": bar.volume_relative_to_session_mean,
        "EfficiencyRatio": efficiency_ratio(bar),
        "BranchEntropy": entropy.get(node, 0.0),
        "MemoryStrength": memory.get(node, 0.0),
        "PolarityFlipFlag": polarity_flip(bars, index),
    }
    return mapping[signal]


def event_type_for(nodes: list[Node], index: int) -> str | None:
    if index <= 0:
        return None
    previous, current = nodes[index - 1], nodes[index]
    if previous == AGE4 and current == AGE5:
        return "A_Age4_to_Age5"
    if previous == AGE5 and current == AGE6_10:
        return "B_Age5_to_Age6-10"
    if previous == AGE6_10 and not is_neutral_node(current):
        return "C_Age6-10_to_Excursion"
    return None


def find_events(nodes: list[Node], instrument: str) -> list[Event]:
    events = []
    for index in range(1, len(nodes)):
        event_type = event_type_for(nodes, index)
        if event_type:
            events.append(Event(event_type, index, instrument))
    return events


def find_controls(nodes: list[Node], instrument: str) -> list[Event]:
    controls = []
    for index in range(1, len(nodes)):
        previous, current = nodes[index - 1], nodes[index]
        if previous == AGE3 and current == AGE4:
            controls.append(Event("ControlStableNeutral", index, instrument))
            continue
        if previous == AGE4 and current == AGE4:
            controls.append(Event("ControlStableNeutral", index, instrument))
            continue
        no_near_failure = True
        if current == AGE4:
            for step in (1, 2, 3):
                if index + step < len(nodes) and (nodes[index + step] == AGE5 or not is_neutral_node(nodes[index + step])):
                    no_near_failure = False
                    break
        if current == AGE4 and no_near_failure:
            controls.append(Event("ControlStableNeutral", index, instrument))
    return controls


def samples_for(events: list[Event], bars: list, nodes: list[Node], memory: dict[Node, float],
                entropy: dict[Node, float]) -> dict[tuple[str, int], list[float]]:
    samples: dict[tuple[str, int], list[float]] = defaultdict(list)
    for event in events:
        for horizon in LOOKBACKS:
            index = event.index - horizon
            for signal in SIGNALS:
                value = signal_value(signal, bars, nodes, index, memory, entropy)
                if value is not None:
                    samples[(signal, horizon)].append(value)
    return samples


def shift_rows(events_by_type: dict[str, list[Event]], controls: list[Event], bars: list, nodes: list[Node],
               memory: dict[Node, float], entropy: dict[Node, float]) -> list[ShiftRow]:
    control_samples = samples_for(controls, bars, nodes, memory, entropy)
    rows = []
    for event_type in EVENT_TYPES:
        failure_samples = samples_for(events_by_type[event_type], bars, nodes, memory, entropy)
        for signal in SIGNALS:
            for horizon in LOOKBACKS:
                fail_values = failure_samples[(signal, horizon)]
                control_values = control_samples[(signal, horizon)]
                failure_mean = mean(fail_values)
                control_mean = mean(control_values)
                spread = pooled_stdev(fail_values, control_values)
                effect = (failure_mean - control_mean) / spread if spread else 0.0
                rows.append(ShiftRow(
                    event_type, signal, horizon, failure_mean, median(fail_values), control_mean,
                    median(control_values), failure_mean - control_mean,
                    median(fail_values) - median(control_values), effect,
                ))
    return rows


def lead_rows(shifts: list[ShiftRow]) -> list[LeadRow]:
    rows = []
    grouped: dict[tuple[str, str], list[ShiftRow]] = defaultdict(list)
    for row in shifts:
        grouped[(row.event_type, row.signal)].append(row)
    for (event_type, signal), values in grouped.items():
        material = [row for row in values if abs(row.effect_size) >= MATERIAL_EFFECT]
        if material:
            earliest = max(row.horizon for row in material)
            earliest_text = f"t-{earliest}"
        else:
            earliest_text = "None"
        max_effect = max((abs(row.effect_size) for row in values), default=0.0)
        rows.append(LeadRow(event_type, signal, earliest_text, max_effect))
    return rows


def signal_mode(signal: str, effect: float) -> str:
    if signal == "RangeRelativeToPrevious" and effect > 0:
        return "RangeFailure"
    if signal == "BodyRelativeToPrevious" and effect > 0:
        return "BodyFailure"
    if signal == "TrueRangeRelativeToPrevious" and effect > 0:
        return "TrueRangeFailure"
    if signal in {"VolumeRelativeToPrevious", "VolumeRelativeToRollingMean", "VolumeRelativeToSessionMean"}:
        return "VolumeExpansionFailure" if effect > 0 else "VolumeContractionFailure"
    if signal == "EfficiencyRatio" and effect > 0:
        return "EfficiencyFailure"
    if signal == "BranchEntropy" and effect > 0:
        return "EntropyFailure"
    if signal == "MemoryStrength" and effect < 0:
        return "MemoryFailure"
    if signal == "PolarityFlipFlag" and effect > 0:
        return "PolarityInstabilityFailure"
    return "MixedFailure"


def mode_rows(shifts: list[ShiftRow], events_by_type: dict[str, list[Event]]) -> list[ModeRow]:
    rows = []
    by_event: dict[str, list[ShiftRow]] = defaultdict(list)
    for row in shifts:
        if row.horizon == 1:
            by_event[row.event_type].append(row)
    for event_type in EVENT_TYPES:
        ranked = sorted(by_event[event_type], key=lambda row: abs(row.effect_size), reverse=True)
        if not ranked:
            continue
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        mode = signal_mode(top.signal, top.effect_size)
        if second and abs(abs(top.effect_size) - abs(second.effect_size)) <= 0.10 * max(abs(top.effect_size), EPSILON):
            mode = "MixedFailure"
        count = len(events_by_type[event_type])
        rows.append(ModeRow(
            event_type,
            mode,
            count,
            1.0 if count else 0.0,
            abs(top.effect_size),
            abs(top.effect_size),
        ))
    return rows


def event_summaries(modes: list[ModeRow], leads: list[LeadRow], shifts: list[ShiftRow]) -> list[EventSummary]:
    rows = []
    for event_type in EVENT_TYPES:
        event_modes = [row for row in modes if row.event_type == event_type]
        dominant = event_modes[0].failure_mode if event_modes else "N/A"
        secondary = "N/A"
        top_shift = sorted([row for row in shifts if row.event_type == event_type], key=lambda row: abs(row.effect_size), reverse=True)
        top_signals = ", ".join(f"{row.signal}:{fmt(row.effect_size)}" for row in top_shift[:3])
        material_leads = [row for row in leads if row.event_type == event_type and row.earliest_horizon != "None"]
        earliest = sorted(material_leads, key=lambda row: (-int(row.earliest_horizon.split("-")[1]), -row.max_effect_size))[0] if material_leads else None
        rows.append(EventSummary(
            event_type,
            dominant,
            secondary,
            top_signals,
            earliest.signal if earliest else "None",
            earliest.earliest_horizon if earliest else "None",
            0,
        ))
    return rows


def polarity_rows(events_by_type: dict[str, list[Event]], controls: list[Event], bars: list) -> dict[str, PolarityRow]:
    control_flips = [polarity_flip(bars, event.index - 1) for event in controls]
    control_flip_rate = mean(control_flips)
    output = {}
    for event_type, events in events_by_type.items():
        flips, same, black_red, red_black = [], [], [], []
        for event in events:
            index = event.index - 1
            if index <= 0:
                continue
            prev = bars[index - 1].volume_polarity or "Neutral"
            cur = bars[index].volume_polarity or "Neutral"
            flips.append(cur != prev)
            same.append(cur == prev)
            black_red.append(prev == "Black" and cur == "Red")
            red_black.append(prev == "Red" and cur == "Black")
        output[event_type] = PolarityRow(event_type, mean(flips), control_flip_rate, mean(flips) - control_flip_rate, mean(same), mean(black_red), mean(red_black))
    return output


def divergence_rows(events_by_type: dict[str, list[Event]], bars: list) -> dict[str, DivergenceRow]:
    output = {}
    for event_type, events in events_by_type.items():
        uu, ud, du, dd = [], [], [], []
        for event in events:
            bar = bars[event.index - 1] if event.index > 0 else None
            if not bar:
                continue
            range_up = bar.range_expansion_flag is True
            range_down = bar.range_contraction_flag is True
            volume_up = bar.volume_expansion_flag is True
            volume_down = bar.volume_contraction_flag is True
            uu.append(range_up and volume_up)
            ud.append(range_up and volume_down)
            du.append(range_down and volume_up)
            dd.append(range_down and volume_down)
        output[event_type] = DivergenceRow(event_type, mean(uu), mean(ud), mean(du), mean(dd))
    return output


def efficiency_rows(events_by_type: dict[str, list[Event]], controls: list[Event], bars: list) -> dict[str, EfficiencyRow]:
    control_eff = [efficiency_ratio(bars[event.index - 1]) for event in controls if event.index > 0]
    control_extreme = [
        bars[event.index - 1].close_location <= 0.20 or bars[event.index - 1].close_location >= 0.80
        for event in controls if event.index > 0 and bars[event.index - 1].close_location is not None
    ]
    output = {}
    for event_type, events in events_by_type.items():
        eff, extreme = [], []
        for event in events:
            if event.index <= 0:
                continue
            bar = bars[event.index - 1]
            eff.append(efficiency_ratio(bar))
            if bar.close_location is not None:
                extreme.append(bar.close_location <= 0.20 or bar.close_location >= 0.80)
        output[event_type] = EfficiencyRow(event_type, mean(eff), mean(control_eff), mean(eff) - mean(control_eff), mean(extreme), mean(control_extreme))
    return output


def slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    xs = list(range(len(values)))
    mx, my = mean(xs), mean(values)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, values)) / den if den else 0.0


def entropy_memory_rows(events_by_type: dict[str, list[Event]], bars: list, nodes: list[Node],
                        memory: dict[Node, float], entropy: dict[Node, float]) -> dict[str, EntropyMemoryRow]:
    output = {}
    for event_type, events in events_by_type.items():
        memory_slopes, entropy_slopes = [], []
        for event in events:
            idxs = [event.index - horizon for horizon in (5, 3, 2, 1, 0) if 0 <= event.index - horizon < len(nodes)]
            memory_values = [memory.get(nodes[index], 0.0) for index in idxs]
            entropy_values = [entropy.get(nodes[index], 0.0) for index in idxs]
            memory_slopes.append(slope(memory_values))
            entropy_slopes.append(slope(entropy_values))
        ms, es = mean(memory_slopes), mean(entropy_slopes)
        output[event_type] = EntropyMemoryRow(event_type, ms, es, ms < 0 and abs(ms) >= abs(es), es > 0 and abs(es) > abs(ms))
    return output


def classify_event_mode(event: Event, bars: list, nodes: list[Node], memory: dict[Node, float],
                        entropy: dict[Node, float], control_means: dict[str, float],
                        control_stds: dict[str, float]) -> str:
    scored = []
    index = event.index - 1
    for signal in SIGNALS:
        value = signal_value(signal, bars, nodes, index, memory, entropy)
        if value is None:
            continue
        spread = control_stds.get(signal, 0.0)
        effect = (value - control_means.get(signal, 0.0)) / spread if spread else 0.0
        scored.append((signal, effect))
    if not scored:
        return "MixedFailure"
    scored.sort(key=lambda item: abs(item[1]), reverse=True)
    top = scored[0]
    if len(scored) > 1 and abs(abs(top[1]) - abs(scored[1][1])) <= 0.10 * max(abs(top[1]), EPSILON):
        return "MixedFailure"
    return signal_mode(top[0], top[1])


def outcome_rows_by_mode(events_by_type: dict[str, list[Event]], bars: list, modes_by_event: dict[tuple[str, int], str]) -> dict[str, OutcomeRow]:
    grouped: dict[str, list[Event]] = defaultdict(list)
    for events in events_by_type.values():
        for event in events:
            grouped[modes_by_event.get((event.event_type, event.index), "MixedFailure")].append(event)
    output = {}
    for mode, events in grouped.items():
        values_by_horizon = {
            horizon: [value for event in events if (value := directional_forward(bars, event.index, horizon)) is not None]
            for horizon in OUTCOME_HORIZONS
        }
        output[mode] = OutcomeRow(
            mode,
            {h: mean(v) for h, v in values_by_horizon.items()},
            {h: mean(value > 0 for value in v) for h, v in values_by_horizon.items()},
            {h: mean(value < 0 for value in v) for h, v in values_by_horizon.items()},
            {h: mean(value == 0 for value in v) for h, v in values_by_horizon.items()},
        )
    return output


def build_result(instrument: str, source_paths: list, bars: list, nodes: list[Node],
                 stream_rows: list, node_rows: dict[Node, object]) -> Result:
    memory, entropy = node_metrics(node_rows, stream_rows)
    events = find_events(nodes, instrument)
    controls = find_controls(nodes, instrument)
    events_by_type = {event_type: [event for event in events if event.event_type == event_type] for event_type in EVENT_TYPES}
    neutral_event_count = max(len(events) + len(controls), 1)
    inventory = {
        event_type: InventoryRow(
            event_type,
            len(events_by_type[event_type]),
            Counter(event.instrument for event in events_by_type[event_type]),
            len(events_by_type[event_type]) / neutral_event_count,
            1 if events_by_type[event_type] else 0,
        )
        for event_type in EVENT_TYPES
    }
    shifts = shift_rows(events_by_type, controls, bars, nodes, memory, entropy)
    leads = lead_rows(shifts)
    modes = mode_rows(shifts, events_by_type)
    summaries = event_summaries(modes, leads, shifts)
    polarity = polarity_rows(events_by_type, controls, bars)
    divergence = divergence_rows(events_by_type, bars)
    efficiency = efficiency_rows(events_by_type, controls, bars)
    entropy_memory = entropy_memory_rows(events_by_type, bars, nodes, memory, entropy)
    control_samples = samples_for(controls, bars, nodes, memory, entropy)
    control_means = {signal: mean(control_samples[(signal, 1)]) for signal in SIGNALS}
    control_stds = {signal: stdev(control_samples[(signal, 1)]) for signal in SIGNALS}
    modes_by_event = {
        (event.event_type, event.index): classify_event_mode(event, bars, nodes, memory, entropy, control_means, control_stds)
        for event in events
    }
    mode_counts = Counter(modes_by_event.values())
    total_events = max(sum(mode_counts.values()), 1)
    mode_effects = {row.failure_mode: row.mean_effect_size for row in modes}
    detailed_modes = [
        ModeRow("AllEvents", mode, count, count / total_events, mode_effects.get(mode, 0.0), mode_effects.get(mode, 0.0))
        for mode, count in mode_counts.items()
    ]
    modes.extend(detailed_modes)
    outcomes = outcome_rows_by_mode(events_by_type, bars, modes_by_event)
    replication = {(row.event_type, row.signal): {} for row in shifts}
    recommendation = make_recommendation(modes, shifts, leads)
    return Result(
        instrument, source_paths, bars, nodes, stream_rows, inventory, shifts, leads, modes, summaries,
        polarity, divergence, efficiency, entropy_memory, replication, outcomes, recommendation,
    )


def make_recommendation(modes: list[ModeRow], shifts: list[ShiftRow], leads: list[LeadRow]) -> Recommendation:
    robust_modes = [row for row in modes if row.robustness == "RobustFailureMode"]
    top_shift = max(shifts, key=lambda row: abs(row.effect_size), default=None)
    reliable = sorted({row.signal for row in shifts if abs(row.effect_size) >= MATERIAL_EFFECT})
    rejected = sorted({row.signal for row in shifts if all(abs(s.effect_size) < MATERIAL_EFFECT for s in shifts if s.signal == row.signal)})
    if robust_modes:
        dominant = robust_modes[0].failure_mode
    elif top_shift:
        dominant = signal_mode(top_shift.signal, top_shift.effect_size)
    else:
        dominant = "NoReliableFailureMode"
    if "Volume" in dominant:
        classification = "VolumeDrivenFailure"
    elif "Range" in dominant or "Body" in dominant or "TrueRange" in dominant:
        classification = "RangeDrivenFailure"
    elif "Efficiency" in dominant:
        classification = "EfficiencyDrivenFailure"
    elif dominant in {"EntropyFailure", "MemoryFailure"}:
        classification = "EntropyMemoryFailure"
    elif dominant == "MixedFailure":
        classification = "MixedFailureProcess"
    else:
        classification = "NoReliableFailureMode"
    if len(reliable) >= 3:
        classification = "MixedFailureProcess"
    return Recommendation(
        classification,
        dominant,
        ", ".join(reliable) if reliable else "None",
        ", ".join(rejected) if rejected else "None",
        "Strengthened: Age4 is the reference equilibrium, Age5/6-10 show measurable deterioration windows." if reliable else "Weakened: no material pre-failure signal replicated mechanically.",
        "Classification is based only on fixed pre-event OHLCV, entropy, memory, and polarity shifts versus stable Neutral controls.",
    )


def attach_replication(aggregate: Result, instruments: list[Result]) -> None:
    for event_type, row in aggregate.event_inventory.items():
        row.instrument_counts = Counter({result.instrument: result.event_inventory[event_type].count for result in instruments})
        row.replication_count = sum(result.event_inventory[event_type].count > 0 for result in instruments)
    effect_by_inst = {
        result.instrument: {(row.event_type, row.signal): row.effect_size for row in result.shifts if row.horizon == 1}
        for result in instruments
    }
    for row in aggregate.shifts:
        if row.horizon != 1:
            continue
        effects = {inst: values.get((row.event_type, row.signal), 0.0) for inst, values in effect_by_inst.items()}
        row.replication_count = sum(
            math.copysign(1, value) == math.copysign(1, row.effect_size) and abs(value) >= MATERIAL_EFFECT
            for value in effects.values()
            if value != 0 and row.effect_size != 0
        )
        aggregate.replication[(row.event_type, row.signal)] = effects
    replicated_by_mode = Counter()
    for row in aggregate.shifts:
        if row.horizon == 1 and row.replication_count >= 2:
            replicated_by_mode[signal_mode(row.signal, row.effect_size)] += 1
    for mode in aggregate.modes:
        mode.replication_count = replicated_by_mode[mode.failure_mode]
        mode.robustness = "RobustFailureMode" if mode.replication_count >= 2 and mode.mean_effect_size >= MATERIAL_EFFECT else "WeakFailureMode"
    for summary in aggregate.summaries:
        relevant = [row for row in aggregate.shifts if row.event_type == summary.event_type and row.signal == summary.earliest_signal and row.horizon == 1]
        summary.replication_count = relevant[0].replication_count if relevant else 0
    for row in aggregate.polarity.values():
        matching = [shift for shift in aggregate.shifts if shift.event_type == row.event_type and shift.signal == "PolarityFlipFlag" and shift.horizon == 1]
        row.replication_count = matching[0].replication_count if matching else 0
    aggregate.recommendation = make_recommendation(aggregate.modes, aggregate.shifts, aggregate.leads)


def inventory_line(row: InventoryRow) -> str:
    return f"{row.event_type} | {row.count} | {row.instrument_counts.get('6E', 0)} | {row.instrument_counts.get('CL', 0)} | {row.instrument_counts.get('NQ', 0)} | {pct(row.percent_of_neutral_events)} | {row.replication_count}"


def append_common(lines: list[str], result: Result, aggregate: bool = False) -> None:
    lines += ["", "1. Failure Event Inventory", "EventType | Count | Count_6E | Count_CL | Count_NQ | PercentOfNeutralEvents | ReplicationCount"]
    lines += [inventory_line(row) for row in result.event_inventory.values()]
    lines += ["", "2. Signal Shift Table", "EventType | Signal | Horizon | FailureMean | ControlMean | DeltaMean | EffectSize | ReplicationCount"]
    for row in sorted(result.shifts, key=lambda row: (row.event_type, row.signal, row.horizon)):
        lines.append(f"{row.event_type} | {row.signal} | t-{row.horizon} | {fmt(row.failure_mean)} | {fmt(row.control_mean)} | {fmt(row.delta_mean)} | {fmt(row.effect_size)} | {row.replication_count}")
    lines += ["", "3. Lead-Time Ranking", "EventType | Signal | EarliestMaterialHorizon | MaxEffectSize | ReplicationCount"]
    for row in sorted(result.leads, key=lambda row: (row.event_type, row.earliest_horizon == "None", row.earliest_horizon, -row.max_effect_size)):
        lines.append(f"{row.event_type} | {row.signal} | {row.earliest_horizon} | {fmt(row.max_effect_size)} | {row.replication_count}")
    lines += ["", "4. Failure Mode Table", "EventType | FailureMode | Count | Percent | MeanEffectSize | ReplicationCount | RobustnessClass"]
    for row in result.modes:
        lines.append(f"{row.event_type} | {row.failure_mode} | {row.count} | {pct(row.percent)} | {fmt(row.mean_effect_size)} | {row.replication_count} | {row.robustness}")
    lines += ["", "5. Event-Type Failure Summary", "EventType | DominantFailureMode | SecondaryFailureMode | EarliestSignal | EarliestLeadTime | ReplicationCount | TopSignals"]
    for row in result.summaries:
        lines.append(f"{row.event_type} | {row.dominant_mode} | {row.secondary_mode} | {row.earliest_signal} | {row.earliest_lead_time} | {row.replication_count} | {row.top_signals}")
    lines += ["", "6. Polarity Instability Table", "EventType | PolarityFlipRate | ControlPolarityFlipRate | Delta | SamePolarityRate | BlackToRedRate | RedToBlackRate | ReplicationCount"]
    for row in result.polarity.values():
        lines.append(f"{row.event_type} | {pct(row.flip_rate)} | {pct(row.control_flip_rate)} | {pct(row.delta)} | {pct(row.same_rate)} | {pct(row.black_to_red_rate)} | {pct(row.red_to_black_rate)} | {row.replication_count}")
    lines += ["", "7. Range-Volume Divergence Table", "EventType | RangeUpVolumeUp | RangeUpVolumeDown | RangeDownVolumeUp | RangeDownVolumeDown"]
    for row in result.divergence.values():
        lines.append(f"{row.event_type} | {pct(row.range_up_volume_up)} | {pct(row.range_up_volume_down)} | {pct(row.range_down_volume_up)} | {pct(row.range_down_volume_down)}")
    lines += ["", "8. Efficiency Breakdown Table", "EventType | MeanEfficiency | ControlEfficiency | EfficiencyDelta | CloseExtremeRate | ControlCloseExtremeRate"]
    for row in result.efficiency.values():
        lines.append(f"{row.event_type} | {fmt(row.mean_efficiency)} | {fmt(row.control_efficiency)} | {fmt(row.efficiency_delta)} | {pct(row.close_extreme_rate)} | {pct(row.control_close_extreme_rate)}")
    lines += ["", "9. Entropy-Memory Sequence Table", "EventType | MemorySlope | EntropySlope | MemoryWeakensFirst | EntropyRisesFirst"]
    for row in result.entropy_memory.values():
        lines.append(f"{row.event_type} | {fmt(row.memory_slope)} | {fmt(row.entropy_slope)} | {row.memory_weakens_first} | {row.entropy_rises_first}")
    if aggregate:
        lines += ["", "10. Cross-Instrument Replication Table", "EventType | Signal | Effect_6E | Effect_CL | Effect_NQ | ReplicationCount"]
        for (event_type, signal), effects in sorted(result.replication.items()):
            shift = next((row for row in result.shifts if row.event_type == event_type and row.signal == signal and row.horizon == 1), None)
            lines.append(f"{event_type} | {signal} | {fmt(effects.get('6E', 0.0))} | {fmt(effects.get('CL', 0.0))} | {fmt(effects.get('NQ', 0.0))} | {shift.replication_count if shift else 0}")
    lines += ["", "11. Outcome Diagnostics Table", "FailureMode | DRFwd1 | DRFwd3 | DRFwd5 | DRFwd10 | ContinuationRate5 | FailureRate5 | FlatRate5"]
    for row in result.outcomes.values():
        lines.append(f"{row.failure_mode} | {fmt(row.dr.get(1, 0.0))} | {fmt(row.dr.get(3, 0.0))} | {fmt(row.dr.get(5, 0.0))} | {fmt(row.dr.get(10, 0.0))} | {pct(row.continuation.get(5, 0.0))} | {pct(row.failure.get(5, 0.0))} | {pct(row.flat.get(5, 0.0))}")
    lines += ["", "12. Recommendation"]
    rec = result.recommendation
    lines += [
        f"Classification: {rec.classification}",
        f"DominantFailureProcess: {rec.dominant_process}",
        f"ReliableSignals: {rec.reliable_signals}",
        f"RejectedSignals: {rec.rejected_signals}",
        f"Study74Assessment: {rec.study74_assessment}",
        f"Reason: {rec.reason}",
    ]
    lines += rankings(result)
    lines += [
        "",
        "16. Comparison With Study 74",
        "Study74Interpretation: Age4 = Maximum Reliable Equilibrium; Age5 = Late/Destabilizing Equilibrium; Age6-10 = Stale Equilibrium.",
        f"Assessment: {rec.study74_assessment}",
        "",
        "17. Low-DoF Audit",
        "Uses only existing APVA state variables and real OHLCV-derived metrics.",
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
        "No forward returns used in failure-mode construction.",
        "",
        "RESEARCH NOTES",
        "Mechanical only.",
        "What breaks first when Neutral fails?",
        "Is failure range-driven, volume-driven, entropy-driven, or mixed?",
        "Does the failure process replicate across 6E, CL, and NQ?",
        "Does this provide a bridge from APVA state structure to real-time market observation?",
    ]


def rankings(result: Result) -> list[str]:
    material = [row for row in result.shifts if row.horizon == 1]
    lines = ["", "RANKINGS"]
    lines += ["", "1. Strongest early failure signals"] + [f"{row.event_type} | {row.signal} | {fmt(row.effect_size)}" for row in sorted(material, key=lambda row: -abs(row.effect_size))[:20]]
    lines += ["", "2. Most replicated failure signals"] + [f"{row.event_type} | {row.signal} | Replication={row.replication_count} | Effect={fmt(row.effect_size)}" for row in sorted(material, key=lambda row: (-row.replication_count, -abs(row.effect_size)))[:20]]
    for number, event_type in ((3, "A_Age4_to_Age5"), (4, "B_Age5_to_Age6-10"), (5, "C_Age6-10_to_Excursion")):
        lines += ["", f"{number}. Strongest {event_type} signals"] + [f"{row.signal} | {fmt(row.effect_size)}" for row in sorted([row for row in material if row.event_type == event_type], key=lambda row: -abs(row.effect_size))[:10]]
    lines += ["", "6. Most robust failure modes"] + [f"{row.failure_mode} | {row.robustness} | {fmt(row.mean_effect_size)}" for row in sorted(result.modes, key=lambda row: (-row.replication_count, -row.mean_effect_size))]
    lines += ["", "7. Weakest failure modes"] + [f"{row.failure_mode} | {row.robustness} | {fmt(row.mean_effect_size)}" for row in sorted(result.modes, key=lambda row: (row.mean_effect_size, row.failure_mode))]
    lines += ["", "8. Strongest polarity instability events"] + [f"{row.event_type} | Delta={pct(row.delta)}" for row in sorted(result.polarity.values(), key=lambda row: -abs(row.delta))]
    lines += ["", "9. Strongest range-volume divergence events"] + [f"{row.event_type} | RangeUpVolumeDown={pct(row.range_up_volume_down)}" for row in sorted(result.divergence.values(), key=lambda row: -row.range_up_volume_down)]
    lines += ["", "10. Recommended Neutral failure model", f"{result.recommendation.classification}: {result.recommendation.dominant_process}"]
    return lines


def write_per_instrument(result: Result, out_root: Path) -> None:
    path = out_root / result.instrument / f"NeutralFailureModes75_{result.instrument}.txt"
    ensure_dir(path.parent)
    lines = [
        "APVA Neutral Failure Modes Study v0.1",
        "=" * 96,
        f"Instrument: {result.instrument}",
        "Input path(s): " + ", ".join(str(path) for path in result.source_paths),
        f"Total rows: {len(result.bars)}",
    ]
    append_common(lines, result)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aggregate(result: Result, instruments: list[Result], out_root: Path) -> None:
    attach_replication(result, instruments)
    path = out_root / "NeutralFailureModes75" / "NeutralFailureModes75_All.txt"
    ensure_dir(path.parent)
    lines = [
        "APVA Neutral Failure Modes Study v0.1 - Aggregate",
        "=" * 96,
        "Instruments: " + ", ".join(instrument_columns(instruments)),
    ]
    append_common(lines, result, aggregate=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(result: Result) -> None:
    if not result.bars:
        raise RuntimeError(f"{result.instrument}: no bars.")
    if not any(getattr(bar, "has_ohlc", False) and getattr(bar, "has_volume", False) for bar in result.bars):
        raise RuntimeError(f"{result.instrument}: Study75 requires upgraded OHLCV loader fields.")
    if not result.shifts:
        raise RuntimeError(f"{result.instrument}: no signal shift rows.")
    if any(math.isnan(row.effect_size) for row in result.shifts):
        raise RuntimeError(f"{result.instrument}: invalid effect size.")
    for event_type in EVENT_TYPES:
        if event_type not in result.event_inventory:
            raise RuntimeError(f"{result.instrument}: missing event inventory {event_type}.")


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
    aggregate_node_rows = aggregate_rows(decay)
    score_rows(aggregate_node_rows)
    memory_thresholds = thresholds(loaded, aggregate_node_rows)
    instrument_results = []
    aggregate_bars, aggregate_nodes, aggregate_stream, aggregate_paths = [], [], [], []
    for loaded_row, decay_row in zip(loaded, decay):
        local_node_rows = local_rows(decay_row)
        score_rows(local_node_rows)
        stream = build_stream(loaded_row, local_node_rows, memory_thresholds, (0.0, 0.0), (0.0, 0.0))
        nodes = [node_for(bar) for bar in loaded_row.bars]
        result = build_result(loaded_row.instrument, loaded_row.source_paths, loaded_row.bars, nodes, stream, local_node_rows)
        validate(result)
        instrument_results.append(result)
        offset = len(aggregate_bars)
        aggregate_stream.extend(build_stream(loaded_row, aggregate_node_rows, memory_thresholds, (0.0, 0.0), (0.0, 0.0), offset))
        aggregate_nodes.extend(nodes)
        aggregate_bars.extend(loaded_row.bars)
        aggregate_paths.extend(loaded_row.source_paths)
    aggregate = build_result("Aggregate", aggregate_paths, aggregate_bars, aggregate_nodes, aggregate_stream, aggregate_node_rows)
    validate(aggregate)
    out_root = Path(args.out_root)
    for result in instrument_results:
        write_per_instrument(result, out_root)
    write_aggregate(aggregate, instrument_results, out_root)
    print(f"Wrote {len(instrument_results)} per-instrument Study75 report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
