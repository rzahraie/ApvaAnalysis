#!/usr/bin/env python3
"""APVA Excursion Destinations Study v0.1.

Study 76 asks whether the destination family of a Neutral_Age6-10 excursion
can be distinguished before the excursion begins.

Research only. No trading, optimization, fitting, machine learning, or forward
returns in destination classification.
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
from APVA_NeutralFailureModes_75 import (
    AGE6_10,
    LOOKBACKS,
    MATERIAL_EFFECT,
    OUTCOME_HORIZONS,
    branch_entropy,
    directional_forward,
    efficiency_ratio,
    mean,
    median,
    node_metrics,
    polarity_flip,
    pooled_stdev,
    safe_div,
    signal_value,
    slope,
    stdev,
)
from APVA_NodeImportance_58 import aggregate_rows, local_rows, score_rows
from APVA_ProcessGraph_54 import Node, node_for, node_text
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

DESTINATION_FAMILIES = (
    "CompressionProcessing",
    "MixedStructure",
    "ReassertionProcessing",
    "RecoveryResolution",
    "ExhaustionPersistence",
    "ConstructiveEmergence",
    "DestructiveRotation",
    "DecayToNeutral",
)

SIGNALS = (
    "RangeRelativeToPrevious",
    "BodyRelativeToPrevious",
    "TrueRangeRelativeToPrevious",
    "VolumeRelativeToPrevious",
    "VolumeRelativeToRollingMean",
    "VolumeRelativeToSessionMean",
    "EfficiencyRatio",
    "CloseLocation",
    "MemoryStrength",
    "BranchEntropy",
    "PolarityFlipFlag",
)

PAIRWISE_TESTS = (
    ("CompressionProcessing", "MixedStructure"),
    ("CompressionProcessing", "ReassertionProcessing"),
    ("CompressionProcessing", "RecoveryResolution"),
    ("MixedStructure", "ReassertionProcessing"),
    ("MixedStructure", "RecoveryResolution"),
    ("ReassertionProcessing", "RecoveryResolution"),
    ("ConstructiveEmergence", "DestructiveRotation"),
    ("ExhaustionPersistence", "DestructiveRotation"),
    ("DecayToNeutral", "MixedStructure"),
)


@dataclass
class DestinationEvent:
    index: int
    instrument: str
    destination: str


@dataclass
class InventoryRow:
    destination: str
    count: int
    instrument_counts: Counter[str]
    probability: float
    replication_count: int


@dataclass
class ProfileRow:
    destination: str
    signal: str
    horizon: int
    mean_value: float
    median_value: float
    stddev_value: float


@dataclass
class SeparationRow:
    destination: str
    signal: str
    horizon: int
    family_mean: float
    other_mean: float
    delta_mean: float
    effect_size: float
    replication_count: int = 0


@dataclass
class PairwiseRow:
    family_a: str
    family_b: str
    signal: str
    horizon: int
    effect_size: float
    replication_count: int = 0


@dataclass
class SignatureRow:
    destination: str
    label: str
    dominant_signal: str
    dominant_horizon: str
    effect_size: float
    replication_count: int


@dataclass
class SharedSignalRow:
    signal: str
    horizon: int
    mean_effect: float
    shared_replication_count: int


@dataclass
class UniqueSignalRow:
    destination: str
    signal: str
    horizon: int
    effect_size: float
    replication_count: int


@dataclass
class TimingRow:
    destination: str
    earliest_signal: str
    earliest_horizon: str
    max_effect_size: float
    replication_count: int


@dataclass
class RangeVolumeRow:
    destination: str
    range_up_volume_up: float
    range_up_volume_down: float
    range_down_volume_up: float
    range_down_volume_down: float


@dataclass
class EfficiencyCloseRow:
    destination: str
    mean_efficiency: float
    median_efficiency: float
    close_near_high_rate: float
    close_near_low_rate: float
    middle_close_rate: float


@dataclass
class PolarityRow:
    destination: str
    black_rate: float
    red_rate: float
    flip_rate: float
    same_rate: float
    black_to_red_rate: float
    red_to_black_rate: float


@dataclass
class MemoryEntropyRow:
    destination: str
    memory_slope: float
    entropy_slope: float
    memory_weakening_rate: float
    entropy_rising_rate: float


@dataclass
class ReplicationRow:
    destination: str
    signal: str
    horizon: int
    effects: dict[str, float] = field(default_factory=dict)
    replication_count: int = 0


@dataclass
class PredictabilityRow:
    destination: str
    score: float
    classification: str
    replicated_signal_count: int
    earliest_lead_time: str


@dataclass
class OutcomeRow:
    destination: str
    dr: dict[int, float] = field(default_factory=dict)
    continuation: dict[int, float] = field(default_factory=dict)
    failure: dict[int, float] = field(default_factory=dict)
    flat: dict[int, float] = field(default_factory=dict)


@dataclass
class Recommendation:
    classification: str
    predictable_destinations: str
    unpredictable_destinations: str
    shared_failure_signals: str
    unique_destination_signals: str
    study75_assessment: str
    reason: str


@dataclass
class Result:
    instrument: str
    source_paths: list
    bars: list
    nodes: list[Node]
    stream_rows: list
    events_by_destination: dict[str, list[DestinationEvent]]
    inventory: dict[str, InventoryRow]
    profiles: list[ProfileRow]
    separation: list[SeparationRow]
    pairwise: list[PairwiseRow]
    signatures: list[SignatureRow]
    shared_signals: list[SharedSignalRow]
    unique_signals: list[UniqueSignalRow]
    timing: list[TimingRow]
    range_volume: dict[str, RangeVolumeRow]
    efficiency_close: dict[str, EfficiencyCloseRow]
    polarity: dict[str, PolarityRow]
    memory_entropy: dict[str, MemoryEntropyRow]
    replication: list[ReplicationRow]
    predictability: list[PredictabilityRow]
    outcomes: dict[str, OutcomeRow]
    recommendation: Recommendation


def destination_signal_value(signal: str, bars: list, nodes: list[Node], index: int,
                             memory: dict[Node, float], entropy: dict[Node, float]) -> float | None:
    if index < 0 or index >= len(bars):
        return None
    if signal == "CloseLocation":
        return bars[index].close_location
    return signal_value(signal, bars, nodes, index, memory, entropy)


def find_events(nodes: list[Node], instrument: str) -> list[DestinationEvent]:
    events = []
    for index in range(1, len(nodes)):
        previous = nodes[index - 1]
        current = nodes[index]
        if previous == AGE6_10 and current[0] in DESTINATION_FAMILIES:
            events.append(DestinationEvent(index, instrument, current[0]))
    return events


def samples_for(events: Iterable[DestinationEvent], bars: list, nodes: list[Node],
                memory: dict[Node, float], entropy: dict[Node, float]) -> dict[tuple[str, int], list[float]]:
    samples: dict[tuple[str, int], list[float]] = defaultdict(list)
    for event in events:
        for signal in SIGNALS:
            for horizon in LOOKBACKS:
                value = destination_signal_value(signal, bars, nodes, event.index - horizon, memory, entropy)
                if value is not None:
                    samples[(signal, horizon)].append(float(value))
    return samples


def destination_inventory(events_by_destination: dict[str, list[DestinationEvent]]) -> dict[str, InventoryRow]:
    total = sum(len(events) for events in events_by_destination.values())
    rows = {}
    for destination in DESTINATION_FAMILIES:
        events = events_by_destination.get(destination, [])
        counts = Counter(event.instrument for event in events)
        rows[destination] = InventoryRow(
            destination=destination,
            count=len(events),
            instrument_counts=counts,
            probability=safe_div(len(events), total) or 0.0,
            replication_count=sum(1 for value in counts.values() if value > 0),
        )
    return rows


def profile_rows(events_by_destination: dict[str, list[DestinationEvent]], bars: list, nodes: list[Node],
                 memory: dict[Node, float], entropy: dict[Node, float]) -> list[ProfileRow]:
    rows = []
    for destination in DESTINATION_FAMILIES:
        samples = samples_for(events_by_destination.get(destination, []), bars, nodes, memory, entropy)
        for signal in SIGNALS:
            for horizon in LOOKBACKS:
                values = samples.get((signal, horizon), [])
                rows.append(ProfileRow(destination, signal, horizon, mean(values), median(values), stdev(values)))
    return rows


def separation_rows(events_by_destination: dict[str, list[DestinationEvent]], bars: list, nodes: list[Node],
                    memory: dict[Node, float], entropy: dict[Node, float]) -> list[SeparationRow]:
    all_events = [event for events in events_by_destination.values() for event in events]
    rows = []
    for destination in DESTINATION_FAMILIES:
        family_events = events_by_destination.get(destination, [])
        other_events = [event for event in all_events if event.destination != destination]
        family_samples = samples_for(family_events, bars, nodes, memory, entropy)
        other_samples = samples_for(other_events, bars, nodes, memory, entropy)
        for signal in SIGNALS:
            for horizon in LOOKBACKS:
                family_values = family_samples.get((signal, horizon), [])
                other_values = other_samples.get((signal, horizon), [])
                family_mean = mean(family_values)
                other_mean = mean(other_values)
                delta = family_mean - other_mean
                pooled = pooled_stdev(family_values, other_values)
                effect = safe_div(delta, pooled) or 0.0
                if not family_values or not other_values:
                    effect = 0.0
                rows.append(SeparationRow(destination, signal, horizon, family_mean, other_mean, delta, effect))
    return rows


def pairwise_rows(events_by_destination: dict[str, list[DestinationEvent]], bars: list, nodes: list[Node],
                  memory: dict[Node, float], entropy: dict[Node, float]) -> list[PairwiseRow]:
    rows = []
    for family_a, family_b in PAIRWISE_TESTS:
        events_a = events_by_destination.get(family_a, [])
        events_b = events_by_destination.get(family_b, [])
        if len(events_a) < 25 or len(events_b) < 25:
            continue
        samples_a = samples_for(events_a, bars, nodes, memory, entropy)
        samples_b = samples_for(events_b, bars, nodes, memory, entropy)
        for signal in SIGNALS:
            for horizon in LOOKBACKS:
                values_a = samples_a.get((signal, horizon), [])
                values_b = samples_b.get((signal, horizon), [])
                delta = mean(values_a) - mean(values_b)
                pooled = pooled_stdev(values_a, values_b)
                effect = safe_div(delta, pooled) or 0.0
                if not values_a or not values_b:
                    effect = 0.0
                rows.append(PairwiseRow(family_a, family_b, signal, horizon, effect))
    return rows


def signature_label(signal: str, effect: float) -> str:
    if signal == "RangeRelativeToPrevious" and effect > 0:
        return "RangeExpansionSignature"
    if signal == "BodyRelativeToPrevious" and effect > 0:
        return "BodyExpansionSignature"
    if signal == "TrueRangeRelativeToPrevious" and effect > 0:
        return "TrueRangeExpansionSignature"
    if signal.startswith("Volume") and effect > 0:
        return "VolumeExpansionSignature"
    if signal.startswith("Volume") and effect < 0:
        return "VolumeContractionSignature"
    if signal == "EfficiencyRatio":
        return "EfficiencySignature"
    if signal == "CloseLocation":
        return "CloseLocationSignature"
    if signal == "BranchEntropy":
        return "EntropySignature"
    if signal == "MemoryStrength":
        return "MemorySignature"
    return "NoReliableSignature"


def signature_rows(separation: list[SeparationRow]) -> list[SignatureRow]:
    rows = []
    by_destination: dict[str, list[SeparationRow]] = defaultdict(list)
    for row in separation:
        by_destination[row.destination].append(row)
    for destination in DESTINATION_FAMILIES:
        candidates = [
            row for row in by_destination.get(destination, [])
            if row.replication_count >= 2 and abs(row.effect_size) >= MATERIAL_EFFECT
        ]
        top = max(candidates, key=lambda row: abs(row.effect_size), default=None)
        if top is None:
            rows.append(SignatureRow(destination, "NoReliableSignature", "None", "N/A", 0.0, 0))
        else:
            rows.append(SignatureRow(
                destination,
                signature_label(top.signal, top.effect_size),
                top.signal,
                f"t-{top.horizon}",
                top.effect_size,
                top.replication_count,
            ))
    return rows


def shared_unique_rows(separation: list[SeparationRow]) -> tuple[list[SharedSignalRow], list[UniqueSignalRow]]:
    shared = []
    unique = []
    by_signal: dict[tuple[str, int], list[SeparationRow]] = defaultdict(list)
    for row in separation:
        if row.replication_count >= 2 and abs(row.effect_size) >= MATERIAL_EFFECT:
            unique.append(UniqueSignalRow(row.destination, row.signal, row.horizon, row.effect_size, row.replication_count))
            by_signal[(row.signal, row.horizon)].append(row)
    for (signal, horizon), rows in sorted(by_signal.items()):
        if len({row.destination for row in rows}) >= 2:
            shared.append(SharedSignalRow(
                signal,
                horizon,
                mean([abs(row.effect_size) for row in rows]),
                len({row.destination for row in rows}),
            ))
    unique.sort(key=lambda row: abs(row.effect_size), reverse=True)
    shared.sort(key=lambda row: (row.shared_replication_count, abs(row.mean_effect)), reverse=True)
    return shared, unique


def timing_rows(separation: list[SeparationRow]) -> list[TimingRow]:
    rows = []
    by_destination: dict[str, list[SeparationRow]] = defaultdict(list)
    for row in separation:
        by_destination[row.destination].append(row)
    for destination in DESTINATION_FAMILIES:
        material = [
            row for row in by_destination.get(destination, [])
            if row.replication_count >= 2 and abs(row.effect_size) >= MATERIAL_EFFECT
        ]
        material.sort(key=lambda row: (-row.horizon, -abs(row.effect_size)))
        earliest = material[0] if material else None
        max_effect = max([abs(row.effect_size) for row in material], default=0.0)
        rows.append(TimingRow(
            destination,
            earliest.signal if earliest else "None",
            f"t-{earliest.horizon}" if earliest else "N/A",
            max_effect,
            earliest.replication_count if earliest else 0,
        ))
    return rows


def range_volume_rows(events_by_destination: dict[str, list[DestinationEvent]], bars: list) -> dict[str, RangeVolumeRow]:
    output = {}
    for destination, events in events_by_destination.items():
        buckets = Counter()
        for event in events:
            index = event.index - 1
            if index < 0:
                continue
            bar = bars[index]
            range_up = bar.range_relative_to_previous is not None and bar.range_relative_to_previous > 1.0
            volume_up = bar.volume_relative_to_previous is not None and bar.volume_relative_to_previous > 1.0
            buckets[(range_up, volume_up)] += 1
        total = sum(buckets.values())
        output[destination] = RangeVolumeRow(
            destination,
            safe_div(buckets[(True, True)], total) or 0.0,
            safe_div(buckets[(True, False)], total) or 0.0,
            safe_div(buckets[(False, True)], total) or 0.0,
            safe_div(buckets[(False, False)], total) or 0.0,
        )
    return output


def efficiency_close_rows(events_by_destination: dict[str, list[DestinationEvent]], bars: list) -> dict[str, EfficiencyCloseRow]:
    output = {}
    for destination, events in events_by_destination.items():
        efficiencies = []
        near_high = []
        near_low = []
        middle = []
        for event in events:
            index = event.index - 1
            if index < 0:
                continue
            bar = bars[index]
            eff = efficiency_ratio(bar)
            if eff is not None:
                efficiencies.append(eff)
            if bar.close_location is not None:
                near_high.append(1.0 if bar.close_location >= 0.80 else 0.0)
                near_low.append(1.0 if bar.close_location <= 0.20 else 0.0)
                middle.append(1.0 if 0.20 < bar.close_location < 0.80 else 0.0)
        output[destination] = EfficiencyCloseRow(
            destination,
            mean(efficiencies),
            median(efficiencies),
            mean(near_high),
            mean(near_low),
            mean(middle),
        )
    return output


def polarity_rows(events_by_destination: dict[str, list[DestinationEvent]], bars: list) -> dict[str, PolarityRow]:
    output = {}
    for destination, events in events_by_destination.items():
        black = red = 0
        flips = []
        black_to_red = []
        red_to_black = []
        for event in events:
            index = event.index - 1
            if index < 0:
                continue
            polarity = str(bars[index].volume_polarity or "").lower()
            if polarity == "black":
                black += 1
            elif polarity == "red":
                red += 1
            flip = polarity_flip(bars, index)
            if flip is not None:
                flips.append(flip)
                prev = str(bars[index - 1].volume_polarity or "").lower() if index > 0 else ""
                black_to_red.append(1.0 if prev == "black" and polarity == "red" else 0.0)
                red_to_black.append(1.0 if prev == "red" and polarity == "black" else 0.0)
        total = black + red
        output[destination] = PolarityRow(
            destination,
            safe_div(black, total) or 0.0,
            safe_div(red, total) or 0.0,
            mean(flips),
            1.0 - mean(flips) if flips else 0.0,
            mean(black_to_red),
            mean(red_to_black),
        )
    return output


def memory_entropy_rows(events_by_destination: dict[str, list[DestinationEvent]], bars: list, nodes: list[Node],
                        memory: dict[Node, float], entropy: dict[Node, float]) -> dict[str, MemoryEntropyRow]:
    output = {}
    for destination, events in events_by_destination.items():
        memory_slopes = []
        entropy_slopes = []
        memory_weak = []
        entropy_rise = []
        for event in events:
            indexes = [event.index - horizon for horizon in (5, 3, 2, 1, 0) if 0 <= event.index - horizon < len(nodes)]
            mem_values = [memory.get(nodes[index], 0.0) for index in indexes]
            ent_values = [entropy.get(nodes[index], 0.0) for index in indexes]
            mem_slope = slope(mem_values)
            ent_slope = slope(ent_values)
            memory_slopes.append(mem_slope)
            entropy_slopes.append(ent_slope)
            memory_weak.append(1.0 if mem_slope < 0 else 0.0)
            entropy_rise.append(1.0 if ent_slope > 0 else 0.0)
        output[destination] = MemoryEntropyRow(
            destination,
            mean(memory_slopes),
            mean(entropy_slopes),
            mean(memory_weak),
            mean(entropy_rise),
        )
    return output


def outcome_rows(events_by_destination: dict[str, list[DestinationEvent]], bars: list) -> dict[str, OutcomeRow]:
    output = {}
    for destination, events in events_by_destination.items():
        row = OutcomeRow(destination)
        for horizon in OUTCOME_HORIZONS:
            values = [directional_forward(bars, event.index, horizon) for event in events]
            values = [value for value in values if value is not None]
            row.dr[horizon] = mean(values)
            row.continuation[horizon] = mean([1.0 if value > 0 else 0.0 for value in values])
            row.failure[horizon] = mean([1.0 if value < 0 else 0.0 for value in values])
            row.flat[horizon] = mean([1.0 if value == 0 else 0.0 for value in values])
        output[destination] = row
    return output


def normalize(values: dict[str, float], inverse: bool = False) -> dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high == low:
        return {key: 1.0 for key in values}
    normalized = {key: (value - low) / (high - low) for key, value in values.items()}
    if inverse:
        normalized = {key: 1.0 - value for key, value in normalized.items()}
    return normalized


def predictability_rows(inventory: dict[str, InventoryRow], separation: list[SeparationRow],
                        timing: list[TimingRow]) -> list[PredictabilityRow]:
    max_effect = {}
    replicated_count = {}
    for destination in DESTINATION_FAMILIES:
        replicated = [
            row for row in separation
            if row.destination == destination
            and row.replication_count >= 2
            and abs(row.effect_size) >= MATERIAL_EFFECT
        ]
        max_effect[destination] = max([abs(row.effect_size) for row in replicated], default=0.0)
        replicated_count[destination] = len({(row.signal, row.horizon) for row in replicated})
    earliest_scores = {}
    timing_lookup = {row.destination: row for row in timing}
    for destination in DESTINATION_FAMILIES:
        text = timing_lookup.get(destination, TimingRow(destination, "None", "N/A", 0.0, 0)).earliest_horizon
        earliest_scores[destination] = {"t-5": 1.0, "t-3": 0.75, "t-2": 0.50, "t-1": 0.25}.get(text, 0.0)
    sample_scores = {
        destination: min(1.0, safe_div(inventory[destination].count, 100.0) or 0.0)
        for destination in DESTINATION_FAMILIES
    }
    parts = [
        normalize(max_effect),
        normalize(replicated_count),
        earliest_scores,
        sample_scores,
    ]
    rows = []
    for destination in DESTINATION_FAMILIES:
        if inventory[destination].count < 25:
            rows.append(PredictabilityRow(destination, 0.0, "NoReliablePrediction", 0, "N/A"))
            continue
        score = mean([part.get(destination, 0.0) for part in parts])
        if replicated_count[destination] == 0:
            classification = "NoReliablePrediction"
        elif score >= 0.67:
            classification = "HighPredictabilityDestination"
        elif score >= 0.34:
            classification = "ModeratePredictabilityDestination"
        else:
            classification = "WeakPredictabilityDestination"
        rows.append(PredictabilityRow(
            destination,
            score,
            classification,
            replicated_count[destination],
            timing_lookup.get(destination, TimingRow(destination, "None", "N/A", 0.0, 0)).earliest_horizon,
        ))
    rows.sort(key=lambda row: row.score, reverse=True)
    return rows


def make_recommendation(predictability: list[PredictabilityRow], shared: list[SharedSignalRow],
                        unique: list[UniqueSignalRow], signatures: list[SignatureRow]) -> Recommendation:
    predictable = [row.destination for row in predictability if row.classification in {"HighPredictabilityDestination", "ModeratePredictabilityDestination"}]
    unpredictable = [row.destination for row in predictability if row.classification in {"WeakPredictabilityDestination", "NoReliablePrediction"}]
    high_count = sum(1 for row in predictability if row.classification == "HighPredictabilityDestination")
    moderate_count = sum(1 for row in predictability if row.classification == "ModeratePredictabilityDestination")
    if high_count >= 3:
        classification = "StrongDestinationPredictability"
    elif high_count + moderate_count >= 3:
        classification = "PartialDestinationPredictability"
    elif high_count + moderate_count >= 1:
        classification = "WeakDestinationPredictability"
    else:
        classification = "NoDestinationPredictability"
    shared_signals = ", ".join(f"{row.signal}@t-{row.horizon}" for row in shared[:8]) or "None"
    unique_signals = ", ".join(f"{row.destination}:{row.signal}@t-{row.horizon}" for row in unique[:8]) or "None"
    volume_unique = any(row.signal.startswith("Volume") for row in unique[:12])
    volume_shared = any(row.signal.startswith("Volume") for row in shared[:12])
    if volume_unique and volume_shared:
        study75 = "Study75Refined: volume expansion is partly shared and partly destination-specific."
    elif volume_unique:
        study75 = "Study75Refined: volume expansion points to specific destination families."
    elif volume_shared:
        study75 = "Study75Confirmed: volume expansion is broadly shared across destinations."
    else:
        study75 = "Study75Weakened: destination separation is not led by replicated volume effects."
    labels = Counter(row.label for row in signatures if row.label != "NoReliableSignature")
    dominant = labels.most_common(1)[0][0] if labels else "NoReliableSignature"
    reason = f"Destination predictability uses replicated one-vs-rest pre-excursion effects only; dominant signature family is {dominant}."
    return Recommendation(
        classification,
        ", ".join(predictable) or "None",
        ", ".join(unpredictable) or "None",
        shared_signals,
        unique_signals,
        study75,
        reason,
    )


def build_result(instrument: str, source_paths: list, bars: list, nodes: list[Node],
                 stream_rows: list, node_rows: dict[Node, object]) -> Result:
    memory, entropy = node_metrics(node_rows, stream_rows)
    events = find_events(nodes, instrument)
    events_by_destination: dict[str, list[DestinationEvent]] = {destination: [] for destination in DESTINATION_FAMILIES}
    for event in events:
        events_by_destination[event.destination].append(event)
    inventory = destination_inventory(events_by_destination)
    profiles = profile_rows(events_by_destination, bars, nodes, memory, entropy)
    separation = separation_rows(events_by_destination, bars, nodes, memory, entropy)
    pairwise = pairwise_rows(events_by_destination, bars, nodes, memory, entropy)
    signatures = signature_rows(separation)
    shared, unique = shared_unique_rows(separation)
    timing = timing_rows(separation)
    range_volume = range_volume_rows(events_by_destination, bars)
    efficiency_close = efficiency_close_rows(events_by_destination, bars)
    polarity = polarity_rows(events_by_destination, bars)
    mem_entropy = memory_entropy_rows(events_by_destination, bars, nodes, memory, entropy)
    replication = replication_rows(separation)
    predictability = predictability_rows(inventory, separation, timing)
    outcomes = outcome_rows(events_by_destination, bars)
    recommendation = make_recommendation(predictability, shared, unique, signatures)
    return Result(
        instrument,
        source_paths,
        bars,
        nodes,
        stream_rows,
        events_by_destination,
        inventory,
        profiles,
        separation,
        pairwise,
        signatures,
        shared,
        unique,
        timing,
        range_volume,
        efficiency_close,
        polarity,
        mem_entropy,
        replication,
        predictability,
        outcomes,
        recommendation,
    )


def replication_rows(separation: list[SeparationRow]) -> list[ReplicationRow]:
    rows = []
    for row in separation:
        rows.append(ReplicationRow(row.destination, row.signal, row.horizon, {}, row.replication_count))
    return rows


def attach_replication(aggregate: Result, instruments: list[Result]) -> None:
    for destination, row in aggregate.inventory.items():
        row.instrument_counts = Counter({result.instrument: result.inventory[destination].count for result in instruments})
        row.replication_count = sum(1 for count in row.instrument_counts.values() if count > 0)
    effect_map: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    for result in instruments:
        for row in result.separation:
            effect_map[(row.destination, row.signal, row.horizon)][result.instrument] = row.effect_size
    for row in aggregate.separation:
        effects = effect_map.get((row.destination, row.signal, row.horizon), {})
        signs = [1 if value > 0 else -1 if value < 0 else 0 for value in effects.values() if abs(value) >= MATERIAL_EFFECT]
        positive = signs.count(1)
        negative = signs.count(-1)
        row.replication_count = max(positive, negative)
    pair_map: dict[tuple[str, str, str, int], dict[str, float]] = defaultdict(dict)
    for result in instruments:
        for row in result.pairwise:
            pair_map[(row.family_a, row.family_b, row.signal, row.horizon)][result.instrument] = row.effect_size
    for row in aggregate.pairwise:
        effects = pair_map.get((row.family_a, row.family_b, row.signal, row.horizon), {})
        signs = [1 if value > 0 else -1 if value < 0 else 0 for value in effects.values() if abs(value) >= MATERIAL_EFFECT]
        row.replication_count = max(signs.count(1), signs.count(-1))
    aggregate.signatures = signature_rows(aggregate.separation)
    aggregate.shared_signals, aggregate.unique_signals = shared_unique_rows(aggregate.separation)
    aggregate.timing = timing_rows(aggregate.separation)
    aggregate.replication = []
    for key, effects in sorted(effect_map.items()):
        destination, signal, horizon = key
        signs = [1 if value > 0 else -1 if value < 0 else 0 for value in effects.values() if abs(value) >= MATERIAL_EFFECT]
        aggregate.replication.append(ReplicationRow(destination, signal, horizon, effects, max(signs.count(1), signs.count(-1))))
    aggregate.predictability = predictability_rows(aggregate.inventory, aggregate.separation, aggregate.timing)
    aggregate.recommendation = make_recommendation(
        aggregate.predictability,
        aggregate.shared_signals,
        aggregate.unique_signals,
        aggregate.signatures,
    )


def inventory_line(row: InventoryRow) -> str:
    return (
        f"{row.destination} | {row.count} | {row.instrument_counts.get('6E', 0)} | "
        f"{row.instrument_counts.get('CL', 0)} | {row.instrument_counts.get('NQ', 0)} | "
        f"{pct(row.probability)} | {row.replication_count}"
    )


def append_common(lines: list[str], result: Result, aggregate: bool = False) -> None:
    lines += ["", "1. Destination Inventory", "DestinationFamily | Count | Count_6E | Count_CL | Count_NQ | Probability | ReplicationCount"]
    lines += [inventory_line(row) for row in result.inventory.values()]

    lines += ["", "2. Destination Profile Table", "DestinationFamily | Signal | Horizon | Mean | Median | StdDev"]
    for row in result.profiles[:400]:
        lines.append(f"{row.destination} | {row.signal} | t-{row.horizon} | {fmt(row.mean_value)} | {fmt(row.median_value)} | {fmt(row.stddev_value)}")

    lines += ["", "3. One-vs-Rest Separation Table", "DestinationFamily | Signal | Horizon | FamilyMean | OtherMean | DeltaMean | EffectSize | ReplicationCount"]
    for row in sorted(result.separation, key=lambda item: abs(item.effect_size), reverse=True)[:240]:
        lines.append(f"{row.destination} | {row.signal} | t-{row.horizon} | {fmt(row.family_mean)} | {fmt(row.other_mean)} | {fmt(row.delta_mean)} | {fmt(row.effect_size)} | {row.replication_count}")

    lines += ["", "4. Pairwise Separation Table", "FamilyA | FamilyB | Signal | Horizon | EffectSize | ReplicationCount"]
    for row in sorted(result.pairwise, key=lambda item: abs(item.effect_size), reverse=True)[:180]:
        lines.append(f"{row.family_a} | {row.family_b} | {row.signal} | t-{row.horizon} | {fmt(row.effect_size)} | {row.replication_count}")

    lines += ["", "5. Destination Signature Table", "DestinationFamily | SignatureLabel | DominantSignal | DominantHorizon | EffectSize | ReplicationCount"]
    for row in result.signatures:
        lines.append(f"{row.destination} | {row.label} | {row.dominant_signal} | {row.dominant_horizon} | {fmt(row.effect_size)} | {row.replication_count}")

    lines += ["", "6. Shared Signal Table", "Signal | Horizon | MeanEffectAcrossDestinations | SharedReplicationCount"]
    for row in result.shared_signals[:80]:
        lines.append(f"{row.signal} | t-{row.horizon} | {fmt(row.mean_effect)} | {row.shared_replication_count}")

    lines += ["", "7. Unique Signal Table", "DestinationFamily | Signal | Horizon | EffectSize | ReplicationCount"]
    for row in result.unique_signals[:120]:
        lines.append(f"{row.destination} | {row.signal} | t-{row.horizon} | {fmt(row.effect_size)} | {row.replication_count}")

    lines += ["", "8. Destination Timing Table", "DestinationFamily | EarliestSignal | EarliestHorizon | MaxEffectSize | ReplicationCount"]
    for row in result.timing:
        lines.append(f"{row.destination} | {row.earliest_signal} | {row.earliest_horizon} | {fmt(row.max_effect_size)} | {row.replication_count}")

    lines += ["", "9. Range-Volume Destination Matrix", "DestinationFamily | RangeUpVolumeUp | RangeUpVolumeDown | RangeDownVolumeUp | RangeDownVolumeDown"]
    for row in result.range_volume.values():
        lines.append(f"{row.destination} | {pct(row.range_up_volume_up)} | {pct(row.range_up_volume_down)} | {pct(row.range_down_volume_up)} | {pct(row.range_down_volume_down)}")

    lines += ["", "10. Efficiency Close Location Table", "DestinationFamily | MeanEfficiency | MedianEfficiency | CloseNearHighRate | CloseNearLowRate | MiddleCloseRate"]
    for row in result.efficiency_close.values():
        lines.append(f"{row.destination} | {fmt(row.mean_efficiency)} | {fmt(row.median_efficiency)} | {pct(row.close_near_high_rate)} | {pct(row.close_near_low_rate)} | {pct(row.middle_close_rate)}")

    lines += ["", "11. Polarity Path Table", "DestinationFamily | BlackRate | RedRate | PolarityFlipRate | SamePolarityRate | BlackToRedRate | RedToBlackRate"]
    for row in result.polarity.values():
        lines.append(f"{row.destination} | {pct(row.black_rate)} | {pct(row.red_rate)} | {pct(row.flip_rate)} | {pct(row.same_rate)} | {pct(row.black_to_red_rate)} | {pct(row.red_to_black_rate)}")

    lines += ["", "12. Memory Entropy Path Table", "DestinationFamily | MemorySlope | EntropySlope | MemoryWeakeningRate | EntropyRisingRate"]
    for row in result.memory_entropy.values():
        lines.append(f"{row.destination} | {fmt(row.memory_slope)} | {fmt(row.entropy_slope)} | {pct(row.memory_weakening_rate)} | {pct(row.entropy_rising_rate)}")

    if aggregate:
        lines += ["", "13. Cross-Instrument Replication Table", "DestinationFamily | Signal | Horizon | Effect_6E | Effect_CL | Effect_NQ | ReplicationCount"]
        for row in sorted(result.replication, key=lambda item: (item.replication_count, max(abs(v) for v in item.effects.values()) if item.effects else 0.0), reverse=True)[:240]:
            lines.append(f"{row.destination} | {row.signal} | t-{row.horizon} | {fmt(row.effects.get('6E', 0.0))} | {fmt(row.effects.get('CL', 0.0))} | {fmt(row.effects.get('NQ', 0.0))} | {row.replication_count}")

    lines += ["", "14. Predictability Table", "DestinationFamily | PredictabilityScore | Classification | ReplicatedSignalCount | EarliestLeadTime"]
    for row in result.predictability:
        lines.append(f"{row.destination} | {fmt(row.score)} | {row.classification} | {row.replicated_signal_count} | {row.earliest_lead_time}")

    lines += ["", "15. Outcome Diagnostics Table", "DestinationFamily | DRFwd1 | DRFwd3 | DRFwd5 | DRFwd10 | ContinuationRate5 | FailureRate5 | FlatRate5"]
    for row in result.outcomes.values():
        lines.append(f"{row.destination} | {fmt(row.dr.get(1, 0.0))} | {fmt(row.dr.get(3, 0.0))} | {fmt(row.dr.get(5, 0.0))} | {fmt(row.dr.get(10, 0.0))} | {pct(row.continuation.get(5, 0.0))} | {pct(row.failure.get(5, 0.0))} | {pct(row.flat.get(5, 0.0))}")

    rec = result.recommendation
    lines += [
        "",
        "16. Recommendation",
        f"Classification: {rec.classification}",
        f"PredictableDestinations: {rec.predictable_destinations}",
        f"UnpredictableDestinations: {rec.unpredictable_destinations}",
        f"SharedFailureSignals: {rec.shared_failure_signals}",
        f"UniqueDestinationSignals: {rec.unique_destination_signals}",
        f"Study75Assessment: {rec.study75_assessment}",
        f"Reason: {rec.reason}",
        "",
        "17. Comparison With Study 75",
        rec.study75_assessment,
        "",
        "18. Low-DoF Audit",
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
        "No forward returns used in destination classification.",
    ]


def rankings(result: Result) -> list[str]:
    lines = ["", "RANKINGS"]
    top_inventory = sorted(result.inventory.values(), key=lambda row: row.count, reverse=True)
    top_predictable = sorted(result.predictability, key=lambda row: row.score, reverse=True)
    top_separation = sorted(result.separation, key=lambda row: abs(row.effect_size), reverse=True)
    top_pairwise = sorted(result.pairwise, key=lambda row: abs(row.effect_size), reverse=True)
    replicated = sorted([row for row in result.replication if row.replication_count >= 2], key=lambda row: row.replication_count, reverse=True)
    lines.append("1. Most common destinations: " + "; ".join(f"{row.destination}={row.count}" for row in top_inventory[:8]))
    lines.append("2. Most predictable destinations: " + "; ".join(f"{row.destination}={fmt(row.score)}" for row in top_predictable[:8]))
    lines.append("3. Least predictable destinations: " + "; ".join(f"{row.destination}={fmt(row.score)}" for row in reversed(top_predictable[-8:])))
    lines.append("4. Strongest one-vs-rest signals: " + "; ".join(f"{row.destination}:{row.signal}@t-{row.horizon}={fmt(row.effect_size)}" for row in top_separation[:8]))
    lines.append("5. Strongest pairwise destination signals: " + "; ".join(f"{row.family_a}/{row.family_b}:{row.signal}@t-{row.horizon}={fmt(row.effect_size)}" for row in top_pairwise[:8]))
    lines.append("6. Most replicated destination signals: " + "; ".join(f"{row.destination}:{row.signal}@t-{row.horizon}=R{row.replication_count}" for row in replicated[:8]))
    lines.append("7. Earliest destination signals: " + "; ".join(f"{row.destination}:{row.earliest_signal}@{row.earliest_horizon}" for row in result.timing[:8]))
    lines.append("8. Strongest range-volume destination patterns: " + "; ".join(f"{row.destination}:UpUp={pct(row.range_up_volume_up)}" for row in result.range_volume.values()))
    lines.append("9. Strongest efficiency/close-location signatures: " + "; ".join(f"{row.destination}:Eff={fmt(row.mean_efficiency)}" for row in result.efficiency_close.values()))
    lines.append(f"10. Recommended excursion destination model: {result.recommendation.classification}")
    return lines


def write_per_instrument(result: Result, out_root: Path) -> None:
    path = out_root / result.instrument / f"ExcursionDestinations76_{result.instrument}.txt"
    ensure_dir(path.parent)
    lines = [
        f"APVA Excursion Destinations Study v0.1 - {result.instrument}",
        "=" * 96,
        f"Instrument: {result.instrument}",
        f"Input path(s): {', '.join(str(path) for path in result.source_paths)}",
        f"Total rows: {len(result.bars)}",
        f"Destination event count: {sum(row.count for row in result.inventory.values())}",
    ]
    append_common(lines, result)
    lines += rankings(result)
    lines += [
        "",
        "RESEARCH NOTES",
        "Mechanical only.",
        "Once Neutral fails, this report audits whether destination families have pre-excursion fingerprints.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aggregate(result: Result, instruments: list[Result], out_root: Path) -> None:
    attach_replication(result, instruments)
    path = out_root / "ExcursionDestinations76" / "ExcursionDestinations76_All.txt"
    ensure_dir(path.parent)
    lines = [
        "APVA Excursion Destinations Study v0.1 - Aggregate",
        "=" * 96,
        f"Instruments: {', '.join(result.instrument for result in instruments)}",
    ]
    append_common(lines, result, aggregate=True)
    lines += rankings(result)
    lines += [
        "",
        "RESEARCH NOTES",
        "Mechanical only.",
        "Questions: Can destination family be distinguished before excursion? Are destination signals shared or unique? Is Study 75 volume expansion universal or destination-specific?",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(result: Result) -> None:
    if not result.bars:
        raise RuntimeError(f"{result.instrument}: no bars.")
    if len(result.bars) != len(result.nodes):
        raise RuntimeError(f"{result.instrument}: bar/node length mismatch.")
    if set(result.inventory) != set(DESTINATION_FAMILIES):
        raise RuntimeError(f"{result.instrument}: destination inventory incomplete.")
    expected_profiles = len(DESTINATION_FAMILIES) * len(SIGNALS) * len(LOOKBACKS)
    if len(result.profiles) != expected_profiles:
        raise RuntimeError(f"{result.instrument}: destination profile table incomplete.")
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
    node_rows = aggregate_rows(decay)
    score_rows(node_rows)
    aggregate_thresholds = thresholds(loaded, node_rows)

    instrument_results = []
    aggregate_bars = []
    aggregate_nodes = []
    aggregate_stream = []
    aggregate_paths = []
    for loaded_row, decay_row in zip(loaded, decay):
        local_node_rows = local_rows(decay_row)
        score_rows(local_node_rows)
        local_stream = build_stream(loaded_row, local_node_rows, aggregate_thresholds, (0.0, 0.0), (0.0, 0.0))
        nodes = [node_for(bar) for bar in loaded_row.bars]
        result = build_result(loaded_row.instrument, loaded_row.source_paths, loaded_row.bars, nodes, local_stream, local_node_rows)
        validate(result)
        instrument_results.append(result)

        offset = len(aggregate_bars)
        aggregate_stream.extend(build_stream(loaded_row, node_rows, aggregate_thresholds, (0.0, 0.0), (0.0, 0.0), offset))
        aggregate_bars.extend(loaded_row.bars)
        aggregate_nodes.extend(nodes)
        aggregate_paths.extend(loaded_row.source_paths)

    aggregate_result = build_result("Aggregate", aggregate_paths, aggregate_bars, aggregate_nodes, aggregate_stream, node_rows)
    validate(aggregate_result)

    out_root = Path(args.out_root)
    for result in instrument_results:
        write_per_instrument(result, out_root)
    write_aggregate(aggregate_result, instrument_results, out_root)
    print(f"Wrote {len(instrument_results)} per-instrument Study76 report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
