#!/usr/bin/env python3
"""APVA Neutral Failure Timing Study v0.1.

Study 83 tests whether NeutralFailure is better represented as a causal warning
overlay than as a hard real-time phase.

Research only. No trading, optimization, fitting, machine learning, future
NextNode predictors, completed-loop hindsight predictors, or forward returns in
warning construction.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from APVA_DestinationRobustness_77 import build_contexts
from APVA_ExcursionDestinations_76 import DESTINATION_FAMILIES
from APVA_ExcursionLifecycle_78 import efficiency_ratio
from APVA_InformationContribution_63 import thresholds
from APVA_InformationDecay_57 import study as decay_study
from APVA_MinimalEngine_52 import load_results
from APVA_NeutralFailureModes_75 import (
    directional_forward,
    mean,
    pooled_stdev,
    safe_div,
    stdev,
)
from APVA_NodeImportance_58 import aggregate_rows, local_rows, score_rows
from APVA_ProcessGraph_54 import Node, node_text
from APVA_RealTimeStateMachine_82 import (
    build_records,
    failure_detection_row,
    is_neutral,
)
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

LOOKBACKS = (0, 1, 2, 3, 5)
THRESHOLDS = (1, 2, 3, 4)
OUTCOME_HORIZONS = (1, 3, 5, 10)
NEUTRAL_AGES = ("2", "3", "4", "5", "6-10", "11-20", "21+")
SIGNALS = (
    "RangeRelativeToPrevious",
    "TrueRangeRelativeToPrevious",
    "BodyRelativeToPrevious",
    "VolumeRelativeToRollingMean",
    "VolumeRelativeToPrevious",
    "VolumeRelativeToSessionMean",
    "EfficiencyRatio",
    "CloseLocation",
    "PolarityFlipFlag",
)


@dataclass(frozen=True)
class FailureEvent:
    instrument: str
    index: int
    node: Node
    destination: str


@dataclass(frozen=True)
class ControlEvent:
    instrument: str
    index: int
    node: Node
    horizon: int


@dataclass
class InventoryRow:
    failure_node: str
    count: int
    instrument_counts: Counter = field(default_factory=Counter)
    probability: float = 0.0


@dataclass
class ShiftRow:
    failure_node: str
    age_bucket: str
    horizon: int
    signal: str
    failure_mean: float
    control_mean: float
    delta: float
    effect_size: float
    replication_count: int = 0


@dataclass
class ScoreDistributionRow:
    score: int
    failure_count: int
    control_count: int
    failure_percent: float
    control_percent: float


@dataclass
class ThresholdRow:
    threshold: int
    detection_at_failure: float
    early1: float
    early2: float
    early3: float
    false_positive: float
    precision: float
    recall: float


@dataclass
class AgeThresholdRow:
    age_bucket: str
    threshold: int
    detection_rate: float
    false_positive: float
    precision: float
    recall: float


@dataclass
class HardVsWarningRow:
    method: str
    detection_rate: float
    false_positive: float
    precision: float
    recall: float
    phase_agreement: float


@dataclass
class DestinationRow:
    destination: str
    mean_score: float
    detection_rate: float
    top_signal: str


@dataclass
class FalsePositiveRow:
    age_bucket: str
    signal_pattern: str
    count: int
    subsequent_phase: str


@dataclass
class MissedRow:
    failure_node: str
    destination: str
    count: int
    top_missing_signal: str


@dataclass
class ReplicationRow:
    instrument: str
    threshold: int
    detection_rate: float
    false_positive: float
    precision: float
    recall: float


@dataclass
class OutcomeRow:
    score: int
    dr: dict[int, float] = field(default_factory=dict)
    continuation5: float = 0.0
    failure5: float = 0.0
    flat5: float = 0.0
    count: int = 0


@dataclass
class Recommendation:
    classification: str
    logic: str
    threshold: int
    reliable_signals: str
    rejected_signals: str
    next_step: str
    reason: str


@dataclass
class Result:
    instrument: str
    source_paths: list
    contexts: list
    records: list
    failures: list[FailureEvent]
    controls: list[ControlEvent]
    inventory: list[InventoryRow]
    shifts: list[ShiftRow]
    score_distribution: list[ScoreDistributionRow]
    thresholds: list[ThresholdRow]
    age_thresholds: list[AgeThresholdRow]
    hard_vs_warning: list[HardVsWarningRow]
    destinations: list[DestinationRow]
    false_positives: list[FalsePositiveRow]
    missed: list[MissedRow]
    replication: list[ReplicationRow]
    outcomes: list[OutcomeRow]
    recommendation: Recommendation


def bar_value(bar, signal: str, bars: list, index: int) -> float | None:
    if index < 0 or index >= len(bars):
        return None
    if signal == "RangeRelativeToPrevious":
        return bar.range_relative_to_previous
    if signal == "TrueRangeRelativeToPrevious":
        return bar.true_range_relative_to_previous
    if signal == "BodyRelativeToPrevious":
        return bar.body_relative_to_previous
    if signal == "VolumeRelativeToRollingMean":
        return bar.volume_relative_to_rolling_mean
    if signal == "VolumeRelativeToPrevious":
        return bar.volume_relative_to_previous
    if signal == "VolumeRelativeToSessionMean":
        return bar.volume_relative_to_session_mean
    if signal == "EfficiencyRatio":
        return efficiency_ratio(bar)
    if signal == "CloseLocation":
        return bar.close_location
    if signal == "PolarityFlipFlag":
        if index <= 0:
            return None
        previous = bars[index - 1].volume_polarity or "Neutral"
        current = bars[index].volume_polarity or "Neutral"
        return float(previous != current)
    return None


def signal_value(context, index: int, signal: str) -> float | None:
    if index < 0 or index >= len(context.bars):
        return None
    return bar_value(context.bars[index], signal, context.bars, index)


def active_components(context, index: int) -> dict[str, bool]:
    bar = context.bars[index]
    efficiency = efficiency_ratio(bar)
    prev_efficiency = efficiency_ratio(context.bars[index - 1]) if index > 0 else None
    close_location = bar.close_location
    return {
        "RangePressure": (bar.range_relative_to_previous or 0.0) > 1.0 or (bar.true_range_relative_to_previous or 0.0) > 1.0,
        "BodyPressure": (bar.body_relative_to_previous or 0.0) > 1.0,
        "VolumePressure": (bar.volume_relative_to_rolling_mean or 0.0) > 1.0 or (bar.volume_relative_to_previous or 0.0) > 1.0,
        "EfficiencyPressure": (efficiency is not None and efficiency >= 0.60) or (efficiency is not None and prev_efficiency is not None and efficiency > prev_efficiency),
        "CloseExtremePressure": close_location is not None and (close_location <= 0.20 or close_location >= 0.80),
        "PolarityInstability": (signal_value(context, index, "PolarityFlipFlag") or 0.0) == 1.0,
    }


def warning_score(context, index: int) -> int:
    return sum(1 for active in active_components(context, index).values() if active)


def context_lookup(contexts: list) -> dict[str, object]:
    return {context.instrument: context for context in contexts}


def failure_events(contexts: list, records: list) -> list[FailureEvent]:
    lookup = context_lookup(contexts)
    events = []
    for record in records:
        if record.reference_phase != "NeutralFailure":
            continue
        context = lookup[record.instrument]
        destination = "None"
        for scan in range(record.index + 1, min(record.index + 6, len(context.nodes))):
            node = context.nodes[scan]
            if node[0] in DESTINATION_FAMILIES:
                destination = node[0]
                break
        events.append(FailureEvent(record.instrument, record.index, record.node, destination))
    return events


def stable_controls(contexts: list, records: list) -> list[ControlEvent]:
    reference_failures = {(row.instrument, row.index) for row in records if row.reference_phase == "NeutralFailure"}
    controls = []
    for context in contexts:
        labeled = {row.index for row in records if row.instrument == context.instrument}
        for index, node in enumerate(context.nodes):
            if index not in labeled or (context.instrument, index) in reference_failures:
                continue
            if not is_neutral(node) or node[1] not in NEUTRAL_AGES:
                continue
            for horizon in (1, 2, 3, 5):
                future = context.nodes[index + 1:min(index + horizon + 1, len(context.nodes))]
                if not any(candidate[0] in DESTINATION_FAMILIES for candidate in future):
                    controls.append(ControlEvent(context.instrument, index, node, horizon))
    return controls


def inventory_rows(events: list[FailureEvent], instruments: list[str]) -> list[InventoryRow]:
    total = len(events)
    grouped = defaultdict(list)
    for event in events:
        grouped[node_text(event.node)].append(event)
    rows = []
    for node, values in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        counts = Counter(event.instrument for event in values)
        rows.append(InventoryRow(node, len(values), Counter({instrument: counts[instrument] for instrument in instruments}), safe_div(len(values), total) or 0.0))
    return rows


def samples_for_events(contexts: list, events: list[FailureEvent], signal: str, horizon: int) -> list[float]:
    lookup = context_lookup(contexts)
    values = []
    for event in events:
        value = signal_value(lookup[event.instrument], event.index - horizon, signal)
        if value is not None:
            values.append(value)
    return values


def samples_for_controls(contexts: list, controls: list[ControlEvent], signal: str, horizon: int, age_bucket: str | None = None) -> list[float]:
    lookup = context_lookup(contexts)
    values = []
    for control in controls:
        if control.horizon < max(1, horizon):
            continue
        if age_bucket and control.node[1] != age_bucket:
            continue
        value = signal_value(lookup[control.instrument], control.index - horizon, signal)
        if value is not None:
            values.append(value)
    return values


def shift_rows(contexts: list, events: list[FailureEvent], controls: list[ControlEvent], instrument_results: list[Result] | None = None) -> list[ShiftRow]:
    rows = []
    for age in ("All",) + NEUTRAL_AGES:
        age_events = events if age == "All" else [event for event in events if event.node[1] == age]
        if not age_events:
            continue
        for horizon in LOOKBACKS:
            for signal in SIGNALS:
                failure_values = samples_for_events(contexts, age_events, signal, horizon)
                control_values = samples_for_controls(contexts, controls, signal, horizon, None if age == "All" else age)
                failure_mean = mean(failure_values)
                control_mean = mean(control_values)
                pooled = pooled_stdev(failure_values, control_values)
                effect = safe_div(failure_mean - control_mean, pooled) or 0.0
                rows.append(ShiftRow(
                    "AllFailureNodes",
                    age,
                    horizon,
                    signal,
                    failure_mean,
                    control_mean,
                    failure_mean - control_mean,
                    effect,
                ))
    if instrument_results:
        by_key = defaultdict(dict)
        for result in instrument_results:
            for row in result.shifts:
                by_key[(row.age_bucket, row.horizon, row.signal)][result.instrument] = row.effect_size
        for row in rows:
            values = by_key.get((row.age_bucket, row.horizon, row.signal), {})
            sign = 1 if row.effect_size >= 0 else -1
            row.replication_count = sum(1 for effect in values.values() if abs(effect) >= 0.25 and (1 if effect >= 0 else -1) == sign)
    return rows


def score_distribution_rows(contexts: list, events: list[FailureEvent], controls: list[ControlEvent]) -> list[ScoreDistributionRow]:
    lookup = context_lookup(contexts)
    failure_counts = Counter(warning_score(lookup[event.instrument], event.index) for event in events)
    control_counts = Counter(warning_score(lookup[control.instrument], control.index) for control in controls if control.horizon == 5)
    failure_total = sum(failure_counts.values())
    control_total = sum(control_counts.values())
    return [
        ScoreDistributionRow(
            score,
            failure_counts[score],
            control_counts[score],
            safe_div(failure_counts[score], failure_total) or 0.0,
            safe_div(control_counts[score], control_total) or 0.0,
        )
        for score in range(7)
    ]


def failure_windows(events: list[FailureEvent], threshold: int, lookup: dict[str, object], early: int = 0) -> set[tuple[str, int]]:
    hits = set()
    for event in events:
        context = lookup[event.instrument]
        for distance in range(early + 1):
            idx = event.index - distance
            if idx >= 0 and warning_score(context, idx) >= threshold:
                hits.add((event.instrument, event.index))
                break
    return hits


def false_positive_count(contexts: list, events: list[FailureEvent], threshold: int, age_bucket: str | None = None) -> tuple[int, int]:
    failure_keys = {(event.instrument, event.index) for event in events}
    count = population = 0
    for context in contexts:
        failure_indexes = {event.index for event in events if event.instrument == context.instrument}
        for index, node in enumerate(context.nodes):
            if not is_neutral(node) or node[1] not in NEUTRAL_AGES:
                continue
            if age_bucket and node[1] != age_bucket:
                continue
            population += 1
            if (context.instrument, index) in failure_keys:
                continue
            future_failure = any((index + distance) in failure_indexes for distance in range(1, 6))
            if not future_failure and warning_score(context, index) >= threshold:
                count += 1
    return count, population


def threshold_rows(contexts: list, events: list[FailureEvent]) -> list[ThresholdRow]:
    lookup = context_lookup(contexts)
    rows = []
    for threshold in THRESHOLDS:
        at_failure = failure_windows(events, threshold, lookup, 0)
        early1 = failure_windows(events, threshold, lookup, 1)
        early2 = failure_windows(events, threshold, lookup, 2)
        early3 = failure_windows(events, threshold, lookup, 3)
        false_count, population = false_positive_count(contexts, events, threshold)
        tp = len(at_failure)
        fp = false_count
        rows.append(ThresholdRow(
            threshold,
            safe_div(len(at_failure), len(events)) or 0.0,
            safe_div(len(early1), len(events)) or 0.0,
            safe_div(len(early2), len(events)) or 0.0,
            safe_div(len(early3), len(events)) or 0.0,
            safe_div(false_count, population) or 0.0,
            safe_div(tp, tp + fp) or 0.0,
            safe_div(tp, len(events)) or 0.0,
        ))
    return rows


def recommended_threshold(rows: list[ThresholdRow]) -> ThresholdRow:
    constrained = [row for row in rows if row.false_positive <= 0.25]
    if constrained:
        return max(constrained, key=lambda row: (row.recall, row.precision))
    return max(rows, key=lambda row: (row.recall - row.false_positive, row.precision), default=ThresholdRow(2, 0, 0, 0, 0, 0, 0, 0))


def age_threshold_rows(contexts: list, events: list[FailureEvent]) -> list[AgeThresholdRow]:
    rows = []
    lookup = context_lookup(contexts)
    for age in NEUTRAL_AGES:
        age_events = [event for event in events if event.node[1] == age]
        if not age_events:
            continue
        for threshold in THRESHOLDS:
            hits = failure_windows(age_events, threshold, lookup, 0)
            false_count, population = false_positive_count(contexts, age_events, threshold, age)
            rows.append(AgeThresholdRow(
                age,
                threshold,
                safe_div(len(hits), len(age_events)) or 0.0,
                safe_div(false_count, population) or 0.0,
                safe_div(len(hits), len(hits) + false_count) or 0.0,
                safe_div(len(hits), len(age_events)) or 0.0,
            ))
    return rows


def hard_vs_warning_rows(contexts: list, records: list, events: list[FailureEvent], thresholds: list[ThresholdRow]) -> list[HardVsWarningRow]:
    hard = failure_detection_row("HardNeutralFailurePhase", records)
    hard_precision = safe_div(hard.detected, hard.detected + hard.false_positive) or 0.0
    best = recommended_threshold(thresholds)
    phase_agreement = safe_div(sum(1 for row in records if row.realtime_phase == row.reference_phase), len(records)) or 0.0
    return [
        HardVsWarningRow("HardNeutralFailurePhase", hard.detection_rate, hard.false_positive_rate, hard_precision, hard.detection_rate, phase_agreement),
        HardVsWarningRow(f"FailureWarningScore>={best.threshold}", best.detection_at_failure, best.false_positive, best.precision, best.recall, phase_agreement),
    ]


def destination_rows(contexts: list, events: list[FailureEvent], shifts: list[ShiftRow], threshold: int) -> list[DestinationRow]:
    lookup = context_lookup(contexts)
    rows = []
    for destination in DESTINATION_FAMILIES:
        selected = [event for event in events if event.destination == destination]
        if not selected:
            rows.append(DestinationRow(destination, 0.0, 0.0, "N/A"))
            continue
        scores = [warning_score(lookup[event.instrument], event.index) for event in selected]
        detected = sum(1 for score in scores if score >= threshold)
        top_signal = max(
            ((row.signal, abs(row.effect_size)) for row in shifts if row.age_bucket == "All" and row.horizon == 0),
            key=lambda item: item[1],
            default=("N/A", 0.0),
        )[0]
        rows.append(DestinationRow(destination, mean(scores), safe_div(detected, len(selected)) or 0.0, top_signal))
    return rows


def signal_pattern(context, index: int) -> str:
    active = [name for name, value in active_components(context, index).items() if value]
    return "+".join(active) if active else "None"


def false_positive_rows(contexts: list, events: list[FailureEvent], threshold: int) -> list[FalsePositiveRow]:
    failure_indexes = defaultdict(set)
    for event in events:
        failure_indexes[event.instrument].add(event.index)
    grouped = Counter()
    for context in contexts:
        for index, node in enumerate(context.nodes):
            if not is_neutral(node) or node[1] not in NEUTRAL_AGES:
                continue
            if any(index + distance in failure_indexes[context.instrument] for distance in range(0, 6)):
                continue
            if warning_score(context, index) < threshold:
                continue
            future = "End"
            if index + 1 < len(context.nodes):
                future = node_text(context.nodes[index + 1])
            grouped[(node[1], signal_pattern(context, index), future)] += 1
    return [FalsePositiveRow(age, pattern, count, future) for (age, pattern, future), count in grouped.most_common(40)]


def missed_rows(contexts: list, events: list[FailureEvent], threshold: int) -> list[MissedRow]:
    lookup = context_lookup(contexts)
    grouped = Counter()
    for event in events:
        context = lookup[event.instrument]
        if warning_score(context, event.index) >= threshold:
            continue
        missing = []
        components = active_components(context, event.index)
        for name, active in components.items():
            if not active:
                missing.append(name)
        grouped[(node_text(event.node), event.destination, "+".join(missing[:3]) or "None")] += 1
    return [MissedRow(node, destination, count, signal) for (node, destination, signal), count in grouped.most_common(40)]


def replication_rows(instrument_results: list[Result]) -> list[ReplicationRow]:
    rows = []
    for result in instrument_results:
        for row in result.thresholds:
            rows.append(ReplicationRow(result.instrument, row.threshold, row.detection_at_failure, row.false_positive, row.precision, row.recall))
    return rows


def outcome_rows(contexts: list) -> list[OutcomeRow]:
    grouped = defaultdict(list)
    for context in contexts:
        for index in range(len(context.bars)):
            if is_neutral(context.nodes[index]):
                grouped[warning_score(context, index)].append((context, index))
    rows = []
    for score in range(7):
        row = OutcomeRow(score)
        samples = grouped.get(score, [])
        values5 = []
        for horizon in OUTCOME_HORIZONS:
            values = []
            for context, index in samples:
                value = directional_forward(context.bars, index, horizon)
                if value is not None:
                    values.append(value)
                    if horizon == 5:
                        values5.append(value)
            row.dr[horizon] = mean(values)
        row.count = len(values5)
        row.continuation5 = safe_div(sum(1 for value in values5 if value > 0), len(values5)) or 0.0
        row.failure5 = safe_div(sum(1 for value in values5 if value < 0), len(values5)) or 0.0
        row.flat5 = safe_div(sum(1 for value in values5 if value == 0), len(values5)) or 0.0
        rows.append(row)
    return rows


def make_recommendation(thresholds: list[ThresholdRow], hard_warning: list[HardVsWarningRow], shifts: list[ShiftRow]) -> Recommendation:
    hard = next((row for row in hard_warning if row.method == "HardNeutralFailurePhase"), None)
    warning = recommended_threshold(thresholds) if thresholds else None
    reliable = sorted({row.signal for row in shifts if row.replication_count >= 2 and abs(row.effect_size) >= 0.25})
    rejected = sorted({signal for signal in SIGNALS if signal not in reliable})
    if warning and warning.recall > 0.60 and warning.false_positive <= 0.15:
        classification = "HybridFailureLogicPreferred"
        logic = "Option C: use FailureWarningScore overlay while retaining phase labels for audit only."
        next_step = "ProceedToTradeTimingAudit"
    elif warning and hard and warning.recall > hard.recall + 0.20 and warning.false_positive <= 0.25:
        classification = "WarningOverlayPreferred"
        logic = "Option C: keep loop phases, remove hard NeutralFailure from core machine, and add FailureWarningScore overlay."
        next_step = "ReviseRealTimeMachine"
    elif hard and warning and hard.recall >= warning.recall:
        classification = "HardFailurePhasePreferred"
        logic = "Option A: keep hard NeutralFailure phase."
        next_step = "ReviseRealTimeMachine"
    else:
        classification = "NoReliableFailureTiming"
        logic = "No causal warning threshold materially improves failure timing."
        next_step = "CollectMoreData"
    reason = "Recommended warning threshold is " + (str(warning.threshold) if warning else "N/A")
    if warning:
        reason += f" with recall {pct(warning.recall)}, precision {pct(warning.precision)}, and false positives {pct(warning.false_positive)}."
    return Recommendation(classification, logic, warning.threshold if warning else 0, ", ".join(reliable) or "None", ", ".join(rejected) or "None", next_step, reason)


def build_result(instrument: str, source_paths: list, contexts: list, instrument_results: list[Result] | None = None) -> Result:
    records = build_records(contexts)
    events = failure_events(contexts, records)
    controls = stable_controls(contexts, records)
    inventory = inventory_rows(events, sorted({context.instrument for context in contexts}))
    shifts = shift_rows(contexts, events, controls, instrument_results)
    distribution = score_distribution_rows(contexts, events, controls)
    thresholds = threshold_rows(contexts, events)
    age_thresholds = age_threshold_rows(contexts, events)
    hard_warning = hard_vs_warning_rows(contexts, records, events, thresholds)
    best_threshold = recommended_threshold(thresholds).threshold
    destinations = destination_rows(contexts, events, shifts, best_threshold)
    false_rows = false_positive_rows(contexts, events, best_threshold)
    missed = missed_rows(contexts, events, best_threshold)
    outcomes = outcome_rows(contexts)
    recommendation = make_recommendation(thresholds, hard_warning, shifts)
    return Result(
        instrument, source_paths, contexts, records, events, controls, inventory, shifts,
        distribution, thresholds, age_thresholds, hard_warning, destinations,
        false_rows, missed, [], outcomes, recommendation,
    )


def append_common(lines: list[str], result: Result) -> None:
    lines += ["", "1. Reference Failure Inventory", "FailureNode | Count | Count_6E | Count_CL | Count_NQ | Probability"]
    for row in result.inventory:
        lines.append(f"{row.failure_node} | {row.count} | {row.instrument_counts.get('6E', 0)} | {row.instrument_counts.get('CL', 0)} | {row.instrument_counts.get('NQ', 0)} | {pct(row.probability)}")

    lines += ["", "2. Warning Window Signal Table", "FailureNode | AgeBucket | Horizon | Signal | FailureMean | ControlMean | EffectSize | ReplicationCount"]
    for row in sorted(result.shifts, key=lambda item: (-abs(item.effect_size), item.age_bucket, item.horizon, item.signal))[:120]:
        lines.append(f"{row.failure_node} | {row.age_bucket} | t-{row.horizon} | {row.signal} | {fmt(row.failure_mean)} | {fmt(row.control_mean)} | {fmt(row.effect_size)} | {row.replication_count}")

    lines += ["", "3. Warning Score Distribution", "Score | FailureCount | ControlCount | FailurePercent | ControlPercent"]
    for row in result.score_distribution:
        lines.append(f"{row.score} | {row.failure_count} | {row.control_count} | {pct(row.failure_percent)} | {pct(row.control_percent)}")

    lines += ["", "4. Threshold Performance Table", "Threshold | DetectionAtFailure | EarlyDetection1 | EarlyDetection2 | EarlyDetection3 | FalsePositiveRate | Precision | Recall"]
    for row in result.thresholds:
        lines.append(f">={row.threshold} | {pct(row.detection_at_failure)} | {pct(row.early1)} | {pct(row.early2)} | {pct(row.early3)} | {pct(row.false_positive)} | {pct(row.precision)} | {pct(row.recall)}")

    lines += ["", "5. Warning Score by Neutral Age Table", "AgeBucket | Threshold | DetectionRate | FalsePositiveRate | Precision | Recall"]
    for row in result.age_thresholds:
        lines.append(f"{row.age_bucket} | >={row.threshold} | {pct(row.detection_rate)} | {pct(row.false_positive)} | {pct(row.precision)} | {pct(row.recall)}")

    lines += ["", "6. Hard vs Warning Table", "Method | DetectionRate | FalsePositiveRate | Precision | Recall | PhaseAgreement"]
    for row in result.hard_vs_warning:
        lines.append(f"{row.method} | {pct(row.detection_rate)} | {pct(row.false_positive)} | {pct(row.precision)} | {pct(row.recall)} | {pct(row.phase_agreement)}")

    lines += ["", "7. Destination Conditioning Table", "DestinationFamily | MeanWarningScore | DetectionRate | TopSignal"]
    for row in result.destinations:
        lines.append(f"{row.destination} | {fmt(row.mean_score)} | {pct(row.detection_rate)} | {row.top_signal}")

    lines += ["", "8. False Positive Anatomy Table", "AgeBucket | SignalPattern | Count | SubsequentPhase"]
    for row in result.false_positives[:40]:
        lines.append(f"{row.age_bucket} | {row.signal_pattern} | {row.count} | {row.subsequent_phase}")

    lines += ["", "9. Missed Failure Anatomy Table", "FailureNode | DestinationFamily | Count | TopMissingSignal"]
    for row in result.missed[:40]:
        lines.append(f"{row.failure_node} | {row.destination} | {row.count} | {row.top_missing_signal}")

    lines += ["", "10. Cross-Instrument Replication Table", "Instrument | Threshold | DetectionRate | FalsePositiveRate | Precision | Recall"]
    for row in result.replication:
        lines.append(f"{row.instrument} | >={row.threshold} | {pct(row.detection_rate)} | {pct(row.false_positive)} | {pct(row.precision)} | {pct(row.recall)}")

    lines += ["", "11. Revised State Machine Table", "State | EntryRule | ExitRule | OverlayRule"]
    revised = (
        ("NeutralFormation", "NeutralProcessing_Age1", "Age advances to mature Neutral", "FailureWarningScore available but not state-defining"),
        ("NeutralMaturation", "NeutralProcessing_Age2-4", "Age enters LateNeutral or non-neutral appears", "Score >= threshold means FailurePressure"),
        ("LateNeutral", "NeutralProcessing_Age5+", "Non-neutral state appears", "Score >= threshold means FailurePressure"),
        ("DestinationSelection", "First non-neutral state appears", "Next non-neutral bar or return path", "DestinationFamily becomes known causally"),
        ("Excursion", "Non-neutral continuation after destination", "DecayToNeutral or NeutralProcessing_Age1", "Score not used for phase assignment"),
        ("ReturnToNeutral", "DecayToNeutral or final non-neutral reset context", "NeutralProcessing_Age1", "Return pressure observed from state only"),
    )
    for state, entry, exit_rule, overlay in revised:
        lines.append(f"{state} | {entry} | {exit_rule} | {overlay}")

    lines += ["", "12. Outcome Diagnostics Table", "WarningScore | Count | DRFwd1 | DRFwd3 | DRFwd5 | DRFwd10 | ContinuationRate5 | FailureRate5 | FlatRate5"]
    for row in result.outcomes:
        lines.append(f"{row.score} | {row.count} | {fmt(row.dr.get(1, 0.0))} | {fmt(row.dr.get(3, 0.0))} | {fmt(row.dr.get(5, 0.0))} | {fmt(row.dr.get(10, 0.0))} | {pct(row.continuation5)} | {pct(row.failure5)} | {pct(row.flat5)}")

    rec = result.recommendation
    lines += ["", "13. Recommendation", "Classification | RecommendedFailureLogic | RecommendedThreshold | ReliableSignals | RejectedSignals | RecommendedNextStep | Reason"]
    lines.append(f"{rec.classification} | {rec.logic} | >={rec.threshold} | {rec.reliable_signals} | {rec.rejected_signals} | {rec.next_step} | {rec.reason}")


def rankings(result: Result) -> list[str]:
    shifts = sorted(result.shifts, key=lambda row: abs(row.effect_size), reverse=True)
    thresholds = sorted(result.thresholds, key=lambda row: (row.recall - row.false_positive, row.precision), reverse=True)
    easy_dest = sorted(result.destinations, key=lambda row: row.detection_rate, reverse=True)
    hard_dest = sorted(result.destinations, key=lambda row: row.detection_rate)
    return [
        "",
        "RANKINGS",
        "1. Strongest causal failure signals: " + "; ".join(f"{row.signal}@{row.age_bucket}/t-{row.horizon}={fmt(row.effect_size)}" for row in shifts[:8]),
        "2. Most replicated failure signals: " + "; ".join(f"{row.signal}@t-{row.horizon}=R{row.replication_count}" for row in sorted(result.shifts, key=lambda row: row.replication_count, reverse=True)[:8]),
        "3. Best warning thresholds: " + "; ".join(f">={row.threshold}=Recall{pct(row.recall)}/FP{pct(row.false_positive)}" for row in thresholds),
        "4. Best age-specific warning profiles: " + "; ".join(f"{row.age_bucket}>={row.threshold}=R{pct(row.recall)}" for row in sorted(result.age_thresholds, key=lambda row: row.recall, reverse=True)[:8]),
        "5. Easiest destination failures to detect: " + "; ".join(f"{row.destination}={pct(row.detection_rate)}" for row in easy_dest[:5]),
        "6. Hardest destination failures to detect: " + "; ".join(f"{row.destination}={pct(row.detection_rate)}" for row in hard_dest[:5]),
        "7. Most common false positive patterns: " + "; ".join(f"{row.age_bucket}:{row.signal_pattern}={row.count}" for row in result.false_positives[:5]),
        "8. Most common missed failure patterns: " + "; ".join(f"{row.failure_node}->{row.destination}={row.count}" for row in result.missed[:5]),
        "9. Recommended failure timing logic: " + result.recommendation.classification,
        "10. Recommended revised real-time state machine: NeutralFailure as warning overlay, not mandatory hard phase.",
        "",
        "RESEARCH NOTES",
        "Mechanical only.",
        "Is NeutralFailure a phase or a warning condition? " + result.recommendation.classification,
        "Can real-time failure detection be improved without lookahead? See threshold performance and hard-vs-warning tables.",
        "Can APVA proceed to a real trade timing audit? " + result.recommendation.next_step,
    ]


def write_report(result: Result, out_root: Path, aggregate: bool = False) -> None:
    if aggregate:
        path = out_root / "NeutralFailureTiming83" / "NeutralFailureTiming83_All.txt"
        title = "APVA Neutral Failure Timing Study v0.1 - Aggregate"
    else:
        path = out_root / result.instrument / f"NeutralFailureTiming83_{result.instrument}.txt"
        title = f"APVA Neutral Failure Timing Study v0.1 - {result.instrument}"
    ensure_dir(path.parent)
    lines = [
        title,
        "=" * 96,
        f"Instrument: {result.instrument}",
        "Input path(s): " + "; ".join(str(path) for path in result.source_paths),
        f"ReferenceFailureCount: {len(result.failures)}",
        f"StableControlCount: {len(result.controls)}",
        "Purpose: Test causal NeutralFailure timing as a warning score overlay.",
    ]
    append_common(lines, result)
    lines += [
        "",
        "14. Low-DoF Audit",
        "No future data used.",
        "No NextNode used.",
        "No future StructuralState used.",
        "No completed-loop hindsight used as predictor.",
        "No forward returns used in warning construction.",
        "No optimization.",
        "No fitting.",
        "No machine learning.",
        "No trading rules.",
    ]
    lines += rankings(result)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(result: Result) -> None:
    if not result.failures:
        raise RuntimeError(f"{result.instrument}: no reference failures.")
    if not result.controls:
        raise RuntimeError(f"{result.instrument}: no stable neutral controls.")
    if not result.thresholds:
        raise RuntimeError(f"{result.instrument}: no threshold performance rows.")
    if not result.recommendation.classification:
        raise RuntimeError(f"{result.instrument}: missing recommendation.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Evidence CSV files or directories")
    parser.add_argument("--out-root", default="Evidence/Output", help="Output root directory")
    return parser.parse_args()


def load_contexts(inputs: list[str]) -> tuple[list, list]:
    loaded = load_results(inputs)
    decay = [decay_study(row) for row in loaded]
    aggregate_node_rows = aggregate_rows(decay)
    score_rows(aggregate_node_rows)
    aggregate_thresholds = thresholds(loaded, aggregate_node_rows)
    local_node_rows = []
    for decay_row in decay:
        rows = local_rows(decay_row)
        score_rows(rows)
        local_node_rows.append(rows)
    return loaded, build_contexts(loaded, local_node_rows, aggregate_thresholds)


def main() -> None:
    args = parse_args()
    loaded, contexts = load_contexts(args.inputs)
    instrument_results = []
    for context in contexts:
        result = build_result(context.instrument, context.source_paths, [context])
        validate(result)
        instrument_results.append(result)

    aggregate_source_paths = [path for context in contexts for path in context.source_paths]
    aggregate = build_result("ALL", aggregate_source_paths, contexts, instrument_results)
    aggregate.replication = replication_rows(instrument_results)
    validate(aggregate)

    out_root = Path(args.out_root)
    for result in instrument_results:
        result.replication = replication_rows([result])
        write_report(result, out_root)
    write_report(aggregate, out_root, aggregate=True)


if __name__ == "__main__":
    main()
