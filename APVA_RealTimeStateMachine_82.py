#!/usr/bin/env python3
"""APVA Real-Time State Machine Study v0.1.

Study 82 asks whether APVA can be represented as a deterministic real-time
state machine using only current and past-bar information.

Research only. No trading, optimization, fitting, machine learning, parameter
search, future NextNode predictors, completed-loop hindsight, or forward returns
in phase assignment.
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
from APVA_LoopClosure_80 import (
    PHASE_MACHINE,
    build_loops,
    complete_loops,
    loop_phases,
    machine_phase,
)
from APVA_MinimalEngine_52 import load_results
from APVA_NeutralFailureModes_75 import directional_forward, mean, safe_div
from APVA_NodeImportance_58 import aggregate_rows, local_rows, score_rows
from APVA_ProcessGraph_54 import Node, node_for, node_text
from APVA_SignalMinimalism_81 import bucket_numeric
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

OUTCOME_HORIZONS = (1, 3, 5, 10)
NEUTRAL = "NeutralProcessing"
UNKNOWN = "Unlabeled"

REAL_TIME_PHASES = (
    "NeutralFormation",
    "NeutralMaturation",
    "LateNeutral",
    "NeutralFailure",
    "DestinationSelection",
    "Excursion",
    "ReturnToNeutral",
)

RULES = (
    ("NeutralFormation", "Current StructuralState is NeutralProcessing and AgeBucket is 1."),
    ("NeutralMaturation", "Current StructuralState is NeutralProcessing and AgeBucket is 2, 3, or 4 unless the causal failure signal fires."),
    ("LateNeutral", "Current StructuralState is NeutralProcessing and AgeBucket is 5, 6-10, 11-20, or 21+ unless the causal failure signal fires."),
    ("NeutralFailure", "Current bar is NeutralProcessing and shows causal failure pressure: range expansion, body expansion, or volume expansion while in Age4+ or after LateNeutral."),
    ("DestinationSelection", "Current StructuralState is non-neutral and prior real-time phase was NeutralFailure or LateNeutral."),
    ("Excursion", "Current StructuralState is non-neutral after a destination has already appeared."),
    ("ReturnToNeutral", "Current StructuralState is non-neutral and current StateAgeNode is DecayToNeutral or the next real-time transition is expected to reset from excursion context."),
)

MINIMAL_STUDY81 = (
    "LoopPhase",
    "NeutralAgeBucket",
    "DestinationFamily",
    "BodyRelativeToPrevious",
    "VolumeRelativeToRollingMean",
    "CloseLocation",
    "EfficiencyRatio",
    "VolumeRelativeToPrevious",
    "RangeRelativeToPrevious",
    "VolumeRelativeToSessionMean",
)

PRACTICAL_SET = (
    "LoopPhase",
    "StructuralState",
    "AgeBucket",
    "RangeRelativeToPrevious",
    "VolumeRelativeToRollingMean",
    "EfficiencyRatio",
    "CloseLocation",
)


@dataclass
class BarRecord:
    instrument: str
    index: int
    node: Node
    previous_node: Node | None
    reference_phase: str
    realtime_phase: str
    practical_phase: str
    destination_reference: str
    destination_realtime: str
    signal_values: dict[str, str]


@dataclass
class AgreementRow:
    scope: str
    count: int
    top1: float
    top2: float


@dataclass
class PhaseMetricRow:
    phase: str
    precision: float
    recall: float
    support: int


@dataclass
class TransitionRow:
    from_phase: str
    to_phase: str
    count: int
    probability: float
    replication_count: int


@dataclass
class FailureDetectionRow:
    scope: str
    reference_failures: int
    realtime_failures: int
    detected: int
    early: int
    late: int
    false_positive: int
    detection_rate: float
    early_rate: float
    late_rate: float
    false_positive_rate: float


@dataclass
class DestinationMetricRow:
    destination: str
    precision: float
    recall: float
    support: int


@dataclass
class MinimalComparisonRow:
    signal_set: str
    agreement: float
    confusion_entropy: float
    phases_observed: int


@dataclass
class InstrumentReplicationRow:
    instrument: str
    agreement: float
    failure_detection: float
    destination_accuracy: float
    top_confusion: str
    recommendation: str


@dataclass
class OutcomeRow:
    phase: str
    dr: dict[int, float] = field(default_factory=dict)
    continuation5: float = 0.0
    failure5: float = 0.0
    flat5: float = 0.0
    count: int = 0


@dataclass
class Recommendation:
    classification: str
    next_step: str
    reliable_phases: str
    ambiguous_phases: str
    reason: str


@dataclass
class Result:
    instrument: str
    source_paths: list
    records: list[BarRecord]
    phase_inventory: Counter
    rule_rows: list[tuple[str, str]]
    agreement: AgreementRow
    confusion: Counter
    phase_metrics: list[PhaseMetricRow]
    transitions: list[TransitionRow]
    failure_detection: FailureDetectionRow
    destination_accuracy: float
    destination_confusion: Counter
    destination_metrics: list[DestinationMetricRow]
    minimal_comparison: list[MinimalComparisonRow]
    replication: list[InstrumentReplicationRow]
    outcomes: list[OutcomeRow]
    recommendation: Recommendation


def normalized_entropy(counter: Counter) -> float:
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


def is_neutral(node: Node) -> bool:
    return node[0] == NEUTRAL


def age_bucket(node: Node) -> str:
    return node[1]


def age_rank(bucket: str) -> int:
    return {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6-10": 6, "11-20": 7, "21+": 8}.get(bucket, 0)


def is_non_neutral(node: Node) -> bool:
    return node[0] in DESTINATION_FAMILIES


def phase_neighbors(phase: str) -> set[str]:
    order = list(REAL_TIME_PHASES)
    if phase not in order:
        return {phase}
    index = order.index(phase)
    neighbors = {phase}
    if index > 0:
        neighbors.add(order[index - 1])
    if index < len(order) - 1:
        neighbors.add(order[index + 1])
    if phase == "ReturnToNeutral":
        neighbors.add("NeutralFormation")
    if phase == "NeutralFormation":
        neighbors.add("ReturnToNeutral")
    return neighbors


def failure_pressure(bar, node: Node, prior_phase: str | None) -> bool:
    if not is_neutral(node):
        return False
    mature = age_rank(age_bucket(node)) >= 4 or prior_phase in {"LateNeutral", "NeutralFailure"}
    if not mature:
        return False
    range_up = (bar.range_relative_to_previous or 0.0) > 1.0
    body_up = (bar.body_relative_to_previous or 0.0) > 1.0
    volume_up = (bar.volume_relative_to_rolling_mean or bar.volume_relative_to_previous or 0.0) > 1.0
    efficient = (efficiency_ratio(bar) or 0.0) >= 0.66
    return (range_up and (body_up or volume_up)) or (volume_up and efficient and age_rank(age_bucket(node)) >= 5)


def realtime_phase(node: Node, previous_node: Node | None, bar, prior_phase: str | None, in_excursion: bool) -> str:
    if is_neutral(node):
        if age_bucket(node) == "1":
            return "NeutralFormation"
        if failure_pressure(bar, node, prior_phase):
            return "NeutralFailure"
        if age_bucket(node) in {"2", "3", "4"}:
            return "NeutralMaturation"
        return "LateNeutral"
    if is_non_neutral(node):
        if node[0] == "DecayToNeutral" or prior_phase == "ReturnToNeutral":
            return "ReturnToNeutral"
        if not previous_node or is_neutral(previous_node) or prior_phase in {"NeutralFailure", "LateNeutral", "NeutralMaturation"}:
            return "DestinationSelection"
        return "Excursion" if in_excursion else "DestinationSelection"
    return "Excursion"


def practical_phase(node: Node, previous_node: Node | None, bar, prior_phase: str | None, in_excursion: bool) -> str:
    if is_neutral(node):
        if age_bucket(node) == "1":
            return "NeutralFormation"
        range_up = (bar.range_relative_to_previous or 0.0) > 1.0
        volume_up = (bar.volume_relative_to_rolling_mean or 0.0) > 1.0
        efficient = (efficiency_ratio(bar) or 0.0) >= 0.66
        if age_rank(age_bucket(node)) >= 4 and range_up and (volume_up or efficient):
            return "NeutralFailure"
        if age_bucket(node) in {"2", "3", "4"}:
            return "NeutralMaturation"
        return "LateNeutral"
    if is_non_neutral(node):
        if node[0] == "DecayToNeutral":
            return "ReturnToNeutral"
        if not previous_node or is_neutral(previous_node) or prior_phase in {"NeutralFailure", "LateNeutral"}:
            return "DestinationSelection"
        return "Excursion" if in_excursion else "DestinationSelection"
    return "Excursion"


def signal_values(bar, node: Node, prior_phase: str | None) -> dict[str, str]:
    return {
        "LoopPhase": prior_phase or "None",
        "StructuralState": node[0],
        "AgeBucket": node[1],
        "NeutralAgeBucket": node[1] if is_neutral(node) else "NonNeutral",
        "DestinationFamily": node[0] if is_non_neutral(node) else "None",
        "RangeRelativeToPrevious": bucket_numeric(bar.range_relative_to_previous, "relative"),
        "BodyRelativeToPrevious": bucket_numeric(bar.body_relative_to_previous, "relative"),
        "VolumeRelativeToRollingMean": bucket_numeric(bar.volume_relative_to_rolling_mean, "relative"),
        "VolumeRelativeToPrevious": bucket_numeric(bar.volume_relative_to_previous, "relative"),
        "VolumeRelativeToSessionMean": bucket_numeric(bar.volume_relative_to_session_mean, "relative"),
        "EfficiencyRatio": bucket_numeric(efficiency_ratio(bar), "efficiency"),
        "CloseLocation": bucket_numeric(bar.close_location, "close"),
        "VolumePolarity": getattr(bar, "volume_polarity", None) or "Missing",
    }


def reference_records_for_context(context) -> list[tuple[int, str]]:
    labels = {}
    for loop in complete_loops(build_loops([context])):
        phases = loop_phases(loop)
        for offset, phase in enumerate(phases):
            index = loop.start + offset
            labels[index] = machine_phase(phase)
    return sorted(labels.items())


def build_records(contexts: list) -> list[BarRecord]:
    records = []
    for context in contexts:
        reference = dict(reference_records_for_context(context))
        prior_rt = None
        prior_practical = None
        in_excursion = False
        practical_in_excursion = False
        for index, bar in enumerate(context.bars):
            if index not in reference:
                continue
            node = context.nodes[index]
            previous_node = context.nodes[index - 1] if index > 0 else None
            rt = realtime_phase(node, previous_node, bar, prior_rt, in_excursion)
            practical = practical_phase(node, previous_node, bar, prior_practical, practical_in_excursion)
            if rt == "DestinationSelection":
                in_excursion = True
            if rt == "NeutralFormation":
                in_excursion = False
            if practical == "DestinationSelection":
                practical_in_excursion = True
            if practical == "NeutralFormation":
                practical_in_excursion = False
            destination_ref = node[0] if reference[index] == "DestinationSelection" and is_non_neutral(node) else "None"
            destination_rt = node[0] if rt == "DestinationSelection" and is_non_neutral(node) else "None"
            records.append(BarRecord(
                context.instrument,
                index,
                node,
                previous_node,
                reference[index],
                rt,
                practical,
                destination_ref,
                destination_rt,
                signal_values(bar, node, prior_rt),
            ))
            prior_rt = rt
            prior_practical = practical
    return records


def agreement_row(scope: str, records: list[BarRecord], attr: str = "realtime_phase") -> AgreementRow:
    if not records:
        return AgreementRow(scope, 0, 0.0, 0.0)
    top1 = sum(1 for row in records if getattr(row, attr) == row.reference_phase)
    top2 = sum(1 for row in records if row.reference_phase in phase_neighbors(getattr(row, attr)))
    return AgreementRow(scope, len(records), top1 / len(records), top2 / len(records))


def confusion_matrix(records: list[BarRecord], attr: str = "realtime_phase") -> Counter:
    return Counter((row.reference_phase, getattr(row, attr)) for row in records)


def phase_metric_rows(records: list[BarRecord], attr: str = "realtime_phase") -> list[PhaseMetricRow]:
    rows = []
    for phase in REAL_TIME_PHASES:
        tp = sum(1 for row in records if row.reference_phase == phase and getattr(row, attr) == phase)
        predicted = sum(1 for row in records if getattr(row, attr) == phase)
        actual = sum(1 for row in records if row.reference_phase == phase)
        rows.append(PhaseMetricRow(phase, safe_div(tp, predicted) or 0.0, safe_div(tp, actual) or 0.0, actual))
    return rows


def transition_rows(records: list[BarRecord], instrument_results: list[Result] | None = None) -> list[TransitionRow]:
    counts = Counter()
    instruments = defaultdict(set)
    by_instrument = defaultdict(list)
    for row in records:
        by_instrument[row.instrument].append(row)
    for instrument, rows in by_instrument.items():
        rows.sort(key=lambda item: item.index)
        for left, right in zip(rows, rows[1:]):
            if left.realtime_phase == right.realtime_phase:
                continue
            key = (left.realtime_phase, right.realtime_phase)
            counts[key] += 1
            instruments[key].add(instrument)
    output = []
    totals = Counter()
    for (left, _), count in counts.items():
        totals[left] += count
    for key, count in counts.most_common():
        output.append(TransitionRow(key[0], key[1], count, safe_div(count, totals[key[0]]) or 0.0, len(instruments[key])))
    return output


def failure_detection_row(scope: str, records: list[BarRecord]) -> FailureDetectionRow:
    by_instrument = defaultdict(list)
    for row in records:
        by_instrument[row.instrument].append(row)
    reference = realtime = detected = early = late = false_positive = 0
    for rows in by_instrument.values():
        rows.sort(key=lambda item: item.index)
        ref_indexes = {row.index for row in rows if row.reference_phase == "NeutralFailure"}
        rt_indexes = {row.index for row in rows if row.realtime_phase == "NeutralFailure"}
        reference += len(ref_indexes)
        realtime += len(rt_indexes)
        for idx in ref_indexes:
            if any(candidate in rt_indexes for candidate in (idx - 1, idx)):
                detected += 1
                if idx - 1 in rt_indexes:
                    early += 1
                else:
                    late += 1
        matched_rt = set()
        for idx in ref_indexes:
            matched_rt.update(candidate for candidate in (idx - 1, idx) if candidate in rt_indexes)
        false_positive += len(rt_indexes - matched_rt)
    total_non_failure = max(len(records) - reference, 1)
    return FailureDetectionRow(
        scope,
        reference,
        realtime,
        detected,
        early,
        late,
        false_positive,
        safe_div(detected, reference) or 0.0,
        safe_div(early, reference) or 0.0,
        safe_div(late, reference) or 0.0,
        safe_div(false_positive, total_non_failure) or 0.0,
    )


def destination_accuracy(records: list[BarRecord]) -> tuple[float, Counter, list[DestinationMetricRow]]:
    destination_rows = [row for row in records if row.reference_phase == "DestinationSelection" and row.destination_reference != "None"]
    if not destination_rows:
        return 0.0, Counter(), []
    confusion = Counter((row.destination_reference, row.destination_realtime) for row in destination_rows)
    correct = sum(1 for row in destination_rows if row.destination_reference == row.destination_realtime)
    metrics = []
    for destination in DESTINATION_FAMILIES:
        tp = confusion[(destination, destination)]
        predicted = sum(count for (actual, pred), count in confusion.items() if pred == destination)
        actual = sum(count for (act, pred), count in confusion.items() if act == destination)
        metrics.append(DestinationMetricRow(destination, safe_div(tp, predicted) or 0.0, safe_div(tp, actual) or 0.0, actual))
    return correct / len(destination_rows), confusion, metrics


def minimal_comparison_rows(records: list[BarRecord]) -> list[MinimalComparisonRow]:
    rows = []
    for name, attr in (("Study81MinimalSignals", "realtime_phase"), ("PracticalRealTimeSet", "practical_phase")):
        confusion = confusion_matrix(records, attr)
        agree = agreement_row(name, records, attr)
        predicted = Counter(getattr(row, attr) for row in records)
        rows.append(MinimalComparisonRow(name, agree.top1, normalized_entropy(confusion), len(predicted)))
    return rows


def instrument_replication_rows(results: list[Result]) -> list[InstrumentReplicationRow]:
    rows = []
    for result in results:
        common = result.confusion.most_common(1)
        top_confusion = f"{common[0][0][0]}->{common[0][0][1]}" if common else "None"
        if result.agreement.top1 >= 0.75 and result.failure_detection.detection_rate >= 0.50:
            label = "Replicated"
        elif result.agreement.top1 >= 0.60:
            label = "Partial"
        else:
            label = "Weak"
        rows.append(InstrumentReplicationRow(
            result.instrument,
            result.agreement.top1,
            result.failure_detection.detection_rate,
            result.destination_accuracy,
            top_confusion,
            label,
        ))
    return rows


def outcome_rows(records: list[BarRecord], contexts: list) -> list[OutcomeRow]:
    lookup = {context.instrument: context for context in contexts}
    rows = []
    for phase in REAL_TIME_PHASES:
        phase_records = [row for row in records if row.realtime_phase == phase]
        out = OutcomeRow(phase)
        values5 = []
        for horizon in OUTCOME_HORIZONS:
            values = []
            for record in phase_records:
                context = lookup[record.instrument]
                value = directional_forward(context.bars, record.index, horizon)
                if value is not None:
                    values.append(value)
                    if horizon == 5:
                        values5.append(value)
            out.dr[horizon] = mean(values)
        out.count = len(values5)
        out.continuation5 = safe_div(sum(1 for value in values5 if value > 0), len(values5)) or 0.0
        out.failure5 = safe_div(sum(1 for value in values5 if value < 0), len(values5)) or 0.0
        out.flat5 = safe_div(sum(1 for value in values5 if value == 0), len(values5)) or 0.0
        rows.append(out)
    return rows


def make_recommendation(result: Result) -> Recommendation:
    reliable = [row.phase for row in result.phase_metrics if row.precision >= 0.70 and row.recall >= 0.50]
    ambiguous = [row.phase for row in result.phase_metrics if row.phase not in reliable]
    if result.agreement.top1 >= 0.80 and result.failure_detection.detection_rate >= 0.60:
        classification = "StrongRealTimeMachine"
        next_step = "ProceedToTradeTimingAudit"
    elif result.agreement.top1 >= 0.65 and result.destination_accuracy >= 0.75:
        classification = "PracticalRealTimeMachine"
        next_step = "ProceedToTradeTimingAudit"
    elif result.agreement.top1 >= 0.50:
        classification = "WeakRealTimeMachine"
        next_step = "RevisePhaseRules"
    else:
        classification = "NoRealTimeMachine"
        next_step = "RevisePhaseRules"
    reason = (
        f"Top1 causal agreement is {pct(result.agreement.top1)}, destination accuracy is "
        f"{pct(result.destination_accuracy)}, and failure detection is {pct(result.failure_detection.detection_rate)}."
    )
    return Recommendation(classification, next_step, ", ".join(reliable) or "None", ", ".join(ambiguous) or "None", reason)


def build_result(instrument: str, source_paths: list, contexts: list, instrument_results: list[Result] | None = None) -> Result:
    records = build_records(contexts)
    inventory = Counter(row.reference_phase for row in records)
    agreement = agreement_row("Top1/Top2", records)
    confusion = confusion_matrix(records)
    metrics = phase_metric_rows(records)
    transitions = transition_rows(records, instrument_results)
    failures = failure_detection_row(instrument, records)
    dest_accuracy, dest_confusion, dest_metrics = destination_accuracy(records)
    minimal = minimal_comparison_rows(records)
    outcomes = outcome_rows(records, contexts)
    result = Result(
        instrument,
        source_paths,
        records,
        inventory,
        list(RULES),
        agreement,
        confusion,
        metrics,
        transitions,
        failures,
        dest_accuracy,
        dest_confusion,
        dest_metrics,
        minimal,
        [],
        outcomes,
        Recommendation("", "", "", "", ""),
    )
    result.recommendation = make_recommendation(result)
    return result


def append_common(lines: list[str], result: Result) -> None:
    lines += ["", "1. Reference Phase Inventory", "LoopPhase | Count | Percent"]
    total = sum(result.phase_inventory.values())
    for phase in REAL_TIME_PHASES:
        count = result.phase_inventory[phase]
        lines.append(f"{phase} | {count} | {pct(safe_div(count, total) or 0.0)}")

    lines += ["", "2. Real-Time Rule Table", "AssignedPhase | DeterministicRule"]
    for phase, rule in result.rule_rows:
        lines.append(f"{phase} | {rule}")

    lines += ["", "3. Agreement Table", "Scope | Count | Top1Agreement | Top2Agreement"]
    lines.append(f"{result.agreement.scope} | {result.agreement.count} | {pct(result.agreement.top1)} | {pct(result.agreement.top2)}")

    lines += ["", "4. Confusion Matrix", "ReferencePhase | RealTimePhase | Count"]
    for (reference, realtime), count in result.confusion.most_common():
        lines.append(f"{reference} | {realtime} | {count}")

    lines += ["", "Per-Phase Precision / Recall", "Phase | Precision | Recall | Support"]
    for row in result.phase_metrics:
        lines.append(f"{row.phase} | {pct(row.precision)} | {pct(row.recall)} | {row.support}")

    lines += ["", "5. Transition Table", "FromRealTimePhase | ToRealTimePhase | Count | Probability | ReplicationCount"]
    for row in result.transitions:
        lines.append(f"{row.from_phase} | {row.to_phase} | {row.count} | {pct(row.probability)} | {row.replication_count}")

    lines += ["", "6. Failure Detection Table", "Scope | ReferenceFailures | RealTimeFailures | Detected | DetectionRate | EarlyDetectionRate | LateDetectionRate | FalsePositiveRate"]
    row = result.failure_detection
    lines.append(
        f"{row.scope} | {row.reference_failures} | {row.realtime_failures} | {row.detected} | "
        f"{pct(row.detection_rate)} | {pct(row.early_rate)} | {pct(row.late_rate)} | {pct(row.false_positive_rate)}"
    )

    lines += ["", "7. Destination Detection Table", "Metric | Value"]
    lines.append(f"DestinationAccuracy | {pct(result.destination_accuracy)}")
    lines += ["ReferenceDestination | RealTimeDestination | Count"]
    for (reference, realtime), count in result.destination_confusion.most_common():
        lines.append(f"{reference} | {realtime} | {count}")
    lines += ["Destination | Precision | Recall | Support"]
    for row in result.destination_metrics:
        lines.append(f"{row.destination} | {pct(row.precision)} | {pct(row.recall)} | {row.support}")

    lines += ["", "8. Minimal Signal Comparison", "SignalSet | Agreement | ConfusionEntropy | PhasesObserved"]
    for row in result.minimal_comparison:
        lines.append(f"{row.signal_set} | {pct(row.agreement)} | {fmt(row.confusion_entropy)} | {row.phases_observed}")

    lines += ["", "9. Cross-Instrument Replication", "Instrument | Agreement | FailureDetection | DestinationAccuracy | TopConfusion | ReplicationAssessment"]
    for row in result.replication:
        lines.append(
            f"{row.instrument} | {pct(row.agreement)} | {pct(row.failure_detection)} | "
            f"{pct(row.destination_accuracy)} | {row.top_confusion} | {row.recommendation}"
        )

    lines += ["", "10. Final State Machine Spec"]
    lines += [
        "States | " + ", ".join(REAL_TIME_PHASES),
        "Inputs | Current StructuralState, Current StateAgeNode, Current AgeBucket, Previous StructuralState, Previous StateAgeNode, prior LoopPhase, real OHLCV-derived relative metrics, EfficiencyRatio, CloseLocation, VolumePolarity",
        "Transitions | NeutralFormation -> NeutralMaturation -> LateNeutral -> NeutralFailure -> DestinationSelection -> Excursion -> ReturnToNeutral -> NeutralFormation",
        "Entry conditions | NeutralFormation enters on NeutralProcessing_Age1; DestinationSelection enters on first non-neutral bar after neutral context.",
        "Exit conditions | Neutral phases exit through causal failure pressure or observed non-neutral state; excursions exit through DecayToNeutral or NeutralProcessing_Age1 reset.",
        "Failure conditions | Mature Neutral with range/body/volume expansion pressure using current and past-bar metrics only.",
        "NinjaTrader portability | Rules are deterministic, causal, and require only current/past bar APVA state plus OHLCV-derived metrics.",
    ]

    lines += ["", "11. Outcome Diagnostics", "RealTimePhase | Count | DRFwd1 | DRFwd3 | DRFwd5 | DRFwd10 | ContinuationRate5 | FailureRate5 | FlatRate5"]
    for row in result.outcomes:
        lines.append(
            f"{row.phase} | {row.count} | {fmt(row.dr.get(1, 0.0))} | {fmt(row.dr.get(3, 0.0))} | "
            f"{fmt(row.dr.get(5, 0.0))} | {fmt(row.dr.get(10, 0.0))} | {pct(row.continuation5)} | "
            f"{pct(row.failure5)} | {pct(row.flat5)}"
        )

    lines += ["", "12. Recommendation", "Classification | RecommendedNextStep | ReliablePhases | AmbiguousPhases | Reason"]
    rec = result.recommendation
    lines.append(f"{rec.classification} | {rec.next_step} | {rec.reliable_phases} | {rec.ambiguous_phases} | {rec.reason}")


def rankings(result: Result) -> list[str]:
    reliable = sorted(result.phase_metrics, key=lambda row: row.precision + row.recall, reverse=True)
    weak = sorted(result.phase_metrics, key=lambda row: row.precision + row.recall)
    transitions = result.transitions[:5]
    return [
        "",
        "RESEARCH NOTES",
        "Main question: Can APVA move from retrospective study framework to causal real-time engine?",
        "Mechanical answer: " + result.recommendation.classification,
        "No lookahead, no future NextNode predictor, no completed-loop hindsight, and no forward returns were used in phase assignment.",
        "",
        "RANKINGS",
        "1. Most reliable phases: " + "; ".join(f"{row.phase}=P{pct(row.precision)}/R{pct(row.recall)}" for row in reliable[:5]),
        "2. Most ambiguous phases: " + "; ".join(f"{row.phase}=P{pct(row.precision)}/R{pct(row.recall)}" for row in weak[:5]),
        "3. Strongest transitions: " + "; ".join(f"{row.from_phase}->{row.to_phase}={row.count}" for row in transitions),
        "4. Destination recall leaders: " + "; ".join(f"{row.destination}={pct(row.recall)}" for row in sorted(result.destination_metrics, key=lambda item: item.recall, reverse=True)[:5]),
        "5. Minimal signal comparison: " + "; ".join(f"{row.signal_set}={pct(row.agreement)}" for row in result.minimal_comparison),
    ]


def write_report(result: Result, out_root: Path, aggregate: bool = False) -> None:
    if aggregate:
        path = out_root / "RealTimeStateMachine82" / "RealTimeStateMachine82_All.txt"
        title = "APVA Real-Time State Machine Study v0.1 - Aggregate"
    else:
        path = out_root / result.instrument / f"RealTimeStateMachine82_{result.instrument}.txt"
        title = f"APVA Real-Time State Machine Study v0.1 - {result.instrument}"
    ensure_dir(path.parent)
    lines = [
        title,
        "=" * 96,
        f"Instrument: {result.instrument}",
        "Input path(s): " + "; ".join(str(path) for path in result.source_paths),
        f"Rows with offline reference labels: {len(result.records)}",
        "Purpose: Test whether APVA loop phases can be assigned causally in real time.",
    ]
    append_common(lines, result)
    lines += [
        "",
        "13. Low-DoF Audit",
        "No future data used in real-time classification.",
        "No NextNode as predictor.",
        "No completed-loop hindsight as predictor.",
        "No optimization.",
        "No fitting.",
        "No machine learning.",
        "No trading rules.",
        "No forward returns used in phase assignment.",
    ]
    lines += rankings(result)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(result: Result) -> None:
    if not result.records:
        raise RuntimeError(f"{result.instrument}: no records with reference labels.")
    if not result.rule_rows:
        raise RuntimeError(f"{result.instrument}: missing real-time rules.")
    if not result.transitions:
        raise RuntimeError(f"{result.instrument}: missing transition table.")
    if result.agreement.top1 < 0 or result.agreement.top1 > 1:
        raise RuntimeError(f"{result.instrument}: invalid agreement.")
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
    instrument_results = []
    for context in contexts:
        result = build_result(context.instrument, context.source_paths, [context])
        validate(result)
        instrument_results.append(result)

    aggregate_source_paths = [path for context in contexts for path in context.source_paths]
    aggregate_result = build_result("ALL", aggregate_source_paths, contexts, instrument_results)
    aggregate_result.replication = instrument_replication_rows(instrument_results)
    validate(aggregate_result)

    out_root = Path(args.out_root)
    for result in instrument_results:
        result.replication = instrument_replication_rows([result])
        write_report(result, out_root)
    write_report(aggregate_result, out_root, aggregate=True)


if __name__ == "__main__":
    main()
