#!/usr/bin/env python3
"""APVA Neutral Formation Study v0.1.

Study 79 asks what market and state conditions precede entry into
NeutralProcessing_Age1 from non-neutral families.

Research only. No trading, optimization, fitting, machine learning, or forward
returns in formation construction.
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
from APVA_NeutralFailureModes_75 import directional_forward, mean, median, pooled_stdev, safe_div, stdev
from APVA_NodeImportance_58 import aggregate_rows, local_rows, score_rows
from APVA_ProcessGraph_54 import Node, node_for, node_text
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

NEUTRAL_AGE1 = ("NeutralProcessing", "1")
LOOKBACKS = (1, 2, 3, 5)
PATH_OFFSETS = (5, 3, 2, 1, 0)
OUTCOME_HORIZONS = (1, 3, 5, 10)
MATERIAL_EFFECT = 0.25

SIGNALS = (
    "RangeRelativeToPrevious",
    "TrueRangeRelativeToPrevious",
    "BodyRelativeToPrevious",
    "VolumeRelativeToRollingMean",
    "VolumeRelativeToPrevious",
    "VolumeRelativeToSessionMean",
    "EfficiencyRatio",
    "CloseLocation",
    "MemoryStrength",
    "BranchEntropy",
)

MARKET_SIGNALS = SIGNALS[:8]
NORMALIZATION_SIGNALS = (
    "RangeRelativeToPrevious",
    "TrueRangeRelativeToPrevious",
    "BodyRelativeToPrevious",
    "VolumeRelativeToRollingMean",
    "EfficiencyRatio",
)


@dataclass
class FormationEvent:
    instrument: str
    index: int
    source_family: str
    source_node: Node


@dataclass
class InventoryRow:
    source_family: str
    count: int
    instrument_counts: Counter[str]
    probability: float
    replication_count: int


@dataclass
class PathRow:
    path: str
    count: int
    probability: float
    replication_count: int


@dataclass
class ProfileRow:
    source_family: str
    horizon: int
    signal: str
    mean_value: float
    median_value: float
    stddev_value: float


@dataclass
class ShiftRow:
    source_family: str
    signal: str
    horizon: int
    formation_mean: float
    control_mean: float
    delta_mean: float
    effect_size: float
    replication_count: int = 0


@dataclass
class ModeRow:
    source_family: str
    mode: str
    count: int
    percent: float
    replication_count: int


@dataclass
class SummaryRow:
    source_family: str
    dominant_mode: str
    secondary_mode: str
    top_signal: str
    replication_count: int


@dataclass
class SpecialRow:
    section: str
    metric: str
    value: str
    interpretation: str


@dataclass
class RecoveryExhaustionRow:
    metric: str
    recovery: str
    exhaustion: str
    difference: str


@dataclass
class DecayAuditRow:
    count: int
    replication_count: int
    mean_length_before_neutral: float
    market_profile: str


@dataclass
class TimingRow:
    source_family: str
    mean_bars: float
    median_bars: float
    max_bars: int


@dataclass
class NormalizationRow:
    source_family: str
    signal: str
    values: dict[int, float] = field(default_factory=dict)


@dataclass
class ReplicationRow:
    source_family: str
    signal: str
    effects: dict[str, float] = field(default_factory=dict)
    replication_count: int = 0


@dataclass
class OutcomeRow:
    source_family: str
    mode: str
    dr: dict[int, float] = field(default_factory=dict)
    continuation5: float = 0.0
    failure5: float = 0.0
    flat5: float = 0.0


@dataclass
class Recommendation:
    classification: str
    dominant_process: str
    source_processes: str
    study78_assessment: str
    reason: str


@dataclass
class Result:
    instrument: str
    source_paths: list
    events: list[FormationEvent]
    controls: list[FormationEvent]
    inventory: dict[str, InventoryRow]
    paths: list[PathRow]
    profiles: list[ProfileRow]
    shifts: list[ShiftRow]
    modes: list[ModeRow]
    summaries: list[SummaryRow]
    special: list[SpecialRow]
    recovery_exhaustion: list[RecoveryExhaustionRow]
    decay_audit: DecayAuditRow
    timing: list[TimingRow]
    normalization: list[NormalizationRow]
    replication: list[ReplicationRow]
    outcomes: list[OutcomeRow]
    recommendation: Recommendation


def is_non_neutral_family(family: str) -> bool:
    return family in DESTINATION_FAMILIES


def signal_value(context, index: int, signal: str) -> float | None:
    if index < 0 or index >= len(context.bars):
        return None
    bar = context.bars[index]
    node = context.nodes[index]
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
    if signal == "MemoryStrength":
        return context.memory.get(node, 0.0)
    if signal == "BranchEntropy":
        return context.entropy.get(node, 0.0)
    return None


def find_events(contexts: list) -> list[FormationEvent]:
    events = []
    for context in contexts:
        for index in range(1, len(context.nodes)):
            previous = context.nodes[index - 1]
            current = context.nodes[index]
            if current == NEUTRAL_AGE1 and is_non_neutral_family(previous[0]):
                events.append(FormationEvent(context.instrument, index, previous[0], previous))
    return events


def find_controls(contexts: list) -> list[FormationEvent]:
    controls = []
    for context in contexts:
        for index, node in enumerate(context.nodes):
            if not is_non_neutral_family(node[0]):
                continue
            followed_by_neutral = False
            for step in (1, 2, 3):
                if index + step < len(context.nodes) and context.nodes[index + step] == NEUTRAL_AGE1:
                    followed_by_neutral = True
                    break
            if not followed_by_neutral:
                controls.append(FormationEvent(context.instrument, index, node[0], node))
    return controls


def context_lookup(contexts: list) -> dict[str, object]:
    return {context.instrument: context for context in contexts}


def event_context(contexts_by_instrument: dict[str, object], event: FormationEvent):
    return contexts_by_instrument[event.instrument]


def samples(events: list[FormationEvent], contexts_by_instrument: dict[str, object],
            source_family: str, signal: str, horizon: int) -> list[float]:
    values = []
    for event in events:
        if event.source_family != source_family:
            continue
        context = event_context(contexts_by_instrument, event)
        value = signal_value(context, event.index - horizon, signal)
        if value is not None:
            values.append(float(value))
    return values


def effect(formation_values: list[float], control_values: list[float]) -> float:
    if not formation_values or not control_values:
        return 0.0
    pooled = pooled_stdev(formation_values, control_values)
    return safe_div(mean(formation_values) - mean(control_values), pooled) or 0.0


def instrument_counts(events: list[FormationEvent], family: str) -> Counter[str]:
    return Counter(event.instrument for event in events if event.source_family == family)


def inventory_rows(events: list[FormationEvent]) -> dict[str, InventoryRow]:
    total = len(events)
    output = {}
    for family in DESTINATION_FAMILIES:
        counts = instrument_counts(events, family)
        count = sum(counts.values())
        output[family] = InventoryRow(
            family,
            count,
            counts,
            safe_div(count, total) or 0.0,
            sum(1 for value in counts.values() if value > 0),
        )
    return output


def path_rows(events: list[FormationEvent], contexts_by_instrument: dict[str, object]) -> list[PathRow]:
    counts = Counter()
    instruments = defaultdict(set)
    for event in events:
        context = event_context(contexts_by_instrument, event)
        parts = []
        for offset in PATH_OFFSETS:
            index = event.index - offset
            if 0 <= index < len(context.nodes):
                parts.append(node_text(context.nodes[index]))
        path = " -> ".join(parts)
        counts[path] += 1
        instruments[path].add(event.instrument)
    total = sum(counts.values())
    return [
        PathRow(path, count, safe_div(count, total) or 0.0, len(instruments[path]))
        for path, count in counts.most_common(100)
    ]


def profile_rows(events: list[FormationEvent], contexts_by_instrument: dict[str, object]) -> list[ProfileRow]:
    output = []
    for family in DESTINATION_FAMILIES:
        for horizon in LOOKBACKS:
            for signal in MARKET_SIGNALS:
                values = samples(events, contexts_by_instrument, family, signal, horizon)
                output.append(ProfileRow(family, horizon, signal, mean(values), median(values), stdev(values)))
    return output


def shift_rows(events: list[FormationEvent], controls: list[FormationEvent], contexts_by_instrument: dict[str, object]) -> list[ShiftRow]:
    output = []
    for family in DESTINATION_FAMILIES:
        for signal in SIGNALS:
            for horizon in LOOKBACKS:
                formation = samples(events, contexts_by_instrument, family, signal, horizon)
                control = samples(controls, contexts_by_instrument, family, signal, horizon)
                output.append(ShiftRow(
                    family,
                    signal,
                    horizon,
                    mean(formation),
                    mean(control),
                    mean(formation) - mean(control),
                    effect(formation, control),
                ))
    return output


def attach_replication(aggregate: Result, instrument_results: list[Result]) -> None:
    effect_map = defaultdict(dict)
    for result in instrument_results:
        for row in result.shifts:
            effect_map[(row.source_family, row.signal, row.horizon)][result.instrument] = row.effect_size
    for row in aggregate.shifts:
        effects = effect_map.get((row.source_family, row.signal, row.horizon), {})
        signs = [1 if value > 0 else -1 if value < 0 else 0 for value in effects.values() if abs(value) >= MATERIAL_EFFECT]
        row.replication_count = max(signs.count(1), signs.count(-1))
    aggregate.replication = []
    for (family, signal, _horizon), effects in effect_map.items():
        if _horizon != 1:
            continue
        signs = [1 if value > 0 else -1 if value < 0 else 0 for value in effects.values() if abs(value) >= MATERIAL_EFFECT]
        aggregate.replication.append(ReplicationRow(family, signal, effects, max(signs.count(1), signs.count(-1))))
    aggregate.modes = mode_rows(aggregate.events, aggregate.shifts, aggregate.contexts_by_instrument if hasattr(aggregate, "contexts_by_instrument") else {})
    aggregate.summaries = summary_rows(aggregate.modes, aggregate.shifts)
    aggregate.recommendation = make_recommendation(aggregate.summaries)


def mode_for_effects(family: str, effects: dict[str, float]) -> str:
    if family == "DecayToNeutral":
        return "DecayFormation"
    candidates = {signal: value for signal, value in effects.items() if abs(value) >= MATERIAL_EFFECT}
    if not candidates:
        return "MixedFormation"
    ordered = sorted(candidates.items(), key=lambda item: abs(item[1]), reverse=True)
    if len(ordered) > 1 and abs(abs(ordered[0][1]) - abs(ordered[1][1])) <= abs(ordered[0][1]) * 0.10:
        return "MixedFormation"
    signal, value = ordered[0]
    if signal in {"RangeRelativeToPrevious", "TrueRangeRelativeToPrevious", "BodyRelativeToPrevious"} and value < 0:
        return "RangeCollapseFormation"
    if signal.startswith("Volume") and value < 0:
        return "VolumeCollapseFormation"
    if signal == "EfficiencyRatio" and value < 0:
        return "EfficiencyCollapseFormation"
    if signal in {"RangeRelativeToPrevious", "TrueRangeRelativeToPrevious"} and value > 0:
        return "ExhaustionFormation"
    if signal.startswith("Volume") and value > 0:
        return "ExhaustionFormation"
    return "CompressionFormation"


def event_mode(event: FormationEvent, shift_lookup: dict[tuple[str, str, int], float]) -> str:
    effects = {signal: shift_lookup.get((event.source_family, signal, 1), 0.0) for signal in SIGNALS}
    return mode_for_effects(event.source_family, effects)


def mode_rows(events: list[FormationEvent], shifts: list[ShiftRow], contexts_by_instrument: dict[str, object]) -> list[ModeRow]:
    shift_lookup = {(row.source_family, row.signal, row.horizon): row.effect_size for row in shifts}
    grouped = defaultdict(Counter)
    instruments = defaultdict(lambda: defaultdict(set))
    for event in events:
        mode = event_mode(event, shift_lookup)
        grouped[event.source_family][mode] += 1
        instruments[event.source_family][mode].add(event.instrument)
    output = []
    for family in DESTINATION_FAMILIES:
        total = sum(grouped[family].values())
        for mode, count in grouped[family].most_common():
            output.append(ModeRow(family, mode, count, safe_div(count, total) or 0.0, len(instruments[family][mode])))
    return output


def summary_rows(modes: list[ModeRow], shifts: list[ShiftRow]) -> list[SummaryRow]:
    output = []
    modes_by_family = defaultdict(list)
    for row in modes:
        modes_by_family[row.source_family].append(row)
    shifts_by_family = defaultdict(list)
    for row in shifts:
        shifts_by_family[row.source_family].append(row)
    for family in DESTINATION_FAMILIES:
        mode_rows_for_family = sorted(modes_by_family[family], key=lambda row: row.count, reverse=True)
        dominant = mode_rows_for_family[0].mode if mode_rows_for_family else "None"
        secondary = mode_rows_for_family[1].mode if len(mode_rows_for_family) > 1 else "None"
        top = max(shifts_by_family[family], key=lambda row: abs(row.effect_size), default=None)
        output.append(SummaryRow(
            family,
            dominant,
            secondary,
            f"{top.signal}@t-{top.horizon}" if top else "None",
            max([row.replication_count for row in shifts_by_family[family]], default=0),
        ))
    return output


def special_rows(inventory: dict[str, InventoryRow], shifts: list[ShiftRow], summaries: list[SummaryRow]) -> list[SpecialRow]:
    sections = {
        "CompressionProcessing": "8. Compression To Neutral Table",
        "MixedStructure": "9. Mixed To Neutral Table",
        "ReassertionProcessing": "10. Reassertion To Neutral Table",
    }
    summary_lookup = {row.source_family: row for row in summaries}
    output = []
    for family, section in sections.items():
        top_shifts = sorted([row for row in shifts if row.source_family == family], key=lambda row: abs(row.effect_size), reverse=True)
        rows = [
            ("Count", str(inventory[family].count), "Formation sample size."),
            ("DominantMode", summary_lookup[family].dominant_mode, "Mechanical dominant formation process."),
            ("TopSignal", summary_lookup[family].top_signal, "Largest formation/control shift."),
        ]
        rows.extend((f"Signal{i+1}", f"{row.signal}@t-{row.horizon} {fmt(row.effect_size)}", "Ranked by absolute effect size.") for i, row in enumerate(top_shifts[:3]))
        output.extend(SpecialRow(section, metric, value, interp) for metric, value, interp in rows)
    return output


def recovery_exhaustion_rows(inventory: dict[str, InventoryRow], shifts: list[ShiftRow], summaries: list[SummaryRow]) -> list[RecoveryExhaustionRow]:
    summary = {row.source_family: row for row in summaries}
    def top_effect(family: str, signal: str) -> float:
        return max([row.effect_size for row in shifts if row.source_family == family and row.signal == signal], key=abs, default=0.0)
    rows = []
    metrics = {
        "Count": (inventory["RecoveryResolution"].count, inventory["ExhaustionPersistence"].count),
        "DominantMode": (summary["RecoveryResolution"].dominant_mode, summary["ExhaustionPersistence"].dominant_mode),
        "RangeEffect": (top_effect("RecoveryResolution", "RangeRelativeToPrevious"), top_effect("ExhaustionPersistence", "RangeRelativeToPrevious")),
        "VolumeEffect": (top_effect("RecoveryResolution", "VolumeRelativeToRollingMean"), top_effect("ExhaustionPersistence", "VolumeRelativeToRollingMean")),
        "EfficiencyEffect": (top_effect("RecoveryResolution", "EfficiencyRatio"), top_effect("ExhaustionPersistence", "EfficiencyRatio")),
    }
    for metric, (left, right) in metrics.items():
        if isinstance(left, str) or isinstance(right, str):
            diff = "Different" if left != right else "Same"
            rows.append(RecoveryExhaustionRow(metric, str(left), str(right), diff))
        else:
            rows.append(RecoveryExhaustionRow(metric, fmt(left), fmt(right), fmt(float(left) - float(right))))
    return rows


def mean_length_before_neutral(events: list[FormationEvent], family: str, contexts_by_instrument: dict[str, object]) -> float:
    lengths = []
    for event in events:
        if event.source_family != family:
            continue
        context = event_context(contexts_by_instrument, event)
        length = 0
        index = event.index - 1
        while index >= 0 and context.nodes[index][0] == family:
            length += 1
            index -= 1
        lengths.append(length)
    return mean(lengths)


def decay_audit(events: list[FormationEvent], inventory: dict[str, InventoryRow], contexts_by_instrument: dict[str, object], profiles: list[ProfileRow]) -> DecayAuditRow:
    decay_profiles = [row for row in profiles if row.source_family == "DecayToNeutral" and row.horizon == 1]
    top = sorted(decay_profiles, key=lambda row: abs(row.mean_value), reverse=True)[:3]
    summary = "; ".join(f"{row.signal}={fmt(row.mean_value)}" for row in top) or "No profile"
    row = inventory["DecayToNeutral"]
    return DecayAuditRow(row.count, row.replication_count, mean_length_before_neutral(events, "DecayToNeutral", contexts_by_instrument), summary)


def timing_rows(events: list[FormationEvent], contexts_by_instrument: dict[str, object]) -> list[TimingRow]:
    output = []
    for family in DESTINATION_FAMILIES:
        lengths = []
        for event in events:
            if event.source_family != family:
                continue
            context = event_context(contexts_by_instrument, event)
            length = 0
            index = event.index - 1
            while index >= 0 and is_non_neutral_family(context.nodes[index][0]):
                length += 1
                index -= 1
            lengths.append(length)
        output.append(TimingRow(family, mean(lengths), median(lengths), max(lengths) if lengths else 0))
    return output


def normalization_rows(events: list[FormationEvent], contexts_by_instrument: dict[str, object]) -> list[NormalizationRow]:
    output = []
    for family in DESTINATION_FAMILIES:
        for signal in NORMALIZATION_SIGNALS:
            values = {}
            for offset in PATH_OFFSETS:
                vals = []
                for event in events:
                    if event.source_family != family:
                        continue
                    context = event_context(contexts_by_instrument, event)
                    value = signal_value(context, event.index - offset, signal)
                    if value is not None:
                        vals.append(float(value))
                values[offset] = mean(vals)
            output.append(NormalizationRow(family, signal, values))
    return output


def outcome_rows(events: list[FormationEvent], contexts_by_instrument: dict[str, object], shifts: list[ShiftRow]) -> list[OutcomeRow]:
    shift_lookup = {(row.source_family, row.signal, row.horizon): row.effect_size for row in shifts}
    grouped = defaultdict(list)
    for event in events:
        grouped[(event.source_family, event_mode(event, shift_lookup))].append(event)
    output = []
    for (family, mode), rows_for_group in grouped.items():
        row = OutcomeRow(family, mode)
        for horizon in OUTCOME_HORIZONS:
            values = []
            for event in rows_for_group:
                context = event_context(contexts_by_instrument, event)
                value = directional_forward(context.bars, event.index, horizon)
                if value is not None:
                    values.append(value)
            row.dr[horizon] = mean(values)
            if horizon == 5:
                row.continuation5 = mean(1.0 if value > 0 else 0.0 for value in values)
                row.failure5 = mean(1.0 if value < 0 else 0.0 for value in values)
                row.flat5 = mean(1.0 if value == 0 else 0.0 for value in values)
        output.append(row)
    return output


def make_recommendation(summaries: list[SummaryRow]) -> Recommendation:
    families_with_modes = [row for row in summaries if row.dominant_mode != "None"]
    distinct_modes = {row.dominant_mode for row in families_with_modes}
    replicated = [row for row in families_with_modes if row.replication_count >= 2]
    if len(replicated) >= 5 and len(distinct_modes) >= 3:
        classification = "StrongFormationSeparation"
    elif len(replicated) >= 3:
        classification = "PartialFormationSeparation"
    elif replicated:
        classification = "WeakFormationSeparation"
    else:
        classification = "NoReliableFormationModel"
    dominant = Counter(row.dominant_mode for row in families_with_modes).most_common(1)
    process = dominant[0][0] if dominant else "None"
    source_processes = "; ".join(f"{row.source_family}:{row.dominant_mode}" for row in families_with_modes) or "None"
    study78 = "Study78Refined: this study identifies the pre-Neutral conditions completing the return loop."
    reason = "Classification uses fixed formation/control effects, replication, and source-family mode separation."
    return Recommendation(classification, process, source_processes, study78, reason)


def build_result(instrument: str, source_paths: list, contexts: list) -> Result:
    contexts_by_instrument = context_lookup(contexts)
    events = find_events(contexts)
    controls = find_controls(contexts)
    inventory = inventory_rows(events)
    paths = path_rows(events, contexts_by_instrument)
    profiles = profile_rows(events, contexts_by_instrument)
    shifts = shift_rows(events, controls, contexts_by_instrument)
    modes = mode_rows(events, shifts, contexts_by_instrument)
    summaries = summary_rows(modes, shifts)
    special = special_rows(inventory, shifts, summaries)
    rec_exh = recovery_exhaustion_rows(inventory, shifts, summaries)
    decay = decay_audit(events, inventory, contexts_by_instrument, profiles)
    timing = timing_rows(events, contexts_by_instrument)
    normalization = normalization_rows(events, contexts_by_instrument)
    outcomes = outcome_rows(events, contexts_by_instrument, shifts)
    recommendation = make_recommendation(summaries)
    result = Result(
        instrument,
        source_paths,
        events,
        controls,
        inventory,
        paths,
        profiles,
        shifts,
        modes,
        summaries,
        special,
        rec_exh,
        decay,
        timing,
        normalization,
        [],
        outcomes,
        recommendation,
    )
    result.contexts_by_instrument = contexts_by_instrument
    return result


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def append_common(lines: list[str], result: Result) -> None:
    lines += ["", "1. Neutral Formation Inventory", "SourceFamily | Count | Count_6E | Count_CL | Count_NQ | ProbabilityOfNeutralFormation | ReplicationCount"]
    for row in result.inventory.values():
        lines.append(f"{row.source_family} | {row.count} | {row.instrument_counts.get('6E', 0)} | {row.instrument_counts.get('CL', 0)} | {row.instrument_counts.get('NQ', 0)} | {pct(row.probability)} | {row.replication_count}")

    lines += ["", "2. Formation Path Table", "Path | Count | Probability | ReplicationCount"]
    for row in result.paths[:80]:
        lines.append(f"{row.path} | {row.count} | {pct(row.probability)} | {row.replication_count}")

    lines += ["", "3. Pre-Formation Market Profile", "SourceFamily | Horizon | Signal | Mean | Median | StdDev"]
    for row in result.profiles:
        lines.append(f"{row.source_family} | t-{row.horizon} | {row.signal} | {fmt(row.mean_value)} | {fmt(row.median_value)} | {fmt(row.stddev_value)}")

    lines += ["", "4. Formation vs Control Table", "SourceFamily | Signal | Horizon | FormationMean | ControlMean | DeltaMean | EffectSize | ReplicationCount"]
    for row in sorted(result.shifts, key=lambda item: abs(item.effect_size), reverse=True)[:240]:
        lines.append(f"{row.source_family} | {row.signal} | t-{row.horizon} | {fmt(row.formation_mean)} | {fmt(row.control_mean)} | {fmt(row.delta_mean)} | {fmt(row.effect_size)} | {row.replication_count}")

    lines += ["", "5. Formation Signal Ranking", "SourceFamily | Signal | Horizon | EffectSize | ReplicationCount"]
    for row in sorted(result.shifts, key=lambda item: (item.source_family, -abs(item.effect_size))):
        lines.append(f"{row.source_family} | {row.signal} | t-{row.horizon} | {fmt(row.effect_size)} | {row.replication_count}")

    lines += ["", "6. Formation Mode Table", "SourceFamily | FormationMode | Count | Percent | ReplicationCount"]
    for row in result.modes:
        lines.append(f"{row.source_family} | {row.mode} | {row.count} | {pct(row.percent)} | {row.replication_count}")

    lines += ["", "7. Source Family Formation Summary", "SourceFamily | DominantFormationMode | SecondaryFormationMode | TopSignal | ReplicationCount"]
    for row in result.summaries:
        lines.append(f"{row.source_family} | {row.dominant_mode} | {row.secondary_mode} | {row.top_signal} | {row.replication_count}")

    for section in ("8. Compression To Neutral Table", "9. Mixed To Neutral Table", "10. Reassertion To Neutral Table"):
        lines += ["", section, "Metric | Value | Interpretation"]
        for row in [item for item in result.special if item.section == section]:
            lines.append(f"{row.metric} | {row.value} | {row.interpretation}")

    lines += ["", "11. Recovery / Exhaustion To Neutral Table", "Metric | RecoveryResolution | ExhaustionPersistence | Difference"]
    for row in result.recovery_exhaustion:
        lines.append(f"{row.metric} | {row.recovery} | {row.exhaustion} | {row.difference}")

    decay = result.decay_audit
    lines += ["", "12. DecayToNeutral Audit Table", "Count | ReplicationCount | MeanLengthBeforeNeutral | MarketProfileSummary"]
    lines.append(f"{decay.count} | {decay.replication_count} | {fmt(decay.mean_length_before_neutral)} | {decay.market_profile}")

    lines += ["", "13. Formation Timing Table", "SourceFamily | MeanBarsToNeutral | MedianBarsToNeutral | MaxBarsToNeutral"]
    for row in result.timing:
        lines.append(f"{row.source_family} | {fmt(row.mean_bars)} | {fmt(row.median_bars)} | {row.max_bars}")

    lines += ["", "14. Market Normalization Sequence Table", "SourceFamily | Signal | Value_t5 | Value_t3 | Value_t2 | Value_t1 | Value_t0"]
    for row in result.normalization:
        lines.append(f"{row.source_family} | {row.signal} | {fmt(row.values.get(5, 0.0))} | {fmt(row.values.get(3, 0.0))} | {fmt(row.values.get(2, 0.0))} | {fmt(row.values.get(1, 0.0))} | {fmt(row.values.get(0, 0.0))}")

    lines += ["", "15. Cross-Instrument Replication Table", "SourceFamily | Signal | Effect_6E | Effect_CL | Effect_NQ | ReplicationCount"]
    for row in result.replication[:160]:
        lines.append(f"{row.source_family} | {row.signal} | {fmt(row.effects.get('6E', 0.0))} | {fmt(row.effects.get('CL', 0.0))} | {fmt(row.effects.get('NQ', 0.0))} | {row.replication_count}")

    lines += ["", "16. Outcome Diagnostics Table", "SourceFamily | FormationMode | DRFwd1 | DRFwd3 | DRFwd5 | DRFwd10 | ContinuationRate5 | FailureRate5 | FlatRate5"]
    for row in result.outcomes:
        lines.append(f"{row.source_family} | {row.mode} | {fmt(row.dr.get(1, 0.0))} | {fmt(row.dr.get(3, 0.0))} | {fmt(row.dr.get(5, 0.0))} | {fmt(row.dr.get(10, 0.0))} | {pct(row.continuation5)} | {pct(row.failure5)} | {pct(row.flat5)}")

    rec = result.recommendation
    lines += [
        "",
        "17. Neutral Formation Model",
        f"Classification: {rec.classification}",
        f"DominantFormationProcess: {rec.dominant_process}",
        f"SourceSpecificProcesses: {rec.source_processes}",
        f"Study78Assessment: {rec.study78_assessment}",
        f"Reason: {rec.reason}",
        "",
        "18. Comparison With Study 78",
        rec.study78_assessment,
        "",
        "19. Low-DoF Audit",
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
        "No forward returns used in formation construction.",
    ]


def rankings(result: Result) -> list[str]:
    top_sources = sorted(result.inventory.values(), key=lambda row: row.count, reverse=True)
    top_signals = sorted(result.shifts, key=lambda row: abs(row.effect_size), reverse=True)
    replicated = sorted([row for row in result.shifts if row.replication_count >= 2], key=lambda row: abs(row.effect_size), reverse=True)
    timing_fast = sorted([row for row in result.timing if row.mean_bars > 0], key=lambda row: row.mean_bars)
    timing_slow = sorted(result.timing, key=lambda row: row.mean_bars, reverse=True)
    range_collapse = sorted([row for row in result.shifts if row.signal in {"RangeRelativeToPrevious", "TrueRangeRelativeToPrevious", "BodyRelativeToPrevious"}], key=lambda row: row.effect_size)
    volume_collapse = sorted([row for row in result.shifts if row.signal.startswith("Volume")], key=lambda row: row.effect_size)
    efficiency_collapse = sorted([row for row in result.shifts if row.signal == "EfficiencyRatio"], key=lambda row: row.effect_size)
    return [
        "",
        "RANKINGS",
        "1. Largest Neutral source families: " + "; ".join(f"{row.source_family}={row.count}" for row in top_sources[:8]),
        "2. Strongest formation signals: " + "; ".join(f"{row.source_family}:{row.signal}@t-{row.horizon}={fmt(row.effect_size)}" for row in top_signals[:8]),
        "3. Most replicated formation signals: " + "; ".join(f"{row.source_family}:{row.signal}@t-{row.horizon}=R{row.replication_count}" for row in replicated[:8]),
        "4. Fastest Neutral-forming families: " + "; ".join(f"{row.source_family}={fmt(row.mean_bars)}" for row in timing_fast[:8]),
        "5. Slowest Neutral-forming families: " + "; ".join(f"{row.source_family}={fmt(row.mean_bars)}" for row in timing_slow[:8]),
        "6. Strongest range-collapse formations: " + "; ".join(f"{row.source_family}:{row.signal}={fmt(row.effect_size)}" for row in range_collapse[:8]),
        "7. Strongest volume-collapse formations: " + "; ".join(f"{row.source_family}:{row.signal}={fmt(row.effect_size)}" for row in volume_collapse[:8]),
        "8. Strongest efficiency-collapse formations: " + "; ".join(f"{row.source_family}={fmt(row.effect_size)}" for row in efficiency_collapse[:8]),
        "9. Strongest transition-basin formations: " + "; ".join(f"{row.source_family}:{row.dominant_mode}" for row in result.summaries if row.source_family == "MixedStructure"),
        f"10. Recommended Neutral formation model: {result.recommendation.classification}",
    ]


def write_report(result: Result, out_root: Path, aggregate: bool = False) -> None:
    if aggregate:
        path = out_root / "NeutralFormation79" / "NeutralFormation79_All.txt"
        title = "APVA Neutral Formation Study v0.1 - Aggregate"
    else:
        path = out_root / result.instrument / f"NeutralFormation79_{result.instrument}.txt"
        title = f"APVA Neutral Formation Study v0.1 - {result.instrument}"
    ensure_dir(path.parent)
    lines = [
        title,
        "=" * 96,
        f"Instrument: {result.instrument}",
        f"Input path(s): {', '.join(str(path) for path in result.source_paths)}",
        f"Formation event count: {len(result.events)}",
        f"Control event count: {len(result.controls)}",
    ]
    append_common(lines, result)
    lines += rankings(result)
    lines += [
        "",
        "RESEARCH NOTES",
        "Mechanical only.",
        "Questions: How is Neutral born? What market behavior precedes Neutral_Age1? Can APVA complete the formation/lifecycle/failure/destination/return loop?",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(result: Result) -> None:
    if set(result.inventory) != set(DESTINATION_FAMILIES):
        raise RuntimeError(f"{result.instrument}: inventory incomplete.")
    if not result.events:
        raise RuntimeError(f"{result.instrument}: no formation events.")
    if len(result.profiles) != len(DESTINATION_FAMILIES) * len(LOOKBACKS) * len(MARKET_SIGNALS):
        raise RuntimeError(f"{result.instrument}: profile table incomplete.")
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
    aggregate_result = build_result("Aggregate", [path for context in contexts for path in context.source_paths], contexts)
    attach_replication(aggregate_result, instrument_results)
    validate(aggregate_result)
    write_report(aggregate_result, out_root, aggregate=True)
    print(f"Wrote {len(instrument_results)} per-instrument Study79 report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
