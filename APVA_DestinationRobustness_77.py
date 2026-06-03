#!/usr/bin/env python3
"""APVA Destination Robustness Study v0.1.

Study 77 validates whether Study 76 destination signatures survive sample,
instrument, bootstrap, jackknife, and outlier stress tests.

Research only. No trading, optimization, fitting, machine learning, or forward
returns in robustness scoring.
"""

from __future__ import annotations

import argparse
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from APVA_BranchForecast_65 import build_stream
from APVA_ExcursionDestinations_76 import (
    DESTINATION_FAMILIES,
    MATERIAL_EFFECT,
    SIGNALS,
    DestinationEvent,
    destination_signal_value,
    find_events,
)
from APVA_InformationContribution_63 import thresholds
from APVA_InformationDecay_57 import study as decay_study
from APVA_MinimalEngine_52 import load_results, safe_mean
from APVA_NeutralFailureModes_75 import (
    LOOKBACKS,
    OUTCOME_HORIZONS,
    directional_forward,
    mean,
    median,
    node_metrics,
    pooled_stdev,
    safe_div,
    stdev,
)
from APVA_NodeImportance_58 import aggregate_rows, local_rows, score_rows
from APVA_ProcessGraph_54 import Node, node_for
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

SEED = 77
BOOTSTRAP_ITERATIONS = 200
DOWNSAMPLE_ITERATIONS = 100
DOWNSAMPLE_SIZES = (25, 50, 100)

KEY_SIGNATURES = (
    ("RecoveryResolution", "RangeRelativeToPrevious", 1),
    ("RecoveryResolution", "TrueRangeRelativeToPrevious", 1),
    ("RecoveryResolution", "VolumeRelativeToRollingMean", 1),
    ("RecoveryResolution", "BodyRelativeToPrevious", 1),
    ("RecoveryResolution", "EfficiencyRatio", 1),
    ("ExhaustionPersistence", "TrueRangeRelativeToPrevious", 1),
    ("ExhaustionPersistence", "RangeRelativeToPrevious", 1),
    ("ExhaustionPersistence", "VolumeRelativeToRollingMean", 1),
    ("ExhaustionPersistence", "BodyRelativeToPrevious", 1),
    ("ExhaustionPersistence", "EfficiencyRatio", 1),
    ("ReassertionProcessing", "VolumeRelativeToRollingMean", 1),
    ("ReassertionProcessing", "TrueRangeRelativeToPrevious", 1),
    ("ReassertionProcessing", "RangeRelativeToPrevious", 1),
    ("MixedStructure", "VolumeRelativeToRollingMean", 3),
    ("MixedStructure", "RangeRelativeToPrevious", 3),
    ("MixedStructure", "TrueRangeRelativeToPrevious", 3),
    ("CompressionProcessing", "VolumeRelativeToRollingMean", 1),
)

LARGE_DESTINATIONS = {"MixedStructure", "ReassertionProcessing", "CompressionProcessing"}
SMALL_DESTINATIONS = {"RecoveryResolution", "ExhaustionPersistence", "ConstructiveEmergence"}

SIGNAL_FAMILIES = {
    "RangeFamily": ("RangeRelativeToPrevious", "TrueRangeRelativeToPrevious", "BodyRelativeToPrevious"),
    "VolumeFamily": ("VolumeRelativeToPrevious", "VolumeRelativeToRollingMean", "VolumeRelativeToSessionMean"),
    "EfficiencyFamily": ("EfficiencyRatio", "CloseLocation"),
    "GraphFamily": ("MemoryStrength", "BranchEntropy"),
}


@dataclass
class Context:
    instrument: str
    source_paths: list
    bars: list
    nodes: list[Node]
    memory: dict[Node, float]
    entropy: dict[Node, float]
    events: list[DestinationEvent]


@dataclass(frozen=True)
class SignatureKey:
    destination: str
    signal: str
    horizon: int


@dataclass
class EffectParts:
    effect: float
    family_mean: float
    other_mean: float
    family_values: list[float]
    other_values: list[float]


@dataclass
class BaselineRow:
    key: SignatureKey
    count: int
    effect: float
    replication_count: int
    direction: str
    rank: int


@dataclass
class SufficiencyRow:
    destination: str
    count: int
    instrument_counts: Counter[str]
    min_instrument_count: int
    max_instrument_share: float
    effective_instrument_count: int
    sample_class: str


@dataclass
class LeaveOneOutRow:
    key: SignatureKey
    full_effect: float
    effects_without: dict[str, float] = field(default_factory=dict)
    direction_stable: bool = False
    material_stable: bool = False


@dataclass
class InstrumentPairRow:
    key: SignatureKey
    effects: dict[str, float] = field(default_factory=dict)
    direction_agreement: bool = False
    material_agreement: bool = False


@dataclass
class BootstrapRow:
    key: SignatureKey
    mean_effect: float
    median_effect: float
    stddev_effect: float
    lower5: float
    upper95: float
    sign_stability: float
    material_stability: float


@dataclass
class DownsampleRow:
    key: SignatureKey
    downsample_n: int
    mean_effect: float
    stddev_effect: float
    sign_stability: float
    material_stability: float


@dataclass
class JackknifeRow:
    key: SignatureKey
    mean_effect: float
    min_effect: float
    max_effect: float
    sign_stability: float
    material_stability: float


@dataclass
class OutlierRow:
    key: SignatureKey
    original_effect: float
    trimmed_effect: float
    effect_change: float
    direction_stable: bool


@dataclass
class HorizonRow:
    destination: str
    signal: str
    effects: dict[int, float] = field(default_factory=dict)
    temporal_coherence: bool = False


@dataclass
class FamilyRow:
    destination: str
    signal_family: str
    family_mean_effect: float
    family_sign_agreement: float
    family_material_count: int


@dataclass
class RobustSignatureRow:
    key: SignatureKey
    classification: str
    reason: str


@dataclass
class DestinationRobustnessRow:
    destination: str
    score: float
    classification: str
    robust_count: int
    fragile_count: int
    rejected_count: int


@dataclass
class ReassessmentRow:
    destination: str
    study76_score: float
    study77_score: float
    reassessment: str


@dataclass
class OutcomeRow:
    destination: str
    signature: str
    dr: dict[int, float] = field(default_factory=dict)
    continuation5: float = 0.0
    failure5: float = 0.0
    flat5: float = 0.0


@dataclass
class Recommendation:
    classification: str
    confirmed_destinations: str
    weakened_destinations: str
    rejected_destinations: str
    next_step: str
    reason: str


@dataclass
class Result:
    instrument: str
    source_paths: list
    contexts: list[Context]
    baseline: list[BaselineRow]
    sufficiency: list[SufficiencyRow]
    leave_one_out: list[LeaveOneOutRow]
    instrument_pairs: list[InstrumentPairRow]
    bootstrap: list[BootstrapRow]
    downsample: list[DownsampleRow]
    jackknife: list[JackknifeRow]
    outliers: list[OutlierRow]
    horizons: list[HorizonRow]
    families: list[FamilyRow]
    robust_signatures: list[RobustSignatureRow]
    destination_robustness: list[DestinationRobustnessRow]
    reassessment: list[ReassessmentRow]
    outcomes: list[OutcomeRow]
    recommendation: Recommendation


def direction(effect: float) -> str:
    if effect > 0:
        return "Positive"
    if effect < 0:
        return "Negative"
    return "Flat"


def same_sign(values: list[float]) -> bool:
    nonzero = [value for value in values if value != 0.0]
    if not nonzero:
        return False
    return all(value > 0 for value in nonzero) or all(value < 0 for value in nonzero)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return ordered[max(0, min(index, len(ordered) - 1))]


def effect_from_values(family_values: list[float], other_values: list[float]) -> float:
    if not family_values or not other_values:
        return 0.0
    delta = mean(family_values) - mean(other_values)
    pooled = pooled_stdev(family_values, other_values)
    return safe_div(delta, pooled) or 0.0


def trim_values(values: list[float]) -> list[float]:
    if len(values) < 20:
        return list(values)
    ordered = sorted(values)
    cut = max(1, int(len(ordered) * 0.05))
    return ordered[cut:-cut] or ordered


def event_value(context: Context, event: DestinationEvent, signal: str, horizon: int) -> float | None:
    return destination_signal_value(
        signal,
        context.bars,
        context.nodes,
        event.index - horizon,
        context.memory,
        context.entropy,
    )


def collect_values(contexts: list[Context], key: SignatureKey,
                   instruments: set[str] | None = None) -> EffectParts:
    family_values = []
    other_values = []
    for context in contexts:
        if instruments is not None and context.instrument not in instruments:
            continue
        for event in context.events:
            value = event_value(context, event, key.signal, key.horizon)
            if value is None:
                continue
            if event.destination == key.destination:
                family_values.append(float(value))
            else:
                other_values.append(float(value))
    effect = effect_from_values(family_values, other_values)
    return EffectParts(effect, mean(family_values), mean(other_values), family_values, other_values)


def all_events_for_destination(contexts: list[Context], destination: str) -> list[tuple[Context, DestinationEvent]]:
    events = []
    for context in contexts:
        for event in context.events:
            if event.destination == destination:
                events.append((context, event))
    return events


def baseline_rows(contexts: list[Context]) -> list[BaselineRow]:
    rows = []
    effect_by_key = {}
    count_by_destination = Counter()
    for context in contexts:
        count_by_destination.update(event.destination for event in context.events)
    for destination in DESTINATION_FAMILIES:
        for signal in SIGNALS:
            for horizon in LOOKBACKS:
                key = SignatureKey(destination, signal, horizon)
                parts = collect_values(contexts, key)
                effect_by_key[key] = parts.effect
    ranks_by_destination: dict[str, dict[SignatureKey, int]] = defaultdict(dict)
    for destination in DESTINATION_FAMILIES:
        keys = [key for key in effect_by_key if key.destination == destination]
        keys.sort(key=lambda item: abs(effect_by_key[item]), reverse=True)
        for rank, key in enumerate(keys, 1):
            ranks_by_destination[destination][key] = rank
    for key, effect in effect_by_key.items():
        rows.append(BaselineRow(
            key,
            count_by_destination[key.destination],
            effect,
            replication_count_for_key(contexts, key),
            direction(effect),
            ranks_by_destination[key.destination][key],
        ))
    rows.sort(key=lambda row: (row.key.destination, row.rank))
    return rows


def replication_count_for_key(contexts: list[Context], key: SignatureKey) -> int:
    effects = []
    for context in contexts:
        parts = collect_values([context], key)
        if abs(parts.effect) >= MATERIAL_EFFECT:
            effects.append(parts.effect)
    positives = sum(1 for effect in effects if effect > 0)
    negatives = sum(1 for effect in effects if effect < 0)
    return max(positives, negatives)


def sufficiency_rows(contexts: list[Context]) -> list[SufficiencyRow]:
    counts_by_destination: dict[str, Counter[str]] = {destination: Counter() for destination in DESTINATION_FAMILIES}
    for context in contexts:
        counts_by_destination.setdefault(context.instrument, Counter())
        for event in context.events:
            counts_by_destination[event.destination][context.instrument] += 1
    rows = []
    for destination in DESTINATION_FAMILIES:
        counts = counts_by_destination[destination]
        total = sum(counts.values())
        nonzero = [value for value in counts.values() if value > 0]
        replication = len(nonzero)
        if total >= 100 and replication >= 2:
            sample_class = "AdequateSample"
        elif 25 <= total < 100 and replication >= 2:
            sample_class = "MarginalSample"
        elif total < 25:
            sample_class = "SparseSample"
        else:
            sample_class = "SparseSample"
        rows.append(SufficiencyRow(
            destination,
            total,
            counts,
            min(nonzero) if nonzero else 0,
            safe_div(max(counts.values(), default=0), total) or 0.0,
            replication,
            sample_class,
        ))
    return rows


def leave_one_out_rows(contexts: list[Context], keys: list[SignatureKey]) -> list[LeaveOneOutRow]:
    instruments = sorted({context.instrument for context in contexts})
    rows = []
    for key in keys:
        full = collect_values(contexts, key).effect
        effects = {}
        for instrument in instruments:
            included = set(instruments) - {instrument}
            effects[instrument] = collect_values(contexts, key, included).effect
        available = [value for value in effects.values() if value != 0.0]
        material = [value for value in effects.values() if abs(value) >= MATERIAL_EFFECT]
        rows.append(LeaveOneOutRow(
            key,
            full,
            effects,
            direction_stable=same_sign([full] + available),
            material_stable=len(material) >= 2,
        ))
    return rows


def instrument_pair_rows(contexts: list[Context], keys: list[SignatureKey]) -> list[InstrumentPairRow]:
    pairs = (("6E", "CL"), ("6E", "NQ"), ("CL", "NQ"))
    rows = []
    for key in keys:
        effects = {}
        for left, right in pairs:
            effects[f"{left}_{right}"] = collect_values(contexts, key, {left, right}).effect
        values = list(effects.values())
        material = [value for value in values if abs(value) >= MATERIAL_EFFECT]
        rows.append(InstrumentPairRow(
            key,
            effects,
            direction_agreement=same_sign(values),
            material_agreement=len(material) >= 2,
        ))
    return rows


def bootstrap_rows(contexts: list[Context], keys: list[SignatureKey]) -> list[BootstrapRow]:
    rows = []
    for key in keys:
        parts = collect_values(contexts, key)
        if len(parts.family_values) < 25 or not parts.other_values:
            rows.append(BootstrapRow(key, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        rng = random.Random(f"{SEED}:{key.destination}:{key.signal}:{key.horizon}:bootstrap")
        effects = []
        for _ in range(BOOTSTRAP_ITERATIONS):
            sampled = [rng.choice(parts.family_values) for _ in parts.family_values]
            effects.append(effect_from_values(sampled, parts.other_values))
        base_sign = 1 if parts.effect > 0 else -1 if parts.effect < 0 else 0
        rows.append(BootstrapRow(
            key,
            mean(effects),
            median(effects),
            stdev(effects),
            percentile(effects, 0.05),
            percentile(effects, 0.95),
            mean([1.0 if (effect > 0) == (base_sign > 0) and base_sign != 0 else 0.0 for effect in effects]),
            mean([1.0 if abs(effect) >= MATERIAL_EFFECT else 0.0 for effect in effects]),
        ))
    return rows


def downsample_rows(contexts: list[Context], keys: list[SignatureKey]) -> list[DownsampleRow]:
    rows = []
    for key in keys:
        if key.destination not in LARGE_DESTINATIONS:
            continue
        parts = collect_values(contexts, key)
        base_sign = 1 if parts.effect > 0 else -1 if parts.effect < 0 else 0
        for size in DOWNSAMPLE_SIZES:
            if len(parts.family_values) < size or not parts.other_values:
                continue
            rng = random.Random(f"{SEED}:{key.destination}:{key.signal}:{key.horizon}:down:{size}")
            effects = []
            for _ in range(DOWNSAMPLE_ITERATIONS):
                sampled = rng.sample(parts.family_values, size)
                effects.append(effect_from_values(sampled, parts.other_values))
            rows.append(DownsampleRow(
                key,
                size,
                mean(effects),
                stdev(effects),
                mean([1.0 if (effect > 0) == (base_sign > 0) and base_sign != 0 else 0.0 for effect in effects]),
                mean([1.0 if abs(effect) >= MATERIAL_EFFECT else 0.0 for effect in effects]),
            ))
    return rows


def jackknife_rows(contexts: list[Context], keys: list[SignatureKey]) -> list[JackknifeRow]:
    rows = []
    for key in keys:
        if key.destination not in SMALL_DESTINATIONS:
            continue
        parts = collect_values(contexts, key)
        if len(parts.family_values) < 2 or not parts.other_values:
            rows.append(JackknifeRow(key, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        base_sign = 1 if parts.effect > 0 else -1 if parts.effect < 0 else 0
        effects = []
        for index in range(len(parts.family_values)):
            subset = parts.family_values[:index] + parts.family_values[index + 1:]
            effects.append(effect_from_values(subset, parts.other_values))
        rows.append(JackknifeRow(
            key,
            mean(effects),
            min(effects),
            max(effects),
            mean([1.0 if (effect > 0) == (base_sign > 0) and base_sign != 0 else 0.0 for effect in effects]),
            mean([1.0 if abs(effect) >= MATERIAL_EFFECT else 0.0 for effect in effects]),
        ))
    return rows


def outlier_rows(contexts: list[Context], keys: list[SignatureKey]) -> list[OutlierRow]:
    rows = []
    for key in keys:
        parts = collect_values(contexts, key)
        trimmed = effect_from_values(trim_values(parts.family_values), trim_values(parts.other_values))
        rows.append(OutlierRow(
            key,
            parts.effect,
            trimmed,
            trimmed - parts.effect,
            same_sign([parts.effect, trimmed]),
        ))
    return rows


def horizon_rows(contexts: list[Context], keys: list[SignatureKey]) -> list[HorizonRow]:
    rows = []
    seen = sorted({(key.destination, key.signal) for key in keys})
    for destination, signal in seen:
        effects = {}
        for horizon in LOOKBACKS:
            effects[horizon] = collect_values(contexts, SignatureKey(destination, signal, horizon)).effect
        ordered = [effects[horizon] for horizon in (1, 2, 3, 5)]
        coherent = any(same_sign([ordered[i], ordered[i + 1]]) for i in range(len(ordered) - 1))
        rows.append(HorizonRow(destination, signal, effects, coherent))
    return rows


def family_rows(contexts: list[Context]) -> list[FamilyRow]:
    rows = []
    for destination in DESTINATION_FAMILIES:
        for family, signals in SIGNAL_FAMILIES.items():
            effects = []
            for signal in signals:
                for horizon in LOOKBACKS:
                    effects.append(collect_values(contexts, SignatureKey(destination, signal, horizon)).effect)
            positives = sum(1 for effect in effects if effect > 0)
            negatives = sum(1 for effect in effects if effect < 0)
            rows.append(FamilyRow(
                destination,
                family,
                mean([abs(effect) for effect in effects]),
                safe_div(max(positives, negatives), len(effects)) or 0.0,
                sum(1 for effect in effects if abs(effect) >= MATERIAL_EFFECT),
            ))
    return rows


def robust_signature_rows(keys: list[SignatureKey], baseline: dict[SignatureKey, BaselineRow],
                          loo: dict[SignatureKey, LeaveOneOutRow], bootstrap: dict[SignatureKey, BootstrapRow],
                          outliers: dict[SignatureKey, OutlierRow]) -> list[RobustSignatureRow]:
    rows = []
    for key in keys:
        base = baseline[key]
        leave = loo[key]
        boot = bootstrap.get(key, BootstrapRow(key, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        outlier = outliers[key]
        if (
            leave.direction_stable
            and leave.material_stable
            and boot.sign_stability >= 0.80
            and boot.material_stability >= 0.60
            and outlier.direction_stable
            and base.replication_count >= 2
        ):
            classification = "RobustSignature"
        elif leave.direction_stable:
            classification = "FragileSignature"
        else:
            classification = "RejectedSignature"
        reason = (
            f"LOO direction={leave.direction_stable}, material={leave.material_stable}, "
            f"bootstrap sign={pct(boot.sign_stability)}, bootstrap material={pct(boot.material_stability)}, "
            f"trim direction={outlier.direction_stable}, replication={base.replication_count}"
        )
        rows.append(RobustSignatureRow(key, classification, reason))
    return rows


def normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if low == high:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def destination_robustness_rows(sufficiency: list[SufficiencyRow], robust_signatures: list[RobustSignatureRow],
                                bootstrap: list[BootstrapRow], loo: list[LeaveOneOutRow],
                                outliers: list[OutlierRow]) -> list[DestinationRobustnessRow]:
    robust_counts = Counter(row.key.destination for row in robust_signatures if row.classification == "RobustSignature")
    fragile_counts = Counter(row.key.destination for row in robust_signatures if row.classification == "FragileSignature")
    rejected_counts = Counter(row.key.destination for row in robust_signatures if row.classification == "RejectedSignature")
    boot_sign = defaultdict(list)
    boot_material = defaultdict(list)
    for row in bootstrap:
        boot_sign[row.key.destination].append(row.sign_stability)
        boot_material[row.key.destination].append(row.material_stability)
    instrument_agreement = defaultdict(list)
    for row in loo:
        instrument_agreement[row.key.destination].append(1.0 if row.direction_stable else 0.0)
    outlier_resistance = defaultdict(list)
    for row in outliers:
        outlier_resistance[row.key.destination].append(1.0 if row.direction_stable else 0.0)
    sample_score = {
        row.destination: {"AdequateSample": 1.0, "MarginalSample": 0.5, "SparseSample": 0.0}.get(row.sample_class, 0.0)
        for row in sufficiency
    }
    parts = [
        normalize({destination: robust_counts[destination] for destination in DESTINATION_FAMILIES}),
        {destination: mean(boot_sign[destination]) for destination in DESTINATION_FAMILIES},
        {destination: mean(boot_material[destination]) for destination in DESTINATION_FAMILIES},
        {destination: mean(instrument_agreement[destination]) for destination in DESTINATION_FAMILIES},
        {destination: mean(outlier_resistance[destination]) for destination in DESTINATION_FAMILIES},
        sample_score,
    ]
    rows = []
    for destination in DESTINATION_FAMILIES:
        score = mean([part.get(destination, 0.0) for part in parts])
        if robust_counts[destination] > 0 and score >= 0.67:
            classification = "RobustDestination"
        elif robust_counts[destination] > 0 or score >= 0.45:
            classification = "MarginalDestination"
        elif score > 0.20:
            classification = "FragileDestination"
        else:
            classification = "UnvalidatedDestination"
        rows.append(DestinationRobustnessRow(
            destination,
            score,
            classification,
            robust_counts[destination],
            fragile_counts[destination],
            rejected_counts[destination],
        ))
    rows.sort(key=lambda row: row.score, reverse=True)
    return rows


def study76_scores(contexts: list[Context]) -> dict[str, float]:
    scores = {}
    for destination in DESTINATION_FAMILIES:
        effects = []
        for signal in SIGNALS:
            for horizon in LOOKBACKS:
                key = SignatureKey(destination, signal, horizon)
                if replication_count_for_key(contexts, key) >= 2:
                    effects.append(abs(collect_values(contexts, key).effect))
        scores[destination] = mean(effects)
    return scores


def reassessment_rows(destination_rows: list[DestinationRobustnessRow], contexts: list[Context]) -> list[ReassessmentRow]:
    scores76 = study76_scores(contexts)
    lookup = {row.destination: row for row in destination_rows}
    rows = []
    for destination in (
        "RecoveryResolution",
        "ExhaustionPersistence",
        "ReassertionProcessing",
        "MixedStructure",
        "CompressionProcessing",
        "ConstructiveEmergence",
    ):
        score77 = lookup.get(destination, DestinationRobustnessRow(destination, 0.0, "UnvalidatedDestination", 0, 0, 0)).score
        classification = lookup.get(destination, DestinationRobustnessRow(destination, 0.0, "UnvalidatedDestination", 0, 0, 0)).classification
        if classification == "RobustDestination":
            reassessment = "Confirmed"
        elif classification == "MarginalDestination":
            reassessment = "Weakened"
        else:
            reassessment = "Rejected"
        rows.append(ReassessmentRow(destination, scores76.get(destination, 0.0), score77, reassessment))
    return rows


def outcome_rows(contexts: list[Context], robust_signatures: list[RobustSignatureRow]) -> list[OutcomeRow]:
    rows = []
    selected = [row for row in robust_signatures if row.classification in {"RobustSignature", "RejectedSignature"}]
    for row in selected:
        values_by_horizon = defaultdict(list)
        for context in contexts:
            for event in context.events:
                if event.destination != row.key.destination:
                    continue
                for horizon in OUTCOME_HORIZONS:
                    value = directional_forward(context.bars, event.index, horizon)
                    if value is not None:
                        values_by_horizon[horizon].append(value)
        outcome = OutcomeRow(row.key.destination, f"{row.key.signal}@t-{row.key.horizon}")
        for horizon in OUTCOME_HORIZONS:
            values = values_by_horizon[horizon]
            outcome.dr[horizon] = mean(values)
        values5 = values_by_horizon[5]
        outcome.continuation5 = mean([1.0 if value > 0 else 0.0 for value in values5])
        outcome.failure5 = mean([1.0 if value < 0 else 0.0 for value in values5])
        outcome.flat5 = mean([1.0 if value == 0 else 0.0 for value in values5])
        rows.append(outcome)
    return rows


def make_recommendation(reassessment: list[ReassessmentRow], destination_rows: list[DestinationRobustnessRow]) -> Recommendation:
    confirmed = [row.destination for row in reassessment if row.reassessment == "Confirmed"]
    weakened = [row.destination for row in reassessment if row.reassessment == "Weakened"]
    rejected = [row.destination for row in reassessment if row.reassessment == "Rejected"]
    if len(confirmed) >= 4:
        classification = "Study76Confirmed"
        next_step = "ProceedToExcursionLifecycle"
    elif confirmed:
        classification = "Study76PartiallyConfirmed"
        next_step = "RestrictToRobustDestinations"
    elif weakened:
        classification = "Study76Weakened"
        next_step = "CollectMoreData"
    else:
        classification = "Study76Rejected"
        next_step = "StopDestinationBranch"
    robust_names = [row.destination for row in destination_rows if row.classification == "RobustDestination"]
    reason = (
        f"Robust destinations: {', '.join(robust_names) or 'None'}. "
        "Classification uses fixed sample, instrument, bootstrap, jackknife, and outlier checks."
    )
    return Recommendation(
        classification,
        ", ".join(confirmed) or "None",
        ", ".join(weakened) or "None",
        ", ".join(rejected) or "None",
        next_step,
        reason,
    )


def build_contexts(loaded_rows: list, node_rows_by_instrument: list[dict[Node, object]], aggregate_thresholds) -> list[Context]:
    contexts = []
    for loaded_row, local_node_rows in zip(loaded_rows, node_rows_by_instrument):
        stream = build_stream(loaded_row, local_node_rows, aggregate_thresholds, (0.0, 0.0), (0.0, 0.0))
        memory, entropy = node_metrics(local_node_rows, stream)
        nodes = [node_for(bar) for bar in loaded_row.bars]
        events = find_events(nodes, loaded_row.instrument)
        contexts.append(Context(loaded_row.instrument, loaded_row.source_paths, loaded_row.bars, nodes, memory, entropy, events))
    return contexts


def build_result(instrument: str, source_paths: list, contexts: list[Context]) -> Result:
    keys = [SignatureKey(*signature) for signature in KEY_SIGNATURES]
    base = baseline_rows(contexts)
    base_lookup = {row.key: row for row in base}
    suff = sufficiency_rows(contexts)
    loo = leave_one_out_rows(contexts, keys)
    pairs = instrument_pair_rows(contexts, keys)
    boot = bootstrap_rows(contexts, keys)
    down = downsample_rows(contexts, keys)
    jack = jackknife_rows(contexts, keys)
    out = outlier_rows(contexts, keys)
    horiz = horizon_rows(contexts, keys)
    fam = family_rows(contexts)
    robust = robust_signature_rows(
        keys,
        base_lookup,
        {row.key: row for row in loo},
        {row.key: row for row in boot},
        {row.key: row for row in out},
    )
    dest = destination_robustness_rows(suff, robust, boot, loo, out)
    reassess = reassessment_rows(dest, contexts)
    outcomes = outcome_rows(contexts, robust)
    rec = make_recommendation(reassess, dest)
    return Result(
        instrument,
        source_paths,
        contexts,
        base,
        suff,
        loo,
        pairs,
        boot,
        down,
        jack,
        out,
        horiz,
        fam,
        robust,
        dest,
        reassess,
        outcomes,
        rec,
    )


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def key_text(key: SignatureKey) -> str:
    return f"{key.destination}:{key.signal}@t-{key.horizon}"


def append_common(lines: list[str], result: Result, aggregate: bool = False) -> None:
    lines += ["", "1. Baseline Signature Table", "DestinationFamily | Signal | Horizon | EffectSize | ReplicationCount | Direction | Rank"]
    for row in sorted(result.baseline, key=lambda item: (item.key.destination, item.rank))[:420]:
        lines.append(f"{row.key.destination} | {row.key.signal} | t-{row.key.horizon} | {fmt(row.effect)} | {row.replication_count} | {row.direction} | {row.rank}")

    lines += ["", "2. Sample Sufficiency Table", "DestinationFamily | Count | Count_6E | Count_CL | Count_NQ | MinInstrumentCount | MaxInstrumentShare | EffectiveInstrumentCount | SampleClass"]
    for row in result.sufficiency:
        lines.append(f"{row.destination} | {row.count} | {row.instrument_counts.get('6E', 0)} | {row.instrument_counts.get('CL', 0)} | {row.instrument_counts.get('NQ', 0)} | {row.min_instrument_count} | {pct(row.max_instrument_share)} | {row.effective_instrument_count} | {row.sample_class}")

    lines += ["", "3. Leave-One-Instrument-Out Table", "DestinationFamily | Signal | Horizon | FullEffect | EffectWithout6E | EffectWithoutCL | EffectWithoutNQ | DirectionStable | MaterialStable"]
    for row in result.leave_one_out:
        lines.append(f"{row.key.destination} | {row.key.signal} | t-{row.key.horizon} | {fmt(row.full_effect)} | {fmt(row.effects_without.get('6E', 0.0))} | {fmt(row.effects_without.get('CL', 0.0))} | {fmt(row.effects_without.get('NQ', 0.0))} | {bool_text(row.direction_stable)} | {bool_text(row.material_stable)}")

    lines += ["", "4. Instrument Pair Table", "DestinationFamily | Signal | Horizon | Effect_6E_CL | Effect_6E_NQ | Effect_CL_NQ | DirectionAgreement | MaterialAgreement"]
    for row in result.instrument_pairs:
        lines.append(f"{row.key.destination} | {row.key.signal} | t-{row.key.horizon} | {fmt(row.effects.get('6E_CL', 0.0))} | {fmt(row.effects.get('6E_NQ', 0.0))} | {fmt(row.effects.get('CL_NQ', 0.0))} | {bool_text(row.direction_agreement)} | {bool_text(row.material_agreement)}")

    lines += ["", "5. Bootstrap Stability Table", "DestinationFamily | Signal | Horizon | MeanEffect | MedianEffect | StdDevEffect | Lower5 | Upper95 | SignStabilityPercent | MaterialStabilityPercent"]
    for row in result.bootstrap:
        lines.append(f"{row.key.destination} | {row.key.signal} | t-{row.key.horizon} | {fmt(row.mean_effect)} | {fmt(row.median_effect)} | {fmt(row.stddev_effect)} | {fmt(row.lower5)} | {fmt(row.upper95)} | {pct(row.sign_stability)} | {pct(row.material_stability)}")

    lines += ["", "6. Downsample Stability Table", "DestinationFamily | Signal | Horizon | DownsampleN | MeanEffect | StdDevEffect | SignStabilityPercent | MaterialStabilityPercent"]
    for row in result.downsample:
        lines.append(f"{row.key.destination} | {row.key.signal} | t-{row.key.horizon} | {row.downsample_n} | {fmt(row.mean_effect)} | {fmt(row.stddev_effect)} | {pct(row.sign_stability)} | {pct(row.material_stability)}")

    lines += ["", "7. Jackknife Table", "DestinationFamily | Signal | Horizon | MeanEffect | MinEffect | MaxEffect | SignStabilityPercent | MaterialStabilityPercent"]
    for row in result.jackknife:
        lines.append(f"{row.key.destination} | {row.key.signal} | t-{row.key.horizon} | {fmt(row.mean_effect)} | {fmt(row.min_effect)} | {fmt(row.max_effect)} | {pct(row.sign_stability)} | {pct(row.material_stability)}")

    lines += ["", "8. Outlier Sensitivity Table", "DestinationFamily | Signal | Horizon | OriginalEffect | TrimmedEffect | EffectChange | DirectionStable"]
    for row in result.outliers:
        lines.append(f"{row.key.destination} | {row.key.signal} | t-{row.key.horizon} | {fmt(row.original_effect)} | {fmt(row.trimmed_effect)} | {fmt(row.effect_change)} | {bool_text(row.direction_stable)}")

    lines += ["", "9. Horizon Stability Table", "DestinationFamily | Signal | Effect_t1 | Effect_t2 | Effect_t3 | Effect_t5 | TemporalCoherence"]
    for row in result.horizons:
        lines.append(f"{row.destination} | {row.signal} | {fmt(row.effects.get(1, 0.0))} | {fmt(row.effects.get(2, 0.0))} | {fmt(row.effects.get(3, 0.0))} | {fmt(row.effects.get(5, 0.0))} | {bool_text(row.temporal_coherence)}")

    lines += ["", "10. Signal Family Stability Table", "DestinationFamily | SignalFamily | FamilyMeanEffect | FamilySignAgreement | FamilyMaterialCount"]
    for row in result.families:
        lines.append(f"{row.destination} | {row.signal_family} | {fmt(row.family_mean_effect)} | {pct(row.family_sign_agreement)} | {row.family_material_count}")

    lines += ["", "11. Robust Signature Table", "DestinationFamily | Signal | Horizon | Classification | Reason"]
    for row in result.robust_signatures:
        lines.append(f"{row.key.destination} | {row.key.signal} | t-{row.key.horizon} | {row.classification} | {row.reason}")

    lines += ["", "12. Destination Robustness Table", "DestinationFamily | RobustnessScore | Classification | RobustSignatureCount | FragileSignatureCount | RejectedSignatureCount"]
    for row in result.destination_robustness:
        lines.append(f"{row.destination} | {fmt(row.score)} | {row.classification} | {row.robust_count} | {row.fragile_count} | {row.rejected_count}")

    lines += ["", "13. Study 76 Reassessment Table", "DestinationFamily | Study76PredictabilityScore | Study77RobustnessScore | Reassessment"]
    for row in result.reassessment:
        lines.append(f"{row.destination} | {fmt(row.study76_score)} | {fmt(row.study77_score)} | {row.reassessment}")

    lines += ["", "14. Outcome Diagnostics Table", "DestinationFamily | Signature | DRFwd1 | DRFwd3 | DRFwd5 | DRFwd10 | ContinuationRate5 | FailureRate5 | FlatRate5"]
    for row in result.outcomes:
        lines.append(f"{row.destination} | {row.signature} | {fmt(row.dr.get(1, 0.0))} | {fmt(row.dr.get(3, 0.0))} | {fmt(row.dr.get(5, 0.0))} | {fmt(row.dr.get(10, 0.0))} | {pct(row.continuation5)} | {pct(row.failure5)} | {pct(row.flat5)}")

    rec = result.recommendation
    lines += [
        "",
        "15. Recommendation",
        f"Classification: {rec.classification}",
        f"ConfirmedDestinations: {rec.confirmed_destinations}",
        f"WeakenedDestinations: {rec.weakened_destinations}",
        f"RejectedDestinations: {rec.rejected_destinations}",
        f"RecommendedNextStep: {rec.next_step}",
        f"Reason: {rec.reason}",
        "",
        "16. Low-DoF Audit",
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
        "No forward returns used in robustness scoring.",
    ]


def rankings(result: Result) -> list[str]:
    robust = [row for row in result.robust_signatures if row.classification == "RobustSignature"]
    fragile = [row for row in result.robust_signatures if row.classification == "FragileSignature"]
    outlier_sensitive = sorted(result.outliers, key=lambda row: abs(row.effect_change), reverse=True)
    instrument_stable = [row for row in result.leave_one_out if row.direction_stable and row.material_stable]
    bootstrap_stable = sorted(result.bootstrap, key=lambda row: (row.sign_stability, row.material_stability), reverse=True)
    family_stable = sorted(result.families, key=lambda row: (row.family_material_count, row.family_mean_effect), reverse=True)
    lines = ["", "RANKINGS"]
    lines.append("1. Most robust destination signatures: " + "; ".join(key_text(row.key) for row in robust[:8]) or "None")
    lines.append("2. Most fragile destination signatures: " + "; ".join(key_text(row.key) for row in fragile[:8]) or "None")
    lines.append("3. Most robust destinations: " + "; ".join(f"{row.destination}={fmt(row.score)}" for row in result.destination_robustness[:8]))
    lines.append("4. Most weakened destinations: " + "; ".join(f"{row.destination}:{row.reassessment}" for row in result.reassessment if row.reassessment != "Confirmed"))
    lines.append("5. Most outlier-sensitive signatures: " + "; ".join(f"{key_text(row.key)}={fmt(row.effect_change)}" for row in outlier_sensitive[:8]))
    lines.append("6. Most instrument-stable signatures: " + "; ".join(key_text(row.key) for row in instrument_stable[:8]))
    lines.append("7. Most bootstrap-stable signatures: " + "; ".join(f"{key_text(row.key)}={pct(row.sign_stability)}" for row in bootstrap_stable[:8]))
    lines.append("8. Strongest signal-family signatures: " + "; ".join(f"{row.destination}:{row.signal_family}={fmt(row.family_mean_effect)}" for row in family_stable[:8]))
    lines.append(f"9. Recommended validated destination model: {result.recommendation.classification}")
    lines.append(f"10. Recommended next research branch: {result.recommendation.next_step}")
    return lines


def write_report(result: Result, out_root: Path, aggregate: bool = False) -> None:
    if aggregate:
        path = out_root / "DestinationRobustness77" / "DestinationRobustness77_All.txt"
        title = "APVA Destination Robustness Study v0.1 - Aggregate"
    else:
        path = out_root / result.instrument / f"DestinationRobustness77_{result.instrument}.txt"
        title = f"APVA Destination Robustness Study v0.1 - {result.instrument}"
    ensure_dir(path.parent)
    lines = [
        title,
        "=" * 96,
        f"Instrument: {result.instrument}",
        f"Input path(s): {', '.join(str(path) for path in result.source_paths)}",
        f"Destination event count: {sum(row.count for row in result.sufficiency)}",
    ]
    append_common(lines, result, aggregate)
    lines += rankings(result)
    lines += [
        "",
        "RESEARCH NOTES",
        "Mechanical only.",
        "Questions: Did Study 76 overstate destination predictability? Which signatures are reliable? Can APVA proceed to excursion lifecycle studies?",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(result: Result) -> None:
    if not result.contexts:
        raise RuntimeError(f"{result.instrument}: no contexts.")
    if len(result.baseline) != len(DESTINATION_FAMILIES) * len(SIGNALS) * len(LOOKBACKS):
        raise RuntimeError(f"{result.instrument}: baseline table incomplete.")
    if len(result.sufficiency) != len(DESTINATION_FAMILIES):
        raise RuntimeError(f"{result.instrument}: sufficiency table incomplete.")
    if not result.robust_signatures:
        raise RuntimeError(f"{result.instrument}: robust signature table missing.")
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
    aggregate_result = build_result("Aggregate", [path for context in contexts for path in context.source_paths], contexts)
    validate(aggregate_result)
    write_report(aggregate_result, out_root, aggregate=True)
    print(f"Wrote {len(instrument_results)} per-instrument Study77 report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
