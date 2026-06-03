#!/usr/bin/env python3
"""APVA Signal Minimalism Study v0.1.

Study 81 asks what smallest signal set can represent APVA's closed loop.

Research only. No trading, optimization, fitting, machine learning, or forward
returns in signal selection.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from APVA_DestinationRobustness_77 import build_contexts
from APVA_ExcursionLifecycle_78 import efficiency_ratio
from APVA_InformationContribution_63 import thresholds
from APVA_InformationDecay_57 import study as decay_study
from APVA_LoopClosure_80 import (
    PHASE_MACHINE,
    build_loops,
    complete_loops,
    destination_family,
    failure_node,
    loop_phases,
    machine_phase,
    max_neutral_age,
    return_family,
)
from APVA_MinimalEngine_52 import load_results
from APVA_NeutralFailureModes_75 import directional_forward, mean, safe_div
from APVA_NodeImportance_58 import aggregate_rows, local_rows, score_rows
from APVA_ProcessGraph_54 import Node, node_for, node_text
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

OUTCOME_HORIZONS = (1, 3, 5, 10)

CANDIDATE_SIGNALS = (
    "LoopPhase",
    "NeutralAgeBucket",
    "DestinationFamily",
    "ReturnFamily",
    "FormationSourceFamily",
    "FailureNode",
    "MaxNeutralAgeReached",
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

SIGNAL_FAMILIES = {
    "StateFamily": (
        "LoopPhase",
        "NeutralAgeBucket",
        "DestinationFamily",
        "ReturnFamily",
        "FailureNode",
        "MaxNeutralAgeReached",
    ),
    "RangeFamily": ("RangeRelativeToPrevious", "TrueRangeRelativeToPrevious", "BodyRelativeToPrevious"),
    "VolumeFamily": ("VolumeRelativeToRollingMean", "VolumeRelativeToPrevious", "VolumeRelativeToSessionMean"),
    "EfficiencyFamily": ("EfficiencyRatio", "CloseLocation"),
    "GraphFamily": ("MemoryStrength", "BranchEntropy"),
}

LOW_DOF_SET = (
    "LoopPhase",
    "NeutralAgeBucket",
    "DestinationFamily",
    "RangeRelativeToPrevious",
    "VolumeRelativeToRollingMean",
)
MARKET_ONLY_SET = (
    "RangeRelativeToPrevious",
    "TrueRangeRelativeToPrevious",
    "BodyRelativeToPrevious",
    "VolumeRelativeToRollingMean",
    "EfficiencyRatio",
)
STATE_ONLY_SET = (
    "LoopPhase",
    "NeutralAgeBucket",
    "DestinationFamily",
    "ReturnFamily",
    "FailureNode",
    "MaxNeutralAgeReached",
)
GRAPH_ONLY_SET = ("MemoryStrength", "BranchEntropy")
STATE_MARKET_SET = (
    "LoopPhase",
    "NeutralAgeBucket",
    "DestinationFamily",
    "RangeRelativeToPrevious",
    "TrueRangeRelativeToPrevious",
    "VolumeRelativeToRollingMean",
    "EfficiencyRatio",
)
INTERPRETABILITY = {
    "LoopPhase": "Where the current bar sits in the closed APVA loop.",
    "NeutralAgeBucket": "Neutral maturity and deterioration level.",
    "DestinationFamily": "Excursion type after Neutral failure.",
    "ReturnFamily": "Family that routes the loop back toward Neutral.",
    "FormationSourceFamily": "Source family immediately before Neutral_Age1 formation.",
    "FailureNode": "Last Neutral node before excursion.",
    "MaxNeutralAgeReached": "Maximum Neutral maturity reached inside the loop.",
    "RangeRelativeToPrevious": "Price range expansion or collapse.",
    "TrueRangeRelativeToPrevious": "Gap-aware range expansion or collapse.",
    "BodyRelativeToPrevious": "Directional body expansion or collapse.",
    "VolumeRelativeToRollingMean": "Participation versus recent activity.",
    "VolumeRelativeToPrevious": "Immediate participation expansion or contraction.",
    "VolumeRelativeToSessionMean": "Participation versus session baseline.",
    "EfficiencyRatio": "Directional efficiency within the bar.",
    "CloseLocation": "Close position inside the bar range.",
    "BranchEntropy": "Structural uncertainty of the current node.",
    "MemoryStrength": "State persistence inherited from information decay logic.",
}


@dataclass
class Observation:
    instrument: str
    index: int
    loop_id: str
    values: dict[str, str]
    target: str
    state_age_node: str


@dataclass
class Metrics:
    top1: float
    top2: float
    brier: float
    entropy: float


@dataclass
class InventoryRow:
    signal: str
    available: int
    missing: int
    percent: float


@dataclass
class PredictionRow:
    name: str
    signals: tuple[str, ...]
    metrics: Metrics


@dataclass
class AblationRow:
    removed: str
    delta_top1: float
    delta_top2: float
    delta_brier: float
    delta_entropy: float


@dataclass
class GreedyRow:
    step: int
    signal: str
    metrics: Metrics
    improvement: float


@dataclass
class NecessityRow:
    signal: str
    classification: str
    reason: str


@dataclass
class ReplicationRow:
    signal_set: str
    top1_by_instrument: dict[str, float]
    variance: float
    assessment: str


@dataclass
class OutcomeRow:
    signal_set: str
    loop_phase: str
    dr: dict[int, float] = field(default_factory=dict)
    continuation5: float = 0.0
    failure5: float = 0.0
    flat5: float = 0.0


@dataclass
class Recommendation:
    classification: str
    minimal_set: str
    removed: str
    core: str
    support: str
    redundant: str
    harmful: str
    next_step: str
    reason: str


@dataclass
class Result:
    instrument: str
    source_paths: list
    observations: list[Observation]
    inventory: list[InventoryRow]
    baseline: PredictionRow
    single: list[PredictionRow]
    families: list[PredictionRow]
    ablations: list[AblationRow]
    family_ablations: list[AblationRow]
    greedy: list[GreedyRow]
    evaluations: list[PredictionRow]
    low_dof: PredictionRow
    market_only: PredictionRow
    state_only: PredictionRow
    graph_only: PredictionRow
    state_market: PredictionRow
    replication: list[ReplicationRow]
    necessity: list[NecessityRow]
    outcomes: list[OutcomeRow]
    recommendation: Recommendation


def bucket_numeric(value: float | None, kind: str) -> str:
    if value is None:
        return "Missing"
    if kind == "efficiency" or kind == "close":
        if value < 0.33:
            return "Low"
        if value < 0.66:
            return "Medium"
        return "High"
    if kind == "graph":
        if value < 0.25:
            return "Low"
        if value < 0.50:
            return "Medium"
        if value < 0.75:
            return "High"
        return "VeryHigh"
    if value < 0.90:
        return "Contracting"
    if value <= 1.10:
        return "Stable"
    return "Expanding"


def context_lookup(contexts: list) -> dict[str, object]:
    return {context.instrument: context for context in contexts}


def signal_values_for_bar(context, loop, offset: int, phase: str) -> dict[str, str]:
    index = loop.start + offset
    node = loop.nodes[offset]
    bar = context.bars[index]
    fail = failure_node(loop)
    values = {
        "LoopPhase": machine_phase(phase),
        "NeutralAgeBucket": node[1] if node[0] == "NeutralProcessing" else "NonNeutral",
        "DestinationFamily": destination_family(loop),
        "ReturnFamily": return_family(loop),
        "FormationSourceFamily": loop.nodes[-2][0] if len(loop.nodes) > 1 else "None",
        "FailureNode": node_text(fail) if fail else "None",
        "MaxNeutralAgeReached": max_neutral_age(loop),
        "RangeRelativeToPrevious": bucket_numeric(bar.range_relative_to_previous, "relative"),
        "TrueRangeRelativeToPrevious": bucket_numeric(bar.true_range_relative_to_previous, "relative"),
        "BodyRelativeToPrevious": bucket_numeric(bar.body_relative_to_previous, "relative"),
        "VolumeRelativeToRollingMean": bucket_numeric(bar.volume_relative_to_rolling_mean, "relative"),
        "VolumeRelativeToPrevious": bucket_numeric(bar.volume_relative_to_previous, "relative"),
        "VolumeRelativeToSessionMean": bucket_numeric(bar.volume_relative_to_session_mean, "relative"),
        "EfficiencyRatio": bucket_numeric(efficiency_ratio(bar), "efficiency"),
        "CloseLocation": bucket_numeric(bar.close_location, "close"),
        "MemoryStrength": bucket_numeric(context.memory.get(node), "graph"),
        "BranchEntropy": bucket_numeric(context.entropy.get(node), "graph"),
    }
    return values


def build_observations(contexts: list) -> list[Observation]:
    lookup = context_lookup(contexts)
    observations = []
    for context in contexts:
        for loop in complete_loops(build_loops([context])):
            phases = loop_phases(loop)
            for offset in range(len(loop.nodes) - 1):
                phase = phases[offset]
                target = machine_phase(phases[offset + 1])
                values = signal_values_for_bar(context, loop, offset, phase)
                observations.append(Observation(
                    context.instrument,
                    loop.start + offset,
                    loop.loop_id,
                    values,
                    target,
                    node_text(loop.nodes[offset]),
                ))
    return observations


def distribution(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: count / total for key, count in counter.items()}


def normalized_entropy(probs: dict[str, float]) -> float:
    if not probs:
        return 0.0
    value = 0.0
    for probability in probs.values():
        if probability > 0:
            value -= probability * math.log2(probability)
    maximum = math.log2(len(probs)) if len(probs) > 1 else 1.0
    return safe_div(value, maximum) or 0.0


def key_for(obs: Observation, signals: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(obs.values.get(signal, "Missing") for signal in signals)


def evaluate(observations: list[Observation], signals: tuple[str, ...]) -> Metrics:
    if not observations:
        return Metrics(0.0, 0.0, 0.0, 0.0)
    grouped = defaultdict(Counter)
    for obs in observations:
        grouped[key_for(obs, signals)][obs.target] += 1
    top1 = top2 = 0
    brier_total = entropy_total = 0.0
    phase_set = set(PHASE_MACHINE)
    for obs in observations:
        probs = distribution(grouped[key_for(obs, signals)])
        ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        if ranked and ranked[0][0] == obs.target:
            top1 += 1
        if obs.target in [key for key, _value in ranked[:2]]:
            top2 += 1
        brier_total += sum(((1.0 if phase == obs.target else 0.0) - probs.get(phase, 0.0)) ** 2 for phase in phase_set) / len(phase_set)
        entropy_total += normalized_entropy(probs)
    total = len(observations)
    return Metrics(top1 / total, top2 / total, brier_total / total, entropy_total / total)


def prediction_row(name: str, observations: list[Observation], signals: tuple[str, ...]) -> PredictionRow:
    return PredictionRow(name, signals, evaluate(observations, signals))


def inventory_rows(observations: list[Observation]) -> list[InventoryRow]:
    rows = []
    total = len(observations)
    for signal in CANDIDATE_SIGNALS:
        available = sum(1 for obs in observations if obs.values.get(signal, "Missing") != "Missing")
        missing = total - available
        rows.append(InventoryRow(signal, available, missing, safe_div(available, total) or 0.0))
    return rows


def ablation_rows(observations: list[Observation], full_metrics: Metrics) -> list[AblationRow]:
    rows = []
    full = tuple(CANDIDATE_SIGNALS)
    for signal in CANDIDATE_SIGNALS:
        reduced = tuple(item for item in full if item != signal)
        metrics = evaluate(observations, reduced)
        rows.append(AblationRow(
            signal,
            full_metrics.top1 - metrics.top1,
            full_metrics.top2 - metrics.top2,
            metrics.brier - full_metrics.brier,
            metrics.entropy - full_metrics.entropy,
        ))
    return rows


def family_ablation_rows(observations: list[Observation], full_metrics: Metrics) -> list[AblationRow]:
    rows = []
    full = tuple(CANDIDATE_SIGNALS)
    for family, signals in SIGNAL_FAMILIES.items():
        reduced = tuple(signal for signal in full if signal not in signals)
        metrics = evaluate(observations, reduced)
        rows.append(AblationRow(
            family,
            full_metrics.top1 - metrics.top1,
            full_metrics.top2 - metrics.top2,
            metrics.brier - full_metrics.brier,
            metrics.entropy - full_metrics.entropy,
        ))
    return rows


def greedy_rows(observations: list[Observation], single_rows: list[PredictionRow]) -> list[GreedyRow]:
    if not single_rows:
        return []
    best = max(single_rows, key=lambda row: row.metrics.top1)
    selected = [best.signals[0]]
    current = best.metrics
    rows = [GreedyRow(1, selected[0], current, current.top1)]
    step = 2
    while True:
        candidates = []
        for signal in CANDIDATE_SIGNALS:
            if signal in selected:
                continue
            trial = tuple(selected + [signal])
            metrics = evaluate(observations, trial)
            candidates.append((signal, metrics))
        if not candidates:
            break
        signal, metrics = max(candidates, key=lambda item: item[1].top1)
        improvement = metrics.top1 - current.top1
        if improvement < 0.01:
            break
        selected.append(signal)
        current = metrics
        rows.append(GreedyRow(step, signal, current, improvement))
        step += 1
    return rows


def necessity_rows(ablations: list[AblationRow]) -> list[NecessityRow]:
    rows = []
    for row in ablations:
        if row.delta_top1 < 0 or row.delta_brier < 0:
            classification = "HarmfulSignal"
            reason = "Removal improves Top1 or Brier."
        elif row.delta_top1 >= 0.02 or row.delta_brier >= 0.01:
            classification = "CoreSignal"
            reason = "Removal causes >=2% Top1 loss or material Brier worsening."
        elif row.delta_top1 >= 0.005:
            classification = "SupportSignal"
            reason = "Removal causes 0.5% to 2% Top1 loss."
        else:
            classification = "RedundantSignal"
            reason = "Removal causes <0.5% Top1 loss."
        rows.append(NecessityRow(row.removed, classification, reason))
    return rows


def replication_rows(context_results: list[Result] | None, set_rows: list[PredictionRow]) -> list[ReplicationRow]:
    if not context_results:
        return []
    rows = []
    for set_row in set_rows:
        values = {}
        for result in context_results:
            match = next((row for row in result.evaluations if row.name == set_row.name), None)
            if match:
                values[result.instrument] = match.metrics.top1
        variance = statistics.pvariance(values.values()) if len(values) > 1 else 0.0
        assessment = "Replicated" if len(values) >= 2 and variance <= 0.01 else "Variable"
        rows.append(ReplicationRow(set_row.name, values, variance, assessment))
    return rows


def outcome_rows(observations: list[Observation], contexts: list, signal_sets: list[PredictionRow]) -> list[OutcomeRow]:
    lookup = context_lookup(contexts)
    selected_names = {row.name for row in signal_sets[:6]}
    output = []
    for set_name in selected_names:
        for phase in PHASE_MACHINE:
            obs_rows = [obs for obs in observations if obs.values.get("LoopPhase") == phase]
            row = OutcomeRow(set_name, phase)
            for horizon in OUTCOME_HORIZONS:
                values = []
                for obs in obs_rows:
                    context = lookup[obs.instrument]
                    value = directional_forward(context.bars, obs.index, horizon)
                    if value is not None:
                        values.append(value)
                row.dr[horizon] = mean(values)
                if horizon == 5:
                    row.continuation5 = mean(1.0 if value > 0 else 0.0 for value in values)
                    row.failure5 = mean(1.0 if value < 0 else 0.0 for value in values)
                    row.flat5 = mean(1.0 if value == 0 else 0.0 for value in values)
            output.append(row)
    return output


def make_recommendation(greedy: list[GreedyRow], evaluations: list[PredictionRow], necessity: list[NecessityRow],
                        baseline: PredictionRow) -> Recommendation:
    minimal = tuple(row.signal for row in greedy)
    minimal_metrics = greedy[-1].metrics if greedy else Metrics(0.0, 0.0, 0.0, 0.0)
    retained = safe_div(minimal_metrics.top1, baseline.metrics.top1) or 0.0
    if len(minimal) <= 5 and retained >= 0.95:
        classification = "StrongMinimalSet"
        next_step = "ProceedToRealTimeStateMachine"
    elif retained >= 0.90:
        classification = "PracticalMinimalSet"
        next_step = "ProceedToRealTimeStateMachine"
    elif retained >= 0.80:
        classification = "WeakMinimalSet"
        next_step = "ProceedToTradeTimingAudit"
    else:
        classification = "NoMinimalSet"
        next_step = "CollectMoreData"
    by_class = defaultdict(list)
    for row in necessity:
        by_class[row.classification].append(row.signal)
    removed = [signal for signal in CANDIDATE_SIGNALS if signal not in minimal]
    return Recommendation(
        classification,
        ", ".join(minimal) or "None",
        ", ".join(removed) or "None",
        ", ".join(by_class["CoreSignal"]) or "None",
        ", ".join(by_class["SupportSignal"]) or "None",
        ", ".join(by_class["RedundantSignal"]) or "None",
        ", ".join(by_class["HarmfulSignal"]) or "None",
        next_step,
        f"Greedy minimal Top1 retained {pct(retained)} of the MinimalLoopPhase baseline.",
    )


def build_result(instrument: str, source_paths: list, contexts: list, instrument_results: list[Result] | None = None) -> Result:
    observations = build_observations(contexts)
    inventory = inventory_rows(observations)
    baseline = prediction_row("MinimalLoopPhase", observations, ("LoopPhase",))
    single = [prediction_row(signal, observations, (signal,)) for signal in CANDIDATE_SIGNALS]
    single.sort(key=lambda row: row.metrics.top1, reverse=True)
    families = [prediction_row(family, observations, tuple(signals)) for family, signals in SIGNAL_FAMILIES.items()]
    families.sort(key=lambda row: row.metrics.top1, reverse=True)
    full = prediction_row("FullCandidateSignalSet", observations, tuple(CANDIDATE_SIGNALS))
    ablations = ablation_rows(observations, full.metrics)
    family_ablations = family_ablation_rows(observations, full.metrics)
    greedy = greedy_rows(observations, single)
    minimal_signals = tuple(row.signal for row in greedy) if greedy else ("LoopPhase",)
    minimal = prediction_row("GreedyMinimalSet", observations, minimal_signals)
    for obs in observations:
        obs.values["StateAgeNode"] = obs.state_age_node
    full_state = prediction_row("FullStateAge", observations, ("StateAgeNode",))
    low_dof = prediction_row("LowDoFSignalSet", observations, LOW_DOF_SET)
    market = prediction_row("MarketOnly", observations, MARKET_ONLY_SET)
    state = prediction_row("StateOnly", observations, STATE_ONLY_SET)
    graph = prediction_row("GraphOnly", observations, GRAPH_ONLY_SET)
    hybrid = prediction_row("StateMarketHybrid", observations, STATE_MARKET_SET)
    evaluations = [full_state, baseline, full, minimal, low_dof, market, state, graph, hybrid]
    necessity = necessity_rows(ablations)
    replication = replication_rows(instrument_results, evaluations) if instrument_results else []
    outcomes = outcome_rows(observations, contexts, evaluations)
    recommendation = make_recommendation(greedy, evaluations, necessity, baseline)
    return Result(
        instrument,
        source_paths,
        observations,
        inventory,
        baseline,
        single,
        families,
        ablations,
        family_ablations,
        greedy,
        evaluations,
        low_dof,
        market,
        state,
        graph,
        hybrid,
        replication,
        necessity,
        outcomes,
        recommendation,
    )


def metric_line(row: PredictionRow) -> str:
    return f"{row.name} | {pct(row.metrics.top1)} | {pct(row.metrics.top2)} | {fmt(row.metrics.brier)} | {fmt(row.metrics.entropy)}"


def append_common(lines: list[str], result: Result) -> None:
    lines += ["", "1. Baseline Loop Phase Model", "Model | Top1Accuracy | Top2Accuracy | BrierScore | Entropy"]
    lines.append(metric_line(result.baseline))

    lines += ["", "2. Candidate Signal Inventory", "Signal | AvailableCount | MissingCount | PercentAvailable"]
    for row in result.inventory:
        lines.append(f"{row.signal} | {row.available} | {row.missing} | {pct(row.percent)}")

    lines += ["", "3. Single Signal Prediction Table", "Signal | Top1Accuracy | Top2Accuracy | BrierScore | Entropy"]
    for row in result.single:
        lines.append(metric_line(row))

    lines += ["", "4. Signal Family Prediction Table", "SignalFamily | Top1Accuracy | Top2Accuracy | BrierScore | Entropy"]
    for row in result.families:
        lines.append(metric_line(row))

    lines += ["", "5. Signal Ablation Table", "RemovedSignal | DeltaTop1 | DeltaTop2 | DeltaBrier | DeltaEntropy"]
    for row in sorted(result.ablations, key=lambda item: item.delta_top1, reverse=True):
        lines.append(f"{row.removed} | {pct(row.delta_top1)} | {pct(row.delta_top2)} | {fmt(row.delta_brier)} | {fmt(row.delta_entropy)}")

    lines += ["", "6. Family Ablation Table", "RemovedFamily | DeltaTop1 | DeltaTop2 | DeltaBrier | DeltaEntropy"]
    for row in sorted(result.family_ablations, key=lambda item: item.delta_top1, reverse=True):
        lines.append(f"{row.removed} | {pct(row.delta_top1)} | {pct(row.delta_top2)} | {fmt(row.delta_brier)} | {fmt(row.delta_entropy)}")

    lines += ["", "7. Greedy Compression Table", "Step | AddedSignal | Top1Accuracy | Top2Accuracy | BrierScore | Entropy | Improvement"]
    for row in result.greedy:
        lines.append(f"{row.step} | {row.signal} | {pct(row.metrics.top1)} | {pct(row.metrics.top2)} | {fmt(row.metrics.brier)} | {fmt(row.metrics.entropy)} | {pct(row.improvement)}")

    lines += ["", "8. Minimal Signal Set Evaluation", "SignalSet | Signals | Top1Accuracy | Top2Accuracy | BrierScore | Entropy | InformationRetained"]
    base_top1 = result.baseline.metrics.top1
    for row in result.evaluations:
        retained = safe_div(row.metrics.top1, base_top1) or 0.0
        lines.append(f"{row.name} | {', '.join(row.signals)} | {pct(row.metrics.top1)} | {pct(row.metrics.top2)} | {fmt(row.metrics.brier)} | {fmt(row.metrics.entropy)} | {pct(retained)}")

    for number, title, row in (
        (9, "Low-DoF Signal Set Table", result.low_dof),
        (10, "Market-Only Table", result.market_only),
        (11, "State-Only Table", result.state_only),
        (12, "Graph-Only Table", result.graph_only),
        (13, "State + Market Hybrid Table", result.state_market),
    ):
        lines += ["", f"{number}. {title}", "SignalSet | Top1Accuracy | Top2Accuracy | BrierScore | Entropy"]
        lines.append(metric_line(row))

    lines += ["", "14. Cross-Instrument Replication Table", "SignalSet | Top1_6E | Top1_CL | Top1_NQ | Variance | ReplicationAssessment"]
    for row in result.replication:
        lines.append(f"{row.signal_set} | {pct(row.top1_by_instrument.get('6E', 0.0))} | {pct(row.top1_by_instrument.get('CL', 0.0))} | {pct(row.top1_by_instrument.get('NQ', 0.0))} | {fmt(row.variance)} | {row.assessment}")

    lines += ["", "15. Signal Necessity Table", "Signal | NecessityClass | Reason"]
    for row in result.necessity:
        lines.append(f"{row.signal} | {row.classification} | {row.reason}")

    lines += ["", "16. Interpretability Table", "Signal | MechanicalRole"]
    for signal in CANDIDATE_SIGNALS:
        lines.append(f"{signal} | {INTERPRETABILITY.get(signal, 'Mechanical APVA signal.')}")

    lines += ["", "17. Outcome Diagnostics Table", "SignalSet | LoopPhase | DRFwd1 | DRFwd3 | DRFwd5 | DRFwd10 | ContinuationRate5 | FailureRate5 | FlatRate5"]
    for row in result.outcomes[:120]:
        lines.append(f"{row.signal_set} | {row.loop_phase} | {fmt(row.dr.get(1, 0.0))} | {fmt(row.dr.get(3, 0.0))} | {fmt(row.dr.get(5, 0.0))} | {fmt(row.dr.get(10, 0.0))} | {pct(row.continuation5)} | {pct(row.failure5)} | {pct(row.flat5)}")

    rec = result.recommendation
    lines += [
        "",
        "18. Recommendation",
        f"Classification: {rec.classification}",
        f"MinimalSignalSet: {rec.minimal_set}",
        f"RemovedSignals: {rec.removed}",
        f"CoreSignals: {rec.core}",
        f"SupportSignals: {rec.support}",
        f"RedundantSignals: {rec.redundant}",
        f"HarmfulSignals: {rec.harmful}",
        f"RecommendedNextStep: {rec.next_step}",
        f"Reason: {rec.reason}",
        "",
        "19. Low-DoF Audit",
        "Uses only existing APVA states and OHLCV-derived variables.",
        "No new APVA states.",
        "No new APVA families.",
        "No context.",
        "No arbitration.",
        "No persistence.",
        "No discretionary phase.",
        "No optimization.",
        "No fitting.",
        "No machine learning.",
        "No trading logic.",
        "No forward returns used in signal selection.",
    ]


def rankings(result: Result) -> list[str]:
    core = [row for row in result.necessity if row.classification == "CoreSignal"]
    redundant = [row for row in result.necessity if row.classification == "RedundantSignal"]
    harmful = [row for row in result.necessity if row.classification == "HarmfulSignal"]
    return [
        "",
        "RANKINGS",
        "1. Best single signals: " + "; ".join(f"{row.name}={pct(row.metrics.top1)}" for row in result.single[:8]),
        "2. Best signal families: " + "; ".join(f"{row.name}={pct(row.metrics.top1)}" for row in result.families),
        "3. Most necessary signals: " + "; ".join(row.signal for row in core) or "None",
        "4. Most redundant signals: " + "; ".join(row.signal for row in redundant[:8]) or "None",
        "5. Harmful signals: " + "; ".join(row.signal for row in harmful) or "None",
        "6. Best minimal signal sets: " + "; ".join(f"{row.name}={pct(row.metrics.top1)}" for row in sorted(result.evaluations, key=lambda item: item.metrics.top1, reverse=True)[:6]),
        "7. Best cross-instrument replicated sets: " + "; ".join(f"{row.signal_set}:{row.assessment}" for row in result.replication[:8]),
        f"8. Best market-only representation: {pct(result.market_only.metrics.top1)}",
        f"9. Best state-market hybrid representation: {pct(result.state_market.metrics.top1)}",
        f"10. Recommended APVA minimal signal set: {result.recommendation.minimal_set}",
    ]


def write_report(result: Result, out_root: Path, aggregate: bool = False) -> None:
    if aggregate:
        path = out_root / "SignalMinimalism81" / "SignalMinimalism81_All.txt"
        title = "APVA Signal Minimalism Study v0.1 - Aggregate"
    else:
        path = out_root / result.instrument / f"SignalMinimalism81_{result.instrument}.txt"
        title = f"APVA Signal Minimalism Study v0.1 - {result.instrument}"
    ensure_dir(path.parent)
    lines = [
        title,
        "=" * 96,
        f"Instrument: {result.instrument}",
        f"Input path(s): {', '.join(str(path) for path in result.source_paths)}",
        f"Observation count: {len(result.observations)}",
    ]
    append_common(lines, result)
    lines += rankings(result)
    lines += [
        "",
        "RESEARCH NOTES",
        "Mechanical only.",
        "Questions: What is the smallest APVA representation? Can StateAge be replaced by LoopPhase? Are graph signals necessary? Are range and volume enough for market behavior?",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(result: Result) -> None:
    if not result.observations:
        raise RuntimeError(f"{result.instrument}: no observations.")
    if len(result.inventory) != len(CANDIDATE_SIGNALS):
        raise RuntimeError(f"{result.instrument}: inventory incomplete.")
    if not result.evaluations:
        raise RuntimeError(f"{result.instrument}: evaluations missing.")
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
    print(f"Wrote {len(instrument_results)} per-instrument Study81 report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
