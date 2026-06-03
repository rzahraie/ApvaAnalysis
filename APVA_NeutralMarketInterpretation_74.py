#!/usr/bin/env python3
"""APVA Neutral Market Interpretation Study v2.0.

Rerun the NeutralProcessing lifecycle interpretation with the real OHLCV
observables exposed by the upgraded shared evidence loader.

Research only. No trading, fitting, optimization, or machine learning.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from APVA_BranchForecast_65 import build_stream
from APVA_InformationContribution_63 import thresholds
from APVA_InformationDecay_57 import study as decay_study
from APVA_MinimalEngine_52 import instrument_columns, load_results, safe_mean
from APVA_NeutralBackbone_71 import NEUTRAL_CHAIN, NEUTRAL_NODES, family_for_state, is_neutral_node
from APVA_NeutralMarketInterpretation_72 import (
    DestinationRow,
    ExcursionRisk,
    LifecycleRow,
    OutcomeRow,
    ReturnRow,
    attach_replication,
    destination_rows,
    excursion_risks,
    lifecycle_rows,
    neutral_indices,
    outcome_rows,
    return_rows,
)
from APVA_NodeImportance_58 import aggregate_rows, local_rows, score_rows
from APVA_NodeNecessity_70 import build_result as necessity_result
from APVA_NodeNecessity_70 import validate as validate_necessity
from APVA_ProcessGraph_54 import Node, node_for, node_text
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

EPSILON = 1e-12
HORIZONS = (1, 3, 5, 10)

STUDY72_BASELINE = {
    ("NeutralProcessing", "1"): "Gateway",
    ("NeutralProcessing", "2"): "Gateway",
    ("NeutralProcessing", "3"): "Stabilization",
    ("NeutralProcessing", "4"): "Equilibrium",
    ("NeutralProcessing", "5"): "Equilibrium",
    ("NeutralProcessing", "6-10"): "StaleEquilibrium",
    ("NeutralProcessing", "11-20"): "LongDrift",
    ("NeutralProcessing", "21+"): "LongDrift",
}


@dataclass
class PriceProfile:
    node: Node
    mean_range: float
    median_range: float
    mean_body: float
    median_body: float
    mean_true_range: float
    median_true_range: float
    mean_close_location: float
    mean_directional_move: float
    std_range: float
    std_body: float
    std_true_range: float


@dataclass
class VolumeProfile:
    node: Node
    mean_volume: float
    median_volume: float
    mean_relative_previous: float
    mean_relative_session: float
    mean_relative_rolling: float
    expansion_rate: float
    contraction_rate: float
    participation_label: str


@dataclass
class RangeExpansionProfile:
    node: Node
    range_expansion_rate: float
    range_contraction_rate: float
    mean_range_relative_previous: float
    mean_body_relative_previous: float
    mean_true_range_relative_previous: float


@dataclass
class EfficiencyProfile:
    node: Node
    mean_efficiency: float
    median_efficiency: float


@dataclass
class VolumeEfficiencyProfile:
    node: Node
    range_volume_ratio: float
    volume_range_ratio: float
    correlation: float


@dataclass
class PolarityProfile:
    node: Node
    black_rate: float
    red_rate: float
    flip_rate: float
    same_rate: float


@dataclass
class EvolutionRow:
    transition: str
    delta_range: float
    delta_body: float
    delta_true_range: float
    delta_volume: float
    delta_efficiency: float
    delta_memory: float
    delta_entropy: float
    delta_excursion: float


@dataclass
class ScoreRow:
    node: Node
    score: float
    components: dict[str, float]


@dataclass
class InterpretationRow:
    node: Node
    interpretation: str
    evidence: str


@dataclass
class ComparisonRow:
    node: Node
    study72: str
    study74: str
    result: str
    reason: str


@dataclass
class Result:
    instrument: str
    source_paths: list
    bars: list
    rows: list
    nodes: list[Node]
    necessity: object
    lifecycle: dict[Node, LifecycleRow]
    price: dict[Node, PriceProfile]
    volume: dict[Node, VolumeProfile]
    range_expansion: dict[Node, RangeExpansionProfile]
    efficiency: dict[Node, EfficiencyProfile]
    volume_efficiency: dict[Node, VolumeEfficiencyProfile]
    polarity: dict[Node, PolarityProfile]
    excursion: dict[Node, ExcursionRisk]
    destinations: dict[tuple[Node, str], DestinationRow]
    returns: dict[Node, ReturnRow]
    evolution: dict[str, EvolutionRow]
    equilibrium: dict[Node, ScoreRow]
    staleness: dict[Node, ScoreRow]
    gateway: dict[Node, ScoreRow]
    interpretations: dict[Node, InterpretationRow]
    comparison: dict[Node, ComparisonRow]
    outcomes: dict[Node, OutcomeRow]


def mean(values: Iterable[float | bool | None]) -> float:
    return safe_mean([float(value) for value in values if value is not None])


def median(values: Iterable[float | None]) -> float:
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else 0.0


def stddev(values: Iterable[float | None]) -> float:
    values = [value for value in values if value is not None]
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def correlation(left: Iterable[float | None], right: Iterable[float | None]) -> float:
    pairs = [(a, b) for a, b in zip(left, right) if a is not None and b is not None]
    if len(pairs) < 2:
        return 0.0
    xs, ys = zip(*pairs)
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in pairs)
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else 0.0


def normalize_map(values: dict[Node, float], inverse: bool = False) -> dict[Node, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high == low:
        normalized = {node: 0.0 for node in values}
    else:
        normalized = {node: (value - low) / (high - low) for node, value in values.items()}
    if inverse:
        return {node: 1.0 - value for node, value in normalized.items()}
    return normalized


def attr(bar, name: str) -> float | None:
    return getattr(bar, name, None)


def real_ohlcv_available(bars: list) -> tuple[bool, bool]:
    return (
        any(getattr(bar, "has_ohlc", False) for bar in bars),
        any(getattr(bar, "has_volume", False) for bar in bars),
    )


def price_profiles(bars: list, indices: dict[Node, list[int]]) -> dict[Node, PriceProfile]:
    output = {}
    for node, indexes in indices.items():
        ranges = [attr(bars[index], "bar_range") for index in indexes]
        bodies = [attr(bars[index], "body") for index in indexes]
        true_ranges = [attr(bars[index], "true_range") for index in indexes]
        close_locations = [attr(bars[index], "close_location") for index in indexes]
        directions = [
            bars[index].close - bars[index].open
            for index in indexes
            if bars[index].close is not None and bars[index].open is not None
        ]
        output[node] = PriceProfile(
            node,
            mean(ranges),
            median(ranges),
            mean(bodies),
            median(bodies),
            mean(true_ranges),
            median(true_ranges),
            mean(close_locations),
            mean(directions),
            stddev(ranges),
            stddev(bodies),
            stddev(true_ranges),
        )
    return output


def volume_profiles(bars: list, indices: dict[Node, list[int]]) -> dict[Node, VolumeProfile]:
    output = {}
    for node, indexes in indices.items():
        volumes = [attr(bars[index], "volume") for index in indexes]
        rel_prev = [attr(bars[index], "volume_relative_to_previous") for index in indexes]
        rel_session = [attr(bars[index], "volume_relative_to_session_mean") for index in indexes]
        rel_rolling = [attr(bars[index], "volume_relative_to_rolling_mean") for index in indexes]
        expansion = [getattr(bars[index], "volume_expansion_flag", None) for index in indexes]
        contraction = [getattr(bars[index], "volume_contraction_flag", None) for index in indexes]
        rolling_mean = mean(rel_rolling)
        if rolling_mean > 1.03:
            label = "IncreasingParticipation"
        elif rolling_mean < 0.97 and rolling_mean > 0:
            label = "DecliningParticipation"
        else:
            label = "StableParticipation"
        output[node] = VolumeProfile(
            node,
            mean(volumes),
            median(volumes),
            mean(rel_prev),
            mean(rel_session),
            rolling_mean,
            mean(expansion),
            mean(contraction),
            label,
        )
    return output


def range_expansion_profiles(bars: list, indices: dict[Node, list[int]]) -> dict[Node, RangeExpansionProfile]:
    output = {}
    for node, indexes in indices.items():
        range_expansion = [getattr(bars[index], "range_expansion_flag", None) for index in indexes]
        range_contraction = [getattr(bars[index], "range_contraction_flag", None) for index in indexes]
        output[node] = RangeExpansionProfile(
            node,
            mean(range_expansion),
            mean(range_contraction),
            mean(attr(bars[index], "range_relative_to_previous") for index in indexes),
            mean(attr(bars[index], "body_relative_to_previous") for index in indexes),
            mean(attr(bars[index], "true_range_relative_to_previous") for index in indexes),
        )
    return output


def efficiency_profiles(bars: list, indices: dict[Node, list[int]]) -> dict[Node, EfficiencyProfile]:
    output = {}
    for node, indexes in indices.items():
        values = []
        for index in indexes:
            body = attr(bars[index], "body")
            bar_range = attr(bars[index], "bar_range")
            if body is not None and bar_range is not None:
                values.append(body / max(bar_range, EPSILON))
        output[node] = EfficiencyProfile(node, mean(values), median(values))
    return output


def volume_efficiency_profiles(bars: list, indices: dict[Node, list[int]]) -> dict[Node, VolumeEfficiencyProfile]:
    output = {}
    for node, indexes in indices.items():
        ranges = [attr(bars[index], "bar_range") for index in indexes]
        volumes = [attr(bars[index], "volume") for index in indexes]
        rv = [rng / max(vol, EPSILON) for rng, vol in zip(ranges, volumes) if rng is not None and vol is not None]
        vr = [vol / max(rng, EPSILON) for rng, vol in zip(ranges, volumes) if rng is not None and vol is not None]
        output[node] = VolumeEfficiencyProfile(node, mean(rv), mean(vr), correlation(ranges, volumes))
    return output


def polarity_profiles(bars: list, indices: dict[Node, list[int]]) -> dict[Node, PolarityProfile]:
    output = {}
    for node, indexes in indices.items():
        polarities = Counter((bars[index].volume_polarity or "Neutral") for index in indexes)
        flips, same = [], []
        for index in indexes:
            if index == 0:
                continue
            current = bars[index].volume_polarity or "Neutral"
            previous = bars[index - 1].volume_polarity or "Neutral"
            flips.append(current != previous)
            same.append(current == previous)
        total = sum(polarities.values())
        output[node] = PolarityProfile(
            node,
            polarities["Black"] / total if total else 0.0,
            polarities["Red"] / total if total else 0.0,
            mean(flips),
            mean(same),
        )
    return output


def evolution_rows(price: dict[Node, PriceProfile], volume: dict[Node, VolumeProfile],
                   efficiency: dict[Node, EfficiencyProfile], lifecycle: dict[Node, LifecycleRow],
                   excursion: dict[Node, ExcursionRisk]) -> dict[str, EvolutionRow]:
    output = {}
    for source, target in NEUTRAL_CHAIN:
        key = f"{node_text(source)} -> {node_text(target)}"
        output[key] = EvolutionRow(
            key,
            price[target].mean_range - price[source].mean_range,
            price[target].mean_body - price[source].mean_body,
            price[target].mean_true_range - price[source].mean_true_range,
            volume[target].mean_volume - volume[source].mean_volume,
            efficiency[target].mean_efficiency - efficiency[source].mean_efficiency,
            lifecycle[target].memory_strength - lifecycle[source].memory_strength,
            lifecycle[target].branch_entropy - lifecycle[source].branch_entropy,
            excursion[target].exit_5 - excursion[source].exit_5,
        )
    return output


def gateway_scores(lifecycle: dict[Node, LifecycleRow], rows: list, excursion: dict[Node, ExcursionRisk]) -> dict[Node, ScoreRow]:
    incoming: dict[Node, set[str]] = defaultdict(set)
    outgoing: dict[Node, set[str]] = defaultdict(set)
    for row in rows:
        if is_neutral_node(row.node):
            incoming[row.node].add(row.previous)
            outgoing[row.node].add(row.next_node)
    incoming_count = {node: len(incoming[node]) for node in lifecycle}
    outgoing_count = {node: len(outgoing[node]) for node in lifecycle}
    turnover = {node: excursion[node].exit_1 for node in lifecycle}
    low_necessity = normalize_map({node: row.necessity for node, row in lifecycle.items()}, inverse=True)
    incoming_score = normalize_map(incoming_count)
    outgoing_score = normalize_map(outgoing_count)
    turnover_score = normalize_map(turnover)
    output = {}
    for node in lifecycle:
        components = {
            "IncomingCount": incoming_score[node],
            "OutgoingCount": outgoing_score[node],
            "LowNecessity": low_necessity[node],
            "HighTurnover": turnover_score[node],
        }
        output[node] = ScoreRow(node, mean(components.values()), components)
    return output


def equilibrium_scores(lifecycle: dict[Node, LifecycleRow], excursion: dict[Node, ExcursionRisk],
                       volume: dict[Node, VolumeProfile], price: dict[Node, PriceProfile]) -> dict[Node, ScoreRow]:
    inv_entropy = normalize_map({node: row.branch_entropy for node, row in lifecycle.items()}, inverse=True)
    memory = normalize_map({node: row.memory_strength for node, row in lifecycle.items()})
    inv_excursion = normalize_map({node: excursion[node].exit_5 for node in lifecycle}, inverse=True)
    volume_stability = normalize_map({node: abs(volume[node].mean_relative_rolling - 1.0) for node in lifecycle}, inverse=True)
    range_stability = normalize_map({node: price[node].std_range for node in lifecycle}, inverse=True)
    output = {}
    for node in lifecycle:
        components = {
            "InverseBranchEntropy": inv_entropy[node],
            "MemoryStrength": memory[node],
            "InverseExcursionRisk": inv_excursion[node],
            "VolumeStability": volume_stability[node],
            "RangeStability": range_stability[node],
        }
        output[node] = ScoreRow(node, mean(components.values()), components)
    return output


def staleness_scores(lifecycle: dict[Node, LifecycleRow], excursion: dict[Node, ExcursionRisk],
                     volume: dict[Node, VolumeProfile], price: dict[Node, PriceProfile]) -> dict[Node, ScoreRow]:
    entropy_growth = {}
    volume_decay = {}
    range_decay = {}
    previous = None
    for node in NEUTRAL_NODES:
        if previous is None:
            entropy_growth[node] = 0.0
            volume_decay[node] = 0.0
            range_decay[node] = 0.0
        else:
            entropy_growth[node] = max(0.0, lifecycle[node].branch_entropy - lifecycle[previous].branch_entropy)
            volume_decay[node] = max(0.0, volume[previous].mean_relative_rolling - volume[node].mean_relative_rolling)
            range_decay[node] = max(0.0, price[previous].mean_range - price[node].mean_range)
        previous = node
    excursion_score = normalize_map({node: excursion[node].exit_5 for node in lifecycle})
    entropy_score = normalize_map(entropy_growth)
    volume_score = normalize_map(volume_decay)
    range_score = normalize_map(range_decay)
    output = {}
    for node in lifecycle:
        components = {
            "ExcursionRisk": excursion_score[node],
            "EntropyGrowth": entropy_score[node],
            "VolumeDecay": volume_score[node],
            "RangeDecay": range_score[node],
        }
        output[node] = ScoreRow(node, mean(components.values()), components)
    return output


def interpret_nodes(equilibrium: dict[Node, ScoreRow], staleness: dict[Node, ScoreRow],
                    gateway: dict[Node, ScoreRow], excursion: dict[Node, ExcursionRisk],
                    volume: dict[Node, VolumeProfile], range_expansion: dict[Node, RangeExpansionProfile],
                    efficiency: dict[Node, EfficiencyProfile]) -> dict[Node, InterpretationRow]:
    top_equilibrium = max(equilibrium.values(), key=lambda row: row.score).node
    top_stale = max(staleness.values(), key=lambda row: row.score).node
    top_gateway = max(gateway.values(), key=lambda row: row.score).node
    top_excursion = max(excursion.values(), key=lambda row: row.exit_5).node
    output = {}
    for node in NEUTRAL_NODES:
        if node == top_gateway or node[1] == "1":
            label = "Gateway"
        elif node == top_equilibrium:
            label = "MaximumEquilibrium"
        elif node == top_stale:
            label = "StaleEquilibrium"
        elif node == top_excursion and node[1] not in {"1", "2"}:
            label = "PreExcursion"
        elif node[1] == "2":
            label = "EarlyStabilization"
        elif node[1] == "3":
            label = "HealthyStabilization"
        elif node[1] == "4":
            label = "MaximumEquilibrium" if equilibrium[node].score >= 0.75 else "LateEquilibrium"
        elif node[1] == "5":
            label = "LateEquilibrium"
        elif node[1] == "6-10":
            label = "StaleEquilibrium"
        elif node[1] in {"11-20", "21+"}:
            label = "LongDrift"
        else:
            label = "Unclear"
        evidence = (
            f"Equilibrium={fmt(equilibrium[node].score)}, "
            f"Staleness={fmt(staleness[node].score)}, "
            f"Gateway={fmt(gateway[node].score)}, "
            f"Exit5={pct(excursion[node].exit_5)}, "
            f"Volume={volume[node].participation_label}, "
            f"RangeExpansion={pct(range_expansion[node].range_expansion_rate)}, "
            f"Efficiency={fmt(efficiency[node].mean_efficiency)}"
        )
        output[node] = InterpretationRow(node, label, evidence)
    return output


def compare_with_study72(interpretations: dict[Node, InterpretationRow]) -> dict[Node, ComparisonRow]:
    equivalent = {
        "Gateway": {"Gateway"},
        "Stabilization": {"EarlyStabilization", "HealthyStabilization"},
        "Equilibrium": {"MaximumEquilibrium", "LateEquilibrium"},
        "StaleEquilibrium": {"StaleEquilibrium", "PreExcursion"},
        "LongDrift": {"LongDrift"},
    }
    output = {}
    for node in NEUTRAL_NODES:
        old = STUDY72_BASELINE[node]
        new = interpretations[node].interpretation
        if new == old or new in equivalent.get(old, set()):
            result = "Strengthened" if new in {"MaximumEquilibrium", "HealthyStabilization"} else "Confirmed"
            reason = "Study74 OHLCV interpretation matches the Study72 lifecycle bucket."
        elif old == "Equilibrium" and new == "StaleEquilibrium":
            result = "Weakened"
            reason = "The node still sits in the lifecycle but real market metrics shift it toward staleness."
        elif old == "StaleEquilibrium" and new == "LateEquilibrium":
            result = "Weakened"
            reason = "Real market metrics make the node look less stale than Study72 implied."
        else:
            result = "Rejected"
            reason = "Study74 OHLCV interpretation does not match the Study72 lifecycle bucket."
        output[node] = ComparisonRow(node, old, new, result, reason)
    return output


def recommendation(result: Result) -> list[str]:
    equilibrium_node = max(result.equilibrium.values(), key=lambda row: row.score).node
    stale_node = max(result.staleness.values(), key=lambda row: row.score).node
    gateway_node = max(result.gateway.values(), key=lambda row: row.score).node
    elevated_excursion = max(result.excursion.values(), key=lambda row: row.exit_5).node
    confirmed = sum(row.result in {"Confirmed", "Strengthened"} for row in result.comparison.values())
    classification = "NeutralLifecycleMarketConfirmed" if confirmed >= 6 else "NeutralLifecyclePartiallyConfirmed"
    if confirmed < 4:
        classification = "NeutralLifecycleMarketWeak"
    return [
        f"Classification: {classification}",
        f"EquilibriumNode: {node_text(equilibrium_node)}",
        f"StaleEquilibriumNode: {node_text(stale_node)}",
        f"GatewayNode: {node_text(gateway_node)}",
        f"ElevatedExcursionRiskNode: {node_text(elevated_excursion)}",
        f"Study72ConfirmedOrStrengthenedCount: {confirmed}",
        "Reason: Real OHLCV metrics support the lifecycle when gateway, stabilization, equilibrium, staleness, and drift labels align with Study72 buckets.",
    ]


def build_result(instrument: str, source_paths: list, bars: list, rows: list, nodes: list[Node],
                 node_rows: dict[Node, object]) -> Result:
    necessity = necessity_result(instrument, source_paths, bars, rows, nodes, node_rows)
    validate_necessity(necessity)
    indices = neutral_indices(nodes)
    lifecycle = lifecycle_rows(rows, indices, necessity, node_rows)
    price = price_profiles(bars, indices)
    volume = volume_profiles(bars, indices)
    range_expansion = range_expansion_profiles(bars, indices)
    efficiency = efficiency_profiles(bars, indices)
    volume_efficiency = volume_efficiency_profiles(bars, indices)
    polarity = polarity_profiles(bars, indices)
    excursion = excursion_risks(nodes, indices)
    destinations = destination_rows(nodes)
    returns = return_rows(nodes)
    evolution = evolution_rows(price, volume, efficiency, lifecycle, excursion)
    equilibrium = equilibrium_scores(lifecycle, excursion, volume, price)
    staleness = staleness_scores(lifecycle, excursion, volume, price)
    gateway = gateway_scores(lifecycle, rows, excursion)
    interp = interpret_nodes(equilibrium, staleness, gateway, excursion, volume, range_expansion, efficiency)
    comparison = compare_with_study72(interp)
    return Result(
        instrument, source_paths, bars, rows, nodes, necessity, lifecycle, price, volume, range_expansion,
        efficiency, volume_efficiency, polarity, excursion, destinations, returns, evolution, equilibrium,
        staleness, gateway, interp, comparison, outcome_rows(bars, indices),
    )


def lifecycle_line(row: LifecycleRow) -> str:
    return f"{node_text(row.node)} | {row.count} | {pct(row.occupancy)} | {row.replication_count} | {fmt(row.memory_strength)} | {fmt(row.branch_entropy)} | {fmt(row.necessity)}"


def components_text(row: ScoreRow) -> str:
    return ", ".join(f"{key}={fmt(value)}" for key, value in row.components.items())


def append_common(lines: list[str], result: Result) -> None:
    lines += ["", "1. Neutral Lifecycle Inventory", "Node | Count | Occupancy | ReplicationCount | MemoryStrength | BranchEntropy | NecessityScore"]
    lines += [lifecycle_line(row) for row in result.lifecycle.values()]
    lines += ["", "2. Real Price Profile", "Node | MeanBarRange | MedianBarRange | MeanBody | MedianBody | MeanTrueRange | MedianTrueRange | MeanCloseLocation | MeanDirectionalMove | StdDevRange | StdDevBody | StdDevTrueRange"]
    for row in result.price.values():
        lines.append(f"{node_text(row.node)} | {fmt(row.mean_range)} | {fmt(row.median_range)} | {fmt(row.mean_body)} | {fmt(row.median_body)} | {fmt(row.mean_true_range)} | {fmt(row.median_true_range)} | {fmt(row.mean_close_location)} | {fmt(row.mean_directional_move)} | {fmt(row.std_range)} | {fmt(row.std_body)} | {fmt(row.std_true_range)}")
    lines += ["", "3. Real Volume Profile", "Node | MeanVolume | MedianVolume | VolumeRelativeToPrevious | VolumeRelativeToSessionMean | VolumeRelativeToRollingMean | VolumeExpansionRate | VolumeContractionRate | ParticipationLabel"]
    for row in result.volume.values():
        lines.append(f"{node_text(row.node)} | {fmt(row.mean_volume)} | {fmt(row.median_volume)} | {fmt(row.mean_relative_previous)} | {fmt(row.mean_relative_session)} | {fmt(row.mean_relative_rolling)} | {pct(row.expansion_rate)} | {pct(row.contraction_rate)} | {row.participation_label}")
    lines += ["", "4. Range Expansion Profile", "Node | RangeExpansionRate | RangeContractionRate | MeanRangeRelativeToPrevious | MeanBodyRelativeToPrevious | MeanTrueRangeRelativeToPrevious"]
    for row in result.range_expansion.values():
        lines.append(f"{node_text(row.node)} | {pct(row.range_expansion_rate)} | {pct(row.range_contraction_rate)} | {fmt(row.mean_range_relative_previous)} | {fmt(row.mean_body_relative_previous)} | {fmt(row.mean_true_range_relative_previous)}")
    lines += ["", "5. Price Efficiency", "Node | MeanEfficiency | MedianEfficiency"]
    lines += [f"{node_text(row.node)} | {fmt(row.mean_efficiency)} | {fmt(row.median_efficiency)}" for row in result.efficiency.values()]
    lines += ["", "6. Volume Efficiency", "Node | RangeVolumeRatio | VolumeRangeRatio | RangeVsVolumeCorrelation"]
    lines += [f"{node_text(row.node)} | {fmt(row.range_volume_ratio)} | {fmt(row.volume_range_ratio)} | {fmt(row.correlation)}" for row in result.volume_efficiency.values()]
    lines += ["", "7. Polarity Structure", "Node | BlackRate | RedRate | PolarityFlipRate | SamePolarityRate"]
    lines += [f"{node_text(row.node)} | {pct(row.black_rate)} | {pct(row.red_rate)} | {pct(row.flip_rate)} | {pct(row.same_rate)}" for row in result.polarity.values()]
    lines += ["", "8. Excursion Risk", "Node | ExitNeutralWithin1 | ExitNeutralWithin2 | ExitNeutralWithin3 | ExitNeutralWithin5 | ExpectedBarsToExit"]
    lines += [f"{node_text(row.node)} | {pct(row.exit_1)} | {pct(row.exit_2)} | {pct(row.exit_3)} | {pct(row.exit_5)} | {fmt(row.expected_bars)}" for row in result.excursion.values()]
    lines += ["", "9. Excursion Destination", "NeutralNode | DestinationFamily | Count | Probability | ReplicationCount"]
    lines += [f"{node_text(row.node)} | {row.family} | {row.count} | {pct(row.probability)} | {row.replication_count}" for row in sorted(result.destinations.values(), key=lambda row: (-row.count, node_text(row.node), row.family))]
    lines += ["", "10. Lifecycle Evolution", "Transition | DeltaRange | DeltaBody | DeltaTrueRange | DeltaVolume | DeltaEfficiency | DeltaMemory | DeltaEntropy | DeltaExcursionRisk"]
    lines += [f"{row.transition} | {fmt(row.delta_range)} | {fmt(row.delta_body)} | {fmt(row.delta_true_range)} | {fmt(row.delta_volume)} | {fmt(row.delta_efficiency)} | {fmt(row.delta_memory)} | {fmt(row.delta_entropy)} | {pct(row.delta_excursion)}" for row in result.evolution.values()]
    lines += ["", "11. Equilibrium Audit", "Node | EquilibriumScore | Components"]
    lines += [f"{node_text(row.node)} | {fmt(row.score)} | {components_text(row)}" for row in sorted(result.equilibrium.values(), key=lambda row: (-row.score, node_text(row.node)))]
    lines += ["", "12. Staleness Audit", "Node | StalenessScore | Components"]
    lines += [f"{node_text(row.node)} | {fmt(row.score)} | {components_text(row)}" for row in sorted(result.staleness.values(), key=lambda row: (-row.score, node_text(row.node)))]
    lines += ["", "13. Gateway Audit", "Node | GatewayScore | Components"]
    lines += [f"{node_text(row.node)} | {fmt(row.score)} | {components_text(row)}" for row in sorted(result.gateway.values(), key=lambda row: (-row.score, node_text(row.node)))]
    lines += ["", "14. Market Interpretation", "Node | Interpretation | SupportingEvidence"]
    lines += [f"{node_text(row.node)} | {row.interpretation} | {row.evidence}" for row in result.interpretations.values()]
    lines += ["", "15. Cross-Instrument Replication", "Node | Interpretation | ReplicationCount"]
    lines += [f"{node_text(node)} | {result.interpretations[node].interpretation} | {result.lifecycle[node].replication_count}" for node in NEUTRAL_NODES]
    lines += ["", "16. Outcome Diagnostics", "Node | DRFwd1 | DRFwd3 | DRFwd5 | DRFwd10 | ContinuationRate1 | ContinuationRate3 | ContinuationRate5 | ContinuationRate10 | FailureRate5 | FlatRate5"]
    for row in result.outcomes.values():
        lines.append(f"{node_text(row.node)} | {fmt(row.mean_by_horizon[1])} | {fmt(row.mean_by_horizon[3])} | {fmt(row.mean_by_horizon[5])} | {fmt(row.mean_by_horizon[10])} | {pct(row.continuation_by_horizon[1])} | {pct(row.continuation_by_horizon[3])} | {pct(row.continuation_by_horizon[5])} | {pct(row.continuation_by_horizon[10])} | {pct(row.failure_by_horizon[5])} | {pct(row.flat_by_horizon[5])}")
    lines += ["", "17. Neutral Lifecycle Recommendation"] + recommendation(result)
    lines += ["", "18. Comparison With Study 72", "Node | Study72Interpretation | Study74Interpretation | Result | Reason"]
    lines += [f"{node_text(row.node)} | {row.study72} | {row.study74} | {row.result} | {row.reason}" for row in result.comparison.values()]
    append_audit(lines)
    lines += ["", "RESEARCH NOTES", "Mechanical only.", "Main question: Did Study 72 discover a genuine market lifecycle, or merely a graph lifecycle?", "Study 74 answers with real OHLCV behavior from the upgraded shared loader."]


def append_audit(lines: list[str]) -> None:
    lines += [
        "",
        "19. Low-DoF Audit",
        "Uses real OHLCV observables.",
        "No new APVA states.",
        "No new APVA families.",
        "No optimization.",
        "No fitting.",
        "No machine learning.",
        "No trading logic.",
    ]


def replication_lines(result: Result, instrument_results: list[Result]) -> list[str]:
    by_name = {row.instrument: row for row in instrument_results}
    names = instrument_columns(instrument_results)
    lines = ["", "Aggregate Cross-Instrument Interpretation Table", "Node | " + " | ".join(f"Interpretation_{name}" for name in names) + " | ReplicationCount"]
    for node in NEUTRAL_NODES:
        values = [by_name[name].interpretations[node].interpretation for name in names]
        agree = sum(value == result.interpretations[node].interpretation for value in values)
        lines.append(f"{node_text(node)} | " + " | ".join(values) + f" | {agree}")
    return lines


def rankings(result: Result) -> list[str]:
    lines = ["", "AGGREGATE RANKINGS"]
    lines += ["", "1. Highest equilibrium nodes"] + [f"{node_text(row.node)} | {fmt(row.score)}" for row in sorted(result.equilibrium.values(), key=lambda row: (-row.score, node_text(row.node)))]
    lines += ["", "2. Highest staleness nodes"] + [f"{node_text(row.node)} | {fmt(row.score)}" for row in sorted(result.staleness.values(), key=lambda row: (-row.score, node_text(row.node)))]
    lines += ["", "3. Highest gateway nodes"] + [f"{node_text(row.node)} | {fmt(row.score)}" for row in sorted(result.gateway.values(), key=lambda row: (-row.score, node_text(row.node)))]
    lines += ["", "4. Highest real range nodes"] + [f"{node_text(row.node)} | {fmt(row.mean_range)}" for row in sorted(result.price.values(), key=lambda row: (-row.mean_range, node_text(row.node)))]
    lines += ["", "5. Highest real volume nodes"] + [f"{node_text(row.node)} | {fmt(row.mean_volume)}" for row in sorted(result.volume.values(), key=lambda row: (-row.mean_volume, node_text(row.node)))]
    lines += ["", "6. Highest efficiency nodes"] + [f"{node_text(row.node)} | {fmt(row.mean_efficiency)}" for row in sorted(result.efficiency.values(), key=lambda row: (-row.mean_efficiency, node_text(row.node)))]
    lines += ["", "7. Highest excursion-risk nodes"] + [f"{node_text(row.node)} | {pct(row.exit_5)}" for row in sorted(result.excursion.values(), key=lambda row: (-row.exit_5, node_text(row.node)))]
    lines += ["", "8. Study72 comparison"] + [f"{node_text(row.node)} | {row.result}" for row in result.comparison.values()]
    lines += ["", "9. Strongest outcome diagnostics"] + [f"{node_text(row.node)} | DRFwd5={fmt(row.mean_by_horizon[5])} | Continuation5={pct(row.continuation_by_horizon[5])}" for row in sorted(result.outcomes.values(), key=lambda row: (-row.mean_by_horizon[5], node_text(row.node)))]
    lines += ["", "10. Recommended Neutral lifecycle model"] + recommendation(result)
    return lines


def write_per_instrument(result: Result, out_root: Path) -> None:
    path = out_root / result.instrument / f"NeutralMarketInterpretation74_{result.instrument}.txt"
    ensure_dir(path.parent)
    neutral_rows = sum(row.count for row in result.lifecycle.values())
    has_ohlc, has_volume = real_ohlcv_available(result.bars)
    lines = [
        "APVA Neutral Market Interpretation Study v2.0",
        "=" * 108,
        f"Instrument: {result.instrument}",
        "Input path(s): " + ", ".join(str(path) for path in result.source_paths),
        f"Total rows: {len(result.bars)}",
        f"Neutral rows: {neutral_rows}",
        f"Real OHLC available: {has_ohlc}",
        f"Real Volume available: {has_volume}",
    ]
    append_common(lines, result)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aggregate(result: Result, instrument_results: list[Result], out_root: Path) -> None:
    attach_replication(result, instrument_results)
    path = out_root / "NeutralMarketInterpretation74" / "NeutralMarketInterpretation74_All.txt"
    ensure_dir(path.parent)
    has_ohlc, has_volume = real_ohlcv_available(result.bars)
    lines = [
        "APVA Neutral Market Interpretation Study v2.0 - Aggregate",
        "=" * 108,
        "Instruments: " + ", ".join(instrument_columns(instrument_results)),
        f"Real OHLC available: {has_ohlc}",
        f"Real Volume available: {has_volume}",
    ]
    append_common(lines, result)
    lines += replication_lines(result, instrument_results)
    lines += rankings(result)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(result: Result) -> None:
    if not result.rows:
        raise RuntimeError(f"{result.instrument}: no stream rows.")
    has_ohlc, has_volume = real_ohlcv_available(result.bars)
    if not has_ohlc or not has_volume:
        raise RuntimeError(f"{result.instrument}: Study74 requires real OHLCV fields from the upgraded loader.")
    for node in NEUTRAL_NODES:
        if node not in result.lifecycle:
            raise RuntimeError(f"{result.instrument}: missing neutral node {node_text(node)}.")
    if any(not 0 <= row.exit_5 <= 1 for row in result.excursion.values()):
        raise RuntimeError(f"{result.instrument}: invalid excursion risk.")
    if any(math.isnan(row.mean_range) for row in result.price.values()):
        raise RuntimeError(f"{result.instrument}: invalid price profile.")


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
    aggregate_bars, aggregate_paths, aggregate_nodes, aggregate_stream = [], [], [], []
    for loaded_row, decay_row in zip(loaded, decay):
        local_node_rows = local_rows(decay_row)
        score_rows(local_node_rows)
        stream = build_stream(loaded_row, local_node_rows, memory_thresholds, (0.0, 0.0), (0.0, 0.0))
        nodes = [node_for(bar) for bar in loaded_row.bars]
        result = build_result(loaded_row.instrument, loaded_row.source_paths, loaded_row.bars, stream, nodes, local_node_rows)
        validate(result)
        instrument_results.append(result)

        offset = len(aggregate_bars)
        aggregate_stream.extend(build_stream(loaded_row, aggregate_node_rows, memory_thresholds, (0.0, 0.0), (0.0, 0.0), offset))
        aggregate_nodes.extend(nodes)
        aggregate_bars.extend(loaded_row.bars)
        aggregate_paths.extend(loaded_row.source_paths)

    aggregate = build_result("Aggregate", aggregate_paths, aggregate_bars, aggregate_stream, aggregate_nodes, aggregate_node_rows)
    validate(aggregate)
    out_root = Path(args.out_root)
    for result in instrument_results:
        write_per_instrument(result, out_root)
    write_aggregate(aggregate, instrument_results, out_root)
    print(f"Wrote {len(instrument_results)} per-instrument Study74 report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
