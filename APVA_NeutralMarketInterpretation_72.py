#!/usr/bin/env python3
"""APVA Neutral Market Interpretation Study v0.1.

Map the fixed NeutralProcessing lifecycle to observable price-volume behavior.
Forward price outcomes are diagnostics only.
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
from APVA_NeutralBackbone_71 import node_from_text
from APVA_NodeImportance_58 import aggregate_rows, local_rows, score_rows
from APVA_NodeNecessity_70 import build_result as necessity_result
from APVA_NodeNecessity_70 import validate as validate_necessity
from APVA_ProcessGraph_54 import Node, node_for, node_text
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

EPSILON = 1e-9
HORIZONS = (1, 3, 5, 10)


@dataclass
class LifecycleRow:
    node: Node
    count: int
    occupancy: float
    replication_count: int
    memory_strength: float
    branch_entropy: float
    necessity: float


@dataclass
class PriceProfile:
    node: Node
    mean_range: float
    mean_body: float
    mean_close_location: float
    mean_direction: float
    mean_true_range: float


@dataclass
class VolumeProfile:
    node: Node
    mean_volume: float
    median_volume: float
    volume_percentile: float
    relative_previous: float
    relative_session_mean: float
    black_rate: float
    red_rate: float
    neutral_rate: float


@dataclass
class CouplingProfile:
    node: Node
    range_volume_ratio: float
    volume_range_ratio: float
    correlation: float


@dataclass
class CompressionProfile:
    node: Node
    compression_rate: float
    expansion_rate: float
    volume_expansion_rate: float
    volume_contraction_rate: float


@dataclass
class PolarityProfile:
    node: Node
    same_rate: float
    flip_rate: float
    black_continuation: float
    red_continuation: float


@dataclass
class ExcursionRisk:
    node: Node
    exit_1: float
    exit_2: float
    exit_3: float
    exit_5: float
    expected_bars: float


@dataclass
class DestinationRow:
    node: Node
    family: str
    count: int
    probability: float
    replication_count: int = 1


@dataclass
class ReturnRow:
    node: Node
    within_1: float
    within_2: float
    within_3: float
    within_5: float
    mean_length: float
    median_length: float
    max_length: int


@dataclass
class TransitionMeaning:
    transition: str
    delta_range: float
    delta_body: float
    delta_volume: float
    delta_memory: float
    delta_entropy: float
    delta_excursion: float


@dataclass
class ScoreRow:
    node: Node
    score: float
    components: tuple[float, float, float]


@dataclass
class InterpretationRow:
    node: Node
    interpretation: str
    evidence: str


@dataclass
class OutcomeRow:
    node: Node
    mean_by_horizon: dict[int, float]
    median_by_horizon: dict[int, float]
    continuation_by_horizon: dict[int, float]
    failure_by_horizon: dict[int, float]
    flat_by_horizon: dict[int, float]


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
    coupling: dict[Node, CouplingProfile]
    compression: dict[Node, CompressionProfile]
    polarity: dict[Node, PolarityProfile]
    excursion: dict[Node, ExcursionRisk]
    destinations: dict[tuple[Node, str], DestinationRow]
    returns: dict[Node, ReturnRow]
    transitions: dict[str, TransitionMeaning]
    equilibrium: dict[Node, ScoreRow]
    staleness: dict[Node, ScoreRow]
    gateway: dict[Node, ScoreRow]
    interpretations: dict[Node, InterpretationRow]
    outcomes: dict[Node, OutcomeRow]


def mean(values: Iterable[float]) -> float:
    return safe_mean(list(values))


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def correlation(left: Iterable[float], right: Iterable[float]) -> float:
    pairs = [(a, b) for a, b in zip(left, right)]
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
        base = {node: 0.0 for node in values}
    else:
        base = {node: (value - low) / (high - low) for node, value in values.items()}
    if inverse:
        return {node: 1.0 - value for node, value in base.items()}
    return base


def value_from_bar(bar, *names: str) -> float | None:
    for name in names:
        if hasattr(bar, name):
            value = getattr(bar, name)
            if value in (None, ""):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def bar_open(bar) -> float:
    close = bar_close(bar)
    return value_from_bar(bar, "open", "Open") if value_from_bar(bar, "open", "Open") is not None else close


def bar_high(bar) -> float:
    close = bar_close(bar)
    return value_from_bar(bar, "high", "High") if value_from_bar(bar, "high", "High") is not None else close


def bar_low(bar) -> float:
    close = bar_close(bar)
    return value_from_bar(bar, "low", "Low") if value_from_bar(bar, "low", "Low") is not None else close


def bar_close(bar) -> float:
    value = value_from_bar(bar, "close", "Close")
    return value if value is not None else 0.0


def bar_volume(bar) -> float | None:
    return value_from_bar(bar, "volume", "Volume")


def bar_polarity(bar) -> str:
    for name in ("polarity", "volume_polarity", "VolumePolarity"):
        if hasattr(bar, name):
            value = getattr(bar, name)
            return str(value) if value else "Neutral"
    return "Neutral"


def has_explicit_ohlc(bars: list) -> bool:
    return bool(bars) and all(any(hasattr(bar, name) for name in ("open", "Open", "high", "High", "low", "Low")) for bar in bars[:1])


def has_explicit_volume(bars: list) -> bool:
    return bool(bars) and any(hasattr(bars[0], name) for name in ("volume", "Volume"))


def bar_range(bar) -> float:
    return bar_high(bar) - bar_low(bar)


def bar_body(bar) -> float:
    return abs(bar_close(bar) - bar_open(bar))


def close_location(bar) -> float:
    return (bar_close(bar) - bar_low(bar)) / max(bar_high(bar) - bar_low(bar), EPSILON)


def true_range(bars: list, index: int) -> float:
    current = bars[index]
    if index == 0:
        return bar_range(current)
    previous_close = bar_close(bars[index - 1])
    return max(bar_high(current) - bar_low(current), abs(bar_high(current) - previous_close), abs(bar_low(current) - previous_close))


def neutral_indices(nodes: list[Node]) -> dict[Node, list[int]]:
    grouped = {node: [] for node in NEUTRAL_NODES}
    for index, node in enumerate(nodes):
        if is_neutral_node(node):
            grouped.setdefault(node, []).append(index)
    return grouped


def branch_entropy(rows: list) -> dict[Node, float]:
    grouped: dict[Node, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if is_neutral_node(row.node):
            grouped[row.node][row.next_node] += 1
    output = {}
    for node, counts in grouped.items():
        total = sum(counts.values())
        probs = [count / total for count in counts.values()] if total else []
        output[node] = -sum(value * math.log(value) for value in probs if value > 0)
    return output


def lifecycle_rows(rows: list, indices: dict[Node, list[int]], necessity, node_rows: dict[Node, object],
                   instrument_count: int = 1) -> dict[Node, LifecycleRow]:
    entropies = branch_entropy(rows)
    total = len(rows)
    output = {}
    for node in NEUTRAL_NODES:
        impact = necessity.impacts.get(node)
        node_row = node_rows.get(node)
        count = len(indices.get(node, []))
        output[node] = LifecycleRow(
            node, count, count / total if total else 0.0,
            instrument_count if count else 0,
            getattr(node_row, "memory_strength", 0.0) if node_row else 0.0,
            entropies.get(node, 0.0),
            impact.necessity_score if impact else 0.0,
        )
    return output


def price_profiles(bars: list, indices: dict[Node, list[int]]) -> dict[Node, PriceProfile]:
    output = {}
    for node, indexes in indices.items():
        output[node] = PriceProfile(
            node,
            mean(bar_range(bars[index]) for index in indexes),
            mean(bar_body(bars[index]) for index in indexes),
            mean(close_location(bars[index]) for index in indexes),
            mean(bar_close(bars[index]) - bar_open(bars[index]) for index in indexes),
            mean(true_range(bars, index) for index in indexes),
        )
    return output


def percentile_values(values: list[float]) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    denom = max(len(ordered) - 1, 1)
    ranks = {}
    for index, value in enumerate(ordered):
        ranks.setdefault(value, index / denom)
    return [ranks[value] for value in values]


def volume_profiles(bars: list, indices: dict[Node, list[int]]) -> dict[Node, VolumeProfile]:
    raw_volumes = [bar_volume(bar) for bar in bars]
    volumes = [value if value is not None else 0.0 for value in raw_volumes]
    percentiles = percentile_values(volumes)
    session_mean = mean(volumes)
    output = {}
    for node, indexes in indices.items():
        node_volumes = [volumes[index] for index in indexes]
        polarities = Counter(bar_polarity(bars[index]) for index in indexes)
        total = sum(polarities.values())
        output[node] = VolumeProfile(
            node,
            mean(node_volumes),
            median(node_volumes),
            mean(percentiles[index] for index in indexes),
            mean(volumes[index] / max(volumes[index - 1], EPSILON) for index in indexes if index > 0 and raw_volumes[index] is not None and raw_volumes[index - 1] is not None),
            mean(volumes[index] / max(session_mean, EPSILON) for index in indexes if raw_volumes[index] is not None),
            polarities["Black"] / total if total else 0.0,
            polarities["Red"] / total if total else 0.0,
            (total - polarities["Black"] - polarities["Red"]) / total if total else 0.0,
        )
    return output


def coupling_profiles(bars: list, indices: dict[Node, list[int]]) -> dict[Node, CouplingProfile]:
    raw_volumes = [bar_volume(bar) for bar in bars]
    volumes_by_index = [value if value is not None else 0.0 for value in raw_volumes]
    output = {}
    for node, indexes in indices.items():
        ranges = [bar_range(bars[index]) for index in indexes]
        volumes = [volumes_by_index[index] for index in indexes]
        output[node] = CouplingProfile(
            node,
            mean(ranges[index] / max(volumes[index], EPSILON) for index in range(len(ranges)) if raw_volumes[indexes[index]] is not None),
            mean(volumes[index] / max(ranges[index], EPSILON) for index in range(len(ranges)) if raw_volumes[indexes[index]] is not None),
            correlation([ranges[index] for index in range(len(ranges)) if raw_volumes[indexes[index]] is not None],
                        [volumes[index] for index in range(len(ranges)) if raw_volumes[indexes[index]] is not None]),
        )
    return output


def compression_profiles(bars: list, indices: dict[Node, list[int]]) -> dict[Node, CompressionProfile]:
    ohlc_available = has_explicit_ohlc(bars)
    raw_volumes = [bar_volume(bar) for bar in bars]
    volumes = [value if value is not None else 0.0 for value in raw_volumes]
    output = {}
    for node, indexes in indices.items():
        range_ratios, body_ratios, volume_ratios = [], [], []
        for index in indexes:
            if index == 0:
                continue
            if ohlc_available:
                range_ratios.append(bar_range(bars[index]) / max(bar_range(bars[index - 1]), EPSILON))
                body_ratios.append(bar_body(bars[index]) / max(bar_body(bars[index - 1]), EPSILON))
            if raw_volumes[index] is not None and raw_volumes[index - 1] is not None:
                volume_ratios.append(volumes[index] / max(volumes[index - 1], EPSILON))
        output[node] = CompressionProfile(
            node,
            mean(value < 1.0 for value in range_ratios),
            mean(value > 1.0 for value in range_ratios),
            mean(value > 1.0 for value in volume_ratios),
            mean(value < 1.0 for value in volume_ratios),
        )
    return output


def polarity_profiles(bars: list, indices: dict[Node, list[int]]) -> dict[Node, PolarityProfile]:
    output = {}
    for node, indexes in indices.items():
        same, flip, black_continue, red_continue = [], [], [], []
        for index in indexes:
            if index == 0:
                continue
            current = bar_polarity(bars[index])
            previous = bar_polarity(bars[index - 1])
            if current and previous:
                same.append(float(current == previous))
                flip.append(float(current != previous))
            if previous == "Black":
                black_continue.append(float(current == "Black"))
            if previous == "Red":
                red_continue.append(float(current == "Red"))
        output[node] = PolarityProfile(node, mean(same), mean(flip), mean(black_continue), mean(red_continue))
    return output


def excursion_risks(nodes: list[Node], indices: dict[Node, list[int]]) -> dict[Node, ExcursionRisk]:
    output = {}
    for node, indexes in indices.items():
        rates = {}
        bars_to_exit = []
        for horizon in (1, 2, 3, 5):
            hits = []
            for index in indexes:
                hit = False
                for step in range(1, horizon + 1):
                    if index + step < len(nodes) and not is_neutral_node(nodes[index + step]):
                        hit = True
                        break
                hits.append(float(hit))
            rates[horizon] = mean(hits)
        for index in indexes:
            distance = None
            for step in range(1, 21):
                if index + step >= len(nodes):
                    break
                if not is_neutral_node(nodes[index + step]):
                    distance = step
                    break
            if distance is not None:
                bars_to_exit.append(distance)
        output[node] = ExcursionRisk(node, rates[1], rates[2], rates[3], rates[5], mean(bars_to_exit))
    return output


def destination_rows(nodes: list[Node]) -> dict[tuple[Node, str], DestinationRow]:
    counts: Counter[tuple[Node, str]] = Counter()
    totals: Counter[Node] = Counter()
    for index in range(len(nodes) - 1):
        node = nodes[index]
        nxt = nodes[index + 1]
        if is_neutral_node(node) and not is_neutral_node(nxt):
            family = family_for_state(nxt[0])
            counts[(node, family)] += 1
            totals[node] += 1
    return {
        key: DestinationRow(key[0], key[1], count, count / totals[key[0]] if totals[key[0]] else 0.0)
        for key, count in counts.items()
    }


def return_rows(nodes: list[Node]) -> dict[Node, ReturnRow]:
    samples = {node: {1: [], 2: [], 3: [], 5: [], "length": []} for node in NEUTRAL_NODES}
    for index in range(len(nodes) - 1):
        node = nodes[index]
        if not is_neutral_node(node) or is_neutral_node(nodes[index + 1]):
            continue
        length = 0
        return_distance = None
        cursor = index + 1
        while cursor < len(nodes) and not is_neutral_node(nodes[cursor]):
            length += 1
            cursor += 1
        if cursor < len(nodes):
            return_distance = cursor - index
        samples[node]["length"].append(length)
        for horizon in (1, 2, 3, 5):
            samples[node][horizon].append(float(return_distance is not None and return_distance <= horizon))
    return {
        node: ReturnRow(node, mean(values[1]), mean(values[2]), mean(values[3]), mean(values[5]),
                        mean(values["length"]), median(values["length"]), max(values["length"]) if values["length"] else 0)
        for node, values in samples.items()
    }


def transition_meanings(price: dict[Node, PriceProfile], volume: dict[Node, VolumeProfile],
                        lifecycle: dict[Node, LifecycleRow], excursion: dict[Node, ExcursionRisk]) -> dict[str, TransitionMeaning]:
    output = {}
    for source, target_node in NEUTRAL_CHAIN:
        key = f"{node_text(source)} -> {node_text(target_node)}"
        output[key] = TransitionMeaning(
            key,
            price[target_node].mean_range - price[source].mean_range,
            price[target_node].mean_body - price[source].mean_body,
            volume[target_node].mean_volume - volume[source].mean_volume,
            lifecycle[target_node].memory_strength - lifecycle[source].memory_strength,
            lifecycle[target_node].branch_entropy - lifecycle[source].branch_entropy,
            excursion[target_node].exit_5 - excursion[source].exit_5,
        )
    return output


def neutral_score_rows(lifecycle: dict[Node, LifecycleRow], volume: dict[Node, VolumeProfile],
                       coupling: dict[Node, CouplingProfile], excursion: dict[Node, ExcursionRisk],
                       price: dict[Node, PriceProfile]) -> tuple[dict[Node, ScoreRow], dict[Node, ScoreRow], dict[Node, ScoreRow]]:
    inv_entropy = normalize_map({node: row.branch_entropy for node, row in lifecycle.items()}, inverse=True)
    memory = normalize_map({node: row.memory_strength for node, row in lifecycle.items()})
    inv_excursion = normalize_map({node: excursion[node].exit_5 for node in lifecycle}, inverse=True)
    equilibrium = {
        node: ScoreRow(node, mean((inv_entropy[node], memory[node], inv_excursion[node])), (inv_entropy[node], memory[node], inv_excursion[node]))
        for node in lifecycle
    }

    memory_change = {}
    volume_eff_change = {}
    return_cycle = {}
    previous = None
    for node in NEUTRAL_NODES:
        if previous is None:
            memory_change[node] = 0.0
            volume_eff_change[node] = 0.0
        else:
            memory_change[node] = lifecycle[previous].memory_strength - lifecycle[node].memory_strength
            volume_eff_change[node] = coupling[previous].volume_range_ratio - coupling[node].volume_range_ratio
        return_cycle[node] = 1.0 - inv_excursion[node]
        previous = node
    falling_memory = normalize_map(memory_change)
    efficiency_fall = normalize_map(volume_eff_change)
    rising_excursion = normalize_map({node: excursion[node].exit_5 for node in lifecycle})
    staleness = {
        node: ScoreRow(node, mean((falling_memory[node], rising_excursion[node], efficiency_fall[node])),
                       (falling_memory[node], rising_excursion[node], efficiency_fall[node]))
        for node in lifecycle
    }

    incoming = {node: 0 for node in lifecycle}
    outgoing = {node: 0 for node in lifecycle}
    # Gateway uses lifecycle-side branch entropy plus inverse necessity; incoming/outgoing are filled later by caller if needed.
    inverse_necessity = normalize_map({node: row.necessity for node, row in lifecycle.items()}, inverse=True)
    entropy = normalize_map({node: row.branch_entropy for node, row in lifecycle.items()})
    gateway = {
        node: ScoreRow(node, mean((inverse_necessity[node], entropy[node], 0.0)), (inverse_necessity[node], entropy[node], 0.0))
        for node in lifecycle
    }
    return equilibrium, staleness, gateway


def gateway_scores(lifecycle: dict[Node, LifecycleRow], rows: list, prior_gateway: dict[Node, ScoreRow]) -> dict[Node, ScoreRow]:
    incoming: dict[Node, set[str]] = defaultdict(set)
    outgoing: dict[Node, set[str]] = defaultdict(set)
    for row in rows:
        if is_neutral_node(row.node):
            incoming[row.node].add(row.previous)
            outgoing[row.node].add(row.next_node)
    branch_factor = {node: len(incoming[node]) * len(outgoing[node]) for node in lifecycle}
    factor_score = normalize_map(branch_factor)
    inverse_necessity = normalize_map({node: row.necessity for node, row in lifecycle.items()}, inverse=True)
    entropy = normalize_map({node: row.branch_entropy for node, row in lifecycle.items()})
    return {
        node: ScoreRow(node, mean((factor_score[node], inverse_necessity[node], entropy[node])),
                       (factor_score[node], inverse_necessity[node], entropy[node]))
        for node in lifecycle
    }


def directional_forward(bars: list, index: int, horizon: int) -> float | None:
    if index + horizon >= len(bars):
        return None
    polarity = bar_polarity(bars[index])
    if polarity == "Black":
        return bar_close(bars[index + horizon]) - bar_close(bars[index])
    if polarity == "Red":
        return bar_close(bars[index]) - bar_close(bars[index + horizon])
    return None


def outcome_rows(bars: list, indices: dict[Node, list[int]]) -> dict[Node, OutcomeRow]:
    output = {}
    for node, indexes in indices.items():
        by_horizon = {}
        for horizon in HORIZONS:
            by_horizon[horizon] = [value for index in indexes if (value := directional_forward(bars, index, horizon)) is not None]
        output[node] = OutcomeRow(
            node,
            {horizon: mean(values) for horizon, values in by_horizon.items()},
            {horizon: median(values) for horizon, values in by_horizon.items()},
            {horizon: mean(value > 0 for value in values) for horizon, values in by_horizon.items()},
            {horizon: mean(value < 0 for value in values) for horizon, values in by_horizon.items()},
            {horizon: mean(value == 0 for value in values) for horizon, values in by_horizon.items()},
        )
    return output


def interpretations(equilibrium: dict[Node, ScoreRow], staleness: dict[Node, ScoreRow],
                    gateway: dict[Node, ScoreRow], excursion: dict[Node, ExcursionRisk]) -> dict[Node, InterpretationRow]:
    top_equilibrium = max(equilibrium, key=lambda node: equilibrium[node].score)
    top_stale = max(staleness, key=lambda node: staleness[node].score)
    top_gateway = max(gateway, key=lambda node: gateway[node].score)
    top_excursion = max(excursion, key=lambda node: excursion[node].exit_5)
    output = {}
    for node in NEUTRAL_NODES:
        if node == top_gateway:
            label = "Gateway"
        elif node == top_equilibrium and node[1] in {"4", "5"}:
            label = "MatureEquilibrium"
        elif node == top_equilibrium:
            label = "Equilibrium"
        elif node == top_stale:
            label = "StaleEquilibrium"
        elif node == top_excursion:
            label = "PreExcursion"
        elif node[1] in {"1", "2"}:
            label = "Gateway"
        elif node[1] == "3":
            label = "Stabilizing"
        elif node[1] in {"4", "5"}:
            label = "Equilibrium"
        elif node[1] == "6-10":
            label = "StaleEquilibrium"
        elif node[1] in {"11-20", "21+"}:
            label = "LongNeutralDrift"
        else:
            label = "Unclear"
        evidence = (
            f"Equilibrium={fmt(equilibrium[node].score)}, "
            f"Staleness={fmt(staleness[node].score)}, "
            f"Gateway={fmt(gateway[node].score)}, "
            f"ExitWithin5={pct(excursion[node].exit_5)}"
        )
        output[node] = InterpretationRow(node, label, evidence)
    return output


def build_result(instrument: str, source_paths: list, bars: list, rows: list, nodes: list[Node],
                 node_rows: dict[Node, object]) -> Result:
    necessity = necessity_result(instrument, source_paths, bars, rows, nodes, node_rows)
    validate_necessity(necessity)
    indices = neutral_indices(nodes)
    lifecycle = lifecycle_rows(rows, indices, necessity, node_rows)
    price = price_profiles(bars, indices)
    volume = volume_profiles(bars, indices)
    coupling = coupling_profiles(bars, indices)
    compression = compression_profiles(bars, indices)
    polarity = polarity_profiles(bars, indices)
    excursion = excursion_risks(nodes, indices)
    destinations = destination_rows(nodes)
    returns = return_rows(nodes)
    transitions = transition_meanings(price, volume, lifecycle, excursion)
    equilibrium, staleness, gateway = neutral_score_rows(lifecycle, volume, coupling, excursion, price)
    gateway = gateway_scores(lifecycle, rows, gateway)
    interp = interpretations(equilibrium, staleness, gateway, excursion)
    return Result(
        instrument, source_paths, bars, rows, nodes, necessity, lifecycle, price, volume, coupling,
        compression, polarity, excursion, destinations, returns, transitions,
        equilibrium, staleness, gateway, interp, outcome_rows(bars, indices),
    )


def attach_replication(aggregate: Result, instruments: list[Result]) -> None:
    for node, row in aggregate.lifecycle.items():
        row.replication_count = sum(node in result.lifecycle and result.lifecycle[node].count > 0 for result in instruments)
    for key, row in aggregate.destinations.items():
        row.replication_count = sum(key in result.destinations for result in instruments)


def lifecycle_line(row: LifecycleRow) -> str:
    return f"{node_text(row.node)} | {row.count} | {pct(row.occupancy)} | {row.replication_count} | {fmt(row.memory_strength)} | {fmt(row.branch_entropy)} | {fmt(row.necessity)}"


def price_line(row: PriceProfile) -> str:
    return f"{node_text(row.node)} | {fmt(row.mean_range)} | {fmt(row.mean_body)} | {fmt(row.mean_close_location)} | {fmt(row.mean_direction)} | {fmt(row.mean_true_range)}"


def volume_line(row: VolumeProfile) -> str:
    return f"{node_text(row.node)} | {fmt(row.mean_volume)} | {fmt(row.median_volume)} | {pct(row.volume_percentile)} | {fmt(row.relative_previous)} | {fmt(row.relative_session_mean)} | {pct(row.black_rate)} | {pct(row.red_rate)} | {pct(row.neutral_rate)}"


def availability_lines(result: Result) -> list[str]:
    lines = [
        f"OHLC available in loaded evidence rows: {has_explicit_ohlc(result.bars)}",
        f"Raw Volume available in loaded evidence rows: {has_explicit_volume(result.bars)}",
    ]
    if not has_explicit_ohlc(result.bars):
        lines.append("OHLC note: Open/High/Low are not exposed by the current shared evidence loader; price range/body/true-range fields are close-only fallbacks and should not be interpreted as full bar ranges.")
    if not has_explicit_volume(result.bars):
        lines.append("Volume note: raw Volume is not exposed by the current shared evidence loader; volume magnitude, relative-volume, and range-volume coupling fields are reported as unavailable zeroes. VolumePolarity remains available.")
    return lines


def append_common(lines: list[str], result: Result) -> None:
    lines += ["", "1. Neutral Lifecycle Inventory", "Node | Count | Occupancy | ReplicationCount | MemoryStrength | BranchEntropy | NecessityScore"]
    lines += [lifecycle_line(row) for row in result.lifecycle.values()]
    lines += ["", "2. Price Behavior Profile", "Node | MeanRange | MeanBody | MeanCloseLocation | MeanDirection | MeanTrueRange"]
    lines += [price_line(row) for row in result.price.values()]
    lines += ["", "3. Volume Behavior Profile", "Node | MeanVolume | MedianVolume | VolumePercentile | VolumeRelativeToPrevious | VolumeRelativeToSessionMean | BlackRate | RedRate | NeutralRate"]
    lines += [volume_line(row) for row in result.volume.values()]
    lines += ["", "4. Range/Volume Coupling", "Node | RangeVolumeRatio | VolumeRangeRatio | RangeVolumeCorrelation"]
    lines += [f"{node_text(row.node)} | {fmt(row.range_volume_ratio)} | {fmt(row.volume_range_ratio)} | {fmt(row.correlation)}" for row in result.coupling.values()]
    lines += ["", "5. Compression/Expansion Profile", "Node | CompressionLikeRate | ExpansionLikeRate | VolumeExpansionRate | VolumeContractionRate"]
    lines += [f"{node_text(row.node)} | {pct(row.compression_rate)} | {pct(row.expansion_rate)} | {pct(row.volume_expansion_rate)} | {pct(row.volume_contraction_rate)}" for row in result.compression.values()]
    lines += ["", "6. Polarity Continuity", "Node | SamePolarityRate | PolarityFlipRate | BlackContinuationRate | RedContinuationRate"]
    lines += [f"{node_text(row.node)} | {pct(row.same_rate)} | {pct(row.flip_rate)} | {pct(row.black_continuation)} | {pct(row.red_continuation)}" for row in result.polarity.values()]
    lines += ["", "7. Excursion Risk", "Node | ExitWithin1 | ExitWithin2 | ExitWithin3 | ExitWithin5 | ExpectedBarsToExit"]
    lines += [f"{node_text(row.node)} | {pct(row.exit_1)} | {pct(row.exit_2)} | {pct(row.exit_3)} | {pct(row.exit_5)} | {fmt(row.expected_bars)}" for row in result.excursion.values()]
    lines += ["", "8. Excursion Destination", "NeutralNode | DestinationFamily | Count | Probability | ReplicationCount"]
    lines += [f"{node_text(row.node)} | {row.family} | {row.count} | {pct(row.probability)} | {row.replication_count}" for row in sorted(result.destinations.values(), key=lambda row: (-row.count, node_text(row.node)))]
    lines += ["", "9. Return Behavior", "NeutralNode | ReturnWithin1 | ReturnWithin2 | ReturnWithin3 | ReturnWithin5 | MeanExcursionLength | MedianExcursionLength | MaxExcursionLength"]
    lines += [f"{node_text(row.node)} | {pct(row.within_1)} | {pct(row.within_2)} | {pct(row.within_3)} | {pct(row.within_5)} | {fmt(row.mean_length)} | {fmt(row.median_length)} | {row.max_length}" for row in result.returns.values()]
    lines += ["", "10. Neutral Age Transition Meaning", "Transition | DeltaRange | DeltaBody | DeltaVolume | DeltaMemory | DeltaEntropy | DeltaExcursionRisk"]
    lines += [f"{row.transition} | {fmt(row.delta_range)} | {fmt(row.delta_body)} | {fmt(row.delta_volume)} | {fmt(row.delta_memory)} | {fmt(row.delta_entropy)} | {pct(row.delta_excursion)}" for row in result.transitions.values()]
    lines += ["", "11. Neutral Equilibrium Test", "Node | EquilibriumScore | MemoryStrength | InverseBranchEntropy | InverseExcursionRisk"]
    lines += [f"{node_text(row.node)} | {fmt(row.score)} | {fmt(row.components[1])} | {fmt(row.components[0])} | {fmt(row.components[2])}" for row in sorted(result.equilibrium.values(), key=lambda row: (-row.score, node_text(row.node)))]
    lines += ["", "12. Neutral Staleness Test", "Node | StalenessScore | MemoryChange | ExcursionRisk | VolumeEfficiencyChange"]
    lines += [f"{node_text(row.node)} | {fmt(row.score)} | {fmt(row.components[0])} | {fmt(row.components[1])} | {fmt(row.components[2])}" for row in sorted(result.staleness.values(), key=lambda row: (-row.score, node_text(row.node)))]
    lines += ["", "13. Neutral Gateway Test", "Node | GatewayScore | IncomingOutgoing | InverseNecessity | BranchEntropy"]
    lines += [f"{node_text(row.node)} | {fmt(row.score)} | {fmt(row.components[0])} | {fmt(row.components[1])} | {fmt(row.components[2])}" for row in sorted(result.gateway.values(), key=lambda row: (-row.score, node_text(row.node)))]
    lines += ["", "14. Price Outcome Diagnostics", "Node | DRFwd1 | DRFwd3 | DRFwd5 | DRFwd10 | ContinuationRate1 | ContinuationRate3 | ContinuationRate5 | ContinuationRate10"]
    for row in result.outcomes.values():
        lines.append(f"{node_text(row.node)} | {fmt(row.mean_by_horizon[1])} | {fmt(row.mean_by_horizon[3])} | {fmt(row.mean_by_horizon[5])} | {fmt(row.mean_by_horizon[10])} | {pct(row.continuation_by_horizon[1])} | {pct(row.continuation_by_horizon[3])} | {pct(row.continuation_by_horizon[5])} | {pct(row.continuation_by_horizon[10])}")
    lines += ["", "15. Neutral Lifecycle Interpretation", "Node | Interpretation | Evidence"]
    lines += [f"{node_text(row.node)} | {row.interpretation} | {row.evidence}" for row in result.interpretations.values()]
    lines += ["", "16. Cross-Instrument Replication", "Node | Interpretation | ReplicationCount"]
    lines += [f"{node_text(node)} | {result.interpretations[node].interpretation} | {result.lifecycle[node].replication_count}" for node in NEUTRAL_NODES]
    lines += ["", "17. Neutral Market Model Recommendation"] + recommendation(result)
    append_audit(lines)
    lines += ["", "19. Mechanical Research Notes", "This study maps fixed NeutralProcessing stages to OHLC, volume, polarity, and excursion diagnostics. It does not use forward returns in interpretation construction."]


def recommendation(result: Result) -> list[str]:
    interpretations_seen = Counter(row.interpretation for row in result.interpretations.values())
    equilibrium_node = max(result.equilibrium.values(), key=lambda row: row.score).node
    stale_node = max(result.staleness.values(), key=lambda row: row.score).node
    gateway_node = max(result.gateway.values(), key=lambda row: row.score).node
    pre_excursion_node = max(result.excursion.values(), key=lambda row: row.exit_5).node
    if interpretations_seen["Gateway"] >= 1 and interpretations_seen["Equilibrium"] + interpretations_seen["MatureEquilibrium"] >= 1 and interpretations_seen["StaleEquilibrium"] >= 1:
        classification = "NeutralAsLifecycle"
        description = "Age1-2 Gateway/Reset; Age3 Stabilization; Age4-5 Equilibrium/Mature balance; Age6-10 Stale equilibrium; Age11+ Long neutral drift."
    elif interpretations_seen["Equilibrium"] + interpretations_seen["MatureEquilibrium"] >= 2:
        classification = "NeutralAsEquilibriumProcess"
        description = "Neutral is primarily an equilibrium process with weaker lifecycle separation."
    elif interpretations_seen["Gateway"] >= 3:
        classification = "NeutralAsGatewayOnly"
        description = "Neutral acts mainly as an entry/exit gateway."
    else:
        classification = "NeutralInterpretationWeak"
        description = "Neutral behavior does not separate cleanly under fixed mechanical scores."
    return [
        f"Classification: {classification}",
        f"NeutralLifecycleDescription: {description}",
        f"EquilibriumNode: {node_text(equilibrium_node)}",
        f"StaleNode: {node_text(stale_node)}",
        f"GatewayNode: {node_text(gateway_node)}",
        f"PreExcursionNode: {node_text(pre_excursion_node)}",
        "Reason: Interpretation uses fixed rank-based equilibrium, staleness, gateway, and excursion diagnostics from observable market behavior.",
    ]


def append_audit(lines: list[str]) -> None:
    lines += [
        "",
        "18. Low-DoF Audit",
        "Variables used:",
        "StateAgeNode",
        "StructuralState",
        "AgeBucket",
        "OHLC",
        "Volume",
        "VolumePolarity",
        "PreviousNode",
        "CurrentNode",
        "NextNode",
        "MemoryStrength",
        "BranchProbability",
        "ExcursionFlag",
        "",
        "OHLC and raw Volume are used when present in loaded evidence rows; this script explicitly reports when the current loader does not expose them.",
        "",
        "No Context",
        "No Arbitration",
        "No Persistence",
        "No Phase",
        "No Optimization",
        "No Fitting",
        "No Machine Learning",
        "No Forward Returns used in market interpretation construction",
    ]


def write_per_instrument(result: Result, out_root: Path) -> None:
    path = out_root / result.instrument / f"NeutralMarketInterpretation_{result.instrument}.txt"
    ensure_dir(path.parent)
    neutral_rows = sum(row.count for row in result.lifecycle.values())
    lines = [
        "APVA Neutral Market Interpretation Study v0.1",
        "=" * 108,
        f"Instrument: {result.instrument}",
        "Input path(s): " + ", ".join(str(path) for path in result.source_paths),
        f"Total rows: {len(result.bars)}",
        f"Neutral rows: {neutral_rows}",
    ]
    lines += availability_lines(result)
    append_common(lines, result)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_replication_lines(result: Result, instruments: list[Result]) -> list[str]:
    by_name = {row.instrument: row for row in instruments}
    names = instrument_columns(instruments)
    lines = ["", "Aggregate Replication Table", "Node | " + " | ".join(f"Interpretation_{name}" for name in names) + " | ReplicationCount"]
    for node in NEUTRAL_NODES:
        values = []
        for name in names:
            values.append(by_name[name].interpretations[node].interpretation)
        replication = len(set(values))
        # ReplicationCount here means instruments agreeing with aggregate interpretation.
        agree = sum(value == result.interpretations[node].interpretation for value in values)
        lines.append(f"{node_text(node)} | " + " | ".join(values) + f" | {agree}")
    return lines


def write_aggregate(result: Result, instrument_results: list[Result], out_root: Path) -> None:
    attach_replication(result, instrument_results)
    path = out_root / "NeutralMarketInterpretation" / "NeutralMarketInterpretation_All.txt"
    ensure_dir(path.parent)
    lines = ["APVA Neutral Market Interpretation Study v0.1 - Aggregate", "=" * 108, "Instruments: " + ", ".join(instrument_columns(instrument_results))]
    lines += availability_lines(result)
    lines += ["", "Aggregate Neutral Lifecycle Table", "Node | Count | Occupancy | MemoryStrength | BranchEntropy | NecessityScore"]
    lines += [f"{node_text(row.node)} | {row.count} | {pct(row.occupancy)} | {fmt(row.memory_strength)} | {fmt(row.branch_entropy)} | {fmt(row.necessity)}" for row in result.lifecycle.values()]
    lines += ["", "Aggregate Price Profile Table", "Node | MeanRange | MeanBody | MeanCloseLocation | MeanDirection | MeanTrueRange"]
    lines += [price_line(row) for row in result.price.values()]
    lines += ["", "Aggregate Volume Profile Table", "Node | MeanVolume | MedianVolume | VolumePercentile | VolumeRelativeToPrevious | VolumeRelativeToSessionMean | BlackRate | RedRate | NeutralRate"]
    lines += [volume_line(row) for row in result.volume.values()]
    lines += ["", "Aggregate Range Volume Table", "Node | RangeVolumeRatio | VolumeRangeRatio | RangeVolumeCorrelation"]
    lines += [f"{node_text(row.node)} | {fmt(row.range_volume_ratio)} | {fmt(row.volume_range_ratio)} | {fmt(row.correlation)}" for row in result.coupling.values()]
    lines += ["", "Aggregate Compression Expansion Table", "Node | CompressionLikeRate | ExpansionLikeRate | VolumeExpansionRate | VolumeContractionRate"]
    lines += [f"{node_text(row.node)} | {pct(row.compression_rate)} | {pct(row.expansion_rate)} | {pct(row.volume_expansion_rate)} | {pct(row.volume_contraction_rate)}" for row in result.compression.values()]
    lines += ["", "Aggregate Polarity Table", "Node | SamePolarityRate | PolarityFlipRate | BlackContinuationRate | RedContinuationRate"]
    lines += [f"{node_text(row.node)} | {pct(row.same_rate)} | {pct(row.flip_rate)} | {pct(row.black_continuation)} | {pct(row.red_continuation)}" for row in result.polarity.values()]
    lines += ["", "Aggregate Excursion Risk Table", "Node | ExitWithin1 | ExitWithin2 | ExitWithin3 | ExitWithin5 | ExpectedBarsToExit"]
    lines += [f"{node_text(row.node)} | {pct(row.exit_1)} | {pct(row.exit_2)} | {pct(row.exit_3)} | {pct(row.exit_5)} | {fmt(row.expected_bars)}" for row in result.excursion.values()]
    lines += ["", "Aggregate Excursion Destination Table", "NeutralNode | DestinationFamily | Count | Probability | ReplicationCount"]
    lines += [f"{node_text(row.node)} | {row.family} | {row.count} | {pct(row.probability)} | {row.replication_count}" for row in sorted(result.destinations.values(), key=lambda row: (-row.count, node_text(row.node)))]
    lines += ["", "Aggregate Return Table", "NeutralNode | ReturnWithin1 | ReturnWithin2 | ReturnWithin3 | ReturnWithin5 | MeanExcursionLength"]
    lines += [f"{node_text(row.node)} | {pct(row.within_1)} | {pct(row.within_2)} | {pct(row.within_3)} | {pct(row.within_5)} | {fmt(row.mean_length)}" for row in result.returns.values()]
    lines += ["", "Aggregate Transition Meaning Table", "Transition | DeltaRange | DeltaBody | DeltaVolume | DeltaMemory | DeltaEntropy | DeltaExcursionRisk"]
    lines += [f"{row.transition} | {fmt(row.delta_range)} | {fmt(row.delta_body)} | {fmt(row.delta_volume)} | {fmt(row.delta_memory)} | {fmt(row.delta_entropy)} | {pct(row.delta_excursion)}" for row in result.transitions.values()]
    lines += ["", "Aggregate Equilibrium Table", "Node | EquilibriumScore | MemoryStrength | InverseBranchEntropy | InverseExcursionRisk"]
    lines += [f"{node_text(row.node)} | {fmt(row.score)} | {fmt(row.components[1])} | {fmt(row.components[0])} | {fmt(row.components[2])}" for row in sorted(result.equilibrium.values(), key=lambda row: (-row.score, node_text(row.node)))]
    lines += ["", "Aggregate Staleness Table", "Node | StalenessScore | MemoryChange | ExcursionRisk | VolumeEfficiencyChange"]
    lines += [f"{node_text(row.node)} | {fmt(row.score)} | {fmt(row.components[0])} | {fmt(row.components[1])} | {fmt(row.components[2])}" for row in sorted(result.staleness.values(), key=lambda row: (-row.score, node_text(row.node)))]
    lines += ["", "Aggregate Gateway Table", "Node | GatewayScore | IncomingCount | OutgoingCount | NecessityScore | BranchEntropy"]
    lines += [f"{node_text(row.node)} | {fmt(row.score)} | {fmt(row.components[0])} | {fmt(row.components[0])} | {fmt(row.components[1])} | {fmt(row.components[2])}" for row in sorted(result.gateway.values(), key=lambda row: (-row.score, node_text(row.node)))]
    lines += ["", "Aggregate Outcome Table", "Node | DRFwd1 | DRFwd3 | DRFwd5 | DRFwd10 | ContinuationRate1 | ContinuationRate3 | ContinuationRate5 | ContinuationRate10"]
    for row in result.outcomes.values():
        lines.append(f"{node_text(row.node)} | {fmt(row.mean_by_horizon[1])} | {fmt(row.mean_by_horizon[3])} | {fmt(row.mean_by_horizon[5])} | {fmt(row.mean_by_horizon[10])} | {pct(row.continuation_by_horizon[1])} | {pct(row.continuation_by_horizon[3])} | {pct(row.continuation_by_horizon[5])} | {pct(row.continuation_by_horizon[10])}")
    lines += ["", "Aggregate Interpretation Table", "Node | Interpretation | Evidence"]
    lines += [f"{node_text(row.node)} | {row.interpretation} | {row.evidence}" for row in result.interpretations.values()]
    lines += aggregate_replication_lines(result, instrument_results)
    lines += ["", "Aggregate Recommendation"] + recommendation(result)
    lines += rankings(result)
    lines += ["", "RESEARCH NOTES", "Mechanical only.", "What does Neutral_Age4 represent?", "What does Neutral_Age5 represent?", "Why is Neutral_Age1 high occupancy but low necessity?", "Does Neutral_Age6-10 represent staleness?", "Can APVA Neutral structure be mapped to observable price-volume behavior?"]
    append_audit(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rankings(result: Result) -> list[str]:
    lines = ["", "AGGREGATE RANKINGS"]
    lines += ["", "1. Highest equilibrium nodes"] + [f"{node_text(row.node)} | {fmt(row.score)}" for row in sorted(result.equilibrium.values(), key=lambda row: (-row.score, node_text(row.node)))]
    lines += ["", "2. Highest staleness nodes"] + [f"{node_text(row.node)} | {fmt(row.score)}" for row in sorted(result.staleness.values(), key=lambda row: (-row.score, node_text(row.node)))]
    lines += ["", "3. Highest gateway nodes"] + [f"{node_text(row.node)} | {fmt(row.score)}" for row in sorted(result.gateway.values(), key=lambda row: (-row.score, node_text(row.node)))]
    lines += ["", "4. Highest excursion-risk Neutral nodes"] + [f"{node_text(row.node)} | {pct(row.exit_5)}" for row in sorted(result.excursion.values(), key=lambda row: (-row.exit_5, node_text(row.node)))]
    lines += ["", "5. Lowest excursion-risk Neutral nodes"] + [f"{node_text(row.node)} | {pct(row.exit_5)}" for row in sorted(result.excursion.values(), key=lambda row: (row.exit_5, node_text(row.node)))]
    lines += ["", "6. Highest volume-efficiency Neutral nodes"] + [f"{node_text(row.node)} | {fmt(row.volume_range_ratio)}" for row in sorted(result.coupling.values(), key=lambda row: (-row.volume_range_ratio, node_text(row.node)))]
    lines += ["", "7. Lowest volume-efficiency Neutral nodes"] + [f"{node_text(row.node)} | {fmt(row.volume_range_ratio)}" for row in sorted(result.coupling.values(), key=lambda row: (row.volume_range_ratio, node_text(row.node)))]
    lines += ["", "8. Strongest polarity-continuity nodes"] + [f"{node_text(row.node)} | {pct(row.same_rate)}" for row in sorted(result.polarity.values(), key=lambda row: (-row.same_rate, node_text(row.node)))]
    lines += ["", "9. Strongest outcome diagnostics"] + [f"{node_text(row.node)} | DRFwd5={fmt(row.mean_by_horizon[5])} | Continuation5={pct(row.continuation_by_horizon[5])}" for row in sorted(result.outcomes.values(), key=lambda row: (-row.mean_by_horizon[5], node_text(row.node)))]
    lines += ["", "10. Recommended Neutral market interpretation"] + recommendation(result)
    return lines


def validate(result: Result) -> None:
    if not result.rows:
        raise RuntimeError(f"{result.instrument}: no stream rows.")
    if not result.lifecycle:
        raise RuntimeError(f"{result.instrument}: missing lifecycle rows.")
    for node in NEUTRAL_NODES:
        if node not in result.lifecycle:
            raise RuntimeError(f"{result.instrument}: missing neutral node {node_text(node)}.")
    if any(not 0 <= row.exit_5 <= 1 for row in result.excursion.values()):
        raise RuntimeError(f"{result.instrument}: invalid excursion probability.")
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
    print(f"Wrote {len(instrument_results)} per-instrument report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
