#!/usr/bin/env python3
"""APVA Structural Forecast Study v0.1.

Forecast future APVA structural states from the observed StructuralState +
AgeBucket transition graph. Forward price outcomes are diagnostics only.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable

from APVA_MinimalEngine_52 import age_zone, directional_return, instrument_columns, load_results, safe_mean
from APVA_ProcessGraph_54 import node_for, node_text
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

HORIZONS = (1, 2, 3, 5)
CALIBRATION_BINS = tuple((index / 10, (index + 1) / 10) for index in range(10))
ENTROPY_BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, math.inf))
Node = tuple[str, str]


@dataclass
class Forecast:
    index: int
    horizon: int
    current: Hashable
    actual: Hashable
    distribution: dict[Hashable, float]
    ranked: list[tuple[Hashable, float]]
    probability: float
    confidence: float
    entropy: float
    brier: float
    top1: bool
    top2: bool
    top3: bool


@dataclass
class Metrics:
    count: int
    top1: float
    top2: float
    top3: float
    confidence: float
    entropy: float
    brier: float


@dataclass
class Calibration:
    label: str
    count: int
    probability: float
    observed: float
    error: float


@dataclass
class Outcome:
    count: int
    mean_dr: float
    median_dr: float
    continuation: float
    failure: float
    flat: float
    skew: float


@dataclass
class StudyResult:
    instrument: str
    source_paths: list
    bars: list
    nodes: list[Node]
    graph: dict[Node, dict[Node, float]]
    forecasts: list[Forecast]
    metrics: dict[int, Metrics]
    state_forecasts: list[Forecast]
    state_metrics: dict[int, Metrics]
    zone_forecasts: list[Forecast]
    zone_metrics: dict[int, Metrics]
    calibration: list[Calibration]
    node_metrics: dict[Node, Metrics]
    entropy_correlation: float
    outcomes: dict[str, Outcome]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def graph(stream: list[Hashable]) -> dict[Hashable, dict[Hashable, float]]:
    counts: dict[Hashable, Counter] = defaultdict(Counter)
    for source, destination in zip(stream, stream[1:]):
        counts[source][destination] += 1
    rows = {}
    for source, destinations in counts.items():
        total = sum(destinations.values())
        rows[source] = {destination: count / total for destination, count in destinations.items()}
    return rows


def propagate(start: Hashable, transitions: dict[Hashable, dict[Hashable, float]], horizon: int) -> dict[Hashable, float]:
    distribution = {start: 1.0}
    for _ in range(horizon):
        future: dict[Hashable, float] = defaultdict(float)
        for source, mass in distribution.items():
            destinations = transitions.get(source)
            if not destinations:
                future[source] += mass
                continue
            for destination, probability in destinations.items():
                future[destination] += mass * probability
        distribution = dict(future)
    return distribution


def ranked_distribution(distribution: dict[Hashable, float]) -> list[tuple[Hashable, float]]:
    return sorted(distribution.items(), key=lambda item: (-item[1], str(item[0])))


def distribution_entropy(distribution: dict[Hashable, float]) -> float:
    return -sum(probability * math.log(probability) for probability in distribution.values() if probability > 0)


def brier_score(distribution: dict[Hashable, float], actual: Hashable, universe: set[Hashable]) -> float:
    return sum((distribution.get(item, 0.0) - (1.0 if item == actual else 0.0)) ** 2 for item in universe)


def build_forecasts(stream: list[Hashable], transitions: dict[Hashable, dict[Hashable, float]]) -> list[Forecast]:
    universe = set(stream)
    rows = []
    for horizon in HORIZONS:
        for index in range(len(stream) - horizon):
            current = stream[index]
            actual = stream[index + horizon]
            distribution = propagate(current, transitions, horizon)
            ranked = ranked_distribution(distribution)
            first = ranked[0][1] if ranked else 0.0
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            choices = [item for item, _ in ranked]
            rows.append(Forecast(
                index, horizon, current, actual, distribution, ranked, first,
                first - second, distribution_entropy(distribution),
                brier_score(distribution, actual, universe),
                actual in choices[:1], actual in choices[:2], actual in choices[:3],
            ))
    return rows


def summarize(forecasts: Iterable[Forecast]) -> Metrics:
    rows = list(forecasts)
    if not rows:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return Metrics(
        len(rows), mean(row.top1 for row in rows), mean(row.top2 for row in rows),
        mean(row.top3 for row in rows), mean(row.confidence for row in rows),
        mean(row.entropy for row in rows), mean(row.brier for row in rows),
    )


def horizon_metrics(forecasts: list[Forecast]) -> dict[int, Metrics]:
    return {horizon: summarize(row for row in forecasts if row.horizon == horizon) for horizon in HORIZONS}


def bin_label(low: float, high: float) -> str:
    return f"{low:.1f}-{high:.1f}" if math.isfinite(high) else f"{low:.1f}+"


def in_bin(value: float, low: float, high: float) -> bool:
    return low <= value <= high if high == 1.0 else low <= value < high


def calibration(forecasts: Iterable[Forecast]) -> list[Calibration]:
    rows = list(forecasts)
    output = []
    for low, high in CALIBRATION_BINS:
        selected = [row for row in rows if in_bin(row.probability, low, high)]
        probability = mean(row.probability for row in selected)
        observed = mean(row.top1 for row in selected)
        output.append(Calibration(bin_label(low, high), len(selected), probability, observed, abs(probability - observed)))
    return output


def correlation(xs: Iterable[float], ys: Iterable[float]) -> float:
    xs, ys = list(xs), list(ys)
    if len(xs) < 2:
        return 0.0
    x_mean, y_mean = mean(xs), mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_sum = sum((x - x_mean) ** 2 for x in xs)
    y_sum = sum((y - y_mean) ** 2 for y in ys)
    return numerator / math.sqrt(x_sum * y_sum) if x_sum and y_sum else 0.0


def entropy_accuracy_correlation(forecasts: Iterable[Forecast]) -> float:
    rows = list(forecasts)
    return correlation((row.entropy for row in rows), (float(row.top1) for row in rows))


def outcome(values: Iterable[float]) -> Outcome:
    values = list(values)
    count = len(values)
    continuation = sum(value > 0 for value in values) / count if count else 0.0
    failure = sum(value < 0 for value in values) / count if count else 0.0
    flat = sum(value == 0 for value in values) / count if count else 0.0
    return Outcome(count, mean(values), median(values), continuation, failure, flat, continuation - failure)


def outcome_diagnostics(bars: list, forecasts: list[Forecast]) -> dict[str, Outcome]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in forecasts:
        value = directional_return(bars, row.index, 5)
        if value is not None:
            bucket = f"t+{row.horizon} {'Correct' if row.top1 else 'Incorrect'}"
            values[bucket].append(value)
    return {bucket: outcome(samples) for bucket, samples in values.items()}


def study(result) -> StudyResult:
    bars = result.bars
    nodes = [node_for(bar) for bar in bars]
    transitions = graph(nodes)
    forecasts = build_forecasts(nodes, transitions)
    states = [bar.state for bar in bars]
    state_forecasts = build_forecasts(states, graph(states))
    zones = [(bar.state, age_zone(bar.age_bucket)) for bar in bars]
    zone_forecasts = build_forecasts(zones, graph(zones))
    node_metrics = {node: summarize(row for row in forecasts if row.current == node) for node in sorted(set(nodes))}
    return StudyResult(
        result.instrument, result.source_paths, bars, nodes, transitions, forecasts,
        horizon_metrics(forecasts), state_forecasts, horizon_metrics(state_forecasts),
        zone_forecasts, horizon_metrics(zone_forecasts), calibration(forecasts), node_metrics,
        entropy_accuracy_correlation(forecasts), outcome_diagnostics(bars, forecasts),
    )


def recommendation(results: list[StudyResult]) -> tuple[str, str]:
    valid = len(results)
    top1 = safe_mean(result.metrics[horizon].top1 for result in results for horizon in HORIZONS)
    top3 = safe_mean(result.metrics[horizon].top3 for result in results for horizon in HORIZONS)
    if top1 >= 0.50 and top3 >= 0.75 and valid >= 2:
        label = "Strong Structural Forecast"
    elif top1 >= 0.35 and top3 >= 0.60 and valid >= 2:
        label = "Moderate Structural Forecast"
    else:
        label = "Weak Structural Forecast"
    reason = f"MeanTop1Accuracy={pct(top1)}; MeanTop3Accuracy={pct(top3)}; ValidInstrumentCount={valid}."
    return label, reason


def top_node(row: Forecast) -> str:
    return node_text(row.ranked[0][0]) if row.ranked else "N/A"


def item_text(item: Hashable) -> str:
    return node_text(item) if isinstance(item, tuple) else str(item)


def model_table(lines: list[str], title: str, rows: dict[int, Metrics]) -> None:
    lines += ["", title, "Horizon | ForecastCount | Top1Accuracy | Top2Accuracy | Top3Accuracy | AverageConfidence | ForecastEntropy | BrierScore"]
    for horizon in HORIZONS:
        row = rows[horizon]
        lines.append(f"t+{horizon} | {row.count} | {pct(row.top1)} | {pct(row.top2)} | {pct(row.top3)} | {pct(row.confidence)} | {fmt(row.entropy)} | {fmt(row.brier)}")


def write_per_instrument(result: StudyResult, out_root: Path) -> None:
    path = out_root / result.instrument / f"StructuralForecast_{result.instrument}.txt"
    ensure_dir(path.parent)
    lines = [
        "APVA Structural Forecast Study v0.1", "=" * 100, "Diagnostics",
        f"Instrument: {result.instrument}",
        "Input path(s): " + ", ".join(str(item) for item in result.source_paths),
        f"Total rows: {len(result.bars)}", f"Node count: {len(set(result.nodes))}",
        f"Forecast count: {len(result.forecasts)}",
        "", "1. One-Step Forecast",
        "CurrentNode | ForecastNode | ForecastProbability | ForecastConfidence | ActualNode | Correct",
    ]
    for row in (item for item in result.forecasts if item.horizon == 1):
        lines.append(f"{item_text(row.current)} | {top_node(row)} | {pct(row.probability)} | {pct(row.confidence)} | {item_text(row.actual)} | {row.top1}")
    lines += ["", "2. Multi-Step Forecast", "Horizon | ForecastCount | Method"]
    for horizon in HORIZONS[1:]:
        lines.append(f"t+{horizon} | {result.metrics[horizon].count} | Graph probability-mass propagation")
    model_table(lines, "3. Top-N Accuracy", result.metrics)
    lines += ["", "4. State-Level Forecast Accuracy", "StateAgeNode | ForecastCount | Top1Accuracy | Top2Accuracy | Top3Accuracy | AverageConfidence"]
    for node, row in sorted(result.node_metrics.items(), key=lambda item: (-item[1].top1, item[0])):
        lines.append(f"{node_text(node)} | {row.count} | {pct(row.top1)} | {pct(row.top2)} | {pct(row.top3)} | {pct(row.confidence)}")
    model_table(lines, "5. State-Only Forecast", result.state_metrics)
    lines += ["", "6. Calibration Test", "ProbabilityBin | ForecastCount | ForecastProbability | ObservedFrequency | CalibrationError"]
    for row in result.calibration:
        lines.append(f"{row.label} | {row.count} | {pct(row.probability)} | {pct(row.observed)} | {pct(row.error)}")
    lines += ["", "7. Brier Score", "Horizon | BrierScore"]
    for horizon in HORIZONS:
        lines.append(f"t+{horizon} | {fmt(result.metrics[horizon].brier)}")
    lines += ["", "8. Entropy vs Accuracy", f"CorrelationEntropyAccuracy: {fmt(result.entropy_correlation)}"]
    model_table(lines, "9. Young/Middle/Late Forecast", result.zone_metrics)
    lines += ["", "10. Outcome Diagnostics", "Diagnostic only. Forward outcomes are not used in forecast construction.", "ForecastBucket | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew"]
    for bucket, row in sorted(result.outcomes.items()):
        lines.append(f"{bucket} | {row.count} | {fmt(row.mean_dr)} | {fmt(row.median_dr)} | {pct(row.continuation)} | {pct(row.failure)} | {pct(row.flat)} | {pct(row.skew)}")
    label, reason = recommendation([result])
    lines += [
        "", "11. Forecast Recommendation", f"Instrument-only diagnostic: {label}", reason,
        "The aggregate report applies the required cross-instrument rule.",
        "", "12. Low-DoF Audit", "Variables used:", "StructuralState", "AgeBucket",
        "TransitionProbability", "", "No Context", "No Arbitration", "No Persistence",
        "No Phase", "No Optimization", "No Fitting", "No Machine Learning",
        "No Forward Returns used in forecast construction",
        "", "13. Mechanical Research Notes",
        "- Forecasts propagate observed historical transition probabilities through the State+Age graph.",
        "- State-only and Young/Middle/Late forecasts are fixed structural reductions.",
        "- Forward outcomes are anchored at the current bar and reported only after forecasts are built.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_calibration(results: list[StudyResult]) -> list[Calibration]:
    return calibration(row for result in results for row in result.forecasts)


def aggregate_node_metrics(results: list[StudyResult]) -> dict[Node, tuple[Metrics, int]]:
    rows = {}
    for node in sorted({node for result in results for node in result.node_metrics}):
        forecasts = [row for result in results for row in result.forecasts if row.current == node]
        rows[node] = summarize(forecasts), sum(node in result.node_metrics for result in results)
    return rows


def aggregate_zone_metrics(results: list[StudyResult]) -> dict[str, Metrics]:
    rows = {}
    for zone in ("Young", "Middle", "Late"):
        selected = [row for result in results for row in result.zone_forecasts if row.current[1] == zone]
        rows[zone] = summarize(selected)
    return rows


def aggregate_outcomes(results: list[StudyResult]) -> list[str]:
    instruments = instrument_columns(results)
    by_instrument = {result.instrument: result for result in results}
    buckets = sorted({bucket for result in results for bucket in result.outcomes})
    lines = ["", "Aggregate Outcome Table", "ForecastBucket | " + " | ".join(f"Count_{instrument} | Skew_{instrument} | MeanDR_{instrument}" for instrument in instruments) + " | ValidInstrumentCount | MeanSkew | MeanDR"]
    for bucket in buckets:
        values = [by_instrument[instrument].outcomes.get(bucket) for instrument in instruments]
        valid = [row for row in values if row and row.count]
        cells = [value for row in values for value in ((str(row.count), pct(row.skew), fmt(row.mean_dr)) if row else ("0", "N/A", "N/A"))]
        lines.append(f"{bucket} | " + " | ".join(cells) + f" | {len(valid)} | {pct(safe_mean(row.skew for row in valid))} | {fmt(safe_mean(row.mean_dr for row in valid))}")
    return lines


def write_aggregate(results: list[StudyResult], out_root: Path) -> None:
    path = out_root / "StructuralForecast" / "StructuralForecast_All.txt"
    ensure_dir(path.parent)
    instruments = instrument_columns(results)
    by_instrument = {result.instrument: result for result in results}
    lines = ["APVA Structural Forecast Study v0.1 - Aggregate", "=" * 108, "Instruments: " + ", ".join(instruments)]
    lines += ["", "Aggregate Accuracy Table", "Horizon | " + " | ".join(f"Top1_{i} | Top2_{i} | Top3_{i}" for i in instruments) + " | ValidInstrumentCount | MeanTop1 | MeanTop2 | MeanTop3"]
    for horizon in HORIZONS:
        values = [by_instrument[i].metrics[horizon] for i in instruments]
        cells = [value for row in values for value in (pct(row.top1), pct(row.top2), pct(row.top3))]
        lines.append(f"t+{horizon} | " + " | ".join(cells) + f" | {len(values)} | {pct(safe_mean(row.top1 for row in values))} | {pct(safe_mean(row.top2 for row in values))} | {pct(safe_mean(row.top3 for row in values))}")
    lines += ["", "Cross-Instrument Replication", "Horizon | " + " | ".join(f"Accuracy_{i} | Top2_{i} | Top3_{i} | Brier_{i}" for i in instruments) + " | ValidInstrumentCount | MeanAccuracy | MeanBrier"]
    for horizon in HORIZONS:
        values = [by_instrument[i].metrics[horizon] for i in instruments]
        cells = [value for row in values for value in (pct(row.top1), pct(row.top2), pct(row.top3), fmt(row.brier))]
        lines.append(f"t+{horizon} | " + " | ".join(cells) + f" | {len(values)} | {pct(safe_mean(row.top1 for row in values))} | {fmt(safe_mean(row.brier for row in values))}")
    lines += ["", "Aggregate Calibration Table", "ProbabilityBin | " + " | ".join(f"ForecastProbability_{i} | ObservedFrequency_{i} | CalibrationError_{i}" for i in instruments) + " | ForecastProbability_All | ObservedFrequency_All | CalibrationError_All"]
    all_calibration = {row.label: row for row in aggregate_calibration(results)}
    for label in (bin_label(low, high) for low, high in CALIBRATION_BINS):
        values = [{row.label: row for row in by_instrument[i].calibration}[label] for i in instruments]
        total = all_calibration[label]
        cells = [value for row in values for value in (pct(row.probability), pct(row.observed), pct(row.error))]
        lines.append(f"{label} | " + " | ".join(cells) + f" | {pct(total.probability)} | {pct(total.observed)} | {pct(total.error)}")
    lines += ["", "Aggregate Brier Table", "Horizon | " + " | ".join(f"Brier_{i}" for i in instruments) + " | MeanBrier"]
    for horizon in HORIZONS:
        values = [by_instrument[i].metrics[horizon].brier for i in instruments]
        lines.append(f"t+{horizon} | " + " | ".join(fmt(value) for value in values) + f" | {fmt(safe_mean(values))}")
    lines += ["", "Aggregate Entropy Table", "EntropyBin | " + " | ".join(f"Accuracy_{i}" for i in instruments) + " | MeanAccuracy | CorrelationEntropyAccuracy"]
    for low, high in ENTROPY_BINS:
        values = []
        for instrument in instruments:
            selected = [row for row in by_instrument[instrument].forecasts if in_bin(row.entropy, low, high)]
            values.append(mean(row.top1 for row in selected) if selected else 0.0)
        selected = [row for result in results for row in result.forecasts if in_bin(row.entropy, low, high)]
        lines.append(f"{bin_label(low, high)} | " + " | ".join(pct(value) for value in values) + f" | {pct(mean(row.top1 for row in selected))} | {fmt(entropy_accuracy_correlation(row for result in results for row in result.forecasts))}")
    node_rows = aggregate_node_metrics(results)
    lines += ["", "Aggregate State Forecast Table", "StateAgeNode | Count | Top1Accuracy | Top2Accuracy | Top3Accuracy | AverageConfidence | ReplicationCount"]
    for node, (row, replication) in sorted(node_rows.items(), key=lambda item: (-item[1][0].top1, item[0])):
        lines.append(f"{node_text(node)} | {row.count} | {pct(row.top1)} | {pct(row.top2)} | {pct(row.top3)} | {pct(row.confidence)} | {replication}")
    zone_rows = aggregate_zone_metrics(results)
    lines += ["", "Aggregate Young/Middle/Late Table", "Zone | Top1Accuracy | Top2Accuracy | Top3Accuracy | Entropy | Brier"]
    for zone in ("Young", "Middle", "Late"):
        row = zone_rows[zone]
        lines.append(f"{zone} | {pct(row.top1)} | {pct(row.top2)} | {pct(row.top3)} | {fmt(row.entropy)} | {fmt(row.brier)}")
    lines += aggregate_outcomes(results)
    label, reason = recommendation(results)
    mean_calibration = safe_mean(row.error for row in aggregate_calibration(results) if row.count)
    mean_brier = safe_mean(result.metrics[h].brier for result in results for h in HORIZONS)
    lines += [
        "", "Aggregate Structural Forecast Recommendation", f"Classification: {label}",
        f"Reason: {reason}", f"Calibration: MeanAbsoluteCalibrationError={pct(mean_calibration)}",
        f"Brier: MeanBrier={fmt(mean_brier)}", f"CrossInstrumentReplication: {len(results)} instruments",
        "", "Aggregate Rankings", "", "1. Most predictable nodes",
    ]
    for node, (row, replication) in sorted(node_rows.items(), key=lambda item: (-item[1][0].top1, -item[1][1], item[0]))[:20]:
        lines.append(f"{node_text(node)} | Top1Accuracy={pct(row.top1)} | ReplicationCount={replication} | Count={row.count}")
    lines += ["", "2. Least predictable nodes"]
    for node, (row, replication) in sorted(node_rows.items(), key=lambda item: (item[1][0].top1, -item[1][1], item[0]))[:20]:
        lines.append(f"{node_text(node)} | Top1Accuracy={pct(row.top1)} | ReplicationCount={replication} | Count={row.count}")
    lines += ["", "3. Highest confidence forecasts"]
    for node, (row, replication) in sorted(node_rows.items(), key=lambda item: (-item[1][0].confidence, item[0]))[:20]:
        lines.append(f"{node_text(node)} | AverageConfidence={pct(row.confidence)} | ReplicationCount={replication}")
    lines += ["", "4. Best calibrated forecasts"]
    for row in sorted((row for row in aggregate_calibration(results) if row.count), key=lambda item: item.error):
        lines.append(f"{row.label} | CalibrationError={pct(row.error)} | ForecastCount={row.count}")
    lines += ["", "5. Best horizons"]
    for horizon in sorted(HORIZONS, key=lambda h: -safe_mean(result.metrics[h].top1 for result in results)):
        lines.append(f"t+{horizon} | MeanTop1Accuracy={pct(safe_mean(result.metrics[horizon].top1 for result in results))}")
    lines += ["", "6. Worst horizons"]
    for horizon in sorted(HORIZONS, key=lambda h: safe_mean(result.metrics[h].top1 for result in results)):
        lines.append(f"t+{horizon} | MeanTop1Accuracy={pct(safe_mean(result.metrics[horizon].top1 for result in results))}")
    lines += ["", "7. Strongest age effects"]
    for horizon in HORIZONS:
        full = safe_mean(result.metrics[horizon].top1 for result in results)
        state = safe_mean(result.state_metrics[horizon].top1 for result in results)
        lines.append(f"t+{horizon} | StateAgeMinusStateOnlyTop1={pct(full - state)}")
    lines += ["", "8. Best Young/Middle/Late model"]
    for zone, row in sorted(zone_rows.items(), key=lambda item: -item[1].top1):
        lines.append(f"{zone} | Top1Accuracy={pct(row.top1)} | Brier={fmt(row.brier)}")
    lines += ["", "9. Best cross-instrument replication"]
    for horizon in sorted(HORIZONS, key=lambda h: -safe_mean(result.metrics[h].top1 for result in results)):
        lines.append(f"t+{horizon} | ValidInstrumentCount={len(results)} | MeanTop1Accuracy={pct(safe_mean(result.metrics[horizon].top1 for result in results))}")
    lines += [
        "", "10. Recommended APVA forecasting model", f"State+Age+TransitionForecast: {reason}",
        "", "Low-DoF Audit", "Variables used:", "StructuralState", "AgeBucket",
        "TransitionProbability", "", "No Context", "No Arbitration", "No Persistence",
        "No Phase", "No Optimization", "No Fitting", "No Machine Learning",
        "No Forward Returns used in forecast construction",
        "", "Research Notes",
        "- Can APVA forecast future structure? See the fixed aggregate classification above.",
        "- Which states are most predictable? See the replicated node rankings.",
        "- Does age improve forecasting? Compare State+Age to State-only accuracy by horizon.",
        "- Does lower entropy imply higher accuracy? See the entropy table and correlation.",
        "- Can Young/Middle/Late retain forecast power? Compare the fixed reduced model metrics.",
        "- Can APVA be reduced to State + Age + Transition Forecast? This report tests that representation mechanically.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_invariants(results: list[StudyResult]) -> None:
    for result in results:
        for destinations in result.graph.values():
            if abs(sum(destinations.values()) - 1.0) > 1e-12:
                raise RuntimeError(f"{result.instrument}: transition probabilities do not sum to one.")
        expected = sum(len(result.bars) - horizon for horizon in HORIZONS)
        if len(result.forecasts) != expected:
            raise RuntimeError(f"{result.instrument}: forecast count mismatch.")
        for row in result.forecasts:
            if abs(sum(row.distribution.values()) - 1.0) > 1e-12:
                raise RuntimeError(f"{result.instrument}: propagated mass does not sum to one.")
            if not (row.top1 <= row.top2 <= row.top3):
                raise RuntimeError(f"{result.instrument}: Top-N accuracy invariant failed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Evidence CSV files or directories")
    parser.add_argument("--out-root", default="Evidence/Output", help="Output root directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [study(result) for result in load_results(args.inputs)]
    if not results:
        raise SystemExit("No APVA evidence CSV files found.")
    validate_invariants(results)
    out_root = Path(args.out_root)
    for result in results:
        write_per_instrument(result, out_root)
    write_aggregate(results, out_root)
    print(f"Wrote {len(results)} per-instrument report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
