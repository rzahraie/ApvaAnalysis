#!/usr/bin/env python3
"""Study fixed gradient movement inside lateral frames before breakout.

This research-only script reuses lateral detection, anatomy features, fixed
score formulas, and DRFwd conventions. It does not optimize or fit parameters.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

from APVA_BreakoutContext_08 import (
    CompletedFrame,
    EvidenceBar,
    detect_completed_frames,
    lateral_start,
    load_rows,
)
from APVA_DissipationCompressionGradient_20 import pearson, rank_values
from APVA_DominanceGradient_21 import SCORES, SCORE_BINS, score_value
from APVA_LateralAnatomy_19 import Case, case_features, effect_size, on
from APVA_RegimeTransition_16 import CANONICAL_INSTRUMENTS, instrument_name, write_text


MIN_VALID_NON_FLAT = 30
TOP_LIMIT = 25
AGGREGATE_OUTPUT = Path(
    "Evidence/Output/LateralPreBreakoutGradient/LateralPreBreakoutGradient_All.txt"
)
METRICS = (
    "GradientDelta",
    "GradientVelocity",
    "GradientAcceleration",
    "Final3GradientMean",
    "Final5GradientMean",
)
FEATURES = (
    "GradientDelta",
    "GradientVelocity",
    "GradientAcceleration",
    "Final3GradientMean",
    "Final5GradientMean",
    "Final3CompressionMean",
    "Final3DissipationMean",
    "Final3AcceptedMean",
    "Final3ExpansionMean",
    "LateralDuration",
    "LateralRange",
    "BreakoutBarRange",
    "BreakoutBarVolume",
)


@dataclass(frozen=True)
class LateralCase:
    lateral_start_bar: int
    lateral_end_bar: int
    breakout_bar: int
    breakout_direction: str
    drfwd5: float
    features: dict[str, float | None]


@dataclass(frozen=True)
class MetricStats:
    score_name: str
    metric_name: str
    count: int
    rho: float
    delta: float


@dataclass(frozen=True)
class FeatureStats:
    feature: str
    success_count: int
    failure_count: int
    success_mean: float
    failure_mean: float
    delta: float
    effect_size: float


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    total_rows: int
    cases: list[LateralCase]
    metric_stats: dict[tuple[str, str], MetricStats]
    feature_stats: dict[str, FeatureStats]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study gradient velocity and acceleration inside laterals before breakout."
    )
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def breakout_drfwd5(rows: list[EvidenceBar], frame: CompletedFrame) -> float | None:
    future = frame.breakout_index + 5
    if future >= len(rows):
        return None
    if frame.direction == "Up":
        return rows[future].close - rows[frame.breakout_index].close
    return rows[frame.breakout_index].close - rows[future].close


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def score_series(rows: list[EvidenceBar], frame: CompletedFrame, score_name: str) -> list[float]:
    return [
        score_value(Case(index, 0.0, case_features(rows, index)), score_name)
        for index in range(frame.start_index, frame.end_index + 1)
    ]


def score_features(rows: list[EvidenceBar], frame: CompletedFrame, score_name: str) -> dict[str, float | None]:
    values = score_series(rows, frame, score_name)
    duration = len(values)
    delta = values[-1] - values[0]
    acceleration = None
    if duration >= 6:
        third = duration // 3
        acceleration = mean(values[-third:]) - mean(values[:third])
    final3_rows = rows[max(frame.start_index, frame.end_index - 2) : frame.end_index + 1]
    final5_rows = rows[max(frame.start_index, frame.end_index - 4) : frame.end_index + 1]
    breakout = rows[frame.breakout_index]
    prefix = score_name + "_"
    return {
        prefix + "GradientStart": values[0],
        prefix + "GradientEnd": values[-1],
        prefix + "GradientDelta": delta,
        prefix + "GradientVelocity": delta / duration if duration else 0.0,
        prefix + "GradientAcceleration": acceleration,
        prefix + "Final3GradientMean": mean(values[-3:]),
        prefix + "Final5GradientMean": mean(values[-5:]),
        "Final3CompressionMean": mean([float(on(row.compression)) for row in final3_rows]),
        "Final3DissipationMean": mean([float(on(row.dissipation)) for row in final3_rows]),
        "Final3AcceptedMean": mean([float(row.acceptance == "Accepted") for row in final3_rows]),
        "Final5AcceptedMean": mean([float(row.acceptance == "Accepted") for row in final5_rows]),
        "Final3ExpansionMean": mean([float(on(row.expansion)) for row in final3_rows]),
        "LateralDuration": float(duration),
        "LateralRange": frame.high - frame.low,
        "BreakoutBarRange": breakout.high - breakout.low,
        "BreakoutBarVolume": breakout.volume,
        "BreakoutBarExpansionOn": float(on(breakout.expansion)),
        "BreakoutBarDissipationOn": float(on(breakout.dissipation)),
    }


def build_cases(rows: list[EvidenceBar]) -> list[LateralCase]:
    frames = detect_completed_frames(rows, "Lateral", lateral_start)
    cases = []
    for frame in frames:
        drfwd5 = breakout_drfwd5(rows, frame)
        if drfwd5 is None:
            continue
        features: dict[str, float | None] = {}
        for score_name in SCORES:
            features.update(score_features(rows, frame, score_name))
        cases.append(
            LateralCase(
                rows[frame.start_index].bar_index,
                rows[frame.end_index].bar_index,
                rows[frame.breakout_index].bar_index,
                frame.direction,
                drfwd5,
                features,
            )
        )
    return cases


def score_metric(cases: list[LateralCase], score_name: str, metric_name: str) -> MetricStats:
    feature = score_name + "_" + metric_name
    selected = [
        case for case in cases if case.drfwd5 != 0.0 and case.features.get(feature) is not None
    ]
    if len(selected) < 3:
        rho = 0.0
    else:
        values = [float(case.features[feature]) for case in selected]
        outcomes = [1.0 if case.drfwd5 > 0.0 else 0.0 for case in selected]
        rho = pearson(rank_values(values), rank_values(outcomes))
    success = [float(case.features[feature]) for case in selected if case.drfwd5 > 0.0]
    failure = [float(case.features[feature]) for case in selected if case.drfwd5 < 0.0]
    return MetricStats(score_name, metric_name, len(selected), rho, mean(success) - mean(failure))


def build_feature_stats(cases: list[LateralCase]) -> dict[str, FeatureStats]:
    output = {}
    for feature in FEATURES:
        actual_feature = "DominanceGradient_" + feature if feature.startswith("Gradient") or feature.startswith("Final") and "Gradient" in feature else feature
        success = [
            float(case.features[actual_feature])
            for case in cases
            if case.drfwd5 > 0.0 and case.features.get(actual_feature) is not None
        ]
        failure = [
            float(case.features[actual_feature])
            for case in cases
            if case.drfwd5 < 0.0 and case.features.get(actual_feature) is not None
        ]
        output[feature] = FeatureStats(
            feature,
            len(success),
            len(failure),
            mean(success),
            mean(failure),
            mean(success) - mean(failure),
            effect_size(success, failure),
        )
    return output


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    cases = build_cases(rows)
    return InstrumentStudy(
        instrument_name(path),
        path,
        len(rows),
        cases,
        {
            (score_name, metric): score_metric(cases, score_name, metric)
            for score_name in SCORES
            for metric in METRICS
        },
        build_feature_stats(cases),
    )


def binned_stats(cases: list[LateralCase], feature: str, labels: list[tuple[str, callable]]) -> list[tuple[str, int, float, float, float, float, float]]:
    output = []
    for label, predicate in labels:
        selected = [case for case in cases if case.features.get(feature) is not None and predicate(float(case.features[feature]))]
        values = [case.drfwd5 for case in selected]
        count = len(values)
        success = sum(value > 0.0 for value in values)
        failure = sum(value < 0.0 for value in values)
        flat = count - success - failure
        denominator = count if count else 1
        output.append((label, count, mean(values), statistics.median(values) if values else 0.0, success / denominator, failure / denominator, flat / denominator))
    return output


def append_bins(lines: list[str], title: str, rows: list[tuple[str, int, float, float, float, float, float]]) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(f"{'Bin':<12} {'Count':>8} {'MeanDRFwd5':>12} {'Median':>12} {'ContRate5':>10} {'FailRate5':>10} {'FlatRate5':>10}")
    for label, count, mean_value, median_value, cont, fail, flat in rows:
        lines.append(f"{label:<12} {count:>8} {mean_value:>12.6f} {median_value:>12.6f} {cont:>9.2%} {fail:>9.2%} {flat:>9.2%}")


def instrument_report(study: InstrumentStudy) -> str:
    cases = study.cases
    durations = [float(case.features["LateralDuration"]) for case in cases]
    success = sum(case.drfwd5 > 0.0 for case in cases)
    failure = sum(case.drfwd5 < 0.0 for case in cases)
    flat = len(cases) - success - failure
    lines = [
        f"APVA Lateral Pre-Breakout Gradient Study v0.1 - {study.instrument}",
        "=" * (49 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {study.total_rows}",
        f"Lateral breakouts found: {len(cases)}",
        f"Valid outcome count: {len(cases)}",
        f"Success count: {success}",
        f"Failure count: {failure}",
        f"Flat count: {flat}",
        f"Average lateral duration: {mean(durations):.2f}",
        f"Median lateral duration: {statistics.median(durations) if durations else 0.0:.2f}",
        "Completed lateral metadata is exposed by APVA_BreakoutContext_08.py without changing prior detector behavior.",
        "",
        "Lateral Breakout Ledger",
        "=======================",
        "StartBar EndBar BreakoutBar Direction Duration LateralRange DRFwd5",
    ]
    for case in cases:
        lines.append(
            f"{case.lateral_start_bar:8d} {case.lateral_end_bar:6d} "
            f"{case.breakout_bar:11d} {case.breakout_direction:9s} "
            f"{float(case.features['LateralDuration']):8.2f} "
            f"{float(case.features['LateralRange']):12.6f} {case.drfwd5:10.6f}"
        )
    delta_bins = [(label, predicate) for label, predicate in SCORE_BINS]
    velocity_bins = [("negative", lambda value: value <= -0.25), ("near-zero", lambda value: abs(value) < 0.25), ("positive", lambda value: value >= 0.25)]
    acceleration_bins = [("negative", lambda value: value <= -0.5), ("near-zero", lambda value: abs(value) < 0.5), ("positive", lambda value: value >= 0.5)]
    for score_name in SCORES:
        lines.extend([f"\nScore: {score_name}", "=" * (7 + len(score_name))])
        append_bins(lines, "Delta Bins", binned_stats(cases, score_name + "_GradientDelta", delta_bins))
        append_bins(lines, "Velocity Bins", binned_stats(cases, score_name + "_GradientVelocity", velocity_bins))
        append_bins(lines, "Acceleration Bins", binned_stats(cases, score_name + "_GradientAcceleration", acceleration_bins))
        lines.extend(["\nCorrelation Summary", "-------------------"])
        for metric in METRICS:
            item = study.metric_stats[(score_name, metric)]
            lines.append(f"{metric:<24} Count={item.count:>6} SpearmanRho={item.rho:>8.4f} SuccessMinusFailureMean={item.delta:>10.6f}")
    lines.extend(["\nCase-Control Feature Comparison", "==============================="])
    lines.append(f"{'Feature':<28} {'SuccessN':>9} {'FailureN':>9} {'SuccessMean':>12} {'FailureMean':>12} {'Delta':>12} {'EffectSize':>12}")
    for item in study.feature_stats.values():
        lines.append(f"{item.feature:<28} {item.success_count:>9} {item.failure_count:>9} {item.success_mean:>12.6f} {item.failure_mean:>12.6f} {item.delta:>12.6f} {item.effect_size:>12.4f}")
    ranked = list(study.feature_stats.values())
    lines.extend(["\nStrongest Success Features", "=========================="])
    for item in sorted(ranked, key=lambda item: (-item.effect_size, item.feature))[:TOP_LIMIT]:
        lines.append(f"{item.feature:<28} EffectSize={item.effect_size:>8.4f} Delta={item.delta:>10.6f}")
    lines.extend(["\nStrongest Failure Features", "=========================="])
    for item in sorted(ranked, key=lambda item: (item.effect_size, item.feature))[:TOP_LIMIT]:
        lines.append(f"{item.feature:<28} EffectSize={item.effect_size:>8.4f} Delta={item.delta:>10.6f}")
    lines.extend(["\nResearch Notes", "=============="])
    primary = study.metric_stats[("DominanceGradient", "GradientVelocity")]
    acceleration = study.metric_stats[("DominanceGradient", "GradientAcceleration")]
    final3 = study.metric_stats[("DominanceGradient", "Final3GradientMean")]
    delta = study.metric_stats[("DominanceGradient", "GradientDelta")]
    lines.append(f"- DominanceGradient GradientVelocity SpearmanRho={primary.rho:.4f}.")
    lines.append(f"- DominanceGradient GradientAcceleration SpearmanRho={acceleration.rho:.4f}.")
    lines.append(f"- Final3GradientMean SpearmanRho={final3.rho:.4f}; full-lateral GradientDelta SpearmanRho={delta.rho:.4f}.")
    for feature in ("Final3CompressionMean", "Final3DissipationMean", "LateralDuration"):
        item = study.feature_stats[feature]
        lines.append(f"- {feature}: Delta={item.delta:.6f}, EffectSize={item.effect_size:.4f}.")
    if success < 10 or failure < 10:
        lines.append("- Low sample warning: fewer than 10 success or failure outcomes.")
    return "\n".join(lines) + "\n"


def instrument_columns(studies: list[InstrumentStudy]) -> list[str]:
    available = [study.instrument for study in studies]
    return list(CANONICAL_INSTRUMENTS) + sorted(set(available) - set(CANONICAL_INSTRUMENTS))


def aggregate_metric_rows(studies: list[InstrumentStudy]) -> list[dict[str, object]]:
    rows = []
    for score_name in SCORES:
        for metric in METRICS:
            values = {study.instrument: study.metric_stats[(score_name, metric)] for study in studies}
            valid = [item for item in values.values() if item.count >= MIN_VALID_NON_FLAT]
            rows.append({"score": score_name, "metric": metric, "values": values, "valid_count": len(valid), "positive_count": sum(item.rho > 0.0 for item in valid), "negative_count": sum(item.rho < 0.0 for item in valid), "mean_rho": mean([item.rho for item in valid]), "mean_delta": mean([item.delta for item in valid])})
    return rows


def aggregate_feature_rows(studies: list[InstrumentStudy]) -> list[dict[str, object]]:
    rows = []
    for feature in FEATURES:
        values = {study.instrument: study.feature_stats[feature] for study in studies}
        valid = [item for item in values.values() if item.success_count >= 10 and item.failure_count >= 10]
        rows.append({"feature": feature, "values": values, "valid_count": len(valid), "success_count": sum(item.delta > 0.0 for item in valid), "failure_count": sum(item.delta < 0.0 for item in valid), "mean_effect": mean([item.effect_size for item in valid])})
    return rows


def append_ranked(lines: list[str], title: str, rows: list[dict[str, object]], feature: bool = False) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    if feature:
        for rank, row in enumerate(rows[:TOP_LIMIT], start=1):
            lines.append(f"{rank:>4} {str(row['feature']):<28} Valid={row['valid_count']} Success={row['success_count']} Failure={row['failure_count']} MeanEffect={float(row['mean_effect']):.4f}")
    else:
        for rank, row in enumerate(rows[:TOP_LIMIT], start=1):
            lines.append(f"{rank:>4} {str(row['score']):<36} {str(row['metric']):<24} Valid={row['valid_count']} Pos={row['positive_count']} Neg={row['negative_count']} MeanRho={float(row['mean_rho']):.4f} MeanDelta={float(row['mean_delta']):.6f}")


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    metric_rows = aggregate_metric_rows(studies)
    feature_rows = aggregate_feature_rows(studies)
    columns = instrument_columns(studies)
    lines = ["APVA Lateral Pre-Breakout Gradient Study v0.1 - Cross-Instrument Aggregate", "===========================================================================", f"Input instruments: {', '.join(study.instrument for study in studies)}", f"Valid metric threshold: non-flat count >= {MIN_VALID_NON_FLAT}.", "\nCross-Instrument Metric Table", "============================="]
    header = f"{'Score':<36} {'Metric':<24}"
    for instrument in columns:
        header += f" {('Count_' + instrument):>9} {('Rho_' + instrument):>9} {('Delta_' + instrument):>11}"
    header += f" {'Valid':>5} {'Pos':>4} {'Neg':>4} {'MeanRho':>9} {'MeanDelta':>11}"
    lines.append(header)
    for row in metric_rows:
        text = f"{str(row['score']):<36} {str(row['metric']):<24}"
        for instrument in columns:
            item = row["values"].get(instrument)
            text += f" {item.count:>9} {item.rho:>9.4f} {item.delta:>11.6f}" if item else f" {'NA':>9} {'NA':>9} {'NA':>11}"
        lines.append(text + f" {row['valid_count']:>5} {row['positive_count']:>4} {row['negative_count']:>4} {float(row['mean_rho']):>9.4f} {float(row['mean_delta']):>11.6f}")
    lines.extend(["\nCross-Instrument Feature Effects", "================================"])
    for row in feature_rows:
        lines.append(f"{str(row['feature']):<28} Valid={row['valid_count']} SuccessAgreement={row['success_count']} FailureAgreement={row['failure_count']} MeanEffect={float(row['mean_effect']):.4f}")
    eligible_metrics = [row for row in metric_rows if row["valid_count"] >= 2]
    eligible_features = [row for row in feature_rows if row["valid_count"] >= 2]
    append_ranked(lines, "Most Replicated Positive Gradient Metrics", sorted(eligible_metrics, key=lambda row: (-row["positive_count"], -row["mean_rho"], row["score"], row["metric"])))
    append_ranked(lines, "Most Replicated Negative Gradient Metrics", sorted(eligible_metrics, key=lambda row: (-row["negative_count"], row["mean_rho"], row["score"], row["metric"])))
    append_ranked(lines, "Strongest Success-Side Features", sorted(eligible_features, key=lambda row: (-row["success_count"], -row["mean_effect"], row["feature"])), True)
    append_ranked(lines, "Strongest Failure-Side Features", sorted(eligible_features, key=lambda row: (-row["failure_count"], row["mean_effect"], row["feature"])), True)
    score_strength = []
    for score_name in SCORES:
        selected = [row for row in eligible_metrics if row["score"] == score_name]
        score_strength.append((score_name, mean([float(row["mean_rho"]) for row in selected])))
    lines.extend(["\nBest Score Variant", "=================="])
    for score_name, value in sorted(score_strength, key=lambda item: (-item[1], item[0])):
        lines.append(f"{score_name:<36} MeanMetricRho={value:.4f}")
    lines.extend(["\nCross-Instrument Research Notes", "==============================="])
    for score_name in SCORES:
        velocity = next(row for row in metric_rows if row["score"] == score_name and row["metric"] == "GradientVelocity")
        acceleration = next(row for row in metric_rows if row["score"] == score_name and row["metric"] == "GradientAcceleration")
        lines.append(f"- {score_name}: GradientVelocity MeanRho={float(velocity['mean_rho']):.4f}; GradientAcceleration MeanRho={float(acceleration['mean_rho']):.4f}.")
    for feature in ("Final3CompressionMean", "Final3DissipationMean", "LateralDuration"):
        row = next(item for item in feature_rows if item["feature"] == feature)
        lines.append(f"- {feature}: valid instruments={row['valid_count']}, MeanEffect={float(row['mean_effect']):.4f}.")
    lines.append(f"- Metrics excluded from replicated rankings because fewer than two valid instruments: {sum(row['valid_count'] < 2 for row in metric_rows)}.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    studies = [study_instrument(path) for path in args.input_paths]
    seen: set[str] = set()
    for study in studies:
        if study.instrument in seen:
            raise ValueError(f"Duplicate instrument input: {study.instrument}")
        seen.add(study.instrument)
        write_text(Path("Evidence") / "Output" / study.instrument / f"LateralPreBreakoutGradient_{study.instrument}.txt", instrument_report(study))
    aggregate = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, aggregate)
    print(aggregate, end="")


if __name__ == "__main__":
    main()
