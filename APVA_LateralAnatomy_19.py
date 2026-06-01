#!/usr/bin/env python3
"""Study anatomy of successful and failed mature aligned lateral observations.

This research-only case-control study reuses prior APVA breakout, stage,
alignment, event, and DRFwd logic. It does not fit parameters or create trades.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

from APVA_BreakoutContext_08 import (
    EvidenceBar,
    detect_frames,
    direction_relative_return,
    ibsym_start,
    lateral_start,
    load_rows,
)
from APVA_RegimeTransition_16 import CANONICAL_INSTRUMENTS, instrument_name, write_text
from APVA_StageAlignment_13 import alignment_for, event_definitions
from APVA_PostBreakoutOOE_10 import SegmentBar, build_segment_bars


AGGREGATE_OUTPUT = Path("Evidence/Output/LateralAnatomy/LateralAnatomy_All.txt")
TOP_LIMIT = 25
MIN_VALID_OUTCOMES = 10


@dataclass(frozen=True)
class Case:
    index: int
    drfwd5: float
    features: dict[str, float]


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
    rows: list[EvidenceBar]
    cases: list[Case]
    flats: int
    feature_stats: dict[str, FeatureStats]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study anatomy of mature aligned lateral DissipationContained observations."
    )
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def on(value: str) -> bool:
    return value not in {"Absent", "Other"}


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def window_features(rows: list[EvidenceBar], index: int, size: int) -> dict[str, float]:
    window = rows[max(0, index - size) : index]
    volumes = [row.volume for row in window]
    ranges = [row.high - row.low for row in window]
    polarity = rows[index].polarity
    prefix = f"Prev{size}_"
    return {
        prefix + "MeanVolume": statistics.fmean(volumes) if volumes else 0.0,
        prefix + "MedianVolume": statistics.median(volumes) if volumes else 0.0,
        prefix + "MeanRange": statistics.fmean(ranges) if ranges else 0.0,
        prefix + "MedianRange": statistics.median(ranges) if ranges else 0.0,
        prefix + "CompressionCount": float(sum(on(row.compression) for row in window)),
        prefix + "ExpansionCount": float(sum(on(row.expansion) for row in window)),
        prefix + "DissipationCount": float(sum(on(row.dissipation) for row in window)),
        prefix + "PeakCount": float(sum(row.participation == "Peak" for row in window)),
        prefix + "ClimacticCount": float(sum(row.participation == "Climactic" for row in window)),
        prefix + "FallingCount": float(sum(row.participation == "Falling" for row in window)),
        prefix + "RisingCount": float(sum(row.participation == "Rising" for row in window)),
        prefix + "AcceptedCount": float(sum(row.acceptance == "Accepted" for row in window)),
        prefix + "ContainedCount": float(sum(row.acceptance == "Contained" for row in window)),
        prefix + "SamePolarityCount": float(sum(row.polarity == polarity for row in window)),
        prefix + "OppositePolarityCount": float(
            sum(
                row.polarity in {"Black", "Red"} and row.polarity != polarity
                for row in window
            )
        ),
    }


def anchor_features(row: EvidenceBar) -> dict[str, float]:
    bar_range = row.high - row.low
    body = abs(row.close - row.open)
    upper_wick = row.high - max(row.open, row.close)
    lower_wick = min(row.open, row.close) - row.low
    return {
        "AnchorVolume": row.volume,
        "AnchorRange": bar_range,
        "AnchorBody": body,
        "AnchorBodyToRange": safe_ratio(body, bar_range),
        "AnchorUpperWick": upper_wick,
        "AnchorLowerWick": lower_wick,
        "AnchorWickImbalance": upper_wick - lower_wick,
        "AnchorParticipation_Falling": float(row.participation == "Falling"),
        "AnchorParticipation_Rising": float(row.participation == "Rising"),
        "AnchorParticipation_Peak": float(row.participation == "Peak"),
        "AnchorParticipation_Climactic": float(row.participation == "Climactic"),
        "AnchorParticipation_Other": float(
            row.participation not in {"Falling", "Rising", "Peak", "Climactic"}
        ),
        "AnchorAcceptance_Accepted": float(row.acceptance == "Accepted"),
        "AnchorAcceptance_Contained": float(row.acceptance == "Contained"),
        "AnchorAcceptance_Other": float(row.acceptance not in {"Accepted", "Contained"}),
        "AnchorCompressionOn": float(on(row.compression)),
        "AnchorExpansionOn": float(on(row.expansion)),
        "AnchorDissipationOn": float(on(row.dissipation)),
        "AnchorPolarity_Black": float(row.polarity == "Black"),
        "AnchorPolarity_Red": float(row.polarity == "Red"),
    }


def case_features(rows: list[EvidenceBar], index: int) -> dict[str, float]:
    features = anchor_features(rows[index])
    features.update(window_features(rows, index, 5))
    features.update(window_features(rows, index, 10))
    return features


def qualifying_cases(rows: list[EvidenceBar], segment_bars: list[SegmentBar]) -> tuple[list[Case], int]:
    dissipation_contained = next(
        event for event in event_definitions() if event.name == "DissipationContained"
    )
    cases: list[Case] = []
    flats = 0
    for segment_bar in segment_bars:
        if segment_bar.breakout_type != "Lateral":
            continue
        if segment_bar.sequence_stage != 4:
            continue
        if alignment_for(rows, segment_bar) != "Aligned":
            continue
        if not dissipation_contained.predicate(rows[segment_bar.index]):
            continue
        drfwd5 = direction_relative_return(rows, segment_bar.index, 5)
        if drfwd5 is None:
            continue
        if drfwd5 == 0.0:
            flats += 1
            continue
        cases.append(Case(segment_bar.index, drfwd5, case_features(rows, segment_bar.index)))
    return cases, flats


def effect_size(success_values: list[float], failure_values: list[float]) -> float:
    if len(success_values) < 2 or len(failure_values) < 2:
        return 0.0
    success_variance = statistics.variance(success_values)
    failure_variance = statistics.variance(failure_values)
    pooled_denominator = len(success_values) + len(failure_values) - 2
    if pooled_denominator <= 0:
        return 0.0
    pooled_variance = (
        (len(success_values) - 1) * success_variance
        + (len(failure_values) - 1) * failure_variance
    ) / pooled_denominator
    pooled_std = pooled_variance**0.5
    return (
        (statistics.fmean(success_values) - statistics.fmean(failure_values)) / pooled_std
        if pooled_std
        else 0.0
    )


def build_feature_stats(cases: list[Case]) -> dict[str, FeatureStats]:
    if not cases:
        return {}
    output = {}
    for feature in cases[0].features:
        success_values = [case.features[feature] for case in cases if case.drfwd5 > 0.0]
        failure_values = [case.features[feature] for case in cases if case.drfwd5 < 0.0]
        success_mean = statistics.fmean(success_values) if success_values else 0.0
        failure_mean = statistics.fmean(failure_values) if failure_values else 0.0
        output[feature] = FeatureStats(
            feature,
            len(success_values),
            len(failure_values),
            success_mean,
            failure_mean,
            success_mean - failure_mean,
            effect_size(success_values, failure_values),
        )
    return output


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    ibsym_breakouts, _ = detect_frames(rows, "IBSYM", ibsym_start)
    lateral_breakouts, _ = detect_frames(rows, "Lateral", lateral_start)
    segment_bars, _ = build_segment_bars(rows, ibsym_breakouts + lateral_breakouts)
    cases, flats = qualifying_cases(rows, segment_bars)
    return InstrumentStudy(instrument_name(path), path, rows, cases, flats, build_feature_stats(cases))


def append_ranked(lines: list[str], title: str, stats: list[FeatureStats], reverse: bool) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(f"{'Rank':>4} {'Feature':<34} {'EffectSize':>12} {'Delta':>12} {'SuccessMean':>12} {'FailureMean':>12}")
    ordered = sorted(stats, key=lambda item: (item.effect_size, item.feature), reverse=reverse)
    for rank, item in enumerate(ordered[:TOP_LIMIT], start=1):
        lines.append(f"{rank:>4} {item.feature:<34} {item.effect_size:>12.4f} {item.delta:>12.6f} {item.success_mean:>12.6f} {item.failure_mean:>12.6f}")


def instrument_report(study: InstrumentStudy) -> str:
    success = [case for case in study.cases if case.drfwd5 > 0.0]
    failure = [case for case in study.cases if case.drfwd5 < 0.0]
    values = [case.drfwd5 for case in study.cases]
    denominator = len(study.cases) + study.flats
    lines = [
        f"APVA Lateral Anatomy Case-Control Study v0.1 - {study.instrument}",
        "=" * (49 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {len(study.rows)}",
        f"Qualifying cases: {denominator}",
        f"Success count: {len(success)}",
        f"Failure count: {len(failure)}",
        f"Flat count: {study.flats}",
        "Population: Lateral + DissipationContained + Stage4 Mature + Aligned.",
        "Lateral frame metadata is skipped because reused segment bars do not expose it.",
        "\nFeature Comparison Table",
        "========================",
        f"{'Feature':<34} {'SuccessN':>9} {'FailureN':>9} {'SuccessMean':>12} {'FailureMean':>12} {'Delta':>12} {'EffectSize':>12}",
    ]
    for item in study.feature_stats.values():
        lines.append(f"{item.feature:<34} {item.success_count:>9} {item.failure_count:>9} {item.success_mean:>12.6f} {item.failure_mean:>12.6f} {item.delta:>12.6f} {item.effect_size:>12.4f}")
    stats = list(study.feature_stats.values())
    append_ranked(lines, "Top 25 Success-Leaning Features", stats, True)
    append_ranked(lines, "Top 25 Failure-Leaning Features", stats, False)
    lines.extend(["\nOutcome Sanity", "=============="])
    lines.append(f"MeanDRFwd5: {statistics.fmean(values) if values else 0.0:.6f}")
    lines.append(f"MedianDRFwd5: {statistics.median(values) if values else 0.0:.6f}")
    lines.append(f"ContinuationRate5: {len(success) / denominator if denominator else 0.0:.2%}")
    lines.append(f"FailureRate5: {len(failure) / denominator if denominator else 0.0:.2%}")
    lines.extend(["\nResearch Notes", "=============="])
    positive = sorted(stats, key=lambda item: (-item.effect_size, item.feature))
    negative = sorted(stats, key=lambda item: (item.effect_size, item.feature))
    lines.append(f"- Highest success-leaning effects: {', '.join(item.feature for item in positive[:5]) or 'none'}.")
    lines.append(f"- Highest failure-leaning effects: {', '.join(item.feature for item in negative[:5]) or 'none'}.")
    for feature in ("Prev5_ExpansionCount", "Prev5_CompressionCount", "Prev5_DissipationCount", "AnchorBodyToRange", "AnchorWickImbalance"):
        item = study.feature_stats.get(feature)
        if item:
            lines.append(f"- {feature}: Delta={item.delta:.6f}, EffectSize={item.effect_size:.4f}.")
    if len(success) < MIN_VALID_OUTCOMES or len(failure) < MIN_VALID_OUTCOMES:
        lines.append(f"- Low sample warning: fewer than {MIN_VALID_OUTCOMES} success or failure cases.")
    return "\n".join(lines) + "\n"


def aggregate_rows(studies: list[InstrumentStudy]) -> list[dict[str, object]]:
    features = sorted({feature for study in studies for feature in study.feature_stats})
    rows = []
    for feature in features:
        values = {study.instrument: study.feature_stats.get(feature) for study in studies}
        valid = [
            item
            for item in values.values()
            if item is not None
            and item.success_count >= MIN_VALID_OUTCOMES
            and item.failure_count >= MIN_VALID_OUTCOMES
        ]
        rows.append({
            "feature": feature,
            "values": values,
            "valid_count": len(valid),
            "success_agreement": sum(item.delta > 0.0 for item in valid),
            "failure_agreement": sum(item.delta < 0.0 for item in valid),
            "mean_effect": statistics.fmean(item.effect_size for item in valid) if valid else 0.0,
        })
    return rows


def instrument_columns(studies: list[InstrumentStudy]) -> list[str]:
    available = [study.instrument for study in studies]
    return list(CANONICAL_INSTRUMENTS) + sorted(set(available) - set(CANONICAL_INSTRUMENTS))


def append_aggregate_ranked(lines: list[str], title: str, rows: list[dict[str, object]]) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(f"{'Rank':>4} {'Feature':<34} {'Valid':>5} {'Success':>7} {'Failure':>7} {'MeanEffect':>11}")
    for rank, row in enumerate(rows[:TOP_LIMIT], start=1):
        lines.append(f"{rank:>4} {str(row['feature']):<34} {int(row['valid_count']):>5} {int(row['success_agreement']):>7} {int(row['failure_agreement']):>7} {float(row['mean_effect']):>11.4f}")


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    rows = aggregate_rows(studies)
    columns = instrument_columns(studies)
    lines = [
        "APVA Lateral Anatomy Case-Control Study v0.1 - Cross-Instrument Aggregate",
        "=======================================================================",
        f"Input instruments: {', '.join(study.instrument for study in studies)}",
        f"Valid instrument threshold: SuccessCount >= {MIN_VALID_OUTCOMES} and FailureCount >= {MIN_VALID_OUTCOMES}.",
        "\nCross-Instrument Feature Table",
        "==============================",
    ]
    header = f"{'Feature':<34}"
    for instrument in columns:
        header += f" {('Delta_' + instrument):>12} {('Effect_' + instrument):>12}"
    header += f" {'Valid':>5} {'Success':>7} {'Failure':>7} {'MeanEffect':>11}"
    lines.append(header)
    for row in rows:
        text = f"{str(row['feature']):<34}"
        values = row["values"]
        for instrument in columns:
            item = values.get(instrument)
            if item is None:
                text += f" {'NA':>12} {'NA':>12}"
            else:
                text += f" {item.delta:>12.6f} {item.effect_size:>12.4f}"
        text += f" {int(row['valid_count']):>5} {int(row['success_agreement']):>7} {int(row['failure_agreement']):>7} {float(row['mean_effect']):>11.4f}"
        lines.append(text)
    eligible = [row for row in rows if row["valid_count"] >= 2]
    append_aggregate_ranked(lines, "Most Replicated Success Features", sorted(eligible, key=lambda row: (-row["success_agreement"], -row["mean_effect"], row["feature"])))
    append_aggregate_ranked(lines, "Most Replicated Failure Features", sorted(eligible, key=lambda row: (-row["failure_agreement"], row["mean_effect"], row["feature"])))
    append_aggregate_ranked(lines, "Strongest Average Positive Effects", sorted(eligible, key=lambda row: (-row["mean_effect"], row["feature"])))
    append_aggregate_ranked(lines, "Strongest Average Negative Effects", sorted(eligible, key=lambda row: (row["mean_effect"], row["feature"])))
    lines.extend(["\nCross-Instrument Research Notes", "==============================="])
    replicated_success = [row for row in eligible if row["success_agreement"] >= 2]
    replicated_failure = [row for row in eligible if row["failure_agreement"] >= 2]
    lines.append(f"- Features replicating toward success in at least two valid instruments: {', '.join(str(row['feature']) for row in replicated_success) or 'none'}.")
    lines.append(f"- Features replicating toward failure in at least two valid instruments: {', '.join(str(row['feature']) for row in replicated_failure) or 'none'}.")
    if eligible:
        positive = max(eligible, key=lambda row: (row["mean_effect"], row["feature"]))
        negative = min(eligible, key=lambda row: (row["mean_effect"], row["feature"]))
        lines.append(f"- Strongest average positive effect: {positive['feature']} ({float(positive['mean_effect']):.4f}).")
        lines.append(f"- Strongest average negative effect: {negative['feature']} ({float(negative['mean_effect']):.4f}).")
    for feature in ("Prev5_ExpansionCount", "Prev5_DissipationCount", "Prev5_CompressionCount", "AnchorBodyToRange", "AnchorWickImbalance"):
        row = next((item for item in rows if item["feature"] == feature), None)
        if row:
            lines.append(f"- {feature}: valid instruments={row['valid_count']}, MeanEffect={float(row['mean_effect']):.4f}.")
    limited = sum(row["valid_count"] < 2 for row in rows)
    lines.append(f"- Features excluded from replicated rankings because fewer than two valid instruments: {limited}.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    studies = [study_instrument(path) for path in args.input_paths]
    seen: set[str] = set()
    for study in studies:
        if study.instrument in seen:
            raise ValueError(f"Duplicate instrument input: {study.instrument}")
        seen.add(study.instrument)
        write_text(Path("Evidence") / "Output" / study.instrument / f"LateralAnatomy_{study.instrument}.txt", instrument_report(study))
    aggregate = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, aggregate)
    print(aggregate, end="")


if __name__ == "__main__":
    main()
