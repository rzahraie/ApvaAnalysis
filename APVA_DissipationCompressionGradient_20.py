#!/usr/bin/env python3
"""Study fixed-bin gradients in mature aligned lateral APVA observations.

This research-only script reuses the study 19 population and features. It does
not optimize bins, fit parameters, or create trading logic.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

from APVA_BreakoutContext_08 import detect_frames, ibsym_start, lateral_start, load_rows
from APVA_LateralAnatomy_19 import Case, qualifying_cases
from APVA_PostBreakoutOOE_10 import build_segment_bars
from APVA_RegimeTransition_16 import CANONICAL_INSTRUMENTS, instrument_name, write_text


BINS = (
    ("0-2", 0, 2),
    ("3-4", 3, 4),
    ("5-6", 5, 6),
    ("7-8", 7, 8),
    ("9-10", 9, 10),
)
VARIABLES = (
    "Prev10_DissipationCount",
    "Prev10_CompressionCount",
    "Prev10_AcceptedCount",
    "Prev10_ClimacticCount",
    "Prev10_ExpansionCount",
)
MIN_CELL_COUNT = 10
MIN_VALID_NON_FLAT = 30
TOP_LIMIT = 25
AGGREGATE_OUTPUT = Path(
    "Evidence/Output/DissipationCompressionGradient/DissipationCompressionGradient_All.txt"
)


@dataclass(frozen=True)
class OutcomeStats:
    count: int
    mean_drfwd5: float
    median_drfwd5: float
    continuation_rate: float
    failure_rate: float
    flat_rate: float


@dataclass(frozen=True)
class GradientStats:
    variable: str
    population_count: int
    non_flat_count: int
    lowest_bin: str
    highest_bin: str
    lowest_continuation_rate: float
    highest_continuation_rate: float
    delta_continuation_rate: float
    spearman_rho: float


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    total_rows: int
    cases: list[Case]
    flats: int
    gradients: dict[str, GradientStats]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study dissipation and compression gradients in mature aligned lateral observations."
    )
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def bin_label(value: float) -> str:
    for label, low, high in BINS:
        if low <= value <= high:
            return label
    return "Other"


def outcome_stats(cases: list[Case]) -> OutcomeStats:
    values = [case.drfwd5 for case in cases]
    count = len(values)
    success = sum(value > 0.0 for value in values)
    failure = sum(value < 0.0 for value in values)
    flat = count - success - failure
    denominator = count if count else 1
    return OutcomeStats(
        count,
        statistics.fmean(values) if values else 0.0,
        statistics.median(values) if values else 0.0,
        success / denominator,
        failure / denominator,
        flat / denominator,
    )


def rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        stop = start + 1
        while stop < len(indexed) and indexed[stop][1] == indexed[start][1]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        for position in range(start, stop):
            ranks[indexed[position][0]] = average_rank
        start = stop
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 3 or len(left) != len(right):
        return 0.0
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_sum = sum((x - left_mean) ** 2 for x in left)
    right_sum = sum((y - right_mean) ** 2 for y in right)
    denominator = (left_sum * right_sum) ** 0.5
    return numerator / denominator if denominator else 0.0


def spearman(cases: list[Case], variable: str) -> float:
    non_flat = [case for case in cases if case.drfwd5 != 0.0]
    if len(non_flat) < 3:
        return 0.0
    values = [case.features[variable] for case in non_flat]
    outcomes = [1.0 if case.drfwd5 > 0.0 else 0.0 for case in non_flat]
    return pearson(rank_values(values), rank_values(outcomes))


def binned_cases(cases: list[Case], variable: str) -> dict[str, list[Case]]:
    output = {label: [] for label, _, _ in BINS}
    for case in cases:
        label = bin_label(case.features[variable])
        if label in output:
            output[label].append(case)
    return output


def gradient_stats(cases: list[Case], variable: str) -> GradientStats:
    grouped = binned_cases(cases, variable)
    occupied = [(label, outcome_stats(grouped[label])) for label, _, _ in BINS if grouped[label]]
    if occupied:
        lowest_label, lowest_stats = occupied[0]
        highest_label, highest_stats = occupied[-1]
    else:
        lowest_label = highest_label = "N/A"
        lowest_stats = highest_stats = outcome_stats([])
    return GradientStats(
        variable,
        len(cases),
        sum(case.drfwd5 != 0.0 for case in cases),
        lowest_label,
        highest_label,
        lowest_stats.continuation_rate,
        highest_stats.continuation_rate,
        highest_stats.continuation_rate - lowest_stats.continuation_rate,
        spearman(cases, variable),
    )


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    ibsym_breakouts, _ = detect_frames(rows, "IBSYM", ibsym_start)
    lateral_breakouts, _ = detect_frames(rows, "Lateral", lateral_start)
    segment_bars, _ = build_segment_bars(rows, ibsym_breakouts + lateral_breakouts)
    non_flat_cases, flats = qualifying_cases(rows, segment_bars)
    # Study 19 excludes flats from its case objects. Reconstruct the full population
    # is unnecessary for feature gradients because flats have no stored anatomy.
    return InstrumentStudy(
        instrument_name(path),
        path,
        len(rows),
        non_flat_cases,
        flats,
        {variable: gradient_stats(non_flat_cases, variable) for variable in VARIABLES},
    )


def append_variable_table(lines: list[str], cases: list[Case], variable: str) -> None:
    lines.extend([f"\n{variable}", "-" * len(variable)])
    lines.append(
        f"{'Bin':<6} {'Count':>8} {'MeanDRFwd5':>12} {'MedianDRFwd5':>14} "
        f"{'ContRate5':>10} {'FailRate5':>10} {'FlatRate5':>10}"
    )
    grouped = binned_cases(cases, variable)
    for label, _, _ in BINS:
        stats = outcome_stats(grouped[label])
        lines.append(
            f"{label:<6} {stats.count:>8} {stats.mean_drfwd5:>12.6f} "
            f"{stats.median_drfwd5:>14.6f} {stats.continuation_rate:>9.2%} "
            f"{stats.failure_rate:>9.2%} {stats.flat_rate:>9.2%}"
        )


def append_interaction(lines: list[str], cases: list[Case]) -> None:
    lines.extend(["\nDissipation x Compression Interaction", "===================================="])
    lines.append("Cell format: Count / ContinuationRate5; rates are N/A when Count < 10.")
    lines.append(f"{'Dissipation':<12} " + " ".join(f"{label:>16}" for label, _, _ in BINS))
    for diss_label, _, _ in BINS:
        cells = []
        for compression_label, _, _ in BINS:
            selected = [
                case
                for case in cases
                if bin_label(case.features["Prev10_DissipationCount"]) == diss_label
                and bin_label(case.features["Prev10_CompressionCount"]) == compression_label
            ]
            stats = outcome_stats(selected)
            rate = f"{stats.continuation_rate:.2%}" if stats.count >= MIN_CELL_COUNT else "N/A"
            cells.append(f"{stats.count:>5} / {rate:<8}")
        lines.append(f"{diss_label:<12} " + " ".join(f"{cell:>16}" for cell in cells))


def instrument_report(study: InstrumentStudy) -> str:
    cases = study.cases
    stats = outcome_stats(cases)
    lines = [
        f"APVA Dissipation/Compression Gradient Study v0.1 - {study.instrument}",
        "=" * (54 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {study.total_rows}",
        f"Qualifying population count: {len(cases) + study.flats}",
        f"Non-flat outcome count: {len(cases)}",
        f"Success count: {sum(case.drfwd5 > 0.0 for case in cases)}",
        f"Failure count: {sum(case.drfwd5 < 0.0 for case in cases)}",
        f"Flat count: {study.flats}",
        "Population and features reuse APVA_LateralAnatomy_19.py.",
        "Study 19 does not retain anatomy for flat cases, so flats are counted diagnostically but omitted from feature bins.",
        "\nVariable Bin Tables",
        "===================",
    ]
    for variable in VARIABLES:
        append_variable_table(lines, cases, variable)
    lines.extend(["\nGradient Summary", "================"])
    lines.append(
        f"{'Variable':<30} {'LowestBin':<10} {'HighestBin':<10} {'LowCont5':>10} "
        f"{'HighCont5':>10} {'DeltaCont5':>10} {'Spearman':>10}"
    )
    for item in study.gradients.values():
        lines.append(
            f"{item.variable:<30} {item.lowest_bin:<10} {item.highest_bin:<10} "
            f"{item.lowest_continuation_rate:>9.2%} {item.highest_continuation_rate:>9.2%} "
            f"{item.delta_continuation_rate:>9.2%} {item.spearman_rho:>10.4f}"
        )
    append_interaction(lines, cases)
    lines.extend(["\nResearch Notes", "=============="])
    for variable in VARIABLES:
        item = study.gradients[variable]
        lines.append(
            f"- {variable}: DeltaContinuationRate5={item.delta_continuation_rate:.2%}, "
            f"SpearmanRho={item.spearman_rho:.4f}."
        )
    lines.append(
        f"- Apparent threshold review: inspect occupied fixed-bin tables; no threshold is fit or optimized."
    )
    if len(cases) < MIN_VALID_NON_FLAT:
        lines.append(f"- Low sample warning: fewer than {MIN_VALID_NON_FLAT} non-flat outcomes.")
    if study.flats:
        lines.append(f"- Flat cases excluded from feature bins because reused study 19 population does not retain their feature records: {study.flats}.")
    return "\n".join(lines) + "\n"


def instrument_columns(studies: list[InstrumentStudy]) -> list[str]:
    available = [study.instrument for study in studies]
    return list(CANONICAL_INSTRUMENTS) + sorted(set(available) - set(CANONICAL_INSTRUMENTS))


def aggregate_rows(studies: list[InstrumentStudy]) -> list[dict[str, object]]:
    rows = []
    for variable in VARIABLES:
        values = {study.instrument: study.gradients[variable] for study in studies}
        valid = [item for item in values.values() if item.non_flat_count >= MIN_VALID_NON_FLAT]
        rows.append(
            {
                "variable": variable,
                "values": values,
                "valid_count": len(valid),
                "positive_count": sum(item.spearman_rho > 0.0 for item in valid),
                "negative_count": sum(item.spearman_rho < 0.0 for item in valid),
                "mean_spearman": statistics.fmean(item.spearman_rho for item in valid) if valid else 0.0,
                "mean_delta": statistics.fmean(item.delta_continuation_rate for item in valid) if valid else 0.0,
            }
        )
    return rows


def append_aggregate_ranked(lines: list[str], title: str, rows: list[dict[str, object]]) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(
        f"{'Rank':>4} {'Variable':<30} {'Valid':>5} {'Positive':>8} "
        f"{'Negative':>8} {'MeanRho':>10} {'MeanDelta':>10}"
    )
    for rank, row in enumerate(rows[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {str(row['variable']):<30} {int(row['valid_count']):>5} "
            f"{int(row['positive_count']):>8} {int(row['negative_count']):>8} "
            f"{float(row['mean_spearman']):>10.4f} {float(row['mean_delta']):>9.2%}"
        )


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    rows = aggregate_rows(studies)
    columns = instrument_columns(studies)
    lines = [
        "APVA Dissipation/Compression Gradient Study v0.1 - Cross-Instrument Aggregate",
        "============================================================================",
        f"Input instruments: {', '.join(study.instrument for study in studies)}",
        f"Valid instrument threshold: non-flat outcome count >= {MIN_VALID_NON_FLAT}.",
        "\nCross-Instrument Gradient Table",
        "===============================",
    ]
    header = f"{'Variable':<30}"
    for instrument in columns:
        header += f" {('Count_' + instrument):>9} {('Delta_' + instrument):>10} {('Rho_' + instrument):>9}"
    header += f" {'Valid':>5} {'Pos':>4} {'Neg':>4} {'MeanRho':>9} {'MeanDelta':>10}"
    lines.append(header)
    for row in rows:
        text = f"{str(row['variable']):<30}"
        values = row["values"]
        for instrument in columns:
            item = values.get(instrument)
            if item is None:
                text += f" {'NA':>9} {'NA':>10} {'NA':>9}"
            else:
                text += f" {item.non_flat_count:>9} {item.delta_continuation_rate:>9.2%} {item.spearman_rho:>9.4f}"
        text += (
            f" {int(row['valid_count']):>5} {int(row['positive_count']):>4} "
            f"{int(row['negative_count']):>4} {float(row['mean_spearman']):>9.4f} "
            f"{float(row['mean_delta']):>9.2%}"
        )
        lines.append(text)
    eligible = [row for row in rows if row["valid_count"] >= 2]
    append_aggregate_ranked(lines, "Most Replicated Positive Gradients", sorted(eligible, key=lambda row: (-row["positive_count"], -row["mean_spearman"], row["variable"])))
    append_aggregate_ranked(lines, "Most Replicated Negative Gradients", sorted(eligible, key=lambda row: (-row["negative_count"], row["mean_spearman"], row["variable"])))
    append_aggregate_ranked(lines, "Strongest Average Positive Gradients", sorted(eligible, key=lambda row: (-row["mean_spearman"], row["variable"])))
    append_aggregate_ranked(lines, "Strongest Average Negative Gradients", sorted(eligible, key=lambda row: (row["mean_spearman"], row["variable"])))
    lines.extend(["\nCross-Instrument Research Notes", "==============================="])
    positive = [row for row in eligible if row["positive_count"] >= 2]
    negative = [row for row in eligible if row["negative_count"] >= 2]
    lines.append(f"- Variables replicating positive gradients in at least two valid instruments: {', '.join(str(row['variable']) for row in positive) or 'none'}.")
    lines.append(f"- Variables replicating negative gradients in at least two valid instruments: {', '.join(str(row['variable']) for row in negative) or 'none'}.")
    for variable in ("Prev10_DissipationCount", "Prev10_CompressionCount"):
        row = next(item for item in rows if item["variable"] == variable)
        lines.append(
            f"- {variable}: valid instruments={row['valid_count']}, positive slopes={row['positive_count']}, "
            f"negative slopes={row['negative_count']}, MeanSpearman={float(row['mean_spearman']):.4f}."
        )
    lines.append(f"- Variables excluded from replicated rankings because fewer than two valid instruments: {sum(row['valid_count'] < 2 for row in rows)}.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    studies = [study_instrument(path) for path in args.input_paths]
    seen: set[str] = set()
    for study in studies:
        if study.instrument in seen:
            raise ValueError(f"Duplicate instrument input: {study.instrument}")
        seen.add(study.instrument)
        write_text(
            Path("Evidence") / "Output" / study.instrument / f"DissipationCompressionGradient_{study.instrument}.txt",
            instrument_report(study),
        )
    aggregate = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, aggregate)
    print(aggregate, end="")


if __name__ == "__main__":
    main()
