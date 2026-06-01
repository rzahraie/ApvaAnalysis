#!/usr/bin/env python3
"""Study fixed combined gradient scores in reused APVA lateral observations.

The requested dominance-gradient names are descriptive score labels only. This
research-only script does not optimize weights, fit parameters, or create trades.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from APVA_BreakoutContext_08 import detect_frames, ibsym_start, lateral_start, load_rows
from APVA_DissipationCompressionGradient_20 import spearman
from APVA_LateralAnatomy_19 import Case, qualifying_cases
from APVA_PostBreakoutOOE_10 import build_segment_bars
from APVA_RegimeTransition_16 import CANONICAL_INSTRUMENTS, instrument_name, write_text


SCORE_BINS = (
    ("<= -5", lambda value: value <= -5),
    ("-4 to -2", lambda value: -4 <= value <= -2),
    ("-1 to 1", lambda value: -1 <= value <= 1),
    ("2 to 4", lambda value: 2 <= value <= 4),
    ("5 to 7", lambda value: 5 <= value <= 7),
    (">= 8", lambda value: value >= 8),
)
SCORES = (
    "DominanceGradient",
    "DominanceGradient_WithExpansion",
    "DominanceGradient_Weighted",
)
MIN_VALID_NON_FLAT = 30
TOP_LIMIT = 25
AGGREGATE_OUTPUT = Path("Evidence/Output/DominanceGradient/DominanceGradient_All.txt")


@dataclass(frozen=True)
class OutcomeStats:
    count: int
    mean_drfwd5: float
    median_drfwd5: float
    continuation_rate: float
    failure_rate: float
    flat_rate: float


@dataclass(frozen=True)
class ScoreStats:
    score_name: str
    non_flat_count: int
    spearman_rho: float
    lowest_bin: str
    highest_bin: str
    lowest_continuation_rate: float
    highest_continuation_rate: float
    delta_continuation_rate: float
    low_tercile_continuation_rate: float
    high_tercile_continuation_rate: float
    high_minus_low_continuation_rate: float
    high_minus_low_mean_drfwd5: float


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    total_rows: int
    cases: list[Case]
    flats: int
    score_stats: dict[str, ScoreStats]
    individual_gradients: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study fixed combined dominance-gradient score labels."
    )
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def score_value(case: Case, score_name: str) -> float:
    features = case.features
    dissipation = features["Prev10_DissipationCount"]
    accepted = features["Prev10_AcceptedCount"]
    compression = features["Prev10_CompressionCount"]
    expansion = features["Prev10_ExpansionCount"]
    if score_name == "DominanceGradient":
        return dissipation + accepted - compression
    if score_name == "DominanceGradient_WithExpansion":
        return dissipation + accepted + expansion - compression
    return 2 * dissipation + accepted - 2 * compression


def bin_label(value: float) -> str:
    return next(label for label, predicate in SCORE_BINS if predicate(value))


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


def grouped_cases(cases: list[Case], score_name: str) -> dict[str, list[Case]]:
    output = {label: [] for label, _ in SCORE_BINS}
    for case in cases:
        output[bin_label(score_value(case, score_name))].append(case)
    return output


def tercile_groups(cases: list[Case], score_name: str) -> dict[str, list[Case]]:
    ordered = sorted(cases, key=lambda case: (score_value(case, score_name), case.index))
    size = len(ordered)
    low_stop = size // 3
    high_start = size - size // 3
    return {
        "Low": ordered[:low_stop],
        "Middle": ordered[low_stop:high_start],
        "High": ordered[high_start:],
    }


def score_stats(cases: list[Case], score_name: str) -> ScoreStats:
    grouped = grouped_cases(cases, score_name)
    occupied = [(label, outcome_stats(grouped[label])) for label, _ in SCORE_BINS if grouped[label]]
    if occupied:
        lowest_label, lowest = occupied[0]
        highest_label, highest = occupied[-1]
    else:
        lowest_label = highest_label = "N/A"
        lowest = highest = outcome_stats([])
    terciles = tercile_groups(cases, score_name)
    low = outcome_stats(terciles["Low"])
    high = outcome_stats(terciles["High"])
    score_cases = [
        Case(case.index, case.drfwd5, {score_name: score_value(case, score_name)})
        for case in cases
    ]
    return ScoreStats(
        score_name,
        sum(case.drfwd5 != 0.0 for case in cases),
        spearman(score_cases, score_name),
        lowest_label,
        highest_label,
        lowest.continuation_rate,
        highest.continuation_rate,
        highest.continuation_rate - lowest.continuation_rate,
        low.continuation_rate,
        high.continuation_rate,
        high.continuation_rate - low.continuation_rate,
        high.mean_drfwd5 - low.mean_drfwd5,
    )


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    ibsym_breakouts, _ = detect_frames(rows, "IBSYM", ibsym_start)
    lateral_breakouts, _ = detect_frames(rows, "Lateral", lateral_start)
    segment_bars, _ = build_segment_bars(rows, ibsym_breakouts + lateral_breakouts)
    cases, flats = qualifying_cases(rows, segment_bars)
    return InstrumentStudy(
        instrument_name(path),
        path,
        len(rows),
        cases,
        flats,
        {score_name: score_stats(cases, score_name) for score_name in SCORES},
        {
            "Prev10_DissipationCount": spearman(cases, "Prev10_DissipationCount"),
            "Prev10_CompressionCount": spearman(cases, "Prev10_CompressionCount"),
        },
    )


def append_score_table(lines: list[str], cases: list[Case], score_name: str) -> None:
    lines.extend([f"\n{score_name}", "-" * len(score_name)])
    lines.append(
        f"{'Bin':<10} {'Count':>8} {'MeanDRFwd5':>12} {'MedianDRFwd5':>14} "
        f"{'ContRate5':>10} {'FailRate5':>10} {'FlatRate5':>10}"
    )
    grouped = grouped_cases(cases, score_name)
    for label, _ in SCORE_BINS:
        stats = outcome_stats(grouped[label])
        lines.append(
            f"{label:<10} {stats.count:>8} {stats.mean_drfwd5:>12.6f} "
            f"{stats.median_drfwd5:>14.6f} {stats.continuation_rate:>9.2%} "
            f"{stats.failure_rate:>9.2%} {stats.flat_rate:>9.2%}"
        )
    lines.extend(["\nTercile Split", "-------------"])
    lines.append(f"{'Tercile':<8} {'Count':>8} {'MeanDRFwd5':>12} {'ContRate5':>10}")
    for label, selected in tercile_groups(cases, score_name).items():
        stats = outcome_stats(selected)
        lines.append(f"{label:<8} {stats.count:>8} {stats.mean_drfwd5:>12.6f} {stats.continuation_rate:>9.2%}")


def instrument_report(study: InstrumentStudy) -> str:
    cases = study.cases
    success = sum(case.drfwd5 > 0.0 for case in cases)
    failure = sum(case.drfwd5 < 0.0 for case in cases)
    lines = [
        f"APVA Dominance Gradient Study v0.1 - {study.instrument}",
        "=" * (38 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Qualifying population count: {len(cases) + study.flats}",
        f"Success count: {success}",
        f"Failure count: {failure}",
        f"Flat count: {study.flats}",
        "Dominance-gradient names are fixed descriptive score labels only.",
        "Population and features reuse APVA_LateralAnatomy_19.py.",
        "Flat feature records are unavailable from reused study 19 and are counted diagnostically only.",
        "\nScore Tables",
        "============",
    ]
    for score_name in SCORES:
        append_score_table(lines, cases, score_name)
    lines.extend(["\nScore Comparison Summary", "========================"])
    lines.append(
        f"{'ScoreName':<36} {'Spearman':>10} {'LowestBin':<10} {'HighestBin':<10} "
        f"{'DeltaCont5':>10} {'HighCont5':>10} {'LowCont5':>10} {'High-Low':>10}"
    )
    for item in study.score_stats.values():
        lines.append(
            f"{item.score_name:<36} {item.spearman_rho:>10.4f} {item.lowest_bin:<10} "
            f"{item.highest_bin:<10} {item.delta_continuation_rate:>9.2%} "
            f"{item.high_tercile_continuation_rate:>9.2%} "
            f"{item.low_tercile_continuation_rate:>9.2%} "
            f"{item.high_minus_low_continuation_rate:>9.2%}"
        )
    lines.extend(["\nResearch Notes", "=============="])
    best = max(study.score_stats.values(), key=lambda item: (item.spearman_rho, item.score_name))
    lines.append(f"- Highest combined-score SpearmanRho: {best.score_name} ({best.spearman_rho:.4f}).")
    for item in study.score_stats.values():
        lines.append(
            f"- {item.score_name}: SpearmanRho={item.spearman_rho:.4f}, "
            f"HighMinusLowContinuationRate5={item.high_minus_low_continuation_rate:.2%}."
        )
    lines.append(
        f"- Individual gradient reference: Prev10_DissipationCount rho="
        f"{study.individual_gradients['Prev10_DissipationCount']:.4f}; "
        f"Prev10_CompressionCount rho={study.individual_gradients['Prev10_CompressionCount']:.4f}."
    )
    lines.append("- Monotonicity review is mechanical: inspect fixed-bin and tercile tables; no curve is fit.")
    if len(cases) < MIN_VALID_NON_FLAT:
        lines.append(f"- Low sample warning: fewer than {MIN_VALID_NON_FLAT} non-flat outcomes.")
    return "\n".join(lines) + "\n"


def instrument_columns(studies: list[InstrumentStudy]) -> list[str]:
    available = [study.instrument for study in studies]
    return list(CANONICAL_INSTRUMENTS) + sorted(set(available) - set(CANONICAL_INSTRUMENTS))


def aggregate_rows(studies: list[InstrumentStudy]) -> list[dict[str, object]]:
    rows = []
    for score_name in SCORES:
        values = {study.instrument: study.score_stats[score_name] for study in studies}
        valid = [item for item in values.values() if item.non_flat_count >= MIN_VALID_NON_FLAT]
        rows.append(
            {
                "score": score_name,
                "values": values,
                "valid_count": len(valid),
                "positive_spearman_count": sum(item.spearman_rho > 0.0 for item in valid),
                "positive_high_low_count": sum(item.high_minus_low_continuation_rate > 0.0 for item in valid),
                "mean_spearman": statistics.fmean(item.spearman_rho for item in valid) if valid else 0.0,
                "mean_high_low": statistics.fmean(item.high_minus_low_continuation_rate for item in valid) if valid else 0.0,
                "stability": statistics.pstdev(item.spearman_rho for item in valid) if len(valid) > 1 else 0.0,
            }
        )
    return rows


def append_aggregate_ranked(lines: list[str], title: str, rows: list[dict[str, object]]) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(
        f"{'Rank':>4} {'Score':<36} {'Valid':>5} {'PosRho':>6} {'PosHiLo':>7} "
        f"{'MeanRho':>10} {'MeanHiLo':>10} {'RhoStd':>10}"
    )
    for rank, row in enumerate(rows[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {str(row['score']):<36} {int(row['valid_count']):>5} "
            f"{int(row['positive_spearman_count']):>6} {int(row['positive_high_low_count']):>7} "
            f"{float(row['mean_spearman']):>10.4f} {float(row['mean_high_low']):>9.2%} "
            f"{float(row['stability']):>10.4f}"
        )


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    rows = aggregate_rows(studies)
    columns = instrument_columns(studies)
    lines = [
        "APVA Dominance Gradient Study v0.1 - Cross-Instrument Aggregate",
        "==============================================================",
        f"Input instruments: {', '.join(study.instrument for study in studies)}",
        f"Valid instrument threshold: non-flat outcome count >= {MIN_VALID_NON_FLAT}.",
        "Dominance-gradient names are fixed descriptive score labels only.",
        "\nCross-Instrument Score Table",
        "============================",
    ]
    header = f"{'Score':<36}"
    for instrument in columns:
        header += f" {('Count_' + instrument):>9} {('Rho_' + instrument):>9} {('HiLo_' + instrument):>9}"
    header += f" {'Valid':>5} {'PosRho':>6} {'PosHiLo':>7} {'MeanRho':>9} {'MeanHiLo':>10}"
    lines.append(header)
    for row in rows:
        text = f"{str(row['score']):<36}"
        values = row["values"]
        for instrument in columns:
            item = values.get(instrument)
            if item is None:
                text += f" {'NA':>9} {'NA':>9} {'NA':>9}"
            else:
                text += (
                    f" {item.non_flat_count:>9} {item.spearman_rho:>9.4f} "
                    f"{item.high_minus_low_continuation_rate:>8.2%}"
                )
        text += (
            f" {int(row['valid_count']):>5} {int(row['positive_spearman_count']):>6} "
            f"{int(row['positive_high_low_count']):>7} {float(row['mean_spearman']):>9.4f} "
            f"{float(row['mean_high_low']):>9.2%}"
        )
        lines.append(text)
    eligible = [row for row in rows if row["valid_count"] >= 2]
    append_aggregate_ranked(lines, "Most Replicated Positive Scores", sorted(eligible, key=lambda row: (-row["positive_spearman_count"], -row["positive_high_low_count"], -row["mean_spearman"], row["score"])))
    append_aggregate_ranked(lines, "Strongest Average Spearman", sorted(eligible, key=lambda row: (-row["mean_spearman"], row["score"])))
    append_aggregate_ranked(lines, "Strongest High-Minus-Low Continuation Advantage", sorted(eligible, key=lambda row: (-row["mean_high_low"], row["score"])))
    append_aggregate_ranked(lines, "Most Stable Score across Instruments", sorted(eligible, key=lambda row: (row["stability"], -row["mean_spearman"], row["score"])))
    lines.extend(["\nCross-Instrument Research Notes", "==============================="])
    by_name = {row["score"]: row for row in rows}
    for score_name in SCORES:
        row = by_name[score_name]
        lines.append(
            f"- {score_name}: valid instruments={row['valid_count']}, positive Spearman slopes="
            f"{row['positive_spearman_count']}, positive HighMinusLow values={row['positive_high_low_count']}, "
            f"MeanSpearman={float(row['mean_spearman']):.4f}."
        )
    base = by_name["DominanceGradient"]
    expansion = by_name["DominanceGradient_WithExpansion"]
    weighted = by_name["DominanceGradient_Weighted"]
    lines.append(
        f"- Expansion effect on MeanSpearman: {float(expansion['mean_spearman']) - float(base['mean_spearman']):.4f}."
    )
    lines.append(
        f"- Weighting effect on MeanSpearman: {float(weighted['mean_spearman']) - float(base['mean_spearman']):.4f}."
    )
    lines.append(
        "- Comparison to individual Prev10_DissipationCount / Prev10_CompressionCount gradients remains visible in per-instrument reports."
    )
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
            Path("Evidence") / "Output" / study.instrument / f"DominanceGradient_{study.instrument}.txt",
            instrument_report(study),
        )
    aggregate = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, aggregate)
    print(aggregate, end="")


if __name__ == "__main__":
    main()
