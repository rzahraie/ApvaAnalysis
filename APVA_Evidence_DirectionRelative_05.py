#!/usr/bin/env python3
"""Print direction-relative APVA Evidence v0.1 consequence statistics.

This research-only study measures continuation or failure relative to terminal
bar polarity. It does not infer higher-layer structure or create trading signals.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from APVA_Evidence_Consequences_02 import DEFAULT_INPUT, HORIZONS, event_sets, load_rows
from APVA_Evidence_Sequences_03 import (
    DEFAULT_LENGTHS,
    SequenceKey,
    display_sequence,
    label_rows,
    sequence_occurrences,
)

DEFAULT_OUTPUT = Path("DirectionRelative.txt")
MIN_RANKED_SAMPLES = 30
TOP_LIMIT = 25

Row = dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print APVA Evidence v0.1 direction-relative consequence tables."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Evidence CSV path (default: {DEFAULT_INPUT.as_posix()})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Report output path (default: {DEFAULT_OUTPUT.as_posix()})",
    )
    return parser.parse_args()


def direction_relative_value(row: Row, forward_close: float) -> float | None:
    close = float(row["Close"])
    polarity = row["VolumePolarity"]
    if polarity == "Black":
        return forward_close - close
    if polarity == "Red":
        return close - forward_close
    return None


def direction_relative_returns(
    rows: list[Row],
    terminal_indexes: list[int],
    horizon: int,
) -> list[float]:
    values: list[float] = []
    for terminal_index in terminal_indexes:
        if terminal_index + horizon >= len(rows):
            continue
        value = direction_relative_value(
            rows[terminal_index], float(rows[terminal_index + horizon]["Close"])
        )
        if value is not None:
            values.append(value)
    return values


def summarize(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    continuation = sum(value > 0.0 for value in values)
    failure = sum(value < 0.0 for value in values)
    flat = count - continuation - failure
    denominator = count if count > 0 else 1
    return {
        "Count": count,
        "MeanDRFwd": statistics.fmean(values) if values else 0.0,
        "MedianDRFwd": statistics.median(values) if values else 0.0,
        "ContinuationCount": continuation,
        "FailureCount": failure,
        "FlatCount": flat,
        "ContinuationRate": continuation / denominator,
        "FailureRate": failure / denominator,
        "FlatRate": flat / denominator,
    }


def event_occurrences(rows: list[Row]) -> list[tuple[str, list[int]]]:
    return [
        (name, [index for index, row in enumerate(rows) if predicate(row)])
        for name, predicate in event_sets()
    ]


def append_table(
    lines: list[str],
    title: str,
    rows: list[Row],
    terminal_indexes: list[int],
) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(
        f"{'Horizon':<8} {'Count':>8} {'MeanDRFwd':>12} {'MedianDRFwd':>14} "
        f"{'Continue':>10} {'Fail':>8} {'Flat':>8} {'ContRate':>10} {'FailRate':>10} {'FlatRate':>10}"
    )
    for horizon in HORIZONS:
        stats = summarize(direction_relative_returns(rows, terminal_indexes, horizon))
        lines.append(
            f"DRFwd{horizon:<2} {stats['Count']:>8} {stats['MeanDRFwd']:>12.4f} "
            f"{stats['MedianDRFwd']:>14.4f} {stats['ContinuationCount']:>10} "
            f"{stats['FailureCount']:>8} {stats['FlatCount']:>8} "
            f"{stats['ContinuationRate']:>9.2%} {stats['FailureRate']:>9.2%} "
            f"{stats['FlatRate']:>9.2%}"
        )


def sequence_fwd5_stats(
    rows: list[Row],
    occurrences: dict[SequenceKey, list[int]],
) -> list[tuple[SequenceKey, dict[str, float | int]]]:
    return [
        (sequence, summarize(direction_relative_returns(rows, indexes, 5)))
        for sequence, indexes in occurrences.items()
    ]


def append_ranked_table(
    lines: list[str],
    title: str,
    ranked: list[tuple[SequenceKey, dict[str, float | int]]],
) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(
        f"{'Rank':>4} {'Sequence':<116} {'Count':>8} {'MeanDRFwd5':>12} "
        f"{'Median':>12} {'ContRate':>10} {'FailRate':>10}"
    )
    for rank, (sequence, stats) in enumerate(ranked[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {display_sequence(sequence):<116} {stats['Count']:>8} "
            f"{stats['MeanDRFwd']:>12.4f} {stats['MedianDRFwd']:>12.4f} "
            f"{stats['ContinuationRate']:>9.2%} {stats['FailureRate']:>9.2%}"
        )


def append_research_notes(
    lines: list[str],
    ranked_stats: list[tuple[SequenceKey, dict[str, float | int]]],
    skipped_count: int,
) -> None:
    lines.extend(["\nResearch Notes", "=============="])
    if not ranked_stats:
        lines.append("- No sequences had at least 30 available DRFwd5 samples.")
        lines.append(f"- Sequences with Count < 30 skipped from ranked tables: {skipped_count}.")
        return

    highest_mean = max(
        ranked_stats, key=lambda item: (item[1]["MeanDRFwd"], display_sequence(item[0]))
    )
    lowest_mean = min(
        ranked_stats, key=lambda item: (item[1]["MeanDRFwd"], display_sequence(item[0]))
    )
    highest_continuation = max(
        ranked_stats,
        key=lambda item: (item[1]["ContinuationRate"], display_sequence(item[0])),
    )
    highest_failure = max(
        ranked_stats, key=lambda item: (item[1]["FailureRate"], display_sequence(item[0]))
    )

    lines.append(
        f"- Highest MeanDRFwd5: {display_sequence(highest_mean[0])} "
        f"({highest_mean[1]['MeanDRFwd']:.4f}, n={highest_mean[1]['Count']})."
    )
    lines.append(
        f"- Lowest MeanDRFwd5: {display_sequence(lowest_mean[0])} "
        f"({lowest_mean[1]['MeanDRFwd']:.4f}, n={lowest_mean[1]['Count']})."
    )
    lines.append(
        f"- Highest Fwd5 continuation rate: {display_sequence(highest_continuation[0])} "
        f"({highest_continuation[1]['ContinuationRate']:.2%}, "
        f"mean={highest_continuation[1]['MeanDRFwd']:.4f}, "
        f"n={highest_continuation[1]['Count']})."
    )
    lines.append(
        f"- Highest Fwd5 failure rate: {display_sequence(highest_failure[0])} "
        f"({highest_failure[1]['FailureRate']:.2%}, "
        f"mean={highest_failure[1]['MeanDRFwd']:.4f}, "
        f"n={highest_failure[1]['Count']})."
    )
    lines.append(f"- Sequences with Count < 30 skipped from ranked tables: {skipped_count}.")

    highest_mean_name = display_sequence(highest_mean[0])
    highest_rate_name = display_sequence(highest_continuation[0])
    if highest_mean_name != highest_rate_name:
        lines.append(
            "- Expectancy/rate conflict: the highest MeanDRFwd5 sequence differs from "
            "the highest Fwd5 continuation-rate sequence."
        )
    else:
        lines.append(
            "- No expectancy/rate conflict at the top rank: the highest MeanDRFwd5 "
            "sequence also has the highest Fwd5 continuation rate."
        )


def build_report(path: Path) -> str:
    rows = load_rows(path)
    valid_polarity_rows = sum(row["VolumePolarity"] in {"Black", "Red"} for row in rows)
    sequence_map = sequence_occurrences(label_rows(rows), list(DEFAULT_LENGTHS))
    sequence_stats = sequence_fwd5_stats(rows, sequence_map)
    ranked_stats = [item for item in sequence_stats if item[1]["Count"] >= MIN_RANKED_SAMPLES]
    skipped_count = sum(item[1]["Count"] < MIN_RANKED_SAMPLES for item in sequence_stats)

    lines = [
        "APVA Evidence Direction-Relative Study v0.1",
        "===========================================",
        f"Input: {path}",
        f"Total rows: {len(rows)}",
        f"Total valid polarity rows: {valid_polarity_rows}",
        "Terminal polarity: Black=up continuation, Red=down continuation",
        "Ordered-sequence consequence origin: final bar of each completed sequence",
        f"Ranked-sequence minimum DRFwd5 samples: {MIN_RANKED_SAMPLES}",
    ]

    lines.extend(["\nSingle-Bar Direction-Relative Consequences", "=========================================="])
    for name, indexes in event_occurrences(rows):
        append_table(lines, f"Event: {name}", rows, indexes)

    lines.extend(["\nOrdered Sequence Direction-Relative Consequences", "================================================"])
    lines.append(f"Observed length-2 and length-3 sequences: {len(sequence_map)}")
    for sequence, indexes in sorted(
        sequence_map.items(), key=lambda item: (len(item[0]), display_sequence(item[0]))
    ):
        append_table(lines, f"Sequence: {display_sequence(sequence)}", rows, indexes)

    ranked_by_mean_desc = sorted(
        ranked_stats,
        key=lambda item: (-item[1]["MeanDRFwd"], display_sequence(item[0])),
    )
    ranked_by_mean_asc = sorted(
        ranked_stats,
        key=lambda item: (item[1]["MeanDRFwd"], display_sequence(item[0])),
    )
    ranked_by_continuation = sorted(
        ranked_stats,
        key=lambda item: (-item[1]["ContinuationRate"], display_sequence(item[0])),
    )
    ranked_by_failure = sorted(
        ranked_stats,
        key=lambda item: (-item[1]["FailureRate"], display_sequence(item[0])),
    )

    append_ranked_table(lines, "Top 25 Continuation Expectancy Sequences", ranked_by_mean_desc)
    append_ranked_table(lines, "Top 25 Failure Expectancy Sequences", ranked_by_mean_asc)
    append_ranked_table(lines, "Top 25 Continuation Rate Sequences", ranked_by_continuation)
    append_ranked_table(lines, "Top 25 Failure Rate Sequences", ranked_by_failure)
    append_research_notes(lines, ranked_stats, skipped_count)

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report = build_report(args.input)
    try:
        args.output.write_text(report, encoding="utf-8")
    except PermissionError:
        # PowerShell may already hold DirectionRelative.txt open when stdout is
        # redirected to that same path. Printing below still writes the report.
        pass
    print(report, end="")


if __name__ == "__main__":
    main()
