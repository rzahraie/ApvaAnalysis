#!/usr/bin/env python3
"""Print descriptive consequences for ordered APVA Evidence v0.1 sequences.

Sequences are adjacent evidence-event labels observed in bar order. Consequences
are measured from the final bar of each completed sequence. This research-only
study does not infer higher-layer structure or create trading signals.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from io import StringIO
from itertools import product
from pathlib import Path

from APVA_Evidence_Report_IO import add_input_arguments, report_path, resolve_input, write_report
from APVA_Evidence_Consequences_02 import (
    DEFAULT_INPUT,
    HORIZONS,
    POLARITIES,
    event_sets,
    load_rows,
    summarize,
)

DEFAULT_LENGTHS = (2, 3)
DEFAULT_MIN_SAMPLES = 30

Row = dict[str, object]
SequenceKey = tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print APVA Evidence v0.1 ordered-sequence consequence tables."
    )
    add_input_arguments(parser, DEFAULT_INPUT)
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_LENGTHS),
        help="Adjacent sequence lengths to study (default: 2 3)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="Minimum available Fwd5 samples required for a sequence report (default: 30)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.lengths or any(length < 2 for length in args.lengths):
        raise ValueError("Sequence lengths must be integers greater than or equal to 2.")
    if args.min_samples < 1:
        raise ValueError("--min-samples must be greater than or equal to 1.")


def label_rows(rows: list[Row]) -> list[tuple[str, ...]]:
    predicates = [(name, predicate) for name, predicate in event_sets() if name != "AllBars"]
    labels: list[tuple[str, ...]] = []
    for row in rows:
        labels.append(tuple(name for name, predicate in predicates if predicate(row)))
    return labels


def sequence_occurrences(
    labels_by_row: list[tuple[str, ...]], lengths: list[int]
) -> dict[SequenceKey, list[int]]:
    occurrences: dict[SequenceKey, list[int]] = defaultdict(list)
    for length in sorted(set(lengths)):
        for end_index in range(length - 1, len(labels_by_row)):
            label_window = labels_by_row[end_index - length + 1 : end_index + 1]
            if any(not labels for labels in label_window):
                continue
            for sequence in product(*label_window):
                occurrences[sequence].append(end_index)
    return dict(occurrences)


def baseline_returns(rows: list[Row], horizon: int, polarity: str | None = None) -> list[float]:
    values: list[float] = []
    for index in range(len(rows) - horizon):
        row = rows[index]
        if polarity is None or row["VolumePolarity"] == polarity:
            values.append(float(rows[index + horizon]["Close"]) - float(row["Close"]))
    return values


def sequence_returns(
    rows: list[Row],
    end_indexes: list[int],
    horizon: int,
    polarity: str | None = None,
) -> list[float]:
    values: list[float] = []
    for end_index in end_indexes:
        if end_index + horizon >= len(rows):
            continue
        row = rows[end_index]
        if polarity is not None and row["VolumePolarity"] != polarity:
            continue
        values.append(float(rows[end_index + horizon]["Close"]) - float(row["Close"]))
    return values


def display_sequence(sequence: SequenceKey) -> str:
    return " -> ".join(sequence)


def print_consequence_table(title: str, values_for_horizon) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(
        f"{'Horizon':<8} {'Count':>8} {'Mean':>12} {'Median':>12} "
        f"{'Pos':>8} {'Neg':>8} {'Flat':>8} {'PosRate':>10} {'NegRate':>10} {'FlatRate':>10}"
    )
    for horizon in HORIZONS:
        stats = summarize(values_for_horizon(horizon))
        print(
            f"Fwd{horizon:<4} {stats['Count']:>8} {stats['Mean']:>12.4f} "
            f"{stats['Median']:>12.4f} {stats['Positive']:>8} {stats['Negative']:>8} "
            f"{stats['Flat']:>8} {stats['PositiveRate']:>9.2%} "
            f"{stats['NegativeRate']:>9.2%} {stats['FlatRate']:>9.2%}"
        )


def print_directional_table(title: str, values_for_horizon) -> None:
    print(f"\nDirectional Conditioning: {title}")
    print("-" * (26 + len(title)))
    print(
        f"{'Polarity':<10} {'Horizon':<8} {'Count':>8} {'Mean':>12} "
        f"{'Median':>12} {'PosRate':>10} {'NegRate':>10}"
    )
    for polarity in POLARITIES:
        for horizon in HORIZONS:
            stats = summarize(values_for_horizon(horizon, polarity))
            print(
                f"{polarity:<10} Fwd{horizon:<4} {stats['Count']:>8} "
                f"{stats['Mean']:>12.4f} {stats['Median']:>12.4f} "
                f"{stats['PositiveRate']:>9.2%} {stats['NegativeRate']:>9.2%}"
            )


def select_sequences(
    rows: list[Row],
    occurrences: dict[SequenceKey, list[int]],
    min_samples: int,
) -> list[tuple[SequenceKey, list[int]]]:
    selected: list[tuple[SequenceKey, list[int]]] = []
    for sequence, end_indexes in occurrences.items():
        sample_count = len(sequence_returns(rows, end_indexes, 5))
        if sample_count >= min_samples:
            selected.append((sequence, end_indexes))
    return sorted(selected, key=lambda item: (len(item[0]), display_sequence(item[0])))


def print_sequence_inventory(
    rows: list[Row],
    occurrences: dict[SequenceKey, list[int]],
    selected: list[tuple[SequenceKey, list[int]]],
    min_samples: int,
) -> None:
    by_length = Counter(len(sequence) for sequence in occurrences)
    selected_by_length = Counter(len(sequence) for sequence, _ in selected)
    print("\nSequence Inventory")
    print("==================")
    print(f"Minimum Fwd5 samples for detailed report: {min_samples}")
    for length in sorted(by_length):
        print(
            f"Length {length}: observed={by_length[length]}, "
            f"reported={selected_by_length[length]}"
        )

    print("\nReported Sequence Frequencies")
    print("-----------------------------")
    print(f"{'Sequence':<92} {'Occurrences':>12} {'Fwd5 Samples':>14}")
    for sequence, end_indexes in selected:
        fwd5_count = len(sequence_returns(rows, end_indexes, 5))
        print(f"{display_sequence(sequence):<92} {len(end_indexes):>12} {fwd5_count:>14}")


def print_research_notes(
    rows: list[Row],
    occurrences: dict[SequenceKey, list[int]],
    selected: list[tuple[SequenceKey, list[int]]],
    min_samples: int,
) -> None:
    fwd5 = [
        (sequence, summarize(sequence_returns(rows, end_indexes, 5)))
        for sequence, end_indexes in selected
    ]
    print("\nResearch Notes")
    print("--------------")
    if not fwd5:
        print(f"- No ordered sequences had at least {min_samples} available Fwd5 samples.")
        return

    max_mean = max(fwd5, key=lambda item: (item[1]["Mean"], display_sequence(item[0])))
    min_mean = min(fwd5, key=lambda item: (item[1]["Mean"], display_sequence(item[0])))
    max_positive_rate = max(
        fwd5, key=lambda item: (item[1]["PositiveRate"], display_sequence(item[0]))
    )
    max_negative_rate = max(
        fwd5, key=lambda item: (item[1]["NegativeRate"], display_sequence(item[0]))
    )
    low_confidence = sorted(
        display_sequence(sequence)
        for sequence, end_indexes in occurrences.items()
        if 0 < len(sequence_returns(rows, end_indexes, 5)) < 30
    )

    print(
        f"- Largest positive Fwd5 mean among reported sequences: {display_sequence(max_mean[0])} "
        f"({max_mean[1]['Mean']:.4f}, n={max_mean[1]['Count']})."
    )
    print(
        f"- Largest negative Fwd5 mean among reported sequences: {display_sequence(min_mean[0])} "
        f"({min_mean[1]['Mean']:.4f}, n={min_mean[1]['Count']})."
    )
    print(
        f"- Highest positive Fwd5 rate among reported sequences: "
        f"{display_sequence(max_positive_rate[0])} "
        f"({max_positive_rate[1]['PositiveRate']:.2%}, n={max_positive_rate[1]['Count']})."
    )
    print(
        f"- Highest negative Fwd5 rate among reported sequences: "
        f"{display_sequence(max_negative_rate[0])} "
        f"({max_negative_rate[1]['NegativeRate']:.2%}, n={max_negative_rate[1]['Count']})."
    )
    print(f"- Ordered sequences with fewer than 30 Fwd5 samples: {len(low_confidence)}.")


def analyze(path: Path, lengths: list[int], min_samples: int) -> None:
    rows = load_rows(path)
    labels_by_row = label_rows(rows)
    occurrences = sequence_occurrences(labels_by_row, lengths)
    selected = select_sequences(rows, occurrences, min_samples)

    print("APVA Evidence Ordered Sequences Study v0.1")
    print("==========================================")
    print(f"Input: {path}")
    print(f"Total rows: {len(rows)}")
    print("Sequence consequence origin: final bar of each completed sequence")

    print_sequence_inventory(rows, occurrences, selected, min_samples)

    print("\nBaseline Consequences")
    print("=====================")
    print_consequence_table("Sequence: AllBars", lambda horizon: baseline_returns(rows, horizon))
    print_directional_table(
        "Sequence: AllBars",
        lambda horizon, polarity: baseline_returns(rows, horizon, polarity),
    )

    print("\nOrdered Sequence Consequences")
    print("=============================")
    for sequence, end_indexes in selected:
        title = "Sequence: " + display_sequence(sequence)
        print_consequence_table(
            title,
            lambda horizon, indexes=end_indexes: sequence_returns(rows, indexes, horizon),
        )
        print_directional_table(
            title,
            lambda horizon, polarity, indexes=end_indexes: sequence_returns(
                rows, indexes, horizon, polarity
            ),
        )

    print_research_notes(rows, occurrences, selected, min_samples)


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_path = resolve_input(args, DEFAULT_INPUT)
    buffer = StringIO()
    with redirect_stdout(buffer):
        analyze(input_path, args.lengths, args.min_samples)
    write_report(buffer.getvalue(), report_path("Sequences", input_path))


if __name__ == "__main__":
    main()
