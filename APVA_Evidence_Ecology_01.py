#!/usr/bin/env python3
"""Print descriptive APVA Evidence v0.1 ecology tables.

This script reports evidence-layer frequencies and co-occurrences only. It does
not infer higher-layer structure, predictions, or trading decisions.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

DEFAULT_INPUT = Path("Evidence/NQ_5 Minute_apva_bar_evidence_v01.csv")

FREQUENCY_COLUMNS = [
    "Geometry",
    "VolumePolarity",
    "VolumeDelta",
    "RangeDelta",
    "BodyDelta",
    "ParticipationState",
    "ExpansionState",
    "CompressionState",
    "DissipationState",
    "AcceptanceState",
    "SignificanceState",
]

CROSSTABS = [
    ("DissipationState", "AcceptanceState"),
    ("ParticipationState", "AcceptanceState"),
    ("ParticipationState", "CompressionState"),
    ("ParticipationState", "ExpansionState"),
    ("SignificanceState", "AcceptanceState"),
    ("Geometry", "AcceptanceState"),
    ("Geometry", "CompressionState"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print APVA Evidence v0.1 frequency tables and crosstabs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Evidence CSV path (default: {DEFAULT_INPUT.as_posix()})",
    )
    return parser.parse_args()


def clean(value: str | None) -> str:
    text = (value or "").strip()
    return text if text else "(empty)"


def require_columns(fieldnames: Iterable[str] | None) -> None:
    available = set(fieldnames or [])
    required = set(FREQUENCY_COLUMNS) | {"EvidenceFlags"}
    for left, right in CROSSTABS:
        required.add(left)
        required.add(right)
    missing = sorted(required - available)
    if missing:
        raise ValueError("Missing required evidence columns: " + ", ".join(missing))


def print_counter(title: str, counts: Counter[str], total_rows: int) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'Value':<42} {'Count':>10} {'Percent':>10}")
    denominator = total_rows if total_rows > 0 else 1
    for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"{value:<42} {count:>10} {100.0 * count / denominator:>9.2f}%")


def print_crosstab(
    left: str,
    right: str,
    counts: Counter[tuple[str, str]],
    total_rows: int,
) -> None:
    title = f"Crosstab: {left} | {right}"
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'Pair':<52} {'Count':>10} {'Percent':>10}")
    denominator = total_rows if total_rows > 0 else 1
    for (left_value, right_value), count in sorted(
        counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
    ):
        pair = f"{left_value}|{right_value}"
        print(f"{pair:<52} {count:>10} {100.0 * count / denominator:>9.2f}%")


def analyze(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing evidence CSV: {path}")

    frequencies = {column: Counter() for column in FREQUENCY_COLUMNS}
    crosstabs = {pair: Counter() for pair in CROSSTABS}
    flag_counts: Counter[str] = Counter()
    total_rows = 0

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames)

        for row in reader:
            total_rows += 1

            for column in FREQUENCY_COLUMNS:
                frequencies[column][clean(row.get(column))] += 1

            for left, right in CROSSTABS:
                crosstabs[(left, right)][
                    (clean(row.get(left)), clean(row.get(right)))
                ] += 1

            for flag in (row.get("EvidenceFlags") or "").split(";"):
                flag = flag.strip()
                if flag:
                    flag_counts[flag] += 1

    print("APVA Evidence v0.1 Ecology")
    print("========================")
    print(f"Input: {path}")
    print(f"Total rows: {total_rows}")

    for column in FREQUENCY_COLUMNS:
        print_counter(f"Frequency: {column}", frequencies[column], total_rows)

    print_counter("Frequency: EvidenceFlags", flag_counts, total_rows)

    for left, right in CROSSTABS:
        print_crosstab(left, right, crosstabs[(left, right)], total_rows)


def main() -> None:
    args = parse_args()
    analyze(args.input)


if __name__ == "__main__":
    main()
