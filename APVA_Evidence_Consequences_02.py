#!/usr/bin/env python3
"""Print descriptive forward consequences for APVA Evidence v0.1 events.

This study measures raw close-to-forward-close differences after evidence-layer
conditions. It does not infer higher-layer structure or create trading signals.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Callable, Iterable

from APVA_Evidence_Report_IO import add_input_arguments, report_path, resolve_input, write_report

DEFAULT_INPUT = Path("Evidence/NQ_5 Minute_apva_bar_evidence_v01.csv")
HORIZONS = (1, 3, 5, 10)
POLARITIES = ("Black", "Red")
REQUIRED_COLUMNS = {
    "Close",
    "VolumePolarity",
    "ParticipationState",
    "DissipationState",
    "AcceptanceState",
    "CompressionState",
    "ExpansionState",
    "SignificanceState",
    "EvidenceFlags",
}

Row = dict[str, object]
EventPredicate = Callable[[Row], bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print APVA Evidence v0.1 forward consequence tables."
    )
    add_input_arguments(parser, DEFAULT_INPUT)
    return parser.parse_args()


def require_columns(fieldnames: Iterable[str] | None) -> None:
    available = set(fieldnames or [])
    missing = sorted(REQUIRED_COLUMNS - available)
    if missing:
        raise ValueError("Missing required evidence columns: " + ", ".join(missing))


def parse_flags(value: str | None) -> set[str]:
    return {flag.strip() for flag in (value or "").split(";") if flag.strip()}


def load_rows(path: Path) -> list[Row]:
    if not path.exists():
        raise FileNotFoundError(f"Missing evidence CSV: {path}")

    rows: list[Row] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames)
        raw_rows = list(reader)

        for row_index, raw in enumerate(raw_rows):
            line_number = row_index + 2
            if not any((value or "").strip() for value in raw.values()):
                continue

            close_text = (raw.get("Close") or "").strip()
            if not close_text and row_index == len(raw_rows) - 1:
                continue

            try:
                close = float(close_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid Close value at CSV line {line_number}: {close_text!r}"
                ) from error

            rows.append(
                {
                    "Close": close,
                    "VolumePolarity": (raw.get("VolumePolarity") or "").strip(),
                    "ParticipationState": (raw.get("ParticipationState") or "").strip(),
                    "DissipationState": (raw.get("DissipationState") or "").strip(),
                    "AcceptanceState": (raw.get("AcceptanceState") or "").strip(),
                    "CompressionState": (raw.get("CompressionState") or "").strip(),
                    "ExpansionState": (raw.get("ExpansionState") or "").strip(),
                    "SignificanceState": (raw.get("SignificanceState") or "").strip(),
                    "EvidenceFlags": parse_flags(raw.get("EvidenceFlags")),
                }
            )

    return rows


def flag_is(name: str) -> EventPredicate:
    return lambda row: name in row["EvidenceFlags"]


def state_is(column: str, value: str) -> EventPredicate:
    return lambda row: row[column] == value


def all_of(*predicates: EventPredicate) -> EventPredicate:
    return lambda row: all(predicate(row) for predicate in predicates)


def event_sets() -> list[tuple[str, EventPredicate]]:
    dissipation_any = lambda row: row["DissipationState"] != "Absent"
    return [
        ("AllBars", lambda row: True),
        ("DissipationAny", dissipation_any),
        (
            "DissipationContained",
            all_of(dissipation_any, state_is("AcceptanceState", "Contained")),
        ),
        (
            "DissipationAccepted",
            all_of(dissipation_any, state_is("AcceptanceState", "Accepted")),
        ),
        ("ParticipationPeak", state_is("ParticipationState", "Peak")),
        ("ParticipationClimactic", state_is("ParticipationState", "Climactic")),
        (
            "PeakContained",
            all_of(
                state_is("ParticipationState", "Peak"),
                state_is("AcceptanceState", "Contained"),
            ),
        ),
        (
            "PeakAccepted",
            all_of(
                state_is("ParticipationState", "Peak"),
                state_is("AcceptanceState", "Accepted"),
            ),
        ),
        (
            "ClimacticContained",
            all_of(
                state_is("ParticipationState", "Climactic"),
                state_is("AcceptanceState", "Contained"),
            ),
        ),
        (
            "ClimacticAccepted",
            all_of(
                state_is("ParticipationState", "Climactic"),
                state_is("AcceptanceState", "Accepted"),
            ),
        ),
        ("HighVolumeLowRange", flag_is("HighVolumeLowRange")),
        ("LowVolumeHighRange", flag_is("LowVolumeHighRange")),
    ]


def forward_returns(
    rows: list[Row],
    predicate: EventPredicate,
    horizon: int,
    polarity: str | None = None,
) -> list[float]:
    values: list[float] = []
    limit = len(rows) - horizon
    for index in range(limit):
        row = rows[index]
        if polarity is not None and row["VolumePolarity"] != polarity:
            continue
        if predicate(row):
            values.append(float(rows[index + horizon]["Close"]) - float(row["Close"]))
    return values


def summarize(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    flat = count - positive - negative
    denominator = count if count > 0 else 1
    return {
        "Count": count,
        "Mean": statistics.fmean(values) if values else 0.0,
        "Median": statistics.median(values) if values else 0.0,
        "Positive": positive,
        "Negative": negative,
        "Flat": flat,
        "PositiveRate": positive / denominator,
        "NegativeRate": negative / denominator,
        "FlatRate": flat / denominator,
    }


def print_event_table(name: str, rows: list[Row], predicate: EventPredicate) -> None:
    print(f"\nEvent Set: {name}")
    print("-" * (11 + len(name)))
    print(
        f"{'Horizon':<8} {'Count':>8} {'Mean':>12} {'Median':>12} "
        f"{'Pos':>8} {'Neg':>8} {'Flat':>8} {'PosRate':>10} {'NegRate':>10} {'FlatRate':>10}"
    )
    for horizon in HORIZONS:
        stats = summarize(forward_returns(rows, predicate, horizon))
        print(
            f"Fwd{horizon:<4} {stats['Count']:>8} {stats['Mean']:>12.4f} "
            f"{stats['Median']:>12.4f} {stats['Positive']:>8} {stats['Negative']:>8} "
            f"{stats['Flat']:>8} {stats['PositiveRate']:>9.2%} "
            f"{stats['NegativeRate']:>9.2%} {stats['FlatRate']:>9.2%}"
        )


def print_directional_table(name: str, rows: list[Row], predicate: EventPredicate) -> None:
    print(f"\nDirectional Conditioning: {name}")
    print("-" * (26 + len(name)))
    print(
        f"{'Polarity':<10} {'Horizon':<8} {'Count':>8} {'Mean':>12} "
        f"{'Median':>12} {'PosRate':>10} {'NegRate':>10}"
    )
    for polarity in POLARITIES:
        for horizon in HORIZONS:
            stats = summarize(forward_returns(rows, predicate, horizon, polarity))
            print(
                f"{polarity:<10} Fwd{horizon:<4} {stats['Count']:>8} "
                f"{stats['Mean']:>12.4f} {stats['Median']:>12.4f} "
                f"{stats['PositiveRate']:>9.2%} {stats['NegativeRate']:>9.2%}"
            )


def print_research_notes(rows: list[Row], sets: list[tuple[str, EventPredicate]]) -> None:
    fwd5 = [(name, summarize(forward_returns(rows, predicate, 5))) for name, predicate in sets]
    populated = [(name, stats) for name, stats in fwd5 if stats["Count"] > 0]

    print("\nResearch Notes")
    print("--------------")
    if not populated:
        print("- No event sets had available Fwd5 samples.")
        return

    max_mean = max(populated, key=lambda item: (item[1]["Mean"], item[0]))
    min_mean = min(populated, key=lambda item: (item[1]["Mean"], item[0]))
    max_positive_rate = max(
        populated, key=lambda item: (item[1]["PositiveRate"], item[0])
    )
    max_negative_rate = max(
        populated, key=lambda item: (item[1]["NegativeRate"], item[0])
    )
    low_confidence = [name for name, stats in fwd5 if stats["Count"] < 30]

    print(
        f"- Largest positive Fwd5 mean: {max_mean[0]} "
        f"({max_mean[1]['Mean']:.4f}, n={max_mean[1]['Count']})."
    )
    print(
        f"- Largest negative Fwd5 mean: {min_mean[0]} "
        f"({min_mean[1]['Mean']:.4f}, n={min_mean[1]['Count']})."
    )
    print(
        f"- Highest positive Fwd5 rate: {max_positive_rate[0]} "
        f"({max_positive_rate[1]['PositiveRate']:.2%}, n={max_positive_rate[1]['Count']})."
    )
    print(
        f"- Highest negative Fwd5 rate: {max_negative_rate[0]} "
        f"({max_negative_rate[1]['NegativeRate']:.2%}, n={max_negative_rate[1]['Count']})."
    )
    if low_confidence:
        print("- Fwd5 event sets with fewer than 30 samples: " + ", ".join(low_confidence) + ".")
    else:
        print("- No Fwd5 event sets had fewer than 30 samples.")


def analyze(path: Path) -> None:
    rows = load_rows(path)
    sets = event_sets()

    print("APVA Evidence Consequences Study v0.1")
    print("====================================")
    print(f"Input: {path}")
    print(f"Total rows: {len(rows)}")

    print("\nForward Return Tables")
    print("=====================")
    for name, predicate in sets:
        print_event_table(name, rows, predicate)

    print("\nDirectional Conditioning")
    print("========================")
    for name, predicate in sets:
        print_directional_table(name, rows, predicate)

    print_research_notes(rows, sets)


def main() -> None:
    args = parse_args()
    input_path = resolve_input(args, DEFAULT_INPUT)
    buffer = StringIO()
    with redirect_stdout(buffer):
        analyze(input_path)
    write_report(buffer.getvalue(), report_path("Consequences", input_path))


if __name__ == "__main__":
    main()
