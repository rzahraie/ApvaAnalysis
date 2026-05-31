#!/usr/bin/env python3
"""Print APVA Evidence v0.1 one-step and two-step transition statistics.

This research-only study measures evidence-state transitions and persistence. It
does not infer higher-layer structure or create trading signals.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

DEFAULT_INPUT = Path("Evidence/NQ_5 Minute_apva_bar_evidence_v01.csv")
DEFAULT_OUTPUT = Path("Transitions.txt")
TOP_LIMIT = 25

EVIDENCE_COLUMNS = [
    "ParticipationState",
    "AcceptanceState",
    "DissipationState",
    "CompressionState",
    "ExpansionState",
    "SignificanceState",
    "Geometry",
    "VolumePolarity",
]

FOCUSED_TRANSITIONS = {
    "ParticipationState": [
        ("Peak", "Peak"),
        ("Peak", "Falling"),
        ("Peak", "Rising"),
        ("Peak", "Climactic"),
        ("Climactic", "Peak"),
        ("Climactic", "Falling"),
        ("Climactic", "Rising"),
    ],
    "AcceptanceState": [
        ("Contained", "Contained"),
        ("Contained", "Accepted"),
        ("Accepted", "Contained"),
        ("Accepted", "Accepted"),
        ("Unresolved", "Accepted"),
        ("Unresolved", "Contained"),
    ],
    "DissipationState": [
        ("Local", "Local"),
        ("Local", "Absent"),
        ("Absent", "Local"),
        ("Absent", "Absent"),
    ],
    "CompressionState": [
        ("Clustered", "Clustered"),
        ("Clustered", "Local"),
        ("Clustered", "Absent"),
        ("Local", "Clustered"),
        ("Local", "Absent"),
    ],
}

Transition = tuple[str, str]
Path3 = tuple[str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print APVA Evidence v0.1 transition and persistence tables."
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


def require_columns(fieldnames: Iterable[str] | None) -> None:
    available = set(fieldnames or [])
    missing = sorted(set(EVIDENCE_COLUMNS) - available)
    if missing:
        raise ValueError("Missing required evidence columns: " + ", ".join(missing))


def clean(value: str | None) -> str:
    text = (value or "").strip()
    return text if text else "(empty)"


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing evidence CSV: {path}")

    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames)
        raw_rows = list(reader)

        for row_index, raw in enumerate(raw_rows):
            if not any((value or "").strip() for value in raw.values()):
                continue

            incomplete = any(not (raw.get(column) or "").strip() for column in EVIDENCE_COLUMNS)
            if incomplete and row_index == len(raw_rows) - 1:
                continue
            if incomplete:
                raise ValueError(f"Incomplete evidence row at CSV line {row_index + 2}.")

            rows.append({column: clean(raw.get(column)) for column in EVIDENCE_COLUMNS})

    return rows


def count_one_step(states: list[str]) -> Counter[Transition]:
    return Counter(zip(states, states[1:]))


def count_two_step(states: list[str]) -> Counter[Path3]:
    return Counter(zip(states, states[1:], states[2:]))


def count_runs(states: list[str]) -> dict[str, list[int]]:
    runs: dict[str, list[int]] = defaultdict(list)
    if not states:
        return dict(runs)

    current = states[0]
    length = 1
    for state in states[1:]:
        if state == current:
            length += 1
        else:
            runs[current].append(length)
            current = state
            length = 1
    runs[current].append(length)
    return dict(runs)


def top_items(counter: Counter, limit: int = TOP_LIMIT):
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]


def append_one_step_section(lines: list[str], column: str, states: list[str]) -> None:
    counts = count_one_step(states)
    from_counts = Counter(current for current, _ in zip(states, states[1:]))
    total = sum(counts.values())

    lines.extend([f"\n{column}: Top {TOP_LIMIT} One-Step Transitions", "-" * (len(column) + 29)])
    lines.append(
        f"{'Transition':<56} {'Count':>10} {'PercentAll':>12} {'PercentFromCurrent':>20}"
    )
    for (current, next_state), count in top_items(counts):
        pct_all = 100.0 * count / total if total else 0.0
        pct_from = 100.0 * count / from_counts[current] if from_counts[current] else 0.0
        lines.append(
            f"{current + ' -> ' + next_state:<56} {count:>10} {pct_all:>11.2f}% {pct_from:>19.2f}%"
        )


def append_two_step_section(lines: list[str], column: str, states: list[str]) -> None:
    counts = count_two_step(states)
    prefix_counts = Counter(zip(states, states[1:]))
    total = sum(counts.values())

    lines.extend([f"\n{column}: Top {TOP_LIMIT} Two-Step Paths", "-" * (len(column) + 23)])
    lines.append(
        f"{'Path':<74} {'Count':>10} {'PercentAll':>12} {'PercentFromPrefix':>19}"
    )
    for (state_a, state_b, state_c), count in top_items(counts):
        pct_all = 100.0 * count / total if total else 0.0
        prefix_count = prefix_counts[(state_a, state_b)]
        pct_from = 100.0 * count / prefix_count if prefix_count else 0.0
        lines.append(
            f"{state_a + ' -> ' + state_b + ' -> ' + state_c:<74} "
            f"{count:>10} {pct_all:>11.2f}% {pct_from:>18.2f}%"
        )


def append_persistence_section(lines: list[str], column: str, states: list[str]) -> None:
    runs = count_runs(states)
    lines.extend([f"\n{column}: Persistence", "-" * (len(column) + 13)])
    lines.append(
        f"{'State':<28} {'RunCount':>10} {'MeanRun':>12} {'MaxRun':>10} "
        f"{'Length1':>10} {'Length2':>10} {'Length3Plus':>13}"
    )
    for state in sorted(runs):
        lengths = runs[state]
        lines.append(
            f"{state:<28} {len(lengths):>10} {statistics.fmean(lengths):>12.2f} "
            f"{max(lengths):>10} {sum(length == 1 for length in lengths):>10} "
            f"{sum(length == 2 for length in lengths):>10} "
            f"{sum(length >= 3 for length in lengths):>13}"
        )


def append_focused_section(lines: list[str], rows: list[dict[str, str]]) -> None:
    lines.extend(["\nFocused Transition Probabilities", "================================"])
    for column, questions in FOCUSED_TRANSITIONS.items():
        states = [row[column] for row in rows]
        counts = count_one_step(states)
        from_counts = Counter(current for current, _ in zip(states, states[1:]))
        lines.extend([f"\n{column}", "-" * len(column)])
        lines.append(f"{'Transition':<48} {'Count':>10} {'FromState':>12} {'Probability':>13}")
        for current, next_state in questions:
            count = counts[(current, next_state)]
            from_count = from_counts[current]
            probability = 100.0 * count / from_count if from_count else 0.0
            lines.append(
                f"{current + ' -> ' + next_state:<48} {count:>10} "
                f"{from_count:>12} {probability:>12.2f}%"
            )


def append_research_notes(lines: list[str], rows: list[dict[str, str]]) -> None:
    lines.extend(["\nResearch Notes", "=============="])

    persistence_rows: list[tuple[float, str, str, int]] = []
    for column in EVIDENCE_COLUMNS:
        states = [row[column] for row in rows]
        runs = count_runs(states)
        for state, lengths in runs.items():
            persistence_rows.append((statistics.fmean(lengths), column, state, max(lengths)))

    if persistence_rows:
        mean_run, column, state, max_run = max(
            persistence_rows, key=lambda item: (item[0], item[1], item[2])
        )
        lines.append(
            f"- Highest persistence state by mean run length: {column}={state} "
            f"(mean={mean_run:.2f}, max={max_run})."
        )

    for column in EVIDENCE_COLUMNS:
        states = [row[column] for row in rows]
        one_step = count_one_step(states)
        two_step = count_two_step(states)
        if one_step:
            transition, count = top_items(one_step, 1)[0]
            lines.append(
                f"- Most common one-step transition for {column}: "
                f"{' -> '.join(transition)} (count={count})."
            )
        if two_step:
            path, count = top_items(two_step, 1)[0]
            lines.append(
                f"- Most common two-step path for {column}: "
                f"{' -> '.join(path)} (count={count})."
            )

    long_runs = sorted(
        (column, state, max_run)
        for _, column, state, max_run in persistence_rows
        if max_run >= 5
    )
    if long_runs:
        lines.append("- States with max run length >= 5:")
        for column, state, max_run in long_runs:
            lines.append(f"  {column}={state}: max={max_run}")
    else:
        lines.append("- No states had max run length >= 5.")

    lines.append("- Focused transition probabilities are listed in the section above.")


def build_report(path: Path) -> str:
    rows = load_rows(path)
    lines = [
        "APVA Evidence Transitions Study v0.1",
        "====================================",
        f"Input: {path}",
        f"Complete rows: {len(rows)}",
    ]

    for column in EVIDENCE_COLUMNS:
        states = [row[column] for row in rows]
        lines.extend([f"\n{column}", "=" * len(column)])
        append_one_step_section(lines, column, states)
        append_two_step_section(lines, column, states)
        append_persistence_section(lines, column, states)

    append_focused_section(lines, rows)
    append_research_notes(lines, rows)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report = build_report(args.input)
    try:
        args.output.write_text(report, encoding="utf-8")
    except PermissionError:
        # PowerShell may already hold Transitions.txt open when stdout is
        # redirected to that same path. Printing below still writes the report.
        pass
    print(report, end="")


if __name__ == "__main__":
    main()
