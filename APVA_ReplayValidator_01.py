#!/usr/bin/env python3
"""
APVA Replay Validator v0.1

Behavioral replay validation for APVA v1.0 state-machine exports.

No trading logic. No optimization. No fitting. No machine learning.
"""

from __future__ import annotations

import csv
import glob
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_INPUT_DIR = r"C:\Users\rz0\Documents\ApvaAnalysis\NT8Export"
OUTPUT_TEXT = "APVA_ReplayValidator_01.txt"
DESTINATION_CSV = "DestinationBehavior.csv"
FAILURE_SCORE_CSV = "FailureScoreBehavior.csv"
LATE_NEUTRAL_CSV = "LateNeutralBehavior.csv"

DESTINATION_FAMILIES = [
    "CompressionProcessing",
    "ReassertionProcessing",
    "RecoveryResolution",
    "MixedStructure",
    "ExhaustionPersistence",
]

FORWARD_HORIZONS = [1, 2, 3, 5, 10]


@dataclass
class Row:
    source_file: str
    row_number: int
    timestamp: str
    instrument: str
    close: Optional[float]
    structural_state: str
    loop_phase: str
    previous_loop_phase: str
    destination_family: str
    failure_warning_score: Optional[int]


@dataclass
class DestinationEvent:
    index: int
    row: Row
    forward: Dict[int, Dict[str, Optional[float]]]
    stability_bars: Optional[int]


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_float(value: object) -> Optional[float]:
    text = clean_text(value).replace(",", "")
    if not text or text.upper() in {"N/A", "NA", "NONE", "NULL"}:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def parse_int(value: object) -> Optional[int]:
    number = parse_float(value)
    if number is None:
        return None
    return int(number)


def input_files(paths: Sequence[str]) -> List[str]:
    if paths:
        candidates: List[str] = []
        for path in paths:
            if os.path.isdir(path):
                candidates.extend(glob.glob(os.path.join(path, "APVA_V10_StateMachine_*.csv")))
            else:
                candidates.extend(glob.glob(path))
        return sorted(set(candidates))

    return sorted(glob.glob(os.path.join(DEFAULT_INPUT_DIR, "APVA_V10_StateMachine_*.csv")))


def load_rows(files: Sequence[str]) -> List[Row]:
    rows: List[Row] = []
    for path in files:
        with open(path, "r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row_number, raw in enumerate(reader, start=2):
                rows.append(
                    Row(
                        source_file=path,
                        row_number=row_number,
                        timestamp=clean_text(raw.get("Timestamp")),
                        instrument=clean_text(raw.get("Instrument")),
                        close=parse_float(raw.get("Close")),
                        structural_state=clean_text(raw.get("StructuralState")),
                        loop_phase=clean_text(raw.get("LoopPhase")),
                        previous_loop_phase=clean_text(raw.get("PreviousLoopPhase")),
                        destination_family=clean_text(raw.get("DestinationFamily")),
                        failure_warning_score=parse_int(raw.get("FailureWarningScore")),
                    )
                )
    return rows


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def median(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return statistics.median(numeric)


def maximum(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return max(numeric)


def format_number(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def forward_metrics(rows: Sequence[Row], index: int, horizon: int) -> Dict[str, Optional[float]]:
    start = rows[index].close
    if start is None:
        return {"mfe": None, "mae": None, "net": None}

    forward_closes: List[float] = []
    for offset in range(1, horizon + 1):
        target_index = index + offset
        if target_index >= len(rows):
            break
        close = rows[target_index].close
        if close is not None:
            forward_closes.append(close)

    if not forward_closes:
        return {"mfe": None, "mae": None, "net": None}

    changes = [close - start for close in forward_closes]
    net = forward_closes[-1] - start
    return {
        "mfe": max(changes),
        "mae": min(changes),
        "net": net,
    }


def stability_bars(rows: Sequence[Row], index: int) -> Optional[int]:
    for offset in range(1, len(rows) - index):
        phase = rows[index + offset].loop_phase
        if phase in {"NeutralFormation", "ReturnToNeutral"}:
            return offset
    return None


def extract_destination_events(rows: Sequence[Row]) -> List[DestinationEvent]:
    events: List[DestinationEvent] = []
    for index, row in enumerate(rows):
        if row.loop_phase != "DestinationSelection":
            continue
        forward = {horizon: forward_metrics(rows, index, horizon) for horizon in FORWARD_HORIZONS}
        events.append(
            DestinationEvent(
                index=index,
                row=row,
                forward=forward,
                stability_bars=stability_bars(rows, index),
            )
        )
    return events


def group_destination_behavior(events: Sequence[DestinationEvent]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[DestinationEvent]] = defaultdict(list)
    for event in events:
        grouped[event.row.destination_family].append(event)

    rows: List[Dict[str, object]] = []
    families = DESTINATION_FAMILIES + sorted(set(grouped) - set(DESTINATION_FAMILIES))
    for family in families:
        family_events = grouped.get(family, [])
        if not family_events:
            rows.append(
                {
                    "DestinationFamily": family,
                    "Count": 0,
                    "Average3BarMove": None,
                    "Average5BarMove": None,
                    "Average10BarMove": None,
                    "AverageMFE5": None,
                    "AverageMAE5": None,
                }
            )
            continue
        rows.append(
            {
                "DestinationFamily": family,
                "Count": len(family_events),
                "Average3BarMove": mean(event.forward[3]["net"] for event in family_events),
                "Average5BarMove": mean(event.forward[5]["net"] for event in family_events),
                "Average10BarMove": mean(event.forward[10]["net"] for event in family_events),
                "AverageMFE5": mean(event.forward[5]["mfe"] for event in family_events),
                "AverageMAE5": mean(event.forward[5]["mae"] for event in family_events),
            }
        )
    return rows


def group_failure_score_behavior(events: Sequence[DestinationEvent]) -> List[Dict[str, object]]:
    grouped: Dict[int, List[DestinationEvent]] = defaultdict(list)
    for event in events:
        if event.row.failure_warning_score is not None:
            grouped[event.row.failure_warning_score].append(event)

    rows: List[Dict[str, object]] = []
    for score in range(0, 7):
        score_events = grouped.get(score, [])
        rows.append(
            {
                "FailureWarningScore": score,
                "Count": len(score_events),
                "Average1BarMove": mean(event.forward[1]["net"] for event in score_events),
                "Average2BarMove": mean(event.forward[2]["net"] for event in score_events),
                "Average3BarMove": mean(event.forward[3]["net"] for event in score_events),
                "Average5BarMove": mean(event.forward[5]["net"] for event in score_events),
                "Average10BarMove": mean(event.forward[10]["net"] for event in score_events),
                "AverageMFE5": mean(event.forward[5]["mfe"] for event in score_events),
                "AverageMAE5": mean(event.forward[5]["mae"] for event in score_events),
            }
        )
    return rows


def late_neutral_behavior(rows: Sequence[Row]) -> Tuple[List[Dict[str, object]], List[int]]:
    distances: List[int] = []
    records: List[Dict[str, object]] = []

    for index, row in enumerate(rows):
        if row.loop_phase != "LateNeutral":
            continue

        distance: Optional[int] = None
        for offset in range(1, len(rows) - index):
            if rows[index + offset].loop_phase == "DestinationSelection":
                distance = offset
                break
        if distance is not None:
            distances.append(distance)
        records.append(
            {
                "Timestamp": row.timestamp,
                "Instrument": row.instrument,
                "BarsUntilDestinationSelection": distance,
            }
        )

    return records, distances


def transition_outcomes(events: Sequence[DestinationEvent]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[DestinationEvent]] = defaultdict(list)
    for event in events:
        grouped[event.row.previous_loop_phase].append(event)

    rows: List[Dict[str, object]] = []
    for phase in ["LateNeutral", "NeutralMaturation", "NeutralFormation"]:
        phase_events = grouped.get(phase, [])
        rows.append(
            {
                "PreviousLoopPhase": phase,
                "DestinationSelectionCount": len(phase_events),
                "Average3BarMove": mean(event.forward[3]["net"] for event in phase_events),
                "Average5BarMove": mean(event.forward[5]["net"] for event in phase_events),
                "Average10BarMove": mean(event.forward[10]["net"] for event in phase_events),
                "AverageMFE5": mean(event.forward[5]["mfe"] for event in phase_events),
                "AverageMAE5": mean(event.forward[5]["mae"] for event in phase_events),
            }
        )
    return rows


def destination_stability(events: Sequence[DestinationEvent]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[DestinationEvent]] = defaultdict(list)
    for event in events:
        grouped[event.row.destination_family].append(event)

    rows: List[Dict[str, object]] = []
    families = DESTINATION_FAMILIES + sorted(set(grouped) - set(DESTINATION_FAMILIES))
    for family in families:
        bars = [event.stability_bars for event in grouped.get(family, [])]
        rows.append(
            {
                "DestinationFamily": family,
                "Count": len(grouped.get(family, [])),
                "MeanBarsToNeutralOrReturn": mean(bars),
                "MedianBarsToNeutralOrReturn": median(bars),
                "MaxBarsToNeutralOrReturn": maximum(bars),
            }
        )
    return rows


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_csv_value(value) for key, value in row.items()})


def format_csv_value(value: object) -> object:
    if isinstance(value, float):
        return format_number(value)
    if value is None:
        return "N/A"
    return value


def best_and_worst_destination(destination_rows: Sequence[Dict[str, object]]) -> Tuple[str, str]:
    candidates = [
        row for row in destination_rows
        if row.get("Count", 0) and row.get("Average5BarMove") is not None
    ]
    if not candidates:
        return "N/A", "N/A"

    largest = max(candidates, key=lambda row: abs(float(row["Average5BarMove"])))
    smallest = min(candidates, key=lambda row: abs(float(row["Average5BarMove"])))
    return str(largest["DestinationFamily"]), str(smallest["DestinationFamily"])


def score_correlation_summary(score_rows: Sequence[Dict[str, object]]) -> str:
    pairs = [
        (int(row["FailureWarningScore"]), row["Average5BarMove"])
        for row in score_rows
        if row.get("Count", 0) and row.get("Average5BarMove") is not None
    ]
    if len(pairs) < 2:
        return "Insufficient score coverage."

    low = [abs(float(move)) for score, move in pairs if score <= 2]
    high = [abs(float(move)) for score, move in pairs if score >= 4]
    low_mean = mean(low)
    high_mean = mean(high)
    if low_mean is None or high_mean is None:
        return "Insufficient low/high score comparison."
    if high_mean > low_mean:
        return "Higher warning scores show larger average absolute 5-bar movement."
    if high_mean < low_mean:
        return "Higher warning scores do not show larger average absolute 5-bar movement."
    return "Warning score movement is flat in this sample."


def write_report(
    path: str,
    files: Sequence[str],
    rows: Sequence[Row],
    events: Sequence[DestinationEvent],
    destination_rows: Sequence[Dict[str, object]],
    score_rows: Sequence[Dict[str, object]],
    late_records: Sequence[Dict[str, object]],
    late_distances: Sequence[int],
    transition_rows: Sequence[Dict[str, object]],
    stability_rows: Sequence[Dict[str, object]],
) -> None:
    phase_counts = Counter(row.loop_phase for row in rows)
    destination_counts = Counter(event.row.destination_family for event in events)
    instruments = sorted({row.instrument for row in rows if row.instrument})
    timestamps = [row.timestamp for row in rows if row.timestamp]
    largest_destination, least_destination = best_and_worst_destination(destination_rows)
    score_summary = score_correlation_summary(score_rows)

    lines: List[str] = []
    lines.append("APVA Replay Validator v0.1")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Purpose")
    lines.append("Validate APVA v1.0 destination events against subsequent close-price behavior.")
    lines.append("Research only. No trading system. No optimization. No fitting. No ML.")
    lines.append("")

    lines.append("SECTION 1 - Destination Event Extraction")
    lines.append("-" * 72)
    lines.append(f"Input files: {len(files)}")
    for file_path in files:
        lines.append(f"  {file_path}")
    lines.append(f"Rows read: {len(rows)}")
    lines.append(f"Instruments: {', '.join(instruments) if instruments else 'N/A'}")
    lines.append(f"Date range: {min(timestamps) if timestamps else 'N/A'} to {max(timestamps) if timestamps else 'N/A'}")
    lines.append(f"DestinationSelection events: {len(events)}")
    lines.append("")
    lines.append("Destination event counts")
    for family, count in destination_counts.most_common():
        lines.append(f"  {family}: {count}")
    lines.append("")

    lines.append("SECTION 2 - Forward Excursion")
    lines.append("-" * 72)
    lines.append("Metrics use Close prices only. MFE/MAE are signed close changes from the event close.")
    for horizon in FORWARD_HORIZONS:
        lines.append(
            f"Horizon {horizon}: "
            f"AvgNet={format_number(mean(event.forward[horizon]['net'] for event in events))} "
            f"AvgMFE={format_number(mean(event.forward[horizon]['mfe'] for event in events))} "
            f"AvgMAE={format_number(mean(event.forward[horizon]['mae'] for event in events))}"
        )
    lines.append("")

    lines.append("SECTION 3 - Destination Family Behavior")
    lines.append("-" * 72)
    append_table(lines, destination_rows)
    lines.append("")

    lines.append("SECTION 4 - Failure Score Behavior")
    lines.append("-" * 72)
    append_table(lines, score_rows)
    lines.append("")

    lines.append("SECTION 5 - LateNeutral Analysis")
    lines.append("-" * 72)
    lines.append(f"LateNeutral rows: {phase_counts.get('LateNeutral', 0)}")
    lines.append(f"LateNeutral rows with later DestinationSelection: {len(late_distances)}")
    lines.append(f"Mean bars until DestinationSelection: {format_number(mean(late_distances))}")
    lines.append(f"Median bars until DestinationSelection: {format_number(median(late_distances))}")
    lines.append(f"Maximum bars until DestinationSelection: {format_number(maximum(late_distances), digits=0)}")
    lines.append("")

    lines.append("SECTION 6 - Transition Analysis")
    lines.append("-" * 72)
    append_table(lines, transition_rows)
    lines.append("")

    lines.append("SECTION 7 - Destination Stability")
    lines.append("-" * 72)
    append_table(lines, stability_rows)
    lines.append("")

    lines.append("SECTION 8 - Summary")
    lines.append("-" * 72)
    lines.append(f"Largest average absolute 5-bar destination movement: {largest_destination}")
    lines.append(f"Least useful by absolute 5-bar movement: {least_destination}")
    lines.append(f"FailureWarningScore behavior: {score_summary}")
    lines.append(
        "LateNeutral matter: "
        + ("Yes, it precedes DestinationSelection in this sample." if late_distances else "Not established in this sample.")
    )
    lines.append(
        "APVA predictive appearance: behavioral differences are present if destination/score groups show distinct "
        "forward movement and stability profiles. This report does not create trading rules."
    )
    lines.append("")
    lines.append("Low-DoF / Research Audit")
    lines.append("-" * 72)
    lines.append("No optimization.")
    lines.append("No machine learning.")
    lines.append("No fitting.")
    lines.append("No trading strategy.")
    lines.append("No Sharpe ratio.")
    lines.append("No profit metrics.")
    lines.append("Close-price replay behavior only.")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def append_table(lines: List[str], rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        lines.append("N/A")
        return

    headers = list(rows[0].keys())
    widths = {header: len(header) for header in headers}
    formatted_rows: List[Dict[str, str]] = []
    for row in rows:
        formatted = {header: format_csv_value(row.get(header, "")) for header in headers}
        formatted_text = {header: str(value) for header, value in formatted.items()}
        formatted_rows.append(formatted_text)
        for header, value in formatted_text.items():
            widths[header] = max(widths[header], len(value))

    lines.append(" | ".join(header.ljust(widths[header]) for header in headers))
    lines.append("-+-".join("-" * widths[header] for header in headers))
    for row in formatted_rows:
        lines.append(" | ".join(row[header].ljust(widths[header]) for header in headers))


def main(argv: Sequence[str]) -> int:
    files = input_files(argv[1:])
    if not files:
        print(f"No APVA_V10_StateMachine_*.csv files found in {DEFAULT_INPUT_DIR}", file=sys.stderr)
        return 1

    rows = load_rows(files)
    events = extract_destination_events(rows)
    destination_rows = group_destination_behavior(events)
    score_rows = group_failure_score_behavior(events)
    late_records, late_distances = late_neutral_behavior(rows)
    transition_rows = transition_outcomes(events)
    stability_rows = destination_stability(events)

    output_dir = os.path.dirname(files[0]) if files else DEFAULT_INPUT_DIR
    destination_path = os.path.join(output_dir, DESTINATION_CSV)
    score_path = os.path.join(output_dir, FAILURE_SCORE_CSV)
    late_path = os.path.join(output_dir, LATE_NEUTRAL_CSV)
    report_path = os.path.join(output_dir, OUTPUT_TEXT)

    write_csv(destination_path, destination_rows)
    write_csv(score_path, score_rows)
    write_csv(late_path, late_records)
    write_report(
        report_path,
        files,
        rows,
        events,
        destination_rows,
        score_rows,
        late_records,
        late_distances,
        transition_rows,
        stability_rows,
    )

    print(f"Wrote {report_path}")
    print(f"Wrote {destination_path}")
    print(f"Wrote {score_path}")
    print(f"Wrote {late_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
