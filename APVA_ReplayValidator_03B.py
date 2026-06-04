#!/usr/bin/env python3
"""
APVA Replay Validator 03B - Prior Neutral Persistence

Validates completed NeutralProcessing age before DestinationSelection using
PriorNeutralPersistenceBars exported by xApvaV10StateMachine.

Research only. No trading system. No optimization. No fitting. No ML.
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
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_INPUT_DIR = r"C:\Users\rz0\Documents\ApvaAnalysis\NT8Export"
OUTPUT_TEXT = "APVA_ReplayValidator_03B.txt"
PRIOR_PERSISTENCE_CSV = "PriorNeutralPersistenceQuality.csv"
HORIZONS = [5, 10]


@dataclass
class Row:
    timestamp: str
    instrument: str
    close: Optional[float]
    loop_phase: str
    previous_loop_phase: str
    destination_family: str
    volume_polarity: str
    prior_neutral_persistence_bars: Optional[int]


@dataclass
class Event:
    row: Row
    dr5: Optional[float]
    dr10: Optional[float]
    mfe5: Optional[float]
    mae5: Optional[float]
    persistence_group: str


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
        files: List[str] = []
        for path in paths:
            if os.path.isdir(path):
                files.extend(glob.glob(os.path.join(path, "APVA_V10_StateMachine_*.csv")))
            else:
                files.extend(glob.glob(path))
        return sorted(set(files))
    return sorted(glob.glob(os.path.join(DEFAULT_INPUT_DIR, "APVA_V10_StateMachine_*.csv")))


def load_rows(files: Sequence[str]) -> tuple[List[Row], bool]:
    rows: List[Row] = []
    prior_column_present = True
    for path in files:
        with open(path, "r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "PriorNeutralPersistenceBars" not in reader.fieldnames:
                prior_column_present = False
            for raw in reader:
                rows.append(
                    Row(
                        timestamp=clean_text(raw.get("Timestamp")),
                        instrument=clean_text(raw.get("Instrument")),
                        close=parse_float(raw.get("Close")),
                        loop_phase=clean_text(raw.get("LoopPhase")),
                        previous_loop_phase=clean_text(raw.get("PreviousLoopPhase")),
                        destination_family=clean_text(raw.get("DestinationFamily")),
                        volume_polarity=clean_text(raw.get("VolumePolarity")),
                        prior_neutral_persistence_bars=parse_int(raw.get("PriorNeutralPersistenceBars")),
                    )
                )
    return rows, prior_column_present


def persistence_group(value: Optional[int]) -> str:
    if value is None:
        return "Unknown"
    if value <= 2:
        return "1-2"
    if value <= 4:
        return "3-4"
    if value <= 6:
        return "5-6"
    if value <= 10:
        return "7-10"
    return "11+"


def directional_event(rows: Sequence[Row], index: int) -> Optional[Event]:
    row = rows[index]
    if row.loop_phase != "DestinationSelection":
        return None
    if row.close is None or row.volume_polarity not in {"Black", "Red"}:
        return None
    if index + 10 >= len(rows):
        return None

    future_5 = [rows[index + offset].close for offset in range(1, 6)]
    future_10_close = rows[index + 10].close
    close_5 = rows[index + 5].close
    if future_10_close is None or close_5 is None or any(value is None for value in future_5):
        return None

    close = row.close
    future_5_values = [value for value in future_5 if value is not None]
    if row.volume_polarity == "Black":
        dr5 = close_5 - close
        dr10 = future_10_close - close
        mfe5 = max(future_5_values) - close
        mae5 = min(future_5_values) - close
    else:
        dr5 = close - close_5
        dr10 = close - future_10_close
        mfe5 = close - min(future_5_values)
        mae5 = close - max(future_5_values)

    return Event(
        row=row,
        dr5=dr5,
        dr10=dr10,
        mfe5=mfe5,
        mae5=mae5,
        persistence_group=persistence_group(row.prior_neutral_persistence_bars),
    )


def extract_events(rows: Sequence[Row]) -> List[Event]:
    events: List[Event] = []
    for index in range(len(rows)):
        event = directional_event(rows, index)
        if event is not None:
            events.append(event)
    return events


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


def minimum(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return min(numeric)


def win_rate(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return sum(1 for value in numeric if value > 0) / len(numeric)


def summarize_group(label: str, events: Sequence[Event]) -> Dict[str, object]:
    mfe5 = mean(event.mfe5 for event in events)
    mae5 = mean(event.mae5 for event in events)
    ratio = None
    if mfe5 is not None and mae5 not in {None, 0}:
        ratio = mfe5 / abs(mae5)
    return {
        "PriorNeutralPersistenceGroup": label,
        "Count": len(events),
        "AvgDR5": mean(event.dr5 for event in events),
        "AvgDR10": mean(event.dr10 for event in events),
        "WinRate5": win_rate(event.dr5 for event in events),
        "WinRate10": win_rate(event.dr10 for event in events),
        "AvgMFE5": mfe5,
        "AvgMAE5": mae5,
        "MfeMaeRatio5": ratio,
    }


def persistence_quality(events: Sequence[Event]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Event]] = defaultdict(list)
    for event in events:
        grouped[event.persistence_group].append(event)

    rows = []
    for group in ["1-2", "3-4", "5-6", "7-10", "11+", "Unknown"]:
        if group == "Unknown" and group not in grouped:
            continue
        rows.append(summarize_group(group, grouped.get(group, [])))
    return rows


def distribution(events: Sequence[Event]) -> Counter:
    return Counter(event.row.prior_neutral_persistence_bars for event in events if event.row.prior_neutral_persistence_bars is not None)


def consistency_count(events: Sequence[Event]) -> int:
    return sum(
        1
        for event in events
        if event.row.previous_loop_phase == "LateNeutral"
        and event.row.prior_neutral_persistence_bars is not None
        and event.row.prior_neutral_persistence_bars < 5
    )


def format_number(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def format_csv_value(value: object) -> object:
    if isinstance(value, float):
        return format_number(value)
    if value is None:
        return "N/A"
    return value


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
            writer.writerow({key: format_csv_value(row.get(key)) for key in fieldnames})


def append_table(lines: List[str], rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        lines.append("N/A")
        return
    headers = list(rows[0].keys())
    formatted_rows: List[Dict[str, str]] = []
    widths = {header: len(header) for header in headers}
    for row in rows:
        formatted = {header: str(format_csv_value(row.get(header))) for header in headers}
        formatted_rows.append(formatted)
        for header, value in formatted.items():
            widths[header] = max(widths[header], len(value))
    lines.append(" | ".join(header.ljust(widths[header]) for header in headers))
    lines.append("-+-".join("-" * widths[header] for header in headers))
    for row in formatted_rows:
        lines.append(" | ".join(row[header].ljust(widths[header]) for header in headers))


def write_report(
    path: str,
    files: Sequence[str],
    rows: Sequence[Row],
    events: Sequence[Event],
    quality_rows: Sequence[Dict[str, object]],
    prior_column_present: bool,
) -> None:
    values = [event.row.prior_neutral_persistence_bars for event in events if event.row.prior_neutral_persistence_bars is not None]
    dist = distribution(events)
    inconsistent = consistency_count(events)
    multiple_buckets = sum(1 for row in quality_rows if row["Count"] and row["PriorNeutralPersistenceGroup"] != "Unknown") > 1

    warnings: List[str] = []
    if not prior_column_present:
        warnings.append("PriorNeutralPersistenceBarsMissing")
    if inconsistent:
        warnings.append("NeutralPersistenceCaptureInconsistent")
    if not multiple_buckets:
        warnings.append("PriorNeutralPersistenceSingleBucket")

    lines: List[str] = []
    lines.append("APVA Replay Validator 03B - Prior Neutral Persistence")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Purpose")
    lines.append("Validate completed NeutralProcessing age before DestinationSelection.")
    lines.append("Research only. No trading system. No optimization. No fitting. No ML.")
    lines.append("")
    lines.append("Input")
    lines.append("-" * 78)
    lines.append(f"Files: {len(files)}")
    for file_path in files:
        lines.append(f"  {file_path}")
    lines.append(f"Rows read: {len(rows)}")
    lines.append(f"Usable DestinationSelection events: {len(events)}")
    lines.append(f"PriorNeutralPersistenceBars column present: {prior_column_present}")
    lines.append("")

    lines.append("PriorNeutralPersistenceBars Diagnostics")
    lines.append("-" * 78)
    lines.append(f"Min PriorNeutralPersistenceBars: {format_number(minimum(values), 0)}")
    lines.append(f"Max PriorNeutralPersistenceBars: {format_number(maximum(values), 0)}")
    lines.append(f"Mean PriorNeutralPersistenceBars: {format_number(mean(values))}")
    lines.append(f"Median PriorNeutralPersistenceBars: {format_number(median(values))}")
    lines.append("")
    lines.append("Distribution table")
    if dist:
        distribution_rows = [
            {"PriorNeutralPersistenceBars": key, "Count": value}
            for key, value in sorted(dist.items())
        ]
        append_table(lines, distribution_rows)
    else:
        lines.append("N/A")
    lines.append("")

    lines.append("Prior Neutral Persistence Quality")
    lines.append("-" * 78)
    append_table(lines, quality_rows)
    lines.append("")

    lines.append("Consistency Check")
    lines.append("-" * 78)
    lines.append(f"LateNeutral -> DestinationSelection events with PriorNeutralPersistenceBars < 5: {inconsistent}")
    if inconsistent:
        lines.append("WARNING: NeutralPersistenceCaptureInconsistent")
    lines.append("")

    lines.append("Summary")
    lines.append("-" * 78)
    lines.append(
        "Completed neutral persistence has multiple buckets: "
        + ("true" if multiple_buckets else "false")
    )
    if warnings:
        lines.append("Warnings: " + ", ".join(warnings))
    else:
        lines.append("Warnings: None")
    lines.append("If all events remain in 1-2 or Unknown after a fresh NT8 export, capture logic is still wrong.")
    lines.append("")

    lines.append("Low-DoF / Research Audit")
    lines.append("-" * 78)
    lines.append("No optimization.")
    lines.append("No fitting.")
    lines.append("No machine learning.")
    lines.append("No trading rules.")
    lines.append("No entries.")
    lines.append("No exits.")
    lines.append("No profit model.")
    lines.append("Behavioral validation only.")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main(argv: Sequence[str]) -> int:
    files = input_files(argv[1:])
    if not files:
        print(f"No APVA_V10_StateMachine_*.csv files found in {DEFAULT_INPUT_DIR}", file=sys.stderr)
        return 1

    rows, prior_column_present = load_rows(files)
    events = extract_events(rows)
    quality_rows = persistence_quality(events)

    output_dir = os.path.dirname(files[0]) if files else DEFAULT_INPUT_DIR
    write_csv(os.path.join(output_dir, PRIOR_PERSISTENCE_CSV), quality_rows)
    write_report(
        os.path.join(output_dir, OUTPUT_TEXT),
        files,
        rows,
        events,
        quality_rows,
        prior_column_present,
    )

    print(f"Wrote {os.path.join(output_dir, OUTPUT_TEXT)}")
    print(f"Wrote {os.path.join(output_dir, PRIOR_PERSISTENCE_CSV)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
