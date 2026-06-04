#!/usr/bin/env python3
"""
APVA Replay Validator 04 - Red Polarity Study

Behavioral validation of Black vs Red VolumePolarity directional behavior.

No trading system. No optimization. No fitting. No ML. No new states.
"""

from __future__ import annotations

import csv
import glob
import math
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_INPUT_DIR = r"C:\Users\rz0\Documents\ApvaAnalysis\NT8Export"
OUTPUT_TEXT = "APVA_ReplayValidator_04_RedPolarityStudy.txt"
SUMMARY_CSV = "RedBlackSummary.csv"
DESTINATION_CSV = "RedBlackDestinationBehavior.csv"
PREVIOUS_PHASE_CSV = "RedBlackPreviousPhaseBehavior.csv"
PERSISTENCE_CSV = "RedBlackPersistenceBehavior.csv"
TIMING_CSV = "RedBlackTimingTest.csv"

HORIZONS = [1, 2, 3, 5, 10]
DESTINATION_FAMILIES = [
    "CompressionProcessing",
    "ReassertionProcessing",
    "RecoveryResolution",
    "MixedStructure",
    "ExhaustionPersistence",
]
PREVIOUS_PHASES = ["NeutralFormation", "NeutralMaturation", "LateNeutral"]
PERSISTENCE_GROUPS = ["1-2", "3-4", "5-6", "7-10", "11+"]


@dataclass
class Row:
    timestamp: str
    instrument: str
    close: Optional[float]
    loop_phase: str
    destination_family: str
    previous_loop_phase: str
    volume_polarity: str
    prior_neutral_persistence_bars: Optional[int]


@dataclass
class Event:
    index: int
    row: Row
    raw_returns: Dict[int, Optional[float]]
    directional_returns: Dict[int, Optional[float]]
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
    value = parse_float(value)
    if value is None:
        return None
    return int(value)


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


def load_rows(files: Sequence[str]) -> List[Row]:
    rows: List[Row] = []
    for path in files:
        with open(path, "r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "PriorNeutralPersistenceBars" not in reader.fieldnames:
                raise ValueError(
                    f"PriorNeutralPersistenceBars missing from {path}. Regenerate the NT8 export first."
                )
            for raw in reader:
                rows.append(
                    Row(
                        timestamp=clean_text(raw.get("Timestamp")),
                        instrument=clean_text(raw.get("Instrument")),
                        close=parse_float(raw.get("Close")),
                        loop_phase=clean_text(raw.get("LoopPhase")),
                        destination_family=clean_text(raw.get("DestinationFamily")),
                        previous_loop_phase=clean_text(raw.get("PreviousLoopPhase")),
                        volume_polarity=clean_text(raw.get("VolumePolarity")),
                        prior_neutral_persistence_bars=parse_int(raw.get("PriorNeutralPersistenceBars")),
                    )
                )
    return rows


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


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def win_rate(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return sum(1 for value in numeric if value > 0) / len(numeric)


def format_number(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def raw_return(rows: Sequence[Row], index: int, horizon: int) -> Optional[float]:
    if index + horizon >= len(rows):
        return None
    start = rows[index].close
    end = rows[index + horizon].close
    if start is None or end is None:
        return None
    return end - start


def directional_return(rows: Sequence[Row], index: int, horizon: int, polarity: str) -> Optional[float]:
    raw = raw_return(rows, index, horizon)
    if raw is None:
        return None
    if polarity == "Black":
        return raw
    if polarity == "Red":
        return -raw
    return None


def mfe_mae_5(rows: Sequence[Row], index: int, polarity: str) -> tuple[Optional[float], Optional[float]]:
    if index + 5 >= len(rows) or rows[index].close is None:
        return None, None
    future = [rows[index + offset].close for offset in range(1, 6)]
    if any(value is None for value in future):
        return None, None
    start = rows[index].close
    assert start is not None
    future_values = [value for value in future if value is not None]
    if polarity == "Black":
        return max(future_values) - start, min(future_values) - start
    if polarity == "Red":
        return start - min(future_values), start - max(future_values)
    return None, None


def extract_events(rows: Sequence[Row]) -> List[Event]:
    events: List[Event] = []
    for index, row in enumerate(rows):
        if row.loop_phase != "DestinationSelection":
            continue
        if row.volume_polarity not in {"Black", "Red"}:
            continue
        if index + 10 >= len(rows):
            continue

        raw_returns = {horizon: raw_return(rows, index, horizon) for horizon in HORIZONS}
        directional_returns = {
            horizon: directional_return(rows, index, horizon, row.volume_polarity)
            for horizon in HORIZONS
        }
        mfe5, mae5 = mfe_mae_5(rows, index, row.volume_polarity)
        if directional_returns[10] is None:
            continue
        events.append(
            Event(
                index=index,
                row=row,
                raw_returns=raw_returns,
                directional_returns=directional_returns,
                mfe5=mfe5,
                mae5=mae5,
                persistence_group=persistence_group(row.prior_neutral_persistence_bars),
            )
        )
    return events


def group_by(events: Sequence[Event], key_fn) -> Dict[object, List[Event]]:
    grouped: Dict[object, List[Event]] = defaultdict(list)
    for event in events:
        grouped[key_fn(event)].append(event)
    return grouped


def mfe_mae_ratio(events: Sequence[Event]) -> Optional[float]:
    mfe = mean(event.mfe5 for event in events)
    mae = mean(event.mae5 for event in events)
    if mfe is None or mae in {None, 0}:
        return None
    return mfe / abs(mae)


def summary_row(label_key: str, label: object, events: Sequence[Event], include_raw: bool = False) -> Dict[str, object]:
    row: Dict[str, object] = {
        label_key: label,
        "Count": len(events),
        "AvgDR1": mean(event.directional_returns[1] for event in events),
        "AvgDR2": mean(event.directional_returns[2] for event in events),
        "AvgDR3": mean(event.directional_returns[3] for event in events),
        "AvgDR5": mean(event.directional_returns[5] for event in events),
        "AvgDR10": mean(event.directional_returns[10] for event in events),
        "WinRate1": win_rate(event.directional_returns[1] for event in events),
        "WinRate3": win_rate(event.directional_returns[3] for event in events),
        "WinRate5": win_rate(event.directional_returns[5] for event in events),
        "WinRate10": win_rate(event.directional_returns[10] for event in events),
        "AvgMFE5": mean(event.mfe5 for event in events),
        "AvgMAE5": mean(event.mae5 for event in events),
        "MfeMaeRatio5": mfe_mae_ratio(events),
    }
    if include_raw:
        row.update(
            {
                "AvgRaw1": mean(event.raw_returns[1] for event in events),
                "AvgRaw3": mean(event.raw_returns[3] for event in events),
                "AvgRaw5": mean(event.raw_returns[5] for event in events),
                "AvgRaw10": mean(event.raw_returns[10] for event in events),
            }
        )
    return row


def red_black_summary(events: Sequence[Event]) -> List[Dict[str, object]]:
    grouped = group_by(events, lambda event: event.row.volume_polarity)
    return [summary_row("VolumePolarity", polarity, grouped.get(polarity, []), include_raw=True) for polarity in ["Black", "Red"]]


def destination_behavior(events: Sequence[Event]) -> List[Dict[str, object]]:
    grouped = group_by(events, lambda event: (event.row.volume_polarity, event.row.destination_family))
    rows = []
    for polarity in ["Black", "Red"]:
        for family in DESTINATION_FAMILIES:
            family_events = grouped.get((polarity, family), [])
            row = summary_row("Group", f"{polarity} + {family}", family_events)
            row["VolumePolarity"] = polarity
            row["DestinationFamily"] = family
            row["SampleClass"] = "Interpretable" if len(family_events) >= 10 else "LowSample"
            rows.append(row)
    return rows


def previous_phase_behavior(events: Sequence[Event]) -> List[Dict[str, object]]:
    grouped = group_by(events, lambda event: (event.row.volume_polarity, event.row.previous_loop_phase))
    rows = []
    for polarity in ["Black", "Red"]:
        for phase in PREVIOUS_PHASES:
            phase_events = grouped.get((polarity, phase), [])
            row = summary_row("Group", f"{polarity} + {phase}", phase_events)
            row["VolumePolarity"] = polarity
            row["PreviousLoopPhase"] = phase
            row["SampleClass"] = "Interpretable" if len(phase_events) >= 10 else "LowSample"
            rows.append(row)
    return rows


def persistence_behavior(events: Sequence[Event]) -> List[Dict[str, object]]:
    grouped = group_by(events, lambda event: (event.row.volume_polarity, event.persistence_group))
    rows = []
    for polarity in ["Black", "Red"]:
        for group in PERSISTENCE_GROUPS:
            group_events = grouped.get((polarity, group), [])
            row = summary_row("Group", f"{polarity} + {group}", group_events)
            row["VolumePolarity"] = polarity
            row["PriorNeutralPersistenceGroup"] = group
            row["SampleClass"] = "Interpretable" if len(group_events) >= 10 else "LowSample"
            rows.append(row)
    return rows


def timing_return(rows: Sequence[Row], event: Event, anchor_offset: int, horizon: int) -> Optional[float]:
    anchor_index = event.index - anchor_offset
    if anchor_index < 0:
        return None
    return directional_return(rows, anchor_index, horizon, event.row.volume_polarity)


def timing_behavior(rows: Sequence[Row], events: Sequence[Event]) -> List[Dict[str, object]]:
    grouped = group_by(events, lambda event: event.row.volume_polarity)
    rows_out = []
    for polarity in ["Red", "Black"]:
        polarity_events = grouped.get(polarity, [])
        row = {"VolumePolarity": polarity, "Count": len(polarity_events)}
        for offset, label in [(0, "Event"), (1, "Minus1"), (2, "Minus2")]:
            dr5_values = [timing_return(rows, event, offset, 5) for event in polarity_events]
            dr10_values = [timing_return(rows, event, offset, 10) for event in polarity_events]
            row[f"{label}AvgDR5"] = mean(dr5_values)
            row[f"{label}AvgDR10"] = mean(dr10_values)
            row[f"{label}WinRate5"] = win_rate(dr5_values)
            row[f"{label}WinRate10"] = win_rate(dr10_values)
        rows_out.append(row)
    return rows_out


def red_inversion_check(events: Sequence[Event]) -> Dict[str, object]:
    red_events = [event for event in events if event.row.volume_polarity == "Red"]
    normal = [event.directional_returns[5] for event in red_events]
    inverted = [-value if value is not None else None for value in normal]
    normal_avg = mean(normal)
    inverted_avg = mean(inverted)
    normal_win = win_rate(normal)
    inverted_win = win_rate(inverted)
    flag = (
        normal_avg is not None
        and inverted_avg is not None
        and inverted_win is not None
        and normal_win is not None
        and inverted_avg > normal_avg
        and inverted_win > normal_win
    )
    return {
        "RedNormalAvgDR5": normal_avg,
        "RedInvertedAvgDR5": inverted_avg,
        "RedNormalWinRate5": normal_win,
        "RedInvertedWinRate5": inverted_win,
        "RedPolarityDirectionPossiblyWrong": flag,
    }


def sample_bias_check(summary_rows: Sequence[Dict[str, object]]) -> bool:
    black = next((row for row in summary_rows if row["VolumePolarity"] == "Black"), None)
    red = next((row for row in summary_rows if row["VolumePolarity"] == "Red"), None)
    if not black or not red:
        return False
    black_raw = black.get("AvgRaw5")
    red_raw = red.get("AvgRaw5")
    return black_raw is not None and red_raw is not None and float(black_raw) > 0 and float(red_raw) > 0


def red_late_flag(timing_rows: Sequence[Dict[str, object]]) -> bool:
    red = next((row for row in timing_rows if row["VolumePolarity"] == "Red"), None)
    if not red:
        return False
    event = red.get("EventAvgDR5")
    minus1 = red.get("Minus1AvgDR5")
    minus2 = red.get("Minus2AvgDR5")
    if event is None:
        return False
    better = [value for value in [minus1, minus2] if value is not None and float(value) > float(event)]
    return bool(better)


def conclusion(summary_rows: Sequence[Dict[str, object]], inversion: Dict[str, object], timing_rows: Sequence[Dict[str, object]]) -> str:
    red = next((row for row in summary_rows if row["VolumePolarity"] == "Red"), None)
    if not red:
        return "Inconclusive"
    if inversion.get("RedPolarityDirectionPossiblyWrong"):
        return "RedPolarityDirectionWrong"
    if red_late_flag(timing_rows):
        return "RedPolarityLate"
    if sample_bias_check(summary_rows):
        return "RedPolaritySampleBias"
    red_avg = red.get("AvgDR5")
    red_win = red.get("WinRate5")
    if red_avg is not None and red_win is not None and float(red_avg) > 0 and float(red_win) >= 0.50:
        return "RedPolarityUsable"
    if red_avg is not None and red_win is not None and float(red_avg) < 0 and float(red_win) < 0.50:
        return "RedPolarityUnusable"
    return "Inconclusive"


def recommendation(result: str) -> str:
    if result == "RedPolarityDirectionWrong":
        return "RevisePolarityLogic"
    if result in {"RedPolarityLate", "RedPolaritySampleBias", "Inconclusive"}:
        return "CollectMoreData"
    if result == "RedPolarityUnusable":
        return "SwitchToContainerLateralGaussian"
    return "ContinueAPVAStateMachine"


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
    headers = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_csv_value(row.get(key)) for key in headers})


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
    summary_rows: Sequence[Dict[str, object]],
    destination_rows: Sequence[Dict[str, object]],
    previous_phase_rows: Sequence[Dict[str, object]],
    persistence_rows: Sequence[Dict[str, object]],
    timing_rows: Sequence[Dict[str, object]],
    inversion: Dict[str, object],
) -> None:
    result = conclusion(summary_rows, inversion, timing_rows)
    next_step = recommendation(result)
    instruments = sorted({row.instrument for row in rows if row.instrument})
    timestamps = [row.timestamp for row in rows if row.timestamp]

    lines: List[str] = []
    lines.append("APVA Replay Validator 04 - Red Polarity Study")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Purpose")
    lines.append("Determine why Red VolumePolarity events underperform.")
    lines.append("Behavioral validation only. No trading system. No optimization. No fitting. No ML.")
    lines.append("")
    lines.append("Input")
    lines.append("-" * 78)
    lines.append(f"Files: {len(files)}")
    for file_path in files:
        lines.append(f"  {file_path}")
    lines.append(f"Rows read: {len(rows)}")
    lines.append(f"Usable DestinationSelection events: {len(events)}")
    lines.append(f"Instruments: {', '.join(instruments) if instruments else 'N/A'}")
    lines.append(f"Date range: {min(timestamps) if timestamps else 'N/A'} to {max(timestamps) if timestamps else 'N/A'}")
    lines.append("")

    lines.append("SECTION 1 - Red / Black Event Summary")
    lines.append("-" * 78)
    append_table(lines, summary_rows)
    lines.append("")

    lines.append("SECTION 2 - Raw Return Bias")
    lines.append("-" * 78)
    raw_rows = [
        {
            "VolumePolarity": row["VolumePolarity"],
            "Count": row["Count"],
            "AvgRaw1": row.get("AvgRaw1"),
            "AvgRaw3": row.get("AvgRaw3"),
            "AvgRaw5": row.get("AvgRaw5"),
            "AvgRaw10": row.get("AvgRaw10"),
        }
        for row in summary_rows
    ]
    append_table(lines, raw_rows)
    lines.append("Sample bullish bias flag: " + str(sample_bias_check(summary_rows)))
    lines.append("")

    lines.append("SECTION 3 - Polarity Direction Sanity Check")
    lines.append("-" * 78)
    append_table(lines, [inversion])
    lines.append("")

    lines.append("SECTION 4 - Red by DestinationFamily")
    lines.append("-" * 78)
    append_table(lines, [row for row in destination_rows if row["VolumePolarity"] == "Red"])
    lines.append("")

    lines.append("SECTION 5 - Black by DestinationFamily")
    lines.append("-" * 78)
    append_table(lines, [row for row in destination_rows if row["VolumePolarity"] == "Black"])
    lines.append("")

    lines.append("SECTION 6 - Red by PreviousLoopPhase")
    lines.append("-" * 78)
    append_table(lines, [row for row in previous_phase_rows if row["VolumePolarity"] == "Red"])
    lines.append("")

    lines.append("SECTION 7 - Black by PreviousLoopPhase")
    lines.append("-" * 78)
    append_table(lines, [row for row in previous_phase_rows if row["VolumePolarity"] == "Black"])
    lines.append("")

    lines.append("SECTION 8 - Red by PriorNeutralPersistenceGroup")
    lines.append("-" * 78)
    append_table(lines, [row for row in persistence_rows if row["VolumePolarity"] == "Red"])
    lines.append("")

    lines.append("SECTION 9 - Black by PriorNeutralPersistenceGroup")
    lines.append("-" * 78)
    append_table(lines, [row for row in persistence_rows if row["VolumePolarity"] == "Black"])
    lines.append("")

    lines.append("SECTION 10 / 11 - Timing Tests")
    lines.append("-" * 78)
    append_table(lines, timing_rows)
    lines.append("RedEventsLate flag: " + str(red_late_flag(timing_rows)))
    lines.append("")

    lines.append("SECTION 12 - Summary")
    lines.append("-" * 78)
    lines.append(f"Conclusion: {result}")
    lines.append(f"Recommended next step: {next_step}")
    lines.append(f"Red polarity direction possibly wrong: {inversion.get('RedPolarityDirectionPossiblyWrong')}")
    lines.append(f"Red events late: {red_late_flag(timing_rows)}")
    lines.append(f"Sample bullish bias: {sample_bias_check(summary_rows)}")
    lines.append("Red weakness is concentrated where Red subgroup DR/win rates remain negative in the tables above.")
    lines.append("")

    lines.append("Low-DoF / Research Audit")
    lines.append("-" * 78)
    lines.append("No threshold tuning.")
    lines.append("No optimization.")
    lines.append("No fitting.")
    lines.append("No machine learning.")
    lines.append("No trading rules.")
    lines.append("Behavioral validation only.")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main(argv: Sequence[str]) -> int:
    files = input_files(argv[1:])
    if not files:
        print(f"No APVA_V10_StateMachine_*.csv files found in {DEFAULT_INPUT_DIR}", file=sys.stderr)
        return 1

    try:
        rows = load_rows(files)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    events = extract_events(rows)
    summary_rows = red_black_summary(events)
    destination_rows = destination_behavior(events)
    previous_phase_rows = previous_phase_behavior(events)
    persistence_rows = persistence_behavior(events)
    timing_rows = timing_behavior(rows, events)
    inversion = red_inversion_check(events)

    output_dir = os.path.dirname(files[0]) if files else DEFAULT_INPUT_DIR
    write_csv(os.path.join(output_dir, SUMMARY_CSV), summary_rows)
    write_csv(os.path.join(output_dir, DESTINATION_CSV), destination_rows)
    write_csv(os.path.join(output_dir, PREVIOUS_PHASE_CSV), previous_phase_rows)
    write_csv(os.path.join(output_dir, PERSISTENCE_CSV), persistence_rows)
    write_csv(os.path.join(output_dir, TIMING_CSV), timing_rows)
    write_report(
        os.path.join(output_dir, OUTPUT_TEXT),
        files,
        rows,
        events,
        summary_rows,
        destination_rows,
        previous_phase_rows,
        persistence_rows,
        timing_rows,
        inversion,
    )

    print(f"Wrote {os.path.join(output_dir, OUTPUT_TEXT)}")
    print(f"Wrote {os.path.join(output_dir, SUMMARY_CSV)}")
    print(f"Wrote {os.path.join(output_dir, DESTINATION_CSV)}")
    print(f"Wrote {os.path.join(output_dir, PREVIOUS_PHASE_CSV)}")
    print(f"Wrote {os.path.join(output_dir, PERSISTENCE_CSV)}")
    print(f"Wrote {os.path.join(output_dir, TIMING_CSV)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
