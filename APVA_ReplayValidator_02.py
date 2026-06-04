#!/usr/bin/env python3
"""
APVA Replay Validator 02 - Directional Behavior

Tests whether APVA v1.0 DestinationSelection events align with the bar's
VolumePolarity-implied direction.

Research only. No trading system. No optimization. No fitting. No ML.
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_INPUT_DIR = r"C:\Users\rz0\Documents\ApvaAnalysis\NT8Export"
OUTPUT_TEXT = "APVA_ReplayValidator_02.txt"
DESTINATION_CSV = "DestinationDirectionalBehavior.csv"
FAILURE_SCORE_CSV = "FailureScoreDirectionalBehavior.csv"
PREVIOUS_PHASE_CSV = "PreviousPhaseDirectionalBehavior.csv"
INTERACTION_CSV = "InteractionDirectionalBehavior.csv"
POLARITY_CSV = "PolarityDirectionalBehavior.csv"

HORIZONS = [1, 2, 3, 5, 10]
DESTINATION_FAMILIES = [
    "CompressionProcessing",
    "ReassertionProcessing",
    "RecoveryResolution",
    "MixedStructure",
    "ExhaustionPersistence",
]
PREVIOUS_PHASES = [
    "NeutralFormation",
    "NeutralMaturation",
    "LateNeutral",
    "ReturnToNeutral",
    "Excursion",
]


@dataclass
class Row:
    source_file: str
    row_number: int
    timestamp: str
    instrument: str
    close: Optional[float]
    loop_phase: str
    previous_loop_phase: str
    destination_family: str
    failure_warning_score: Optional[int]
    volume_polarity: str


@dataclass
class DirectionalEvent:
    index: int
    row: Row
    raw_returns: Dict[int, Optional[float]]
    directional_returns: Dict[int, Optional[float]]
    directional_mfe: Dict[int, Optional[float]]
    directional_mae: Dict[int, Optional[float]]


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
    value_float = parse_float(value)
    if value_float is None:
        return None
    return int(value_float)


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
            for row_number, raw in enumerate(reader, start=2):
                rows.append(
                    Row(
                        source_file=path,
                        row_number=row_number,
                        timestamp=clean_text(raw.get("Timestamp")),
                        instrument=clean_text(raw.get("Instrument")),
                        close=parse_float(raw.get("Close")),
                        loop_phase=clean_text(raw.get("LoopPhase")),
                        previous_loop_phase=clean_text(raw.get("PreviousLoopPhase")),
                        destination_family=clean_text(raw.get("DestinationFamily")),
                        failure_warning_score=parse_int(raw.get("FailureWarningScore")),
                        volume_polarity=clean_text(raw.get("VolumePolarity")),
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


def win_rate(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return sum(1 for value in numeric if value > 0) / len(numeric)


def format_number(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def format_percent(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def directional_metrics(rows: Sequence[Row], index: int, horizon: int) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    event = rows[index]
    if event.close is None or event.volume_polarity not in {"Black", "Red"}:
        return None, None, None, None

    end_index = index + horizon
    if end_index >= len(rows) or rows[end_index].close is None:
        return None, None, None, None

    future_closes = [
        rows[index + offset].close
        for offset in range(1, horizon + 1)
        if index + offset < len(rows) and rows[index + offset].close is not None
    ]
    if len(future_closes) < horizon:
        return None, None, None, None

    raw_return = rows[end_index].close - event.close
    if event.volume_polarity == "Black":
        directional_return = raw_return
        directional_mfe = max(future_closes) - event.close
        directional_mae = min(future_closes) - event.close
    else:
        directional_return = event.close - rows[end_index].close
        directional_mfe = event.close - min(future_closes)
        directional_mae = event.close - max(future_closes)

    return raw_return, directional_return, directional_mfe, directional_mae


def extract_events(rows: Sequence[Row]) -> Tuple[List[DirectionalEvent], Dict[str, int]]:
    found = 0
    skipped_missing_future = 0
    skipped_missing_polarity = 0
    events: List[DirectionalEvent] = []

    for index, row in enumerate(rows):
        if row.loop_phase != "DestinationSelection":
            continue
        found += 1

        if row.volume_polarity not in {"Black", "Red"}:
            skipped_missing_polarity += 1
            continue

        if index + max(HORIZONS) >= len(rows):
            skipped_missing_future += 1
            continue

        raw_returns: Dict[int, Optional[float]] = {}
        directional_returns: Dict[int, Optional[float]] = {}
        directional_mfe: Dict[int, Optional[float]] = {}
        directional_mae: Dict[int, Optional[float]] = {}
        usable = True

        for horizon in HORIZONS:
            raw_ret, directional_ret, mfe, mae = directional_metrics(rows, index, horizon)
            raw_returns[horizon] = raw_ret
            directional_returns[horizon] = directional_ret
            directional_mfe[horizon] = mfe
            directional_mae[horizon] = mae
            if horizon == max(HORIZONS) and directional_ret is None:
                usable = False

        if not usable:
            skipped_missing_future += 1
            continue

        events.append(
            DirectionalEvent(
                index=index,
                row=row,
                raw_returns=raw_returns,
                directional_returns=directional_returns,
                directional_mfe=directional_mfe,
                directional_mae=directional_mae,
            )
        )

    return events, {
        "EventsFound": found,
        "EventsUsable": len(events),
        "SkippedMissingFuture": skipped_missing_future,
        "SkippedMissingPolarity": skipped_missing_polarity,
    }


def summarize_events(events: Sequence[DirectionalEvent]) -> Dict[str, object]:
    summary: Dict[str, object] = {"Count": len(events)}
    for horizon in HORIZONS:
        summary[f"AvgDR{horizon}"] = mean(event.directional_returns[horizon] for event in events)
        summary[f"MedianDR{horizon}"] = median(event.directional_returns[horizon] for event in events)
        summary[f"WinRate{horizon}"] = win_rate(event.directional_returns[horizon] for event in events)
        summary[f"AverageRawReturn{horizon}"] = mean(event.raw_returns[horizon] for event in events)
        summary[f"AvgMFE{horizon}"] = mean(event.directional_mfe[horizon] for event in events)
        summary[f"AvgMAE{horizon}"] = mean(event.directional_mae[horizon] for event in events)
    return summary


def destination_behavior(events: Sequence[DirectionalEvent]) -> List[Dict[str, object]]:
    grouped = group_by(events, lambda event: event.row.destination_family)
    rows: List[Dict[str, object]] = []
    families = DESTINATION_FAMILIES + sorted(set(grouped) - set(DESTINATION_FAMILIES))
    for family in families:
        family_events = grouped.get(family, [])
        rows.append(
            {
                "DestinationFamily": family,
                "Count": len(family_events),
                "AvgDR1": mean(event.directional_returns[1] for event in family_events),
                "AvgDR2": mean(event.directional_returns[2] for event in family_events),
                "AvgDR3": mean(event.directional_returns[3] for event in family_events),
                "AvgDR5": mean(event.directional_returns[5] for event in family_events),
                "AvgDR10": mean(event.directional_returns[10] for event in family_events),
                "WinRate3": win_rate(event.directional_returns[3] for event in family_events),
                "WinRate5": win_rate(event.directional_returns[5] for event in family_events),
                "WinRate10": win_rate(event.directional_returns[10] for event in family_events),
                "AvgMFE5": mean(event.directional_mfe[5] for event in family_events),
                "AvgMAE5": mean(event.directional_mae[5] for event in family_events),
            }
        )
    return rows


def failure_score_behavior(events: Sequence[DirectionalEvent]) -> List[Dict[str, object]]:
    grouped = group_by(events, lambda event: event.row.failure_warning_score)
    rows: List[Dict[str, object]] = []
    for score in range(0, 7):
        score_events = grouped.get(score, [])
        rows.append(
            {
                "FailureWarningScore": score,
                "Count": len(score_events),
                "AvgDR3": mean(event.directional_returns[3] for event in score_events),
                "AvgDR5": mean(event.directional_returns[5] for event in score_events),
                "AvgDR10": mean(event.directional_returns[10] for event in score_events),
                "WinRate3": win_rate(event.directional_returns[3] for event in score_events),
                "WinRate5": win_rate(event.directional_returns[5] for event in score_events),
                "WinRate10": win_rate(event.directional_returns[10] for event in score_events),
                "AvgMFE5": mean(event.directional_mfe[5] for event in score_events),
                "AvgMAE5": mean(event.directional_mae[5] for event in score_events),
            }
        )
    return rows


def previous_phase_behavior(events: Sequence[DirectionalEvent]) -> List[Dict[str, object]]:
    grouped = group_by(events, lambda event: event.row.previous_loop_phase)
    rows: List[Dict[str, object]] = []
    phases = PREVIOUS_PHASES + sorted(set(grouped) - set(PREVIOUS_PHASES))
    for phase in phases:
        phase_events = grouped.get(phase, [])
        rows.append(
            {
                "PreviousLoopPhase": phase,
                "Count": len(phase_events),
                "AvgDR3": mean(event.directional_returns[3] for event in phase_events),
                "AvgDR5": mean(event.directional_returns[5] for event in phase_events),
                "AvgDR10": mean(event.directional_returns[10] for event in phase_events),
                "WinRate3": win_rate(event.directional_returns[3] for event in phase_events),
                "WinRate5": win_rate(event.directional_returns[5] for event in phase_events),
                "WinRate10": win_rate(event.directional_returns[10] for event in phase_events),
            }
        )
    return rows


def interaction_behavior(events: Sequence[DirectionalEvent]) -> List[Dict[str, object]]:
    grouped = group_by(events, lambda event: (event.row.previous_loop_phase, event.row.destination_family))
    rows: List[Dict[str, object]] = []
    for key in sorted(grouped, key=lambda item: (str(item[0]), str(item[1]))):
        pair_events = grouped[key]
        rows.append(
            {
                "PreviousLoopPhase": key[0],
                "DestinationFamily": key[1],
                "Count": len(pair_events),
                "AvgDR5": mean(event.directional_returns[5] for event in pair_events),
                "AvgDR10": mean(event.directional_returns[10] for event in pair_events),
                "WinRate5": win_rate(event.directional_returns[5] for event in pair_events),
                "WinRate10": win_rate(event.directional_returns[10] for event in pair_events),
                "SampleClass": "Interpretable" if len(pair_events) >= 10 else "LowSample",
            }
        )
    return rows


def polarity_behavior(events: Sequence[DirectionalEvent]) -> List[Dict[str, object]]:
    grouped = group_by(events, lambda event: event.row.volume_polarity)
    rows: List[Dict[str, object]] = []
    for polarity in ["Black", "Red"]:
        polarity_events = grouped.get(polarity, [])
        rows.append(
            {
                "VolumePolarity": polarity,
                "Count": len(polarity_events),
                "AvgDR5": mean(event.directional_returns[5] for event in polarity_events),
                "AvgDR10": mean(event.directional_returns[10] for event in polarity_events),
                "WinRate5": win_rate(event.directional_returns[5] for event in polarity_events),
                "WinRate10": win_rate(event.directional_returns[10] for event in polarity_events),
            }
        )
    return rows


def group_by(events: Sequence[DirectionalEvent], key_fn) -> Dict[object, List[DirectionalEvent]]:
    grouped: Dict[object, List[DirectionalEvent]] = defaultdict(list)
    for event in events:
        grouped[key_fn(event)].append(event)
    return grouped


def best_worst_classes(rows: Sequence[Dict[str, object]], key_name: str) -> List[Dict[str, object]]:
    candidates = [row for row in rows if int(row.get("Count", 0)) >= 10]
    rankings: List[Dict[str, object]] = []
    for metric in ["AvgDR5", "AvgDR10", "WinRate5", "WinRate10"]:
        valid = [row for row in candidates if row.get(metric) is not None]
        if not valid:
            continue
        best = max(valid, key=lambda row: float(row[metric]))
        worst = min(valid, key=lambda row: float(row[metric]))
        rankings.append(
            {
                "Metric": metric,
                "BestClass": best.get(key_name, composite_label(best)),
                "BestValue": best.get(metric),
                "WorstClass": worst.get(key_name, composite_label(worst)),
                "WorstValue": worst.get(metric),
            }
        )

    valid_ratio = [
        row for row in candidates
        if row.get("AvgMFE5") is not None and row.get("AvgMAE5") not in {None, 0}
    ]
    if valid_ratio:
        def ratio(row: Dict[str, object]) -> float:
            return float(row["AvgMFE5"]) / abs(float(row["AvgMAE5"]))

        best = max(valid_ratio, key=ratio)
        worst = min(valid_ratio, key=ratio)
        rankings.append(
            {
                "Metric": "AvgMFE5 / abs(AvgMAE5)",
                "BestClass": best.get(key_name, composite_label(best)),
                "BestValue": ratio(best),
                "WorstClass": worst.get(key_name, composite_label(worst)),
                "WorstValue": ratio(worst),
            }
        )
    return rankings


def composite_label(row: Dict[str, object]) -> str:
    if "PreviousLoopPhase" in row and "DestinationFamily" in row:
        return f"{row['PreviousLoopPhase']} -> {row['DestinationFamily']}"
    return "N/A"


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
            writer.writerow({key: format_csv_value(value) for key, value in row.items()})


def format_csv_value(value: object) -> object:
    if isinstance(value, float):
        return format_number(value)
    if value is None:
        return "N/A"
    return value


def append_table(lines: List[str], rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        lines.append("N/A")
        return
    headers = list(rows[0].keys())
    formatted_rows = []
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


def classification(destination_rows: Sequence[Dict[str, object]], previous_phase_rows: Sequence[Dict[str, object]]) -> str:
    interpretable = [row for row in destination_rows if int(row.get("Count", 0)) >= 10]
    if not interpretable:
        return "NoDirectionalSeparation"

    avg_dr5_values = [float(row["AvgDR5"]) for row in interpretable if row.get("AvgDR5") is not None]
    win5_values = [float(row["WinRate5"]) for row in interpretable if row.get("WinRate5") is not None]
    if not avg_dr5_values or not win5_values:
        return "NoDirectionalSeparation"

    spread = max(avg_dr5_values) - min(avg_dr5_values)
    best_win = max(win5_values)
    late_row = next((row for row in previous_phase_rows if row["PreviousLoopPhase"] == "LateNeutral"), None)
    late_positive = bool(late_row and late_row.get("AvgDR5") is not None and float(late_row["AvgDR5"]) > 0)

    if spread >= 15 and best_win >= 0.58 and late_positive:
        return "StrongDirectionalSeparation"
    if spread >= 8 and best_win >= 0.54:
        return "ModerateDirectionalSeparation"
    if spread >= 4 or best_win >= 0.52:
        return "WeakDirectionalSeparation"
    return "NoDirectionalSeparation"


def recommendation(summary_class: str) -> str:
    if summary_class in {"ModerateDirectionalSeparation", "StrongDirectionalSeparation"}:
        return "ReplayValidator03_EventQuality"
    if summary_class == "WeakDirectionalSeparation":
        return "CollectMoreData"
    return "ReviseDestinationClassifier"


def write_report(
    path: str,
    files: Sequence[str],
    rows: Sequence[Row],
    events: Sequence[DirectionalEvent],
    extraction: Dict[str, int],
    directional_summary: Dict[str, object],
    destination_rows: Sequence[Dict[str, object]],
    score_rows: Sequence[Dict[str, object]],
    previous_phase_rows: Sequence[Dict[str, object]],
    interaction_rows: Sequence[Dict[str, object]],
    polarity_rows: Sequence[Dict[str, object]],
) -> None:
    instruments = sorted({row.instrument for row in rows if row.instrument})
    timestamps = [row.timestamp for row in rows if row.timestamp]
    class_result = classification(destination_rows, previous_phase_rows)
    next_step = recommendation(class_result)
    destination_rankings = best_worst_classes(destination_rows, "DestinationFamily")
    interaction_rankings = best_worst_classes(interaction_rows, "Interaction")

    strongest_destination = best_class_name(destination_rows, "DestinationFamily", "AvgDR5")
    weakest_destination = worst_class_name(destination_rows, "DestinationFamily", "AvgDR5")
    late_row = next((row for row in previous_phase_rows if row["PreviousLoopPhase"] == "LateNeutral"), None)
    polarity_black = next((row for row in polarity_rows if row["VolumePolarity"] == "Black"), None)
    polarity_red = next((row for row in polarity_rows if row["VolumePolarity"] == "Red"), None)

    lines: List[str] = []
    lines.append("APVA Replay Validator 02 - Directional Behavior")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Purpose")
    lines.append("Test whether DestinationSelection movement aligns with VolumePolarity-implied direction.")
    lines.append("Research only. No trading system. No optimization. No fitting. No ML.")
    lines.append("")

    lines.append("Input")
    lines.append("-" * 78)
    lines.append(f"Files: {len(files)}")
    for file_path in files:
        lines.append(f"  {file_path}")
    lines.append(f"Rows read: {len(rows)}")
    lines.append(f"Instruments: {', '.join(instruments) if instruments else 'N/A'}")
    lines.append(f"Date range: {min(timestamps) if timestamps else 'N/A'} to {max(timestamps) if timestamps else 'N/A'}")
    lines.append("")

    lines.append("SECTION 1 - Event Extraction")
    lines.append("-" * 78)
    for key in ["EventsFound", "EventsUsable", "SkippedMissingFuture", "SkippedMissingPolarity"]:
        lines.append(f"{key}: {extraction[key]}")
    lines.append("")

    lines.append("SECTION 2 - Directional Returns")
    lines.append("-" * 78)
    for horizon in HORIZONS:
        lines.append(
            f"Horizon {horizon}: "
            f"AvgDR={format_number(directional_summary.get(f'AvgDR{horizon}'))} "
            f"MedianDR={format_number(directional_summary.get(f'MedianDR{horizon}'))} "
            f"WinRate={format_percent(directional_summary.get(f'WinRate{horizon}'))} "
            f"AvgRaw={format_number(directional_summary.get(f'AverageRawReturn{horizon}'))} "
            f"AvgMFE={format_number(directional_summary.get(f'AvgMFE{horizon}'))} "
            f"AvgMAE={format_number(directional_summary.get(f'AvgMAE{horizon}'))}"
        )
    lines.append("")

    lines.append("SECTION 3 - Destination Family Directional Behavior")
    lines.append("-" * 78)
    append_table(lines, destination_rows)
    lines.append("")

    lines.append("SECTION 4 - Failure Score Directional Behavior")
    lines.append("-" * 78)
    append_table(lines, score_rows)
    lines.append("")

    lines.append("SECTION 5 - Previous Phase Directional Behavior")
    lines.append("-" * 78)
    append_table(lines, previous_phase_rows)
    lines.append("")

    lines.append("SECTION 6 - Destination x PreviousPhase Interaction")
    lines.append("-" * 78)
    append_table(lines, interaction_rows)
    lines.append("")

    lines.append("SECTION 7 - Polarity Split")
    lines.append("-" * 78)
    append_table(lines, polarity_rows)
    lines.append("")

    lines.append("SECTION 8 - Best / Worst Event Classes")
    lines.append("-" * 78)
    lines.append("Destination rankings")
    append_table(lines, destination_rankings)
    lines.append("")
    lines.append("Interaction rankings")
    append_table(lines, interaction_rankings)
    lines.append("")

    lines.append("SECTION 9 - Summary")
    lines.append("-" * 78)
    lines.append(f"Directional separation classification: {class_result}")
    lines.append(f"Recommended next step: {next_step}")
    lines.append(f"Directionally strongest destination by AvgDR5: {strongest_destination}")
    lines.append(f"Directionally weakest destination by AvgDR5: {weakest_destination}")
    if late_row:
        lines.append(
            "LateNeutral after directional normalization: "
            f"AvgDR5={format_number(late_row.get('AvgDR5'))}, "
            f"WinRate5={format_percent(late_row.get('WinRate5'))}"
        )
    lines.append("FailureWarningScore directional behavior: see score table; higher score helps only if DR/win rate rises by score.")
    if polarity_black and polarity_red:
        lines.append(
            "Black/Red symmetry: "
            f"Black AvgDR5={format_number(polarity_black.get('AvgDR5'))}, "
            f"Red AvgDR5={format_number(polarity_red.get('AvgDR5'))}"
        )
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


def best_class_name(rows: Sequence[Dict[str, object]], label_key: str, metric: str) -> str:
    candidates = [row for row in rows if int(row.get("Count", 0)) >= 10 and row.get(metric) is not None]
    if not candidates:
        return "N/A"
    best = max(candidates, key=lambda row: float(row[metric]))
    return str(best.get(label_key, composite_label(best)))


def worst_class_name(rows: Sequence[Dict[str, object]], label_key: str, metric: str) -> str:
    candidates = [row for row in rows if int(row.get("Count", 0)) >= 10 and row.get(metric) is not None]
    if not candidates:
        return "N/A"
    worst = min(candidates, key=lambda row: float(row[metric]))
    return str(worst.get(label_key, composite_label(worst)))


def main(argv: Sequence[str]) -> int:
    files = input_files(argv[1:])
    if not files:
        print(f"No APVA_V10_StateMachine_*.csv files found in {DEFAULT_INPUT_DIR}", file=sys.stderr)
        return 1

    rows = load_rows(files)
    events, extraction = extract_events(rows)
    directional_summary = summarize_events(events)
    destination_rows = destination_behavior(events)
    score_rows = failure_score_behavior(events)
    previous_phase_rows = previous_phase_behavior(events)
    interaction_rows = interaction_behavior(events)
    polarity_rows = polarity_behavior(events)

    output_dir = os.path.dirname(files[0]) if files else DEFAULT_INPUT_DIR
    write_csv(os.path.join(output_dir, DESTINATION_CSV), destination_rows)
    write_csv(os.path.join(output_dir, FAILURE_SCORE_CSV), score_rows)
    write_csv(os.path.join(output_dir, PREVIOUS_PHASE_CSV), previous_phase_rows)
    write_csv(os.path.join(output_dir, INTERACTION_CSV), interaction_rows)
    write_csv(os.path.join(output_dir, POLARITY_CSV), polarity_rows)
    write_report(
        os.path.join(output_dir, OUTPUT_TEXT),
        files,
        rows,
        events,
        extraction,
        directional_summary,
        destination_rows,
        score_rows,
        previous_phase_rows,
        interaction_rows,
        polarity_rows,
    )

    print(f"Wrote {os.path.join(output_dir, OUTPUT_TEXT)}")
    print(f"Wrote {os.path.join(output_dir, DESTINATION_CSV)}")
    print(f"Wrote {os.path.join(output_dir, FAILURE_SCORE_CSV)}")
    print(f"Wrote {os.path.join(output_dir, PREVIOUS_PHASE_CSV)}")
    print(f"Wrote {os.path.join(output_dir, INTERACTION_CSV)}")
    print(f"Wrote {os.path.join(output_dir, POLARITY_CSV)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
