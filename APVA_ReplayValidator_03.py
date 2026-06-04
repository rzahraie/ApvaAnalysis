#!/usr/bin/env python3
"""
APVA Replay Validator 03 - Event Quality

Ranks APVA v1.0 DestinationSelection event classes by directional behavior.

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
OUTPUT_TEXT = "APVA_ReplayValidator_03.txt"
EVENT_QUALITY_CSV = "EventQualityRanking.csv"
DESTINATION_QUALITY_CSV = "DestinationQuality.csv"
NEUTRAL_PERSISTENCE_CSV = "NeutralPersistenceQuality.csv"
INTERACTION_QUALITY_CSV = "InteractionQuality.csv"

DESTINATION_FAMILIES = [
    "CompressionProcessing",
    "ReassertionProcessing",
    "RecoveryResolution",
    "MixedStructure",
    "ExhaustionPersistence",
]
PREVIOUS_PHASES = ["NeutralFormation", "NeutralMaturation", "LateNeutral"]


@dataclass
class Row:
    timestamp: str
    instrument: str
    close: Optional[float]
    loop_phase: str
    destination_family: str
    failure_warning_score: Optional[int]
    previous_loop_phase: str
    volume_polarity: str
    neutral_persistence_bars: Optional[int]
    structural_state: str
    age_bucket: str


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
            for raw in reader:
                rows.append(
                    Row(
                        timestamp=clean_text(raw.get("Timestamp")),
                        instrument=clean_text(raw.get("Instrument")),
                        close=parse_float(raw.get("Close")),
                        loop_phase=clean_text(raw.get("LoopPhase")),
                        destination_family=clean_text(raw.get("DestinationFamily")),
                        failure_warning_score=parse_int(raw.get("FailureWarningScore")),
                        previous_loop_phase=clean_text(raw.get("PreviousLoopPhase")),
                        volume_polarity=clean_text(raw.get("VolumePolarity")),
                        neutral_persistence_bars=parse_int(raw.get("NeutralPersistenceBars")),
                        structural_state=clean_text(raw.get("StructuralState")),
                        age_bucket=clean_text(raw.get("AgeBucket")),
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
    if future_10_close is None or any(value is None for value in future_5):
        return None

    close = row.close
    close_5 = rows[index + 5].close
    if close_5 is None:
        return None

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
        persistence_group=persistence_group(row.neutral_persistence_bars),
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


def win_rate(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return sum(1 for value in numeric if value > 0) / len(numeric)


def format_number(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def group_by(events: Sequence[Event], key_fn) -> Dict[object, List[Event]]:
    grouped: Dict[object, List[Event]] = defaultdict(list)
    for event in events:
        grouped[key_fn(event)].append(event)
    return grouped


def summarize_group(label_key: str, label: object, events: Sequence[Event]) -> Dict[str, object]:
    mfe5 = mean(event.mfe5 for event in events)
    mae5 = mean(event.mae5 for event in events)
    ratio = None
    if mfe5 is not None and mae5 not in {None, 0}:
        ratio = mfe5 / abs(mae5)

    return {
        label_key: label,
        "Count": len(events),
        "AvgDR5": mean(event.dr5 for event in events),
        "AvgDR10": mean(event.dr10 for event in events),
        "WinRate5": win_rate(event.dr5 for event in events),
        "WinRate10": win_rate(event.dr10 for event in events),
        "AvgMFE5": mfe5,
        "AvgMAE5": mae5,
        "MfeMaeRatio5": ratio,
        "SampleClass": "Interpretable" if len(events) >= 10 else "LowSample",
    }


def neutral_persistence_quality(events: Sequence[Event]) -> List[Dict[str, object]]:
    grouped = group_by(events, lambda event: event.persistence_group)
    order = ["1-2", "3-4", "5-6", "7-10", "11+", "Unknown"]
    return [summarize_group("NeutralPersistenceGroup", group, grouped.get(group, [])) for group in order if group in grouped or group != "Unknown"]


def previous_phase_quality(events: Sequence[Event]) -> List[Dict[str, object]]:
    grouped = group_by(events, lambda event: event.row.previous_loop_phase)
    rows = []
    for phase in PREVIOUS_PHASES:
        rows.append(summarize_group("PreviousLoopPhase", phase, grouped.get(phase, [])))
    return rows


def destination_quality(events: Sequence[Event]) -> List[Dict[str, object]]:
    grouped = group_by(events, lambda event: event.row.destination_family)
    rows = [summarize_group("DestinationFamily", family, grouped.get(family, [])) for family in DESTINATION_FAMILIES]
    return sorted(rows, key=lambda row: (float(row["AvgDR5"]) if row["AvgDR5"] is not None else -999999), reverse=True)


def failure_score_quality(events: Sequence[Event]) -> List[Dict[str, object]]:
    grouped = group_by(events, lambda event: event.row.failure_warning_score)
    rows = []
    for score in [3, 4, 5, 6]:
        rows.append(summarize_group("FailureWarningScore", score, grouped.get(score, [])))
    return rows


def polarity_quality(events: Sequence[Event]) -> List[Dict[str, object]]:
    grouped = group_by(events, lambda event: event.row.volume_polarity)
    return [summarize_group("VolumePolarity", polarity, grouped.get(polarity, [])) for polarity in ["Black", "Red"]]


def interaction_quality(events: Sequence[Event]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    grouped_phase_destination = group_by(events, lambda event: (event.row.previous_loop_phase, event.row.destination_family))
    for key in sorted(grouped_phase_destination, key=lambda item: (str(item[0]), str(item[1]))):
        row = summarize_group("EventClass", f"{key[0]} + {key[1]}", grouped_phase_destination[key])
        row["InteractionType"] = "PreviousLoopPhase x DestinationFamily"
        rows.append(row)

    grouped_persistence_destination = group_by(events, lambda event: (event.persistence_group, event.row.destination_family))
    for key in sorted(grouped_persistence_destination, key=lambda item: (str(item[0]), str(item[1]))):
        row = summarize_group("EventClass", f"NeutralPersistence {key[0]} + {key[1]}", grouped_persistence_destination[key])
        row["InteractionType"] = "NeutralPersistenceGroup x DestinationFamily"
        rows.append(row)

    grouped_polarity_destination = group_by(events, lambda event: (event.row.volume_polarity, event.row.destination_family))
    for key in sorted(grouped_polarity_destination, key=lambda item: (str(item[0]), str(item[1]))):
        row = summarize_group("EventClass", f"{key[0]} + {key[1]}", grouped_polarity_destination[key])
        row["InteractionType"] = "VolumePolarity x DestinationFamily"
        rows.append(row)

    return rows


def normalize(value: Optional[float], minimum: float, maximum: float) -> Optional[float]:
    if value is None:
        return None
    if maximum == minimum:
        return 0.5
    return (value - minimum) / (maximum - minimum)


def quality_rankings(interaction_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    candidates = [row for row in interaction_rows if int(row.get("Count", 0)) >= 10]
    if not candidates:
        return []

    dr_values = [float(row["AvgDR5"]) for row in candidates if row.get("AvgDR5") is not None]
    win_values = [float(row["WinRate5"]) for row in candidates if row.get("WinRate5") is not None]
    ratio_values = [float(row["MfeMaeRatio5"]) for row in candidates if row.get("MfeMaeRatio5") is not None]
    if not dr_values or not win_values or not ratio_values:
        return []

    min_dr, max_dr = min(dr_values), max(dr_values)
    min_win, max_win = min(win_values), max(win_values)
    min_ratio, max_ratio = min(ratio_values), max(ratio_values)

    ranked: List[Dict[str, object]] = []
    for row in candidates:
        norm_dr = normalize(row.get("AvgDR5"), min_dr, max_dr)
        norm_win = normalize(row.get("WinRate5"), min_win, max_win)
        norm_ratio = normalize(row.get("MfeMaeRatio5"), min_ratio, max_ratio)
        if norm_dr is None or norm_win is None or norm_ratio is None:
            continue
        quality_score = 0.40 * norm_dr + 0.40 * norm_win + 0.20 * norm_ratio
        ranked_row = dict(row)
        ranked_row["NormalizedAvgDR5"] = norm_dr
        ranked_row["NormalizedWinRate5"] = norm_win
        ranked_row["NormalizedMfeMaeRatio5"] = norm_ratio
        ranked_row["QualityScore"] = quality_score
        ranked.append(ranked_row)

    ranked.sort(key=lambda item: item["QualityScore"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["Rank"] = rank
    return ranked


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


def format_csv_value(value: object) -> object:
    if isinstance(value, float):
        return format_number(value)
    if value is None:
        return "N/A"
    return value


def append_table(lines: List[str], rows: Sequence[Dict[str, object]], limit: Optional[int] = None) -> None:
    if limit is not None:
        rows = rows[:limit]
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


def strongest_dimension(rows_by_dimension: Sequence[Tuple[str, Sequence[Dict[str, object]]]]) -> str:
    spreads: List[Tuple[str, float]] = []
    for name, rows in rows_by_dimension:
        values = [float(row["AvgDR5"]) for row in rows if int(row.get("Count", 0)) >= 10 and row.get("AvgDR5") is not None]
        if len(values) >= 2:
            spreads.append((name, max(values) - min(values)))
    if not spreads:
        return "N/A"
    return max(spreads, key=lambda item: item[1])[0]


def classify_quality(ranked_rows: Sequence[Dict[str, object]]) -> str:
    if not ranked_rows:
        return "NoQualityClusters"
    top = float(ranked_rows[0]["QualityScore"])
    bottom = float(ranked_rows[-1]["QualityScore"])
    spread = top - bottom
    if spread >= 0.70:
        return "StrongQualityClusters"
    if spread >= 0.45:
        return "ModerateQualityClusters"
    if spread >= 0.25:
        return "WeakQualityClusters"
    return "NoQualityClusters"


def recommendation(classification: str) -> str:
    if classification in {"ModerateQualityClusters", "StrongQualityClusters"}:
        return "ReplayValidator04_EventClusters"
    if classification == "WeakQualityClusters":
        return "ReplayValidator04_Symmetry"
    return "CollectMoreData"


def write_report(
    path: str,
    files: Sequence[str],
    rows: Sequence[Row],
    events: Sequence[Event],
    persistence_rows: Sequence[Dict[str, object]],
    phase_rows: Sequence[Dict[str, object]],
    destination_rows: Sequence[Dict[str, object]],
    score_rows: Sequence[Dict[str, object]],
    polarity_rows: Sequence[Dict[str, object]],
    interaction_rows: Sequence[Dict[str, object]],
    ranking_rows: Sequence[Dict[str, object]],
) -> None:
    instruments = sorted({row.instrument for row in rows if row.instrument})
    timestamps = [row.timestamp for row in rows if row.timestamp]
    quality_class = classify_quality(ranking_rows)
    next_step = recommendation(quality_class)
    strongest = strongest_dimension(
        [
            ("DestinationFamily", destination_rows),
            ("NeutralPersistence", persistence_rows),
            ("PreviousLoopPhase", phase_rows),
            ("VolumePolarity", polarity_rows),
            ("FailureWarningScore", score_rows),
        ]
    )

    lines: List[str] = []
    lines.append("APVA Replay Validator 03 - Event Quality")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Purpose")
    lines.append("Identify APVA DestinationSelection event characteristics associated with strongest directional behavior.")
    lines.append("Research only. No trading system. No optimization. No fitting. No ML.")
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

    lines.append("SECTION 1 - Event Quality Dimensions")
    lines.append("-" * 78)
    lines.append("Captured: DestinationFamily, FailureWarningScore, PreviousLoopPhase, VolumePolarity,")
    lines.append("NeutralPersistenceBars, StructuralState, AgeBucket.")
    lines.append("")

    lines.append("SECTION 2 - Neutral Persistence Study")
    lines.append("-" * 78)
    append_table(lines, persistence_rows)
    lines.append("")

    lines.append("SECTION 3 - LateNeutral Quality Study")
    lines.append("-" * 78)
    append_table(lines, phase_rows)
    lines.append("")

    lines.append("SECTION 4 - Destination Quality Study")
    lines.append("-" * 78)
    append_table(lines, destination_rows)
    lines.append("")

    lines.append("SECTION 5 - Failure Score Study")
    lines.append("-" * 78)
    append_table(lines, score_rows)
    lines.append("")

    lines.append("SECTION 6 - Polarity Study")
    lines.append("-" * 78)
    append_table(lines, polarity_rows)
    lines.append("")

    lines.append("SECTION 7 - Interaction Analysis")
    lines.append("-" * 78)
    append_table(lines, interaction_rows)
    lines.append("")

    lines.append("SECTION 8 - Quality Score")
    lines.append("-" * 78)
    lines.append("QualityScore = 0.40 * normalized AvgDR5 + 0.40 * normalized WinRate5 + 0.20 * normalized MFE5/abs(MAE5)")
    lines.append("Ranking only. No threshold recommendations.")
    append_table(lines, ranking_rows)
    lines.append("")

    lines.append("SECTION 9 - Best Event Classes")
    lines.append("-" * 78)
    append_table(lines, ranking_rows[:10])
    lines.append("")

    lines.append("SECTION 10 - Worst Event Classes")
    lines.append("-" * 78)
    append_table(lines, list(reversed(ranking_rows[-10:])))
    lines.append("")

    lines.append("SECTION 11 - Summary")
    lines.append("-" * 78)
    lines.append(f"Conclusion: {quality_class}")
    lines.append(f"Recommended next step: {next_step}")
    lines.append(f"Most separating dimension by AvgDR5 spread: {strongest}")
    if ranking_rows:
        lines.append(f"Top event class: {ranking_rows[0]['EventClass']} QualityScore={format_number(ranking_rows[0]['QualityScore'])}")
        lines.append(f"Bottom event class: {ranking_rows[-1]['EventClass']} QualityScore={format_number(ranking_rows[-1]['QualityScore'])}")
    lines.append("APVA contains high-quality subsets if interpretable classes separate by DR, win rate, and MFE/MAE ratio.")
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

    rows = load_rows(files)
    events = extract_events(rows)
    persistence_rows = neutral_persistence_quality(events)
    phase_rows = previous_phase_quality(events)
    destination_rows = destination_quality(events)
    score_rows = failure_score_quality(events)
    polarity_rows = polarity_quality(events)
    interaction_rows = interaction_quality(events)
    ranking_rows = quality_rankings(interaction_rows)

    output_dir = os.path.dirname(files[0]) if files else DEFAULT_INPUT_DIR
    write_csv(os.path.join(output_dir, EVENT_QUALITY_CSV), ranking_rows)
    write_csv(os.path.join(output_dir, DESTINATION_QUALITY_CSV), destination_rows)
    write_csv(os.path.join(output_dir, NEUTRAL_PERSISTENCE_CSV), persistence_rows)
    write_csv(os.path.join(output_dir, INTERACTION_QUALITY_CSV), interaction_rows)
    write_report(
        os.path.join(output_dir, OUTPUT_TEXT),
        files,
        rows,
        events,
        persistence_rows,
        phase_rows,
        destination_rows,
        score_rows,
        polarity_rows,
        interaction_rows,
        ranking_rows,
    )

    print(f"Wrote {os.path.join(output_dir, OUTPUT_TEXT)}")
    print(f"Wrote {os.path.join(output_dir, EVENT_QUALITY_CSV)}")
    print(f"Wrote {os.path.join(output_dir, DESTINATION_QUALITY_CSV)}")
    print(f"Wrote {os.path.join(output_dir, NEUTRAL_PERSISTENCE_CSV)}")
    print(f"Wrote {os.path.join(output_dir, INTERACTION_QUALITY_CSV)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
