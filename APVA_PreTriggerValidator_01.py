#!/usr/bin/env python3
"""
APVA PreTrigger Validator 01

Direct behavioral validation of APVA_V11_PreTrigger.csv.

This validates PreTriggerState itself as the event. It does not anchor on
DestinationSelection and does not use optimization, fitting, ML, entries, exits,
or profit metrics.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_INPUT = r"C:\Users\rz0\Documents\ApvaAnalysis\NT8Export\APVA_V11_PreTrigger.csv"
OUTPUT_TEXT = "APVA_PreTriggerValidator_01.txt"
STATE_BEHAVIOR_CSV = "PreTriggerStateBehavior.csv"
INTERACTION_BEHAVIOR_CSV = "PreTriggerInteractionBehavior.csv"

HORIZONS = [1, 2, 3, 5, 10]
PRE_TRIGGER_STATES = ["CandidateB", "CandidateC", "CandidateBC"]
PRE_TRIGGER_SCORES = [1, 2]
MIN_INTERPRETABLE = 20


@dataclass
class Row:
    timestamp: str
    instrument: str
    close: Optional[float]
    loop_phase: str
    destination_family: str
    pre_trigger_state: str
    pre_trigger_score: Optional[int]
    volume_polarity: str
    prior_neutral_persistence_bars: Optional[int]


@dataclass
class Event:
    index: int
    row: Row
    dr: Dict[int, Optional[float]]
    mfe5: Optional[float]
    mae5: Optional[float]
    prior_persistence_group: str


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


def input_path(args: Sequence[str]) -> str:
    if len(args) > 1:
        return args[1]
    return DEFAULT_INPUT


def load_rows(path: str) -> List[Row]:
    required = {
        "Timestamp",
        "Instrument",
        "Close",
        "LoopPhase",
        "DestinationFamily",
        "PreTriggerState",
        "PreTriggerScore",
        "VolumePolarity",
        "PriorNeutralPersistenceBars",
    }
    rows: List[Row] = []
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")
        for raw in reader:
            rows.append(
                Row(
                    timestamp=clean_text(raw.get("Timestamp")),
                    instrument=clean_text(raw.get("Instrument")),
                    close=parse_float(raw.get("Close")),
                    loop_phase=clean_text(raw.get("LoopPhase")),
                    destination_family=clean_text(raw.get("DestinationFamily")),
                    pre_trigger_state=clean_text(raw.get("PreTriggerState")),
                    pre_trigger_score=parse_int(raw.get("PreTriggerScore")),
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
    if index + 5 >= len(rows):
        return None, None
    start = rows[index].close
    future = [rows[index + offset].close for offset in range(1, 6)]
    if start is None or any(value is None for value in future):
        return None, None
    future_values = [value for value in future if value is not None]
    if polarity == "Black":
        return max(future_values) - start, min(future_values) - start
    if polarity == "Red":
        return start - min(future_values), start - max(future_values)
    return None, None


def extract_events(rows: Sequence[Row]) -> List[Event]:
    events: List[Event] = []
    for index, row in enumerate(rows):
        if row.pre_trigger_state == "None":
            continue
        if row.pre_trigger_state not in PRE_TRIGGER_STATES:
            continue
        if row.volume_polarity not in {"Black", "Red"}:
            continue
        if row.close is None or index + 10 >= len(rows):
            continue
        dr = {horizon: directional_return(rows, index, horizon, row.volume_polarity) for horizon in HORIZONS}
        if dr[10] is None:
            continue
        mfe5, mae5 = mfe_mae_5(rows, index, row.volume_polarity)
        events.append(
            Event(
                index=index,
                row=row,
                dr=dr,
                mfe5=mfe5,
                mae5=mae5,
                prior_persistence_group=persistence_group(row.prior_neutral_persistence_bars),
            )
        )
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


def mfe_mae_ratio(events: Sequence[Event]) -> Optional[float]:
    avg_mfe = mean(event.mfe5 for event in events)
    avg_mae = mean(event.mae5 for event in events)
    if avg_mfe is None or avg_mae in {None, 0}:
        return None
    return avg_mfe / abs(avg_mae)


def sample_class(count: int) -> str:
    return "Interpretable" if count >= MIN_INTERPRETABLE else "LowSample"


def behavior_row(group_type: str, group: str, events: Sequence[Event]) -> Dict[str, object]:
    return {
        "GroupType": group_type,
        "Group": group,
        "Count": len(events),
        "AvgDR1": mean(event.dr[1] for event in events),
        "AvgDR2": mean(event.dr[2] for event in events),
        "AvgDR3": mean(event.dr[3] for event in events),
        "AvgDR5": mean(event.dr[5] for event in events),
        "AvgDR10": mean(event.dr[10] for event in events),
        "WinRate1": win_rate(event.dr[1] for event in events),
        "WinRate3": win_rate(event.dr[3] for event in events),
        "WinRate5": win_rate(event.dr[5] for event in events),
        "WinRate10": win_rate(event.dr[10] for event in events),
        "AvgMFE5": mean(event.mfe5 for event in events),
        "AvgMAE5": mean(event.mae5 for event in events),
        "MfeMaeRatio5": mfe_mae_ratio(events),
        "SampleClass": sample_class(len(events)),
    }


def group_by(events: Sequence[Event], key_fn) -> Dict[object, List[Event]]:
    grouped: Dict[object, List[Event]] = defaultdict(list)
    for event in events:
        grouped[key_fn(event)].append(event)
    return grouped


def state_behavior(events: Sequence[Event]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    by_state = group_by(events, lambda event: event.row.pre_trigger_state)
    for state in PRE_TRIGGER_STATES:
        rows.append(behavior_row("PreTriggerState", state, by_state.get(state, [])))

    by_score = group_by(events, lambda event: event.row.pre_trigger_score)
    for score in PRE_TRIGGER_SCORES:
        rows.append(behavior_row("PreTriggerScore", str(score), by_score.get(score, [])))
    return rows


def interaction_behavior(events: Sequence[Event]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    group_specs = [
        ("PreTriggerState x VolumePolarity", lambda event: (event.row.pre_trigger_state, event.row.volume_polarity)),
        ("PreTriggerState x LoopPhase", lambda event: (event.row.pre_trigger_state, event.row.loop_phase)),
        ("PreTriggerState x PriorNeutralPersistenceGroup", lambda event: (event.row.pre_trigger_state, event.prior_persistence_group)),
        ("PreTriggerState x DestinationFamily", lambda event: (event.row.pre_trigger_state, event.row.destination_family or "N/A")),
    ]
    for group_type, key_fn in group_specs:
        grouped = group_by(events, key_fn)
        for key in sorted(grouped, key=lambda item: " + ".join(str(part) for part in item)):
            label = " + ".join(str(part) for part in key)
            rows.append(behavior_row(group_type, label, grouped[key]))
    return rows


def conclusion(rows: Sequence[Dict[str, object]]) -> str:
    interpretable = [
        row for row in rows
        if row.get("SampleClass") == "Interpretable"
        and row.get("AvgDR5") is not None
        and row.get("WinRate5") is not None
        and row.get("MfeMaeRatio5") is not None
    ]
    if any(row["AvgDR5"] > 20 and row["WinRate5"] >= 0.65 and row["MfeMaeRatio5"] > 1.5 for row in interpretable):
        return "StrongPreTriggerSignal"
    if any(row["AvgDR5"] > 10 and row["WinRate5"] >= 0.58 and row["MfeMaeRatio5"] > 1.2 for row in interpretable):
        return "ModeratePreTriggerSignal"
    if any(row["AvgDR5"] is not None and row["AvgDR5"] > 0 for row in interpretable):
        return "WeakPreTriggerSignal"
    return "PreTriggerFails"


def format_number(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def stringify(value: object) -> str:
    if isinstance(value, float):
        return format_number(value)
    if value is None:
        return "N/A"
    return str(value)


def write_csv(path: str, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: stringify(row.get(field)) for field in fields})


def render_table(rows: Sequence[Dict[str, object]], fields: Sequence[str], limit: Optional[int] = None) -> str:
    selected = list(rows if limit is None else rows[:limit])
    if not selected:
        return "(no rows)"
    rendered = [{field: stringify(row.get(field)) for field in fields} for row in selected]
    widths = {field: max(len(field), *(len(row[field]) for row in rendered)) for field in fields}
    header = " | ".join(field.ljust(widths[field]) for field in fields)
    divider = "-+-".join("-" * widths[field] for field in fields)
    lines = [header, divider]
    for row in rendered:
        lines.append(" | ".join(row[field].ljust(widths[field]) for field in fields))
    return "\n".join(lines)


def sort_best(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            row.get("SampleClass") == "Interpretable",
            row.get("AvgDR5") if row.get("AvgDR5") is not None else float("-inf"),
            row.get("WinRate5") if row.get("WinRate5") is not None else float("-inf"),
            row.get("MfeMaeRatio5") if row.get("MfeMaeRatio5") is not None else float("-inf"),
        ),
        reverse=True,
    )


def build_report(
    input_file: str,
    rows: Sequence[Row],
    events: Sequence[Event],
    state_rows: Sequence[Dict[str, object]],
    interaction_rows: Sequence[Dict[str, object]],
    classification: str,
) -> List[str]:
    instruments = sorted({row.instrument for row in rows if row.instrument})
    timestamps = [row.timestamp for row in rows if row.timestamp]
    fields = [
        "GroupType",
        "Group",
        "Count",
        "AvgDR1",
        "AvgDR2",
        "AvgDR3",
        "AvgDR5",
        "AvgDR10",
        "WinRate1",
        "WinRate3",
        "WinRate5",
        "WinRate10",
        "AvgMFE5",
        "AvgMAE5",
        "MfeMaeRatio5",
        "SampleClass",
    ]
    lines: List[str] = []
    lines.append("APVA PreTrigger Validator 01")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Purpose")
    lines.append("Validate APVA_V11_PreTrigger.csv directly, using PreTriggerState as the event.")
    lines.append("No DestinationSelection anchoring. Behavioral validation only.")
    lines.append("")
    lines.append("Input")
    lines.append("-" * 78)
    lines.append(f"File: {input_file}")
    lines.append(f"Rows read: {len(rows)}")
    lines.append(f"PreTrigger events: {len(events)}")
    lines.append(f"Instruments: {', '.join(instruments) if instruments else 'N/A'}")
    lines.append(f"Date range: {min(timestamps) if timestamps else 'N/A'} to {max(timestamps) if timestamps else 'N/A'}")
    lines.append("")
    lines.append("SECTION 1 - PreTrigger State / Score Behavior")
    lines.append("-" * 78)
    lines.append(render_table(state_rows, fields))
    lines.append("")
    lines.append("SECTION 2 - Interaction Behavior")
    lines.append("-" * 78)
    lines.append(render_table(sort_best(interaction_rows), fields, limit=40))
    lines.append("")
    lines.append("SECTION 3 - Quality Decision")
    lines.append("-" * 78)
    lines.append(f"Conclusion: {classification}")
    lines.append("Strong rule: Count >= 20, AvgDR5 > 20, WinRate5 >= 0.65, MfeMaeRatio5 > 1.5")
    lines.append("Moderate rule: Count >= 20, AvgDR5 > 10, WinRate5 >= 0.58, MfeMaeRatio5 > 1.2")
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
    return lines


def main(argv: Sequence[str]) -> int:
    path = input_path(argv)
    if not os.path.exists(path):
        print(f"Missing input file: {path}", file=sys.stderr)
        return 1

    rows = load_rows(path)
    events = extract_events(rows)
    if not events:
        print("No PreTriggerState events found.", file=sys.stderr)
        return 1

    state_rows = state_behavior(events)
    interaction_rows = interaction_behavior(events)
    classification = conclusion(list(state_rows) + list(interaction_rows))

    output_dir = os.path.dirname(path)
    fields = [
        "GroupType",
        "Group",
        "Count",
        "AvgDR1",
        "AvgDR2",
        "AvgDR3",
        "AvgDR5",
        "AvgDR10",
        "WinRate1",
        "WinRate3",
        "WinRate5",
        "WinRate10",
        "AvgMFE5",
        "AvgMAE5",
        "MfeMaeRatio5",
        "SampleClass",
    ]
    write_csv(os.path.join(output_dir, STATE_BEHAVIOR_CSV), state_rows, fields)
    write_csv(os.path.join(output_dir, INTERACTION_BEHAVIOR_CSV), interaction_rows, fields)

    report_path = os.path.join(output_dir, OUTPUT_TEXT)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(build_report(path, rows, events, state_rows, interaction_rows, classification)) + "\n")

    print(f"Wrote {report_path}")
    print(f"Wrote {os.path.join(output_dir, STATE_BEHAVIOR_CSV)}")
    print(f"Wrote {os.path.join(output_dir, INTERACTION_BEHAVIOR_CSV)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
