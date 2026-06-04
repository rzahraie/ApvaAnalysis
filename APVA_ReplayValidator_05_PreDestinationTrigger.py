#!/usr/bin/env python3
"""
APVA Replay Validator 05 - PreDestination Trigger Study

Behavioral validation of whether APVA contains causal pre-DestinationSelection
conditions one or two bars before the runtime DestinationSelection event.

No trading system. No optimization. No fitting. No ML. No entries. No exits.
"""

from __future__ import annotations

import csv
import glob
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence


DEFAULT_INPUT_DIR = r"C:\Users\rz0\Documents\ApvaAnalysis\NT8Export"
OUTPUT_TEXT = "APVA_ReplayValidator_05_PreDestinationTrigger.txt"
TIMING_CSV = "PreDestinationTimingSummary.csv"
FEATURE_CSV = "PreDestinationFeatureFrequency.csv"
CANDIDATE_CSV = "PreDestinationCandidateResults.csv"
RED_DIAGNOSIS_CSV = "PreDestinationRedDiagnosis.csv"
BLACK_DIAGNOSIS_CSV = "PreDestinationBlackDiagnosis.csv"

HORIZONS = [5, 10]
ANCHORS = [("T0", 0), ("Tminus1", -1), ("Tminus2", -2)]
POLARITIES = ["All", "Black", "Red"]
PRESSURE_FIELDS = [
    "RangePressure",
    "BodyPressure",
    "VolumePressure",
    "EfficiencyPressure",
    "CloseExtremePressure",
    "PolarityInstability",
]
PRESSURE_ATTRS = {
    "RangePressure": "range_pressure",
    "BodyPressure": "body_pressure",
    "VolumePressure": "volume_pressure",
    "EfficiencyPressure": "efficiency_pressure",
    "CloseExtremePressure": "close_extreme_pressure",
    "PolarityInstability": "polarity_instability",
}
PERSISTENCE_GROUPS = ["1-2", "3-4", "5-6", "7-10", "11+", "Unknown"]


@dataclass
class Row:
    timestamp: str
    instrument: str
    close: Optional[float]
    loop_phase: str
    structural_state: str
    destination_family: str
    failure_warning_score: Optional[int]
    range_pressure: bool
    body_pressure: bool
    volume_pressure: bool
    efficiency_pressure: bool
    close_extreme_pressure: bool
    polarity_instability: bool
    neutral_persistence_bars: Optional[int]
    prior_neutral_persistence_bars: Optional[int]
    volume_polarity: str


@dataclass
class DestinationEvent:
    t0_index: int
    t0: Row


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


def parse_bool(value: object) -> bool:
    text = clean_text(value).lower()
    return text in {"true", "1", "yes", "y"}


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
    required = {
        "Timestamp",
        "Instrument",
        "Close",
        "LoopPhase",
        "StructuralState",
        "DestinationFamily",
        "FailureWarningScore",
        "RangePressure",
        "BodyPressure",
        "VolumePressure",
        "EfficiencyPressure",
        "CloseExtremePressure",
        "PolarityInstability",
        "NeutralPersistenceBars",
        "PriorNeutralPersistenceBars",
        "VolumePolarity",
    }
    for path in files:
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
                        structural_state=clean_text(raw.get("StructuralState")),
                        destination_family=clean_text(raw.get("DestinationFamily")),
                        failure_warning_score=parse_int(raw.get("FailureWarningScore")),
                        range_pressure=parse_bool(raw.get("RangePressure")),
                        body_pressure=parse_bool(raw.get("BodyPressure")),
                        volume_pressure=parse_bool(raw.get("VolumePressure")),
                        efficiency_pressure=parse_bool(raw.get("EfficiencyPressure")),
                        close_extreme_pressure=parse_bool(raw.get("CloseExtremePressure")),
                        polarity_instability=parse_bool(raw.get("PolarityInstability")),
                        neutral_persistence_bars=parse_int(raw.get("NeutralPersistenceBars")),
                        prior_neutral_persistence_bars=parse_int(raw.get("PriorNeutralPersistenceBars")),
                        volume_polarity=clean_text(raw.get("VolumePolarity")),
                    )
                )
    return rows


def extract_events(rows: Sequence[Row]) -> List[DestinationEvent]:
    events: List[DestinationEvent] = []
    for index, row in enumerate(rows):
        if row.loop_phase != "DestinationSelection":
            continue
        if row.volume_polarity not in {"Black", "Red"}:
            continue
        if index < 2 or index + 10 >= len(rows):
            continue
        if any(rows[index + offset].close is None for offset in range(-2, 11)):
            continue
        events.append(DestinationEvent(t0_index=index, t0=row))
    return events


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


def raw_return(rows: Sequence[Row], anchor_index: int, horizon: int) -> Optional[float]:
    if anchor_index < 0 or anchor_index + horizon >= len(rows):
        return None
    start = rows[anchor_index].close
    end = rows[anchor_index + horizon].close
    if start is None or end is None:
        return None
    return end - start


def directional_return(
    rows: Sequence[Row],
    anchor_index: int,
    horizon: int,
    t0_polarity: str,
    invert_red: bool = False,
) -> Optional[float]:
    raw = raw_return(rows, anchor_index, horizon)
    if raw is None:
        return None
    if t0_polarity == "Black":
        return raw
    if t0_polarity == "Red":
        return raw if invert_red else -raw
    return None


def mfe_mae_5(
    rows: Sequence[Row],
    anchor_index: int,
    t0_polarity: str,
    invert_red: bool = False,
) -> tuple[Optional[float], Optional[float]]:
    if anchor_index < 0 or anchor_index + 5 >= len(rows):
        return None, None
    start = rows[anchor_index].close
    future = [rows[anchor_index + offset].close for offset in range(1, 6)]
    if start is None or any(value is None for value in future):
        return None, None
    future_values = [value for value in future if value is not None]
    if t0_polarity == "Black":
        return max(future_values) - start, min(future_values) - start
    if t0_polarity == "Red":
        if invert_red:
            return max(future_values) - start, min(future_values) - start
        return start - min(future_values), start - max(future_values)
    return None, None


def group_events(events: Sequence[DestinationEvent], key_fn: Callable[[DestinationEvent], object]) -> Dict[object, List[DestinationEvent]]:
    grouped: Dict[object, List[DestinationEvent]] = defaultdict(list)
    for event in events:
        grouped[key_fn(event)].append(event)
    return grouped


def selected_events_by_polarity(events: Sequence[DestinationEvent], polarity: str) -> List[DestinationEvent]:
    if polarity == "All":
        return list(events)
    return [event for event in events if event.t0.volume_polarity == polarity]


def mfa_ratio(avg_mfe: Optional[float], avg_mae: Optional[float]) -> Optional[float]:
    if avg_mfe is None or avg_mae in {None, 0}:
        return None
    return avg_mfe / abs(avg_mae)


def behavior_row(
    label_values: Dict[str, object],
    rows: Sequence[Row],
    events: Sequence[DestinationEvent],
    anchor_offset: int,
    invert_red: bool = False,
) -> Dict[str, object]:
    dr5 = [directional_return(rows, event.t0_index + anchor_offset, 5, event.t0.volume_polarity, invert_red) for event in events]
    dr10 = [directional_return(rows, event.t0_index + anchor_offset, 10, event.t0.volume_polarity, invert_red) for event in events]
    mfe_mae = [
        mfe_mae_5(rows, event.t0_index + anchor_offset, event.t0.volume_polarity, invert_red)
        for event in events
    ]
    avg_mfe = mean(mfe for mfe, _ in mfe_mae)
    avg_mae = mean(mae for _, mae in mfe_mae)
    row: Dict[str, object] = dict(label_values)
    row.update(
        {
            "Count": len(events),
            "AvgDR5": mean(dr5),
            "AvgDR10": mean(dr10),
            "WinRate5": win_rate(dr5),
            "WinRate10": win_rate(dr10),
            "AvgMFE5": avg_mfe,
            "AvgMAE5": avg_mae,
            "MfeMaeRatio5": mfa_ratio(avg_mfe, avg_mae),
        }
    )
    return row


def timing_summary(rows: Sequence[Row], events: Sequence[DestinationEvent]) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    for anchor_name, offset in ANCHORS:
        for polarity in POLARITIES:
            selected = selected_events_by_polarity(events, polarity)
            result.append(behavior_row({"Anchor": anchor_name, "VolumePolarity": polarity}, rows, selected, offset))
    return result


def anchor_row(rows: Sequence[Row], event: DestinationEvent, anchor_offset: int) -> Row:
    return rows[event.t0_index + anchor_offset]


def feature_key_values(rows: Sequence[Row], event: DestinationEvent, anchor_offset: int) -> List[tuple[str, str]]:
    row = anchor_row(rows, event, anchor_offset)
    values: List[tuple[str, str]] = [
        ("LoopPhase", row.loop_phase or "Unknown"),
        ("StructuralState", row.structural_state or "Unknown"),
        ("FailureWarningScore", str(row.failure_warning_score) if row.failure_warning_score is not None else "Unknown"),
        ("VolumePolarityMatchWithT0", str(row.volume_polarity == event.t0.volume_polarity)),
        ("PriorNeutralPersistenceGroup", persistence_group(row.prior_neutral_persistence_bars)),
        ("NeutralPersistenceGroup", persistence_group(row.neutral_persistence_bars)),
    ]
    for field in PRESSURE_FIELDS:
        attr = PRESSURE_ATTRS[field]
        values.append((field, str(bool(getattr(row, attr)))))
    return values


def feature_frequency(rows: Sequence[Row], events: Sequence[DestinationEvent]) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    for anchor_name, offset in [("Tminus1", -1), ("Tminus2", -2)]:
        winner_events = [
            event for event in events
            if (directional_return(rows, event.t0_index + offset, 5, event.t0.volume_polarity) or 0) > 0
        ]
        loser_events = [
            event for event in events
            if (directional_return(rows, event.t0_index + offset, 5, event.t0.volume_polarity) or 0) <= 0
        ]
        observed: set[tuple[str, str]] = set()
        for event in events:
            observed.update(feature_key_values(rows, event, offset))
        for feature, value in sorted(observed):
            winner_count = sum(1 for event in winner_events if (feature, value) in feature_key_values(rows, event, offset))
            loser_count = sum(1 for event in loser_events if (feature, value) in feature_key_values(rows, event, offset))
            winner_frequency = winner_count / len(winner_events) if winner_events else None
            loser_frequency = loser_count / len(loser_events) if loser_events else None
            difference = None
            if winner_frequency is not None and loser_frequency is not None:
                difference = winner_frequency - loser_frequency
            result.append(
                {
                    "Anchor": anchor_name,
                    "Feature": feature,
                    "Value": value,
                    "WinnerFrequency": winner_frequency,
                    "LoserFrequency": loser_frequency,
                    "Difference": difference,
                    "WinnerCount": winner_count,
                    "LoserCount": loser_count,
                }
            )
    return sorted(result, key=lambda row: (row["Anchor"], -abs(row["Difference"] or 0), row["Feature"], row["Value"]))


def candidate_a(row: Row, _event: DestinationEvent) -> bool:
    return row.loop_phase in {"LateNeutral", "NeutralMaturation"} and (row.failure_warning_score or 0) >= 3


def candidate_b(row: Row, _event: DestinationEvent) -> bool:
    return (row.failure_warning_score or 0) >= 4 and row.range_pressure and row.volume_pressure


def candidate_c(row: Row, _event: DestinationEvent) -> bool:
    return (row.failure_warning_score or 0) >= 4 and row.body_pressure and row.efficiency_pressure


def candidate_d(row: Row, _event: DestinationEvent) -> bool:
    return (row.prior_neutral_persistence_bars or 0) >= 5 or (row.neutral_persistence_bars or 0) >= 5


def candidate_e(row: Row, event: DestinationEvent) -> bool:
    return (
        row.loop_phase == "LateNeutral"
        and (row.failure_warning_score or 0) >= 3
        and row.volume_polarity == event.t0.volume_polarity
    )


def candidate_f(row: Row, _event: DestinationEvent) -> bool:
    return row.loop_phase == "NeutralMaturation" and (row.failure_warning_score or 0) >= 4 and row.range_pressure


CANDIDATES: List[tuple[str, str, Callable[[Row, DestinationEvent], bool]]] = [
    ("CandidateA", "LoopPhase in LateNeutral/NeutralMaturation AND FailureWarningScore >= 3", candidate_a),
    ("CandidateB", "FailureWarningScore >= 4 AND RangePressure AND VolumePressure", candidate_b),
    ("CandidateC", "FailureWarningScore >= 4 AND BodyPressure AND EfficiencyPressure", candidate_c),
    ("CandidateD", "PriorNeutralPersistenceBars >= 5 OR NeutralPersistenceBars >= 5", candidate_d),
    ("CandidateE", "LateNeutral AND FailureWarningScore >= 3 AND anchor polarity matches T0", candidate_e),
    ("CandidateF", "NeutralMaturation AND FailureWarningScore >= 4 AND RangePressure", candidate_f),
]


def candidate_results(rows: Sequence[Row], events: Sequence[DestinationEvent]) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    for anchor_name, offset in [("Tminus1", -1), ("Tminus2", -2)]:
        for candidate_name, description, predicate in CANDIDATES:
            matched = [
                event for event in events
                if predicate(anchor_row(rows, event, offset), event)
            ]
            for polarity in POLARITIES:
                selected = selected_events_by_polarity(matched, polarity)
                row = behavior_row(
                    {
                        "Anchor": anchor_name,
                        "Candidate": candidate_name,
                        "Condition": description,
                        "VolumePolarity": polarity,
                    },
                    rows,
                    selected,
                    offset,
                )
                result.append(row)
    return result


def polarity_diagnosis(rows: Sequence[Row], events: Sequence[DestinationEvent], polarity: str) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    selected = selected_events_by_polarity(events, polarity)
    for anchor_name, offset in [("Tminus1", -1), ("Tminus2", -2)]:
        normal = behavior_row({"Anchor": anchor_name}, rows, selected, offset, invert_red=False)
        inverted = behavior_row({"Anchor": anchor_name}, rows, selected, offset, invert_red=True)
        result.append(
            {
                "VolumePolarity": polarity,
                "Anchor": anchor_name,
                "NormalAvgDR5": normal["AvgDR5"],
                "InvertedAvgDR5": inverted["AvgDR5"],
                "NormalWinRate5": normal["WinRate5"],
                "InvertedWinRate5": inverted["WinRate5"],
                "NormalAvgDR10": normal["AvgDR10"],
                "InvertedAvgDR10": inverted["AvgDR10"],
                "NormalWinRate10": normal["WinRate10"],
                "InvertedWinRate10": inverted["WinRate10"],
            }
        )
    return result


def best_candidate_rows(candidate_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = [
        row for row in candidate_rows
        if row.get("VolumePolarity") == "All" and isinstance(row.get("Count"), int) and row["Count"] >= 20
    ]
    return sorted(
        rows,
        key=lambda row: (
            row.get("AvgDR5") if row.get("AvgDR5") is not None else float("-inf"),
            row.get("WinRate5") if row.get("WinRate5") is not None else float("-inf"),
            row.get("MfeMaeRatio5") if row.get("MfeMaeRatio5") is not None else float("-inf"),
        ),
        reverse=True,
    )


def red_proxy_invalid(red_diagnosis: Sequence[Dict[str, object]]) -> bool:
    checks = []
    for row in red_diagnosis:
        normal = row.get("NormalAvgDR5")
        inverted = row.get("InvertedAvgDR5")
        normal_wr = row.get("NormalWinRate5")
        inverted_wr = row.get("InvertedWinRate5")
        if normal is None or inverted is None or normal_wr is None or inverted_wr is None:
            continue
        checks.append(inverted > normal and inverted_wr > normal_wr)
    return bool(checks) and all(checks)


def red_late_but_recoverable(red_diagnosis: Sequence[Dict[str, object]]) -> bool:
    for row in red_diagnosis:
        normal = row.get("NormalAvgDR5")
        normal_wr = row.get("NormalWinRate5")
        if normal is not None and normal_wr is not None and normal > 0 and normal_wr >= 0.50:
            return True
    return False


def pre_destination_signal_found(candidate_rows: Sequence[Dict[str, object]]) -> bool:
    for row in candidate_rows:
        if row.get("VolumePolarity") != "All":
            continue
        count = row.get("Count")
        avg_dr5 = row.get("AvgDR5")
        win_rate5 = row.get("WinRate5")
        if isinstance(count, int) and count >= 20 and avg_dr5 is not None and win_rate5 is not None:
            if avg_dr5 > 0 and win_rate5 >= 0.58:
                return True
    return False


def conclusion_and_recommendation(has_signal: bool, red_invalid: bool) -> tuple[str, str]:
    if not has_signal and red_invalid:
        return "SwitchFrameworkRecommended", "SwitchToContainerLateralGaussian"
    if red_invalid:
        return "DirectionalProxyInvalid", "ReviseVolumePolarity"
    if has_signal:
        return "PreDestinationSignalFound", "ContinueAPVA_PreTrigger"
    return "NoPreDestinationSignal", "SwitchToContainerLateralGaussian"


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


def report_lines(
    files: Sequence[str],
    rows: Sequence[Row],
    events: Sequence[DestinationEvent],
    timing_rows: Sequence[Dict[str, object]],
    feature_rows: Sequence[Dict[str, object]],
    candidate_rows: Sequence[Dict[str, object]],
    red_rows: Sequence[Dict[str, object]],
    black_rows: Sequence[Dict[str, object]],
    conclusion: str,
    recommendation: str,
    has_signal: bool,
    red_invalid: bool,
    red_recoverable: bool,
) -> List[str]:
    instruments = sorted({row.instrument for row in rows if row.instrument})
    timestamps = [row.timestamp for row in rows if row.timestamp]
    top_candidates = best_candidate_rows(candidate_rows)[:10]

    lines: List[str] = []
    lines.append("APVA Replay Validator 05 - PreDestination Trigger Study")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Purpose")
    lines.append("Determine whether APVA contains pre-Destination trigger conditions before DestinationSelection.")
    lines.append("Behavioral validation only. No trading system. No optimization. No fitting. No ML.")
    lines.append("")
    lines.append("Input")
    lines.append("-" * 78)
    lines.append(f"Files: {len(files)}")
    for path in files:
        lines.append(f"  {path}")
    lines.append(f"Rows read: {len(rows)}")
    lines.append(f"Usable DestinationSelection events with Tminus2 and forward 10 bars: {len(events)}")
    lines.append(f"Instruments: {', '.join(instruments) if instruments else 'N/A'}")
    lines.append(f"Date range: {min(timestamps) if timestamps else 'N/A'} to {max(timestamps) if timestamps else 'N/A'}")
    lines.append("")

    timing_fields = ["Anchor", "VolumePolarity", "Count", "AvgDR5", "AvgDR10", "WinRate5", "WinRate10", "AvgMFE5", "AvgMAE5", "MfeMaeRatio5"]
    lines.append("SECTION 1 - Timing Recap")
    lines.append("-" * 78)
    lines.append(render_table(timing_rows, timing_fields))
    lines.append("")

    lines.append("SECTION 2 - Pre-Destination Feature Snapshot")
    lines.append("-" * 78)
    lines.append("Captured at Tminus1 and Tminus2:")
    lines.append("LoopPhase, StructuralState, PriorNeutralPersistenceBars, NeutralPersistenceBars,")
    lines.append("FailureWarningScore, pressure flags, DestinationFamily at T0, VolumePolarity at T0,")
    lines.append("VolumePolarity at anchor, and polarity match.")
    lines.append("")

    feature_fields = ["Anchor", "Feature", "Value", "WinnerFrequency", "LoserFrequency", "Difference", "WinnerCount", "LoserCount"]
    lines.append("SECTION 3 - Feature Frequency at Winning Anchors")
    lines.append("-" * 78)
    lines.append(render_table(feature_rows, feature_fields, limit=40))
    lines.append("")

    candidate_fields = ["Anchor", "Candidate", "VolumePolarity", "Count", "AvgDR5", "AvgDR10", "WinRate5", "WinRate10", "AvgMFE5", "AvgMAE5", "MfeMaeRatio5"]
    lines.append("SECTION 4 - Candidate Pre-Destination Conditions")
    lines.append("-" * 78)
    lines.append(render_table(candidate_rows, candidate_fields, limit=60))
    lines.append("")

    diagnosis_fields = ["VolumePolarity", "Anchor", "NormalAvgDR5", "InvertedAvgDR5", "NormalWinRate5", "InvertedWinRate5", "NormalAvgDR10", "InvertedAvgDR10"]
    lines.append("SECTION 5 - Red Pre-Destination Diagnosis")
    lines.append("-" * 78)
    lines.append(render_table(red_rows, diagnosis_fields))
    lines.append(f"RedDirectionalProxyInvalid: {red_invalid}")
    lines.append(f"RedLateButRecoverable: {red_recoverable}")
    lines.append("")

    lines.append("SECTION 6 - Black Pre-Destination Diagnosis")
    lines.append("-" * 78)
    lines.append(render_table(black_rows, diagnosis_fields))
    lines.append("")

    lines.append("SECTION 7 - Best Pre-Destination Candidate")
    lines.append("-" * 78)
    lines.append(render_table(top_candidates, candidate_fields))
    lines.append(f"Any predefined candidate meets Count >= 20, WinRate5 >= 0.58, AvgDR5 > 0: {has_signal}")
    lines.append("")

    lines.append("SECTION 8 - Kill-Switch Decision")
    lines.append("-" * 78)
    lines.append(f"Conclusion: {conclusion}")
    lines.append(f"Recommended next step: {recommendation}")
    lines.append(f"NoPreDestinationSignal condition: {not has_signal}")
    lines.append(f"DirectionalProxyInvalid condition: {red_invalid}")
    lines.append("")
    lines.append("Low-DoF / Research Audit")
    lines.append("-" * 78)
    lines.append("No fitting.")
    lines.append("No optimization.")
    lines.append("No machine learning.")
    lines.append("No trading rules.")
    lines.append("No entries.")
    lines.append("No exits.")
    lines.append("No profit model.")
    lines.append("Behavioral validation only.")
    return lines


def main(argv: Sequence[str]) -> int:
    files = input_files(argv[1:])
    if not files:
        print(f"No APVA_V10_StateMachine_*.csv files found in {DEFAULT_INPUT_DIR}", file=sys.stderr)
        return 1

    rows = load_rows(files)
    events = extract_events(rows)
    if not events:
        print("No usable DestinationSelection events found.", file=sys.stderr)
        return 1

    output_dir = os.path.dirname(files[0]) if files else DEFAULT_INPUT_DIR
    timing_rows = timing_summary(rows, events)
    feature_rows = feature_frequency(rows, events)
    candidate_rows = candidate_results(rows, events)
    red_rows = polarity_diagnosis(rows, events, "Red")
    black_rows = polarity_diagnosis(rows, events, "Black")

    has_signal = pre_destination_signal_found(candidate_rows)
    red_invalid = red_proxy_invalid(red_rows)
    red_recoverable = red_late_but_recoverable(red_rows)
    conclusion, recommendation = conclusion_and_recommendation(has_signal, red_invalid)

    write_csv(
        os.path.join(output_dir, TIMING_CSV),
        timing_rows,
        ["Anchor", "VolumePolarity", "Count", "AvgDR5", "AvgDR10", "WinRate5", "WinRate10", "AvgMFE5", "AvgMAE5", "MfeMaeRatio5"],
    )
    write_csv(
        os.path.join(output_dir, FEATURE_CSV),
        feature_rows,
        ["Anchor", "Feature", "Value", "WinnerFrequency", "LoserFrequency", "Difference", "WinnerCount", "LoserCount"],
    )
    write_csv(
        os.path.join(output_dir, CANDIDATE_CSV),
        candidate_rows,
        ["Anchor", "Candidate", "Condition", "VolumePolarity", "Count", "AvgDR5", "AvgDR10", "WinRate5", "WinRate10", "AvgMFE5", "AvgMAE5", "MfeMaeRatio5"],
    )
    write_csv(
        os.path.join(output_dir, RED_DIAGNOSIS_CSV),
        red_rows,
        ["VolumePolarity", "Anchor", "NormalAvgDR5", "InvertedAvgDR5", "NormalWinRate5", "InvertedWinRate5", "NormalAvgDR10", "InvertedAvgDR10", "NormalWinRate10", "InvertedWinRate10"],
    )
    write_csv(
        os.path.join(output_dir, BLACK_DIAGNOSIS_CSV),
        black_rows,
        ["VolumePolarity", "Anchor", "NormalAvgDR5", "InvertedAvgDR5", "NormalWinRate5", "InvertedWinRate5", "NormalAvgDR10", "InvertedAvgDR10", "NormalWinRate10", "InvertedWinRate10"],
    )

    lines = report_lines(
        files,
        rows,
        events,
        timing_rows,
        feature_rows,
        candidate_rows,
        red_rows,
        black_rows,
        conclusion,
        recommendation,
        has_signal,
        red_invalid,
        red_recoverable,
    )
    report_path = os.path.join(output_dir, OUTPUT_TEXT)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    for filename in [OUTPUT_TEXT, TIMING_CSV, FEATURE_CSV, CANDIDATE_CSV, RED_DIAGNOSIS_CSV, BLACK_DIAGNOSIS_CSV]:
        print(f"Wrote {os.path.join(output_dir, filename)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
