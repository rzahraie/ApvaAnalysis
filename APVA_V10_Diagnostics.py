#!/usr/bin/env python3
"""
APVA v1.0 Diagnostics Analyzer.

Behavior validation for CSV exports produced by xApvaV10StateMachine.cs.
This is not a discovery study and does not define new APVA theory.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_INPUT_DIR = Path(r"C:\Users\rz0\Documents\ApvaAnalysis\NT8Export")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "Diagnostics"

LOOP_PHASES = [
    "NeutralFormation",
    "NeutralMaturation",
    "LateNeutral",
    "DestinationSelection",
    "Excursion",
    "ReturnToNeutral",
]

STRUCTURAL_STATES = [
    "NeutralProcessing",
    "CompressionProcessing",
    "MixedStructure",
    "RecoveryResolution",
    "ReassertionProcessing",
    "ExhaustionPersistence",
    "DecayToNeutral",
]

DESTINATION_FAMILIES = [
    "CompressionProcessing",
    "MixedStructure",
    "RecoveryResolution",
    "ReassertionProcessing",
    "ExhaustionPersistence",
]

WARNING_SCORES = list(range(7))

LEGAL_TRANSITIONS = {
    ("NeutralFormation", "NeutralFormation"),
    ("NeutralFormation", "NeutralMaturation"),
    ("NeutralFormation", "DestinationSelection"),
    ("NeutralMaturation", "NeutralMaturation"),
    ("NeutralMaturation", "LateNeutral"),
    ("NeutralMaturation", "DestinationSelection"),
    ("LateNeutral", "LateNeutral"),
    ("LateNeutral", "DestinationSelection"),
    ("DestinationSelection", "Excursion"),
    ("DestinationSelection", "ReturnToNeutral"),
    ("DestinationSelection", "NeutralFormation"),
    ("Excursion", "Excursion"),
    ("Excursion", "ReturnToNeutral"),
    ("Excursion", "NeutralFormation"),
    ("ReturnToNeutral", "ReturnToNeutral"),
    ("ReturnToNeutral", "NeutralFormation"),
    ("ReturnToNeutral", "Excursion"),
}


def parse_args(argv: Sequence[str]) -> Tuple[Path, Path]:
    if len(argv) == 1:
        return DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR
    if len(argv) == 2:
        input_dir = Path(argv[1])
        return input_dir, input_dir / "Diagnostics"
    if len(argv) == 3:
        return Path(argv[1]), Path(argv[2])
    raise SystemExit("Usage: py -3 APVA_V10_Diagnostics.py [input_dir] [output_dir]")


def clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_float(value: object) -> Optional[float]:
    text = clean(value)
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def parse_int(value: object) -> Optional[int]:
    numeric = parse_float(value)
    if numeric is None:
        return None
    return int(numeric)


def parse_timestamp(value: object) -> Optional[datetime]:
    text = clean(value)
    if not text:
        return None
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def fmt_int(value: int) -> str:
    return str(int(value))


def fmt_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def fmt_float(value: Optional[float], digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return "N/A"
    return f"{value:.{digits}f}"


def percent(count: int, total: int) -> float:
    return count / total if total else 0.0


def mean(values: Iterable[float]) -> Optional[float]:
    material = [v for v in values if v is not None]
    return statistics.fmean(material) if material else None


def median(values: Iterable[float]) -> Optional[float]:
    material = [v for v in values if v is not None]
    return statistics.median(material) if material else None


def stdev(values: Iterable[float]) -> Optional[float]:
    material = [v for v in values if v is not None]
    if len(material) < 2:
        return 0.0 if material else None
    return statistics.stdev(material)


def discover_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        return []
    return sorted(
        path
        for path in input_dir.glob("*.csv")
        if path.name.startswith("APVA_V10_StateMachine_")
    )


def load_rows(files: Sequence[Path]) -> Tuple[List[Dict[str, str]], Dict[str, int], List[str]]:
    rows: List[Dict[str, str]] = []
    rejects = Counter()
    missing_columns: List[str] = []
    required = ["LoopPhase", "StructuralState"]

    for path in files:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            for column in required:
                if column not in fieldnames and column not in missing_columns:
                    missing_columns.append(column)
            for raw in reader:
                raw["_SourceFile"] = str(path)
                raw["_TimestampParsed"] = parse_timestamp(raw.get("Timestamp"))
                if not clean(raw.get("LoopPhase")):
                    rejects["Missing LoopPhase"] += 1
                    continue
                if not clean(raw.get("StructuralState")):
                    rejects["Missing StructuralState"] += 1
                    continue
                rows.append(raw)
    return rows, dict(rejects), missing_columns


def instrument_key(row: Dict[str, str]) -> str:
    return clean(row.get("Instrument")) or "Unknown"


def day_key(row: Dict[str, str]) -> str:
    timestamp = row.get("_TimestampParsed")
    if isinstance(timestamp, datetime):
        return timestamp.date().isoformat()
    text = clean(row.get("Timestamp"))
    return text[:10] if len(text) >= 10 else "Unknown"


def phase_counts(rows: Sequence[Dict[str, str]]) -> Counter:
    return Counter(clean(row.get("LoopPhase")) for row in rows)


def state_counts(rows: Sequence[Dict[str, str]]) -> Counter:
    return Counter(clean(row.get("StructuralState")) for row in rows)


def warning_counts(rows: Sequence[Dict[str, str]]) -> Counter:
    counts = Counter()
    for row in rows:
        score = parse_int(row.get("FailureWarningScore"))
        if score is not None:
            counts[score] += 1
    return counts


def transition_counts(rows: Sequence[Dict[str, str]]) -> Counter:
    counts = Counter()
    for prior, current in zip(rows, rows[1:]):
        if instrument_key(prior) != instrument_key(current):
            continue
        from_phase = clean(prior.get("LoopPhase"))
        to_phase = clean(current.get("LoopPhase"))
        if from_phase and to_phase:
            counts[(from_phase, to_phase)] += 1
    return counts


def run_lengths(rows: Sequence[Dict[str, str]], field: str) -> Dict[str, List[int]]:
    durations: Dict[str, List[int]] = defaultdict(list)
    prior_key = None
    prior_value = None
    length = 0

    for row in rows:
        key = instrument_key(row)
        value = clean(row.get(field))
        if prior_key is None:
            prior_key, prior_value, length = key, value, 1
            continue
        if key == prior_key and value == prior_value:
            length += 1
        else:
            if prior_value:
                durations[prior_value].append(length)
            prior_key, prior_value, length = key, value, 1

    if prior_value:
        durations[prior_value].append(length)
    return durations


def destination_counts(rows: Sequence[Dict[str, str]]) -> Counter:
    counts = Counter()
    for row in rows:
        if clean(row.get("LoopPhase")) != "DestinationSelection":
            continue
        destination = clean(row.get("DestinationFamily"))
        if destination and destination != "N/A":
            counts[destination] += 1
    return counts


def failure_pressure_by_phase(rows: Sequence[Dict[str, str]], threshold: int = 3) -> Dict[str, Counter]:
    table: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        phase = clean(row.get("LoopPhase"))
        score = parse_int(row.get("FailureWarningScore"))
        if score is None:
            continue
        table[phase]["Total"] += 1
        if score >= threshold:
            table[phase]["Pressure"] += 1
    return table


def age_bucket(row: Dict[str, str]) -> str:
    return clean(row.get("AgeBucket"))


def consistency_checks(rows: Sequence[Dict[str, str]]) -> Counter:
    checks = Counter()
    seen_late = False
    prior_phase = ""
    prior_state = ""
    prior_age = ""
    prior_instrument = ""
    last_non_return_phase = ""

    for row in rows:
        inst = instrument_key(row)
        phase = clean(row.get("LoopPhase"))
        structural = clean(row.get("StructuralState"))
        age = age_bucket(row)

        if phase == "LateNeutral":
            seen_late = True

        if inst != prior_instrument:
            prior_phase = ""
            prior_state = ""
            prior_age = ""
            last_non_return_phase = ""

        if phase == "DestinationSelection" and prior_phase == "NeutralFormation":
            checks["DestinationSelection directly after NeutralFormation"] += 1
        if (
            phase == "DestinationSelection"
            and prior_state == "NeutralProcessing"
            and prior_age == "1"
        ):
            checks["DestinationSelection directly after Age1 Neutral"] += 1
        if phase == "Excursion" and prior_phase not in ("DestinationSelection", "Excursion", "ReturnToNeutral"):
            checks["Excursion without DestinationSelection"] += 1
        if phase == "ReturnToNeutral" and last_non_return_phase not in ("DestinationSelection", "Excursion", "ReturnToNeutral"):
            checks["ReturnToNeutral without Excursion"] += 1
        if (prior_phase, phase) not in LEGAL_TRANSITIONS and prior_phase and phase:
            checks[f"Illegal transition {prior_phase}->{phase}"] += 1

        if phase != "ReturnToNeutral":
            last_non_return_phase = phase
        prior_phase = phase
        prior_state = structural
        prior_age = age
        prior_instrument = inst

    if not seen_late:
        checks["LateNeutral never observed"] += 1
    return checks


def distribution_variation(rows: Sequence[Dict[str, str]], field: str, values: Sequence[str]) -> List[Dict[str, object]]:
    by_day: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_day[day_key(row)].append(row)

    output = []
    for value in values:
        daily_rates = []
        for day_rows in by_day.values():
            total = len(day_rows)
            count = sum(1 for row in day_rows if clean(row.get(field)) == value)
            daily_rates.append(percent(count, total))
        output.append(
            {
                "Value": value,
                "MeanDailyPercent": mean(daily_rates),
                "StdDevDailyPercent": stdev(daily_rates),
                "MinDailyPercent": min(daily_rates) if daily_rates else None,
                "MaxDailyPercent": max(daily_rates) if daily_rates else None,
            }
        )
    return output


def warning_variation(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    by_day: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_day[day_key(row)].append(row)

    output = []
    for score in WARNING_SCORES:
        daily_rates = []
        for day_rows in by_day.values():
            total = len(day_rows)
            count = sum(1 for row in day_rows if parse_int(row.get("FailureWarningScore")) == score)
            daily_rates.append(percent(count, total))
        output.append(
            {
                "Score": score,
                "MeanDailyPercent": mean(daily_rates),
                "StdDevDailyPercent": stdev(daily_rates),
                "MinDailyPercent": min(daily_rates) if daily_rates else None,
                "MaxDailyPercent": max(daily_rates) if daily_rates else None,
            }
        )
    return output


def write_csv(path: Path, headers: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def phase_frequency_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    counts = phase_counts(rows)
    total = len(rows)
    return [
        {"LoopPhase": phase, "Count": counts.get(phase, 0), "Percent": fmt_pct(percent(counts.get(phase, 0), total))}
        for phase in LOOP_PHASES
    ]


def warning_distribution_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    counts = warning_counts(rows)
    total = sum(counts.values())
    return [
        {"Score": score, "Count": counts.get(score, 0), "Percent": fmt_pct(percent(counts.get(score, 0), total))}
        for score in WARNING_SCORES
    ]


def transition_matrix_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    counts = transition_counts(rows)
    totals = Counter()
    for (from_phase, _to_phase), count in counts.items():
        totals[from_phase] += count

    output = []
    for (from_phase, to_phase), count in sorted(counts.items()):
        output.append(
            {
                "FromLoopPhase": from_phase,
                "ToLoopPhase": to_phase,
                "Count": count,
                "Probability": fmt_pct(percent(count, totals[from_phase])),
            }
        )
    return output


def classify_machine(
    rows: Sequence[Dict[str, str]],
    checks: Counter,
    phase_freq: Sequence[Dict[str, object]],
    transition_rows_out: Sequence[Dict[str, object]],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    if not rows:
        return "StateMachineBroken", ["No valid rows loaded."]

    phase_counts_map = {row["LoopPhase"]: int(row["Count"]) for row in phase_freq}
    absent = [phase for phase in LOOP_PHASES if phase_counts_map.get(phase, 0) == 0]
    if absent:
        reasons.append("Absent phases: " + ", ".join(absent))

    illegal_total = sum(count for name, count in checks.items() if name.startswith("Illegal transition"))
    if illegal_total:
        reasons.append(f"Illegal transitions observed: {illegal_total}")

    destination_count = phase_counts_map.get("DestinationSelection", 0)
    if destination_count > len(rows) * 0.35:
        reasons.append("DestinationSelection appears over-triggered.")
    if phase_counts_map.get("LateNeutral", 0) == 0:
        reasons.append("LateNeutral is not functioning.")

    if absent and illegal_total:
        return "StateMachineBroken", reasons
    if absent or illegal_total or destination_count > len(rows) * 0.35:
        return "StateMachineQuestionable", reasons
    if checks:
        reasons.append("Suspicious patterns exist but core phases are present.")
        return "StateMachinePlausible", reasons
    reasons.append("Core phases present and no suspicious transition patterns were detected.")
    return "StateMachineHealthy", reasons


def table_lines(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> List[str]:
    rendered = [" | ".join(headers), " | ".join("-" for _ in headers)]
    for row in rows:
        rendered.append(" | ".join(str(value) for value in row))
    return rendered


def build_report(
    input_dir: Path,
    output_dir: Path,
    files: Sequence[Path],
    rows: Sequence[Dict[str, str]],
    rejects: Dict[str, int],
    missing_columns: Sequence[str],
) -> str:
    instruments = sorted({instrument_key(row) for row in rows})
    timestamps = [row.get("_TimestampParsed") for row in rows if isinstance(row.get("_TimestampParsed"), datetime)]
    date_range = "N/A"
    if timestamps:
        date_range = f"{min(timestamps)} to {max(timestamps)}"

    phase_freq = phase_frequency_rows(rows)
    structural_counts = state_counts(rows)
    destination_counter = destination_counts(rows)
    warning_rows = warning_distribution_rows(rows)
    warning_values = [parse_int(row.get("FailureWarningScore")) for row in rows]
    warning_values = [value for value in warning_values if value is not None]
    transition_rows_out = transition_matrix_rows(rows)
    durations = run_lengths(rows, "LoopPhase")
    pressure_table = failure_pressure_by_phase(rows)
    checks = consistency_checks(rows)
    classification, reasons = classify_machine(rows, checks, phase_freq, transition_rows_out)

    lines: List[str] = []
    lines.append("APVA v1.0 Diagnostics Analyzer")
    lines.append("=" * 36)
    lines.append("")

    lines.append("SECTION 1 - Input Validation")
    lines.extend(
        table_lines(
            ["Metric", "Value"],
            [
                ["InputFolder", input_dir],
                ["OutputFolder", output_dir],
                ["FilesFound", len(files)],
                ["RowsRead", len(rows)],
                ["InstrumentsDetected", ", ".join(instruments) if instruments else "N/A"],
                ["DateRange", date_range],
                ["MissingRequiredColumns", ", ".join(missing_columns) if missing_columns else "None"],
                ["Rejected Missing LoopPhase", rejects.get("Missing LoopPhase", 0)],
                ["Rejected Missing StructuralState", rejects.get("Missing StructuralState", 0)],
            ],
        )
    )
    lines.append("")

    lines.append("SECTION 2 - LoopPhase Frequency")
    lines.extend(table_lines(["LoopPhase", "Count", "Percent"], [[r["LoopPhase"], r["Count"], r["Percent"]] for r in phase_freq]))
    absent = [r["LoopPhase"] for r in phase_freq if int(r["Count"]) == 0]
    rare = [r["LoopPhase"] for r in phase_freq if 0 < int(r["Count"]) < max(5, len(rows) * 0.005)]
    lines.append("Absent states: " + (", ".join(absent) if absent else "None"))
    lines.append("Extremely rare states: " + (", ".join(rare) if rare else "None"))
    lines.append("")

    lines.append("SECTION 3 - StructuralState Frequency")
    structural_rows = []
    for state, count in structural_counts.most_common():
        structural_rows.append([state, count, fmt_pct(percent(count, len(rows)))])
    for state in STRUCTURAL_STATES:
        if state not in structural_counts:
            structural_rows.append([state, 0, fmt_pct(0)])
    lines.extend(table_lines(["StructuralState", "Count", "Percent"], structural_rows))
    lines.append("")

    lines.append("SECTION 4 - Transition Matrix")
    lines.extend(
        table_lines(
            ["FromLoopPhase", "ToLoopPhase", "Count", "Probability"],
            [[r["FromLoopPhase"], r["ToLoopPhase"], r["Count"], r["Probability"]] for r in transition_rows_out],
        )
    )
    illegal = [(name, count) for name, count in checks.items() if name.startswith("Illegal transition")]
    lines.append("Illegal transition count: " + str(sum(count for _name, count in illegal)))
    lines.append("")

    lines.append("SECTION 5 - State Duration")
    duration_rows = []
    for phase in LOOP_PHASES:
        values = durations.get(phase, [])
        duration_rows.append(
            [
                phase,
                len(values),
                fmt_float(mean(values), 2),
                fmt_float(median(values), 2),
                max(values) if values else 0,
            ]
        )
    lines.extend(table_lines(["LoopPhase", "Occurrences", "MeanDurationBars", "MedianDurationBars", "MaxDurationBars"], duration_rows))
    longest = sorted(duration_rows, key=lambda row: float(row[2]) if row[2] != "N/A" else -1, reverse=True)
    lines.append("Longest phases: " + ", ".join(row[0] for row in longest[:3]))
    lines.append("Shortest phases: " + ", ".join(row[0] for row in reversed(longest[-3:])))
    lines.append("")

    lines.append("SECTION 6 - Destination Analysis")
    total_destinations = sum(destination_counter.values())
    destination_rows = []
    for family in DESTINATION_FAMILIES:
        count = destination_counter.get(family, 0)
        destination_rows.append([family, count, fmt_pct(percent(count, total_destinations))])
    extra_destinations = [
        (family, count)
        for family, count in destination_counter.most_common()
        if family not in DESTINATION_FAMILIES
    ]
    for family, count in extra_destinations:
        destination_rows.append([family, count, fmt_pct(percent(count, total_destinations))])
    lines.extend(table_lines(["DestinationFamily", "Count", "Percent"], destination_rows))
    dominant = destination_counter.most_common(1)
    if dominant and percent(dominant[0][1], total_destinations) >= 0.60:
        lines.append(f"Dominance warning: {dominant[0][0]} accounts for {fmt_pct(percent(dominant[0][1], total_destinations))}.")
    else:
        lines.append("Dominance warning: None")
    lines.append("")

    lines.append("SECTION 7 - FailureWarningScore Distribution")
    lines.extend(table_lines(["Score", "Count", "Percent"], [[r["Score"], r["Count"], r["Percent"]] for r in warning_rows]))
    lines.append("MeanScore: " + fmt_float(mean(warning_values), 2))
    lines.append("MedianScore: " + fmt_float(median(warning_values), 2))
    lines.append("")

    lines.append("SECTION 8 - Failure Pressure Analysis")
    pressure_rows = []
    for phase in LOOP_PHASES:
        total = pressure_table[phase]["Total"]
        pressure = pressure_table[phase]["Pressure"]
        pressure_rows.append([phase, pressure, total, fmt_pct(percent(pressure, total))])
    lines.extend(table_lines(["LoopPhase", "PressureCount", "TotalCount", "PressurePercent"], pressure_rows))
    lines.append("")

    lines.append("SECTION 9 - Phase Consistency Checks")
    if checks:
        lines.extend(table_lines(["Check", "Count"], [[name, count] for name, count in checks.most_common()]))
    else:
        lines.append("No suspicious patterns detected.")
    lines.append("")

    lines.append("SECTION 10 - Cross-Day Stability")
    day_count = len({day_key(row) for row in rows})
    lines.append(f"DaysDetected: {day_count}")
    lines.append("LoopPhase daily variation")
    phase_variation = distribution_variation(rows, "LoopPhase", LOOP_PHASES)
    lines.extend(
        table_lines(
            ["LoopPhase", "MeanDailyPercent", "StdDevDailyPercent", "MinDailyPercent", "MaxDailyPercent"],
            [
                [
                    row["Value"],
                    fmt_pct(row["MeanDailyPercent"] or 0),
                    fmt_pct(row["StdDevDailyPercent"] or 0),
                    fmt_pct(row["MinDailyPercent"] or 0),
                    fmt_pct(row["MaxDailyPercent"] or 0),
                ]
                for row in phase_variation
            ],
        )
    )
    lines.append("Destination daily variation")
    destination_variation = distribution_variation(rows, "DestinationFamily", DESTINATION_FAMILIES)
    lines.extend(
        table_lines(
            ["DestinationFamily", "MeanDailyPercent", "StdDevDailyPercent", "MinDailyPercent", "MaxDailyPercent"],
            [
                [
                    row["Value"],
                    fmt_pct(row["MeanDailyPercent"] or 0),
                    fmt_pct(row["StdDevDailyPercent"] or 0),
                    fmt_pct(row["MinDailyPercent"] or 0),
                    fmt_pct(row["MaxDailyPercent"] or 0),
                ]
                for row in destination_variation
            ],
        )
    )
    lines.append("WarningScore daily variation")
    warn_variation = warning_variation(rows)
    lines.extend(
        table_lines(
            ["Score", "MeanDailyPercent", "StdDevDailyPercent", "MinDailyPercent", "MaxDailyPercent"],
            [
                [
                    row["Score"],
                    fmt_pct(row["MeanDailyPercent"] or 0),
                    fmt_pct(row["StdDevDailyPercent"] or 0),
                    fmt_pct(row["MinDailyPercent"] or 0),
                    fmt_pct(row["MaxDailyPercent"] or 0),
                ]
                for row in warn_variation
            ],
        )
    )
    lines.append("")

    lines.append("SECTION 11 - Summary")
    lines.append("Classification: " + classification)
    lines.append("Reason:")
    for reason in reasons:
        lines.append("- " + reason)
    lines.append("Highest-priority fixes:")
    fixes = []
    if any(r["LoopPhase"] == "LateNeutral" and int(r["Count"]) == 0 for r in phase_freq):
        fixes.append("LateNeutral age bucketing or neutral persistence needs adjustment.")
    if destination_counter and destination_counter.most_common(1)[0][1] > max(1, total_destinations) * 0.60:
        fixes.append("Destination detector may be over-selecting one family.")
    if checks:
        fixes.append("Review suspicious transition patterns before adding trading interpretation.")
    if not fixes:
        fixes.append("No urgent structural fix detected; continue NT8 replay validation.")
    for fix in fixes:
        lines.append("- " + fix)
    lines.append("")
    lines.append("Research Notes")
    lines.append("This is behavior validation only. No discovery, optimization, fitting, machine learning, or trading logic.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str]) -> int:
    input_dir, output_dir = parse_args(argv)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(input_dir)
    rows, rejects, missing_columns = load_rows(files)

    write_csv(
        output_dir / "APVA_V10_PhaseFrequency.csv",
        ["LoopPhase", "Count", "Percent"],
        phase_frequency_rows(rows),
    )
    write_csv(
        output_dir / "APVA_V10_TransitionMatrix.csv",
        ["FromLoopPhase", "ToLoopPhase", "Count", "Probability"],
        transition_matrix_rows(rows),
    )
    write_csv(
        output_dir / "APVA_V10_WarningDistribution.csv",
        ["Score", "Count", "Percent"],
        warning_distribution_rows(rows),
    )

    report = build_report(input_dir, output_dir, files, rows, rejects, missing_columns)
    (output_dir / "APVA_V10_Diagnostics.txt").write_text(report, encoding="utf-8")

    print(f"FilesFound={len(files)}")
    print(f"RowsRead={len(rows)}")
    print(f"Output={output_dir / 'APVA_V10_Diagnostics.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
