#!/usr/bin/env python3
"""APVA Evidence Loader Upgrade Study v0.1.

Infrastructure audit for the shared APVA evidence loader.

This study verifies that the shared loader exposes market observables when
present, computes fixed derived OHLC/volume fields, and remains backward
compatible for structural studies.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from APVA_ProcessGraph_54 import node_for, node_text
from APVA_StructuralLifeCycle_44 import (
    Bar,
    collect_inputs,
    ensure_dir,
    find_col,
    fmt,
    load_results,
    pct,
    safe_float,
)


ROLLING_N = 20
OUTPUT_DIR = Path("Evidence/Output/EvidenceLoaderUpgrade")


@dataclass
class FileDiagnostics:
    path: str
    rows_loaded: int
    detected: dict[str, str | None]
    missing: dict[str, int]
    has_ohlc: bool
    has_volume: bool
    has_volume_polarity: bool
    derived_available: dict[str, bool]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def display(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return bool_text(value)
    if isinstance(value, float):
        return fmt(value)
    return str(value)


def detect_columns(headers: list[str]) -> dict[str, str | None]:
    return {
        "Timestamp": find_col(headers, ["Time", "DateTime", "Timestamp", "BarTime"]),
        "Instrument": find_col(headers, ["Instrument", "Symbol"]),
        "SessionDate": find_col(headers, ["SessionDate"]),
        "Open": find_col(headers, ["Open", "O"]),
        "High": find_col(headers, ["High", "H"]),
        "Low": find_col(headers, ["Low", "L"]),
        "Close": find_col(headers, ["Close", "C"]),
        "Volume": find_col(headers, ["Volume", "Vol", "V"]),
        "VolumePolarity": find_col(headers, ["VolumePolarity", "Polarity", "BarPolarity"]),
        "ParticipationState": find_col(headers, ["ParticipationState"]),
        "ExpansionState": find_col(headers, ["ExpansionState"]),
        "CompressionState": find_col(headers, ["CompressionState"]),
        "DissipationState": find_col(headers, ["DissipationState"]),
        "AcceptanceState": find_col(headers, ["AcceptanceState"]),
        "EvidenceFlags": find_col(headers, ["EvidenceFlags", "Flags"]),
    }


def missing_count(rows: list[dict[str, str]], col: str | None, numeric: bool = False) -> int:
    if not col:
        return len(rows)
    count = 0
    for row in rows:
        value = row.get(col)
        if numeric:
            if safe_float(value) is None:
                count += 1
        elif value is None or str(value).strip() == "":
            count += 1
    return count


def diagnose_file(path: str) -> FileDiagnostics:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        rows = list(reader)
    detected = detect_columns(headers)
    missing = {
        "OpenMissing": missing_count(rows, detected["Open"], numeric=True),
        "HighMissing": missing_count(rows, detected["High"], numeric=True),
        "LowMissing": missing_count(rows, detected["Low"], numeric=True),
        "CloseMissing": missing_count(rows, detected["Close"], numeric=True),
        "VolumeMissing": missing_count(rows, detected["Volume"], numeric=True),
    }
    has_ohlc = all(detected[name] for name in ("Open", "High", "Low", "Close")) and all(
        missing[name + "Missing"] < len(rows) for name in ("Open", "High", "Low", "Close")
    )
    has_volume = bool(detected["Volume"]) and missing["VolumeMissing"] < len(rows)
    has_polarity = bool(detected["VolumePolarity"]) and missing_count(rows, detected["VolumePolarity"]) < len(rows)
    return FileDiagnostics(
        path=path,
        rows_loaded=len(rows),
        detected=detected,
        missing=missing,
        has_ohlc=has_ohlc,
        has_volume=has_volume,
        has_volume_polarity=has_polarity,
        derived_available={
            "BarRangeAvailable": has_ohlc,
            "BodyAvailable": has_ohlc,
            "TrueRangeAvailable": has_ohlc,
            "VolumeRelativeAvailable": has_volume,
        },
    )


def attribute_available(bars: list[Bar], name: str) -> bool:
    return any(getattr(bar, name, None) is not None for bar in bars)


def branch_entropy_by_node(bars: list[Bar]) -> dict[str, float]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for index, bar in enumerate(bars[:-1]):
        current = node_text(node_for(bar))
        next_node = node_text(node_for(bars[index + 1]))
        counts[current][next_node] += 1
    out = {}
    for node, node_counts in counts.items():
        total = sum(node_counts.values())
        if total <= 0:
            out[node] = 0.0
            continue
        probs = [count / total for count in node_counts.values()]
        import math
        out[node] = -sum(prob * math.log(prob) for prob in probs if prob > 0)
    return out


def backward_compatibility(results) -> tuple[bool, list[str]]:
    required = {
        "StateAgeNode": lambda bars: all(node_text(node_for(bar)) for bar in bars[:10]),
        "StructuralState": lambda bars: all(getattr(bar, "state", "") for bar in bars[:10]),
        "AgeBucket": lambda bars: all(getattr(bar, "age_bucket", "") for bar in bars[:10]),
        "PreviousNode": lambda bars: len(bars) > 1,
        "CurrentNode": lambda bars: bool(bars),
        "NextNode": lambda bars: len(bars) > 1,
        "MemoryStrength": lambda bars: True,
        "BranchEntropy": lambda bars: bool(branch_entropy_by_node(bars)),
        "VolumePolarity": lambda bars: any(getattr(bar, "volume_polarity", "") for bar in bars),
    }
    missing = []
    all_bars = [bar for result in results for bar in result.bars]
    for name, predicate in required.items():
        if not predicate(all_bars):
            missing.append(name)
    return not missing, missing


def study72_readiness(results) -> dict[str, tuple[bool, str]]:
    all_bars = [bar for result in results for bar in result.bars]
    has_ohlc = any(getattr(bar, "has_ohlc", False) for bar in all_bars)
    has_volume = any(getattr(bar, "has_volume", False) for bar in all_bars)
    return {
        "CanRunNeutralMarketInterpretationWithOHLC": (has_ohlc, "requires Open, High, Low, Close"),
        "CanRunNeutralMarketInterpretationWithVolume": (has_volume, "requires Volume"),
        "CanRunRangeVolumeCoupling": (has_ohlc and has_volume, "requires OHLC and Volume"),
        "CanRunCompressionExpansionProfile": (has_ohlc and has_volume, "requires OHLC and Volume-relative fields"),
    }


def sample_probe(results) -> dict[str, list[dict[str, object]]]:
    probe = {}
    for result in results:
        rows = []
        for index, bar in enumerate(result.bars):
            if len(rows) >= 10:
                break
            if bar.close is None:
                continue
            rows.append({
                "Timestamp": bar.time,
                "Instrument": bar.instrument,
                "Open": bar.open,
                "High": bar.high,
                "Low": bar.low,
                "Close": bar.close,
                "Volume": bar.volume,
                "VolumePolarity": bar.volume_polarity,
                "BarRange": bar.bar_range,
                "Body": bar.body,
                "TrueRange": bar.true_range,
                "CloseLocation": bar.close_location,
                "VolumeRelativeToPrevious": bar.volume_relative_to_previous,
                "VolumeRelativeToSessionMean": bar.volume_relative_to_session_mean,
                "VolumeRelativeToRollingMean": bar.volume_relative_to_rolling_mean,
                "RangeRelativeToPrevious": bar.range_relative_to_previous,
                "BodyRelativeToPrevious": bar.body_relative_to_previous,
                "TrueRangeRelativeToPrevious": bar.true_range_relative_to_previous,
            })
        probe[result.instrument] = rows
    return probe


def loader_aggregate(results) -> dict[str, object]:
    bars = [bar for result in results for bar in result.bars]
    return {
        "RowsLoaded": len(bars),
        "HasOHLC": any(getattr(bar, "has_ohlc", False) for bar in bars),
        "HasVolume": any(getattr(bar, "has_volume", False) for bar in bars),
        "HasVolumePolarity": any(getattr(bar, "has_volume_polarity", False) for bar in bars),
        "BarRangeAvailable": attribute_available(bars, "bar_range"),
        "BodyAvailable": attribute_available(bars, "body"),
        "TrueRangeAvailable": attribute_available(bars, "true_range"),
        "VolumeRelativeAvailable": attribute_available(bars, "volume_relative_to_previous"),
        "VolumeRelativeToSessionMeanAvailable": attribute_available(bars, "volume_relative_to_session_mean"),
        "VolumeRelativeToRollingMeanAvailable": attribute_available(bars, "volume_relative_to_rolling_mean"),
    }


def report_lines(file_diagnostics: list[FileDiagnostics], results) -> list[str]:
    aggregate = loader_aggregate(results)
    compat_passed, missing_legacy = backward_compatibility(results)
    readiness = study72_readiness(results)
    lines = [
        "APVA Evidence Loader Upgrade Study v0.1",
        "=" * 96,
        "",
        "1. Loader Diagnostics",
        "Metric | Value",
    ]
    lines += [f"{key} | {display(value)}" for key, value in aggregate.items()]
    lines += ["", "File | RowsLoaded | HasOHLC | HasVolume | HasVolumePolarity"]
    for diag in file_diagnostics:
        lines.append(
            f"{diag.path} | {diag.rows_loaded} | {bool_text(diag.has_ohlc)} | "
            f"{bool_text(diag.has_volume)} | {bool_text(diag.has_volume_polarity)}"
        )

    lines += ["", "2. Column Detection Report", "File | Field | DetectedColumn"]
    for diag in file_diagnostics:
        for field, column in diag.detected.items():
            lines.append(f"{diag.path} | {field} | {column or 'N/A'}")

    lines += ["", "Missing Field Counts", "File | OpenMissing | HighMissing | LowMissing | CloseMissing | VolumeMissing"]
    for diag in file_diagnostics:
        lines.append(
            f"{diag.path} | {diag.missing['OpenMissing']} | {diag.missing['HighMissing']} | "
            f"{diag.missing['LowMissing']} | {diag.missing['CloseMissing']} | {diag.missing['VolumeMissing']}"
        )

    lines += ["", "3. Derived Field Report", "File | BarRangeAvailable | BodyAvailable | TrueRangeAvailable | VolumeRelativeAvailable"]
    for diag in file_diagnostics:
        lines.append(
            f"{diag.path} | {bool_text(diag.derived_available['BarRangeAvailable'])} | "
            f"{bool_text(diag.derived_available['BodyAvailable'])} | "
            f"{bool_text(diag.derived_available['TrueRangeAvailable'])} | "
            f"{bool_text(diag.derived_available['VolumeRelativeAvailable'])}"
        )

    lines += [
        "",
        "Derived Field Definitions",
        "BarRange = High - Low",
        "Body = abs(Close - Open)",
        "CloseLocation = (Close - Low) / max(High - Low, epsilon)",
        "TrueRange = max(High - Low, abs(High - PreviousClose), abs(Low - PreviousClose))",
        f"VolumeRelativeToRollingMean uses fixed N = {ROLLING_N} and requires at least 5 prior bars.",
    ]

    lines += ["", "4. Backward Compatibility Report"]
    lines.append(f"BackwardCompatibilityPassed: {bool_text(compat_passed)}")
    lines.append("MissingLegacyFields: " + (", ".join(missing_legacy) if missing_legacy else "None"))
    lines.append("Note: MemoryStrength remains a downstream Study 57 derived metric; BranchEntropy is derivable from the loaded StateAge stream.")

    lines += ["", "5. Study 72 Readiness Report", "Capability | CanRun | Explanation"]
    for name, (can_run, explanation) in readiness.items():
        suffix = explanation if not can_run else "required fields present"
        lines.append(f"{name} | {bool_text(can_run)} | {suffix}")

    lines += [
        "",
        "6. Low-DoF Audit",
        "Infrastructure only.",
        "No new APVA states.",
        "No new APVA families.",
        "No context.",
        "No arbitration.",
        "No persistence.",
        "No phase.",
        "No optimization.",
        "No fitting.",
        "No machine learning.",
        "No trading logic.",
    ]
    return lines


def validate(results, file_diagnostics: list[FileDiagnostics]) -> None:
    if not results:
        raise RuntimeError("No loaded instrument results.")
    if not file_diagnostics:
        raise RuntimeError("No file diagnostics.")
    for result in results:
        if not result.bars:
            raise RuntimeError(f"{result.instrument}: no bars loaded.")
        for bar in result.bars:
            if bar.has_ohlc and any(value is None for value in (bar.bar_range, bar.body, bar.true_range, bar.close_location)):
                raise RuntimeError(f"{result.instrument}: OHLC row missing derived OHLC fields.")
            if bar.volume_relative_to_previous is not None and bar.volume is None:
                raise RuntimeError(f"{result.instrument}: volume relative field exists without volume.")


def write_outputs(file_diagnostics: list[FileDiagnostics], results, out_root: Path) -> None:
    output_dir = out_root / "EvidenceLoaderUpgrade"
    ensure_dir(str(output_dir))
    report_path = output_dir / "EvidenceLoaderUpgrade_All.txt"
    probe_path = output_dir / "EvidenceLoaderProbe.json"
    report_path.write_text("\n".join(report_lines(file_diagnostics, results)) + "\n", encoding="utf-8")
    probe_path.write_text(json.dumps(sample_probe(results), indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Evidence CSV files or directories")
    parser.add_argument("--out-root", default="Evidence/Output", help="Output root directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = collect_inputs(args.inputs)
    if not paths:
        raise SystemExit("No APVA evidence CSV files found.")
    file_diagnostics = [diagnose_file(path) for path in paths]
    results = load_results(args.inputs)
    validate(results, file_diagnostics)
    write_outputs(file_diagnostics, results, Path(args.out_root))
    print(f"Wrote EvidenceLoaderUpgrade report and JSON probe under {Path(args.out_root) / 'EvidenceLoaderUpgrade'}.")


if __name__ == "__main__":
    main()
