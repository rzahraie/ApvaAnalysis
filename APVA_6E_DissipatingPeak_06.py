#!/usr/bin/env python3
"""Study strict dissipating-peak attempts in 6E APVA evidence exports.

This targeted evidence-layer study measures direction-relative consequences
after a lower-volume contained test of a Peak or Climactic anchor extreme.
It does not create trading signals or infer higher-layer structure.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from APVA_Evidence_Report_IO import write_report


DEFAULT_INPUT_FOLDER = Path("Evidence/6E")
DEFAULT_OUTPUT = Path("Evidence/Output/6E/DissipatingPeak_6E.txt")
HORIZONS = (1, 3, 5, 10, 20)
MAX_ATTEMPT_LOOKAHEAD = 12
TOP_LIMIT = 20
REQUIRED_COLUMNS = {
    "BarIndex",
    "Time",
    "High",
    "Low",
    "Close",
    "Volume",
    "VolumePolarity",
    "ParticipationState",
}


@dataclass(frozen=True)
class EvidenceBar:
    bar_index: int
    time: str
    high: float
    low: float
    close: float
    volume: float
    polarity: str
    participation: str


@dataclass(frozen=True)
class DissipatingPeakEvent:
    anchor_index: int
    attempt_index: int
    anchor: EvidenceBar
    attempt: EvidenceBar
    bars_after_anchor: int
    volume_ratio: float
    strict_contained: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study strict 6E dissipating-peak evidence events."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        help="6E evidence CSV path. Defaults to the latest CSV in Evidence/6E.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Report output path (default: {DEFAULT_OUTPUT.as_posix()})",
    )
    return parser.parse_args()


def latest_csv(folder: Path) -> Path:
    files = [path for path in folder.glob("*.csv") if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No 6E evidence CSV files found in: {folder}")
    return max(files, key=lambda path: (path.stat().st_mtime_ns, path.name))


def require_columns(fieldnames: Iterable[str] | None) -> None:
    available = set(fieldnames or [])
    missing = sorted(REQUIRED_COLUMNS - available)
    if missing:
        raise ValueError("Missing required evidence columns: " + ", ".join(missing))


def parse_float(raw: dict[str, str], column: str, line_number: int) -> float:
    text = (raw.get(column) or "").strip()
    try:
        return float(text)
    except ValueError as error:
        raise ValueError(
            f"Invalid {column} value at CSV line {line_number}: {text!r}"
        ) from error


def load_rows(path: Path) -> list[EvidenceBar]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing evidence CSV: {path}")

    rows: list[EvidenceBar] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames)
        for row_number, raw in enumerate(reader, start=2):
            if not any((value or "").strip() for value in raw.values()):
                continue
            bar_index_text = (raw.get("BarIndex") or "").strip()
            try:
                bar_index = int(bar_index_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid BarIndex value at CSV line {row_number}: {bar_index_text!r}"
                ) from error
            rows.append(
                EvidenceBar(
                    bar_index=bar_index,
                    time=(raw.get("Time") or "").strip(),
                    high=parse_float(raw, "High", row_number),
                    low=parse_float(raw, "Low", row_number),
                    close=parse_float(raw, "Close", row_number),
                    volume=parse_float(raw, "Volume", row_number),
                    polarity=(raw.get("VolumePolarity") or "").strip(),
                    participation=(raw.get("ParticipationState") or "").strip(),
                )
            )
    return rows


def is_anchor(row: EvidenceBar) -> bool:
    return row.participation in {"Peak", "Climactic"} and row.polarity in {"Black", "Red"}


def qualifies(anchor: EvidenceBar, attempt: EvidenceBar) -> bool:
    if attempt.volume >= anchor.volume:
        return False
    if anchor.polarity == "Black":
        return attempt.high >= anchor.high and attempt.close <= anchor.high
    return attempt.low <= anchor.low and attempt.close >= anchor.low


def find_events(rows: list[EvidenceBar]) -> tuple[int, list[DissipatingPeakEvent]]:
    anchors = 0
    events: list[DissipatingPeakEvent] = []
    for anchor_index, anchor in enumerate(rows):
        if not is_anchor(anchor):
            continue
        anchors += 1
        stop = min(len(rows), anchor_index + MAX_ATTEMPT_LOOKAHEAD + 1)
        for attempt_index in range(anchor_index + 1, stop):
            attempt = rows[attempt_index]
            if not qualifies(anchor, attempt):
                continue
            events.append(
                DissipatingPeakEvent(
                    anchor_index=anchor_index,
                    attempt_index=attempt_index,
                    anchor=anchor,
                    attempt=attempt,
                    bars_after_anchor=attempt_index - anchor_index,
                    volume_ratio=attempt.volume / anchor.volume if anchor.volume else 0.0,
                    strict_contained=anchor.low <= attempt.close <= anchor.high,
                )
            )
            break
    return anchors, events


def direction_relative_return(
    rows: list[EvidenceBar],
    event: DissipatingPeakEvent,
    horizon: int,
) -> float | None:
    forward_index = event.attempt_index + horizon
    if forward_index >= len(rows):
        return None
    forward_close = rows[forward_index].close
    if event.anchor.polarity == "Black":
        return event.attempt.close - forward_close
    return forward_close - event.attempt.close


def returns_for(
    rows: list[EvidenceBar],
    events: list[DissipatingPeakEvent],
    horizon: int,
) -> list[float]:
    values: list[float] = []
    for event in events:
        value = direction_relative_return(rows, event, horizon)
        if value is not None:
            values.append(value)
    return values


def summarize(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    success = sum(value > 0.0 for value in values)
    failure = sum(value < 0.0 for value in values)
    flat = count - success - failure
    denominator = count if count else 1
    return {
        "Count": count,
        "MeanDRFwd": statistics.fmean(values) if values else 0.0,
        "MedianDRFwd": statistics.median(values) if values else 0.0,
        "SuccessCount": success,
        "FailureCount": failure,
        "FlatCount": flat,
        "SuccessRate": success / denominator,
        "FailureRate": failure / denominator,
        "FlatRate": flat / denominator,
    }


def append_metrics_table(
    lines: list[str],
    title: str,
    rows: list[EvidenceBar],
    events: list[DissipatingPeakEvent],
) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(
        f"{'Horizon':<8} {'Count':>8} {'MeanDRFwd':>12} {'MedianDRFwd':>14} "
        f"{'Success':>9} {'Failure':>9} {'Flat':>8} "
        f"{'SuccessRate':>12} {'FailureRate':>12} {'FlatRate':>10}"
    )
    for horizon in HORIZONS:
        stats = summarize(returns_for(rows, events, horizon))
        lines.append(
            f"DRFwd{horizon:<2} {stats['Count']:>8} {stats['MeanDRFwd']:>12.6f} "
            f"{stats['MedianDRFwd']:>14.6f} {stats['SuccessCount']:>9} "
            f"{stats['FailureCount']:>9} {stats['FlatCount']:>8} "
            f"{stats['SuccessRate']:>11.2%} {stats['FailureRate']:>11.2%} "
            f"{stats['FlatRate']:>9.2%}"
        )


def append_breakdown(
    lines: list[str],
    title: str,
    rows: list[EvidenceBar],
    events: list[DissipatingPeakEvent],
    groups: list[tuple[str, Callable[[DissipatingPeakEvent], bool]]],
) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    for label, predicate in groups:
        append_metrics_table(lines, label, rows, [event for event in events if predicate(event)])


def volume_ratio_bucket(event: DissipatingPeakEvent) -> str:
    ratio = event.volume_ratio
    if ratio <= 0.25:
        return "<=0.25"
    if ratio <= 0.50:
        return ">0.25 to <=0.50"
    if ratio <= 0.75:
        return ">0.50 to <=0.75"
    return ">0.75 to <1.00"


def append_event_ledger(
    lines: list[str],
    rows: list[EvidenceBar],
    events: list[DissipatingPeakEvent],
) -> None:
    lines.extend(["\nValid Event Ledger", "=================="])
    lines.append(
        f"{'AnchorBar':>10} {'AnchorTime':<19} {'Polarity':<8} {'Part':<10} "
        f"{'AnchorHigh':>12} {'AnchorLow':>12} {'AnchorClose':>12} {'AnchorVol':>12} "
        f"{'AttemptBar':>10} {'AttemptTime':<19} {'After':>5} {'AttemptHigh':>12} "
        f"{'AttemptLow':>12} {'AttemptClose':>12} {'AttemptVol':>12} "
        f"{'VolRatio':>9} {'Strict':>7} "
        + " ".join(f"{'DRFwd' + str(horizon):>12}" for horizon in HORIZONS)
    )
    for event in events:
        values = [direction_relative_return(rows, event, horizon) for horizon in HORIZONS]
        formatted_values = " ".join(
            f"{value:>12.6f}" if value is not None else f"{'NA':>12}"
            for value in values
        )
        lines.append(
            f"{event.anchor.bar_index:>10} {event.anchor.time:<19} "
            f"{event.anchor.polarity:<8} {event.anchor.participation:<10} "
            f"{event.anchor.high:>12.6f} {event.anchor.low:>12.6f} "
            f"{event.anchor.close:>12.6f} {event.anchor.volume:>12.2f} "
            f"{event.attempt.bar_index:>10} {event.attempt.time:<19} "
            f"{event.bars_after_anchor:>5} {event.attempt.high:>12.6f} "
            f"{event.attempt.low:>12.6f} {event.attempt.close:>12.6f} "
            f"{event.attempt.volume:>12.2f} {event.volume_ratio:>9.4f} "
            f"{str(event.strict_contained):>7} {formatted_values}"
        )


def append_ranked_events(
    lines: list[str],
    title: str,
    rows: list[EvidenceBar],
    events: list[DissipatingPeakEvent],
    reverse: bool,
) -> None:
    available = [
        (event, direction_relative_return(rows, event, 10))
        for event in events
        if direction_relative_return(rows, event, 10) is not None
    ]
    ranked = sorted(available, key=lambda item: float(item[1]), reverse=reverse)[:TOP_LIMIT]
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(
        f"{'AnchorBar':>10} {'AnchorTime':<19} {'Polarity':<8} {'Part':<10} "
        f"{'AttemptBar':>10} {'AttemptTime':<19} {'After':>5} {'VolRatio':>9} "
        f"{'Strict':>7} {'DRFwd10':>10}"
    )
    for event, value in ranked:
        lines.append(
            f"{event.anchor.bar_index:>10} {event.anchor.time:<19} "
            f"{event.anchor.polarity:<8} {event.anchor.participation:<10} "
            f"{event.attempt.bar_index:>10} {event.attempt.time:<19} "
            f"{event.bars_after_anchor:>5} {event.volume_ratio:>9.4f} "
            f"{str(event.strict_contained):>7} {float(value):>10.6f}"
        )


def comparison_note(
    rows: list[EvidenceBar],
    events: list[DissipatingPeakEvent],
    left_label: str,
    left: Callable[[DissipatingPeakEvent], bool],
    right_label: str,
    right: Callable[[DissipatingPeakEvent], bool],
    horizon: int = 10,
) -> str:
    left_stats = summarize(returns_for(rows, [event for event in events if left(event)], horizon))
    right_stats = summarize(returns_for(rows, [event for event in events if right(event)], horizon))
    if left_stats["SuccessRate"] > right_stats["SuccessRate"]:
        relation = "higher"
    elif left_stats["SuccessRate"] < right_stats["SuccessRate"]:
        relation = "lower"
    else:
        relation = "equal"
    return (
        f"- DRFwd{horizon} success rate is {relation} for {left_label} "
        f"({left_stats['SuccessRate']:.2%}, n={left_stats['Count']}) than {right_label} "
        f"({right_stats['SuccessRate']:.2%}, n={right_stats['Count']})."
    )


def append_research_notes(
    lines: list[str],
    rows: list[EvidenceBar],
    anchors: int,
    events: list[DissipatingPeakEvent],
) -> None:
    lines.extend(["\nResearch Notes", "=============="])
    lines.append(f"- Total Peak or Climactic anchors with Black/Red polarity: {anchors}.")
    lines.append(f"- Total valid first-attempt dissipating-peak events: {len(events)}.")
    rate = len(events) / anchors if anchors else 0.0
    lines.append(f"- Valid event rate per anchor: {rate:.2%}.")

    horizon_stats = [(horizon, summarize(returns_for(rows, events, horizon))) for horizon in HORIZONS]
    best_horizon, best_stats = max(
        horizon_stats, key=lambda item: (item[1]["SuccessRate"], -item[0])
    )
    lines.append(
        f"- Highest overall success rate occurs at DRFwd{best_horizon}: "
        f"{best_stats['SuccessRate']:.2%} (n={best_stats['Count']})."
    )
    lines.append(
        comparison_note(
            rows,
            events,
            "StrictContained=true",
            lambda event: event.strict_contained,
            "StrictContained=false",
            lambda event: not event.strict_contained,
        )
    )
    lines.append(
        comparison_note(
            rows,
            events,
            "VolumeRatio<=0.50",
            lambda event: event.volume_ratio <= 0.50,
            "VolumeRatio>0.50",
            lambda event: event.volume_ratio > 0.50,
        )
    )
    lines.append(
        comparison_note(
            rows,
            events,
            "Peak anchors",
            lambda event: event.anchor.participation == "Peak",
            "Climactic anchors",
            lambda event: event.anchor.participation == "Climactic",
        )
    )
    lines.append(
        comparison_note(
            rows,
            events,
            "Black anchors",
            lambda event: event.anchor.polarity == "Black",
            "Red anchors",
            lambda event: event.anchor.polarity == "Red",
        )
    )


def build_report(path: Path) -> str:
    rows = load_rows(path)
    anchors, events = find_events(rows)
    lines = [
        "APVA 6E Dissipating Peak Study v0.1",
        "==================================",
        f"Input: {path}",
        f"Total rows: {len(rows)}",
        f"Attempt search window: anchor+1 through anchor+{MAX_ATTEMPT_LOOKAHEAD}",
        "Selection rule: first qualifying attempt per anchor",
        "DRFwd > 0: reversal/opposite-anchor success",
        "DRFwd < 0: anchor-direction continuation/failure",
    ]

    append_metrics_table(lines, "All Valid Dissipating-Peak Events", rows, events)
    append_event_ledger(lines, rows, events)
    append_breakdown(
        lines,
        "Breakdown by Anchor Polarity",
        rows,
        events,
        [
            ("Black", lambda event: event.anchor.polarity == "Black"),
            ("Red", lambda event: event.anchor.polarity == "Red"),
        ],
    )
    append_breakdown(
        lines,
        "Breakdown by Anchor Participation State",
        rows,
        events,
        [
            ("Peak", lambda event: event.anchor.participation == "Peak"),
            ("Climactic", lambda event: event.anchor.participation == "Climactic"),
        ],
    )
    append_breakdown(
        lines,
        "Breakdown by Bars After Anchor",
        rows,
        events,
        [
            (f"BarsAfterAnchor={offset}", lambda event, offset=offset: event.bars_after_anchor == offset)
            for offset in range(1, MAX_ATTEMPT_LOOKAHEAD + 1)
        ],
    )
    append_breakdown(
        lines,
        "Breakdown by Strict Containment",
        rows,
        events,
        [
            ("StrictContained=true", lambda event: event.strict_contained),
            ("StrictContained=false", lambda event: not event.strict_contained),
        ],
    )
    ratio_labels = ("<=0.25", ">0.25 to <=0.50", ">0.50 to <=0.75", ">0.75 to <1.00")
    append_breakdown(
        lines,
        "Breakdown by Volume Ratio",
        rows,
        events,
        [
            (label, lambda event, label=label: volume_ratio_bucket(event) == label)
            for label in ratio_labels
        ],
    )
    append_ranked_events(lines, "Top 20 Strongest Reversal Events by DRFwd10", rows, events, True)
    append_ranked_events(lines, "Top 20 Strongest Failure Events by DRFwd10", rows, events, False)
    append_research_notes(lines, rows, anchors, events)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_path = args.input_path or latest_csv(DEFAULT_INPUT_FOLDER)
    write_report(build_report(input_path), args.output)


if __name__ == "__main__":
    main()
