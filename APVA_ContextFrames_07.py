#!/usr/bin/env python3
"""Study APVA evidence events inside simple, descriptive context frames.

This evidence-layer research script asks where events occurred. It does not
create trading signals or infer higher-layer structure.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from APVA_Evidence_Report_IO import report_path, write_report


DEFAULT_INPUT_FOLDER = Path("Evidence/6E")
HORIZONS = (1, 3, 5, 10, 20)
MIN_RANKED_SAMPLES = 30
TOP_LIMIT = 25
FRAME_NAMES = (
    "CompressionFrame",
    "ExpansionFrame",
    "DirectionalPersistenceFrame",
    "AlternatingFrame",
    "NoNamedFrame",
)
REQUIRED_COLUMNS = {
    "BarIndex",
    "Time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "VolumePolarity",
    "ParticipationState",
    "CompressionState",
    "ExpansionState",
    "DissipationState",
    "AcceptanceState",
}


@dataclass(frozen=True)
class EvidenceBar:
    bar_index: int
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    polarity: str
    participation: str
    compression: str
    expansion: str
    dissipation: str
    acceptance: str


@dataclass(frozen=True)
class EventDefinition:
    name: str
    predicate: Callable[[EvidenceBar], bool]


@dataclass(frozen=True)
class FrameDefinition:
    name: str
    membership: tuple[bool, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study APVA evidence events inside descriptive context frames."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        help="Evidence CSV path. Defaults to the latest CSV in Evidence/6E.",
    )
    return parser.parse_args()


def latest_csv(folder: Path) -> Path:
    files = [path for path in folder.glob("*.csv") if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No evidence CSV files found in: {folder}")
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
        for line_number, raw in enumerate(reader, start=2):
            if not any((value or "").strip() for value in raw.values()):
                continue
            bar_index_text = (raw.get("BarIndex") or "").strip()
            try:
                bar_index = int(bar_index_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid BarIndex value at CSV line {line_number}: {bar_index_text!r}"
                ) from error
            rows.append(
                EvidenceBar(
                    bar_index=bar_index,
                    time=(raw.get("Time") or "").strip(),
                    open=parse_float(raw, "Open", line_number),
                    high=parse_float(raw, "High", line_number),
                    low=parse_float(raw, "Low", line_number),
                    close=parse_float(raw, "Close", line_number),
                    volume=parse_float(raw, "Volume", line_number),
                    polarity=(raw.get("VolumePolarity") or "").strip(),
                    participation=(raw.get("ParticipationState") or "").strip(),
                    compression=(raw.get("CompressionState") or "").strip(),
                    expansion=(raw.get("ExpansionState") or "").strip(),
                    dissipation=(raw.get("DissipationState") or "").strip(),
                    acceptance=(raw.get("AcceptanceState") or "").strip(),
                )
            )
    return rows


def event_definitions() -> list[EventDefinition]:
    return [
        EventDefinition("AllBars", lambda row: True),
        EventDefinition("DissipationAny", lambda row: row.dissipation != "Absent"),
        EventDefinition(
            "DissipationContained",
            lambda row: row.dissipation != "Absent" and row.acceptance == "Contained",
        ),
        EventDefinition(
            "DissipationAccepted",
            lambda row: row.dissipation != "Absent" and row.acceptance == "Accepted",
        ),
        EventDefinition("ParticipationPeak", lambda row: row.participation == "Peak"),
        EventDefinition(
            "ParticipationClimactic", lambda row: row.participation == "Climactic"
        ),
        EventDefinition(
            "PeakContained",
            lambda row: row.participation == "Peak" and row.acceptance == "Contained",
        ),
        EventDefinition(
            "PeakAccepted",
            lambda row: row.participation == "Peak" and row.acceptance == "Accepted",
        ),
        EventDefinition(
            "ClimacticContained",
            lambda row: row.participation == "Climactic"
            and row.acceptance == "Contained",
        ),
        EventDefinition(
            "ClimacticAccepted",
            lambda row: row.participation == "Climactic" and row.acceptance == "Accepted",
        ),
    ]


def all_non_absent(values: list[str]) -> bool:
    return bool(values) and all(value != "Absent" for value in values)


def same_direction(values: list[str]) -> bool:
    return len(values) == 5 and values[0] in {"Black", "Red"} and len(set(values)) == 1


def alternating(values: list[str]) -> bool:
    return (
        len(values) == 4
        and all(value in {"Black", "Red"} for value in values)
        and all(values[index] != values[index - 1] for index in range(1, len(values)))
    )


def build_frames(rows: list[EvidenceBar]) -> list[FrameDefinition]:
    compression: list[bool] = []
    expansion: list[bool] = []
    persistence: list[bool] = []
    alternation: list[bool] = []

    for index in range(len(rows)):
        compression.append(
            index >= 2
            and all_non_absent([row.compression for row in rows[index - 2 : index + 1]])
        )
        expansion.append(
            index >= 2
            and all_non_absent([row.expansion for row in rows[index - 2 : index + 1]])
        )
        persistence.append(
            index >= 4 and same_direction([row.polarity for row in rows[index - 4 : index + 1]])
        )
        alternation.append(
            index >= 3 and alternating([row.polarity for row in rows[index - 3 : index + 1]])
        )

    no_named = [
        not (compression[index] or expansion[index] or persistence[index] or alternation[index])
        for index in range(len(rows))
    ]
    return [
        FrameDefinition("CompressionFrame", tuple(compression)),
        FrameDefinition("ExpansionFrame", tuple(expansion)),
        FrameDefinition("DirectionalPersistenceFrame", tuple(persistence)),
        FrameDefinition("AlternatingFrame", tuple(alternation)),
        FrameDefinition("NoNamedFrame", tuple(no_named)),
    ]


def direction_relative_return(
    rows: list[EvidenceBar],
    index: int,
    horizon: int,
) -> float | None:
    forward_index = index + horizon
    if forward_index >= len(rows):
        return None
    if rows[index].polarity == "Black":
        return rows[forward_index].close - rows[index].close
    if rows[index].polarity == "Red":
        return rows[index].close - rows[forward_index].close
    return None


def consequence_values(
    rows: list[EvidenceBar],
    event: EventDefinition,
    frame: FrameDefinition,
    horizon: int,
    inside: bool,
) -> list[float]:
    values: list[float] = []
    for index, row in enumerate(rows):
        if frame.membership[index] != inside or not event.predicate(row):
            continue
        value = direction_relative_return(rows, index, horizon)
        if value is not None:
            values.append(value)
    return values


def summarize(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    continuation = sum(value > 0.0 for value in values)
    failure = sum(value < 0.0 for value in values)
    flat = count - continuation - failure
    denominator = count if count else 1
    return {
        "Count": count,
        "MeanDRFwd": statistics.fmean(values) if values else 0.0,
        "MedianDRFwd": statistics.median(values) if values else 0.0,
        "ContinuationCount": continuation,
        "FailureCount": failure,
        "FlatCount": flat,
        "ContinuationRate": continuation / denominator,
        "FailureRate": failure / denominator,
        "FlatRate": flat / denominator,
    }


def append_full_metrics(
    lines: list[str],
    rows: list[EvidenceBar],
    events: list[EventDefinition],
    frames: list[FrameDefinition],
) -> None:
    lines.extend(["\nInside-Frame Direction-Relative Consequences", "============================================"])
    for frame in frames:
        lines.extend([f"\nFrame: {frame.name}", "-" * (7 + len(frame.name))])
        for event in events:
            lines.append(f"\nEvent: {event.name}")
            lines.append(
                f"{'Horizon':<8} {'Count':>8} {'MeanDRFwd':>12} {'MedianDRFwd':>14} "
                f"{'Continue':>10} {'Fail':>8} {'Flat':>8} "
                f"{'ContRate':>10} {'FailRate':>10} {'FlatRate':>10}"
            )
            for horizon in HORIZONS:
                stats = summarize(consequence_values(rows, event, frame, horizon, True))
                lines.append(
                    f"DRFwd{horizon:<2} {stats['Count']:>8} {stats['MeanDRFwd']:>12.6f} "
                    f"{stats['MedianDRFwd']:>14.6f} {stats['ContinuationCount']:>10} "
                    f"{stats['FailureCount']:>8} {stats['FlatCount']:>8} "
                    f"{stats['ContinuationRate']:>9.2%} {stats['FailureRate']:>9.2%} "
                    f"{stats['FlatRate']:>9.2%}"
                )


def append_comparisons(
    lines: list[str],
    rows: list[EvidenceBar],
    events: list[EventDefinition],
    frames: list[FrameDefinition],
) -> None:
    lines.extend(["\nInside-Frame vs Outside-Frame Comparisons", "========================================="])
    for frame in frames:
        lines.extend([f"\nFrame: {frame.name}", "-" * (7 + len(frame.name))])
        for event in events:
            lines.append(f"\nEvent: {event.name}")
            lines.append(
                f"{'Horizon':<8} {'InCount':>8} {'InMean':>12} {'InCont':>10} "
                f"{'OutCount':>9} {'OutMean':>12} {'OutCont':>10} "
                f"{'DeltaMean':>12} {'DeltaCont':>10}"
            )
            for horizon in HORIZONS:
                inside = summarize(consequence_values(rows, event, frame, horizon, True))
                outside = summarize(consequence_values(rows, event, frame, horizon, False))
                lines.append(
                    f"DRFwd{horizon:<2} {inside['Count']:>8} {inside['MeanDRFwd']:>12.6f} "
                    f"{inside['ContinuationRate']:>9.2%} {outside['Count']:>9} "
                    f"{outside['MeanDRFwd']:>12.6f} {outside['ContinuationRate']:>9.2%} "
                    f"{inside['MeanDRFwd'] - outside['MeanDRFwd']:>12.6f} "
                    f"{inside['ContinuationRate'] - outside['ContinuationRate']:>9.2%}"
                )


def fwd5_ranked_stats(
    rows: list[EvidenceBar],
    events: list[EventDefinition],
    frames: list[FrameDefinition],
) -> tuple[list[tuple[str, str, dict[str, float | int]]], int]:
    ranked: list[tuple[str, str, dict[str, float | int]]] = []
    skipped = 0
    for frame in frames:
        for event in events:
            stats = summarize(consequence_values(rows, event, frame, 5, True))
            if stats["Count"] >= MIN_RANKED_SAMPLES:
                ranked.append((event.name, frame.name, stats))
            else:
                skipped += 1
    return ranked, skipped


def append_ranked_table(
    lines: list[str],
    title: str,
    ranked: list[tuple[str, str, dict[str, float | int]]],
) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(
        f"{'Rank':>4} {'Event':<24} {'Frame':<30} {'Count':>8} {'MeanDRFwd5':>12} "
        f"{'Median':>12} {'ContRate':>10} {'FailRate':>10}"
    )
    for rank, (event_name, frame_name, stats) in enumerate(ranked[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {event_name:<24} {frame_name:<30} {stats['Count']:>8} "
            f"{stats['MeanDRFwd']:>12.6f} {stats['MedianDRFwd']:>12.6f} "
            f"{stats['ContinuationRate']:>9.2%} {stats['FailureRate']:>9.2%}"
        )


def peak_contained_delta(
    rows: list[EvidenceBar],
    event: EventDefinition,
    frame: FrameDefinition,
) -> float:
    inside = summarize(consequence_values(rows, event, frame, 5, True))
    outside = summarize(consequence_values(rows, event, frame, 5, False))
    return float(inside["MeanDRFwd"]) - float(outside["MeanDRFwd"])


def append_research_notes(
    lines: list[str],
    rows: list[EvidenceBar],
    events: list[EventDefinition],
    frames: list[FrameDefinition],
    ranked: list[tuple[str, str, dict[str, float | int]]],
    skipped: int,
) -> None:
    lines.extend(["\nResearch Notes", "=============="])
    frame_counts = [(frame.name, sum(frame.membership)) for frame in frames]
    most_bars = max(frame_counts, key=lambda item: (item[1], item[0]))
    fewest_bars = min(frame_counts, key=lambda item: (item[1], item[0]))
    lines.append(f"- Frame with most bars: {most_bars[0]} ({most_bars[1]}).")
    lines.append(f"- Frame with fewest bars: {fewest_bars[0]} ({fewest_bars[1]}).")

    if ranked:
        highest = max(ranked, key=lambda item: (item[2]["MeanDRFwd"], item[0], item[1]))
        lowest = min(ranked, key=lambda item: (item[2]["MeanDRFwd"], item[0], item[1]))
        lines.append(
            f"- Highest MeanDRFwd5 event-frame combination: {highest[0]} inside "
            f"{highest[1]} ({highest[2]['MeanDRFwd']:.6f}, n={highest[2]['Count']})."
        )
        lines.append(
            f"- Lowest MeanDRFwd5 event-frame combination: {lowest[0]} inside "
            f"{lowest[1]} ({lowest[2]['MeanDRFwd']:.6f}, n={lowest[2]['Count']})."
        )
    else:
        lines.append("- No event-frame combinations met the ranked-table sample threshold.")

    peak_contained = next(event for event in events if event.name == "PeakContained")
    deltas = [(frame.name, peak_contained_delta(rows, peak_contained, frame)) for frame in frames]
    best = max(deltas, key=lambda item: (item[1], item[0]))
    worst = min(deltas, key=lambda item: (item[1], item[0]))
    lines.append(
        f"- Largest PeakContained inside-vs-outside MeanDRFwd5 improvement: "
        f"{best[0]} ({best[1]:.6f})."
    )
    lines.append(
        f"- Largest PeakContained inside-vs-outside MeanDRFwd5 worsening: "
        f"{worst[0]} ({worst[1]:.6f})."
    )
    lines.append(
        f"- Event-frame combinations skipped from ranked tables because Count < "
        f"{MIN_RANKED_SAMPLES}: {skipped}."
    )


def build_report(path: Path) -> str:
    rows = load_rows(path)
    events = event_definitions()
    frames = build_frames(rows)
    ranked, skipped = fwd5_ranked_stats(rows, events, frames)
    by_mean_desc = sorted(ranked, key=lambda item: (-item[2]["MeanDRFwd"], item[0], item[1]))
    by_mean_asc = sorted(ranked, key=lambda item: (item[2]["MeanDRFwd"], item[0], item[1]))
    by_continuation = sorted(
        ranked, key=lambda item: (-item[2]["ContinuationRate"], item[0], item[1])
    )
    by_failure = sorted(ranked, key=lambda item: (-item[2]["FailureRate"], item[0], item[1]))

    lines = [
        "APVA Context Frames Study v0.1",
        "==============================",
        f"Input: {path}",
        f"Total rows: {len(rows)}",
        f"Valid Black/Red polarity rows: {sum(row.polarity in {'Black', 'Red'} for row in rows)}",
        "DRFwd > 0: continuation of terminal event-bar polarity",
        "DRFwd < 0: failure of terminal event-bar polarity",
        f"Ranked-table minimum DRFwd5 samples: {MIN_RANKED_SAMPLES}",
        "\nFrame Membership Counts",
        "=======================",
    ]
    for frame in frames:
        lines.append(f"{frame.name:<30} {sum(frame.membership):>8}")

    append_full_metrics(lines, rows, events, frames)
    append_comparisons(lines, rows, events, frames)
    append_ranked_table(lines, "Top 25 Combinations by Highest MeanDRFwd5", by_mean_desc)
    append_ranked_table(lines, "Top 25 Combinations by Lowest MeanDRFwd5", by_mean_asc)
    append_ranked_table(lines, "Top 25 Combinations by Highest ContinuationRate5", by_continuation)
    append_ranked_table(lines, "Top 25 Combinations by Highest FailureRate5", by_failure)
    append_research_notes(lines, rows, events, frames, ranked, skipped)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_path = args.input_path or latest_csv(DEFAULT_INPUT_FOLDER)
    write_report(build_report(input_path), report_path("ContextFrames", input_path))


if __name__ == "__main__":
    main()
