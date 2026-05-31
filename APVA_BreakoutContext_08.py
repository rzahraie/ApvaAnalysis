#!/usr/bin/env python3
"""Study APVA evidence events after explicit IB/SYM and lateral breakouts.

This research-only script measures evidence-layer consequences in descriptive
post-breakout contexts. It does not create trading signals or infer structure.
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
WINDOWS = (1, 3, 5, 10)
HORIZONS = (1, 3, 5, 10, 20)
MIN_RANKED_SAMPLES = 30
TOP_LIMIT = 25
MATERIAL_DELTA_RATE = 0.05
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
    "AcceptanceState",
    "DissipationState",
    "CompressionState",
    "ExpansionState",
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
    acceptance: str
    dissipation: str
    compression: str
    expansion: str


@dataclass(frozen=True)
class EventDefinition:
    name: str
    predicate: Callable[[EvidenceBar], bool]


@dataclass
class FrameDiagnostics:
    frames: int = 0
    breakout_up: int = 0
    breakout_down: int = 0
    failed_breakouts: int = 0
    total_length: int = 0

    @property
    def average_length(self) -> float:
        return self.total_length / self.frames if self.frames else 0.0


@dataclass
class ActiveFrame:
    start_index: int
    high: float
    low: float


@dataclass(frozen=True)
class Breakout:
    index: int
    source: str
    direction: str


@dataclass(frozen=True)
class CombinationStats:
    event_name: str
    context_name: str
    inside: dict[str, float | int]
    outside: dict[str, float | int]

    @property
    def delta_mean(self) -> float:
        return float(self.inside["MeanDRFwd"]) - float(self.outside["MeanDRFwd"])

    @property
    def delta_continuation_rate(self) -> float:
        return float(self.inside["ContinuationRate"]) - float(
            self.outside["ContinuationRate"]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study APVA evidence events after IB/SYM and lateral breakouts."
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
                    acceptance=(raw.get("AcceptanceState") or "").strip(),
                    dissipation=(raw.get("DissipationState") or "").strip(),
                    compression=(raw.get("CompressionState") or "").strip(),
                    expansion=(raw.get("ExpansionState") or "").strip(),
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
        EventDefinition("ParticipationPeak", lambda row: row.participation == "Peak"),
        EventDefinition(
            "ParticipationClimactic", lambda row: row.participation == "Climactic"
        ),
        EventDefinition(
            "PeakContained",
            lambda row: row.participation == "Peak" and row.acceptance == "Contained",
        ),
        EventDefinition(
            "ClimacticContained",
            lambda row: row.participation == "Climactic"
            and row.acceptance == "Contained",
        ),
    ]


def inside_bar(rows: list[EvidenceBar], index: int) -> bool:
    return (
        index > 0
        and rows[index].high <= rows[index - 1].high
        and rows[index].low >= rows[index - 1].low
    )


def symmetric_candidate(rows: list[EvidenceBar], index: int) -> bool:
    if index < 20:
        return False
    prior_ranges = [row.high - row.low for row in rows[index - 20 : index]]
    median_range = statistics.median(prior_ranges)
    return rows[index].high - rows[index].low <= median_range * 0.60


def classify_breakout(bar: EvidenceBar, frame: ActiveFrame) -> str | None:
    if bar.high > frame.high and bar.close > frame.high:
        return "Up"
    if bar.low < frame.low and bar.close < frame.low:
        return "Down"
    if bar.high > frame.high or bar.low < frame.low:
        return "Failed"
    return None


def ibsym_start(rows: list[EvidenceBar], index: int) -> ActiveFrame | None:
    if index >= 1 and inside_bar(rows, index - 1) and inside_bar(rows, index):
        start = index - 1
        bars = rows[start : index + 1]
        return ActiveFrame(start, max(row.high for row in bars), min(row.low for row in bars))
    if index >= 2 and all(symmetric_candidate(rows, offset) for offset in range(index - 2, index + 1)):
        start = index - 2
        bars = rows[start : index + 1]
        return ActiveFrame(start, max(row.high for row in bars), min(row.low for row in bars))
    return None


def lateral_start(rows: list[EvidenceBar], index: int) -> ActiveFrame | None:
    if index < 2:
        return None
    first = rows[index - 2]
    if (
        rows[index - 1].high <= first.high
        and rows[index - 1].low >= first.low
        and rows[index].high <= first.high
        and rows[index].low >= first.low
    ):
        return ActiveFrame(index - 2, first.high, first.low)
    return None


def detect_frames(
    rows: list[EvidenceBar],
    source: str,
    start_detector: Callable[[list[EvidenceBar], int], ActiveFrame | None],
) -> tuple[list[Breakout], FrameDiagnostics]:
    diagnostics = FrameDiagnostics()
    breakouts: list[Breakout] = []
    active: ActiveFrame | None = None

    for index, bar in enumerate(rows):
        if active is None:
            active = start_detector(rows, index)
            if active is not None:
                diagnostics.frames += 1
            continue

        result = classify_breakout(bar, active)
        if result == "Failed":
            diagnostics.failed_breakouts += 1
            continue
        if result in {"Up", "Down"}:
            diagnostics.total_length += index - active.start_index
            if result == "Up":
                diagnostics.breakout_up += 1
            else:
                diagnostics.breakout_down += 1
            breakouts.append(Breakout(index, source, result))
            active = None

    if active is not None:
        diagnostics.total_length += len(rows) - active.start_index
    return breakouts, diagnostics


def build_contexts(
    row_count: int,
    breakouts: list[Breakout],
) -> dict[str, tuple[bool, ...]]:
    contexts: dict[str, list[bool]] = {}
    for source in ("IBSYM", "Lateral"):
        for direction in ("Up", "Down"):
            for window in WINDOWS:
                contexts[f"Post{source}Breakout{direction}_{window}"] = [False] * row_count
    for window in WINDOWS:
        contexts[f"PostAnyBreakout_{window}"] = [False] * row_count

    for breakout in breakouts:
        for window in WINDOWS:
            stop = min(row_count, breakout.index + window + 1)
            specific = f"Post{breakout.source}Breakout{breakout.direction}_{window}"
            any_breakout = f"PostAnyBreakout_{window}"
            for index in range(breakout.index + 1, stop):
                contexts[specific][index] = True
                contexts[any_breakout][index] = True
    return {name: tuple(values) for name, values in contexts.items()}


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


def values_for(
    rows: list[EvidenceBar],
    event: EventDefinition,
    context: tuple[bool, ...],
    horizon: int,
    inside: bool,
) -> list[float]:
    values: list[float] = []
    for index, row in enumerate(rows):
        if context[index] != inside or not event.predicate(row):
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
        "ContinuationRate": continuation / denominator,
        "FailureRate": failure / denominator,
        "FlatRate": flat / denominator,
    }


def build_combinations(
    rows: list[EvidenceBar],
    events: list[EventDefinition],
    contexts: dict[str, tuple[bool, ...]],
) -> list[CombinationStats]:
    combinations: list[CombinationStats] = []
    for context_name, context in contexts.items():
        for event in events:
            combinations.append(
                CombinationStats(
                    event_name=event.name,
                    context_name=context_name,
                    inside=summarize(values_for(rows, event, context, 5, True)),
                    outside=summarize(values_for(rows, event, context, 5, False)),
                )
            )
    return combinations


def append_frame_diagnostics(
    lines: list[str],
    ibsym: FrameDiagnostics,
    lateral: FrameDiagnostics,
) -> None:
    lines.extend(["\nFrame Diagnostics", "================="])
    lines.append(f"IB/SYM frames:              {ibsym.frames}")
    lines.append(f"IB/SYM breakout up:         {ibsym.breakout_up}")
    lines.append(f"IB/SYM breakout down:       {ibsym.breakout_down}")
    lines.append(f"IB/SYM failed breakouts:    {ibsym.failed_breakouts}")
    lines.append(f"IB/SYM average frame length:{ibsym.average_length:>10.2f}")
    lines.append("")
    lines.append(f"Lateral frames:              {lateral.frames}")
    lines.append(f"Lateral breakout up:         {lateral.breakout_up}")
    lines.append(f"Lateral breakout down:       {lateral.breakout_down}")
    lines.append(f"Lateral failed breakouts:    {lateral.failed_breakouts}")
    lines.append(f"Lateral average frame length:{lateral.average_length:>10.2f}")


def append_main_report(lines: list[str], combinations: list[CombinationStats]) -> None:
    lines.extend(["\nEvent and Breakout-Context Comparisons", "======================================"])
    lines.append(
        f"{'Event':<24} {'Context':<32} {'InCount':>8} {'InMean5':>11} "
        f"{'InMedian5':>11} {'InCont5':>9} {'InFail5':>9} {'OutCount':>9} "
        f"{'OutMean5':>11} {'OutCont5':>9} {'OutFail5':>9} {'DeltaMean5':>11} "
        f"{'DeltaCont5':>10}"
    )
    for item in combinations:
        lines.append(
            f"{item.event_name:<24} {item.context_name:<32} "
            f"{item.inside['Count']:>8} {item.inside['MeanDRFwd']:>11.6f} "
            f"{item.inside['MedianDRFwd']:>11.6f} {item.inside['ContinuationRate']:>8.2%} "
            f"{item.inside['FailureRate']:>8.2%} {item.outside['Count']:>9} "
            f"{item.outside['MeanDRFwd']:>11.6f} {item.outside['ContinuationRate']:>8.2%} "
            f"{item.outside['FailureRate']:>8.2%} {item.delta_mean:>11.6f} "
            f"{item.delta_continuation_rate:>9.2%}"
        )


def append_horizon_appendix(
    lines: list[str],
    rows: list[EvidenceBar],
    events: list[EventDefinition],
    contexts: dict[str, tuple[bool, ...]],
) -> None:
    lines.extend(["\nInside-Context Horizon Appendix", "==============================="])
    lines.append(
        f"{'Event':<24} {'Context':<32} {'Horizon':<8} {'Count':>8} "
        f"{'MeanDRFwd':>12} {'MedianDRFwd':>12} {'ContRate':>10} "
        f"{'FailRate':>10} {'FlatRate':>10}"
    )
    for context_name, context in contexts.items():
        for event in events:
            for horizon in HORIZONS:
                stats = summarize(values_for(rows, event, context, horizon, True))
                lines.append(
                    f"{event.name:<24} {context_name:<32} DRFwd{horizon:<2} "
                    f"{stats['Count']:>8} {stats['MeanDRFwd']:>12.6f} "
                    f"{stats['MedianDRFwd']:>12.6f} {stats['ContinuationRate']:>9.2%} "
                    f"{stats['FailureRate']:>9.2%} {stats['FlatRate']:>9.2%}"
                )


def append_ranked_table(
    lines: list[str],
    title: str,
    ranked: list[CombinationStats],
) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(
        f"{'Rank':>4} {'Event':<24} {'Context':<32} {'Count':>8} "
        f"{'MeanDRFwd5':>12} {'ContRate5':>10} {'FailRate5':>10} "
        f"{'DeltaCont5':>10}"
    )
    for rank, item in enumerate(ranked[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {item.event_name:<24} {item.context_name:<32} "
            f"{item.inside['Count']:>8} {item.inside['MeanDRFwd']:>12.6f} "
            f"{item.inside['ContinuationRate']:>9.2%} {item.inside['FailureRate']:>9.2%} "
            f"{item.delta_continuation_rate:>9.2%}"
        )


def average_delta_for(
    combinations: list[CombinationStats],
    event_name: str,
    context_prefix: str,
) -> float:
    values = [
        item.delta_continuation_rate
        for item in combinations
        if item.event_name == event_name and item.context_name.startswith(context_prefix)
    ]
    return statistics.fmean(values) if values else 0.0


def append_research_notes(
    lines: list[str],
    combinations: list[CombinationStats],
    ranked: list[CombinationStats],
    skipped: int,
) -> None:
    lines.extend(["\nResearch Notes", "=============="])
    dissipation = [
        item for item in combinations if item.event_name == "DissipationContained"
    ]
    reduced = [item for item in dissipation if item.delta_continuation_rate <= -MATERIAL_DELTA_RATE]
    lines.append(
        f"- DissipationContained post-breakout contexts with continuation-rate reduction "
        f"of at least {MATERIAL_DELTA_RATE:.0%}: {len(reduced)} of {len(dissipation)}."
    )

    ibsym_delta = average_delta_for(combinations, "DissipationContained", "PostIBSYM")
    lateral_delta = average_delta_for(combinations, "DissipationContained", "PostLateral")
    lines.append(
        f"- Mean DissipationContained DeltaContinuationRate5 across IB/SYM contexts: "
        f"{ibsym_delta:.2%}; across lateral contexts: {lateral_delta:.2%}."
    )

    any_windows = [
        item
        for item in combinations
        if item.event_name == "DissipationContained"
        and item.context_name.startswith("PostAnyBreakout_")
    ]
    any_windows.sort(key=lambda item: int(item.context_name.rsplit("_", 1)[1]))
    window_text = ", ".join(
        f"{item.context_name.rsplit('_', 1)[1]} bars={item.delta_continuation_rate:.2%}"
        for item in any_windows
    )
    lines.append(
        f"- DissipationContained PostAnyBreakout DeltaContinuationRate5 by window: "
        f"{window_text}."
    )

    peak = [item for item in combinations if item.event_name == "PeakContained"]
    worsens = min(peak, key=lambda item: (item.delta_continuation_rate, item.context_name))
    improves = max(peak, key=lambda item: (item.delta_continuation_rate, item.context_name))
    lines.append(
        f"- Context that most worsens PeakContained continuation rate: "
        f"{worsens.context_name} ({worsens.delta_continuation_rate:.2%})."
    )
    lines.append(
        f"- Context that most improves PeakContained continuation rate: "
        f"{improves.context_name} ({improves.delta_continuation_rate:.2%})."
    )
    lines.append(
        f"- Event/context combinations skipped from ranked tables because Count < "
        f"{MIN_RANKED_SAMPLES}: {skipped}."
    )
    if ranked:
        best = max(ranked, key=lambda item: (item.inside["MeanDRFwd"], item.event_name))
        worst = min(ranked, key=lambda item: (item.inside["MeanDRFwd"], item.event_name))
        lines.append(
            f"- Highest ranked MeanDRFwd5: {best.event_name} inside {best.context_name} "
            f"({best.inside['MeanDRFwd']:.6f}, n={best.inside['Count']})."
        )
        lines.append(
            f"- Lowest ranked MeanDRFwd5: {worst.event_name} inside {worst.context_name} "
            f"({worst.inside['MeanDRFwd']:.6f}, n={worst.inside['Count']})."
        )


def build_report(path: Path) -> str:
    rows = load_rows(path)
    events = event_definitions()
    ibsym_breakouts, ibsym_diagnostics = detect_frames(rows, "IBSYM", ibsym_start)
    lateral_breakouts, lateral_diagnostics = detect_frames(rows, "Lateral", lateral_start)
    contexts = build_contexts(len(rows), ibsym_breakouts + lateral_breakouts)
    combinations = build_combinations(rows, events, contexts)
    ranked = [item for item in combinations if item.inside["Count"] >= MIN_RANKED_SAMPLES]
    skipped = len(combinations) - len(ranked)

    lines = [
        "APVA IB/SYM and Lateral Breakout Context Study v0.1",
        "=================================================",
        f"Input: {path}",
        f"Total rows: {len(rows)}",
        f"Valid Black/Red polarity rows: {sum(row.polarity in {'Black', 'Red'} for row in rows)}",
        "Post-breakout windows exclude the breakout bar.",
        "Successful breakouts require boundary acceptance by close.",
        "Failed boundary pierces are counted and leave the original frame active.",
        "DRFwd > 0: continuation of terminal event-bar polarity",
        "DRFwd < 0: failure of terminal event-bar polarity",
        f"Ranked-table minimum DRFwd5 samples: {MIN_RANKED_SAMPLES}",
    ]
    append_frame_diagnostics(lines, ibsym_diagnostics, lateral_diagnostics)
    append_main_report(lines, combinations)
    append_horizon_appendix(lines, rows, events, contexts)
    append_ranked_table(
        lines,
        "Top 25 Best Combinations by MeanDRFwd5",
        sorted(ranked, key=lambda item: (-item.inside["MeanDRFwd"], item.event_name, item.context_name)),
    )
    append_ranked_table(
        lines,
        "Top 25 Worst Combinations by MeanDRFwd5",
        sorted(ranked, key=lambda item: (item.inside["MeanDRFwd"], item.event_name, item.context_name)),
    )
    append_ranked_table(
        lines,
        "Top 25 Largest Positive DeltaContinuationRate5",
        sorted(ranked, key=lambda item: (-item.delta_continuation_rate, item.event_name, item.context_name)),
    )
    append_ranked_table(
        lines,
        "Top 25 Largest Negative DeltaContinuationRate5",
        sorted(ranked, key=lambda item: (item.delta_continuation_rate, item.event_name, item.context_name)),
    )
    append_research_notes(lines, combinations, ranked, skipped)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_path = args.input_path or latest_csv(DEFAULT_INPUT_FOLDER)
    write_report(build_report(input_path), report_path("BreakoutContext", input_path))


if __name__ == "__main__":
    main()
