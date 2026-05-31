#!/usr/bin/env python3
"""Study a crude post-breakout OOE reset proxy in APVA evidence exports.

This research-only script measures evidence consequences before and after an
ordered polarity/participation proxy completes. The proxy is not true PVA OOE.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from APVA_BreakoutContext_08 import (
    DEFAULT_INPUT_FOLDER,
    HORIZONS,
    MIN_RANKED_SAMPLES,
    TOP_LIMIT,
    Breakout,
    EvidenceBar,
    detect_frames,
    direction_relative_return,
    ibsym_start,
    latest_csv,
    lateral_start,
    load_rows,
)
from APVA_Evidence_Report_IO import report_path, write_report


MAX_SEGMENT_BARS = 20


@dataclass(frozen=True)
class EventDefinition:
    name: str
    predicate: Callable[[EvidenceBar], bool]


@dataclass(frozen=True)
class SegmentBar:
    index: int
    breakout_type: str
    breakout_direction: str
    bars_since_breakout: int
    alignment: str | None
    phase_token: str
    sequence_stage: int
    complete_sequence_formed: bool


@dataclass(frozen=True)
class ComparisonStats:
    event_name: str
    breakout_type: str
    breakout_direction: str
    alignment: str
    incomplete: dict[str, float | int]
    complete: dict[str, float | int]

    @property
    def delta_mean(self) -> float:
        return float(self.complete["MeanDRFwd"]) - float(self.incomplete["MeanDRFwd"])

    @property
    def delta_continuation_rate(self) -> float:
        return float(self.complete["ContinuationRate"]) - float(
            self.incomplete["ContinuationRate"]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study a crude ordered post-breakout OOE reset proxy."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        help="Evidence CSV path. Defaults to the latest CSV in Evidence/6E.",
    )
    return parser.parse_args()


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


def classify_alignment(direction: str, polarity: str) -> str | None:
    if direction == "Up" and polarity == "Black":
        return "Aligned"
    if direction == "Up" and polarity == "Red":
        return "Opposed"
    if direction == "Down" and polarity == "Red":
        return "Aligned"
    if direction == "Down" and polarity == "Black":
        return "Opposed"
    return None


def phase_token(row: EvidenceBar) -> str:
    if row.polarity == "Black" and row.participation == "Falling":
        return "BlackFalling"
    if row.polarity == "Black" and row.participation in {"Rising", "Peak", "Climactic"}:
        return "BlackRisingOrPeak"
    if row.polarity == "Red" and row.participation == "Falling":
        return "RedFalling"
    if row.polarity == "Red" and row.participation in {"Rising", "Peak", "Climactic"}:
        return "RedRisingOrPeak"
    return "Other"


def sequence_tokens(direction: str) -> tuple[set[str], set[str], set[str], set[str]]:
    if direction == "Up":
        return (
            {"BlackFalling", "BlackRisingOrPeak"},
            {"RedRisingOrPeak"},
            {"RedFalling"},
            {"BlackRisingOrPeak"},
        )
    return (
        {"RedFalling", "RedRisingOrPeak"},
        {"BlackRisingOrPeak"},
        {"BlackFalling"},
        {"RedRisingOrPeak"},
    )


def advance_stage(stage: int, token: str, direction: str) -> int:
    if stage >= 4:
        return 4
    required = sequence_tokens(direction)
    if token in required[stage]:
        return stage + 1
    return stage


def next_distinct_breakout_indexes(breakouts: list[Breakout]) -> dict[int, int | None]:
    indexes = sorted({breakout.index for breakout in breakouts})
    return {
        index: indexes[position + 1] if position + 1 < len(indexes) else None
        for position, index in enumerate(indexes)
    }


def build_segment_bars(
    rows: list[EvidenceBar],
    breakouts: list[Breakout],
) -> tuple[list[SegmentBar], list[int]]:
    segment_bars: list[SegmentBar] = []
    completion_bars: list[int] = []
    next_indexes = next_distinct_breakout_indexes(breakouts)

    for breakout in sorted(breakouts, key=lambda item: (item.index, item.source, item.direction)):
        stop = min(len(rows), breakout.index + MAX_SEGMENT_BARS + 1)
        next_breakout = next_indexes[breakout.index]
        if next_breakout is not None:
            stop = min(stop, next_breakout)

        stage = 0
        reached_at: int | None = None
        for index in range(breakout.index + 1, stop):
            token = phase_token(rows[index])
            stage = advance_stage(stage, token, breakout.direction)
            bars_since = index - breakout.index
            if stage == 4 and reached_at is None:
                reached_at = bars_since
            segment_bars.append(
                SegmentBar(
                    index=index,
                    breakout_type=breakout.source,
                    breakout_direction=breakout.direction,
                    bars_since_breakout=bars_since,
                    alignment=classify_alignment(breakout.direction, rows[index].polarity),
                    phase_token=token,
                    sequence_stage=stage,
                    complete_sequence_formed=stage == 4,
                )
            )
        if reached_at is not None:
            completion_bars.append(reached_at)
    return segment_bars, completion_bars


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


def values_for(
    rows: list[EvidenceBar],
    segment_bars: list[SegmentBar],
    event: EventDefinition,
    horizon: int,
    predicate: Callable[[SegmentBar], bool],
) -> list[float]:
    values: list[float] = []
    for segment_bar in segment_bars:
        if not predicate(segment_bar) or not event.predicate(rows[segment_bar.index]):
            continue
        value = direction_relative_return(rows, segment_bar.index, horizon)
        if value is not None:
            values.append(value)
    return values


def build_comparisons(
    rows: list[EvidenceBar],
    segment_bars: list[SegmentBar],
    events: list[EventDefinition],
) -> list[ComparisonStats]:
    comparisons: list[ComparisonStats] = []
    for event in events:
        for breakout_type in ("IBSYM", "Lateral"):
            for direction in ("Up", "Down"):
                for alignment in ("Aligned", "Opposed"):
                    base = lambda item, breakout_type=breakout_type, direction=direction, alignment=alignment: (
                        item.breakout_type == breakout_type
                        and item.breakout_direction == direction
                        and item.alignment == alignment
                    )
                    incomplete = summarize(
                        values_for(
                            rows,
                            segment_bars,
                            event,
                            5,
                            lambda item, base=base: base(item)
                            and not item.complete_sequence_formed,
                        )
                    )
                    complete = summarize(
                        values_for(
                            rows,
                            segment_bars,
                            event,
                            5,
                            lambda item, base=base: base(item)
                            and item.complete_sequence_formed,
                        )
                    )
                    comparisons.append(
                        ComparisonStats(
                            event.name,
                            breakout_type,
                            direction,
                            alignment,
                            incomplete,
                            complete,
                        )
                    )
    return comparisons


def append_comparison_report(lines: list[str], comparisons: list[ComparisonStats]) -> None:
    lines.extend(["\nIncomplete vs Complete OOE-Proxy Comparisons", "============================================"])
    lines.append(
        f"{'Event':<24} {'Type':<8} {'Dir':<5} {'Alignment':<8} "
        f"{'IncCount':>8} {'IncMean5':>11} {'IncCont5':>9} {'IncFail5':>9} "
        f"{'CmpCount':>8} {'CmpMean5':>11} {'CmpCont5':>9} {'CmpFail5':>9} "
        f"{'DeltaMean5':>11} {'DeltaCont5':>10}"
    )
    for item in comparisons:
        lines.append(
            f"{item.event_name:<24} {item.breakout_type:<8} {item.breakout_direction:<5} "
            f"{item.alignment:<8} {item.incomplete['Count']:>8} "
            f"{item.incomplete['MeanDRFwd']:>11.6f} {item.incomplete['ContinuationRate']:>8.2%} "
            f"{item.incomplete['FailureRate']:>8.2%} {item.complete['Count']:>8} "
            f"{item.complete['MeanDRFwd']:>11.6f} {item.complete['ContinuationRate']:>8.2%} "
            f"{item.complete['FailureRate']:>8.2%} {item.delta_mean:>11.6f} "
            f"{item.delta_continuation_rate:>9.2%}"
        )


def append_stage_report(
    lines: list[str],
    rows: list[EvidenceBar],
    segment_bars: list[SegmentBar],
    events: list[EventDefinition],
) -> None:
    lines.extend(["\nSequence Stage DRFwd5 Report", "============================"])
    lines.append(
        f"{'Event':<24} {'Stage':>5} {'Count':>8} {'MeanDRFwd5':>12} "
        f"{'ContRate5':>10} {'FailRate5':>10}"
    )
    for event in events:
        for stage in range(5):
            stats = summarize(
                values_for(
                    rows,
                    segment_bars,
                    event,
                    5,
                    lambda item, stage=stage: item.sequence_stage == stage,
                )
            )
            lines.append(
                f"{event.name:<24} {stage:>5} {stats['Count']:>8} "
                f"{stats['MeanDRFwd']:>12.6f} {stats['ContinuationRate']:>9.2%} "
                f"{stats['FailureRate']:>9.2%}"
            )


def bars_since_bucket(value: int) -> str:
    if value == 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 5:
        return "4-5"
    if value <= 10:
        return "6-10"
    return "11-20"


def append_bars_since_report(
    lines: list[str],
    rows: list[EvidenceBar],
    segment_bars: list[SegmentBar],
    events: list[EventDefinition],
) -> None:
    lines.extend(["\nBars-Since-Breakout DRFwd5 Report", "================================="])
    lines.append(
        f"{'Event':<24} {'Bucket':<6} {'Count':>8} {'MeanDRFwd5':>12} "
        f"{'ContRate5':>10} {'FailRate5':>10}"
    )
    for event in events:
        for bucket in ("1", "2-3", "4-5", "6-10", "11-20"):
            stats = summarize(
                values_for(
                    rows,
                    segment_bars,
                    event,
                    5,
                    lambda item, bucket=bucket: bars_since_bucket(item.bars_since_breakout)
                    == bucket,
                )
            )
            lines.append(
                f"{event.name:<24} {bucket:<6} {stats['Count']:>8} "
                f"{stats['MeanDRFwd']:>12.6f} {stats['ContinuationRate']:>9.2%} "
                f"{stats['FailureRate']:>9.2%}"
            )


def append_horizon_report(
    lines: list[str],
    rows: list[EvidenceBar],
    segment_bars: list[SegmentBar],
    events: list[EventDefinition],
) -> None:
    lines.extend(["\nIncomplete vs Complete Horizon Appendix", "======================================="])
    lines.append(
        f"{'Event':<24} {'State':<10} {'Horizon':<8} {'Count':>8} "
        f"{'MeanDRFwd':>12} {'Median':>12} {'ContRate':>10} {'FailRate':>10} {'FlatRate':>10}"
    )
    for event in events:
        for complete in (False, True):
            state = "Complete" if complete else "Incomplete"
            for horizon in HORIZONS:
                stats = summarize(
                    values_for(
                        rows,
                        segment_bars,
                        event,
                        horizon,
                        lambda item, complete=complete: item.complete_sequence_formed == complete,
                    )
                )
                lines.append(
                    f"{event.name:<24} {state:<10} DRFwd{horizon:<2} {stats['Count']:>8} "
                    f"{stats['MeanDRFwd']:>12.6f} {stats['MedianDRFwd']:>12.6f} "
                    f"{stats['ContinuationRate']:>9.2%} {stats['FailureRate']:>9.2%} "
                    f"{stats['FlatRate']:>9.2%}"
                )


def append_ranked_table(
    lines: list[str],
    title: str,
    ranked: list[ComparisonStats],
    state: str,
) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(
        f"{'Rank':>4} {'Event':<24} {'Type':<8} {'Dir':<5} {'Alignment':<8} "
        f"{'Count':>8} {'MeanDRFwd5':>12} {'ContRate5':>10} {'FailRate5':>10} "
        f"{'DeltaCont5':>10}"
    )
    for rank, item in enumerate(ranked[:TOP_LIMIT], start=1):
        stats = item.complete if state == "Complete" else item.incomplete
        lines.append(
            f"{rank:>4} {item.event_name:<24} {item.breakout_type:<8} "
            f"{item.breakout_direction:<5} {item.alignment:<8} {stats['Count']:>8} "
            f"{stats['MeanDRFwd']:>12.6f} {stats['ContinuationRate']:>9.2%} "
            f"{stats['FailureRate']:>9.2%} {item.delta_continuation_rate:>9.2%}"
        )


def mean_delta(
    comparisons: list[ComparisonStats],
    predicate: Callable[[ComparisonStats], bool],
) -> float:
    values = [item.delta_continuation_rate for item in comparisons if predicate(item)]
    return statistics.fmean(values) if values else 0.0


def append_research_notes(
    lines: list[str],
    comparisons: list[ComparisonStats],
    ranked: list[ComparisonStats],
    skipped: int,
) -> None:
    lines.extend(["\nResearch Notes", "=============="])
    overall_delta = mean_delta(comparisons, lambda item: True)
    lines.append(
        f"- Mean complete-minus-incomplete DeltaContinuationRate5 across all comparisons: "
        f"{overall_delta:.2%}."
    )
    lines.append(
        f"- Mean delta for IB/SYM comparisons: "
        f"{mean_delta(comparisons, lambda item: item.breakout_type == 'IBSYM'):.2%}; "
        f"for lateral comparisons: "
        f"{mean_delta(comparisons, lambda item: item.breakout_type == 'Lateral'):.2%}."
    )
    lines.append(
        f"- Mean delta after breakout up: "
        f"{mean_delta(comparisons, lambda item: item.breakout_direction == 'Up'):.2%}; "
        f"after breakout down: "
        f"{mean_delta(comparisons, lambda item: item.breakout_direction == 'Down'):.2%}."
    )
    lines.append(
        f"- Mean delta for aligned comparisons: "
        f"{mean_delta(comparisons, lambda item: item.alignment == 'Aligned'):.2%}; "
        f"for opposed comparisons: "
        f"{mean_delta(comparisons, lambda item: item.alignment == 'Opposed'):.2%}."
    )
    if ranked:
        best = max(ranked, key=lambda item: (item.delta_continuation_rate, item.event_name))
        worst = min(ranked, key=lambda item: (item.delta_continuation_rate, item.event_name))
        lines.append(
            f"- Largest improvement after complete OOE proxy: {best.event_name}, "
            f"{best.breakout_type}, {best.breakout_direction}, {best.alignment} "
            f"({best.delta_continuation_rate:.2%})."
        )
        lines.append(
            f"- Largest deterioration after complete OOE proxy: {worst.event_name}, "
            f"{worst.breakout_type}, {worst.breakout_direction}, {worst.alignment} "
            f"({worst.delta_continuation_rate:.2%})."
        )
    lines.append(
        f"- Comparisons skipped from ranked tables because incomplete or complete Count < "
        f"{MIN_RANKED_SAMPLES}: {skipped}."
    )


def build_report(path: Path) -> str:
    rows = load_rows(path)
    events = event_definitions()
    ibsym_breakouts, ibsym_diagnostics = detect_frames(rows, "IBSYM", ibsym_start)
    lateral_breakouts, lateral_diagnostics = detect_frames(rows, "Lateral", lateral_start)
    breakouts = ibsym_breakouts + lateral_breakouts
    segment_bars, completion_bars = build_segment_bars(rows, breakouts)
    comparisons = build_comparisons(rows, segment_bars, events)
    ranked = [
        item
        for item in comparisons
        if item.incomplete["Count"] >= MIN_RANKED_SAMPLES
        and item.complete["Count"] >= MIN_RANKED_SAMPLES
    ]
    skipped = len(comparisons) - len(ranked)
    completed_bars = sum(item.complete_sequence_formed for item in segment_bars)

    lines = [
        "APVA Post-Breakout OOE Reset Study v0.1",
        "======================================",
        f"Input: {path}",
        f"Total rows: {len(rows)}",
        "OOE proxy: crude ordered VolumePolarity + ParticipationState sequence only.",
        "This is not true PVA OOE.",
        "Segments begin on the first bar after an accepted breakout.",
        f"Segments end at the next accepted breakout, after {MAX_SEGMENT_BARS} bars, or at end of data.",
        "Same-bar IB/SYM and lateral breakouts create parallel source-specific segments.",
        "DRFwd > 0: continuation of terminal event-bar polarity",
        "DRFwd < 0: failure of terminal event-bar polarity",
        f"Ranked comparisons require incomplete and complete DRFwd5 Count >= {MIN_RANKED_SAMPLES}.",
        "\nDiagnostics",
        "===========",
        f"Successful IB/SYM breakouts:    {ibsym_diagnostics.breakout_up + ibsym_diagnostics.breakout_down}",
        f"Successful lateral breakouts:   {lateral_diagnostics.breakout_up + lateral_diagnostics.breakout_down}",
        f"Post-breakout segments:         {len(breakouts)}",
        f"Post-breakout segment bars:     {len(segment_bars)}",
        f"Bars reaching SequenceStage 4:  {completed_bars / len(segment_bars) if segment_bars else 0.0:.2%}",
        f"Average bars to SequenceStage 4:{statistics.fmean(completion_bars) if completion_bars else 0.0:>10.2f}",
    ]
    append_comparison_report(lines, comparisons)
    append_stage_report(lines, rows, segment_bars, events)
    append_bars_since_report(lines, rows, segment_bars, events)
    append_horizon_report(lines, rows, segment_bars, events)
    append_ranked_table(
        lines,
        "Largest Improvements after CompleteSequenceFormed",
        sorted(ranked, key=lambda item: (-item.delta_continuation_rate, item.event_name)),
        "Complete",
    )
    append_ranked_table(
        lines,
        "Largest Deteriorations after CompleteSequenceFormed",
        sorted(ranked, key=lambda item: (item.delta_continuation_rate, item.event_name)),
        "Complete",
    )
    append_ranked_table(
        lines,
        "Worst Incomplete-Sequence Contexts",
        sorted(ranked, key=lambda item: (item.incomplete["MeanDRFwd"], item.event_name)),
        "Incomplete",
    )
    append_ranked_table(
        lines,
        "Best Complete-Sequence Contexts",
        sorted(ranked, key=lambda item: (-item.complete["MeanDRFwd"], item.event_name)),
        "Complete",
    )
    append_research_notes(lines, comparisons, ranked, skipped)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_path = args.input_path or latest_csv(DEFAULT_INPUT_FOLDER)
    write_report(build_report(input_path), report_path("PostBreakoutOOE", input_path))


if __name__ == "__main__":
    main()
