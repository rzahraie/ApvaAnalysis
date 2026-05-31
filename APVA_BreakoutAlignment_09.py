#!/usr/bin/env python3
"""Study event-polarity alignment after APVA IB/SYM and lateral breakouts.

This research-only script compares evidence events that align with or oppose a
recent accepted breakout. It does not create trading signals or infer structure.
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
    WINDOWS,
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


@dataclass(frozen=True)
class EventDefinition:
    name: str
    predicate: Callable[[EvidenceBar], bool]


@dataclass(frozen=True)
class AlignmentOccurrence:
    index: int
    alignment: str


@dataclass(frozen=True)
class ComparisonStats:
    event_name: str
    context_type: str
    window: int
    aligned: dict[str, float | int]
    opposed: dict[str, float | int]

    @property
    def delta_mean(self) -> float:
        return float(self.aligned["MeanDRFwd"]) - float(self.opposed["MeanDRFwd"])

    @property
    def delta_continuation_rate(self) -> float:
        return float(self.aligned["ContinuationRate"]) - float(
            self.opposed["ContinuationRate"]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study aligned and opposed APVA events after accepted breakouts."
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
            lambda row: row.participation == "Climactic"
            and row.acceptance == "Accepted",
        ),
    ]


def context_type(breakout: Breakout) -> str:
    return f"Post{breakout.source}Breakout{breakout.direction}"


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


def build_occurrences(
    rows: list[EvidenceBar],
    breakouts: list[Breakout],
) -> dict[tuple[str, int], list[AlignmentOccurrence]]:
    occurrences: dict[tuple[str, int], list[AlignmentOccurrence]] = {}
    for source in ("IBSYM", "Lateral"):
        for direction in ("Up", "Down"):
            for window in WINDOWS:
                occurrences[(f"Post{source}Breakout{direction}", window)] = []

    for breakout in breakouts:
        name = context_type(breakout)
        for window in WINDOWS:
            stop = min(len(rows), breakout.index + window + 1)
            for index in range(breakout.index + 1, stop):
                alignment = classify_alignment(breakout.direction, rows[index].polarity)
                if alignment is not None:
                    occurrences[(name, window)].append(AlignmentOccurrence(index, alignment))
    return occurrences


def values_for(
    rows: list[EvidenceBar],
    event: EventDefinition,
    occurrences: list[AlignmentOccurrence],
    alignment: str,
    horizon: int,
) -> list[float]:
    values: list[float] = []
    for occurrence in occurrences:
        if occurrence.alignment != alignment or not event.predicate(rows[occurrence.index]):
            continue
        value = direction_relative_return(rows, occurrence.index, horizon)
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


def append_main_report(
    lines: list[str],
    rows: list[EvidenceBar],
    events: list[EventDefinition],
    occurrences: dict[tuple[str, int], list[AlignmentOccurrence]],
) -> None:
    lines.extend(["\nAlignment Consequence Tables", "============================"])
    for (name, window), context_occurrences in occurrences.items():
        lines.extend([f"\nContext: {name}, Window={window}", "-" * (19 + len(name) + len(str(window)))])
        for event in events:
            for alignment in ("Aligned", "Opposed"):
                lines.append(f"\nEvent: {event.name}, Alignment={alignment}")
                lines.append(
                    f"{'Horizon':<8} {'Count':>8} {'MeanDRFwd':>12} "
                    f"{'MedianDRFwd':>14} {'ContRate':>10} {'FailRate':>10} {'FlatRate':>10}"
                )
                for horizon in HORIZONS:
                    stats = summarize(
                        values_for(rows, event, context_occurrences, alignment, horizon)
                    )
                    lines.append(
                        f"DRFwd{horizon:<2} {stats['Count']:>8} {stats['MeanDRFwd']:>12.6f} "
                        f"{stats['MedianDRFwd']:>14.6f} {stats['ContinuationRate']:>9.2%} "
                        f"{stats['FailureRate']:>9.2%} {stats['FlatRate']:>9.2%}"
                    )


def build_comparisons(
    rows: list[EvidenceBar],
    events: list[EventDefinition],
    occurrences: dict[tuple[str, int], list[AlignmentOccurrence]],
) -> list[ComparisonStats]:
    comparisons: list[ComparisonStats] = []
    for (name, window), context_occurrences in occurrences.items():
        for event in events:
            comparisons.append(
                ComparisonStats(
                    event_name=event.name,
                    context_type=name,
                    window=window,
                    aligned=summarize(values_for(rows, event, context_occurrences, "Aligned", 5)),
                    opposed=summarize(values_for(rows, event, context_occurrences, "Opposed", 5)),
                )
            )
    return comparisons


def append_comparison_report(lines: list[str], comparisons: list[ComparisonStats]) -> None:
    lines.extend(["\nAligned vs Opposed DRFwd5 Comparisons", "===================================="])
    lines.append(
        f"{'Event':<24} {'Context':<28} {'Window':>6} {'AlnCount':>8} "
        f"{'AlnMean5':>11} {'AlnCont5':>9} {'AlnFail5':>9} {'OppCount':>8} "
        f"{'OppMean5':>11} {'OppCont5':>9} {'OppFail5':>9} {'DeltaMean5':>11} "
        f"{'DeltaCont5':>10}"
    )
    for item in comparisons:
        lines.append(
            f"{item.event_name:<24} {item.context_type:<28} {item.window:>6} "
            f"{item.aligned['Count']:>8} {item.aligned['MeanDRFwd']:>11.6f} "
            f"{item.aligned['ContinuationRate']:>8.2%} {item.aligned['FailureRate']:>8.2%} "
            f"{item.opposed['Count']:>8} {item.opposed['MeanDRFwd']:>11.6f} "
            f"{item.opposed['ContinuationRate']:>8.2%} {item.opposed['FailureRate']:>8.2%} "
            f"{item.delta_mean:>11.6f} {item.delta_continuation_rate:>9.2%}"
        )


def append_ranked_table(
    lines: list[str],
    title: str,
    ranked: list[ComparisonStats],
    side: str,
) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(
        f"{'Rank':>4} {'Event':<24} {'Context':<28} {'Window':>6} {'Count':>8} "
        f"{'MeanDRFwd5':>12} {'ContRate5':>10} {'FailRate5':>10} {'DeltaCont5':>10}"
    )
    for rank, item in enumerate(ranked[:TOP_LIMIT], start=1):
        stats = item.aligned if side == "Aligned" else item.opposed
        lines.append(
            f"{rank:>4} {item.event_name:<24} {item.context_type:<28} {item.window:>6} "
            f"{stats['Count']:>8} {stats['MeanDRFwd']:>12.6f} "
            f"{stats['ContinuationRate']:>9.2%} {stats['FailureRate']:>9.2%} "
            f"{item.delta_continuation_rate:>9.2%}"
        )


def average_advantage(
    comparisons: list[ComparisonStats],
    event_name: str,
    context_fragment: str,
) -> float:
    values = [
        item.delta_continuation_rate
        for item in comparisons
        if item.event_name == event_name and context_fragment in item.context_type
    ]
    return statistics.fmean(values) if values else 0.0


def append_research_notes(
    lines: list[str],
    comparisons: list[ComparisonStats],
    ranked: list[ComparisonStats],
    skipped: int,
) -> None:
    lines.extend(["\nResearch Notes", "=============="])
    if ranked:
        best = max(ranked, key=lambda item: (item.delta_continuation_rate, item.event_name))
        worst = min(ranked, key=lambda item: (item.delta_continuation_rate, item.event_name))
        lines.append(
            f"- Largest alignment advantage: {best.event_name}, {best.context_type}, "
            f"window={best.window} ({best.delta_continuation_rate:.2%})."
        )
        lines.append(
            f"- Largest alignment disadvantage: {worst.event_name}, {worst.context_type}, "
            f"window={worst.window} ({worst.delta_continuation_rate:.2%})."
        )
    else:
        lines.append("- No comparisons met the two-sided ranked-table sample threshold.")

    dissipation = average_advantage(comparisons, "DissipationContained", "")
    peak = average_advantage(comparisons, "PeakContained", "")
    lines.append(
        f"- Mean alignment advantage across DissipationContained comparisons: "
        f"{dissipation:.2%}; across PeakContained comparisons: {peak:.2%}."
    )
    ibsym = average_advantage(comparisons, "DissipationContained", "IBSYM")
    lateral = average_advantage(comparisons, "DissipationContained", "Lateral")
    lines.append(
        f"- Mean DissipationContained alignment advantage across IB/SYM comparisons: "
        f"{ibsym:.2%}; across lateral comparisons: {lateral:.2%}."
    )
    up = average_advantage(comparisons, "DissipationContained", "BreakoutUp")
    down = average_advantage(comparisons, "DissipationContained", "BreakoutDown")
    lines.append(
        f"- Mean DissipationContained alignment advantage after breakout up: "
        f"{up:.2%}; after breakout down: {down:.2%}."
    )
    lines.append(
        f"- Comparisons skipped from ranked tables because aligned or opposed Count < "
        f"{MIN_RANKED_SAMPLES}: {skipped}."
    )


def build_report(path: Path) -> str:
    rows = load_rows(path)
    events = event_definitions()
    ibsym_breakouts, ibsym_diagnostics = detect_frames(rows, "IBSYM", ibsym_start)
    lateral_breakouts, lateral_diagnostics = detect_frames(rows, "Lateral", lateral_start)
    breakouts = ibsym_breakouts + lateral_breakouts
    occurrences = build_occurrences(rows, breakouts)
    comparisons = build_comparisons(rows, events, occurrences)
    ranked = [
        item
        for item in comparisons
        if item.aligned["Count"] >= MIN_RANKED_SAMPLES
        and item.opposed["Count"] >= MIN_RANKED_SAMPLES
    ]
    skipped = len(comparisons) - len(ranked)
    total_aligned = sum(
        occurrence.alignment == "Aligned"
        for context_occurrences in occurrences.values()
        for occurrence in context_occurrences
    )
    total_opposed = sum(
        occurrence.alignment == "Opposed"
        for context_occurrences in occurrences.values()
        for occurrence in context_occurrences
    )

    lines = [
        "APVA Breakout Alignment Study v0.1",
        "==================================",
        f"Input: {path}",
        f"Total rows: {len(rows)}",
        "Post-breakout windows exclude the breakout bar.",
        "Breakout detection reuses APVA_BreakoutContext_08.py.",
        "Occurrences are counted per breakout context/window; overlapping contexts may include the same bar.",
        "DRFwd > 0: continuation of terminal event-bar polarity",
        "DRFwd < 0: failure of terminal event-bar polarity",
        f"Ranked comparisons require both aligned and opposed DRFwd5 Count >= {MIN_RANKED_SAMPLES}.",
        "\nFrame Diagnostics",
        "=================",
        f"IB/SYM breakout up:     {ibsym_diagnostics.breakout_up}",
        f"IB/SYM breakout down:   {ibsym_diagnostics.breakout_down}",
        f"Lateral breakout up:    {lateral_diagnostics.breakout_up}",
        f"Lateral breakout down:  {lateral_diagnostics.breakout_down}",
        f"Aligned event occurrences across context windows: {total_aligned}",
        f"Opposed event occurrences across context windows: {total_opposed}",
    ]
    append_main_report(lines, rows, events, occurrences)
    append_comparison_report(lines, comparisons)
    append_ranked_table(
        lines,
        "Top 25 Aligned Combinations by MeanDRFwd5",
        sorted(ranked, key=lambda item: (-item.aligned["MeanDRFwd"], item.event_name, item.context_type, item.window)),
        "Aligned",
    )
    append_ranked_table(
        lines,
        "Top 25 Opposed Combinations by MeanDRFwd5",
        sorted(ranked, key=lambda item: (-item.opposed["MeanDRFwd"], item.event_name, item.context_type, item.window)),
        "Opposed",
    )
    append_ranked_table(
        lines,
        "Top 25 Largest Positive Alignment Advantages",
        sorted(ranked, key=lambda item: (-item.delta_continuation_rate, item.event_name, item.context_type, item.window)),
        "Aligned",
    )
    append_ranked_table(
        lines,
        "Top 25 Largest Negative Alignment Advantages",
        sorted(ranked, key=lambda item: (item.delta_continuation_rate, item.event_name, item.context_type, item.window)),
        "Aligned",
    )
    append_research_notes(lines, comparisons, ranked, skipped)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_path = args.input_path or latest_csv(DEFAULT_INPUT_FOLDER)
    write_report(build_report(input_path), report_path("BreakoutAlignment", input_path))


if __name__ == "__main__":
    main()
