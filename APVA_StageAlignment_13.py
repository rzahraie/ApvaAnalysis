#!/usr/bin/env python3
"""Measure APVA SequenceStage and breakout-alignment interactions.

This research-only script reuses the crude post-breakout stage proxy from study
10 and the alignment definition from study 09. It is not a trade study and does
not model true PVA OOE.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from APVA_BreakoutAlignment_09 import classify_alignment
from APVA_BreakoutContext_08 import (
    DEFAULT_INPUT_FOLDER,
    MIN_RANKED_SAMPLES,
    TOP_LIMIT,
    EvidenceBar,
    detect_frames,
    direction_relative_return,
    ibsym_start,
    latest_csv,
    lateral_start,
    load_rows,
)
from APVA_Evidence_Report_IO import report_path, write_report
from APVA_PostBreakoutOOE_10 import SegmentBar, build_segment_bars


MATURITY_STAGES = {
    "Immature": {0, 1},
    "Developing": {2, 3},
    "Mature": {4},
}


@dataclass(frozen=True)
class EventDefinition:
    name: str
    predicate: Callable[[EvidenceBar], bool]


@dataclass(frozen=True)
class Comparison:
    event_name: str
    left_label: str
    right_label: str
    left: dict[str, float | int]
    right: dict[str, float | int]

    @property
    def delta_mean(self) -> float:
        return float(self.left["MeanDRFwd"]) - float(self.right["MeanDRFwd"])

    @property
    def delta_continuation_rate(self) -> float:
        return float(self.left["ContinuationRate"]) - float(
            self.right["ContinuationRate"]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure APVA SequenceStage and breakout-alignment interactions."
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
            lambda row: row.participation == "Climactic" and row.acceptance == "Accepted",
        ),
    ]


def alignment_for(rows: list[EvidenceBar], segment_bar: SegmentBar) -> str | None:
    return classify_alignment(
        segment_bar.breakout_direction,
        rows[segment_bar.index].polarity,
    )


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
    stages: set[int],
    alignment: str,
    horizon: int = 5,
) -> list[float]:
    values: list[float] = []
    for segment_bar in segment_bars:
        if segment_bar.sequence_stage not in stages:
            continue
        if alignment_for(rows, segment_bar) != alignment:
            continue
        if not event.predicate(rows[segment_bar.index]):
            continue
        value = direction_relative_return(rows, segment_bar.index, horizon)
        if value is not None:
            values.append(value)
    return values


def stats_for(
    rows: list[EvidenceBar],
    segment_bars: list[SegmentBar],
    event: EventDefinition,
    stages: set[int],
    alignment: str,
) -> dict[str, float | int]:
    return summarize(values_for(rows, segment_bars, event, stages, alignment))


def append_diagnostics(
    lines: list[str],
    rows: list[EvidenceBar],
    segment_bars: list[SegmentBar],
    events: list[EventDefinition],
) -> None:
    valid = [bar for bar in segment_bars if alignment_for(rows, bar) is not None]
    lines.extend(["\nDiagnostics", "==========="])
    lines.append(f"Total stage-alignment observations: {len(valid)}")
    for stage in range(5):
        lines.append(f"Stage{stage} observations: {sum(bar.sequence_stage == stage for bar in valid)}")
    for alignment in ("Aligned", "Opposed"):
        lines.append(
            f"{alignment} observations: "
            f"{sum(alignment_for(rows, bar) == alignment for bar in valid)}"
        )

    lines.extend(["\nStage x Alignment Counts", "========================"])
    lines.append(f"{'Stage':<8} {'Aligned':>10} {'Opposed':>10}")
    for stage in range(5):
        lines.append(
            f"Stage{stage:<2} "
            f"{sum(bar.sequence_stage == stage and alignment_for(rows, bar) == 'Aligned' for bar in valid):>10} "
            f"{sum(bar.sequence_stage == stage and alignment_for(rows, bar) == 'Opposed' for bar in valid):>10}"
        )

    lines.extend(["\nEvent Counts by Stage x Alignment", "================================="])
    lines.append(
        f"{'Event':<24} "
        + " ".join(
            f"{'S' + str(stage) + 'A':>7} {'S' + str(stage) + 'O':>7}"
            for stage in range(5)
        )
    )
    for event in events:
        counts = []
        for stage in range(5):
            for alignment in ("Aligned", "Opposed"):
                counts.append(
                    sum(
                        bar.sequence_stage == stage
                        and alignment_for(rows, bar) == alignment
                        and event.predicate(rows[bar.index])
                        for bar in valid
                    )
                )
        lines.append(f"{event.name:<24} " + " ".join(f"{count:>7}" for count in counts))


def append_main_table(
    lines: list[str],
    rows: list[EvidenceBar],
    segment_bars: list[SegmentBar],
    events: list[EventDefinition],
) -> None:
    lines.extend(["\nEvent x Stage x Alignment DRFwd5 Table", "====================================="])
    lines.append(
        f"{'Event':<24} {'Stage':>7} {'Alignment':<8} {'Count':>8} "
        f"{'MeanDRFwd5':>12} {'Median':>12} {'ContRate5':>10} "
        f"{'FailRate5':>10} {'FlatRate5':>10}"
    )
    for event in events:
        for stage in range(5):
            for alignment in ("Aligned", "Opposed"):
                stats = stats_for(rows, segment_bars, event, {stage}, alignment)
                lines.append(
                    f"{event.name:<24} {'Stage' + str(stage):>7} {alignment:<8} "
                    f"{stats['Count']:>8} {stats['MeanDRFwd']:>12.6f} "
                    f"{stats['MedianDRFwd']:>12.6f} {stats['ContinuationRate']:>9.2%} "
                    f"{stats['FailureRate']:>9.2%} {stats['FlatRate']:>9.2%}"
                )


def make_comparisons(
    rows: list[EvidenceBar],
    segment_bars: list[SegmentBar],
    events: list[EventDefinition],
    kind: str,
) -> list[Comparison]:
    comparisons: list[Comparison] = []
    for event in events:
        if kind == "Stage4Alignment":
            left = stats_for(rows, segment_bars, event, {4}, "Aligned")
            right = stats_for(rows, segment_bars, event, {4}, "Opposed")
            labels = ("Stage4 Aligned", "Stage4 Opposed")
        elif kind == "ImmatureAlignment":
            left = stats_for(rows, segment_bars, event, {0, 1}, "Aligned")
            right = stats_for(rows, segment_bars, event, {0, 1}, "Opposed")
            labels = ("Immature Aligned", "Immature Opposed")
        elif kind == "AlignedMaturity":
            left = stats_for(rows, segment_bars, event, {4}, "Aligned")
            right = stats_for(rows, segment_bars, event, {0, 1}, "Aligned")
            labels = ("Aligned Mature", "Aligned Immature")
        else:
            left = stats_for(rows, segment_bars, event, {4}, "Opposed")
            right = stats_for(rows, segment_bars, event, {0, 1}, "Opposed")
            labels = ("Opposed Mature", "Opposed Immature")
        comparisons.append(Comparison(event.name, labels[0], labels[1], left, right))
    return comparisons


def append_comparison_table(lines: list[str], title: str, comparisons: list[Comparison]) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(
        f"{'Event':<24} {'LeftCount':>9} {'LeftMean5':>11} {'LeftCont5':>10} "
        f"{'RightCount':>10} {'RightMean5':>11} {'RightCont5':>11} "
        f"{'DeltaMean5':>11} {'DeltaCont5':>10}"
    )
    for item in comparisons:
        lines.append(
            f"{item.event_name:<24} {item.left['Count']:>9} "
            f"{item.left['MeanDRFwd']:>11.6f} {item.left['ContinuationRate']:>9.2%} "
            f"{item.right['Count']:>10} {item.right['MeanDRFwd']:>11.6f} "
            f"{item.right['ContinuationRate']:>10.2%} {item.delta_mean:>11.6f} "
            f"{item.delta_continuation_rate:>9.2%}"
        )


def ranked(comparisons: list[Comparison]) -> list[Comparison]:
    return [
        item
        for item in comparisons
        if item.left["Count"] >= MIN_RANKED_SAMPLES
        and item.right["Count"] >= MIN_RANKED_SAMPLES
    ]


def append_ranked_table(
    lines: list[str],
    title: str,
    items: list[Comparison],
    sort_key: Callable[[Comparison], float],
    reverse: bool,
) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(
        f"{'Rank':>4} {'Event':<24} {'LeftCount':>9} {'RightCount':>10} "
        f"{'LeftMean5':>11} {'RightMean5':>11} {'DeltaMean5':>11} "
        f"{'DeltaCont5':>10}"
    )
    ordered = sorted(items, key=lambda item: (sort_key(item), item.event_name), reverse=reverse)
    for rank, item in enumerate(ordered[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {item.event_name:<24} {item.left['Count']:>9} "
            f"{item.right['Count']:>10} {item.left['MeanDRFwd']:>11.6f} "
            f"{item.right['MeanDRFwd']:>11.6f} {item.delta_mean:>11.6f} "
            f"{item.delta_continuation_rate:>9.2%}"
        )


def comparison_for(items: list[Comparison], event_name: str) -> Comparison:
    return next(item for item in items if item.event_name == event_name)


def append_research_notes(
    lines: list[str],
    stage4_alignment: list[Comparison],
    immature_alignment: list[Comparison],
    aligned_maturity: list[Comparison],
    opposed_maturity: list[Comparison],
    low_sample: int,
) -> None:
    lines.extend(["\nResearch Notes", "=============="])
    mean_stage4 = statistics.fmean(item.delta_continuation_rate for item in stage4_alignment)
    mean_immature = statistics.fmean(item.delta_continuation_rate for item in immature_alignment)
    lines.append(
        f"- Mean aligned-minus-opposed DeltaContinuationRate5: Stage4={mean_stage4:.2%}; "
        f"Immature={mean_immature:.2%}."
    )

    peak_stage4 = comparison_for(stage4_alignment, "PeakContained")
    peak_immature = comparison_for(immature_alignment, "PeakContained")
    lines.append(
        f"- PeakContained alignment advantage: Stage4={peak_stage4.delta_continuation_rate:.2%}; "
        f"Immature={peak_immature.delta_continuation_rate:.2%}."
    )

    diss_stage4 = comparison_for(stage4_alignment, "DissipationContained")
    diss_aligned_maturity = comparison_for(aligned_maturity, "DissipationContained")
    lines.append(
        f"- DissipationContained Stage4 alignment advantage={diss_stage4.delta_continuation_rate:.2%}; "
        f"Aligned Mature-minus-Immature advantage={diss_aligned_maturity.delta_continuation_rate:.2%}."
    )

    climactic = [
        item.delta_continuation_rate
        for item in stage4_alignment
        if "Climactic" in item.event_name
    ]
    other = [
        item.delta_continuation_rate
        for item in stage4_alignment
        if "Climactic" not in item.event_name
    ]
    lines.append(
        f"- Mean Stage4 alignment advantage: climactic-named events="
        f"{statistics.fmean(climactic) if climactic else 0.0:.2%}; other events="
        f"{statistics.fmean(other) if other else 0.0:.2%}."
    )

    mature_combined = [
        (item.event_name, "Mature Aligned", item.left)
        for item in stage4_alignment
    ] + [
        (item.event_name, "Mature Opposed", item.right)
        for item in stage4_alignment
    ]
    strongest = max(mature_combined, key=lambda item: (item[2]["MeanDRFwd"], item[0], item[1]))
    weakest = min(mature_combined, key=lambda item: (item[2]["MeanDRFwd"], item[0], item[1]))
    lines.append(
        f"- Strongest mature continuation combination by MeanDRFwd5: {strongest[0]} "
        f"{strongest[1]} ({strongest[2]['MeanDRFwd']:.6f}, n={strongest[2]['Count']})."
    )
    lines.append(
        f"- Strongest mature failure combination by MeanDRFwd5: {weakest[0]} "
        f"{weakest[1]} ({weakest[2]['MeanDRFwd']:.6f}, n={weakest[2]['Count']})."
    )
    lines.append(
        f"- Comparison entries skipped from ranked tables because one side has Count < "
        f"{MIN_RANKED_SAMPLES}: {low_sample}."
    )


def build_report(path: Path) -> str:
    rows = load_rows(path)
    events = event_definitions()
    ibsym_breakouts, _ = detect_frames(rows, "IBSYM", ibsym_start)
    lateral_breakouts, _ = detect_frames(rows, "Lateral", lateral_start)
    segment_bars, _ = build_segment_bars(rows, ibsym_breakouts + lateral_breakouts)
    stage4_alignment = make_comparisons(rows, segment_bars, events, "Stage4Alignment")
    immature_alignment = make_comparisons(rows, segment_bars, events, "ImmatureAlignment")
    aligned_maturity = make_comparisons(rows, segment_bars, events, "AlignedMaturity")
    opposed_maturity = make_comparisons(rows, segment_bars, events, "OpposedMaturity")

    ranked_stage4 = ranked(stage4_alignment)
    ranked_immature = ranked(immature_alignment)
    ranked_aligned_maturity = ranked(aligned_maturity)
    ranked_opposed_maturity = ranked(opposed_maturity)
    low_sample = (
        len(stage4_alignment) - len(ranked_stage4)
        + len(immature_alignment) - len(ranked_immature)
        + len(aligned_maturity) - len(ranked_aligned_maturity)
        + len(opposed_maturity) - len(ranked_opposed_maturity)
    )

    lines = [
        "APVA Stage + Alignment Interaction Study v0.1",
        "=============================================",
        f"Input: {path}",
        f"Total rows: {len(rows)}",
        "SequenceStage logic reuses APVA_PostBreakoutOOE_10.py.",
        "Alignment logic reuses APVA_BreakoutAlignment_09.py.",
        "This is not true PVA OOE.",
        "Each source-specific segment-bar observation is counted; overlapping segments may include the same bar.",
        "SequenceStage progression is direction-aware, so Stage and Alignment are descriptive interacting factors.",
        f"Ranked comparisons require both sides Count >= {MIN_RANKED_SAMPLES}.",
    ]
    append_diagnostics(lines, rows, segment_bars, events)
    append_main_table(lines, rows, segment_bars, events)
    append_comparison_table(lines, "Stage4 Aligned vs Stage4 Opposed", stage4_alignment)
    append_comparison_table(lines, "Immature Aligned vs Immature Opposed", immature_alignment)
    append_comparison_table(lines, "Aligned Mature vs Aligned Immature", aligned_maturity)
    append_comparison_table(lines, "Opposed Mature vs Opposed Immature", opposed_maturity)
    append_ranked_table(
        lines,
        "Top 25 Mature Aligned Events by MeanDRFwd5",
        ranked_stage4,
        lambda item: float(item.left["MeanDRFwd"]),
        True,
    )
    append_ranked_table(
        lines,
        "Top 25 Mature Opposed Events by MeanDRFwd5",
        ranked_stage4,
        lambda item: float(item.right["MeanDRFwd"]),
        True,
    )
    append_ranked_table(
        lines,
        "Largest Alignment Advantages inside Mature",
        ranked_stage4,
        lambda item: item.delta_continuation_rate,
        True,
    )
    append_ranked_table(
        lines,
        "Largest Alignment Advantages inside Immature",
        ranked_immature,
        lambda item: item.delta_continuation_rate,
        True,
    )
    append_ranked_table(
        lines,
        "Largest Maturity Improvements for Aligned Events",
        ranked_aligned_maturity,
        lambda item: item.delta_continuation_rate,
        True,
    )
    append_ranked_table(
        lines,
        "Largest Maturity Improvements for Opposed Events",
        ranked_opposed_maturity,
        lambda item: item.delta_continuation_rate,
        True,
    )
    append_ranked_table(
        lines,
        "Largest Maturity Deteriorations for Aligned Events",
        ranked_aligned_maturity,
        lambda item: item.delta_continuation_rate,
        False,
    )
    append_ranked_table(
        lines,
        "Largest Maturity Deteriorations for Opposed Events",
        ranked_opposed_maturity,
        lambda item: item.delta_continuation_rate,
        False,
    )
    append_research_notes(
        lines,
        stage4_alignment,
        immature_alignment,
        aligned_maturity,
        opposed_maturity,
        low_sample,
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_path = args.input_path or latest_csv(DEFAULT_INPUT_FOLDER)
    write_report(build_report(input_path), report_path("StageAlignment", input_path))


if __name__ == "__main__":
    main()
