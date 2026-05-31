#!/usr/bin/env python3
"""Measure APVA evidence consequences conditioned on crude proxy SequenceStage.

This research-only script reuses the post-breakout stage model from study 10.
It is not a trade study and does not model true PVA OOE.
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
class EventStageStats:
    event_name: str
    stage: int
    stats: dict[str, float | int]


@dataclass(frozen=True)
class EventComparison:
    event_name: str
    baseline: dict[str, float | int]
    target: dict[str, float | int]

    @property
    def delta_mean(self) -> float:
        return float(self.target["MeanDRFwd"]) - float(self.baseline["MeanDRFwd"])

    @property
    def delta_continuation_rate(self) -> float:
        return float(self.target["ContinuationRate"]) - float(
            self.baseline["ContinuationRate"]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure APVA evidence consequences by crude proxy SequenceStage."
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
    stages: set[int],
) -> list[float]:
    values: list[float] = []
    for segment_bar in segment_bars:
        if segment_bar.sequence_stage not in stages or not event.predicate(rows[segment_bar.index]):
            continue
        value = direction_relative_return(rows, segment_bar.index, horizon)
        if value is not None:
            values.append(value)
    return values


def stage_stats(
    rows: list[EvidenceBar],
    segment_bars: list[SegmentBar],
    event: EventDefinition,
    stage: int,
    horizon: int = 5,
) -> dict[str, float | int]:
    return summarize(values_for(rows, segment_bars, event, horizon, {stage}))


def group_stats(
    rows: list[EvidenceBar],
    segment_bars: list[SegmentBar],
    event: EventDefinition,
    stages: set[int],
    horizon: int = 5,
) -> dict[str, float | int]:
    return summarize(values_for(rows, segment_bars, event, horizon, stages))


def append_diagnostics(
    lines: list[str],
    rows: list[EvidenceBar],
    segment_bars: list[SegmentBar],
    events: list[EventDefinition],
) -> None:
    lines.extend(["\nDiagnostics", "==========="])
    lines.append(f"Total post-breakout stage bars: {len(segment_bars)}")
    for stage in range(5):
        lines.append(
            f"Total Stage{stage} bars:              "
            f"{sum(bar.sequence_stage == stage for bar in segment_bars)}"
        )
    lines.extend(["\nEvent Counts per Stage", "======================"])
    lines.append(f"{'Event':<24} " + " ".join(f"{'Stage' + str(stage):>9}" for stage in range(5)))
    for event in events:
        counts = [
            sum(
                bar.sequence_stage == stage and event.predicate(rows[bar.index])
                for bar in segment_bars
            )
            for stage in range(5)
        ]
        lines.append(f"{event.name:<24} " + " ".join(f"{count:>9}" for count in counts))


def append_main_stage_report(
    lines: list[str],
    rows: list[EvidenceBar],
    segment_bars: list[SegmentBar],
    events: list[EventDefinition],
) -> None:
    lines.extend(["\nStage-Conditioned Direction-Relative Consequences", "==============================================="])
    for event in events:
        lines.extend([f"\nEvent: {event.name}", "-" * (7 + len(event.name))])
        for stage in range(5):
            lines.append(f"\nStage{stage}")
            lines.append(
                f"{'Horizon':<8} {'Count':>8} {'MeanDRFwd':>12} {'MedianDRFwd':>14} "
                f"{'ContRate':>10} {'FailRate':>10} {'FlatRate':>10}"
            )
            for horizon in HORIZONS:
                stats = stage_stats(rows, segment_bars, event, stage, horizon)
                lines.append(
                    f"DRFwd{horizon:<2} {stats['Count']:>8} {stats['MeanDRFwd']:>12.6f} "
                    f"{stats['MedianDRFwd']:>14.6f} {stats['ContinuationRate']:>9.2%} "
                    f"{stats['FailureRate']:>9.2%} {stats['FlatRate']:>9.2%}"
                )


def append_compact_stage_report(
    lines: list[str],
    rows: list[EvidenceBar],
    segment_bars: list[SegmentBar],
    events: list[EventDefinition],
) -> None:
    lines.extend(["\nCompact DRFwd5 Stage Comparison", "==============================="])
    lines.append(
        f"{'Event':<24} "
        + " ".join(
            f"{'S' + str(stage) + 'Count':>8} {'S' + str(stage) + 'Mean':>10} "
            f"{'S' + str(stage) + 'Cont':>9}"
            for stage in range(5)
        )
    )
    for event in events:
        parts = []
        for stage in range(5):
            stats = stage_stats(rows, segment_bars, event, stage)
            parts.append(
                f"{stats['Count']:>8} {stats['MeanDRFwd']:>10.6f} "
                f"{stats['ContinuationRate']:>8.2%}"
            )
        lines.append(f"{event.name:<24} " + " ".join(parts))


def append_stage4_comparisons(
    lines: list[str],
    rows: list[EvidenceBar],
    segment_bars: list[SegmentBar],
    events: list[EventDefinition],
) -> list[EventComparison]:
    stage0_comparisons: list[EventComparison] = []
    lines.extend(["\nStage4 Comparisons", "=================="])
    lines.append(
        f"{'Event':<24} {'S0Count':>8} {'S4Count':>8} {'S4-S0Mean':>12} "
        f"{'S4-S0Cont':>11} {'S1-3Count':>10} {'S4-S1-3Mean':>13} {'S4-S1-3Cont':>13}"
    )
    for event in events:
        stage0 = group_stats(rows, segment_bars, event, {0})
        middle = group_stats(rows, segment_bars, event, {1, 2, 3})
        stage4 = group_stats(rows, segment_bars, event, {4})
        stage0_comparison = EventComparison(event.name, stage0, stage4)
        middle_comparison = EventComparison(event.name, middle, stage4)
        stage0_comparisons.append(stage0_comparison)
        lines.append(
            f"{event.name:<24} {stage0['Count']:>8} {stage4['Count']:>8} "
            f"{stage0_comparison.delta_mean:>12.6f} "
            f"{stage0_comparison.delta_continuation_rate:>10.2%} "
            f"{middle['Count']:>10} {middle_comparison.delta_mean:>13.6f} "
            f"{middle_comparison.delta_continuation_rate:>12.2%}"
        )
    return stage0_comparisons


def append_maturity_report(
    lines: list[str],
    rows: list[EvidenceBar],
    segment_bars: list[SegmentBar],
    events: list[EventDefinition],
) -> dict[str, list[EventComparison]]:
    comparisons = {"Immature": [], "Developing": []}
    lines.extend(["\nMaturity Bucket DRFwd5 Report", "============================"])
    lines.append(
        f"{'Event':<24} {'Maturity':<11} {'Count':>8} {'MeanDRFwd5':>12} "
        f"{'Median':>12} {'ContRate5':>10} {'FailRate5':>10} {'FlatRate5':>10}"
    )
    for event in events:
        maturity_stats = {
            name: group_stats(rows, segment_bars, event, stages)
            for name, stages in MATURITY_STAGES.items()
        }
        for name in ("Immature", "Developing", "Mature"):
            stats = maturity_stats[name]
            lines.append(
                f"{event.name:<24} {name:<11} {stats['Count']:>8} "
                f"{stats['MeanDRFwd']:>12.6f} {stats['MedianDRFwd']:>12.6f} "
                f"{stats['ContinuationRate']:>9.2%} {stats['FailureRate']:>9.2%} "
                f"{stats['FlatRate']:>9.2%}"
            )
        comparisons["Immature"].append(
            EventComparison(event.name, maturity_stats["Immature"], maturity_stats["Mature"])
        )
        comparisons["Developing"].append(
            EventComparison(event.name, maturity_stats["Developing"], maturity_stats["Mature"])
        )
    return comparisons


def append_mature_comparisons(
    lines: list[str],
    maturity_comparisons: dict[str, list[EventComparison]],
) -> None:
    lines.extend(["\nMature Comparisons", "=================="])
    lines.append(
        f"{'Event':<24} {'Mature-ImmMean':>14} {'Mature-ImmCont':>15} "
        f"{'Mature-DevMean':>14} {'Mature-DevCont':>15}"
    )
    developing = {item.event_name: item for item in maturity_comparisons["Developing"]}
    for immature in maturity_comparisons["Immature"]:
        dev = developing[immature.event_name]
        lines.append(
            f"{immature.event_name:<24} {immature.delta_mean:>14.6f} "
            f"{immature.delta_continuation_rate:>14.2%} {dev.delta_mean:>14.6f} "
            f"{dev.delta_continuation_rate:>14.2%}"
        )


def append_ranked_event_stage(
    lines: list[str],
    title: str,
    ranked: list[EventStageStats],
) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(
        f"{'Rank':>4} {'Event':<24} {'Stage':>7} {'Count':>8} {'MeanDRFwd5':>12} "
        f"{'ContRate5':>10} {'FailRate5':>10}"
    )
    for rank, item in enumerate(ranked[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {item.event_name:<24} {'Stage' + str(item.stage):>7} "
            f"{item.stats['Count']:>8} {item.stats['MeanDRFwd']:>12.6f} "
            f"{item.stats['ContinuationRate']:>9.2%} {item.stats['FailureRate']:>9.2%}"
        )


def append_ranked_comparison(
    lines: list[str],
    title: str,
    ranked: list[EventComparison],
) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(
        f"{'Rank':>4} {'Event':<24} {'BaseCount':>10} {'TargetCount':>12} "
        f"{'DeltaMean5':>12} {'DeltaCont5':>11}"
    )
    for rank, item in enumerate(ranked[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {item.event_name:<24} {item.baseline['Count']:>10} "
            f"{item.target['Count']:>12} {item.delta_mean:>12.6f} "
            f"{item.delta_continuation_rate:>10.2%}"
        )


def append_rankings(
    lines: list[str],
    rows: list[EvidenceBar],
    segment_bars: list[SegmentBar],
    events: list[EventDefinition],
    stage0_comparisons: list[EventComparison],
    maturity_comparisons: dict[str, list[EventComparison]],
) -> int:
    combinations = [
        EventStageStats(event.name, stage, stage_stats(rows, segment_bars, event, stage))
        for event in events
        for stage in range(5)
    ]
    ranked_stages = [item for item in combinations if item.stats["Count"] >= MIN_RANKED_SAMPLES]
    immature = [
        item
        for item in maturity_comparisons["Immature"]
        if item.baseline["Count"] >= MIN_RANKED_SAMPLES
        and item.target["Count"] >= MIN_RANKED_SAMPLES
    ]
    stage0 = [
        item
        for item in stage0_comparisons
        if item.baseline["Count"] >= MIN_RANKED_SAMPLES
        and item.target["Count"] >= MIN_RANKED_SAMPLES
    ]
    append_ranked_event_stage(
        lines,
        "Top 25 Event/Stage Combinations by MeanDRFwd5",
        sorted(ranked_stages, key=lambda item: (-item.stats["MeanDRFwd"], item.event_name, item.stage)),
    )
    append_ranked_event_stage(
        lines,
        "Bottom 25 Event/Stage Combinations by MeanDRFwd5",
        sorted(ranked_stages, key=lambda item: (item.stats["MeanDRFwd"], item.event_name, item.stage)),
    )
    append_ranked_comparison(
        lines,
        "Largest Mature Improvements over Immature by ContinuationRate5",
        sorted(immature, key=lambda item: (-item.delta_continuation_rate, item.event_name)),
    )
    append_ranked_comparison(
        lines,
        "Largest Mature Deteriorations versus Immature by ContinuationRate5",
        sorted(immature, key=lambda item: (item.delta_continuation_rate, item.event_name)),
    )
    append_ranked_comparison(
        lines,
        "Largest Stage4 Improvements over Stage0 by ContinuationRate5",
        sorted(stage0, key=lambda item: (-item.delta_continuation_rate, item.event_name)),
    )
    append_ranked_comparison(
        lines,
        "Largest Stage4 Deteriorations versus Stage0 by ContinuationRate5",
        sorted(stage0, key=lambda item: (item.delta_continuation_rate, item.event_name)),
    )
    return (len(combinations) - len(ranked_stages)) + (
        len(maturity_comparisons["Immature"]) - len(immature)
    ) + (len(stage0_comparisons) - len(stage0))


def comparison_for(
    comparisons: list[EventComparison],
    event_name: str,
) -> EventComparison:
    return next(item for item in comparisons if item.event_name == event_name)


def append_research_notes(
    lines: list[str],
    stage0_comparisons: list[EventComparison],
    maturity_comparisons: dict[str, list[EventComparison]],
    low_sample: int,
) -> None:
    lines.extend(["\nResearch Notes", "=============="])
    improved = [item for item in stage0_comparisons if item.delta_continuation_rate > 0.0]
    deteriorated = [item for item in stage0_comparisons if item.delta_continuation_rate < 0.0]
    lines.append(
        f"- Events with higher Stage4 than Stage0 continuation rate: "
        f"{', '.join(item.event_name for item in improved) or 'none'}."
    )
    lines.append(
        f"- Events with lower Stage4 than Stage0 continuation rate: "
        f"{', '.join(item.event_name for item in deteriorated) or 'none'}."
    )
    immature = maturity_comparisons["Immature"]
    for event_name in ("DissipationContained", "PeakContained"):
        item = comparison_for(immature, event_name)
        lines.append(
            f"- Mature-minus-Immature {event_name} DeltaContinuationRate5: "
            f"{item.delta_continuation_rate:.2%}; DeltaMeanDRFwd5: {item.delta_mean:.6f}."
        )
    climactic = [
        item.delta_continuation_rate
        for item in stage0_comparisons
        if "Climactic" in item.event_name
    ]
    non_climactic = [
        item.delta_continuation_rate
        for item in stage0_comparisons
        if "Climactic" not in item.event_name
    ]
    lines.append(
        f"- Mean Stage4-minus-Stage0 DeltaContinuationRate5: climactic-named events="
        f"{statistics.fmean(climactic) if climactic else 0.0:.2%}; other events="
        f"{statistics.fmean(non_climactic) if non_climactic else 0.0:.2%}."
    )
    material = [
        item for item in stage0_comparisons if abs(item.delta_continuation_rate) >= 0.05
    ]
    lines.append(
        f"- Events with absolute Stage4-minus-Stage0 continuation-rate difference >= 5%: "
        f"{len(material)} of {len(stage0_comparisons)}."
    )
    lines.append(
        f"- Ranked entries skipped because required Count < {MIN_RANKED_SAMPLES}: "
        f"{low_sample}."
    )


def build_report(path: Path) -> str:
    rows = load_rows(path)
    events = event_definitions()
    ibsym_breakouts, _ = detect_frames(rows, "IBSYM", ibsym_start)
    lateral_breakouts, _ = detect_frames(rows, "Lateral", lateral_start)
    segment_bars, _ = build_segment_bars(rows, ibsym_breakouts + lateral_breakouts)
    lines = [
        "APVA Stage-Conditioned Consequences Study v0.1",
        "==============================================",
        f"Input: {path}",
        f"Total rows: {len(rows)}",
        "SequenceStage logic reuses APVA_PostBreakoutOOE_10.py.",
        "This is not true PVA OOE.",
        "Each source-specific segment-bar observation is counted; overlapping segments may include the same bar.",
        "DRFwd > 0: continuation of terminal event-bar polarity",
        "DRFwd < 0: failure of terminal event-bar polarity",
        f"Ranked-table minimum Count: {MIN_RANKED_SAMPLES}",
    ]
    append_diagnostics(lines, rows, segment_bars, events)
    append_main_stage_report(lines, rows, segment_bars, events)
    append_compact_stage_report(lines, rows, segment_bars, events)
    stage0_comparisons = append_stage4_comparisons(lines, rows, segment_bars, events)
    maturity_comparisons = append_maturity_report(lines, rows, segment_bars, events)
    append_mature_comparisons(lines, maturity_comparisons)
    low_sample = append_rankings(
        lines,
        rows,
        segment_bars,
        events,
        stage0_comparisons,
        maturity_comparisons,
    )
    append_research_notes(lines, stage0_comparisons, maturity_comparisons, low_sample)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_path = args.input_path or latest_csv(DEFAULT_INPUT_FOLDER)
    write_report(build_report(input_path), report_path("StageConditionedConsequences", input_path))


if __name__ == "__main__":
    main()
