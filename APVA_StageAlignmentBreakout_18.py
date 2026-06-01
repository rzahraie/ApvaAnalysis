#!/usr/bin/env python3
"""Decompose APVA Stage x Alignment consequences by breakout type.

This research-only script reuses prior breakout, stage, alignment, event, and
direction-relative-return definitions without adding a new model.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

from APVA_BreakoutContext_08 import (
    HORIZONS,
    MIN_RANKED_SAMPLES,
    TOP_LIMIT,
    EvidenceBar,
    detect_frames,
    direction_relative_return,
    ibsym_start,
    lateral_start,
    load_rows,
)
from APVA_RegimeTransition_16 import CANONICAL_INSTRUMENTS, instrument_name, write_text
from APVA_StageAlignment_13 import (
    MATURITY_STAGES,
    EventDefinition,
    alignment_for,
    event_definitions,
    summarize,
)
from APVA_PostBreakoutOOE_10 import SegmentBar, build_segment_bars


BREAKOUT_TYPES = ("IBSYM", "Lateral")
ALIGNMENTS = ("Aligned", "Opposed")
AGGREGATE_OUTPUT = Path(
    "Evidence/Output/StageAlignmentBreakout/StageAlignmentBreakout_All.txt"
)


@dataclass(frozen=True)
class Comparison:
    event_name: str
    breakout_type: str
    comparison_type: str
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


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    rows: list[EvidenceBar]
    segment_bars: list[SegmentBar]
    comparisons: list[Comparison]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose APVA Stage x Alignment consequences by breakout type."
    )
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def values_for(
    rows: list[EvidenceBar],
    segment_bars: list[SegmentBar],
    event: EventDefinition,
    breakout_type: str,
    stages: set[int],
    alignment: str,
    horizon: int = 5,
) -> list[float]:
    values: list[float] = []
    for segment_bar in segment_bars:
        if segment_bar.breakout_type != breakout_type:
            continue
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
    breakout_type: str,
    stages: set[int],
    alignment: str,
    horizon: int = 5,
) -> dict[str, float | int]:
    return summarize(values_for(rows, segment_bars, event, breakout_type, stages, alignment, horizon))


def make_comparisons(
    rows: list[EvidenceBar],
    segment_bars: list[SegmentBar],
    events: list[EventDefinition],
) -> list[Comparison]:
    comparisons: list[Comparison] = []
    for event in events:
        for breakout_type in BREAKOUT_TYPES:
            mature_aligned = stats_for(rows, segment_bars, event, breakout_type, {4}, "Aligned")
            mature_opposed = stats_for(rows, segment_bars, event, breakout_type, {4}, "Opposed")
            immature_aligned = stats_for(rows, segment_bars, event, breakout_type, {0, 1}, "Aligned")
            immature_opposed = stats_for(rows, segment_bars, event, breakout_type, {0, 1}, "Opposed")
            comparisons.extend(
                [
                    Comparison(
                        event.name,
                        breakout_type,
                        "MatureAlignmentAdvantage",
                        "Mature Aligned",
                        "Mature Opposed",
                        mature_aligned,
                        mature_opposed,
                    ),
                    Comparison(
                        event.name,
                        breakout_type,
                        "ImmatureAlignmentAdvantage",
                        "Immature Aligned",
                        "Immature Opposed",
                        immature_aligned,
                        immature_opposed,
                    ),
                    Comparison(
                        event.name,
                        breakout_type,
                        "AlignedMaturityImprovement",
                        "Mature Aligned",
                        "Immature Aligned",
                        mature_aligned,
                        immature_aligned,
                    ),
                ]
            )
    return comparisons


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    ibsym_breakouts, _ = detect_frames(rows, "IBSYM", ibsym_start)
    lateral_breakouts, _ = detect_frames(rows, "Lateral", lateral_start)
    segment_bars, _ = build_segment_bars(rows, ibsym_breakouts + lateral_breakouts)
    return InstrumentStudy(
        instrument_name(path),
        path,
        rows,
        segment_bars,
        make_comparisons(rows, segment_bars, event_definitions()),
    )


def comparison_map(study: InstrumentStudy) -> dict[tuple[str, str, str], Comparison]:
    return {
        (item.event_name, item.breakout_type, item.comparison_type): item
        for item in study.comparisons
    }


def append_comparison_table(lines: list[str], title: str, items: list[Comparison]) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(
        f"{'Event':<24} {'Type':<8} {'LeftCount':>9} {'LeftCont5':>10} "
        f"{'RightCount':>10} {'RightCont5':>11} {'DeltaCont5':>10} {'DeltaMean5':>11}"
    )
    for item in items:
        lines.append(
            f"{item.event_name:<24} {item.breakout_type:<8} {item.left['Count']:>9} "
            f"{item.left['ContinuationRate']:>9.2%} {item.right['Count']:>10} "
            f"{item.right['ContinuationRate']:>10.2%} "
            f"{item.delta_continuation_rate:>9.2%} {item.delta_mean:>11.6f}"
        )


def ranked(items: list[Comparison]) -> list[Comparison]:
    return [
        item
        for item in items
        if item.left["Count"] >= MIN_RANKED_SAMPLES
        and item.right["Count"] >= MIN_RANKED_SAMPLES
    ]


def append_ranked_comparisons(
    lines: list[str],
    title: str,
    items: list[Comparison],
    reverse: bool,
) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(
        f"{'Rank':>4} {'Event':<24} {'Type':<8} {'LeftCount':>9} "
        f"{'RightCount':>10} {'DeltaCont5':>10} {'DeltaMean5':>11}"
    )
    ordered = sorted(
        ranked(items),
        key=lambda item: (item.delta_continuation_rate, item.event_name, item.breakout_type),
        reverse=reverse,
    )
    for rank, item in enumerate(ordered[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {item.event_name:<24} {item.breakout_type:<8} "
            f"{item.left['Count']:>9} {item.right['Count']:>10} "
            f"{item.delta_continuation_rate:>9.2%} {item.delta_mean:>11.6f}"
        )


def append_ranked_strength(
    lines: list[str],
    title: str,
    items: list[Comparison],
    use_left: bool,
) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(
        f"{'Rank':>4} {'Event':<24} {'Type':<8} {'Count':>8} "
        f"{'MeanDRFwd5':>12} {'ContRate5':>10}"
    )
    eligible = []
    for item in items:
        stats = item.left if use_left else item.right
        if stats["Count"] >= MIN_RANKED_SAMPLES:
            eligible.append((item, stats))
    eligible.sort(key=lambda pair: (-pair[1]["MeanDRFwd"], pair[0].event_name, pair[0].breakout_type))
    for rank, (item, stats) in enumerate(eligible[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {item.event_name:<24} {item.breakout_type:<8} "
            f"{stats['Count']:>8} {stats['MeanDRFwd']:>12.6f} "
            f"{stats['ContinuationRate']:>9.2%}"
        )


def instrument_report(study: InstrumentStudy) -> str:
    rows = study.rows
    segment_bars = study.segment_bars
    valid = [bar for bar in segment_bars if alignment_for(rows, bar) is not None]
    events = event_definitions()
    comparisons_by_type = {
        comparison_type: [
            item for item in study.comparisons if item.comparison_type == comparison_type
        ]
        for comparison_type in (
            "MatureAlignmentAdvantage",
            "ImmatureAlignmentAdvantage",
            "AlignedMaturityImprovement",
        )
    }
    lines = [
        f"APVA Stage + Alignment + Breakout Type Study v0.1 - {study.instrument}",
        "=" * (52 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {len(rows)}",
        f"Total stage-alignment-breakout observations: {len(valid)}",
        "Breakout, SequenceStage, alignment, event, and DRFwd logic are reused from prior studies.",
        "This is a descriptive decomposition of the prior Stage x Alignment result.",
        "\nObservation Counts by BreakoutType",
        "==================================",
    ]
    for breakout_type in BREAKOUT_TYPES:
        lines.append(f"{breakout_type:<12} {sum(bar.breakout_type == breakout_type for bar in valid):>8}")
    lines.extend(["\nObservation Counts by Stage", "==========================="])
    for stage in range(5):
        lines.append(f"Stage{stage:<2} {sum(bar.sequence_stage == stage for bar in valid):>8}")
    lines.extend(["\nObservation Counts by Alignment", "==============================="])
    for alignment in ALIGNMENTS:
        lines.append(f"{alignment:<12} {sum(alignment_for(rows, bar) == alignment for bar in valid):>8}")

    lines.extend(["\nEvent x Stage x Alignment x BreakoutType", "========================================"])
    lines.append(
        f"{'Event':<24} {'Type':<8} {'Stage':>7} {'Alignment':<8} {'Count':>8} "
        f"{'MeanDRFwd5':>12} {'Median':>12} {'ContRate5':>10} {'FailRate5':>10} {'FlatRate5':>10}"
    )
    for event in events:
        for breakout_type in BREAKOUT_TYPES:
            for stage in range(5):
                for alignment in ALIGNMENTS:
                    stats = stats_for(rows, segment_bars, event, breakout_type, {stage}, alignment)
                    lines.append(
                        f"{event.name:<24} {breakout_type:<8} {'Stage' + str(stage):>7} "
                        f"{alignment:<8} {stats['Count']:>8} {stats['MeanDRFwd']:>12.6f} "
                        f"{stats['MedianDRFwd']:>12.6f} {stats['ContinuationRate']:>9.2%} "
                        f"{stats['FailureRate']:>9.2%} {stats['FlatRate']:>9.2%}"
                    )
    append_comparison_table(lines, "Mature Aligned versus Mature Opposed by BreakoutType", comparisons_by_type["MatureAlignmentAdvantage"])
    append_comparison_table(lines, "Immature Aligned versus Immature Opposed by BreakoutType", comparisons_by_type["ImmatureAlignmentAdvantage"])
    append_comparison_table(lines, "Mature Aligned versus Immature Aligned by BreakoutType", comparisons_by_type["AlignedMaturityImprovement"])
    append_ranked_comparisons(lines, "Largest Mature Alignment Advantages by BreakoutType", comparisons_by_type["MatureAlignmentAdvantage"], True)
    append_ranked_comparisons(lines, "Largest Immature Alignment Advantages by BreakoutType", comparisons_by_type["ImmatureAlignmentAdvantage"], True)
    append_ranked_comparisons(lines, "Largest Maturity Improvements for Aligned Events by BreakoutType", comparisons_by_type["AlignedMaturityImprovement"], True)
    append_ranked_comparisons(lines, "Largest Maturity Deteriorations for Aligned Events by BreakoutType", comparisons_by_type["AlignedMaturityImprovement"], False)
    append_ranked_strength(lines, "Strongest Mature Aligned Combinations", comparisons_by_type["MatureAlignmentAdvantage"], True)
    append_ranked_strength(lines, "Strongest Mature Opposed Combinations", comparisons_by_type["MatureAlignmentAdvantage"], False)
    lines.extend(["\nResearch Notes", "=============="])
    mature = ranked(comparisons_by_type["MatureAlignmentAdvantage"])
    if mature:
        best = max(mature, key=lambda item: (item.delta_continuation_rate, item.event_name))
        lines.append(f"- Largest mature alignment advantage: {best.event_name}, {best.breakout_type} ({best.delta_continuation_rate:.2%}).")
    for event_name in ("DissipationContained", "ParticipationPeak", "AllBars"):
        choices = [item for item in mature if item.event_name == event_name]
        if choices:
            best = max(choices, key=lambda item: (item.delta_continuation_rate, item.breakout_type))
            lines.append(f"- Best mature alignment breakout type for {event_name}: {best.breakout_type} ({best.delta_continuation_rate:.2%}).")
    lines.append(
        f"- Mature-alignment comparisons skipped from ranked tables because one side Count < "
        f"{MIN_RANKED_SAMPLES}: {len(comparisons_by_type['MatureAlignmentAdvantage']) - len(mature)}."
    )
    return "\n".join(lines) + "\n"


def aggregate_rows(studies: list[InstrumentStudy]) -> list[dict[str, object]]:
    maps = {study.instrument: comparison_map(study) for study in studies}
    output = []
    for event in event_definitions():
        for breakout_type in BREAKOUT_TYPES:
            for comparison_type in (
                "MatureAlignmentAdvantage",
                "ImmatureAlignmentAdvantage",
                "AlignedMaturityImprovement",
            ):
                key = (event.name, breakout_type, comparison_type)
                values = {instrument: mapping[key] for instrument, mapping in maps.items()}
                valid = [
                    item
                    for item in values.values()
                    if item.left["Count"] >= MIN_RANKED_SAMPLES
                    and item.right["Count"] >= MIN_RANKED_SAMPLES
                ]
                output.append(
                    {
                        "event": event.name,
                        "type": breakout_type,
                        "comparison": comparison_type,
                        "values": values,
                        "valid_count": len(valid),
                        "positive_count": sum(item.delta_continuation_rate > 0.0 for item in valid),
                        "negative_count": sum(item.delta_continuation_rate < 0.0 for item in valid),
                        "mean_delta": statistics.fmean(item.delta_continuation_rate for item in valid) if valid else 0.0,
                    }
                )
    return output


def instrument_columns(studies: list[InstrumentStudy]) -> list[str]:
    available = [study.instrument for study in studies]
    return list(CANONICAL_INSTRUMENTS) + sorted(set(available) - set(CANONICAL_INSTRUMENTS))


def append_aggregate_ranked(lines: list[str], title: str, rows: list[dict[str, object]]) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(
        f"{'Rank':>4} {'Event':<24} {'Type':<8} {'Comparison':<28} "
        f"{'Valid':>5} {'Pos':>4} {'Neg':>4} {'MeanDelta':>10}"
    )
    for rank, row in enumerate(rows[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {str(row['event']):<24} {str(row['type']):<8} "
            f"{str(row['comparison']):<28} {int(row['valid_count']):>5} "
            f"{int(row['positive_count']):>4} {int(row['negative_count']):>4} "
            f"{float(row['mean_delta']):>9.2%}"
        )


def best_family_note(rows: list[dict[str, object]], event_name: str) -> str:
    choices = [
        row
        for row in rows
        if row["event"] == event_name
        and row["comparison"] == "MatureAlignmentAdvantage"
        and row["valid_count"] >= 1
    ]
    if not choices:
        return f"{event_name}: no valid mature-alignment breakout family"
    best = max(choices, key=lambda row: (row["mean_delta"], row["type"]))
    return f"{event_name}: {best['type']} ({float(best['mean_delta']):.2%}, valid instruments={best['valid_count']})"


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    rows = aggregate_rows(studies)
    columns = instrument_columns(studies)
    lines = [
        "APVA Stage + Alignment + Breakout Type Study v0.1 - Cross-Instrument Aggregate",
        "===============================================================================",
        f"Input instruments: {', '.join(study.instrument for study in studies)}",
        f"Replication minimum: both comparison sides Count >= {MIN_RANKED_SAMPLES} in at least two instruments.",
        "\nCross-Instrument Comparison Table",
        "=================================",
    ]
    header = f"{'Event':<24} {'Type':<8} {'Comparison':<28}"
    for instrument in columns:
        header += f" {('L_' + instrument):>7} {('R_' + instrument):>7} {('Delta_' + instrument):>9}"
    header += f" {'Valid':>5} {'Pos':>4} {'Neg':>4} {'MeanDelta':>10}"
    lines.append(header)
    for row in rows:
        text = f"{str(row['event']):<24} {str(row['type']):<8} {str(row['comparison']):<28}"
        values = row["values"]
        for instrument in columns:
            item = values.get(instrument)
            if item is None:
                text += f" {'NA':>7} {'NA':>7} {'NA':>9}"
            else:
                text += (
                    f" {int(item.left['Count']):>7} {int(item.right['Count']):>7} "
                    f"{item.delta_continuation_rate:>8.2%}"
                )
        text += (
            f" {int(row['valid_count']):>5} {int(row['positive_count']):>4} "
            f"{int(row['negative_count']):>4} {float(row['mean_delta']):>9.2%}"
        )
        lines.append(text)
    eligible = [row for row in rows if row["valid_count"] >= 2]
    mature = [row for row in eligible if row["comparison"] == "MatureAlignmentAdvantage"]
    immature = [row for row in eligible if row["comparison"] == "ImmatureAlignmentAdvantage"]
    maturity = [row for row in eligible if row["comparison"] == "AlignedMaturityImprovement"]
    append_aggregate_ranked(lines, "Most Replicated Positive Mature Alignment Advantages", sorted(mature, key=lambda row: (-row["positive_count"], -row["mean_delta"], row["event"], row["type"])))
    append_aggregate_ranked(lines, "Most Replicated Negative Mature Alignment Advantages", sorted(mature, key=lambda row: (-row["negative_count"], row["mean_delta"], row["event"], row["type"])))
    append_aggregate_ranked(lines, "Most Replicated Positive Immature Alignment Advantages", sorted(immature, key=lambda row: (-row["positive_count"], -row["mean_delta"], row["event"], row["type"])))
    append_aggregate_ranked(lines, "Most Replicated Positive Aligned Maturity Improvements", sorted(maturity, key=lambda row: (-row["positive_count"], -row["mean_delta"], row["event"], row["type"])))
    lines.extend(["\nBest Breakout Families", "======================"])
    for event_name in ("DissipationContained", "ParticipationPeak", "AllBars"):
        lines.append(f"- {best_family_note(rows, event_name)}.")
    lines.extend(["\nCross-Instrument Research Notes", "==============================="])
    for breakout_type in BREAKOUT_TYPES:
        selected = [row for row in mature if row["type"] == breakout_type]
        lines.append(
            f"- {breakout_type} mature alignment rows replicating positively in at least two instruments: "
            f"{sum(row['positive_count'] >= 2 for row in selected)} of {len(selected)}."
        )
    dissipation = [row for row in mature if row["event"] == "DissipationContained"]
    peak = [row for row in mature if row["event"] == "ParticipationPeak"]
    lines.append(f"- {best_family_note(rows, 'DissipationContained')}.")
    lines.append(f"- {best_family_note(rows, 'ParticipationPeak')}.")
    surviving = [row for row in mature if row["positive_count"] >= 2]
    lines.append(f"- Mature alignment advantages replicating positively in at least two instruments: {', '.join(str(row['event']) + '/' + str(row['type']) for row in surviving) or 'none'}.")
    if dissipation and peak:
        lines.append("- Breakout decomposition is reported directly; no causal attribution is inferred.")
    lines.append("- Agreement is measured mechanically through per-instrument valid-side counts and DeltaContinuationRate5 signs.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    studies = [study_instrument(path) for path in args.input_paths]
    seen: set[str] = set()
    for study in studies:
        if study.instrument in seen:
            raise ValueError(f"Duplicate instrument input: {study.instrument}")
        seen.add(study.instrument)
        write_text(
            Path("Evidence") / "Output" / study.instrument / f"StageAlignmentBreakout_{study.instrument}.txt",
            instrument_report(study),
        )
    aggregate = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, aggregate)
    print(aggregate, end="")


if __name__ == "__main__":
    main()
