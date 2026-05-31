#!/usr/bin/env python3
"""Study evolution of the crude APVA post-breakout OOE proxy.

This sequence-evolution research script is not a trade study and does not model
true PVA OOE. It reuses the deliberately crude proxy from study 10.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from APVA_BreakoutContext_08 import (
    DEFAULT_INPUT_FOLDER,
    Breakout,
    EvidenceBar,
    detect_frames,
    ibsym_start,
    latest_csv,
    lateral_start,
    load_rows,
)
from APVA_Evidence_Report_IO import report_path, write_report
from APVA_PostBreakoutOOE_10 import (
    MAX_SEGMENT_BARS,
    advance_stage,
    next_distinct_breakout_indexes,
    phase_token,
)


MIN_RANKED_SAMPLES = 30
LOOKAHEAD_BARS = 5


@dataclass(frozen=True)
class EvidenceCondition:
    name: str
    predicate: Callable[[EvidenceBar], bool]


@dataclass(frozen=True)
class SegmentBar:
    index: int
    bars_since_breakout: int
    stage: int


@dataclass(frozen=True)
class Segment:
    segment_id: int
    breakout_type: str
    breakout_direction: str
    start_bar: int
    end_bar: int
    bars: tuple[SegmentBar, ...]
    max_stage_reached: int
    bars_to_stage: tuple[int | None, int | None, int | None, int | None]

    @property
    def length(self) -> int:
        return len(self.bars)


@dataclass(frozen=True)
class ConditionRate:
    condition_name: str
    condition_count: int
    transitioned_count: int
    transition_rate: float
    baseline_rate: float

    @property
    def delta_rate(self) -> float:
        return self.transition_rate - self.baseline_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study evolution of the crude APVA post-breakout OOE proxy."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        help="Evidence CSV path. Defaults to the latest CSV in Evidence/6E.",
    )
    return parser.parse_args()


def evidence_conditions() -> list[EvidenceCondition]:
    return [
        EvidenceCondition("DissipationAny", lambda row: row.dissipation != "Absent"),
        EvidenceCondition(
            "DissipationContained",
            lambda row: row.dissipation != "Absent" and row.acceptance == "Contained",
        ),
        EvidenceCondition(
            "DissipationAccepted",
            lambda row: row.dissipation != "Absent" and row.acceptance == "Accepted",
        ),
        EvidenceCondition("ParticipationPeak", lambda row: row.participation == "Peak"),
        EvidenceCondition(
            "ParticipationClimactic", lambda row: row.participation == "Climactic"
        ),
        EvidenceCondition(
            "PeakContained",
            lambda row: row.participation == "Peak" and row.acceptance == "Contained",
        ),
        EvidenceCondition(
            "PeakAccepted",
            lambda row: row.participation == "Peak" and row.acceptance == "Accepted",
        ),
        EvidenceCondition(
            "ClimacticContained",
            lambda row: row.participation == "Climactic"
            and row.acceptance == "Contained",
        ),
        EvidenceCondition(
            "ClimacticAccepted",
            lambda row: row.participation == "Climactic" and row.acceptance == "Accepted",
        ),
        EvidenceCondition("CompressionAny", lambda row: row.compression != "Absent"),
        EvidenceCondition("ExpansionAny", lambda row: row.expansion != "Absent"),
    ]


def build_segments(rows: list[EvidenceBar], breakouts: list[Breakout]) -> list[Segment]:
    segments: list[Segment] = []
    next_indexes = next_distinct_breakout_indexes(breakouts)
    ordered = sorted(breakouts, key=lambda item: (item.index, item.source, item.direction))

    for segment_id, breakout in enumerate(ordered, start=1):
        stop = min(len(rows), breakout.index + MAX_SEGMENT_BARS + 1)
        next_breakout = next_indexes[breakout.index]
        if next_breakout is not None:
            stop = min(stop, next_breakout)

        stage = 0
        stage_times: list[int | None] = [None, None, None, None]
        bars: list[SegmentBar] = []
        for index in range(breakout.index + 1, stop):
            bars_since = index - breakout.index
            prior_stage = stage
            stage = advance_stage(stage, phase_token(rows[index]), breakout.direction)
            if stage > prior_stage and stage_times[stage - 1] is None:
                stage_times[stage - 1] = bars_since
            bars.append(SegmentBar(index, bars_since, stage))

        start_bar = rows[bars[0].index].bar_index if bars else rows[breakout.index].bar_index + 1
        end_bar = rows[bars[-1].index].bar_index if bars else rows[breakout.index].bar_index
        segments.append(
            Segment(
                segment_id=segment_id,
                breakout_type=breakout.source,
                breakout_direction=breakout.direction,
                start_bar=start_bar,
                end_bar=end_bar,
                bars=tuple(bars),
                max_stage_reached=stage,
                bars_to_stage=tuple(stage_times),
            )
        )
    return segments


def stage_distribution(segments: list[Segment]) -> list[int]:
    return [sum(segment.max_stage_reached == stage for segment in segments) for stage in range(5)]


def append_diagnostics(lines: list[str], segments: list[Segment]) -> None:
    lengths = [segment.length for segment in segments]
    lines.extend(["\nDiagnostics", "==========="])
    lines.append(f"Total segments:         {len(segments)}")
    lines.append(f"IBSYM segments:         {sum(segment.breakout_type == 'IBSYM' for segment in segments)}")
    lines.append(f"Lateral segments:       {sum(segment.breakout_type == 'Lateral' for segment in segments)}")
    lines.append(f"Up segments:            {sum(segment.breakout_direction == 'Up' for segment in segments)}")
    lines.append(f"Down segments:          {sum(segment.breakout_direction == 'Down' for segment in segments)}")
    lines.append(f"Average segment length: {statistics.fmean(lengths) if lengths else 0.0:.2f}")
    lines.append(f"Median segment length:  {statistics.median(lengths) if lengths else 0.0:.2f}")
    lines.append(f"Max segment length:     {max(lengths) if lengths else 0}")
    lines.append(f"Zero-length segments:   {sum(segment.length == 0 for segment in segments)}")


def append_segment_ledger(lines: list[str], segments: list[Segment]) -> None:
    lines.extend(["\nSegment Ledger", "=============="])
    lines.append(
        f"{'SegmentId':>9} {'Type':<8} {'Direction':<9} {'StartBar':>10} "
        f"{'EndBar':>10} {'Length':>8} {'MaxStage':>9} {'ToStage1':>9} "
        f"{'ToStage2':>9} {'ToStage3':>9} {'ToStage4':>9}"
    )
    for segment in segments:
        stage_times = [
            str(value) if value is not None else "NA" for value in segment.bars_to_stage
        ]
        lines.append(
            f"{segment.segment_id:>9} {segment.breakout_type:<8} "
            f"{segment.breakout_direction:<9} {segment.start_bar:>10} "
            f"{segment.end_bar:>10} {segment.length:>8} "
            f"{segment.max_stage_reached:>9} {stage_times[0]:>9} "
            f"{stage_times[1]:>9} {stage_times[2]:>9} {stage_times[3]:>9}"
        )


def append_distribution(lines: list[str], title: str, segments: list[Segment]) -> None:
    distribution = stage_distribution(segments)
    labels = ("Ended at Stage0", "Ended at Stage1", "Ended at Stage2", "Ended at Stage3", "Reached Stage4")
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(f"{'Outcome':<20} {'Count':>8} {'Percent':>10}")
    denominator = len(segments) if segments else 1
    for label, count in zip(labels, distribution):
        lines.append(f"{label:<20} {count:>8} {count / denominator:>9.2%}")


def transition_timings(segments: list[Segment], stage: int) -> list[int]:
    values: list[int] = []
    for segment in segments:
        target = segment.bars_to_stage[stage]
        if target is None:
            continue
        prior = segment.bars_to_stage[stage - 1] if stage > 0 else 0
        values.append(target - int(prior or 0))
    return values


def append_transition_timing(lines: list[str], segments: list[Segment]) -> None:
    lines.extend(["\nStage Transition Timing", "======================="])
    lines.append(
        f"{'Transition':<18} {'Eligible':>8} {'Transitions':>12} {'Probability':>12} "
        f"{'MeanBars':>10} {'MedianBars':>12}"
    )
    for stage in range(4):
        eligible = sum(
            segment.length > 0 if stage == 0 else segment.max_stage_reached >= stage
            for segment in segments
        )
        timings = transition_timings(segments, stage)
        probability = len(timings) / eligible if eligible else 0.0
        lines.append(
            f"Stage{stage} -> Stage{stage + 1:<2} {eligible:>8} {len(timings):>12} "
            f"{probability:>11.2%} {statistics.fmean(timings) if timings else 0.0:>10.2f} "
            f"{statistics.median(timings) if timings else 0.0:>12.2f}"
        )


def stage_runs(segments: list[Segment], stage: int) -> list[int]:
    runs: list[int] = []
    for segment in segments:
        current = 0
        for bar in segment.bars:
            if bar.stage == stage:
                current += 1
            elif current:
                runs.append(current)
                current = 0
        if current:
            runs.append(current)
    return runs


def append_stage_persistence(lines: list[str], segments: list[Segment]) -> None:
    lines.extend(["\nStage Persistence", "================="])
    lines.append(
        f"{'Stage':<8} {'Occurrences':>12} {'Runs':>8} {'MeanRun':>10} "
        f"{'MedianRun':>12} {'MaxRun':>8}"
    )
    for stage in range(5):
        runs = stage_runs(segments, stage)
        lines.append(
            f"Stage{stage:<2} {sum(runs):>12} {len(runs):>8} "
            f"{statistics.fmean(runs) if runs else 0.0:>10.2f} "
            f"{statistics.median(runs) if runs else 0.0:>12.2f} "
            f"{max(runs) if runs else 0:>8}"
        )


def transition_bars(segment: Segment, target_stage: int) -> list[SegmentBar]:
    for bar in segment.bars:
        if bar.stage == target_stage:
            return [bar]
    return []


def append_evidence_before_transition(
    lines: list[str],
    rows: list[EvidenceBar],
    segments: list[Segment],
    conditions: list[EvidenceCondition],
) -> None:
    lines.extend(["\nEvidence Before Successful Transition", "====================================="])
    lines.append(
        f"{'Transition':<18} {'Condition':<24} {'Transitions':>12} "
        f"{'WithEvidence':>12} {'Percent':>10}"
    )
    for stage in range(4):
        successful = [
            bar
            for segment in segments
            for bar in transition_bars(segment, stage + 1)
        ]
        for condition in conditions:
            count = 0
            for bar in successful:
                start = max(0, bar.index - 2)
                if any(condition.predicate(rows[index]) for index in range(start, bar.index + 1)):
                    count += 1
            lines.append(
                f"Stage{stage} -> Stage{stage + 1:<2} {condition.name:<24} "
                f"{len(successful):>12} {count:>12} "
                f"{count / len(successful) if successful else 0.0:>9.2%}"
            )


def transitions_within_five(segment: Segment, position: int, stage: int) -> bool:
    stop = min(len(segment.bars), position + LOOKAHEAD_BARS + 1)
    return any(segment.bars[offset].stage >= stage + 1 for offset in range(position + 1, stop))


def condition_rates(
    rows: list[EvidenceBar],
    segments: list[Segment],
    conditions: list[EvidenceCondition],
    stage: int,
) -> tuple[float, list[ConditionRate]]:
    candidates = [
        (segment, position, bar)
        for segment in segments
        for position, bar in enumerate(segment.bars)
        if bar.stage == stage
    ]
    baseline_count = sum(
        transitions_within_five(segment, position, stage)
        for segment, position, _ in candidates
    )
    baseline_rate = baseline_count / len(candidates) if candidates else 0.0
    rates: list[ConditionRate] = []
    for condition in conditions:
        matched = [
            (segment, position)
            for segment, position, bar in candidates
            if condition.predicate(rows[bar.index])
        ]
        transitioned = sum(
            transitions_within_five(segment, position, stage)
            for segment, position in matched
        )
        rates.append(
            ConditionRate(
                condition.name,
                len(matched),
                transitioned,
                transitioned / len(matched) if matched else 0.0,
                baseline_rate,
            )
        )
    return baseline_rate, rates


def append_transition_rate_comparisons(
    lines: list[str],
    rows: list[EvidenceBar],
    segments: list[Segment],
    conditions: list[EvidenceCondition],
) -> dict[int, list[ConditionRate]]:
    all_rates: dict[int, list[ConditionRate]] = {}
    lines.extend(["\nTransition-Within-5 Rate Comparison", "==================================="])
    lines.append(
        f"{'Transition':<18} {'Condition':<24} {'CondCount':>10} {'Within5':>9} "
        f"{'Rate':>10} {'Baseline':>10} {'DeltaRate':>10}"
    )
    for stage in range(4):
        _, rates = condition_rates(rows, segments, conditions, stage)
        all_rates[stage] = rates
        for item in rates:
            lines.append(
                f"Stage{stage} -> Stage{stage + 1:<2} {item.condition_name:<24} "
                f"{item.condition_count:>10} {item.transitioned_count:>9} "
                f"{item.transition_rate:>9.2%} {item.baseline_rate:>9.2%} "
                f"{item.delta_rate:>9.2%}"
            )
    return all_rates


def append_ranked_conditions(lines: list[str], all_rates: dict[int, list[ConditionRate]]) -> int:
    low_sample = 0
    lines.extend(["\nRanked Transition Conditions", "============================"])
    for stage in range(4):
        rates = all_rates[stage]
        ranked = [item for item in rates if item.condition_count >= MIN_RANKED_SAMPLES]
        low_sample += len(rates) - len(ranked)
        lines.extend([f"\nStage{stage} -> Stage{stage + 1}", "-" * 18])
        lines.append("Top positive DeltaRate conditions:")
        for item in sorted(ranked, key=lambda value: (-value.delta_rate, value.condition_name)):
            lines.append(
                f"  {item.condition_name:<24} Delta={item.delta_rate:>8.2%} "
                f"Rate={item.transition_rate:>8.2%} Baseline={item.baseline_rate:>8.2%} "
                f"n={item.condition_count}"
            )
        lines.append("Worst negative DeltaRate conditions:")
        for item in sorted(ranked, key=lambda value: (value.delta_rate, value.condition_name)):
            lines.append(
                f"  {item.condition_name:<24} Delta={item.delta_rate:>8.2%} "
                f"Rate={item.transition_rate:>8.2%} Baseline={item.baseline_rate:>8.2%} "
                f"n={item.condition_count}"
            )
    return low_sample


def completion_rate(segments: list[Segment]) -> float:
    return sum(segment.max_stage_reached == 4 for segment in segments) / len(segments) if segments else 0.0


def append_research_notes(
    lines: list[str],
    segments: list[Segment],
    all_rates: dict[int, list[ConditionRate]],
    low_sample: int,
) -> None:
    lines.extend(["\nResearch Notes", "=============="])
    probabilities = []
    for stage in range(4):
        eligible = sum(
            segment.length > 0 if stage == 0 else segment.max_stage_reached >= stage
            for segment in segments
        )
        transitions = sum(segment.max_stage_reached >= stage + 1 for segment in segments)
        probabilities.append((stage, transitions / eligible if eligible else 0.0))
    easiest = max(probabilities, key=lambda item: (item[1], -item[0]))
    hardest = min(probabilities, key=lambda item: (item[1], item[0]))
    lines.append(f"- Easiest transition: Stage{easiest[0]} -> Stage{easiest[0] + 1} ({easiest[1]:.2%}).")
    lines.append(f"- Hardest transition: Stage{hardest[0]} -> Stage{hardest[0] + 1} ({hardest[1]:.2%}).")
    for stage in range(4):
        ranked = [item for item in all_rates[stage] if item.condition_count >= MIN_RANKED_SAMPLES]
        if not ranked:
            lines.append(f"- Stage{stage} -> Stage{stage + 1}: no conditions met n >= {MIN_RANKED_SAMPLES}.")
            continue
        best = max(ranked, key=lambda item: (item.delta_rate, item.condition_name))
        worst = min(ranked, key=lambda item: (item.delta_rate, item.condition_name))
        lines.append(
            f"- Stage{stage} -> Stage{stage + 1}: largest positive DeltaRate is "
            f"{best.condition_name} ({best.delta_rate:.2%}); largest negative DeltaRate is "
            f"{worst.condition_name} ({worst.delta_rate:.2%})."
        )
    ibsym = [segment for segment in segments if segment.breakout_type == "IBSYM"]
    lateral = [segment for segment in segments if segment.breakout_type == "Lateral"]
    up = [segment for segment in segments if segment.breakout_direction == "Up"]
    down = [segment for segment in segments if segment.breakout_direction == "Down"]
    lines.append(
        f"- Stage4 completion rate: IBSYM={completion_rate(ibsym):.2%}; "
        f"Lateral={completion_rate(lateral):.2%}."
    )
    lines.append(
        f"- Stage4 completion rate: Up={completion_rate(up):.2%}; "
        f"Down={completion_rate(down):.2%}."
    )
    lines.append(
        f"- Ranked-condition entries skipped because ConditionCount < "
        f"{MIN_RANKED_SAMPLES}: {low_sample}."
    )


def build_report(path: Path) -> str:
    rows = load_rows(path)
    conditions = evidence_conditions()
    ibsym_breakouts, _ = detect_frames(rows, "IBSYM", ibsym_start)
    lateral_breakouts, _ = detect_frames(rows, "Lateral", lateral_start)
    segments = build_segments(rows, ibsym_breakouts + lateral_breakouts)
    lines = [
        "APVA Sequence Evolution Study v0.1",
        "==================================",
        f"Input: {path}",
        f"Total rows: {len(rows)}",
        "SequenceStage logic reuses the crude proxy from APVA_PostBreakoutOOE_10.py.",
        "This is not true PVA OOE.",
        "Same-bar IB/SYM and lateral breakouts create parallel source-specific segments.",
        f"Transition-rate lookahead: next {LOOKAHEAD_BARS} bars of the same segment.",
        f"Ranked-condition minimum Count: {MIN_RANKED_SAMPLES}",
    ]
    append_diagnostics(lines, segments)
    append_segment_ledger(lines, segments)
    append_distribution(lines, "Stage Completion Distribution", segments)
    append_transition_timing(lines, segments)
    append_stage_persistence(lines, segments)
    append_evidence_before_transition(lines, rows, segments, conditions)
    all_rates = append_transition_rate_comparisons(lines, rows, segments, conditions)
    append_distribution(
        lines,
        "Stage Completion Distribution by BreakoutType: IBSYM",
        [segment for segment in segments if segment.breakout_type == "IBSYM"],
    )
    append_distribution(
        lines,
        "Stage Completion Distribution by BreakoutType: Lateral",
        [segment for segment in segments if segment.breakout_type == "Lateral"],
    )
    append_distribution(
        lines,
        "Stage Completion Distribution by BreakoutDirection: Up",
        [segment for segment in segments if segment.breakout_direction == "Up"],
    )
    append_distribution(
        lines,
        "Stage Completion Distribution by BreakoutDirection: Down",
        [segment for segment in segments if segment.breakout_direction == "Down"],
    )
    low_sample = append_ranked_conditions(lines, all_rates)
    append_research_notes(lines, segments, all_rates, low_sample)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_path = args.input_path or latest_csv(DEFAULT_INPUT_FOLDER)
    write_report(build_report(input_path), report_path("SequenceEvolution", input_path))


if __name__ == "__main__":
    main()
