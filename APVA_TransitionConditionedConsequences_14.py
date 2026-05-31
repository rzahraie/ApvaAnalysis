#!/usr/bin/env python3
"""Measure consequences after crude APVA post-breakout stage transitions.

This research-only study reuses the post-breakout proxy stages from study 10.
It is not a trade study and does not model true PVA OOE.
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
from APVA_SequenceEvolution_11 import Segment, build_segments, evidence_conditions


TRANSITIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
)


@dataclass(frozen=True)
class TransitionObservation:
    from_stage: int
    to_stage: int
    index: int
    breakout_type: str
    breakout_direction: str
    alignment: str | None

    @property
    def name(self) -> str:
        return f"Stage{self.from_stage} -> Stage{self.to_stage}"


@dataclass(frozen=True)
class ContextStats:
    transition_name: str
    context_name: str
    stats: dict[str, float | int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure consequences after crude APVA proxy stage transitions."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        help="Evidence CSV path. Defaults to the latest CSV in Evidence/6E.",
    )
    return parser.parse_args()


def transition_name(from_stage: int, to_stage: int) -> str:
    return f"Stage{from_stage} -> Stage{to_stage}"


def build_transition_observations(
    rows: list[EvidenceBar],
    segments: list[Segment],
) -> list[TransitionObservation]:
    observations: list[TransitionObservation] = []
    for segment in segments:
        for from_stage, to_stage in TRANSITIONS:
            target = next((bar for bar in segment.bars if bar.stage == to_stage), None)
            if target is None:
                continue
            observations.append(
                TransitionObservation(
                    from_stage=from_stage,
                    to_stage=to_stage,
                    index=target.index,
                    breakout_type=segment.breakout_type,
                    breakout_direction=segment.breakout_direction,
                    alignment=classify_alignment(
                        segment.breakout_direction,
                        rows[target.index].polarity,
                    ),
                )
            )
    return observations


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
    observations: list[TransitionObservation],
    horizon: int,
    predicate: Callable[[TransitionObservation], bool],
) -> list[float]:
    values: list[float] = []
    for observation in observations:
        if not predicate(observation):
            continue
        value = direction_relative_return(rows, observation.index, horizon)
        if value is not None:
            values.append(value)
    return values


def stats_for(
    rows: list[EvidenceBar],
    observations: list[TransitionObservation],
    horizon: int,
    predicate: Callable[[TransitionObservation], bool],
) -> dict[str, float | int]:
    return summarize(values_for(rows, observations, horizon, predicate))


def append_diagnostics(
    lines: list[str],
    segments: list[Segment],
    observations: list[TransitionObservation],
) -> None:
    valid = [item for item in observations if item.alignment is not None]
    lines.extend(["\nDiagnostics", "==========="])
    lines.append(f"Total segments:              {len(segments)}")
    lines.append(f"Total transitions:           {len(observations)}")
    for from_stage, to_stage in TRANSITIONS:
        name = transition_name(from_stage, to_stage)
        lines.append(f"{name:<28} {sum(item.name == name for item in observations)}")
    lines.append(f"IBSYM transitions:           {sum(item.breakout_type == 'IBSYM' for item in observations)}")
    lines.append(f"Lateral transitions:         {sum(item.breakout_type == 'Lateral' for item in observations)}")
    lines.append(f"Up transitions:              {sum(item.breakout_direction == 'Up' for item in observations)}")
    lines.append(f"Down transitions:            {sum(item.breakout_direction == 'Down' for item in observations)}")
    lines.append(f"Aligned transitions:         {sum(item.alignment == 'Aligned' for item in valid)}")
    lines.append(f"Opposed transitions:         {sum(item.alignment == 'Opposed' for item in valid)}")
    lines.append(f"Neutral/missing transitions: {sum(item.alignment is None for item in observations)}")


def append_horizon_table(
    lines: list[str],
    title: str,
    rows: list[EvidenceBar],
    observations: list[TransitionObservation],
    predicate: Callable[[TransitionObservation], bool],
) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(
        f"{'Horizon':<8} {'Count':>8} {'MeanDRFwd':>12} {'MedianDRFwd':>14} "
        f"{'ContRate':>10} {'FailRate':>10} {'FlatRate':>10}"
    )
    for horizon in HORIZONS:
        stats = stats_for(rows, observations, horizon, predicate)
        lines.append(
            f"DRFwd{horizon:<2} {stats['Count']:>8} {stats['MeanDRFwd']:>12.6f} "
            f"{stats['MedianDRFwd']:>14.6f} {stats['ContinuationRate']:>9.2%} "
            f"{stats['FailureRate']:>9.2%} {stats['FlatRate']:>9.2%}"
        )


def append_overall(
    lines: list[str],
    rows: list[EvidenceBar],
    observations: list[TransitionObservation],
) -> None:
    lines.extend(["\nOverall Transition Consequences", "==============================="])
    for from_stage, to_stage in TRANSITIONS:
        name = transition_name(from_stage, to_stage)
        append_horizon_table(lines, name, rows, observations, lambda item, name=name: item.name == name)


def append_breakdown(
    lines: list[str],
    title: str,
    rows: list[EvidenceBar],
    observations: list[TransitionObservation],
    groups: list[tuple[str, Callable[[TransitionObservation], bool]]],
) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    for from_stage, to_stage in TRANSITIONS:
        name = transition_name(from_stage, to_stage)
        for label, condition in groups:
            append_horizon_table(
                lines,
                f"{name}, {label}",
                rows,
                observations,
                lambda item, name=name, condition=condition: item.name == name and condition(item),
            )


def append_evidence_breakdown(
    lines: list[str],
    rows: list[EvidenceBar],
    observations: list[TransitionObservation],
) -> None:
    conditions = evidence_conditions()
    lines.extend(["\nTransition Consequences by Evidence Condition", "============================================="])
    for from_stage, to_stage in TRANSITIONS:
        name = transition_name(from_stage, to_stage)
        for condition in conditions:
            append_horizon_table(
                lines,
                f"{name}, {condition.name}=true",
                rows,
                observations,
                lambda item, name=name, condition=condition: (
                    item.name == name and condition.predicate(rows[item.index])
                ),
            )


def append_stage4_focus(
    lines: list[str],
    rows: list[EvidenceBar],
    observations: list[TransitionObservation],
) -> None:
    conditions = {item.name: item for item in evidence_conditions()}
    focus = [
        ("Overall", lambda item: True),
        ("Aligned", lambda item: item.alignment == "Aligned"),
        ("Opposed", lambda item: item.alignment == "Opposed"),
        ("IBSYM", lambda item: item.breakout_type == "IBSYM"),
        ("Lateral", lambda item: item.breakout_type == "Lateral"),
        ("Up", lambda item: item.breakout_direction == "Up"),
        ("Down", lambda item: item.breakout_direction == "Down"),
        (
            "DissipationContained=true",
            lambda item: conditions["DissipationContained"].predicate(rows[item.index]),
        ),
        (
            "PeakContained=true",
            lambda item: conditions["PeakContained"].predicate(rows[item.index]),
        ),
        (
            "ParticipationPeak=true",
            lambda item: conditions["ParticipationPeak"].predicate(rows[item.index]),
        ),
        (
            "CompressionAny=true",
            lambda item: conditions["CompressionAny"].predicate(rows[item.index]),
        ),
        (
            "ExpansionAny=true",
            lambda item: conditions["ExpansionAny"].predicate(rows[item.index]),
        ),
    ]
    lines.extend(["\nStage3 -> Stage4 Focus", "======================"])
    for label, condition in focus:
        append_horizon_table(
            lines,
            label,
            rows,
            observations,
            lambda item, condition=condition: item.name == "Stage3 -> Stage4" and condition(item),
        )


def context_stats(
    rows: list[EvidenceBar],
    observations: list[TransitionObservation],
) -> list[ContextStats]:
    conditions = evidence_conditions()
    contexts: list[tuple[str, Callable[[TransitionObservation], bool]]] = [
        ("Overall", lambda item: True),
        ("Aligned", lambda item: item.alignment == "Aligned"),
        ("Opposed", lambda item: item.alignment == "Opposed"),
        ("IBSYM", lambda item: item.breakout_type == "IBSYM"),
        ("Lateral", lambda item: item.breakout_type == "Lateral"),
        ("Up", lambda item: item.breakout_direction == "Up"),
        ("Down", lambda item: item.breakout_direction == "Down"),
    ]
    contexts.extend(
        (
            f"{condition.name}=true",
            lambda item, condition=condition: condition.predicate(rows[item.index]),
        )
        for condition in conditions
    )
    output: list[ContextStats] = []
    for from_stage, to_stage in TRANSITIONS:
        name = transition_name(from_stage, to_stage)
        for context_name, condition in contexts:
            output.append(
                ContextStats(
                    name,
                    context_name,
                    stats_for(
                        rows,
                        observations,
                        5,
                        lambda item, name=name, condition=condition: item.name == name
                        and condition(item),
                    ),
                )
            )
    return output


def append_ranked_table(
    lines: list[str],
    title: str,
    items: list[ContextStats],
    key: Callable[[ContextStats], float],
    reverse: bool,
) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(
        f"{'Rank':>4} {'Transition':<18} {'Context':<28} {'Count':>8} "
        f"{'MeanDRFwd5':>12} {'ContRate5':>10} {'FailRate5':>10}"
    )
    ordered = sorted(items, key=lambda item: (key(item), item.transition_name, item.context_name), reverse=reverse)
    for rank, item in enumerate(ordered[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {item.transition_name:<18} {item.context_name:<28} "
            f"{item.stats['Count']:>8} {item.stats['MeanDRFwd']:>12.6f} "
            f"{item.stats['ContinuationRate']:>9.2%} {item.stats['FailureRate']:>9.2%}"
        )


def find_context(
    items: list[ContextStats],
    transition: str,
    context: str,
) -> ContextStats:
    return next(item for item in items if item.transition_name == transition and item.context_name == context)


def comparison_note(
    items: list[ContextStats],
    transition: str,
    left: str,
    right: str,
) -> str:
    left_item = find_context(items, transition, left)
    right_item = find_context(items, transition, right)
    delta = float(left_item.stats["ContinuationRate"]) - float(right_item.stats["ContinuationRate"])
    return f"{left} minus {right} DeltaContinuationRate5={delta:.2%}"


def append_research_notes(lines: list[str], items: list[ContextStats], skipped: int) -> None:
    lines.extend(["\nResearch Notes", "=============="])
    overall = [item for item in items if item.context_name == "Overall"]
    strongest = max(overall, key=lambda item: (item.stats["MeanDRFwd"], item.transition_name))
    lines.append(
        f"- Strongest overall transition by MeanDRFwd5: {strongest.transition_name} "
        f"({strongest.stats['MeanDRFwd']:.6f}, n={strongest.stats['Count']})."
    )
    stage4 = find_context(items, "Stage3 -> Stage4", "Overall")
    earlier = [item for item in overall if item.transition_name != "Stage3 -> Stage4"]
    earlier_mean = statistics.fmean(float(item.stats["MeanDRFwd"]) for item in earlier)
    lines.append(
        f"- Stage3 -> Stage4 MeanDRFwd5={stage4.stats['MeanDRFwd']:.6f}; "
        f"mean of earlier transition MeanDRFwd5 values={earlier_mean:.6f}."
    )
    transition = "Stage3 -> Stage4"
    lines.append(f"- Stage3 -> Stage4: {comparison_note(items, transition, 'Aligned', 'Opposed')}.")
    lines.append(f"- Stage3 -> Stage4: {comparison_note(items, transition, 'IBSYM', 'Lateral')}.")
    lines.append(f"- Stage3 -> Stage4: {comparison_note(items, transition, 'Up', 'Down')}.")
    lines.append(
        f"- Stage3 -> Stage4 DissipationContained minus Overall DeltaContinuationRate5="
        f"{float(find_context(items, transition, 'DissipationContained=true').stats['ContinuationRate']) - float(stage4.stats['ContinuationRate']):.2%}."
    )
    lines.append(
        f"- Stage3 -> Stage4 PeakContained minus Overall DeltaContinuationRate5="
        f"{float(find_context(items, transition, 'PeakContained=true').stats['ContinuationRate']) - float(stage4.stats['ContinuationRate']):.2%}."
    )
    lines.append(
        f"- Transition/context combinations skipped from ranked tables because Count < "
        f"{MIN_RANKED_SAMPLES}: {skipped}."
    )


def build_report(path: Path) -> str:
    rows = load_rows(path)
    ibsym_breakouts, _ = detect_frames(rows, "IBSYM", ibsym_start)
    lateral_breakouts, _ = detect_frames(rows, "Lateral", lateral_start)
    segments = build_segments(rows, ibsym_breakouts + lateral_breakouts)
    observations = build_transition_observations(rows, segments)
    contexts = context_stats(rows, observations)
    ranked = [item for item in contexts if item.stats["Count"] >= MIN_RANKED_SAMPLES]
    skipped = len(contexts) - len(ranked)
    lines = [
        "APVA Transition-Conditioned Consequences Study v0.1",
        "==================================================",
        f"Input: {path}",
        f"Total rows: {len(rows)}",
        "Transition bars are the first bars where the reused proxy stage appears.",
        "SequenceStage logic reuses APVA_PostBreakoutOOE_10.py.",
        "Alignment logic reuses APVA_BreakoutAlignment_09.py.",
        "This is not true PVA OOE.",
        "Same-bar IB/SYM and lateral breakouts create parallel source-specific segments.",
        f"Ranked-table minimum Count: {MIN_RANKED_SAMPLES}",
    ]
    append_diagnostics(lines, segments, observations)
    append_overall(lines, rows, observations)
    append_breakdown(
        lines,
        "Transition Consequences by Alignment",
        rows,
        observations,
        [("Aligned", lambda item: item.alignment == "Aligned"), ("Opposed", lambda item: item.alignment == "Opposed")],
    )
    append_breakdown(
        lines,
        "Transition Consequences by Breakout Type",
        rows,
        observations,
        [("IBSYM", lambda item: item.breakout_type == "IBSYM"), ("Lateral", lambda item: item.breakout_type == "Lateral")],
    )
    append_breakdown(
        lines,
        "Transition Consequences by Breakout Direction",
        rows,
        observations,
        [("Up", lambda item: item.breakout_direction == "Up"), ("Down", lambda item: item.breakout_direction == "Down")],
    )
    append_evidence_breakdown(lines, rows, observations)
    append_stage4_focus(lines, rows, observations)
    append_ranked_table(lines, "Best Transition/Context Combinations by MeanDRFwd5", ranked, lambda item: float(item.stats["MeanDRFwd"]), True)
    append_ranked_table(lines, "Worst Transition/Context Combinations by MeanDRFwd5", ranked, lambda item: float(item.stats["MeanDRFwd"]), False)
    append_ranked_table(lines, "Best Transition/Context Combinations by ContinuationRate5", ranked, lambda item: float(item.stats["ContinuationRate"]), True)
    append_ranked_table(lines, "Worst Transition/Context Combinations by ContinuationRate5", ranked, lambda item: float(item.stats["ContinuationRate"]), False)
    append_research_notes(lines, contexts, skipped)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_path = args.input_path or latest_csv(DEFAULT_INPUT_FOLDER)
    write_report(build_report(input_path), report_path("TransitionConditionedConsequences", input_path))


if __name__ == "__main__":
    main()
