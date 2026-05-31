#!/usr/bin/env python3
"""Study market neighborhoods around crude APVA proxy stage transitions.

This research-only script reuses the transition observations from study 14.
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
    EvidenceBar,
    detect_frames,
    ibsym_start,
    latest_csv,
    lateral_start,
    load_rows,
)
from APVA_Evidence_Report_IO import report_path, write_report
from APVA_SequenceEvolution_11 import build_segments
from APVA_TransitionConditionedConsequences_14 import (
    TRANSITIONS,
    TransitionObservation,
    build_transition_observations,
    transition_name,
)


OFFSETS = tuple(range(-5, 6))
MIN_RANKED_SAMPLES = 30


@dataclass(frozen=True)
class EvidenceCondition:
    name: str
    predicate: Callable[[EvidenceBar], bool]


@dataclass(frozen=True)
class NeighborhoodStats:
    count: int
    mean_return: float
    median_return: float
    mean_volume: float
    falling_rate: float
    rising_rate: float
    peak_rate: float
    climactic_rate: float
    accepted_rate: float
    contained_rate: float
    other_acceptance_rate: float
    dissipation_rate: float
    compression_rate: float
    expansion_rate: float


@dataclass(frozen=True)
class NeighborhoodRow:
    transition: str
    context: str
    offset: int
    stats: NeighborhoodStats


@dataclass(frozen=True)
class Enrichment:
    condition: str
    transition_count: int
    transition_rate: float
    all_bars_rate: float
    ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study neighborhoods around crude APVA proxy stage transitions."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        help="Evidence CSV path. Defaults to the latest CSV in Evidence/6E.",
    )
    return parser.parse_args()


def density_conditions() -> list[EvidenceCondition]:
    return [
        EvidenceCondition("DissipationAny", lambda row: row.dissipation != "Absent"),
        EvidenceCondition(
            "DissipationContained",
            lambda row: row.dissipation != "Absent" and row.acceptance == "Contained",
        ),
        EvidenceCondition("ParticipationPeak", lambda row: row.participation == "Peak"),
        EvidenceCondition(
            "ParticipationClimactic", lambda row: row.participation == "Climactic"
        ),
        EvidenceCondition("CompressionAny", lambda row: row.compression != "Absent"),
        EvidenceCondition("ExpansionAny", lambda row: row.expansion != "Absent"),
    ]


def enrichment_conditions() -> list[EvidenceCondition]:
    return [
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
        EvidenceCondition("CompressionAny", lambda row: row.compression != "Absent"),
        EvidenceCondition("ExpansionAny", lambda row: row.expansion != "Absent"),
    ]


def relative_return(
    rows: list[EvidenceBar],
    observation: TransitionObservation,
    offset: int,
) -> float | None:
    index = observation.index + offset
    if index < 0 or index >= len(rows):
        return None
    transition = rows[observation.index]
    if transition.polarity == "Black":
        return rows[index].close - transition.close
    if transition.polarity == "Red":
        return transition.close - rows[index].close
    return None


def neighborhood_stats(
    rows: list[EvidenceBar],
    observations: list[TransitionObservation],
    offset: int,
) -> NeighborhoodStats:
    pairs = [
        (relative_return(rows, observation, offset), rows[observation.index + offset])
        for observation in observations
        if 0 <= observation.index + offset < len(rows)
        and relative_return(rows, observation, offset) is not None
    ]
    values = [float(value) for value, _ in pairs]
    bars = [bar for _, bar in pairs]
    count = len(bars)
    denominator = count if count else 1
    return NeighborhoodStats(
        count=count,
        mean_return=statistics.fmean(values) if values else 0.0,
        median_return=statistics.median(values) if values else 0.0,
        mean_volume=statistics.fmean(bar.volume for bar in bars) if bars else 0.0,
        falling_rate=sum(bar.participation == "Falling" for bar in bars) / denominator,
        rising_rate=sum(bar.participation == "Rising" for bar in bars) / denominator,
        peak_rate=sum(bar.participation == "Peak" for bar in bars) / denominator,
        climactic_rate=sum(bar.participation == "Climactic" for bar in bars) / denominator,
        accepted_rate=sum(bar.acceptance == "Accepted" for bar in bars) / denominator,
        contained_rate=sum(bar.acceptance == "Contained" for bar in bars) / denominator,
        other_acceptance_rate=sum(
            bar.acceptance not in {"Accepted", "Contained"} for bar in bars
        )
        / denominator,
        dissipation_rate=sum(bar.dissipation != "Absent" for bar in bars) / denominator,
        compression_rate=sum(bar.compression != "Absent" for bar in bars) / denominator,
        expansion_rate=sum(bar.expansion != "Absent" for bar in bars) / denominator,
    )


def append_neighborhood_table(
    lines: list[str],
    title: str,
    rows: list[EvidenceBar],
    observations: list[TransitionObservation],
) -> list[NeighborhoodRow]:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(
        f"{'Offset':>6} {'Count':>7} {'MeanRet':>10} {'MedianRet':>10} {'MeanVol':>10} "
        f"{'Fall':>7} {'Rise':>7} {'Peak':>7} {'Clim':>7} {'Accept':>7} "
        f"{'Contain':>8} {'Other':>7} {'Diss':>7} {'Compr':>7} {'Expand':>7}"
    )
    output: list[NeighborhoodRow] = []
    for offset in OFFSETS:
        stats = neighborhood_stats(rows, observations, offset)
        lines.append(
            f"{offset:>+6} {stats.count:>7} {stats.mean_return:>10.6f} "
            f"{stats.median_return:>10.6f} {stats.mean_volume:>10.2f} "
            f"{stats.falling_rate:>6.2%} {stats.rising_rate:>6.2%} "
            f"{stats.peak_rate:>6.2%} {stats.climactic_rate:>6.2%} "
            f"{stats.accepted_rate:>6.2%} {stats.contained_rate:>7.2%} "
            f"{stats.other_acceptance_rate:>6.2%} {stats.dissipation_rate:>6.2%} "
            f"{stats.compression_rate:>6.2%} {stats.expansion_rate:>6.2%}"
        )
        output.append(NeighborhoodRow(title.split(",")[0], title, offset, stats))
    return output


def append_neighborhood_sections(
    lines: list[str],
    rows: list[EvidenceBar],
    observations: list[TransitionObservation],
) -> list[NeighborhoodRow]:
    ranked_rows: list[NeighborhoodRow] = []
    sections = [
        (
            "Overall Transition Neighborhoods",
            [("Overall", lambda item: True)],
        ),
        (
            "Alignment Neighborhoods",
            [
                ("Aligned", lambda item: item.alignment == "Aligned"),
                ("Opposed", lambda item: item.alignment == "Opposed"),
            ],
        ),
        (
            "Breakout Type Neighborhoods",
            [
                ("IBSYM", lambda item: item.breakout_type == "IBSYM"),
                ("Lateral", lambda item: item.breakout_type == "Lateral"),
            ],
        ),
        (
            "Breakout Direction Neighborhoods",
            [
                ("Up", lambda item: item.breakout_direction == "Up"),
                ("Down", lambda item: item.breakout_direction == "Down"),
            ],
        ),
    ]
    for section_title, contexts in sections:
        lines.extend([f"\n{section_title}", "=" * len(section_title)])
        for from_stage, to_stage in TRANSITIONS:
            name = transition_name(from_stage, to_stage)
            matching = [item for item in observations if item.name == name]
            for label, predicate in contexts:
                selected = [item for item in matching if predicate(item)]
                ranked_rows.extend(
                    append_neighborhood_table(lines, f"{name}, {label}", rows, selected)
                )
    return ranked_rows


def append_density_report(
    lines: list[str],
    rows: list[EvidenceBar],
    observations: list[TransitionObservation],
) -> tuple[list[tuple[str, str, float, int]], list[tuple[str, str, float, int]]]:
    conditions = density_conditions()
    output: list[tuple[str, str, float, int]] = []
    pre_transition: list[tuple[str, str, float, int]] = []
    lines.extend(["\nEvent Density within +/-5 Bars", "============================="])
    lines.append(f"{'Transition':<18} {'Condition':<24} {'Bars':>8} {'Density':>10}")
    for from_stage, to_stage in TRANSITIONS:
        name = transition_name(from_stage, to_stage)
        selected = [item for item in observations if item.name == name]
        neighborhood = [
            rows[item.index + offset]
            for item in selected
            for offset in OFFSETS
            if 0 <= item.index + offset < len(rows)
        ]
        pre_neighborhood = [
            rows[item.index + offset]
            for item in selected
            for offset in range(-5, 0)
            if 0 <= item.index + offset < len(rows)
        ]
        denominator = len(neighborhood) if neighborhood else 1
        pre_denominator = len(pre_neighborhood) if pre_neighborhood else 1
        for condition in conditions:
            count = sum(condition.predicate(row) for row in neighborhood)
            density = count / denominator
            lines.append(f"{name:<18} {condition.name:<24} {len(neighborhood):>8} {density:>9.2%}")
            output.append((name, condition.name, density, len(neighborhood)))
            pre_count = sum(condition.predicate(row) for row in pre_neighborhood)
            pre_transition.append(
                (name, condition.name, pre_count / pre_denominator, len(pre_neighborhood))
            )
    return output, pre_transition


def enrichment_stats(
    rows: list[EvidenceBar],
    observations: list[TransitionObservation],
) -> list[Enrichment]:
    transition_bars = [rows[item.index] for item in observations]
    output: list[Enrichment] = []
    for condition in enrichment_conditions():
        transition_count = sum(condition.predicate(row) for row in transition_bars)
        transition_rate = transition_count / len(transition_bars) if transition_bars else 0.0
        all_bars_rate = sum(condition.predicate(row) for row in rows) / len(rows) if rows else 0.0
        ratio = transition_rate / all_bars_rate if all_bars_rate else 0.0
        output.append(
            Enrichment(condition.name, transition_count, transition_rate, all_bars_rate, ratio)
        )
    return output


def append_enrichment(lines: list[str], items: list[Enrichment]) -> None:
    lines.extend(["\nTransition-Bar Evidence Enrichment", "=================================="])
    lines.append(
        f"{'Condition':<24} {'TransitionCount':>15} {'TransitionRate':>15} "
        f"{'AllBarsRate':>12} {'Enrichment':>11}"
    )
    for item in items:
        lines.append(
            f"{item.condition:<24} {item.transition_count:>15} {item.transition_rate:>14.2%} "
            f"{item.all_bars_rate:>11.2%} {item.ratio:>10.2f}x"
        )


def append_ranked_neighborhoods(
    lines: list[str],
    title: str,
    items: list[NeighborhoodRow],
    reverse: bool,
) -> None:
    eligible = [
        item for item in items if item.offset in {1, 2, 3, 4, 5}
        and item.stats.count >= MIN_RANKED_SAMPLES
    ]
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(
        f"{'Rank':>4} {'Context':<38} {'Offset':>7} {'Count':>8} "
        f"{'MeanReturn':>12} {'Diss':>8} {'Compr':>8} {'Expand':>8}"
    )
    ordered = sorted(
        eligible,
        key=lambda item: (item.stats.mean_return, item.context, item.offset),
        reverse=reverse,
    )
    for rank, item in enumerate(ordered[:25], start=1):
        lines.append(
            f"{rank:>4} {item.context:<38} {item.offset:>+7} {item.stats.count:>8} "
            f"{item.stats.mean_return:>12.6f} {item.stats.dissipation_rate:>7.2%} "
            f"{item.stats.compression_rate:>7.2%} {item.stats.expansion_rate:>7.2%}"
        )


def append_rankings(
    lines: list[str],
    enrichments: list[Enrichment],
    neighborhood_rows: list[NeighborhoodRow],
    pre_transition_densities: list[tuple[str, str, float, int]],
) -> int:
    eligible_enrichment = [
        item for item in enrichments if item.transition_count >= MIN_RANKED_SAMPLES
    ]
    lines.extend(["\nMost Enriched Evidence on Transition Bars", "========================================="])
    for rank, item in enumerate(sorted(eligible_enrichment, key=lambda item: (-item.ratio, item.condition)), start=1):
        lines.append(f"{rank:>4} {item.condition:<24} {item.ratio:>8.2f}x n={item.transition_count}")
    lines.extend(["\nLeast Enriched Evidence on Transition Bars", "=========================================="])
    for rank, item in enumerate(sorted(eligible_enrichment, key=lambda item: (item.ratio, item.condition)), start=1):
        lines.append(f"{rank:>4} {item.condition:<24} {item.ratio:>8.2f}x n={item.transition_count}")
    append_ranked_neighborhoods(lines, "Strongest Positive Post-Transition Continuation", neighborhood_rows, True)
    append_ranked_neighborhoods(lines, "Strongest Negative Post-Transition Continuation", neighborhood_rows, False)
    lines.extend(["\nStrongest Pre-Transition Buildup Signatures", "=========================================="])
    eligible_density = [
        item for item in pre_transition_densities if item[3] >= MIN_RANKED_SAMPLES
    ]
    for rank, item in enumerate(sorted(eligible_density, key=lambda item: (-item[2], item[0], item[1]))[:25], start=1):
        lines.append(f"{rank:>4} {item[0]:<18} {item[1]:<24} {item[2]:>9.2%} bars={item[3]}")
    return len(enrichments) - len(eligible_enrichment)


def find_row(items: list[NeighborhoodRow], transition: str, context: str, offset: int) -> NeighborhoodRow:
    return next(item for item in items if item.context == f"{transition}, {context}" and item.offset == offset)


def append_research_notes(
    lines: list[str],
    rows: list[NeighborhoodRow],
    enrichments: list[Enrichment],
    low_sample: int,
) -> None:
    lines.extend(["\nResearch Notes", "=============="])
    overall_post = [
        item for item in rows if item.context.endswith(", Overall") and item.offset in {1, 2, 3, 4, 5}
    ]
    strongest = max(overall_post, key=lambda item: (item.stats.mean_return, item.context, item.offset))
    weakest = min(overall_post, key=lambda item: (item.stats.mean_return, item.context, item.offset))
    lines.append(
        f"- Strongest overall directional follow-through: {strongest.context}, "
        f"offset={strongest.offset:+d} ({strongest.stats.mean_return:.6f})."
    )
    lines.append(
        f"- Strongest overall failure: {weakest.context}, offset={weakest.offset:+d} "
        f"({weakest.stats.mean_return:.6f})."
    )
    stage4 = find_row(rows, "Stage3 -> Stage4", "Overall", 5)
    earlier = [
        find_row(rows, transition_name(start, end), "Overall", 5)
        for start, end in TRANSITIONS[:-1]
    ]
    lines.append(
        f"- Stage3 -> Stage4 offset +5 MeanReturn={stage4.stats.mean_return:.6f}; "
        f"earlier-transition mean={statistics.fmean(item.stats.mean_return for item in earlier):.6f}."
    )
    enrichment_map = {item.condition: item for item in enrichments}
    for condition in ("DissipationContained", "CompressionAny", "ExpansionAny"):
        item = enrichment_map[condition]
        lines.append(f"- Transition-bar {condition} enrichment: {item.ratio:.2f}x.")
    aligned = find_row(rows, "Stage3 -> Stage4", "Aligned", 5)
    opposed = find_row(rows, "Stage3 -> Stage4", "Opposed", 5)
    lines.append(
        f"- Stage3 -> Stage4 offset +5 Aligned-minus-Opposed MeanReturn="
        f"{aligned.stats.mean_return - opposed.stats.mean_return:.6f}."
    )
    ibsym = find_row(rows, "Stage3 -> Stage4", "IBSYM", 5)
    lateral = find_row(rows, "Stage3 -> Stage4", "Lateral", 5)
    lines.append(
        f"- Stage3 -> Stage4 offset +5 IBSYM-minus-Lateral MeanReturn="
        f"{ibsym.stats.mean_return - lateral.stats.mean_return:.6f}."
    )
    lines.append(
        f"- Enrichment entries skipped from ranked tables because transition Count < "
        f"{MIN_RANKED_SAMPLES}: {low_sample}."
    )


def build_report(path: Path) -> str:
    rows = load_rows(path)
    ibsym_breakouts, _ = detect_frames(rows, "IBSYM", ibsym_start)
    lateral_breakouts, _ = detect_frames(rows, "Lateral", lateral_start)
    segments = build_segments(rows, ibsym_breakouts + lateral_breakouts)
    observations = build_transition_observations(rows, segments)
    lines = [
        "APVA Transition Neighborhood Study v0.1",
        "=======================================",
        f"Input: {path}",
        f"Total rows: {len(rows)}",
        f"Total transition observations: {len(observations)}",
        "Transition bars reuse APVA_TransitionConditionedConsequences_14.py.",
        "SequenceStage logic reuses APVA_PostBreakoutOOE_10.py.",
        "Alignment logic reuses APVA_BreakoutAlignment_09.py.",
        "Neighborhood offsets use surrounding market rows and may cross segment boundaries.",
        "This is not true PVA OOE.",
        f"Ranked-table minimum Count: {MIN_RANKED_SAMPLES}",
    ]
    neighborhood_rows = append_neighborhood_sections(lines, rows, observations)
    _, pre_transition_densities = append_density_report(lines, rows, observations)
    enrichments = enrichment_stats(rows, observations)
    append_enrichment(lines, enrichments)
    low_sample = append_rankings(
        lines, enrichments, neighborhood_rows, pre_transition_densities
    )
    append_research_notes(lines, neighborhood_rows, enrichments, low_sample)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_path = args.input_path or latest_csv(DEFAULT_INPUT_FOLDER)
    write_report(build_report(input_path), report_path("TransitionNeighborhood", input_path))


if __name__ == "__main__":
    main()
