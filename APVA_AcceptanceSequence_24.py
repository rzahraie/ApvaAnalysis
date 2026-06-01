"""APVA Acceptance Sequence Study v0.1.

Research-only study of what follows existing AcceptanceState observations.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

from APVA_AcceptanceAnatomy_23 import (
    Case,
    cases_for_indexes,
    mature_aligned_lateral_indexes,
    summarize,
)
from APVA_BreakoutContext_08 import EvidenceBar, direction_relative_return, load_rows
from APVA_LateralAnatomy_19 import effect_size, on
from APVA_RegimeTransition_16 import CANONICAL_INSTRUMENTS, instrument_name, write_text


AGGREGATE_OUTPUT = Path("Evidence/Output/AcceptanceSequence/AcceptanceSequence_All.txt")
OFFSETS = (1, 2, 3, 5)
NEIGHBORHOODS = (1, 2, 3, 5, 10)
SEQUENCE_LENGTHS = (2, 3, 4, 5)
SEQUENCE_MIN_COUNT = 20
TOP_LIMIT = 25
TOKENS = ("A", "C", "D", "E", "P", "K", "N")
TOKEN_NAMES = {
    "A": "Accepted",
    "C": "Compression",
    "D": "Dissipation",
    "E": "Expansion",
    "P": "Peak",
    "K": "Climactic",
    "N": "None",
}


@dataclass(frozen=True)
class TransitionStats:
    count: int
    probability: float


@dataclass(frozen=True)
class SequenceStats:
    sequence: str
    count: int
    mean_drfwd5: float
    continuation_rate: float
    failure_rate: float


@dataclass(frozen=True)
class ComparisonStats:
    name: str
    accepted_count: int
    other_count: int
    accepted_mean: float
    other_mean: float
    delta: float
    effect: float


@dataclass(frozen=True)
class OutcomeStats:
    count: int
    mean: float
    median: float
    continuation_rate: float
    failure_rate: float


@dataclass(frozen=True)
class PopulationBStudy:
    success_transitions: dict[tuple[int, str], TransitionStats]
    failure_transitions: dict[tuple[int, str], TransitionStats]
    transition_deltas: dict[tuple[int, str], float]
    transition_effects: dict[tuple[int, str], float]
    success_persistence: dict[str, OutcomeStats]
    failure_persistence: dict[str, OutcomeStats]
    success_sequences: dict[str, SequenceStats]
    failure_sequences: dict[str, SequenceStats]
    sequence_effects: dict[str, float]


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    rows: list[EvidenceBar]
    accepted_indexes: list[int]
    transitions: dict[tuple[int, str], TransitionStats]
    persistence: dict[str, OutcomeStats]
    sequences: dict[str, SequenceStats]
    neighborhoods: list[ComparisonStats]
    population_b: PopulationBStudy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study AcceptanceState as an evidence-layer transition process."
    )
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def outcome(values: list[float]) -> OutcomeStats:
    stats = summarize(values)
    return OutcomeStats(stats.count, stats.mean, stats.median, stats.continuation_rate, stats.failure_rate)


def state_token(row: EvidenceBar) -> str:
    # Stable precedence for overlapping evidence fields. This is encoding only.
    if row.acceptance == "Accepted":
        return "A"
    if on(row.compression):
        return "C"
    if on(row.dissipation):
        return "D"
    if on(row.expansion):
        return "E"
    if row.participation == "Peak":
        return "P"
    if row.participation == "Climactic":
        return "K"
    return "N"


def transition_match(row: EvidenceBar, token: str) -> bool:
    if token == "A":
        return row.acceptance == "Accepted"
    if token == "C":
        return on(row.compression)
    if token == "D":
        return on(row.dissipation)
    if token == "E":
        return on(row.expansion)
    if token == "P":
        return row.participation == "Peak"
    if token == "K":
        return row.participation == "Climactic"
    return not any(transition_match(row, candidate) for candidate in TOKENS[:-1])


def accepted_indexes(rows: list[EvidenceBar]) -> list[int]:
    return [index for index, row in enumerate(rows) if row.acceptance == "Accepted"]


def transition_matrix(
    rows: list[EvidenceBar],
    anchors: list[int],
) -> dict[tuple[int, str], TransitionStats]:
    output = {}
    for offset in OFFSETS:
        available = [index for index in anchors if index + offset < len(rows)]
        denominator = len(available) if available else 1
        for token in TOKENS:
            count = sum(transition_match(rows[index + offset], token) for index in available)
            output[(offset, token)] = TransitionStats(count, count / denominator)
    return output


def accepted_runs(rows: list[EvidenceBar]) -> list[tuple[int, int]]:
    runs = []
    index = 0
    while index < len(rows):
        if rows[index].acceptance != "Accepted":
            index += 1
            continue
        start = index
        while index + 1 < len(rows) and rows[index + 1].acceptance == "Accepted":
            index += 1
        runs.append((start, index))
        index += 1
    return runs


def run_labels(length: int) -> list[str]:
    labels = [str(length) if length < 5 else "5+"]
    if length >= 10:
        labels.append("10+")
    return labels


def persistence_stats(
    rows: list[EvidenceBar],
    runs: list[tuple[int, int]] | None = None,
) -> dict[str, OutcomeStats]:
    grouped = {label: [] for label in ("1", "2", "3", "4", "5+", "10+")}
    for start, end in runs if runs is not None else accepted_runs(rows):
        value = direction_relative_return(rows, end, 5)
        if value is None:
            continue
        for label in run_labels(end - start + 1):
            grouped[label].append(value)
    return {label: outcome(values) for label, values in grouped.items()}


def following_acceptance_persistence(rows: list[EvidenceBar], cases: list[Case]) -> dict[str, OutcomeStats]:
    grouped = {label: [] for label in ("1", "2", "3", "4", "5+", "10+")}
    for case in cases:
        start = case.index + 1
        if start >= len(rows) or rows[start].acceptance != "Accepted":
            continue
        end = start
        while end + 1 < len(rows) and rows[end + 1].acceptance == "Accepted":
            end += 1
        for label in run_labels(end - start + 1):
            grouped[label].append(case.drfwd5)
    return {label: outcome(values) for label, values in grouped.items()}


def sequence_string(rows: list[EvidenceBar], start: int, length: int) -> str:
    return "->".join(state_token(rows[index]) for index in range(start, start + length))


def sequence_stats(
    rows: list[EvidenceBar],
    anchors: list[int],
    minimum_count: int = SEQUENCE_MIN_COUNT,
) -> dict[str, SequenceStats]:
    grouped: dict[str, list[float]] = {}
    for index in anchors:
        value = direction_relative_return(rows, index, 5)
        if value is None:
            continue
        for length in SEQUENCE_LENGTHS:
            if index + length > len(rows):
                continue
            key = sequence_string(rows, index, length)
            grouped.setdefault(key, []).append(value)
    output = {}
    for key, values in grouped.items():
        if len(values) < minimum_count:
            continue
        stats = outcome(values)
        output[key] = SequenceStats(key, stats.count, stats.mean, stats.continuation_rate, stats.failure_rate)
    return output


def future_count(rows: list[EvidenceBar], index: int, window: int, token: str) -> float:
    return float(sum(state_token(rows[position]) == token for position in range(index + 1, min(len(rows), index + window + 1))))


def neighborhood_stats(rows: list[EvidenceBar]) -> list[ComparisonStats]:
    accepted = accepted_indexes(rows)
    other = [index for index, row in enumerate(rows) if row.acceptance != "Accepted"]
    output = []
    for window in NEIGHBORHOODS:
        for token in ("A", "C", "D", "E", "P", "K"):
            left = [future_count(rows, index, window, token) for index in accepted]
            right = [future_count(rows, index, window, token) for index in other]
            output.append(
                ComparisonStats(
                    f"Fwd{window}_{TOKEN_NAMES[token]}Count",
                    len(left),
                    len(right),
                    mean(left),
                    mean(right),
                    mean(left) - mean(right),
                    effect_size(left, right),
                )
            )
    return output


def transition_effects(
    success: dict[tuple[int, str], TransitionStats],
    failure: dict[tuple[int, str], TransitionStats],
) -> dict[tuple[int, str], float]:
    return {key: success[key].probability - failure[key].probability for key in success}


def transition_effect_sizes(
    rows: list[EvidenceBar],
    success: list[int],
    failure: list[int],
) -> dict[tuple[int, str], float]:
    output = {}
    for offset in OFFSETS:
        success_available = [index for index in success if index + offset < len(rows)]
        failure_available = [index for index in failure if index + offset < len(rows)]
        for token in TOKENS:
            left = [float(transition_match(rows[index + offset], token)) for index in success_available]
            right = [float(transition_match(rows[index + offset], token)) for index in failure_available]
            output[(offset, token)] = effect_size(left, right)
    return output


def sequence_effects(
    success: dict[str, SequenceStats],
    failure: dict[str, SequenceStats],
) -> dict[str, float]:
    names = set(success) | set(failure)
    return {
        name: success.get(name, SequenceStats(name, 0, 0.0, 0.0, 0.0)).continuation_rate
        - failure.get(name, SequenceStats(name, 0, 0.0, 0.0, 0.0)).continuation_rate
        for name in names
    }


def population_b_study(rows: list[EvidenceBar]) -> PopulationBStudy:
    indexes = mature_aligned_lateral_indexes(rows)
    cases = cases_for_indexes(rows, indexes)
    success = [case.index for case in cases if case.drfwd5 > 0.0]
    failure = [case.index for case in cases if case.drfwd5 < 0.0]
    success_transitions = transition_matrix(rows, success)
    failure_transitions = transition_matrix(rows, failure)
    success_sequences = sequence_stats(rows, success)
    failure_sequences = sequence_stats(rows, failure)
    return PopulationBStudy(
        success_transitions,
        failure_transitions,
        transition_effects(success_transitions, failure_transitions),
        transition_effect_sizes(rows, success, failure),
        following_acceptance_persistence(rows, [case for case in cases if case.drfwd5 > 0.0]),
        following_acceptance_persistence(rows, [case for case in cases if case.drfwd5 < 0.0]),
        success_sequences,
        failure_sequences,
        sequence_effects(success_sequences, failure_sequences),
    )


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    anchors = accepted_indexes(rows)
    return InstrumentStudy(
        instrument_name(path),
        path,
        rows,
        anchors,
        transition_matrix(rows, anchors),
        persistence_stats(rows),
        sequence_stats(rows, anchors),
        neighborhood_stats(rows),
        population_b_study(rows),
    )


def append_transition_table(lines: list[str], title: str, matrix: dict[tuple[int, str], TransitionStats]) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(f"{'Offset':>6} {'Transition':<28} {'Count':>8} {'Probability':>12}")
    for offset in OFFSETS:
        for token in TOKENS:
            item = matrix[(offset, token)]
            lines.append(f"{offset:>6} {'Accepted -> ' + TOKEN_NAMES[token]:<28} {item.count:>8} {item.probability:>11.2%}")


def append_persistence(lines: list[str], title: str, grouped: dict[str, OutcomeStats]) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(f"{'RunLength':<10} {'Count':>8} {'MeanDRFwd5':>12} {'Median':>12} {'ContRate5':>10} {'FailRate5':>10}")
    for label, item in grouped.items():
        lines.append(f"{label:<10} {item.count:>8} {item.mean:>12.6f} {item.median:>12.6f} {item.continuation_rate:>9.2%} {item.failure_rate:>9.2%}")


def append_sequence_table(lines: list[str], title: str, items: list[SequenceStats]) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(f"{'Sequence':<28} {'Count':>8} {'MeanDRFwd5':>12} {'ContRate5':>10} {'FailRate5':>10}")
    for item in items[:TOP_LIMIT]:
        lines.append(f"{item.sequence:<28} {item.count:>8} {item.mean_drfwd5:>12.6f} {item.continuation_rate:>9.2%} {item.failure_rate:>9.2%}")


def append_sequence_rankings(lines: list[str], sequences: dict[str, SequenceStats], prefix: str = "") -> None:
    items = list(sequences.values())
    append_sequence_table(lines, prefix + "Most Common Sequences", sorted(items, key=lambda item: (-item.count, item.sequence)))
    append_sequence_table(lines, prefix + "Highest Continuation Sequences", sorted(items, key=lambda item: (-item.continuation_rate, -item.count, item.sequence)))
    append_sequence_table(lines, prefix + "Highest Failure Sequences", sorted(items, key=lambda item: (-item.failure_rate, -item.count, item.sequence)))


def instrument_report(study: InstrumentStudy) -> str:
    lines = [
        f"APVA Acceptance Sequence Study v0.1 - {study.instrument}",
        "=" * (38 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {len(study.rows)}",
        f"Accepted bars: {len(study.accepted_indexes)}",
        "Sequence encoding precedence: A Accepted, C Compression, D Dissipation, E Expansion, P Peak, K Climactic, N None.",
    ]
    lines.extend(["\nSection 1 - Acceptance Transition Matrix", "----------------------------------------"])
    append_transition_table(lines, "All Accepted Bars", study.transitions)
    lines.extend(["\nSection 2 - Acceptance Persistence", "----------------------------------"])
    append_persistence(lines, "Accepted Run Lengths", study.persistence)
    lines.extend(["\nSection 3 - Acceptance Sequence Mining", "--------------------------------------"])
    append_sequence_rankings(lines, study.sequences)
    lines.extend(["\nSection 4 - Acceptance Neighborhood", "-----------------------------------"])
    lines.append(f"{'Feature':<28} {'AcceptedN':>9} {'OtherN':>9} {'AcceptedMean':>13} {'OtherMean':>12} {'Delta':>10} {'Effect':>9}")
    for item in study.neighborhoods:
        lines.append(f"{item.name:<28} {item.accepted_count:>9} {item.other_count:>9} {item.accepted_mean:>13.6f} {item.other_mean:>12.6f} {item.delta:>10.6f} {item.effect:>9.4f}")
    lines.extend(["\nSection 5 - Population B Analysis", "---------------------------------"])
    lines.append("Population B tables are measured after qualifying event anchors, separated by DRFwd5 success and failure.")
    append_transition_table(lines, "Population B Success-Anchor Transitions", study.population_b.success_transitions)
    append_transition_table(lines, "Population B Failure-Anchor Transitions", study.population_b.failure_transitions)
    lines.extend(["\nPopulation B Transition Probability Effects", "-------------------------------------------"])
    lines.append(f"{'Offset':>6} {'Transition':<28} {'SuccessMinusFailure':>20} {'EffectSize':>12}")
    for offset in OFFSETS:
        for token in TOKENS:
            lines.append(
                f"{offset:>6} {'Accepted -> ' + TOKEN_NAMES[token]:<28} "
                f"{study.population_b.transition_deltas[(offset, token)]:>19.2%} "
                f"{study.population_b.transition_effects[(offset, token)]:>12.4f}"
            )
    lines.append("\nPopulation B persistence measures immediate Accepted runs beginning at anchor +1.")
    append_persistence(lines, "Population B Success-Anchor Persistence", study.population_b.success_persistence)
    append_persistence(lines, "Population B Failure-Anchor Persistence", study.population_b.failure_persistence)
    append_sequence_rankings(lines, study.population_b.success_sequences, "Population B Success ")
    append_sequence_rankings(lines, study.population_b.failure_sequences, "Population B Failure ")
    lines.extend(["\nSection 6 - Acceptance Hub Test", "-------------------------------"])
    append_transition_table(lines, "Acceptance Hub Probabilities", study.transitions)
    lines.extend(["\nSection 7 - Graph-Oriented Summary", "----------------------------------"])
    for offset in OFFSETS:
        lines.append(f"\nAccepted outgoing graph at +{offset}:")
        ordered = sorted(((token, study.transitions[(offset, token)]) for token in TOKENS), key=lambda item: (-item[1].probability, item[0]))
        for token, item in ordered:
            lines.append(f"  -> {TOKEN_NAMES[token]:<12} (p={item.probability:.2%}, n={item.count})")
    lines.extend(["\nResearch Notes", "=============="])
    for token in ("A", "D", "E", "C"):
        item = study.transitions[(1, token)]
        lines.append(f"- Accepted -> {TOKEN_NAMES[token]} at +1: p={item.probability:.2%}, n={item.count}.")
    best = max(study.transitions.items(), key=lambda item: item[1].probability)
    lines.append(f"- Largest outgoing probability: +{best[0][0]} Accepted -> {TOKEN_NAMES[best[0][1]]}, p={best[1].probability:.2%}.")
    lines.append("- Classification as state, transition, hub, or terminal condition is intentionally left unresolved.")
    return "\n".join(lines) + "\n"


def instrument_columns(studies: list[InstrumentStudy]) -> list[str]:
    available = [study.instrument for study in studies]
    return list(CANONICAL_INSTRUMENTS) + sorted(set(available) - set(CANONICAL_INSTRUMENTS))


def append_aggregate_transitions(lines: list[str], studies: list[InstrumentStudy], columns: list[str]) -> None:
    lines.extend(["\nAggregate Transition Matrix", "==========================="])
    header = f"{'Transition':<34}"
    for instrument in columns:
        header += f" {('P_' + instrument):>10}"
    header += f" {'Valid':>5} {'PositiveAgreement':>17} {'MeanProbability':>16}"
    lines.append(header)
    ranked = []
    for offset in OFFSETS:
        for token in TOKENS:
            values = {study.instrument: study.transitions[(offset, token)].probability for study in studies}
            valid = list(values.values())
            ranked.append((f"+{offset} Accepted -> {TOKEN_NAMES[token]}", mean(valid)))
            row = f"{'+%s Accepted -> %s' % (offset, TOKEN_NAMES[token]):<34}"
            for instrument in columns:
                row += f" {values.get(instrument, 0.0):>9.2%}"
            row += f" {len(valid):>5} {sum(value > 0.0 for value in valid):>17} {mean(valid):>15.2%}"
            lines.append(row)
    lines.extend(["\nMost Common Transitions", "======================="])
    for rank, (name, probability) in enumerate(sorted(ranked, key=lambda item: (-item[1], item[0]))[:TOP_LIMIT], start=1):
        lines.append(f"{rank:>4} {name:<34} MeanProbability={probability:.2%}")


def append_aggregate_persistence(lines: list[str], studies: list[InstrumentStudy]) -> None:
    lines.extend(["\nAggregate Acceptance Persistence", "================================"])
    lines.append(f"{'RunLength':<10} {'MeanCount':>10} {'MeanContRate5':>15}")
    for label in ("1", "2", "3", "4", "5+", "10+"):
        items = [study.persistence[label] for study in studies]
        lines.append(f"{label:<10} {mean([float(item.count) for item in items]):>10.2f} {mean([item.continuation_rate for item in items]):>14.2%}")


def append_aggregate_sequences(lines: list[str], studies: list[InstrumentStudy]) -> None:
    sequences = sorted(set().union(*(study.sequences for study in studies)))
    rows = []
    for sequence in sequences:
        items = [study.sequences[sequence] for study in studies if sequence in study.sequences]
        rows.append((sequence, sum(item.count for item in items), len(items), mean([item.continuation_rate for item in items]), mean([item.failure_rate for item in items])))
    lines.extend(["\nAggregate Acceptance Sequences", "=============================="])
    lines.append(f"{'Sequence':<28} {'TotalCount':>10} {'Instruments':>11} {'MeanContRate5':>14} {'MeanFailRate5':>14}")
    for sequence, count, instruments, continuation, failure in sorted(rows, key=lambda item: (-item[1], item[0]))[:TOP_LIMIT]:
        lines.append(f"{sequence:<28} {count:>10} {instruments:>11} {continuation:>13.2%} {failure:>13.2%}")
    for title, ordered in (
        ("Most Persistent Acceptance Chains", sorted(rows, key=lambda item: (-item[0].count('A'), -item[1], item[0]))),
        ("Highest Continuation Chains", sorted(rows, key=lambda item: (-item[3], -item[1], item[0]))),
        ("Highest Failure Chains", sorted(rows, key=lambda item: (-item[4], -item[1], item[0]))),
    ):
        lines.extend([f"\n{title}", "=" * len(title)])
        for rank, (sequence, count, instruments, continuation, failure) in enumerate(ordered[:TOP_LIMIT], start=1):
            lines.append(f"{rank:>4} {sequence:<28} Count={count:>7} Instruments={instruments} ContRate5={continuation:.2%} FailRate5={failure:.2%}")


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    columns = instrument_columns(studies)
    lines = [
        "APVA Acceptance Sequence Study v0.1 - Cross-Instrument Aggregate",
        "================================================================",
        f"Input instruments: {', '.join(study.instrument for study in studies)}",
        "Sequence encoding precedence: A Accepted, C Compression, D Dissipation, E Expansion, P Peak, K Climactic, N None.",
    ]
    append_aggregate_transitions(lines, studies, columns)
    append_aggregate_persistence(lines, studies)
    append_aggregate_sequences(lines, studies)
    lines.extend(["\nCross-Instrument Research Notes", "==============================="])
    for token in ("A", "D", "E", "C"):
        probabilities = [study.transitions[(1, token)].probability for study in studies]
        lines.append(f"- +1 Accepted -> {TOKEN_NAMES[token]} MeanProbability={mean(probabilities):.2%}.")
    lines.append("- State, transition, hub, and terminal-condition labels remain mechanical questions, not conclusions.")
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
            Path("Evidence") / "Output" / study.instrument / f"AcceptanceSequence_{study.instrument}.txt",
            instrument_report(study),
        )
    report = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, report)
    print(report, end="")


if __name__ == "__main__":
    main()
