"""APVA State Grammar Study v0.1.

Research-only rolling sequence study over existing APVA evidence states.
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from APVA_AcceptanceSequence_24 import TOKEN_NAMES, TOKENS, state_token
from APVA_BreakoutContext_08 import EvidenceBar, direction_relative_return, load_rows
from APVA_RegimeTransition_16 import CANONICAL_INSTRUMENTS, instrument_name, write_text


AGGREGATE_OUTPUT = Path("Evidence/Output/StateGrammar/StateGrammar_All.txt")
LENGTHS = (3, 4, 5, 6)
PER_INSTRUMENT_MIN_COUNT = 20
AGGREGATE_MIN_COUNT = 50
TOP_LIMIT = 25


@dataclass(frozen=True)
class OutcomeStats:
    count: int
    mean: float
    median: float
    continuation_rate: float
    failure_rate: float
    flat_rate: float


@dataclass(frozen=True)
class SequenceStats:
    sequence: str
    length: int
    count: int
    outcome: OutcomeStats
    expected_count: float
    oer: float | None
    outcome_values: tuple[float, ...]


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    rows: list[EvidenceBar]
    tokens: list[str]
    state_counts: dict[str, int]
    frequencies: dict[str, float]
    windows: dict[int, int]
    sequences: dict[tuple[int, str], SequenceStats]


@dataclass(frozen=True)
class AggregateStats:
    sequence: str
    length: int
    count: int
    outcome: OutcomeStats
    values: dict[str, SequenceStats]
    valid_instruments: int
    positive_count: int
    negative_count: int
    mean_continuation: float
    mean_failure: float
    mean_oer: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover recurring APVA evidence-state grammar and measure consequences."
    )
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summarize(values: list[float]) -> OutcomeStats:
    count = len(values)
    denominator = count if count else 1
    return OutcomeStats(
        count,
        mean(values),
        statistics.median(values) if values else 0.0,
        sum(value > 0.0 for value in values) / denominator,
        sum(value < 0.0 for value in values) / denominator,
        sum(value == 0.0 for value in values) / denominator,
    )


def sequence_string(tokens: list[str], start: int, length: int) -> str:
    return "->".join(tokens[start : start + length])


def expected_count(sequence: str, total_windows: int, frequencies: dict[str, float]) -> float:
    probability = 1.0
    for token in sequence.split("->"):
        probability *= frequencies[token]
    return total_windows * probability


def build_sequences(
    rows: list[EvidenceBar],
    tokens: list[str],
    frequencies: dict[str, float],
) -> tuple[dict[int, int], dict[tuple[int, str], SequenceStats]]:
    windows = {}
    output = {}
    for length in LENGTHS:
        window_count = max(0, len(rows) - length + 1)
        windows[length] = window_count
        raw_counts: dict[str, int] = {}
        outcomes: dict[str, list[float]] = {}
        for start in range(window_count):
            sequence = sequence_string(tokens, start, length)
            raw_counts[sequence] = raw_counts.get(sequence, 0) + 1
            anchor = start + length - 1
            value = direction_relative_return(rows, anchor, 5)
            if value is not None:
                outcomes.setdefault(sequence, []).append(value)
        for sequence, count in raw_counts.items():
            expected = expected_count(sequence, window_count, frequencies)
            output[(length, sequence)] = SequenceStats(
                sequence,
                length,
                count,
                summarize(outcomes.get(sequence, [])),
                expected,
                count / expected if expected else None,
                tuple(outcomes.get(sequence, [])),
            )
    return windows, output


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    tokens = [state_token(row) for row in rows]
    counts = {token: tokens.count(token) for token in TOKENS}
    denominator = len(tokens) if tokens else 1
    frequencies = {token: counts[token] / denominator for token in TOKENS}
    windows, sequences = build_sequences(rows, tokens, frequencies)
    return InstrumentStudy(
        instrument_name(path),
        path,
        rows,
        tokens,
        counts,
        frequencies,
        windows,
        sequences,
    )


def format_oer(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def eligible(
    study: InstrumentStudy,
    length: int,
    root: str | None = None,
) -> list[SequenceStats]:
    return [
        item
        for (item_length, _), item in study.sequences.items()
        if item_length == length
        and item.count >= PER_INSTRUMENT_MIN_COUNT
        and (root is None or item.sequence.startswith(root + "->"))
    ]


def append_table(lines: list[str], title: str, items: list[SequenceStats]) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(
        f"{'Sequence':<31} {'Count':>7} {'OutcomeN':>9} {'OER':>9} "
        f"{'MeanDRFwd5':>12} {'Median':>12} {'ContRate5':>10} {'FailRate5':>10} {'FlatRate5':>10}"
    )
    for item in items:
        lines.append(
            f"{item.sequence:<31} {item.count:>7} {item.outcome.count:>9} "
            f"{format_oer(item.oer):>9} {item.outcome.mean:>12.6f} "
            f"{item.outcome.median:>12.6f} {item.outcome.continuation_rate:>9.2%} "
            f"{item.outcome.failure_rate:>9.2%} {item.outcome.flat_rate:>9.2%}"
        )


def ranked_items(items: list[SequenceStats], mode: str) -> list[SequenceStats]:
    if mode == "common":
        return sorted(items, key=lambda item: (-item.count, item.sequence))
    if mode == "oer":
        return sorted(items, key=lambda item: (-(item.oer or 0.0), -item.count, item.sequence))
    if mode == "continuation":
        return sorted(items, key=lambda item: (-item.outcome.continuation_rate, -item.count, item.sequence))
    if mode == "failure":
        return sorted(items, key=lambda item: (-item.outcome.failure_rate, -item.count, item.sequence))
    if mode == "positive_mean":
        return sorted(items, key=lambda item: (-item.outcome.mean, -item.count, item.sequence))
    return sorted(items, key=lambda item: (item.outcome.mean, -item.count, item.sequence))


def append_rankings(lines: list[str], title: str, items: list[SequenceStats]) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    for heading, mode in (
        ("Most Common", "common"),
        ("Most Overrepresented", "oer"),
        ("Highest Continuation", "continuation"),
        ("Highest Failure", "failure"),
        ("Highest Positive Mean DRFwd5", "positive_mean"),
        ("Highest Negative Mean DRFwd5", "negative_mean"),
    ):
        lines.extend([f"\n{heading}", "-" * len(heading)])
        lines.append(f"{'Rank':>4} {'Sequence':<31} {'Count':>7} {'OutcomeN':>9} {'OER':>9} {'Mean':>12} {'ContRate5':>10} {'FailRate5':>10}")
        for rank, item in enumerate(ranked_items(items, mode)[:TOP_LIMIT], start=1):
            lines.append(
                f"{rank:>4} {item.sequence:<31} {item.count:>7} {item.outcome.count:>9} "
                f"{format_oer(item.oer):>9} {item.outcome.mean:>12.6f} "
                f"{item.outcome.continuation_rate:>9.2%} {item.outcome.failure_rate:>9.2%}"
            )


def motif_note(study: InstrumentStudy, sequence: str) -> str:
    key = (sequence.count("->") + 1, sequence)
    item = study.sequences.get(key)
    if item is None:
        return f"{sequence}: absent"
    return (
        f"{sequence}: Count={item.count}, OER={format_oer(item.oer)}, "
        f"ContinuationRate5={item.outcome.continuation_rate:.2%}, "
        f"FailureRate5={item.outcome.failure_rate:.2%}"
    )


def instrument_report(study: InstrumentStudy) -> str:
    valid_polarity = sum(row.polarity in {"Black", "Red"} for row in study.rows)
    lines = [
        f"APVA State Grammar Study v0.1 - {study.instrument}",
        "=" * (32 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {len(study.rows)}",
        f"Valid polarity rows: {valid_polarity}",
        "State encoding precedence: A Accepted, C Compression, D Dissipation, E Expansion, P Peak, K Climactic, N None.",
        "OER is a crude independence baseline: observed raw windows divided by expected windows.",
        "",
        "Section 1 - Diagnostics",
        "=======================",
        f"{'State':<10} {'Count':>10} {'Frequency':>12}",
    ]
    for token in TOKENS:
        lines.append(f"{token + ' ' + TOKEN_NAMES[token]:<10} {study.state_counts[token]:>10} {study.frequencies[token]:>11.2%}")
    lines.extend(["\nTotal Windows", "-------------"])
    for length in LENGTHS:
        lines.append(f"Length {length}: {study.windows[length]}")

    lines.extend(["\nSection 2 - Sequence Consequence Tables", "======================================="])
    for length in LENGTHS:
        append_table(lines, f"Length {length} Sequences (Count >= {PER_INSTRUMENT_MIN_COUNT})", ranked_items(eligible(study, length), "common"))

    lines.extend(["\nSection 3 - Ranked Sections", "==========================="])
    for length in LENGTHS:
        append_rankings(lines, f"Length {length} Rankings", eligible(study, length))

    lines.extend(["\nSection 4 - Acceptance-Rooted Sequences", "======================================"])
    for length in LENGTHS:
        items = eligible(study, length, "A")
        append_table(lines, f"A-Rooted Length {length}", ranked_items(items, "common"))
        append_rankings(lines, f"A-Rooted Length {length} Rankings", items)

    lines.extend(["\nSection 5 - Compression-Rooted Sequences", "======================================="])
    for length in LENGTHS:
        items = eligible(study, length, "C")
        append_table(lines, f"C-Rooted Length {length}", ranked_items(items, "common"))
        append_rankings(lines, f"C-Rooted Length {length} Rankings", items)

    lines.extend(["\nSection 6 - Research Notes", "=========================="])
    common = ranked_items([item for length in LENGTHS for item in eligible(study, length)], "common")
    oer = ranked_items([item for length in LENGTHS for item in eligible(study, length)], "oer")
    if common:
        lines.append(f"- Most common reported motif: {common[0].sequence}, Count={common[0].count}.")
    if oer:
        lines.append(f"- Most overrepresented reported motif: {oer[0].sequence}, OER={format_oer(oer[0].oer)}.")
    for sequence in ("A->A->A", "A->C->A", "A->A->A->C->C"):
        lines.append(f"- {motif_note(study, sequence)}.")
    lines.append("- OER is descriptive only and should not be overinterpreted.")
    return "\n".join(lines) + "\n"


def instrument_columns(studies: list[InstrumentStudy]) -> list[str]:
    available = [study.instrument for study in studies]
    return list(CANONICAL_INSTRUMENTS) + sorted(set(available) - set(CANONICAL_INSTRUMENTS))


def aggregate_sequences(studies: list[InstrumentStudy]) -> list[AggregateStats]:
    keys = sorted(set().union(*(study.sequences for study in studies)))
    output = []
    for key in keys:
        length, sequence = key
        values = {study.instrument: study.sequences[key] for study in studies if key in study.sequences}
        valid = [item for item in values.values() if item.count >= PER_INSTRUMENT_MIN_COUNT]
        consequence_values = [value for item in values.values() for value in item.outcome_values]
        output.append(
            AggregateStats(
                sequence,
                length,
                sum(item.count for item in values.values()),
                summarize(consequence_values),
                values,
                len(valid),
                sum(item.outcome.continuation_rate > 0.5 for item in valid),
                sum(item.outcome.continuation_rate < 0.5 for item in valid),
                mean([item.outcome.continuation_rate for item in valid]),
                mean([item.outcome.failure_rate for item in valid]),
                mean([item.oer for item in valid if item.oer is not None]),
            )
        )
    return [item for item in output if item.count >= AGGREGATE_MIN_COUNT]


def append_aggregate_table(lines: list[str], items: list[AggregateStats], columns: list[str]) -> None:
    lines.extend(["\nAggregate Sequence Table", "========================"])
    header = f"{'Sequence':<31} {'Len':>3}"
    for instrument in columns:
        header += f" {('Count_' + instrument):>9} {('Cont_' + instrument):>9} {('Fail_' + instrument):>9} {('OER_' + instrument):>9}"
    header += f" {'Valid':>5} {'Pos':>4} {'Neg':>4} {'MeanCont':>9} {'MeanFail':>9} {'MeanOER':>9}"
    lines.append(header)
    for item in sorted(items, key=lambda row: (row.length, -row.count, row.sequence)):
        row = f"{item.sequence:<31} {item.length:>3}"
        for instrument in columns:
            value = item.values.get(instrument)
            row += (
                f" {value.count:>9} {value.outcome.continuation_rate:>8.2%} "
                f"{value.outcome.failure_rate:>8.2%} {format_oer(value.oer):>9}"
                if value
                else f" {0:>9} {0.0:>8.2%} {0.0:>8.2%} {'N/A':>9}"
            )
        row += (
            f" {item.valid_instruments:>5} {item.positive_count:>4} {item.negative_count:>4} "
            f"{item.mean_continuation:>8.2%} {item.mean_failure:>8.2%} {item.mean_oer:>9.4f}"
        )
        lines.append(row)


def aggregate_ranked(items: list[AggregateStats], mode: str) -> list[AggregateStats]:
    eligible_items = [item for item in items if item.valid_instruments >= 2]
    if mode == "continuation":
        return sorted(eligible_items, key=lambda item: (-item.positive_count, -item.mean_continuation, -item.count, item.sequence))
    if mode == "failure":
        return sorted(eligible_items, key=lambda item: (-item.negative_count, -item.mean_failure, -item.count, item.sequence))
    if mode == "oer":
        return sorted(eligible_items, key=lambda item: (-item.mean_oer, -item.count, item.sequence))
    return sorted(eligible_items, key=lambda item: (-item.count, item.sequence))


def append_aggregate_ranked(lines: list[str], title: str, items: list[AggregateStats]) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(f"{'Rank':>4} {'Sequence':<31} {'Len':>3} {'Count':>8} {'Valid':>5} {'Pos':>4} {'Neg':>4} {'MeanCont':>9} {'MeanFail':>9} {'MeanOER':>9}")
    for rank, item in enumerate(items[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {item.sequence:<31} {item.length:>3} {item.count:>8} "
            f"{item.valid_instruments:>5} {item.positive_count:>4} {item.negative_count:>4} "
            f"{item.mean_continuation:>8.2%} {item.mean_failure:>8.2%} {item.mean_oer:>9.4f}"
        )


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    columns = instrument_columns(studies)
    items = aggregate_sequences(studies)
    lines = [
        "APVA State Grammar Study v0.1 - Cross-Instrument Aggregate",
        "===========================================================",
        f"Input instruments: {', '.join(study.instrument for study in studies)}",
        f"Aggregate raw-count threshold: {AGGREGATE_MIN_COUNT}.",
        f"Replication threshold: Count >= {PER_INSTRUMENT_MIN_COUNT} in at least two instruments.",
        "State encoding precedence: A Accepted, C Compression, D Dissipation, E Expansion, P Peak, K Climactic, N None.",
        "OER is a crude independence baseline.",
    ]
    append_aggregate_table(lines, items, columns)
    append_aggregate_ranked(lines, "Most Replicated Continuation Sequences", aggregate_ranked(items, "continuation"))
    append_aggregate_ranked(lines, "Most Replicated Failure Sequences", aggregate_ranked(items, "failure"))
    append_aggregate_ranked(lines, "Most Market-General Overrepresented Sequences", aggregate_ranked(items, "oer"))
    append_aggregate_ranked(lines, "Best A-Rooted Continuation Sequences", aggregate_ranked([item for item in items if item.sequence.startswith("A->")], "continuation"))
    append_aggregate_ranked(lines, "Worst A-Rooted Failure Sequences", aggregate_ranked([item for item in items if item.sequence.startswith("A->")], "failure"))
    append_aggregate_ranked(lines, "Best C-Rooted Continuation Sequences", aggregate_ranked([item for item in items if item.sequence.startswith("C->")], "continuation"))
    append_aggregate_ranked(lines, "Worst C-Rooted Failure Sequences", aggregate_ranked([item for item in items if item.sequence.startswith("C->")], "failure"))
    append_aggregate_ranked(
        lines,
        "Overrepresented Continuation Motifs",
        sorted(
            [item for item in items if item.valid_instruments >= 2 and item.mean_oer > 1.5 and item.mean_continuation > 0.55],
            key=lambda item: (-item.mean_continuation, -item.mean_oer, -item.count, item.sequence),
        ),
    )
    append_aggregate_ranked(
        lines,
        "Overrepresented Failure Motifs",
        sorted(
            [item for item in items if item.valid_instruments >= 2 and item.mean_oer > 1.5 and item.mean_failure > 0.55],
            key=lambda item: (-item.mean_failure, -item.mean_oer, -item.count, item.sequence),
        ),
    )
    lines.extend(["\nCross-Instrument Research Notes", "==============================="])
    for sequence in ("A->A->A", "A->C->A", "A->A->A->C->C"):
        matching = next((item for item in items if item.sequence == sequence), None)
        if matching:
            lines.append(f"- {sequence}: ValidInstruments={matching.valid_instruments}, MeanContRate5={matching.mean_continuation:.2%}, MeanFailRate5={matching.mean_failure:.2%}, MeanOER={matching.mean_oer:.4f}.")
        else:
            lines.append(f"- {sequence}: below aggregate reporting threshold or absent.")
    lines.append("- Recurrence and consequence are reported separately; OER remains a crude independence baseline.")
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
            Path("Evidence") / "Output" / study.instrument / f"StateGrammar_{study.instrument}.txt",
            instrument_report(study),
        )
    report = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, report)
    print(report, end="")


if __name__ == "__main__":
    main()
