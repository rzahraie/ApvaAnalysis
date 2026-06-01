"""APVA State Transition Graph Study v0.1.

Research-only graph view over the existing APVA evidence-state alphabet.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

from APVA_AcceptanceSequence_24 import TOKEN_NAMES, TOKENS, state_token
from APVA_BreakoutContext_08 import EvidenceBar, direction_relative_return, load_rows
from APVA_RegimeTransition_16 import CANONICAL_INSTRUMENTS, instrument_name, write_text
from APVA_StateGrammar_25 import OutcomeStats, format_oer, mean, summarize


AGGREGATE_OUTPUT = Path("Evidence/Output/StateTransitionGraph/StateTransitionGraph_All.txt")
PER_INSTRUMENT_MIN_COUNT = 20
AGGREGATE_MIN_COUNT = 50
TOP_LIMIT = 25

PERSISTENCE_EDGES = {f"{token}->{token}" for token in TOKENS}
ALTERNATION_EDGES = {
    "A->C", "C->A", "A->D", "D->A", "C->D",
    "D->C", "C->E", "E->C", "A->E", "E->A",
}
TRIPLET_CLASSES = {
    "SameDirection": {"A->C->A", "C->A->C", "D->E->D", "E->D->E"},
    "ContinuationOfState": {"A->A->A", "C->C->C", "D->D->D", "E->E->E"},
    "BreakFromPersistence": {"A->A->C", "C->C->A", "C->C->E", "A->A->D"},
    "ReturnToState": {"A->C->A", "C->A->C", "A->D->A", "D->A->D"},
}
EDGE_CLASS_NAMES = ("Persistence", "Alternation", "Other")
TRIPLET_CLASS_NAMES = ("SameDirection", "ContinuationOfState", "BreakFromPersistence", "ReturnToState", "Other")


@dataclass(frozen=True)
class ItemStats:
    item_type: str
    item: str
    classes: tuple[str, ...]
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
    total_edges: int
    total_triplets: int
    items: dict[tuple[str, str], ItemStats]


@dataclass(frozen=True)
class AggregateItem:
    item_type: str
    item: str
    classes: tuple[str, ...]
    count: int
    outcome: OutcomeStats
    values: dict[str, ItemStats]
    valid_instruments: int
    positive_count: int
    negative_count: int
    mean_continuation: float
    mean_failure: float
    mean_oer: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study APVA evidence-state graph edges and transition pairs."
    )
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def edge_classes(item: str) -> tuple[str, ...]:
    if item in PERSISTENCE_EDGES:
        return ("Persistence",)
    if item in ALTERNATION_EDGES:
        return ("Alternation",)
    return ("Other",)


def triplet_classes(item: str) -> tuple[str, ...]:
    classes = tuple(name for name, values in TRIPLET_CLASSES.items() if item in values)
    return classes or ("Other",)


def expected_count(item: str, total_windows: int, frequencies: dict[str, float]) -> float:
    probability = 1.0
    for token in item.split("->"):
        probability *= frequencies[token]
    return total_windows * probability


def build_items(
    rows: list[EvidenceBar],
    tokens: list[str],
    frequencies: dict[str, float],
) -> tuple[int, int, dict[tuple[str, str], ItemStats]]:
    output: dict[tuple[str, str], ItemStats] = {}
    totals = {"Edge": max(0, len(rows) - 1), "Triplet": max(0, len(rows) - 2)}
    for item_type, length, classifier in (
        ("Edge", 2, edge_classes),
        ("Triplet", 3, triplet_classes),
    ):
        raw_counts: dict[str, int] = {}
        outcomes: dict[str, list[float]] = {}
        for start in range(totals[item_type]):
            item = "->".join(tokens[start : start + length])
            raw_counts[item] = raw_counts.get(item, 0) + 1
            value = direction_relative_return(rows, start + length - 1, 5)
            if value is not None:
                outcomes.setdefault(item, []).append(value)
        for item, count in raw_counts.items():
            expected = expected_count(item, totals[item_type], frequencies)
            values = outcomes.get(item, [])
            output[(item_type, item)] = ItemStats(
                item_type,
                item,
                classifier(item),
                count,
                summarize(values),
                expected,
                count / expected if expected else None,
                tuple(values),
            )
    return totals["Edge"], totals["Triplet"], output


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    tokens = [state_token(row) for row in rows]
    counts = {token: tokens.count(token) for token in TOKENS}
    denominator = len(tokens) if tokens else 1
    frequencies = {token: counts[token] / denominator for token in TOKENS}
    total_edges, total_triplets, items = build_items(rows, tokens, frequencies)
    return InstrumentStudy(
        instrument_name(path),
        path,
        rows,
        tokens,
        counts,
        frequencies,
        total_edges,
        total_triplets,
        items,
    )


def item_label(item: ItemStats | AggregateItem) -> str:
    return ",".join(item.classes)


def filter_items(
    items: list[ItemStats],
    item_type: str | None = None,
    class_name: str | None = None,
) -> list[ItemStats]:
    return [
        item for item in items
        if item.count >= PER_INSTRUMENT_MIN_COUNT
        and (item_type is None or item.item_type == item_type)
        and (class_name is None or class_name in item.classes)
    ]


def ranked(items: list[ItemStats], mode: str) -> list[ItemStats]:
    if mode == "common":
        return sorted(items, key=lambda item: (-item.count, item.item))
    if mode == "oer":
        return sorted(items, key=lambda item: (-(item.oer or 0.0), -item.count, item.item))
    if mode == "continuation":
        return sorted(items, key=lambda item: (-item.outcome.continuation_rate, -item.count, item.item))
    if mode == "failure":
        return sorted(items, key=lambda item: (-item.outcome.failure_rate, -item.count, item.item))
    return sorted(items, key=lambda item: (item.item_type, item.item))


def append_item_table(lines: list[str], title: str, items: list[ItemStats]) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(
        f"{'Item':<17} {'Class':<38} {'Count':>7} {'OutcomeN':>9} {'OER':>9} "
        f"{'MeanDRFwd5':>12} {'Median':>12} {'ContRate5':>10} {'FailRate5':>10} {'FlatRate5':>10}"
    )
    for item in items:
        lines.append(
            f"{item.item:<17} {item_label(item):<38} {item.count:>7} {item.outcome.count:>9} "
            f"{format_oer(item.oer):>9} {item.outcome.mean:>12.6f} {item.outcome.median:>12.6f} "
            f"{item.outcome.continuation_rate:>9.2%} {item.outcome.failure_rate:>9.2%} "
            f"{item.outcome.flat_rate:>9.2%}"
        )


def append_ranked(lines: list[str], title: str, items: list[ItemStats]) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(
        f"{'Rank':>4} {'Item':<17} {'Class':<38} {'Count':>7} {'OutcomeN':>9} "
        f"{'OER':>9} {'Mean':>12} {'ContRate5':>10} {'FailRate5':>10}"
    )
    for rank, item in enumerate(items[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {item.item:<17} {item_label(item):<38} {item.count:>7} "
            f"{item.outcome.count:>9} {format_oer(item.oer):>9} {item.outcome.mean:>12.6f} "
            f"{item.outcome.continuation_rate:>9.2%} {item.outcome.failure_rate:>9.2%}"
        )


def summarize_class(items: list[ItemStats], class_name: str) -> OutcomeStats:
    return summarize([
        value
        for item in items
        if class_name in item.classes
        for value in item.outcome_values
    ])


def append_class_summary(lines: list[str], title: str, items: list[ItemStats], classes: tuple[str, ...]) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(f"{'Class':<22} {'Count':>9} {'OutcomeN':>9} {'MeanDRFwd5':>12} {'ContRate5':>10} {'FailRate5':>10}")
    for class_name in classes:
        raw_count = sum(item.count for item in items if class_name in item.classes)
        stats = summarize_class(items, class_name)
        lines.append(
            f"{class_name:<22} {raw_count:>9} {stats.count:>9} {stats.mean:>12.6f} "
            f"{stats.continuation_rate:>9.2%} {stats.failure_rate:>9.2%}"
        )


def lookup_note(study: InstrumentStudy, item_type: str, item: str) -> str:
    value = study.items.get((item_type, item))
    if value is None:
        return f"{item}: absent"
    return (
        f"{item}: Count={value.count}, OER={format_oer(value.oer)}, "
        f"ContinuationRate5={value.outcome.continuation_rate:.2%}, "
        f"FailureRate5={value.outcome.failure_rate:.2%}"
    )


def instrument_report(study: InstrumentStudy) -> str:
    all_items = list(study.items.values())
    edges = sorted((item for item in all_items if item.item_type == "Edge"), key=lambda item: item.item)
    triplets = sorted((item for item in all_items if item.item_type == "Triplet"), key=lambda item: item.item)
    eligible_edges = filter_items(all_items, "Edge")
    eligible_triplets = filter_items(all_items, "Triplet")
    lines = [
        f"APVA State Transition Graph Study v0.1 - {study.instrument}",
        "=" * (42 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {len(study.rows)}",
        f"Valid polarity rows: {sum(row.polarity in {'Black', 'Red'} for row in study.rows)}",
        "State encoding precedence: A Accepted, C Compression, D Dissipation, E Expansion, P Peak, K Climactic, N None.",
        "OER is a crude independence baseline over raw recurrence counts.",
        "Triplet class membership may overlap where a requested motif belongs to more than one class.",
        "",
        "Section 1 - Diagnostics",
        "=======================",
        f"{'State':<14} {'Count':>10} {'Frequency':>12}",
    ]
    for token in TOKENS:
        lines.append(f"{token + ' ' + TOKEN_NAMES[token]:<14} {study.state_counts[token]:>10} {study.frequencies[token]:>11.2%}")
    lines.extend([f"\nTotal edges: {study.total_edges}", f"Total edge-pairs: {study.total_triplets}"])
    append_item_table(lines, "Section 2 - Edge Table", edges)
    append_item_table(lines, "Section 3 - Edge-Pair Table", triplets)
    append_class_summary(lines, "Edge Class Summary", edges, EDGE_CLASS_NAMES)
    append_class_summary(lines, "Transition Momentum Summary", triplets, TRIPLET_CLASS_NAMES)
    lines.extend(["\nSection 4 - Ranked Edge Sections", "================================"])
    append_ranked(lines, "Most Common Edges", ranked(eligible_edges, "common"))
    append_ranked(lines, "Most Overrepresented Edges", ranked(eligible_edges, "oer"))
    append_ranked(lines, "Highest Continuation Edges", ranked(eligible_edges, "continuation"))
    append_ranked(lines, "Highest Failure Edges", ranked(eligible_edges, "failure"))
    append_ranked(lines, "Best Alternation Edges", ranked(filter_items(all_items, "Edge", "Alternation"), "continuation"))
    append_ranked(lines, "Worst Persistence Edges", ranked(filter_items(all_items, "Edge", "Persistence"), "failure"))
    lines.extend(["\nSection 5 - Ranked Edge-Pair Sections", "====================================="])
    append_ranked(lines, "Most Common Triplets", ranked(eligible_triplets, "common"))
    append_ranked(lines, "Most Overrepresented Triplets", ranked(eligible_triplets, "oer"))
    append_ranked(lines, "Highest Continuation Triplets", ranked(eligible_triplets, "continuation"))
    append_ranked(lines, "Highest Failure Triplets", ranked(eligible_triplets, "failure"))
    append_ranked(lines, "Best ReturnToState Triplets", ranked(filter_items(all_items, "Triplet", "ReturnToState"), "continuation"))
    append_ranked(lines, "Worst ContinuationOfState Triplets", ranked(filter_items(all_items, "Triplet", "ContinuationOfState"), "failure"))
    lines.extend(["\nSection 6 - Research Notes", "=========================="])
    edge_class = {name: summarize_class(edges, name) for name in EDGE_CLASS_NAMES}
    lines.append(f"- Alternation continuation rate: {edge_class['Alternation'].continuation_rate:.2%}; persistence continuation rate: {edge_class['Persistence'].continuation_rate:.2%}.")
    for item_type, item in (("Edge", "A->C"), ("Edge", "C->A"), ("Edge", "A->A"), ("Triplet", "A->C->A"), ("Triplet", "A->A->A")):
        lines.append(f"- {lookup_note(study, item_type, item)}.")
    low_sample = sum(item.count < PER_INSTRUMENT_MIN_COUNT for item in all_items)
    lines.append(f"- {low_sample} observed graph items are below the per-instrument ranked-table threshold of {PER_INSTRUMENT_MIN_COUNT}.")
    return "\n".join(lines) + "\n"


def instrument_columns(studies: list[InstrumentStudy]) -> list[str]:
    available = [study.instrument for study in studies]
    return list(CANONICAL_INSTRUMENTS) + sorted(set(available) - set(CANONICAL_INSTRUMENTS))


def aggregate_items(studies: list[InstrumentStudy]) -> list[AggregateItem]:
    keys = sorted(set().union(*(study.items for study in studies)))
    output = []
    for key in keys:
        item_type, item = key
        values = {study.instrument: study.items[key] for study in studies if key in study.items}
        valid = [value for value in values.values() if value.count >= PER_INSTRUMENT_MIN_COUNT]
        outcome_values = [value for entry in values.values() for value in entry.outcome_values]
        classes = next(iter(values.values())).classes
        output.append(
            AggregateItem(
                item_type,
                item,
                classes,
                sum(value.count for value in values.values()),
                summarize(outcome_values),
                values,
                len(valid),
                sum(value.outcome.continuation_rate > 0.5 for value in valid),
                sum(value.outcome.continuation_rate < 0.5 for value in valid),
                mean([value.outcome.continuation_rate for value in valid]),
                mean([value.outcome.failure_rate for value in valid]),
                mean([value.oer for value in valid if value.oer is not None]),
            )
        )
    return [item for item in output if item.count >= AGGREGATE_MIN_COUNT]


def append_aggregate_table(lines: list[str], items: list[AggregateItem], columns: list[str]) -> None:
    lines.extend(["\nAggregate Graph Item Table", "=========================="])
    header = f"{'Type':<8} {'Item':<17} {'Class':<38}"
    for instrument in columns:
        header += f" {('Count_' + instrument):>9} {('Cont_' + instrument):>9} {('Fail_' + instrument):>9} {('OER_' + instrument):>9}"
    header += f" {'Valid':>5} {'Pos':>4} {'Neg':>4} {'MeanCont':>9} {'MeanFail':>9} {'MeanOER':>9}"
    lines.append(header)
    for item in sorted(items, key=lambda row: (row.item_type, -row.count, row.item)):
        row = f"{item.item_type:<8} {item.item:<17} {item_label(item):<38}"
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


def aggregate_class_summary(studies: list[InstrumentStudy], columns: list[str]) -> list[str]:
    lines = ["\nAggregate Class Summary", "======================="]
    header = f"{'ItemType':<10} {'Class':<22}"
    for instrument in columns:
        header += f" {('Count_' + instrument):>9} {('N_' + instrument):>9} {('Cont_' + instrument):>9} {('Fail_' + instrument):>9}"
    header += f" {'Count_All':>10} {'N_All':>9} {'Cont_All':>9} {'Fail_All':>9}"
    lines.append(header)
    for item_type, classes in (("Edge", EDGE_CLASS_NAMES), ("Triplet", TRIPLET_CLASS_NAMES)):
        for class_name in classes:
            per_instrument = {}
            raw_counts = {}
            all_values = []
            all_raw_count = 0
            for study in studies:
                raw_count = sum(
                    item.count for item in study.items.values()
                    if item.item_type == item_type and class_name in item.classes
                )
                values = [
                    value for item in study.items.values()
                    if item.item_type == item_type and class_name in item.classes
                    for value in item.outcome_values
                ]
                raw_counts[study.instrument] = raw_count
                per_instrument[study.instrument] = summarize(values)
                all_raw_count += raw_count
                all_values.extend(values)
            combined = summarize(all_values)
            row = f"{item_type:<10} {class_name:<22}"
            for instrument in columns:
                stats = per_instrument.get(instrument, summarize([]))
                row += f" {raw_counts.get(instrument, 0):>9} {stats.count:>9} {stats.continuation_rate:>8.2%} {stats.failure_rate:>8.2%}"
            row += f" {all_raw_count:>10} {combined.count:>9} {combined.continuation_rate:>8.2%} {combined.failure_rate:>8.2%}"
            lines.append(row)
    return lines


def aggregate_ranked(
    items: list[AggregateItem],
    mode: str,
    item_type: str | None = None,
    class_name: str | None = None,
) -> list[AggregateItem]:
    eligible = [
        item for item in items
        if item.valid_instruments >= 2
        and (item_type is None or item.item_type == item_type)
        and (class_name is None or class_name in item.classes)
    ]
    if mode == "continuation":
        return sorted(eligible, key=lambda item: (-item.positive_count, -item.mean_continuation, -item.count, item.item))
    if mode == "failure":
        return sorted(eligible, key=lambda item: (-item.negative_count, -item.mean_failure, -item.count, item.item))
    if mode == "oer_positive":
        return sorted(eligible, key=lambda item: (-item.mean_oer, -item.mean_continuation, -item.count, item.item))
    if mode == "oer_negative":
        return sorted(eligible, key=lambda item: (-item.mean_oer, -item.mean_failure, -item.count, item.item))
    return sorted(eligible, key=lambda item: (-item.count, item.item))


def append_aggregate_ranked(lines: list[str], title: str, items: list[AggregateItem]) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(
        f"{'Rank':>4} {'Type':<8} {'Item':<17} {'Class':<38} {'Count':>7} "
        f"{'Valid':>5} {'Pos':>4} {'Neg':>4} {'MeanCont':>9} {'MeanFail':>9} {'MeanOER':>9}"
    )
    for rank, item in enumerate(items[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {item.item_type:<8} {item.item:<17} {item_label(item):<38} "
            f"{item.count:>7} {item.valid_instruments:>5} {item.positive_count:>4} "
            f"{item.negative_count:>4} {item.mean_continuation:>8.2%} "
            f"{item.mean_failure:>8.2%} {item.mean_oer:>9.4f}"
        )


def aggregate_lookup(items: list[AggregateItem], item_type: str, item: str) -> str:
    value = next((entry for entry in items if entry.item_type == item_type and entry.item == item), None)
    if value is None:
        return f"{item}: below aggregate reporting threshold or absent"
    return (
        f"{item}: ValidInstruments={value.valid_instruments}, MeanContRate5={value.mean_continuation:.2%}, "
        f"MeanFailRate5={value.mean_failure:.2%}, MeanOER={value.mean_oer:.4f}"
    )


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    columns = instrument_columns(studies)
    items = aggregate_items(studies)
    lines = [
        "APVA State Transition Graph Study v0.1 - Cross-Instrument Aggregate",
        "====================================================================",
        f"Input instruments: {', '.join(study.instrument for study in studies)}",
        f"Aggregate raw-count threshold: {AGGREGATE_MIN_COUNT}.",
        f"Replication threshold: Count >= {PER_INSTRUMENT_MIN_COUNT} in at least two instruments.",
        "State encoding precedence: A Accepted, C Compression, D Dissipation, E Expansion, P Peak, K Climactic, N None.",
        "OER is a crude independence baseline over raw recurrence counts.",
        "Triplet class membership may overlap where a requested motif belongs to more than one class.",
    ]
    append_aggregate_table(lines, items, columns)
    lines.extend(aggregate_class_summary(studies, columns))
    append_aggregate_ranked(lines, "Most Replicated Positive Edges", aggregate_ranked(items, "continuation", "Edge"))
    append_aggregate_ranked(lines, "Most Replicated Negative Edges", aggregate_ranked(items, "failure", "Edge"))
    append_aggregate_ranked(lines, "Most Replicated Positive Triplets", aggregate_ranked(items, "continuation", "Triplet"))
    append_aggregate_ranked(lines, "Most Replicated Negative Triplets", aggregate_ranked(items, "failure", "Triplet"))
    append_aggregate_ranked(lines, "Most Overrepresented Positive Edges and Triplets", aggregate_ranked(items, "oer_positive"))
    append_aggregate_ranked(lines, "Most Overrepresented Negative Edges and Triplets", aggregate_ranked(items, "oer_negative"))
    append_aggregate_ranked(lines, "Best Alternation Items", aggregate_ranked(items, "continuation", class_name="Alternation"))
    append_aggregate_ranked(lines, "Worst Persistence Items", aggregate_ranked(items, "failure", class_name="Persistence"))
    append_aggregate_ranked(lines, "Best ReturnToState Triplets", aggregate_ranked(items, "continuation", "Triplet", "ReturnToState"))
    append_aggregate_ranked(lines, "Worst ContinuationOfState Triplets", aggregate_ranked(items, "failure", "Triplet", "ContinuationOfState"))
    lines.extend(["\nCross-Instrument Research Notes", "==============================="])
    for item_type, item in (
        ("Edge", "A->C"),
        ("Edge", "C->A"),
        ("Triplet", "A->C->A"),
        ("Triplet", "C->A->C"),
        ("Triplet", "A->A->A"),
    ):
        lines.append(f"- {aggregate_lookup(items, item_type, item)}.")
    lines.append("- Class summaries provide the mechanical comparison of alternation and persistence.")
    lines.append("- OER is descriptive only; recurrence and consequence remain separate measurements.")
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
            Path("Evidence") / "Output" / study.instrument / f"StateTransitionGraph_{study.instrument}.txt",
            instrument_report(study),
        )
    report = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, report)
    print(report, end="")


if __name__ == "__main__":
    main()
