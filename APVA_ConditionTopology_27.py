"""APVA Condition Topology Study v0.1.

Research-only co-occurrence study over existing APVA evidence conditions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from APVA_AcceptanceAnatomy_23 import mature_aligned_lateral_indexes
from APVA_BreakoutContext_08 import EvidenceBar, direction_relative_return, load_rows
from APVA_LateralAnatomy_19 import on
from APVA_RegimeTransition_16 import CANONICAL_INSTRUMENTS, instrument_name, write_text
from APVA_StateGrammar_25 import OutcomeStats, mean, summarize


AGGREGATE_OUTPUT = Path("Evidence/Output/ConditionTopology/ConditionTopology_All.txt")
PER_INSTRUMENT_MIN_COUNT = 20
AGGREGATE_MIN_COUNT = 50
TOP_LIMIT = 25

CONDITIONS = ("Accepted", "Compressed", "Dissipating", "Expanding", "Peak", "Climactic")
PAIRS = (
    ("Accepted", "Compressed"),
    ("Accepted", "Dissipating"),
    ("Accepted", "Expanding"),
    ("Accepted", "Peak"),
    ("Accepted", "Climactic"),
    ("Compressed", "Dissipating"),
    ("Compressed", "Expanding"),
    ("Compressed", "Peak"),
    ("Dissipating", "Peak"),
    ("Dissipating", "Climactic"),
    ("Expanding", "Peak"),
    ("Expanding", "Climactic"),
)
TRIPLES = (
    ("Accepted", "Compressed", "Dissipating"),
    ("Accepted", "Compressed", "Peak"),
    ("Accepted", "Dissipating", "Peak"),
    ("Compressed", "Dissipating", "Peak"),
)
QUADRUPLES = (("Accepted", "Compressed", "Dissipating", "Peak"),)
COMBINATIONS = PAIRS + TRIPLES + QUADRUPLES
TOPOLOGY_VIEWS = {
    "Acceptance": (
        ("Accepted alone", ("Accepted",), True),
        ("Accepted+Compressed", ("Accepted", "Compressed"), False),
        ("Accepted+Dissipating", ("Accepted", "Dissipating"), False),
        ("Accepted+Peak", ("Accepted", "Peak"), False),
        ("Accepted+Compressed+Dissipating", ("Accepted", "Compressed", "Dissipating"), False),
        ("Accepted+Compressed+Peak", ("Accepted", "Compressed", "Peak"), False),
        ("Accepted+Dissipating+Peak", ("Accepted", "Dissipating", "Peak"), False),
    ),
    "Compression": (
        ("Compressed alone", ("Compressed",), True),
        ("Compressed+Accepted", ("Compressed", "Accepted"), False),
        ("Compressed+Dissipating", ("Compressed", "Dissipating"), False),
        ("Compressed+Peak", ("Compressed", "Peak"), False),
        ("Compressed+Dissipating+Peak", ("Compressed", "Dissipating", "Peak"), False),
    ),
    "Dissipation": (
        ("Dissipating alone", ("Dissipating",), True),
        ("Dissipating+Accepted", ("Dissipating", "Accepted"), False),
        ("Dissipating+Compressed", ("Dissipating", "Compressed"), False),
        ("Dissipating+Peak", ("Dissipating", "Peak"), False),
        ("Dissipating+Accepted+Peak", ("Dissipating", "Accepted", "Peak"), False),
    ),
}


@dataclass(frozen=True)
class ComboStats:
    name: str
    conditions: tuple[str, ...]
    item_type: str
    count: int
    frequency: float
    expected_frequency: float
    lift: float | None
    outcome: OutcomeStats
    outcome_values: tuple[float, ...]


@dataclass(frozen=True)
class PopulationStudy:
    name: str
    indexes: list[int]
    condition_counts: dict[str, int]
    condition_frequencies: dict[str, float]
    combinations: dict[str, ComboStats]


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    rows: list[EvidenceBar]
    populations: dict[str, PopulationStudy]


@dataclass(frozen=True)
class AggregateCombo:
    name: str
    item_type: str
    values: dict[str, ComboStats]
    count: int
    valid_instruments: int
    positive_count: int
    negative_count: int
    mean_continuation: float
    mean_failure: float
    mean_lift: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study APVA evidence-condition co-occurrence topology."
    )
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def flags(row: EvidenceBar) -> dict[str, bool]:
    return {
        "Accepted": row.acceptance == "Accepted",
        "Compressed": on(row.compression),
        "Dissipating": on(row.dissipation),
        "Expanding": on(row.expansion),
        "Peak": row.participation == "Peak",
        "Climactic": row.participation == "Climactic",
    }


def combo_name(conditions: tuple[str, ...]) -> str:
    return "+".join(conditions)


def combo_type(conditions: tuple[str, ...]) -> str:
    return {2: "Pair", 3: "Triple", 4: "Quadruple"}[len(conditions)]


def matches(row_flags: dict[str, bool], conditions: tuple[str, ...], exact: bool = False) -> bool:
    if not all(row_flags[condition] for condition in conditions):
        return False
    return not exact or sum(row_flags.values()) == len(conditions)


def build_combo_stats(
    rows: list[EvidenceBar],
    indexes: list[int],
    condition_frequencies: dict[str, float],
    conditions: tuple[str, ...],
) -> ComboStats:
    selected = [index for index in indexes if matches(flags(rows[index]), conditions)]
    denominator = len(indexes) if indexes else 1
    frequency = len(selected) / denominator
    expected = 1.0
    for condition in conditions:
        expected *= condition_frequencies[condition]
    values = [
        value
        for index in selected
        if (value := direction_relative_return(rows, index, 5)) is not None
    ]
    return ComboStats(
        combo_name(conditions),
        conditions,
        combo_type(conditions),
        len(selected),
        frequency,
        expected,
        frequency / expected if expected else None,
        summarize(values),
        tuple(values),
    )


def build_population(rows: list[EvidenceBar], name: str, indexes: list[int]) -> PopulationStudy:
    row_flags = {index: flags(rows[index]) for index in indexes}
    counts = {
        condition: sum(value[condition] for value in row_flags.values())
        for condition in CONDITIONS
    }
    denominator = len(indexes) if indexes else 1
    frequencies = {condition: counts[condition] / denominator for condition in CONDITIONS}
    combinations = {
        combo_name(conditions): build_combo_stats(rows, indexes, frequencies, conditions)
        for conditions in COMBINATIONS
    }
    return PopulationStudy(name, indexes, counts, frequencies, combinations)


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    population_b = mature_aligned_lateral_indexes(rows)
    return InstrumentStudy(
        instrument_name(path),
        path,
        rows,
        {
            "Full Population": build_population(rows, "Full Population", list(range(len(rows)))),
            "Population B": build_population(rows, "Population B", population_b),
        },
    )


def format_lift(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def append_frequency_table(lines: list[str], population: PopulationStudy) -> None:
    denominator = len(population.indexes)
    lines.extend([f"\n{population.name} Condition Frequencies", "-" * (len(population.name) + 22)])
    lines.append(f"Population bars: {denominator}")
    lines.append(f"{'Condition':<38} {'Type':<10} {'Count':>8} {'Frequency':>10}")
    for condition in CONDITIONS:
        lines.append(
            f"{condition:<38} {'Condition':<10} {population.condition_counts[condition]:>8} "
            f"{population.condition_frequencies[condition]:>9.2%}"
        )
    for item in population.combinations.values():
        lines.append(f"{item.name:<38} {item.item_type:<10} {item.count:>8} {item.frequency:>9.2%}")


def append_consequence_table(
    lines: list[str],
    title: str,
    items: list[ComboStats],
    include_low_count: bool = False,
) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(
        f"{'Combination':<38} {'Type':<10} {'Count':>8} {'OutcomeN':>9} {'Lift':>9} "
        f"{'MeanDRFwd5':>12} {'Median':>12} {'ContRate5':>10} {'FailRate5':>10} {'FlatRate5':>10}"
    )
    for item in items:
        if not include_low_count and item.count < PER_INSTRUMENT_MIN_COUNT:
            continue
        lines.append(
            f"{item.name:<38} {item.item_type:<10} {item.count:>8} {item.outcome.count:>9} "
            f"{format_lift(item.lift):>9} {item.outcome.mean:>12.6f} {item.outcome.median:>12.6f} "
            f"{item.outcome.continuation_rate:>9.2%} {item.outcome.failure_rate:>9.2%} "
            f"{item.outcome.flat_rate:>9.2%}"
        )


def append_lift_rankings(lines: list[str], title: str, items: list[ComboStats]) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(f"{'Rank':>4} {'Combination':<38} {'Type':<10} {'Count':>8} {'Observed':>10} {'Expected':>10} {'Lift':>9}")
    ranked = sorted(
        [item for item in items if item.count >= PER_INSTRUMENT_MIN_COUNT],
        key=lambda item: (-(item.lift or 0.0), -item.count, item.name),
    )
    for rank, item in enumerate(ranked[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {item.name:<38} {item.item_type:<10} {item.count:>8} "
            f"{item.frequency:>9.2%} {item.expected_frequency:>9.2%} {format_lift(item.lift):>9}"
        )


def append_population_comparison(
    lines: list[str],
    full: PopulationStudy,
    population_b: PopulationStudy,
) -> None:
    lines.extend(["\nPopulation B vs Full Population", "-------------------------------"])
    lines.append(
        f"{'Combination':<38} {'FullN':>7} {'FullCont':>9} {'FullFail':>9} {'FullLift':>9} "
        f"{'PopBN':>7} {'PopBCont':>9} {'PopBFail':>9} {'PopBLift':>9} "
        f"{'DeltaCont':>10} {'DeltaFail':>10} {'DeltaLift':>10}"
    )
    for name, b_item in population_b.combinations.items():
        full_item = full.combinations[name]
        lift_delta = (
            b_item.lift - full_item.lift
            if b_item.lift is not None and full_item.lift is not None
            else None
        )
        lines.append(
            f"{name:<38} {full_item.count:>7} {full_item.outcome.continuation_rate:>8.2%} "
            f"{full_item.outcome.failure_rate:>8.2%} {format_lift(full_item.lift):>9} "
            f"{b_item.count:>7} {b_item.outcome.continuation_rate:>8.2%} "
            f"{b_item.outcome.failure_rate:>8.2%} {format_lift(b_item.lift):>9} "
            f"{b_item.outcome.continuation_rate - full_item.outcome.continuation_rate:>9.2%} "
            f"{b_item.outcome.failure_rate - full_item.outcome.failure_rate:>9.2%} "
            f"{format_lift(lift_delta):>10}"
        )


def topology_stats(rows: list[EvidenceBar], indexes: list[int], label: str, conditions: tuple[str, ...], exact: bool) -> ComboStats:
    denominator = len(indexes) if indexes else 1
    selected = [index for index in indexes if matches(flags(rows[index]), conditions, exact)]
    values = [
        value
        for index in selected
        if (value := direction_relative_return(rows, index, 5)) is not None
    ]
    return ComboStats(
        label,
        conditions,
        "Alone" if exact else "Inclusive",
        len(selected),
        len(selected) / denominator,
        0.0,
        None,
        summarize(values),
        tuple(values),
    )


def append_topology_view(
    lines: list[str],
    title: str,
    rows: list[EvidenceBar],
    indexes: list[int],
    definitions: tuple[tuple[str, tuple[str, ...], bool], ...],
) -> None:
    items = [topology_stats(rows, indexes, *definition) for definition in definitions]
    append_consequence_table(lines, title, items, include_low_count=True)


def instrument_report(study: InstrumentStudy) -> str:
    full = study.populations["Full Population"]
    population_b = study.populations["Population B"]
    full_items = list(full.combinations.values())
    b_items = list(population_b.combinations.values())
    lines = [
        f"APVA Condition Topology Study v0.1 - {study.instrument}",
        "=" * (37 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {len(study.rows)}",
        f"Population B rows: {len(population_b.indexes)}",
        "Conditions are independent binary flags; a bar may belong to multiple combinations.",
        "Population B reuses the mature aligned lateral DissipationContained selector.",
        "",
        "Section 1 - Condition Frequencies",
        "=================================",
    ]
    append_frequency_table(lines, full)
    lines.extend(["\nSection 2 - Pair Consequences", "============================="])
    append_consequence_table(lines, "Full Population Pairs", [full.combinations[combo_name(item)] for item in PAIRS])
    lines.extend(["\nSection 3 - Triple Consequences", "==============================="])
    append_consequence_table(lines, "Full Population Triples", [full.combinations[combo_name(item)] for item in TRIPLES + QUADRUPLES])
    lines.extend(["\nSection 4 - Observed vs Expected", "================================"])
    append_lift_rankings(lines, "Most Overrepresented Pairs", [full.combinations[combo_name(item)] for item in PAIRS])
    append_lift_rankings(lines, "Most Overrepresented Triples", [full.combinations[combo_name(item)] for item in TRIPLES + QUADRUPLES])
    lines.extend(["\nSection 5 - Acceptance Topology", "==============================="])
    append_topology_view(lines, "Acceptance-Centered Comparison", study.rows, full.indexes, TOPOLOGY_VIEWS["Acceptance"])
    lines.extend(["\nSection 6 - Compression Topology", "================================"])
    append_topology_view(lines, "Compression-Centered Comparison", study.rows, full.indexes, TOPOLOGY_VIEWS["Compression"])
    lines.extend(["\nSection 7 - Dissipation Topology", "================================"])
    append_topology_view(lines, "Dissipation-Centered Comparison", study.rows, full.indexes, TOPOLOGY_VIEWS["Dissipation"])
    lines.extend(["\nSection 8 - Population B", "========================"])
    append_frequency_table(lines, population_b)
    append_consequence_table(lines, "Population B Pairs and Triples", b_items, include_low_count=True)
    append_lift_rankings(lines, "Population B Most Overrepresented Combinations", b_items)
    append_population_comparison(lines, full, population_b)
    lines.extend(["\nResearch Notes", "=============="])
    full_ranked = sorted(full_items, key=lambda item: (-item.outcome.continuation_rate, -item.count, item.name))
    lift_ranked = sorted(full_items, key=lambda item: (-(item.lift or 0.0), -item.count, item.name))
    if full_ranked:
        lines.append(f"- Highest full-population continuation combination: {full_ranked[0].name}, ContinuationRate5={full_ranked[0].outcome.continuation_rate:.2%}, Count={full_ranked[0].count}.")
    if lift_ranked:
        lines.append(f"- Highest full-population lift combination: {lift_ranked[0].name}, Lift={format_lift(lift_ranked[0].lift)}, Count={lift_ranked[0].count}.")
    lines.append(f"- {sum(item.count < PER_INSTRUMENT_MIN_COUNT for item in full_items)} full-population combinations are below the ranked-table threshold of {PER_INSTRUMENT_MIN_COUNT}.")
    lines.append(f"- Population B contains {len(population_b.indexes)} rows; its table is printed without suppressing low counts.")
    lines.append("- Co-occurrence lift is descriptive only and does not imply a causal relationship.")
    return "\n".join(lines) + "\n"


def instrument_columns(studies: list[InstrumentStudy]) -> list[str]:
    available = [study.instrument for study in studies]
    return list(CANONICAL_INSTRUMENTS) + sorted(set(available) - set(CANONICAL_INSTRUMENTS))


def aggregate_combinations(studies: list[InstrumentStudy]) -> list[AggregateCombo]:
    output = []
    for conditions in COMBINATIONS:
        name = combo_name(conditions)
        values = {
            study.instrument: study.populations["Full Population"].combinations[name]
            for study in studies
        }
        valid = [value for value in values.values() if value.count >= PER_INSTRUMENT_MIN_COUNT]
        output.append(
            AggregateCombo(
                name,
                combo_type(conditions),
                values,
                sum(value.count for value in values.values()),
                len(valid),
                sum(value.outcome.continuation_rate > 0.5 for value in valid),
                sum(value.outcome.continuation_rate < 0.5 for value in valid),
                mean([value.outcome.continuation_rate for value in valid]),
                mean([value.outcome.failure_rate for value in valid]),
                mean([value.lift for value in valid if value.lift is not None]),
            )
        )
    return [item for item in output if item.count >= AGGREGATE_MIN_COUNT]


def append_aggregate_table(lines: list[str], items: list[AggregateCombo], columns: list[str]) -> None:
    lines.extend(["\nAggregate Combination Table", "==========================="])
    header = f"{'Combination':<38} {'Type':<10}"
    for instrument in columns:
        header += f" {('Count_' + instrument):>9} {('Cont_' + instrument):>9} {('Fail_' + instrument):>9} {('Lift_' + instrument):>9}"
    header += f" {'Valid':>5} {'Pos':>4} {'Neg':>4} {'MeanCont':>9} {'MeanFail':>9} {'MeanLift':>9}"
    lines.append(header)
    for item in sorted(items, key=lambda value: (value.item_type, -value.count, value.name)):
        row = f"{item.name:<38} {item.item_type:<10}"
        for instrument in columns:
            value = item.values.get(instrument)
            row += (
                f" {value.count:>9} {value.outcome.continuation_rate:>8.2%} "
                f"{value.outcome.failure_rate:>8.2%} {format_lift(value.lift):>9}"
                if value
                else f" {0:>9} {0.0:>8.2%} {0.0:>8.2%} {'N/A':>9}"
            )
        row += (
            f" {item.valid_instruments:>5} {item.positive_count:>4} {item.negative_count:>4} "
            f"{item.mean_continuation:>8.2%} {item.mean_failure:>8.2%} {item.mean_lift:>9.4f}"
        )
        lines.append(row)


def aggregate_ranked(
    items: list[AggregateCombo],
    mode: str,
    item_type: str | None = None,
    contains: str | None = None,
) -> list[AggregateCombo]:
    eligible = [
        item for item in items
        if item.valid_instruments >= 2
        and (item_type is None or item.item_type == item_type)
        and (contains is None or contains in item.name.split("+"))
    ]
    if mode == "positive":
        return sorted(eligible, key=lambda item: (-item.positive_count, -item.mean_continuation, -item.count, item.name))
    if mode == "negative":
        return sorted(eligible, key=lambda item: (-item.negative_count, -item.mean_failure, -item.count, item.name))
    return sorted(eligible, key=lambda item: (-item.mean_lift, -item.count, item.name))


def append_aggregate_ranked(lines: list[str], title: str, items: list[AggregateCombo]) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(
        f"{'Rank':>4} {'Combination':<38} {'Type':<10} {'Count':>8} {'Valid':>5} "
        f"{'Pos':>4} {'Neg':>4} {'MeanCont':>9} {'MeanFail':>9} {'MeanLift':>9}"
    )
    for rank, item in enumerate(items[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {item.name:<38} {item.item_type:<10} {item.count:>8} "
            f"{item.valid_instruments:>5} {item.positive_count:>4} {item.negative_count:>4} "
            f"{item.mean_continuation:>8.2%} {item.mean_failure:>8.2%} {item.mean_lift:>9.4f}"
        )


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    columns = instrument_columns(studies)
    items = aggregate_combinations(studies)
    lines = [
        "APVA Condition Topology Study v0.1 - Cross-Instrument Aggregate",
        "===============================================================",
        f"Input instruments: {', '.join(study.instrument for study in studies)}",
        f"Aggregate raw-count threshold: {AGGREGATE_MIN_COUNT}.",
        f"Replication threshold: Count >= {PER_INSTRUMENT_MIN_COUNT} in at least two instruments.",
        "Conditions are independent binary flags; combination rows are inclusive.",
        "Lift is observed co-occurrence divided by the independent-frequency baseline.",
    ]
    append_aggregate_table(lines, items, columns)
    append_aggregate_ranked(lines, "Most Replicated Positive Pairs", aggregate_ranked(items, "positive", "Pair"))
    append_aggregate_ranked(lines, "Most Replicated Negative Pairs", aggregate_ranked(items, "negative", "Pair"))
    append_aggregate_ranked(lines, "Most Replicated Positive Triples", aggregate_ranked(items, "positive", "Triple"))
    append_aggregate_ranked(lines, "Most Replicated Negative Triples", aggregate_ranked(items, "negative", "Triple"))
    append_aggregate_ranked(lines, "Highest Lift Pairs", aggregate_ranked(items, "lift", "Pair"))
    append_aggregate_ranked(lines, "Highest Lift Triples", aggregate_ranked(items, "lift", "Triple"))
    append_aggregate_ranked(lines, "Best Acceptance Combinations", aggregate_ranked(items, "positive", contains="Accepted"))
    append_aggregate_ranked(lines, "Worst Acceptance Combinations", aggregate_ranked(items, "negative", contains="Accepted"))
    append_aggregate_ranked(lines, "Best Compression Combinations", aggregate_ranked(items, "positive", contains="Compressed"))
    append_aggregate_ranked(lines, "Best Dissipation Combinations", aggregate_ranked(items, "positive", contains="Dissipating"))
    lines.extend(["\nCross-Instrument Research Notes", "==============================="])
    if items:
        positive = aggregate_ranked(items, "positive")
        lift = aggregate_ranked(items, "lift")
        if positive:
            lines.append(f"- Most replicated positive combination: {positive[0].name}, PositiveContinuationCount={positive[0].positive_count}, MeanContinuation={positive[0].mean_continuation:.2%}.")
        if lift:
            lines.append(f"- Highest replicated lift combination: {lift[0].name}, MeanLift={lift[0].mean_lift:.4f}.")
    lines.append("- Pair and triple topology results are descriptive comparisons against the prior state-sequence studies.")
    lines.append("- Co-occurrence lift does not imply causality or establish a new ontology.")
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
            Path("Evidence") / "Output" / study.instrument / f"ConditionTopology_{study.instrument}.txt",
            instrument_report(study),
        )
    report = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, report)
    print(report, end="")


if __name__ == "__main__":
    main()
