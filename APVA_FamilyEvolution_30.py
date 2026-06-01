"""APVA Family Evolution Study v0.1.

Research-only projection of existing APVA condition topology into requested families.
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from APVA_AcceptanceAnatomy_23 import mature_aligned_lateral_indexes
from APVA_BreakoutContext_08 import EvidenceBar, direction_relative_return, load_rows
from APVA_ConditionTopology_27 import flags
from APVA_ContextTopology_28 import CONTEXT_WINDOWS, context_metrics
from APVA_LateralAnatomy_19 import effect_size
from APVA_RegimeTransition_16 import CANONICAL_INSTRUMENTS, instrument_name, write_text
from APVA_StateGrammar_25 import OutcomeStats, format_oer, mean, summarize


AGGREGATE_OUTPUT = Path("Evidence/Output/FamilyEvolution/FamilyEvolution_All.txt")
FAMILIES = ("A", "B", "C", "D", "N")
PATH_LENGTHS = (2, 3, 4)
PER_INSTRUMENT_MIN_COUNT = 20
AGGREGATE_MIN_COUNT = 50
TOP_LIMIT = 25
FEATURES = tuple(
    f"Prev{size}{suffix}"
    for size in CONTEXT_WINDOWS
    for suffix in (
        "AcceptedCount",
        "CompressedCount",
        "DissipatingCount",
        "ExpandingCount",
        "PeakCount",
        "ClimacticCount",
        "Dominance",
        "NetAcceptance",
    )
)
FAMILY_CLASS = {"A": "Constructive", "B": "Constructive", "C": "Destructive", "D": "Destructive", "N": "Neutral"}
CLASS_TRANSITIONS = tuple(
    f"{source}->{target}"
    for source in ("Constructive", "Destructive", "Neutral")
    for target in ("Constructive", "Destructive", "Neutral")
)


@dataclass(frozen=True)
class TransitionStats:
    transition: str
    lag: int
    count: int
    probability: float
    expected_count: float
    lift: float | None


@dataclass(frozen=True)
class PersistenceStats:
    family: str
    run_count: int
    mean_duration: float
    median_duration: float
    percentile90: float
    max_duration: int


@dataclass(frozen=True)
class PathwayStats:
    pathway: str
    length: int
    count: int
    expected_count: float
    oer: float | None
    outcome: OutcomeStats
    outcome_values: tuple[float, ...]


@dataclass(frozen=True)
class ClassTransitionStats:
    transition: str
    count: int
    probability: float
    outcome: OutcomeStats


@dataclass(frozen=True)
class ContextFeature:
    feature: str
    present_count: int
    absent_count: int
    present_mean: float
    absent_mean: float
    delta: float
    effect: float


@dataclass(frozen=True)
class FamilyContext:
    family: str
    features: list[ContextFeature]


@dataclass(frozen=True)
class PopulationStudy:
    name: str
    indexes: list[int]
    family_by_index: dict[int, str]
    counts: dict[str, int]
    frequencies: dict[str, float]
    outcomes: dict[str, OutcomeStats]
    transitions: dict[tuple[int, str], TransitionStats]
    persistence: dict[str, PersistenceStats]
    pathways: dict[tuple[int, str], PathwayStats]
    class_transitions: dict[str, ClassTransitionStats]
    contexts: dict[str, FamilyContext]


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    rows: list[EvidenceBar]
    populations: dict[str, PopulationStudy]


@dataclass(frozen=True)
class AggregateOutcome:
    family: str
    values: dict[str, OutcomeStats]
    counts: dict[str, int]
    valid_instruments: int
    positive_count: int
    negative_count: int
    mean_continuation: float
    mean_failure: float


@dataclass(frozen=True)
class AggregateTransition:
    transition: str
    lag: int
    values: dict[str, TransitionStats]
    mean_lift: float


@dataclass(frozen=True)
class AggregatePathway:
    pathway: str
    length: int
    values: dict[str, PathwayStats]
    valid_instruments: int
    mean_continuation: float
    mean_failure: float
    mean_oer: float


@dataclass(frozen=True)
class AggregateContext:
    family: str
    feature: str
    values: dict[str, ContextFeature]
    valid_instruments: int
    positive_count: int
    negative_count: int
    mean_effect: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study requested APVA topology families and their evolution.")
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def family_for(row: EvidenceBar) -> str:
    value = flags(row)
    if value["Accepted"] and value["Compressed"]:
        return "A"
    if value["Compressed"] and value["Peak"]:
        return "B"
    if value["Accepted"] and (value["Peak"] or value["Climactic"]):
        return "C"
    if value["Accepted"] and value["Expanding"]:
        return "D"
    return "N"


def frequencies(family_by_index: dict[int, str]) -> tuple[dict[str, int], dict[str, float]]:
    counts = {family: sum(value == family for value in family_by_index.values()) for family in FAMILIES}
    denominator = len(family_by_index) if family_by_index else 1
    return counts, {family: counts[family] / denominator for family in FAMILIES}


def outcome_layer(rows: list[EvidenceBar], family_by_index: dict[int, str]) -> dict[str, OutcomeStats]:
    return {
        family: summarize([
            value
            for index, current in family_by_index.items()
            if current == family
            if (value := direction_relative_return(rows, index, 5)) is not None
        ])
        for family in FAMILIES
    }


def build_transitions(
    family_by_index: dict[int, str],
    family_frequencies: dict[str, float],
) -> dict[tuple[int, str], TransitionStats]:
    output = {}
    for lag in (1, 5):
        observed = {}
        total = 0
        for index, target in family_by_index.items():
            source = family_by_index.get(index - lag)
            if source is None:
                continue
            total += 1
            transition = f"{source}->{target}"
            observed[transition] = observed.get(transition, 0) + 1
        for source in FAMILIES:
            for target in FAMILIES:
                transition = f"{source}->{target}"
                count = observed.get(transition, 0)
                expected = total * family_frequencies[source] * family_frequencies[target]
                source_count = sum(value for name, value in observed.items() if name.startswith(source + "->"))
                output[(lag, transition)] = TransitionStats(
                    transition,
                    lag,
                    count,
                    count / source_count if source_count else 0.0,
                    expected,
                    count / expected if expected else None,
                )
    return output


def percentile90(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)])


def build_persistence(family_by_index: dict[int, str]) -> dict[str, PersistenceStats]:
    runs = {family: [] for family in FAMILIES}
    ordered = sorted(family_by_index)
    previous_index = None
    previous_family = None
    duration = 0
    for index in ordered:
        current = family_by_index[index]
        if previous_index is not None and index == previous_index + 1 and current == previous_family:
            duration += 1
        else:
            if previous_family is not None:
                runs[previous_family].append(duration)
            duration = 1
        previous_index = index
        previous_family = current
    if previous_family is not None:
        runs[previous_family].append(duration)
    return {
        family: PersistenceStats(
            family,
            len(values),
            mean(values),
            statistics.median(values) if values else 0.0,
            percentile90(values),
            max(values) if values else 0,
        )
        for family, values in runs.items()
    }


def pathway_expected(pathway: str, total: int, family_frequencies: dict[str, float]) -> float:
    probability = 1.0
    for family in pathway.split("->"):
        probability *= family_frequencies[family]
    return total * probability


def build_pathways(
    rows: list[EvidenceBar],
    family_by_index: dict[int, str],
    family_frequencies: dict[str, float],
) -> dict[tuple[int, str], PathwayStats]:
    output = {}
    allowed = set(family_by_index)
    for length in PATH_LENGTHS:
        counts = {}
        values = {}
        anchors = [index for index in sorted(allowed) if all(index - offset in allowed for offset in range(length))]
        for anchor in anchors:
            pathway = "->".join(family_by_index[index] for index in range(anchor - length + 1, anchor + 1))
            counts[pathway] = counts.get(pathway, 0) + 1
            outcome = direction_relative_return(rows, anchor, 5)
            if outcome is not None:
                values.setdefault(pathway, []).append(outcome)
        for pathway, count in counts.items():
            expected = pathway_expected(pathway, len(anchors), family_frequencies)
            outcomes = values.get(pathway, [])
            output[(length, pathway)] = PathwayStats(
                pathway,
                length,
                count,
                expected,
                count / expected if expected else None,
                summarize(outcomes),
                tuple(outcomes),
            )
    return output


def build_class_transitions(rows: list[EvidenceBar], family_by_index: dict[int, str]) -> dict[str, ClassTransitionStats]:
    raw = {transition: 0 for transition in CLASS_TRANSITIONS}
    values = {transition: [] for transition in CLASS_TRANSITIONS}
    for index, target in family_by_index.items():
        source = family_by_index.get(index - 1)
        if source is None:
            continue
        transition = f"{FAMILY_CLASS[source]}->{FAMILY_CLASS[target]}"
        raw[transition] += 1
        outcome = direction_relative_return(rows, index, 5)
        if outcome is not None:
            values[transition].append(outcome)
    total = sum(raw.values())
    return {
        transition: ClassTransitionStats(
            transition,
            raw[transition],
            raw[transition] / total if total else 0.0,
            summarize(values[transition]),
        )
        for transition in CLASS_TRANSITIONS
    }


def build_contexts(rows: list[EvidenceBar], family_by_index: dict[int, str]) -> dict[str, FamilyContext]:
    metrics = {index: context_metrics(rows, index) for index in family_by_index}
    output = {}
    for family in FAMILIES:
        present_indexes = [index for index, current in family_by_index.items() if current == family]
        absent_indexes = [index for index, current in family_by_index.items() if current != family]
        features = []
        for feature in FEATURES:
            present = [metrics[index][feature] for index in present_indexes]
            absent = [metrics[index][feature] for index in absent_indexes]
            features.append(
                ContextFeature(
                    feature,
                    len(present),
                    len(absent),
                    mean(present),
                    mean(absent),
                    mean(present) - mean(absent),
                    effect_size(present, absent),
                )
            )
        output[family] = FamilyContext(family, features)
    return output


def build_population(rows: list[EvidenceBar], name: str, indexes: list[int]) -> PopulationStudy:
    family_by_index = {index: family_for(rows[index]) for index in indexes}
    counts, family_frequencies = frequencies(family_by_index)
    return PopulationStudy(
        name,
        indexes,
        family_by_index,
        counts,
        family_frequencies,
        outcome_layer(rows, family_by_index),
        build_transitions(family_by_index, family_frequencies),
        build_persistence(family_by_index),
        build_pathways(rows, family_by_index, family_frequencies),
        build_class_transitions(rows, family_by_index),
        build_contexts(rows, family_by_index),
    )


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    return InstrumentStudy(
        instrument_name(path),
        path,
        rows,
        {
            "Full Population": build_population(rows, "Full Population", list(range(len(rows)))),
            "Population B": build_population(rows, "Population B", mature_aligned_lateral_indexes(rows)),
        },
    )


def append_frequency_table(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'Family':<8} {'Count':>10} {'Frequency':>10}")
    for family in FAMILIES:
        lines.append(f"{family:<8} {population.counts[family]:>10} {population.frequencies[family]:>9.2%}")


def append_outcome_table(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'Family':<8} {'Count':>9} {'MeanDRFwd5':>12} {'Median':>12} {'ContRate5':>10} {'FailRate5':>10} {'FlatRate5':>10}")
    for family in FAMILIES:
        value = population.outcomes[family]
        lines.append(
            f"{family:<8} {value.count:>9} {value.mean:>12.6f} {value.median:>12.6f} "
            f"{value.continuation_rate:>9.2%} {value.failure_rate:>9.2%} {value.flat_rate:>9.2%}"
        )


def append_transition_table(lines: list[str], population: PopulationStudy, lag: int) -> None:
    lines.append(f"{'Transition':<12} {'Count':>8} {'Probability':>12} {'Expected':>12} {'Lift':>9}")
    for transition in sorted(value for value_lag, value in population.transitions if value_lag == lag):
        value = population.transitions[(lag, transition)]
        lines.append(f"{transition:<12} {value.count:>8} {value.probability:>11.2%} {value.expected_count:>12.4f} {format_oer(value.lift):>9}")


def append_persistence_table(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'Family':<8} {'Runs':>8} {'Mean':>10} {'Median':>10} {'P90':>10} {'Max':>8}")
    for family in FAMILIES:
        value = population.persistence[family]
        lines.append(
            f"{family:<8} {value.run_count:>8} {value.mean_duration:>10.4f} "
            f"{value.median_duration:>10.4f} {value.percentile90:>10.4f} {value.max_duration:>8}"
        )


def append_pathway_table(lines: list[str], population: PopulationStudy, include_low_count: bool = False) -> None:
    lines.append(f"{'Pathway':<16} {'Len':>3} {'Count':>8} {'OutcomeN':>9} {'OER':>9} {'MeanDRFwd5':>12} {'ContRate5':>10} {'FailRate5':>10}")
    items = sorted(population.pathways.values(), key=lambda value: (value.length, -value.count, value.pathway))
    for value in items:
        if not include_low_count and value.count < PER_INSTRUMENT_MIN_COUNT:
            continue
        lines.append(
            f"{value.pathway:<16} {value.length:>3} {value.count:>8} {value.outcome.count:>9} "
            f"{format_oer(value.oer):>9} {value.outcome.mean:>12.6f} "
            f"{value.outcome.continuation_rate:>9.2%} {value.outcome.failure_rate:>9.2%}"
        )


def append_class_table(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'Transition':<30} {'Count':>8} {'Probability':>12} {'OutcomeN':>9} {'MeanDRFwd5':>12} {'ContRate5':>10} {'FailRate5':>10}")
    for transition in CLASS_TRANSITIONS:
        value = population.class_transitions[transition]
        lines.append(
            f"{transition:<30} {value.count:>8} {value.probability:>11.2%} {value.outcome.count:>9} "
            f"{value.outcome.mean:>12.6f} {value.outcome.continuation_rate:>9.2%} {value.outcome.failure_rate:>9.2%}"
        )


def append_context_tables(lines: list[str], population: PopulationStudy, include_rankings: bool = True) -> None:
    for family in FAMILIES:
        value = population.contexts[family]
        lines.extend([f"\nFamily {family} Context Signature", "-" * 26])
        lines.append(f"{'Feature':<32} {'PresentN':>8} {'AbsentN':>8} {'PresentMean':>12} {'AbsentMean':>12} {'Delta':>12} {'Effect':>11}")
        for item in value.features:
            lines.append(
                f"{item.feature:<32} {item.present_count:>8} {item.absent_count:>8} "
                f"{item.present_mean:>12.6f} {item.absent_mean:>12.6f} {item.delta:>12.6f} {item.effect:>11.6f}"
            )
        if include_rankings:
            for title, ordered in (
                ("Top Positive Predictors", sorted(value.features, key=lambda item: (-item.effect, item.feature))),
                ("Top Negative Predictors", sorted(value.features, key=lambda item: (item.effect, item.feature))),
            ):
                lines.extend([f"\n{title}", "." * len(title)])
                for rank, item in enumerate(ordered[:TOP_LIMIT], start=1):
                    lines.append(f"{rank:>3}. {item.feature:<32} Delta={item.delta:>10.6f} Effect={item.effect:>10.6f}")


def instrument_report(study: InstrumentStudy) -> str:
    full = study.populations["Full Population"]
    population_b = study.populations["Population B"]
    lines = [
        f"APVA Family Evolution Study v0.1 - {study.instrument}",
        "=" * (34 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {len(study.rows)}",
        f"Valid polarity rows: {sum(row.polarity in {'Black', 'Red'} for row in study.rows)}",
        "Family precedence: A, then B, then C, then D, then N.",
        "Family assignment is a requested research projection over existing condition flags.",
        "",
        "Section 1 - Family Frequencies",
        "==============================",
    ]
    append_frequency_table(lines, full)
    lines.extend(["\nSection 2 - Family Outcome Layer", "================================"])
    append_outcome_table(lines, full)
    lines.extend(["\nSection 3 - Family Transition Matrix", "===================================="])
    lines.extend(["\n1-Bar Transitions", "-----------------"])
    append_transition_table(lines, full, 1)
    lines.extend(["\n5-Bar Transitions", "-----------------"])
    append_transition_table(lines, full, 5)
    lines.extend(["\nSection 4 - Family Persistence", "=============================="])
    append_persistence_table(lines, full)
    lines.extend(["\nSection 5 - Family Pathways", "==========================="])
    append_pathway_table(lines, full)
    lines.extend(["\nSection 6 - Constructive vs Destructive", "======================================="])
    append_class_table(lines, full)
    lines.extend(["\nSection 7 - Context -> Family", "============================="])
    append_context_tables(lines, full)
    lines.extend(["\nSection 8 - Population B", "========================"])
    lines.append(f"Population B bars: {len(population_b.indexes)}")
    lines.extend(["\nFamily Frequencies", "------------------"])
    append_frequency_table(lines, population_b)
    lines.extend(["\nFamily Outcomes", "---------------"])
    append_outcome_table(lines, population_b)
    lines.extend(["\n1-Bar Transition Counts", "-----------------------"])
    append_transition_table(lines, population_b, 1)
    lines.extend(["\nContext -> Family Signatures", "----------------------------"])
    append_context_tables(lines, population_b, include_rankings=False)
    lines.extend(["\nSection 9 - Mechanical Research Notes", "====================================="])
    best = max(FAMILIES, key=lambda family: full.outcomes[family].continuation_rate)
    worst = min(FAMILIES, key=lambda family: full.outcomes[family].continuation_rate)
    persistent = max(FAMILIES, key=lambda family: full.persistence[family].mean_duration)
    lines.append(f"- Highest continuation family: {best}, ContinuationRate5={full.outcomes[best].continuation_rate:.2%}.")
    lines.append(f"- Lowest continuation family: {worst}, ContinuationRate5={full.outcomes[worst].continuation_rate:.2%}.")
    lines.append(f"- Most persistent family by mean duration: {persistent}, MeanDuration={full.persistence[persistent].mean_duration:.4f}.")
    lines.append(f"- Population B contains {len(population_b.indexes)} rows and is reported separately.")
    lines.append("- Family projection and context signatures are descriptive only.")
    return "\n".join(lines) + "\n"


def instrument_columns(studies: list[InstrumentStudy]) -> list[str]:
    available = [study.instrument for study in studies]
    return list(CANONICAL_INSTRUMENTS) + sorted(set(available) - set(CANONICAL_INSTRUMENTS))


def aggregate_outcomes(studies: list[InstrumentStudy]) -> list[AggregateOutcome]:
    output = []
    for family in FAMILIES:
        values = {study.instrument: study.populations["Full Population"].outcomes[family] for study in studies}
        counts = {study.instrument: study.populations["Full Population"].counts[family] for study in studies}
        valid = [values[name] for name, count in counts.items() if count >= PER_INSTRUMENT_MIN_COUNT]
        output.append(
            AggregateOutcome(
                family,
                values,
                counts,
                len(valid),
                sum(item.continuation_rate > 0.5 for item in valid),
                sum(item.continuation_rate < 0.5 for item in valid),
                mean([item.continuation_rate for item in valid]),
                mean([item.failure_rate for item in valid]),
            )
        )
    return output


def aggregate_transitions(studies: list[InstrumentStudy]) -> list[AggregateTransition]:
    output = []
    for lag in (1, 5):
        for source in FAMILIES:
            for target in FAMILIES:
                transition = f"{source}->{target}"
                values = {study.instrument: study.populations["Full Population"].transitions[(lag, transition)] for study in studies}
                output.append(AggregateTransition(transition, lag, values, mean([item.lift for item in values.values() if item.lift is not None])))
    return output


def aggregate_pathways(studies: list[InstrumentStudy]) -> list[AggregatePathway]:
    keys = sorted(set().union(*(study.populations["Full Population"].pathways for study in studies)))
    output = []
    for length, pathway in keys:
        values = {
            study.instrument: study.populations["Full Population"].pathways[(length, pathway)]
            for study in studies
            if (length, pathway) in study.populations["Full Population"].pathways
        }
        if sum(item.count for item in values.values()) < AGGREGATE_MIN_COUNT:
            continue
        valid = [item for item in values.values() if item.count >= PER_INSTRUMENT_MIN_COUNT]
        output.append(
            AggregatePathway(
                pathway,
                length,
                values,
                len(valid),
                mean([item.outcome.continuation_rate for item in valid]),
                mean([item.outcome.failure_rate for item in valid]),
                mean([item.oer for item in valid if item.oer is not None]),
            )
        )
    return output


def aggregate_contexts(studies: list[InstrumentStudy], population: str = "Full Population") -> list[AggregateContext]:
    output = []
    for family in FAMILIES:
        for feature in FEATURES:
            values = {
                study.instrument: next(
                    item for item in study.populations[population].contexts[family].features
                    if item.feature == feature
                )
                for study in studies
            }
            valid = [item for item in values.values() if item.present_count >= PER_INSTRUMENT_MIN_COUNT]
            output.append(
                AggregateContext(
                    family,
                    feature,
                    values,
                    len(valid),
                    sum(item.effect > 0.0 for item in valid),
                    sum(item.effect < 0.0 for item in valid),
                    mean([item.effect for item in valid]),
                )
            )
    return output


def append_aggregate_outcomes(lines: list[str], items: list[AggregateOutcome], columns: list[str]) -> None:
    lines.extend(["\nAggregate Family Outcome Table", "=============================="])
    header = f"{'Family':<8}"
    for instrument in columns:
        header += f" {('Count_' + instrument):>9} {('Cont_' + instrument):>9} {('Fail_' + instrument):>9} {('Mean_' + instrument):>12}"
    header += f" {'Valid':>5} {'Pos':>4} {'Neg':>4} {'MeanCont':>9} {'MeanFail':>9}"
    lines.append(header)
    for item in items:
        row = f"{item.family:<8}"
        for instrument in columns:
            value = item.values.get(instrument)
            row += (
                f" {item.counts.get(instrument, 0):>9} {value.continuation_rate:>8.2%} "
                f"{value.failure_rate:>8.2%} {value.mean:>12.6f}"
                if value else f" {0:>9} {0.0:>8.2%} {0.0:>8.2%} {0.0:>12.6f}"
            )
        row += f" {item.valid_instruments:>5} {item.positive_count:>4} {item.negative_count:>4} {item.mean_continuation:>8.2%} {item.mean_failure:>8.2%}"
        lines.append(row)


def append_aggregate_transitions(lines: list[str], items: list[AggregateTransition], columns: list[str]) -> None:
    lines.extend(["\nAggregate Transition Table", "=========================="])
    header = f"{'Transition':<12} {'Lag':>3}"
    for instrument in columns:
        header += f" {('Count_' + instrument):>9} {('Prob_' + instrument):>9} {('Lift_' + instrument):>9}"
    header += f" {'MeanLift':>9}"
    lines.append(header)
    for item in items:
        row = f"{item.transition:<12} {item.lag:>3}"
        for instrument in columns:
            value = item.values.get(instrument)
            row += (
                f" {value.count:>9} {value.probability:>8.2%} {format_oer(value.lift):>9}"
                if value else f" {0:>9} {0.0:>8.2%} {'N/A':>9}"
            )
        row += f" {item.mean_lift:>9.4f}"
        lines.append(row)


def append_aggregate_pathways(lines: list[str], items: list[AggregatePathway], columns: list[str]) -> None:
    lines.extend(["\nAggregate Pathway Table", "======================="])
    header = f"{'Pathway':<16} {'Len':>3}"
    for instrument in columns:
        header += f" {('Count_' + instrument):>9} {('Cont_' + instrument):>9} {('OER_' + instrument):>9}"
    header += f" {'Valid':>5} {'MeanCont':>9} {'MeanFail':>9} {'MeanOER':>9}"
    lines.append(header)
    for item in sorted(items, key=lambda value: (value.length, -sum(entry.count for entry in value.values.values()), value.pathway)):
        row = f"{item.pathway:<16} {item.length:>3}"
        for instrument in columns:
            value = item.values.get(instrument)
            row += (
                f" {value.count:>9} {value.outcome.continuation_rate:>8.2%} {format_oer(value.oer):>9}"
                if value else f" {0:>9} {0.0:>8.2%} {'N/A':>9}"
            )
        row += f" {item.valid_instruments:>5} {item.mean_continuation:>8.2%} {item.mean_failure:>8.2%} {item.mean_oer:>9.4f}"
        lines.append(row)


def append_aggregate_contexts(lines: list[str], items: list[AggregateContext], columns: list[str]) -> None:
    lines.extend(["\nAggregate Context -> Family Table", "================================="])
    header = f"{'Family':<8} {'Feature':<32}"
    for instrument in columns:
        header += f" {('Effect_' + instrument):>11}"
    header += f" {'Valid':>5} {'Pos':>4} {'Neg':>4} {'MeanEffect':>11}"
    lines.append(header)
    for item in items:
        row = f"{item.family:<8} {item.feature:<32}"
        for instrument in columns:
            value = item.values.get(instrument)
            row += f" {(value.effect if value else 0.0):>11.6f}"
        row += f" {item.valid_instruments:>5} {item.positive_count:>4} {item.negative_count:>4} {item.mean_effect:>11.6f}"
        lines.append(row)


def append_ranked(lines: list[str], title: str, rows: list[str]) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.extend(rows[:TOP_LIMIT])


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    columns = instrument_columns(studies)
    outcomes = aggregate_outcomes(studies)
    transitions = aggregate_transitions(studies)
    pathways = aggregate_pathways(studies)
    contexts = aggregate_contexts(studies)
    population_b = aggregate_contexts(studies, "Population B")
    context_lookup = {(item.family, item.feature): item for item in contexts}
    population_b_differences = [
        (
            item,
            item.mean_effect - context_lookup[(item.family, item.feature)].mean_effect,
        )
        for item in population_b
        if item.valid_instruments >= 2
        and context_lookup[(item.family, item.feature)].valid_instruments >= 2
    ]
    persistence = {
        family: mean([study.populations["Full Population"].persistence[family].mean_duration for study in studies])
        for family in FAMILIES
    }
    class_values = {
        transition: [
            study.populations["Full Population"].class_transitions[transition]
            for study in studies
        ]
        for transition in CLASS_TRANSITIONS
    }
    lines = [
        "APVA Family Evolution Study v0.1 - Cross-Instrument Aggregate",
        "============================================================",
        f"Input instruments: {', '.join(study.instrument for study in studies)}",
        "Family precedence: A, then B, then C, then D, then N.",
        f"Pathway aggregate threshold: {AGGREGATE_MIN_COUNT}; replication threshold: Count >= {PER_INSTRUMENT_MIN_COUNT} in at least two instruments.",
    ]
    append_aggregate_outcomes(lines, outcomes, columns)
    append_aggregate_transitions(lines, transitions, columns)
    append_aggregate_pathways(lines, pathways, columns)
    append_aggregate_contexts(lines, contexts, columns)
    eligible_outcomes = [item for item in outcomes if item.valid_instruments >= 2]
    eligible_paths = [item for item in pathways if item.valid_instruments >= 2]
    eligible_contexts = [item for item in contexts if item.valid_instruments >= 2]
    append_ranked(lines, "Best-Performing Families", [f"{item.family}: MeanContinuation={item.mean_continuation:.2%}, MeanFailure={item.mean_failure:.2%}" for item in sorted(eligible_outcomes, key=lambda value: -value.mean_continuation)])
    append_ranked(lines, "Worst-Performing Families", [f"{item.family}: MeanContinuation={item.mean_continuation:.2%}, MeanFailure={item.mean_failure:.2%}" for item in sorted(eligible_outcomes, key=lambda value: value.mean_continuation)])
    append_ranked(lines, "Most Persistent Families", [f"{family}: MeanDurationAcrossInstruments={value:.4f}" for family, value in sorted(persistence.items(), key=lambda pair: -pair[1])])
    for lag in (1, 5):
        append_ranked(lines, f"Strongest {lag}-Bar Transition Lifts", [f"{item.transition}: MeanLift={item.mean_lift:.4f}" for item in sorted([value for value in transitions if value.lag == lag], key=lambda value: -value.mean_lift)])
    append_ranked(lines, "Most Common Family Pathways", [f"{item.pathway}: Length={item.length}, AggregateCount={sum(value.count for value in item.values.values())}, MeanOER={item.mean_oer:.4f}" for item in sorted(eligible_paths, key=lambda value: -sum(item.count for item in value.values.values()))])
    append_ranked(lines, "Highest Continuation Family Pathways", [f"{item.pathway}: Length={item.length}, MeanContinuation={item.mean_continuation:.2%}, MeanOER={item.mean_oer:.4f}" for item in sorted(eligible_paths, key=lambda value: -value.mean_continuation)])
    append_ranked(lines, "Highest Failure Family Pathways", [f"{item.pathway}: Length={item.length}, MeanFailure={item.mean_failure:.2%}, MeanOER={item.mean_oer:.4f}" for item in sorted(eligible_paths, key=lambda value: -value.mean_failure)])
    append_ranked(lines, "Best Constructive/Destructive Transitions", [f"{transition}: MeanContinuation={mean([item.outcome.continuation_rate for item in values]):.2%}, MeanFailure={mean([item.outcome.failure_rate for item in values]):.2%}" for transition, values in sorted(class_values.items(), key=lambda pair: -mean([item.outcome.continuation_rate for item in pair[1]]))])
    for family in ("A", "B", "C", "D"):
        append_ranked(lines, f"Strongest Context Predictors of Family {family}", [f"{item.feature}: MeanEffect={item.mean_effect:.6f}, PositiveAgreement={item.positive_count}, NegativeAgreement={item.negative_count}" for item in sorted([value for value in eligible_contexts if value.family == family], key=lambda value: -abs(value.mean_effect))])
    append_ranked(lines, "Population-B Differences", [f"Family {item.family} | {item.feature}: PopulationBMeanEffect={item.mean_effect:.6f}, DeltaVsFull={delta:.6f}" for item, delta in sorted(population_b_differences, key=lambda pair: -abs(pair[1]))])
    lines.extend(["\nResearch Notes", "=============="])
    if eligible_outcomes:
        lines.append(f"- Highest aggregate continuation family: {max(eligible_outcomes, key=lambda item: item.mean_continuation).family}.")
        lines.append(f"- Lowest aggregate continuation family: {min(eligible_outcomes, key=lambda item: item.mean_continuation).family}.")
    lines.append("- Family transitions, pathways, and context signatures are descriptive research projections.")
    lines.append("- Population B summaries reuse the existing mature aligned lateral DissipationContained selector.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    studies = [study_instrument(path) for path in args.input_paths]
    seen = set()
    for study in studies:
        if study.instrument in seen:
            raise ValueError(f"Duplicate instrument input: {study.instrument}")
        seen.add(study.instrument)
        write_text(Path("Evidence") / "Output" / study.instrument / f"FamilyEvolution_{study.instrument}.txt", instrument_report(study))
    report = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, report)
    print(report, end="")


if __name__ == "__main__":
    main()
