"""APVA Trajectory Study v0.1.

Research-only analysis of rolling paths over the existing Study 30 families.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from APVA_AcceptanceAnatomy_23 import mature_aligned_lateral_indexes
from APVA_BreakoutContext_08 import EvidenceBar, direction_relative_return, load_rows
from APVA_FamilyEvolution_30 import FAMILIES, FAMILY_CLASS, family_for, instrument_columns
from APVA_RegimeTransition_16 import instrument_name, write_text
from APVA_StateGrammar_25 import OutcomeStats, format_oer, mean, summarize


AGGREGATE_OUTPUT = Path("Evidence/Output/Trajectory/Trajectory_All.txt")
PATH_LENGTHS = (3, 4, 5)
BRANCH_LENGTHS = (2, 3)
PER_INSTRUMENT_MIN_COUNT = 20
AGGREGATE_MIN_COUNT = 50
TOP_LIMIT = 25


@dataclass(frozen=True)
class StabilityStats:
    count: int
    repeat10: float
    repeat20: float
    return_last10: float
    return_last20: float


@dataclass(frozen=True)
class PathStats:
    path: str
    length: int
    count: int
    frequency: float
    expected_count: float
    oer: float | None
    outcome: OutcomeStats
    anchors: tuple[int, ...]
    stability: StabilityStats


@dataclass(frozen=True)
class BranchStats:
    pattern: str
    family: str
    count: int
    probability: float
    baseline_frequency: float
    lift: float | None


@dataclass(frozen=True)
class ClassPathStats:
    path: str
    length: int
    count: int
    expected_count: float
    oer: float | None
    outcome: OutcomeStats


@dataclass(frozen=True)
class PopulationStudy:
    name: str
    indexes: list[int]
    families: dict[int, str]
    counts: dict[str, int]
    frequencies: dict[str, float]
    windows: dict[int, int]
    paths: dict[tuple[int, str], PathStats]
    divergence: dict[tuple[str, str], BranchStats]
    convergence: dict[tuple[str, str], BranchStats]
    class_paths: dict[tuple[int, str], ClassPathStats]


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    rows: list[EvidenceBar]
    populations: dict[str, PopulationStudy]


@dataclass(frozen=True)
class AggregatePath:
    path: str
    length: int
    values: dict[str, PathStats]
    valid_instruments: int
    positive_count: int
    negative_count: int
    mean_continuation: float
    mean_failure: float
    mean_oer: float
    mean_repeat10: float
    mean_repeat20: float
    mean_return_last10: float
    mean_return_last20: float


@dataclass(frozen=True)
class AggregateBranch:
    pattern: str
    family: str
    values: dict[str, BranchStats]
    valid_instruments: int
    mean_probability: float
    mean_lift: float


@dataclass(frozen=True)
class AggregateClassPath:
    path: str
    length: int
    values: dict[str, ClassPathStats]
    valid_instruments: int
    mean_continuation: float
    mean_failure: float
    mean_oer: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study rolling trajectories over the Study 30 APVA families.")
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def sequence(mapping: dict[int, str], anchor: int, length: int) -> str:
    return "->".join(mapping[index] for index in range(anchor - length + 1, anchor + 1))


def contiguous_anchors(allowed: set[int], length: int, future: int = 0) -> list[int]:
    return [
        anchor
        for anchor in sorted(allowed)
        if all(anchor - offset in allowed for offset in range(length))
        and all(anchor + offset in allowed for offset in range(1, future + 1))
    ]


def probabilities(mapping: dict[int, str]) -> tuple[dict[str, int], dict[str, float]]:
    counts = {family: sum(value == family for value in mapping.values()) for family in FAMILIES}
    denominator = len(mapping) if mapping else 1
    return counts, {family: counts[family] / denominator for family in FAMILIES}


def expected_count(path: str, total: int, frequencies: dict[str, float]) -> float:
    probability = 1.0
    for family in path.split("->"):
        probability *= frequencies[family]
    return total * probability


def stability(mapping: dict[int, str], anchors: list[int], path: str, length: int) -> StabilityStats:
    anchor_set = set(anchors)
    last_family = path.split("->")[-1]
    denominator = len(anchors) if anchors else 1
    repeat10 = 0
    repeat20 = 0
    return10 = 0
    return20 = 0
    for anchor in anchors:
        repeat_offsets = [
            offset
            for offset in range(1, 21)
            if anchor + offset in anchor_set and sequence(mapping, anchor + offset, length) == path
        ]
        return_offsets = [offset for offset in range(1, 21) if mapping.get(anchor + offset) == last_family]
        repeat10 += any(offset <= 10 for offset in repeat_offsets)
        repeat20 += bool(repeat_offsets)
        return10 += any(offset <= 10 for offset in return_offsets)
        return20 += bool(return_offsets)
    return StabilityStats(len(anchors), repeat10 / denominator, repeat20 / denominator, return10 / denominator, return20 / denominator)


def build_paths(rows: list[EvidenceBar], mapping: dict[int, str], frequencies: dict[str, float]) -> tuple[dict[int, int], dict[tuple[int, str], PathStats]]:
    allowed = set(mapping)
    windows = {}
    output = {}
    for length in PATH_LENGTHS:
        anchors = contiguous_anchors(allowed, length)
        windows[length] = len(anchors)
        grouped = {}
        values = {}
        for anchor in anchors:
            path = sequence(mapping, anchor, length)
            grouped.setdefault(path, []).append(anchor)
            outcome = direction_relative_return(rows, anchor, 5)
            if outcome is not None:
                values.setdefault(path, []).append(outcome)
        for path, path_anchors in grouped.items():
            expected = expected_count(path, len(anchors), frequencies)
            output[(length, path)] = PathStats(
                path,
                length,
                len(path_anchors),
                len(path_anchors) / len(anchors) if anchors else 0.0,
                expected,
                len(path_anchors) / expected if expected else None,
                summarize(values.get(path, [])),
                tuple(path_anchors),
                stability(mapping, path_anchors, path, length),
            )
    return windows, output


def build_branches(mapping: dict[int, str], frequencies: dict[str, float], convergence: bool) -> dict[tuple[str, str], BranchStats]:
    allowed = set(mapping)
    output = {}
    for length in BRANCH_LENGTHS:
        grouped = {}
        anchors = contiguous_anchors(allowed, length, future=0)
        for anchor in anchors:
            if convergence:
                prior = anchor - length
                if prior not in allowed:
                    continue
                pattern = sequence(mapping, anchor, length)
                family = mapping[prior]
            else:
                future = anchor + 1
                if future not in allowed:
                    continue
                pattern = sequence(mapping, anchor, length)
                family = mapping[future]
            grouped.setdefault(pattern, []).append(family)
        for pattern, families in grouped.items():
            denominator = len(families)
            for family in FAMILIES:
                count = sum(value == family for value in families)
                probability = count / denominator if denominator else 0.0
                baseline = frequencies[family]
                output[(pattern, family)] = BranchStats(pattern, family, count, probability, baseline, probability / baseline if baseline else None)
    return output


def class_frequencies(mapping: dict[int, str]) -> dict[str, float]:
    classes = ("Constructive", "Destructive", "Neutral")
    denominator = len(mapping) if mapping else 1
    return {name: sum(FAMILY_CLASS[value] == name for value in mapping.values()) / denominator for name in classes}


def class_path(path: str) -> str:
    return "->".join(FAMILY_CLASS[family] for family in path.split("->"))


def build_class_paths(rows: list[EvidenceBar], paths: dict[tuple[int, str], PathStats], mapping: dict[int, str]) -> dict[tuple[int, str], ClassPathStats]:
    frequencies = class_frequencies(mapping)
    grouped = {}
    for item in paths.values():
        key = (item.length, class_path(item.path))
        grouped.setdefault(key, []).extend(item.anchors)
    output = {}
    for (length, path), anchors in grouped.items():
        expected = expected_count(path, sum(len(item.anchors) for item in paths.values() if item.length == length), frequencies)
        values = [value for anchor in anchors if (value := direction_relative_return(rows, anchor, 5)) is not None]
        output[(length, path)] = ClassPathStats(path, length, len(anchors), expected, len(anchors) / expected if expected else None, summarize(values))
    return output


def build_population(rows: list[EvidenceBar], name: str, indexes: list[int]) -> PopulationStudy:
    mapping = {index: family_for(rows[index]) for index in indexes}
    counts, frequencies = probabilities(mapping)
    windows, paths = build_paths(rows, mapping, frequencies)
    return PopulationStudy(
        name,
        indexes,
        mapping,
        counts,
        frequencies,
        windows,
        paths,
        build_branches(mapping, frequencies, False),
        build_branches(mapping, frequencies, True),
        build_class_paths(rows, paths, mapping),
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


def append_path_frequency(lines: list[str], population: PopulationStudy, minimum: int = PER_INSTRUMENT_MIN_COUNT) -> None:
    lines.append(f"{'Path':<34} {'Len':>3} {'Count':>8} {'Frequency':>11} {'Expected':>11} {'OER':>9}")
    items = sorted(population.paths.values(), key=lambda item: (item.length, -item.count, item.path))
    for item in items:
        if item.count < minimum:
            continue
        lines.append(f"{item.path:<34} {item.length:>3} {item.count:>8} {item.frequency:>10.2%} {item.expected_count:>11.4f} {format_oer(item.oer):>9}")


def append_path_outcomes(lines: list[str], population: PopulationStudy, minimum: int = PER_INSTRUMENT_MIN_COUNT) -> None:
    lines.append(f"{'Path':<34} {'Len':>3} {'Count':>8} {'OutcomeN':>9} {'Mean':>11} {'Median':>11} {'Cont':>9} {'Fail':>9} {'Flat':>9}")
    items = sorted(population.paths.values(), key=lambda item: (item.length, -item.count, item.path))
    for item in items:
        if item.count < minimum:
            continue
        value = item.outcome
        lines.append(f"{item.path:<34} {item.length:>3} {item.count:>8} {value.count:>9} {value.mean:>11.6f} {value.median:>11.6f} {value.continuation_rate:>8.2%} {value.failure_rate:>8.2%} {value.flat_rate:>8.2%}")


def append_ranked_paths(lines: list[str], title: str, population: PopulationStudy, key, reverse: bool = True) -> None:
    lines.extend([f"\n{title}", "." * len(title)])
    items = [item for item in population.paths.values() if item.count >= PER_INSTRUMENT_MIN_COUNT]
    for rank, item in enumerate(sorted(items, key=key, reverse=reverse)[:TOP_LIMIT], start=1):
        lines.append(f"{rank:>3}. {item.path:<34} Len={item.length} Count={item.count:>6} OER={format_oer(item.oer):>8} Cont={item.outcome.continuation_rate:>7.2%} Fail={item.outcome.failure_rate:>7.2%}")


def append_branches(lines: list[str], values: dict[tuple[str, str], BranchStats]) -> None:
    lines.append(f"{'Pattern':<22} {'Family':<8} {'Count':>8} {'Probability':>12} {'Baseline':>10} {'Lift':>9}")
    for item in sorted(values.values(), key=lambda value: (len(value.pattern.split("->")), value.pattern, value.family)):
        lines.append(f"{item.pattern:<22} {item.family:<8} {item.count:>8} {item.probability:>11.2%} {item.baseline_frequency:>9.2%} {format_oer(item.lift):>9}")


def append_stability(lines: list[str], population: PopulationStudy, minimum: int = PER_INSTRUMENT_MIN_COUNT) -> None:
    lines.append(f"{'Path':<34} {'Len':>3} {'Count':>8} {'Repeat10':>10} {'Repeat20':>10} {'Return10':>10} {'Return20':>10}")
    for item in sorted(population.paths.values(), key=lambda value: (value.length, -value.count, value.path)):
        if item.count < minimum:
            continue
        value = item.stability
        lines.append(f"{item.path:<34} {item.length:>3} {item.count:>8} {value.repeat10:>9.2%} {value.repeat20:>9.2%} {value.return_last10:>9.2%} {value.return_last20:>9.2%}")


def append_class_paths(lines: list[str], population: PopulationStudy, minimum: int = PER_INSTRUMENT_MIN_COUNT) -> None:
    lines.append(f"{'ClassPath':<58} {'Len':>3} {'Count':>8} {'OER':>9} {'Cont':>9} {'Fail':>9}")
    for item in sorted(population.class_paths.values(), key=lambda value: (value.length, -value.count, value.path)):
        if item.count < minimum:
            continue
        lines.append(f"{item.path:<58} {item.length:>3} {item.count:>8} {format_oer(item.oer):>9} {item.outcome.continuation_rate:>8.2%} {item.outcome.failure_rate:>8.2%}")


def instrument_report(study: InstrumentStudy) -> str:
    full = study.populations["Full Population"]
    population_b = study.populations["Population B"]
    lines = [
        f"APVA Trajectory Study v0.1 - {study.instrument}",
        "=" * (29 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {len(study.rows)}",
        f"Valid polarity rows: {sum(row.polarity in {'Black', 'Red'} for row in study.rows)}",
        f"Family counts: {full.counts}",
        f"Total path windows by length: {full.windows}",
        "Study 30 family precedence reused exactly: A, then B, then C, then D, then N.",
        "",
        "Section 1 - Family Path Tables",
        "==============================",
    ]
    append_path_frequency(lines, full)
    lines.extend(["\nSection 2 - Outcome Layer by Path", "================================="])
    append_path_outcomes(lines, full)
    append_ranked_paths(lines, "Section 3 - Most Common Paths", full, lambda item: (item.count, item.path))
    append_ranked_paths(lines, "Section 4 - Most Overrepresented Paths", full, lambda item: (item.oer or 0.0, item.path))
    append_ranked_paths(lines, "Section 5 - Highest Continuation Paths", full, lambda item: (item.outcome.continuation_rate, item.path))
    append_ranked_paths(lines, "Section 6 - Highest Failure Paths", full, lambda item: (item.outcome.failure_rate, item.path))
    lines.extend(["\nSection 7 - Path Divergence Tables", "=================================="])
    append_branches(lines, full.divergence)
    lines.extend(["\nSection 8 - Path Convergence Tables", "==================================="])
    append_branches(lines, full.convergence)
    lines.extend(["\nSection 9 - Path Stability", "=========================="])
    append_stability(lines, full)
    lines.extend(["\nSection 10 - Constructive / Destructive Path Classes", "===================================================="])
    append_class_paths(lines, full)
    lines.extend(["\nSection 11 - Population B Trajectory Summary", "============================================"])
    lines.append(f"Population B rows: {len(population_b.indexes)}")
    lines.append(f"Population B path windows by length: {population_b.windows}")
    lines.extend(["\nPopulation B Path Frequencies", "-----------------------------"])
    append_path_frequency(lines, population_b)
    lines.extend(["\nPopulation B Path Outcomes", "--------------------------"])
    append_path_outcomes(lines, population_b)
    lines.extend(["\nPopulation B Divergence", "-----------------------"])
    append_branches(lines, population_b.divergence)
    lines.extend(["\nPopulation B Convergence", "------------------------"])
    append_branches(lines, population_b.convergence)
    lines.extend(["\nPopulation B Constructive / Destructive Paths", "---------------------------------------------"])
    append_class_paths(lines, population_b)
    lines.extend(["\nSection 12 - Mechanical Research Notes", "======================================"])
    eligible = [item for item in full.paths.values() if item.count >= PER_INSTRUMENT_MIN_COUNT]
    common = max(eligible, key=lambda item: item.count, default=None)
    overrepresented = max(eligible, key=lambda item: item.oer or 0.0, default=None)
    lines.append(f"- Most common reported path: {common.path if common else 'N/A'}.")
    lines.append(f"- Most overrepresented reported path: {overrepresented.path if overrepresented else 'N/A'}.")
    lines.append(f"- Population B contains {len(population_b.indexes)} rows and preserves original-bar contiguity.")
    lines.append("- Paths, branching, convergence, stability, and class paths are descriptive only.")
    return "\n".join(lines) + "\n"


def aggregate_paths(studies: list[InstrumentStudy], population: str = "Full Population") -> list[AggregatePath]:
    keys = sorted(set().union(*(study.populations[population].paths for study in studies)))
    output = []
    for length, path in keys:
        values = {study.instrument: study.populations[population].paths[(length, path)] for study in studies if (length, path) in study.populations[population].paths}
        if sum(item.count for item in values.values()) < AGGREGATE_MIN_COUNT:
            continue
        valid = [item for item in values.values() if item.count >= PER_INSTRUMENT_MIN_COUNT]
        output.append(AggregatePath(path, length, values, len(valid), sum(item.outcome.continuation_rate > 0.5 for item in valid), sum(item.outcome.continuation_rate < 0.5 for item in valid), mean([item.outcome.continuation_rate for item in valid]), mean([item.outcome.failure_rate for item in valid]), mean([item.oer for item in valid if item.oer is not None]), mean([item.stability.repeat10 for item in valid]), mean([item.stability.repeat20 for item in valid]), mean([item.stability.return_last10 for item in valid]), mean([item.stability.return_last20 for item in valid])))
    return output


def aggregate_branches(studies: list[InstrumentStudy], attribute: str) -> list[AggregateBranch]:
    keys = sorted(set().union(*(getattr(study.populations["Full Population"], attribute) for study in studies)))
    output = []
    for pattern, family in keys:
        values = {study.instrument: getattr(study.populations["Full Population"], attribute)[(pattern, family)] for study in studies if (pattern, family) in getattr(study.populations["Full Population"], attribute)}
        valid = [item for item in values.values() if item.count >= PER_INSTRUMENT_MIN_COUNT]
        output.append(AggregateBranch(pattern, family, values, len(valid), mean([item.probability for item in valid]), mean([item.lift for item in valid if item.lift is not None])))
    return output


def aggregate_class_paths(studies: list[InstrumentStudy]) -> list[AggregateClassPath]:
    keys = sorted(set().union(*(study.populations["Full Population"].class_paths for study in studies)))
    output = []
    for length, path in keys:
        values = {study.instrument: study.populations["Full Population"].class_paths[(length, path)] for study in studies if (length, path) in study.populations["Full Population"].class_paths}
        valid = [item for item in values.values() if item.count >= PER_INSTRUMENT_MIN_COUNT]
        output.append(AggregateClassPath(path, length, values, len(valid), mean([item.outcome.continuation_rate for item in valid]), mean([item.outcome.failure_rate for item in valid]), mean([item.oer for item in valid if item.oer is not None])))
    return output


def append_aggregate_paths(lines: list[str], items: list[AggregatePath], columns: list[str]) -> None:
    lines.append(f"{'Path':<34} {'Len':>3}" + "".join(f" {name + '_N':>7} {name + '_Cont':>9} {name + '_Fail':>9} {name + '_OER':>8}" for name in columns) + " ValidN PosN NegN MeanCont MeanFail MeanOER")
    for item in items:
        cells = []
        for name in columns:
            value = item.values.get(name)
            cells.append(f" {value.count if value else 0:>7} {value.outcome.continuation_rate if value else 0.0:>8.2%} {value.outcome.failure_rate if value else 0.0:>8.2%} {value.oer if value and value.oer is not None else 0.0:>8.3f}")
        lines.append(f"{item.path:<34} {item.length:>3}{''.join(cells)} {item.valid_instruments:>6} {item.positive_count:>4} {item.negative_count:>4} {item.mean_continuation:>8.2%} {item.mean_failure:>8.2%} {item.mean_oer:>7.3f}")


def append_aggregate_branches(lines: list[str], items: list[AggregateBranch], columns: list[str]) -> None:
    lines.append(f"{'Pattern':<22} {'Family':<8}" + "".join(f" {name + '_N':>7} {name + '_Prob':>9} {name + '_Lift':>9}" for name in columns) + " ValidN MeanProb MeanLift")
    for item in items:
        cells = []
        for name in columns:
            value = item.values.get(name)
            cells.append(f" {value.count if value else 0:>7} {value.probability if value else 0.0:>8.2%} {value.lift if value and value.lift is not None else 0.0:>9.3f}")
        lines.append(f"{item.pattern:<22} {item.family:<8}{''.join(cells)} {item.valid_instruments:>6} {item.mean_probability:>8.2%} {item.mean_lift:>8.3f}")


def append_aggregate_stability(lines: list[str], items: list[AggregatePath], columns: list[str]) -> None:
    lines.append(f"{'Path':<34} {'Len':>3}" + "".join(f" {name + '_R10':>8} {name + '_R20':>8} {name + '_L10':>8} {name + '_L20':>8}" for name in columns) + " MeanR10 MeanR20 MeanL10 MeanL20")
    for item in items:
        cells = []
        for name in columns:
            value = item.values.get(name)
            stats = value.stability if value else StabilityStats(0, 0.0, 0.0, 0.0, 0.0)
            cells.append(f" {stats.repeat10:>7.2%} {stats.repeat20:>7.2%} {stats.return_last10:>7.2%} {stats.return_last20:>7.2%}")
        lines.append(f"{item.path:<34} {item.length:>3}{''.join(cells)} {item.mean_repeat10:>7.2%} {item.mean_repeat20:>7.2%} {item.mean_return_last10:>7.2%} {item.mean_return_last20:>7.2%}")


def append_aggregate_class_paths(lines: list[str], items: list[AggregateClassPath], columns: list[str]) -> None:
    lines.append(f"{'ClassPath':<58} {'Len':>3}" + "".join(f" {name + '_N':>7} {name + '_Cont':>9} {name + '_Fail':>9} {name + '_OER':>8}" for name in columns) + " ValidN MeanCont MeanFail MeanOER")
    for item in items:
        cells = []
        for name in columns:
            value = item.values.get(name)
            cells.append(f" {value.count if value else 0:>7} {value.outcome.continuation_rate if value else 0.0:>8.2%} {value.outcome.failure_rate if value else 0.0:>8.2%} {value.oer if value and value.oer is not None else 0.0:>8.3f}")
        lines.append(f"{item.path:<58} {item.length:>3}{''.join(cells)} {item.valid_instruments:>6} {item.mean_continuation:>8.2%} {item.mean_failure:>8.2%} {item.mean_oer:>7.3f}")


def append_ranked(lines: list[str], title: str, items: list, key, formatter, reverse: bool = True) -> None:
    lines.extend([f"\n{title}", "." * len(title)])
    eligible = [item for item in items if item.valid_instruments >= 2]
    if not eligible:
        lines.append("No items met the two-instrument minimum.")
        return
    for rank, item in enumerate(sorted(eligible, key=key, reverse=reverse)[:TOP_LIMIT], start=1):
        lines.append(f"{rank:>3}. {formatter(item)}")


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    columns = instrument_columns(studies)
    paths = aggregate_paths(studies)
    divergence = aggregate_branches(studies, "divergence")
    convergence = aggregate_branches(studies, "convergence")
    classes = aggregate_class_paths(studies)
    population_b = aggregate_paths(studies, "Population B")
    lines = [
        "APVA Trajectory Study v0.1 - Aggregate",
        "======================================",
        f"Instruments: {', '.join(study.instrument for study in studies)}",
        "Study 30 family precedence reused exactly: A, then B, then C, then D, then N.",
        "",
        "Aggregate Path Table",
        "====================",
    ]
    append_aggregate_paths(lines, paths, columns)
    lines.extend(["\nAggregate Divergence Table", "=========================="])
    append_aggregate_branches(lines, divergence, columns)
    lines.extend(["\nAggregate Convergence Table", "==========================="])
    append_aggregate_branches(lines, convergence, columns)
    lines.extend(["\nAggregate Stability Table", "========================="])
    append_aggregate_stability(lines, paths, columns)
    lines.extend(["\nAggregate Constructive / Destructive Path Table", "==============================================="])
    append_aggregate_class_paths(lines, classes, columns)
    lines.extend(["\nAggregate Rankings", "=================="])
    append_ranked(lines, "1. Most common replicated paths", paths, lambda item: (sum(value.count for value in item.values.values()), item.path), lambda item: f"{item.path:<34} Len={item.length} TotalN={sum(value.count for value in item.values.values())} ValidN={item.valid_instruments}")
    append_ranked(lines, "2. Most overrepresented replicated paths", paths, lambda item: (item.mean_oer, item.path), lambda item: f"{item.path:<34} Len={item.length} MeanOER={item.mean_oer:.4f} ValidN={item.valid_instruments}")
    append_ranked(lines, "3. Highest continuation paths", paths, lambda item: (item.mean_continuation, item.path), lambda item: f"{item.path:<34} MeanCont={item.mean_continuation:.2%} MeanFail={item.mean_failure:.2%} ValidN={item.valid_instruments}")
    append_ranked(lines, "4. Highest failure paths", paths, lambda item: (item.mean_failure, item.path), lambda item: f"{item.path:<34} MeanFail={item.mean_failure:.2%} MeanCont={item.mean_continuation:.2%} ValidN={item.valid_instruments}")
    append_ranked(lines, "5. Strongest path divergences", divergence, lambda item: (item.mean_lift, item.pattern, item.family), lambda item: f"{item.pattern}->{item.family:<8} MeanProb={item.mean_probability:.2%} MeanLift={item.mean_lift:.4f} ValidN={item.valid_instruments}")
    append_ranked(lines, "6. Strongest path convergences", convergence, lambda item: (item.mean_lift, item.pattern, item.family), lambda item: f"{item.family}->{item.pattern:<22} MeanProb={item.mean_probability:.2%} MeanLift={item.mean_lift:.4f} ValidN={item.valid_instruments}")
    append_ranked(lines, "7. Most stable paths", paths, lambda item: (item.mean_repeat20, item.path), lambda item: f"{item.path:<34} Repeat10={item.mean_repeat10:.2%} Repeat20={item.mean_repeat20:.2%} ValidN={item.valid_instruments}")
    append_ranked(lines, "8. Highest return-to-last-family paths", paths, lambda item: (item.mean_return_last20, item.path), lambda item: f"{item.path:<34} Return10={item.mean_return_last10:.2%} Return20={item.mean_return_last20:.2%} ValidN={item.valid_instruments}")
    append_ranked(lines, "9. Best constructive / destructive paths", classes, lambda item: (item.mean_continuation, item.path), lambda item: f"{item.path:<58} MeanCont={item.mean_continuation:.2%} MeanOER={item.mean_oer:.4f} ValidN={item.valid_instruments}")
    append_ranked(lines, "10. Worst constructive / destructive paths", classes, lambda item: (item.mean_failure, item.path), lambda item: f"{item.path:<58} MeanFail={item.mean_failure:.2%} MeanOER={item.mean_oer:.4f} ValidN={item.valid_instruments}")
    full_by_key = {(item.length, item.path): item for item in paths}
    differences = []
    for item in population_b:
        baseline = full_by_key.get((item.length, item.path))
        if baseline and item.valid_instruments >= 2 and baseline.valid_instruments >= 2:
            differences.append((item.path, item.length, item.mean_continuation - baseline.mean_continuation, item.valid_instruments))
    lines.extend(["\n11. Population-B strongest trajectory differences", "." * 48])
    if differences:
        for path, length, delta, valid in sorted(differences, key=lambda item: (-abs(item[2]), item[0]))[:TOP_LIMIT]:
            lines.append(f"{path:<34} Len={length} DeltaMeanContinuation={delta:>9.2%} PopulationBValidN={valid}")
    else:
        lines.append("No Population B paths met the replicated comparison minimum.")
    lines.extend(["\nCross-Instrument Mechanical Research Notes", "=========================================="])
    common = max((item for item in paths if item.valid_instruments >= 2), key=lambda item: sum(value.count for value in item.values.values()), default=None)
    strongest = max((item for item in paths if item.valid_instruments >= 2), key=lambda item: item.mean_oer, default=None)
    lines.append(f"- Most common replicated path: {common.path if common else 'N/A'}.")
    lines.append(f"- Most overrepresented replicated path: {strongest.path if strongest else 'N/A'}.")
    lines.append("- Divergence and convergence lifts use unconditional family frequencies as the baseline.")
    lines.append("- Population B paths preserve original-bar contiguity and are reported only when replicated thresholds are met.")
    lines.append("- Trajectory statistics are descriptive only.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    studies = [study_instrument(path) for path in args.input_paths]
    seen = set()
    for study in studies:
        if study.instrument in seen:
            raise ValueError(f"Duplicate instrument input: {study.instrument}")
        seen.add(study.instrument)
        write_text(Path("Evidence") / "Output" / study.instrument / f"Trajectory_{study.instrument}.txt", instrument_report(study))
    report = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, report)
    print(report, end="")


if __name__ == "__main__":
    main()
