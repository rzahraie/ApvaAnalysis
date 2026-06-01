"""APVA Narrative Archetype Study v0.1.

Research-only mechanical grouping of existing Study 34 family trajectories.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from APVA_BreakoutContext_08 import EvidenceBar, direction_relative_return, load_rows
from APVA_FamilyEvolution_30 import instrument_columns
from APVA_RegimeTransition_16 import instrument_name, write_text
from APVA_StateGrammar_25 import OutcomeStats, format_oer, mean, summarize
from APVA_Trajectory_34 import (
    AGGREGATE_MIN_COUNT,
    PER_INSTRUMENT_MIN_COUNT,
    PathStats,
    PopulationStudy as TrajectoryPopulation,
    build_population as build_trajectory_population,
)
from APVA_AcceptanceAnatomy_23 import mature_aligned_lateral_indexes


AGGREGATE_OUTPUT = Path("Evidence/Output/Archetypes/Archetypes_All.txt")
TOP_LIMIT = 20
ARCHETYPES = (
    "Recovery",
    "Exhaustion",
    "Reassertion",
    "Decay",
    "Compression Resolution",
    "Destructive Persistence",
    "Constructive Emergence",
    "Neutral Drift",
    "Unclassified",
)


@dataclass(frozen=True)
class ArchetypeStats:
    name: str
    member_path_count: int
    total_occurrences: int
    mean_oer: float
    outcome: OutcomeStats
    members: tuple[PathStats, ...]


@dataclass(frozen=True)
class StabilityStats:
    count: int
    repeat10: float
    repeat20: float
    return10: float
    return20: float


@dataclass(frozen=True)
class TransitionStats:
    source: str
    target: str
    count: int
    probability: float
    lift: float | None
    outcome: OutcomeStats


@dataclass(frozen=True)
class BranchStats:
    source: str
    target: str
    count: int
    probability: float
    lift: float | None


@dataclass(frozen=True)
class ArchetypePopulation:
    name: str
    trajectory: TrajectoryPopulation
    primary_by_anchor: dict[int, str]
    membership: dict[str, ArchetypeStats]
    transitions: dict[tuple[str, str], TransitionStats]
    stability: dict[str, StabilityStats]
    divergence: dict[tuple[str, str], BranchStats]
    convergence: dict[tuple[str, str], BranchStats]


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    rows: list[EvidenceBar]
    populations: dict[str, ArchetypePopulation]


@dataclass(frozen=True)
class AggregateArchetype:
    name: str
    values: dict[str, ArchetypeStats]
    valid_instruments: int
    positive_count: int
    negative_count: int
    mean_continuation: float
    mean_failure: float
    mean_oer: float


@dataclass(frozen=True)
class AggregateTransition:
    source: str
    target: str
    values: dict[str, TransitionStats]
    valid_instruments: int
    mean_probability: float
    mean_lift: float
    mean_continuation: float
    mean_failure: float


@dataclass(frozen=True)
class AggregateStability:
    name: str
    values: dict[str, StabilityStats]
    valid_instruments: int
    mean_repeat10: float
    mean_repeat20: float
    mean_return10: float
    mean_return20: float


@dataclass(frozen=True)
class AggregateBranch:
    source: str
    target: str
    values: dict[str, BranchStats]
    valid_instruments: int
    mean_probability: float
    mean_lift: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group APVA Study 34 trajectories into mechanical narrative archetypes.")
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def tokens(path: str) -> list[str]:
    return path.split("->")


def dominated_by_dn(values: list[str]) -> bool:
    return sum(value in {"D", "N"} for value in values) > len(values) / 2


def archetypes_for(path: str) -> tuple[str, ...]:
    values = tokens(path)
    joined = "->".join(values)
    matches = []
    if joined.endswith("C->B") or "C->C->B" in joined:
        matches.append("Recovery")
    if "C->C->C" in joined or joined.endswith("C->C"):
        matches.append("Exhaustion")
    if dominated_by_dn(values) and (values[-1] == "D" or joined.endswith("N->D")):
        matches.append("Reassertion")
    if len(values) >= 2 and values[-1] == "N" and values[-2] in {"B", "C"}:
        matches.append("Decay")
    if any(values[index] == "B" and values[index + 1] in {"N", "C"} for index in range(len(values) - 1)):
        matches.append("Compression Resolution")
    if any(values[index] == values[index + 1] and values[index] in {"C", "D"} for index in range(len(values) - 1)):
        matches.append("Destructive Persistence")
    if len(values) >= 2 and values[-1] in {"A", "B"} and values[-2] in {"N", "D"}:
        matches.append("Constructive Emergence")
    if values.count("N") > len(values) / 2:
        matches.append("Neutral Drift")
    return tuple(matches or ["Unclassified"])


def primary_archetype(path: str) -> str:
    matches = set(archetypes_for(path))
    return next(name for name in ARCHETYPES if name in matches)


def anchor_paths(trajectory: TrajectoryPopulation) -> dict[int, str]:
    output = {}
    for item in trajectory.paths.values():
        for anchor in item.anchors:
            current = output.get(anchor)
            if current is None or len(item.path.split("->")) > len(current.split("->")):
                output[anchor] = item.path
    return output


def archetype_membership_with_rows(rows: list[EvidenceBar], trajectory: TrajectoryPopulation) -> dict[str, ArchetypeStats]:
    grouped = {name: [] for name in ARCHETYPES}
    for item in trajectory.paths.values():
        for name in archetypes_for(item.path):
            grouped[name].append(item)
    output = {}
    for name, members in grouped.items():
        values = [
            value
            for item in members
            for anchor in item.anchors
            if (value := direction_relative_return(rows, anchor, 5)) is not None
        ]
        output[name] = ArchetypeStats(name, len(members), sum(item.count for item in members), mean([item.oer for item in members if item.oer is not None]), summarize(values), tuple(members))
    return output


def archetype_probabilities(primary: dict[int, str]) -> dict[str, float]:
    denominator = len(primary) if primary else 1
    return {name: sum(value == name for value in primary.values()) / denominator for name in ARCHETYPES}


def build_transitions(rows: list[EvidenceBar], primary: dict[int, str], probabilities: dict[str, float]) -> dict[tuple[str, str], TransitionStats]:
    counts = {}
    values = {}
    for index, target in primary.items():
        source = primary.get(index - 1)
        if source is None:
            continue
        key = (source, target)
        counts[key] = counts.get(key, 0) + 1
        outcome = direction_relative_return(rows, index, 5)
        if outcome is not None:
            values.setdefault(key, []).append(outcome)
    total = sum(counts.values())
    output = {}
    for source in ARCHETYPES:
        source_total = sum(counts.get((source, target), 0) for target in ARCHETYPES)
        for target in ARCHETYPES:
            count = counts.get((source, target), 0)
            expected = total * probabilities[source] * probabilities[target]
            output[(source, target)] = TransitionStats(source, target, count, count / source_total if source_total else 0.0, count / expected if expected else None, summarize(values.get((source, target), [])))
    return output


def build_stability(primary: dict[int, str]) -> dict[str, StabilityStats]:
    output = {}
    for name in ARCHETYPES:
        anchors = [index for index, value in primary.items() if value == name]
        denominator = len(anchors) if anchors else 1
        repeat10 = repeat20 = return10 = return20 = 0
        for anchor in anchors:
            offsets = [offset for offset in range(1, 21) if primary.get(anchor + offset) == name]
            repeat10 += any(offset <= 10 for offset in offsets)
            repeat20 += bool(offsets)
            return10 += any(offset <= 10 for offset in offsets)
            return20 += bool(offsets)
        output[name] = StabilityStats(len(anchors), repeat10 / denominator, repeat20 / denominator, return10 / denominator, return20 / denominator)
    return output


def build_branches(primary: dict[int, str], probabilities: dict[str, float], convergence: bool) -> dict[tuple[str, str], BranchStats]:
    grouped = {}
    for index, current in primary.items():
        other = primary.get(index - 1 if convergence else index + 1)
        if other is not None:
            grouped.setdefault(current, []).append(other)
    output = {}
    for current in ARCHETYPES:
        values = grouped.get(current, [])
        for other in ARCHETYPES:
            count = sum(value == other for value in values)
            probability = count / len(values) if values else 0.0
            baseline = probabilities[other]
            output[(current, other)] = BranchStats(current, other, count, probability, probability / baseline if baseline else None)
    return output


def build_population(rows: list[EvidenceBar], name: str, indexes: list[int]) -> ArchetypePopulation:
    trajectory = build_trajectory_population(rows, name, indexes)
    primary = {anchor: primary_archetype(path) for anchor, path in anchor_paths(trajectory).items()}
    probabilities = archetype_probabilities(primary)
    return ArchetypePopulation(name, trajectory, primary, archetype_membership_with_rows(rows, trajectory), build_transitions(rows, primary, probabilities), build_stability(primary), build_branches(primary, probabilities, False), build_branches(primary, probabilities, True))


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    return InstrumentStudy(instrument_name(path), path, rows, {"Full Population": build_population(rows, "Full Population", list(range(len(rows)))), "Population B": build_population(rows, "Population B", mature_aligned_lateral_indexes(rows))})


def append_path_summary(lines: list[str], population: ArchetypePopulation) -> None:
    lines.append(f"{'Path':<34} {'Len':>3} {'Count':>8} {'Frequency':>11} {'OER':>9} {'Cont':>9} {'Fail':>9} {'Mean':>11}")
    for item in sorted(population.trajectory.paths.values(), key=lambda value: (value.length, -value.count, value.path)):
        if item.count < PER_INSTRUMENT_MIN_COUNT:
            continue
        lines.append(f"{item.path:<34} {item.length:>3} {item.count:>8} {item.frequency:>10.2%} {format_oer(item.oer):>9} {item.outcome.continuation_rate:>8.2%} {item.outcome.failure_rate:>8.2%} {item.outcome.mean:>11.6f}")


def append_membership(lines: list[str], population: ArchetypePopulation) -> None:
    lines.append(f"{'Archetype':<28} {'MemberPaths':>11} {'Occurrences':>12} {'MeanOER':>9} {'Cont':>9} {'Fail':>9} {'Mean':>11}")
    for name in ARCHETYPES:
        item = population.membership[name]
        lines.append(f"{name:<28} {item.member_path_count:>11} {item.total_occurrences:>12} {item.mean_oer:>9.3f} {item.outcome.continuation_rate:>8.2%} {item.outcome.failure_rate:>8.2%} {item.outcome.mean:>11.6f}")


def append_member_rankings(lines: list[str], population: ArchetypePopulation) -> None:
    for name in ARCHETYPES:
        members = [item for item in population.membership[name].members if item.count >= PER_INSTRUMENT_MIN_COUNT]
        lines.extend([f"\n{name} Member Paths", "." * (len(name) + 13)])
        for title, key in (("Top by Count", lambda item: item.count), ("Top by OER", lambda item: item.oer or 0.0), ("Top by Continuation", lambda item: item.outcome.continuation_rate), ("Top by Failure", lambda item: item.outcome.failure_rate)):
            lines.append(title + ":")
            for item in sorted(members, key=key, reverse=True)[:5]:
                lines.append(f"  {item.path:<34} Count={item.count:>6} OER={format_oer(item.oer):>8} Cont={item.outcome.continuation_rate:>7.2%} Fail={item.outcome.failure_rate:>7.2%}")


def append_transitions(lines: list[str], population: ArchetypePopulation) -> None:
    lines.append(f"{'Transition':<58} {'Count':>8} {'Probability':>12} {'Lift':>9} {'Cont':>9} {'Fail':>9}")
    for item in sorted(population.transitions.values(), key=lambda value: (value.source, value.target)):
        if item.count:
            lines.append(f"{item.source + ' -> ' + item.target:<58} {item.count:>8} {item.probability:>11.2%} {format_oer(item.lift):>9} {item.outcome.continuation_rate:>8.2%} {item.outcome.failure_rate:>8.2%}")


def append_stability(lines: list[str], population: ArchetypePopulation) -> None:
    lines.append(f"{'Archetype':<28} {'Count':>8} {'Repeat10':>10} {'Repeat20':>10} {'Return10':>10} {'Return20':>10}")
    for name in ARCHETYPES:
        item = population.stability[name]
        lines.append(f"{name:<28} {item.count:>8} {item.repeat10:>9.2%} {item.repeat20:>9.2%} {item.return10:>9.2%} {item.return20:>9.2%}")


def append_branches(lines: list[str], values: dict[tuple[str, str], BranchStats]) -> None:
    lines.append(f"{'Source':<28} {'Target':<28} {'Count':>8} {'Probability':>12} {'Lift':>9}")
    for item in sorted(values.values(), key=lambda value: (value.source, value.target)):
        if item.count:
            lines.append(f"{item.source:<28} {item.target:<28} {item.count:>8} {item.probability:>11.2%} {format_oer(item.lift):>9}")


def instrument_report(study: InstrumentStudy) -> str:
    full = study.populations["Full Population"]
    population_b = study.populations["Population B"]
    lines = [
        f"APVA Narrative Archetype Study v0.1 - {study.instrument}",
        "=" * (39 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {len(study.rows)}",
        f"Family counts: {full.trajectory.counts}",
        f"Path counts: {full.trajectory.windows}",
        f"Primary archetype counts: {dict((name, sum(value == name for value in full.primary_by_anchor.values())) for name in ARCHETYPES)}",
        "Archetype membership is multi-match; primary labels use the requested precedence.",
        "",
        "Section 1 - Path Generation Summary",
        "===================================",
    ]
    append_path_summary(lines, full)
    lines.extend(["\nSection 2 - Archetype Membership", "================================"])
    append_membership(lines, full)
    lines.extend(["\nSection 3 - Archetype Statistics", "================================"])
    append_member_rankings(lines, full)
    lines.extend(["\nSection 4 - Archetype Transitions", "================================="])
    append_transitions(lines, full)
    lines.extend(["\nSection 5 - Archetype Stability", "==============================="])
    append_stability(lines, full)
    lines.extend(["\nSection 6 - Divergence", "======================"])
    append_branches(lines, full.divergence)
    lines.extend(["\nSection 7 - Convergence", "======================="])
    append_branches(lines, full.convergence)
    lines.extend(["\nSection 8 - Population B", "========================"])
    lines.append(f"Population B rows: {len(population_b.trajectory.indexes)}")
    append_membership(lines, population_b)
    lines.extend(["\nPopulation B Transitions", "------------------------"])
    append_transitions(lines, population_b)
    lines.extend(["\nPopulation B Stability", "----------------------"])
    append_stability(lines, population_b)
    lines.extend(["\nPopulation B Divergence", "-----------------------"])
    append_branches(lines, population_b.divergence)
    lines.extend(["\nPopulation B Convergence", "------------------------"])
    append_branches(lines, population_b.convergence)
    lines.extend(["\nSection 9 - Mechanical Research Notes", "====================================="])
    common = max(ARCHETYPES, key=lambda name: full.membership[name].total_occurrences)
    stable = max(ARCHETYPES, key=lambda name: full.stability[name].return20)
    lines.append(f"- Most common multi-match archetype: {common}.")
    lines.append(f"- Highest ReturnWithin20 primary archetype: {stable}.")
    lines.append(f"- Population B contains {len(population_b.primary_by_anchor)} primary-labeled contiguous path anchors.")
    lines.append("- Archetype rules, transitions, and stability metrics are mechanical and descriptive only.")
    return "\n".join(lines) + "\n"


def aggregate_archetypes(studies: list[InstrumentStudy], population: str = "Full Population") -> list[AggregateArchetype]:
    output = []
    for name in ARCHETYPES:
        values = {study.instrument: study.populations[population].membership[name] for study in studies}
        valid = [item for item in values.values() if item.total_occurrences >= PER_INSTRUMENT_MIN_COUNT]
        output.append(AggregateArchetype(name, values, len(valid), sum(item.outcome.continuation_rate > 0.5 for item in valid), sum(item.outcome.continuation_rate < 0.5 for item in valid), mean([item.outcome.continuation_rate for item in valid]), mean([item.outcome.failure_rate for item in valid]), mean([item.mean_oer for item in valid])))
    return output


def aggregate_transitions(studies: list[InstrumentStudy]) -> list[AggregateTransition]:
    output = []
    for source in ARCHETYPES:
        for target in ARCHETYPES:
            values = {study.instrument: study.populations["Full Population"].transitions[(source, target)] for study in studies}
            valid = [item for item in values.values() if item.count >= PER_INSTRUMENT_MIN_COUNT]
            output.append(AggregateTransition(source, target, values, len(valid), mean([item.probability for item in valid]), mean([item.lift for item in valid if item.lift is not None]), mean([item.outcome.continuation_rate for item in valid]), mean([item.outcome.failure_rate for item in valid])))
    return output


def aggregate_stability(studies: list[InstrumentStudy]) -> list[AggregateStability]:
    output = []
    for name in ARCHETYPES:
        values = {study.instrument: study.populations["Full Population"].stability[name] for study in studies}
        valid = [item for item in values.values() if item.count >= PER_INSTRUMENT_MIN_COUNT]
        output.append(AggregateStability(name, values, len(valid), mean([item.repeat10 for item in valid]), mean([item.repeat20 for item in valid]), mean([item.return10 for item in valid]), mean([item.return20 for item in valid])))
    return output


def aggregate_branches(studies: list[InstrumentStudy], attribute: str) -> list[AggregateBranch]:
    output = []
    for source in ARCHETYPES:
        for target in ARCHETYPES:
            values = {study.instrument: getattr(study.populations["Full Population"], attribute)[(source, target)] for study in studies}
            valid = [item for item in values.values() if item.count >= PER_INSTRUMENT_MIN_COUNT]
            output.append(AggregateBranch(source, target, values, len(valid), mean([item.probability for item in valid]), mean([item.lift for item in valid if item.lift is not None])))
    return output


def append_aggregate_archetypes(lines: list[str], items: list[AggregateArchetype], columns: list[str]) -> None:
    lines.append(f"{'Archetype':<28}" + "".join(f" {name + '_N':>9} {name + '_Cont':>10} {name + '_Fail':>10} {name + '_OER':>9}" for name in columns) + " ValidN PosN NegN MeanCont MeanFail MeanOER")
    for item in items:
        cells = []
        for name in columns:
            value = item.values.get(name)
            cells.append(f" {value.total_occurrences if value else 0:>9} {value.outcome.continuation_rate if value else 0.0:>9.2%} {value.outcome.failure_rate if value else 0.0:>9.2%} {value.mean_oer if value else 0.0:>9.3f}")
        lines.append(f"{item.name:<28}{''.join(cells)} {item.valid_instruments:>6} {item.positive_count:>4} {item.negative_count:>4} {item.mean_continuation:>8.2%} {item.mean_failure:>8.2%} {item.mean_oer:>7.3f}")


def append_aggregate_transitions(lines: list[str], items: list[AggregateTransition], columns: list[str]) -> None:
    lines.append(f"{'Transition':<58}" + "".join(f" {name + '_N':>8} {name + '_Prob':>10} {name + '_Lift':>9}" for name in columns) + " ValidN MeanProb MeanLift MeanCont MeanFail")
    for item in items:
        cells = []
        for name in columns:
            value = item.values.get(name)
            cells.append(f" {value.count if value else 0:>8} {value.probability if value else 0.0:>9.2%} {value.lift if value and value.lift is not None else 0.0:>9.3f}")
        lines.append(f"{item.source + ' -> ' + item.target:<58}{''.join(cells)} {item.valid_instruments:>6} {item.mean_probability:>8.2%} {item.mean_lift:>8.3f} {item.mean_continuation:>8.2%} {item.mean_failure:>8.2%}")


def append_aggregate_stability(lines: list[str], items: list[AggregateStability], columns: list[str]) -> None:
    lines.append(f"{'Archetype':<28}" + "".join(f" {name + '_R10':>9} {name + '_R20':>9} {name + '_L10':>9} {name + '_L20':>9}" for name in columns) + " ValidN MeanR10 MeanR20 MeanL10 MeanL20")
    for item in items:
        cells = []
        for name in columns:
            value = item.values.get(name, StabilityStats(0, 0.0, 0.0, 0.0, 0.0))
            cells.append(f" {value.repeat10:>8.2%} {value.repeat20:>8.2%} {value.return10:>8.2%} {value.return20:>8.2%}")
        lines.append(f"{item.name:<28}{''.join(cells)} {item.valid_instruments:>6} {item.mean_repeat10:>8.2%} {item.mean_repeat20:>8.2%} {item.mean_return10:>8.2%} {item.mean_return20:>8.2%}")


def append_aggregate_branches(lines: list[str], items: list[AggregateBranch], columns: list[str]) -> None:
    lines.append(f"{'Source':<28} {'Target':<28}" + "".join(f" {name + '_N':>8} {name + '_Prob':>10} {name + '_Lift':>9}" for name in columns) + " ValidN MeanProb MeanLift")
    for item in items:
        cells = []
        for name in columns:
            value = item.values.get(name)
            cells.append(f" {value.count if value else 0:>8} {value.probability if value else 0.0:>9.2%} {value.lift if value and value.lift is not None else 0.0:>9.3f}")
        lines.append(f"{item.source:<28} {item.target:<28}{''.join(cells)} {item.valid_instruments:>6} {item.mean_probability:>8.2%} {item.mean_lift:>8.3f}")


def append_ranked(lines: list[str], title: str, items: list, key, formatter) -> None:
    lines.extend([f"\n{title}", "." * len(title)])
    eligible = [item for item in items if item.valid_instruments >= 2]
    if not eligible:
        lines.append("No items met the two-instrument minimum.")
        return
    for rank, item in enumerate(sorted(eligible, key=key, reverse=True)[:TOP_LIMIT], start=1):
        lines.append(f"{rank:>3}. {formatter(item)}")


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    columns = instrument_columns(studies)
    archetypes = aggregate_archetypes(studies)
    transitions = aggregate_transitions(studies)
    stability = aggregate_stability(studies)
    divergence = aggregate_branches(studies, "divergence")
    convergence = aggregate_branches(studies, "convergence")
    population_b = aggregate_archetypes(studies, "Population B")
    lines = ["APVA Narrative Archetype Study v0.1 - Aggregate", "===============================================", f"Instruments: {', '.join(study.instrument for study in studies)}", "Archetype membership is multi-match; primary labels drive transition and stability studies.", "", "Aggregate Archetype Table", "========================="]
    append_aggregate_archetypes(lines, archetypes, columns)
    lines.extend(["\nAggregate Archetype Transition Table", "===================================="])
    append_aggregate_transitions(lines, transitions, columns)
    lines.extend(["\nAggregate Stability Table", "========================="])
    append_aggregate_stability(lines, stability, columns)
    lines.extend(["\nAggregate Divergence Table", "=========================="])
    append_aggregate_branches(lines, divergence, columns)
    lines.extend(["\nAggregate Convergence Table", "==========================="])
    append_aggregate_branches(lines, convergence, columns)
    lines.extend(["\nPopulation B Summary", "===================="])
    append_aggregate_archetypes(lines, population_b, columns)
    lines.extend(["\nAggregate Rankings", "=================="])
    append_ranked(lines, "1. Most common archetypes", archetypes, lambda item: (sum(value.total_occurrences for value in item.values.values()), item.name), lambda item: f"{item.name:<28} TotalN={sum(value.total_occurrences for value in item.values.values())} ValidN={item.valid_instruments}")
    append_ranked(lines, "2. Most overrepresented archetypes", archetypes, lambda item: (item.mean_oer, item.name), lambda item: f"{item.name:<28} MeanOER={item.mean_oer:.4f} ValidN={item.valid_instruments}")
    append_ranked(lines, "3. Highest continuation archetypes", archetypes, lambda item: (item.mean_continuation, item.name), lambda item: f"{item.name:<28} MeanCont={item.mean_continuation:.2%} MeanFail={item.mean_failure:.2%}")
    append_ranked(lines, "4. Highest failure archetypes", archetypes, lambda item: (item.mean_failure, item.name), lambda item: f"{item.name:<28} MeanFail={item.mean_failure:.2%} MeanCont={item.mean_continuation:.2%}")
    append_ranked(lines, "5. Most stable archetypes", stability, lambda item: (item.mean_return20, item.name), lambda item: f"{item.name:<28} MeanReturn10={item.mean_return10:.2%} MeanReturn20={item.mean_return20:.2%}")
    append_ranked(lines, "6. Strongest archetype transitions", transitions, lambda item: (item.mean_lift, item.source, item.target), lambda item: f"{item.source} -> {item.target:<28} MeanLift={item.mean_lift:.4f} MeanProb={item.mean_probability:.2%}")
    append_ranked(lines, "7. Strongest archetype divergences", divergence, lambda item: (item.mean_lift, item.source, item.target), lambda item: f"{item.source} -> {item.target:<28} MeanLift={item.mean_lift:.4f} MeanProb={item.mean_probability:.2%}")
    append_ranked(lines, "8. Strongest archetype convergences", convergence, lambda item: (item.mean_lift, item.source, item.target), lambda item: f"{item.target} -> {item.source:<28} MeanLift={item.mean_lift:.4f} MeanProb={item.mean_probability:.2%}")
    full = {item.name: item for item in archetypes}
    lines.extend(["\n9. Population-B archetype differences", "." * 37])
    for item in sorted(population_b, key=lambda value: (-abs(value.mean_continuation - full[value.name].mean_continuation), value.name)):
        lines.append(f"{item.name:<28} DeltaMeanContinuation={item.mean_continuation - full[item.name].mean_continuation:>9.2%} PopulationBValidN={item.valid_instruments}")
    lines.extend(["\nCross-Instrument Mechanical Research Notes", "=========================================="])
    common = max(archetypes, key=lambda item: sum(value.total_occurrences for value in item.values.values()))
    stable = max(stability, key=lambda item: item.mean_return20)
    lines.append(f"- Most common multi-match archetype: {common.name}.")
    lines.append(f"- Highest replicated ReturnWithin20 primary archetype: {stable.name}.")
    lines.append("- Recovery includes C->C->B paths mechanically; Exhaustion includes C->C->C paths mechanically.")
    lines.append("- Archetype rules, transitions, divergence, convergence, and stability metrics are descriptive only.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    studies = [study_instrument(path) for path in args.input_paths]
    seen = set()
    for study in studies:
        if study.instrument in seen:
            raise ValueError(f"Duplicate instrument input: {study.instrument}")
        seen.add(study.instrument)
        write_text(Path("Evidence") / "Output" / study.instrument / f"Archetypes_{study.instrument}.txt", instrument_report(study))
    report = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, report)
    print(report, end="")


if __name__ == "__main__":
    main()
