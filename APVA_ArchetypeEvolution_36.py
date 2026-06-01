"""APVA Archetype Evolution Study v0.1.

Research-only analysis of mechanical narrative evolution over Study 35 archetypes.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

from APVA_AcceptanceAnatomy_23 import mature_aligned_lateral_indexes
from APVA_Archetypes_35 import ARCHETYPES, primary_archetype
from APVA_BreakoutContext_08 import EvidenceBar, direction_relative_return, load_rows
from APVA_ConditionTopology_27 import flags
from APVA_ContextTopology_28 import context_metrics
from APVA_FamilyEvolution_30 import FAMILIES, family_for, instrument_columns
from APVA_LateralAnatomy_19 import effect_size
from APVA_RegimeTransition_16 import instrument_name, write_text
from APVA_StateGrammar_25 import OutcomeStats, format_oer, mean, summarize


AGGREGATE_OUTPUT = Path("Evidence/Output/ArchetypeEvolution/ArchetypeEvolution_All.txt")
HORIZONS = (1, 2, 3, 5)
TOP_LIMIT = 20
PER_INSTRUMENT_MIN_COUNT = 20
POSITIONS = ("First", "Middle", "Last", "Only")
DIRECT_COMPARISONS = (
    ("Recovery", "Recovery", "Decay"),
    ("Recovery", "Recovery", "Compression Resolution"),
    ("Exhaustion", "Exhaustion", "Recovery"),
    ("Exhaustion", "Exhaustion", "Decay"),
    ("Decay", "Compression Resolution", "Neutral Drift"),
    ("Decay", "Compression Resolution", "Reassertion"),
    ("Compression Resolution", "Compression Resolution", "Decay"),
    ("Reassertion", "Neutral Drift", "Destructive Persistence"),
)


@dataclass(frozen=True)
class RunInfo:
    so_far: int
    total: int
    position: str


@dataclass(frozen=True)
class Observation:
    index: int
    current: str
    family: str
    path3: str
    path4: str
    path5: str
    next_by_horizon: dict[int, str]
    features: dict[str, float]
    outcome: float | None


@dataclass(frozen=True)
class TransitionStats:
    current: str
    next_archetype: str
    horizon: int
    count: int
    probability: float
    lift: float | None
    outcome: OutcomeStats


@dataclass(frozen=True)
class FeatureComparison:
    feature: str
    left_count: int
    right_count: int
    left_mean: float
    right_mean: float
    delta: float
    effect: float


@dataclass(frozen=True)
class PopulationStudy:
    name: str
    indexes: list[int]
    family_by_index: dict[int, str]
    archetype_by_index: dict[int, str]
    observations: list[Observation]
    transitions: dict[tuple[int, str, str], TransitionStats]
    branch_predictors: dict[tuple[str, str], list[FeatureComparison]]
    direct_comparisons: dict[str, list[FeatureComparison]]


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    rows: list[EvidenceBar]
    populations: dict[str, PopulationStudy]


@dataclass(frozen=True)
class AggregateTransition:
    current: str
    next_archetype: str
    horizon: int
    values: dict[str, TransitionStats]
    valid_instruments: int
    mean_probability: float
    mean_lift: float
    mean_continuation: float
    mean_failure: float
    mean_skew: float


@dataclass(frozen=True)
class AggregateFeature:
    category: str
    feature: str
    values: dict[str, FeatureComparison]
    valid_instruments: int
    positive_count: int
    negative_count: int
    mean_effect: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study bar-aligned evolution of Study 35 APVA archetypes.")
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def path_for(mapping: dict[int, str], index: int, length: int) -> str:
    if not all(offset in mapping for offset in range(index - length + 1, index + 1)):
        return ""
    return "->".join(mapping[offset] for offset in range(index - length + 1, index + 1))


def archetype_stream(mapping: dict[int, str]) -> dict[int, str]:
    output = {}
    for index in sorted(mapping):
        paths = [path_for(mapping, index, length) for length in (3, 4, 5)]
        available = [path for path in paths if path]
        if available:
            output[index] = primary_archetype(available[-1])
    return output


def run_info(mapping: dict[int, str]) -> dict[int, RunInfo]:
    output = {}
    ordered = sorted(mapping)
    cursor = 0
    while cursor < len(ordered):
        start = cursor
        while cursor + 1 < len(ordered) and ordered[cursor + 1] == ordered[cursor] + 1 and mapping[ordered[cursor + 1]] == mapping[ordered[start]]:
            cursor += 1
        total = cursor - start + 1
        for position_index in range(start, cursor + 1):
            index = ordered[position_index]
            so_far = position_index - start + 1
            if total == 1:
                position = "Only"
            elif position_index == start:
                position = "First"
            elif position_index == cursor:
                position = "Last"
            else:
                position = "Middle"
            output[index] = RunInfo(so_far, total, position)
        cursor += 1
    return output


def normalized(value: str) -> str:
    return value or "Unknown"


def build_features(
    rows: list[EvidenceBar],
    index: int,
    family: str,
    paths: dict[int, str],
    archetype_run: RunInfo,
    family_run: RunInfo,
) -> dict[str, float]:
    row = rows[index]
    values = flags(row)
    output = {f"Flag_{name}": float(enabled) for name, enabled in values.items()}
    output.update(context_metrics(rows, index))
    output.update(
        {
            "ArchetypeRunLengthSoFar": float(archetype_run.so_far),
            "ArchetypeTotalRunLength": float(archetype_run.total),
            "FamilyRunLengthSoFar": float(family_run.so_far),
            "FamilyTotalRunLength": float(family_run.total),
        }
    )
    for position in POSITIONS:
        output[f"PositionInArchetypeRun_{position}"] = float(archetype_run.position == position)
    for name in ARCHETYPES:
        output[f"CurrentArchetype_{name}"] = float(primary_archetype(paths[max(paths)]) == name)
    for name in FAMILIES:
        output[f"CurrentFamily_{name}"] = float(family == name)
    for label, value in (
        ("ParticipationState", normalized(row.participation)),
        ("AcceptanceState", normalized(row.acceptance)),
        ("CompressionState", normalized(row.compression)),
        ("DissipationState", normalized(row.dissipation)),
        ("ExpansionState", normalized(row.expansion)),
    ):
        output[f"{label}_{value}"] = 1.0
    return output


def feature_names(observations: list[Observation]) -> list[str]:
    return sorted(set().union(*(item.features for item in observations))) if observations else []


def compare(feature: str, left: list[float], right: list[float]) -> FeatureComparison:
    return FeatureComparison(feature, len(left), len(right), mean(left), mean(right), mean(left) - mean(right), effect_size(left, right))


def compare_groups(left: list[Observation], right: list[Observation], features: list[str]) -> list[FeatureComparison]:
    return [compare(feature, [item.features.get(feature, 0.0) for item in left], [item.features.get(feature, 0.0) for item in right]) for feature in features]


def build_observations(rows: list[EvidenceBar], family_mapping: dict[int, str], archetypes: dict[int, str]) -> list[Observation]:
    archetype_runs = run_info(archetypes)
    family_runs = run_info(family_mapping)
    output = []
    for index, current in archetypes.items():
        paths = {length: path for length in (3, 4, 5) if (path := path_for(family_mapping, index, length))}
        next_by_horizon = {horizon: archetypes[index + horizon] for horizon in HORIZONS if index + horizon in archetypes}
        output.append(
            Observation(
                index,
                current,
                family_mapping[index],
                paths.get(3, ""),
                paths.get(4, ""),
                paths.get(5, ""),
                next_by_horizon,
                build_features(rows, index, family_mapping[index], paths, archetype_runs[index], family_runs[index]),
                direction_relative_return(rows, index, 5),
            )
        )
    return output


def add_reportable_path_features(observations: list[Observation]) -> None:
    for length, attribute in ((3, "path3"), (4, "path4"), (5, "path5")):
        counts = {}
        for item in observations:
            path = getattr(item, attribute)
            if path:
                counts[path] = counts.get(path, 0) + 1
        reportable = [path for path, count in counts.items() if count >= PER_INSTRUMENT_MIN_COUNT]
        for item in observations:
            current = getattr(item, attribute)
            if current in reportable:
                item.features[f"CurrentPath{length}_{current}"] = 1.0


def build_transitions(observations: list[Observation]) -> dict[tuple[int, str, str], TransitionStats]:
    output = {}
    for horizon in HORIZONS:
        eligible = [item for item in observations if horizon in item.next_by_horizon]
        next_frequencies = {name: sum(item.next_by_horizon[horizon] == name for item in eligible) / len(eligible) if eligible else 0.0 for name in ARCHETYPES}
        for current in ARCHETYPES:
            current_items = [item for item in eligible if item.current == current]
            for next_name in ARCHETYPES:
                selected = [item for item in current_items if item.next_by_horizon[horizon] == next_name]
                probability = len(selected) / len(current_items) if current_items else 0.0
                baseline = next_frequencies[next_name]
                output[(horizon, current, next_name)] = TransitionStats(current, next_name, horizon, len(selected), probability, probability / baseline if baseline else None, summarize([item.outcome for item in selected if item.outcome is not None]))
    return output


def build_population(rows: list[EvidenceBar], name: str, indexes: list[int]) -> PopulationStudy:
    family_mapping = {index: family_for(rows[index]) for index in indexes}
    archetypes = archetype_stream(family_mapping)
    observations = build_observations(rows, family_mapping, archetypes)
    add_reportable_path_features(observations)
    features = feature_names(observations)
    predictors = {}
    for current in ARCHETYPES:
        current_items = [item for item in observations if item.current == current and 1 in item.next_by_horizon]
        next_names = sorted(set(item.next_by_horizon[1] for item in current_items))
        for next_name in next_names:
            predictors[(current, next_name)] = compare_groups([item for item in current_items if item.next_by_horizon[1] == next_name], [item for item in current_items if item.next_by_horizon[1] != next_name], features)
    direct = {}
    for current, left, right in DIRECT_COMPARISONS:
        current_items = [item for item in observations if item.current == current and 1 in item.next_by_horizon]
        direct[f"{current}->{left} vs {current}->{right}"] = compare_groups([item for item in current_items if item.next_by_horizon[1] == left], [item for item in current_items if item.next_by_horizon[1] == right], features)
    return PopulationStudy(name, indexes, family_mapping, archetypes, observations, build_transitions(observations), predictors, direct)


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    return InstrumentStudy(instrument_name(path), path, rows, {"Full Population": build_population(rows, "Full Population", list(range(len(rows)))), "Population B": build_population(rows, "Population B", mature_aligned_lateral_indexes(rows))})


def quality(item: TransitionStats) -> str:
    skew = item.outcome.continuation_rate - item.outcome.failure_rate
    if skew > 0.05:
        return "Favorable"
    if skew < -0.05:
        return "Unfavorable"
    return "Neutral"


def append_transition_table(lines: list[str], population: PopulationStudy, horizons: tuple[int, ...] = HORIZONS) -> None:
    lines.append(f"{'H':>2} {'Current':<28} {'Next':<28} {'Count':>8} {'Probability':>12} {'Lift':>9}")
    for horizon in horizons:
        for current in ARCHETYPES:
            for next_name in ARCHETYPES:
                item = population.transitions[(horizon, current, next_name)]
                if item.count:
                    lines.append(f"{horizon:>2} {current:<28} {next_name:<28} {item.count:>8} {item.probability:>11.2%} {format_oer(item.lift):>9}")


def append_predictors(lines: list[str], population: PopulationStudy, limit: int = TOP_LIMIT) -> None:
    for (current, next_name), comparisons in population.branch_predictors.items():
        selected = [item for item in comparisons if item.left_count and item.right_count]
        if not selected:
            continue
        lines.extend([f"\n{current} -> {next_name}", "." * (len(current) + len(next_name) + 4)])
        for rank, item in enumerate(sorted(selected, key=lambda value: (-abs(value.effect), value.feature))[:limit], start=1):
            lines.append(f"{rank:>3}. {item.feature:<52} PresentN={item.left_count:>6} AbsentN={item.right_count:>6} Delta={item.delta:>10.6f} Effect={item.effect:>10.6f}")


def append_direct(lines: list[str], population: PopulationStudy) -> None:
    for comparison_name, comparisons in population.direct_comparisons.items():
        lines.extend([f"\n{comparison_name}", "." * len(comparison_name)])
        selected = [item for item in comparisons if item.left_count and item.right_count]
        if not selected:
            lines.append("Insufficient observations.")
            continue
        for rank, item in enumerate(sorted(selected, key=lambda value: (-abs(value.effect), value.feature))[:TOP_LIMIT], start=1):
            lines.append(f"{rank:>3}. {item.feature:<52} LeftN={item.left_count:>6} RightN={item.right_count:>6} Delta={item.delta:>10.6f} Effect={item.effect:>10.6f}")


def append_outcomes(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'Current':<28} {'Next':<28} {'Count':>8} {'Mean':>11} {'Median':>11} {'Cont':>9} {'Fail':>9} {'Flat':>9}")
    for current in ARCHETYPES:
        for next_name in ARCHETYPES:
            item = population.transitions[(1, current, next_name)]
            if item.count:
                value = item.outcome
                lines.append(f"{current:<28} {next_name:<28} {item.count:>8} {value.mean:>11.6f} {value.median:>11.6f} {value.continuation_rate:>8.2%} {value.failure_rate:>8.2%} {value.flat_rate:>8.2%}")


def append_quality(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'Transition':<58} {'Count':>8} {'Probability':>12} {'Lift':>9} {'Cont':>9} {'Fail':>9} {'Skew':>9} {'Quality':<12}")
    for current in ARCHETYPES:
        for next_name in ARCHETYPES:
            item = population.transitions[(1, current, next_name)]
            if item.count:
                skew = item.outcome.continuation_rate - item.outcome.failure_rate
                lines.append(f"{current + ' -> ' + next_name:<58} {item.count:>8} {item.probability:>11.2%} {format_oer(item.lift):>9} {item.outcome.continuation_rate:>8.2%} {item.outcome.failure_rate:>8.2%} {skew:>8.2%} {quality(item):<12}")


def instrument_report(study: InstrumentStudy) -> str:
    full = study.populations["Full Population"]
    population_b = study.populations["Population B"]
    counts = {name: sum(value == name for value in full.archetype_by_index.values()) for name in ARCHETYPES}
    lines = [f"APVA Archetype Evolution Study v0.1 - {study.instrument}", "=" * (40 + len(study.instrument)), f"Instrument: {study.instrument}", f"Input: {study.path}", f"Total rows: {len(study.rows)}", f"Family counts: {dict((name, sum(value == name for value in full.family_by_index.values())) for name in FAMILIES)}", f"Archetype counts: {counts}", f"Valid archetype observations: {len(full.observations)}", "Primary archetypes reuse Study 35 precedence exactly.", "", "Section 1 - Archetype Transition Probability Tables", "================================================="]
    append_transition_table(lines, full, (1,))
    lines.extend(["\nSection 2 - Horizon Comparison t+1/t+2/t+3/t+5", "=============================================="])
    append_transition_table(lines, full)
    lines.extend(["\nSection 3 - Branch Predictor Rankings", "====================================="])
    append_predictors(lines, full)
    lines.extend(["\nSection 4 - Branch-vs-Branch Comparisons", "========================================"])
    append_direct(lines, full)
    lines.extend(["\nSection 5 - Outcome Layer", "========================="])
    append_outcomes(lines, full)
    lines.extend(["\nSection 6 - Transition Quality", "=============================="])
    append_quality(lines, full)
    lines.extend(["\nSection 7 - Population B Summary", "================================"])
    lines.append(f"Population B rows: {len(population_b.indexes)}")
    lines.append(f"Population B valid archetype observations: {len(population_b.observations)}")
    append_transition_table(lines, population_b)
    lines.extend(["\nPopulation B Branch Predictors", "------------------------------"])
    append_predictors(lines, population_b, limit=10)
    lines.extend(["\nPopulation B Outcome Layer", "--------------------------"])
    append_outcomes(lines, population_b)
    lines.extend(["\nSection 8 - Mechanical Research Notes", "====================================="])
    persistent = max(ARCHETYPES, key=lambda name: full.transitions[(1, name, name)].probability)
    lines.append(f"- Highest t+1 primary-archetype persistence: {persistent}, Probability={full.transitions[(1, persistent, persistent)].probability:.2%}.")
    lines.append(f"- Population B contains {len(population_b.observations)} valid bar-aligned archetype observations.")
    lines.append("- Transition quality is a mechanical outcome-skew label, not a recommendation.")
    return "\n".join(lines) + "\n"


def aggregate_transitions(studies: list[InstrumentStudy], population: str = "Full Population") -> list[AggregateTransition]:
    output = []
    for horizon in HORIZONS:
        for current in ARCHETYPES:
            for next_name in ARCHETYPES:
                values = {study.instrument: study.populations[population].transitions[(horizon, current, next_name)] for study in studies}
                valid = [item for item in values.values() if item.count >= PER_INSTRUMENT_MIN_COUNT]
                output.append(AggregateTransition(current, next_name, horizon, values, len(valid), mean([item.probability for item in valid]), mean([item.lift for item in valid if item.lift is not None]), mean([item.outcome.continuation_rate for item in valid]), mean([item.outcome.failure_rate for item in valid]), mean([item.outcome.continuation_rate - item.outcome.failure_rate for item in valid])))
    return output


def aggregate_features(studies: list[InstrumentStudy], attribute: str, population: str = "Full Population") -> list[AggregateFeature]:
    categories = sorted(set().union(*(getattr(study.populations[population], attribute) for study in studies)))
    output = []
    for category in categories:
        mappings = {
            study.instrument: {
                item.feature: item
                for item in getattr(study.populations[population], attribute).get(category, [])
            }
            for study in studies
        }
        features = sorted(set().union(*(mapping for mapping in mappings.values())))
        for feature in features:
            values = {name: mapping[feature] for name, mapping in mappings.items() if feature in mapping}
            valid = [item for item in values.values() if item.left_count >= PER_INSTRUMENT_MIN_COUNT and item.right_count >= PER_INSTRUMENT_MIN_COUNT]
            label = f"{category[0]}->{category[1]}" if isinstance(category, tuple) else category
            output.append(AggregateFeature(label, feature, values, len(valid), sum(item.effect > 0.0 for item in valid), sum(item.effect < 0.0 for item in valid), mean([item.effect for item in valid])))
    return output


def append_aggregate_transitions(lines: list[str], items: list[AggregateTransition], columns: list[str]) -> None:
    lines.append(f"{'H':>2} {'Current':<28} {'Next':<28}" + "".join(f" {name + '_N':>8} {name + '_Prob':>10} {name + '_Lift':>9}" for name in columns) + " ValidN MeanProb MeanLift MeanCont MeanFail MeanSkew")
    for item in items:
        cells = []
        for name in columns:
            value = item.values.get(name)
            cells.append(f" {value.count if value else 0:>8} {value.probability if value else 0.0:>9.2%} {value.lift if value and value.lift is not None else 0.0:>9.3f}")
        lines.append(f"{item.horizon:>2} {item.current:<28} {item.next_archetype:<28}{''.join(cells)} {item.valid_instruments:>6} {item.mean_probability:>8.2%} {item.mean_lift:>8.3f} {item.mean_continuation:>8.2%} {item.mean_failure:>8.2%} {item.mean_skew:>8.2%}")


def append_aggregate_features(lines: list[str], items: list[AggregateFeature], columns: list[str]) -> None:
    lines.append(f"{'Category':<58} {'Feature':<52}" + "".join(f" {name + '_Effect':>12}" for name in columns) + " ValidN PosN NegN MeanEffect")
    for item in items:
        if item.valid_instruments < 2:
            continue
        cells = "".join(f" {item.values[name].effect if name in item.values else 0.0:>12.6f}" for name in columns)
        lines.append(f"{item.category:<58} {item.feature:<52}{cells} {item.valid_instruments:>6} {item.positive_count:>4} {item.negative_count:>4} {item.mean_effect:>10.6f}")


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
    transitions = aggregate_transitions(studies)
    predictors = aggregate_features(studies, "branch_predictors")
    direct = aggregate_features(studies, "direct_comparisons")
    population_b = aggregate_transitions(studies, "Population B")
    lines = ["APVA Archetype Evolution Study v0.1 - Aggregate", "===============================================", f"Instruments: {', '.join(study.instrument for study in studies)}", "Primary archetypes reuse Study 35 precedence exactly.", "", "Aggregate Archetype Transition Table", "===================================="]
    append_aggregate_transitions(lines, transitions, columns)
    lines.extend(["\nAggregate Branch Predictor Table", "================================"])
    append_aggregate_features(lines, predictors, columns)
    lines.extend(["\nAggregate Branch-vs-Branch Table", "================================"])
    append_aggregate_features(lines, direct, columns)
    lines.extend(["\nAggregate Outcome Table", "======================="])
    append_aggregate_transitions(lines, [item for item in transitions if item.horizon == 1], columns)
    lines.extend(["\nPopulation B Summary", "===================="])
    append_aggregate_transitions(lines, population_b, columns)
    lines.extend(["\nAggregate Rankings", "=================="])
    t1 = [item for item in transitions if item.horizon == 1]
    append_ranked(lines, "1. Most likely next archetype transitions", t1, lambda item: (item.mean_probability, item.current, item.next_archetype), lambda item: f"{item.current} -> {item.next_archetype:<28} MeanProb={item.mean_probability:.2%} MeanLift={item.mean_lift:.4f}")
    append_ranked(lines, "2. Strongest transition lifts", t1, lambda item: (item.mean_lift, item.current, item.next_archetype), lambda item: f"{item.current} -> {item.next_archetype:<28} MeanLift={item.mean_lift:.4f} MeanProb={item.mean_probability:.2%}")
    append_ranked(lines, "3. Most persistent archetypes", [item for item in t1 if item.current == item.next_archetype], lambda item: (item.mean_probability, item.current), lambda item: f"{item.current:<28} MeanPersistence={item.mean_probability:.2%} MeanLift={item.mean_lift:.4f}")
    for number, category, title in ((4, "Recovery->Recovery", "Strongest predictors of Recovery continuation"), (5, "Exhaustion->Exhaustion", "Strongest predictors of Exhaustion persistence"), (6, "Decay->Compression Resolution", "Strongest predictors of Decay -> Compression Resolution"), (7, "Compression Resolution->Compression Resolution", "Strongest predictors of Compression Resolution persistence")):
        append_ranked(lines, f"{number}. {title}", [item for item in predictors if item.category == category], lambda item: (abs(item.mean_effect), item.feature), lambda item: f"{item.feature:<52} MeanEffect={item.mean_effect:>10.6f} PosN={item.positive_count} NegN={item.negative_count}")
    append_ranked(lines, "8. Best outcome archetype transitions", t1, lambda item: (item.mean_skew, item.current, item.next_archetype), lambda item: f"{item.current} -> {item.next_archetype:<28} MeanSkew={item.mean_skew:.2%} MeanCont={item.mean_continuation:.2%}")
    append_ranked(lines, "9. Worst outcome archetype transitions", t1, lambda item: (-item.mean_skew, item.current, item.next_archetype), lambda item: f"{item.current} -> {item.next_archetype:<28} MeanSkew={item.mean_skew:.2%} MeanFail={item.mean_failure:.2%}")
    full_lookup = {(item.horizon, item.current, item.next_archetype): item for item in transitions}
    differences = []
    for item in population_b:
        baseline = full_lookup[(item.horizon, item.current, item.next_archetype)]
        if item.valid_instruments >= 2 and baseline.valid_instruments >= 2:
            differences.append((item, item.mean_probability - baseline.mean_probability))
    lines.extend(["\n10. Population-B differences", "." * 28])
    if differences:
        for item, delta in sorted(differences, key=lambda pair: (-abs(pair[1]), pair[0].current, pair[0].next_archetype))[:TOP_LIMIT]:
            lines.append(f"t+{item.horizon} {item.current} -> {item.next_archetype:<28} DeltaMeanProbability={delta:>9.2%}")
    else:
        lines.append("No Population B transitions met the replicated comparison minimum.")
    lines.extend(["\nCross-Instrument Mechanical Research Notes", "=========================================="])
    persistent = max((item for item in t1 if item.current == item.next_archetype), key=lambda item: item.mean_probability)
    lines.append(f"- Highest replicated t+1 archetype persistence: {persistent.current}, MeanProbability={persistent.mean_probability:.2%}.")
    lines.append("- Predictor rankings use t+1 next-archetype branches; horizon tables retain t+1, t+2, t+3, and t+5.")
    lines.append("- Transition quality and outcome skew are mechanical descriptions only.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    studies = [study_instrument(path) for path in args.input_paths]
    seen = set()
    for study in studies:
        if study.instrument in seen:
            raise ValueError(f"Duplicate instrument input: {study.instrument}")
        seen.add(study.instrument)
        write_text(Path("Evidence") / "Output" / study.instrument / f"ArchetypeEvolution_{study.instrument}.txt", instrument_report(study))
    report = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, report)
    print(report, end="")


if __name__ == "__main__":
    main()
