"""APVA Family B Resolution Study v0.1B.

Research-only study of how the existing Study 30 Family B projection resolves.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

from APVA_AcceptanceAnatomy_23 import mature_aligned_lateral_indexes
from APVA_BreakoutContext_08 import EvidenceBar, direction_relative_return, load_rows
from APVA_ConditionTopology_27 import flags
from APVA_ContextTopology_28 import context_metrics
from APVA_FamilyEvolution_30 import FAMILIES, family_for, instrument_columns
from APVA_LateralAnatomy_19 import effect_size
from APVA_RegimeTransition_16 import instrument_name, write_text
from APVA_StateGrammar_25 import OutcomeStats, mean, summarize


AGGREGATE_OUTPUT = Path("Evidence/Output/FamilyBResolution/FamilyBResolution_All.txt")
RESOLUTIONS = ("B->A", "B->C", "B->D", "B->N", "B->B_Persist")
EXIT_FAMILIES = ("A", "C", "D", "N")
RUN_BUCKETS = ("1", "2", "3", "4", "5+", "10+")
BRANCH_COMPARISONS = (
    ("B->N", "B->D"),
    ("B->N", "B->C"),
    ("B->D", "B->C"),
    ("B->D", "B->B_Persist"),
    ("B->N", "B->B_Persist"),
)
TOP_LIMIT = 20
PER_INSTRUMENT_MIN_COUNT = 20

PARTICIPATION_VALUES = ("Falling", "Rising", "Peak", "Climactic", "Normal", "Silent")
ACCEPTANCE_VALUES = ("Accepted", "Contained", "Rejected", "Unresolved", "Unknown")
COMPRESSION_VALUES = ("Absent", "Local", "Clustered", "Lateral", "Resolving", "FailedResolution", "Unknown")
DISSIPATION_VALUES = ("Absent", "Local", "Repeated", "Strong", "Climactic", "Unknown")
EXPANSION_VALUES = ("Absent", "Local", "Strong", "Failed", "Climactic", "Unknown")
POLARITY_VALUES = ("Black", "Red", "Neutral")
POSITION_VALUES = ("First", "Middle", "Last", "Only")


@dataclass(frozen=True)
class RunInfo:
    start: int
    end: int
    so_far: int
    total: int
    position: str


@dataclass(frozen=True)
class ResolutionOccurrence:
    index: int
    resolution: str
    resolution_family: str
    bars_to_resolution: int
    features: dict[str, float]
    context: dict[str, float]
    outcome: float | None


@dataclass(frozen=True)
class ResolutionStats:
    resolution: str
    count: int
    percent: float
    mean_bars: float
    median_bars: float
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
class SourceStats:
    family: str
    count: int
    probability: float
    baseline_frequency: float
    lift: float | None


@dataclass(frozen=True)
class ExitOccurrence:
    run_start: int
    run_end: int
    run_length: int
    bucket: str
    exit_family: str
    exit_direction: str


@dataclass(frozen=True)
class ExitStats:
    bucket: str
    exit_family: str
    count: int
    percent: float


@dataclass(frozen=True)
class PopulationStudy:
    name: str
    occurrences: list[ResolutionOccurrence]
    resolutions: dict[str, ResolutionStats]
    branch_predictors: dict[str, list[FeatureComparison]]
    branch_pairs: dict[str, list[FeatureComparison]]
    sources: dict[str, SourceStats]
    exits: list[ExitOccurrence]
    exit_stats: dict[tuple[str, str], ExitStats]


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    rows: list[EvidenceBar]
    family_b_count: int
    populations: dict[str, PopulationStudy]


@dataclass(frozen=True)
class AggregateResolution:
    resolution: str
    values: dict[str, ResolutionStats]
    valid_instruments: int
    mean_percent: float
    mean_median_bars: float


@dataclass(frozen=True)
class AggregatePredictor:
    category: str
    feature: str
    values: dict[str, FeatureComparison]
    valid_instruments: int
    positive_count: int
    negative_count: int
    mean_effect: float


@dataclass(frozen=True)
class AggregateOutcome:
    resolution: str
    values: dict[str, ResolutionStats]
    valid_instruments: int
    positive_count: int
    negative_count: int
    mean_continuation: float
    mean_failure: float


@dataclass(frozen=True)
class AggregateSource:
    family: str
    values: dict[str, SourceStats]
    valid_instruments: int
    mean_probability: float
    mean_lift: float


@dataclass(frozen=True)
class AggregateExit:
    bucket: str
    exit_family: str
    values: dict[str, ExitStats]
    valid_instruments: int
    mean_percent: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study how the Study 30 APVA Family B projection resolves.")
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def family_b_runs(families: list[str]) -> tuple[dict[int, RunInfo], list[tuple[int, int]]]:
    output = {}
    runs = []
    index = 0
    while index < len(families):
        if families[index] != "B":
            index += 1
            continue
        end = index
        while end + 1 < len(families) and families[end + 1] == "B":
            end += 1
        runs.append((index, end))
        total = end - index + 1
        for current in range(index, end + 1):
            so_far = current - index + 1
            if total == 1:
                position = "Only"
            elif current == index:
                position = "First"
            elif current == end:
                position = "Last"
            else:
                position = "Middle"
            output[current] = RunInfo(index, end, so_far, total, position)
        index = end + 1
    return output, runs


def classify_resolution(families: list[str], index: int) -> tuple[str, str, int] | None:
    if index + 5 >= len(families):
        return None
    for offset in range(1, 6):
        future = families[index + offset]
        if future != "B":
            return f"B->{future}", future, offset
    return "B->B_Persist", "B", 5


def run_bucket(length: int) -> str:
    if length >= 10:
        return "10+"
    if length >= 5:
        return "5+"
    return str(length)


def indicator(value: str, expected: str) -> float:
    return float(value == expected)


def current_features(row: EvidenceBar, run: RunInfo) -> dict[str, float]:
    current_flags = flags(row)
    output = {f"CurrentFlag_{name}": float(value) for name, value in current_flags.items()}
    output.update(
        {
            "FamilyBRunLengthSoFar": float(run.so_far),
            "FamilyBTotalRunLength": float(run.total),
            "CompressionOn": float(row.compression != "Absent"),
            "DissipationOn": float(row.dissipation != "Absent"),
            "ExpansionOn": float(row.expansion != "Absent"),
        }
    )
    for value in POLARITY_VALUES:
        output[f"VolumePolarity_{value}"] = indicator(row.polarity, value)
    for value in PARTICIPATION_VALUES:
        output[f"ParticipationState_{value}"] = indicator(row.participation, value)
    for value in ACCEPTANCE_VALUES:
        output[f"AcceptanceState_{value}"] = indicator(row.acceptance, value)
    for value in COMPRESSION_VALUES:
        output[f"CompressionState_{value}"] = indicator(row.compression, value)
    for value in DISSIPATION_VALUES:
        output[f"DissipationState_{value}"] = indicator(row.dissipation, value)
    for value in EXPANSION_VALUES:
        output[f"ExpansionState_{value}"] = indicator(row.expansion, value)
    for value in POSITION_VALUES:
        output[f"PositionInRun_{value}"] = indicator(run.position, value)
    return output


def comparison(feature: str, left: list[float], right: list[float]) -> FeatureComparison:
    return FeatureComparison(feature, len(left), len(right), mean(left), mean(right), mean(left) - mean(right), effect_size(left, right))


def feature_names(occurrences: list[ResolutionOccurrence]) -> list[str]:
    if not occurrences:
        return []
    return sorted(set(occurrences[0].features) | set(occurrences[0].context))


def feature_value(occurrence: ResolutionOccurrence, feature: str) -> float:
    return occurrence.features.get(feature, occurrence.context.get(feature, 0.0))


def compare_groups(left: list[ResolutionOccurrence], right: list[ResolutionOccurrence], features: list[str]) -> list[FeatureComparison]:
    return [
        comparison(feature, [feature_value(item, feature) for item in left], [feature_value(item, feature) for item in right])
        for feature in features
    ]


def resolution_stats(occurrences: list[ResolutionOccurrence]) -> dict[str, ResolutionStats]:
    denominator = len(occurrences) if occurrences else 1
    output = {}
    for resolution in RESOLUTIONS:
        selected = [item for item in occurrences if item.resolution == resolution]
        bars = [item.bars_to_resolution for item in selected]
        output[resolution] = ResolutionStats(
            resolution,
            len(selected),
            len(selected) / denominator,
            mean(bars),
            statistics.median(bars) if bars else 0.0,
            summarize([item.outcome for item in selected if item.outcome is not None]),
        )
    return output


def source_stats(occurrences: list[ResolutionOccurrence], frequencies: dict[str, float]) -> dict[str, SourceStats]:
    selected = [item for item in occurrences if item.resolution_family in EXIT_FAMILIES]
    denominator = len(selected) if selected else 1
    output = {}
    for family in EXIT_FAMILIES:
        count = sum(item.resolution_family == family for item in selected)
        probability = count / denominator
        baseline = frequencies[family]
        output[family] = SourceStats(family, count, probability, baseline, probability / baseline if baseline else None)
    return output


def build_exits(rows: list[EvidenceBar], families: list[str], runs: list[tuple[int, int]]) -> list[ExitOccurrence]:
    output = []
    for start, end in runs:
        exit_index = end + 1
        if exit_index >= len(families):
            continue
        length = end - start + 1
        output.append(ExitOccurrence(start, end, length, run_bucket(length), families[exit_index], rows[exit_index].polarity))
    return output


def exit_stats(exits: list[ExitOccurrence]) -> dict[tuple[str, str], ExitStats]:
    output = {}
    for bucket in RUN_BUCKETS:
        bucket_total = sum(item.bucket == bucket for item in exits)
        for family in EXIT_FAMILIES:
            count = sum(item.bucket == bucket and item.exit_family == family for item in exits)
            output[(bucket, family)] = ExitStats(bucket, family, count, count / bucket_total if bucket_total else 0.0)
    return output


def build_population(
    name: str,
    occurrences: list[ResolutionOccurrence],
    frequencies: dict[str, float],
    exits: list[ExitOccurrence],
) -> PopulationStudy:
    features = feature_names(occurrences)
    branch_predictors = {
        resolution: compare_groups(
            [item for item in occurrences if item.resolution == resolution],
            [item for item in occurrences if item.resolution != resolution],
            features,
        )
        for resolution in RESOLUTIONS
    }
    branch_pairs = {
        f"{left} vs {right}": compare_groups(
            [item for item in occurrences if item.resolution == left],
            [item for item in occurrences if item.resolution == right],
            features,
        )
        for left, right in BRANCH_COMPARISONS
    }
    return PopulationStudy(name, occurrences, resolution_stats(occurrences), branch_predictors, branch_pairs, source_stats(occurrences, frequencies), exits, exit_stats(exits))


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    families = [family_for(row) for row in rows]
    frequencies = {family: families.count(family) / len(families) if families else 0.0 for family in FAMILIES}
    run_info, runs = family_b_runs(families)
    occurrences = []
    for index, family in enumerate(families):
        if family != "B":
            continue
        classified = classify_resolution(families, index)
        if classified is None:
            continue
        resolution, resolution_family, bars_to_resolution = classified
        occurrences.append(
            ResolutionOccurrence(
                index,
                resolution,
                resolution_family,
                bars_to_resolution,
                current_features(rows[index], run_info[index]),
                context_metrics(rows, index),
                direction_relative_return(rows, index, 5),
            )
        )
    exits = build_exits(rows, families, runs)
    population_b_indexes = set(mature_aligned_lateral_indexes(rows))
    return InstrumentStudy(
        instrument_name(path),
        path,
        rows,
        families.count("B"),
        {
            "Full Population": build_population("Full Population", occurrences, frequencies, exits),
            "Population B": build_population(
                "Population B",
                [item for item in occurrences if item.index in population_b_indexes],
                frequencies,
                [item for item in exits if item.run_end in population_b_indexes],
            ),
        },
    )


def append_resolution_table(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'Resolution':<16} {'Count':>8} {'Percent':>10} {'MeanBars':>10} {'MedianBars':>12}")
    for resolution in RESOLUTIONS:
        item = population.resolutions[resolution]
        lines.append(f"{resolution:<16} {item.count:>8} {item.percent:>9.2%} {item.mean_bars:>10.4f} {item.median_bars:>12.4f}")


def append_outcome_table(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'Resolution':<16} {'Count':>8} {'OutcomeN':>9} {'MeanDRFwd5':>12} {'Median':>12} {'ContRate5':>10} {'FailRate5':>10} {'FlatRate5':>10}")
    for resolution in RESOLUTIONS:
        item = population.resolutions[resolution]
        value = item.outcome
        lines.append(f"{resolution:<16} {item.count:>8} {value.count:>9} {value.mean:>12.6f} {value.median:>12.6f} {value.continuation_rate:>9.2%} {value.failure_rate:>9.2%} {value.flat_rate:>9.2%}")


def append_feature_means(lines: list[str], population: PopulationStudy, features: list[str]) -> None:
    lines.append(f"{'Feature':<40}" + "".join(f" {resolution:>14}" for resolution in RESOLUTIONS))
    for feature in features:
        values = [mean([feature_value(item, feature) for item in population.occurrences if item.resolution == resolution]) for resolution in RESOLUTIONS]
        lines.append(f"{feature:<40}" + "".join(f" {value:>14.6f}" for value in values))


def append_ranked_comparisons(lines: list[str], title: str, comparisons: list[FeatureComparison], limit: int = TOP_LIMIT) -> None:
    lines.extend([f"\n{title}", "." * len(title)])
    if not comparisons:
        lines.append("No observations.")
        return
    for rank, item in enumerate(sorted(comparisons, key=lambda value: (-abs(value.effect), value.feature))[:limit], start=1):
        lines.append(f"{rank:>3}. {item.feature:<40} LeftN={item.left_count:>6} RightN={item.right_count:>6} Delta={item.delta:>10.6f} Effect={item.effect:>10.6f}")


def append_source_table(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'GeneratedFamily':<18} {'Count':>8} {'Probability':>12} {'Baseline':>10} {'Lift':>10}")
    for family in EXIT_FAMILIES:
        item = population.sources[family]
        lift = f"{item.lift:.6f}" if item.lift is not None else "N/A"
        lines.append(f"{family:<18} {item.count:>8} {item.probability:>11.2%} {item.baseline_frequency:>9.2%} {lift:>10}")


def append_exit_table(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'Bucket':<8} {'ExitFamily':<12} {'Count':>8} {'Percent':>10}")
    for bucket in RUN_BUCKETS:
        for family in EXIT_FAMILIES:
            item = population.exit_stats[(bucket, family)]
            lines.append(f"{bucket:<8} {family:<12} {item.count:>8} {item.percent:>9.2%}")
    lines.extend(["\nExit Direction Counts", "---------------------"])
    directions = sorted(set(item.exit_direction for item in population.exits))
    for direction in directions:
        lines.append(f"{direction:<12} {sum(item.exit_direction == direction for item in population.exits):>8}")


def instrument_report(study: InstrumentStudy) -> str:
    full = study.populations["Full Population"]
    population_b = study.populations["Population B"]
    current = [feature for feature in feature_names(full.occurrences) if not feature.startswith("Prev")]
    historical = [feature for feature in feature_names(full.occurrences) if feature.startswith("Prev")]
    lines = [
        f"APVA Family B Resolution Study v0.1B - {study.instrument}",
        "=" * (39 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {len(study.rows)}",
        f"Family B count: {study.family_b_count}",
        f"Valid Family B resolution observations: {len(full.occurrences)}",
        "Family B resolution uses the existing Study 30 family precedence.",
        "",
        "Section 1 - Resolution Counts",
        "=============================",
    ]
    append_resolution_table(lines, full)
    lines.extend(["\nSection 2 - Current Family B Composition", "========================================"])
    append_feature_means(lines, full, current)
    lines.extend(["\nSection 3 - Historical Context", "=============================="])
    append_feature_means(lines, full, historical)
    lines.extend(["\nSection 4 - Branch Predictors", "============================="])
    for resolution in RESOLUTIONS:
        append_ranked_comparisons(lines, f"{resolution} vs all other Family B occurrences", full.branch_predictors[resolution])
    lines.extend(["\nSection 5 - Branch-vs-Branch", "============================"])
    for pair, comparisons in full.branch_pairs.items():
        append_ranked_comparisons(lines, pair, comparisons)
    lines.extend(["\nSection 6 - Outcome Layer", "========================="])
    append_outcome_table(lines, full)
    lines.extend(["\nSection 7 - Source Generation", "============================="])
    append_source_table(lines, full)
    lines.extend(["\nSection 8 - Attractor Exit Study", "================================"])
    append_exit_table(lines, full)
    lines.extend(["\nSection 9 - Population B", "========================"])
    lines.append(f"Population B Family B resolution observations: {len(population_b.occurrences)}")
    append_resolution_table(lines, population_b)
    lines.extend(["\nPopulation B Outcome Layer", "--------------------------"])
    append_outcome_table(lines, population_b)
    lines.extend(["\nPopulation B Source Generation", "------------------------------"])
    append_source_table(lines, population_b)
    lines.extend(["\nPopulation B Attractor Exit Study", "---------------------------------"])
    append_exit_table(lines, population_b)
    for resolution in RESOLUTIONS:
        append_ranked_comparisons(lines, f"Population B: {resolution} vs all other Family B occurrences", population_b.branch_predictors[resolution], limit=10)
    lines.extend(["\nSection 10 - Mechanical Research Notes", "======================================"])
    common = max(RESOLUTIONS, key=lambda name: full.resolutions[name].count)
    dominant_exit = max(EXIT_FAMILIES, key=lambda name: full.sources[name].probability)
    longest_bucket = max(RUN_BUCKETS, key=lambda bucket: sum(full.exit_stats[(bucket, family)].count for family in EXIT_FAMILIES))
    lines.append(f"- Most common resolution: {common}, Count={full.resolutions[common].count}.")
    lines.append(f"- Dominant generated future family: {dominant_exit}, Probability={full.sources[dominant_exit].probability:.2%}.")
    lines.append(f"- Most populated run-length bucket: {longest_bucket}.")
    lines.append(f"- Population B contains {len(population_b.occurrences)} valid Family B resolution observations.")
    lines.append("- Resolution, source generation, and exit statistics are descriptive only.")
    return "\n".join(lines) + "\n"


def aggregate_resolutions(studies: list[InstrumentStudy], population: str = "Full Population") -> list[AggregateResolution]:
    output = []
    for resolution in RESOLUTIONS:
        values = {study.instrument: study.populations[population].resolutions[resolution] for study in studies}
        valid = [value for value in values.values() if value.count >= PER_INSTRUMENT_MIN_COUNT]
        output.append(AggregateResolution(resolution, values, len(valid), mean([value.percent for value in valid]), mean([value.median_bars for value in valid])))
    return output


def aggregate_predictors(studies: list[InstrumentStudy], attribute: str) -> list[AggregatePredictor]:
    categories = RESOLUTIONS if attribute == "branch_predictors" else tuple(f"{left} vs {right}" for left, right in BRANCH_COMPARISONS)
    output = []
    for category in categories:
        per_study = {study.instrument: {item.feature: item for item in getattr(study.populations["Full Population"], attribute)[category]} for study in studies}
        features = sorted(set().union(*(mapping for mapping in per_study.values())))
        for feature in features:
            values = {name: mapping[feature] for name, mapping in per_study.items() if feature in mapping}
            valid = [value for value in values.values() if value.left_count >= PER_INSTRUMENT_MIN_COUNT and value.right_count >= PER_INSTRUMENT_MIN_COUNT]
            output.append(AggregatePredictor(category, feature, values, len(valid), sum(value.effect > 0.0 for value in valid), sum(value.effect < 0.0 for value in valid), mean([value.effect for value in valid])))
    return output


def aggregate_outcomes(studies: list[InstrumentStudy]) -> list[AggregateOutcome]:
    output = []
    for resolution in RESOLUTIONS:
        values = {study.instrument: study.populations["Full Population"].resolutions[resolution] for study in studies}
        valid = [value.outcome for value in values.values() if value.outcome.count >= PER_INSTRUMENT_MIN_COUNT]
        output.append(AggregateOutcome(resolution, values, len(valid), sum(value.continuation_rate > 0.5 for value in valid), sum(value.continuation_rate < 0.5 for value in valid), mean([value.continuation_rate for value in valid]), mean([value.failure_rate for value in valid])))
    return output


def aggregate_sources(studies: list[InstrumentStudy]) -> list[AggregateSource]:
    output = []
    for family in EXIT_FAMILIES:
        values = {study.instrument: study.populations["Full Population"].sources[family] for study in studies}
        valid = [value for value in values.values() if value.count >= PER_INSTRUMENT_MIN_COUNT]
        output.append(AggregateSource(family, values, len(valid), mean([value.probability for value in valid]), mean([value.lift for value in valid if value.lift is not None])))
    return output


def aggregate_exits(studies: list[InstrumentStudy]) -> list[AggregateExit]:
    output = []
    for bucket in RUN_BUCKETS:
        for family in EXIT_FAMILIES:
            values = {study.instrument: study.populations["Full Population"].exit_stats[(bucket, family)] for study in studies}
            valid = [value for value in values.values() if value.count >= PER_INSTRUMENT_MIN_COUNT]
            output.append(AggregateExit(bucket, family, values, len(valid), mean([value.percent for value in valid])))
    return output


def append_aggregate_resolution_table(lines: list[str], items: list[AggregateResolution], columns: list[str]) -> None:
    lines.append(f"{'Resolution':<16}" + "".join(f" {name + '_Count':>12} {name + '_Pct':>10} {name + '_Med':>9}" for name in columns) + " ValidN    MeanPct  MeanMedian")
    for item in items:
        cells = [f" {item.values[name].count if name in item.values else 0:>12} {item.values[name].percent if name in item.values else 0.0:>9.2%} {item.values[name].median_bars if name in item.values else 0.0:>9.4f}" for name in columns]
        lines.append(f"{item.resolution:<16}{''.join(cells)} {item.valid_instruments:>6} {item.mean_percent:>9.2%} {item.mean_median_bars:>11.4f}")


def append_aggregate_predictor_table(lines: list[str], items: list[AggregatePredictor], columns: list[str]) -> None:
    lines.append(f"{'Category':<24} {'Feature':<40}" + "".join(f" {name + '_Effect':>12}" for name in columns) + " ValidN  PosN  NegN  MeanEffect")
    for item in items:
        cells = "".join(f" {item.values[name].effect if name in item.values else 0.0:>12.6f}" for name in columns)
        lines.append(f"{item.category:<24} {item.feature:<40}{cells} {item.valid_instruments:>6} {item.positive_count:>5} {item.negative_count:>5} {item.mean_effect:>11.6f}")


def append_aggregate_outcome_table(lines: list[str], items: list[AggregateOutcome], columns: list[str]) -> None:
    lines.append(f"{'Resolution':<16}" + "".join(f" {name + '_N':>9} {name + '_Cont':>10} {name + '_Fail':>10} {name + '_Mean':>11}" for name in columns) + " ValidN  PosN  NegN  MeanCont  MeanFail")
    for item in items:
        cells = []
        for name in columns:
            outcome = item.values[name].outcome if name in item.values else summarize([])
            cells.append(f" {outcome.count:>9} {outcome.continuation_rate:>9.2%} {outcome.failure_rate:>9.2%} {outcome.mean:>11.6f}")
        lines.append(f"{item.resolution:<16}{''.join(cells)} {item.valid_instruments:>6} {item.positive_count:>5} {item.negative_count:>5} {item.mean_continuation:>9.2%} {item.mean_failure:>9.2%}")


def append_aggregate_source_table(lines: list[str], items: list[AggregateSource], columns: list[str]) -> None:
    lines.append(f"{'Family':<10}" + "".join(f" {name + '_N':>8} {name + '_Prob':>10} {name + '_Lift':>10}" for name in columns) + " ValidN  MeanProb  MeanLift")
    for item in items:
        cells = []
        for name in columns:
            value = item.values.get(name)
            cells.append(f" {value.count if value else 0:>8} {value.probability if value else 0.0:>9.2%} {value.lift if value and value.lift is not None else 0.0:>10.6f}")
        lines.append(f"{item.family:<10}{''.join(cells)} {item.valid_instruments:>6} {item.mean_probability:>9.2%} {item.mean_lift:>9.6f}")


def append_aggregate_exit_table(lines: list[str], items: list[AggregateExit], columns: list[str]) -> None:
    lines.append(f"{'Bucket':<8} {'ExitFamily':<12}" + "".join(f" {name + '_N':>8} {name + '_Pct':>10}" for name in columns) + " ValidN   MeanPct")
    for item in items:
        cells = "".join(f" {item.values[name].count if name in item.values else 0:>8} {item.values[name].percent if name in item.values else 0.0:>9.2%}" for name in columns)
        lines.append(f"{item.bucket:<8} {item.exit_family:<12}{cells} {item.valid_instruments:>6} {item.mean_percent:>9.2%}")


def append_ranked_aggregate(lines: list[str], title: str, items: list, key, formatter, reverse: bool = True) -> None:
    lines.extend([f"\n{title}", "." * len(title)])
    eligible = [item for item in items if item.valid_instruments >= 2]
    if not eligible:
        lines.append("No items met the two-instrument minimum.")
        return
    for rank, item in enumerate(sorted(eligible, key=key, reverse=reverse)[:TOP_LIMIT], start=1):
        lines.append(f"{rank:>3}. {formatter(item)}")


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    columns = instrument_columns(studies)
    resolutions = aggregate_resolutions(studies)
    predictors = aggregate_predictors(studies, "branch_predictors")
    pair_predictors = aggregate_predictors(studies, "branch_pairs")
    outcomes = aggregate_outcomes(studies)
    sources = aggregate_sources(studies)
    exits = aggregate_exits(studies)
    population_b = aggregate_resolutions(studies, "Population B")
    lines = [
        "APVA Family B Resolution Study v0.1B - Aggregate",
        "================================================",
        f"Instruments: {', '.join(study.instrument for study in studies)}",
        "Family B resolution uses the existing Study 30 family projection.",
        "",
        "Aggregate Resolution Counts",
        "===========================",
    ]
    append_aggregate_resolution_table(lines, resolutions, columns)
    lines.extend(["\nAggregate Branch Predictors", "==========================="])
    append_aggregate_predictor_table(lines, predictors, columns)
    lines.extend(["\nAggregate Branch-vs-Branch", "=========================="])
    append_aggregate_predictor_table(lines, pair_predictors, columns)
    lines.extend(["\nAggregate Outcome Layer", "======================="])
    append_aggregate_outcome_table(lines, outcomes, columns)
    lines.extend(["\nAggregate Source Generation", "==========================="])
    append_aggregate_source_table(lines, sources, columns)
    lines.extend(["\nAggregate Attractor Exit", "========================"])
    append_aggregate_exit_table(lines, exits, columns)
    lines.extend(["\nAggregate Rankings", "=================="])
    append_ranked_aggregate(lines, "1. Most common B resolutions", resolutions, lambda item: (item.mean_percent, item.resolution), lambda item: f"{item.resolution:<16} MeanPct={item.mean_percent:.2%} ValidN={item.valid_instruments}")
    append_ranked_aggregate(lines, "2. Fastest B resolutions", resolutions, lambda item: (item.mean_median_bars, item.resolution), lambda item: f"{item.resolution:<16} MeanMedianBars={item.mean_median_bars:.4f} ValidN={item.valid_instruments}", reverse=False)
    for number, branch in ((3, "B->N"), (4, "B->D"), (5, "B->C"), (6, "B->B_Persist")):
        append_ranked_aggregate(lines, f"{number}. Strongest predictors of {branch}", [item for item in predictors if item.category == branch], lambda item: (abs(item.mean_effect), item.feature), lambda item: f"{item.feature:<40} MeanEffect={item.mean_effect:>10.6f} PosN={item.positive_count} NegN={item.negative_count} ValidN={item.valid_instruments}")
    append_ranked_aggregate(lines, "7. Strongest attractor exits", sources, lambda item: (item.mean_lift, item.family), lambda item: f"B->{item.family:<12} MeanProb={item.mean_probability:.2%} MeanLift={item.mean_lift:.6f} ValidN={item.valid_instruments}")
    append_ranked_aggregate(lines, "8. Best outcome resolutions", outcomes, lambda item: (item.mean_continuation, item.resolution), lambda item: f"{item.resolution:<16} MeanCont={item.mean_continuation:.2%} MeanFail={item.mean_failure:.2%} ValidN={item.valid_instruments}")
    append_ranked_aggregate(lines, "9. Worst outcome resolutions", outcomes, lambda item: (item.mean_continuation, item.resolution), lambda item: f"{item.resolution:<16} MeanCont={item.mean_continuation:.2%} MeanFail={item.mean_failure:.2%} ValidN={item.valid_instruments}", reverse=False)
    full_by_name = {item.resolution: item for item in resolutions}
    lines.extend(["\n10. Population-B differences", "." * 28])
    for item in sorted(population_b, key=lambda value: (-abs(value.mean_percent - full_by_name[value.resolution].mean_percent), value.resolution)):
        lines.append(f"{item.resolution:<16} DeltaMeanPct={item.mean_percent - full_by_name[item.resolution].mean_percent:>10.2%} PopulationBValidN={item.valid_instruments}")
    lines.extend(["\nCross-Instrument Mechanical Research Notes", "=========================================="])
    common = max(resolutions, key=lambda item: item.mean_percent)
    dominant = max(sources, key=lambda item: item.mean_probability)
    strongest = max(sources, key=lambda item: item.mean_lift)
    lines.append(f"- Most common replicated B resolution: {common.resolution}, MeanPct={common.mean_percent:.2%}.")
    lines.append(f"- Dominant generated future family: {dominant.family}, MeanProbability={dominant.mean_probability:.2%}.")
    lines.append(f"- Strongest generated-family lift: B->{strongest.family}, MeanLift={strongest.mean_lift:.6f}.")
    lines.append("- Resolution, source generation, attractor exits, and Population B differences are descriptive only.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    studies = [study_instrument(path) for path in args.input_paths]
    seen = set()
    for study in studies:
        if study.instrument in seen:
            raise ValueError(f"Duplicate instrument input: {study.instrument}")
        seen.add(study.instrument)
        write_text(Path("Evidence") / "Output" / study.instrument / f"FamilyBResolution_{study.instrument}.txt", instrument_report(study))
    report = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, report)
    print(report, end="")


if __name__ == "__main__":
    main()
