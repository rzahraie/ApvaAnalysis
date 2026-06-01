"""APVA Family Resolution Study v0.1.

Research-only study of how the existing Study 30 Family C projection resolves.
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
from APVA_FamilyEvolution_30 import FEATURES, family_for, instrument_columns
from APVA_LateralAnatomy_19 import effect_size
from APVA_RegimeTransition_16 import instrument_name, write_text
from APVA_StateGrammar_25 import OutcomeStats, mean, summarize


AGGREGATE_OUTPUT = Path("Evidence/Output/FamilyResolution/FamilyResolution_All.txt")
RESOLUTIONS = ("C->A", "C->B", "C->D", "C->N", "C->C_Persist")
BRANCH_COMPARISONS = (
    ("C->B", "C->D"),
    ("C->B", "C->N"),
    ("C->B", "C->C_Persist"),
    ("C->D", "C->N"),
    ("C->D", "C->C_Persist"),
    ("C->N", "C->C_Persist"),
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
class PopulationStudy:
    name: str
    occurrences: list[ResolutionOccurrence]
    resolutions: dict[str, ResolutionStats]
    branch_predictors: dict[str, list[FeatureComparison]]
    branch_pairs: dict[str, list[FeatureComparison]]


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    rows: list[EvidenceBar]
    family_c_count: int
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study how the Study 30 APVA Family C projection resolves.")
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def family_c_runs(families: list[str]) -> dict[int, RunInfo]:
    output = {}
    index = 0
    while index < len(families):
        if families[index] != "C":
            index += 1
            continue
        end = index
        while end + 1 < len(families) and families[end + 1] == "C":
            end += 1
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
            output[current] = RunInfo(so_far, total, position)
        index = end + 1
    return output


def classify_resolution(families: list[str], index: int) -> tuple[str, str, int] | None:
    if index + 5 >= len(families):
        return None
    for offset in range(1, 6):
        future = families[index + offset]
        if future != "C":
            return f"C->{future}", future, offset
    return "C->C_Persist", "C", 5


def indicator(value: str, expected: str) -> float:
    return float(value == expected)


def current_features(row: EvidenceBar, run: RunInfo) -> dict[str, float]:
    current_flags = flags(row)
    output = {
        f"CurrentFlag_{name}": float(value)
        for name, value in current_flags.items()
    }
    output.update(
        {
            "FamilyCRunLengthSoFar": float(run.so_far),
            "FamilyCTotalRunLength": float(run.total),
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
    return FeatureComparison(
        feature,
        len(left),
        len(right),
        mean(left),
        mean(right),
        mean(left) - mean(right),
        effect_size(left, right),
    )


def feature_names(occurrences: list[ResolutionOccurrence]) -> list[str]:
    if not occurrences:
        return []
    return sorted(set(occurrences[0].features) | set(occurrences[0].context))


def feature_value(occurrence: ResolutionOccurrence, feature: str) -> float:
    return occurrence.features.get(feature, occurrence.context.get(feature, 0.0))


def compare_groups(
    left: list[ResolutionOccurrence],
    right: list[ResolutionOccurrence],
    features: list[str],
) -> list[FeatureComparison]:
    return [
        comparison(
            feature,
            [feature_value(item, feature) for item in left],
            [feature_value(item, feature) for item in right],
        )
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


def build_population(name: str, occurrences: list[ResolutionOccurrence]) -> PopulationStudy:
    features = feature_names(occurrences)
    branch_predictors = {}
    for resolution in RESOLUTIONS:
        branch_predictors[resolution] = compare_groups(
            [item for item in occurrences if item.resolution == resolution],
            [item for item in occurrences if item.resolution != resolution],
            features,
        )
    branch_pairs = {}
    for left, right in BRANCH_COMPARISONS:
        branch_pairs[f"{left} vs {right}"] = compare_groups(
            [item for item in occurrences if item.resolution == left],
            [item for item in occurrences if item.resolution == right],
            features,
        )
    return PopulationStudy(name, occurrences, resolution_stats(occurrences), branch_predictors, branch_pairs)


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    families = [family_for(row) for row in rows]
    runs = family_c_runs(families)
    occurrences = []
    for index, family in enumerate(families):
        if family != "C":
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
                current_features(rows[index], runs[index]),
                context_metrics(rows, index),
                direction_relative_return(rows, index, 5),
            )
        )
    population_b_indexes = set(mature_aligned_lateral_indexes(rows))
    return InstrumentStudy(
        instrument_name(path),
        path,
        rows,
        sum(family == "C" for family in families),
        {
            "Full Population": build_population("Full Population", occurrences),
            "Population B": build_population(
                "Population B",
                [item for item in occurrences if item.index in population_b_indexes],
            ),
        },
    )


def append_resolution_table(lines: list[str], population: PopulationStudy) -> None:
    lines.append(
        f"{'Resolution':<16} {'Count':>8} {'Percent':>10} {'MeanBars':>10} "
        f"{'MedianBars':>12}"
    )
    for resolution in RESOLUTIONS:
        item = population.resolutions[resolution]
        lines.append(
            f"{resolution:<16} {item.count:>8} {item.percent:>9.2%} "
            f"{item.mean_bars:>10.4f} {item.median_bars:>12.4f}"
        )


def append_outcome_table(lines: list[str], population: PopulationStudy) -> None:
    lines.append(
        f"{'Resolution':<16} {'Count':>8} {'OutcomeN':>9} {'MeanDRFwd5':>12} "
        f"{'Median':>12} {'ContRate5':>10} {'FailRate5':>10} {'FlatRate5':>10}"
    )
    for resolution in RESOLUTIONS:
        item = population.resolutions[resolution]
        outcome = item.outcome
        lines.append(
            f"{resolution:<16} {item.count:>8} {outcome.count:>9} {outcome.mean:>12.6f} "
            f"{outcome.median:>12.6f} {outcome.continuation_rate:>9.2%} "
            f"{outcome.failure_rate:>9.2%} {outcome.flat_rate:>9.2%}"
        )


def append_feature_means(lines: list[str], population: PopulationStudy, features: list[str]) -> None:
    lines.append(f"{'Feature':<40}" + "".join(f" {resolution:>14}" for resolution in RESOLUTIONS))
    for feature in features:
        values = []
        for resolution in RESOLUTIONS:
            selected = [
                feature_value(item, feature)
                for item in population.occurrences
                if item.resolution == resolution
            ]
            values.append(mean(selected))
        lines.append(f"{feature:<40}" + "".join(f" {value:>14.6f}" for value in values))


def append_ranked_comparisons(
    lines: list[str],
    title: str,
    comparisons: list[FeatureComparison],
    limit: int = TOP_LIMIT,
) -> None:
    lines.extend([f"\n{title}", "." * len(title)])
    if not comparisons:
        lines.append("No observations.")
        return
    for rank, item in enumerate(sorted(comparisons, key=lambda value: (-abs(value.effect), value.feature))[:limit], start=1):
        lines.append(
            f"{rank:>3}. {item.feature:<40} LeftN={item.left_count:>6} RightN={item.right_count:>6} "
            f"Delta={item.delta:>10.6f} Effect={item.effect:>10.6f}"
        )


def instrument_report(study: InstrumentStudy) -> str:
    full = study.populations["Full Population"]
    population_b = study.populations["Population B"]
    current = [feature for feature in feature_names(full.occurrences) if not feature.startswith("Prev")]
    historical = [feature for feature in feature_names(full.occurrences) if feature.startswith("Prev")]
    lines = [
        f"APVA Family Resolution Study v0.1 - {study.instrument}",
        "=" * (35 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {len(study.rows)}",
        f"Family C count: {study.family_c_count}",
        f"Valid Family C resolution observations: {len(full.occurrences)}",
        "Family C resolution uses the existing Study 30 family precedence.",
        "",
        "Section 1 - Resolution Classification Table",
        "===========================================",
    ]
    append_resolution_table(lines, full)
    lines.extend(["\nSection 2 - Current Feature Distribution by Resolution", "======================================================"])
    append_feature_means(lines, full, current)
    lines.extend(["\nSection 3 - Historical Context by Resolution", "============================================"])
    append_feature_means(lines, full, historical)
    lines.extend(["\nSection 4 - Branch Comparison Rankings", "======================================"])
    for resolution in RESOLUTIONS:
        append_ranked_comparisons(lines, f"{resolution} vs all other Family C occurrences", full.branch_predictors[resolution])
    lines.extend(["\nSection 5 - Branch-vs-Branch Rankings", "====================================="])
    for pair, comparisons in full.branch_pairs.items():
        append_ranked_comparisons(lines, pair, comparisons)
    lines.extend(["\nSection 6 - Resolution Outcome Layer", "===================================="])
    append_outcome_table(lines, full)
    lines.extend(["\nSection 7 - Population B Summary", "================================"])
    lines.append(f"Population B Family C resolution observations: {len(population_b.occurrences)}")
    append_resolution_table(lines, population_b)
    lines.extend(["\nPopulation B Resolution Outcome Layer", "-------------------------------------"])
    append_outcome_table(lines, population_b)
    for resolution in RESOLUTIONS:
        append_ranked_comparisons(
            lines,
            f"Population B: {resolution} vs all other Family C occurrences",
            population_b.branch_predictors[resolution],
            limit=10,
        )
    lines.extend(["\nSection 8 - Mechanical Research Notes", "====================================="])
    most_common = max(RESOLUTIONS, key=lambda name: full.resolutions[name].count)
    fastest = min(
        (name for name in RESOLUTIONS if full.resolutions[name].count),
        key=lambda name: full.resolutions[name].median_bars,
        default="N/A",
    )
    best = max(RESOLUTIONS, key=lambda name: full.resolutions[name].outcome.continuation_rate)
    worst = min(RESOLUTIONS, key=lambda name: full.resolutions[name].outcome.continuation_rate)
    lines.append(f"- Most common resolution: {most_common}, Count={full.resolutions[most_common].count}.")
    lines.append(f"- Fastest median resolution: {fastest}.")
    lines.append(f"- Highest continuation resolution: {best}, ContinuationRate5={full.resolutions[best].outcome.continuation_rate:.2%}.")
    lines.append(f"- Lowest continuation resolution: {worst}, ContinuationRate5={full.resolutions[worst].outcome.continuation_rate:.2%}.")
    lines.append(f"- Population B contains {len(population_b.occurrences)} valid Family C resolution observations.")
    lines.append("- Branch comparisons, outcomes, and context effects are descriptive only.")
    return "\n".join(lines) + "\n"


def aggregate_resolutions(studies: list[InstrumentStudy], population: str = "Full Population") -> list[AggregateResolution]:
    output = []
    for resolution in RESOLUTIONS:
        values = {study.instrument: study.populations[population].resolutions[resolution] for study in studies}
        valid = [value for value in values.values() if value.count >= PER_INSTRUMENT_MIN_COUNT]
        output.append(
            AggregateResolution(
                resolution,
                values,
                len(valid),
                mean([value.percent for value in valid]),
                mean([value.median_bars for value in valid]),
            )
        )
    return output


def aggregate_predictors(
    studies: list[InstrumentStudy],
    attribute: str,
    population: str = "Full Population",
) -> list[AggregatePredictor]:
    categories = RESOLUTIONS if attribute == "branch_predictors" else tuple(
        f"{left} vs {right}" for left, right in BRANCH_COMPARISONS
    )
    output = []
    for category in categories:
        per_study = {
            study.instrument: {
                item.feature: item
                for item in getattr(study.populations[population], attribute)[category]
            }
            for study in studies
        }
        features = sorted(set().union(*(mapping for mapping in per_study.values())))
        for feature in features:
            values = {name: mapping[feature] for name, mapping in per_study.items() if feature in mapping}
            valid = [
                value
                for value in values.values()
                if value.left_count >= PER_INSTRUMENT_MIN_COUNT and value.right_count >= PER_INSTRUMENT_MIN_COUNT
            ]
            output.append(
                AggregatePredictor(
                    category,
                    feature,
                    values,
                    len(valid),
                    sum(value.effect > 0.0 for value in valid),
                    sum(value.effect < 0.0 for value in valid),
                    mean([value.effect for value in valid]),
                )
            )
    return output


def aggregate_outcomes(studies: list[InstrumentStudy], population: str = "Full Population") -> list[AggregateOutcome]:
    output = []
    for resolution in RESOLUTIONS:
        values = {study.instrument: study.populations[population].resolutions[resolution] for study in studies}
        valid = [value.outcome for value in values.values() if value.outcome.count >= PER_INSTRUMENT_MIN_COUNT]
        output.append(
            AggregateOutcome(
                resolution,
                values,
                len(valid),
                sum(value.continuation_rate > 0.5 for value in valid),
                sum(value.continuation_rate < 0.5 for value in valid),
                mean([value.continuation_rate for value in valid]),
                mean([value.failure_rate for value in valid]),
            )
        )
    return output


def append_aggregate_resolution_table(
    lines: list[str],
    items: list[AggregateResolution],
    columns: list[str],
) -> None:
    lines.append(f"{'Resolution':<16}" + "".join(f" {name + '_Count':>12} {name + '_Pct':>10} {name + '_Med':>9}" for name in columns) + " ValidN    MeanPct  MeanMedian")
    for item in items:
        cells = []
        for name in columns:
            value = item.values.get(name)
            cells.append(
                f" {value.count if value else 0:>12} {value.percent if value else 0.0:>9.2%} "
                f"{value.median_bars if value else 0.0:>9.4f}"
            )
        lines.append(f"{item.resolution:<16}{''.join(cells)} {item.valid_instruments:>6} {item.mean_percent:>9.2%} {item.mean_median_bars:>11.4f}")


def append_aggregate_predictor_table(
    lines: list[str],
    items: list[AggregatePredictor],
    columns: list[str],
) -> None:
    lines.append(f"{'Category':<24} {'Feature':<40}" + "".join(f" {name + '_Effect':>12}" for name in columns) + " ValidN  PosN  NegN  MeanEffect")
    for item in items:
        cells = "".join(f" {item.values[name].effect if name in item.values else 0.0:>12.6f}" for name in columns)
        lines.append(f"{item.category:<24} {item.feature:<40}{cells} {item.valid_instruments:>6} {item.positive_count:>5} {item.negative_count:>5} {item.mean_effect:>11.6f}")


def append_aggregate_outcome_table(
    lines: list[str],
    items: list[AggregateOutcome],
    columns: list[str],
) -> None:
    lines.append(f"{'Resolution':<16}" + "".join(f" {name + '_N':>9} {name + '_Cont':>10} {name + '_Fail':>10} {name + '_Mean':>11}" for name in columns) + " ValidN  PosN  NegN  MeanCont  MeanFail")
    for item in items:
        cells = []
        for name in columns:
            value = item.values.get(name)
            outcome = value.outcome if value else summarize([])
            cells.append(f" {outcome.count:>9} {outcome.continuation_rate:>9.2%} {outcome.failure_rate:>9.2%} {outcome.mean:>11.6f}")
        lines.append(f"{item.resolution:<16}{''.join(cells)} {item.valid_instruments:>6} {item.positive_count:>5} {item.negative_count:>5} {item.mean_continuation:>9.2%} {item.mean_failure:>9.2%}")


def append_ranked_aggregate(
    lines: list[str],
    title: str,
    items: list,
    key,
    formatter,
    reverse: bool = True,
) -> None:
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
    population_b_resolutions = aggregate_resolutions(studies, "Population B")
    lines = [
        "APVA Family Resolution Study v0.1 - Aggregate",
        "==============================================",
        f"Instruments: {', '.join(study.instrument for study in studies)}",
        "Family C resolution uses the existing Study 30 family projection.",
        "",
        "Aggregate Resolution Counts",
        "===========================",
    ]
    append_aggregate_resolution_table(lines, resolutions, columns)
    lines.extend(["\nAggregate Branch Predictor Table", "================================"])
    append_aggregate_predictor_table(lines, predictors, columns)
    lines.extend(["\nAggregate Branch-vs-Branch Table", "================================"])
    append_aggregate_predictor_table(lines, pair_predictors, columns)
    lines.extend(["\nAggregate Outcome Table", "======================="])
    append_aggregate_outcome_table(lines, outcomes, columns)
    lines.extend(["\nAggregate Population-B Resolution Counts", "========================================"])
    append_aggregate_resolution_table(lines, population_b_resolutions, columns)
    lines.extend(["\nAggregate Rankings", "=================="])
    append_ranked_aggregate(lines, "1. Most common Family C resolutions", resolutions, lambda item: (item.mean_percent, item.resolution), lambda item: f"{item.resolution:<16} MeanPct={item.mean_percent:.2%} ValidN={item.valid_instruments}")
    append_ranked_aggregate(lines, "2. Fastest Family C resolutions", resolutions, lambda item: (item.mean_median_bars, item.resolution), lambda item: f"{item.resolution:<16} MeanMedianBars={item.mean_median_bars:.4f} ValidN={item.valid_instruments}", reverse=False)
    for number, branch in ((3, "C->B"), (4, "C->D"), (5, "C->N"), (6, "C->C_Persist")):
        append_ranked_aggregate(lines, f"{number}. Strongest predictors of {branch}", [item for item in predictors if item.category == branch], lambda item: (abs(item.mean_effect), item.feature), lambda item: f"{item.feature:<40} MeanEffect={item.mean_effect:>10.6f} PosN={item.positive_count} NegN={item.negative_count} ValidN={item.valid_instruments}")
    for number, pair in ((7, "C->B vs C->D"), (8, "C->B vs C->N")):
        append_ranked_aggregate(lines, f"{number}. Strongest {pair} differentiators", [item for item in pair_predictors if item.category == pair], lambda item: (abs(item.mean_effect), item.feature), lambda item: f"{item.feature:<40} MeanEffect={item.mean_effect:>10.6f} PosN={item.positive_count} NegN={item.negative_count} ValidN={item.valid_instruments}")
    append_ranked_aggregate(lines, "9. Best resolution outcomes", outcomes, lambda item: (item.mean_continuation, item.resolution), lambda item: f"{item.resolution:<16} MeanCont={item.mean_continuation:.2%} MeanFail={item.mean_failure:.2%} ValidN={item.valid_instruments}")
    append_ranked_aggregate(lines, "10. Worst resolution outcomes", outcomes, lambda item: (item.mean_continuation, item.resolution), lambda item: f"{item.resolution:<16} MeanCont={item.mean_continuation:.2%} MeanFail={item.mean_failure:.2%} ValidN={item.valid_instruments}", reverse=False)
    population_b_delta = []
    full_by_name = {item.resolution: item for item in resolutions}
    for item in population_b_resolutions:
        baseline = full_by_name[item.resolution]
        population_b_delta.append((item.resolution, item.valid_instruments, item.mean_percent - baseline.mean_percent))
    lines.extend(["\n11. Population-B differences", "." * 28])
    for resolution, valid, delta in sorted(population_b_delta, key=lambda item: (-abs(item[2]), item[0])):
        lines.append(f"{resolution:<16} DeltaMeanPct={delta:>10.2%} PopulationBValidN={valid}")
    lines.extend(["\nCross-Instrument Mechanical Research Notes", "=========================================="])
    common = max(resolutions, key=lambda item: item.mean_percent)
    fastest = min((item for item in resolutions if item.valid_instruments), key=lambda item: item.mean_median_bars, default=None)
    best = max(outcomes, key=lambda item: item.mean_continuation)
    worst = min(outcomes, key=lambda item: item.mean_continuation)
    lines.append(f"- Most common replicated resolution: {common.resolution}, MeanPct={common.mean_percent:.2%}.")
    lines.append(f"- Fastest replicated resolution: {fastest.resolution if fastest else 'N/A'}.")
    lines.append(f"- Highest replicated continuation outcome: {best.resolution}, MeanContinuation={best.mean_continuation:.2%}.")
    lines.append(f"- Lowest replicated continuation outcome: {worst.resolution}, MeanContinuation={worst.mean_continuation:.2%}.")
    lines.append("- Resolution, branch predictors, outcomes, and Population B differences are descriptive only.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    studies = [study_instrument(path) for path in args.input_paths]
    seen = set()
    for study in studies:
        if study.instrument in seen:
            raise ValueError(f"Duplicate instrument input: {study.instrument}")
        seen.add(study.instrument)
        write_text(
            Path("Evidence") / "Output" / study.instrument / f"FamilyResolution_{study.instrument}.txt",
            instrument_report(study),
        )
    report = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, report)
    print(report, end="")


if __name__ == "__main__":
    main()
