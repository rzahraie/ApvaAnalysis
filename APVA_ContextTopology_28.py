"""APVA Context Topology Study v0.1.

Research-only historical-context study over Study 27 condition combinations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from APVA_AcceptanceAnatomy_23 import mature_aligned_lateral_indexes
from APVA_BreakoutContext_08 import EvidenceBar, direction_relative_return, load_rows
from APVA_ConditionTopology_27 import COMBINATIONS, combo_name, flags, matches
from APVA_LateralAnatomy_19 import effect_size
from APVA_RegimeTransition_16 import CANONICAL_INSTRUMENTS, instrument_name, write_text
from APVA_StateGrammar_25 import mean


AGGREGATE_OUTPUT = Path("Evidence/Output/ContextTopology/ContextTopology_All.txt")
TOP_LIMIT = 25
PER_INSTRUMENT_MIN_COUNT = 20
CONTEXT_WINDOWS = (5, 10, 20)
BASE_CONDITIONS = ("Accepted", "Compressed", "Dissipating", "Expanding", "Peak", "Climactic")
SLOPE_CONDITIONS = ("Accepted", "Compressed", "Dissipating", "Peak", "Dominance")
CLUSTERS = ("AcceptanceHeavy", "CompressionHeavy", "DissipationHeavy", "PeakHeavy", "Mixed")

POSITIVE_NAMES = (
    "Accepted+Compressed",
    "Accepted+Compressed+Dissipating",
    "Compressed+Dissipating",
    "Compressed+Peak",
    "Compressed+Dissipating+Peak",
)
NEGATIVE_NAMES = (
    "Accepted+Peak",
    "Accepted+Climactic",
    "Accepted+Expanding",
    "Accepted+Dissipating",
    "Accepted+Dissipating+Peak",
)
TARGET_NAMES = POSITIVE_NAMES + NEGATIVE_NAMES
IMPORTED_COMBINATIONS = {combo_name(conditions): conditions for conditions in COMBINATIONS}
TARGETS = {
    name: IMPORTED_COMBINATIONS[name]
    for name in TARGET_NAMES
    if name in IMPORTED_COMBINATIONS
}


@dataclass(frozen=True)
class Occurrence:
    combination: str
    group: str
    index: int
    outcome: float
    metrics: dict[str, float]
    slopes: dict[str, float]
    cluster: str


@dataclass(frozen=True)
class FeatureComparison:
    feature: str
    success_count: int
    failure_count: int
    success_mean: float
    failure_mean: float
    delta: float
    effect: float


@dataclass(frozen=True)
class ClusterStats:
    cluster: str
    count: int
    continuation_rate: float
    failure_rate: float


@dataclass(frozen=True)
class ComboAnalysis:
    name: str
    group: str
    occurrences: list[Occurrence]
    comparisons: list[FeatureComparison]
    clusters: list[ClusterStats]


@dataclass(frozen=True)
class PopulationStudy:
    name: str
    analyses: dict[str, ComboAnalysis]
    grouped: dict[str, ComboAnalysis]


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    rows: list[EvidenceBar]
    populations: dict[str, PopulationStudy]


@dataclass(frozen=True)
class AggregateCombo:
    name: str
    group: str
    values: dict[str, ComboAnalysis]
    valid_instruments: int
    agreement_count: int
    mean_effect: float


@dataclass(frozen=True)
class AggregateFeature:
    label: str
    values: dict[str, FeatureComparison]
    valid_instruments: int
    positive_count: int
    negative_count: int
    mean_effect: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study historical context around APVA condition-topology combinations."
    )
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def context_metrics(rows: list[EvidenceBar], index: int) -> dict[str, float]:
    output = {}
    for size in CONTEXT_WINDOWS:
        window = rows[max(0, index - size) : index]
        counts = {
            condition: float(sum(flags(row)[condition] for row in window))
            for condition in BASE_CONDITIONS
        }
        for condition, count in counts.items():
            output[f"Prev{size}{condition}Count"] = count
        output[f"Prev{size}NetAcceptance"] = counts["Accepted"] - counts["Peak"]
        output[f"Prev{size}Dominance"] = counts["Accepted"] + counts["Dissipating"] - counts["Compressed"]
    return output


def context_slopes(metrics: dict[str, float]) -> dict[str, float]:
    output = {}
    for condition in SLOPE_CONDITIONS:
        suffix = condition if condition == "Dominance" else condition + "Count"
        output[f"ContextSlope{condition}"] = metrics[f"Prev5{suffix}"] - metrics[f"Prev20{suffix}"] / 4.0
    return output


def context_cluster(metrics: dict[str, float]) -> str:
    values = {
        "AcceptanceHeavy": metrics["Prev10AcceptedCount"],
        "CompressionHeavy": metrics["Prev10CompressedCount"],
        "DissipationHeavy": metrics["Prev10DissipatingCount"],
        "PeakHeavy": metrics["Prev10PeakCount"],
    }
    maximum = max(values.values())
    winners = [name for name, value in values.items() if value == maximum]
    return winners[0] if len(winners) == 1 else "Mixed"


def occurrence(rows: list[EvidenceBar], index: int, combination: str, group: str) -> Occurrence | None:
    outcome = direction_relative_return(rows, index, 5)
    if outcome is None:
        return None
    metrics = context_metrics(rows, index)
    return Occurrence(combination, group, index, outcome, metrics, context_slopes(metrics), context_cluster(metrics))


def build_occurrences(rows: list[EvidenceBar], indexes: list[int], combination: str) -> list[Occurrence]:
    conditions = TARGETS[combination]
    group = "Positive" if combination in POSITIVE_NAMES else "Negative"
    output = []
    for index in indexes:
        if matches(flags(rows[index]), conditions):
            item = occurrence(rows, index, combination, group)
            if item is not None:
                output.append(item)
    return output


def feature_comparison(occurrences: list[Occurrence], feature: str) -> FeatureComparison:
    success = [
        (item.metrics | item.slopes)[feature]
        for item in occurrences
        if item.outcome > 0.0
    ]
    failure = [
        (item.metrics | item.slopes)[feature]
        for item in occurrences
        if item.outcome < 0.0
    ]
    return FeatureComparison(
        feature,
        len(success),
        len(failure),
        mean(success),
        mean(failure),
        mean(success) - mean(failure),
        effect_size(success, failure),
    )


def all_features() -> tuple[str, ...]:
    metrics = tuple(
        f"Prev{size}{condition}Count"
        for size in CONTEXT_WINDOWS
        for condition in BASE_CONDITIONS
    )
    derived = tuple(
        f"Prev{size}{name}"
        for size in CONTEXT_WINDOWS
        for name in ("NetAcceptance", "Dominance")
    )
    slopes = tuple(f"ContextSlope{condition}" for condition in SLOPE_CONDITIONS)
    return metrics + derived + slopes


FEATURES = all_features()


def cluster_stats(occurrences: list[Occurrence]) -> list[ClusterStats]:
    output = []
    for cluster in CLUSTERS:
        selected = [item.outcome for item in occurrences if item.cluster == cluster]
        denominator = len(selected) if selected else 1
        output.append(
            ClusterStats(
                cluster,
                len(selected),
                sum(value > 0.0 for value in selected) / denominator,
                sum(value < 0.0 for value in selected) / denominator,
            )
        )
    return output


def analyze(name: str, group: str, occurrences: list[Occurrence]) -> ComboAnalysis:
    return ComboAnalysis(
        name,
        group,
        occurrences,
        [feature_comparison(occurrences, feature) for feature in FEATURES],
        cluster_stats(occurrences),
    )


def build_population(rows: list[EvidenceBar], name: str, indexes: list[int]) -> PopulationStudy:
    analyses = {
        combination: analyze(
            combination,
            "Positive" if combination in POSITIVE_NAMES else "Negative",
            build_occurrences(rows, indexes, combination),
        )
        for combination in TARGETS
    }
    grouped = {
        group: analyze(
            f"All {group} Combinations",
            group,
            [item for analysis in analyses.values() if analysis.group == group for item in analysis.occurrences],
        )
        for group in ("Positive", "Negative")
    }
    return PopulationStudy(name, analyses, grouped)


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


def counts(analysis: ComboAnalysis) -> tuple[int, int, int, int]:
    total = len(analysis.occurrences)
    success = sum(item.outcome > 0.0 for item in analysis.occurrences)
    failure = sum(item.outcome < 0.0 for item in analysis.occurrences)
    return total, success, failure, total - success - failure


def ranked_comparisons(analysis: ComboAnalysis, reverse: bool = True) -> list[FeatureComparison]:
    return sorted(
        analysis.comparisons,
        key=lambda item: (-item.delta if reverse else item.delta, -abs(item.effect), item.feature),
    )


def append_context_table(lines: list[str], analysis: ComboAnalysis) -> None:
    total, success, failure, flat = counts(analysis)
    lines.extend([f"\n{analysis.name}", "-" * len(analysis.name)])
    lines.append(f"Group: {analysis.group}; Occurrences={total}; Success={success}; Failure={failure}; Flat={flat}")
    lines.append(
        f"{'Feature':<34} {'SuccessN':>8} {'FailureN':>8} {'SuccessMean':>12} "
        f"{'FailureMean':>12} {'Delta':>12} {'EffectSize':>11}"
    )
    for item in analysis.comparisons:
        lines.append(
            f"{item.feature:<34} {item.success_count:>8} {item.failure_count:>8} "
            f"{item.success_mean:>12.6f} {item.failure_mean:>12.6f} "
            f"{item.delta:>12.6f} {item.effect:>11.6f}"
        )
    for title, items in (
        ("Largest Positive Deltas", ranked_comparisons(analysis)),
        ("Largest Negative Deltas", ranked_comparisons(analysis, False)),
    ):
        lines.extend([f"\n{title}", "." * len(title)])
        for rank, item in enumerate(items[:TOP_LIMIT], start=1):
            lines.append(f"{rank:>3}. {item.feature:<34} Delta={item.delta:>10.6f} Effect={item.effect:>10.6f}")


def append_cluster_table(lines: list[str], analysis: ComboAnalysis) -> None:
    lines.extend([f"\n{analysis.name} Context Clusters", "-" * (len(analysis.name) + 17)])
    lines.append(f"{'Cluster':<20} {'Count':>8} {'ContRate5':>10} {'FailRate5':>10}")
    for item in analysis.clusters:
        lines.append(f"{item.cluster:<20} {item.count:>8} {item.continuation_rate:>9.2%} {item.failure_rate:>9.2%}")


def append_population(lines: list[str], population: PopulationStudy, include_details: bool) -> None:
    lines.extend([f"\n{population.name}", "=" * len(population.name)])
    for analysis in population.analyses.values():
        append_context_table(lines, analysis)
        append_cluster_table(lines, analysis)
    lines.extend([f"\n{population.name} Positive vs Negative Comparison", "-" * (len(population.name) + 32)])
    for analysis in population.grouped.values():
        append_context_table(lines, analysis)
        append_cluster_table(lines, analysis)
    if include_details:
        return


def append_population_b_comparison(lines: list[str], full: PopulationStudy, population_b: PopulationStudy) -> None:
    lines.extend(["\nPopulation B vs Full Population", "-------------------------------"])
    lines.append(
        f"{'Combination':<38} {'FullN':>7} {'FullCont':>9} {'PopBN':>7} "
        f"{'PopBCont':>9} {'DeltaCont':>10} {'FullTopEffect':>14} {'PopBTopEffect':>14}"
    )
    for name, b_analysis in population_b.analyses.items():
        full_analysis = full.analyses[name]
        full_total, full_success, _, _ = counts(full_analysis)
        b_total, b_success, _, _ = counts(b_analysis)
        full_cont = full_success / full_total if full_total else 0.0
        b_cont = b_success / b_total if b_total else 0.0
        full_effect = max((abs(item.effect) for item in full_analysis.comparisons), default=0.0)
        b_effect = max((abs(item.effect) for item in b_analysis.comparisons), default=0.0)
        lines.append(
            f"{name:<38} {full_total:>7} {full_cont:>8.2%} {b_total:>7} "
            f"{b_cont:>8.2%} {b_cont - full_cont:>9.2%} {full_effect:>14.6f} {b_effect:>14.6f}"
        )


def instrument_report(study: InstrumentStudy) -> str:
    full = study.populations["Full Population"]
    population_b = study.populations["Population B"]
    all_occurrences = [item for analysis in full.analyses.values() for item in analysis.occurrences]
    total = len(all_occurrences)
    success = sum(item.outcome > 0.0 for item in all_occurrences)
    failure = sum(item.outcome < 0.0 for item in all_occurrences)
    lines = [
        f"APVA Context Topology Study v0.1 - {study.instrument}",
        "=" * (35 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {len(study.rows)}",
        f"Imported target combinations: {len(TARGETS)}",
        f"Target occurrences: {total}",
        f"Success count: {success}",
        f"Failure count: {failure}",
        f"Flat count: {total - success - failure}",
        f"Population B rows: {len(mature_aligned_lateral_indexes(study.rows))}",
        "Context clusters use Prev10 counts; ties are Mixed.",
        "Context slopes use Prev5 count minus Prev20 count divided by four.",
        "",
        "Section 1 - Combination Success Context",
        "=======================================",
    ]
    for analysis in full.analyses.values():
        append_context_table(lines, analysis)
    lines.extend(["\nSection 2 - Context Gradient", "============================"])
    lines.append("Slope metrics are included in every combination context table as ContextSlope* rows.")
    lines.extend(["\nSection 3 - Context Clustering", "=============================="])
    for analysis in full.analyses.values():
        append_cluster_table(lines, analysis)
    lines.extend(["\nSection 4 - Positive vs Negative Combination Comparison", "========================================================"])
    for analysis in full.grouped.values():
        append_context_table(lines, analysis)
        append_cluster_table(lines, analysis)
    lines.extend(["\nSection 5 - Population B", "========================"])
    append_population(lines, population_b, include_details=True)
    append_population_b_comparison(lines, full, population_b)
    lines.extend(["\nResearch Notes", "=============="])
    ac = full.analyses.get("Accepted+Compressed")
    if ac:
        strongest = max(ac.comparisons, key=lambda item: abs(item.effect))
        lines.append(f"- Accepted+Compressed strongest context difference: {strongest.feature}, Delta={strongest.delta:.6f}, EffectSize={strongest.effect:.6f}.")
    pos = full.grouped["Positive"]
    neg = full.grouped["Negative"]
    pos_feature = max(pos.comparisons, key=lambda item: abs(item.effect))
    neg_feature = max(neg.comparisons, key=lambda item: abs(item.effect))
    lines.append(f"- Positive-combination strongest context difference: {pos_feature.feature}, EffectSize={pos_feature.effect:.6f}.")
    lines.append(f"- Negative-combination strongest context difference: {neg_feature.feature}, EffectSize={neg_feature.effect:.6f}.")
    lines.append("- Context measurements are descriptive; they do not establish that combinations are sufficient by themselves.")
    return "\n".join(lines) + "\n"


def instrument_columns(studies: list[InstrumentStudy]) -> list[str]:
    available = [study.instrument for study in studies]
    return list(CANONICAL_INSTRUMENTS) + sorted(set(available) - set(CANONICAL_INSTRUMENTS))


def strongest_comparison(analysis: ComboAnalysis) -> FeatureComparison:
    return max(analysis.comparisons, key=lambda item: abs(item.effect))


def aggregate_combinations(studies: list[InstrumentStudy]) -> list[AggregateCombo]:
    output = []
    for name in TARGETS:
        values = {study.instrument: study.populations["Full Population"].analyses[name] for study in studies}
        valid = [analysis for analysis in values.values() if len(analysis.occurrences) >= PER_INSTRUMENT_MIN_COUNT]
        effects = [strongest_comparison(analysis).effect for analysis in valid]
        output.append(
            AggregateCombo(
                name,
                "Positive" if name in POSITIVE_NAMES else "Negative",
                values,
                len(valid),
                max(
                    sum(effect > 0.0 for effect in effects),
                    sum(effect < 0.0 for effect in effects),
                ),
                mean(effects),
            )
        )
    return output


def aggregate_features(studies: list[InstrumentStudy], population: str) -> list[AggregateFeature]:
    output = []
    for name in TARGETS:
        for feature in FEATURES:
            values = {
                study.instrument: next(
                    item for item in study.populations[population].analyses[name].comparisons
                    if item.feature == feature
                )
                for study in studies
            }
            valid = [
                item for item in values.values()
                if item.success_count + item.failure_count >= PER_INSTRUMENT_MIN_COUNT
            ]
            output.append(
                AggregateFeature(
                    f"{name} | {feature}",
                    values,
                    len(valid),
                    sum(item.effect > 0.0 for item in valid),
                    sum(item.effect < 0.0 for item in valid),
                    mean([item.effect for item in valid]),
                )
            )
    return output


def append_aggregate_combo_table(lines: list[str], items: list[AggregateCombo], columns: list[str]) -> None:
    lines.extend(["\nAggregate Combination Summary", "============================="])
    header = f"{'Combination':<38} {'Group':<9}"
    for instrument in columns:
        header += f" {('Count_' + instrument):>9} {('Success_' + instrument):>11} {('LargestDiff_' + instrument):>18}"
    header += f" {'Valid':>5} {'Agree':>5} {'MeanEffect':>11}"
    lines.append(header)
    for item in items:
        row = f"{item.name:<38} {item.group:<9}"
        for instrument in columns:
            analysis = item.values.get(instrument)
            if analysis:
                total, success, _, _ = counts(analysis)
                largest = strongest_comparison(analysis)
                row += f" {total:>9} {success:>11} {(largest.feature + ':' + format(largest.effect, '.3f')):>18}"
            else:
                row += f" {0:>9} {0:>11} {'N/A':>18}"
        row += f" {item.valid_instruments:>5} {item.agreement_count:>5} {item.mean_effect:>11.6f}"
        lines.append(row)


def ranked_features(items: list[AggregateFeature], mode: str) -> list[AggregateFeature]:
    eligible = [item for item in items if item.valid_instruments >= 2]
    if mode == "positive":
        return sorted(eligible, key=lambda item: (-item.positive_count, -item.mean_effect, item.label))
    if mode == "negative":
        return sorted(eligible, key=lambda item: (-item.negative_count, item.mean_effect, item.label))
    return sorted(eligible, key=lambda item: (-abs(item.mean_effect), item.label))


def append_aggregate_ranked(lines: list[str], title: str, items: list[AggregateFeature]) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(f"{'Rank':>4} {'Combination | Feature':<74} {'Valid':>5} {'Pos':>4} {'Neg':>4} {'MeanEffect':>11}")
    for rank, item in enumerate(items[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {item.label:<74} {item.valid_instruments:>5} "
            f"{item.positive_count:>4} {item.negative_count:>4} {item.mean_effect:>11.6f}"
        )


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    columns = instrument_columns(studies)
    combos = aggregate_combinations(studies)
    full_features = aggregate_features(studies, "Full Population")
    b_features = aggregate_features(studies, "Population B")
    slopes = [item for item in full_features if "ContextSlope" in item.label]
    lines = [
        "APVA Context Topology Study v0.1 - Cross-Instrument Aggregate",
        "=============================================================",
        f"Input instruments: {', '.join(study.instrument for study in studies)}",
        f"Imported target combinations: {len(TARGETS)}.",
        f"Replication threshold: at least {PER_INSTRUMENT_MIN_COUNT} non-flat observations in two instruments.",
    ]
    append_aggregate_combo_table(lines, combos, columns)
    append_aggregate_ranked(lines, "Most Replicated Positive-Context Features", ranked_features(full_features, "positive"))
    append_aggregate_ranked(lines, "Most Replicated Negative-Context Features", ranked_features(full_features, "negative"))
    append_aggregate_ranked(lines, "Strongest Success Context Patterns", ranked_features(full_features, "positive"))
    append_aggregate_ranked(lines, "Strongest Failure Context Patterns", ranked_features(full_features, "negative"))
    append_aggregate_ranked(lines, "Most Important Context Slopes", ranked_features(slopes, "absolute"))
    append_aggregate_ranked(lines, "Most Important Population-B Context Features", ranked_features(b_features, "absolute"))
    lines.extend(["\nCross-Instrument Research Notes", "==============================="])
    ranked = ranked_features(full_features, "absolute")
    if ranked:
        lines.append(f"- Strongest replicated context feature: {ranked[0].label}, MeanEffect={ranked[0].mean_effect:.6f}.")
    lines.append("- Context comparisons describe histories preceding topology occurrences; they do not establish sufficiency.")
    lines.append("- Population B rankings reuse the existing mature aligned lateral DissipationContained selector.")
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
            Path("Evidence") / "Output" / study.instrument / f"ContextTopology_{study.instrument}.txt",
            instrument_report(study),
        )
    report = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, report)
    print(report, end="")


if __name__ == "__main__":
    main()
