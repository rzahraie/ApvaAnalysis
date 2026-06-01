"""APVA Branch Predictor Study v0.1.

Research-only comparison of evidence and memory features across competing
Study 35 narrative branches.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from APVA_AcceptanceAnatomy_23 import mature_aligned_lateral_indexes
from APVA_ArchetypeEvolution_36 import Observation, archetype_stream, build_observations, run_info
from APVA_Archetypes_35 import ARCHETYPES
from APVA_BreakoutContext_08 import EvidenceBar, load_rows
from APVA_FamilyEvolution_30 import FAMILIES, family_for, instrument_columns
from APVA_LateralAnatomy_19 import effect_size
from APVA_RegimeTransition_16 import instrument_name, write_text
from APVA_StateGrammar_25 import OutcomeStats, mean, summarize


AGGREGATE_OUTPUT = Path("Evidence/Output/BranchPredictor/BranchPredictor_All.txt")
TOP_LIMIT = 25
INSTRUMENT_BRANCH_MIN = 20
AGGREGATE_BRANCH_MIN = 100
INSTRUMENT_PATH_MIN = 10
AGGREGATE_PATH_MIN = 50
POSITIONS = ("First", "Middle", "Last", "Only")
BRANCHES = (
    ("Recovery", "Recovery", "Decay"),
    ("Recovery", "Recovery", "Compression Resolution"),
    ("Exhaustion", "Exhaustion", "Recovery"),
    ("Exhaustion", "Exhaustion", "Decay"),
    ("Decay", "Compression Resolution", "Neutral Drift"),
    ("Compression Resolution", "Compression Resolution", "Decay"),
    ("Reassertion", "Neutral Drift", "Destructive Persistence"),
)


@dataclass(frozen=True)
class FeatureComparison:
    feature: str
    group: str
    left_count: int
    right_count: int
    left_mean: float
    right_mean: float
    delta: float
    effect: float


@dataclass(frozen=True)
class GroupStats:
    group: str
    feature_count: int
    mean_abs_effect: float
    max_abs_effect: float
    replicated_count: int = 0


@dataclass(frozen=True)
class BranchStudy:
    name: str
    current: str
    left: str
    right: str
    left_items: list[Observation]
    right_items: list[Observation]
    features: list[FeatureComparison]
    groups: dict[str, GroupStats]
    left_outcome: OutcomeStats
    right_outcome: OutcomeStats


@dataclass(frozen=True)
class PopulationStudy:
    name: str
    indexes: list[int]
    families: dict[int, str]
    archetypes: dict[int, str]
    observations: list[Observation]
    branches: dict[str, BranchStudy]
    path_counts: dict[str, int]


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    rows: list[EvidenceBar]
    populations: dict[str, PopulationStudy]


@dataclass(frozen=True)
class AggregateFeature:
    branch: str
    feature: str
    group: str
    values: dict[str, FeatureComparison]
    valid_instruments: int
    positive_count: int
    negative_count: int
    agreement_count: int
    mean_effect: float
    mean_abs_effect: float


@dataclass(frozen=True)
class AggregateGroup:
    branch: str
    group: str
    values: dict[str, GroupStats]
    valid_instruments: int
    mean_abs_effect: float
    max_abs_effect: float
    replicated_count: int


@dataclass(frozen=True)
class AggregateOutcome:
    branch: str
    side: str
    values: dict[str, OutcomeStats]
    counts: dict[str, int]
    valid_instruments: int
    mean_continuation: float
    mean_failure: float
    mean_skew: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare evidence and memory features across competing APVA narrative branches.")
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def branch_name(current: str, left: str, right: str) -> str:
    return f"{current}->{left} vs {current}->{right}"


def feature_group(feature: str) -> str:
    if feature.startswith("Flag_"):
        return "Evidence"
    if feature.startswith(("ParticipationState_", "AcceptanceState_", "CompressionState_", "DissipationState_", "ExpansionState_")):
        return "States"
    if feature.startswith("Prev5"):
        return "Memory5"
    if feature.startswith("Prev10"):
        return "Memory10"
    if feature.startswith("Prev20"):
        return "Memory20"
    if "RunLength" in feature or "PositionIn" in feature:
        return "Runs"
    if feature.startswith("CurrentFamily_"):
        return "Family"
    if feature.startswith("CurrentArchetype_"):
        return "Archetype"
    if feature.startswith("CurrentPath3_"):
        return "Path3"
    if feature.startswith("CurrentPath4_"):
        return "Path4"
    if feature.startswith("CurrentPath5_"):
        return "Path5"
    return "States"


def augment_family_positions(observations: list[Observation], families: dict[int, str]) -> None:
    runs = run_info(families)
    for item in observations:
        current = runs[item.index]
        for position in POSITIONS:
            item.features[f"FamilyPositionInRun_{position}"] = float(current.position == position)


def add_path_indicators(observations: list[Observation]) -> dict[str, int]:
    counts = {}
    for item in observations:
        for length, attribute in ((3, "path3"), (4, "path4"), (5, "path5")):
            path = getattr(item, attribute)
            if path:
                feature = f"CurrentPath{length}_{path}"
                counts[feature] = counts.get(feature, 0) + 1
    reportable = {feature for feature, count in counts.items() if count >= INSTRUMENT_PATH_MIN}
    for item in observations:
        for length, attribute in ((3, "path3"), (4, "path4"), (5, "path5")):
            feature = f"CurrentPath{length}_{getattr(item, attribute)}"
            if feature in reportable:
                item.features[feature] = 1.0
    return counts


def comparison(feature: str, left: list[Observation], right: list[Observation]) -> FeatureComparison:
    left_values = [item.features.get(feature, 0.0) for item in left]
    right_values = [item.features.get(feature, 0.0) for item in right]
    return FeatureComparison(feature, feature_group(feature), len(left), len(right), mean(left_values), mean(right_values), mean(left_values) - mean(right_values), effect_size(left_values, right_values))


def group_stats(features: list[FeatureComparison]) -> dict[str, GroupStats]:
    output = {}
    for group in ("Evidence", "States", "Memory5", "Memory10", "Memory20", "Runs", "Family", "Archetype", "Path3", "Path4", "Path5"):
        values = [item for item in features if item.group == group]
        output[group] = GroupStats(group, len(values), mean([abs(item.effect) for item in values]), max([abs(item.effect) for item in values], default=0.0))
    return output


def build_branch(current: str, left_name: str, right_name: str, observations: list[Observation]) -> BranchStudy:
    eligible = [item for item in observations if item.current == current and 1 in item.next_by_horizon]
    left = [item for item in eligible if item.next_by_horizon[1] == left_name]
    right = [item for item in eligible if item.next_by_horizon[1] == right_name]
    names = sorted(set().union(*(item.features for item in left + right))) if left or right else []
    features = [comparison(feature, left, right) for feature in names]
    return BranchStudy(branch_name(current, left_name, right_name), current, left_name, right_name, left, right, features, group_stats(features), summarize([item.outcome for item in left if item.outcome is not None]), summarize([item.outcome for item in right if item.outcome is not None]))


def build_population(rows: list[EvidenceBar], name: str, indexes: list[int]) -> PopulationStudy:
    families = {index: family_for(rows[index]) for index in indexes}
    archetypes = archetype_stream(families)
    observations = build_observations(rows, families, archetypes)
    augment_family_positions(observations, families)
    path_counts = add_path_indicators(observations)
    branches = {branch_name(current, left, right): build_branch(current, left, right, observations) for current, left, right in BRANCHES}
    return PopulationStudy(name, indexes, families, archetypes, observations, branches, path_counts)


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    return InstrumentStudy(instrument_name(path), path, rows, {"Full Population": build_population(rows, "Full Population", list(range(len(rows)))), "Population B": build_population(rows, "Population B", mature_aligned_lateral_indexes(rows))})


def branch_aggregate_counts(studies: list[InstrumentStudy], population: str = "Full Population") -> dict[str, int]:
    return {name: sum(len(study.populations[population].branches[name].left_items) + len(study.populations[population].branches[name].right_items) for study in studies) for name in (branch_name(*value) for value in BRANCHES)}


def eligible_branch_names(studies: list[InstrumentStudy], population: str = "Full Population") -> set[str]:
    return {name for name, count in branch_aggregate_counts(studies, population).items() if count >= AGGREGATE_BRANCH_MIN}


def append_frequency(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'Comparison':<92} {'LeftN':>8} {'RightN':>8} {'TotalN':>8}")
    for item in population.branches.values():
        lines.append(f"{item.name:<92} {len(item.left_items):>8} {len(item.right_items):>8} {len(item.left_items) + len(item.right_items):>8}")


def append_outcomes(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'Comparison':<92} {'Side':<8} {'Count':>8} {'Mean':>11} {'Median':>11} {'Cont':>9} {'Fail':>9} {'Flat':>9}")
    for item in population.branches.values():
        for side, value in (("Left", item.left_outcome), ("Right", item.right_outcome)):
            lines.append(f"{item.name:<92} {side:<8} {value.count:>8} {value.mean:>11.6f} {value.median:>11.6f} {value.continuation_rate:>8.2%} {value.failure_rate:>8.2%} {value.flat_rate:>8.2%}")


def append_separators(lines: list[str], population: PopulationStudy, limit: int = TOP_LIMIT) -> None:
    for item in population.branches.values():
        lines.extend([f"\n{item.name}", "." * len(item.name)])
        if not item.left_items or not item.right_items:
            lines.append("Insufficient observations.")
            continue
        for title, values in (("Top Positive Separators", sorted(item.features, key=lambda feature: (-feature.effect, feature.feature))), ("Top Negative Separators", sorted(item.features, key=lambda feature: (feature.effect, feature.feature)))):
            lines.append(title + ":")
            for rank, feature in enumerate(values[:limit], start=1):
                lines.append(f"{rank:>3}. {feature.feature:<54} Group={feature.group:<10} Delta={feature.delta:>10.6f} Effect={feature.effect:>10.6f}")


def append_groups(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'Comparison':<92} {'Group':<10} {'Features':>8} {'MeanAbs':>10} {'MaxAbs':>10}")
    for item in population.branches.values():
        for value in item.groups.values():
            lines.append(f"{item.name:<92} {value.group:<10} {value.feature_count:>8} {value.mean_abs_effect:>10.6f} {value.max_abs_effect:>10.6f}")


def append_group_subset(lines: list[str], population: PopulationStudy, groups: tuple[str, ...]) -> None:
    lines.append(f"{'Comparison':<92} {'Group':<10} {'MeanAbs':>10} {'MaxAbs':>10}")
    for item in population.branches.values():
        for group in groups:
            value = item.groups[group]
            lines.append(f"{item.name:<92} {group:<10} {value.mean_abs_effect:>10.6f} {value.max_abs_effect:>10.6f}")


def instrument_report(study: InstrumentStudy) -> str:
    full = study.populations["Full Population"]
    population_b = study.populations["Population B"]
    valid = sum(len(item.left_items) >= INSTRUMENT_BRANCH_MIN and len(item.right_items) >= INSTRUMENT_BRANCH_MIN for item in full.branches.values())
    lines = [f"APVA Branch Predictor Study v0.1 - {study.instrument}", "=" * (36 + len(study.instrument)), f"Instrument: {study.instrument}", f"Input: {study.path}", f"Total rows: {len(study.rows)}", f"Family counts: {dict((name, sum(value == name for value in full.families.values())) for name in FAMILIES)}", f"Archetype counts: {dict((name, sum(value == name for value in full.archetypes.values())) for name in ARCHETYPES)}", f"Valid archetype transitions: {sum(1 in item.next_by_horizon for item in full.observations)}", f"Valid branch comparisons: {valid}", "", "Section 1 - Branch Frequency Summary", "===================================="]
    append_frequency(lines, full)
    lines.extend(["\nSection 2 - Branch Outcome Summary", "=================================="])
    append_outcomes(lines, full)
    lines.extend(["\nSection 3 - Top Branch Separators", "================================="])
    append_separators(lines, full)
    lines.extend(["\nSection 4 - Cross-Feature Group Importance", "=========================================="])
    append_groups(lines, full)
    lines.extend(["\nSection 5 - Memory Depth Analysis", "================================="])
    append_group_subset(lines, full, ("Memory5", "Memory10", "Memory20"))
    lines.extend(["\nSection 6 - Trajectory Importance", "================================="])
    append_group_subset(lines, full, ("Family", "Archetype", "Path3", "Path4", "Path5"))
    lines.extend(["\nSection 7 - Population B Branch Analysis", "========================================"])
    lines.append(f"Population B observations: {len(population_b.observations)}")
    append_frequency(lines, population_b)
    lines.extend(["\nPopulation B Outcomes", "---------------------"])
    append_outcomes(lines, population_b)
    lines.extend(["\nPopulation B Separators", "-----------------------"])
    append_separators(lines, population_b, limit=10)
    lines.extend(["\nPopulation B Feature Groups", "---------------------------"])
    append_groups(lines, population_b)
    lines.extend(["\nSection 8 - Mechanical Research Notes", "====================================="])
    lines.append(f"- {valid} branch comparisons have at least {INSTRUMENT_BRANCH_MIN} observations on both sides.")
    lines.append(f"- Path indicators are materialized at instrument Count >= {INSTRUMENT_PATH_MIN}; aggregate reporting also requires Count >= {AGGREGATE_PATH_MIN}.")
    lines.append(f"- Population B contains {len(population_b.observations)} bar-aligned observations and is reported sparsely.")
    lines.append("- Separators and outcome summaries are descriptive only.")
    return "\n".join(lines) + "\n"


def aggregate_path_counts(studies: list[InstrumentStudy]) -> dict[str, int]:
    output = {}
    for study in studies:
        for feature, count in study.populations["Full Population"].path_counts.items():
            output[feature] = output.get(feature, 0) + count
    return output


def aggregate_features(studies: list[InstrumentStudy]) -> list[AggregateFeature]:
    allowed_branches = eligible_branch_names(studies)
    paths = aggregate_path_counts(studies)
    output = []
    for branch in sorted(allowed_branches):
        mappings = {study.instrument: {item.feature: item for item in study.populations["Full Population"].branches[branch].features} for study in studies}
        features = sorted(set().union(*(mapping for mapping in mappings.values())))
        for feature in features:
            group = feature_group(feature)
            if group.startswith("Path") and paths.get(feature, 0) < AGGREGATE_PATH_MIN:
                continue
            values = {name: mapping[feature] for name, mapping in mappings.items() if feature in mapping}
            valid = [item for item in values.values() if item.left_count >= INSTRUMENT_BRANCH_MIN and item.right_count >= INSTRUMENT_BRANCH_MIN]
            positive = sum(item.effect > 0.0 for item in valid)
            negative = sum(item.effect < 0.0 for item in valid)
            output.append(AggregateFeature(branch, feature, group, values, len(valid), positive, negative, max(positive, negative), mean([item.effect for item in valid]), mean([abs(item.effect) for item in valid])))
    return output


def aggregate_groups(studies: list[InstrumentStudy], features: list[AggregateFeature]) -> list[AggregateGroup]:
    output = []
    for branch in sorted(eligible_branch_names(studies)):
        for group in ("Evidence", "States", "Memory5", "Memory10", "Memory20", "Runs", "Family", "Archetype", "Path3", "Path4", "Path5"):
            selected = [item for item in features if item.branch == branch and item.group == group]
            by_instrument = {}
            for study in studies:
                instrument_values = [item.values[study.instrument] for item in selected if study.instrument in item.values]
                by_instrument[study.instrument] = GroupStats(group, len(instrument_values), mean([abs(item.effect) for item in instrument_values]), max([abs(item.effect) for item in instrument_values], default=0.0), sum(item.valid_instruments >= 2 and item.agreement_count == item.valid_instruments for item in selected))
            valid = [value for value in by_instrument.values() if value.feature_count]
            output.append(AggregateGroup(branch, group, by_instrument, len(valid), mean([value.mean_abs_effect for value in valid]), max([value.max_abs_effect for value in valid], default=0.0), sum(item.valid_instruments >= 2 and item.agreement_count == item.valid_instruments for item in selected)))
    return output


def aggregate_outcomes(studies: list[InstrumentStudy]) -> list[AggregateOutcome]:
    output = []
    for branch in sorted(eligible_branch_names(studies)):
        for side in ("Left", "Right"):
            values = {}
            counts = {}
            for study in studies:
                item = study.populations["Full Population"].branches[branch]
                values[study.instrument] = item.left_outcome if side == "Left" else item.right_outcome
                counts[study.instrument] = len(item.left_items) if side == "Left" else len(item.right_items)
            valid = [values[name] for name, count in counts.items() if count >= INSTRUMENT_BRANCH_MIN]
            output.append(AggregateOutcome(branch, side, values, counts, len(valid), mean([item.continuation_rate for item in valid]), mean([item.failure_rate for item in valid]), mean([item.continuation_rate - item.failure_rate for item in valid])))
    return output


def append_aggregate_features(lines: list[str], items: list[AggregateFeature], columns: list[str]) -> None:
    lines.append(f"{'BranchComparison':<92} {'Feature':<54} {'Group':<10}" + "".join(f" {name + '_Effect':>12}" for name in columns) + " ValidN PosN NegN Agree MeanEffect MeanAbs")
    for item in items:
        if item.valid_instruments < 2:
            continue
        cells = "".join(f" {item.values[name].effect if name in item.values else 0.0:>12.6f}" for name in columns)
        lines.append(f"{item.branch:<92} {item.feature:<54} {item.group:<10}{cells} {item.valid_instruments:>6} {item.positive_count:>4} {item.negative_count:>4} {item.agreement_count:>5} {item.mean_effect:>10.6f} {item.mean_abs_effect:>8.6f}")


def append_aggregate_groups(lines: list[str], items: list[AggregateGroup], columns: list[str]) -> None:
    lines.append(f"{'BranchComparison':<92} {'Group':<10}" + "".join(f" {name + '_Mean':>10} {name + '_Max':>10} {name + '_Rep':>8}" for name in columns) + " ValidN MeanAbsAll MaxAbsAll RepAll")
    for item in items:
        cells = "".join(f" {item.values[name].mean_abs_effect if name in item.values else 0.0:>10.6f} {item.values[name].max_abs_effect if name in item.values else 0.0:>10.6f} {item.values[name].replicated_count if name in item.values else 0:>8}" for name in columns)
        lines.append(f"{item.branch:<92} {item.group:<10}{cells} {item.valid_instruments:>6} {item.mean_abs_effect:>10.6f} {item.max_abs_effect:>9.6f} {item.replicated_count:>6}")


def append_aggregate_group_subset(lines: list[str], items: list[AggregateGroup], groups: tuple[str, ...]) -> None:
    lines.append(f"{'BranchComparison':<92} {'Group':<10} {'MeanAbsAll':>10} {'MaxAbsAll':>10} {'Replicated':>10}")
    for item in items:
        if item.group in groups:
            lines.append(f"{item.branch:<92} {item.group:<10} {item.mean_abs_effect:>10.6f} {item.max_abs_effect:>10.6f} {item.replicated_count:>10}")


def append_aggregate_outcomes(lines: list[str], items: list[AggregateOutcome], columns: list[str]) -> None:
    lines.append(f"{'BranchComparison':<92} {'Side':<6}" + "".join(f" {name + '_N':>8} {name + '_Cont':>10} {name + '_Fail':>10} {name + '_Mean':>11}" for name in columns) + " ValidN MeanCont MeanFail MeanSkew")
    for item in items:
        cells = "".join(f" {item.counts.get(name, 0):>8} {item.values[name].continuation_rate if name in item.values else 0.0:>9.2%} {item.values[name].failure_rate if name in item.values else 0.0:>9.2%} {item.values[name].mean if name in item.values else 0.0:>11.6f}" for name in columns)
        lines.append(f"{item.branch:<92} {item.side:<6}{cells} {item.valid_instruments:>6} {item.mean_continuation:>8.2%} {item.mean_failure:>8.2%} {item.mean_skew:>8.2%}")


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
    features = aggregate_features(studies)
    groups = aggregate_groups(studies, features)
    outcomes = aggregate_outcomes(studies)
    lines = ["APVA Branch Predictor Study v0.1 - Aggregate", "===========================================", f"Instruments: {', '.join(study.instrument for study in studies)}", f"Aggregate branch minimum: {AGGREGATE_BRANCH_MIN}; valid-instrument side minimum: {INSTRUMENT_BRANCH_MIN}.", f"Path reporting minimums: aggregate {AGGREGATE_PATH_MIN}; instrument {INSTRUMENT_PATH_MIN}.", "", "Aggregate Branch Predictor Table", "================================"]
    append_aggregate_features(lines, features, columns)
    lines.extend(["\nAggregate Feature Group Table", "============================="])
    append_aggregate_groups(lines, groups, columns)
    lines.extend(["\nAggregate Memory Depth Table", "============================"])
    append_aggregate_group_subset(lines, groups, ("Memory5", "Memory10", "Memory20"))
    lines.extend(["\nAggregate Trajectory Importance Table", "====================================="])
    append_aggregate_group_subset(lines, groups, ("Family", "Archetype", "Path3", "Path4", "Path5"))
    lines.extend(["\nAggregate Outcome Table", "======================="])
    append_aggregate_outcomes(lines, outcomes, columns)
    lines.extend(["\nAggregate Rankings", "=================="])
    replicated = [item for item in features if item.valid_instruments >= 2 and item.agreement_count == item.valid_instruments]
    append_ranked(lines, "1. Strongest replicated branch separators", replicated, lambda item: (item.mean_abs_effect, item.feature), lambda item: f"{item.branch} | {item.feature:<44} Group={item.group:<10} MeanEffect={item.mean_effect:>10.6f}")
    ranking_specs = (
        ("2. Strongest Recovery->Recovery vs Recovery->Decay separators", branch_name("Recovery", "Recovery", "Decay")),
        ("3. Strongest Recovery->Recovery vs Recovery->Compression Resolution separators", branch_name("Recovery", "Recovery", "Compression Resolution")),
        ("4. Strongest Exhaustion->Exhaustion vs Exhaustion->Recovery separators", branch_name("Exhaustion", "Exhaustion", "Recovery")),
        ("5. Strongest Exhaustion->Exhaustion vs Exhaustion->Decay separators", branch_name("Exhaustion", "Exhaustion", "Decay")),
        ("6. Strongest Decay->Compression Resolution vs Decay->Neutral Drift separators", branch_name("Decay", "Compression Resolution", "Neutral Drift")),
        ("7. Strongest Compression Resolution persistence separators", branch_name("Compression Resolution", "Compression Resolution", "Decay")),
        ("8. Strongest Reassertion branch separators", branch_name("Reassertion", "Neutral Drift", "Destructive Persistence")),
    )
    for title, branch in ranking_specs:
        append_ranked(lines, title, [item for item in replicated if item.branch == branch], lambda item: (item.mean_abs_effect, item.feature), lambda item: f"{item.feature:<54} Group={item.group:<10} MeanEffect={item.mean_effect:>10.6f}")
    append_ranked(lines, "9. Most important feature groups", groups, lambda item: (item.mean_abs_effect, item.branch, item.group), lambda item: f"{item.branch} | {item.group:<10} MeanAbs={item.mean_abs_effect:.6f} MaxAbs={item.max_abs_effect:.6f}")
    append_ranked(lines, "10. Most important memory depths", [item for item in groups if item.group in {"Memory5", "Memory10", "Memory20"}], lambda item: (item.mean_abs_effect, item.branch, item.group), lambda item: f"{item.branch} | {item.group:<10} MeanAbs={item.mean_abs_effect:.6f}")
    append_ranked(lines, "11. Most important trajectory variables", [item for item in groups if item.group in {"Family", "Archetype", "Path3", "Path4", "Path5"}], lambda item: (item.mean_abs_effect, item.branch, item.group), lambda item: f"{item.branch} | {item.group:<10} MeanAbs={item.mean_abs_effect:.6f}")
    append_ranked(lines, "12. Branches with best outcome skew", outcomes, lambda item: (item.mean_skew, item.branch, item.side), lambda item: f"{item.branch} | {item.side:<5} MeanSkew={item.mean_skew:.2%} MeanCont={item.mean_continuation:.2%}")
    append_ranked(lines, "13. Branches with worst outcome skew", outcomes, lambda item: (-item.mean_skew, item.branch, item.side), lambda item: f"{item.branch} | {item.side:<5} MeanSkew={item.mean_skew:.2%} MeanFail={item.mean_failure:.2%}")
    lines.extend(["\n14. Population-B differences", "." * 28])
    lines.append("Population B remains sparse; per-instrument Population B comparisons are reported without aggregate inference.")
    lines.extend(["\nCross-Instrument Mechanical Research Notes", "=========================================="])
    strongest = max(replicated, key=lambda item: item.mean_abs_effect, default=None)
    strongest_group = max((item for item in groups if item.valid_instruments >= 2), key=lambda item: item.mean_abs_effect, default=None)
    lines.append(f"- Strongest replicated separator: {strongest.feature if strongest else 'N/A'}.")
    lines.append(f"- Highest mean absolute-effect feature group: {strongest_group.group if strongest_group else 'N/A'}.")
    lines.append("- Path indicators are retained only when their occurrence counts meet the requested thresholds.")
    lines.append("- Separators, group effects, memory depth, trajectory importance, and outcome skew are descriptive only.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    studies = [study_instrument(path) for path in args.input_paths]
    seen = set()
    for study in studies:
        if study.instrument in seen:
            raise ValueError(f"Duplicate instrument input: {study.instrument}")
        seen.add(study.instrument)
        write_text(Path("Evidence") / "Output" / study.instrument / f"BranchPredictor_{study.instrument}.txt", instrument_report(study))
    report = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, report)
    print(report, end="")


if __name__ == "__main__":
    main()
