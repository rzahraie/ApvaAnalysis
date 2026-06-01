"""APVA Structural Compression Study v0.1.

Research-only comparison of exact family paths and smaller mechanical
structural signatures across competing Study 37 narrative branches.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import pstdev

from APVA_AcceptanceAnatomy_23 import mature_aligned_lateral_indexes
from APVA_ArchetypeEvolution_36 import Observation, archetype_stream, build_observations
from APVA_Archetypes_35 import ARCHETYPES
from APVA_BranchPredictor_37 import BRANCHES, POSITIONS, augment_family_positions, branch_name
from APVA_BreakoutContext_08 import EvidenceBar, load_rows
from APVA_FamilyEvolution_30 import FAMILIES, instrument_columns, family_for
from APVA_LateralAnatomy_19 import effect_size
from APVA_RegimeTransition_16 import instrument_name, write_text
from APVA_StateGrammar_25 import OutcomeStats, mean, summarize


AGGREGATE_OUTPUT = Path("Evidence/Output/StructuralCompression/StructuralCompression_All.txt")
INSTRUMENT_BRANCH_MIN = 20
AGGREGATE_BRANCH_MIN = 100
INSTRUMENT_PATH_MIN = 10
AGGREGATE_PATH_MIN = 50
TOP_LIMIT = 25
FEATURE_SETS = ("ExactPath", "StructuralSignature", "FamilyOnly", "ArchetypeOnly", "RunOnly")
SIGNATURE_GROUPS = ("Endpoint", "Composition", "Repetition", "Motif", "NarrativeLike")
FAMILY_CLASS = {"A": "Constructive", "B": "Constructive", "C": "Destructive", "D": "Destructive", "N": "Neutral"}


@dataclass(frozen=True)
class FeatureEffect:
    feature: str
    feature_set: str
    signature_group: str
    left_mean: float
    right_mean: float
    delta: float
    effect: float


@dataclass(frozen=True)
class PowerStats:
    feature_set: str
    feature_count: int
    mean_abs_top5: float
    mean_abs_top10: float
    max_abs: float
    replicated_count: int = 0


@dataclass(frozen=True)
class BranchStudy:
    name: str
    left_items: list[Observation]
    right_items: list[Observation]
    effects: dict[str, list[FeatureEffect]]
    powers: dict[str, PowerStats]
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
    exact_counts: dict[str, int]


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
    signature_group: str
    effects: dict[str, float]
    valid_instruments: int
    agreement_count: int
    mean_effect: float
    mean_abs_effect: float


@dataclass(frozen=True)
class AggregatePower:
    branch: str
    feature_set: str
    powers: dict[str, float]
    valid_instruments: int
    mean_power: float
    power_stddev: float
    replicated_count: int


@dataclass(frozen=True)
class CompressionStats:
    branch: str
    exact: float
    structural: float
    family: float
    archetype: float
    run: float
    retention: float | None
    verdict: str
    valid_instruments: int
    replicated_structural_count: int


@dataclass(frozen=True)
class AggregateOutcome:
    branch: str
    side: str
    stats: dict[str, OutcomeStats]
    counts: dict[str, int]
    valid_instruments: int
    mean_continuation: float
    mean_failure: float
    mean_skew: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress exact APVA family paths into mechanical structural signatures.")
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def repeated_alternation(parts: list[str]) -> bool:
    return len(parts) >= 3 and all(parts[index] == parts[index - 2] and parts[index] != parts[index - 1] for index in range(2, len(parts)))


def one_hot(output: dict[str, float], prefix: str, value: str) -> None:
    output[f"{prefix}_{value}"] = 1.0


def path_signature(path: str, length: int) -> dict[str, float]:
    parts = path.split("->")
    counts = {family: parts.count(family) for family in FAMILIES}
    classes = [FAMILY_CLASS[family] for family in parts]
    dominant_count = max(counts.values())
    dominant = [family for family, count in counts.items() if count == dominant_count]
    dominant_name = dominant[0] if len(dominant) == 1 else "Mixed"
    repeat_count = sum(parts[index] == parts[index - 1] for index in range(1, len(parts)))
    transition_count = sum(parts[index] != parts[index - 1] for index in range(1, len(parts)))
    if len(set(parts)) == 1:
        persistence = "AllSame"
    elif repeated_alternation(parts):
        persistence = "Alternating"
    elif dominant_count > 1:
        persistence = "RepeatedDominantFamily"
    else:
        persistence = "Mixed"
    compact = "".join(parts)
    output = {
        f"Structural_L{length}_Contains{family}": float(counts[family] > 0) for family in FAMILIES
    }
    output.update({
        f"Structural_L{length}_ContainsRepeated{family}": float(counts[family] > 1) for family in FAMILIES
    })
    output.update({
        f"Structural_L{length}_RepeatCount": float(repeat_count),
        f"Structural_L{length}_TransitionCount": float(transition_count),
        f"Structural_L{length}_UniqueFamilyCount": float(len(set(parts))),
        f"Structural_L{length}_ConstructiveCount": float(sum(value in ("A", "B") for value in parts)),
        f"Structural_L{length}_DestructiveCount": float(sum(value in ("C", "D") for value in parts)),
        f"Structural_L{length}_NeutralCount": float(counts["N"]),
        f"Structural_L{length}_StartsConstructive": float(classes[0] == "Constructive"),
        f"Structural_L{length}_StartsDestructive": float(classes[0] == "Destructive"),
        f"Structural_L{length}_StartsNeutral": float(classes[0] == "Neutral"),
        f"Structural_L{length}_EndsConstructive": float(classes[-1] == "Constructive"),
        f"Structural_L{length}_EndsDestructive": float(classes[-1] == "Destructive"),
        f"Structural_L{length}_EndsNeutral": float(classes[-1] == "Neutral"),
        f"Structural_L{length}_ContainsCB": float("CB" in compact),
        f"Structural_L{length}_ContainsCCB": float("CCB" in compact),
        f"Structural_L{length}_ContainsCC": float("CC" in compact),
        f"Structural_L{length}_ContainsCCC": float("CCC" in compact),
        f"Structural_L{length}_ContainsBN": float("BN" in compact),
        f"Structural_L{length}_ContainsCN": float("CN" in compact),
        f"Structural_L{length}_ContainsDN": float("DN" in compact),
        f"Structural_L{length}_ContainsNDN": float("NDN" in compact),
        f"Structural_L{length}_ContainsDND": float("DND" in compact),
        f"Structural_L{length}_ContainsNNN": float("NNN" in compact),
        f"Structural_L{length}_CompressionLike": float(counts["B"] > 0 or counts["N"] > 1 or (parts[-1] == "N" and any(value in ("B", "C") for value in parts[:-1]))),
        f"Structural_L{length}_ExhaustionLike": float(counts["C"] > 1 or parts[-1] == "C"),
        f"Structural_L{length}_RecoveryLike": float(compact.endswith("CB") or "CCB" in compact or (classes[0] == "Destructive" and classes[-1] == "Constructive")),
        f"Structural_L{length}_DecayLike": float(parts[-1] == "N" and any(value in ("B", "C") for value in parts[:-1])),
        f"Structural_L{length}_ReassertionLike": float(repeated_alternation([value for value in parts if value in ("D", "N")]) and all(value in ("D", "N") for value in parts) or compact.endswith("ND")),
        f"Structural_L{length}_NeutralDriftLike": float(counts["N"] >= len(parts) - 1),
    })
    one_hot(output, f"Structural_L{length}_StartFamily", parts[0])
    one_hot(output, f"Structural_L{length}_EndFamily", parts[-1])
    one_hot(output, f"Structural_L{length}_MiddleFamilySet", "".join(sorted(set(parts[1:-1]))) or "None")
    one_hot(output, f"Structural_L{length}_DirectionClass", f"{classes[0]}To{classes[-1]}")
    one_hot(output, f"Structural_L{length}_PersistenceClass", persistence)
    one_hot(output, f"Structural_L{length}_DominantFamily", dominant_name)
    return output


def signature_group(feature: str) -> str:
    if any(token in feature for token in ("StartFamily", "EndFamily", "DirectionClass")):
        return "Endpoint"
    if any(token in feature for token in ("ContainsCB", "ContainsCCB", "ContainsCCC", "ContainsBN", "ContainsCN", "ContainsDN", "ContainsNDN", "ContainsDND", "ContainsNNN")):
        return "Motif"
    if any(token in feature for token in ("CompressionLike", "ExhaustionLike", "RecoveryLike", "DecayLike", "ReassertionLike", "NeutralDriftLike")):
        return "NarrativeLike"
    if any(token in feature for token in ("ContainsRepeated", "RepeatCount", "PersistenceClass")):
        return "Repetition"
    return "Composition"


def feature_set(feature: str) -> str:
    if feature.startswith("ExactPath"):
        return "ExactPath"
    if feature.startswith("Structural_"):
        return "StructuralSignature"
    if feature.startswith("CurrentFamily_"):
        return "FamilyOnly"
    if feature.startswith("CurrentArchetype_"):
        return "ArchetypeOnly"
    if "RunLength" in feature or "PositionIn" in feature:
        return "RunOnly"
    return ""


def add_path_features(observations: list[Observation]) -> dict[str, int]:
    counts = {}
    for item in observations:
        for length, attribute in ((3, "path3"), (4, "path4"), (5, "path5")):
            path = getattr(item, attribute)
            if path:
                key = f"ExactPath_L{length}_{path}"
                counts[key] = counts.get(key, 0) + 1
                item.features.update(path_signature(path, length))
    reportable = {key for key, count in counts.items() if count >= INSTRUMENT_PATH_MIN}
    for item in observations:
        for length, attribute in ((3, "path3"), (4, "path4"), (5, "path5")):
            key = f"ExactPath_L{length}_{getattr(item, attribute)}"
            if key in reportable:
                item.features[key] = 1.0
    return counts


def selected_feature_names(items: list[Observation]) -> list[str]:
    names = set().union(*(item.features for item in items)) if items else set()
    return sorted(name for name in names if feature_set(name) in FEATURE_SETS)


def compare_feature(name: str, left: list[Observation], right: list[Observation]) -> FeatureEffect:
    left_values = [item.features.get(name, 0.0) for item in left]
    right_values = [item.features.get(name, 0.0) for item in right]
    return FeatureEffect(name, feature_set(name), signature_group(name) if name.startswith("Structural_") else "", mean(left_values), mean(right_values), mean(left_values) - mean(right_values), effect_size(left_values, right_values))


def top_mean(values: list[float], limit: int) -> float:
    return mean(sorted((abs(value) for value in values), reverse=True)[:limit])


def power_stats(feature_set_name: str, effects: list[FeatureEffect]) -> PowerStats:
    values = [item.effect for item in effects]
    return PowerStats(feature_set_name, len(values), top_mean(values, 5), top_mean(values, 10), max([abs(value) for value in values], default=0.0))


def build_branch(current: str, left_name: str, right_name: str, observations: list[Observation]) -> BranchStudy:
    eligible = [item for item in observations if item.current == current and 1 in item.next_by_horizon]
    left = [item for item in eligible if item.next_by_horizon[1] == left_name]
    right = [item for item in eligible if item.next_by_horizon[1] == right_name]
    effects = {name: [] for name in FEATURE_SETS}
    for name in selected_feature_names(left + right):
        item = compare_feature(name, left, right)
        effects[item.feature_set].append(item)
    powers = {name: power_stats(name, effects[name]) for name in FEATURE_SETS}
    return BranchStudy(branch_name(current, left_name, right_name), left, right, effects, powers, summarize([item.outcome for item in left if item.outcome is not None]), summarize([item.outcome for item in right if item.outcome is not None]))


def build_population(rows: list[EvidenceBar], name: str, indexes: list[int]) -> PopulationStudy:
    families = {index: family_for(rows[index]) for index in indexes}
    archetypes = archetype_stream(families)
    observations = build_observations(rows, families, archetypes)
    augment_family_positions(observations, families)
    exact_counts = add_path_features(observations)
    branches = {branch_name(*branch): build_branch(*branch, observations) for branch in BRANCHES}
    return PopulationStudy(name, indexes, families, archetypes, observations, branches, exact_counts)


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    return InstrumentStudy(instrument_name(path), path, rows, {"Full Population": build_population(rows, "Full Population", list(range(len(rows)))), "Population B": build_population(rows, "Population B", mature_aligned_lateral_indexes(rows))})


def aggregate_side_counts(studies: list[InstrumentStudy], population: str = "Full Population") -> dict[str, tuple[int, int]]:
    output = {}
    for branch in (branch_name(*value) for value in BRANCHES):
        output[branch] = (
            sum(len(study.populations[population].branches[branch].left_items) for study in studies),
            sum(len(study.populations[population].branches[branch].right_items) for study in studies),
        )
    return output


def eligible_branches(studies: list[InstrumentStudy], population: str = "Full Population") -> set[str]:
    return {name for name, (left, right) in aggregate_side_counts(studies, population).items() if left >= AGGREGATE_BRANCH_MIN and right >= AGGREGATE_BRANCH_MIN}


def aggregate_exact_counts(studies: list[InstrumentStudy]) -> dict[str, int]:
    output = {}
    for study in studies:
        for name, count in study.populations["Full Population"].exact_counts.items():
            output[name] = output.get(name, 0) + count
    return output


def valid_instrument_branch(study: InstrumentStudy, branch: str, population: str = "Full Population") -> bool:
    value = study.populations[population].branches[branch]
    return len(value.left_items) >= INSTRUMENT_BRANCH_MIN and len(value.right_items) >= INSTRUMENT_BRANCH_MIN


def aggregate_structural_features(studies: list[InstrumentStudy]) -> list[AggregateFeature]:
    output = []
    for branch in sorted(eligible_branches(studies)):
        names = sorted({item.feature for study in studies for item in study.populations["Full Population"].branches[branch].effects["StructuralSignature"]})
        for name in names:
            effects = {}
            group = ""
            for study in studies:
                if not valid_instrument_branch(study, branch):
                    continue
                mapping = {item.feature: item for item in study.populations["Full Population"].branches[branch].effects["StructuralSignature"]}
                if name in mapping:
                    effects[study.instrument] = mapping[name].effect
                    group = mapping[name].signature_group
            positive = sum(value > 0 for value in effects.values())
            negative = sum(value < 0 for value in effects.values())
            output.append(AggregateFeature(branch, name, group, effects, len(effects), max(positive, negative), mean(list(effects.values())), mean([abs(value) for value in effects.values()])))
    return output


def aggregate_powers(studies: list[InstrumentStudy], structural: list[AggregateFeature]) -> list[AggregatePower]:
    exact_counts = aggregate_exact_counts(studies)
    output = []
    for branch in sorted(eligible_branches(studies)):
        for set_name in FEATURE_SETS:
            powers = {}
            for study in studies:
                if not valid_instrument_branch(study, branch):
                    continue
                effects = study.populations["Full Population"].branches[branch].effects[set_name]
                if set_name == "ExactPath":
                    effects = [item for item in effects if exact_counts.get(item.feature, 0) >= AGGREGATE_PATH_MIN]
                powers[study.instrument] = power_stats(set_name, effects).mean_abs_top10
            replicated = sum(item.valid_instruments >= 2 and item.agreement_count == item.valid_instruments for item in structural if item.branch == branch) if set_name == "StructuralSignature" else 0
            values = list(powers.values())
            output.append(AggregatePower(branch, set_name, powers, len(values), mean(values), pstdev(values) if len(values) > 1 else 0.0, replicated))
    return output


def verdict(retention: float | None) -> str:
    if retention is None:
        return "N/A"
    if retention >= 0.70:
        return "LikelyStructural"
    if retention >= 0.40:
        return "Mixed"
    return "LikelyPathSpecific"


def aggregate_compression(powers: list[AggregatePower]) -> list[CompressionStats]:
    output = []
    for branch in sorted({item.branch for item in powers}):
        mapping = {item.feature_set: item for item in powers if item.branch == branch}
        exact = mapping["ExactPath"].mean_power
        structural = mapping["StructuralSignature"].mean_power
        retention = structural / exact if exact else None
        output.append(CompressionStats(branch, exact, structural, mapping["FamilyOnly"].mean_power, mapping["ArchetypeOnly"].mean_power, mapping["RunOnly"].mean_power, retention, verdict(retention), mapping["StructuralSignature"].valid_instruments, mapping["StructuralSignature"].replicated_count))
    return output


def aggregate_outcomes(studies: list[InstrumentStudy]) -> list[AggregateOutcome]:
    output = []
    for branch in sorted(eligible_branches(studies)):
        for side in ("Left", "Right"):
            counts = {}
            stats = {}
            for study in studies:
                item = study.populations["Full Population"].branches[branch]
                values = item.left_items if side == "Left" else item.right_items
                counts[study.instrument] = len(values)
                stats[study.instrument] = item.left_outcome if side == "Left" else item.right_outcome
            valid = [stats[name] for name, count in counts.items() if count >= INSTRUMENT_BRANCH_MIN]
            output.append(AggregateOutcome(branch, side, stats, counts, len(valid), mean([item.continuation_rate for item in valid]), mean([item.failure_rate for item in valid]), mean([item.continuation_rate - item.failure_rate for item in valid])))
    return output


def format_retention(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def append_heading(lines: list[str], title: str) -> None:
    lines.extend(["", title, "=" * len(title)])


def append_branch_counts(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'Comparison':<92} {'LeftN':>8} {'RightN':>8}")
    for item in population.branches.values():
        lines.append(f"{item.name:<92} {len(item.left_items):>8} {len(item.right_items):>8}")


def append_effects(lines: list[str], population: PopulationStudy, set_name: str, limit: int = TOP_LIMIT) -> None:
    for branch in population.branches.values():
        lines.extend(["", branch.name, "." * len(branch.name)])
        effects = sorted(branch.effects[set_name], key=lambda item: abs(item.effect), reverse=True)[:limit]
        if not effects:
            lines.append("No reportable features.")
            continue
        lines.append(f"{'Feature':<72} {'Group':<13} {'LeftMean':>10} {'RightMean':>10} {'Delta':>10} {'Effect':>10}")
        for item in effects:
            lines.append(f"{item.feature:<72} {item.signature_group:<13} {item.left_mean:>10.4f} {item.right_mean:>10.4f} {item.delta:>10.4f} {item.effect:>10.4f}")


def branch_compression(branch: BranchStudy) -> CompressionStats:
    exact = branch.powers["ExactPath"].mean_abs_top10
    structural = branch.powers["StructuralSignature"].mean_abs_top10
    retention = structural / exact if exact else None
    return CompressionStats(branch.name, exact, structural, branch.powers["FamilyOnly"].mean_abs_top10, branch.powers["ArchetypeOnly"].mean_abs_top10, branch.powers["RunOnly"].mean_abs_top10, retention, verdict(retention), 1, 0)


def append_powers(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'Comparison':<92} {'FeatureSet':<20} {'Features':>8} {'Top5':>10} {'Top10':>10} {'MaxAbs':>10}")
    for branch in population.branches.values():
        for item in branch.powers.values():
            lines.append(f"{branch.name:<92} {item.feature_set:<20} {item.feature_count:>8} {item.mean_abs_top5:>10.4f} {item.mean_abs_top10:>10.4f} {item.max_abs:>10.4f}")


def append_compression(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'Comparison':<92} {'Exact':>9} {'Structure':>10} {'Family':>9} {'Archetype':>10} {'Run':>9} {'Retention':>10} {'Verdict':<20}")
    for branch in population.branches.values():
        item = branch_compression(branch)
        lines.append(f"{item.branch:<92} {item.exact:>9.4f} {item.structural:>10.4f} {item.family:>9.4f} {item.archetype:>10.4f} {item.run:>9.4f} {format_retention(item.retention):>10} {item.verdict:<20}")


def append_signature_groups(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'Comparison':<92} {'Group':<14} {'Top5':>10} {'MaxAbs':>10}")
    for branch in population.branches.values():
        for group in SIGNATURE_GROUPS:
            effects = [item.effect for item in branch.effects["StructuralSignature"] if item.signature_group == group]
            lines.append(f"{branch.name:<92} {group:<14} {top_mean(effects, 5):>10.4f} {max([abs(value) for value in effects], default=0.0):>10.4f}")


def append_outcomes(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'Comparison':<92} {'Side':<6} {'Count':>8} {'Mean':>11} {'Median':>11} {'Cont':>9} {'Fail':>9} {'Flat':>9}")
    for branch in population.branches.values():
        for side, item in (("Left", branch.left_outcome), ("Right", branch.right_outcome)):
            lines.append(f"{branch.name:<92} {side:<6} {item.count:>8} {item.mean:>11.6f} {item.median:>11.6f} {item.continuation_rate:>8.2%} {item.failure_rate:>8.2%} {item.flat_rate:>8.2%}")


def instrument_report(study: InstrumentStudy) -> str:
    full = study.populations["Full Population"]
    population_b = study.populations["Population B"]
    valid = sum(len(item.left_items) >= INSTRUMENT_BRANCH_MIN and len(item.right_items) >= INSTRUMENT_BRANCH_MIN for item in full.branches.values())
    lines = [
        f"APVA Structural Compression Study v0.1 - {study.instrument}",
        "=" * (41 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {len(study.rows)}",
        f"Family counts: {dict((name, sum(value == name for value in full.families.values())) for name in FAMILIES)}",
        f"Archetype counts: {dict((name, sum(value == name for value in full.archetypes.values())) for name in ARCHETYPES)}",
        f"Valid branch comparisons: {valid}",
    ]
    append_heading(lines, "Section 1 - Branch Comparison Diagnostics")
    append_branch_counts(lines, full)
    append_heading(lines, "Section 2 - Exact Path Separation")
    append_effects(lines, full, "ExactPath")
    append_heading(lines, "Section 3 - Structural Signature Separation")
    append_effects(lines, full, "StructuralSignature")
    append_heading(lines, "Section 4 - Family-Only and Archetype-Only Baselines")
    append_effects(lines, full, "FamilyOnly", 10)
    append_effects(lines, full, "ArchetypeOnly", 10)
    append_heading(lines, "Section 5 - Feature Set Power Comparison")
    append_powers(lines, full)
    append_heading(lines, "Section 6 - Retention Ratio and Compression Verdict")
    append_compression(lines, full)
    append_heading(lines, "Section 7 - Structural Feature Group Importance")
    append_signature_groups(lines, full)
    append_heading(lines, "Section 8 - Outcome Layer")
    append_outcomes(lines, full)
    append_heading(lines, "Section 9 - Population B")
    lines.append(f"Population B observations: {len(population_b.observations)}")
    append_branch_counts(lines, population_b)
    append_compression(lines, population_b)
    append_heading(lines, "Section 10 - Mechanical Research Notes")
    ratios = [branch_compression(item) for item in full.branches.values() if branch_compression(item).retention is not None]
    best = max(ratios, key=lambda item: item.retention or 0.0, default=None)
    lines.extend([
        f"- Highest structural retention: {best.branch} ({format_retention(best.retention)})." if best else "- No branch had measurable exact-path power.",
        "- Structural signatures are mechanical compressions of exact family paths.",
        "- Verdict thresholds are descriptive only: LikelyStructural >= 0.70, Mixed >= 0.40, LikelyPathSpecific < 0.40.",
        "- Population B remains separately reported and may be sparse.",
    ])
    return "\n".join(lines) + "\n"


def append_aggregate_powers(lines: list[str], items: list[AggregatePower], columns: list[str]) -> None:
    lines.append(f"{'BranchComparison':<92} {'FeatureSet':<20}" + "".join(f" {name + '_Power':>10}" for name in columns) + " ValidN  MeanPower  PowerStd  Replicated")
    for item in items:
        cells = "".join(f" {item.powers.get(name, 0.0):>10.4f}" for name in columns)
        lines.append(f"{item.branch:<92} {item.feature_set:<20}{cells} {item.valid_instruments:>6} {item.mean_power:>10.4f} {item.power_stddev:>9.4f} {item.replicated_count:>10}")


def append_aggregate_compression(lines: list[str], items: list[CompressionStats]) -> None:
    lines.append(f"{'BranchComparison':<92} {'Exact':>9} {'Structure':>10} {'Family':>9} {'Archetype':>10} {'Run':>9} {'Retention':>10} {'Verdict':<20} {'ValidN':>6} {'RepStruct':>9}")
    for item in items:
        lines.append(f"{item.branch:<92} {item.exact:>9.4f} {item.structural:>10.4f} {item.family:>9.4f} {item.archetype:>10.4f} {item.run:>9.4f} {format_retention(item.retention):>10} {item.verdict:<20} {item.valid_instruments:>6} {item.replicated_structural_count:>9}")


def append_aggregate_structural(lines: list[str], items: list[AggregateFeature], columns: list[str]) -> None:
    lines.append(f"{'BranchComparison':<92} {'Feature':<72} {'Group':<14}" + "".join(f" {name + '_Effect':>10}" for name in columns) + " ValidN Agree MeanEffect MeanAbs")
    for item in items:
        if item.valid_instruments < 2:
            continue
        cells = "".join(f" {item.effects.get(name, 0.0):>10.4f}" for name in columns)
        lines.append(f"{item.branch:<92} {item.feature:<72} {item.signature_group:<14}{cells} {item.valid_instruments:>6} {item.agreement_count:>5} {item.mean_effect:>10.4f} {item.mean_abs_effect:>7.4f}")


def aggregate_signature_groups(structural: list[AggregateFeature]) -> list[tuple[str, str, float, float, int]]:
    output = []
    for branch in sorted({item.branch for item in structural}):
        for group in SIGNATURE_GROUPS:
            items = [item for item in structural if item.branch == branch and item.signature_group == group and item.valid_instruments >= 2]
            output.append((branch, group, top_mean([item.mean_abs_effect for item in items], 5), max([item.mean_abs_effect for item in items], default=0.0), sum(item.agreement_count == item.valid_instruments for item in items)))
    return output


def append_aggregate_groups(lines: list[str], items: list[tuple[str, str, float, float, int]]) -> None:
    lines.append(f"{'BranchComparison':<92} {'Group':<14} {'Top5':>10} {'MaxAbs':>10} {'Replicated':>10}")
    for branch, group, top5, maximum, replicated in items:
        lines.append(f"{branch:<92} {group:<14} {top5:>10.4f} {maximum:>10.4f} {replicated:>10}")


def append_aggregate_outcomes(lines: list[str], items: list[AggregateOutcome], columns: list[str]) -> None:
    lines.append(f"{'BranchComparison':<92} {'Side':<6}" + "".join(f" {name + '_N':>8} {name + '_Cont':>9} {name + '_Fail':>9} {name + '_Mean':>10}" for name in columns) + " ValidN MeanCont MeanFail MeanSkew")
    for item in items:
        cells = "".join(f" {item.counts.get(name, 0):>8} {item.stats[name].continuation_rate if name in item.stats else 0.0:>8.2%} {item.stats[name].failure_rate if name in item.stats else 0.0:>8.2%} {item.stats[name].mean if name in item.stats else 0.0:>10.5f}" for name in columns)
        lines.append(f"{item.branch:<92} {item.side:<6}{cells} {item.valid_instruments:>6} {item.mean_continuation:>8.2%} {item.mean_failure:>8.2%} {item.mean_skew:>8.2%}")


def append_ranked(lines: list[str], title: str, items: list, key, formatter) -> None:
    lines.extend(["", title, "." * len(title)])
    eligible = [item for item in items if getattr(item, "valid_instruments", 2) >= 2]
    if not eligible:
        lines.append("No items met the two-instrument minimum.")
        return
    for rank, item in enumerate(sorted(eligible, key=key, reverse=True)[:TOP_LIMIT], 1):
        lines.append(f"{rank:>3}. {formatter(item)}")


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    columns = instrument_columns(studies)
    structural = aggregate_structural_features(studies)
    powers = aggregate_powers(studies, structural)
    compression = aggregate_compression(powers)
    groups = aggregate_signature_groups(structural)
    outcomes = aggregate_outcomes(studies)
    lines = [
        "APVA Structural Compression Study v0.1 - Aggregate",
        "=================================================",
        f"Instruments: {', '.join(study.instrument for study in studies)}",
        f"Aggregate branch side minimum: {AGGREGATE_BRANCH_MIN}; valid-instrument side minimum: {INSTRUMENT_BRANCH_MIN}.",
        f"Path reporting minimums: aggregate {AGGREGATE_PATH_MIN}; instrument {INSTRUMENT_PATH_MIN}.",
    ]
    append_heading(lines, "Aggregate Feature Set Power Table")
    append_aggregate_powers(lines, powers, columns)
    append_heading(lines, "Aggregate Compression Table")
    append_aggregate_compression(lines, compression)
    append_heading(lines, "Aggregate Structural Feature Table")
    append_aggregate_structural(lines, structural, columns)
    append_heading(lines, "Aggregate Structural Group Table")
    append_aggregate_groups(lines, groups)
    append_heading(lines, "Aggregate Outcome Table")
    append_aggregate_outcomes(lines, outcomes, columns)
    append_heading(lines, "Aggregate Rankings")
    append_ranked(lines, "1. Branches most likely structural", compression, lambda item: item.retention or -1.0, lambda item: f"{item.branch} | Retention={format_retention(item.retention)} Verdict={item.verdict}")
    append_ranked(lines, "2. Branches most likely path-specific", compression, lambda item: -(item.retention if item.retention is not None else 99.0), lambda item: f"{item.branch} | Retention={format_retention(item.retention)} Verdict={item.verdict}")
    for number, title, group in (
        (3, "Best structural signature separators", None),
        (4, "Best endpoint signatures", "Endpoint"),
        (5, "Best composition signatures", "Composition"),
        (6, "Best repetition signatures", "Repetition"),
        (7, "Best motif signatures", "Motif"),
        (8, "Best narrative-like signatures", "NarrativeLike"),
    ):
        selected = [item for item in structural if group is None or item.signature_group == group]
        append_ranked(lines, f"{number}. {title}", selected, lambda item: item.mean_abs_effect, lambda item: f"{item.branch} | {item.feature} | Group={item.signature_group} MeanAbs={item.mean_abs_effect:.4f} Agree={item.agreement_count}/{item.valid_instruments}")
    append_ranked(lines, "9. Cases where FamilyOnly nearly matches ExactPath", compression, lambda item: item.family / item.exact if item.exact else -1.0, lambda item: f"{item.branch} | Family/Exact={(item.family / item.exact if item.exact else 0.0):.4f}")
    append_ranked(lines, "10. Cases where ArchetypeOnly nearly matches ExactPath", compression, lambda item: item.archetype / item.exact if item.exact else -1.0, lambda item: f"{item.branch} | Archetype/Exact={(item.archetype / item.exact if item.exact else 0.0):.4f}")
    lines.extend(["", "11. Population-B compression differences", "." * 40, "Population B remains sparse; per-instrument compression summaries are reported without aggregate inference."])
    append_heading(lines, "Cross-Instrument Mechanical Research Notes")
    likely = [item for item in compression if item.verdict == "LikelyStructural" and item.valid_instruments >= 2 and item.replicated_structural_count >= 3]
    lines.extend([
        f"- Branches meeting the stronger compressed-signature criteria: {', '.join(item.branch for item in likely) if likely else 'none'}.",
        "- RetentionRatio compares StructuralSignature MeanAbsEffectTop10 against ExactPath MeanAbsEffectTop10.",
        "- Endpoint, composition, repetition, motif, and narrative-like summaries are mechanical feature groups.",
        "- Family-only, archetype-only, run-only, and Population B comparisons are descriptive only.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    studies = [study_instrument(path) for path in args.input_paths]
    seen = set()
    for study in studies:
        if study.instrument in seen:
            raise ValueError(f"Duplicate instrument input: {study.instrument}")
        seen.add(study.instrument)
        output = Path("Evidence/Output") / study.instrument / f"StructuralCompression_{study.instrument}.txt"
        write_text(output, instrument_report(study))
    report = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, report)
    print(report, end="")


if __name__ == "__main__":
    main()
