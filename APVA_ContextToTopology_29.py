"""APVA Context-to-Topology Study v0.1.

Research-only upstream context signature study over existing condition topologies.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from APVA_AcceptanceAnatomy_23 import mature_aligned_lateral_indexes
from APVA_ConditionTopology_27 import flags, matches
from APVA_ContextTopology_28 import (
    CONTEXT_WINDOWS,
    NEGATIVE_NAMES,
    POSITIVE_NAMES,
    TARGETS,
    context_metrics,
)
from APVA_BreakoutContext_08 import EvidenceBar, load_rows
from APVA_LateralAnatomy_19 import effect_size
from APVA_RegimeTransition_16 import CANONICAL_INSTRUMENTS, instrument_name, write_text
from APVA_StateGrammar_25 import mean


AGGREGATE_OUTPUT = Path("Evidence/Output/ContextToTopology/ContextToTopology_All.txt")
PER_INSTRUMENT_MIN_COUNT = 20
TOP_LIMIT = 20
SIMILARITY_THRESHOLD = 0.75

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


@dataclass(frozen=True)
class SignatureFeature:
    feature: str
    present_count: int
    absent_count: int
    present_mean: float
    absent_mean: float
    delta: float
    effect: float


@dataclass(frozen=True)
class TopologySignature:
    topology: str
    group: str
    total_bars: int
    present_count: int
    absent_count: int
    features: list[SignatureFeature]


@dataclass(frozen=True)
class Similarity:
    left: str
    right: str
    similarity: float


@dataclass(frozen=True)
class PopulationStudy:
    name: str
    indexes: list[int]
    signatures: dict[str, TopologySignature]
    similarities: list[Similarity]


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    rows: list[EvidenceBar]
    populations: dict[str, PopulationStudy]


@dataclass(frozen=True)
class AggregateFeature:
    topology: str
    group: str
    feature: str
    values: dict[str, SignatureFeature]
    valid_instruments: int
    positive_count: int
    negative_count: int
    mean_effect: float


@dataclass(frozen=True)
class AggregateSimilarity:
    left: str
    right: str
    values: dict[str, float]
    valid_instruments: int
    mean_similarity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study whether historical APVA context generates current condition topologies."
    )
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def signature(rows: list[EvidenceBar], indexes: list[int], topology: str) -> TopologySignature:
    conditions = TARGETS[topology]
    present_metrics = []
    absent_metrics = []
    for index in indexes:
        metrics = context_metrics(rows, index)
        destination = present_metrics if matches(flags(rows[index]), conditions) else absent_metrics
        destination.append(metrics)
    features = []
    for feature in FEATURES:
        present = [metrics[feature] for metrics in present_metrics]
        absent = [metrics[feature] for metrics in absent_metrics]
        features.append(
            SignatureFeature(
                feature,
                len(present),
                len(absent),
                mean(present),
                mean(absent),
                mean(present) - mean(absent),
                effect_size(present, absent),
            )
        )
    return TopologySignature(
        topology,
        "Positive" if topology in POSITIVE_NAMES else "Negative",
        len(indexes),
        len(present_metrics),
        len(absent_metrics),
        features,
    )


def effect_vector(value: TopologySignature) -> list[float]:
    return [feature.effect for feature in value.features]


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def similarities(signatures: dict[str, TopologySignature]) -> list[Similarity]:
    names = list(signatures)
    output = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            output.append(Similarity(left, right, cosine(effect_vector(signatures[left]), effect_vector(signatures[right]))))
    return output


def build_population(rows: list[EvidenceBar], name: str, indexes: list[int]) -> PopulationStudy:
    values = {topology: signature(rows, indexes, topology) for topology in TARGETS}
    return PopulationStudy(name, indexes, values, similarities(values))


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


def ranked_features(value: TopologySignature, positive: bool) -> list[SignatureFeature]:
    return sorted(
        value.features,
        key=lambda item: (-item.effect if positive else item.effect, item.feature),
    )


def append_signature_table(lines: list[str], title: str, value: TopologySignature) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(
        f"{value.topology}: Group={value.group}; TotalBars={value.total_bars}; "
        f"TopologyPresent={value.present_count}; TopologyAbsent={value.absent_count}"
    )
    lines.append(
        f"{'Feature':<32} {'PresentN':>8} {'AbsentN':>8} {'PresentMean':>12} "
        f"{'AbsentMean':>12} {'Delta':>12} {'EffectSize':>11}"
    )
    for item in value.features:
        lines.append(
            f"{item.feature:<32} {item.present_count:>8} {item.absent_count:>8} "
            f"{item.present_mean:>12.6f} {item.absent_mean:>12.6f} "
            f"{item.delta:>12.6f} {item.effect:>11.6f}"
        )


def append_feature_rankings(lines: list[str], value: TopologySignature) -> None:
    lines.extend([f"\n{value.topology} Ranked Signature Features", "-" * (len(value.topology) + 26)])
    for title, items in (
        ("Top 20 Positive Context Features", ranked_features(value, True)),
        ("Top 20 Negative Context Features", ranked_features(value, False)),
    ):
        lines.extend([f"\n{title}", "." * len(title)])
        for rank, item in enumerate(items[:TOP_LIMIT], start=1):
            lines.append(f"{rank:>3}. {item.feature:<32} Delta={item.delta:>10.6f} Effect={item.effect:>10.6f}")


def append_window_comparison(lines: list[str], signatures: dict[str, TopologySignature]) -> None:
    lines.extend(["\nSection 4 - Context Window Comparison", "====================================="])
    lines.append(f"{'Topology':<38} {'Window':>6} {'BestFeature':<32} {'EffectSize':>11}")
    for value in signatures.values():
        for size in CONTEXT_WINDOWS:
            selected = [item for item in value.features if item.feature.startswith(f"Prev{size}")]
            best = max(selected, key=lambda item: abs(item.effect))
            lines.append(f"{value.topology:<38} {('Prev' + str(size)):>6} {best.feature:<32} {best.effect:>11.6f}")


def append_similarity_matrix(lines: list[str], population: PopulationStudy) -> None:
    names = list(population.signatures)
    lookup = {(item.left, item.right): item.similarity for item in population.similarities}
    lines.extend(["\nSection 5 - Context Similarity Matrix", "====================================="])
    header = f"{'Topology':<38}" + "".join(f" {name[:12]:>12}" for name in names)
    lines.append(header)
    for left in names:
        row = f"{left:<38}"
        for right in names:
            if left == right:
                value = 1.0
            else:
                value = lookup.get((left, right), lookup.get((right, left), 0.0))
            row += f" {value:>12.4f}"
        lines.append(row)


def append_family_candidates(lines: list[str], population: PopulationStudy) -> None:
    lines.extend(["\nSection 6 - Topology Family Grouping", "===================================="])
    lines.append(f"Family-candidate threshold: cosine similarity >= {SIMILARITY_THRESHOLD:.2f}")
    lines.append(f"{'Topology A':<38} {'Topology B':<38} {'Similarity':>11}")
    candidates = sorted(
        [item for item in population.similarities if item.similarity >= SIMILARITY_THRESHOLD],
        key=lambda item: (-item.similarity, item.left, item.right),
    )
    for item in candidates:
        lines.append(f"{item.left:<38} {item.right:<38} {item.similarity:>11.4f}")


def append_population(lines: list[str], population: PopulationStudy, prefix: str) -> None:
    for value in population.signatures.values():
        append_signature_table(lines, f"{prefix}: {value.topology}", value)
        append_feature_rankings(lines, value)


def instrument_report(study: InstrumentStudy) -> str:
    full = study.populations["Full Population"]
    population_b = study.populations["Population B"]
    lines = [
        f"APVA Context-to-Topology Study v0.1 - {study.instrument}",
        "=" * (38 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {len(study.rows)}",
        f"Population B rows: {len(population_b.indexes)}",
        "Historical context is compared for topology-present versus topology-absent bars.",
        "This report measures Context -> Topology only; it does not measure forward outcomes.",
        "",
        "Section 1 - Diagnostics",
        "=======================",
        f"{'Topology':<38} {'Group':<9} {'Present':>9} {'Absent':>9}",
    ]
    for value in full.signatures.values():
        lines.append(f"{value.topology:<38} {value.group:<9} {value.present_count:>9} {value.absent_count:>9}")
    lines.extend(["\nSection 2 - Topology Signature Table", "===================================="])
    append_population(lines, full, "Full Population")
    lines.extend(["\nSection 3 - Ranked Signature Features", "====================================="])
    lines.append("Rankings are printed below each topology signature table.")
    append_window_comparison(lines, full.signatures)
    append_similarity_matrix(lines, full)
    append_family_candidates(lines, full)
    lines.extend(["\nSection 7 - Population B Repeat", "==============================="])
    append_population(lines, population_b, "Population B")
    lines.extend(["\nResearch Notes", "=============="])
    for topology in ("Accepted+Compressed", "Accepted+Compressed+Dissipating", "Accepted+Peak"):
        value = full.signatures.get(topology)
        if value:
            strongest = max(value.features, key=lambda item: abs(item.effect))
            lines.append(f"- {topology}: strongest context signature feature={strongest.feature}, EffectSize={strongest.effect:.6f}.")
    prev10_wins = 0
    for value in full.signatures.values():
        strongest = max(value.features, key=lambda item: abs(item.effect))
        prev10_wins += strongest.feature.startswith("Prev10")
    lines.append(f"- Prev10 supplies the strongest absolute feature for {prev10_wins}/{len(full.signatures)} topologies.")
    lines.append("- Similarity groups are descriptive family candidates only; they do not define a new ontology.")
    return "\n".join(lines) + "\n"


def instrument_columns(studies: list[InstrumentStudy]) -> list[str]:
    available = [study.instrument for study in studies]
    return list(CANONICAL_INSTRUMENTS) + sorted(set(available) - set(CANONICAL_INSTRUMENTS))


def aggregate_features(studies: list[InstrumentStudy], population: str) -> list[AggregateFeature]:
    output = []
    for topology in TARGETS:
        group = "Positive" if topology in POSITIVE_NAMES else "Negative"
        for feature in FEATURES:
            values = {
                study.instrument: next(
                    item for item in study.populations[population].signatures[topology].features
                    if item.feature == feature
                )
                for study in studies
            }
            valid = [item for item in values.values() if item.present_count >= PER_INSTRUMENT_MIN_COUNT]
            output.append(
                AggregateFeature(
                    topology,
                    group,
                    feature,
                    values,
                    len(valid),
                    sum(item.effect > 0.0 for item in valid),
                    sum(item.effect < 0.0 for item in valid),
                    mean([item.effect for item in valid]),
                )
            )
    return output


def aggregate_similarities(studies: list[InstrumentStudy]) -> list[AggregateSimilarity]:
    keys = [(left, right) for index, left in enumerate(TARGETS) for right in list(TARGETS)[index + 1 :]]
    output = []
    for left, right in keys:
        values = {}
        for study in studies:
            population = study.populations["Full Population"]
            item = next(
                value for value in population.similarities
                if {value.left, value.right} == {left, right}
            )
            values[study.instrument] = item.similarity
        output.append(AggregateSimilarity(left, right, values, len(values), mean(list(values.values()))))
    return output


def append_aggregate_table(lines: list[str], items: list[AggregateFeature], columns: list[str]) -> None:
    lines.extend(["\nAggregate Topology-Feature Table", "================================"])
    header = f"{'Topology':<38} {'Group':<9} {'Feature':<32}"
    for instrument in columns:
        header += f" {('Effect_' + instrument):>11}"
    header += f" {'Valid':>5} {'Pos':>4} {'Neg':>4} {'MeanEffect':>11}"
    lines.append(header)
    for item in items:
        row = f"{item.topology:<38} {item.group:<9} {item.feature:<32}"
        for instrument in columns:
            value = item.values.get(instrument)
            row += f" {(value.effect if value else 0.0):>11.6f}"
        row += (
            f" {item.valid_instruments:>5} {item.positive_count:>4} "
            f"{item.negative_count:>4} {item.mean_effect:>11.6f}"
        )
        lines.append(row)


def ranked_aggregate(
    items: list[AggregateFeature],
    mode: str,
    group: str | None = None,
) -> list[AggregateFeature]:
    eligible = [
        item for item in items
        if item.valid_instruments >= 2 and (group is None or item.group == group)
    ]
    if mode == "positive":
        return sorted(eligible, key=lambda item: (-item.positive_count, -item.mean_effect, item.topology, item.feature))
    if mode == "negative":
        return sorted(eligible, key=lambda item: (-item.negative_count, item.mean_effect, item.topology, item.feature))
    return sorted(eligible, key=lambda item: (-abs(item.mean_effect), item.topology, item.feature))


def append_aggregate_ranked(lines: list[str], title: str, items: list[AggregateFeature]) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(f"{'Rank':>4} {'Topology':<38} {'Feature':<32} {'Valid':>5} {'Pos':>4} {'Neg':>4} {'MeanEffect':>11}")
    for rank, item in enumerate(items[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {item.topology:<38} {item.feature:<32} {item.valid_instruments:>5} "
            f"{item.positive_count:>4} {item.negative_count:>4} {item.mean_effect:>11.6f}"
        )


def append_similarity_ranked(lines: list[str], title: str, items: list[AggregateSimilarity], reverse: bool) -> None:
    lines.extend([f"\n{title}", "-" * len(title)])
    lines.append(f"{'Rank':>4} {'Topology A':<38} {'Topology B':<38} {'MeanSimilarity':>14}")
    ordered = sorted(items, key=lambda item: (-item.mean_similarity if reverse else item.mean_similarity, item.left, item.right))
    for rank, item in enumerate(ordered[:TOP_LIMIT], start=1):
        lines.append(f"{rank:>4} {item.left:<38} {item.right:<38} {item.mean_similarity:>14.6f}")


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    columns = instrument_columns(studies)
    full = aggregate_features(studies, "Full Population")
    population_b = aggregate_features(studies, "Population B")
    similarity = aggregate_similarities(studies)
    lines = [
        "APVA Context-to-Topology Study v0.1 - Cross-Instrument Aggregate",
        "================================================================",
        f"Input instruments: {', '.join(study.instrument for study in studies)}",
        f"Replication threshold: topology present count >= {PER_INSTRUMENT_MIN_COUNT} in at least two instruments.",
        "Effects compare historical context for topology-present versus topology-absent bars.",
    ]
    append_aggregate_table(lines, full, columns)
    append_aggregate_ranked(lines, "Most Replicated Positive Context Predictors of Positive Topologies", ranked_aggregate(full, "positive", "Positive"))
    append_aggregate_ranked(lines, "Most Replicated Negative Context Predictors of Positive Topologies", ranked_aggregate(full, "negative", "Positive"))
    append_aggregate_ranked(lines, "Most Replicated Positive Context Predictors of Negative Topologies", ranked_aggregate(full, "positive", "Negative"))
    append_aggregate_ranked(lines, "Most Replicated Negative Context Predictors of Negative Topologies", ranked_aggregate(full, "negative", "Negative"))
    append_aggregate_ranked(lines, "Strongest Positive Topology Signatures", ranked_aggregate(full, "positive"))
    append_aggregate_ranked(lines, "Strongest Negative Topology Signatures", ranked_aggregate(full, "negative"))
    append_similarity_ranked(lines, "Most Similar Topology Pairs", similarity, True)
    append_similarity_ranked(lines, "Most Distinct Topology Pairs", similarity, False)
    append_aggregate_ranked(lines, "Population-B Strongest Topology Predictors", ranked_aggregate(population_b, "absolute"))
    lines.extend(["\nCross-Instrument Research Notes", "==============================="])
    ranked = ranked_aggregate(full, "absolute")
    if ranked:
        lines.append(f"- Strongest replicated context-to-topology feature: {ranked[0].topology} | {ranked[0].feature}, MeanEffect={ranked[0].mean_effect:.6f}.")
    lines.append("- Similarity compares effect-size vectors and is descriptive only.")
    lines.append("- This report studies context upstream of topology; it does not measure topology consequences.")
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
            Path("Evidence") / "Output" / study.instrument / f"ContextToTopology_{study.instrument}.txt",
            instrument_report(study),
        )
    report = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, report)
    print(report, end="")


if __name__ == "__main__":
    main()
