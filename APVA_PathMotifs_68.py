#!/usr/bin/env python3
"""APVA Path Motif Study v0.1.

Audit recurring PreviousNode -> CurrentNode -> NextNode structural motifs.
Forward price outcomes are diagnostics only.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from APVA_BranchForecast_65 import build_result as build_branch_result, build_stream, mean, model_result
from APVA_InformationContribution_63 import thresholds
from APVA_InformationDecay_57 import study as decay_study
from APVA_JunctionAudit_67 import BaseBranch, ConditionalBranch, base_profile, conditional_profile, entropy, state_of
from APVA_MinimalEngine_52 import directional_return, instrument_columns, load_results
from APVA_NodeImportance_58 import aggregate_rows, local_rows, score_rows
from APVA_ProcessGraph_54 import node_for, node_text
from APVA_StructuralForecast_56 import Outcome, outcome
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct

JUNCTION_NODES = {
    "RecoveryResolution_Age1",
    "ReassertionProcessing_Age1",
    "DestructiveRotation_Age1",
    "MixedStructure_Age1",
    "ConstructiveEmergence_Age1",
    "ExhaustionPersistence_Age1",
    "CompressionProcessing_Age1",
}
PIPELINE_NODES = {
    "NeutralProcessing_Age3",
    "NeutralProcessing_Age4",
    "NeutralProcessing_Age5",
    "NeutralProcessing_Age11-20",
}
HORIZONS = (2, 3, 5)


@dataclass
class MotifRow:
    previous: str
    current: str
    next_node: str
    count: int
    probability: float
    base_probability: float
    delta: float
    changed: bool
    conditional_entropy: float
    entropy_reduction: float
    replication_count: int = 0
    contribution: float = 0.0
    top1: float = 0.0
    top2: float = 0.0
    top3: float = 0.0
    brier: float = 0.0
    calibration: float = 0.0
    motif_class: str = ""


@dataclass
class PrefixEntropy:
    previous: str
    current: str
    count: int
    conditional_entropy: float
    base_entropy: float
    entropy_reduction: float


@dataclass
class PersistenceRow:
    previous: str
    current: str
    next_node: str
    horizon: int
    dominant_node: str
    probability: float
    count: int


@dataclass
class ConcentrationRow:
    current: str
    count: int
    top_share: float
    top3_share: float
    top5_share: float


@dataclass
class FamilyRow:
    family: str
    count: int
    motif_count: int
    entropy: float
    replication_count: int


@dataclass
class GrammarRow:
    current: str
    incoming_count: int
    outgoing_count: int
    branch_factor: int


@dataclass
class AgreementRow:
    previous: str
    current: str
    next_node: str
    mean_probability: float
    stdev_probability: float
    probabilities: dict[str, float]


@dataclass
class Result:
    instrument: str
    source_paths: list
    bars: list
    rows: list
    segments: list[list]
    branch_result: object
    base: dict[tuple[str, str], BaseBranch]
    conditional: dict[tuple[str, str, str], ConditionalBranch]
    motifs: dict[tuple[str, str, str], MotifRow]
    prefix_entropy: dict[tuple[str, str], PrefixEntropy]
    persistence: dict[tuple[str, str, str, int], PersistenceRow]
    concentration: dict[str, ConcentrationRow]
    families: dict[str, FamilyRow]
    grammar: dict[str, GrammarRow]
    outcomes: dict[tuple[str, str, str], Outcome]


def motif_key(row) -> tuple[str, str, str]:
    return row.previous, node_text(row.node), row.next_node


def distribution_brier(distribution: dict[str, float], actual: str, universe: Iterable[str]) -> float:
    return sum((distribution.get(item, 0.0) - float(item == actual)) ** 2 for item in universe)


def motif_inventory(rows: list, base: dict[tuple[str, str], BaseBranch],
                    conditional: dict[tuple[str, str, str], ConditionalBranch],
                    branch_result) -> dict[tuple[str, str, str], MotifRow]:
    counts = Counter(motif_key(row) for row in rows)
    grouped: dict[tuple[str, str], list[ConditionalBranch]] = defaultdict(list)
    for row in conditional.values():
        grouped[(row.previous, row.current)].append(row)
    base_dominants = {}
    for (current, _), row in base.items():
        if current not in base_dominants or row.rank < base_dominants[current].rank:
            base_dominants[current] = row
    universe = tuple(sorted({row.next_node for row in rows}))
    output = {}
    for key, count in counts.items():
        previous, current, next_node = key
        cond = conditional.get(key)
        cond_probability = cond.probability if cond else 0.0
        base_probability = base.get((current, next_node)).probability if (current, next_node) in base else 0.0
        base_dom = base_dominants.get(current)
        changed = bool(cond and base_dom and cond.rank == 1 and cond.next_node != base_dom.next_node)
        distribution = {item.next_node: item.probability for item in grouped.get((previous, current), [])}
        conditional_entropy = entropy(distribution.values())
        base_entropy = branch_result.entropy_rows[current].entropy if current in branch_result.entropy_rows else 0.0
        ranked = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
        choices = [item for item, _ in ranked]
        brier = distribution_brier(distribution, next_node, universe)
        output[key] = MotifRow(
            previous, current, next_node, count, cond_probability, base_probability,
            cond_probability - base_probability, changed,
            conditional_entropy, base_entropy - conditional_entropy,
            top1=float(next_node in choices[:1]), top2=float(next_node in choices[:2]),
            top3=float(next_node in choices[:3]), brier=brier,
            calibration=abs(cond_probability - float(next_node in choices[:1])),
        )
    return output


def prefix_entropy_rows(rows: list, branch_result) -> dict[tuple[str, str], PrefixEntropy]:
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[(row.previous, node_text(row.node))][row.next_node] += 1
    output = {}
    for key, counts in grouped.items():
        total = sum(counts.values())
        probabilities = [count / total for count in counts.values()] if total else []
        conditional = entropy(probabilities)
        current = key[1]
        base = branch_result.entropy_rows[current].entropy if current in branch_result.entropy_rows else 0.0
        output[key] = PrefixEntropy(key[0], current, total, conditional, base, base - conditional)
    return output


def persistence_rows(segments: list[list]) -> dict[tuple[str, str, str, int], PersistenceRow]:
    nested: dict[tuple[str, str, str, int], Counter[str]] = defaultdict(Counter)
    for segment in segments:
        for index, row in enumerate(segment):
            key = motif_key(row)
            for horizon in HORIZONS:
                if index + horizon < len(segment):
                    nested[(*key, horizon)][node_text(segment[index + horizon].node)] += 1
    output = {}
    for key, counts in nested.items():
        total = sum(counts.values())
        dominant, count = max(counts.items(), key=lambda item: (item[1], item[0]))
        output[key] = PersistenceRow(key[0], key[1], key[2], key[3], dominant, count / total if total else 0.0, total)
    return output


def concentration_rows(motifs: dict[tuple[str, str, str], MotifRow]) -> dict[str, ConcentrationRow]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in motifs.values():
        grouped[row.current].append(row.count)
    output = {}
    for current, counts in grouped.items():
        total = sum(counts)
        ordered = sorted(counts, reverse=True)
        output[current] = ConcentrationRow(
            current, total,
            ordered[0] / total if total else 0.0,
            sum(ordered[:3]) / total if total else 0.0,
            sum(ordered[:5]) / total if total else 0.0,
        )
    return output


def family_rows(motifs: dict[tuple[str, str, str], MotifRow]) -> dict[str, FamilyRow]:
    grouped: dict[str, list[MotifRow]] = defaultdict(list)
    for row in motifs.values():
        grouped[state_of(row.current)].append(row)
    output = {}
    for family, rows in grouped.items():
        total = sum(row.count for row in rows)
        probabilities = [row.count / total for row in rows] if total else []
        output[family] = FamilyRow(
            family, total, len(rows), entropy(probabilities),
            max((row.replication_count for row in rows), default=0),
        )
    return output


def grammar_rows(motifs: dict[tuple[str, str, str], MotifRow]) -> dict[str, GrammarRow]:
    incoming: dict[str, set[str]] = defaultdict(set)
    outgoing: dict[str, set[str]] = defaultdict(set)
    for row in motifs.values():
        incoming[row.current].add(row.previous)
        outgoing[row.current].add(row.next_node)
    return {
        current: GrammarRow(current, len(incoming[current]), len(outgoing[current]),
                            len(incoming[current]) * len(outgoing[current]))
        for current in sorted(set(incoming) | set(outgoing))
    }


def contribution_scores(rows: list, motifs: dict[tuple[str, str, str], MotifRow], baseline) -> None:
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[(row.previous, node_text(row.node))][row.next_node] += 1
    dominant = {
        key: max(counts.items(), key=lambda item: (item[1], item[0]))[0]
        for key, counts in grouped.items()
    }
    correct_by_prefix = {
        key: counts[dominant[key]]
        for key, counts in grouped.items()
    }
    baseline_correct = sum(correct_by_prefix.values())
    total = len(rows)
    base_top1 = baseline.metrics.top1
    for key, motif in motifs.items():
        prefix = (motif.previous, motif.current)
        remaining_total = total - motif.count
        if remaining_total <= 0:
            motif.contribution = 0.0
            continue
        new_counts = Counter(grouped[prefix])
        new_counts[motif.next_node] -= motif.count
        if new_counts[motif.next_node] <= 0:
            del new_counts[motif.next_node]
        new_prefix_correct = 0
        if new_counts:
            new_dominant = max(new_counts.items(), key=lambda item: (item[1], item[0]))[0]
            new_prefix_correct = new_counts[new_dominant]
        degraded_correct = baseline_correct - correct_by_prefix[prefix] + new_prefix_correct
        degraded_top1 = degraded_correct / remaining_total
        motif.contribution = base_top1 - degraded_top1


def outcome_rows(bars: list, rows: list) -> dict[tuple[str, str, str], Outcome]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = directional_return(bars, row.index, 5)
        if value is not None:
            grouped[motif_key(row)].append(value)
    return {key: outcome(values) for key, values in grouped.items()}


def attach_replication(aggregate: Result, instruments: list[Result]) -> None:
    for key, row in aggregate.motifs.items():
        row.replication_count = sum(key in result.motifs for result in instruments)
    median_entropy = statistics.median(row.conditional_entropy for row in aggregate.motifs.values()) if aggregate.motifs else 0.0
    for row in aggregate.motifs.values():
        if row.count >= 50 and row.replication_count >= 2 and abs(row.delta) >= 0.15:
            row.motif_class = "StrongMotif"
        elif row.count >= 50 and row.replication_count >= 2 and row.conditional_entropy < median_entropy:
            row.motif_class = "StableMotif"
        elif row.count < 50 and row.replication_count >= 2:
            row.motif_class = "EmergentMotif"
        else:
            row.motif_class = "UnclassifiedMotif"
    aggregate.families = family_rows(aggregate.motifs)


def build_result(instrument: str, source_paths: list, bars: list, rows: list, segments: list[list]) -> Result:
    branch = build_branch_result(instrument, source_paths, bars, rows, segments)
    base = base_profile(rows)
    conditional = conditional_profile(rows, base)
    motifs = motif_inventory(rows, base, conditional, branch)
    contribution_scores(rows, motifs, branch.models["Model2_CurrentPreviousNode"])
    return Result(
        instrument, source_paths, bars, rows, segments, branch, base, conditional, motifs,
        prefix_entropy_rows(rows, branch), persistence_rows(segments),
        concentration_rows(motifs), family_rows(motifs), grammar_rows(motifs),
        outcome_rows(bars, rows),
    )


def motif_text(key: tuple[str, str, str]) -> str:
    return f"{key[0]} -> {key[1]} -> {key[2]}"


def motif_line(row: MotifRow) -> str:
    return (f"{row.previous} | {row.current} | {row.next_node} | {row.count} | {pct(row.probability)} | "
            f"{pct(row.base_probability)} | {pct(row.delta)} | {row.changed} | {fmt(row.conditional_entropy)} | "
            f"{fmt(row.entropy_reduction)} | {row.replication_count} | {row.motif_class}")


def limited(rows: Iterable, key, line, limit: int = 25) -> list[str]:
    return [line(row) for row in sorted(rows, key=key)[:limit]]


def append_common_sections(lines: list[str], result: Result, limit: int = 100) -> None:
    lines += ["", "1. Motif Inventory", "PreviousNode | CurrentNode | NextNode | Count | Probability | BaseProbability | ProbabilityDelta | DominantBranchChanged | ConditionalEntropy | EntropyReduction | ReplicationCount | MotifClass"]
    lines += [motif_line(row) for row in sorted(result.motifs.values(), key=lambda row: (-row.count, row.previous, row.current, row.next_node))[:limit]]
    lines += ["", "2. Motif Frequency Ranking", "Top motifs by count."]
    lines += limited(result.motifs.values(), lambda row: (-row.count, row.previous, row.current, row.next_node),
                     lambda row: f"{motif_text((row.previous, row.current, row.next_node))} | {row.count}", limit)
    lines += ["", "3. Motif Stability", "Motif | ConditionalProbability | BaseProbability | ProbabilityDelta | DominantBranchChanged"]
    lines += limited(result.motifs.values(), lambda row: (-abs(row.delta), -row.count, row.current),
                     lambda row: f"{motif_text((row.previous, row.current, row.next_node))} | {pct(row.probability)} | {pct(row.base_probability)} | {pct(row.delta)} | {row.changed}", limit)
    lines += ["", "4. Motif Entropy", "PreviousNode | CurrentNode | Count | ConditionalEntropy | BaseEntropy | EntropyReduction"]
    lines += limited(result.prefix_entropy.values(), lambda row: (-row.entropy_reduction, row.current, row.previous),
                     lambda row: f"{row.previous} | {row.current} | {row.count} | {fmt(row.conditional_entropy)} | {fmt(row.base_entropy)} | {fmt(row.entropy_reduction)}", limit)
    lines += ["", "5. Replication", "Motif | ReplicationCount | ReplicationPercent | ReplicationClass"]
    for row in sorted(result.motifs.values(), key=lambda row: (-row.replication_count, -row.count, row.current))[:limit]:
        label = "StrongReplication" if row.replication_count >= 3 else "ModerateReplication" if row.replication_count == 2 else "WeakReplication"
        lines.append(f"{motif_text((row.previous, row.current, row.next_node))} | {row.replication_count} | {pct(row.replication_count / 3)} | {label}")
    lines += ["", "6. Persistence", "Motif | Horizon | DominantNode | Probability | Count"]
    lines += limited(result.persistence.values(), lambda row: (-row.probability, -row.count, row.current),
                     lambda row: f"{motif_text((row.previous, row.current, row.next_node))} | t+{row.horizon} | {row.dominant_node} | {pct(row.probability)} | {row.count}", limit)
    lines += ["", "7. Junction Motifs", "Motif | Count | Probability | ProbabilityDelta | EntropyReduction"]
    lines += limited((row for row in result.motifs.values() if row.current in JUNCTION_NODES),
                     lambda row: (-abs(row.delta), -row.count, row.current),
                     lambda row: f"{motif_text((row.previous, row.current, row.next_node))} | {row.count} | {pct(row.probability)} | {pct(row.delta)} | {fmt(row.entropy_reduction)}", limit)
    lines += ["", "8. Pipeline Motifs", "Motif | Count | Probability | ConditionalEntropy"]
    lines += limited((row for row in result.motifs.values() if row.current in PIPELINE_NODES),
                     lambda row: (-row.probability, -row.count, row.current),
                     lambda row: f"{motif_text((row.previous, row.current, row.next_node))} | {row.count} | {pct(row.probability)} | {fmt(row.conditional_entropy)}", limit)
    lines += ["", "9. Motif Concentration", "CurrentNode | Count | TopMotifShare | Top3MotifShare | Top5MotifShare"]
    lines += limited(result.concentration.values(), lambda row: (-row.top_share, row.current),
                     lambda row: f"{row.current} | {row.count} | {pct(row.top_share)} | {pct(row.top3_share)} | {pct(row.top5_share)}", limit)
    lines += ["", "10. Motif Families", "Family | FamilyCount | MotifCount | FamilyEntropy | FamilyReplication"]
    lines += limited(result.families.values(), lambda row: (-row.count, row.family),
                     lambda row: f"{row.family} | {row.count} | {row.motif_count} | {fmt(row.entropy)} | {row.replication_count}", limit)
    lines += ["", "11. Contribution", "Motif | ContributionScore | ReplicationCount"]
    lines += limited(result.motifs.values(), lambda row: (-row.contribution, -row.count, row.current),
                     lambda row: f"{motif_text((row.previous, row.current, row.next_node))} | {fmt(row.contribution)} | {row.replication_count}", limit)
    lines += ["", "12. Motif Grammar", "CurrentNode | IncomingNodes | OutgoingNodes | MotifBranchFactor"]
    lines += limited(result.grammar.values(), lambda row: (-row.branch_factor, row.current),
                     lambda row: f"{row.current} | {row.incoming_count} | {row.outgoing_count} | {row.branch_factor}", limit)
    lines += ["", "13. Motif Predictability", "Motif | Top1Accuracy | Top2Accuracy | Top3Accuracy | BrierScore | CalibrationError"]
    lines += limited(result.motifs.values(), lambda row: (-row.top1, -row.top2, row.brier, row.current),
                     lambda row: f"{motif_text((row.previous, row.current, row.next_node))} | {pct(row.top1)} | {pct(row.top2)} | {pct(row.top3)} | {fmt(row.brier)} | {pct(row.calibration)}", limit)
    lines += ["", "14. Cross-Instrument Agreement", "See aggregate report for instrument probability agreement."]
    lines += ["", "15. Outcome Diagnostics", "Diagnostic only. Outcomes are not used in motif construction.", "Motif | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew"]
    for key, row in sorted(result.outcomes.items(), key=lambda item: (-item[1].count, item[0]))[:limit]:
        lines.append(f"{motif_text(key)} | {row.count} | {fmt(row.mean_dr)} | {fmt(row.median_dr)} | {pct(row.continuation)} | {pct(row.failure)} | {pct(row.flat)} | {fmt(row.skew)}")
    lines += ["", "16. Motif Classification", "StrongMotif: Count >= 50, ReplicationCount >= 2, and AbsoluteProbabilityDelta >= 15%. StableMotif: Count >= 50, ReplicationCount >= 2, and ConditionalEntropy below median. EmergentMotif: Count < 50 and ReplicationCount >= 2."]
    lines += [f"{motif_text((row.previous, row.current, row.next_node))} | {row.count} | {pct(abs(row.delta))} | {fmt(row.conditional_entropy)} | {row.replication_count} | {row.motif_class}"
              for row in sorted(result.motifs.values(), key=lambda row: (row.motif_class, -abs(row.delta), -row.count))[:limit]]


def recommendation(result: Result) -> list[str]:
    strong = sum(row.motif_class == "StrongMotif" for row in result.motifs.values())
    stable = sum(row.motif_class == "StableMotif" for row in result.motifs.values())
    emergent = sum(row.motif_class == "EmergentMotif" for row in result.motifs.values())
    node_top1 = result.branch_result.models["Model1_CurrentNode"].metrics.top1
    motif_top1 = result.branch_result.models["Model2_CurrentPreviousNode"].metrics.top1
    improvement = motif_top1 - node_top1
    if improvement >= 0.05 and strong >= 10:
        label = "MotifDominant"
    elif improvement > 0.01 or strong or stable:
        label = "MotifEnhanced"
    else:
        label = "StateGraphDominant"
    return [
        f"Classification: {label}",
        f"StrongMotifCount: {strong}",
        f"StableMotifCount: {stable}",
        f"EmergentMotifCount: {emergent}",
        f"ReplicationAssessment: Max motif replication = {max((row.replication_count for row in result.motifs.values()), default=0)} instruments.",
        f"NodeVsMotifAssessment: CurrentNode Top1={pct(node_top1)}; PreviousNode+CurrentNode Top1={pct(motif_top1)}; Gain={pct(improvement)}.",
    ]


def append_audit(lines: list[str]) -> None:
    lines += [
        "",
        "18. Low-DoF Audit",
        "Variables used:",
        "- PreviousNode",
        "- CurrentNode",
        "- NextNode",
        "- StructuralState",
        "- AgeBucket",
        "- BranchProbability",
        "- BranchEntropy",
        "No Context",
        "No Arbitration",
        "No Persistence",
        "No Phase",
        "No Optimization",
        "No Fitting",
        "No Machine Learning",
        "No Forward Returns used in motif construction",
    ]


def write_per_instrument(result: Result, out_root: Path) -> None:
    path = out_root / result.instrument / f"PathMotifs_{result.instrument}.txt"
    ensure_dir(path.parent)
    lines = [
        "APVA Path Motif Study v0.1",
        "=" * 88,
        f"Instrument: {result.instrument}",
        "Input path(s): " + ", ".join(str(path) for path in result.source_paths),
        f"Total rows: {len(result.bars)}",
        f"Motif count: {len(result.motifs)}",
    ]
    append_common_sections(lines, result)
    lines += ["", "17. Recommendation"] + recommendation(result)
    append_audit(lines)
    lines += [
        "",
        "19. Mechanical Research Notes",
        "- Recurring APVA motifs are audited as observed PreviousNode -> CurrentNode -> NextNode triplets.",
        "- Motif construction uses structural state and age only.",
        "- Outcome diagnostics are reported separately and are not used in motif scoring.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def agreement_rows(aggregate: Result, instruments: list[Result], names: list[str]) -> dict[tuple[str, str, str], AgreementRow]:
    by_name = {item.instrument: item for item in instruments}
    output = {}
    for key in sorted(aggregate.motifs):
        probs = {name: by_name[name].motifs[key].probability if key in by_name[name].motifs else 0.0 for name in names}
        values = list(probs.values())
        output[key] = AgreementRow(key[0], key[1], key[2], mean(values), statistics.pstdev(values) if len(values) > 1 else 0.0, probs)
    return output


def aggregate_outcome_lines(results: list[Result], instruments: list[str]) -> list[str]:
    by_name = {row.instrument: row for row in results}
    keys = sorted({key for result in results for key in result.outcomes})
    lines = ["", "Aggregate Outcome Table", "Motif | " + " | ".join(f"Count_{name} | MeanDR_{name}" for name in instruments) + " | ValidInstrumentCount"]
    for key in keys[:250]:
        cells = []
        valid = 0
        for name in instruments:
            row = by_name[name].outcomes.get(key)
            cells.append(f"{row.count if row else 0} | {fmt(row.mean_dr) if row else 'N/A'}")
            valid += int(row is not None)
        lines.append(f"{motif_text(key)} | " + " | ".join(cells) + f" | {valid}")
    return lines


def write_aggregate(result: Result, instrument_results: list[Result], out_root: Path) -> None:
    attach_replication(result, instrument_results)
    instruments = instrument_columns(instrument_results)
    agreement = agreement_rows(result, instrument_results, instruments)
    path = out_root / "PathMotifs" / "PathMotifs_All.txt"
    ensure_dir(path.parent)
    lines = ["APVA Path Motif Study v0.1 - Aggregate", "=" * 108, "Instruments: " + ", ".join(instruments)]
    lines += ["", "Aggregate Motif Table", "PreviousNode | CurrentNode | NextNode | Count | Probability | ReplicationCount"]
    lines += [f"{row.previous} | {row.current} | {row.next_node} | {row.count} | {pct(row.probability)} | {row.replication_count}"
              for row in sorted(result.motifs.values(), key=lambda row: (-row.count, row.current))[:250]]
    lines += ["", "Aggregate Strong Motif Table", "Motif | Count | ProbabilityDelta | EntropyReduction | ReplicationCount"]
    lines += [f"{motif_text((row.previous, row.current, row.next_node))} | {row.count} | {pct(row.delta)} | {fmt(row.entropy_reduction)} | {row.replication_count}"
              for row in sorted((row for row in result.motifs.values() if row.motif_class == "StrongMotif"), key=lambda row: (-abs(row.delta), -row.count))]
    lines += ["", "Aggregate Junction Motif Table", "PreviousNode | JunctionNode | NextNode | Count | Probability | ProbabilityDelta"]
    lines += [f"{row.previous} | {row.current} | {row.next_node} | {row.count} | {pct(row.probability)} | {pct(row.delta)}"
              for row in sorted((row for row in result.motifs.values() if row.current in JUNCTION_NODES), key=lambda row: (-abs(row.delta), -row.count))[:150]]
    lines += ["", "Aggregate Pipeline Motif Table", "PreviousNode | PipelineNode | NextNode | Count | Probability"]
    lines += [f"{row.previous} | {row.current} | {row.next_node} | {row.count} | {pct(row.probability)}"
              for row in sorted((row for row in result.motifs.values() if row.current in PIPELINE_NODES), key=lambda row: (-row.probability, -row.count))[:150]]
    lines += ["", "Aggregate Motif Contribution Table", "Motif | ContributionScore | ReplicationCount"]
    lines += [f"{motif_text((row.previous, row.current, row.next_node))} | {fmt(row.contribution)} | {row.replication_count}"
              for row in sorted(result.motifs.values(), key=lambda row: (-row.contribution, -row.count))[:150]]
    lines += ["", "Aggregate Grammar Hub Table", "CurrentNode | IncomingNodes | OutgoingNodes | MotifBranchFactor"]
    lines += [f"{row.current} | {row.incoming_count} | {row.outgoing_count} | {row.branch_factor}"
              for row in sorted(result.grammar.values(), key=lambda row: (-row.branch_factor, row.current))]
    lines += ["", "Aggregate Replication Table", "Motif | " + " | ".join(f"Prob_{name}" for name in instruments) + " | MeanProbability | StdDeviationProbability"]
    for key, row in sorted(agreement.items(), key=lambda item: (-result.motifs[item[0]].replication_count, item[0]))[:250]:
        lines.append(f"{motif_text(key)} | " + " | ".join(pct(row.probabilities[name]) for name in instruments) + f" | {pct(row.mean_probability)} | {pct(row.stdev_probability)}")
    lines += aggregate_outcome_lines(instrument_results, instruments)
    lines += ["", "Aggregate Recommendation"] + recommendation(result)
    lines += rankings(result)
    lines += [
        "",
        "RESEARCH NOTES",
        "- Do recurring APVA motifs exist? See frequency, replication, and classification tables.",
        "- Do motifs outperform node-only transitions? See NodeVsMotifAssessment.",
        "- Are junction behaviors explained by motifs? See Aggregate Junction Motif Table.",
        "- Structural grammar is evaluated mechanically without adding new states or variables.",
    ]
    append_audit(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rankings(result: Result) -> list[str]:
    lines = ["", "AGGREGATE RANKINGS"]
    lines += ["", "1. Most frequent motifs"] + [f"{motif_text((row.previous, row.current, row.next_node))} | {row.count}" for row in sorted(result.motifs.values(), key=lambda row: (-row.count, row.current))[:10]]
    lines += ["", "2. Strongest motifs"] + [f"{motif_text((row.previous, row.current, row.next_node))} | {pct(row.delta)}" for row in sorted((row for row in result.motifs.values() if row.motif_class == "StrongMotif"), key=lambda row: (-abs(row.delta), -row.count))[:10]]
    lines += ["", "3. Highest entropy reductions"] + [f"{row.previous} -> {row.current} | {fmt(row.entropy_reduction)}" for row in sorted(result.prefix_entropy.values(), key=lambda row: (-row.entropy_reduction, row.current))[:10]]
    lines += ["", "4. Highest contribution motifs"] + [f"{motif_text((row.previous, row.current, row.next_node))} | {fmt(row.contribution)}" for row in sorted(result.motifs.values(), key=lambda row: (-row.contribution, -row.count))[:10]]
    lines += ["", "5. Most replicated motifs"] + [f"{motif_text((row.previous, row.current, row.next_node))} | {row.replication_count}" for row in sorted(result.motifs.values(), key=lambda row: (-row.replication_count, -row.count))[:10]]
    lines += ["", "6. Strongest junction motifs"] + [f"{motif_text((row.previous, row.current, row.next_node))} | {pct(row.delta)}" for row in sorted((row for row in result.motifs.values() if row.current in JUNCTION_NODES), key=lambda row: (-abs(row.delta), -row.count))[:10]]
    lines += ["", "7. Strongest pipeline motifs"] + [f"{motif_text((row.previous, row.current, row.next_node))} | {pct(row.probability)}" for row in sorted((row for row in result.motifs.values() if row.current in PIPELINE_NODES), key=lambda row: (-row.probability, -row.count))[:10]]
    lines += ["", "8. Largest grammar hubs"] + [f"{row.current} | {row.branch_factor}" for row in sorted(result.grammar.values(), key=lambda row: (-row.branch_factor, row.current))[:10]]
    lines += ["", "9. Most predictable motifs"] + [f"{motif_text((row.previous, row.current, row.next_node))} | Top1={pct(row.top1)} | Brier={fmt(row.brier)}" for row in sorted(result.motifs.values(), key=lambda row: (-row.top1, row.brier, -row.count))[:10]]
    lines += ["", "10. Recommended APVA structural representation"] + recommendation(result)
    return lines


def validate(result: Result) -> None:
    if not result.rows:
        raise RuntimeError(f"{result.instrument}: no motif rows.")
    if not result.motifs:
        raise RuntimeError(f"{result.instrument}: no motifs.")
    if any(not 0 <= row.probability <= 1 for row in result.motifs.values()):
        raise RuntimeError(f"{result.instrument}: invalid motif probability.")
    if any(row.count <= 0 for row in result.motifs.values()):
        raise RuntimeError(f"{result.instrument}: invalid motif count.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Evidence CSV files or directories")
    parser.add_argument("--out-root", default="Evidence/Output", help="Output root directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_results(args.inputs)
    if not loaded:
        raise SystemExit("No APVA evidence CSV files found.")
    decay = [decay_study(row) for row in loaded]
    aggregate_nodes = aggregate_rows(decay)
    score_rows(aggregate_nodes)
    memory_thresholds = thresholds(loaded, aggregate_nodes)
    confidence_values = [aggregate_nodes[node_for(bar)].confidence for row in loaded for bar in row.bars]
    entropy_values = [aggregate_nodes[node_for(bar)].entropy_growth for row in loaded for bar in row.bars]
    confidence_thresholds = statistics.quantiles(confidence_values, n=3)[0], statistics.quantiles(confidence_values, n=3)[1]
    entropy_thresholds = statistics.quantiles(entropy_values, n=3)[0], statistics.quantiles(entropy_values, n=3)[1]

    aggregate_stream, aggregate_segments, aggregate_bars, aggregate_paths = [], [], [], []
    instrument_results = []
    for loaded_row, decay_row in zip(loaded, decay):
        offset = len(aggregate_bars)
        aggregate_segment = build_stream(loaded_row, aggregate_nodes, memory_thresholds, confidence_thresholds, entropy_thresholds, offset)
        aggregate_stream.extend(aggregate_segment)
        aggregate_segments.append(aggregate_segment)
        aggregate_bars.extend(loaded_row.bars)
        aggregate_paths.extend(loaded_row.source_paths)

        local = local_rows(decay_row)
        score_rows(local)
        local_segment = build_stream(loaded_row, local, memory_thresholds, confidence_thresholds, entropy_thresholds)
        result = build_result(loaded_row.instrument, loaded_row.source_paths, loaded_row.bars, local_segment, [local_segment])
        validate(result)
        instrument_results.append(result)

    aggregate_result = build_result("Aggregate", aggregate_paths, aggregate_bars, aggregate_stream, aggregate_segments)
    validate(aggregate_result)
    attach_replication(aggregate_result, instrument_results)
    out_root = Path(args.out_root)
    for result in instrument_results:
        write_per_instrument(result, out_root)
    write_aggregate(aggregate_result, instrument_results, out_root)
    print(f"Wrote {len(instrument_results)} per-instrument report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
