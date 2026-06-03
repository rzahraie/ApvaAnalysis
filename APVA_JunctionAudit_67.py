#!/usr/bin/env python3
"""APVA Junction Audit Study v0.1.

Audit second-order PreviousNode -> CurrentNode -> NextNode path effects for
fixed APVA StateAge branch forecasting. Forward outcomes are diagnostics only.
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
from APVA_MinimalEngine_52 import directional_return, instrument_columns, load_results
from APVA_NodeImportance_58 import aggregate_rows, local_rows, score_rows
from APVA_ProcessGraph_54 import node_for, node_text
from APVA_StructuralForecast_56 import Outcome, outcome
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct


@dataclass
class Candidate:
    current: str
    count: int
    dominant_next: str
    dominant_probability: float
    entropy: float
    concentration: float
    previous_gain: float
    replication_count: int


@dataclass
class BaseBranch:
    current: str
    next_node: str
    count: int
    probability: float
    rank: int


@dataclass
class ConditionalBranch:
    previous: str
    current: str
    next_node: str
    count: int
    probability: float
    rank: int
    base_probability: float
    delta: float
    changed: bool


@dataclass
class PathStrength:
    previous: str
    current: str
    dominant_next: str
    count: int
    conditional_probability: float
    base_dominant_next: str
    base_dominant_probability: float
    delta: float
    max_delta: float
    mean_delta: float
    changed: bool
    conditional_entropy: float
    entropy_reduction: float
    concentration_change: float
    strength: float
    path_class: str
    replication_count: int = 0


@dataclass
class JunctionType:
    current: str
    junction_type: str
    strong_count: int
    moderate_count: int
    weak_count: int
    replication_count: int


@dataclass
class SecondOrderValue:
    current: str
    current_accuracy: float
    previous_accuracy: float
    accuracy_gain: float
    brier_improvement: float
    entropy_reduction: float
    sparse_key_increase: float
    net_value: float


@dataclass
class Directionality:
    current: str
    previous: str
    redirect_state: str
    probability: float
    delta: float


@dataclass
class Concentration:
    current: str
    previous_count: int
    top_share: float
    top3_share: float


@dataclass
class Stability:
    current: str
    mean_entropy: float
    median_entropy: float
    max_entropy: float
    entropy_reduction: float


@dataclass
class Result:
    instrument: str
    source_paths: list
    bars: list
    rows: list
    branch_result: object
    candidates: dict[str, Candidate]
    base: dict[tuple[str, str], BaseBranch]
    conditional: dict[tuple[str, str, str], ConditionalBranch]
    strengths: dict[tuple[str, str], PathStrength]
    types: dict[str, JunctionType]
    second_order: dict[str, SecondOrderValue]
    directionality: dict[tuple[str, str], Directionality]
    concentration: dict[str, Concentration]
    stability: dict[str, Stability]
    outcomes: dict[tuple[str, str, str], Outcome]


def entropy(distribution: Iterable[float]) -> float:
    return -sum(value * math.log(value) for value in distribution if value > 0)


def concentration(distribution: Iterable[float]) -> float:
    values = sorted(distribution, reverse=True)
    if not values:
        return 0.0
    return values[0] - (values[1] if len(values) > 1 else 0.0)


def state_of(node_text_value: str) -> str:
    return node_text_value.rsplit("_Age", 1)[0]


def normalize(values: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if math.isclose(low, high):
        return {key: 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def classify_path(count: int, abs_delta: float, changed: bool) -> str:
    if count >= 50 and abs_delta >= 0.20 and changed:
        return "StrongJunctionPath"
    if count >= 25 and abs_delta >= 0.10:
        return "ModerateJunctionPath"
    return "WeakJunctionPath"


def base_profile(rows: list) -> dict[tuple[str, str], BaseBranch]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[node_text(row.node)][row.next_node] += 1
    output = {}
    for current, counter in grouped.items():
        total = sum(counter.values())
        ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        for rank, (next_node, count) in enumerate(ranked, start=1):
            output[(current, next_node)] = BaseBranch(current, next_node, count, count / total if total else 0.0, rank)
    return output


def conditional_profile(rows: list, base: dict[tuple[str, str], BaseBranch]) -> dict[tuple[str, str, str], ConditionalBranch]:
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[(row.previous, node_text(row.node))][row.next_node] += 1
    base_dominants = {
        current: branch.next_node
        for (current, _), branch in base.items()
        if branch.rank == 1
    }
    output = {}
    for (previous, current), counter in grouped.items():
        total = sum(counter.values())
        ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        conditional_dominant = ranked[0][0] if ranked else ""
        for rank, (next_node, count) in enumerate(ranked, start=1):
            probability = count / total if total else 0.0
            base_probability = base.get((current, next_node), BaseBranch(current, next_node, 0, 0.0, 0)).probability
            output[(previous, current, next_node)] = ConditionalBranch(
                previous, current, next_node, count, probability, rank, base_probability,
                probability - base_probability, conditional_dominant != base_dominants.get(current, ""),
            )
    return output


def candidate_inventory(branch_result, node_replication: dict[str, int]) -> dict[str, Candidate]:
    output = {}
    gain = branch_result.gains["Model2_CurrentPreviousNode"]
    for node, value in gain.items():
        replication = node_replication.get(node, 0)
        if value.gain < 0.05 or replication < 2:
            continue
        entropy_row = branch_result.entropy_rows[node]
        branches = [row for (current, _), row in branch_result.branch_rows.items() if current == node]
        dominant = max(branches, key=lambda row: (row.probability, row.count), default=None)
        output[node] = Candidate(
            node, sum(row.count for row in branches), dominant.next_node if dominant else "",
            dominant.probability if dominant else 0.0, entropy_row.entropy, entropy_row.concentration,
            value.gain, replication,
        )
    return output


def path_strengths(candidates: dict[str, Candidate], base: dict[tuple[str, str], BaseBranch], conditional: dict[tuple[str, str, str], ConditionalBranch]) -> dict[tuple[str, str], PathStrength]:
    by_prev_current: dict[tuple[str, str], list[ConditionalBranch]] = defaultdict(list)
    for row in conditional.values():
        if row.current in candidates:
            by_prev_current[(row.previous, row.current)].append(row)
    output = {}
    raw = {}
    for key, rows in by_prev_current.items():
        previous, current = key
        dominant = min((row for row in rows if row.rank == 1), key=lambda row: row.next_node)
        base_dominant = next(row for row in base.values() if row.current == current and row.rank == 1)
        cond_probs = [row.probability for row in rows]
        base_probs = [base.get((current, row.next_node), BaseBranch(current, row.next_node, 0, 0.0, 0)).probability for row in rows]
        max_delta = max(abs(row.delta) for row in rows)
        mean_delta = mean(abs(row.delta) for row in rows)
        cond_entropy = entropy(cond_probs)
        base_entropy = entropy(base_probs)
        conc_change = concentration(cond_probs) - concentration(base_probs)
        output[key] = PathStrength(
            previous, current, dominant.next_node, sum(row.count for row in rows),
            dominant.probability, base_dominant.next_node, base_dominant.probability,
            dominant.probability - base.get((current, dominant.next_node), BaseBranch(current, dominant.next_node, 0, 0.0, 0)).probability,
            max_delta, mean_delta, dominant.changed, cond_entropy, base_entropy - cond_entropy,
            conc_change, 0.0, classify_path(sum(row.count for row in rows), abs(dominant.delta), dominant.changed),
        )
        raw[key] = output[key]
    max_norm = normalize({key: row.max_delta for key, row in raw.items()})
    ent_norm = normalize({key: max(row.entropy_reduction, 0.0) for key, row in raw.items()})
    con_norm = normalize({key: max(row.concentration_change, 0.0) for key, row in raw.items()})
    for key, row in output.items():
        row.strength = mean((max_norm[key], ent_norm[key], con_norm[key], float(row.changed)))
    return output


def attach_path_replication(aggregate: dict[tuple[str, str], PathStrength], instrument_results: list[Result]) -> None:
    for key, row in aggregate.items():
        row.replication_count = sum(
            key in result.strengths and result.strengths[key].dominant_next == row.dominant_next
            for result in instrument_results
        )


def junction_types(candidates: dict[str, Candidate], strengths: dict[tuple[str, str], PathStrength]) -> dict[str, JunctionType]:
    output = {}
    for current, candidate in candidates.items():
        rows = [row for row in strengths.values() if row.current == current]
        strong = sum(row.path_class == "StrongJunctionPath" for row in rows)
        moderate = sum(row.path_class == "ModerateJunctionPath" for row in rows)
        weak = sum(row.path_class == "WeakJunctionPath" for row in rows)
        if strong >= 2 and candidate.replication_count >= 2:
            label = "TrueJunction"
        elif moderate >= 1 and candidate.replication_count >= 2:
            label = "WeakJunction"
        else:
            label = "FalseJunction"
        output[current] = JunctionType(current, label, strong, moderate, weak, candidate.replication_count)
    return output


def second_order_values(candidates: dict[str, Candidate], rows: list, strengths: dict[tuple[str, str], PathStrength]) -> dict[str, SecondOrderValue]:
    output = {}
    raw = {}
    universe = tuple(sorted({row.next_node for row in rows}))
    for current in candidates:
        selected = [row for row in rows if node_text(row.node) == current]
        if not selected:
            continue
        targets = [row.next_node for row in selected]
        current_model = model_result("CurrentOnly", [current for _ in selected], targets, universe)
        previous_model = model_result("PreviousCurrent", [(row.previous, current) for row in selected], targets, universe)
        accuracy_gain = previous_model.metrics.top1 - current_model.metrics.top1
        brier_improvement = current_model.metrics.brier - previous_model.metrics.brier
        entropy_reduction = current_model.metrics.entropy - previous_model.metrics.entropy
        sparse_increase = previous_model.sparse_rate - current_model.sparse_rate
        raw[current] = (current_model, previous_model, accuracy_gain, brier_improvement, entropy_reduction, sparse_increase)
    acc_norm = normalize({(node, ""): max(values[2], 0.0) for node, values in raw.items()})
    bri_norm = normalize({(node, ""): max(values[3], 0.0) for node, values in raw.items()})
    ent_norm = normalize({(node, ""): max(values[4], 0.0) for node, values in raw.items()})
    sp_norm = normalize({(node, ""): max(values[5], 0.0) for node, values in raw.items()})
    for current, values in raw.items():
        current_model, previous_model, accuracy_gain, brier_improvement, entropy_reduction, sparse_increase = values
        key = (current, "")
        net = mean((acc_norm[key], bri_norm[key], ent_norm[key])) - sp_norm[key]
        output[current] = SecondOrderValue(
            current, current_model.metrics.top1, previous_model.metrics.top1, accuracy_gain,
            brier_improvement, entropy_reduction, sparse_increase, net,
        )
    return output


def directionality(strengths: dict[tuple[str, str], PathStrength]) -> dict[tuple[str, str], Directionality]:
    return {
        key: Directionality(row.current, row.previous, state_of(row.dominant_next), row.conditional_probability, row.delta)
        for key, row in strengths.items()
    }


def concentration_rows(candidates: dict[str, Candidate], rows: list) -> dict[str, Concentration]:
    output = {}
    for current in candidates:
        counts = Counter(row.previous for row in rows if node_text(row.node) == current)
        total = sum(counts.values())
        ranked = sorted(counts.values(), reverse=True)
        output[current] = Concentration(current, len(counts), (ranked[0] / total if total else 0.0), (sum(ranked[:3]) / total if total else 0.0))
    return output


def stability_rows(candidates: dict[str, Candidate], strengths: dict[tuple[str, str], PathStrength]) -> dict[str, Stability]:
    output = {}
    for current, candidate in candidates.items():
        values = [row.conditional_entropy for row in strengths.values() if row.current == current]
        output[current] = Stability(
            current, mean(values), statistics.median(values) if values else 0.0,
            max(values, default=0.0), candidate.entropy - mean(values),
        )
    return output


def outcome_rows(bars: list, stream: list, strengths: dict[tuple[str, str], PathStrength]) -> dict[tuple[str, str, str], Outcome]:
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    dominant = {(row.previous, row.current, row.dominant_next) for row in strengths.values()}
    for row in stream:
        key = (row.previous, node_text(row.node), row.next_node)
        if key not in dominant:
            continue
        value = directional_return(bars, row.index, 5)
        if value is not None:
            values[key].append(value)
    return {key: outcome(samples) for key, samples in values.items()}


def build_result(instrument: str, source_paths: list, bars: list, branch_result, node_replication: dict[str, int]) -> Result:
    candidates = candidate_inventory(branch_result, node_replication)
    base = base_profile(branch_result.rows)
    conditional = conditional_profile(branch_result.rows, base)
    strengths = path_strengths(candidates, base, conditional)
    return Result(
        instrument, source_paths, bars, branch_result.rows, branch_result, candidates, base, conditional, strengths,
        junction_types(candidates, strengths), second_order_values(candidates, branch_result.rows, strengths),
        directionality(strengths), concentration_rows(candidates, branch_result.rows),
        stability_rows(candidates, strengths), outcome_rows(bars, branch_result.rows, strengths),
    )


def recommendation(result: Result, instrument_results: list[Result]) -> tuple[str, str]:
    true = [row for row in result.types.values() if row.junction_type == "TrueJunction"]
    weak = [row for row in result.types.values() if row.junction_type == "WeakJunction"]
    false = [row for row in result.types.values() if row.junction_type == "FalseJunction"]
    positive = [row for row in result.second_order.values() if row.net_value > 0]
    max_rep = max((row.replication_count for row in result.strengths.values()), default=0)
    if len(true) >= 2 and positive and max_rep >= 2:
        label = "SecondOrderSelective"
    elif true or weak:
        label = "JunctionEvidenceWeak"
    else:
        label = "RemainFirstOrder"
    reason = (
        f"TrueJunctions={len(true)}; WeakJunctions={len(weak)}; FalseJunctions={len(false)}; "
        f"PositiveNetSecondOrderNodes={len(positive)}; MaxPathReplication={max_rep}."
    )
    return label, reason


def candidate_line(row: Candidate) -> str:
    return f"{row.current} | {row.count} | {row.dominant_next} | {pct(row.dominant_probability)} | {fmt(row.entropy)} | {pct(row.concentration)} | {pct(row.previous_gain)} | {row.replication_count}"


def base_line(row: BaseBranch) -> str:
    return f"{row.current} | {row.next_node} | {row.count} | {pct(row.probability)} | {row.rank}"


def conditional_line(row: ConditionalBranch) -> str:
    return f"{row.previous} | {row.current} | {row.next_node} | {row.count} | {pct(row.probability)} | {pct(row.base_probability)} | {pct(row.delta)} | {row.changed}"


def strength_line(row: PathStrength) -> str:
    return f"{row.previous} | {row.current} | {pct(row.max_delta)} | {pct(row.mean_delta)} | {fmt(row.entropy_reduction)} | {pct(row.concentration_change)} | {fmt(row.strength)}"


def map_line(row: PathStrength) -> str:
    return f"{row.previous} | {row.current} | {row.dominant_next} | {pct(row.conditional_probability)} | {row.base_dominant_next} | {pct(row.base_dominant_probability)} | {pct(row.delta)} | {row.path_class} | {row.replication_count}"


def type_line(row: JunctionType) -> str:
    return f"{row.current} | {row.junction_type} | {row.strong_count} | {row.moderate_count} | {row.weak_count} | {row.replication_count}"


def second_order_line(row: SecondOrderValue) -> str:
    return f"{row.current} | {pct(row.current_accuracy)} | {pct(row.previous_accuracy)} | {pct(row.accuracy_gain)} | {fmt(row.brier_improvement)} | {fmt(row.entropy_reduction)} | {pct(row.sparse_key_increase)} | {fmt(row.net_value)}"


def write_per_instrument(result: Result, out_root: Path) -> None:
    path = out_root / result.instrument / f"JunctionAudit_{result.instrument}.txt"
    ensure_dir(path.parent)
    lines = [
        "APVA Junction Audit Study v0.1",
        "=" * 108,
        "Diagnostics",
        f"Instrument: {result.instrument}",
        "Input path(s): " + ", ".join(str(item) for item in result.source_paths),
        f"Total rows: {len(result.bars)}",
        f"Junction candidate count: {len(result.candidates)}",
    ]
    append_common_sections(lines, result)
    label, reason = recommendation(result, [result])
    lines += ["", "15. Recommendation", f"Classification: {label}", f"Reason: {reason}"]
    append_audit(lines)
    lines += ["", "17. Mechanical Research Notes", "- Junction paths are audited mechanically from branch distributions.", "- PreviousNode is evaluated only as second-order structure.", "- Forward outcomes are diagnostics only."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_common_sections(lines: list[str], result: Result) -> None:
    lines += ["", "1. Junction Candidate Inventory", "CurrentNode | Count | DominantNextNode | BaseDominantProbability | BranchEntropy | BranchConcentration | PreviousNodeGain | ReplicationCount"]
    lines += [candidate_line(row) for row in sorted(result.candidates.values(), key=lambda row: (-row.previous_gain, row.current))]
    lines += ["", "2. Base Branch Profile", "CurrentNode | NextNode | Count | Probability | Rank"]
    lines += [base_line(row) for row in sorted(result.base.values(), key=lambda row: (row.current, row.rank)) if row.current in result.candidates]
    lines += ["", "3. PreviousNode Conditional Branch Profile", "PreviousNode | CurrentNode | NextNode | Count | ConditionalProbability | BaseProbability | ProbabilityDelta | DominantBranchChanged"]
    lines += [conditional_line(row) for row in sorted(result.conditional.values(), key=lambda row: (row.current, row.previous, row.rank)) if row.current in result.candidates]
    lines += ["", "4. Branch Shift Measurement", "PreviousNode | CurrentNode | NextNode | Count | ConditionalProbability | BaseProbability | ProbabilityDelta | DominantBranchChanged"]
    lines += [conditional_line(row) for row in sorted(result.conditional.values(), key=lambda row: (-abs(row.delta), row.current, row.previous)) if row.current in result.candidates]
    lines += ["", "5. Junction Strength", "PreviousNode | CurrentNode | MaxAbsoluteDelta | MeanAbsoluteDelta | EntropyReduction | BranchConcentrationChange | JunctionStrength"]
    lines += [strength_line(row) for row in sorted(result.strengths.values(), key=lambda row: (-row.strength, row.current, row.previous))]
    lines += ["", "6. Meaningful Path Filter", "PreviousNode | CurrentNode | DominantNextNode | Count | ConditionalProbability | BaseDominantNextNode | BaseDominantProbability | ProbabilityDelta | PathClass | ReplicationCount"]
    lines += [map_line(row) for row in sorted(result.strengths.values(), key=lambda row: (row.path_class, -abs(row.delta), row.current))]
    lines += ["", "7. Junction Map", "PreviousNode | CurrentNode | DominantNextNode | ConditionalProbability | BaseDominantNextNode | BaseDominantProbability | ProbabilityDelta | PathClass | ReplicationCount"]
    lines += [map_line(row) for row in sorted(result.strengths.values(), key=lambda row: (row.current, row.previous))]
    lines += ["", "8. Junction Type Classification", "CurrentNode | JunctionType | StrongPathCount | ModeratePathCount | WeakPathCount | ReplicationCount"]
    lines += [type_line(row) for row in sorted(result.types.values(), key=lambda row: (row.junction_type, row.current))]
    lines += ["", "9. Second-Order Value", "CurrentNode | CurrentNodeModelAccuracy | PreviousCurrentModelAccuracy | AccuracyGain | BrierImprovement | EntropyReduction | SparseKeyIncrease | NetSecondOrderValue"]
    lines += [second_order_line(row) for row in sorted(result.second_order.values(), key=lambda row: (-row.net_value, row.current))]
    lines += ["", "10. Cross-Instrument Replication", "See aggregate report."]
    lines += ["", "11. Junction Directionality", "CurrentNode | PreviousNode | RedirectTargetState | RedirectProbability | RedirectDelta"]
    lines += [f"{row.current} | {row.previous} | {row.redirect_state} | {pct(row.probability)} | {pct(row.delta)}" for row in sorted(result.directionality.values(), key=lambda row: (row.current, row.previous))]
    lines += ["", "12. Junction Concentration", "CurrentNode | NumberOfPreviousNodes | TopPreviousNodeShare | Top3PreviousNodeShare"]
    lines += [f"{row.current} | {row.previous_count} | {pct(row.top_share)} | {pct(row.top3_share)}" for row in sorted(result.concentration.values(), key=lambda row: row.current)]
    lines += ["", "13. Junction Stability", "CurrentNode | ConditionalBranchEntropyMean | ConditionalBranchEntropyMedian | ConditionalBranchEntropyMax | EntropyReduction"]
    lines += [f"{row.current} | {fmt(row.mean_entropy)} | {fmt(row.median_entropy)} | {fmt(row.max_entropy)} | {fmt(row.entropy_reduction)}" for row in sorted(result.stability.values(), key=lambda row: row.current)]
    lines += ["", "14. Outcome Diagnostics", "PreviousNode | CurrentNode | DominantNextNode | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew"]
    for key, row in sorted(result.outcomes.items()):
        lines.append(f"{key[0]} | {key[1]} | {key[2]} | {row.count} | {fmt(row.mean_dr)} | {fmt(row.median_dr)} | {pct(row.continuation)} | {pct(row.failure)} | {pct(row.flat)} | {fmt(row.skew)}")


def append_audit(lines: list[str]) -> None:
    lines += [
        "",
        "16. Low-DoF Audit",
        "Variables used:",
        "PreviousNode",
        "CurrentNode",
        "NextNode",
        "StructuralState",
        "AgeBucket",
        "BranchProbability",
        "PreviousNodeGain",
        "BranchEntropy",
        "BranchConcentration",
        "",
        "No Context",
        "No Arbitration",
        "No Persistence",
        "No Phase",
        "No Optimization",
        "No Fitting",
        "No Machine Learning",
        "No Forward Returns used in junction scoring",
    ]


def aggregate_outcome_lines(results: list[Result], instruments: list[str]) -> list[str]:
    by_name = {row.instrument: row for row in results}
    keys = sorted({key for result in results for key in result.outcomes})
    lines = ["", "Aggregate Outcome Table", "PreviousNode | CurrentNode | DominantNextNode | " + " | ".join(f"Count_{name} | MeanDR_{name}" for name in instruments) + " | ValidInstrumentCount"]
    for key in keys:
        values = [by_name[name].outcomes.get(key) for name in instruments]
        cells = [value for row in values for value in ((str(row.count), fmt(row.mean_dr)) if row else ("0", "N/A"))]
        lines.append(f"{key[0]} | {key[1]} | {key[2]} | " + " | ".join(cells) + f" | {sum(row is not None for row in values)}")
    return lines


def write_aggregate(result: Result, instrument_results: list[Result], out_root: Path) -> None:
    attach_path_replication(result.strengths, instrument_results)
    result.types = junction_types(result.candidates, result.strengths)
    path = out_root / "JunctionAudit" / "JunctionAudit_All.txt"
    ensure_dir(path.parent)
    instruments = instrument_columns(instrument_results)
    by_name = {row.instrument: row for row in instrument_results}
    lines = ["APVA Junction Audit Study v0.1 - Aggregate", "=" * 108, "Instruments: " + ", ".join(instruments)]
    lines += ["", "Aggregate Junction Candidate Table", "CurrentNode | Count | BaseDominantNextNode | BaseDominantProbability | PreviousNodeGain | BranchEntropy | ReplicationCount"]
    lines += [f"{row.current} | {row.count} | {row.dominant_next} | {pct(row.dominant_probability)} | {pct(row.previous_gain)} | {fmt(row.entropy)} | {row.replication_count}" for row in sorted(result.candidates.values(), key=lambda row: (-row.previous_gain, row.current))]
    lines += ["", "Aggregate Base Branch Profile Table", "CurrentNode | NextNode | Count | Probability | Rank"]
    lines += [base_line(row) for row in sorted(result.base.values(), key=lambda row: (row.current, row.rank)) if row.current in result.candidates]
    lines += ["", "Aggregate Conditional Branch Table", "PreviousNode | CurrentNode | NextNode | Count | ConditionalProbability | BaseProbability | ProbabilityDelta | DominantBranchChanged"]
    lines += [conditional_line(row) for row in sorted(result.conditional.values(), key=lambda row: (row.current, row.previous, row.rank)) if row.current in result.candidates]
    lines += ["", "Aggregate Junction Strength Table", "PreviousNode | CurrentNode | MaxAbsoluteDelta | MeanAbsoluteDelta | EntropyReduction | BranchConcentrationChange | JunctionStrength"]
    lines += [strength_line(row) for row in sorted(result.strengths.values(), key=lambda row: (-row.strength, row.current, row.previous))]
    lines += ["", "Aggregate Meaningful Path Table", "PreviousNode | CurrentNode | DominantNextNode | Count | ConditionalProbability | BaseDominantNextNode | BaseDominantProbability | ProbabilityDelta | PathClass | ReplicationCount"]
    lines += [map_line(row) for row in sorted(result.strengths.values(), key=lambda row: (row.path_class, -abs(row.delta), row.current))]
    lines += ["", "Aggregate Junction Map", "PreviousNode | CurrentNode | DominantNextNode | ConditionalProbability | PathClass | ReplicationCount"]
    lines += [f"{row.previous} | {row.current} | {row.dominant_next} | {pct(row.conditional_probability)} | {row.path_class} | {row.replication_count}" for row in sorted(result.strengths.values(), key=lambda row: (row.current, row.previous))]
    lines += ["", "Aggregate Junction Type Table", "CurrentNode | JunctionType | StrongPathCount | ModeratePathCount | WeakPathCount | ReplicationCount"]
    lines += [type_line(row) for row in sorted(result.types.values(), key=lambda row: (row.junction_type, row.current))]
    lines += ["", "Aggregate Second-Order Value Table", "CurrentNode | CurrentNodeAccuracy | PreviousCurrentAccuracy | AccuracyGain | BrierImprovement | EntropyReduction | SparseKeyIncrease | NetSecondOrderValue"]
    lines += [second_order_line(row) for row in sorted(result.second_order.values(), key=lambda row: (-row.net_value, row.current))]
    lines += ["", "Aggregate Directionality Table", "CurrentNode | PreviousNode | RedirectTargetState | RedirectProbability | RedirectDelta"]
    lines += [f"{row.current} | {row.previous} | {row.redirect_state} | {pct(row.probability)} | {pct(row.delta)}" for row in sorted(result.directionality.values(), key=lambda row: (row.current, row.previous))]
    lines += ["", "Aggregate Concentration Table", "CurrentNode | NumberOfPreviousNodes | TopPreviousNodeShare | Top3PreviousNodeShare"]
    lines += [f"{row.current} | {row.previous_count} | {pct(row.top_share)} | {pct(row.top3_share)}" for row in sorted(result.concentration.values(), key=lambda row: row.current)]
    lines += ["", "Aggregate Stability Table", "CurrentNode | ConditionalBranchEntropyMean | ConditionalBranchEntropyMedian | ConditionalBranchEntropyMax | EntropyReduction"]
    lines += [f"{row.current} | {fmt(row.mean_entropy)} | {fmt(row.median_entropy)} | {fmt(row.max_entropy)} | {fmt(row.entropy_reduction)}" for row in sorted(result.stability.values(), key=lambda row: row.current)]
    lines += aggregate_outcome_lines(instrument_results, instruments)
    label, reason = recommendation(result, instrument_results)
    true = [row.current for row in result.types.values() if row.junction_type == "TrueJunction"]
    weak = [row.current for row in result.types.values() if row.junction_type == "WeakJunction"]
    false = [row.current for row in result.types.values() if row.junction_type == "FalseJunction"]
    lines += ["", "Aggregate Recommendation", f"Classification: {label}", f"TrueJunctions: {', '.join(true) if true else 'None'}", f"WeakJunctions: {', '.join(weak) if weak else 'None'}", f"FalseJunctions: {', '.join(false) if false else 'None'}", f"RecommendedSecondOrderNodes: {', '.join(true) if true else 'None'}", f"Reason: {reason}", f"ReplicationAssessment: {len(instruments)} instrument(s)"]
    lines += rankings(result)
    lines += ["", "RESEARCH NOTES", "- Junction candidates are fixed by PreviousNodeGain >= 5% and replication >= 2.", "- Path classes are fixed by count, probability shift, and dominant-branch-change rules.", "- This is a selective second-order structural audit, not a new state family."]
    append_audit(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rankings(result: Result) -> list[str]:
    lines = ["", "AGGREGATE RANKINGS"]
    lines += ["", "1. Strongest junction candidates"] + [f"{row.current} | {pct(row.previous_gain)}" for row in sorted(result.candidates.values(), key=lambda row: (-row.previous_gain, row.current))[:10]]
    lines += ["", "2. Strongest junction paths"] + [f"{row.previous} -> {row.current} -> {row.dominant_next} | {fmt(row.strength)}" for row in sorted(result.strengths.values(), key=lambda row: (-row.strength, row.current))[:10]]
    lines += ["", "3. Most replicated junction paths"] + [f"{row.previous} -> {row.current} -> {row.dominant_next} | {row.replication_count}" for row in sorted(result.strengths.values(), key=lambda row: (-row.replication_count, row.current))[:10]]
    lines += ["", "4. Largest branch shifts"] + [f"{row.previous} -> {row.current} -> {row.dominant_next} | {pct(row.delta)}" for row in sorted(result.strengths.values(), key=lambda row: (-abs(row.delta), row.current))[:10]]
    lines += ["", "5. Largest entropy reductions"] + [f"{row.previous} -> {row.current} | {fmt(row.entropy_reduction)}" for row in sorted(result.strengths.values(), key=lambda row: (-row.entropy_reduction, row.current))[:10]]
    lines += ["", "6. Highest second-order value"] + [f"{row.current} | {fmt(row.net_value)}" for row in sorted(result.second_order.values(), key=lambda row: (-row.net_value, row.current))[:10]]
    lines += ["", "7. Junctions controlled by few prior paths"] + [f"{row.current} | TopPrevious={pct(row.top_share)}" for row in sorted(result.concentration.values(), key=lambda row: (-row.top_share, row.current))[:10]]
    lines += ["", "8. Junctions controlled by many prior paths"] + [f"{row.current} | TopPrevious={pct(row.top_share)}" for row in sorted(result.concentration.values(), key=lambda row: (row.top_share, row.current))[:10]]
    lines += ["", "9. Best redirect-to-neutral paths"] + [f"{row.previous} -> {row.current} | {pct(row.probability)} | Delta={pct(row.delta)}" for row in sorted((row for row in result.directionality.values() if row.redirect_state == "NeutralProcessing"), key=lambda row: (-row.delta, row.current))[:10]]
    recommended = [row.current for row in result.types.values() if row.junction_type == "TrueJunction"]
    lines += ["", "10. Recommended selective second-order APVA nodes"] + (recommended if recommended else ["None"])
    return lines


def validate(result: Result) -> None:
    if not result.rows:
        raise RuntimeError(f"{result.instrument}: no rows.")
    for row in result.base.values():
        if not 0 <= row.probability <= 1:
            raise RuntimeError(f"{result.instrument}: invalid base probability.")
    for row in result.conditional.values():
        if not 0 <= row.probability <= 1:
            raise RuntimeError(f"{result.instrument}: invalid conditional probability.")
    if any(row.path_class not in {"StrongJunctionPath", "ModerateJunctionPath", "WeakJunctionPath"} for row in result.strengths.values()):
        raise RuntimeError(f"{result.instrument}: invalid path class.")


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
    instrument_branches = []
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
        instrument_branches.append(build_branch_result(loaded_row.instrument, loaded_row.source_paths, loaded_row.bars, local_segment, [local_segment]))

    node_replication = Counter()
    for branch in instrument_branches:
        for node in branch.entropy_rows:
            node_replication[node] += 1
    aggregate_branch = build_branch_result("Aggregate", aggregate_paths, aggregate_bars, aggregate_stream, aggregate_segments)
    aggregate_result = build_result("Aggregate", aggregate_paths, aggregate_bars, aggregate_branch, node_replication)
    validate(aggregate_result)
    instrument_results = []
    for branch, loaded_row in zip(instrument_branches, loaded):
        result = build_result(branch.instrument, branch.source_paths, loaded_row.bars, branch, node_replication)
        validate(result)
        instrument_results.append(result)
    attach_path_replication(aggregate_result.strengths, instrument_results)
    aggregate_result.types = junction_types(aggregate_result.candidates, aggregate_result.strengths)
    out_root = Path(args.out_root)
    for result in instrument_results:
        write_per_instrument(result, out_root)
    write_aggregate(aggregate_result, instrument_results, out_root)
    print(f"Wrote {len(instrument_results)} per-instrument report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
