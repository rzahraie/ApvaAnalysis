#!/usr/bin/env python3
"""
APVA Interpretation Arbitration Study v0.1

Mechanically arbitrate among a deliberately small set of structural
interpretations. Scores use only past/current structural labels. Forward
returns are reported later as diagnostics and never enter scoring.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics as stats
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from APVA_StructuralLifeCycle_44 import (
    AGE_BUCKETS,
    STRUCTURAL_STATES,
    Bar,
    InstrumentResult,
    directional_return,
    ensure_dir,
    fmt,
    load_results,
    pct,
)
from APVA_TransitionContext_46 import Observation, build_observations, entropy, normalized_entropy

MIN_INSTRUMENT_RELIABILITY = 10
MIN_AGGREGATE_RELIABILITY = 50
MIN_VALID_INSTRUMENTS = 2
TOP_LIMIT = 25
TYPE_ORDER = {
    "ValidatedContext": 0,
    "PreviousDistinctContext": 1,
    "CurrentStateAge": 2,
}
VALIDATED_CONTEXTS = {
    ("RecoveryResolution", "NeutralProcessing", "1"): ("C", "Positive"),
    ("DecayToNeutral", "NeutralProcessing", "4"): ("D", "Positive"),
    ("RecoveryResolution", "CompressionProcessing", "3"): ("B", "Negative"),
}


@dataclass(frozen=True)
class Interpretation:
    interpretation_type: str
    key: str
    current_state: str
    age_bucket: str
    previous_distinct: str
    validated_flag: bool
    validated_direction: str


@dataclass(frozen=True)
class ReliabilityMetrics:
    count: int
    persistence: float
    exit_rate: float
    entropy: float
    normalized_entropy: float
    dominant_destination: str
    dominant_probability: float
    score: float


@dataclass(frozen=True)
class ScoredInterpretation:
    interpretation: Interpretation
    reliability: ReliabilityMetrics
    bonus: float
    final_score: float


@dataclass
class Outcome:
    values: List[float] = field(default_factory=list)

    def add(self, value: Optional[float]) -> None:
        if value is not None:
            self.values.append(value)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def mean_dr(self) -> float:
        return mean(self.values)

    @property
    def median_dr(self) -> float:
        return stats.median(self.values) if self.values else 0.0

    @property
    def continuation_rate(self) -> float:
        return sum(x > 0 for x in self.values) / len(self.values) if self.values else 0.0

    @property
    def failure_rate(self) -> float:
        return sum(x < 0 for x in self.values) / len(self.values) if self.values else 0.0

    @property
    def flat_rate(self) -> float:
        return sum(x == 0 for x in self.values) / len(self.values) if self.values else 0.0

    @property
    def skew(self) -> float:
        return self.continuation_rate - self.failure_rate


@dataclass
class ArbitrationRow:
    index: int
    interpretations: List[Interpretation]
    scored: List[ScoredInterpretation]
    winner_type: str
    winner_key: str
    winner_score: float
    runner_up_score: float
    margin: float
    dominance_ratio: Optional[float]
    ambiguity: str
    winner_run_age: int = 0
    replacement: bool = False
    previous_winner_run_age: int = 0
    fractal_jump_risk: bool = False


@dataclass
class StudyResult:
    instrument: str
    bars: List[Bar]
    source_paths: List[str]
    observations: List[Observation]
    interpretations: List[List[Interpretation]]
    arbitrations: List[ArbitrationRow]
    reliability: Dict[Tuple[str, str], ReliabilityMetrics]


def mean(values: Iterable[float]) -> float:
    xs = list(values)
    return sum(xs) / len(xs) if xs else 0.0


def baseline_key(observation: Observation) -> str:
    return f"{observation.state}+Age{observation.age_bucket}"


def distinct_key(observation: Observation) -> str:
    return f"{observation.previous_distinct}->{observation.state}+Age{observation.age_bucket}"


def validated_context(observation: Observation) -> Tuple[str, str]:
    return VALIDATED_CONTEXTS.get(
        (observation.previous_distinct, observation.state, observation.age_bucket),
        ("", "None"),
    )


def interpretations_for(observation: Observation) -> List[Interpretation]:
    context_name, context_direction = validated_context(observation)
    interpretations = [
        Interpretation(
            "CurrentStateAge",
            baseline_key(observation),
            observation.state,
            observation.age_bucket,
            observation.previous_distinct,
            bool(context_name),
            context_direction,
        )
    ]
    if observation.previous_distinct:
        interpretations.append(
            Interpretation(
                "PreviousDistinctContext",
                distinct_key(observation),
                observation.state,
                observation.age_bucket,
                observation.previous_distinct,
                bool(context_name),
                context_direction,
            )
        )
    if context_name:
        interpretations.append(
            Interpretation(
                "ValidatedContext",
                f"{context_name}:{distinct_key(observation)}",
                observation.state,
                observation.age_bucket,
                observation.previous_distinct,
                True,
                context_direction,
            )
        )
    return interpretations


def metrics_for(
    interpretation: Interpretation,
    destinations: Counter[str],
    enforce_minimum: bool = True,
) -> ReliabilityMetrics:
    count = sum(destinations.values())
    persistence = destinations[interpretation.current_state] / count if count else 0.0
    exit_rate = 1.0 - persistence if count else 0.0
    ent = entropy(destinations)
    norm_ent = normalized_entropy(destinations)
    dominant = destinations.most_common(1)
    dominant_destination = dominant[0][0] if dominant else "N/A"
    dominant_probability = dominant[0][1] / count if dominant and count else 0.0
    score = persistence + dominant_probability - norm_ent
    if enforce_minimum and count < MIN_INSTRUMENT_RELIABILITY:
        score = 0.0
    return ReliabilityMetrics(
        count,
        persistence,
        exit_rate,
        ent,
        norm_ent,
        dominant_destination,
        dominant_probability,
        score,
    )


def bonus_for(interpretation: Interpretation) -> float:
    if interpretation.interpretation_type != "ValidatedContext":
        return 0.0
    if interpretation.validated_direction == "Positive":
        return 0.25
    if interpretation.validated_direction == "Negative":
        return 0.15
    return 0.0


def ambiguity_class(winner: Optional[ScoredInterpretation], margin: float) -> str:
    if winner is None or margin < 0.05:
        return "Unresolved"
    if margin >= 0.50:
        return "Clear"
    if margin >= 0.20:
        return "Moderate"
    return "Ambiguous"


def build_arbitrations(
    bars: List[Bar],
    interpretations: List[List[Interpretation]],
) -> List[ArbitrationRow]:
    history: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    rows: List[ArbitrationRow] = []
    previous_winner = ""
    previous_type = ""
    run_age = 0
    for index, candidates in enumerate(interpretations):
        if index >= 1:
            for interpretation in interpretations[index - 1]:
                history[(interpretation.interpretation_type, interpretation.key)][bars[index].state] += 1
        scored: List[ScoredInterpretation] = []
        for interpretation in candidates:
            reliability = metrics_for(interpretation, history[(interpretation.interpretation_type, interpretation.key)])
            if reliability.count < MIN_INSTRUMENT_RELIABILITY:
                continue
            bonus = bonus_for(interpretation)
            scored.append(ScoredInterpretation(interpretation, reliability, bonus, reliability.score + bonus))
        scored.sort(key=lambda x: (-x.final_score, TYPE_ORDER[x.interpretation.interpretation_type], x.interpretation.key))
        winner = scored[0] if scored else None
        runner_up = scored[1] if len(scored) >= 2 else None
        winner_score = winner.final_score if winner else 0.0
        runner_up_score = runner_up.final_score if runner_up else 0.0
        margin = winner_score - runner_up_score if winner else 0.0
        dominance_ratio = winner_score / runner_up_score if runner_up_score > 0 else None
        ambiguity = ambiguity_class(winner, margin)
        winner_key = winner.interpretation.key if winner else "Unresolved"
        winner_type = winner.interpretation.interpretation_type if winner else "Unresolved"
        old_run_age = run_age
        replacement = bool(previous_winner and winner_key != previous_winner)
        if winner_key == previous_winner:
            run_age += 1
        else:
            run_age = 1
        fractal_jump = replacement and old_run_age <= 3 and winner_type != previous_type
        rows.append(
            ArbitrationRow(
                index,
                candidates,
                scored,
                winner_type,
                winner_key,
                winner_score,
                runner_up_score,
                margin,
                dominance_ratio,
                ambiguity,
                run_age,
                replacement,
                old_run_age,
                fractal_jump,
            )
        )
        previous_winner = winner_key
        previous_type = winner_type
    return rows


def build_full_reliability(
    bars: List[Bar],
    interpretations: List[List[Interpretation]],
) -> Dict[Tuple[str, str], ReliabilityMetrics]:
    destinations: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    examples: Dict[Tuple[str, str], Interpretation] = {}
    for index in range(len(bars) - 1):
        for interpretation in interpretations[index]:
            key = (interpretation.interpretation_type, interpretation.key)
            destinations[key][bars[index + 1].state] += 1
            examples[key] = interpretation
    return {
        key: metrics_for(examples[key], counts, enforce_minimum=False)
        for key, counts in destinations.items()
    }


def build_result(result: InstrumentResult) -> StudyResult:
    observations = build_observations(result.bars)
    interpretations = [interpretations_for(observation) for observation in observations]
    return StudyResult(
        result.instrument,
        result.bars,
        result.source_paths,
        observations,
        interpretations,
        build_arbitrations(result.bars, interpretations),
        build_full_reliability(result.bars, interpretations),
    )


def resolved(row: ArbitrationRow) -> bool:
    return row.winner_type != "Unresolved"


def outcome_for_rows(result: StudyResult, rows: Iterable[ArbitrationRow]) -> Outcome:
    outcome = Outcome()
    for row in rows:
        outcome.add(directional_return(result.bars, row.index, 5))
    return outcome


def winner_runs(rows: Sequence[ArbitrationRow]) -> List[Tuple[str, str, int]]:
    runs: List[Tuple[str, str, int]] = []
    if not rows:
        return runs
    current_type = rows[0].winner_type
    current_key = rows[0].winner_key
    length = 1
    for row in rows[1:]:
        if row.winner_key == current_key:
            length += 1
        else:
            runs.append((current_type, current_key, length))
            current_type = row.winner_type
            current_key = row.winner_key
            length = 1
    runs.append((current_type, current_key, length))
    return runs


def arbitration_summary(result: StudyResult) -> Dict[str, float]:
    rows = result.arbitrations
    resolved_rows = [row for row in rows if resolved(row)]
    dominance = [row.dominance_ratio for row in resolved_rows if row.dominance_ratio is not None]
    return {
        "total": len(rows),
        "resolved": len(resolved_rows),
        "unresolved_bars": len(rows) - len(resolved_rows),
        "clear": sum(row.ambiguity == "Clear" for row in rows),
        "moderate": sum(row.ambiguity == "Moderate" for row in rows),
        "ambiguous": sum(row.ambiguity == "Ambiguous" for row in rows),
        "unresolved": sum(row.ambiguity == "Unresolved" for row in rows),
        "winner_score": mean(row.winner_score for row in resolved_rows),
        "margin": mean(row.margin for row in resolved_rows),
        "dominance": mean(x for x in dominance if x is not None),
    }


def stability_summary(result: StudyResult) -> Dict[str, float]:
    runs = winner_runs(result.arbitrations)
    lengths = [length for _, _, length in runs]
    total = len(result.arbitrations)
    replacements = sum(row.replacement for row in result.arbitrations)
    risks = sum(row.fractal_jump_risk for row in result.arbitrations)
    return {
        "mean_run": mean(lengths),
        "median_run": stats.median(lengths) if lengths else 0.0,
        "max_run": max(lengths) if lengths else 0.0,
        "replacement_rate": replacements / total if total else 0.0,
        "risk_rate": risks / total if total else 0.0,
    }


def conflict_summary(result: StudyResult) -> Dict[str, float]:
    conflict_count = 0
    context_override = 0
    validated_override = 0
    clear_conflict = 0
    ambiguous_conflict = 0
    for row in result.arbitrations:
        score_by_type = {
            scored.interpretation.interpretation_type: scored.final_score
            for scored in row.scored
        }
        has_baseline = "CurrentStateAge" in score_by_type
        has_context = "PreviousDistinctContext" in score_by_type
        conflict = has_baseline and has_context and score_by_type["CurrentStateAge"] != score_by_type["PreviousDistinctContext"]
        if conflict:
            conflict_count += 1
            if row.winner_type in ("PreviousDistinctContext", "ValidatedContext"):
                context_override += 1
            if row.winner_type == "ValidatedContext":
                validated_override += 1
            if row.ambiguity == "Clear":
                clear_conflict += 1
            if row.ambiguity in ("Ambiguous", "Unresolved"):
                ambiguous_conflict += 1
    total = len(result.arbitrations)
    return {
        "count": conflict_count,
        "rate": conflict_count / total if total else 0.0,
        "context_override": context_override,
        "validated_override": validated_override,
        "clear": clear_conflict,
        "ambiguous": ambiguous_conflict,
    }


def validated_counts(result: StudyResult) -> Counter[str]:
    counts: Counter[str] = Counter()
    for observation in result.observations:
        context_name, _ = validated_context(observation)
        if context_name:
            counts[context_name] += 1
    return counts


def reliability_rows(result: StudyResult) -> List[Tuple[str, str, ReliabilityMetrics]]:
    return sorted(
        ((kind, key, metrics) for (kind, key), metrics in result.reliability.items()),
        key=lambda x: (x[0], x[1]),
    )


def append_reliability(lines: List[str], result: StudyResult) -> None:
    lines.append(
        "InterpretationType | InterpretationKey | Count | PersistenceRate_t1 | ExitRate_t1 | Entropy_t1 | "
        "NormalizedEntropy_t1 | DominantDestination | DominantProbability_t1 | ReliabilityScore"
    )
    for kind, key, metrics in reliability_rows(result):
        lines.append(
            f"{kind} | {key} | {metrics.count} | {pct(metrics.persistence)} | {pct(metrics.exit_rate)} | "
            f"{fmt(metrics.entropy)} | {fmt(metrics.normalized_entropy)} | {metrics.dominant_destination} | "
            f"{pct(metrics.dominant_probability)} | {fmt(metrics.score if metrics.count >= MIN_INSTRUMENT_RELIABILITY else 0.0)}"
        )


def append_examples(lines: List[str], result: StudyResult) -> None:
    lines.append(
        "BarIndex | WinnerType | WinnerKey | WinnerScore | RunnerUpScore | ScoreMargin | DominanceRatio | AmbiguityClass"
    )
    examples = [row for row in result.arbitrations if resolved(row)][:100]
    for row in examples:
        lines.append(
            f"{row.index} | {row.winner_type} | {row.winner_key} | {fmt(row.winner_score)} | {fmt(row.runner_up_score)} | "
            f"{fmt(row.margin)} | {fmt(row.dominance_ratio)} | {row.ambiguity}"
        )


def append_arbitration_summary(lines: List[str], result: StudyResult) -> None:
    summary = arbitration_summary(result)
    lines.extend(
        [
            f"TotalBars: {int(summary['total'])}",
            f"ResolvedBars: {int(summary['resolved'])}",
            f"UnresolvedBars: {int(summary['unresolved_bars'])}",
            f"ClearCount: {int(summary['clear'])}",
            f"ModerateCount: {int(summary['moderate'])}",
            f"AmbiguousCount: {int(summary['ambiguous'])}",
            f"UnresolvedCount: {int(summary['unresolved'])}",
            f"MeanWinnerScore: {fmt(summary['winner_score'])}",
            f"MeanScoreMargin: {fmt(summary['margin'])}",
            f"MeanDominanceRatio: {fmt(summary['dominance'])}",
        ]
    )


def append_stability(lines: List[str], result: StudyResult) -> None:
    summary = stability_summary(result)
    lines.extend(
        [
            f"WinnerRunCount: {len(winner_runs(result.arbitrations))}",
            f"MeanWinnerRunLength: {fmt(summary['mean_run'])}",
            f"MedianWinnerRunLength: {fmt(summary['median_run'])}",
            f"MaxWinnerRunLength: {int(summary['max_run'])}",
            f"ReplacementRate: {pct(summary['replacement_rate'])}",
        ]
    )


def append_conflicts(lines: List[str], result: StudyResult) -> None:
    summary = conflict_summary(result)
    lines.extend(
        [
            f"ConflictCount: {int(summary['count'])}",
            f"ConflictRate: {pct(summary['rate'])}",
            f"ContextOverrideCount: {int(summary['context_override'])}",
            f"ValidatedOverrideCount: {int(summary['validated_override'])}",
            f"ClearConflictCount: {int(summary['clear'])}",
            f"AmbiguousConflictCount: {int(summary['ambiguous'])}",
        ]
    )


def outcome_groups(result: StudyResult) -> Dict[Tuple[str, str, str], Outcome]:
    groups: Dict[Tuple[str, str, str], Outcome] = defaultdict(Outcome)
    for row in result.arbitrations:
        dr = directional_return(result.bars, row.index, 5)
        groups[(row.winner_type, row.winner_key, row.ambiguity)].add(dr)
    return groups


def append_outcomes(lines: List[str], result: StudyResult) -> None:
    lines.append(
        "WinnerType | WinnerKey | AmbiguityClass | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | "
        "FailureRate5 | FlatRate5 | OutcomeSkew"
    )
    for (winner_type, winner_key, ambiguity), outcome in sorted(outcome_groups(result).items()):
        lines.append(
            f"{winner_type} | {winner_key} | {ambiguity} | {outcome.count} | {fmt(outcome.mean_dr)} | "
            f"{fmt(outcome.median_dr)} | {pct(outcome.continuation_rate)} | {pct(outcome.failure_rate)} | "
            f"{pct(outcome.flat_rate)} | {pct(outcome.skew)}"
        )


def append_fractal_proxy(lines: List[str], result: StudyResult) -> None:
    risks = [row for row in result.arbitrations if row.fractal_jump_risk]
    outcome = outcome_for_rows(result, risks)
    total = len(result.arbitrations)
    lines.extend(
        [
            f"FractalJumpRiskCount: {len(risks)}",
            f"FractalJumpRiskRate: {pct(len(risks) / total if total else 0.0)}",
            f"OutcomeSkewAfterRisk: {pct(outcome.skew)}",
            f"MeanDRFwd5AfterRisk: {fmt(outcome.mean_dr)}",
        ]
    )


def append_audit(lines: List[str]) -> None:
    lines.extend(
        [
            "Allowed variables used:",
            "- CurrentState",
            "- AgeBucket",
            "- PreviousDistinctState",
            "- ValidatedContextFlag",
            "- PersistenceRate_t1",
            "- DominantDestinationProbability_t1",
            "- Entropy_t1",
            "",
            "Forbidden variables not used:",
            "- Phase",
            "- New contexts",
            "- Optimized weights",
            "- Forward returns in scoring",
            "- Price outcome in scoring",
            "- Machine learning",
        ]
    )


def write_instrument_report(result: StudyResult, out_root: str) -> str:
    out_dir = os.path.join(out_root, result.instrument)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"InterpretationArbitration_{result.instrument}.txt")
    states = Counter(observation.state for observation in result.observations)
    ages = Counter(observation.age_bucket for observation in result.observations)
    contexts = validated_counts(result)
    candidate_counts = Counter(
        interpretation.interpretation_type
        for interpretations in result.interpretations
        for interpretation in interpretations
    )
    lines = [
        "APVA Interpretation Arbitration Study v0.1",
        "Research only. No optimization, fitting, machine learning, trading rules, or future leakage.",
        "",
        "Diagnostics",
        f"Instrument: {result.instrument}",
        f"Input path(s): {'; '.join(result.source_paths)}",
        f"Total rows: {len(result.bars)}",
        "State counts: " + ", ".join(f"{state}={states[state]}" for state in STRUCTURAL_STATES),
        "Age counts: " + ", ".join(f"{age}={ages[age]}" for age in AGE_BUCKETS),
        "Validated context counts: " + ", ".join(f"{name}={contexts[name]}" for name in ("B", "C", "D")),
        "Candidate interpretation counts: " + ", ".join(f"{kind}={candidate_counts[kind]}" for kind in TYPE_ORDER),
        "",
        "1. Reliability tables",
    ]
    append_reliability(lines, result)
    lines += ["", "2. Interpretation scoring examples"]
    append_examples(lines, result)
    lines += ["", "3. Arbitration summary"]
    append_arbitration_summary(lines, result)
    lines += ["", "4. Winner stability"]
    append_stability(lines, result)
    lines += ["", "5. Conflict analysis"]
    append_conflicts(lines, result)
    lines += ["", "6. Outcome diagnostics"]
    append_outcomes(lines, result)
    lines += ["", "7. Fractal-jumping proxy"]
    append_fractal_proxy(lines, result)
    lines += ["", "8. Low-DoF audit"]
    append_audit(lines)
    lines += [
        "",
        "9. Mechanical research notes",
        "- Arbitration scores are computed online from structural transitions already observed before each scored bar.",
        "- ReliabilityScore is zero until its interpretation key has at least 10 prior structural transitions.",
        "- A validated negative context receives an informational bonus, not a directional score.",
        "- Forward price returns are attached only after arbitration as outcome diagnostics.",
        "- Rapid winner-type replacement after short runs is reported as a mechanical fractal-jump proxy.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def instrument_columns(instruments: Sequence[str], getter) -> str:
    columns: List[str] = []
    for instrument in instruments:
        columns.extend(getter(instrument))
    return " | ".join(columns)


def aggregate_reliability(
    results: Sequence[StudyResult],
) -> Dict[Tuple[str, str], ReliabilityMetrics]:
    combined: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    examples: Dict[Tuple[str, str], Interpretation] = {}
    for result in results:
        for index in range(len(result.bars) - 1):
            for interpretation in result.interpretations[index]:
                key = (interpretation.interpretation_type, interpretation.key)
                combined[key][result.bars[index + 1].state] += 1
                examples[key] = interpretation
    return {key: metrics_for(examples[key], counts, enforce_minimum=False) for key, counts in combined.items()}


def append_aggregate_reliability(lines: List[str], results: Sequence[StudyResult]) -> List[Tuple[str, str, ReliabilityMetrics, int]]:
    instruments = [result.instrument for result in results]
    by_inst = {result.instrument: result for result in results}
    combined = aggregate_reliability(results)
    lines.append(
        "InterpretationType | InterpretationKey | "
        + instrument_columns(instruments, lambda inst: [f"Count_{inst}", f"Persist_{inst}", f"Entropy_{inst}", f"DominantProb_{inst}", f"Score_{inst}"])
        + " | ValidInstrumentCount | MeanPersistence | MeanEntropy | MeanDominantProbability | MeanScore"
    )
    ranking_rows = []
    for (kind, key), aggregate_metrics in sorted(combined.items()):
        values = {inst: by_inst[inst].reliability.get((kind, key)) for inst in instruments}
        valid = [value for value in values.values() if value and value.count >= MIN_INSTRUMENT_RELIABILITY]
        if aggregate_metrics.count < MIN_AGGREGATE_RELIABILITY:
            continue
        cols = instrument_columns(
            instruments,
            lambda inst: [
                str(values[inst].count) if values[inst] else "0",
                pct(values[inst].persistence) if values[inst] else "N/A",
                fmt(values[inst].entropy) if values[inst] else "N/A",
                pct(values[inst].dominant_probability) if values[inst] else "N/A",
                fmt(values[inst].score if values[inst].count >= MIN_INSTRUMENT_RELIABILITY else 0.0) if values[inst] else "N/A",
            ],
        )
        mean_score = mean(value.score for value in valid)
        ranking_rows.append((kind, key, aggregate_metrics, len(valid)))
        lines.append(
            f"{kind} | {key} | {cols} | {len(valid)} | {pct(mean(value.persistence for value in valid)) if valid else 'N/A'} | "
            f"{fmt(mean(value.entropy for value in valid)) if valid else 'N/A'} | "
            f"{pct(mean(value.dominant_probability for value in valid)) if valid else 'N/A'} | {fmt(mean_score) if valid else 'N/A'}"
        )
    return ranking_rows


def grouped_outcomes_by_type_ambiguity(result: StudyResult) -> Dict[Tuple[str, str], Outcome]:
    groups: Dict[Tuple[str, str], Outcome] = defaultdict(Outcome)
    for row in result.arbitrations:
        groups[(row.winner_type, row.ambiguity)].add(directional_return(result.bars, row.index, 5))
    return groups


def write_aggregate_report(results: Sequence[StudyResult], out_root: str) -> str:
    out_dir = os.path.join(out_root, "InterpretationArbitration")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "InterpretationArbitration_All.txt")
    instruments = [result.instrument for result in results]
    lines = [
        "APVA Interpretation Arbitration Study v0.1 - Aggregate",
        "Research only. Fixed low-degree-of-freedom arbitration with past-only structural reliability scoring.",
        f"Instruments: {', '.join(instruments)}",
        "",
        "Aggregate Reliability Table",
    ]
    reliability_rankings = append_aggregate_reliability(lines, results)
    lines += [
        "",
        "Aggregate Arbitration Table",
        "Instrument | TotalBars | ResolvedBars | UnresolvedBars | ClearCount | ModerateCount | AmbiguousCount | "
        "UnresolvedCount | MeanWinnerScore | MeanScoreMargin | MeanDominanceRatio",
    ]
    for result in results:
        summary = arbitration_summary(result)
        lines.append(
            f"{result.instrument} | {int(summary['total'])} | {int(summary['resolved'])} | {int(summary['unresolved_bars'])} | "
            f"{int(summary['clear'])} | {int(summary['moderate'])} | {int(summary['ambiguous'])} | {int(summary['unresolved'])} | "
            f"{fmt(summary['winner_score'])} | {fmt(summary['margin'])} | {fmt(summary['dominance'])}"
        )
    lines += [
        "",
        "Aggregate Winner Stability Table",
        "Instrument | MeanWinnerRunLength | MedianWinnerRunLength | MaxWinnerRunLength | ReplacementRate | FractalJumpRiskRate",
    ]
    for result in results:
        summary = stability_summary(result)
        lines.append(
            f"{result.instrument} | {fmt(summary['mean_run'])} | {fmt(summary['median_run'])} | {int(summary['max_run'])} | "
            f"{pct(summary['replacement_rate'])} | {pct(summary['risk_rate'])}"
        )
    lines += [
        "",
        "Aggregate Conflict Table",
        "Instrument | ConflictCount | ConflictRate | ContextOverrideCount | ValidatedOverrideCount | ClearConflictCount | AmbiguousConflictCount",
    ]
    for result in results:
        summary = conflict_summary(result)
        lines.append(
            f"{result.instrument} | {int(summary['count'])} | {pct(summary['rate'])} | {int(summary['context_override'])} | "
            f"{int(summary['validated_override'])} | {int(summary['clear'])} | {int(summary['ambiguous'])}"
        )
    lines += [
        "",
        "Aggregate Outcome Diagnostics Table",
        "WinnerType | AmbiguityClass | "
        + instrument_columns(instruments, lambda inst: [f"Count_{inst}", f"Skew_{inst}", f"MeanDR_{inst}"])
        + " | ValidInstrumentCount | MeanSkew | MeanDR",
    ]
    outcome_by_inst = {result.instrument: grouped_outcomes_by_type_ambiguity(result) for result in results}
    outcome_keys = sorted({key for groups in outcome_by_inst.values() for key in groups})
    aggregate_outcome_rows = []
    for key in outcome_keys:
        values = {inst: outcome_by_inst[inst].get(key, Outcome()) for inst in instruments}
        valid = [value for value in values.values() if value.count >= MIN_INSTRUMENT_RELIABILITY]
        cols = instrument_columns(
            instruments,
            lambda inst: [str(values[inst].count), pct(values[inst].skew), fmt(values[inst].mean_dr)],
        )
        aggregate_outcome_rows.append((key, len(valid), mean(value.skew for value in valid), mean(value.mean_dr for value in valid)))
        lines.append(
            f"{key[0]} | {key[1]} | {cols} | {len(valid)} | {pct(mean(value.skew for value in valid)) if valid else 'N/A'} | "
            f"{fmt(mean(value.mean_dr for value in valid)) if valid else 'N/A'}"
        )
    lines += [
        "",
        "Aggregate Fractal-Jump Proxy Table",
        "Instrument | RiskCount | RiskRate | OutcomeSkewAfterRisk | MeanDRAfterRisk",
    ]
    for result in results:
        risks = [row for row in result.arbitrations if row.fractal_jump_risk]
        outcome = outcome_for_rows(result, risks)
        lines.append(
            f"{result.instrument} | {len(risks)} | {pct(len(risks) / len(result.arbitrations) if result.arbitrations else 0.0)} | "
            f"{pct(outcome.skew)} | {fmt(outcome.mean_dr)}"
        )

    winner_counts = Counter((row.winner_type, row.winner_key) for result in results for row in result.arbitrations)
    winner_runs_all = Counter()
    for result in results:
        for winner_type, winner_key, length in winner_runs(result.arbitrations):
            winner_runs_all[(winner_type, winner_key)] += length
    validated_winners = Counter(
        row.winner_key for result in results for row in result.arbitrations if row.winner_type == "ValidatedContext"
    )
    context_winners = Counter(
        row.winner_key for result in results for row in result.arbitrations if row.winner_type in ("PreviousDistinctContext", "ValidatedContext")
    )

    def ranking(title: str, rows: Iterable[str]) -> None:
        lines.extend(["", title])
        materialized = list(rows)[:TOP_LIMIT]
        lines.extend(materialized or ["No rows met the validation threshold."])

    valid_reliability = [row for row in reliability_rankings if row[3] >= MIN_VALID_INSTRUMENTS]
    ranking("1. Most reliable interpretation keys", (f"{row[0]} | {row[1]} | Score={fmt(row[2].score)}" for row in sorted(valid_reliability, key=lambda row: -row[2].score)))
    ranking("2. Lowest entropy interpretation keys", (f"{row[0]} | {row[1]} | Entropy={fmt(row[2].entropy)}" for row in sorted(valid_reliability, key=lambda row: row[2].entropy)))
    ranking("3. Highest dominance-probability interpretation keys", (f"{row[0]} | {row[1]} | DominantProb={pct(row[2].dominant_probability)}" for row in sorted(valid_reliability, key=lambda row: -row[2].dominant_probability)))
    ranking("4. Most common winning interpretations", (f"{key[0]} | {key[1]} | Count={count}" for key, count in winner_counts.most_common(TOP_LIMIT)))
    ranking("5. Most stable winning interpretations", (f"{key[0]} | {key[1]} | TotalRunBars={count}" for key, count in winner_runs_all.most_common(TOP_LIMIT)))
    ranking("6. Highest conflict-rate interpretations", (f"{result.instrument} | ConflictRate={pct(conflict_summary(result)['rate'])}" for result in sorted(results, key=lambda result: -conflict_summary(result)["rate"])))
    ranking("7. Most common context overrides", (f"{key} | Count={count}" for key, count in context_winners.most_common(TOP_LIMIT)))
    ranking("8. Most common validated-context overrides", (f"{key} | Count={count}" for key, count in validated_winners.most_common(TOP_LIMIT)))
    clear = [row for row in aggregate_outcome_rows if row[0][1] == "Clear" and row[1] >= MIN_VALID_INSTRUMENTS]
    ambiguous = [row for row in aggregate_outcome_rows if row[0][1] in ("Ambiguous", "Unresolved") and row[1] >= MIN_VALID_INSTRUMENTS]
    ranking("9. Best outcome clear winners", (f"{row[0][0]} | {row[0][1]} | MeanSkew={pct(row[2])} | MeanDR={fmt(row[3])}" for row in sorted(clear, key=lambda row: -row[2])))
    ranking("10. Worst outcome ambiguous winners", (f"{row[0][0]} | {row[0][1]} | MeanSkew={pct(row[2])} | MeanDR={fmt(row[3])}" for row in sorted(ambiguous, key=lambda row: row[2])))
    ranking("11. Highest fractal-jump risk periods", (f"{result.instrument} | RiskRate={pct(stability_summary(result)['risk_rate'])}" for result in sorted(results, key=lambda result: -stability_summary(result)["risk_rate"])))
    lines += ["", "Low-DoF audit"]
    append_audit(lines)
    lines += [
        "",
        "Mechanical research notes",
        "- Interpretation reliability is structural-label reliability, not directional desirability.",
        "- Each bar is scored online using only transitions already observable before that bar.",
        "- The validated negative B context receives a smaller informational bonus.",
        "- Outcome diagnostics remain downstream and cannot influence arbitration.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="APVA Interpretation Arbitration Study v0.1")
    parser.add_argument("inputs", nargs="+", help="Evidence CSV files or directories")
    parser.add_argument("--out-root", default="Evidence/Output", help="Output root directory")
    args = parser.parse_args(argv)
    loaded = load_results(args.inputs)
    if not loaded:
        raise SystemExit("No input rows loaded.")
    results = [build_result(result) for result in loaded]
    for result in results:
        write_instrument_report(result, args.out_root)
    aggregate = write_aggregate_report(results, args.out_root)
    print(f"Wrote InterpretationArbitration reports under {args.out_root}")
    print(f"Aggregate: {aggregate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
