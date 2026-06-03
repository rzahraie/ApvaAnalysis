#!/usr/bin/env python3
"""
APVA Context Disambiguation Study v0.1

Validate a frozen set of prior-context candidates discovered before this
study. No candidate discovery, optimization, fitting, parameter search, or
machine learning is performed. Research only.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics as stats
import zlib
from collections import Counter
from dataclasses import dataclass
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

HORIZONS = (1, 2, 3, 5)
BOOTSTRAP_RESAMPLES = 1000
RANDOMIZATION_REPEATS = 100
MIN_VALID_COUNT = 10
MIN_VALID_INSTRUMENTS = 2
TOP_LIMIT = 25


@dataclass(frozen=True)
class Candidate:
    name: str
    state: str
    age_bucket: str
    previous_distinct: Optional[str]
    description: str
    baseline: Optional[str] = None


CANDIDATES = (
    Candidate("A", "CompressionProcessing", "3", "NeutralProcessing", "NeutralProcessing -> CompressionProcessing Age 3", "E"),
    Candidate("B", "CompressionProcessing", "3", "RecoveryResolution", "RecoveryResolution -> CompressionProcessing Age 3", "E"),
    Candidate("C", "NeutralProcessing", "1", "RecoveryResolution", "RecoveryResolution -> NeutralProcessing Age 1", "F"),
    Candidate("D", "NeutralProcessing", "4", "DecayToNeutral", "DecayToNeutral -> NeutralProcessing Age 4", "G"),
    Candidate("E", "CompressionProcessing", "3", None, "Baseline CompressionProcessing Age 3"),
    Candidate("F", "NeutralProcessing", "1", None, "Baseline NeutralProcessing Age 1"),
    Candidate("G", "NeutralProcessing", "4", None, "Baseline NeutralProcessing Age 4"),
)
CANDIDATE_BY_NAME = {candidate.name: candidate for candidate in CANDIDATES}


@dataclass(frozen=True)
class Sample:
    instrument: str
    index: int
    dr: Optional[float]
    next_states: Tuple[Optional[str], ...]


@dataclass(frozen=True)
class Metrics:
    count: int
    mean_dr: float
    median_dr: float
    continuation_rate: float
    failure_rate: float
    flat_rate: float
    skew: float
    stddev_dr: float
    max_favorable: float
    max_adverse: float


@dataclass(frozen=True)
class TransitionMetrics:
    horizon: int
    count: int
    destinations: Counter[str]
    entropy: float
    normalized_entropy: float
    dominant_destination: str
    dominant_probability: float


@dataclass(frozen=True)
class BootstrapMetrics:
    median_mean_dr: float
    p05_mean_dr: float
    p95_mean_dr: float
    median_skew: float
    p05_skew: float
    p95_skew: float
    median_cont: float
    p05_cont: float
    p95_cont: float
    median_fail: float
    p05_fail: float
    p95_fail: float


@dataclass(frozen=True)
class RandomizedMetrics:
    observed_mean_dr: float
    randomized_mean_dr: float
    observed_skew: float
    randomized_skew: float
    observed_entropy: float
    randomized_entropy: float


@dataclass
class StudyResult:
    instrument: str
    bars: List[Bar]
    source_paths: List[str]
    observations: List[Observation]
    samples: Dict[str, List[Sample]]
    bootstrap: Dict[str, BootstrapMetrics]
    randomized: Dict[str, RandomizedMetrics]


def mean(values: Iterable[float]) -> float:
    xs = list(values)
    return sum(xs) / len(xs) if xs else 0.0


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    position = (len(xs) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(xs) - 1)
    fraction = position - lower
    return xs[lower] * (1.0 - fraction) + xs[upper] * fraction


def stable_seed(*parts: str) -> int:
    return zlib.crc32("|".join(parts).encode("utf-8"))


def matches(candidate: Candidate, observation: Observation, previous_distinct: Optional[str] = None) -> bool:
    if observation.state != candidate.state or observation.age_bucket != candidate.age_bucket:
        return False
    if candidate.previous_distinct is None:
        return True
    value = observation.previous_distinct if previous_distinct is None else previous_distinct
    return value == candidate.previous_distinct


def sample_for(bars: List[Bar], instrument: str, observation: Observation) -> Sample:
    next_states = tuple(
        bars[observation.index + horizon].state if observation.index + horizon < len(bars) else None
        for horizon in HORIZONS
    )
    return Sample(
        instrument,
        observation.index,
        directional_return(bars, observation.index, 5),
        next_states,
    )


def build_samples(
    bars: List[Bar], instrument: str, observations: List[Observation]
) -> Dict[str, List[Sample]]:
    result: Dict[str, List[Sample]] = {candidate.name: [] for candidate in CANDIDATES}
    for observation in observations:
        sample = sample_for(bars, instrument, observation)
        for candidate in CANDIDATES:
            if matches(candidate, observation):
                result[candidate.name].append(sample)
    return result


def outcome_metrics(samples: Sequence[Sample]) -> Metrics:
    values = [sample.dr for sample in samples if sample.dr is not None]
    if not values:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    count = len(values)
    continuation = sum(value > 0 for value in values) / count
    failure = sum(value < 0 for value in values) / count
    flats = sum(value == 0 for value in values) / count
    return Metrics(
        count,
        mean(values),
        stats.median(values),
        continuation,
        failure,
        flats,
        continuation - failure,
        stats.pstdev(values) if len(values) > 1 else 0.0,
        max(values),
        min(values),
    )


def transition_metrics(samples: Sequence[Sample], horizon: int) -> TransitionMetrics:
    offset = HORIZONS.index(horizon)
    destinations = Counter(
        sample.next_states[offset] for sample in samples if sample.next_states[offset] is not None
    )
    count = sum(destinations.values())
    dominant = destinations.most_common(1)
    return TransitionMetrics(
        horizon,
        count,
        destinations,
        entropy(destinations),
        normalized_entropy(destinations),
        dominant[0][0] if dominant else "N/A",
        dominant[0][1] / count if dominant and count else 0.0,
    )


def transition_lift(
    samples: Sequence[Sample], all_samples: Sequence[Sample], horizon: int, destination: str
) -> Optional[float]:
    candidate = transition_metrics(samples, horizon)
    unconditional = transition_metrics(all_samples, horizon)
    probability = candidate.destinations[destination] / candidate.count if candidate.count else 0.0
    baseline = unconditional.destinations[destination] / unconditional.count if unconditional.count else 0.0
    return probability / baseline if baseline else None


def bootstrap_metrics(samples: Sequence[Sample], seed: int) -> BootstrapMetrics:
    values = [sample.dr for sample in samples if sample.dr is not None]
    if not values:
        return BootstrapMetrics(*(0.0 for _ in range(12)))
    rng = random.Random(seed)
    estimates: Dict[str, List[float]] = {"mean": [], "skew": [], "cont": [], "fail": []}
    for _ in range(BOOTSTRAP_RESAMPLES):
        resample = [values[rng.randrange(len(values))] for _ in values]
        continuation = sum(value > 0 for value in resample) / len(resample)
        failure = sum(value < 0 for value in resample) / len(resample)
        estimates["mean"].append(mean(resample))
        estimates["skew"].append(continuation - failure)
        estimates["cont"].append(continuation)
        estimates["fail"].append(failure)
    return BootstrapMetrics(
        percentile(estimates["mean"], 0.50),
        percentile(estimates["mean"], 0.05),
        percentile(estimates["mean"], 0.95),
        percentile(estimates["skew"], 0.50),
        percentile(estimates["skew"], 0.05),
        percentile(estimates["skew"], 0.95),
        percentile(estimates["cont"], 0.50),
        percentile(estimates["cont"], 0.05),
        percentile(estimates["cont"], 0.95),
        percentile(estimates["fail"], 0.50),
        percentile(estimates["fail"], 0.05),
        percentile(estimates["fail"], 0.95),
    )


def randomized_metrics(
    bars: List[Bar],
    instrument: str,
    observations: List[Observation],
    candidate: Candidate,
) -> RandomizedMetrics:
    observed_samples = [
        sample_for(bars, instrument, observation)
        for observation in observations
        if matches(candidate, observation)
    ]
    observed = outcome_metrics(observed_samples)
    observed_entropy = transition_metrics(observed_samples, 1).entropy
    if candidate.previous_distinct is None:
        return RandomizedMetrics(
            observed.mean_dr,
            observed.mean_dr,
            observed.skew,
            observed.skew,
            observed_entropy,
            observed_entropy,
        )
    labels = [observation.previous_distinct for observation in observations]
    rng = random.Random(stable_seed("randomization", instrument, candidate.name))
    randomized_mean_dr: List[float] = []
    randomized_skew: List[float] = []
    randomized_entropy: List[float] = []
    for _ in range(RANDOMIZATION_REPEATS):
        shuffled = list(labels)
        rng.shuffle(shuffled)
        shuffled_samples = [
            sample_for(bars, instrument, observation)
            for observation, label in zip(observations, shuffled)
            if matches(candidate, observation, label)
        ]
        metric = outcome_metrics(shuffled_samples)
        randomized_mean_dr.append(metric.mean_dr)
        randomized_skew.append(metric.skew)
        randomized_entropy.append(transition_metrics(shuffled_samples, 1).entropy)
    return RandomizedMetrics(
        observed.mean_dr,
        mean(randomized_mean_dr),
        observed.skew,
        mean(randomized_skew),
        observed_entropy,
        mean(randomized_entropy),
    )


def build_result(result: InstrumentResult) -> StudyResult:
    observations = build_observations(result.bars)
    samples = build_samples(result.bars, result.instrument, observations)
    bootstrap = {
        candidate.name: bootstrap_metrics(samples[candidate.name], stable_seed("bootstrap", result.instrument, candidate.name))
        for candidate in CANDIDATES
    }
    randomized = {
        candidate.name: randomized_metrics(result.bars, result.instrument, observations, candidate)
        for candidate in CANDIDATES
    }
    return StudyResult(
        result.instrument,
        result.bars,
        result.source_paths,
        observations,
        samples,
        bootstrap,
        randomized,
    )


def combined_samples(results: Sequence[StudyResult], candidate_name: str) -> List[Sample]:
    return [sample for result in results for sample in result.samples[candidate_name]]


def combined_bootstrap(results: Sequence[StudyResult], candidate_name: str) -> BootstrapMetrics:
    return bootstrap_metrics(combined_samples(results, candidate_name), stable_seed("bootstrap", "aggregate", candidate_name))


def combined_randomized(results: Sequence[StudyResult], candidate_name: str) -> RandomizedMetrics:
    observed = outcome_metrics(combined_samples(results, candidate_name))
    observed_entropy = transition_metrics(combined_samples(results, candidate_name), 1).entropy
    candidate = CANDIDATE_BY_NAME[candidate_name]
    if candidate.previous_distinct is None:
        return RandomizedMetrics(observed.mean_dr, observed.mean_dr, observed.skew, observed.skew, observed_entropy, observed_entropy)
    rng = random.Random(stable_seed("randomization", "aggregate", candidate_name))
    randomized_mean_dr: List[float] = []
    randomized_skew: List[float] = []
    randomized_entropy: List[float] = []
    for _ in range(RANDOMIZATION_REPEATS):
        pooled: List[Sample] = []
        for result in results:
            labels = [observation.previous_distinct for observation in result.observations]
            rng.shuffle(labels)
            pooled.extend(
                sample_for(result.bars, result.instrument, observation)
                for observation, label in zip(result.observations, labels)
                if matches(candidate, observation, label)
            )
        metric = outcome_metrics(pooled)
        randomized_mean_dr.append(metric.mean_dr)
        randomized_skew.append(metric.skew)
        randomized_entropy.append(transition_metrics(pooled, 1).entropy)
    return RandomizedMetrics(
        observed.mean_dr,
        mean(randomized_mean_dr),
        observed.skew,
        mean(randomized_skew),
        observed_entropy,
        mean(randomized_entropy),
    )


def split_samples(samples: Sequence[Sample]) -> Tuple[List[Sample], List[Sample]]:
    midpoint = len(samples) // 2
    return list(samples[:midpoint]), list(samples[midpoint:])


def direction(metrics: Metrics) -> str:
    if metrics.skew > 0:
        return "Positive"
    if metrics.skew < 0:
        return "Negative"
    return "Neutral"


def candidate_context_counts(result: StudyResult) -> Counter[str]:
    return Counter(
        observation.contexts.get("ContextDistinct", "N/A")
        for observation in result.observations
        if observation.contexts.get("ContextDistinct")
    )


def append_candidate_summary(lines: List[str], result: StudyResult) -> None:
    lines.append("Candidate | Description | Count | DirectionAgreement")
    for candidate in CANDIDATES:
        metrics = outcome_metrics(result.samples[candidate.name])
        lines.append(f"{candidate.name} | {candidate.description} | {metrics.count} | {direction(metrics)}")


def append_outcomes(lines: List[str], result: StudyResult) -> None:
    lines.append(
        "Candidate | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | "
        "OutcomeSkew | StdDevDRFwd5 | MaxFavorableDRFwd5 | MaxAdverseDRFwd5"
    )
    for candidate in CANDIDATES:
        metrics = outcome_metrics(result.samples[candidate.name])
        lines.append(
            f"{candidate.name} | {metrics.count} | {fmt(metrics.mean_dr)} | {fmt(metrics.median_dr)} | "
            f"{pct(metrics.continuation_rate)} | {pct(metrics.failure_rate)} | {pct(metrics.flat_rate)} | "
            f"{pct(metrics.skew)} | {fmt(metrics.stddev_dr)} | {fmt(metrics.max_favorable)} | {fmt(metrics.max_adverse)}"
        )


def append_transitions(lines: List[str], result: StudyResult) -> None:
    all_samples = [sample_for(result.bars, result.instrument, observation) for observation in result.observations]
    lines.append(
        "Candidate | Horizon | NextState | Count | Probability | Lift | Entropy | NormalizedEntropy | "
        "DominantDestination | DominantProbability"
    )
    for candidate in CANDIDATES:
        samples = result.samples[candidate.name]
        for horizon in HORIZONS:
            metrics = transition_metrics(samples, horizon)
            for destination, count in metrics.destinations.most_common():
                lift = transition_lift(samples, all_samples, horizon, destination)
                lines.append(
                    f"{candidate.name} | t+{horizon} | {destination} | {count} | "
                    f"{pct(count / metrics.count) if metrics.count else 'N/A'} | {fmt(lift)} | "
                    f"{fmt(metrics.entropy)} | {fmt(metrics.normalized_entropy)} | "
                    f"{metrics.dominant_destination} | {pct(metrics.dominant_probability)}"
                )


def comparison_values(samples: Dict[str, List[Sample]], candidate_name: str) -> Tuple[Metrics, Metrics, TransitionMetrics, TransitionMetrics]:
    candidate = CANDIDATE_BY_NAME[candidate_name]
    if not candidate.baseline:
        raise ValueError(f"Candidate {candidate_name} has no baseline.")
    left = outcome_metrics(samples[candidate_name])
    right = outcome_metrics(samples[candidate.baseline])
    left_transition = transition_metrics(samples[candidate_name], 1)
    right_transition = transition_metrics(samples[candidate.baseline], 1)
    return left, right, left_transition, right_transition


def append_comparisons(lines: List[str], samples: Dict[str, List[Sample]]) -> None:
    lines.append(
        "Candidate | Baseline | DeltaContinuationRate | DeltaFailureRate | DeltaOutcomeSkew | "
        "DeltaMeanDR | DeltaEntropy | DeltaDominantDestinationProbability"
    )
    for candidate_name in ("A", "B", "C", "D"):
        candidate = CANDIDATE_BY_NAME[candidate_name]
        left, right, left_transition, right_transition = comparison_values(samples, candidate_name)
        lines.append(
            f"{candidate_name} | {candidate.baseline} | "
            f"{pct(left.continuation_rate - right.continuation_rate)} | "
            f"{pct(left.failure_rate - right.failure_rate)} | {pct(left.skew - right.skew)} | "
            f"{fmt(left.mean_dr - right.mean_dr)} | {fmt(left_transition.entropy - right_transition.entropy)} | "
            f"{pct(left_transition.dominant_probability - right_transition.dominant_probability)}"
        )


def append_time_stability(lines: List[str], result: StudyResult) -> None:
    lines.append(
        "Candidate | EarlyCount | EarlySkew | EarlyMeanDR | EarlyEntropy | EarlyDominant | "
        "LateCount | LateSkew | LateMeanDR | LateEntropy | LateDominant | DeltaSkew | DeltaMeanDR | DeltaEntropy"
    )
    for candidate in CANDIDATES:
        early, late = split_samples(result.samples[candidate.name])
        em = outcome_metrics(early)
        lm = outcome_metrics(late)
        et = transition_metrics(early, 1)
        lt = transition_metrics(late, 1)
        lines.append(
            f"{candidate.name} | {em.count} | {pct(em.skew)} | {fmt(em.mean_dr)} | {fmt(et.entropy)} | {et.dominant_destination} | "
            f"{lm.count} | {pct(lm.skew)} | {fmt(lm.mean_dr)} | {fmt(lt.entropy)} | {lt.dominant_destination} | "
            f"{pct(lm.skew - em.skew)} | {fmt(lm.mean_dr - em.mean_dr)} | {fmt(lt.entropy - et.entropy)}"
        )


def append_bootstrap(lines: List[str], bootstrap: Dict[str, BootstrapMetrics]) -> None:
    lines.append(
        "Candidate | MedianMeanDR | P05MeanDR | P95MeanDR | MedianSkew | P05Skew | P95Skew | "
        "MedianContinuation | P05Continuation | P95Continuation | MedianFailure | P05Failure | P95Failure"
    )
    for candidate in CANDIDATES:
        metric = bootstrap[candidate.name]
        lines.append(
            f"{candidate.name} | {fmt(metric.median_mean_dr)} | {fmt(metric.p05_mean_dr)} | {fmt(metric.p95_mean_dr)} | "
            f"{pct(metric.median_skew)} | {pct(metric.p05_skew)} | {pct(metric.p95_skew)} | "
            f"{pct(metric.median_cont)} | {pct(metric.p05_cont)} | {pct(metric.p95_cont)} | "
            f"{pct(metric.median_fail)} | {pct(metric.p05_fail)} | {pct(metric.p95_fail)}"
        )


def append_randomized(lines: List[str], randomized: Dict[str, RandomizedMetrics]) -> None:
    lines.append(
        "Candidate | ObservedMeanDR | RandomizedMeanDR | ObservedMinusRandomizedDR | ObservedSkew | "
        "RandomizedSkew | ObservedMinusRandomizedSkew | ObservedEntropy | RandomizedEntropy | ObservedMinusRandomizedEntropy"
    )
    for candidate in CANDIDATES:
        metric = randomized[candidate.name]
        lines.append(
            f"{candidate.name} | {fmt(metric.observed_mean_dr)} | {fmt(metric.randomized_mean_dr)} | "
            f"{fmt(metric.observed_mean_dr - metric.randomized_mean_dr)} | {pct(metric.observed_skew)} | "
            f"{pct(metric.randomized_skew)} | {pct(metric.observed_skew - metric.randomized_skew)} | "
            f"{fmt(metric.observed_entropy)} | {fmt(metric.randomized_entropy)} | "
            f"{fmt(metric.observed_entropy - metric.randomized_entropy)}"
        )


def append_compression_focus(
    lines: List[str],
    samples: Dict[str, List[Sample]],
    bootstrap: Dict[str, BootstrapMetrics],
    randomized: Dict[str, RandomizedMetrics],
) -> None:
    lines.append(
        "Candidate | Description | Count | MeanDR | Skew | Entropy | DominantDestination | DominantProbability | "
        "ContinuationRate | FailureRate | BootstrapP05Skew | BootstrapP95Skew | ObservedMinusRandomizedSkew"
    )
    for candidate_name in ("E", "A", "B"):
        candidate = CANDIDATE_BY_NAME[candidate_name]
        outcome = outcome_metrics(samples[candidate_name])
        transition = transition_metrics(samples[candidate_name], 1)
        boot = bootstrap[candidate_name]
        control = randomized[candidate_name]
        lines.append(
            f"{candidate.name} | {candidate.description} | {outcome.count} | {fmt(outcome.mean_dr)} | {pct(outcome.skew)} | "
            f"{fmt(transition.entropy)} | {transition.dominant_destination} | {pct(transition.dominant_probability)} | "
            f"{pct(outcome.continuation_rate)} | {pct(outcome.failure_rate)} | {pct(boot.p05_skew)} | "
            f"{pct(boot.p95_skew)} | {pct(control.observed_skew - control.randomized_skew)}"
        )


def write_instrument_report(result: StudyResult, out_root: str) -> str:
    out_dir = os.path.join(out_root, result.instrument)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"ContextDisambiguation_{result.instrument}.txt")
    state_counts = Counter(observation.state for observation in result.observations)
    age_counts = Counter(observation.age_bucket for observation in result.observations)
    context_counts = candidate_context_counts(result)
    lines = [
        "APVA Context Disambiguation Study v0.1",
        "Frozen candidate validation. Research only. No trades, optimization, fitting, machine learning, or parameter search.",
        "",
        "Diagnostics",
        f"Instrument: {result.instrument}",
        f"Input path(s): {'; '.join(result.source_paths)}",
        f"Total rows: {len(result.bars)}",
        "State counts: " + ", ".join(f"{state}={state_counts[state]}" for state in STRUCTURAL_STATES),
        "Age counts: " + ", ".join(f"{age}={age_counts[age]}" for age in AGE_BUCKETS),
        f"Distinct context count: {len(context_counts)}",
        "",
        "1. Candidate summary",
    ]
    append_candidate_summary(lines, result)
    lines += ["", "2. Outcome validation"]
    append_outcomes(lines, result)
    lines += ["", "3. Transition validation"]
    append_transitions(lines, result)
    lines += ["", "4. Baseline comparison"]
    append_comparisons(lines, result.samples)
    lines += ["", "5. Time stability"]
    append_time_stability(lines, result)
    lines += ["", "6. Bootstrap robustness"]
    append_bootstrap(lines, result.bootstrap)
    lines += ["", "7. Randomization test"]
    append_randomized(lines, result.randomized)
    lines += ["", "8. Compression Age-3 focus"]
    append_compression_focus(lines, result.samples, result.bootstrap, result.randomized)
    lines += [
        "",
        "9. Mechanical research notes",
        "- Candidates A-D are frozen prior-context tests. Candidates E-G are state-age baselines.",
        "- Bootstrap intervals use 1000 deterministic resamples with replacement.",
        "- Randomization controls shuffle PreviousDistinctState labels within this instrument 100 times.",
        "- Transition entropy is reported at each horizon; baseline comparisons use t+1 entropy.",
        "- Outcomes are downstream diagnostics and do not define candidates.",
        "- Review low-count candidate results cautiously.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def instrument_columns(instruments: Sequence[str], getter) -> str:
    parts: List[str] = []
    for instrument in instruments:
        parts.extend(getter(instrument))
    return " | ".join(parts)


def agreement(results: Sequence[StudyResult], candidate_name: str) -> Tuple[int, int, float, str]:
    directions = [
        direction(outcome_metrics(result.samples[candidate_name]))
        for result in results
        if outcome_metrics(result.samples[candidate_name]).count >= MIN_VALID_COUNT
    ]
    if not directions:
        return 0, 0, 0.0, "N/A"
    most_common, count = Counter(directions).most_common(1)[0]
    return count, len(directions), count / len(directions), most_common


def aggregate_report(results: Sequence[StudyResult], out_root: str) -> str:
    out_dir = os.path.join(out_root, "ContextDisambiguation")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "ContextDisambiguation_All.txt")
    instruments = [result.instrument for result in results]
    by_inst = {result.instrument: result for result in results}
    all_samples = {candidate.name: combined_samples(results, candidate.name) for candidate in CANDIDATES}
    aggregate_bootstrap = {candidate.name: combined_bootstrap(results, candidate.name) for candidate in CANDIDATES}
    aggregate_randomized = {candidate.name: combined_randomized(results, candidate.name) for candidate in CANDIDATES}
    lines = [
        "APVA Context Disambiguation Study v0.1 - Aggregate",
        "Frozen candidate validation. Research only. No trades, optimization, fitting, machine learning, or parameter search.",
        f"Instruments: {', '.join(instruments)}",
        "",
        "Aggregate Candidate Table",
        "Candidate | Description | "
        + instrument_columns(instruments, lambda instrument: [f"Count_{instrument}", f"MeanDR_{instrument}", f"Skew_{instrument}", f"Entropy_{instrument}"])
        + " | AgreementCount | AgreementPercent | AgreementDirection | MeanDR_All | Skew_All | Entropy_All",
    ]
    candidate_rows = []
    for candidate in CANDIDATES:
        values = {
            instrument: outcome_metrics(by_inst[instrument].samples[candidate.name])
            for instrument in instruments
        }
        transitions = {
            instrument: transition_metrics(by_inst[instrument].samples[candidate.name], 1)
            for instrument in instruments
        }
        combined = outcome_metrics(all_samples[candidate.name])
        combined_transition = transition_metrics(all_samples[candidate.name], 1)
        agree_count, valid_count, agree_pct, agree_direction = agreement(results, candidate.name)
        cols = instrument_columns(
            instruments,
            lambda instrument: [
                str(values[instrument].count),
                fmt(values[instrument].mean_dr),
                pct(values[instrument].skew),
                fmt(transitions[instrument].entropy),
            ],
        )
        candidate_rows.append((candidate, valid_count, combined, combined_transition, agree_count, agree_pct, agree_direction))
        lines.append(
            f"{candidate.name} | {candidate.description} | {cols} | {agree_count} of {valid_count} | "
            f"{pct(agree_pct)} | {agree_direction} | {fmt(combined.mean_dr)} | {pct(combined.skew)} | {fmt(combined_transition.entropy)}"
        )

    lines += [
        "",
        "Aggregate Baseline Comparison",
        "Candidate | Baseline | DeltaContinuationRate_All | DeltaFailureRate_All | DeltaOutcomeSkew_All | DeltaMeanDR_All | "
        "DeltaEntropy_All | DeltaDominantDestinationProbability_All | ReplicatedInstrumentCount",
    ]
    comparison_rows = []
    for candidate_name in ("A", "B", "C", "D"):
        candidate = CANDIDATE_BY_NAME[candidate_name]
        left, right, left_transition, right_transition = comparison_values(all_samples, candidate_name)
        replicated = 0
        for result in results:
            inst_left = outcome_metrics(result.samples[candidate_name])
            inst_right = outcome_metrics(result.samples[candidate.baseline or ""])
            if inst_left.count >= MIN_VALID_COUNT and inst_right.count >= MIN_VALID_COUNT:
                replicated += 1
        comparison_rows.append((candidate_name, replicated, left, right, left_transition, right_transition))
        lines.append(
            f"{candidate_name} | {candidate.baseline} | {pct(left.continuation_rate - right.continuation_rate)} | "
            f"{pct(left.failure_rate - right.failure_rate)} | {pct(left.skew - right.skew)} | {fmt(left.mean_dr - right.mean_dr)} | "
            f"{fmt(left_transition.entropy - right_transition.entropy)} | "
            f"{pct(left_transition.dominant_probability - right_transition.dominant_probability)} | {replicated}"
        )

    lines += ["", "Aggregate Bootstrap Table"]
    append_bootstrap(lines, aggregate_bootstrap)
    lines += ["", "Aggregate Randomization Table"]
    append_randomized(lines, aggregate_randomized)
    lines += ["", "Compression Age-3 Resolution Table"]
    append_compression_focus(lines, all_samples, aggregate_bootstrap, aggregate_randomized)

    lines += [
        "",
        "Instrument Stability",
        "Candidate | Instrument | Count | OutcomeSkew | MeanDR | Entropy | DominantDestination | DominantProbability | DirectionAgreement",
    ]
    for candidate in CANDIDATES:
        for result in results:
            metric = outcome_metrics(result.samples[candidate.name])
            transition = transition_metrics(result.samples[candidate.name], 1)
            lines.append(
                f"{candidate.name} | {result.instrument} | {metric.count} | {pct(metric.skew)} | {fmt(metric.mean_dr)} | "
                f"{fmt(transition.entropy)} | {transition.dominant_destination} | {pct(transition.dominant_probability)} | {direction(metric)}"
            )

    lines += ["", "Time Stability"]
    lines.append("Candidate | EarlyCount | LateCount | DeltaOutcomeSkew | DeltaMeanDR | DeltaEntropy")
    stability_rows = []
    for candidate in CANDIDATES:
        early: List[Sample] = []
        late: List[Sample] = []
        for result in results:
            instrument_early, instrument_late = split_samples(result.samples[candidate.name])
            early.extend(instrument_early)
            late.extend(instrument_late)
        early_metric = outcome_metrics(early)
        late_metric = outcome_metrics(late)
        early_transition = transition_metrics(early, 1)
        late_transition = transition_metrics(late, 1)
        stability_rows.append(
            (
                candidate.name,
                early_metric.count,
                late_metric.count,
                late_metric.skew - early_metric.skew,
                late_metric.mean_dr - early_metric.mean_dr,
                late_transition.entropy - early_transition.entropy,
            )
        )
        lines.append(
            f"{candidate.name} | {early_metric.count} | {late_metric.count} | {pct(late_metric.skew - early_metric.skew)} | "
            f"{fmt(late_metric.mean_dr - early_metric.mean_dr)} | {fmt(late_transition.entropy - early_transition.entropy)}"
        )

    def ranking(title: str, rows: Iterable[str]) -> None:
        lines.extend(["", title])
        materialized = list(rows)[:TOP_LIMIT]
        lines.extend(materialized or ["No candidates met the minimum validation threshold."])

    valid_candidates = [row for row in candidate_rows if row[1] >= MIN_VALID_INSTRUMENTS]
    validated_contexts = [row for row in comparison_rows if row[1] >= MIN_VALID_INSTRUMENTS]
    ranking(
        "1. Strongest validated context candidate",
        (
            f"{row[0]} | DeltaOutcomeSkew={pct(row[2].skew - row[3].skew)} | DeltaMeanDR={fmt(row[2].mean_dr - row[3].mean_dr)}"
            for row in sorted(validated_contexts, key=lambda row: -(row[2].skew - row[3].skew))
        ),
    )
    ranking(
        "2. Strongest instrument agreement",
        (
            f"{row[0].name} | {row[4]} of {row[1]} | Agreement={pct(row[5])} | Direction={row[6]}"
            for row in sorted(valid_candidates, key=lambda row: (-row[5], -row[4]))
        ),
    )
    ranking(
        "3. Strongest entropy reduction",
        (
            f"{row[0]} | EntropyReduction={fmt(row[5].entropy - row[4].entropy)}"
            for row in sorted(validated_contexts, key=lambda row: -(row[5].entropy - row[4].entropy))
        ),
    )
    ranking(
        "4. Strongest MeanDR improvement",
        (
            f"{row[0]} | DeltaMeanDR={fmt(row[2].mean_dr - row[3].mean_dr)}"
            for row in sorted(validated_contexts, key=lambda row: -(row[2].mean_dr - row[3].mean_dr))
        ),
    )
    ranking(
        "5. Strongest OutcomeSkew improvement",
        (
            f"{row[0]} | DeltaOutcomeSkew={pct(row[2].skew - row[3].skew)}"
            for row in sorted(validated_contexts, key=lambda row: -(row[2].skew - row[3].skew))
        ),
    )
    ranking(
        "6. Most time-stable candidate",
        (
            f"{row[0]} | AbsDeltaSkew={pct(abs(row[3]))} | AbsDeltaMeanDR={fmt(abs(row[4]))} | AbsDeltaEntropy={fmt(abs(row[5]))}"
            for row in sorted(stability_rows, key=lambda row: (abs(row[3]), abs(row[4]), abs(row[5])))
        ),
    )
    ranking(
        "7. Most bootstrap-robust candidate",
        (
            f"{candidate.name} | SkewCI=[{pct(aggregate_bootstrap[candidate.name].p05_skew)}, {pct(aggregate_bootstrap[candidate.name].p95_skew)}] | "
            f"MeanDRCI=[{fmt(aggregate_bootstrap[candidate.name].p05_mean_dr)}, {fmt(aggregate_bootstrap[candidate.name].p95_mean_dr)}]"
            for candidate in CANDIDATES
            if outcome_metrics(all_samples[candidate.name]).count >= MIN_VALID_COUNT
        ),
    )
    ranking(
        "8. Strongest randomization separation",
        (
            f"{candidate.name} | DeltaSkew={pct(aggregate_randomized[candidate.name].observed_skew - aggregate_randomized[candidate.name].randomized_skew)} | "
            f"DeltaMeanDR={fmt(aggregate_randomized[candidate.name].observed_mean_dr - aggregate_randomized[candidate.name].randomized_mean_dr)} | "
            f"DeltaEntropy={fmt(aggregate_randomized[candidate.name].observed_entropy - aggregate_randomized[candidate.name].randomized_entropy)}"
            for candidate in sorted(
                CANDIDATES,
                key=lambda candidate: -abs(aggregate_randomized[candidate.name].observed_skew - aggregate_randomized[candidate.name].randomized_skew),
            )
        ),
    )
    compression_rows = [row for row in candidate_rows if row[0].name in ("E", "A", "B") and row[1] >= MIN_VALID_INSTRUMENTS]
    ranking("9. Best Compression Age-3 context", (f"{row[0].name} | Skew={pct(row[2].skew)} | MeanDR={fmt(row[2].mean_dr)}" for row in sorted(compression_rows, key=lambda row: -row[2].skew)))
    ranking("10. Worst Compression Age-3 context", (f"{row[0].name} | Skew={pct(row[2].skew)} | MeanDR={fmt(row[2].mean_dr)}" for row in sorted(compression_rows, key=lambda row: row[2].skew)))
    lines += [
        "",
        "Mechanical research notes",
        "- This study validates only the seven frozen candidates named in the specification.",
        "- Context candidates A-D are compared with their frozen state-age baselines E-G.",
        "- Instrument agreement, chronological stability, deterministic bootstrap intervals, and shuffled-label controls are reported separately.",
        "- Randomization shuffles PreviousDistinctState labels within instrument; it does not change structural state, age bucket, or price outcome.",
        "- Outcomes are downstream diagnostics and never define candidates.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="APVA Context Disambiguation Study v0.1")
    parser.add_argument("inputs", nargs="+", help="Evidence CSV files or directories")
    parser.add_argument("--out-root", default="Evidence/Output", help="Output root directory")
    args = parser.parse_args(argv)
    loaded = load_results(args.inputs)
    if not loaded:
        raise SystemExit("No input rows loaded.")
    results = [build_result(result) for result in loaded]
    for result in results:
        write_instrument_report(result, args.out_root)
    aggregate = aggregate_report(results, args.out_root)
    print(f"Wrote ContextDisambiguation reports under {args.out_root}")
    print(f"Aggregate: {aggregate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
