#!/usr/bin/env python3
"""APVA State Transition Model Study v0.1.

Represent the minimal APVA engine as a StructuralState + AgeBucket transition
system. Forward price outcomes are reported as diagnostics only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from APVA_MinimalEngine_52 import (
    AGE_BUCKETS,
    MIN_COUNT,
    STRUCTURAL_STATES,
    ZONES,
    age_zone,
    build_comparisons,
    build_outcomes,
    build_zones,
    conditional_entropy,
    dominant,
    entropy,
    instrument_columns,
    load_results,
    normalized_entropy,
    safe_mean,
    study as minimal_study,
    transition_groups,
)
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct


@dataclass
class TransitionLaw:
    state: str
    age_bucket: str
    count: int
    destinations: Counter[str]
    dominant_destination: str
    dominant_probability: float
    second_destination: str
    second_probability: float
    confidence: float
    entropy: float
    normalized_entropy: float
    persistence: float
    pressure: float
    momentum: str


@dataclass
class ConfidenceEffect:
    state: str
    min_age: str
    min_confidence: float
    max_age: str
    max_confidence: float
    spread: float


@dataclass
class Comparison:
    state_entropy: float
    state_age_entropy: float
    reduction: float
    percent_reduction: float
    dominant_probability_increase: float


@dataclass
class Chain:
    start: tuple[str, str]
    destinations: tuple[str, str, str]
    probability: float


@dataclass
class StudyResult:
    instrument: str
    source_paths: list[Path]
    bars: list
    laws: dict[tuple[str, str], TransitionLaw]
    confidence_effects: dict[str, ConfidenceEffect]
    zones: dict
    comparison: Comparison
    chains: dict[tuple[str, str], Chain]
    outcomes: dict


def momentum(persistence: float) -> str:
    if persistence >= 0.70:
        return "MomentumState"
    if persistence >= 0.30:
        return "BalancedState"
    return "TransitionState"


def destinations(counts: Counter[str]) -> tuple[str, float, str, float]:
    total = sum(counts.values())
    if not total:
        return "N/A", 0.0, "N/A", 0.0
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    first_state, first_count = ranked[0]
    second_state, second_count = ranked[1] if len(ranked) > 1 else ("N/A", 0)
    return first_state, first_count / total, second_state, second_count / total


def transition_laws(bars: list) -> dict[tuple[str, str], TransitionLaw]:
    rows = {}
    for (state, bucket), counts in transition_groups(bars, 1, True).items():
        count = sum(counts.values())
        first, first_probability, second, second_probability = destinations(counts)
        persistence = counts[state] / count
        rows[(state, bucket)] = TransitionLaw(
            state, bucket, count, counts, first, first_probability, second,
            second_probability, first_probability - second_probability,
            entropy(counts.values()), normalized_entropy(counts.values()),
            persistence, 1.0 - persistence, momentum(persistence),
        )
    return rows


def confidence_effects(laws: dict[tuple[str, str], TransitionLaw]) -> dict[str, ConfidenceEffect]:
    rows = {}
    for state in STRUCTURAL_STATES:
        candidates = [
            laws[(state, bucket)]
            for bucket in AGE_BUCKETS
            if (state, bucket) in laws and laws[(state, bucket)].count >= MIN_COUNT
        ]
        if not candidates:
            candidates = [laws[(state, bucket)] for bucket in AGE_BUCKETS if (state, bucket) in laws]
        if not candidates:
            rows[state] = ConfidenceEffect(state, "N/A", 0.0, "N/A", 0.0, 0.0)
            continue
        low = min(candidates, key=lambda row: row.confidence)
        high = max(candidates, key=lambda row: row.confidence)
        rows[state] = ConfidenceEffect(
            state, low.age_bucket, low.confidence, high.age_bucket,
            high.confidence, high.confidence - low.confidence,
        )
    return rows


def weighted_dominant_probability(groups: dict) -> float:
    total = sum(sum(counts.values()) for counts in groups.values())
    if not total:
        return 0.0
    return sum(max(counts.values()) for counts in groups.values() if counts) / total


def comparison(bars: list) -> Comparison:
    state_groups = transition_groups(bars, 1, False)
    state_age_groups = transition_groups(bars, 1, True)
    state_entropy = conditional_entropy(state_groups)
    state_age_entropy = conditional_entropy(state_age_groups)
    reduction = state_entropy - state_age_entropy
    return Comparison(
        state_entropy, state_age_entropy, reduction,
        reduction / state_entropy if state_entropy else 0.0,
        weighted_dominant_probability(state_age_groups) - weighted_dominant_probability(state_groups),
    )


def state_only_destinations(bars: list) -> dict[str, tuple[str, float]]:
    rows = {}
    for state, counts in transition_groups(bars, 1, False).items():
        rows[state] = dominant(counts)
    return rows


def process_chains(bars: list, laws: dict[tuple[str, str], TransitionLaw]) -> dict[tuple[str, str], Chain]:
    state_destinations = state_only_destinations(bars)
    rows = {}
    for key, law in laws.items():
        first = law.dominant_destination
        second, second_probability = state_destinations.get(first, ("N/A", 0.0))
        third, third_probability = state_destinations.get(second, ("N/A", 0.0))
        rows[key] = Chain(key, (first, second, third), law.dominant_probability * second_probability * third_probability)
    return rows


def study(result) -> StudyResult:
    minimal = minimal_study(result)
    laws = transition_laws(minimal.bars)
    return StudyResult(
        minimal.instrument, minimal.source_paths, minimal.bars, laws,
        confidence_effects(laws), build_zones(minimal.bars), comparison(minimal.bars),
        process_chains(minimal.bars, laws), build_outcomes(minimal.bars),
    )


def recommendation(results: list[StudyResult]) -> tuple[str, str]:
    percent = safe_mean(result.comparison.percent_reduction for result in results)
    strengths = {
        state: safe_mean(result.confidence_effects[state].spread for result in results)
        for state in STRUCTURAL_STATES
    }
    meaningful = sum(value > 0.20 for value in strengths.values())
    valid = len(results)
    engine = "StateAge" if percent >= 0.03 and meaningful >= 3 and valid >= 2 else "StateOnly"
    reason = (
        f"MeanPercentEntropyReduction={pct(percent)}; "
        f"StatesWithTransitionConfidenceSpreadAbove0.20={meaningful}; "
        f"ValidInstrumentCount={valid}."
    )
    return engine, reason


def matrix_probability(law: TransitionLaw, state: str) -> float:
    return law.destinations[state] / law.count if law.count else 0.0


def write_per_instrument(result: StudyResult, out_root: Path) -> None:
    path = out_root / result.instrument / f"StateTransitionModel_{result.instrument}.txt"
    ensure_dir(path.parent)
    state_counts = Counter(bar.state for bar in result.bars)
    lines = [
        "APVA State Transition Model Study v0.1",
        "=" * 88,
        "Diagnostics",
        f"Instrument: {result.instrument}",
        "Input path(s): " + ", ".join(str(item) for item in result.source_paths),
        f"Total rows: {len(result.bars)}",
        "State counts: " + ", ".join(f"{state}={state_counts[state]}" for state in STRUCTURAL_STATES),
        f"State-age counts: {len(result.laws)}",
        "",
        "1. Transition Matrix",
        "State | AgeBucket | NextState | Count | TransitionCount | TransitionProbability",
    ]
    for key in sorted(result.laws):
        law = result.laws[key]
        for state in STRUCTURAL_STATES:
            lines.append(f"{law.state} | {law.age_bucket} | {state} | {law.count} | {law.destinations[state]} | {pct(matrix_probability(law, state))}")
    lines += ["", "2. Dominant Destinations", "State | AgeBucket | Count | DominantDestination | DominantProbability | SecondDestination | SecondProbability | TransitionConfidence"]
    for key in sorted(result.laws):
        law = result.laws[key]
        lines.append(f"{law.state} | {law.age_bucket} | {law.count} | {law.dominant_destination} | {pct(law.dominant_probability)} | {law.second_destination} | {pct(law.second_probability)} | {pct(law.confidence)}")
    lines += ["", "3. Age-Dependent Transition Laws", "State | AgeBucket | Count | DominantDestination | DominantProbability | TransitionConfidence | Entropy"]
    for state in STRUCTURAL_STATES:
        for bucket in AGE_BUCKETS:
            law = result.laws.get((state, bucket))
            if law:
                lines.append(f"{state} | {bucket} | {law.count} | {law.dominant_destination} | {pct(law.dominant_probability)} | {pct(law.confidence)} | {fmt(law.entropy)}")
    lines += ["", "Transition Confidence Spread by State", "State | MinConfidenceAge | MinConfidence | MaxConfidenceAge | MaxConfidence | TransitionConfidenceSpread"]
    for state in STRUCTURAL_STATES:
        row = result.confidence_effects[state]
        lines.append(f"{state} | {row.min_age} | {pct(row.min_confidence)} | {row.max_age} | {pct(row.max_confidence)} | {pct(row.spread)}")
    lines += ["", "4. Transition Entropy", "State | AgeBucket | Count | TransitionEntropy | NormalizedTransitionEntropy"]
    for key in sorted(result.laws):
        row = result.laws[key]
        lines.append(f"{row.state} | {row.age_bucket} | {row.count} | {fmt(row.entropy)} | {fmt(row.normalized_entropy)}")
    lines += ["", "Lowest entropy buckets"]
    for row in sorted(result.laws.values(), key=lambda item: item.entropy)[:20]:
        lines.append(f"{row.state} + {row.age_bucket} | Count={row.count} | Entropy={fmt(row.entropy)}")
    lines += ["", "Highest entropy buckets"]
    for row in sorted(result.laws.values(), key=lambda item: item.entropy, reverse=True)[:20]:
        lines.append(f"{row.state} + {row.age_bucket} | Count={row.count} | Entropy={fmt(row.entropy)}")
    lines += ["", "5. Structural Momentum", "State | AgeBucket | Count | PersistenceProbability | Classification"]
    for key in sorted(result.laws):
        row = result.laws[key]
        lines.append(f"{row.state} | {row.age_bucket} | {row.count} | {pct(row.persistence)} | {row.momentum}")
    lines += ["", "6. Transition Pressure", "State | AgeBucket | Count | TransitionPressure"]
    for row in sorted(result.laws.values(), key=lambda item: item.pressure, reverse=True):
        lines.append(f"{row.state} | {row.age_bucket} | {row.count} | {pct(row.pressure)}")
    lines += ["", "7. Young / Middle / Late Transition Model", "State | Zone | Count | DominantDestination | DominantProbability | PersistenceProbability | TransitionEntropy"]
    for key in sorted(result.zones):
        row = result.zones[key]
        lines.append(f"{row.state} | {row.zone} | {row.count} | {row.dominant_destination} | {pct(row.dominant_probability)} | {pct(row.persistence)} | {fmt(row.entropy)}")
    row = result.comparison
    lines += [
        "", "8. State vs State+Age Comparison",
        "Horizon | TransitionEntropy_StateOnly | TransitionEntropy_StateAge | EntropyReduction | PercentEntropyReduction | DominantProbabilityIncrease",
        f"t+1 | {fmt(row.state_entropy)} | {fmt(row.state_age_entropy)} | {fmt(row.reduction)} | {pct(row.percent_reduction)} | {pct(row.dominant_probability_increase)}",
        "", "9. Structural Process Chains",
        "StartStateAge | Chain | ChainProbability",
    ]
    for chain in sorted(result.chains.values(), key=lambda item: item.probability, reverse=True):
        lines.append(f"{chain.start[0]} + {chain.start[1]} | {chain.start[0]} + {chain.start[1]} -> {' -> '.join(chain.destinations)} | {pct(chain.probability)}")
    lines += ["", "10. Outcome Diagnostics", "Diagnostic only. Forward outcomes are not used in transition modeling.", "State | AgeBucket | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew"]
    for key in sorted(result.outcomes):
        outcome = result.outcomes[key]
        lines.append(f"{key[0]} | {key[1]} | {outcome.count} | {fmt(outcome.mean_dr)} | {fmt(outcome.median_dr)} | {pct(outcome.continuation)} | {pct(outcome.failure)} | {pct(outcome.flat)} | {pct(outcome.skew)}")
    engine, reason = recommendation([result])
    lines += [
        "", "11. State Machine Recommendation",
        f"Instrument-only diagnostic: {engine}",
        reason,
        "The aggregate report applies the required cross-instrument rule.",
        "", "12. Low-DoF Audit",
        "Variables used:",
        "StructuralState",
        "AgeBucket",
        "",
        "No Context",
        "No Arbitration",
        "No Persistence",
        "No Phase",
        "No Optimization",
        "No Fitting",
        "No Machine Learning",
        "No Forward Returns used in transition modeling",
        "", "13. Mechanical Research Notes",
        "- Transition probabilities are computed from StructuralState and AgeBucket only.",
        "- Every observed StateAgeKey reports a complete destination matrix, including zero-count destinations.",
        "- Process chains follow the dominant StateAge transition once, then state-only dominant transitions for two additional steps.",
        "- Outcome diagnostics remain separate from transition modeling.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_report(results: list[StudyResult], out_root: Path) -> None:
    path = out_root / "StateTransitionModel" / "StateTransitionModel_All.txt"
    ensure_dir(path.parent)
    instruments = instrument_columns(results)
    by_instrument = {result.instrument: result for result in results}
    keys = sorted({key for result in results for key in result.laws})
    lines = ["APVA State Transition Model Study v0.1 - Aggregate", "=" * 104, "Instruments: " + ", ".join(instruments), ""]

    lines += ["Aggregate Transition Matrix Table", "State | AgeBucket | NextState | " + " | ".join(f"Count_{i} | Prob_{i}" for i in instruments) + " | ValidInstrumentCount | MeanProbability"]
    for key in keys:
        for destination in STRUCTURAL_STATES:
            values = [by_instrument[i].laws.get(key) for i in instruments]
            valid = [row for row in values if row and row.count >= MIN_COUNT]
            cells = [value for row in values for value in ((str(row.destinations[destination]), pct(matrix_probability(row, destination))) if row else ("0", "N/A"))]
            lines.append(f"{key[0]} | {key[1]} | {destination} | " + " | ".join(cells) + f" | {len(valid)} | {pct(safe_mean(matrix_probability(row, destination) for row in valid))}")

    lines += ["", "Aggregate Dominant Destination Table", "State | AgeBucket | " + " | ".join(f"Destination_{i} | Prob_{i}" for i in instruments) + " | ValidInstrumentCount | MeanDominantProbability | MeanTransitionConfidence"]
    aggregate_laws = []
    for key in keys:
        values = [by_instrument[i].laws.get(key) for i in instruments]
        valid = [row for row in values if row and row.count >= MIN_COUNT]
        aggregate_laws.append((key, valid))
        cells = [value for row in values for value in ((row.dominant_destination, pct(row.dominant_probability)) if row else ("N/A", "N/A"))]
        lines.append(f"{key[0]} | {key[1]} | " + " | ".join(cells) + f" | {len(valid)} | {pct(safe_mean(row.dominant_probability for row in valid))} | {pct(safe_mean(row.confidence for row in valid))}")

    lines += ["", "Aggregate Entropy Table", "State | AgeBucket | " + " | ".join(f"Entropy_{i}" for i in instruments) + " | ValidInstrumentCount | MeanEntropy"]
    for key, valid in aggregate_laws:
        values = [by_instrument[i].laws.get(key) for i in instruments]
        lines.append(f"{key[0]} | {key[1]} | " + " | ".join(fmt(row.entropy) if row else "N/A" for row in values) + f" | {len(valid)} | {fmt(safe_mean(row.entropy for row in valid))}")

    lines += ["", "Aggregate Momentum Table", "State | AgeBucket | " + " | ".join(f"PersistenceProb_{i} | Classification_{i}" for i in instruments) + " | AgreementCount | FinalClassification"]
    for key in keys:
        values = [by_instrument[i].laws.get(key) for i in instruments]
        classes = [row.momentum if row else "Insufficient" for row in values]
        final, agreement = Counter(classes).most_common(1)[0]
        cells = [value for row in values for value in ((pct(row.persistence), row.momentum) if row else ("N/A", "Insufficient"))]
        lines.append(f"{key[0]} | {key[1]} | " + " | ".join(cells) + f" | {agreement} | {final}")

    lines += ["", "Aggregate Pressure Table", "State | AgeBucket | " + " | ".join(f"Pressure_{i}" for i in instruments) + " | ValidInstrumentCount | MeanPressure"]
    for key, valid in aggregate_laws:
        values = [by_instrument[i].laws.get(key) for i in instruments]
        lines.append(f"{key[0]} | {key[1]} | " + " | ".join(pct(row.pressure) if row else "N/A" for row in values) + f" | {len(valid)} | {pct(safe_mean(row.pressure for row in valid))}")

    lines += ["", "Aggregate Zone Table", "State | Zone | " + " | ".join(f"Destination_{i} | Probability_{i} | Entropy_{i} | PersistenceProbability_{i}" for i in instruments) + " | ValidInstrumentCount | MeanProbability | MeanEntropy | MeanPersistenceProbability"]
    zone_keys = sorted({key for result in results for key in result.zones})
    for key in zone_keys:
        values = [by_instrument[i].zones.get(key) for i in instruments]
        valid = [row for row in values if row and row.count >= MIN_COUNT]
        cells = [value for row in values for value in ((row.dominant_destination, pct(row.dominant_probability), fmt(row.entropy), pct(row.persistence)) if row else ("N/A", "N/A", "N/A", "N/A"))]
        lines.append(f"{key[0]} | {key[1]} | " + " | ".join(cells) + f" | {len(valid)} | {pct(safe_mean(row.dominant_probability for row in valid))} | {fmt(safe_mean(row.entropy for row in valid))} | {pct(safe_mean(row.persistence for row in valid))}")

    comparison_values = [by_instrument[i].comparison for i in instruments]
    lines += [
        "", "Aggregate State vs State+Age Table",
        "Horizon | " + " | ".join(f"StateOnlyEntropy_{i} | StateAgeEntropy_{i} | EntropyReduction_{i} | PercentEntropyReduction_{i} | DominantProbabilityIncrease_{i}" for i in instruments) + " | ValidInstrumentCount | MeanEntropyReduction | MeanPercentEntropyReduction | MeanDominantProbabilityIncrease",
        "t+1 | " + " | ".join(value for row in comparison_values for value in (fmt(row.state_entropy), fmt(row.state_age_entropy), fmt(row.reduction), pct(row.percent_reduction), pct(row.dominant_probability_increase))) + f" | {len(comparison_values)} | {fmt(safe_mean(row.reduction for row in comparison_values))} | {pct(safe_mean(row.percent_reduction for row in comparison_values))} | {pct(safe_mean(row.dominant_probability_increase for row in comparison_values))}",
    ]

    lines += ["", "Aggregate Process Chain Table", "StartStateAge | " + " | ".join(f"Chain_{i} | ChainProbability_{i}" for i in instruments) + " | ValidInstrumentCount | MeanChainProbability"]
    for key in keys:
        values = [by_instrument[i].chains.get(key) for i in instruments]
        valid = [row for instrument, row in zip(instruments, values) if row and by_instrument[instrument].laws[key].count >= MIN_COUNT]
        cells = [value for row in values for value in ((f"{key[0]} + {key[1]} -> {' -> '.join(row.destinations)}", pct(row.probability)) if row else ("N/A", "N/A"))]
        lines.append(f"{key[0]} + {key[1]} | " + " | ".join(cells) + f" | {len(valid)} | {pct(safe_mean(row.probability for row in valid))}")

    lines += ["", "Aggregate Outcome Table", "State | AgeBucket | " + " | ".join(f"Count_{i} | Skew_{i} | MeanDR_{i}" for i in instruments) + " | ValidInstrumentCount | MeanSkew | MeanDR"]
    outcome_keys = sorted({key for result in results for key in result.outcomes})
    aggregate_outcomes = []
    for key in outcome_keys:
        values = [by_instrument[i].outcomes.get(key) for i in instruments]
        valid = [row for row in values if row and row.count >= MIN_COUNT]
        aggregate_outcomes.append((key, valid))
        cells = [value for row in values for value in ((str(row.count), pct(row.skew), fmt(row.mean_dr)) if row else ("0", "N/A", "N/A"))]
        lines.append(f"{key[0]} | {key[1]} | " + " | ".join(cells) + f" | {len(valid)} | {pct(safe_mean(row.skew for row in valid))} | {fmt(safe_mean(row.mean_dr for row in valid))}")

    engine, reason = recommendation(results)
    effect_rows = [(state, safe_mean(result.confidence_effects[state].spread for result in results)) for state in STRUCTURAL_STATES]
    lines += [
        "", "Aggregate State Machine Recommendation",
        f"RecommendedEngine: {engine}",
        f"Reason: {reason}",
        f"EntropyReduction: {pct(safe_mean(row.percent_reduction for row in comparison_values))}",
        f"TransitionConfidence: {pct(safe_mean(row.confidence for _, valid in aggregate_laws for row in valid))}",
        f"AgeEffectStrength: {sum(strength > 0.20 for _, strength in effect_rows)} states above 0.20 transition-confidence spread",
        f"CrossInstrumentReplication: {len(results)} instruments",
        "", "Aggregate Rankings",
        "", "1. Most predictable state-age buckets",
    ]
    ranked_laws = [(key, valid) for key, valid in aggregate_laws if valid]
    for key, valid in sorted(ranked_laws, key=lambda item: safe_mean(row.entropy for row in item[1]))[:20]:
        lines.append(f"{key[0]} + {key[1]} | MeanEntropy={fmt(safe_mean(row.entropy for row in valid))} | ValidInstrumentCount={len(valid)}")
    lines += ["", "2. Least predictable state-age buckets"]
    for key, valid in sorted(ranked_laws, key=lambda item: safe_mean(row.entropy for row in item[1]), reverse=True)[:20]:
        lines.append(f"{key[0]} + {key[1]} | MeanEntropy={fmt(safe_mean(row.entropy for row in valid))} | ValidInstrumentCount={len(valid)}")
    lines += ["", "3. Highest confidence transitions"]
    for key, valid in sorted(ranked_laws, key=lambda item: safe_mean(row.confidence for row in item[1]), reverse=True)[:20]:
        lines.append(f"{key[0]} + {key[1]} | MeanConfidence={pct(safe_mean(row.confidence for row in valid))} | ValidInstrumentCount={len(valid)}")
    lines += ["", "4. Lowest confidence transitions"]
    for key, valid in sorted(ranked_laws, key=lambda item: safe_mean(row.confidence for row in item[1]))[:20]:
        lines.append(f"{key[0]} + {key[1]} | MeanConfidence={pct(safe_mean(row.confidence for row in valid))} | ValidInstrumentCount={len(valid)}")
    lines += ["", "5. Highest momentum states"]
    for key, valid in sorted(ranked_laws, key=lambda item: safe_mean(row.persistence for row in item[1]), reverse=True)[:20]:
        lines.append(f"{key[0]} + {key[1]} | MeanPersistence={pct(safe_mean(row.persistence for row in valid))}")
    lines += ["", "6. Highest transition-pressure states"]
    for key, valid in sorted(ranked_laws, key=lambda item: safe_mean(row.pressure for row in item[1]), reverse=True)[:20]:
        lines.append(f"{key[0]} + {key[1]} | MeanPressure={pct(safe_mean(row.pressure for row in valid))}")
    lines += ["", "7. Strongest age-dependent states"]
    for state, spread in sorted(effect_rows, key=lambda item: item[1], reverse=True):
        lines.append(f"{state} | MeanTransitionConfidenceSpread={pct(spread)}")
    lines += ["", "8. Weakest age-dependent states"]
    for state, spread in sorted(effect_rows, key=lambda item: item[1]):
        lines.append(f"{state} | MeanTransitionConfidenceSpread={pct(spread)}")
    lines += ["", "9. Most probable process chains"]
    chain_rows = []
    for key in keys:
        valid = [result.chains[key] for result in results if key in result.chains and result.laws[key].count >= MIN_COUNT]
        if valid:
            chain_rows.append((key, valid))
    for key, valid in sorted(chain_rows, key=lambda item: safe_mean(row.probability for row in item[1]), reverse=True)[:20]:
        lines.append(f"{key[0]} + {key[1]} | MeanChainProbability={pct(safe_mean(row.probability for row in valid))} | ValidInstrumentCount={len(valid)}")
    lines += ["", "10. Recommended APVA state machine", f"{engine}: {reason}"]

    lines += [
        "", "Low-DoF Audit",
        "Variables used:",
        "StructuralState",
        "AgeBucket",
        "",
        "No Context",
        "No Arbitration",
        "No Persistence",
        "No Phase",
        "No Optimization",
        "No Fitting",
        "No Machine Learning",
        "No Forward Returns used in transition modeling",
        "", "Mechanical Research Notes",
        "- APVA is represented as a complete StateAgeKey -> NextState transition system.",
        "- State + Age is compared mechanically with State-only transition entropy.",
        "- Young / Middle / Late zones summarize the same fixed age buckets.",
        "- Outcome diagnostics remain separate from transition modeling and recommendation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Evidence CSV files or directories")
    parser.add_argument("--out-root", default="Evidence/Output", help="Output root directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [study(result) for result in load_results(args.inputs)]
    if not results:
        raise SystemExit("No APVA evidence CSV files found.")
    out_root = Path(args.out_root)
    for result in results:
        write_per_instrument(result, out_root)
    aggregate_report(results, out_root)
    print(f"Wrote {len(results)} per-instrument report(s) and aggregate report under {out_root}.")


if __name__ == "__main__":
    main()
