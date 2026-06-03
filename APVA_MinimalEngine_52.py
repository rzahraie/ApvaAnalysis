#!/usr/bin/env python3
"""APVA Minimal Engine Study v0.1.

Reconstruct the established APVA structural-state stream, then test whether
StructuralState + AgeBucket is a sufficient low-degree-of-freedom engine.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from APVA_StructuralLifeCycle_44 import (
    AGE_BUCKETS,
    STRUCTURAL_STATES,
    directional_return,
    ensure_dir,
    fmt,
    load_results,
    pct,
)

HORIZONS = (1, 2, 3, 5)
SURVIVAL_HORIZONS = (1, 2, 3, 5, 10, 20)
ZONES = {
    "Young": ("1", "2"),
    "Middle": ("3", "4", "5"),
    "Late": ("6-10", "11-20", "21+"),
}
MIN_COUNT = 20


def entropy(counts: Iterable[int]) -> float:
    values = [value for value in counts if value > 0]
    total = sum(values)
    if total <= 0:
        return 0.0
    return -sum((value / total) * math.log(value / total) for value in values)


def normalized_entropy(counts: Iterable[int]) -> float:
    values = [value for value in counts if value > 0]
    if len(values) <= 1:
        return 0.0
    return entropy(values) / math.log(len(values))


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def distribution_text(counts: Counter[str]) -> str:
    total = sum(counts.values())
    if not total:
        return "N/A"
    return ", ".join(
        f"{state}:{pct(count / total)}"
        for state, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def dominant(counts: Counter[str]) -> tuple[str, float]:
    total = sum(counts.values())
    if not total:
        return "N/A", 0.0
    state, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return state, count / total


def classification(count: int, exit_rate: float) -> str:
    if count < MIN_COUNT:
        return "Insufficient"
    if exit_rate <= 0.20:
        return "Stable"
    if exit_rate <= 0.60:
        return "Transitional"
    return "Terminal"


def age_zone(bucket: str) -> str:
    for zone, buckets in ZONES.items():
        if bucket in buckets:
            return zone
    raise ValueError(f"Unknown age bucket: {bucket}")


@dataclass
class Hazard:
    state: str
    age_bucket: str
    horizon: int
    count: int
    destinations: Counter[str]
    persistence: float
    exit_rate: float
    entropy: float
    normalized_entropy: float
    dominant_destination: str
    dominant_probability: float


@dataclass
class Survival:
    state: str
    count: int
    rates: dict[int, float]


@dataclass
class AgeEffect:
    state: str
    min_exit_age: str
    min_exit_rate: float
    max_exit_age: str
    max_exit_rate: float
    exit_spread: float
    entropy_spread: float
    dominant_spread: float
    strength: float


@dataclass
class ZoneSummary:
    state: str
    zone: str
    count: int
    persistence: float
    exit_rate: float
    entropy: float
    dominant_destination: str
    dominant_probability: float


@dataclass
class Comparison:
    horizon: int
    state_entropy: float
    state_age_entropy: float
    reduction: float
    percent_reduction: float
    sparse_rate_increase: float
    unique_key_increase: int


@dataclass
class Outcome:
    count: int
    mean_dr: float
    median_dr: float
    continuation: float
    failure: float
    flat: float
    skew: float


@dataclass
class StudyResult:
    instrument: str
    source_paths: list[Path]
    bars: list
    hazards: dict[tuple[str, str, int], Hazard]
    survival: dict[str, Survival]
    age_effects: dict[str, AgeEffect]
    zones: dict[tuple[str, str], ZoneSummary]
    comparisons: dict[int, Comparison]
    outcomes: dict[tuple[str, str], Outcome]


def transition_groups(bars: list, horizon: int, with_age: bool = True):
    groups: dict[object, Counter[str]] = defaultdict(Counter)
    for index in range(len(bars) - horizon):
        bar = bars[index]
        key = (bar.state, bar.age_bucket) if with_age else bar.state
        groups[key][bars[index + horizon].state] += 1
    return groups


def conditional_entropy(groups: dict[object, Counter[str]]) -> float:
    total = sum(sum(counts.values()) for counts in groups.values())
    if not total:
        return 0.0
    return sum(sum(counts.values()) * entropy(counts.values()) for counts in groups.values()) / total


def sparse_rate(groups: dict[object, Counter[str]]) -> float:
    if not groups:
        return 0.0
    return sum(sum(counts.values()) < MIN_COUNT for counts in groups.values()) / len(groups)


def build_hazards(bars: list) -> dict[tuple[str, str, int], Hazard]:
    rows = {}
    for horizon in HORIZONS:
        for (state, bucket), counts in transition_groups(bars, horizon).items():
            count = sum(counts.values())
            destination, probability = dominant(counts)
            persistence = counts[state] / count
            rows[(state, bucket, horizon)] = Hazard(
                state, bucket, horizon, count, counts, persistence, 1.0 - persistence,
                entropy(counts.values()), normalized_entropy(counts.values()),
                destination, probability,
            )
    return rows


def build_survival(bars: list) -> dict[str, Survival]:
    rows = {}
    for state in STRUCTURAL_STATES:
        indexes = [index for index, bar in enumerate(bars) if bar.state == state]
        rates = {}
        for horizon in SURVIVAL_HORIZONS:
            eligible = [index for index in indexes if index + horizon < len(bars)]
            survived = sum(
                all(bars[future].state == state for future in range(index + 1, index + horizon + 1))
                for index in eligible
            )
            rates[horizon] = survived / len(eligible) if eligible else 0.0
        rows[state] = Survival(state, len(indexes), rates)
    return rows


def build_age_effects(hazards: dict[tuple[str, str, int], Hazard]) -> dict[str, AgeEffect]:
    rows = {}
    for state in STRUCTURAL_STATES:
        candidates = [
            hazards[(state, bucket, 1)]
            for bucket in AGE_BUCKETS
            if (state, bucket, 1) in hazards and hazards[(state, bucket, 1)].count >= MIN_COUNT
        ]
        if not candidates:
            candidates = [
                hazards[(state, bucket, 1)]
                for bucket in AGE_BUCKETS
                if (state, bucket, 1) in hazards
            ]
        if not candidates:
            rows[state] = AgeEffect(state, "N/A", 0.0, "N/A", 0.0, 0.0, 0.0, 0.0, 0.0)
            continue
        low = min(candidates, key=lambda row: row.exit_rate)
        high = max(candidates, key=lambda row: row.exit_rate)
        exit_spread = high.exit_rate - low.exit_rate
        entropy_spread = max(row.entropy for row in candidates) - min(row.entropy for row in candidates)
        dominant_spread = max(row.dominant_probability for row in candidates) - min(row.dominant_probability for row in candidates)
        rows[state] = AgeEffect(
            state, low.age_bucket, low.exit_rate, high.age_bucket, high.exit_rate,
            exit_spread, entropy_spread, dominant_spread, exit_spread + entropy_spread,
        )
    return rows


def build_zones(bars: list) -> dict[tuple[str, str], ZoneSummary]:
    groups: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for index in range(len(bars) - 1):
        bar = bars[index]
        groups[(bar.state, age_zone(bar.age_bucket))][bars[index + 1].state] += 1
    rows = {}
    for (state, zone), counts in groups.items():
        count = sum(counts.values())
        destination, probability = dominant(counts)
        persistence = counts[state] / count
        rows[(state, zone)] = ZoneSummary(
            state, zone, count, persistence, 1.0 - persistence, entropy(counts.values()),
            destination, probability,
        )
    return rows


def build_comparisons(bars: list) -> dict[int, Comparison]:
    rows = {}
    for horizon in HORIZONS:
        state_groups = transition_groups(bars, horizon, False)
        age_groups = transition_groups(bars, horizon, True)
        state_entropy = conditional_entropy(state_groups)
        age_entropy = conditional_entropy(age_groups)
        reduction = state_entropy - age_entropy
        rows[horizon] = Comparison(
            horizon, state_entropy, age_entropy, reduction,
            reduction / state_entropy if state_entropy else 0.0,
            sparse_rate(age_groups) - sparse_rate(state_groups),
            len(age_groups) - len(state_groups),
        )
    return rows


def outcome(values: list[float]) -> Outcome:
    if not values:
        return Outcome(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    count = len(values)
    continuation = sum(value > 0 for value in values) / count
    failure = sum(value < 0 for value in values) / count
    flat = sum(value == 0 for value in values) / count
    return Outcome(count, safe_mean(values), median(values), continuation, failure, flat, continuation - failure)


def build_outcomes(bars: list) -> dict[tuple[str, str], Outcome]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for index, bar in enumerate(bars):
        value = directional_return(bars, index)
        if value is not None:
            groups[(bar.state, bar.age_bucket)].append(value)
    return {key: outcome(values) for key, values in groups.items()}


def study(result) -> StudyResult:
    bars = result.bars
    hazards = build_hazards(bars)
    return StudyResult(
        result.instrument, result.source_paths, bars, hazards, build_survival(bars),
        build_age_effects(hazards), build_zones(bars), build_comparisons(bars),
        build_outcomes(bars),
    )


def recommendation(results: list[StudyResult]) -> tuple[str, str]:
    percent = safe_mean(row.percent_reduction for result in results for row in result.comparisons.values())
    strengths = {
        state: safe_mean(result.age_effects[state].strength for result in results)
        for state in STRUCTURAL_STATES
    }
    meaningful = sum(value > 0.20 for value in strengths.values())
    sparse = safe_mean(row.sparse_rate_increase for result in results for row in result.comparisons.values())
    valid = len(results)
    engine = "StateAge" if percent >= 0.03 and meaningful >= 3 and valid >= 2 else "StateOnly"
    reason = (
        f"MeanPercentEntropyReduction={pct(percent)}; "
        f"StatesWithAgeEffectStrengthAbove0.20={meaningful}; "
        f"MeanSparseRateIncrease={pct(sparse)}; "
        f"ValidInstrumentCount={valid}."
    )
    return engine, reason


def write_per_instrument(result: StudyResult, out_root: Path) -> None:
    path = out_root / result.instrument / f"MinimalEngine_{result.instrument}.txt"
    ensure_dir(path.parent)
    state_counts = Counter(bar.state for bar in result.bars)
    age_counts = Counter((bar.state, bar.age_bucket) for bar in result.bars)
    lines = [
        "APVA Minimal Engine Study v0.1",
        "=" * 80,
        "Diagnostics",
        f"Instrument: {result.instrument}",
        "Input path(s): " + ", ".join(str(item) for item in result.source_paths),
        f"Total rows: {len(result.bars)}",
        "State counts: " + ", ".join(f"{state}={state_counts[state]}" for state in STRUCTURAL_STATES),
        f"State-age key count: {len(age_counts)}",
        "",
        "1. Transition Hazard Table",
        "State | AgeBucket | Horizon | Count | NextStateDistribution | PersistenceRate | ExitRate | Entropy | NormalizedEntropy | DominantDestination | DominantProbability",
    ]
    for key in sorted(result.hazards):
        row = result.hazards[key]
        lines.append(
            f"{row.state} | {row.age_bucket} | t+{row.horizon} | {row.count} | "
            f"{distribution_text(row.destinations)} | {pct(row.persistence)} | {pct(row.exit_rate)} | "
            f"{fmt(row.entropy)} | {fmt(row.normalized_entropy)} | {row.dominant_destination} | {pct(row.dominant_probability)}"
        )
    lines += ["", "2. Survival Curves", "State | Count | Survival1 | Survival2 | Survival3 | Survival5 | Survival10 | Survival20"]
    for state in STRUCTURAL_STATES:
        row = result.survival[state]
        lines.append(f"{state} | {row.count} | " + " | ".join(pct(row.rates[h]) for h in SURVIVAL_HORIZONS))
    lines += ["", "3. Age Effect Test", "State | MinExitAge | MinExitRate | MaxExitAge | MaxExitRate | ExitRateSpread | EntropySpread | DominantDestinationSpread | AgeEffectStrength"]
    for state in STRUCTURAL_STATES:
        row = result.age_effects[state]
        lines.append(f"{state} | {row.min_exit_age} | {pct(row.min_exit_rate)} | {row.max_exit_age} | {pct(row.max_exit_rate)} | {fmt(row.exit_spread)} | {fmt(row.entropy_spread)} | {fmt(row.dominant_spread)} | {fmt(row.strength)}")
    lines += ["", "4. Young / Middle / Late Zone Table", "State | Zone | Count | PersistenceRate_t1 | ExitRate_t1 | Entropy_t1 | DominantDestination | DominantProbability"]
    for key in sorted(result.zones):
        row = result.zones[key]
        lines.append(f"{row.state} | {row.zone} | {row.count} | {pct(row.persistence)} | {pct(row.exit_rate)} | {fmt(row.entropy)} | {row.dominant_destination} | {pct(row.dominant_probability)}")
    lines += ["", "5. State vs State+Age Comparison", "Horizon | StateOnlyEntropy | StateAgeEntropy | EntropyReduction | PercentEntropyReduction | SparseRateIncrease | UniqueKeyIncrease"]
    for horizon in HORIZONS:
        row = result.comparisons[horizon]
        lines.append(f"t+{horizon} | {fmt(row.state_entropy)} | {fmt(row.state_age_entropy)} | {fmt(row.reduction)} | {pct(row.percent_reduction)} | {pct(row.sparse_rate_increase)} | {row.unique_key_increase}")
    lines += ["", "6. Minimal Engine Classification", "State | AgeBucket | Count | ExitRate_t1 | Entropy_t1 | Classification"]
    for state, bucket, horizon in sorted(result.hazards):
        if horizon == 1:
            row = result.hazards[(state, bucket, horizon)]
            lines.append(f"{state} | {bucket} | {row.count} | {pct(row.exit_rate)} | {fmt(row.entropy)} | {classification(row.count, row.exit_rate)}")
    lines += ["", "7. Outcome Diagnostics", "Diagnostic only. Outcomes are not used to classify engine states.", "State | AgeBucket | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew"]
    for key in sorted(result.outcomes):
        row = result.outcomes[key]
        lines.append(f"{key[0]} | {key[1]} | {row.count} | {fmt(row.mean_dr)} | {fmt(row.median_dr)} | {pct(row.continuation)} | {pct(row.failure)} | {pct(row.flat)} | {pct(row.skew)}")
    local_engine, local_reason = recommendation([result])
    lines += [
        "", "8. Minimal Engine Recommendation",
        f"Instrument-only diagnostic: {local_engine}",
        local_reason,
        "The aggregate report applies the required cross-instrument rule.",
        "", "9. Low-DoF Audit",
        "Variables used:",
        "StructuralState",
        "AgeBucket",
        "",
        "No context used.",
        "No arbitration used.",
        "No persistence used.",
        "No phase used.",
        "No optimization.",
        "No fitting.",
        "No machine learning.",
        "No forward returns used in model selection.",
        "", "10. Mechanical Research Notes",
        "- State + Age transition behavior is reported mechanically.",
        "- Age effects are measured from transition hazard and entropy spreads.",
        "- Young / Middle / Late zones are fixed summaries of the detailed age buckets.",
        "- Outcome diagnostics are reported separately and are not used for model selection.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def instrument_columns(results: list[StudyResult]) -> list[str]:
    preferred = ["6E", "CL", "NQ"]
    names = [result.instrument for result in results]
    return [name for name in preferred if name in names] + sorted(name for name in names if name not in preferred)


def result_map(results: list[StudyResult]) -> dict[str, StudyResult]:
    return {result.instrument: result for result in results}


def aggregate_report(results: list[StudyResult], out_root: Path) -> None:
    path = out_root / "MinimalEngine" / "MinimalEngine_All.txt"
    ensure_dir(path.parent)
    instruments = instrument_columns(results)
    by_instrument = result_map(results)
    lines = ["APVA Minimal Engine Study v0.1 - Aggregate", "=" * 100, "Instruments: " + ", ".join(instruments), ""]

    lines += ["Aggregate Transition Hazard Table", "State | AgeBucket | Horizon | " + " | ".join(f"Count_{i} | ExitRate_{i} | Entropy_{i} | DominantDestination_{i} | DominantProbability_{i}" for i in instruments) + " | ValidInstrumentCount | MeanExitRate | MeanEntropy | MeanDominantProbability"]
    hazard_keys = sorted({key for result in results for key in result.hazards})
    aggregate_hazards = []
    for key in hazard_keys:
        values = [by_instrument[i].hazards.get(key) for i in instruments]
        valid = [row for row in values if row and row.count >= MIN_COUNT]
        aggregate_hazards.append((key, valid))
        cells = []
        for row in values:
            cells += [str(row.count), pct(row.exit_rate), fmt(row.entropy), row.dominant_destination, pct(row.dominant_probability)] if row else ["0", "N/A", "N/A", "N/A", "N/A"]
        lines.append(f"{key[0]} | {key[1]} | t+{key[2]} | " + " | ".join(cells) + f" | {len(valid)} | {pct(safe_mean(r.exit_rate for r in valid))} | {fmt(safe_mean(r.entropy for r in valid))} | {pct(safe_mean(r.dominant_probability for r in valid))}")

    lines += ["", "Aggregate Survival Curve Table", "State | " + " | ".join(f"Survival1_{i} | Survival2_{i} | Survival3_{i} | Survival5_{i} | Survival10_{i} | Survival20_{i}" for i in instruments) + " | ValidInstrumentCount | MeanSurvival1 | MeanSurvival2 | MeanSurvival3 | MeanSurvival5 | MeanSurvival10 | MeanSurvival20"]
    for state in STRUCTURAL_STATES:
        values = [by_instrument[i].survival[state] for i in instruments]
        valid = [row for row in values if row.count >= MIN_COUNT]
        cells = [pct(row.rates[h]) for row in values for h in SURVIVAL_HORIZONS]
        means = [pct(safe_mean(row.rates[h] for row in valid)) for h in SURVIVAL_HORIZONS]
        lines.append(f"{state} | " + " | ".join(cells) + f" | {len(valid)} | " + " | ".join(means))

    lines += ["", "Aggregate Age Effect Table", "State | " + " | ".join(f"MinExitAge_{i} | MinExitRate_{i} | MaxExitAge_{i} | MaxExitRate_{i} | ExitRateSpread_{i} | EntropySpread_{i} | AgeEffectStrength_{i}" for i in instruments) + " | MeanExitRateSpread | MeanEntropySpread | MeanAgeEffectStrength"]
    aggregate_effects = []
    for state in STRUCTURAL_STATES:
        values = [by_instrument[i].age_effects[state] for i in instruments]
        aggregate_effects.append((state, safe_mean(row.strength for row in values)))
        cells = []
        for row in values:
            cells += [row.min_exit_age, pct(row.min_exit_rate), row.max_exit_age, pct(row.max_exit_rate), fmt(row.exit_spread), fmt(row.entropy_spread), fmt(row.strength)]
        lines.append(f"{state} | " + " | ".join(cells) + f" | {fmt(safe_mean(r.exit_spread for r in values))} | {fmt(safe_mean(r.entropy_spread for r in values))} | {fmt(safe_mean(r.strength for r in values))}")

    lines += ["", "Aggregate State vs State+Age Table", "Horizon | " + " | ".join(f"StateOnlyEntropy_{i} | StateAgeEntropy_{i} | EntropyReduction_{i} | PercentReduction_{i}" for i in instruments) + " | ValidInstrumentCount | MeanEntropyReduction | MeanPercentReduction | MeanSparseRateIncrease"]
    for horizon in HORIZONS:
        values = [by_instrument[i].comparisons[horizon] for i in instruments]
        cells = [value for row in values for value in (fmt(row.state_entropy), fmt(row.state_age_entropy), fmt(row.reduction), pct(row.percent_reduction))]
        lines.append(f"t+{horizon} | " + " | ".join(cells) + f" | {len(values)} | {fmt(safe_mean(r.reduction for r in values))} | {pct(safe_mean(r.percent_reduction for r in values))} | {pct(safe_mean(r.sparse_rate_increase for r in values))}")

    lines += ["", "Aggregate Zone Table", "State | Zone | " + " | ".join(f"Count_{i} | ExitRate_{i} | Entropy_{i}" for i in instruments) + " | ValidInstrumentCount | MeanExitRate | MeanEntropy"]
    zone_keys = sorted({key for result in results for key in result.zones})
    for key in zone_keys:
        values = [by_instrument[i].zones.get(key) for i in instruments]
        valid = [row for row in values if row and row.count >= MIN_COUNT]
        cells = [value for row in values for value in ((str(row.count), pct(row.exit_rate), fmt(row.entropy)) if row else ("0", "N/A", "N/A"))]
        lines.append(f"{key[0]} | {key[1]} | " + " | ".join(cells) + f" | {len(valid)} | {pct(safe_mean(r.exit_rate for r in valid))} | {fmt(safe_mean(r.entropy for r in valid))}")

    lines += ["", "Aggregate Classification Table", "State | AgeBucket | " + " | ".join(f"Classification_{i}" for i in instruments) + " | AgreementCount | AgreementPercent | FinalClassification"]
    class_keys = sorted({(state, bucket) for result in results for state, bucket, horizon in result.hazards if horizon == 1})
    for key in class_keys:
        classes = []
        for instrument in instruments:
            row = by_instrument[instrument].hazards.get((key[0], key[1], 1))
            classes.append(classification(row.count, row.exit_rate) if row else "Insufficient")
        final, agreement = Counter(classes).most_common(1)[0]
        lines.append(f"{key[0]} | {key[1]} | " + " | ".join(classes) + f" | {agreement} | {pct(agreement / len(classes))} | {final}")

    lines += ["", "Aggregate Outcome Diagnostic Table", "State | AgeBucket | " + " | ".join(f"Count_{i} | Skew_{i} | MeanDR_{i}" for i in instruments) + " | ValidInstrumentCount | MeanSkew | MeanDR"]
    outcome_keys = sorted({key for result in results for key in result.outcomes})
    aggregate_outcomes = []
    for key in outcome_keys:
        values = [by_instrument[i].outcomes.get(key) for i in instruments]
        valid = [row for row in values if row and row.count >= MIN_COUNT]
        aggregate_outcomes.append((key, valid))
        cells = [value for row in values for value in ((str(row.count), pct(row.skew), fmt(row.mean_dr)) if row else ("0", "N/A", "N/A"))]
        lines.append(f"{key[0]} | {key[1]} | " + " | ".join(cells) + f" | {len(valid)} | {pct(safe_mean(r.skew for r in valid))} | {fmt(safe_mean(r.mean_dr for r in valid))}")

    engine, reason = recommendation(results)
    lines += ["", "Aggregate Minimal Engine Recommendation", f"RecommendedEngine: {engine}", f"Reason: {reason}"]

    lines += ["", "Aggregate Rankings"]
    lines += ["", "1. Most stable state-age buckets"]
    for key, valid in sorted((item for item in aggregate_hazards if item[0][2] == 1 and item[1]), key=lambda item: safe_mean(row.exit_rate for row in item[1]))[:20]:
        lines.append(f"{key[0]} + {key[1]} | MeanExitRate={pct(safe_mean(row.exit_rate for row in valid))} | ValidInstrumentCount={len(valid)}")
    lines += ["", "2. Most terminal state-age buckets"]
    for key, valid in sorted((item for item in aggregate_hazards if item[0][2] == 1 and item[1]), key=lambda item: safe_mean(row.exit_rate for row in item[1]), reverse=True)[:20]:
        lines.append(f"{key[0]} + {key[1]} | MeanExitRate={pct(safe_mean(row.exit_rate for row in valid))} | ValidInstrumentCount={len(valid)}")
    lines += ["", "3. Strongest age effects by state"]
    for state, strength in sorted(aggregate_effects, key=lambda item: item[1], reverse=True):
        lines.append(f"{state} | MeanAgeEffectStrength={fmt(strength)}")
    lines += ["", "4. Weakest age effects by state"]
    for state, strength in sorted(aggregate_effects, key=lambda item: item[1]):
        lines.append(f"{state} | MeanAgeEffectStrength={fmt(strength)}")
    lines += ["", "5. Best survival states"]
    for state in sorted(STRUCTURAL_STATES, key=lambda item: safe_mean(by_instrument[i].survival[item].rates[5] for i in instruments), reverse=True):
        lines.append(f"{state} | MeanSurvival5={pct(safe_mean(by_instrument[i].survival[state].rates[5] for i in instruments))}")
    lines += ["", "6. Worst survival states"]
    for state in sorted(STRUCTURAL_STATES, key=lambda item: safe_mean(by_instrument[i].survival[item].rates[5] for i in instruments)):
        lines.append(f"{state} | MeanSurvival5={pct(safe_mean(by_instrument[i].survival[state].rates[5] for i in instruments))}")
    lines += ["", "7. Largest State+Age entropy improvement"]
    for horizon in sorted(HORIZONS, key=lambda h: safe_mean(by_instrument[i].comparisons[h].percent_reduction for i in instruments), reverse=True):
        lines.append(f"t+{horizon} | MeanPercentReduction={pct(safe_mean(by_instrument[i].comparisons[horizon].percent_reduction for i in instruments))}")
    valid_outcomes = [(key, valid) for key, valid in aggregate_outcomes if len(valid) >= 2]
    lines += ["", "8. Best outcome skew state-age buckets"]
    for key, valid in sorted(valid_outcomes, key=lambda item: safe_mean(row.skew for row in item[1]), reverse=True)[:20]:
        lines.append(f"{key[0]} + {key[1]} | MeanSkew={pct(safe_mean(row.skew for row in valid))} | ValidInstrumentCount={len(valid)}")
    lines += ["", "9. Worst outcome skew state-age buckets"]
    for key, valid in sorted(valid_outcomes, key=lambda item: safe_mean(row.skew for row in item[1]))[:20]:
        lines.append(f"{key[0]} + {key[1]} | MeanSkew={pct(safe_mean(row.skew for row in valid))} | ValidInstrumentCount={len(valid)}")
    lines += ["", "10. Recommended minimal APVA engine", f"{engine}: {reason}"]

    lines += [
        "", "Low-DoF Audit",
        "Variables used:",
        "StructuralState",
        "AgeBucket",
        "",
        "No context used.",
        "No arbitration used.",
        "No persistence used.",
        "No phase used.",
        "No optimization.",
        "No fitting.",
        "No machine learning.",
        "No forward returns used in model selection.",
        "", "Mechanical Research Notes",
        "- The minimal engine uses family, path, and archetype logic only to reconstruct StructuralState.",
        "- All engine analysis after reconstruction uses StructuralState and AgeBucket only.",
        "- State + Age is recommended only when the fixed entropy, age-effect, and replication rules pass.",
        "- Outcome diagnostics remain separate from model selection.",
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
