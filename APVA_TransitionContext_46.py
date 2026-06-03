#!/usr/bin/env python3
"""
APVA Transition Context Matrix Study v0.1

Research whether prior structural context explains state-age transition
behavior beyond the current structural state and age bucket.

No trades. No entries/exits. No optimization. No fitting. No machine
learning. Research only.
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

HORIZONS = (1, 2, 3, 5)
CONTEXT_TYPES = ("Context1", "ContextDistinct", "ContextPath3", "ContextPath5")
MIN_INSTRUMENT_CONTEXT = 10
MIN_AGGREGATE_CONTEXT = 50
MIN_VALID_INSTRUMENTS = 2
TOP_LIMIT = 25


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
        return sum(self.values) / len(self.values) if self.values else 0.0

    @property
    def median_dr(self) -> float:
        return stats.median(self.values) if self.values else 0.0

    @property
    def cont_rate(self) -> float:
        return sum(x > 0 for x in self.values) / len(self.values) if self.values else 0.0

    @property
    def fail_rate(self) -> float:
        return sum(x < 0 for x in self.values) / len(self.values) if self.values else 0.0

    @property
    def flat_rate(self) -> float:
        return sum(x == 0 for x in self.values) / len(self.values) if self.values else 0.0

    @property
    def skew(self) -> float:
        return self.cont_rate - self.fail_rate


@dataclass(frozen=True)
class Observation:
    index: int
    state: str
    age: int
    age_bucket: str
    previous1: str
    previous2: str
    previous3: str
    previous_distinct: str
    previous_distinct_distance: Optional[int]
    contexts: Dict[str, str]


@dataclass(frozen=True)
class MatrixRow:
    kind: str
    key: str
    state: str
    age_bucket: str
    horizon: int
    next_state: str
    source_count: int
    count: int
    probability: float
    lift: Optional[float]
    entropy: float
    normalized_entropy: float


@dataclass(frozen=True)
class EntropyRow:
    context_type: str
    state: str
    age_bucket: str
    horizon: int
    count: int
    baseline_entropy: float
    context_entropy: float
    reduction: float


@dataclass(frozen=True)
class FocusRow:
    context_type: str
    key: str
    count: int
    distribution: str
    entropy: float
    normalized_entropy: float
    outcome: Outcome


@dataclass(frozen=True)
class ConcentrationRow:
    state: str
    age_bucket: str
    context_type: str
    top_context: str
    count: int
    total: int
    share: float
    dominant_next: str
    dominant_probability: float
    outcome_skew: float


@dataclass
class StudyResult:
    instrument: str
    bars: List[Bar]
    source_paths: List[str]
    observations: List[Observation]
    baseline: List[MatrixRow]
    contexts: List[MatrixRow]
    entropy_rows: List[EntropyRow]
    focus_rows: List[FocusRow]
    concentration_rows: List[ConcentrationRow]
    outcomes: Dict[Tuple[str, str], Outcome]


def entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((n / total) * math.log(n / total) for n in counts.values() if n)


def normalized_entropy(counts: Counter[str]) -> float:
    observed = sum(1 for n in counts.values() if n)
    if observed <= 1:
        return 0.0
    return entropy(counts) / math.log(observed)


def mean(values: Iterable[float]) -> float:
    xs = list(values)
    return sum(xs) / len(xs) if xs else 0.0


def stdev(values: Iterable[float]) -> float:
    xs = list(values)
    return stats.pstdev(xs) if len(xs) > 1 else 0.0


def baseline_key(state: str, age_bucket: str) -> str:
    return f"{state}+Age{age_bucket}"


def context_key(states: Sequence[str], state: str, age_bucket: str) -> str:
    return f"{'->'.join(states)}->{state}+Age{age_bucket}"


def build_observations(bars: List[Bar]) -> List[Observation]:
    observations: List[Observation] = []
    for i, bar in enumerate(bars):
        previous1 = bars[i - 1].state if i >= 1 else ""
        previous2 = bars[i - 2].state if i >= 2 else ""
        previous3 = bars[i - 3].state if i >= 3 else ""
        previous_distinct = ""
        previous_distinct_distance: Optional[int] = None
        for prior in range(i - 1, -1, -1):
            if bars[prior].state != bar.state:
                previous_distinct = bars[prior].state
                previous_distinct_distance = i - prior
                break
        contexts: Dict[str, str] = {}
        if previous1:
            contexts["Context1"] = context_key((previous1,), bar.state, bar.age_bucket)
        if previous_distinct:
            contexts["ContextDistinct"] = context_key((previous_distinct,), bar.state, bar.age_bucket)
        if i >= 2:
            contexts["ContextPath3"] = context_key(
                tuple(x.state for x in bars[i - 2:i]), bar.state, bar.age_bucket
            )
        if i >= 4:
            contexts["ContextPath5"] = context_key(
                tuple(x.state for x in bars[i - 4:i]), bar.state, bar.age_bucket
            )
        observations.append(
            Observation(
                i,
                bar.state,
                bar.age,
                bar.age_bucket,
                previous1,
                previous2,
                previous3,
                previous_distinct,
                previous_distinct_distance,
                contexts,
            )
        )
    return observations


def build_matrix(
    bars: List[Bar],
    observations: List[Observation],
    kind: str,
    minimum_count: int = 1,
) -> List[MatrixRow]:
    rows: List[MatrixRow] = []
    for horizon in HORIZONS:
        unconditional: Counter[str] = Counter()
        grouped: Dict[Tuple[str, str, str], Counter[str]] = defaultdict(Counter)
        for obs in observations:
            if obs.index + horizon >= len(bars):
                continue
            nxt = bars[obs.index + horizon].state
            unconditional[nxt] += 1
            if kind == "Baseline":
                key = baseline_key(obs.state, obs.age_bucket)
            else:
                key = obs.contexts.get(kind, "")
            if key:
                grouped[(key, obs.state, obs.age_bucket)][nxt] += 1
        total_unconditional = sum(unconditional.values())
        for (key, state, age_bucket), destination_counts in grouped.items():
            source_count = sum(destination_counts.values())
            if source_count < minimum_count:
                continue
            ent = entropy(destination_counts)
            norm_ent = normalized_entropy(destination_counts)
            for nxt, count in destination_counts.items():
                probability = count / source_count
                baseline = unconditional[nxt] / total_unconditional if total_unconditional else 0.0
                lift = probability / baseline if baseline else None
                rows.append(
                    MatrixRow(
                        kind,
                        key,
                        state,
                        age_bucket,
                        horizon,
                        nxt,
                        source_count,
                        count,
                        probability,
                        lift,
                        ent,
                        norm_ent,
                    )
                )
    return sorted(rows, key=lambda x: (x.kind, x.key, x.horizon, x.next_state))


def build_outcomes(
    bars: List[Bar], observations: List[Observation]
) -> Dict[Tuple[str, str], Outcome]:
    outcomes: Dict[Tuple[str, str], Outcome] = defaultdict(Outcome)
    for obs in observations:
        dr = directional_return(bars, obs.index, 5)
        outcomes[("Baseline", baseline_key(obs.state, obs.age_bucket))].add(dr)
        for kind, key in obs.contexts.items():
            outcomes[(kind, key)].add(dr)
    return outcomes


def build_entropy_rows(
    bars: List[Bar], observations: List[Observation]
) -> List[EntropyRow]:
    rows: List[EntropyRow] = []
    for kind in CONTEXT_TYPES:
        for state in STRUCTURAL_STATES:
            for age_bucket in AGE_BUCKETS:
                subset = [o for o in observations if o.state == state and o.age_bucket == age_bucket and kind in o.contexts]
                for horizon in HORIZONS:
                    eligible = [o for o in subset if o.index + horizon < len(bars)]
                    if not eligible:
                        continue
                    baseline_counts = Counter(bars[o.index + horizon].state for o in eligible)
                    by_context: Dict[str, Counter[str]] = defaultdict(Counter)
                    for obs in eligible:
                        by_context[obs.contexts[kind]][bars[obs.index + horizon].state] += 1
                    context_ent = sum(
                        (sum(counts.values()) / len(eligible)) * entropy(counts)
                        for counts in by_context.values()
                    )
                    baseline_ent = entropy(baseline_counts)
                    rows.append(
                        EntropyRow(
                            kind,
                            state,
                            age_bucket,
                            horizon,
                            len(eligible),
                            baseline_ent,
                            context_ent,
                            baseline_ent - context_ent,
                        )
                    )
    return rows


def focus_value(obs: Observation, kind: str, bars: List[Bar]) -> str:
    if kind == "Context1":
        return obs.previous1
    if kind == "ContextDistinct":
        return obs.previous_distinct
    if kind == "ContextPath3" and obs.index >= 2:
        return "->".join(x.state for x in bars[obs.index - 2:obs.index])
    if kind == "ContextPath5" and obs.index >= 4:
        return "->".join(x.state for x in bars[obs.index - 4:obs.index])
    return ""


def build_focus_rows(bars: List[Bar], observations: List[Observation]) -> List[FocusRow]:
    rows: List[FocusRow] = []
    subset = [o for o in observations if o.state == "CompressionProcessing" and o.age_bucket == "3"]
    for kind in CONTEXT_TYPES:
        groups: Dict[str, List[Observation]] = defaultdict(list)
        for obs in subset:
            value = focus_value(obs, kind, bars)
            if value and obs.index + 1 < len(bars):
                groups[value].append(obs)
        for key, group in groups.items():
            destinations = Counter(bars[o.index + 1].state for o in group)
            distribution = ", ".join(
                f"{state}={pct(count / len(group))}"
                for state, count in destinations.most_common()
            )
            outcome = Outcome()
            for obs in group:
                outcome.add(directional_return(bars, obs.index, 5))
            rows.append(
                FocusRow(
                    kind,
                    key,
                    len(group),
                    distribution,
                    entropy(destinations),
                    normalized_entropy(destinations),
                    outcome,
                )
            )
    return sorted(rows, key=lambda x: (x.context_type, -x.count, x.key))


def build_concentration_rows(
    bars: List[Bar], observations: List[Observation]
) -> List[ConcentrationRow]:
    rows: List[ConcentrationRow] = []
    for state in STRUCTURAL_STATES:
        for age_bucket in AGE_BUCKETS:
            subset = [o for o in observations if o.state == state and o.age_bucket == age_bucket]
            for kind in CONTEXT_TYPES:
                eligible = [o for o in subset if kind in o.contexts and o.index + 1 < len(bars)]
                if not eligible:
                    continue
                groups: Dict[str, List[Observation]] = defaultdict(list)
                for obs in eligible:
                    groups[obs.contexts[kind]].append(obs)
                top_context, group = max(groups.items(), key=lambda item: (len(item[1]), item[0]))
                destinations = Counter(bars[o.index + 1].state for o in group)
                dominant_next, dominant_count = destinations.most_common(1)[0]
                outcome = Outcome()
                for obs in group:
                    outcome.add(directional_return(bars, obs.index, 5))
                rows.append(
                    ConcentrationRow(
                        state,
                        age_bucket,
                        kind,
                        top_context,
                        len(group),
                        len(eligible),
                        len(group) / len(eligible),
                        dominant_next,
                        dominant_count / len(group),
                        outcome.skew,
                    )
                )
    return rows


def build_study_result(result: InstrumentResult) -> StudyResult:
    observations = build_observations(result.bars)
    return StudyResult(
        result.instrument,
        result.bars,
        result.source_paths,
        observations,
        build_matrix(result.bars, observations, "Baseline"),
        [row for kind in CONTEXT_TYPES for row in build_matrix(result.bars, observations, kind, MIN_INSTRUMENT_CONTEXT)],
        build_entropy_rows(result.bars, observations),
        build_focus_rows(result.bars, observations),
        build_concentration_rows(result.bars, observations),
        build_outcomes(result.bars, observations),
    )


def age_sort(age_bucket: str) -> int:
    return AGE_BUCKETS.index(age_bucket) if age_bucket in AGE_BUCKETS else len(AGE_BUCKETS)


def state_age_counts(result: StudyResult) -> Counter[Tuple[str, str]]:
    return Counter((o.state, o.age_bucket) for o in result.observations)


def write_matrix(lines: List[str], rows: Iterable[MatrixRow], include_kind: bool = False) -> None:
    if include_kind:
        lines.append("ContextType | ContextKey | Horizon | NextState | SourceCount | Count | Probability | Lift | Entropy | NormalizedEntropy")
    else:
        lines.append("CurrentState | AgeBucket | Horizon | NextState | SourceCount | Count | Probability | Lift | Entropy | NormalizedEntropy")
    for row in rows:
        prefix = f"{row.kind} | {row.key}" if include_kind else f"{row.state} | {row.age_bucket}"
        lines.append(
            f"{prefix} | t+{row.horizon} | {row.next_state} | {row.source_count} | {row.count} | "
            f"{pct(row.probability)} | {fmt(row.lift)} | {fmt(row.entropy)} | {fmt(row.normalized_entropy)}"
        )


def write_instrument_report(result: StudyResult, out_root: str) -> str:
    out_dir = os.path.join(out_root, result.instrument)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"TransitionContext_{result.instrument}.txt")
    state_counts = Counter(o.state for o in result.observations)
    age_counts = state_age_counts(result)
    lines = [
        "APVA Transition Context Matrix Study v0.1",
        "Research only. No trades, entries/exits, optimization, fitting, or machine learning.",
        "",
        "Diagnostics",
        f"Instrument: {result.instrument}",
        f"Input path(s): {'; '.join(result.source_paths)}",
        f"Total rows: {len(result.bars)}",
        f"Valid context observations: {sum(bool(o.contexts) for o in result.observations)}",
        "Structural state counts: " + ", ".join(f"{x}={state_counts[x]}" for x in STRUCTURAL_STATES),
        "State-age counts: " + ", ".join(
            f"{state}+Age{age}={count}"
            for (state, age), count in sorted(age_counts.items(), key=lambda x: (x[0][0], age_sort(x[0][1])))
        ),
        "",
        "1. Baseline State+Age transition table",
    ]
    write_matrix(lines, result.baseline)
    lines += ["", "2. Context transition tables"]
    write_matrix(lines, result.contexts, include_kind=True)
    lines += [
        "",
        "3. Context vs baseline entropy",
        "ContextType | CurrentState | AgeBucket | Horizon | Count | BaselineEntropy | ContextEntropy | EntropyReduction",
    ]
    for row in sorted(result.entropy_rows, key=lambda x: (x.context_type, x.state, age_sort(x.age_bucket), x.horizon)):
        lines.append(
            f"{row.context_type} | {row.state} | {row.age_bucket} | t+{row.horizon} | {row.count} | "
            f"{fmt(row.baseline_entropy)} | {fmt(row.context_entropy)} | {fmt(row.reduction)}"
        )
    lines += [
        "",
        "4. Compression Age-3 focus",
        "ContextType | ContextKey | Count | NextStateProbabilities_t1 | Entropy | NormalizedEntropy | ContRate5 | FailRate5 | OutcomeSkew | MeanDRFwd5 | MedianDRFwd5",
    ]
    for row in result.focus_rows:
        lines.append(
            f"{row.context_type} | {row.key} | {row.count} | {row.distribution} | {fmt(row.entropy)} | "
            f"{fmt(row.normalized_entropy)} | {pct(row.outcome.cont_rate)} | {pct(row.outcome.fail_rate)} | "
            f"{pct(row.outcome.skew)} | {fmt(row.outcome.mean_dr)} | {fmt(row.outcome.median_dr)}"
        )
    lines += [
        "",
        "5. Outcome layer",
        "ContextType | ContextKey | Count | MeanDRFwd5 | MedianDRFwd5 | ContRate5 | FailRate5 | FlatRate5 | OutcomeSkew",
    ]
    for (kind, key), outcome in sorted(result.outcomes.items()):
        if kind != "Baseline" and outcome.count < MIN_INSTRUMENT_CONTEXT:
            continue
        lines.append(
            f"{kind} | {key} | {outcome.count} | {fmt(outcome.mean_dr)} | {fmt(outcome.median_dr)} | "
            f"{pct(outcome.cont_rate)} | {pct(outcome.fail_rate)} | {pct(outcome.flat_rate)} | {pct(outcome.skew)}"
        )
    lines += [
        "",
        "6. Context concentration",
        "CurrentState | AgeBucket | ContextType | TopContext | Count | Share | DominantNextState | DominantNextProbability | OutcomeSkew",
    ]
    for row in sorted(result.concentration_rows, key=lambda x: (x.state, age_sort(x.age_bucket), x.context_type)):
        lines.append(
            f"{row.state} | {row.age_bucket} | {row.context_type} | {row.top_context} | {row.count} | "
            f"{pct(row.share)} | {row.dominant_next} | {pct(row.dominant_probability)} | {pct(row.outcome_skew)}"
        )
    lines += [
        "",
        "7. Population B",
        "Population B skipped gracefully: the reused structural-state loader does not expose breakout type, maturity stage, or alignment membership.",
        "",
        "8. Mechanical research notes",
        "- Positive EntropyReduction means prior structural context reduces next-state uncertainty.",
        "- ContextDistinct tests the most recent structurally different state rather than immediate persistence.",
        "- ContextPath3 and ContextPath5 test whether short structural memory adds transition information.",
        "- CompressionProcessing Age 3 is reported separately to expose mixed prior pathways.",
        "- Outcome fields are reported after classification and are not used to define states or contexts.",
        "- Review low-count rows cautiously.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def instrument_columns(instruments: Sequence[str], getter) -> str:
    parts: List[str] = []
    for inst in instruments:
        parts.extend(getter(inst))
    return " | ".join(parts)


def aggregate_outcome(results: Sequence[StudyResult], kind: str, key: str) -> Outcome:
    outcome = Outcome()
    for result in results:
        outcome.values.extend(result.outcomes.get((kind, key), Outcome()).values)
    return outcome


def aggregate_report(results: Sequence[StudyResult], out_root: str) -> str:
    instruments = [r.instrument for r in results]
    by_inst = {r.instrument: r for r in results}
    out_dir = os.path.join(out_root, "TransitionContext")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "TransitionContext_All.txt")
    lines = [
        "APVA Transition Context Matrix Study v0.1 - Aggregate",
        "Research only. No trades, entries/exits, optimization, fitting, or machine learning.",
        f"Instruments: {', '.join(instruments)}",
        "",
        "Aggregate Baseline Table",
    ]
    baseline_keys = sorted(
        {(x.state, x.age_bucket, x.horizon, x.next_state) for r in results for x in r.baseline},
        key=lambda x: (x[0], age_sort(x[1]), x[2], x[3]),
    )
    header = "CurrentState | AgeBucket | Horizon | NextState | " + instrument_columns(
        instruments, lambda inst: [f"Count_{inst}", f"Prob_{inst}", f"Lift_{inst}"]
    ) + " | ValidInstrumentCount | MeanProbability | MeanLift"
    lines.append(header)
    for key in baseline_keys:
        values = {}
        for inst, result in by_inst.items():
            values[inst] = next((x for x in result.baseline if (x.state, x.age_bucket, x.horizon, x.next_state) == key), None)
        valid = [x for x in values.values() if x and x.source_count >= MIN_INSTRUMENT_CONTEXT]
        cols = instrument_columns(
            instruments,
            lambda inst: [
                str(values[inst].count) if values[inst] else "0",
                pct(values[inst].probability) if values[inst] else "N/A",
                fmt(values[inst].lift) if values[inst] else "N/A",
            ],
        )
        lines.append(
            f"{key[0]} | {key[1]} | t+{key[2]} | {key[3]} | {cols} | {len(valid)} | "
            f"{pct(mean(x.probability for x in valid)) if valid else 'N/A'} | {fmt(mean(x.lift for x in valid if x.lift is not None)) if valid else 'N/A'}"
        )

    lines += ["", "Aggregate Context Table"]
    lines.append(
        "ContextType | ContextKey | Horizon | NextState | "
        + instrument_columns(instruments, lambda inst: [f"Count_{inst}", f"Prob_{inst}", f"Lift_{inst}", f"Entropy_{inst}", f"Skew_{inst}"])
        + " | ValidInstrumentCount | MeanProbability | MeanLift | MeanEntropy | MeanSkew"
    )
    context_keys = sorted({(x.kind, x.key, x.horizon, x.next_state) for r in results for x in r.contexts})
    aggregate_context_rows = []
    for key in context_keys:
        values = {
            inst: next((x for x in by_inst[inst].contexts if (x.kind, x.key, x.horizon, x.next_state) == key), None)
            for inst in instruments
        }
        total_source = sum(x.source_count for x in values.values() if x)
        if total_source < MIN_AGGREGATE_CONTEXT:
            continue
        valid = [x for x in values.values() if x and x.source_count >= MIN_INSTRUMENT_CONTEXT]
        skews = {
            inst: by_inst[inst].outcomes.get((key[0], key[1]), Outcome()).skew
            for inst in instruments
        }
        cols = instrument_columns(
            instruments,
            lambda inst: [
                str(values[inst].count) if values[inst] else "0",
                pct(values[inst].probability) if values[inst] else "N/A",
                fmt(values[inst].lift) if values[inst] else "N/A",
                fmt(values[inst].entropy) if values[inst] else "N/A",
                pct(skews[inst]) if values[inst] else "N/A",
            ],
        )
        mean_prob = mean(x.probability for x in valid)
        mean_lift = mean(x.lift for x in valid if x.lift is not None)
        mean_ent = mean(x.entropy for x in valid)
        mean_skew = mean(skews[inst] for inst, x in values.items() if x and x.source_count >= MIN_INSTRUMENT_CONTEXT)
        aggregate_context_rows.append((key, len(valid), mean_prob, mean_lift, mean_ent, mean_skew))
        lines.append(
            f"{key[0]} | {key[1]} | t+{key[2]} | {key[3]} | {cols} | {len(valid)} | "
            f"{pct(mean_prob) if valid else 'N/A'} | {fmt(mean_lift) if valid else 'N/A'} | "
            f"{fmt(mean_ent) if valid else 'N/A'} | {pct(mean_skew) if valid else 'N/A'}"
        )

    lines += ["", "Aggregate Entropy Comparison Table"]
    lines.append(
        "ContextType | CurrentState | AgeBucket | Horizon | "
        + instrument_columns(instruments, lambda inst: [f"BaselineEntropy_{inst}", f"ContextEntropy_{inst}", f"EntropyReduction_{inst}"])
        + " | ValidInstrumentCount | MeanBaselineEntropy | MeanContextEntropy | MeanEntropyReduction"
    )
    entropy_keys = sorted(
        {(x.context_type, x.state, x.age_bucket, x.horizon) for r in results for x in r.entropy_rows},
        key=lambda x: (x[0], x[1], age_sort(x[2]), x[3]),
    )
    aggregate_entropy_rows = []
    for key in entropy_keys:
        values = {
            inst: next((x for x in by_inst[inst].entropy_rows if (x.context_type, x.state, x.age_bucket, x.horizon) == key), None)
            for inst in instruments
        }
        valid = [x for x in values.values() if x and x.count >= MIN_INSTRUMENT_CONTEXT]
        cols = instrument_columns(
            instruments,
            lambda inst: [
                fmt(values[inst].baseline_entropy) if values[inst] else "N/A",
                fmt(values[inst].context_entropy) if values[inst] else "N/A",
                fmt(values[inst].reduction) if values[inst] else "N/A",
            ],
        )
        aggregate_entropy_rows.append((key, len(valid), mean(x.reduction for x in valid)))
        lines.append(
            f"{key[0]} | {key[1]} | {key[2]} | t+{key[3]} | {cols} | {len(valid)} | "
            f"{fmt(mean(x.baseline_entropy for x in valid)) if valid else 'N/A'} | "
            f"{fmt(mean(x.context_entropy for x in valid)) if valid else 'N/A'} | "
            f"{fmt(mean(x.reduction for x in valid)) if valid else 'N/A'}"
        )

    lines += ["", "Aggregate Compression Age-3 Table"]
    lines.append(
        "ContextType | ContextKey | "
        + instrument_columns(instruments, lambda inst: [f"Count_{inst}", f"Cont_{inst}", f"Fail_{inst}", f"Skew_{inst}", f"MeanDR_{inst}"])
        + " | ValidInstrumentCount | MeanContinuation | MeanFailure | MeanSkew | MeanDR"
    )
    focus_keys = sorted({(x.context_type, x.key) for r in results for x in r.focus_rows})
    aggregate_focus_rows = []
    for key in focus_keys:
        values = {
            inst: next((x for x in by_inst[inst].focus_rows if (x.context_type, x.key) == key), None)
            for inst in instruments
        }
        valid = [x for x in values.values() if x and x.count >= MIN_INSTRUMENT_CONTEXT]
        cols = instrument_columns(
            instruments,
            lambda inst: [
                str(values[inst].count) if values[inst] else "0",
                pct(values[inst].outcome.cont_rate) if values[inst] else "N/A",
                pct(values[inst].outcome.fail_rate) if values[inst] else "N/A",
                pct(values[inst].outcome.skew) if values[inst] else "N/A",
                fmt(values[inst].outcome.mean_dr) if values[inst] else "N/A",
            ],
        )
        aggregate_focus_rows.append((key, len(valid), mean(x.outcome.skew for x in valid)))
        lines.append(
            f"{key[0]} | {key[1]} | {cols} | {len(valid)} | "
            f"{pct(mean(x.outcome.cont_rate for x in valid)) if valid else 'N/A'} | "
            f"{pct(mean(x.outcome.fail_rate for x in valid)) if valid else 'N/A'} | "
            f"{pct(mean(x.outcome.skew for x in valid)) if valid else 'N/A'} | "
            f"{fmt(mean(x.outcome.mean_dr for x in valid)) if valid else 'N/A'}"
        )

    lines += ["", "Aggregate Context Concentration Table"]
    lines.append(
        "CurrentState | AgeBucket | ContextType | TopContext | "
        + instrument_columns(instruments, lambda inst: [f"Share_{inst}"])
        + " | ValidInstrumentCount | MeanShare | DominantNextState | MeanDominantProbability"
    )
    concentration_rows = []
    concentration_keys = sorted(
        {(x.state, x.age_bucket, x.context_type) for r in results for x in r.concentration_rows},
        key=lambda x: (x[0], age_sort(x[1]), x[2]),
    )
    for key in concentration_keys:
        all_rows = [
            x for r in results for x in r.concentration_rows
            if (x.state, x.age_bucket, x.context_type) == key
        ]
        top_counts: Counter[str] = Counter()
        for row in all_rows:
            top_counts[row.top_context] += row.count
        top_context = top_counts.most_common(1)[0][0]
        values = {
            inst: next((x for x in by_inst[inst].concentration_rows if (x.state, x.age_bucket, x.context_type) == key), None)
            for inst in instruments
        }
        valid = [x for x in values.values() if x and x.total >= MIN_INSTRUMENT_CONTEXT]
        dominant = Counter(x.dominant_next for x in valid).most_common(1)
        dominant_next = dominant[0][0] if dominant else "N/A"
        cols = instrument_columns(instruments, lambda inst: [pct(values[inst].share) if values[inst] else "N/A"])
        concentration_rows.append((key, top_context, len(valid), mean(x.share for x in valid)))
        lines.append(
            f"{key[0]} | {key[1]} | {key[2]} | {top_context} | {cols} | {len(valid)} | "
            f"{pct(mean(x.share for x in valid)) if valid else 'N/A'} | {dominant_next} | "
            f"{pct(mean(x.dominant_probability for x in valid)) if valid else 'N/A'}"
        )

    def ranking(title: str, rows: Iterable[str]) -> None:
        lines.extend(["", title])
        materialized = list(rows)[:TOP_LIMIT]
        lines.extend(materialized or ["No replicated rows met the minimum count threshold."])

    valid_entropy = [x for x in aggregate_entropy_rows if x[1] >= MIN_VALID_INSTRUMENTS]
    ranking(
        "1. Contexts with strongest entropy reduction",
        (f"{x[0][0]} | {x[0][1]} | {x[0][2]} | t+{x[0][3]} | MeanEntropyReduction={fmt(x[2])}" for x in sorted(valid_entropy, key=lambda x: -x[2])),
    )
    ranking(
        "2. Contexts with worst entropy loss",
        (f"{x[0][0]} | {x[0][1]} | {x[0][2]} | t+{x[0][3]} | MeanEntropyReduction={fmt(x[2])}" for x in sorted(valid_entropy, key=lambda x: x[2])),
    )
    for number, kind, title in (
        (3, "ContextDistinct", "Strongest replicated ContextDistinct transitions"),
        (4, "ContextPath3", "Strongest replicated ContextPath3 transitions"),
        (5, "ContextPath5", "Strongest replicated ContextPath5 transitions"),
    ):
        subset = [x for x in aggregate_context_rows if x[0][0] == kind and x[1] >= MIN_VALID_INSTRUMENTS and x[3] > 1.5 and x[2] >= 0.10]
        ranking(
            f"{number}. {title}",
            (f"{x[0][1]} | t+{x[0][2]} -> {x[0][3]} | MeanProbability={pct(x[2])} | MeanLift={fmt(x[3])}" for x in sorted(subset, key=lambda x: (-x[3], -x[2]))),
        )
    valid_focus = [x for x in aggregate_focus_rows if x[1] >= MIN_VALID_INSTRUMENTS]
    ranking("6. Best Compression Age-3 contexts", (f"{x[0][0]} | {x[0][1]} | MeanOutcomeSkew={pct(x[2])}" for x in sorted(valid_focus, key=lambda x: -x[2])))
    ranking("7. Worst Compression Age-3 contexts", (f"{x[0][0]} | {x[0][1]} | MeanOutcomeSkew={pct(x[2])}" for x in sorted(valid_focus, key=lambda x: x[2])))
    outcome_rows = []
    for kind in CONTEXT_TYPES:
        keys = {key for r in results for (row_kind, key), value in r.outcomes.items() if row_kind == kind and value.count >= MIN_INSTRUMENT_CONTEXT}
        for key in keys:
            values = [r.outcomes[(kind, key)] for r in results if r.outcomes.get((kind, key), Outcome()).count >= MIN_INSTRUMENT_CONTEXT]
            total = sum(x.count for x in values)
            if len(values) >= MIN_VALID_INSTRUMENTS and total >= MIN_AGGREGATE_CONTEXT:
                outcome_rows.append((kind, key, mean(x.skew for x in values)))
    ranking("8. Contexts with best outcome skew", (f"{x[0]} | {x[1]} | MeanOutcomeSkew={pct(x[2])}" for x in sorted(outcome_rows, key=lambda x: -x[2])))
    ranking("9. Contexts with worst outcome skew", (f"{x[0]} | {x[1]} | MeanOutcomeSkew={pct(x[2])}" for x in sorted(outcome_rows, key=lambda x: x[2])))
    valid_concentration = [x for x in concentration_rows if x[2] >= MIN_VALID_INSTRUMENTS]
    ranking("10. Current states most dominated by one prior context", (f"{x[0][0]} | {x[0][1]} | {x[0][2]} | {x[1]} | MeanShare={pct(x[3])}" for x in sorted(valid_concentration, key=lambda x: -x[3])))
    ranking("11. Population-B context differences", ["Population B skipped gracefully: the reused structural-state loader does not expose breakout type, maturity stage, or alignment membership."])
    lines += [
        "",
        "Cross-instrument mechanical research notes",
        "- Context entropy is a weighted conditional entropy inside each current state-age bucket.",
        "- Positive entropy reduction indicates that prior state history compresses next-state uncertainty.",
        "- Strong replicated context transitions require at least two valid instruments, MeanLift > 1.5, and MeanProbability >= 10%.",
        "- CompressionProcessing Age 3 contexts are separated explicitly for paradox inspection.",
        "- Outcome fields are downstream diagnostics only.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="APVA Transition Context Matrix Study v0.1")
    parser.add_argument("inputs", nargs="+", help="Evidence CSV files or directories")
    parser.add_argument("--out-root", default="Evidence/Output", help="Output root directory")
    args = parser.parse_args(argv)
    loaded = load_results(args.inputs)
    if not loaded:
        raise SystemExit("No input rows loaded.")
    results = [build_study_result(result) for result in loaded]
    for result in results:
        write_instrument_report(result, args.out_root)
    aggregate = aggregate_report(results, args.out_root)
    print(f"Wrote TransitionContext reports under {args.out_root}")
    print(f"Aggregate: {aggregate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
