"""APVA State Age Analysis Study v0.1.

Research-only analysis of structural-state age, persistence, exits, outcomes,
and evidence composition. Structural states are imported from Study 39.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from APVA_StructuralStateMachine_39 import (
    Bar,
    STRUCTURAL_STATES,
    directional_return,
    fmt,
    instrument_from_path,
    mean,
    median,
    outcome_stats,
    pct,
    read_bars,
)


AGGREGATE_OUTPUT = Path("Evidence/Output/StateAge/StateAge_All.txt")
AGE_BUCKETS = ("1", "2", "3", "4", "5", "6-10", "11-20", "21+")
HORIZONS = (1, 3, 5)
FLAG_NAMES = ("Accepted", "Compressed", "Dissipating", "Expanding", "Peak", "Climactic")
STATE_FIELDS = ("ParticipationState", "ExpansionState", "CompressionState", "DissipationState", "AcceptanceState")
VALID_MIN = 20
TOP_LIMIT = 25


@dataclass(frozen=True)
class AgeObservation:
    index: int
    state: str
    age: int
    age_bucket: str
    run_length: int
    position: str
    population_b: bool


@dataclass(frozen=True)
class TransitionStats:
    state: str
    age_bucket: str
    next_state: str
    horizon: int
    count: int
    probability: float
    lift: float | None


@dataclass(frozen=True)
class CurveStats:
    state: str
    age_bucket: str
    count: int
    persistence: dict[int, float]
    exits: dict[int, float]
    outcome_count: int
    mean_dr: float
    median_dr: float
    continuation: float
    failure: float
    flat: float
    skew: float


@dataclass(frozen=True)
class YoungOldStats:
    state: str
    young_count: int
    old_count: int
    young_continuation: float
    old_continuation: float
    young_failure: float
    old_failure: float
    young_exit: float
    old_exit: float
    young_persistence: float
    old_persistence: float
    young_skew: float
    old_skew: float
    classification: str


@dataclass(frozen=True)
class ConditionStats:
    state: str
    age_bucket: str
    count: int
    flag_rates: dict[str, float]
    distributions: dict[str, dict[str, float]]


@dataclass(frozen=True)
class PopulationStudy:
    name: str
    observations: list[AgeObservation]
    transitions: list[TransitionStats]
    curves: dict[tuple[str, str], CurveStats]
    young_old: dict[str, YoungOldStats]
    conditions: dict[tuple[str, str], ConditionStats]


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    bars: list[Bar]
    observations: list[AgeObservation]
    run_counts: dict[str, int]
    populations: dict[str, PopulationStudy]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study APVA structural-state age using the Study 39 structural-state model.")
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def age_bucket(age: int) -> str:
    if age <= 5:
        return str(age)
    if age <= 10:
        return "6-10"
    if age <= 20:
        return "11-20"
    return "21+"


def state_position(age: int, run_length: int) -> str:
    if run_length == 1:
        return "Only"
    if age == 1:
        return "First"
    if age == run_length:
        return "Last"
    if age <= math.ceil(run_length * 0.33):
        return "Early"
    if age >= math.ceil(run_length * 0.67):
        return "Late"
    return "Middle"


def annotate_runs(bars: list[Bar]) -> tuple[list[AgeObservation], dict[str, int]]:
    output = []
    run_counts = Counter()
    cursor = 0
    while cursor < len(bars):
        end = cursor
        while end + 1 < len(bars) and bars[end + 1].structural_state == bars[cursor].structural_state:
            end += 1
        run_length = end - cursor + 1
        state = bars[cursor].structural_state
        run_counts[state] += 1
        for index in range(cursor, end + 1):
            age = index - cursor + 1
            output.append(AgeObservation(index, state, age, age_bucket(age), run_length, state_position(age, run_length), bars[index].population_b))
        cursor = end + 1
    return output, dict(run_counts)


def selected_observations(observations: list[AgeObservation], population_b_only: bool) -> list[AgeObservation]:
    return [item for item in observations if not population_b_only or item.population_b]


def group_observations(observations: Iterable[AgeObservation]) -> dict[tuple[str, str], list[AgeObservation]]:
    output = defaultdict(list)
    for item in observations:
        output[(item.state, item.age_bucket)].append(item)
    return output


def build_transitions(bars: list[Bar], observations: list[AgeObservation]) -> list[TransitionStats]:
    output = []
    grouped = group_observations(observations)
    for horizon in HORIZONS:
        eligible = [item for item in observations if item.index + horizon < len(bars)]
        next_counts = Counter(bars[item.index + horizon].structural_state for item in eligible)
        total = len(eligible)
        for (state, bucket), items in grouped.items():
            current = [item for item in items if item.index + horizon < len(bars)]
            next_by_state = Counter(bars[item.index + horizon].structural_state for item in current)
            for next_state in STRUCTURAL_STATES:
                count = next_by_state[next_state]
                probability = count / len(current) if current else 0.0
                baseline = next_counts[next_state] / total if total else 0.0
                output.append(TransitionStats(state, bucket, next_state, horizon, count, probability, probability / baseline if baseline else None))
    return output


def transition_lookup(transitions: list[TransitionStats]) -> dict[tuple[str, str, int, str], TransitionStats]:
    return {(item.state, item.age_bucket, item.horizon, item.next_state): item for item in transitions}


def build_curves(bars: list[Bar], observations: list[AgeObservation], transitions: list[TransitionStats]) -> dict[tuple[str, str], CurveStats]:
    lookup = transition_lookup(transitions)
    output = {}
    for (state, bucket), items in group_observations(observations).items():
        outcomes = outcome_stats(bars, [item.index for item in items], 5)
        persistence = {horizon: lookup[(state, bucket, horizon, state)].probability for horizon in HORIZONS}
        exits = {horizon: 1.0 - persistence[horizon] for horizon in HORIZONS}
        output[(state, bucket)] = CurveStats(state, bucket, len(items), persistence, exits, outcomes.count, outcomes.mean_dr, outcomes.median_dr, outcomes.continuation_rate, outcomes.failure_rate, outcomes.flat_rate, outcomes.continuation_rate - outcomes.failure_rate)
    return output


def indexes_for_buckets(observations: list[AgeObservation], state: str, buckets: set[str]) -> list[int]:
    return [item.index for item in observations if item.state == state and item.age_bucket in buckets]


def one_bar_persistence(bars: list[Bar], indexes: list[int], state: str) -> float:
    eligible = [index for index in indexes if index + 1 < len(bars)]
    return sum(bars[index + 1].structural_state == state for index in eligible) / len(eligible) if eligible else 0.0


def classify(delta_persistence: float, delta_exit: float, delta_skew: float) -> str:
    labels = []
    if delta_persistence > 0 and delta_exit < 0:
        labels.append("Matures")
    if delta_persistence < 0 and delta_exit > 0:
        labels.append("Exhausts")
    if delta_skew > 0:
        labels.append("OutcomeImproves")
    if delta_skew < 0:
        labels.append("OutcomeDegrades")
    return "+".join(labels) if labels else "Mixed"


def build_young_old(bars: list[Bar], observations: list[AgeObservation]) -> dict[str, YoungOldStats]:
    output = {}
    for state in STRUCTURAL_STATES:
        young = indexes_for_buckets(observations, state, {"1", "2", "3"})
        old = indexes_for_buckets(observations, state, {"6-10", "11-20", "21+"})
        young_outcome = outcome_stats(bars, young, 5)
        old_outcome = outcome_stats(bars, old, 5)
        young_persistence = one_bar_persistence(bars, young, state)
        old_persistence = one_bar_persistence(bars, old, state)
        young_exit = 1.0 - young_persistence
        old_exit = 1.0 - old_persistence
        young_skew = young_outcome.continuation_rate - young_outcome.failure_rate
        old_skew = old_outcome.continuation_rate - old_outcome.failure_rate
        output[state] = YoungOldStats(state, len(young), len(old), young_outcome.continuation_rate, old_outcome.continuation_rate, young_outcome.failure_rate, old_outcome.failure_rate, young_exit, old_exit, young_persistence, old_persistence, young_skew, old_skew, classify(old_persistence - young_persistence, old_exit - young_exit, old_skew - young_skew))
    return output


def normalize_state(value: str) -> str:
    return value.strip() or "Other"


def build_conditions(bars: list[Bar], observations: list[AgeObservation]) -> dict[tuple[str, str], ConditionStats]:
    output = {}
    for key, items in group_observations(observations).items():
        selected = [bars[item.index] for item in items]
        denominator = len(selected) or 1
        flag_rates = {name: sum(bar.flags[name] for bar in selected) / denominator for name in FLAG_NAMES}
        distributions = {}
        for field in STATE_FIELDS:
            counts = Counter(normalize_state(bar.states.get(field, "")) for bar in selected)
            distributions[field] = {name: count / denominator for name, count in sorted(counts.items())}
        output[key] = ConditionStats(key[0], key[1], len(items), flag_rates, distributions)
    return output


def build_population(bars: list[Bar], observations: list[AgeObservation], name: str, population_b_only: bool) -> PopulationStudy:
    selected = selected_observations(observations, population_b_only)
    transitions = build_transitions(bars, selected)
    return PopulationStudy(name, selected, transitions, build_curves(bars, selected, transitions), build_young_old(bars, selected), build_conditions(bars, selected))


def study_instrument(path: Path) -> InstrumentStudy:
    bars = read_bars(str(path))
    observations, run_counts = annotate_runs(bars)
    instrument = bars[0].instrument if bars else instrument_from_path(str(path))
    populations = {
        "Full Population": build_population(bars, observations, "Full Population", False),
        "Population B": build_population(bars, observations, "Population B", True),
    }
    return InstrumentStudy(instrument, path, bars, observations, run_counts, populations)


def append_heading(lines: list[str], title: str) -> None:
    lines.extend(["", title, "=" * len(title)])


def append_state_counts(lines: list[str], study: InstrumentStudy) -> None:
    counts = Counter(item.state for item in study.observations)
    lines.append(f"{'State':<28} {'Count':>8} {'Frequency':>10} {'Runs':>8}")
    for state in STRUCTURAL_STATES:
        lines.append(f"{state:<28} {counts[state]:>8} {pct(counts[state] / len(study.observations) if study.observations else 0.0):>10} {study.run_counts.get(state, 0):>8}")


def append_run_summary(lines: list[str], observations: list[AgeObservation]) -> None:
    grouped = defaultdict(list)
    seen = set()
    for item in observations:
        key = (item.state, item.index - item.age + 1)
        if key not in seen:
            grouped[item.state].append(item.run_length)
            seen.add(key)
    lines.append(f"{'State':<28} {'RunCount':>8} {'MeanLen':>10} {'MedianLen':>10} {'MaxLen':>8}")
    for state in STRUCTURAL_STATES:
        values = grouped[state]
        lines.append(f"{state:<28} {len(values):>8} {mean(values):>10.3f} {median(values):>10.3f} {max(values, default=0):>8}")


def append_bucket_distribution(lines: list[str], population: PopulationStudy) -> None:
    counts = Counter((item.state, item.age_bucket) for item in population.observations)
    lines.append(f"{'State':<28} {'AgeBucket':<8} {'Count':>8}")
    for state in STRUCTURAL_STATES:
        for bucket in AGE_BUCKETS:
            lines.append(f"{state:<28} {bucket:<8} {counts[(state, bucket)]:>8}")


def append_transitions(lines: list[str], transitions: list[TransitionStats]) -> None:
    lines.append(f"{'State':<28} {'Age':<8} {'NextState':<28} {'H':>2} {'Count':>8} {'Probability':>12} {'Lift':>10}")
    for item in transitions:
        if item.count:
            lines.append(f"{item.state:<28} {item.age_bucket:<8} {item.next_state:<28} {item.horizon:>2} {item.count:>8} {pct(item.probability):>12} {fmt(item.lift, 3):>10}")


def append_curves(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'State':<28} {'Age':<8} {'Count':>8} {'Persist1':>9} {'Exit1':>9} {'Persist3':>9} {'Exit3':>9} {'Persist5':>9} {'Exit5':>9} {'Cont5':>9} {'Fail5':>9} {'Skew':>9}")
    for state in STRUCTURAL_STATES:
        for bucket in AGE_BUCKETS:
            item = population.curves.get((state, bucket))
            if item:
                lines.append(f"{state:<28} {bucket:<8} {item.count:>8} {pct(item.persistence[1]):>9} {pct(item.exits[1]):>9} {pct(item.persistence[3]):>9} {pct(item.exits[3]):>9} {pct(item.persistence[5]):>9} {pct(item.exits[5]):>9} {pct(item.continuation):>9} {pct(item.failure):>9} {pct(item.skew):>9}")


def append_outcomes(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'State':<28} {'Age':<8} {'Count':>8} {'MeanDR':>11} {'MedianDR':>11} {'Cont5':>9} {'Fail5':>9} {'Flat5':>9} {'Skew':>9}")
    for state in STRUCTURAL_STATES:
        for bucket in AGE_BUCKETS:
            item = population.curves.get((state, bucket))
            if item:
                lines.append(f"{state:<28} {bucket:<8} {item.outcome_count:>8} {item.mean_dr:>11.5f} {item.median_dr:>11.5f} {pct(item.continuation):>9} {pct(item.failure):>9} {pct(item.flat):>9} {pct(item.skew):>9}")


def append_young_old(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'State':<28} {'YoungN':>7} {'OldN':>7} {'YoungCont':>10} {'OldCont':>10} {'dPersist':>10} {'dExit':>10} {'dSkew':>10} {'Classification'}")
    for item in population.young_old.values():
        lines.append(f"{item.state:<28} {item.young_count:>7} {item.old_count:>7} {pct(item.young_continuation):>10} {pct(item.old_continuation):>10} {pct(item.old_persistence - item.young_persistence):>10} {pct(item.old_exit - item.young_exit):>10} {pct(item.old_skew - item.young_skew):>10} {item.classification}")


def append_conditions(lines: list[str], population: PopulationStudy) -> None:
    lines.append(f"{'State':<28} {'Age':<8} {'Count':>8}" + "".join(f" {name[:8]:>9}" for name in FLAG_NAMES))
    for state in STRUCTURAL_STATES:
        for bucket in AGE_BUCKETS:
            item = population.conditions.get((state, bucket))
            if item:
                lines.append(f"{state:<28} {bucket:<8} {item.count:>8}" + "".join(f" {pct(item.flag_rates[name]):>9}" for name in FLAG_NAMES))
    lines.append("\nCategorical state distributions")
    lines.append(f"{'State':<28} {'Age':<8} {'Field':<20} Distribution")
    for state in STRUCTURAL_STATES:
        for bucket in AGE_BUCKETS:
            item = population.conditions.get((state, bucket))
            if item:
                for field in STATE_FIELDS:
                    values = ", ".join(f"{name}={pct(rate)}" for name, rate in item.distributions[field].items())
                    lines.append(f"{state:<28} {bucket:<8} {field:<20} {values}")


def instrument_report(study: InstrumentStudy) -> str:
    full = study.populations["Full Population"]
    population_b = study.populations["Population B"]
    lines = [
        f"APVA State Age Analysis Study v0.1 - {study.instrument}",
        "=" * (37 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {len(study.bars)}",
        f"Structural observations: {len(study.observations)}",
        f"Structural state counts: {dict(Counter(item.state for item in study.observations))}",
        f"Structural state run counts: {study.run_counts}",
    ]
    append_heading(lines, "Section 1 - State Count Table")
    append_state_counts(lines, study)
    append_heading(lines, "Section 2 - State Run Summary")
    append_run_summary(lines, study.observations)
    append_heading(lines, "Section 3 - Age Bucket Distribution")
    append_bucket_distribution(lines, full)
    append_heading(lines, "Section 4 - State Age Transition Table")
    append_transitions(lines, full.transitions)
    append_heading(lines, "Section 5 - State Age Outcome Table")
    append_outcomes(lines, full)
    append_heading(lines, "Section 6 - Aging Curves")
    append_curves(lines, full)
    append_heading(lines, "Section 7 - Young-vs-Old Maturation / Exhaustion")
    append_young_old(lines, full)
    append_heading(lines, "Section 8 - Age-Condition Interaction")
    append_conditions(lines, full)
    append_heading(lines, "Section 9 - Population B")
    lines.append(f"Population B observations: {len(population_b.observations)}")
    append_bucket_distribution(lines, population_b)
    append_curves(lines, population_b)
    append_young_old(lines, population_b)
    append_outcomes(lines, population_b)
    append_heading(lines, "Section 10 - Mechanical Research Notes")
    strongest = max(full.young_old.values(), key=lambda item: item.old_persistence - item.young_persistence)
    weakest = min(full.young_old.values(), key=lambda item: item.old_persistence - item.young_persistence)
    lines.extend([
        f"- Strongest old-minus-young persistence increase: {strongest.state} ({pct(strongest.old_persistence - strongest.young_persistence)}).",
        f"- Strongest old-minus-young persistence decay: {weakest.state} ({pct(weakest.old_persistence - weakest.young_persistence)}).",
        "- Structural states, families, paths, and Population B flags are inherited from Study 39.",
        "- Age curves and classifications are descriptive only.",
    ])
    return "\n".join(lines) + "\n"


def instrument_columns(studies: list[InstrumentStudy]) -> list[str]:
    preferred = ["6E", "NQ", "CL"]
    available = [study.instrument for study in studies]
    return preferred + sorted(set(available) - set(preferred))


def append_aggregate_age(lines: list[str], studies: list[InstrumentStudy], columns: list[str]) -> None:
    lines.append(f"{'State':<28} {'Age':<8}" + "".join(f" {name + '_N':>7} {name + '_P1':>8} {name + '_X1':>8} {name + '_Cont':>8} {name + '_Fail':>8} {name + '_Skew':>8}" for name in columns) + " ValidN MeanP1 MeanX1 MeanCont MeanFail MeanSkew")
    for state in STRUCTURAL_STATES:
        for bucket in AGE_BUCKETS:
            values = {study.instrument: study.populations["Full Population"].curves.get((state, bucket)) for study in studies}
            valid = [item for item in values.values() if item and item.count >= VALID_MIN]
            cells = ""
            for name in columns:
                item = values.get(name)
                cells += f" {item.count if item else 0:>7} {item.persistence[1] if item else 0.0:>8.2%} {item.exits[1] if item else 0.0:>8.2%} {item.continuation if item else 0.0:>8.2%} {item.failure if item else 0.0:>8.2%} {item.skew if item else 0.0:>8.2%}"
            lines.append(f"{state:<28} {bucket:<8}{cells} {len(valid):>6} {mean([item.persistence[1] for item in valid]):>6.2%} {mean([item.exits[1] for item in valid]):>6.2%} {mean([item.continuation for item in valid]):>8.2%} {mean([item.failure for item in valid]):>8.2%} {mean([item.skew for item in valid]):>8.2%}")


def aggregate_young_old_rows(studies: list[InstrumentStudy]) -> list[dict]:
    output = []
    for state in STRUCTURAL_STATES:
        values = {study.instrument: study.populations["Full Population"].young_old[state] for study in studies}
        valid = [item for item in values.values() if item.young_count >= VALID_MIN and item.old_count >= VALID_MIN]
        persist_deltas = [item.old_persistence - item.young_persistence for item in valid]
        exit_deltas = [item.old_exit - item.young_exit for item in valid]
        skew_deltas = [item.old_skew - item.young_skew for item in valid]
        persistence_agreement = len(valid) >= 2 and (all(value > 0 for value in persist_deltas) or all(value < 0 for value in persist_deltas))
        outcome_agreement = len(valid) >= 2 and (all(value > 0 for value in skew_deltas) or all(value < 0 for value in skew_deltas))
        output.append({"state": state, "values": values, "valid": len(valid), "persist_agree": persistence_agreement, "outcome_agree": outcome_agreement, "persist": mean(persist_deltas), "exit": mean(exit_deltas), "skew": mean(skew_deltas), "classification": classify(mean(persist_deltas), mean(exit_deltas), mean(skew_deltas))})
    return output


def append_aggregate_young_old(lines: list[str], rows: list[dict], columns: list[str]) -> None:
    lines.append(f"{'State':<28}" + "".join(f" {name + '_YN':>7} {name + '_ON':>7} {name + '_dP':>8} {name + '_dX':>8} {name + '_dS':>8}" for name in columns) + " ValidN PAgree OAgree Mean_dP Mean_dX Mean_dS Classification")
    for row in rows:
        cells = ""
        for name in columns:
            item = row["values"].get(name)
            cells += f" {item.young_count if item else 0:>7} {item.old_count if item else 0:>7} {item.old_persistence - item.young_persistence if item else 0.0:>8.2%} {item.old_exit - item.young_exit if item else 0.0:>8.2%} {item.old_skew - item.young_skew if item else 0.0:>8.2%}"
        lines.append(f"{row['state']:<28}{cells} {row['valid']:>6} {str(row['persist_agree']):>6} {str(row['outcome_agree']):>6} {row['persist']:>7.2%} {row['exit']:>7.2%} {row['skew']:>7.2%} {row['classification']}")


def aggregate_transition_rows(studies: list[InstrumentStudy]) -> list[dict]:
    output = []
    for state in STRUCTURAL_STATES:
        for bucket in AGE_BUCKETS:
            for next_state in STRUCTURAL_STATES:
                for horizon in HORIZONS:
                    values = {}
                    for study in studies:
                        mapping = transition_lookup(study.populations["Full Population"].transitions)
                        values[study.instrument] = mapping.get((state, bucket, horizon, next_state))
                    valid = [item for item in values.values() if item and item.count >= VALID_MIN]
                    output.append({"state": state, "bucket": bucket, "next": next_state, "horizon": horizon, "values": values, "valid": len(valid), "probability": mean([item.probability for item in valid]), "lift": mean([item.lift for item in valid if item.lift is not None])})
    return output


def append_aggregate_transitions(lines: list[str], rows: list[dict], columns: list[str]) -> None:
    lines.append(f"{'State':<28} {'Age':<8} {'NextState':<28} {'H':>2}" + "".join(f" {name + '_N':>7} {name + '_Prob':>9} {name + '_Lift':>8}" for name in columns) + " ValidN MeanProb MeanLift")
    for row in rows:
        if not any(item and item.count for item in row["values"].values()):
            continue
        cells = ""
        for name in columns:
            item = row["values"].get(name)
            cells += f" {item.count if item else 0:>7} {item.probability if item else 0.0:>9.2%} {fmt(item.lift if item else None, 3):>8}"
        lines.append(f"{row['state']:<28} {row['bucket']:<8} {row['next']:<28} {row['horizon']:>2}{cells} {row['valid']:>6} {row['probability']:>8.2%} {row['lift']:>8.3f}")


def append_aggregate_conditions(lines: list[str], studies: list[InstrumentStudy], columns: list[str]) -> None:
    lines.append(f"{'State':<28} {'Age':<8} {'Flag':<12}" + "".join(f" {name + '_Rate':>9}" for name in columns) + " MeanRate")
    for state in STRUCTURAL_STATES:
        for bucket in AGE_BUCKETS:
            for flag in FLAG_NAMES:
                values = {}
                for study in studies:
                    item = study.populations["Full Population"].conditions.get((state, bucket))
                    if item:
                        values[study.instrument] = item.flag_rates[flag]
                cells = "".join(f" {values.get(name, 0.0):>9.2%}" for name in columns)
                lines.append(f"{state:<28} {bucket:<8} {flag:<12}{cells} {mean(list(values.values())):>8.2%}")


def rank(lines: list[str], title: str, items: list, key, formatter) -> None:
    lines.extend(["", title, "." * len(title)])
    eligible = [item for item in items if item.get("valid", 0) >= 2]
    if not eligible:
        lines.append("No items met the two-instrument minimum.")
        return
    for index, item in enumerate(sorted(eligible, key=key, reverse=True)[:TOP_LIMIT], 1):
        lines.append(f"{index:>3}. {formatter(item)}")


def aggregate_curve_rows(studies: list[InstrumentStudy]) -> list[dict]:
    output = []
    for state in STRUCTURAL_STATES:
        for bucket in AGE_BUCKETS:
            values = {study.instrument: study.populations["Full Population"].curves.get((state, bucket)) for study in studies}
            valid = [item for item in values.values() if item and item.count >= VALID_MIN]
            output.append({"state": state, "bucket": bucket, "values": values, "valid": len(valid), "exit": mean([item.exits[1] for item in valid]), "skew": mean([item.skew for item in valid])})
    return output


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    columns = instrument_columns(studies)
    young_old = aggregate_young_old_rows(studies)
    transitions = aggregate_transition_rows(studies)
    curves = aggregate_curve_rows(studies)
    lines = [
        "APVA State Age Analysis Study v0.1 - Aggregate",
        "==============================================",
        f"Instruments: {', '.join(study.instrument for study in studies)}",
        f"Valid-instrument state-age minimum: {VALID_MIN}.",
        "Structural states are imported from APVA_StructuralStateMachine_39.py.",
    ]
    append_heading(lines, "Aggregate State Age Table")
    append_aggregate_age(lines, studies, columns)
    append_heading(lines, "Aggregate Young-vs-Old Table")
    append_aggregate_young_old(lines, young_old, columns)
    append_heading(lines, "Aggregate Transition Table")
    append_aggregate_transitions(lines, transitions, columns)
    append_heading(lines, "Aggregate Age-Condition Table")
    append_aggregate_conditions(lines, studies, columns)
    append_heading(lines, "Aggregate Rankings")
    rank(lines, "1. States with strongest age persistence increase", young_old, lambda item: item["persist"], lambda item: f"{item['state']} | dPersist={pct(item['persist'])} dExit={pct(item['exit'])} Classification={item['classification']}")
    rank(lines, "2. States with strongest age persistence decay", young_old, lambda item: -item["persist"], lambda item: f"{item['state']} | dPersist={pct(item['persist'])} dExit={pct(item['exit'])} Classification={item['classification']}")
    rank(lines, "3. States whose outcome improves most with age", young_old, lambda item: item["skew"], lambda item: f"{item['state']} | dSkew={pct(item['skew'])} Classification={item['classification']}")
    rank(lines, "4. States whose outcome degrades most with age", young_old, lambda item: -item["skew"], lambda item: f"{item['state']} | dSkew={pct(item['skew'])} Classification={item['classification']}")
    rank(lines, "5. Age buckets with highest exit rate", curves, lambda item: item["exit"], lambda item: f"{item['state']} Age={item['bucket']} | Exit1={pct(item['exit'])}")
    rank(lines, "6. Age buckets with lowest exit rate", curves, lambda item: -item["exit"], lambda item: f"{item['state']} Age={item['bucket']} | Exit1={pct(item['exit'])}")
    rank(lines, "7. Best state-age outcome skew", curves, lambda item: item["skew"], lambda item: f"{item['state']} Age={item['bucket']} | Skew={pct(item['skew'])}")
    rank(lines, "8. Worst state-age outcome skew", curves, lambda item: -item["skew"], lambda item: f"{item['state']} Age={item['bucket']} | Skew={pct(item['skew'])}")
    rank(lines, "9. Strongest age-conditioned transitions", transitions, lambda item: item["lift"], lambda item: f"{item['state']} Age={item['bucket']} -> {item['next']} t+{item['horizon']} | Prob={pct(item['probability'])} Lift={item['lift']:.3f}")
    lines.extend(["", "10. Population-B age effects", "." * 28, "Population B remains sparse; per-instrument state-age summaries are reported without aggregate inference."])
    append_heading(lines, "Cross-Instrument Mechanical Research Notes")
    strongest = max([item for item in young_old if item["valid"] >= 2], key=lambda item: item["persist"], default=None)
    weakest = min([item for item in young_old if item["valid"] >= 2], key=lambda item: item["persist"], default=None)
    lines.extend([
        f"- Largest replicated old-minus-young persistence delta: {strongest['state']} ({pct(strongest['persist'])})." if strongest else "- No state met the replicated young-vs-old minimum.",
        f"- Strongest replicated persistence decay with age: {weakest['state']} ({pct(weakest['persist'])})." if weakest else "- No state met the replicated young-vs-old minimum.",
        "- State-age curves, evidence composition, transition lifts, and outcome skew are descriptive only.",
        "- Population B is reported separately and may be sparse when Study 39 source flags are absent.",
    ])
    return "\n".join(lines) + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    studies = [study_instrument(path) for path in args.input_paths]
    seen = set()
    for study in studies:
        if study.instrument in seen:
            raise ValueError(f"Duplicate instrument input: {study.instrument}")
        seen.add(study.instrument)
        write_text(Path("Evidence/Output") / study.instrument / f"StateAge_{study.instrument}.txt", instrument_report(study))
    report = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, report)
    print(report, end="")


if __name__ == "__main__":
    main()
