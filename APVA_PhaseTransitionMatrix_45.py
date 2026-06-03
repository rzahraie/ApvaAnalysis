"""APVA Phase Transition Matrix Study v0.1.

Research-only lifecycle-phase transition study. Phase labels are imported from
Study 44 and are derived from structural-state hazard behavior only.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import pstdev
from typing import Callable, Iterable

from APVA_StructuralLifeCycle_44 import (
    AGE_BUCKETS,
    STRUCTURAL_STATES,
    Bar,
    InstrumentResult,
    Outcome,
    build_lifecycle_rows,
    directional_return,
    load_results,
    mean,
    pct,
)


AGGREGATE_OUTPUT = Path("Evidence/Output/PhaseTransitionMatrix/PhaseTransitionMatrix_All.txt")
PHASES = ("Birth", "Growth", "Maturity", "Decay", "Terminal", "Insufficient")
HORIZONS = (1, 2, 3, 5)
VALID_MIN = 20
TOP_LIMIT = 25


@dataclass(frozen=True)
class MatrixRow:
    source: str
    horizon: int
    destination: str
    count: int
    probability: float
    lift: float | None


@dataclass(frozen=True)
class StatePhaseRow:
    state: str
    phase: str
    horizon: int
    destination: str
    count: int
    probability: float
    lift: float | None


@dataclass(frozen=True)
class EntropyRow:
    phase: str
    horizon: int
    count: int
    entropy: float
    normalized_entropy: float
    dominant_destination: str
    dominant_probability: float
    dominant_lift: float | None


@dataclass(frozen=True)
class CompressionRow:
    state: str
    horizon: int
    age_entropy: float
    phase_entropy: float
    reduction: float


@dataclass(frozen=True)
class OutcomeRow:
    label: str
    state_phase: str
    outcome: Outcome


@dataclass(frozen=True)
class PhaseStudy:
    instrument: str
    sources: list[str]
    bars: list[Bar]
    phases: list[str]
    phase_state: list[MatrixRow]
    phase_phase: list[MatrixRow]
    state_phase: list[StatePhaseRow]
    entropy: list[EntropyRow]
    compression: list[CompressionRow]
    outcomes: list[OutcomeRow]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure APVA lifecycle-phase transition behavior.")
    parser.add_argument("inputs", nargs="+", help="One or more APVA evidence CSV files or directories.")
    return parser.parse_args()


def phase_stream(result: InstrumentResult) -> list[str]:
    rows = build_lifecycle_rows(result.bars)
    mapping = {(row.state, row.age): row.phase for row in rows}
    return [mapping.get((bar.state, bar.age_bucket), "Insufficient") for bar in result.bars]


def matrix(source_labels: list[str], destination_labels: list[str], destinations: tuple[str, ...]) -> list[MatrixRow]:
    output = []
    for horizon in HORIZONS:
        pairs = [(source_labels[index], destination_labels[index + horizon]) for index in range(len(source_labels) - horizon)]
        destination_counts = Counter(destination for _, destination in pairs)
        source_counts = Counter(source for source, _ in pairs)
        pair_counts = Counter(pairs)
        total = len(pairs)
        for source in PHASES:
            for destination in destinations:
                count = pair_counts[(source, destination)]
                probability = count / source_counts[source] if source_counts[source] else 0.0
                baseline = destination_counts[destination] / total if total else 0.0
                output.append(MatrixRow(source, horizon, destination, count, probability, probability / baseline if baseline else None))
    return output


def state_phase_matrix(bars: list[Bar], phases: list[str]) -> list[StatePhaseRow]:
    output = []
    states = [bar.state for bar in bars]
    for horizon in HORIZONS:
        pairs = [((states[index], phases[index]), states[index + horizon]) for index in range(len(bars) - horizon)]
        destinations = Counter(destination for _, destination in pairs)
        sources = Counter(source for source, _ in pairs)
        counts = Counter(pairs)
        total = len(pairs)
        for state in STRUCTURAL_STATES:
            for phase in PHASES:
                for destination in STRUCTURAL_STATES:
                    count = counts[((state, phase), destination)]
                    probability = count / sources[(state, phase)] if sources[(state, phase)] else 0.0
                    baseline = destinations[destination] / total if total else 0.0
                    output.append(StatePhaseRow(state, phase, horizon, destination, count, probability, probability / baseline if baseline else None))
    return output


def shannon(counts: Iterable[int]) -> tuple[float, float]:
    values = [value for value in counts if value > 0]
    total = sum(values)
    if not total:
        return 0.0, 0.0
    entropy = -sum((value / total) * math.log(value / total) for value in values)
    normalized = entropy / math.log(len(values)) if len(values) > 1 else 0.0
    return entropy, normalized


def entropy_rows(phase_state: list[MatrixRow]) -> list[EntropyRow]:
    output = []
    for phase in PHASES:
        for horizon in HORIZONS:
            selected = [row for row in phase_state if row.source == phase and row.horizon == horizon]
            entropy, normalized = shannon(row.count for row in selected)
            dominant = max(selected, key=lambda row: row.count) if selected else None
            output.append(EntropyRow(phase, horizon, sum(row.count for row in selected), entropy, normalized, dominant.destination if dominant else "N/A", dominant.probability if dominant else 0.0, dominant.lift if dominant else None))
    return output


def conditional_entropy(bars: list[Bar], phases: list[str], state: str, horizon: int, category: Callable[[int], str]) -> float:
    groups = defaultdict(Counter)
    for index in range(len(bars) - horizon):
        if bars[index].state == state:
            groups[category(index)][bars[index + horizon].state] += 1
    total = sum(sum(counts.values()) for counts in groups.values())
    if not total:
        return 0.0
    return sum((sum(counts.values()) / total) * shannon(counts.values())[0] for counts in groups.values())


def compression_rows(bars: list[Bar], phases: list[str]) -> list[CompressionRow]:
    output = []
    for state in STRUCTURAL_STATES:
        for horizon in HORIZONS:
            age_entropy = conditional_entropy(bars, phases, state, horizon, lambda index: bars[index].age_bucket)
            phase_entropy = conditional_entropy(bars, phases, state, horizon, lambda index: phases[index])
            output.append(CompressionRow(state, horizon, age_entropy, phase_entropy, age_entropy - phase_entropy))
    return output


def outcome_rows(bars: list[Bar], phases: list[str]) -> list[OutcomeRow]:
    phase_outcomes = {phase: Outcome() for phase in PHASES}
    state_phase_outcomes = {(state, phase): Outcome() for state in STRUCTURAL_STATES for phase in PHASES}
    for index, bar in enumerate(bars):
        value = directional_return(bars, index, 5)
        phase_outcomes[phases[index]].add(value)
        state_phase_outcomes[(bar.state, phases[index])].add(value)
    output = [OutcomeRow(phase, "", outcome) for phase, outcome in phase_outcomes.items()]
    output.extend(OutcomeRow(phase, state, outcome) for (state, phase), outcome in state_phase_outcomes.items())
    return output


def study_result(result: InstrumentResult) -> PhaseStudy:
    phases = phase_stream(result)
    states = [bar.state for bar in result.bars]
    phase_state = matrix(phases, states, STRUCTURAL_STATES)
    return PhaseStudy(result.instrument, result.source_paths, result.bars, phases, phase_state, matrix(phases, phases, PHASES), state_phase_matrix(result.bars, phases), entropy_rows(phase_state), compression_rows(result.bars, phases), outcome_rows(result.bars, phases))


def append_heading(lines: list[str], title: str) -> None:
    lines.extend(["", title, "=" * len(title)])


def fmt(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def append_phase_counts(lines: list[str], study: PhaseStudy) -> None:
    counts = Counter(study.phases)
    lines.append(f"{'Phase':<14} {'Count':>9} {'Frequency':>10}")
    for phase in PHASES:
        lines.append(f"{phase:<14} {counts[phase]:>9} {pct(counts[phase] / len(study.phases) if study.phases else 0.0):>10}")


def append_matrix(lines: list[str], rows: list[MatrixRow], destination_title: str) -> None:
    lines.append(f"{'Phase':<14} {'H':>2} {destination_title:<28} {'Count':>9} {'Probability':>12} {'Lift':>9}")
    for row in rows:
        if row.count:
            lines.append(f"{row.source:<14} {row.horizon:>2} {row.destination:<28} {row.count:>9} {pct(row.probability):>12} {fmt(row.lift):>9}")


def append_state_phase(lines: list[str], rows: list[StatePhaseRow]) -> None:
    lines.append(f"{'CurrentState':<28} {'Phase':<14} {'H':>2} {'NextState':<28} {'Count':>9} {'Probability':>12} {'Lift':>9}")
    for row in rows:
        if row.count:
            lines.append(f"{row.state:<28} {row.phase:<14} {row.horizon:>2} {row.destination:<28} {row.count:>9} {pct(row.probability):>12} {fmt(row.lift):>9}")


def append_entropy(lines: list[str], rows: list[EntropyRow]) -> None:
    lines.append(f"{'Phase':<14} {'H':>2} {'Count':>9} {'Entropy':>10} {'NormEntropy':>12} {'DominantDestination':<28} {'DominantProb':>12} {'DominantLift':>12}")
    for row in rows:
        lines.append(f"{row.phase:<14} {row.horizon:>2} {row.count:>9} {row.entropy:>10.4f} {row.normalized_entropy:>12.4f} {row.dominant_destination:<28} {pct(row.dominant_probability):>12} {fmt(row.dominant_lift):>12}")


def append_dominant(lines: list[str], rows: list[EntropyRow]) -> None:
    lines.append(f"{'Phase':<14} {'H':>2} {'DominantState':<28} {'Probability':>12} {'Lift':>9}")
    for row in rows:
        lines.append(f"{row.phase:<14} {row.horizon:>2} {row.dominant_destination:<28} {pct(row.dominant_probability):>12} {fmt(row.dominant_lift):>9}")


def append_compression(lines: list[str], rows: list[CompressionRow]) -> None:
    lines.append(f"{'State':<28} {'H':>2} {'AgeEntropy':>11} {'PhaseEntropy':>13} {'Reduction':>11}")
    for row in rows:
        lines.append(f"{row.state:<28} {row.horizon:>2} {row.age_entropy:>11.4f} {row.phase_entropy:>13.4f} {row.reduction:>11.4f}")


def append_outcomes(lines: list[str], rows: list[OutcomeRow]) -> None:
    lines.append(f"{'Phase':<14} {'StatePhase':<28} {'Count':>9} {'MeanDR':>11} {'MedianDR':>11} {'Cont':>9} {'Fail':>9} {'Flat':>9} {'Skew':>9}")
    for row in rows:
        if row.outcome.count:
            lines.append(f"{row.label:<14} {row.state_phase:<28} {row.outcome.valid:>9} {row.outcome.mean_dr:>11.5f} {row.outcome.median_dr:>11.5f} {pct(row.outcome.cont_rate):>9} {pct(row.outcome.fail_rate):>9} {pct(row.outcome.flat_rate):>9} {pct(row.outcome.skew):>9}")


def instrument_report(study: PhaseStudy) -> str:
    state_counts = Counter(bar.state for bar in study.bars)
    phase_counts = Counter(study.phases)
    lines = [
        f"APVA Phase Transition Matrix Study v0.1 - {study.instrument}",
        "=" * (43 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input path(s): {', '.join(study.sources)}",
        f"Total rows: {len(study.bars)}",
        f"Structural state counts: {dict(state_counts)}",
        f"Lifecycle phase counts: {dict(phase_counts)}",
        f"Valid phase observations: {sum(phase != 'Insufficient' for phase in study.phases)}",
    ]
    append_heading(lines, "Section 1 - Phase Counts")
    append_phase_counts(lines, study)
    append_heading(lines, "Section 2 - Phase -> NextState Tables")
    append_matrix(lines, study.phase_state, "NextState")
    append_heading(lines, "Section 3 - Phase -> Phase Tables")
    append_matrix(lines, study.phase_phase, "NextPhase")
    append_heading(lines, "Section 4 - State + Phase -> NextState Tables")
    append_state_phase(lines, study.state_phase)
    append_heading(lines, "Section 5 - Entropy Analysis")
    append_entropy(lines, study.entropy)
    append_heading(lines, "Section 6 - Dominant Destinations")
    append_dominant(lines, study.entropy)
    append_heading(lines, "Section 7 - Phase vs Age Entropy Comparison")
    append_compression(lines, study.compression)
    append_heading(lines, "Section 8 - Outcome Layer")
    append_outcomes(lines, study.outcomes)
    append_heading(lines, "Section 9 - Population B")
    lines.extend([
        "Population B unavailable from Study 44 source bars.",
        "Required DissipationContained + Lateral + Mature + Aligned membership fields are not exposed by the reused Study 44 lifecycle loader.",
        "Skipped gracefully; no Population B membership was inferred.",
    ])
    append_heading(lines, "Section 10 - Mechanical Research Notes")
    best = min([row for row in study.entropy if row.phase != "Insufficient"], key=lambda row: row.normalized_entropy, default=None)
    reduction = max(study.compression, key=lambda row: row.reduction, default=None)
    lines.extend([
        f"- Lowest normalized phase entropy: {best.phase} t+{best.horizon} ({best.normalized_entropy:.4f})." if best else "- No sufficient phase entropy rows.",
        f"- Largest phase-vs-age entropy reduction: {reduction.state} t+{reduction.horizon} ({reduction.reduction:.4f})." if reduction else "- No phase-vs-age entropy rows.",
        "- Lifecycle phases are inherited from Study 44 hazard-only inference.",
        "- Outcomes are reported separately and are not used to define phase.",
    ])
    return "\n".join(lines) + "\n"


def instrument_columns(studies: list[PhaseStudy]) -> list[str]:
    available = {study.instrument for study in studies}
    return [name for name in ("6E", "CL", "NQ") if name in available] + sorted(available - {"6E", "CL", "NQ"})


def row_map(rows: list, key: Callable) -> dict:
    return {key(row): row for row in rows}


def aggregate_matrix_rows(studies: list[PhaseStudy], attribute: str, key: Callable) -> list[dict]:
    keys = sorted({key(row) for study in studies for row in getattr(study, attribute)})
    maps = {study.instrument: row_map(getattr(study, attribute), key) for study in studies}
    output = []
    for item_key in keys:
        values = {name: mapping[item_key] for name, mapping in maps.items() if item_key in mapping}
        valid = [row for row in values.values() if row.count >= VALID_MIN]
        lifts = [row.lift for row in valid if row.lift is not None]
        output.append({"key": item_key, "values": values, "valid": len(valid), "probability": mean([row.probability for row in valid]), "lift": mean(lifts), "prob_std": pstdev([row.probability for row in valid]) if len(valid) > 1 else 0.0, "lift_std": pstdev(lifts) if len(lifts) > 1 else 0.0})
    return output


def append_aggregate_matrix(lines: list[str], rows: list[dict], columns: list[str], key_headers: tuple[str, ...]) -> None:
    lines.append(" ".join(f"{header:<28}" for header in key_headers) + "".join(f" {name + '_N':>8} {name + '_Prob':>9} {name + '_Lift':>8}" for name in columns) + " ValidN MeanProb MeanLift ProbStd LiftStd")
    for item in rows:
        if not any(row.count for row in item["values"].values()):
            continue
        keys = item["key"] if isinstance(item["key"], tuple) else (item["key"],)
        prefix = " ".join(f"{str(value):<28}" for value in keys)
        cells = ""
        for name in columns:
            row = item["values"].get(name)
            cells += f" {row.count if row else 0:>8} {row.probability if row else 0.0:>9.2%} {fmt(row.lift if row else None):>8}"
        lines.append(f"{prefix}{cells} {item['valid']:>6} {item['probability']:>8.2%} {item['lift']:>8.3f} {item['prob_std']:>7.3f} {item['lift_std']:>7.3f}")


def aggregate_entropy_rows(studies: list[PhaseStudy]) -> list[dict]:
    maps = {study.instrument: row_map(study.entropy, lambda row: (row.phase, row.horizon)) for study in studies}
    output = []
    for phase in PHASES:
        for horizon in HORIZONS:
            values = {name: mapping[(phase, horizon)] for name, mapping in maps.items()}
            valid = [row for row in values.values() if row.count >= VALID_MIN]
            output.append({"phase": phase, "horizon": horizon, "values": values, "valid": len(valid), "entropy": mean([row.entropy for row in valid]), "normalized": mean([row.normalized_entropy for row in valid]), "dominant": mean([row.dominant_probability for row in valid])})
    return output


def append_aggregate_entropy(lines: list[str], rows: list[dict], columns: list[str]) -> None:
    lines.append(f"{'Phase':<14} {'H':>2}" + "".join(f" {name + '_N':>8} {name + '_Ent':>8} {name + '_Norm':>8} {name + '_Dom':>18} {name + '_Prob':>9}" for name in columns) + " ValidN MeanEnt MeanNorm MeanDomProb")
    for item in rows:
        cells = ""
        for name in columns:
            row = item["values"][name]
            cells += f" {row.count:>8} {row.entropy:>8.3f} {row.normalized_entropy:>8.3f} {row.dominant_destination:>18} {row.dominant_probability:>9.2%}"
        lines.append(f"{item['phase']:<14} {item['horizon']:>2}{cells} {item['valid']:>6} {item['entropy']:>7.3f} {item['normalized']:>8.3f} {item['dominant']:>11.2%}")


def aggregate_compression_rows(studies: list[PhaseStudy]) -> list[dict]:
    maps = {study.instrument: row_map(study.compression, lambda row: (row.state, row.horizon)) for study in studies}
    output = []
    for state in STRUCTURAL_STATES:
        for horizon in HORIZONS:
            values = {name: mapping[(state, horizon)] for name, mapping in maps.items()}
            output.append({"state": state, "horizon": horizon, "values": values, "valid": len(values), "age": mean([row.age_entropy for row in values.values()]), "phase": mean([row.phase_entropy for row in values.values()]), "reduction": mean([row.reduction for row in values.values()])})
    return output


def append_aggregate_compression(lines: list[str], rows: list[dict], columns: list[str]) -> None:
    lines.append(f"{'State':<28} {'H':>2}" + "".join(f" {name + '_Age':>9} {name + '_Phase':>10} {name + '_Red':>9}" for name in columns) + " ValidN MeanAge MeanPhase MeanReduction")
    for item in rows:
        cells = "".join(f" {item['values'][name].age_entropy:>9.3f} {item['values'][name].phase_entropy:>10.3f} {item['values'][name].reduction:>9.3f}" for name in columns)
        lines.append(f"{item['state']:<28} {item['horizon']:>2}{cells} {item['valid']:>6} {item['age']:>7.3f} {item['phase']:>9.3f} {item['reduction']:>13.3f}")


def aggregate_outcome_rows(studies: list[PhaseStudy]) -> list[dict]:
    keys = sorted({(row.label, row.state_phase) for study in studies for row in study.outcomes})
    maps = {study.instrument: row_map(study.outcomes, lambda row: (row.label, row.state_phase)) for study in studies}
    output = []
    for key in keys:
        values = {name: mapping[key].outcome for name, mapping in maps.items() if key in mapping}
        valid = [row for row in values.values() if row.valid >= VALID_MIN]
        output.append({"key": key, "values": values, "valid": len(valid), "cont": mean([row.cont_rate for row in valid]), "fail": mean([row.fail_rate for row in valid]), "skew": mean([row.skew for row in valid]), "mean": mean([row.mean_dr for row in valid])})
    return output


def append_aggregate_outcomes(lines: list[str], rows: list[dict], columns: list[str]) -> None:
    lines.append(f"{'Phase':<14} {'StatePhase':<28}" + "".join(f" {name + '_N':>8} {name + '_Cont':>9} {name + '_Fail':>9} {name + '_Skew':>9} {name + '_Mean':>10}" for name in columns) + " ValidN MeanCont MeanFail MeanSkew MeanDR")
    for item in rows:
        cells = ""
        for name in columns:
            row = item["values"].get(name)
            cells += f" {row.valid if row else 0:>8} {row.cont_rate if row else 0.0:>9.2%} {row.fail_rate if row else 0.0:>9.2%} {row.skew if row else 0.0:>9.2%} {row.mean_dr if row else 0.0:>10.4f}"
        lines.append(f"{item['key'][0]:<14} {item['key'][1]:<28}{cells} {item['valid']:>6} {item['cont']:>8.2%} {item['fail']:>8.2%} {item['skew']:>8.2%} {item['mean']:>7.4f}")


def rank(lines: list[str], title: str, rows: list[dict], key: Callable, formatter: Callable, exclude_insufficient: bool = False) -> None:
    lines.extend(["", title, "." * len(title)])
    eligible = [row for row in rows if row["valid"] >= 2 and (not exclude_insufficient or "Insufficient" not in row.get("key", ()))]
    if not eligible:
        lines.append("No rows met the two-instrument minimum.")
        return
    for index, row in enumerate(sorted(eligible, key=key, reverse=True)[:TOP_LIMIT], 1):
        lines.append(f"{index:>3}. {formatter(row)}")


def aggregate_report(studies: list[PhaseStudy]) -> str:
    columns = instrument_columns(studies)
    phase_state = aggregate_matrix_rows(studies, "phase_state", lambda row: (row.source, row.horizon, row.destination))
    phase_phase = aggregate_matrix_rows(studies, "phase_phase", lambda row: (row.source, row.horizon, row.destination))
    state_phase = aggregate_matrix_rows(studies, "state_phase", lambda row: (row.state, row.phase, row.horizon, row.destination))
    entropy = aggregate_entropy_rows(studies)
    compression = aggregate_compression_rows(studies)
    outcomes = aggregate_outcome_rows(studies)
    lines = [
        "APVA Phase Transition Matrix Study v0.1 - Aggregate",
        "===================================================",
        f"Instruments: {', '.join(columns)}",
        f"Valid-instrument minimum: {VALID_MIN}.",
        "Lifecycle phases are imported from Study 44 hazard-only inference.",
    ]
    append_heading(lines, "Aggregate Phase -> NextState Table")
    append_aggregate_matrix(lines, phase_state, columns, ("Phase", "Horizon", "NextState"))
    append_heading(lines, "Aggregate Phase -> Phase Table")
    append_aggregate_matrix(lines, phase_phase, columns, ("Phase", "Horizon", "NextPhase"))
    append_heading(lines, "Aggregate State + Phase Table")
    append_aggregate_matrix(lines, state_phase, columns, ("CurrentState", "CurrentPhase", "Horizon", "NextState"))
    append_heading(lines, "Aggregate Entropy Table")
    append_aggregate_entropy(lines, entropy, columns)
    append_heading(lines, "Aggregate Phase vs Age Table")
    append_aggregate_compression(lines, compression, columns)
    append_heading(lines, "Aggregate Outcome Table")
    append_aggregate_outcomes(lines, outcomes, columns)
    append_heading(lines, "Aggregate Rankings")
    entropy_rank = [{"key": (row["phase"], row["horizon"]), "valid": row["valid"], "entropy": row["normalized"], "dominant": row["dominant"]} for row in entropy if row["phase"] != "Insufficient"]
    rank(lines, "1. Lowest entropy phases", entropy_rank, lambda row: -row["entropy"], lambda row: f"{row['key'][0]} t+{row['key'][1]} | NormEntropy={row['entropy']:.3f} DominantProb={pct(row['dominant'])}")
    rank(lines, "2. Highest entropy phases", entropy_rank, lambda row: row["entropy"], lambda row: f"{row['key'][0]} t+{row['key'][1]} | NormEntropy={row['entropy']:.3f} DominantProb={pct(row['dominant'])}")
    strong_phase_state = [row for row in phase_state if row["key"][0] != "Insufficient" and row["lift"] > 1.5 and row["probability"] >= 0.10]
    rank(lines, "3. Strongest replicated Phase -> NextState transitions", strong_phase_state, lambda row: row["lift"], lambda row: f"{row['key'][0]} t+{row['key'][1]} -> {row['key'][2]} | Prob={pct(row['probability'])} Lift={row['lift']:.3f}")
    strong_phase_phase = [row for row in phase_phase if row["key"][0] != "Insufficient" and row["key"][2] != "Insufficient"]
    rank(lines, "4. Strongest replicated Phase -> Phase transitions", strong_phase_phase, lambda row: row["lift"], lambda row: f"{row['key'][0]} t+{row['key'][1]} -> {row['key'][2]} | Prob={pct(row['probability'])} Lift={row['lift']:.3f}")
    rank(lines, "5. Strongest State + Phase transitions", state_phase, lambda row: row["lift"], lambda row: f"{row['key'][0]} + {row['key'][1]} t+{row['key'][2]} -> {row['key'][3]} | Prob={pct(row['probability'])} Lift={row['lift']:.3f}", True)
    rank(lines, "6. Phases with strongest dominant destinations", entropy_rank, lambda row: row["dominant"], lambda row: f"{row['key'][0]} t+{row['key'][1]} | DominantProb={pct(row['dominant'])}")
    rank(lines, "7. States where phase reduces entropy most", [row for row in compression if row["reduction"] > 0], lambda row: row["reduction"], lambda row: f"{row['state']} t+{row['horizon']} | EntropyReduction={row['reduction']:.4f}")
    rank(lines, "8. States where phase loses information", [row for row in compression if row["reduction"] < 0], lambda row: -row["reduction"], lambda row: f"{row['state']} t+{row['horizon']} | EntropyReduction={row['reduction']:.4f}")
    phase_outcomes = [row for row in outcomes if not row["key"][1] and row["key"][0] != "Insufficient"]
    rank(lines, "9. Best phase outcome skew", phase_outcomes, lambda row: row["skew"], lambda row: f"{row['key'][0]} | Skew={pct(row['skew'])} MeanDR={row['mean']:.4f}")
    rank(lines, "10. Worst phase outcome skew", phase_outcomes, lambda row: -row["skew"], lambda row: f"{row['key'][0]} | Skew={pct(row['skew'])} MeanDR={row['mean']:.4f}")
    lines.extend(["", "11. Population-B phase differences", "." * 34, "Population B skipped gracefully; the reused Study 44 loader does not expose required membership fields."])
    append_heading(lines, "Cross-Instrument Mechanical Research Notes")
    best = min(entropy_rank, key=lambda row: row["entropy"], default=None)
    reduction = max(compression, key=lambda row: row["reduction"], default=None)
    lines.extend([
        f"- Lowest replicated normalized entropy phase: {best['key'][0]} t+{best['key'][1]} ({best['entropy']:.4f})." if best else "- No replicated sufficient phase entropy rows.",
        f"- Largest mean phase-vs-age entropy reduction: {reduction['state']} t+{reduction['horizon']} ({reduction['reduction']:.4f})." if reduction else "- No entropy-compression rows.",
        "- Lifecycle phase labels are determined by Study 44 state-age hazard behavior only.",
        "- Outcome skew is reported separately and is not used for phase assignment.",
    ])
    return "\n".join(lines) + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    studies = [study_result(result) for result in load_results(args.inputs)]
    if not studies:
        raise ValueError("No evidence CSV inputs were found.")
    for study in studies:
        write_text(Path("Evidence/Output") / study.instrument / f"PhaseTransitionMatrix_{study.instrument}.txt", instrument_report(study))
    report = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, report)
    print(report, end="")


if __name__ == "__main__":
    main()
