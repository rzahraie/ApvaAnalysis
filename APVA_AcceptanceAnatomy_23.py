"""APVA Acceptance Anatomy Study v0.1.

Research-only descriptive study. Reuses existing evidence-layer definitions.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

from APVA_BreakoutContext_08 import (
    EvidenceBar,
    detect_frames,
    direction_relative_return,
    ibsym_start,
    lateral_start,
    load_rows,
)
from APVA_DissipationCompressionGradient_20 import pearson, rank_values, spearman
from APVA_DominanceGradient_21 import score_value
from APVA_LateralAnatomy_19 import Case, case_features, effect_size, on
from APVA_LateralPreBreakoutGradient_22 import build_cases as lateral_breakout_cases
from APVA_PostBreakoutOOE_10 import build_segment_bars
from APVA_RegimeTransition_16 import CANONICAL_INSTRUMENTS, instrument_name, write_text


AGGREGATE_OUTPUT = Path("Evidence/Output/AcceptanceAnatomy/AcceptanceAnatomy_All.txt")
TOP_LIMIT = 25
MIN_AGGREGATE_COUNT = 30
ACCEPTANCE_STATES = ("Accepted", "Contained", "Other/Absent")
PARTICIPATION_STATES = ("Falling", "Rising", "Peak", "Climactic")
POLARITIES = ("Black", "Red", "Other")
LATENT_SCORES = (
    "AcceptanceScore",
    "AcceptanceScore2",
    "AcceptanceScore3",
    "DominanceGradient",
)


@dataclass(frozen=True)
class OutcomeStats:
    count: int
    mean: float
    median: float
    continuation_rate: float
    failure_rate: float
    flat_rate: float


@dataclass(frozen=True)
class ComparisonStats:
    name: str
    accepted_count: int
    non_accepted_count: int
    accepted_mean: float
    non_accepted_mean: float
    delta: float
    effect: float


@dataclass(frozen=True)
class LatentStats:
    name: str
    non_flat_count: int
    rho: float
    low: OutcomeStats
    middle: OutcomeStats
    high: OutcomeStats

    @property
    def high_minus_low(self) -> float:
        return self.high.continuation_rate - self.low.continuation_rate


@dataclass(frozen=True)
class PopulationStudy:
    name: str
    cases: list[Case]
    raw_frequency: dict[str, int]
    frequency: dict[str, OutcomeStats]
    context: list[ComparisonStats]
    neighborhood: list[ComparisonStats]
    persistence: dict[str, OutcomeStats]
    interactions: dict[str, dict[str, OutcomeStats]]
    latent: dict[str, LatentStats]


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    rows: list[EvidenceBar]
    populations: dict[str, PopulationStudy]
    breakout_acceptance: dict[str, OutcomeStats]
    breakout_rho3: float
    breakout_rho5: float


@dataclass(frozen=True)
class AggregateFinding:
    name: str
    values: dict[str, tuple[int, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study acceptance anatomy in all bars and the mature aligned lateral population."
    )
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summarize(values: list[float]) -> OutcomeStats:
    count = len(values)
    denominator = count if count else 1
    return OutcomeStats(
        count,
        mean(values),
        statistics.median(values) if values else 0.0,
        sum(value > 0.0 for value in values) / denominator,
        sum(value < 0.0 for value in values) / denominator,
        sum(value == 0.0 for value in values) / denominator,
    )


def acceptance_state(row: EvidenceBar) -> str:
    return row.acceptance if row.acceptance in {"Accepted", "Contained"} else "Other/Absent"


def polarity_state(row: EvidenceBar) -> str:
    return row.polarity if row.polarity in {"Black", "Red"} else "Other"


def all_bar_cases(rows: list[EvidenceBar]) -> list[Case]:
    cases = []
    for index in range(len(rows)):
        value = direction_relative_return(rows, index, 5)
        if value is not None:
            cases.append(Case(index, value, case_features(rows, index)))
    return cases


def mature_aligned_lateral_indexes(rows: list[EvidenceBar]) -> list[int]:
    ibsym, _ = detect_frames(rows, "IBSYM", ibsym_start)
    lateral, _ = detect_frames(rows, "Lateral", lateral_start)
    segment_bars, _ = build_segment_bars(rows, ibsym + lateral)
    indexes = []
    for segment_bar in segment_bars:
        row = rows[segment_bar.index]
        if segment_bar.breakout_type != "Lateral":
            continue
        if segment_bar.sequence_stage != 4 or segment_bar.alignment != "Aligned":
            continue
        if row.dissipation == "Absent" or row.acceptance != "Contained":
            continue
        indexes.append(segment_bar.index)
    return indexes


def cases_for_indexes(rows: list[EvidenceBar], indexes: list[int]) -> list[Case]:
    cases = []
    for index in indexes:
        value = direction_relative_return(rows, index, 5)
        if value is not None:
            cases.append(Case(index, value, case_features(rows, index)))
    return cases


def values_by_state(rows: list[EvidenceBar], cases: list[Case]) -> dict[str, OutcomeStats]:
    return {
        state: summarize(
            [case.drfwd5 for case in cases if acceptance_state(rows[case.index]) == state]
        )
        for state in ACCEPTANCE_STATES
    }


def binary_comparison(
    rows: list[EvidenceBar],
    cases: list[Case],
    name: str,
    predicate,
) -> ComparisonStats:
    accepted = [float(predicate(rows[case.index], case)) for case in cases if rows[case.index].acceptance == "Accepted"]
    other = [float(predicate(rows[case.index], case)) for case in cases if rows[case.index].acceptance != "Accepted"]
    return ComparisonStats(
        name,
        len(accepted),
        len(other),
        mean(accepted),
        mean(other),
        mean(accepted) - mean(other),
        effect_size(accepted, other),
    )


def context_comparisons(rows: list[EvidenceBar], cases: list[Case]) -> list[ComparisonStats]:
    predicates = [
        ("CompressionOn", lambda row, case: on(row.compression)),
        ("ExpansionOn", lambda row, case: on(row.expansion)),
        ("DissipationOn", lambda row, case: on(row.dissipation)),
    ]
    predicates.extend(
        (f"Participation_{state}", lambda row, case, state=state: row.participation == state)
        for state in PARTICIPATION_STATES
    )
    predicates.extend(
        (f"VolumePolarity_{state}", lambda row, case, state=state: polarity_state(row) == state)
        for state in POLARITIES
    )
    return [binary_comparison(rows, cases, name, predicate) for name, predicate in predicates]


def neighborhood_comparisons(rows: list[EvidenceBar], cases: list[Case]) -> list[ComparisonStats]:
    features = (
        "Prev10_CompressionCount",
        "Prev10_DissipationCount",
        "Prev10_ExpansionCount",
        "Prev10_AcceptedCount",
        "Prev10_PeakCount",
        "Prev10_ClimacticCount",
    )
    output = []
    for feature in features:
        accepted = [case.features[feature] for case in cases if rows[case.index].acceptance == "Accepted"]
        other = [case.features[feature] for case in cases if rows[case.index].acceptance != "Accepted"]
        output.append(
            ComparisonStats(
                feature,
                len(accepted),
                len(other),
                mean(accepted),
                mean(other),
                mean(accepted) - mean(other),
                effect_size(accepted, other),
            )
        )
    return output


def persistence_stats(rows: list[EvidenceBar], cases: list[Case]) -> dict[str, OutcomeStats]:
    case_values = {case.index: case.drfwd5 for case in cases}
    grouped = {label: [] for label in ("1", "2", "3", "4", "5+")}
    index = 0
    while index < len(rows):
        if rows[index].acceptance != "Accepted":
            index += 1
            continue
        start = index
        while index + 1 < len(rows) and rows[index + 1].acceptance == "Accepted":
            index += 1
        length = index - start + 1
        label = str(length) if length < 5 else "5+"
        if index in case_values:
            grouped[label].append(case_values[index])
        index += 1
    return {label: summarize(values) for label, values in grouped.items()}


def interaction_stats(rows: list[EvidenceBar], cases: list[Case]) -> dict[str, dict[str, OutcomeStats]]:
    dimensions = {
        "Acceptance x Compression": lambda row: "CompressionOn" if on(row.compression) else "CompressionOff",
        "Acceptance x Dissipation": lambda row: "DissipationOn" if on(row.dissipation) else "DissipationOff",
        "Acceptance x Expansion": lambda row: "ExpansionOn" if on(row.expansion) else "ExpansionOff",
    }
    output = {}
    for title, classify in dimensions.items():
        grouped: dict[str, list[float]] = {}
        for case in cases:
            row = rows[case.index]
            key = f"{acceptance_state(row)} | {classify(row)}"
            grouped.setdefault(key, []).append(case.drfwd5)
        output[title] = {key: summarize(values) for key, values in sorted(grouped.items())}
    return output


def latent_value(case: Case, name: str) -> float:
    features = case.features
    accepted = features["Prev10_AcceptedCount"]
    dissipation = features["Prev10_DissipationCount"]
    compression = features["Prev10_CompressionCount"]
    if name == "AcceptanceScore":
        return accepted
    if name == "AcceptanceScore2":
        return accepted + dissipation
    if name == "AcceptanceScore3":
        return accepted + dissipation - compression
    return score_value(case, "DominanceGradient")


def latent_stats(cases: list[Case], name: str) -> LatentStats:
    ordered = sorted(cases, key=lambda case: (latent_value(case, name), case.index))
    stop = len(ordered) // 3
    groups = {
        "Low": ordered[:stop],
        "Middle": ordered[stop : len(ordered) - stop],
        "High": ordered[len(ordered) - stop :] if stop else [],
    }
    non_flat = [case for case in cases if case.drfwd5 != 0.0]
    rho = pearson(
        rank_values([latent_value(case, name) for case in non_flat]),
        rank_values([1.0 if case.drfwd5 > 0.0 else 0.0 for case in non_flat]),
    )
    return LatentStats(
        name,
        len(non_flat),
        rho,
        summarize([case.drfwd5 for case in groups["Low"]]),
        summarize([case.drfwd5 for case in groups["Middle"]]),
        summarize([case.drfwd5 for case in groups["High"]]),
    )


def population_study(
    name: str,
    rows: list[EvidenceBar],
    indexes: list[int],
    cases: list[Case],
) -> PopulationStudy:
    return PopulationStudy(
        name,
        cases,
        {state: sum(acceptance_state(rows[index]) == state for index in indexes) for state in ACCEPTANCE_STATES},
        values_by_state(rows, cases),
        context_comparisons(rows, cases),
        neighborhood_comparisons(rows, cases),
        persistence_stats(rows, cases),
        interaction_stats(rows, cases),
        {score: latent_stats(cases, score) for score in LATENT_SCORES},
    )


def lateral_breakout_acceptance(rows: list[EvidenceBar]) -> tuple[dict[str, OutcomeStats], float, float]:
    cases = lateral_breakout_cases(rows)
    bins = {
        "Final3AcceptedMean Low": lambda value: value < 0.34,
        "Final3AcceptedMean Medium": lambda value: 0.34 <= value < 0.67,
        "Final3AcceptedMean High": lambda value: value >= 0.67,
        "Final5AcceptedMean Low": lambda value: value < 0.34,
        "Final5AcceptedMean Medium": lambda value: 0.34 <= value < 0.67,
        "Final5AcceptedMean High": lambda value: value >= 0.67,
    }
    grouped = {}
    for name, predicate in bins.items():
        feature = "Final3AcceptedMean" if name.startswith("Final3") else "Final5AcceptedMean"
        grouped[name] = summarize(
            [case.drfwd5 for case in cases if predicate(float(case.features[feature]))]
        )
    non_flat = [case for case in cases if case.drfwd5 != 0.0]

    def correlation(feature: str) -> float:
        return pearson(
            rank_values([float(case.features[feature]) for case in non_flat]),
            rank_values([1.0 if case.drfwd5 > 0.0 else 0.0 for case in non_flat]),
        )

    return grouped, correlation("Final3AcceptedMean"), correlation("Final5AcceptedMean")


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    population_b_indexes = mature_aligned_lateral_indexes(rows)
    populations = {
        "Population A - All Bars": population_study(
            "Population A - All Bars",
            rows,
            list(range(len(rows))),
            all_bar_cases(rows),
        ),
        "Population B - DissipationContained + Lateral + Mature + Aligned": population_study(
            "Population B - DissipationContained + Lateral + Mature + Aligned",
            rows,
            population_b_indexes,
            cases_for_indexes(rows, population_b_indexes),
        ),
    }
    breakout, rho3, rho5 = lateral_breakout_acceptance(rows)
    return InstrumentStudy(instrument_name(path), path, rows, populations, breakout, rho3, rho5)


def append_outcomes(lines: list[str], grouped: dict[str, OutcomeStats]) -> None:
    total = sum(item.count for item in grouped.values())
    lines.append(f"{'Label':<42} {'Count':>8} {'Percent':>9} {'MeanDRFwd5':>12} {'Median':>12} {'ContRate5':>10} {'FailRate5':>10}")
    for label, item in grouped.items():
        lines.append(
            f"{label:<42} {item.count:>8} {item.count / total if total else 0.0:>8.2%} "
            f"{item.mean:>12.6f} {item.median:>12.6f} "
            f"{item.continuation_rate:>9.2%} {item.failure_rate:>9.2%}"
        )


def append_frequency(lines: list[str], population: PopulationStudy) -> None:
    total = sum(population.raw_frequency.values())
    lines.append(f"{'State':<20} {'RawCount':>10} {'Percent':>9} {'OutcomeN':>9} {'MeanDRFwd5':>12} {'Median':>12} {'ContRate5':>10}")
    for state in ACCEPTANCE_STATES:
        item = population.frequency[state]
        raw_count = population.raw_frequency[state]
        lines.append(
            f"{state:<20} {raw_count:>10} {raw_count / total if total else 0.0:>8.2%} "
            f"{item.count:>9} {item.mean:>12.6f} {item.median:>12.6f} {item.continuation_rate:>9.2%}"
        )


def append_comparisons(lines: list[str], items: list[ComparisonStats]) -> None:
    lines.append(f"{'Feature':<30} {'AcceptedN':>9} {'OtherN':>9} {'AcceptedMean':>13} {'OtherMean':>12} {'Delta':>10} {'Effect':>9}")
    for item in items:
        lines.append(
            f"{item.name:<30} {item.accepted_count:>9} {item.non_accepted_count:>9} "
            f"{item.accepted_mean:>13.6f} {item.non_accepted_mean:>12.6f} "
            f"{item.delta:>10.6f} {item.effect:>9.4f}"
        )


def append_population(lines: list[str], population: PopulationStudy) -> None:
    lines.extend([f"\n{population.name}", "=" * len(population.name), f"Rows with valid DRFwd5: {len(population.cases)}"])
    lines.extend(["\nSection 1 - Acceptance Frequency Analysis", "-----------------------------------------"])
    append_frequency(lines, population)
    lines.extend(["\nSection 2 - Acceptance Context Analysis", "---------------------------------------"])
    append_comparisons(lines, population.context)
    lines.extend(["\nSection 3 - Acceptance Neighborhood", "-----------------------------------"])
    append_comparisons(lines, population.neighborhood)
    lines.extend(["\nSection 4 - Acceptance Persistence", "----------------------------------"])
    append_outcomes(lines, population.persistence)
    lines.extend(["\nSection 6 - Acceptance Interaction Matrix", "-----------------------------------------"])
    for title, grouped in population.interactions.items():
        lines.extend([f"\n{title}", "-" * len(title)])
        lines.append(f"{'Cell':<42} {'Count':>8} {'ContinuationRate5':>18}")
        for label, item in grouped.items():
            rate = f"{item.continuation_rate:.2%}" if item.count >= 10 else "N/A"
            lines.append(f"{label:<42} {item.count:>8} {rate:>18}")
    lines.extend(["\nSection 7 - Acceptance Latent Variable Test", "-------------------------------------------"])
    lines.append(f"{'Score':<22} {'NonFlat':>8} {'Spearman':>10} {'LowRate':>10} {'MiddleRate':>11} {'HighRate':>10} {'High-Low':>10}")
    for item in population.latent.values():
        lines.append(
            f"{item.name:<22} {item.non_flat_count:>8} {item.rho:>10.4f} "
            f"{item.low.continuation_rate:>9.2%} {item.middle.continuation_rate:>10.2%} "
            f"{item.high.continuation_rate:>9.2%} {item.high_minus_low:>9.2%}"
        )


def instrument_report(study: InstrumentStudy) -> str:
    lines = [
        f"APVA Acceptance Anatomy Study v0.1 - {study.instrument}",
        "=" * (37 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {len(study.rows)}",
    ]
    for population in study.populations.values():
        append_population(lines, population)
    lines.extend(["\nSection 5 - Acceptance Before Breakout", "--------------------------------------"])
    append_outcomes(lines, study.breakout_acceptance)
    lines.append(f"Final3AcceptedMean SpearmanRho: {study.breakout_rho3:.4f}")
    lines.append(f"Final5AcceptedMean SpearmanRho: {study.breakout_rho5:.4f}")
    lines.extend(["\nResearch Notes", "=============="])
    all_bars = study.populations["Population A - All Bars"]
    context = {item.name: item for item in all_bars.context}
    neighborhood = {item.name: item for item in all_bars.neighborhood}
    accepted = all_bars.frequency["Accepted"]
    lines.append(f"- Accepted bars: n={accepted.count}, MeanDRFwd5={accepted.mean:.6f}, ContinuationRate5={accepted.continuation_rate:.2%}.")
    lines.append(f"- Accepted versus non-accepted DissipationOn effect={context['DissipationOn'].effect:.4f}.")
    lines.append(f"- Accepted versus non-accepted CompressionOn effect={context['CompressionOn'].effect:.4f}.")
    lines.append(f"- Prev10_AcceptedCount accepted-versus-other effect={neighborhood['Prev10_AcceptedCount'].effect:.4f}.")
    lines.append(f"- Final3AcceptedMean breakout SpearmanRho={study.breakout_rho3:.4f}; Final5AcceptedMean={study.breakout_rho5:.4f}.")
    lines.append(f"- AcceptanceScore SpearmanRho={all_bars.latent['AcceptanceScore'].rho:.4f}; DominanceGradient={all_bars.latent['DominanceGradient'].rho:.4f}.")
    return "\n".join(lines) + "\n"


def aggregate_findings(studies: list[InstrumentStudy]) -> list[AggregateFinding]:
    findings = []
    for population_name in next(iter(studies)).populations:
        prefix = "AllBars" if population_name.startswith("Population A") else "PopulationB"
        for state in ACCEPTANCE_STATES:
            findings.append(AggregateFinding(
                f"{prefix}:AcceptanceState:{state}:ContinuationRate5",
                {study.instrument: (study.populations[population_name].frequency[state].count, study.populations[population_name].frequency[state].continuation_rate) for study in studies},
            ))
        for item_name in ("CompressionOn", "ExpansionOn", "DissipationOn"):
            findings.append(AggregateFinding(
                f"{prefix}:Context:{item_name}:EffectSize",
                {study.instrument: (next(item for item in study.populations[population_name].context if item.name == item_name).accepted_count, next(item for item in study.populations[population_name].context if item.name == item_name).effect) for study in studies},
            ))
        for score in LATENT_SCORES:
            findings.append(AggregateFinding(
                f"{prefix}:Latent:{score}:Spearman",
                {study.instrument: (study.populations[population_name].latent[score].non_flat_count, study.populations[population_name].latent[score].rho) for study in studies},
            ))
            findings.append(AggregateFinding(
                f"{prefix}:Latent:{score}:HighMinusLowContinuation",
                {study.instrument: (study.populations[population_name].latent[score].non_flat_count, study.populations[population_name].latent[score].high_minus_low) for study in studies},
            ))
    findings.append(AggregateFinding("LateralBreakout:Final3AcceptedMean:Spearman", {study.instrument: (sum(item.count for name, item in study.breakout_acceptance.items() if name.startswith("Final3")), study.breakout_rho3) for study in studies}))
    findings.append(AggregateFinding("LateralBreakout:Final5AcceptedMean:Spearman", {study.instrument: (sum(item.count for name, item in study.breakout_acceptance.items() if name.startswith("Final5")), study.breakout_rho5) for study in studies}))
    return findings


def instrument_columns(studies: list[InstrumentStudy]) -> list[str]:
    available = [study.instrument for study in studies]
    return list(CANONICAL_INSTRUMENTS) + sorted(set(available) - set(CANONICAL_INSTRUMENTS))


def valid_values(finding: AggregateFinding) -> list[tuple[int, float]]:
    return [value for value in finding.values.values() if value[0] >= MIN_AGGREGATE_COUNT]


def append_ranked(lines: list[str], title: str, findings: list[AggregateFinding], mode: str) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(f"{'Rank':>4} {'Finding':<68} {'Valid':>5} {'Pos':>4} {'Neg':>4} {'MeanMetric':>12}")
    eligible = [finding for finding in findings if len(valid_values(finding)) >= 2]
    if mode == "positive":
        eligible.sort(key=lambda finding: (sum(value > 0 for _, value in valid_values(finding)), mean([value for _, value in valid_values(finding)])), reverse=True)
    elif mode == "negative":
        eligible.sort(key=lambda finding: (sum(value < 0 for _, value in valid_values(finding)), -mean([value for _, value in valid_values(finding)])), reverse=True)
    else:
        eligible.sort(key=lambda finding: abs(mean([value for _, value in valid_values(finding)])), reverse=True)
    for rank, finding in enumerate(eligible[:TOP_LIMIT], start=1):
        values = valid_values(finding)
        lines.append(f"{rank:>4} {finding.name:<68} {len(values):>5} {sum(value > 0 for _, value in values):>4} {sum(value < 0 for _, value in values):>4} {mean([value for _, value in values]):>12.6f}")


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    columns = instrument_columns(studies)
    findings = aggregate_findings(studies)
    lines = [
        "APVA Acceptance Anatomy Study v0.1 - Cross-Instrument Aggregate",
        "===============================================================",
        f"Input instruments: {', '.join(study.instrument for study in studies)}",
        f"Valid metric threshold: count >= {MIN_AGGREGATE_COUNT}.",
        "",
        "Aggregate Metric Table",
        "======================",
    ]
    header = f"{'Finding':<68}"
    for instrument in columns:
        header += f" {('Count_' + instrument):>10} {('Metric_' + instrument):>12}"
    header += f" {'Valid':>5} {'Pos':>4} {'Neg':>4} {'MeanMetric':>12}"
    lines.append(header)
    for finding in findings:
        row = f"{finding.name:<68}"
        for instrument in columns:
            count, value = finding.values.get(instrument, (0, 0.0))
            row += f" {count:>10} {value:>12.6f}"
        values = valid_values(finding)
        row += f" {len(values):>5} {sum(value > 0 for _, value in values):>4} {sum(value < 0 for _, value in values):>4} {mean([value for _, value in values]):>12.6f}"
        lines.append(row)
    append_ranked(lines, "Most Replicated Positive Findings", findings, "positive")
    append_ranked(lines, "Most Replicated Negative Findings", findings, "negative")
    predictive = [finding for finding in findings if "Latent" in finding.name or "LateralBreakout" in finding.name]
    append_ranked(lines, "Most Predictive Acceptance Metrics", predictive, "magnitude")
    lines.extend(["\nCross-Instrument Research Notes", "==============================="])
    for name in ("AllBars:Context:DissipationOn:EffectSize", "AllBars:Context:CompressionOn:EffectSize", "AllBars:Latent:AcceptanceScore:Spearman", "AllBars:Latent:DominanceGradient:Spearman"):
        finding = next(item for item in findings if item.name == name)
        lines.append(f"- {name}: MeanMetric={mean([value for _, value in valid_values(finding)]):.6f}.")
    lines.append(f"- Findings excluded from replicated rankings because fewer than two instruments were valid: {sum(len(valid_values(item)) < 2 for item in findings)}.")
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
            Path("Evidence") / "Output" / study.instrument / f"AcceptanceAnatomy_{study.instrument}.txt",
            instrument_report(study),
        )
    report = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, report)
    print(report, end="")


if __name__ == "__main__":
    main()
