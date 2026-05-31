#!/usr/bin/env python3
"""Study direct evidence-state transitions across one or more instruments.

This research-only script compares prior-bar evidence state to current-bar
evidence state. It does not use a stage model, an OOE model, or trading logic.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


HORIZONS = (1, 3, 5, 10, 20)
MIN_RANKED_SAMPLES = 30
TOP_LIMIT = 25
CANONICAL_INSTRUMENTS = ("6E", "NQ", "CL")
AGGREGATE_OUTPUT = Path("Evidence/Output/RegimeTransition/RegimeTransition_All.txt")
REQUIRED_COLUMNS = {
    "BarIndex",
    "Time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "VolumePolarity",
    "ParticipationState",
    "AcceptanceState",
    "DissipationState",
    "CompressionState",
    "ExpansionState",
}


@dataclass(frozen=True)
class EvidenceBar:
    bar_index: int
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    polarity: str
    participation: str
    acceptance: str
    dissipation: str
    compression: str
    expansion: str


@dataclass(frozen=True)
class TransitionDefinition:
    name: str
    predicate: Callable[[EvidenceBar, EvidenceBar], bool]


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    rows: list[EvidenceBar]
    occurrences: dict[str, list[int]]
    fwd5_stats: dict[str, dict[str, float | int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study direct APVA evidence-state transitions across instruments."
    )
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def normalized(value: str | None) -> str:
    text = (value or "").strip()
    return "Other" if not text or text.lower() in {"none", "unknown"} else text


def on(value: str) -> bool:
    return normalized(value) not in {"Absent", "Other"}


def off(value: str) -> bool:
    return normalized(value) == "Absent"


def state_transition(attribute: str, prior_value: str, current_value: str) -> Callable[[EvidenceBar, EvidenceBar], bool]:
    return lambda prior, current: (
        getattr(prior, attribute) == prior_value and getattr(current, attribute) == current_value
    )


def transition_definitions() -> list[TransitionDefinition]:
    return [
        TransitionDefinition("Falling -> Rising", state_transition("participation", "Falling", "Rising")),
        TransitionDefinition("Rising -> Peak", state_transition("participation", "Rising", "Peak")),
        TransitionDefinition("Peak -> Falling", state_transition("participation", "Peak", "Falling")),
        TransitionDefinition("Peak -> Climactic", state_transition("participation", "Peak", "Climactic")),
        TransitionDefinition("Climactic -> Falling", state_transition("participation", "Climactic", "Falling")),
        TransitionDefinition("Contained -> Accepted", state_transition("acceptance", "Contained", "Accepted")),
        TransitionDefinition("Accepted -> Contained", state_transition("acceptance", "Accepted", "Contained")),
        TransitionDefinition("CompressionOn -> ExpansionOn", lambda prior, current: on(prior.compression) and on(current.expansion)),
        TransitionDefinition("ExpansionOn -> CompressionOn", lambda prior, current: on(prior.expansion) and on(current.compression)),
        TransitionDefinition("CompressionOff -> CompressionOn", lambda prior, current: off(prior.compression) and on(current.compression)),
        TransitionDefinition("ExpansionOff -> ExpansionOn", lambda prior, current: off(prior.expansion) and on(current.expansion)),
        TransitionDefinition("CompressionOn -> CompressionOff", lambda prior, current: on(prior.compression) and off(current.compression)),
        TransitionDefinition("ExpansionOn -> ExpansionOff", lambda prior, current: on(prior.expansion) and off(current.expansion)),
        TransitionDefinition("DissipationOff -> DissipationOn", lambda prior, current: off(prior.dissipation) and on(current.dissipation)),
        TransitionDefinition("DissipationOn -> DissipationOff", lambda prior, current: on(prior.dissipation) and off(current.dissipation)),
        TransitionDefinition("DissipationOn -> ExpansionOn", lambda prior, current: on(prior.dissipation) and on(current.expansion)),
        TransitionDefinition("DissipationOff -> ExpansionOn", lambda prior, current: off(prior.dissipation) and on(current.expansion)),
        TransitionDefinition("CompressionOn -> DissipationOn", lambda prior, current: on(prior.compression) and on(current.dissipation)),
        TransitionDefinition("Contained -> ExpansionOn", lambda prior, current: prior.acceptance == "Contained" and on(current.expansion)),
        TransitionDefinition("Accepted -> ExpansionOn", lambda prior, current: prior.acceptance == "Accepted" and on(current.expansion)),
        TransitionDefinition("Peak -> ExpansionOn", lambda prior, current: prior.participation == "Peak" and on(current.expansion)),
        TransitionDefinition("Falling -> ExpansionOn", lambda prior, current: prior.participation == "Falling" and on(current.expansion)),
        TransitionDefinition("Rising -> ExpansionOn", lambda prior, current: prior.participation == "Rising" and on(current.expansion)),
    ]


def require_columns(fieldnames: Iterable[str] | None) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(fieldnames or []))
    if missing:
        raise ValueError("Missing required evidence columns: " + ", ".join(missing))


def parse_float(raw: dict[str, str], column: str, line_number: int) -> float:
    text = (raw.get(column) or "").strip()
    try:
        return float(text)
    except ValueError as error:
        raise ValueError(f"Invalid {column} value at CSV line {line_number}: {text!r}") from error


def load_rows(path: Path) -> list[EvidenceBar]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing evidence CSV: {path}")
    rows: list[EvidenceBar] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames)
        for line_number, raw in enumerate(reader, start=2):
            if not any((value or "").strip() for value in raw.values()):
                continue
            try:
                bar_index = int((raw.get("BarIndex") or "").strip())
            except ValueError as error:
                raise ValueError(f"Invalid BarIndex at CSV line {line_number}") from error
            rows.append(
                EvidenceBar(
                    bar_index=bar_index,
                    time=(raw.get("Time") or "").strip(),
                    open=parse_float(raw, "Open", line_number),
                    high=parse_float(raw, "High", line_number),
                    low=parse_float(raw, "Low", line_number),
                    close=parse_float(raw, "Close", line_number),
                    volume=parse_float(raw, "Volume", line_number),
                    polarity=normalized(raw.get("VolumePolarity")),
                    participation=normalized(raw.get("ParticipationState")),
                    acceptance=normalized(raw.get("AcceptanceState")),
                    dissipation=normalized(raw.get("DissipationState")),
                    compression=normalized(raw.get("CompressionState")),
                    expansion=normalized(raw.get("ExpansionState")),
                )
            )
    return rows


def instrument_name(path: Path) -> str:
    parent = path.parent.name.strip()
    raw = parent if parent and parent.lower() != "evidence" else path.stem.split("_", 1)[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_") or "UnknownInstrument"


def direction_relative_return(rows: list[EvidenceBar], index: int, horizon: int) -> float | None:
    if index + horizon >= len(rows):
        return None
    if rows[index].polarity == "Black":
        return rows[index + horizon].close - rows[index].close
    if rows[index].polarity == "Red":
        return rows[index].close - rows[index + horizon].close
    return None


def summarize(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    continuation = sum(value > 0.0 for value in values)
    failure = sum(value < 0.0 for value in values)
    flat = count - continuation - failure
    denominator = count if count else 1
    return {
        "Count": count,
        "MeanDRFwd": statistics.fmean(values) if values else 0.0,
        "MedianDRFwd": statistics.median(values) if values else 0.0,
        "ContinuationRate": continuation / denominator,
        "FailureRate": failure / denominator,
        "FlatRate": flat / denominator,
    }


def transition_values(rows: list[EvidenceBar], indexes: list[int], horizon: int) -> list[float]:
    values = [direction_relative_return(rows, index, horizon) for index in indexes]
    return [value for value in values if value is not None]


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    definitions = transition_definitions()
    occurrences = {
        definition.name: [
            index
            for index in range(1, len(rows))
            if definition.predicate(rows[index - 1], rows[index])
        ]
        for definition in definitions
    }
    return InstrumentStudy(
        instrument=instrument_name(path),
        path=path,
        rows=rows,
        occurrences=occurrences,
        fwd5_stats={
            name: summarize(transition_values(rows, indexes, 5))
            for name, indexes in occurrences.items()
        },
    )


def write_text(path: Path, report: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def append_ranked(lines: list[str], title: str, ranked: list[tuple[str, dict[str, float | int]]]) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(f"{'Rank':>4} {'Transition':<36} {'Count':>8} {'MeanDRFwd5':>12} {'ContRate5':>10} {'FailRate5':>10}")
    for rank, (name, stats) in enumerate(ranked[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {name:<36} {stats['Count']:>8} {stats['MeanDRFwd']:>12.6f} "
            f"{stats['ContinuationRate']:>9.2%} {stats['FailureRate']:>9.2%}"
        )


def transition_note(study: InstrumentStudy, name: str) -> str:
    stats = study.fwd5_stats[name]
    return (
        f"{name}: MeanDRFwd5={stats['MeanDRFwd']:.6f}, "
        f"ContinuationRate5={stats['ContinuationRate']:.2%}, n={stats['Count']}"
    )


def instrument_report(study: InstrumentStudy) -> str:
    lines = [
        f"APVA Regime Transition Study v0.1 - {study.instrument}",
        "=" * (36 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"Input: {study.path}",
        f"Total rows: {len(study.rows)}",
        f"Valid polarity rows: {sum(row.polarity in {'Black', 'Red'} for row in study.rows)}",
        f"Total studied transition observations: {sum(len(indexes) for indexes in study.occurrences.values())}",
        "Transitions compare prior-bar source state to current-bar target state.",
        "CompressionOn -> ExpansionOn is recorded once although it appears in two requested categories.",
        "\nTransition Consequence Table",
        "============================",
        f"{'Transition':<36} {'Count':>8} {'MeanDRFwd5':>12} {'MedianDRFwd5':>14} "
        f"{'ContRate5':>10} {'FailRate5':>10} {'FlatRate5':>10}",
    ]
    for name, stats in study.fwd5_stats.items():
        lines.append(
            f"{name:<36} {stats['Count']:>8} {stats['MeanDRFwd']:>12.6f} "
            f"{stats['MedianDRFwd']:>14.6f} {stats['ContinuationRate']:>9.2%} "
            f"{stats['FailureRate']:>9.2%} {stats['FlatRate']:>9.2%}"
        )
    lines.extend(["\nAll-Horizon Appendix", "===================="])
    for name, indexes in study.occurrences.items():
        lines.extend([f"\n{name}", "-" * len(name)])
        lines.append(f"{'Horizon':<8} {'Count':>8} {'MeanDRFwd':>12} {'Median':>12} {'ContRate':>10} {'FailRate':>10} {'FlatRate':>10}")
        for horizon in HORIZONS:
            stats = summarize(transition_values(study.rows, indexes, horizon))
            lines.append(
                f"DRFwd{horizon:<2} {stats['Count']:>8} {stats['MeanDRFwd']:>12.6f} "
                f"{stats['MedianDRFwd']:>12.6f} {stats['ContinuationRate']:>9.2%} "
                f"{stats['FailureRate']:>9.2%} {stats['FlatRate']:>9.2%}"
            )
    eligible = [(name, stats) for name, stats in study.fwd5_stats.items() if stats["Count"] >= MIN_RANKED_SAMPLES]
    append_ranked(lines, "Top 25 Transitions by MeanDRFwd5", sorted(eligible, key=lambda item: (-item[1]["MeanDRFwd"], item[0])))
    append_ranked(lines, "Bottom 25 Transitions by MeanDRFwd5", sorted(eligible, key=lambda item: (item[1]["MeanDRFwd"], item[0])))
    append_ranked(lines, "Top 25 Transitions by ContinuationRate5", sorted(eligible, key=lambda item: (-item[1]["ContinuationRate"], item[0])))
    append_ranked(lines, "Top 25 Transitions by FailureRate5", sorted(eligible, key=lambda item: (-item[1]["FailureRate"], item[0])))
    lines.extend(["\nResearch Notes", "=============="])
    if eligible:
        strongest = max(eligible, key=lambda item: (item[1]["MeanDRFwd"], item[0]))
        weakest = min(eligible, key=lambda item: (item[1]["MeanDRFwd"], item[0]))
        lines.append(f"- Strongest positive MeanDRFwd5: {strongest[0]} ({strongest[1]['MeanDRFwd']:.6f}, n={strongest[1]['Count']}).")
        lines.append(f"- Strongest negative MeanDRFwd5: {weakest[0]} ({weakest[1]['MeanDRFwd']:.6f}, n={weakest[1]['Count']}).")
    for name in ("DissipationOn -> ExpansionOn", "CompressionOn -> ExpansionOn", "Contained -> Accepted", "Falling -> Rising"):
        lines.append(f"- {transition_note(study, name)}.")
    lines.append(f"- Transitions skipped from ranked tables because Count < {MIN_RANKED_SAMPLES}: {len(study.fwd5_stats) - len(eligible)}.")
    return "\n".join(lines) + "\n"


def instrument_columns(studies: list[InstrumentStudy]) -> list[str]:
    available = [study.instrument for study in studies]
    return list(CANONICAL_INSTRUMENTS) + sorted(set(available) - set(CANONICAL_INSTRUMENTS))


def aggregate_rows(studies: list[InstrumentStudy]) -> list[dict[str, object]]:
    by_instrument = {study.instrument: study for study in studies}
    rows: list[dict[str, object]] = []
    for definition in transition_definitions():
        valid = []
        values: dict[str, dict[str, float | int]] = {}
        for instrument, study in by_instrument.items():
            stats = study.fwd5_stats[definition.name]
            values[instrument] = stats
            if stats["Count"] >= MIN_RANKED_SAMPLES:
                valid.append(stats)
        rows.append(
            {
                "name": definition.name,
                "values": values,
                "instrument_count": len(studies),
                "valid_count": len(valid),
                "positive_mean_count": sum(stats["MeanDRFwd"] > 0.0 for stats in valid),
                "positive_cont_count": sum(stats["ContinuationRate"] > 0.50 for stats in valid),
                "negative_mean_count": sum(stats["MeanDRFwd"] < 0.0 for stats in valid),
                "negative_cont_count": sum(stats["ContinuationRate"] < 0.50 for stats in valid),
                "mean_cont": statistics.fmean(float(stats["ContinuationRate"]) for stats in valid) if valid else 0.0,
                "mean_mean": statistics.fmean(float(stats["MeanDRFwd"]) for stats in valid) if valid else 0.0,
            }
        )
    return rows


def append_aggregate_ranked(lines: list[str], title: str, rows: list[dict[str, object]]) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(f"{'Rank':>4} {'Transition':<36} {'Valid':>6} {'PosCont':>8} {'NegCont':>8} {'MeanCont':>10} {'MeanMean5':>12}")
    for rank, row in enumerate(rows[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {str(row['name']):<36} {int(row['valid_count']):>6} "
            f"{int(row['positive_cont_count']):>8} {int(row['negative_cont_count']):>8} "
            f"{float(row['mean_cont']):>9.2%} {float(row['mean_mean']):>12.6f}"
        )


def aggregate_note(rows: list[dict[str, object]], name: str) -> str:
    row = next(item for item in rows if item["name"] == name)
    return (
        f"{name}: valid instruments={row['valid_count']}, "
        f"positive continuation instruments={row['positive_cont_count']}, "
        f"negative continuation instruments={row['negative_cont_count']}, "
        f"MeanContRateAcrossValid={float(row['mean_cont']):.2%}"
    )


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    columns = instrument_columns(studies)
    rows = aggregate_rows(studies)
    lines = [
        "APVA Regime Transition Study v0.1 - Cross-Instrument Aggregate",
        "==============================================================",
        f"Input instruments: {', '.join(study.instrument for study in studies)}",
        f"Ranked replication minimum: Count >= {MIN_RANKED_SAMPLES} in at least two instruments.",
        "No stage model or OOE model is used.",
        "\nCross-Instrument Transition Table",
        "=================================",
    ]
    header = f"{'Transition':<36} {'Inst':>4} {'GE30':>5}"
    for instrument in columns:
        header += f" {('Count_' + instrument):>10} {('Cont_' + instrument):>9} {('Mean_' + instrument):>11}"
    header += f" {'PosMean':>7} {'PosCont':>7} {'NegMean':>7} {'NegCont':>7} {'MeanCont':>9} {'MeanMean5':>11}"
    lines.append(header)
    for row in rows:
        text = f"{str(row['name']):<36} {int(row['instrument_count']):>4} {int(row['valid_count']):>5}"
        values = row["values"]
        for instrument in columns:
            stats = values.get(instrument)
            if stats is None:
                text += f" {'NA':>10} {'NA':>9} {'NA':>11}"
            else:
                text += f" {int(stats['Count']):>10} {float(stats['ContinuationRate']):>8.2%} {float(stats['MeanDRFwd']):>11.6f}"
        text += (
            f" {int(row['positive_mean_count']):>7} {int(row['positive_cont_count']):>7} "
            f"{int(row['negative_mean_count']):>7} {int(row['negative_cont_count']):>7} "
            f"{float(row['mean_cont']):>8.2%} {float(row['mean_mean']):>11.6f}"
        )
        lines.append(text)
    eligible = [row for row in rows if row["valid_count"] >= 2]
    append_aggregate_ranked(lines, "Most Replicated Positive Transitions", sorted(eligible, key=lambda row: (-row["positive_cont_count"], -row["mean_cont"], row["name"])))
    append_aggregate_ranked(lines, "Most Replicated Negative Transitions", sorted(eligible, key=lambda row: (-row["negative_cont_count"], row["mean_cont"], row["name"])))
    all_valid = [row for row in rows if row["valid_count"] == len(studies) and len(studies) >= 3]
    append_aggregate_ranked(lines, "Most Stable Positive Transitions", [row for row in sorted(all_valid, key=lambda row: (-row["mean_cont"], row["name"])) if row["positive_cont_count"] == len(studies)])
    append_aggregate_ranked(lines, "Most Stable Negative Transitions", [row for row in sorted(all_valid, key=lambda row: (row["mean_cont"], row["name"])) if row["negative_cont_count"] == len(studies)])
    lines.extend(["\nCross-Instrument Research Notes", "==============================="])
    positive = [row for row in eligible if row["positive_cont_count"] >= 2]
    negative = [row for row in eligible if row["negative_cont_count"] >= 2]
    lines.append(f"- Transitions replicating positively in at least two valid instruments: {', '.join(str(row['name']) for row in positive) or 'none'}.")
    lines.append(f"- Transitions replicating negatively in at least two valid instruments: {', '.join(str(row['name']) for row in negative) or 'none'}.")
    for name in ("DissipationOn -> ExpansionOn", "CompressionOn -> ExpansionOn", "Contained -> Accepted"):
        lines.append(f"- {aggregate_note(rows, name)}.")
    lines.append("- Stability versus single evidence events is not computed in this transition-only report.")
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
            Path("Evidence") / "Output" / study.instrument / f"RegimeTransition_{study.instrument}.txt",
            instrument_report(study),
        )
    aggregate = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, aggregate)
    print(aggregate, end="")


if __name__ == "__main__":
    main()
