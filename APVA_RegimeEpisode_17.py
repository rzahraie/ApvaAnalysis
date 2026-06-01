#!/usr/bin/env python3
"""Study persistent multi-bar APVA evidence-regime episodes.

This research-only script measures source clusters followed by target clusters.
It does not use a stage model, optimize parameters, or create trading logic.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from APVA_BreakoutAlignment_09 import classify_alignment
from APVA_RegimeTransition_16 import (
    CANONICAL_INSTRUMENTS,
    EvidenceBar,
    HORIZONS,
    MIN_RANKED_SAMPLES,
    TOP_LIMIT,
    direction_relative_return,
    instrument_name,
    load_rows,
    summarize,
    write_text,
)


AGGREGATE_OUTPUT = Path("Evidence/Output/RegimeEpisode/RegimeEpisode_All.txt")


@dataclass(frozen=True)
class EpisodeFamily:
    name: str
    source: Callable[[EvidenceBar], bool]
    target: Callable[[EvidenceBar], bool]
    alignment_direction: str


@dataclass(frozen=True)
class Episode:
    family: str
    anchor_index: int
    source_count: int
    target_count: int
    source_strength: str
    target_strength: str
    quality: str
    alignment: str | None


@dataclass(frozen=True)
class InstrumentStudy:
    instrument: str
    path: Path
    rows: list[EvidenceBar]
    episodes: list[Episode]
    family_stats: dict[str, dict[str, float | int]]
    quality_stats: dict[tuple[str, str], dict[str, float | int]]
    alignment_stats: dict[tuple[str, str], dict[str, float | int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study persistent APVA evidence-regime episodes across instruments."
    )
    parser.add_argument("input_paths", type=Path, nargs="+", help="One or more evidence CSV files.")
    return parser.parse_args()


def on(value: str) -> bool:
    return value not in {"Absent", "Other"}


def episode_families() -> list[EpisodeFamily]:
    dissipation = lambda row: on(row.dissipation)
    compression = lambda row: on(row.compression)
    expansion = lambda row: on(row.expansion)
    contained = lambda row: row.acceptance == "Contained"
    accepted = lambda row: row.acceptance == "Accepted"
    falling = lambda row: row.participation == "Falling"
    rising = lambda row: row.participation == "Rising"
    peak_like = lambda row: row.participation in {"Peak", "Climactic"}
    return [
        EpisodeFamily("Dissipation Cluster -> Expansion Cluster", dissipation, expansion, "Up"),
        EpisodeFamily("Compression Cluster -> Expansion Cluster", compression, expansion, "Up"),
        EpisodeFamily("Contained Cluster -> Accepted Cluster", contained, accepted, "Up"),
        EpisodeFamily("Falling Cluster -> Rising Cluster", falling, rising, "Up"),
        EpisodeFamily("Peak/Climactic Cluster -> Expansion Cluster", peak_like, expansion, "Up"),
        EpisodeFamily("Dissipation Cluster -> Accepted Cluster", dissipation, accepted, "Up"),
        EpisodeFamily("Compression Cluster -> Accepted Cluster", compression, accepted, "Up"),
        EpisodeFamily("Contained Cluster -> Expansion Cluster", contained, expansion, "Up"),
        EpisodeFamily("Falling Cluster -> Expansion Cluster", falling, expansion, "Up"),
        EpisodeFamily("Peak/Climactic Cluster -> Accepted Cluster", peak_like, accepted, "Up"),
    ]


def find_episodes(rows: list[EvidenceBar]) -> list[Episode]:
    episodes: list[Episode] = []
    for family in episode_families():
        for anchor in range(4, len(rows) - 2):
            source_count = sum(family.source(row) for row in rows[anchor - 4 : anchor])
            target_count = sum(family.target(row) for row in rows[anchor : anchor + 3])
            if source_count < 3 or target_count < 2:
                continue
            source_strength = "Strong" if source_count == 4 else "Weak"
            target_strength = "Strong" if target_count == 3 else "Weak"
            quality = "Strong" if source_strength == "Strong" and target_strength == "Strong" else "Weak"
            episodes.append(
                Episode(
                    family=family.name,
                    anchor_index=anchor,
                    source_count=source_count,
                    target_count=target_count,
                    source_strength=source_strength,
                    target_strength=target_strength,
                    quality=quality,
                    alignment=classify_alignment(family.alignment_direction, rows[anchor].polarity),
                )
            )
    return episodes


def values_for(rows: list[EvidenceBar], episodes: list[Episode], horizon: int) -> list[float]:
    values = [direction_relative_return(rows, episode.anchor_index, horizon) for episode in episodes]
    return [value for value in values if value is not None]


def stats_for(rows: list[EvidenceBar], episodes: list[Episode], horizon: int = 5) -> dict[str, float | int]:
    return summarize(values_for(rows, episodes, horizon))


def study_instrument(path: Path) -> InstrumentStudy:
    rows = load_rows(path)
    episodes = find_episodes(rows)
    family_stats = {}
    quality_stats = {}
    alignment_stats = {}
    for family in episode_families():
        selected = [episode for episode in episodes if episode.family == family.name]
        family_stats[family.name] = stats_for(rows, selected)
        for quality in ("Strong", "Weak"):
            quality_stats[(family.name, quality)] = stats_for(
                rows, [episode for episode in selected if episode.quality == quality]
            )
        for alignment in ("Aligned", "Opposed"):
            alignment_stats[(family.name, alignment)] = stats_for(
                rows, [episode for episode in selected if episode.alignment == alignment]
            )
    return InstrumentStudy(
        instrument_name(path), path, rows, episodes, family_stats, quality_stats, alignment_stats
    )


def append_ranked(
    lines: list[str],
    title: str,
    items: list[tuple[str, dict[str, float | int]]],
) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(f"{'Rank':>4} {'EpisodeFamily':<48} {'Count':>8} {'MeanDRFwd5':>12} {'ContRate5':>10}")
    for rank, (name, stats) in enumerate(items[:TOP_LIMIT], start=1):
        lines.append(f"{rank:>4} {name:<48} {stats['Count']:>8} {stats['MeanDRFwd']:>12.6f} {stats['ContinuationRate']:>9.2%}")


def append_advantage_ranked(
    lines: list[str],
    title: str,
    items: list[tuple[str, dict[str, float | int], dict[str, float | int]]],
) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(f"{'Rank':>4} {'EpisodeFamily':<48} {'LeftCount':>10} {'RightCount':>11} {'DeltaMean5':>12} {'DeltaCont5':>11}")
    for rank, (name, left, right) in enumerate(items[:TOP_LIMIT], start=1):
        lines.append(
            f"{rank:>4} {name:<48} {left['Count']:>10} {right['Count']:>11} "
            f"{float(left['MeanDRFwd']) - float(right['MeanDRFwd']):>12.6f} "
            f"{float(left['ContinuationRate']) - float(right['ContinuationRate']):>10.2%}"
        )


def family_note(study: InstrumentStudy, family: str) -> str:
    stats = study.family_stats[family]
    return f"{family}: MeanDRFwd5={stats['MeanDRFwd']:.6f}, ContinuationRate5={stats['ContinuationRate']:.2%}, n={stats['Count']}"


def instrument_report(study: InstrumentStudy) -> str:
    lines = [
        f"APVA Regime Episode Study v0.1 - {study.instrument}",
        "=" * (33 + len(study.instrument)),
        f"Instrument: {study.instrument}",
        f"InputPath: {study.path}",
        f"TotalRows: {len(study.rows)}",
        f"ValidPolarityRows: {sum(row.polarity in {'Black', 'Red'} for row in study.rows)}",
        f"TotalEpisodes: {len(study.episodes)}",
        "Anchor: first bar of the three-bar target window.",
        "Alignment proxy: Black anchor polarity is Aligned; Red anchor polarity is Opposed.",
        "\nEpisode Counts by Family",
        "========================",
    ]
    for family in episode_families():
        lines.append(f"{family.name:<48} {sum(episode.family == family.name for episode in study.episodes):>8}")
    lines.extend(["\nMain Episode Table", "=================="])
    lines.append(f"{'EpisodeFamily':<48} {'Count':>8} {'MeanDRFwd5':>12} {'MedianDRFwd5':>14} {'ContRate5':>10} {'FailRate5':>10} {'FlatRate5':>10}")
    for family, stats in study.family_stats.items():
        lines.append(f"{family:<48} {stats['Count']:>8} {stats['MeanDRFwd']:>12.6f} {stats['MedianDRFwd']:>14.6f} {stats['ContinuationRate']:>9.2%} {stats['FailureRate']:>9.2%} {stats['FlatRate']:>9.2%}")
    lines.extend(["\nAll-Horizon Appendix", "===================="])
    for family in episode_families():
        selected = [episode for episode in study.episodes if episode.family == family.name]
        lines.extend([f"\n{family.name}", "-" * len(family.name)])
        lines.append(f"{'Horizon':<8} {'Count':>8} {'MeanDRFwd':>12} {'Median':>12} {'ContRate':>10} {'FailRate':>10} {'FlatRate':>10}")
        for horizon in HORIZONS:
            stats = stats_for(study.rows, selected, horizon)
            lines.append(f"DRFwd{horizon:<2} {stats['Count']:>8} {stats['MeanDRFwd']:>12.6f} {stats['MedianDRFwd']:>12.6f} {stats['ContinuationRate']:>9.2%} {stats['FailureRate']:>9.2%} {stats['FlatRate']:>9.2%}")
    lines.extend(["\nQuality Breakdown", "================="])
    lines.append(f"{'EpisodeFamily':<48} {'Quality':<8} {'Count':>8} {'MeanDRFwd5':>12} {'ContRate5':>10}")
    for (family, quality), stats in study.quality_stats.items():
        lines.append(f"{family:<48} {quality:<8} {stats['Count']:>8} {stats['MeanDRFwd']:>12.6f} {stats['ContinuationRate']:>9.2%}")
    lines.extend(["\nAlignment Breakdown", "==================="])
    lines.append(f"{'EpisodeFamily':<48} {'Alignment':<8} {'Count':>8} {'MeanDRFwd5':>12} {'ContRate5':>10}")
    for (family, alignment), stats in study.alignment_stats.items():
        lines.append(f"{family:<48} {alignment:<8} {stats['Count']:>8} {stats['MeanDRFwd']:>12.6f} {stats['ContinuationRate']:>9.2%}")
    eligible = [(name, stats) for name, stats in study.family_stats.items() if stats["Count"] >= MIN_RANKED_SAMPLES]
    append_ranked(lines, "Best Episode Families by MeanDRFwd5", sorted(eligible, key=lambda item: (-item[1]["MeanDRFwd"], item[0])))
    append_ranked(lines, "Worst Episode Families by MeanDRFwd5", sorted(eligible, key=lambda item: (item[1]["MeanDRFwd"], item[0])))
    append_ranked(lines, "Best Episode Families by ContinuationRate5", sorted(eligible, key=lambda item: (-item[1]["ContinuationRate"], item[0])))
    append_ranked(lines, "Worst Episode Families by ContinuationRate5", sorted(eligible, key=lambda item: (item[1]["ContinuationRate"], item[0])))
    quality = [(family.name, study.quality_stats[(family.name, "Strong")], study.quality_stats[(family.name, "Weak")]) for family in episode_families()]
    quality = [item for item in quality if item[1]["Count"] >= MIN_RANKED_SAMPLES and item[2]["Count"] >= MIN_RANKED_SAMPLES]
    alignment = [(family.name, study.alignment_stats[(family.name, "Aligned")], study.alignment_stats[(family.name, "Opposed")]) for family in episode_families()]
    alignment = [item for item in alignment if item[1]["Count"] >= MIN_RANKED_SAMPLES and item[2]["Count"] >= MIN_RANKED_SAMPLES]
    append_advantage_ranked(lines, "Strong vs Weak Quality Advantage", sorted(quality, key=lambda item: (-(item[1]["ContinuationRate"] - item[2]["ContinuationRate"]), item[0])))
    append_advantage_ranked(lines, "Aligned vs Opposed Advantage", sorted(alignment, key=lambda item: (-(item[1]["ContinuationRate"] - item[2]["ContinuationRate"]), item[0])))
    lines.extend(["\nResearch Notes", "=============="])
    if eligible:
        best = max(eligible, key=lambda item: (item[1]["MeanDRFwd"], item[0]))
        worst = min(eligible, key=lambda item: (item[1]["MeanDRFwd"], item[0]))
        lines.append(f"- Best episode family by MeanDRFwd5: {best[0]} ({best[1]['MeanDRFwd']:.6f}, n={best[1]['Count']}).")
        lines.append(f"- Worst episode family by MeanDRFwd5: {worst[0]} ({worst[1]['MeanDRFwd']:.6f}, n={worst[1]['Count']}).")
    if quality:
        mean_quality = statistics.fmean(float(left["ContinuationRate"]) - float(right["ContinuationRate"]) for _, left, right in quality)
        lines.append(f"- Mean Strong-minus-Weak DeltaContinuationRate5 across eligible families: {mean_quality:.2%}.")
    if alignment:
        mean_alignment = statistics.fmean(float(left["ContinuationRate"]) - float(right["ContinuationRate"]) for _, left, right in alignment)
        lines.append(f"- Mean Aligned-minus-Opposed DeltaContinuationRate5 across eligible families: {mean_alignment:.2%}.")
    for family in ("Dissipation Cluster -> Expansion Cluster", "Compression Cluster -> Expansion Cluster", "Contained Cluster -> Accepted Cluster"):
        lines.append(f"- {family_note(study, family)}.")
    lines.append(f"- Families skipped from main ranked tables because Count < {MIN_RANKED_SAMPLES}: {len(study.family_stats) - len(eligible)}.")
    return "\n".join(lines) + "\n"


def instrument_columns(studies: list[InstrumentStudy]) -> list[str]:
    available = [study.instrument for study in studies]
    return list(CANONICAL_INSTRUMENTS) + sorted(set(available) - set(CANONICAL_INSTRUMENTS))


def aggregate_rows(studies: list[InstrumentStudy]) -> list[dict[str, object]]:
    rows = []
    for family in episode_families():
        values = {study.instrument: study.family_stats[family.name] for study in studies}
        valid = [stats for stats in values.values() if stats["Count"] >= MIN_RANKED_SAMPLES]
        quality_deltas = []
        alignment_deltas = []
        for study in studies:
            strong = study.quality_stats[(family.name, "Strong")]
            weak = study.quality_stats[(family.name, "Weak")]
            if strong["Count"] >= MIN_RANKED_SAMPLES and weak["Count"] >= MIN_RANKED_SAMPLES:
                quality_deltas.append(float(strong["ContinuationRate"]) - float(weak["ContinuationRate"]))
            aligned = study.alignment_stats[(family.name, "Aligned")]
            opposed = study.alignment_stats[(family.name, "Opposed")]
            if aligned["Count"] >= MIN_RANKED_SAMPLES and opposed["Count"] >= MIN_RANKED_SAMPLES:
                alignment_deltas.append(float(aligned["ContinuationRate"]) - float(opposed["ContinuationRate"]))
        rows.append({
            "family": family.name,
            "values": values,
            "valid_count": len(valid),
            "positive_count": sum(stats["ContinuationRate"] > 0.50 for stats in valid),
            "negative_count": sum(stats["ContinuationRate"] < 0.50 for stats in valid),
            "mean_cont": statistics.fmean(float(stats["ContinuationRate"]) for stats in valid) if valid else 0.0,
            "mean_mean": statistics.fmean(float(stats["MeanDRFwd"]) for stats in valid) if valid else 0.0,
            "quality_delta": statistics.fmean(quality_deltas) if quality_deltas else 0.0,
            "alignment_delta": statistics.fmean(alignment_deltas) if alignment_deltas else 0.0,
        })
    return rows


def append_aggregate_ranked(lines: list[str], title: str, rows: list[dict[str, object]]) -> None:
    lines.extend([f"\n{title}", "=" * len(title)])
    lines.append(f"{'Rank':>4} {'EpisodeFamily':<48} {'GE30':>5} {'Pos':>4} {'Neg':>4} {'MeanCont':>10} {'MeanMean5':>12} {'QualityDelta':>13} {'AlignDelta':>11}")
    for rank, row in enumerate(rows[:TOP_LIMIT], start=1):
        lines.append(f"{rank:>4} {str(row['family']):<48} {int(row['valid_count']):>5} {int(row['positive_count']):>4} {int(row['negative_count']):>4} {float(row['mean_cont']):>9.2%} {float(row['mean_mean']):>12.6f} {float(row['quality_delta']):>12.2%} {float(row['alignment_delta']):>10.2%}")


def aggregate_note(rows: list[dict[str, object]], family: str) -> str:
    row = next(item for item in rows if item["family"] == family)
    return f"{family}: valid instruments={row['valid_count']}, positive={row['positive_count']}, negative={row['negative_count']}, MeanContRate={float(row['mean_cont']):.2%}"


def aggregate_report(studies: list[InstrumentStudy]) -> str:
    columns = instrument_columns(studies)
    rows = aggregate_rows(studies)
    lines = [
        "APVA Regime Episode Study v0.1 - Cross-Instrument Aggregate",
        "===========================================================",
        f"Input instruments: {', '.join(study.instrument for study in studies)}",
        f"Replication minimum: Count >= {MIN_RANKED_SAMPLES} in at least two instruments.",
        "No stage model is used.",
        "\nCross-Instrument Episode Table",
        "==============================",
    ]
    header = f"{'EpisodeFamily':<48}"
    for instrument in columns:
        header += f" {('Count_' + instrument):>9} {('Cont_' + instrument):>8} {('Mean_' + instrument):>11}"
    header += f" {'GE30':>5} {'Pos':>4} {'Neg':>4} {'MeanCont':>9} {'MeanMean5':>11} {'QualityD':>9} {'AlignD':>8}"
    lines.append(header)
    for row in rows:
        text = f"{str(row['family']):<48}"
        values = row["values"]
        for instrument in columns:
            stats = values.get(instrument)
            if stats is None:
                text += f" {'NA':>9} {'NA':>8} {'NA':>11}"
            else:
                text += f" {int(stats['Count']):>9} {float(stats['ContinuationRate']):>7.2%} {float(stats['MeanDRFwd']):>11.6f}"
        text += f" {int(row['valid_count']):>5} {int(row['positive_count']):>4} {int(row['negative_count']):>4} {float(row['mean_cont']):>8.2%} {float(row['mean_mean']):>11.6f} {float(row['quality_delta']):>8.2%} {float(row['alignment_delta']):>7.2%}"
        lines.append(text)
    eligible = [row for row in rows if row["valid_count"] >= 2]
    append_aggregate_ranked(lines, "Most Replicated Positive Episodes", sorted(eligible, key=lambda row: (-row["positive_count"], -row["mean_cont"], row["family"])))
    append_aggregate_ranked(lines, "Most Replicated Negative Episodes", sorted(eligible, key=lambda row: (-row["negative_count"], row["mean_cont"], row["family"])))
    all_valid = [row for row in rows if row["valid_count"] == len(studies) and len(studies) >= 3]
    append_aggregate_ranked(lines, "Most Stable Positive Episodes", [row for row in sorted(all_valid, key=lambda row: (-row["mean_cont"], row["family"])) if row["positive_count"] == len(studies)])
    append_aggregate_ranked(lines, "Most Stable Negative Episodes", [row for row in sorted(all_valid, key=lambda row: (row["mean_cont"], row["family"])) if row["negative_count"] == len(studies)])
    append_aggregate_ranked(lines, "Strongest Quality-Sensitive Episodes", sorted(eligible, key=lambda row: (-abs(row["quality_delta"]), row["family"])))
    append_aggregate_ranked(lines, "Strongest Alignment-Sensitive Episodes", sorted(eligible, key=lambda row: (-abs(row["alignment_delta"]), row["family"])))
    lines.extend(["\nCross-Instrument Research Notes", "==============================="])
    positive = [row for row in eligible if row["positive_count"] >= 2]
    negative = [row for row in eligible if row["negative_count"] >= 2]
    lines.append(f"- Episodes replicating positively in at least two valid instruments: {', '.join(str(row['family']) for row in positive) or 'none'}.")
    lines.append(f"- Episodes replicating negatively in at least two valid instruments: {', '.join(str(row['family']) for row in negative) or 'none'}.")
    if eligible:
        general = max(eligible, key=lambda row: (row["positive_count"], row["mean_cont"], row["family"]))
        least = max(eligible, key=lambda row: (row["negative_count"], -row["mean_cont"], row["family"]))
        lines.append(f"- Most market-general positive episode by replication metrics: {general['family']}.")
        lines.append(f"- Least market-general episode by negative replication metrics: {least['family']}.")
    for family in ("Dissipation Cluster -> Expansion Cluster", "Compression Cluster -> Expansion Cluster", "Contained Cluster -> Accepted Cluster"):
        lines.append(f"- {aggregate_note(rows, family)}.")
    lines.append("- Replication versus single transitions is not computed inside this episode-only report.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    studies = [study_instrument(path) for path in args.input_paths]
    seen: set[str] = set()
    for study in studies:
        if study.instrument in seen:
            raise ValueError(f"Duplicate instrument input: {study.instrument}")
        seen.add(study.instrument)
        write_text(Path("Evidence") / "Output" / study.instrument / f"RegimeEpisode_{study.instrument}.txt", instrument_report(study))
    aggregate = aggregate_report(studies)
    write_text(AGGREGATE_OUTPUT, aggregate)
    print(aggregate, end="")


if __name__ == "__main__":
    main()
