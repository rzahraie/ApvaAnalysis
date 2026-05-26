#!/usr/bin/env python3
"""Create descriptive cross-era reports for frozen APVA candidates.

This reads completed fixed-candidate extended validation outputs. It does not
construct entries, alter candidate rules, tune thresholds, or search for new
topologies.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


FROZEN_CANDIDATES = [
    "RRCCC",
    "CCRRR",
    "PriorSlope_DominantPressureValue_Q3",
]
VALIDATION_MODES = ["Reference", "Spacing_10", "Spacing_20"]
ERA_LABELS = {
    "2020": "COVID / crisis",
    "2022": "tightening / bear trend",
    "2024": "modern mixed regime",
}
SUMMARY_COLUMNS = [
    "Dataset",
    "RegimeCode",
    "Regime",
    "ValidationMode",
    "Candidate",
    "Count",
    "ES_Count",
    "NQ_Count",
    "Mean",
    "Median",
    "TStat",
    "ProfitFactor",
    "WinRate",
    "StopRate",
    "PositiveBlockFraction",
    "InstrumentWeekPositiveFraction",
    "MaxSingleBlockContributionFraction",
    "ES_Mean",
    "NQ_Mean",
    "ValidationPass",
    "ValidationStatus",
]
OPTIONAL_AGGREGATION_COLUMNS = ["Sum", "GrossWin", "GrossLoss"]


def infer_regime(dataset: str) -> tuple[str, str]:
    """Parse an era from dataset naming without restricting future year tags."""
    stem = Path(str(dataset)).stem.lower()
    date_tags = re.findall(r"(?:19|20)\d{2}", stem)
    year = date_tags[-1] if date_tags else ""
    if year:
        label = ERA_LABELS.get(year, f"{year} observed regime")
        return year, f"{year} | {label}"
    if re.search(r"(?:^|_)v\d+(?:_|$)", stem):
        return "canonical", "canonical | original regime"
    if "generated" in stem:
        return "generated", "generated | legacy generated regime"
    normalized = stem.replace("apva_forward_signed_return_dataset_", "").replace("_", " ")
    return "unclassified", f"unclassified | {normalized}"


def annotate_regime(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    parsed = out["Dataset"].map(infer_regime)
    out.insert(1, "RegimeCode", parsed.map(lambda item: item[0]))
    out.insert(2, "Regime", parsed.map(lambda item: item[1]))
    return out


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def read_sources(indir: Path) -> dict[str, pd.DataFrame]:
    names = [
        "extended_entries.csv",
        "extended_dataset_candidate_summary.csv",
        "extended_block_summary.csv",
        "extended_instrument_summary.csv",
        "extended_candidate_summary.csv",
        "extended_scorecard.csv",
    ]
    missing = [name for name in names if not (indir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing extended validation output files: {missing}")
    return {name: pd.read_csv(indir / name) for name in names}


def regime_summary(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    annotated = annotate_regime(dataset_summary)
    missing = [column for column in SUMMARY_COLUMNS if column not in annotated]
    if missing:
        raise RuntimeError(f"Dataset candidate summary is missing columns: {missing}")
    optional = [column for column in OPTIONAL_AGGREGATION_COLUMNS if column in annotated]
    return annotated[SUMMARY_COLUMNS + optional].sort_values(
        ["RegimeCode", "Dataset", "ValidationMode", "Candidate"]
    ).reset_index(drop=True)


def regime_blocks(blocks: pd.DataFrame) -> pd.DataFrame:
    dataset_blocks = blocks.loc[blocks["Scope"].eq("Dataset")].copy()
    annotated = annotate_regime(dataset_blocks)
    return annotated.sort_values(
        ["RegimeCode", "Dataset", "ValidationMode", "Candidate", "RegimeWeek"]
    ).reset_index(drop=True)


def contribution_rows(
    entries: pd.DataFrame,
    dimension: str,
    groups: list[str],
    label: str,
) -> pd.DataFrame:
    valid = entries.loc[entries["Candidate"].isin(FROZEN_CANDIDATES)].copy()
    grouped = (
        valid.groupby(groups + [dimension], dropna=False)["NormalizedPolicyOutcome"]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={"count": "Count", "sum": "NetContribution"})
    )
    totals = grouped.groupby(groups)["NetContribution"].transform("sum")
    absolute_totals = grouped.groupby(groups)["NetContribution"].transform(lambda values: values.abs().sum())
    grouped["DiagnosticDimension"] = label
    grouped["DiagnosticValue"] = grouped[dimension].astype(str)
    grouped["NetContributionShare"] = grouped["NetContribution"] / totals.replace(0, np.nan)
    grouped["AbsoluteNetContributionShare"] = (
        grouped["NetContribution"].abs() / absolute_totals.replace(0, np.nan)
    )
    return grouped[
        groups
        + [
            "DiagnosticDimension",
            "DiagnosticValue",
            "Count",
            "NetContribution",
            "NetContributionShare",
            "AbsoluteNetContributionShare",
        ]
    ]


def concentration_diagnostics(entries: pd.DataFrame) -> pd.DataFrame:
    annotated = annotate_regime(entries)
    by_regime = contribution_rows(
        annotated, "Regime", ["Candidate", "ValidationMode"], "Regime"
    )
    by_instrument = contribution_rows(
        annotated, "Instrument", ["Candidate", "ValidationMode"], "Instrument"
    )
    by_spacing = contribution_rows(
        annotated, "ValidationMode", ["Candidate"], "ValidationMode"
    )
    by_spacing["ValidationMode"] = "AcrossModes"
    return pd.concat([by_regime, by_instrument, by_spacing], ignore_index=True).sort_values(
        ["Candidate", "ValidationMode", "DiagnosticDimension", "AbsoluteNetContributionShare"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def consistency(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    signs = np.sign(clean)
    return float(max((signs > 0).mean(), (signs < 0).mean(), (signs == 0).mean()))


def stable_variance(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.var(ddof=0)) if len(clean) else np.nan


def candidate_stability(
    regimes: pd.DataFrame,
    instruments: pd.DataFrame,
    pooled: pd.DataFrame,
    concentration: pd.DataFrame,
) -> pd.DataFrame:
    instrument_dataset = annotate_regime(instruments.loc[instruments["Scope"].eq("Dataset")])
    pooled = numeric(pooled, ["Mean", "ProfitFactor"])
    regime_reference = reference_regime_candidate_aggregation(regimes)
    rows = []
    for candidate in FROZEN_CANDIDATES:
        reference = regime_reference.loc[regime_reference["Candidate"].eq(candidate)].copy()
        reference_inst = instrument_dataset.loc[
            instrument_dataset["Candidate"].eq(candidate)
            & instrument_dataset["ValidationMode"].eq("Reference")
        ]
        mode_rows = pooled.loc[pooled["Candidate"].eq(candidate)]
        passes = mode_rows["ValidationPass"].astype(str).str.lower().eq("true")
        regime_conc = concentration.loc[
            concentration["Candidate"].eq(candidate)
            & concentration["ValidationMode"].eq("Reference")
            & concentration["DiagnosticDimension"].eq("Regime")
        ]
        instrument_conc = concentration.loc[
            concentration["Candidate"].eq(candidate)
            & concentration["ValidationMode"].eq("Reference")
            & concentration["DiagnosticDimension"].eq("Instrument")
        ]
        rows.append({
            "Candidate": candidate,
            "ReferenceRegimeCount": int(reference["Regime"].nunique()),
            "ReferenceTotalCount": int(reference["Count"].sum()),
            "ReferenceRegimesCountGE75": int((reference["Count"] >= 75).sum()),
            "RegimeMeanVariance": stable_variance(reference["Mean"]),
            "RegimePFVarianceFinite": stable_variance(reference["ProfitFactor"]),
            "InfinitePFRegimeCount": int(np.isinf(pd.to_numeric(reference["ProfitFactor"], errors="coerce")).sum()),
            "RegimeSignConsistency": consistency(reference["Mean"]),
            "PositiveRegimeFraction": float((reference["Mean"] > 0).mean()),
            "InstrumentSignConsistency": consistency(reference_inst["Mean"]),
            "PositiveInstrumentFraction": float((reference_inst["Mean"] > 0).mean()),
            "SpacingMeanVariance": stable_variance(mode_rows["Mean"]),
            "SpacingPFVariance": stable_variance(mode_rows["ProfitFactor"]),
            "SpacingPassCount": int(passes.sum()),
            "SpacingSurvivalScore": float(passes.mean()),
            "RegimeSurvivalScore": float((reference["Mean"] > 0).mean()),
            "MaxRegimeAbsoluteContributionShare": float(regime_conc["AbsoluteNetContributionShare"].max()),
            "MaxInstrumentAbsoluteContributionShare": float(instrument_conc["AbsoluteNetContributionShare"].max()),
        })
    return pd.DataFrame(rows)


def robustness_rankings(stability: pd.DataFrame) -> pd.DataFrame:
    ranking = stability.copy()
    ranking["ConcentrationPenalty"] = ranking["MaxRegimeAbsoluteContributionShare"]
    ranking["RobustnessScore"] = (
        ranking["SpacingSurvivalScore"]
        + ranking["RegimeSurvivalScore"]
        + ranking["InstrumentSignConsistency"].fillna(0.0)
        - ranking["ConcentrationPenalty"]
    )
    ranking["RobustnessRank"] = ranking["RobustnessScore"].rank(
        ascending=False, method="min"
    ).astype(int)
    return ranking.sort_values(["RobustnessRank", "Candidate"]).reset_index(drop=True)


def reference_regime_candidate_aggregation(regimes: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated Reference-era labels for descriptive diagnostics."""
    ref = regimes.loc[
        regimes["Candidate"].isin(FROZEN_CANDIDATES)
        & regimes["ValidationMode"].eq("Reference")
    ].copy()
    rows = []
    for (regime, candidate), group in ref.groupby(["Regime", "Candidate"], sort=True):
        counts = pd.to_numeric(group["Count"], errors="coerce").fillna(0.0)
        sums = pd.to_numeric(group.get("Sum", pd.Series(index=group.index, dtype=float)), errors="coerce")
        means = pd.to_numeric(group["Mean"], errors="coerce")
        if sums.notna().any():
            total_sum = float(sums.sum())
        else:
            total_sum = float((means * counts).sum())
        total_count = float(counts.sum())
        if {"GrossWin", "GrossLoss"}.issubset(group.columns):
            gross_win = float(pd.to_numeric(group["GrossWin"], errors="coerce").sum())
            gross_loss = float(pd.to_numeric(group["GrossLoss"], errors="coerce").sum())
            weighted_pf = gross_win / abs(gross_loss) if gross_loss != 0 else (np.inf if gross_win > 0 else np.nan)
            pf_method = "recomputed from summed GrossWin and GrossLoss"
        else:
            pfs = pd.to_numeric(group["ProfitFactor"], errors="coerce")
            finite = pfs.notna()
            weighted_pf = (
                float(np.average(pfs.loc[finite], weights=counts.loc[finite]))
                if finite.any() and counts.loc[finite].sum() > 0 else np.nan
            )
            pf_method = "count-weighted average of available dataset PF values"
        rows.append({
            "Regime": regime,
            "Candidate": candidate,
            "DatasetCount": int(group["Dataset"].nunique()),
            "Count": int(total_count),
            "Sum": total_sum,
            "Mean": total_sum / total_count if total_count > 0 else np.nan,
            "ProfitFactor": weighted_pf,
            "ProfitFactorAggregation": pf_method,
        })
    return pd.DataFrame(rows)


def svg_bar_chart(
    values: pd.DataFrame,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    colors = ["#285f8f", "#c26d2e", "#438a5e"]
    width, height = 1180, 650
    left, top, chart_width, chart_height = 90, 70, 1040, 430
    clean = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    maximum = max(float(clean.max().max()), 0.0)
    minimum = min(float(clean.min().min()), 0.0)
    span = maximum - minimum if maximum != minimum else 1.0
    zero_y = top + chart_height * maximum / span
    groups = max(1, len(clean.index))
    group_width = chart_width / groups
    bar_width = group_width / (len(clean.columns) + 1)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="35" text-anchor="middle" font-family="Arial" font-size="20">{html.escape(title)}</text>',
        f'<text transform="translate(22,{top + chart_height / 2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(ylabel)}</text>',
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + chart_width}" y2="{zero_y:.2f}" stroke="#333" stroke-width="1"/>',
    ]
    for row_index, (label, row) in enumerate(clean.iterrows()):
        base_x = left + row_index * group_width + bar_width / 2
        for column_index, column in enumerate(clean.columns):
            value = float(row[column])
            value_y = top + chart_height * (maximum - value) / span
            y = min(value_y, zero_y)
            bar_height = max(1.0, abs(value_y - zero_y))
            x = base_x + column_index * bar_width
            svg.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width * 0.78:.2f}" height="{bar_height:.2f}" fill="{colors[column_index % len(colors)]}"/>'
            )
        svg.append(
            f'<text x="{base_x + (len(clean.columns) - 1) * bar_width / 2:.2f}" y="{top + chart_height + 22}" text-anchor="middle" font-family="Arial" font-size="11">{html.escape(str(label))}</text>'
        )
    for index, column in enumerate(clean.columns):
        x = left + index * 225
        svg.append(f'<rect x="{x}" y="{height - 65}" width="15" height="15" fill="{colors[index % len(colors)]}"/>')
        svg.append(
            f'<text x="{x + 22}" y="{height - 53}" font-family="Arial" font-size="12">{html.escape(str(column))}</text>'
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def svg_heatmap(heat: pd.DataFrame, title: str, path: Path) -> None:
    width, height = 820, 310
    left, top, cell_width, cell_height = 280, 70, 150, 55
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial" font-size="19">{html.escape(title)}</text>',
    ]
    for column_index, column in enumerate(heat.columns):
        svg.append(
            f'<text x="{left + column_index * cell_width + cell_width / 2}" y="{top - 16}" text-anchor="middle" font-family="Arial" font-size="12">{html.escape(str(column))}</text>'
        )
    for row_index, (candidate, row) in enumerate(heat.iterrows()):
        y = top + row_index * cell_height
        svg.append(
            f'<text x="{left - 12}" y="{y + cell_height / 2 + 4}" text-anchor="end" font-family="Arial" font-size="12">{html.escape(str(candidate))}</text>'
        )
        for column_index, value in enumerate(row):
            passed = bool(value)
            color = "#3d7fa6" if passed else "#e5e8ea"
            text_color = "white" if passed else "#333"
            x = left + column_index * cell_width
            svg.append(
                f'<rect x="{x}" y="{y}" width="{cell_width - 4}" height="{cell_height - 4}" fill="{color}" stroke="white"/>'
            )
            svg.append(
                f'<text x="{x + cell_width / 2 - 2}" y="{y + cell_height / 2 + 4}" text-anchor="middle" font-family="Arial" font-size="13" fill="{text_color}">{"Pass" if passed else "Fail"}</text>'
            )
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def write_charts(
    regimes: pd.DataFrame,
    concentration: pd.DataFrame,
    pooled: pd.DataFrame,
    charts_dir: Path,
) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    ref = reference_regime_candidate_aggregation(regimes)
    mean_plot = ref.pivot(index="Regime", columns="Candidate", values="Mean")
    svg_bar_chart(
        mean_plot,
        "Frozen Candidate Mean By Regime (Reference)",
        "Mean Normalized Policy Outcome",
        charts_dir / "candidate_mean_by_regime.svg",
    )

    pf_plot = ref.pivot(index="Regime", columns="Candidate", values="ProfitFactor").replace(np.inf, np.nan)
    svg_bar_chart(
        pf_plot,
        "Frozen Candidate Profit Factor By Regime (Reference)",
        "Profit Factor (finite values)",
        charts_dir / "candidate_pf_by_regime.svg",
    )

    regime_conc = concentration.loc[
        concentration["ValidationMode"].eq("Reference")
        & concentration["DiagnosticDimension"].eq("Regime")
    ]
    shares = regime_conc.pivot(
        index="DiagnosticValue", columns="Candidate", values="AbsoluteNetContributionShare"
    )
    svg_bar_chart(
        shares,
        "Candidate Contribution Share By Regime (Reference)",
        "Absolute Net Contribution Share",
        charts_dir / "contribution_share_by_regime.svg",
    )

    survival = pooled.loc[pooled["Candidate"].isin(FROZEN_CANDIDATES)].copy()
    survival["PassValue"] = survival["ValidationPass"].astype(str).str.lower().eq("true").astype(int)
    heat = survival.pivot(index="Candidate", columns="ValidationMode", values="PassValue").reindex(
        index=FROZEN_CANDIDATES, columns=VALIDATION_MODES
    )
    svg_heatmap(heat, "Pooled Validation Survival By Spacing Mode", charts_dir / "spacing_survival_heatmap.svg")


def scorecard(
    regimes: pd.DataFrame,
    stability: pd.DataFrame,
    rankings: pd.DataFrame,
    concentration: pd.DataFrame,
    pooled: pd.DataFrame,
) -> pd.DataFrame:
    top = rankings.iloc[0]
    top_conc = rankings.sort_values("ConcentrationPenalty", ascending=False).iloc[0]
    mode_passes = (
        pooled.loc[pooled["Candidate"].isin(FROZEN_CANDIDATES)]
        .assign(Pass=lambda frame: frame["ValidationPass"].astype(str).str.lower().eq("true"))
        .groupby("ValidationMode")["Pass"].sum()
        .reindex(VALIDATION_MODES)
    )
    reference_conc = concentration.loc[
        concentration["ValidationMode"].eq("Reference")
        & concentration["DiagnosticDimension"].eq("Regime")
    ]
    max_regime = reference_conc.sort_values("AbsoluteNetContributionShare", ascending=False).iloc[0]
    ccrrr = rankings.loc[rankings["Candidate"].eq("CCRRR")].iloc[0]
    slope = rankings.loc[rankings["Candidate"].eq("PriorSlope_DominantPressureValue_Q3")].iloc[0]
    rows = [
        {"Metric": "report_mode", "Value": "descriptive diagnostics over frozen candidate validation outputs only"},
        {"Metric": "regime_parser", "Value": "filename year tags parsed generally; named context labels supplied for observed 2020/2022/2024 eras"},
        {"Metric": "candidate_count", "Value": len(FROZEN_CANDIDATES)},
        {"Metric": "regime_count", "Value": regimes["Regime"].nunique()},
        {"Metric": "datasets", "Value": ",".join(sorted(regimes["Dataset"].unique()))},
        {"Metric": "robustness_score_formula", "Value": "spacing_survival_score + regime_survival_score + instrument_sign_consistency - concentration_penalty"},
        {"Metric": "concentration_penalty_formula", "Value": "maximum absolute net contribution share across Reference regimes"},
        {"Metric": "top_robustness_candidate", "Value": top["Candidate"]},
        {"Metric": "top_robustness_score", "Value": top["RobustnessScore"]},
        {"Metric": "most_regime_stable_by_rank", "Value": top["Candidate"]},
        {"Metric": "most_concentration_sensitive_candidate", "Value": top_conc["Candidate"]},
        {"Metric": "most_concentration_sensitive_penalty", "Value": top_conc["ConcentrationPenalty"]},
        {"Metric": "ccrrr_rank", "Value": int(ccrrr["RobustnessRank"])},
        {"Metric": "prior_slope_rank", "Value": int(slope["RobustnessRank"])},
        {"Metric": "ccrrr_still_leads_ranking", "Value": bool(ccrrr["RobustnessRank"] < slope["RobustnessRank"])},
        {"Metric": "max_reference_regime_contribution_candidate", "Value": max_regime["Candidate"]},
        {"Metric": "max_reference_regime_contribution_regime", "Value": max_regime["DiagnosticValue"]},
        {"Metric": "max_reference_regime_absolute_contribution_share", "Value": max_regime["AbsoluteNetContributionShare"]},
        {"Metric": "reference_passing_candidates", "Value": ",".join(pooled.loc[
            pooled["ValidationMode"].eq("Reference")
            & pooled["Candidate"].isin(FROZEN_CANDIDATES)
            & pooled["ValidationPass"].astype(str).str.lower().eq("true"), "Candidate"
        ])},
        {"Metric": "spacing_10_passing_candidates", "Value": ",".join(pooled.loc[
            pooled["ValidationMode"].eq("Spacing_10")
            & pooled["Candidate"].isin(FROZEN_CANDIDATES)
            & pooled["ValidationPass"].astype(str).str.lower().eq("true"), "Candidate"
        ])},
        {"Metric": "spacing_20_passing_candidates", "Value": ",".join(pooled.loc[
            pooled["ValidationMode"].eq("Spacing_20")
            & pooled["Candidate"].isin(FROZEN_CANDIDATES)
            & pooled["ValidationPass"].astype(str).str.lower().eq("true"), "Candidate"
        ])},
        {"Metric": "spacing_mode_pass_counts", "Value": ",".join(f"{mode}:{int(mode_passes.loc[mode])}" for mode in VALIDATION_MODES)},
    ]
    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="outputs/fixed_candidate_extended_validation")
    parser.add_argument("--outdir", default="outputs/cross_era_validation_report")
    args = parser.parse_args(argv)
    indir = Path(args.indir)
    outdir = Path(args.outdir)
    charts_dir = outdir / "charts"
    outdir.mkdir(parents=True, exist_ok=True)
    sources = read_sources(indir)
    entries = numeric(sources["extended_entries.csv"], ["NormalizedPolicyOutcome"])
    dataset_summary = numeric(
        sources["extended_dataset_candidate_summary.csv"],
        ["Count", "ES_Count", "NQ_Count", "Mean", "Median", "TStat", "ProfitFactor"],
    )
    blocks = numeric(sources["extended_block_summary.csv"], ["Count", "Mean", "Sum"])
    instruments = numeric(sources["extended_instrument_summary.csv"], ["Count", "Mean", "ProfitFactor"])
    pooled = numeric(sources["extended_candidate_summary.csv"], ["Count", "Mean", "ProfitFactor"])
    regimes = regime_summary(dataset_summary)
    regime_block = regime_blocks(blocks)
    concentration = concentration_diagnostics(entries)
    stability = candidate_stability(regimes, instruments, pooled, concentration)
    rankings = robustness_rankings(stability)
    report_scorecard = scorecard(regimes, stability, rankings, concentration, pooled)
    regimes.to_csv(outdir / "regime_candidate_summary.csv", index=False)
    regime_block.to_csv(outdir / "regime_block_summary.csv", index=False)
    stability.to_csv(outdir / "candidate_stability_summary.csv", index=False)
    concentration.to_csv(outdir / "concentration_diagnostics.csv", index=False)
    rankings.to_csv(outdir / "robustness_rankings.csv", index=False)
    report_scorecard.to_csv(outdir / "cross_era_validation_scorecard.csv", index=False)
    write_charts(regimes, concentration, pooled, charts_dir)
    print("APVA cross-era validation report complete")
    print(rankings[[
        "RobustnessRank", "Candidate", "RobustnessScore", "SpacingSurvivalScore",
        "RegimeSurvivalScore", "ConcentrationPenalty"
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
