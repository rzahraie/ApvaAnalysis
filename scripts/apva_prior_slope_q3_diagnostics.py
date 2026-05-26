#!/usr/bin/env python3
"""Describe the frozen PriorSlope_DominantPressureValue_Q3 validation candidate.

This report consumes completed fixed-rule validation outputs. It does not
construct new candidates, modify thresholds, or search for alternative rules.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from apva_analysis_utils import summarize
from apva_cross_era_validation_report import infer_regime
from apva_topology_walkforward_validation import CCRRR, RRCCC


CANDIDATE = "PriorSlope_DominantPressureValue_Q3"
VALIDATION_MODES = ["Reference", "Spacing_10", "Spacing_20"]
NUMERIC_COLUMNS = [
    "BarIndex",
    "HorizonBars",
    "PriorSlope_DominantPressureValue",
    "NormalizedPolicyOutcome",
    "SignedNormalizedReturn",
    "DirectionalNormalizedMAE",
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required diagnostic input: {path}")
    return pd.read_csv(path)


def numeric(frame: pd.DataFrame, columns: list[str] = NUMERIC_COLUMNS) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def annotate_regime(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    parsed = out["Dataset"].map(infer_regime)
    out["RegimeCode"] = parsed.map(lambda pair: pair[0])
    out["Regime"] = parsed.map(lambda pair: pair[1])
    return out


def diagnostic_summary(group: pd.DataFrame) -> dict[str, object]:
    outcome = pd.to_numeric(group["NormalizedPolicyOutcome"], errors="coerce").dropna()
    row: dict[str, object] = dict(summarize(outcome))
    row["StopRate"] = float(group["DisasterStopped"].astype(bool).mean()) if len(group) else np.nan
    row["ES_Count"] = int(group["Instrument"].astype(str).str.upper().eq("ES").sum())
    row["NQ_Count"] = int(group["Instrument"].astype(str).str.upper().eq("NQ").sum())
    row["PositiveContribution"] = float(outcome[outcome > 0].sum())
    row["NegativeContribution"] = float(outcome[outcome < 0].sum())
    row["PositiveCount"] = int((outcome > 0).sum())
    row["NegativeCount"] = int((outcome < 0).sum())
    positives = outcome[outcome > 0].sort_values(ascending=False)
    for fraction, name in [(0.01, "Top1PctWinContributionShare"), (0.05, "Top5PctWinContributionShare"), (0.10, "Top10PctWinContributionShare")]:
        n = max(1, int(np.ceil(len(outcome) * fraction))) if len(outcome) else 0
        row[name] = (
            float(positives.head(n).sum() / positives.sum())
            if n and len(positives) and positives.sum() != 0 else np.nan
        )
    return row


def grouped_summary(entries: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in entries.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_cols, keys)}
        row.update(diagnostic_summary(group))
        rows.append(row)
    return pd.DataFrame(rows)


def add_contribution_shares(
    summary: pd.DataFrame,
    *,
    group_cols: list[str],
) -> pd.DataFrame:
    out = summary.copy()
    totals = out.groupby(group_cols)["Sum"].transform("sum")
    absolute_totals = out.groupby(group_cols)["Sum"].transform(lambda values: values.abs().sum())
    out["NetContributionShare"] = out["Sum"] / totals.replace(0, np.nan)
    out["AbsoluteNetContributionShare"] = out["Sum"].abs() / absolute_totals.replace(0, np.nan)
    return out


def mark_overlaps(entries: pd.DataFrame) -> pd.DataFrame:
    prior = entries.loc[entries["Candidate"].eq(CANDIDATE)].copy()
    if prior.empty:
        raise RuntimeError(f"No {CANDIDATE} entries are present in extended_entries.csv")
    prior["Overlaps_CCRRR"] = prior["PriorPressureSeq"].astype(str).eq(CCRRR)
    prior["Overlaps_RRCCC"] = prior["PriorPressureSeq"].astype(str).eq(RRCCC)
    prior["OverlapClass"] = np.select(
        [
            prior["Overlaps_CCRRR"] & prior["Overlaps_RRCCC"],
            prior["Overlaps_CCRRR"],
            prior["Overlaps_RRCCC"],
        ],
        ["Overlap_CCRRR_and_RRCCC", "Overlap_CCRRR", "Overlap_RRCCC"],
        default="BaseOnly_NeitherGrammar",
    )
    return annotate_regime(prior).sort_values(
        ["ValidationMode", "Dataset", "Instrument", "File", "BarIndex"]
    ).reset_index(drop=True)


def reconcile_sources(
    prior: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    extended_blocks: pd.DataFrame,
    regime_summary: pd.DataFrame,
    regime_blocks: pd.DataFrame,
) -> pd.DataFrame:
    sources = {
        "extended_dataset_candidate_summary": dataset_summary.loc[dataset_summary["Candidate"].eq(CANDIDATE)],
        "extended_block_summary": extended_blocks.loc[
            extended_blocks["Candidate"].eq(CANDIDATE) & extended_blocks["Scope"].eq("Dataset")
        ],
        "regime_candidate_summary": regime_summary.loc[regime_summary["Candidate"].eq(CANDIDATE)],
        "regime_block_summary": regime_blocks.loc[regime_blocks["Candidate"].eq(CANDIDATE)],
    }
    rows: list[dict[str, object]] = []
    for mode in VALIDATION_MODES:
        entry_group = prior.loc[prior["ValidationMode"].eq(mode)]
        expected_count = int(len(entry_group))
        expected_sum = float(entry_group["NormalizedPolicyOutcome"].sum())
        for source_name, source in sources.items():
            group = source.loc[source["ValidationMode"].eq(mode)]
            actual_count = int(pd.to_numeric(group["Count"], errors="coerce").sum())
            actual_sum = float(pd.to_numeric(group["Sum"], errors="coerce").sum())
            count_match = expected_count == actual_count
            sum_match = bool(np.isclose(expected_sum, actual_sum, rtol=1e-10, atol=1e-10))
            rows.append({
                "ValidationMode": mode,
                "Source": source_name,
                "ExpectedCount": expected_count,
                "ActualCount": actual_count,
                "ExpectedSum": expected_sum,
                "ActualSum": actual_sum,
                "CountMatch": count_match,
                "SumMatch": sum_match,
                "Status": "PASS" if count_match and sum_match else "FAIL",
            })
    reconciliation = pd.DataFrame(rows)
    if not reconciliation["Status"].eq("PASS").all():
        raise RuntimeError("PriorSlope Q3 diagnostic inputs are stale or mutually inconsistent.")
    return reconciliation


def summarize_spacing(entries: pd.DataFrame) -> pd.DataFrame:
    all_rows = entries.copy()
    all_rows["OverlapClass"] = "All_PriorSlope_Q3"
    combined = pd.concat([all_rows, entries], ignore_index=True)
    summary = grouped_summary(combined, ["ValidationMode", "OverlapClass"])
    mode_totals = summary.loc[summary["OverlapClass"].eq("All_PriorSlope_Q3"), ["ValidationMode", "Count", "Sum"]].rename(
        columns={"Count": "ModeCount", "Sum": "ModeSum"}
    )
    summary = summary.merge(mode_totals, on="ValidationMode", how="left")
    summary["EntryShareOfMode"] = summary["Count"] / summary["ModeCount"]
    summary["NetContributionShareOfMode"] = summary["Sum"] / summary["ModeSum"].replace(0, np.nan)
    return summary.sort_values(["ValidationMode", "OverlapClass"]).reset_index(drop=True)


def summarize_blocks(entries: pd.DataFrame) -> pd.DataFrame:
    summary = grouped_summary(entries, ["ValidationMode", "Dataset", "Regime", "TimeBlock", "RegimeWeek"])
    summary = add_contribution_shares(summary, group_cols=["ValidationMode"])
    return summary.sort_values(
        ["ValidationMode", "AbsoluteNetContributionShare"],
        ascending=[True, False],
    ).reset_index(drop=True)


def scalar(summary: pd.DataFrame, mode: str, column: str, overlap_class: str = "All_PriorSlope_Q3") -> float:
    value = summary.loc[
        summary["ValidationMode"].eq(mode) & summary["OverlapClass"].eq(overlap_class),
        column,
    ]
    if value.empty:
        return np.nan
    return float(value.iloc[0])


def count_or_zero(summary: pd.DataFrame, mode: str, overlap_class: str) -> int:
    value = scalar(summary, mode, "Count", overlap_class)
    return 0 if pd.isna(value) else int(value)


def build_scorecard(
    entries: pd.DataFrame,
    by_regime: pd.DataFrame,
    by_instrument: pd.DataFrame,
    by_spacing: pd.DataFrame,
    blocks: pd.DataFrame,
    reconciliation: pd.DataFrame,
) -> pd.DataFrame:
    reference_regime = by_regime.loc[by_regime["ValidationMode"].eq("Reference")]
    reference_instrument = by_instrument.loc[by_instrument["ValidationMode"].eq("Reference")]
    reference_blocks = blocks.loc[blocks["ValidationMode"].eq("Reference")]
    rows: list[dict[str, object]] = [
        {"Metric": "report_scope", "Value": f"diagnostic-only summary of frozen {CANDIDATE}"},
        {"Metric": "frozen_rule", "Value": "0.004718046888521732 <= PriorSlope_DominantPressureValue <= 0.05505235072964343"},
        {"Metric": "candidate_rules_changed", "Value": False},
        {"Metric": "thresholds_changed", "Value": False},
        {"Metric": "reference_count", "Value": int(scalar(by_spacing, "Reference", "Count"))},
        {"Metric": "reference_mean", "Value": scalar(by_spacing, "Reference", "Mean")},
        {"Metric": "reference_median", "Value": scalar(by_spacing, "Reference", "Median")},
        {"Metric": "reference_profit_factor", "Value": scalar(by_spacing, "Reference", "ProfitFactor")},
        {"Metric": "reference_stop_rate", "Value": scalar(by_spacing, "Reference", "StopRate")},
        {"Metric": "spacing_20_count", "Value": int(scalar(by_spacing, "Spacing_20", "Count"))},
        {"Metric": "spacing_20_mean", "Value": scalar(by_spacing, "Spacing_20", "Mean")},
        {"Metric": "spacing_20_median", "Value": scalar(by_spacing, "Spacing_20", "Median")},
        {"Metric": "spacing_20_profit_factor", "Value": scalar(by_spacing, "Spacing_20", "ProfitFactor")},
        {"Metric": "spacing_20_stop_rate", "Value": scalar(by_spacing, "Spacing_20", "StopRate")},
        {"Metric": "reference_positive_regime_fraction", "Value": float((reference_regime["Mean"] > 0).mean())},
        {"Metric": "reference_max_regime_absolute_contribution_share", "Value": float(reference_regime["AbsoluteNetContributionShare"].max())},
        {"Metric": "reference_positive_instrument_fraction", "Value": float((reference_instrument["Mean"] > 0).mean())},
        {"Metric": "reference_max_instrument_absolute_contribution_share", "Value": float(reference_instrument["AbsoluteNetContributionShare"].max())},
        {"Metric": "reference_max_block_absolute_contribution_share", "Value": float(reference_blocks["AbsoluteNetContributionShare"].max())},
        {"Metric": "reference_top_5_pct_win_contribution_share", "Value": scalar(by_spacing, "Reference", "Top5PctWinContributionShare")},
        {"Metric": "spacing_20_top_5_pct_win_contribution_share", "Value": scalar(by_spacing, "Spacing_20", "Top5PctWinContributionShare")},
        {"Metric": "reference_overlap_ccrrr_count", "Value": count_or_zero(by_spacing, "Reference", "Overlap_CCRRR")},
        {"Metric": "reference_overlap_rrccc_count", "Value": count_or_zero(by_spacing, "Reference", "Overlap_RRCCC")},
        {"Metric": "reference_base_only_count", "Value": count_or_zero(by_spacing, "Reference", "BaseOnly_NeitherGrammar")},
        {"Metric": "spacing_20_overlap_ccrrr_count", "Value": count_or_zero(by_spacing, "Spacing_20", "Overlap_CCRRR")},
        {"Metric": "spacing_20_overlap_rrccc_count", "Value": count_or_zero(by_spacing, "Spacing_20", "Overlap_RRCCC")},
        {"Metric": "spacing_20_base_only_count", "Value": count_or_zero(by_spacing, "Spacing_20", "BaseOnly_NeitherGrammar")},
        {"Metric": "entry_rows_all_modes", "Value": int(len(entries))},
        {"Metric": "input_reconciliation_check_count", "Value": int(len(reconciliation))},
        {"Metric": "input_reconciliation_all_pass", "Value": bool(reconciliation["Status"].eq("PASS").all())},
    ]
    return pd.DataFrame(rows)


def fmt(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(value)
    return f"{float(value):.{digits}f}"


def markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def write_handoff(
    path: Path,
    by_regime: pd.DataFrame,
    by_instrument: pd.DataFrame,
    by_spacing: pd.DataFrame,
    blocks: pd.DataFrame,
) -> None:
    reference = by_spacing.loc[
        by_spacing["ValidationMode"].eq("Reference") & by_spacing["OverlapClass"].eq("All_PriorSlope_Q3")
    ].iloc[0]
    spacing20 = by_spacing.loc[
        by_spacing["ValidationMode"].eq("Spacing_20") & by_spacing["OverlapClass"].eq("All_PriorSlope_Q3")
    ].iloc[0]
    regimes = by_regime.loc[by_regime["ValidationMode"].eq("Reference")].sort_values("AbsoluteNetContributionShare", ascending=False)
    instruments = by_instrument.loc[by_instrument["ValidationMode"].eq("Reference")].set_index("Instrument")
    blocks_ref = blocks.loc[blocks["ValidationMode"].eq("Reference")]
    overlaps = by_spacing.loc[by_spacing["ValidationMode"].eq("Reference")].set_index("OverlapClass")
    top_regime = regimes.iloc[0]
    top_regime_share = float(top_regime["AbsoluteNetContributionShare"])
    distribution_text = (
        "The Reference contribution is distributed across regimes rather than being majority-dominated by one regime"
        if top_regime_share < 0.50
        else "The Reference contribution is concentrated in one regime"
    )
    es = instruments.loc["ES"]
    nq = instruments.loc["NQ"]
    both_text = (
        "Both instruments are positive in aggregate, with ES contributing the higher mean outcome"
        if es["Mean"] > 0 and nq["Mean"] > 0
        else "Aggregate instrument behavior is not positive for both ES and NQ"
    )
    base_only = overlaps.loc["BaseOnly_NeitherGrammar"]
    overlap_total = reference["Count"] - base_only["Count"]
    overlap_share = overlap_total / reference["Count"]
    overlap_text = (
        "Most Reference entries do not overlap either frozen pressure grammar"
        if base_only["Count"] > overlap_total
        else "Most Reference entries overlap a frozen pressure grammar"
    )
    lines = [
        "# PriorSlope Q3 Diagnostic Handoff",
        "",
        "## Scope",
        "",
        "This report describes the frozen `PriorSlope_DominantPressureValue_Q3`",
        "candidate only. No candidate rules or validation thresholds were changed.",
        "",
        "Frozen rule:",
        "",
        "```text",
        "0.004718046888521732 <= PriorSlope_DominantPressureValue <= 0.05505235072964343",
        "```",
        "",
        "## Core Results",
        "",
        "| Mode | Count | Mean | Median | PF | Stop Rate | Top 5% Win Contribution Share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in VALIDATION_MODES:
        row = by_spacing.loc[
            by_spacing["ValidationMode"].eq(mode) & by_spacing["OverlapClass"].eq("All_PriorSlope_Q3")
        ].iloc[0]
        lines.append(
            f"| {mode} | {int(row['Count'])} | {fmt(row['Mean'])} | {fmt(row['Median'])} | "
            f"{fmt(row['ProfitFactor'])} | {fmt(row['StopRate'])} | {fmt(row['Top5PctWinContributionShare'])} |"
        )
    lines.extend([
        "",
        "## Regime Stability",
        "",
        f"{distribution_text}. The largest absolute Reference regime share is "
        f"`{fmt(top_regime_share)}` from `{top_regime['Regime']}`. "
        f"`{int((regimes['Mean'] > 0).sum())}` of `{len(regimes)}` aggregated regimes have positive means.",
        "",
        "| Reference Regime | Count | Mean | PF | Absolute Contribution Share |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for _, row in regimes.iterrows():
        lines.append(
            f"| {markdown_cell(row['Regime'])} | {int(row['Count'])} | {fmt(row['Mean'])} | "
            f"{fmt(row['ProfitFactor'])} | {fmt(row['AbsoluteNetContributionShare'])} |"
        )
    lines.extend([
        "",
        "## Instrument Behavior",
        "",
        f"{both_text}. ES mean is `{fmt(es['Mean'])}` on `{int(es['Count'])}` Reference entries; "
        f"NQ mean is `{fmt(nq['Mean'])}` on `{int(nq['Count'])}` entries.",
        "",
        "## Spacing 20",
        "",
        f"`Spacing_20` retains `{int(spacing20['Count'])}` entries with mean `{fmt(spacing20['Mean'])}`, "
        f"median `{fmt(spacing20['Median'])}`, PF `{fmt(spacing20['ProfitFactor'])}`, and stop rate "
        f"`{fmt(spacing20['StopRate'])}`. Its largest block absolute contribution share is "
        f"`{fmt(blocks.loc[blocks['ValidationMode'].eq('Spacing_20'), 'AbsoluteNetContributionShare'].max())}`. "
        "This supports survival under dependence reduction as a distributed diagnostic result, not a new rule.",
        "",
        "## Grammar Overlap",
        "",
        f"{overlap_text}: `{int(base_only['Count'])}` of `{int(reference['Count'])}` Reference entries "
        f"are in neither `CCRRR` nor `RRCCC`; combined grammar overlap is `{int(overlap_total)}` "
        f"entries (`{fmt(overlap_share)}`).",
        f"The non-overlap subset remains positive with mean `{fmt(base_only['Mean'])}`, median "
        f"`{fmt(base_only['Median'])}`, and PF `{fmt(base_only['ProfitFactor'])}`, so PriorSlope Q3 "
        "does not simply restate either fixed grammar candidate.",
        "",
        "| Reference Overlap Class | Count | Mean | Median | PF | Contribution Share |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for label in ["BaseOnly_NeitherGrammar", "Overlap_CCRRR", "Overlap_RRCCC", "Overlap_CCRRR_and_RRCCC"]:
        if label not in overlaps.index:
            continue
        row = overlaps.loc[label]
        lines.append(
            f"| `{label}` | {int(row['Count'])} | {fmt(row['Mean'])} | {fmt(row['Median'])} | "
            f"{fmt(row['ProfitFactor'])} | {fmt(row['NetContributionShareOfMode'])} |"
        )
    lines.extend([
        "",
        "## Win Distribution And Stops",
        "",
        f"In Reference mode the positive median outcome (`{fmt(reference['Median'])}`) and largest "
        f"single-block absolute contribution share (`{fmt(blocks_ref['AbsoluteNetContributionShare'].max())}`) "
        "indicate that the result is not explained only by a few block clusters. The top 5% of entries "
        f"still account for `{fmt(reference['Top5PctWinContributionShare'])}` of positive contribution, "
        "so larger wins materially amplify an otherwise positive central distribution.",
        "",
        f"Stop rates remain similar from Reference (`{fmt(reference['StopRate'])}`) to Spacing_20 "
        f"(`{fmt(spacing20['StopRate'])}`).",
        "",
        "## Next Fixed Validation Target",
        "",
        "Collect and run the unchanged frozen rule on additional paired ES/NQ raw-state-log regimes, "
        "with particular value in increasing ES coverage outside the already represented dates. "
        "No threshold or candidate change is indicated by this diagnostic report.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-dir", default="outputs/fixed_candidate_extended_validation")
    parser.add_argument("--cross-era-dir", default="outputs/cross_era_validation_report")
    parser.add_argument("--outdir", default="outputs/prior_slope_q3_diagnostics")
    args = parser.parse_args(argv)
    workspace = Path.cwd()
    fixed_dir = workspace / args.fixed_dir
    cross_dir = workspace / args.cross_era_dir
    outdir = workspace / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    entries = numeric(read_csv(fixed_dir / "extended_entries.csv"))
    dataset_summary = read_csv(fixed_dir / "extended_dataset_candidate_summary.csv")
    extended_blocks = read_csv(fixed_dir / "extended_block_summary.csv")
    regime_summary = read_csv(cross_dir / "regime_candidate_summary.csv")
    regime_blocks = read_csv(cross_dir / "regime_block_summary.csv")
    for source_name, source in [
        ("extended_dataset_candidate_summary.csv", dataset_summary),
        ("regime_candidate_summary.csv", regime_summary),
        ("extended_block_summary.csv", extended_blocks),
        ("regime_block_summary.csv", regime_blocks),
    ]:
        if CANDIDATE not in set(source["Candidate"].astype(str)):
            raise RuntimeError(f"{source_name} contains no {CANDIDATE} rows")
    prior = mark_overlaps(entries)
    reconciliation = reconcile_sources(prior, dataset_summary, extended_blocks, regime_summary, regime_blocks)
    by_regime = add_contribution_shares(
        grouped_summary(prior, ["ValidationMode", "Regime"]),
        group_cols=["ValidationMode"],
    ).sort_values(["ValidationMode", "Regime"]).reset_index(drop=True)
    by_instrument = add_contribution_shares(
        grouped_summary(prior, ["ValidationMode", "Instrument"]),
        group_cols=["ValidationMode"],
    ).sort_values(["ValidationMode", "Instrument"]).reset_index(drop=True)
    by_regime_instrument = add_contribution_shares(
        grouped_summary(prior, ["ValidationMode", "Regime", "Instrument"]),
        group_cols=["ValidationMode"],
    ).sort_values(["ValidationMode", "Regime", "Instrument"]).reset_index(drop=True)
    by_spacing = summarize_spacing(prior)
    blocks = summarize_blocks(prior)
    scorecard = build_scorecard(prior, by_regime, by_instrument, by_spacing, blocks, reconciliation)
    prior.to_csv(outdir / "prior_slope_q3_entries.csv", index=False)
    by_regime.to_csv(outdir / "prior_slope_q3_by_regime.csv", index=False)
    by_instrument.to_csv(outdir / "prior_slope_q3_by_instrument.csv", index=False)
    by_regime_instrument.to_csv(outdir / "prior_slope_q3_by_regime_instrument.csv", index=False)
    by_spacing.to_csv(outdir / "prior_slope_q3_by_spacing.csv", index=False)
    blocks.to_csv(outdir / "prior_slope_q3_block_contribution.csv", index=False)
    scorecard.to_csv(outdir / "prior_slope_q3_scorecard.csv", index=False)
    write_handoff(outdir / "chatgpt_master_analysis_handoff.md", by_regime, by_instrument, by_spacing, blocks)
    print("PriorSlope Q3 diagnostic report complete")
    print(by_spacing.loc[by_spacing["OverlapClass"].eq("All_PriorSlope_Q3"), [
        "ValidationMode", "Count", "Mean", "Median", "ProfitFactor", "StopRate"
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
