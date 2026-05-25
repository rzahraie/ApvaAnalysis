#!/usr/bin/env python3
"""Dependence-reduction validation for frozen ES/NQ APVA candidates.

The candidate definitions are imported from the fixed-rule walk-forward
validation. This script applies only prespecified de-clustering methods and
evaluates the resulting candidate observations; it does not search or tune.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from apva_analysis_utils import LEAKY_OUTCOME_FIELDS, group_summary, prepare_df, summarize
from apva_topology_walkforward_validation import (
    CANDIDATE_RULES,
    PREDICTOR_FEATURES,
    PRIOR_SLOPE_Q3_MAX,
    PRIOR_SLOPE_Q3_MIN,
    build_base_entries,
    candidate_entries,
)


DECLUSTER_METHODS = {
    "Reference": "no de-clustering",
    "Spacing_5": "minimum bar spacing = 5 within Instrument/File/Candidate",
    "Spacing_10": "minimum bar spacing = 10 within Instrument/File/Candidate",
    "Spacing_20": "minimum bar spacing = 20 within Instrument/File/Candidate",
    "First_Per_Instrument_File_Week": (
        "first chronological entry per Instrument/File/RegimeWeek/Candidate"
    ),
}


def spacing_mask(entries: pd.DataFrame, min_spacing: int) -> pd.Series:
    keep = pd.Series(False, index=entries.index)
    ordered = entries.sort_values(["Instrument", "File", "Candidate", "BarIndex"])
    for _, group in ordered.groupby(["Instrument", "File", "Candidate"], sort=False):
        last_kept: float | None = None
        for index, bar_index in zip(group.index, group["BarIndex"]):
            bar = float(bar_index)
            if last_kept is None or bar - last_kept >= min_spacing:
                keep.loc[index] = True
                last_kept = bar
    return keep


def apply_decluster_methods(entries: pd.DataFrame) -> pd.DataFrame:
    frames = []
    method_masks = {
        "Reference": pd.Series(True, index=entries.index),
        "Spacing_5": spacing_mask(entries, 5),
        "Spacing_10": spacing_mask(entries, 10),
        "Spacing_20": spacing_mask(entries, 20),
        "First_Per_Instrument_File_Week": ~entries.sort_values(
            ["Instrument", "File", "Candidate", "RegimeWeek", "BarIndex"]
        ).duplicated(["Instrument", "File", "RegimeWeek", "Candidate"]).reindex(
            entries.index, fill_value=False
        ),
    }
    for method, mask in method_masks.items():
        selected = entries.loc[mask].copy()
        selected.insert(0, "DeclusterRule", DECLUSTER_METHODS[method])
        selected.insert(0, "DeclusterMethod", method)
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def block_summary(entries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    totals = entries.groupby(["DeclusterMethod", "Candidate"])["NormalizedPolicyOutcome"].sum()
    for (method, candidate, block, week), group in entries.groupby(
        ["DeclusterMethod", "Candidate", "TimeBlock", "RegimeWeek"],
        dropna=False,
        sort=True,
    ):
        row = {
            "DeclusterMethod": method,
            "DeclusterRule": DECLUSTER_METHODS[method],
            "Candidate": candidate,
            "TimeBlock": block,
            "RegimeWeek": week,
        }
        row.update(summarize(group["NormalizedPolicyOutcome"]))
        row["StopRate"] = float(group["DisasterStopped"].mean())
        row["ES_Count"] = int(group["Instrument"].astype(str).str.upper().eq("ES").sum())
        row["NQ_Count"] = int(group["Instrument"].astype(str).str.upper().eq("NQ").sum())
        total_sum = float(totals.loc[(method, candidate)])
        row["NetContributionFraction"] = (
            float(row["Sum"] / total_sum) if total_sum > 0 else np.nan
        )
        row["PositiveBlock"] = bool(row["Mean"] > 0)
        rows.append(row)
    return pd.DataFrame(rows)


def instrument_summary(entries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, candidate, instrument), group in entries.groupby(
        ["DeclusterMethod", "Candidate", "Instrument"], sort=True
    ):
        row = {
            "DeclusterMethod": method,
            "DeclusterRule": DECLUSTER_METHODS[method],
            "Candidate": candidate,
            "Instrument": instrument,
        }
        row.update(summarize(group["NormalizedPolicyOutcome"]))
        row["StopRate"] = float(group["DisasterStopped"].mean())
        row["NegativeAggregateMean"] = bool(row["Mean"] < 0)
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_summary(
    entries: pd.DataFrame,
    by_block: pd.DataFrame,
    by_instrument: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for method in DECLUSTER_METHODS:
        method_entries = entries.loc[entries["DeclusterMethod"].eq(method)]
        base = method_entries.loc[method_entries["Candidate"].eq("Base_ES_NQ")]
        base_pf = float(summarize(base["NormalizedPolicyOutcome"])["ProfitFactor"])
        for candidate in CANDIDATE_RULES:
            group = method_entries.loc[method_entries["Candidate"].eq(candidate)]
            blocks = by_block.loc[
                by_block["DeclusterMethod"].eq(method) & by_block["Candidate"].eq(candidate)
            ]
            instruments = by_instrument.loc[
                by_instrument["DeclusterMethod"].eq(method)
                & by_instrument["Candidate"].eq(candidate)
            ]
            instrument_weeks = group_summary(group, ["Instrument", "RegimeWeek"])
            es = instruments.loc[instruments["Instrument"].astype(str).str.upper().eq("ES")]
            nq = instruments.loc[instruments["Instrument"].astype(str).str.upper().eq("NQ")]
            row: dict[str, object] = {
                "DeclusterMethod": method,
                "DeclusterRule": DECLUSTER_METHODS[method],
                "Candidate": candidate,
                "CandidateRule": CANDIDATE_RULES[candidate],
                "BaseMethodProfitFactor": base_pf,
            }
            row.update(summarize(group["NormalizedPolicyOutcome"]))
            row["ES_Count"] = int(group["Instrument"].astype(str).str.upper().eq("ES").sum())
            row["NQ_Count"] = int(group["Instrument"].astype(str).str.upper().eq("NQ").sum())
            row["StopRate"] = float(group["DisasterStopped"].mean())
            row["BlockCount"] = int(len(blocks))
            row["PositiveBlockFraction"] = (
                float((blocks["Mean"] > 0).mean()) if not blocks.empty else np.nan
            )
            row["InstrumentWeekPositiveFraction"] = (
                float((instrument_weeks["Mean"] > 0).mean())
                if not instrument_weeks.empty else np.nan
            )
            row["MaxSingleBlockContributionFraction"] = (
                float(blocks["NetContributionFraction"].max()) if not blocks.empty else np.nan
            )
            row["ES_Mean"] = float(es["Mean"].iloc[0]) if not es.empty else np.nan
            row["NQ_Mean"] = float(nq["Mean"].iloc[0]) if not nq.empty else np.nan
            row["NegativeInstrumentAggregateFlag"] = bool(
                instruments["NegativeAggregateMean"].any()
            )
            row["CountPass"] = bool(row["Count"] >= args.min_count)
            row["BothInstrumentsPass"] = bool(row["ES_Count"] > 0 and row["NQ_Count"] > 0)
            row["MedianPass"] = bool(row["Median"] > 0)
            row["PFAboveMethodBasePass"] = bool(row["ProfitFactor"] > base_pf)
            row["PositiveBlockFractionPass"] = bool(
                row["PositiveBlockFraction"] > args.min_positive_block_fraction
            )
            row["MaxSingleBlockContributionPass"] = bool(
                row["MaxSingleBlockContributionFraction"] <= args.max_block_contribution
            )
            row["InstrumentMeansPass"] = bool(
                pd.notna(row["ES_Mean"])
                and pd.notna(row["NQ_Mean"])
                and row["ES_Mean"] >= 0
                and row["NQ_Mean"] >= 0
            )
            criteria = [
                "CountPass",
                "BothInstrumentsPass",
                "MedianPass",
                "PFAboveMethodBasePass",
                "PositiveBlockFractionPass",
                "MaxSingleBlockContributionPass",
                "InstrumentMeansPass",
            ]
            row["ValidationPass"] = bool(all(row[name] for name in criteria))
            row["ValidationStatus"] = (
                "reference_only" if candidate == "Base_ES_NQ"
                else ("pass" if row["ValidationPass"] else "fail")
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_scorecard(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = [
        {"Metric": "candidate_mode", "Value": "frozen rules only; no candidate search"},
        {"Metric": "block_definition", "Value": "chronological ISO week"},
        {"Metric": "prior_slope_q3_min_frozen", "Value": PRIOR_SLOPE_Q3_MIN},
        {"Metric": "prior_slope_q3_max_frozen", "Value": PRIOR_SLOPE_Q3_MAX},
        {"Metric": "count_requirement", "Value": f">= {args.min_count}"},
        {"Metric": "median_requirement", "Value": "> 0"},
        {"Metric": "profit_factor_requirement", "Value": "> method-matched Base_ES_NQ PF"},
        {
            "Metric": "positive_block_fraction_requirement",
            "Value": f"> {args.min_positive_block_fraction}",
        },
        {
            "Metric": "max_single_block_contribution_requirement",
            "Value": f"<= {args.max_block_contribution}",
        },
        {
            "Metric": "instrument_mean_requirement",
            "Value": "ES mean >= 0 and NQ mean >= 0",
        },
        {
            "Metric": "predictor_fields_excluded_as_leaky",
            "Value": ",".join(LEAKY_OUTCOME_FIELDS),
        },
    ]
    for method, rule in DECLUSTER_METHODS.items():
        rows.append({"Metric": f"{method}_rule", "Value": rule})
        method_summary = summary.loc[summary["DeclusterMethod"].eq(method)]
        passed = method_summary.loc[
            method_summary["Candidate"].ne("Base_ES_NQ") & method_summary["ValidationPass"]
        ]["Candidate"].tolist()
        rows.append(
            {
                "Metric": f"{method}_passing_candidates",
                "Value": ",".join(passed) if passed else "none",
            }
        )
        for _, candidate in method_summary.iterrows():
            label = f"{method}_{candidate['Candidate']}"
            for key in [
                "ValidationStatus",
                "Count",
                "ES_Count",
                "NQ_Count",
                "Mean",
                "Median",
                "TStat",
                "ProfitFactor",
                "BaseMethodProfitFactor",
                "WinRate",
                "PositiveBlockFraction",
                "InstrumentWeekPositiveFraction",
                "MaxSingleBlockContributionFraction",
                "ES_Mean",
                "NQ_Mean",
                "NegativeInstrumentAggregateFlag",
                "BlockCount",
                "ValidationPass",
            ]:
                rows.append({"Metric": f"{label}_{key}", "Value": candidate[key]})
    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    parser.add_argument("--outdir", default="outputs/topology_decluster_validation")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--pressure", default="CompressionPressure")
    parser.add_argument("--entropy-min", type=float, default=0.88)
    parser.add_argument("--entropy-max", type=float, default=1.30)
    parser.add_argument("--directional-presence", type=float, default=0.0)
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--disaster-stop", type=float, default=2.0)
    parser.add_argument("--min-count", type=int, default=75)
    parser.add_argument("--min-positive-block-fraction", type=float, default=0.60)
    parser.add_argument("--max-block-contribution", type=float, default=0.40)
    args = parser.parse_known_args(argv)[0]

    leaked_predictors = set(PREDICTOR_FEATURES).intersection(LEAKY_OUTCOME_FIELDS)
    if leaked_predictors:
        raise RuntimeError(f"Leaky predictor fields configured: {sorted(leaked_predictors)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frozen = candidate_entries(build_base_entries(prepare_df(args.input), args))
    entries = apply_decluster_methods(frozen)
    blocks = block_summary(entries)
    instruments = instrument_summary(entries)
    summary = candidate_summary(entries, blocks, instruments, args)
    score = build_scorecard(summary, args)

    entries.to_csv(outdir / "decluster_entries.csv", index=False)
    summary.to_csv(outdir / "decluster_candidate_summary.csv", index=False)
    blocks.to_csv(outdir / "decluster_block_summary.csv", index=False)
    instruments.to_csv(outdir / "decluster_instrument_summary.csv", index=False)
    score.to_csv(outdir / "decluster_scorecard.csv", index=False)

    print("APVA frozen topology de-cluster validation complete")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
