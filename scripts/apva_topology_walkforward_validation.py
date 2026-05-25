#!/usr/bin/env python3
"""Time-block validation for frozen ES/NQ APVA topology candidates.

This script evaluates previously selected candidate rules over chronological
week blocks. It does not discover, re-rank, or re-bin candidate features.
Outcome columns are used only after leakage-safe entries and candidate rules
have been identified.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from apva_analysis_utils import (
    INDEX_INSTRUMENTS,
    LEAKY_OUTCOME_FIELDS,
    apply_normalized_policy,
    group_summary,
    mark_target_entries,
    prepare_df,
    summarize,
    week_key,
)


RRCCC = (
    "RotationalPressure > RotationalPressure > CompressionPressure > "
    "CompressionPressure > CompressionPressure"
)
CCRRR = (
    "CompressionPressure > CompressionPressure > RotationalPressure > "
    "RotationalPressure > RotationalPressure"
)

# Frozen from outputs/index_topology_family_validation/topology_family_1d_summary.csv.
PRIOR_SLOPE_Q3_MIN = 0.004718046888521732
PRIOR_SLOPE_Q3_MAX = 0.05505235072964343

CANDIDATE_RULES = {
    "Base_ES_NQ": "base compression-entry topology",
    "RRCCC": f"PriorPressureSeq == {RRCCC}",
    "CCRRR": f"PriorPressureSeq == {CCRRR}",
    "PriorSlope_DominantPressureValue_Q3": (
        f"{PRIOR_SLOPE_Q3_MIN} <= PriorSlope_DominantPressureValue <= "
        f"{PRIOR_SLOPE_Q3_MAX} (frozen prior Q3 interval)"
    ),
}

PREDICTOR_FEATURES = [
    "DominantPressure",
    "RollingDirectionalPresence",
    "RollingEntropy",
    "HorizonBars",
    "PriorPressureSeq",
    "PriorSlope_DominantPressureValue",
]


def slope(vals: pd.Series) -> float:
    x = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(float)
    return float(x[-1] - x[0]) if len(x) >= 2 else np.nan


def build_base_entries(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    marked = mark_target_entries(
        df,
        instruments=INDEX_INSTRUMENTS,
        horizon=args.horizon,
        pressure=args.pressure,
        entropy_min=args.entropy_min,
        entropy_max=args.entropy_max,
        directional_presence=args.directional_presence,
    )
    rows = []
    for idx in marked.index[marked["TargetEntry"]]:
        entry = marked.loc[idx]
        prior = marked[
            marked["Instrument"].eq(entry["Instrument"])
            & marked["File"].eq(entry["File"])
            & (marked["BarIndex"] < entry["BarIndex"])
        ].tail(args.lookback)
        if len(prior) < args.lookback:
            continue
        row = entry.copy()
        row["PriorPressureSeq"] = " > ".join(prior["DominantPressure"].astype(str).tolist())
        row["PriorSlope_DominantPressureValue"] = slope(prior["DominantPressureValue"])
        rows.append(row)
    if not rows:
        raise RuntimeError("No base entries with complete prior histories found")
    entries = apply_normalized_policy(pd.DataFrame(rows).reset_index(drop=True), args.disaster_stop)
    entries["RegimeWeek"] = week_key(entries["Time"]) if "Time" in entries else "UnknownWeek"
    return assign_time_blocks(entries)


def assign_time_blocks(entries: pd.DataFrame) -> pd.DataFrame:
    out = entries.copy()
    weeks = sorted(out["RegimeWeek"].dropna().astype(str).unique())
    block_names = {week: f"B{index:02d}_{week}" for index, week in enumerate(weeks, start=1)}
    out["TimeBlock"] = out["RegimeWeek"].map(block_names)
    return out


def candidate_entries(base: pd.DataFrame) -> pd.DataFrame:
    masks = {
        "Base_ES_NQ": pd.Series(True, index=base.index),
        "RRCCC": base["PriorPressureSeq"].eq(RRCCC),
        "CCRRR": base["PriorPressureSeq"].eq(CCRRR),
        "PriorSlope_DominantPressureValue_Q3": base["PriorSlope_DominantPressureValue"].between(
            PRIOR_SLOPE_Q3_MIN,
            PRIOR_SLOPE_Q3_MAX,
            inclusive="both",
        ),
    }
    rows = []
    for candidate, mask in masks.items():
        selected = base.loc[mask].copy()
        selected.insert(0, "CandidateRule", CANDIDATE_RULES[candidate])
        selected.insert(0, "Candidate", candidate)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def block_summary(entries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    totals = entries.groupby("Candidate")["NormalizedPolicyOutcome"].sum()
    for (candidate, block, week), group in entries.groupby(
        ["Candidate", "TimeBlock", "RegimeWeek"], dropna=False, sort=True
    ):
        row = {"Candidate": candidate, "TimeBlock": block, "RegimeWeek": week}
        row.update(summarize(group["NormalizedPolicyOutcome"]))
        row["StopRate"] = float(group["DisasterStopped"].mean())
        row["ES_Count"] = int(group["Instrument"].astype(str).str.upper().eq("ES").sum())
        row["NQ_Count"] = int(group["Instrument"].astype(str).str.upper().eq("NQ").sum())
        total_sum = float(totals.loc[candidate])
        row["NetContributionFraction"] = (
            float(row["Sum"] / total_sum) if total_sum > 0 else np.nan
        )
        row["PositiveBlock"] = bool(row["Mean"] > 0)
        rows.append(row)
    return pd.DataFrame(rows)


def instrument_summary(entries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, instrument), group in entries.groupby(["Candidate", "Instrument"], sort=True):
        row = {"Candidate": candidate, "Instrument": instrument}
        row.update(summarize(group["NormalizedPolicyOutcome"]))
        row["StopRate"] = float(group["DisasterStopped"].mean())
        # Conservative interpretation of "no catastrophic negative result".
        row["NegativeAggregateMean"] = bool(row["Mean"] < 0)
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_summary(
    entries: pd.DataFrame,
    by_block: pd.DataFrame,
    by_instrument: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    base_pf = float(
        summarize(entries.loc[entries["Candidate"].eq("Base_ES_NQ"), "NormalizedPolicyOutcome"])[
            "ProfitFactor"
        ]
    )
    rows = []
    for candidate, group in entries.groupby("Candidate", sort=False):
        blocks = by_block.loc[by_block["Candidate"].eq(candidate)]
        instruments = by_instrument.loc[by_instrument["Candidate"].eq(candidate)]
        instrument_weeks = group_summary(group, ["Instrument", "RegimeWeek"])
        row: dict[str, object] = {
            "Candidate": candidate,
            "CandidateRule": CANDIDATE_RULES[candidate],
        }
        row.update(summarize(group["NormalizedPolicyOutcome"]))
        row["ES_Count"] = int(group["Instrument"].astype(str).str.upper().eq("ES").sum())
        row["NQ_Count"] = int(group["Instrument"].astype(str).str.upper().eq("NQ").sum())
        row["StopRate"] = float(group["DisasterStopped"].mean())
        row["BlockCount"] = int(len(blocks))
        row["PositiveBlockFraction"] = float((blocks["Mean"] > 0).mean())
        row["PositiveInstrumentFraction"] = float((instruments["Mean"] > 0).mean())
        row["InstrumentWeekPositiveFraction"] = float(
            (instrument_weeks["Mean"] > 0).mean()
        )
        row["MaxSingleBlockContributionFraction"] = float(
            blocks["NetContributionFraction"].max()
        )
        row["CountPass"] = bool(row["Count"] >= args.min_count)
        row["BothInstrumentsPass"] = bool(row["ES_Count"] > 0 and row["NQ_Count"] > 0)
        row["MedianPass"] = bool(row["Median"] > 0)
        row["PFAboveBasePass"] = bool(row["ProfitFactor"] > base_pf)
        row["PositiveBlockFractionPass"] = bool(
            row["PositiveBlockFraction"] > args.min_positive_block_fraction
        )
        row["MaxSingleBlockContributionPass"] = bool(
            row["MaxSingleBlockContributionFraction"] <= args.max_block_contribution
        )
        row["NoCatastrophicInstrumentResultPass"] = bool(
            not instruments["NegativeAggregateMean"].any()
        )
        test_fields = [
            "CountPass",
            "BothInstrumentsPass",
            "MedianPass",
            "PFAboveBasePass",
            "PositiveBlockFractionPass",
            "MaxSingleBlockContributionPass",
            "NoCatastrophicInstrumentResultPass",
        ]
        row["ValidationPass"] = bool(all(row[field] for field in test_fields))
        row["ValidationStatus"] = (
            "reference_only" if candidate == "Base_ES_NQ"
            else ("pass" if row["ValidationPass"] else "fail")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def scorecard(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = [
        {"Metric": "block_definition", "Value": "chronological ISO week"},
        {"Metric": "candidate_mode", "Value": "frozen rules only; no candidate search"},
        {"Metric": "prior_slope_q3_min_frozen", "Value": PRIOR_SLOPE_Q3_MIN},
        {"Metric": "prior_slope_q3_max_frozen", "Value": PRIOR_SLOPE_Q3_MAX},
        {"Metric": "count_requirement", "Value": f">= {args.min_count}"},
        {"Metric": "median_requirement", "Value": "> 0"},
        {"Metric": "profit_factor_requirement", "Value": "> Base_ES_NQ PF"},
        {
            "Metric": "positive_block_fraction_requirement",
            "Value": f"> {args.min_positive_block_fraction}",
        },
        {
            "Metric": "max_single_block_contribution_requirement",
            "Value": f"<= {args.max_block_contribution}",
        },
        {
            "Metric": "max_single_block_contribution_definition",
            "Value": "maximum block policy sum / candidate total policy sum when total sum > 0",
        },
        {
            "Metric": "catastrophic_instrument_definition",
            "Value": "conservative: any ES or NQ aggregate mean below zero fails",
        },
        {
            "Metric": "predictor_fields_excluded_as_leaky",
            "Value": ",".join(LEAKY_OUTCOME_FIELDS),
        },
    ]
    for _, candidate in summary.iterrows():
        label = str(candidate["Candidate"])
        for key in [
            "ValidationStatus",
            "Count",
            "ES_Count",
            "NQ_Count",
            "Mean",
            "Median",
            "TStat",
            "ProfitFactor",
            "WinRate",
            "PositiveBlockFraction",
            "PositiveInstrumentFraction",
            "InstrumentWeekPositiveFraction",
            "MaxSingleBlockContributionFraction",
            "BlockCount",
            "ValidationPass",
        ]:
            rows.append({"Metric": f"{label}_{key}", "Value": candidate[key]})
    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    parser.add_argument("--outdir", default="outputs/topology_walkforward_validation")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--pressure", default="CompressionPressure")
    parser.add_argument("--entropy-min", type=float, default=0.88)
    parser.add_argument("--entropy-max", type=float, default=1.30)
    parser.add_argument("--directional-presence", type=float, default=0.0)
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--disaster-stop", type=float, default=2.0)
    parser.add_argument("--min-count", type=int, default=100)
    parser.add_argument("--min-positive-block-fraction", type=float, default=0.60)
    parser.add_argument("--max-block-contribution", type=float, default=0.35)
    args = parser.parse_known_args(argv)[0]

    leaked_predictors = set(PREDICTOR_FEATURES).intersection(LEAKY_OUTCOME_FIELDS)
    if leaked_predictors:
        raise RuntimeError(f"Leaky predictor fields configured: {sorted(leaked_predictors)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    entries = candidate_entries(build_base_entries(prepare_df(args.input), args))
    blocks = block_summary(entries)
    instruments = instrument_summary(entries)
    summary = candidate_summary(entries, blocks, instruments, args)
    score = scorecard(summary, args)

    entries.to_csv(outdir / "walkforward_entries.csv", index=False)
    summary.to_csv(outdir / "walkforward_candidate_summary.csv", index=False)
    blocks.to_csv(outdir / "walkforward_block_summary.csv", index=False)
    instruments.to_csv(outdir / "walkforward_instrument_summary.csv", index=False)
    score.to_csv(outdir / "walkforward_scorecard.csv", index=False)

    print("APVA frozen topology walk-forward validation complete")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
