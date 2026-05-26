#!/usr/bin/env python3
"""Extended fixed-rule validation across independent compatible APVA datasets.

Candidate definitions and thresholds are frozen from prior validation. This
script inventories row-level CSVs, excludes derived analysis outputs and
row-identical dataset copies, then applies fixed validation modes only.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from apva_analysis_utils import LEAKY_OUTCOME_FIELDS, group_summary, prepare_df, summarize
from apva_topology_decluster_validation import DECLUSTER_METHODS, spacing_mask
from apva_topology_walkforward_validation import (
    CANDIDATE_RULES,
    PREDICTOR_FEATURES,
    PRIOR_SLOPE_Q3_MAX,
    PRIOR_SLOPE_Q3_MIN,
    build_base_entries,
    candidate_entries,
)


REQUIRED_ROW_FIELDS = {
    "Instrument",
    "File",
    "BarIndex",
    "HorizonBars",
    "DominantPressure",
    "RollingDirectionalPresence",
    "RollingEntropy",
    "SignedNormalizedReturn",
    "DirectionalNormalizedMAE",
}
ROW_KEY_FIELDS = ["Instrument", "File", "BarIndex", "HorizonBars"]
VALIDATION_MODES = {
    "Reference": DECLUSTER_METHODS["Reference"],
    "Spacing_10": DECLUSTER_METHODS["Spacing_10"],
    "Spacing_20": DECLUSTER_METHODS["Spacing_20"],
}


def candidate_csv_paths(workspace: Path, outdir: Path) -> list[Path]:
    paths: set[Path] = set()
    for root_name in ["tables", "outputs", "data"]:
        root = workspace / root_name
        if root.exists():
            paths.update(root.rglob("*.csv"))
    paths.update(workspace.glob("*.csv"))
    return sorted(path for path in paths if outdir not in path.parents)


def header_fields(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle), [])


def row_key_set(path: Path) -> set[tuple[str, ...]]:
    df = pd.read_csv(path, usecols=ROW_KEY_FIELDS)
    return set(map(tuple, df[ROW_KEY_FIELDS].astype(str).to_numpy().tolist()))


def inventory_datasets(
    workspace: Path,
    primary: Path,
    outdir: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    primary_keys = row_key_set(primary)
    rows = []
    included = []
    for path in candidate_csv_paths(workspace, outdir):
        rel = path.relative_to(workspace).as_posix()
        fields = header_fields(path)
        missing = sorted(REQUIRED_ROW_FIELDS - set(fields))
        row: dict[str, object] = {
            "Dataset": rel,
            "Location": rel.split("/", maxsplit=1)[0],
            "ColumnCount": len(fields),
            "RequiredSchemaMatch": not missing,
            "MissingRequiredFields": ",".join(missing),
            "Included": False,
            "Reason": "",
            "RowCount": np.nan,
            "InstrumentCount": np.nan,
            "TimeMin": "",
            "TimeMax": "",
            "PrimaryRowKeyOverlapCount": np.nan,
            "PrimaryRowKeyOverlapFraction": np.nan,
        }
        if missing:
            row["Reason"] = "excluded: missing required row-level forward-return fields"
            rows.append(row)
            continue
        df = pd.read_csv(path)
        keys = set(map(tuple, df[ROW_KEY_FIELDS].astype(str).to_numpy().tolist()))
        overlap = len(keys & primary_keys)
        time = pd.to_datetime(df["Time"], errors="coerce") if "Time" in df else pd.Series(dtype="datetime64[ns]")
        row.update({
            "RowCount": int(len(df)),
            "InstrumentCount": int(df["Instrument"].nunique()),
            "TimeMin": str(time.min()) if not time.empty else "",
            "TimeMax": str(time.max()) if not time.empty else "",
            "PrimaryRowKeyOverlapCount": overlap,
            "PrimaryRowKeyOverlapFraction": overlap / len(keys) if keys else np.nan,
        })
        if path.resolve() == primary.resolve():
            row["Included"] = True
            row["Reason"] = "included: canonical primary row-level forward-return dataset"
            included.append(path)
        elif rel.startswith("outputs/"):
            row["Reason"] = "excluded: prior analysis output/subset, not an additional validation dataset"
        elif keys == primary_keys:
            row["Reason"] = "excluded: row-identical derivative of primary dataset; no additional observations"
        else:
            row["Included"] = True
            row["Reason"] = "included: compatible row-level dataset with additional row keys"
            included.append(path)
        rows.append(row)
    return pd.DataFrame(rows), included


def apply_validation_modes(entries: pd.DataFrame) -> pd.DataFrame:
    masks = {
        "Reference": pd.Series(True, index=entries.index),
        "Spacing_10": spacing_mask(entries, 10),
        "Spacing_20": spacing_mask(entries, 20),
    }
    frames = []
    for mode, mask in masks.items():
        selected = entries.loc[mask].copy()
        selected.insert(0, "ValidationModeRule", VALIDATION_MODES[mode])
        selected.insert(0, "ValidationMode", mode)
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def build_entries(paths: list[Path], workspace: Path, args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for path in paths:
        base = build_base_entries(prepare_df(str(path)), args)
        base = base.loc[base["NormalizedPolicyOutcome"].notna()].copy()
        if base.empty:
            continue
        frozen = candidate_entries(base)
        frozen.insert(0, "Dataset", path.relative_to(workspace).as_posix())
        frames.append(apply_validation_modes(frozen))
    if not frames:
        raise RuntimeError("No independent compatible datasets were included for validation")
    return pd.concat(frames, ignore_index=True)


def summarize_blocks(entries: pd.DataFrame, pooled: bool) -> pd.DataFrame:
    scope_cols = ["ValidationMode", "Candidate"] if pooled else ["Dataset", "ValidationMode", "Candidate"]
    total = entries.groupby(scope_cols)["NormalizedPolicyOutcome"].sum()
    rows = []
    block_group = scope_cols + ["Dataset", "TimeBlock", "RegimeWeek"] if pooled else scope_cols + ["TimeBlock", "RegimeWeek"]
    for keys, group in entries.groupby(block_group, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(block_group, keys)}
        row["Scope"] = "Pooled" if pooled else "Dataset"
        row.update(summarize(group["NormalizedPolicyOutcome"]))
        row["StopRate"] = float(group["DisasterStopped"].mean())
        row["ES_Count"] = int(group["Instrument"].astype(str).str.upper().eq("ES").sum())
        row["NQ_Count"] = int(group["Instrument"].astype(str).str.upper().eq("NQ").sum())
        total_key = tuple(row[col] for col in scope_cols)
        total_sum = float(total.loc[total_key if len(total_key) > 1 else total_key[0]])
        row["NetContributionFraction"] = float(row["Sum"] / total_sum) if total_sum > 0 else np.nan
        row["PositiveBlock"] = bool(row["Mean"] > 0)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_instruments(entries: pd.DataFrame, pooled: bool) -> pd.DataFrame:
    scope_cols = ["ValidationMode", "Candidate"] if pooled else ["Dataset", "ValidationMode", "Candidate"]
    rows = []
    for keys, group in entries.groupby(scope_cols + ["Instrument"], dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(scope_cols + ["Instrument"], keys)}
        row["Scope"] = "Pooled" if pooled else "Dataset"
        row.update(summarize(group["NormalizedPolicyOutcome"]))
        row["StopRate"] = float(group["DisasterStopped"].mean())
        row["NegativeAggregateMean"] = bool(row["Mean"] < 0)
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_summaries(
    entries: pd.DataFrame,
    blocks: pd.DataFrame,
    instruments: pd.DataFrame,
    args: argparse.Namespace,
    pooled: bool,
) -> pd.DataFrame:
    scope_cols = ["ValidationMode"] if pooled else ["Dataset", "ValidationMode"]
    rows = []
    for scope_key, scope_entries in entries.groupby(scope_cols, sort=True):
        if not isinstance(scope_key, tuple):
            scope_key = (scope_key,)
        scope = {col: val for col, val in zip(scope_cols, scope_key)}
        mode = scope["ValidationMode"]
        base = scope_entries.loc[scope_entries["Candidate"].eq("Base_ES_NQ")]
        base_pf = float(summarize(base["NormalizedPolicyOutcome"])["ProfitFactor"])
        for candidate in CANDIDATE_RULES:
            group = scope_entries.loc[scope_entries["Candidate"].eq(candidate)]
            block_mask = blocks["Scope"].eq("Pooled" if pooled else "Dataset")
            instrument_mask = instruments["Scope"].eq("Pooled" if pooled else "Dataset")
            for col, val in scope.items():
                block_mask &= blocks[col].eq(val)
                instrument_mask &= instruments[col].eq(val)
            block_mask &= blocks["Candidate"].eq(candidate)
            instrument_mask &= instruments["Candidate"].eq(candidate)
            candidate_blocks = blocks.loc[block_mask]
            candidate_instruments = instruments.loc[instrument_mask]
            inst_week_cols = ["Dataset", "Instrument", "RegimeWeek"] if pooled else ["Instrument", "RegimeWeek"]
            inst_weeks = group_summary(group, inst_week_cols)
            es = candidate_instruments.loc[candidate_instruments["Instrument"].astype(str).str.upper().eq("ES")]
            nq = candidate_instruments.loc[candidate_instruments["Instrument"].astype(str).str.upper().eq("NQ")]
            row: dict[str, object] = {
                **scope,
                "Scope": "Pooled" if pooled else "Dataset",
                "Candidate": candidate,
                "CandidateRule": CANDIDATE_RULES[candidate],
                "ValidationModeRule": VALIDATION_MODES[mode],
                "BaseMethodProfitFactor": base_pf,
            }
            row.update(summarize(group["NormalizedPolicyOutcome"]))
            row["ES_Count"] = int(group["Instrument"].astype(str).str.upper().eq("ES").sum())
            row["NQ_Count"] = int(group["Instrument"].astype(str).str.upper().eq("NQ").sum())
            row["StopRate"] = float(group["DisasterStopped"].mean())
            row["BlockCount"] = int(len(candidate_blocks))
            row["PositiveBlockFraction"] = float((candidate_blocks["Mean"] > 0).mean())
            row["InstrumentWeekPositiveFraction"] = float((inst_weeks["Mean"] > 0).mean())
            row["MaxSingleBlockContributionFraction"] = float(
                candidate_blocks["NetContributionFraction"].max()
            )
            row["ES_Mean"] = float(es["Mean"].iloc[0]) if not es.empty else np.nan
            row["NQ_Mean"] = float(nq["Mean"].iloc[0]) if not nq.empty else np.nan
            row["NegativeInstrumentAggregateFlag"] = bool(candidate_instruments["NegativeAggregateMean"].any())
            row["CountPass"] = bool(row["Count"] >= args.min_count)
            row["BothInstrumentsPass"] = bool(row["ES_Count"] > 0 and row["NQ_Count"] > 0)
            row["MedianPass"] = bool(row["Median"] > 0)
            row["PFAboveMethodBasePass"] = bool(row["ProfitFactor"] > base_pf)
            row["PositiveBlockFractionPass"] = bool(row["PositiveBlockFraction"] > args.min_positive_block_fraction)
            row["MaxSingleBlockContributionPass"] = bool(
                row["MaxSingleBlockContributionFraction"] <= args.max_block_contribution
            )
            row["InstrumentMeansPass"] = bool(
                pd.notna(row["ES_Mean"]) and pd.notna(row["NQ_Mean"])
                and row["ES_Mean"] >= 0 and row["NQ_Mean"] >= 0
            )
            criteria = [
                "CountPass", "BothInstrumentsPass", "MedianPass", "PFAboveMethodBasePass",
                "PositiveBlockFractionPass", "MaxSingleBlockContributionPass", "InstrumentMeansPass",
            ]
            row["ValidationPass"] = bool(all(row[field] for field in criteria))
            row["ValidationStatus"] = (
                "reference_only" if candidate == "Base_ES_NQ"
                else ("pass" if row["ValidationPass"] else "fail")
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_scorecard(inventory: pd.DataFrame, pooled: pd.DataFrame, dataset: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    included = inventory.loc[inventory["Included"], "Dataset"].tolist()
    rows = [
        {"Metric": "candidate_mode", "Value": "frozen rules only; no candidate search"},
        {"Metric": "compatible_independent_dataset_count", "Value": len(included)},
        {"Metric": "included_datasets", "Value": ",".join(included)},
        {
            "Metric": "pooled_interpretation",
            "Value": "single included dataset; pooled summaries do not provide cross-dataset replication"
            if len(included) == 1 else "pooled across independent compatible datasets",
        },
        {"Metric": "prior_slope_q3_min_frozen", "Value": PRIOR_SLOPE_Q3_MIN},
        {"Metric": "prior_slope_q3_max_frozen", "Value": PRIOR_SLOPE_Q3_MAX},
        {"Metric": "count_requirement", "Value": f">= {args.min_count}"},
        {"Metric": "profit_factor_requirement", "Value": "> method-matched Base_ES_NQ PF"},
        {"Metric": "positive_block_fraction_requirement", "Value": f"> {args.min_positive_block_fraction}"},
        {"Metric": "max_single_block_contribution_requirement", "Value": f"<= {args.max_block_contribution}"},
        {"Metric": "instrument_mean_requirement", "Value": "ES mean >= 0 and NQ mean >= 0"},
        {"Metric": "predictor_fields_excluded_as_leaky", "Value": ",".join(LEAKY_OUTCOME_FIELDS)},
    ]
    for mode in VALIDATION_MODES:
        subset = pooled.loc[pooled["ValidationMode"].eq(mode)]
        passed = subset.loc[subset["Candidate"].ne("Base_ES_NQ") & subset["ValidationPass"], "Candidate"].tolist()
        rows.append({"Metric": f"Pooled_{mode}_passing_candidates", "Value": ",".join(passed) if passed else "none"})
        for _, candidate in subset.iterrows():
            label = f"Pooled_{mode}_{candidate['Candidate']}"
            for key in [
                "ValidationStatus", "Count", "ES_Count", "NQ_Count", "Mean", "Median",
                "TStat", "ProfitFactor", "BaseMethodProfitFactor", "WinRate",
                "PositiveBlockFraction", "InstrumentWeekPositiveFraction",
                "MaxSingleBlockContributionFraction", "ES_Mean", "NQ_Mean",
                "NegativeInstrumentAggregateFlag", "BlockCount", "ValidationPass",
            ]:
                rows.append({"Metric": f"{label}_{key}", "Value": candidate[key]})
    rows.append({"Metric": "dataset_summary_row_count", "Value": len(dataset)})
    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    parser.add_argument("--outdir", default="outputs/fixed_candidate_extended_validation")
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

    workspace = Path(args.workspace).resolve()
    primary = (workspace / args.input).resolve()
    outdir = (workspace / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    inventory, included = inventory_datasets(workspace, primary, outdir)
    entries = build_entries(included, workspace, args)
    dataset_blocks = summarize_blocks(entries, pooled=False)
    pooled_blocks = summarize_blocks(entries, pooled=True)
    blocks = pd.concat([dataset_blocks, pooled_blocks], ignore_index=True, sort=False)
    dataset_instruments = summarize_instruments(entries, pooled=False)
    pooled_instruments = summarize_instruments(entries, pooled=True)
    instruments = pd.concat([dataset_instruments, pooled_instruments], ignore_index=True, sort=False)
    dataset_summary = candidate_summaries(entries, blocks, instruments, args, pooled=False)
    pooled_summary = candidate_summaries(entries, blocks, instruments, args, pooled=True)
    scorecard = build_scorecard(inventory, pooled_summary, dataset_summary, args)

    inventory.to_csv(outdir / "extended_dataset_inventory.csv", index=False)
    entries.to_csv(outdir / "extended_entries.csv", index=False)
    pooled_summary.to_csv(outdir / "extended_candidate_summary.csv", index=False)
    dataset_summary.to_csv(outdir / "extended_dataset_candidate_summary.csv", index=False)
    blocks.to_csv(outdir / "extended_block_summary.csv", index=False)
    instruments.to_csv(outdir / "extended_instrument_summary.csv", index=False)
    scorecard.to_csv(outdir / "extended_scorecard.csv", index=False)
    print("APVA fixed-candidate extended validation complete")
    print(inventory.loc[inventory["RequiredSchemaMatch"], ["Dataset", "Included", "Reason"]].to_string(index=False))
    print()
    print(pooled_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
