#!/usr/bin/env python3
"""
APVA state transition hazard analysis.

Purpose:
Move from static topology rules to dynamic state-transition modeling.

Prior key result:
    PriorLast_RollingDirectionalPresence Q3 improved the ES/NQ compression-entry
    topology substantially.

Hypothesis:
    The edge is tied to directional exhaustion / partial directional presence
    immediately before compression equilibrium and release.

This script builds transition statistics around target compression-entry states.

It analyzes:
    - prior state -> entry state transitions
    - entry state -> next state transitions
    - pressure-state transition pairs
    - directional-presence regime transitions
    - entropy-regime transitions
    - transition hazards associated with positive normalized policy outcome

Outputs:
    transition_hazard_entries.csv
    transition_hazard_pressure_pairs.csv
    transition_hazard_context_pairs.csv
    transition_hazard_scorecard.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


INDEX_INSTRUMENTS = {"ES", "NQ"}


def tstat(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    if len(x) < 2:
        return float("nan")
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / (sd / math.sqrt(len(x)))) if sd > 0 else float("nan")


def summarize(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce").dropna()
    wins = x[x > 0]
    losses = x[x < 0]
    gw = wins.sum()
    gl = -losses.sum()

    return {
        "Count": int(len(x)),
        "Mean": float(x.mean()) if len(x) else np.nan,
        "Median": float(x.median()) if len(x) else np.nan,
        "TStat": tstat(x),
        "WinRate": float((x > 0).mean()) if len(x) else np.nan,
        "ProfitFactor": float(gw / gl) if gl > 0 else float("inf"),
        "Min": float(x.min()) if len(x) else np.nan,
        "Q25": float(x.quantile(0.25)) if len(x) else np.nan,
        "Q75": float(x.quantile(0.75)) if len(x) else np.nan,
        "Q90": float(x.quantile(0.90)) if len(x) else np.nan,
        "Max": float(x.max()) if len(x) else np.nan,
        "Sum": float(x.sum()) if len(x) else np.nan,
    }


def bin_quantile(series: pd.Series, labels: list[str]) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series("NA", index=series.index, dtype="object")
    ok = x.notna()

    if ok.sum() < len(labels):
        return out

    try:
        out.loc[ok] = pd.qcut(
            x.loc[ok].rank(method="first"),
            q=len(labels),
            labels=labels,
        ).astype(str)
    except Exception:
        out.loc[ok] = "BIN_FAIL"

    return out


def prepare_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric = [
        "BarIndex",
        "HorizonBars",
        "SignedNormalizedReturn",
        "DirectionalNormalizedMAE",
        "RollingEntropy",
        "RollingDirectionalPresence",
        "RollingVelocity",
        "DominantPressureValue",
    ]

    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    required = [
        "Instrument",
        "File",
        "BarIndex",
        "HorizonBars",
        "DominantPressure",
        "RollingEntropy",
        "RollingDirectionalPresence",
        "SignedNormalizedReturn",
        "DirectionalNormalizedMAE",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df = df.sort_values(["Instrument", "File", "BarIndex"]).reset_index(drop=True)

    # Ex-ante context bins computed globally over dataset.
    df["EntropyBin"] = bin_quantile(df["RollingEntropy"], ["E1", "E2", "E3", "E4"])
    df["DirPresenceBin"] = bin_quantile(
        df["RollingDirectionalPresence"],
        ["D1", "D2", "D3", "D4"],
    )
    df["VelocityBin"] = bin_quantile(df["RollingVelocity"], ["V1", "V2", "V3", "V4"])
    df["PressureValueBin"] = bin_quantile(
        df["DominantPressureValue"],
        ["P1", "P2", "P3", "P4"],
    )

    group_cols = ["Instrument", "File"]

    for col in [
        "DominantPressure",
        "ResolvedArchetype",
        "MacroState",
        "EntropyBin",
        "DirPresenceBin",
        "VelocityBin",
        "PressureValueBin",
    ]:
        if col in df.columns:
            df[f"Prev_{col}"] = df.groupby(group_cols)[col].shift(1)
            df[f"Next_{col}"] = df.groupby(group_cols)[col].shift(-1)

    for col in [
        "RollingEntropy",
        "RollingDirectionalPresence",
        "RollingVelocity",
        "DominantPressureValue",
    ]:
        if col in df.columns:
            df[f"Prev_{col}"] = df.groupby(group_cols)[col].shift(1)
            df[f"Next_{col}"] = df.groupby(group_cols)[col].shift(-1)
            df[f"DeltaPrevToEntry_{col}"] = df[col] - df[f"Prev_{col}"]
            df[f"DeltaEntryToNext_{col}"] = df[f"Next_{col}"] - df[col]

    return df


def apply_policy(entries: pd.DataFrame, disaster_stop: float) -> pd.DataFrame:
    out = entries.copy()
    stopped = out["DirectionalNormalizedMAE"] <= -abs(disaster_stop)
    out["NormalizedPolicyOutcome"] = out["SignedNormalizedReturn"].where(
        ~stopped,
        -abs(disaster_stop),
    )
    out["DisasterStopped"] = stopped
    return out


def find_entries(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    state = (
        df["Instrument"].astype(str).str.upper().isin(INDEX_INSTRUMENTS)
        & df["HorizonBars"].eq(args.horizon)
        & df["DominantPressure"].astype(str).eq(args.pressure)
        & df["RollingEntropy"].between(args.entropy_min, args.entropy_max, inclusive="both")
        & df["RollingDirectionalPresence"].eq(args.directional_presence)
    )

    df["InTargetState"] = state
    prev_state = (
        df.groupby(["Instrument", "File"])["InTargetState"]
        .shift(1)
        .astype("boolean")
        .fillna(False)
    )
    df["TargetEntry"] = df["InTargetState"] & ~prev_state

    entries = df.loc[df["TargetEntry"]].copy()
    if entries.empty:
        raise RuntimeError("No target entries found")

    entries = apply_policy(entries, args.disaster_stop)
    entries["PositiveOutcome"] = entries["NormalizedPolicyOutcome"] > 0

    return entries.reset_index(drop=True)


def pair_summary(entries: pd.DataFrame, pair_cols: list[str], min_count: int) -> pd.DataFrame:
    rows = []

    for key, g in entries.groupby(pair_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)

        if len(g) < min_count:
            continue

        row = {col: val for col, val in zip(pair_cols, key)}
        row.update(summarize(g["NormalizedPolicyOutcome"]))
        row["StopRate"] = float(g["DisasterStopped"].mean())
        row["PositiveOutcomeFraction"] = float(g["PositiveOutcome"].mean())
        row["InstrumentCount"] = int(g["Instrument"].nunique())
        row["ES_Count"] = int((g["Instrument"].astype(str).str.upper() == "ES").sum())
        row["NQ_Count"] = int((g["Instrument"].astype(str).str.upper() == "NQ").sum())
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(
        ["Mean", "Median", "ProfitFactor", "TStat", "Count"],
        ascending=False,
    )


def numeric_delta_summary(entries: pd.DataFrame, min_count: int) -> pd.DataFrame:
    features = [
        "DeltaPrevToEntry_RollingEntropy",
        "DeltaPrevToEntry_RollingDirectionalPresence",
        "DeltaPrevToEntry_RollingVelocity",
        "DeltaPrevToEntry_DominantPressureValue",
        "DeltaEntryToNext_RollingEntropy",
        "DeltaEntryToNext_RollingDirectionalPresence",
        "DeltaEntryToNext_RollingVelocity",
        "DeltaEntryToNext_DominantPressureValue",
    ]

    rows = []

    for feature in features:
        if feature not in entries.columns:
            continue

        vals = pd.to_numeric(entries[feature], errors="coerce")
        valid = entries.loc[vals.notna()].copy()

        if len(valid) < min_count * 2:
            continue

        valid["_rank"] = vals.loc[valid.index].rank(method="first")
        valid["_q"] = pd.qcut(valid["_rank"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

        for q, g in valid.groupby("_q", observed=False):
            if len(g) < min_count:
                continue

            row = {
                "Feature": feature,
                "Gate": str(q),
                "FeatureMean": float(pd.to_numeric(g[feature], errors="coerce").mean()),
                "FeatureMin": float(pd.to_numeric(g[feature], errors="coerce").min()),
                "FeatureMax": float(pd.to_numeric(g[feature], errors="coerce").max()),
            }
            row.update(summarize(g["NormalizedPolicyOutcome"]))
            row["StopRate"] = float(g["DisasterStopped"].mean())
            row["InstrumentCount"] = int(g["Instrument"].nunique())
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(
        ["Mean", "Median", "ProfitFactor", "TStat", "Count"],
        ascending=False,
    )


def build_scorecard(
    entries: pd.DataFrame,
    pressure_pairs: pd.DataFrame,
    context_pairs: pd.DataFrame,
    deltas: pd.DataFrame,
) -> pd.DataFrame:
    base = summarize(entries["NormalizedPolicyOutcome"])

    rows = [
        {"Metric": "entry_count", "Value": float(len(entries))},
        {"Metric": "base_mean", "Value": base["Mean"]},
        {"Metric": "base_median", "Value": base["Median"]},
        {"Metric": "base_tstat", "Value": base["TStat"]},
        {"Metric": "base_profit_factor", "Value": base["ProfitFactor"]},
        {"Metric": "base_win_rate", "Value": base["WinRate"]},
        {"Metric": "base_stop_rate", "Value": float(entries["DisasterStopped"].mean())},
    ]

    if not pressure_pairs.empty:
        b = pressure_pairs.iloc[0]
        rows.extend([
            {"Metric": "best_pressure_prev", "Value": b.get("Prev_DominantPressure", "")},
            {"Metric": "best_pressure_entry", "Value": b.get("DominantPressure", "")},
            {"Metric": "best_pressure_next", "Value": b.get("Next_DominantPressure", "")},
            {"Metric": "best_pressure_count", "Value": float(b["Count"])},
            {"Metric": "best_pressure_mean", "Value": float(b["Mean"])},
            {"Metric": "best_pressure_median", "Value": float(b["Median"])},
            {"Metric": "best_pressure_tstat", "Value": float(b["TStat"])},
            {"Metric": "best_pressure_profit_factor", "Value": float(b["ProfitFactor"])},
        ])

    if not context_pairs.empty:
        b = context_pairs.iloc[0]
        rows.extend([
            {"Metric": "best_context_cols", "Value": "Prev_DirPresenceBin,DirPresenceBin,Next_DirPresenceBin"},
            {"Metric": "best_context_prev", "Value": b.get("Prev_DirPresenceBin", "")},
            {"Metric": "best_context_entry", "Value": b.get("DirPresenceBin", "")},
            {"Metric": "best_context_next", "Value": b.get("Next_DirPresenceBin", "")},
            {"Metric": "best_context_count", "Value": float(b["Count"])},
            {"Metric": "best_context_mean", "Value": float(b["Mean"])},
            {"Metric": "best_context_median", "Value": float(b["Median"])},
            {"Metric": "best_context_tstat", "Value": float(b["TStat"])},
            {"Metric": "best_context_profit_factor", "Value": float(b["ProfitFactor"])},
        ])

    if not deltas.empty:
        b = deltas.iloc[0]
        rows.extend([
            {"Metric": "best_delta_feature", "Value": b["Feature"]},
            {"Metric": "best_delta_gate", "Value": b["Gate"]},
            {"Metric": "best_delta_count", "Value": float(b["Count"])},
            {"Metric": "best_delta_mean", "Value": float(b["Mean"])},
            {"Metric": "best_delta_median", "Value": float(b["Median"])},
            {"Metric": "best_delta_tstat", "Value": float(b["TStat"])},
            {"Metric": "best_delta_profit_factor", "Value": float(b["ProfitFactor"])},
        ])

    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/state_transition_hazard")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--pressure", default="CompressionPressure")
    p.add_argument("--entropy-min", type=float, default=0.88)
    p.add_argument("--entropy-max", type=float, default=1.30)
    p.add_argument("--directional-presence", type=float, default=0.0)
    p.add_argument("--disaster-stop", type=float, default=2.0)
    p.add_argument("--min-count", type=int, default=30)
    args = p.parse_known_args(argv)[0]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = prepare_df(args.input)
    entries = find_entries(df, args)

    pressure_pairs = pair_summary(
        entries,
        ["Prev_DominantPressure", "DominantPressure", "Next_DominantPressure"],
        args.min_count,
    )

    context_pairs = pair_summary(
        entries,
        ["Prev_DirPresenceBin", "DirPresenceBin", "Next_DirPresenceBin"],
        args.min_count,
    )

    entropy_pairs = pair_summary(
        entries,
        ["Prev_EntropyBin", "EntropyBin", "Next_EntropyBin"],
        args.min_count,
    )

    velocity_pairs = pair_summary(
        entries,
        ["Prev_VelocityBin", "VelocityBin", "Next_VelocityBin"],
        args.min_count,
    )

    deltas = numeric_delta_summary(entries, args.min_count)

    score = build_scorecard(entries, pressure_pairs, context_pairs, deltas)

    entries.to_csv(outdir / "transition_hazard_entries.csv", index=False)
    pressure_pairs.to_csv(outdir / "transition_hazard_pressure_pairs.csv", index=False)
    context_pairs.to_csv(outdir / "transition_hazard_dir_presence_pairs.csv", index=False)
    entropy_pairs.to_csv(outdir / "transition_hazard_entropy_pairs.csv", index=False)
    velocity_pairs.to_csv(outdir / "transition_hazard_velocity_pairs.csv", index=False)
    deltas.to_csv(outdir / "transition_hazard_delta_summary.csv", index=False)
    score.to_csv(outdir / "transition_hazard_scorecard.csv", index=False)

    print("APVA state transition hazard analysis complete")
    print(score.to_string(index=False))
    print()
    if not pressure_pairs.empty:
        print("Pressure pairs:")
        print(pressure_pairs.head(15).to_string(index=False))
    print()
    if not context_pairs.empty:
        print("Directional presence transition pairs:")
        print(context_pairs.head(15).to_string(index=False))
    print()
    if not deltas.empty:
        print("Delta summary:")
        print(deltas.head(15).to_string(index=False))

    return 0


if __name__ == "__main__":
    main()