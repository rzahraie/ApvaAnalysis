#!/usr/bin/env python3
"""
APVA low directional-presence persistence validation.

Purpose:
Validate whether the D1 -> D1 -> D1 directional-presence transition is a real
stable APVA regime or a small-sample artifact.

Prior result:
    Prev_DirPresenceBin = D1
    Entry_DirPresenceBin = D1
    Next_DirPresenceBin = D1
    count ~53
    mean ~+1.19
    PF ~3.05

This script tests:
    - all directional-presence transition triplets
    - D1 persistence across instrument
    - D1 persistence across week
    - D1 persistence intersected with pressure transition
    - D1 persistence intersected with entropy-delta quartile
    - robustness under multiple minimum-count thresholds

Universe:
    ES, NQ

Leakage rule:
    Only ex-ante state fields are used for gating.
    SignedNormalizedReturn and DirectionalNormalizedMAE are used only for outcome simulation.
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


def week_key(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    iso = dt.dt.isocalendar()
    out = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    out.loc[dt.isna()] = "UnknownWeek"
    return out


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

    df["DirPresenceBin"] = bin_quantile(
        df["RollingDirectionalPresence"],
        ["D1", "D2", "D3", "D4"],
    )
    df["EntropyBin"] = bin_quantile(
        df["RollingEntropy"],
        ["E1", "E2", "E3", "E4"],
    )

    group_cols = ["Instrument", "File"]

    for col in [
        "DominantPressure",
        "DirPresenceBin",
        "EntropyBin",
        "RollingDirectionalPresence",
        "RollingEntropy",
        "RollingVelocity",
        "DominantPressureValue",
    ]:
        if col in df.columns:
            df[f"Prev_{col}"] = df.groupby(group_cols)[col].shift(1)
            df[f"Next_{col}"] = df.groupby(group_cols)[col].shift(-1)

    for col in [
        "RollingDirectionalPresence",
        "RollingEntropy",
        "RollingVelocity",
        "DominantPressureValue",
    ]:
        if col in df.columns:
            df[f"DeltaPrevToEntry_{col}"] = df[col] - df[f"Prev_{col}"]
            df[f"DeltaEntryToNext_{col}"] = df[f"Next_{col}"] - df[col]

    if "Time" in df.columns:
        df["RegimeWeek"] = week_key(df["Time"])
    else:
        df["RegimeWeek"] = "UnknownWeek"

    return df


def apply_policy(entries: pd.DataFrame, disaster_stop: float) -> pd.DataFrame:
    out = entries.copy()
    stopped = out["DirectionalNormalizedMAE"] <= -abs(disaster_stop)
    out["NormalizedPolicyOutcome"] = out["SignedNormalizedReturn"].where(
        ~stopped,
        -abs(disaster_stop),
    )
    out["DisasterStopped"] = stopped
    out["PositiveOutcome"] = out["NormalizedPolicyOutcome"] > 0
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

    entries["DirPresenceTriplet"] = (
        entries["Prev_DirPresenceBin"].astype(str)
        + " > "
        + entries["DirPresenceBin"].astype(str)
        + " > "
        + entries["Next_DirPresenceBin"].astype(str)
    )

    entries["PressureTriplet"] = (
        entries["Prev_DominantPressure"].astype(str)
        + " > "
        + entries["DominantPressure"].astype(str)
        + " > "
        + entries["Next_DominantPressure"].astype(str)
    )

    entries["EntropyTriplet"] = (
        entries["Prev_EntropyBin"].astype(str)
        + " > "
        + entries["EntropyBin"].astype(str)
        + " > "
        + entries["Next_EntropyBin"].astype(str)
    )

    entries["IsD1Persistence"] = entries["DirPresenceTriplet"].eq("D1 > D1 > D1")
    entries["IsRotCompComp"] = entries["PressureTriplet"].eq(
        "RotationalPressure > CompressionPressure > CompressionPressure"
    )

    entries["EntropyDeltaQuartile"] = bin_quantile(
        entries["DeltaPrevToEntry_RollingEntropy"],
        ["Q1", "Q2", "Q3", "Q4"],
    )

    return entries.reset_index(drop=True)


def group_summary(entries: pd.DataFrame, group_cols: list[str], min_count: int) -> pd.DataFrame:
    rows = []

    for key, g in entries.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)

        if len(g) < min_count:
            continue

        row = {col: val for col, val in zip(group_cols, key)}
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


def summarize_d1(entries: pd.DataFrame, min_count: int) -> dict:
    d1 = entries.loc[entries["IsD1Persistence"]].copy()
    base = summarize(entries["NormalizedPolicyOutcome"])

    row = {
        "BaseCount": int(len(entries)),
        "BaseMean": base["Mean"],
        "BaseMedian": base["Median"],
        "BaseTStat": base["TStat"],
        "BaseProfitFactor": base["ProfitFactor"],
        "D1Count": int(len(d1)),
    }

    if len(d1) >= 1:
        s = summarize(d1["NormalizedPolicyOutcome"])
        row.update({
            "D1Mean": s["Mean"],
            "D1Median": s["Median"],
            "D1TStat": s["TStat"],
            "D1WinRate": s["WinRate"],
            "D1ProfitFactor": s["ProfitFactor"],
            "D1StopRate": float(d1["DisasterStopped"].mean()),
            "D1_ES_Count": int((d1["Instrument"].astype(str).str.upper() == "ES").sum()),
            "D1_NQ_Count": int((d1["Instrument"].astype(str).str.upper() == "NQ").sum()),
            "D1InstrumentCount": int(d1["Instrument"].nunique()),
        })

        by_inst = group_summary(d1, ["Instrument"], min_count=1)
        row["D1InstrumentPositiveFraction"] = (
            float((by_inst["Mean"] > 0).mean()) if not by_inst.empty else np.nan
        )
        row["D1InstrumentMedianMean"] = (
            float(by_inst["Mean"].median()) if not by_inst.empty else np.nan
        )

        by_week = group_summary(d1, ["RegimeWeek"], min_count=1)
        row["D1WeekCount"] = int(len(by_week))
        row["D1WeekPositiveFraction"] = (
            float((by_week["Mean"] > 0).mean()) if not by_week.empty else np.nan
        )
        row["D1WeekMedianMean"] = (
            float(by_week["Mean"].median()) if not by_week.empty else np.nan
        )

    return row


def build_scorecard(
    entries: pd.DataFrame,
    triplets: pd.DataFrame,
    d1_by_inst: pd.DataFrame,
    d1_by_week: pd.DataFrame,
    d1_intersections: pd.DataFrame,
) -> pd.DataFrame:
    d1_stats = summarize_d1(entries, min_count=1)
    rows = [{"Metric": k, "Value": v} for k, v in d1_stats.items()]

    if not triplets.empty:
        best = triplets.iloc[0]
        for k in [
            "DirPresenceTriplet",
            "Count",
            "Mean",
            "Median",
            "TStat",
            "WinRate",
            "ProfitFactor",
            "ES_Count",
            "NQ_Count",
        ]:
            rows.append({"Metric": f"BestTriplet_{k}", "Value": best[k]})

    if not d1_intersections.empty:
        best = d1_intersections.iloc[0]
        for k in [
            "IsRotCompComp",
            "EntropyDeltaQuartile",
            "Count",
            "Mean",
            "Median",
            "TStat",
            "WinRate",
            "ProfitFactor",
            "ES_Count",
            "NQ_Count",
        ]:
            rows.append({"Metric": f"BestD1Intersection_{k}", "Value": best[k]})

    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/low_directional_presence_persistence")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--pressure", default="CompressionPressure")
    p.add_argument("--entropy-min", type=float, default=0.88)
    p.add_argument("--entropy-max", type=float, default=1.30)
    p.add_argument("--directional-presence", type=float, default=0.0)
    p.add_argument("--disaster-stop", type=float, default=2.0)
    p.add_argument("--min-count", type=int, default=20)
    args = p.parse_known_args(argv)[0]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = prepare_df(args.input)
    entries = find_entries(df, args)

    triplets = group_summary(entries, ["DirPresenceTriplet"], args.min_count)

    d1 = entries.loc[entries["IsD1Persistence"]].copy()
    d1_by_inst = group_summary(d1, ["Instrument"], min_count=1)
    d1_by_week = group_summary(d1, ["RegimeWeek"], min_count=1)
    d1_by_pressure = group_summary(d1, ["PressureTriplet"], min_count=5)
    d1_intersections = group_summary(
        d1,
        ["IsRotCompComp", "EntropyDeltaQuartile"],
        min_count=5,
    )

    score = build_scorecard(
        entries,
        triplets,
        d1_by_inst,
        d1_by_week,
        d1_intersections,
    )

    entries.to_csv(outdir / "low_dir_presence_entries.csv", index=False)
    triplets.to_csv(outdir / "low_dir_presence_triplets.csv", index=False)
    d1.to_csv(outdir / "d1_persistence_entries.csv", index=False)
    d1_by_inst.to_csv(outdir / "d1_persistence_by_instrument.csv", index=False)
    d1_by_week.to_csv(outdir / "d1_persistence_by_week.csv", index=False)
    d1_by_pressure.to_csv(outdir / "d1_persistence_by_pressure_triplet.csv", index=False)
    d1_intersections.to_csv(outdir / "d1_persistence_intersections.csv", index=False)
    score.to_csv(outdir / "low_dir_presence_scorecard.csv", index=False)

    print("APVA low directional-presence persistence validation complete")
    print(score.to_string(index=False))
    print()
    if not triplets.empty:
        print("Directional-presence triplets:")
        print(triplets.head(20).to_string(index=False))
    print()
    if not d1_by_inst.empty:
        print("D1 by instrument:")
        print(d1_by_inst.to_string(index=False))
    print()
    if not d1_by_week.empty:
        print("D1 by week:")
        print(d1_by_week.to_string(index=False))
    print()
    if not d1_intersections.empty:
        print("D1 intersections:")
        print(d1_intersections.head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    main()