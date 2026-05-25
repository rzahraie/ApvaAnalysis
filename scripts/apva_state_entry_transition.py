#!/usr/bin/env python3
"""
APVA state entry transition analysis.

Purpose
-------
Determine whether the discovered convex APVA state is strongest at state birth
or during continued state occupancy.

State definition:
    ES / 5-bar horizon / CompressionPressure
    RollingDirectionalPresence == 0
    RollingEntropy in [0.88, 1.30]

The script marks contiguous runs of this state within each source File and then
compares:
    - Entry bars: first bar of each run
    - Continuation bars: subsequent bars in the same run

Recommended Jupyter usage:

    import runpy
    ns = runpy.run_path("scripts/apva_state_entry_transition.py", run_name="not_main")
    ns["main"]([
        "--input", "tables/apva_forward_signed_return_dataset_v1.csv",
        "--outdir", "outputs/state_entry_transition_es_compression_entropy_mid_presence_zero"
    ])
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def tstat(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    if len(x) < 2:
        return float("nan")
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / (sd / math.sqrt(len(x)))) if sd > 0 else float("nan")


def top_share(s: pd.Series, frac: float) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna().sort_values(ascending=False)
    if x.empty:
        return float("nan")
    k = max(1, math.ceil(len(x) * frac))
    pos_sum = x[x > 0].sum()
    return float(x.iloc[:k].sum() / pos_sum) if pos_sum != 0 else float("nan")


def week_key(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    iso = dt.dt.isocalendar()
    out = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    out.loc[dt.isna()] = "UnknownWeek"
    return out


def summarize_group(df: pd.DataFrame, label: str) -> dict:
    ret = pd.to_numeric(df["SignedReturn"], errors="coerce").dropna()
    row = {
        "Group": label,
        "Count": int(len(ret)),
        "MeanSignedReturn": float(ret.mean()) if len(ret) else float("nan"),
        "MedianSignedReturn": float(ret.median()) if len(ret) else float("nan"),
        "StdSignedReturn": float(ret.std(ddof=1)) if len(ret) > 1 else float("nan"),
        "TStat": tstat(ret),
        "PositiveFraction": float((ret > 0).mean()) if len(ret) else float("nan"),
        "MaxSignedReturn": float(ret.max()) if len(ret) else float("nan"),
        "MinSignedReturn": float(ret.min()) if len(ret) else float("nan"),
        "Top5SharePositiveSum": top_share(ret, 0.05),
        "Top10SharePositiveSum": top_share(ret, 0.10),
    }
    for col, prefix in [("DirectionalMFE", "MFE"), ("DirectionalMAE", "MAE")]:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            row[f"Mean{prefix}"] = float(v.mean()) if len(v) else float("nan")
            row[f"Median{prefix}"] = float(v.median()) if len(v) else float("nan")
            row[f"Q90{prefix}"] = float(v.quantile(0.90)) if len(v) else float("nan")
            row[f"Q95{prefix}"] = float(v.quantile(0.95)) if len(v) else float("nan")
    return row


def load_and_label(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.input)
    num_cols = [
        "HorizonBars", "BarIndex", "RollingEntropy", "RollingDirectionalPresence",
        "SignedReturn", "DirectionalMFE", "DirectionalMAE"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    required = [
        "Instrument", "File", "BarIndex", "HorizonBars", "DominantPressure",
        "RollingEntropy", "RollingDirectionalPresence", "SignedReturn"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df = df.sort_values(["File", "BarIndex"]).reset_index(drop=True)

    state = (
        df["Instrument"].astype(str).str.upper().eq(args.instrument.upper())
        & df["HorizonBars"].eq(args.horizon)
        & df["DominantPressure"].astype(str).eq(args.pressure)
        & df["RollingEntropy"].between(args.entropy_min, args.entropy_max, inclusive="both")
        & df["RollingDirectionalPresence"].eq(args.directional_presence)
    )
    df["InTargetState"] = state

    prev_state = df.groupby("File")["InTargetState"].shift(1).fillna(False)
    df["StateEntry"] = df["InTargetState"] & ~prev_state
    df["StateContinuation"] = df["InTargetState"] & prev_state

    run_start = df["StateEntry"].astype(int)
    df["StateRunId"] = run_start.groupby(df["File"]).cumsum()
    df.loc[~df["InTargetState"], "StateRunId"] = np.nan
    df["BarsSinceStateEntry"] = df.groupby(["File", "StateRunId"]).cumcount()

    selected = df.loc[df["InTargetState"]].copy().reset_index(drop=True)
    if selected.empty:
        raise RuntimeError("No target-state rows selected")
    if "Time" in selected.columns:
        selected["RegimeWeek"] = week_key(selected["Time"])
    return selected


def build_week_summary(sel: pd.DataFrame) -> pd.DataFrame:
    if "RegimeWeek" not in sel.columns:
        return pd.DataFrame()
    rows = []
    for label, g0 in [("Entry", sel[sel["StateEntry"]]), ("Continuation", sel[sel["StateContinuation"]])]:
        for week, g in g0.groupby("RegimeWeek", dropna=False, sort=True):
            ret = pd.to_numeric(g["SignedReturn"], errors="coerce").dropna()
            if ret.empty:
                continue
            rows.append({
                "Group": label,
                "RegimeWeek": week,
                "Count": int(len(ret)),
                "MeanSignedReturn": float(ret.mean()),
                "MedianSignedReturn": float(ret.median()),
                "TStat": tstat(ret),
                "PositiveFraction": float((ret > 0).mean()),
            })
    return pd.DataFrame(rows)


def build_run_summary(sel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (file, rid), g in sel.groupby(["File", "StateRunId"], dropna=True, sort=True):
        ret = pd.to_numeric(g["SignedReturn"], errors="coerce").dropna()
        if ret.empty:
            continue
        rows.append({
            "File": file,
            "StateRunId": rid,
            "RunLength": int(len(g)),
            "EntryBarIndex": int(g["BarIndex"].iloc[0]),
            "MeanSignedReturn": float(ret.mean()),
            "MaxSignedReturn": float(ret.max()),
            "EntrySignedReturn": float(g["SignedReturn"].iloc[0]),
            "BestMFE": float(g["DirectionalMFE"].max()) if "DirectionalMFE" in g.columns else float("nan"),
            "WorstMAE": float(g["DirectionalMAE"].min()) if "DirectionalMAE" in g.columns else float("nan"),
        })
    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/state_entry_transition_es_compression_entropy_mid_presence_zero")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--pressure", default="CompressionPressure")
    p.add_argument("--entropy-min", type=float, default=0.88)
    p.add_argument("--entropy-max", type=float, default=1.30)
    p.add_argument("--directional-presence", type=float, default=0.0)
    args = p.parse_known_args(argv)[0]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sel = load_and_label(args)
    entry = sel[sel["StateEntry"]].copy()
    cont = sel[sel["StateContinuation"]].copy()

    summary = pd.DataFrame([
        summarize_group(sel, "AllTargetState"),
        summarize_group(entry, "Entry"),
        summarize_group(cont, "Continuation"),
    ])

    week = build_week_summary(sel)
    run_summary = build_run_summary(sel)

    score_rows = []
    for _, row in summary.iterrows():
        score_rows.append({"Metric": f"{row['Group']}_count", "Value": float(row["Count"])})
        score_rows.append({"Metric": f"{row['Group']}_mean", "Value": float(row["MeanSignedReturn"])})
        score_rows.append({"Metric": f"{row['Group']}_median", "Value": float(row["MedianSignedReturn"])})
        score_rows.append({"Metric": f"{row['Group']}_tstat", "Value": float(row["TStat"])})
        score_rows.append({"Metric": f"{row['Group']}_top5_share", "Value": float(row["Top5SharePositiveSum"])})
    if not week.empty:
        for grp, g in week.groupby("Group"):
            score_rows.append({"Metric": f"{grp}_positive_week_fraction", "Value": float((g["MeanSignedReturn"] > 0).mean())})
            score_rows.append({"Metric": f"{grp}_median_week_mean", "Value": float(g["MeanSignedReturn"].median())})
    score = pd.DataFrame(score_rows)

    sel.to_csv(outdir / "selected_state_entry_rows.csv", index=False)
    summary.to_csv(outdir / "entry_transition_summary.csv", index=False)
    week.to_csv(outdir / "entry_transition_by_week.csv", index=False)
    run_summary.to_csv(outdir / "state_run_summary.csv", index=False)
    score.to_csv(outdir / "entry_transition_scorecard.csv", index=False)

    print("APVA state entry transition complete")
    print(score.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
