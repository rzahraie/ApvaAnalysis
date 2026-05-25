#!/usr/bin/env python3
"""
APVA precursor sequence analysis.

Purpose
-------
Analyze what happens immediately BEFORE the discovered transient APVA state.

Target state:
    ES / 5-bar horizon / CompressionPressure
    RollingDirectionalPresence == 0
    RollingEntropy in [0.88, 1.30]

Prior result:
    The edge is concentrated at state birth, not continued occupancy.

This script examines N bars before each target-state entry and summarizes:
    - prior DominantPressure sequence
    - prior ResolvedArchetype sequence
    - prior MacroState sequence
    - prior entropy / directional presence / velocity levels
    - return and excursion outcomes after target-state entry

Recommended Jupyter usage:

    import runpy
    ns = runpy.run_path("scripts/apva_precursor_sequence.py", run_name="not_main")
    ns["main"]([
        "--input", "tables/apva_forward_signed_return_dataset_v1.csv",
        "--outdir", "outputs/precursor_sequence_es_compression_entropy_mid_presence_zero"
    ])

Outputs
-------
- precursor_entry_rows.csv
- precursor_sequence_summary.csv
- precursor_path_summary.csv
- precursor_offset_summary.csv
- precursor_scorecard.csv
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
    pos = x[x > 0].sum()
    return float(x.iloc[:k].sum() / pos) if pos != 0 else float("nan")


def join_seq(values: list[object]) -> str:
    out = []
    for v in values:
        if pd.isna(v):
            out.append("NA")
        else:
            out.append(str(v).strip().replace(",", ";"))
    return " > ".join(out)


def summarize_returns(df: pd.DataFrame, group_name: str) -> dict:
    r = pd.to_numeric(df["SignedReturn"], errors="coerce").dropna()
    row = {
        "Group": group_name,
        "Count": int(len(r)),
        "MeanSignedReturn": float(r.mean()) if len(r) else float("nan"),
        "MedianSignedReturn": float(r.median()) if len(r) else float("nan"),
        "TStat": tstat(r),
        "PositiveFraction": float((r > 0).mean()) if len(r) else float("nan"),
        "MaxSignedReturn": float(r.max()) if len(r) else float("nan"),
        "MinSignedReturn": float(r.min()) if len(r) else float("nan"),
        "Top5SharePositiveSum": top_share(r, 0.05),
        "Top10SharePositiveSum": top_share(r, 0.10),
    }
    for col in ["DirectionalMFE", "DirectionalMAE", "RollingEntropy", "RollingDirectionalPresence", "RollingVelocity", "DominantPressureValue"]:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce").dropna()
            row[f"Mean_{col}"] = float(x.mean()) if len(x) else float("nan")
            row[f"Median_{col}"] = float(x.median()) if len(x) else float("nan")
    return row


def load_and_mark(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.input)
    numeric_cols = [
        "BarIndex", "HorizonBars", "RollingEntropy", "RollingDirectionalPresence",
        "RollingVelocity", "DominantPressureValue", "SignedReturn", "DirectionalMFE", "DirectionalMAE"
    ]
    for c in numeric_cols:
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
    target = (
        df["Instrument"].astype(str).str.upper().eq(args.instrument.upper())
        & df["HorizonBars"].eq(args.horizon)
        & df["DominantPressure"].astype(str).eq(args.pressure)
        & df["RollingEntropy"].between(args.entropy_min, args.entropy_max, inclusive="both")
        & df["RollingDirectionalPresence"].eq(args.directional_presence)
    )
    df["InTargetState"] = target
    prev = df.groupby("File")["InTargetState"].shift(1).fillna(False)
    df["TargetEntry"] = df["InTargetState"] & ~prev
    return df


def build_entry_precursors(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    entries = df.index[df["TargetEntry"]].tolist()
    rows = []
    seq_cols = [c for c in ["DominantPressure", "ResolvedArchetype", "MacroState"] if c in df.columns]
    numeric_cols = [c for c in ["RollingEntropy", "RollingDirectionalPresence", "RollingVelocity", "DominantPressureValue"] if c in df.columns]

    for idx in entries:
        entry = df.loc[idx]
        file = entry["File"]
        bar = entry["BarIndex"]
        prior = df[(df["File"].eq(file)) & (df["BarIndex"] < bar)].tail(args.lookback).copy()
        if len(prior) < args.min_prior_bars:
            continue

        row = {
            "File": file,
            "EntryBarIndex": int(bar),
            "EntryTime": entry.get("Time", ""),
            "SignedReturn": entry["SignedReturn"],
            "DirectionalMFE": entry.get("DirectionalMFE", np.nan),
            "DirectionalMAE": entry.get("DirectionalMAE", np.nan),
            "EntryRollingEntropy": entry.get("RollingEntropy", np.nan),
            "EntryRollingDirectionalPresence": entry.get("RollingDirectionalPresence", np.nan),
            "EntryDominantPressure": entry.get("DominantPressure", ""),
            "PriorBarCount": int(len(prior)),
        }
        for col in seq_cols:
            vals = prior[col].tolist()
            row[f"Prior{col}Seq"] = join_seq(vals)
            row[f"Prev{col}"] = vals[-1] if vals else ""
        for col in numeric_cols:
            x = pd.to_numeric(prior[col], errors="coerce").dropna()
            row[f"PriorMean_{col}"] = float(x.mean()) if len(x) else np.nan
            row[f"PriorLast_{col}"] = float(x.iloc[-1]) if len(x) else np.nan
            row[f"PriorDelta_{col}"] = float(x.iloc[-1] - x.iloc[0]) if len(x) >= 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def path_summary(entries: pd.DataFrame, seq_col: str, min_count: int) -> pd.DataFrame:
    if seq_col not in entries.columns:
        return pd.DataFrame()
    rows = []
    for path, g in entries.groupby(seq_col, dropna=False, sort=True):
        if len(g) < min_count:
            continue
        row = summarize_returns(g, str(path))
        row["SequenceColumn"] = seq_col
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["MeanSignedReturn", "Count"], ascending=False)


def offset_summary(df: pd.DataFrame, entries: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for offset in range(-args.lookback, 1):
        samples = []
        for _, e in entries.iterrows():
            hit = df[(df["File"].eq(e["File"])) & (df["BarIndex"].eq(e["EntryBarIndex"] + offset))]
            if hit.empty:
                continue
            r = hit.iloc[0].copy()
            r["EntrySignedReturn"] = e["SignedReturn"]
            r["Offset"] = offset
            samples.append(r)
        if not samples:
            continue
        s = pd.DataFrame(samples)
        row = {"Offset": offset, "Count": int(len(s))}
        for col in ["RollingEntropy", "RollingDirectionalPresence", "RollingVelocity", "DominantPressureValue"]:
            if col in s.columns:
                x = pd.to_numeric(s[col], errors="coerce").dropna()
                row[f"Mean_{col}"] = float(x.mean()) if len(x) else np.nan
                row[f"Median_{col}"] = float(x.median()) if len(x) else np.nan
        ret = pd.to_numeric(s["EntrySignedReturn"], errors="coerce").dropna()
        row["EntryMeanSignedReturn"] = float(ret.mean()) if len(ret) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/precursor_sequence_es_compression_entropy_mid_presence_zero")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--pressure", default="CompressionPressure")
    p.add_argument("--entropy-min", type=float, default=0.88)
    p.add_argument("--entropy-max", type=float, default=1.30)
    p.add_argument("--directional-presence", type=float, default=0.0)
    p.add_argument("--lookback", type=int, default=5)
    p.add_argument("--min-prior-bars", type=int, default=3)
    p.add_argument("--min-path-count", type=int, default=5)
    args = p.parse_known_args(argv)[0]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_and_mark(args)
    entries = build_entry_precursors(df, args)
    if entries.empty:
        raise RuntimeError("No entry precursor rows built")

    summary_rows = [summarize_returns(entries, "AllEntries")]
    path_frames = []
    for col in ["PriorDominantPressureSeq", "PriorResolvedArchetypeSeq", "PriorMacroStateSeq", "PrevDominantPressure", "PrevResolvedArchetype", "PrevMacroState"]:
        ps = path_summary(entries, col, args.min_path_count)
        if not ps.empty:
            path_frames.append(ps)
    paths = pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame()
    offsets = offset_summary(df, entries, args)

    if not paths.empty:
        best = paths.iloc[0]
        summary_rows.append({
            "Group": "BestPrecursorPath",
            "Count": best["Count"],
            "MeanSignedReturn": best["MeanSignedReturn"],
            "MedianSignedReturn": best["MedianSignedReturn"],
            "TStat": best["TStat"],
            "PositiveFraction": best["PositiveFraction"],
            "Path": best["Group"],
            "SequenceColumn": best["SequenceColumn"],
        })
    score = pd.DataFrame(summary_rows)

    entries.to_csv(outdir / "precursor_entry_rows.csv", index=False)
    score.to_csv(outdir / "precursor_sequence_summary.csv", index=False)
    paths.to_csv(outdir / "precursor_path_summary.csv", index=False)
    offsets.to_csv(outdir / "precursor_offset_summary.csv", index=False)

    scorecard = []
    allrow = score.iloc[0]
    for k in ["Count", "MeanSignedReturn", "MedianSignedReturn", "TStat", "PositiveFraction", "Top5SharePositiveSum"]:
        if k in allrow.index:
            scorecard.append({"Metric": f"AllEntries_{k}", "Value": allrow[k]})
    if not paths.empty:
        best = paths.iloc[0]
        for k in ["SequenceColumn", "Group", "Count", "MeanSignedReturn", "MedianSignedReturn", "TStat", "PositiveFraction"]:
            scorecard.append({"Metric": f"BestPath_{k}", "Value": best[k]})
    pd.DataFrame(scorecard).to_csv(outdir / "precursor_scorecard.csv", index=False)

    print("APVA precursor sequence analysis complete")
    print(pd.DataFrame(scorecard).to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
