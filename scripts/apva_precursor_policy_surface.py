#!/usr/bin/env python3
"""
APVA precursor-filtered policy surface.

Purpose
-------
Test whether the best discovered precursor path improves policy behavior.

Target state:
    ES / 5-bar horizon / CompressionPressure
    RollingDirectionalPresence == 0
    RollingEntropy in [0.88, 1.30]

Required 5-bar prior DominantPressure path:
    RotationalPressure > RotationalPressure > CompressionPressure > CompressionPressure > CompressionPressure

This tests whether transition topology improves the convex payoff state.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

TARGET_PATH = [
    "RotationalPressure",
    "RotationalPressure",
    "CompressionPressure",
    "CompressionPressure",
    "CompressionPressure",
]


def parse_grid(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def tstat(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    if len(x) < 2:
        return float("nan")
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / (sd / math.sqrt(len(x)))) if sd > 0 else float("nan")


def tail_share(s: pd.Series, frac: float) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna().sort_values(ascending=False)
    if x.empty:
        return float("nan")
    k = max(1, math.ceil(len(x) * frac))
    pos = x[x > 0].sum()
    return float(x.iloc[:k].sum() / pos) if pos != 0 else float("nan")


def policy_outcome(row: pd.Series, stop: float, target: float, collision: str) -> float:
    mfe = float(row["DirectionalMFE"])
    mae = float(row["DirectionalMAE"])
    signed = float(row["SignedReturn"])
    hit_target = mfe >= target
    hit_stop = mae <= -abs(stop)
    if hit_target and hit_stop:
        if collision == "conservative":
            return -abs(stop)
        if collision == "optimistic":
            return target
        if collision == "mfe_priority":
            return target if abs(mfe) >= abs(mae) else -abs(stop)
    if hit_target:
        return target
    if hit_stop:
        return -abs(stop)
    return float(np.clip(signed, -abs(stop), target))


def load_precursor_entries(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.input)
    for c in ["BarIndex", "HorizonBars", "RollingEntropy", "RollingDirectionalPresence", "SignedReturn", "DirectionalMFE", "DirectionalMAE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    required = ["Instrument", "File", "BarIndex", "HorizonBars", "DominantPressure", "RollingEntropy", "RollingDirectionalPresence", "SignedReturn", "DirectionalMFE", "DirectionalMAE"]
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
    prev = df.groupby("File")["InTargetState"].shift(1).fillna(False)
    df["TargetEntry"] = df["InTargetState"] & ~prev

    rows = []
    for idx in df.index[df["TargetEntry"]]:
        e = df.loc[idx]
        prior = df[(df["File"].eq(e["File"])) & (df["BarIndex"] < e["BarIndex"])].tail(args.lookback)
        if len(prior) < args.lookback:
            continue
        path = prior["DominantPressure"].astype(str).tolist()
        if path == TARGET_PATH:
            r = e.copy()
            r["PriorDominantPressureSeq"] = " > ".join(path)
            rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No precursor-filtered entries found")
    return out.reset_index(drop=True)


def summarize_outcomes(x: pd.Series, stop: float, target: float, collision: str) -> dict:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    wins = vals[vals > 0]
    losses = vals[vals < 0]
    gw = wins.sum()
    gl = -losses.sum()
    return {
        "Stop": stop,
        "Target": target,
        "CollisionMode": collision,
        "Count": int(len(vals)),
        "MeanOutcome": float(vals.mean()),
        "MedianOutcome": float(vals.median()),
        "TStat": tstat(vals),
        "WinRate": float((vals > 0).mean()),
        "LossRate": float((vals < 0).mean()),
        "ProfitFactor": float(gw / gl) if gl > 0 else float("inf"),
        "MinOutcome": float(vals.min()),
        "MaxOutcome": float(vals.max()),
        "Top5PctShareOfPositiveSum": tail_share(vals, 0.05),
        "Top10PctShareOfPositiveSum": tail_share(vals, 0.10),
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/precursor_policy_surface_es_compression_entropy_mid_presence_zero")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--pressure", default="CompressionPressure")
    p.add_argument("--entropy-min", type=float, default=0.88)
    p.add_argument("--entropy-max", type=float, default=1.30)
    p.add_argument("--directional-presence", type=float, default=0.0)
    p.add_argument("--lookback", type=int, default=5)
    p.add_argument("--stops", type=parse_grid, default=parse_grid("2,4,6,8,10,12"))
    p.add_argument("--targets", type=parse_grid, default=parse_grid("2,4,6,8,10,12,16,20,24,32"))
    p.add_argument("--collision", choices=["conservative", "optimistic", "mfe_priority"], default="conservative")
    args = p.parse_known_args(argv)[0]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    entries = load_precursor_entries(args)

    rows = []
    for stop in args.stops:
        for target in args.targets:
            outcomes = entries.apply(policy_outcome, axis=1, stop=stop, target=target, collision=args.collision)
            rows.append(summarize_outcomes(outcomes, stop, target, args.collision))
    surface = pd.DataFrame(rows).sort_values(["MeanOutcome", "ProfitFactor", "TStat"], ascending=False)
    best = surface.iloc[0]

    raw = entries["SignedReturn"]
    raw_score = {
        "raw_count": len(entries),
        "raw_mean": float(raw.mean()),
        "raw_median": float(raw.median()),
        "raw_tstat": tstat(raw),
        "raw_win_rate": float((raw > 0).mean()),
        "raw_top5_share_positive_sum": tail_share(raw, 0.05),
    }
    score = pd.DataFrame([
        {"Metric": "entry_count", "Value": float(len(entries))},
        {"Metric": "raw_mean", "Value": raw_score["raw_mean"]},
        {"Metric": "raw_median", "Value": raw_score["raw_median"]},
        {"Metric": "raw_tstat", "Value": raw_score["raw_tstat"]},
        {"Metric": "raw_win_rate", "Value": raw_score["raw_win_rate"]},
        {"Metric": "raw_top5_share_positive_sum", "Value": raw_score["raw_top5_share_positive_sum"]},
        {"Metric": "best_stop", "Value": float(best["Stop"])},
        {"Metric": "best_target", "Value": float(best["Target"])},
        {"Metric": "best_mean_outcome", "Value": float(best["MeanOutcome"])},
        {"Metric": "best_median_outcome", "Value": float(best["MedianOutcome"])},
        {"Metric": "best_tstat", "Value": float(best["TStat"])},
        {"Metric": "best_win_rate", "Value": float(best["WinRate"])},
        {"Metric": "best_profit_factor", "Value": float(best["ProfitFactor"])},
        {"Metric": "best_top5_share_positive_sum", "Value": float(best["Top5PctShareOfPositiveSum"])},
    ])

    entries.to_csv(outdir / "precursor_policy_entries.csv", index=False)
    surface.to_csv(outdir / "precursor_policy_surface.csv", index=False)
    score.to_csv(outdir / "precursor_policy_scorecard.csv", index=False)
    print("APVA precursor policy surface complete")
    print(score.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
