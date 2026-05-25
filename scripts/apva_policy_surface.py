#!/usr/bin/env python3
"""
APVA policy surface.

Purpose
-------
For a convex APVA state, test how fixed stop/target policies reshape payoff.

Primary state:
    ES / 5-bar horizon / CompressionPressure
    RollingDirectionalPresence == 0
    RollingEntropy in [0.88, 1.30]

This uses DirectionalMFE and DirectionalMAE as path summaries. It cannot know
true intrabar sequencing. If both stop and target are touched within the horizon,
the script supports conservative, optimistic, and mfe_priority assumptions.

Recommended Jupyter usage:

    import runpy
    ns = runpy.run_path("scripts/apva_policy_surface.py", run_name="not_main")
    ns["main"]([
        "--input", "tables/apva_forward_signed_return_dataset_v1.csv",
        "--outdir", "outputs/policy_surface_es_compression_entropy_mid_presence_zero",
        "--collision", "conservative"
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


def tail_share(x: pd.Series, frac: float = 0.05) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna().sort_values(ascending=False)
    if vals.empty:
        return float("nan")
    k = max(1, math.ceil(len(vals) * frac))
    pos_sum = vals[vals > 0].sum()
    return float(vals.iloc[:k].sum() / pos_sum) if pos_sum != 0 else float("nan")


def parse_grid(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


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
        raise ValueError(f"Unknown collision mode: {collision}")
    if hit_target:
        return target
    if hit_stop:
        return -abs(stop)
    # Neither level hit: exit at horizon signed return, clipped to avoid exceeding policy bounds.
    return float(np.clip(signed, -abs(stop), target))


def load_state(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.input)
    numeric = ["HorizonBars", "RollingEntropy", "RollingDirectionalPresence", "SignedReturn", "DirectionalMFE", "DirectionalMAE"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["Instrument", "HorizonBars", "DominantPressure", "RollingEntropy", "RollingDirectionalPresence", "SignedReturn", "DirectionalMFE", "DirectionalMAE"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    mask = (
        df["Instrument"].astype(str).str.upper().eq(args.instrument.upper())
        & df["HorizonBars"].eq(args.horizon)
        & df["DominantPressure"].astype(str).eq(args.pressure)
        & df["RollingEntropy"].between(args.entropy_min, args.entropy_max, inclusive="both")
        & df["RollingDirectionalPresence"].eq(args.directional_presence)
    )
    out = df.loc[mask].dropna(subset=["SignedReturn", "DirectionalMFE", "DirectionalMAE"]).copy()
    if out.empty:
        raise RuntimeError("No rows selected")
    return out.reset_index(drop=True)


def summarize_policy(outcomes: pd.Series, stop: float, target: float, collision: str) -> dict:
    x = pd.to_numeric(outcomes, errors="coerce").dropna()
    wins = x[x > 0]
    losses = x[x < 0]
    gross_win = wins.sum()
    gross_loss = -losses.sum()
    return {
        "Stop": stop,
        "Target": target,
        "CollisionMode": collision,
        "Count": int(len(x)),
        "MeanOutcome": float(x.mean()),
        "MedianOutcome": float(x.median()),
        "StdOutcome": float(x.std(ddof=1)) if len(x) > 1 else float("nan"),
        "TStat": tstat(x),
        "WinRate": float((x > 0).mean()),
        "LossRate": float((x < 0).mean()),
        "ScratchRate": float((x == 0).mean()),
        "GrossWin": float(gross_win),
        "GrossLoss": float(gross_loss),
        "ProfitFactor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "MinOutcome": float(x.min()),
        "Q05": float(x.quantile(0.05)),
        "Q25": float(x.quantile(0.25)),
        "Q75": float(x.quantile(0.75)),
        "Q95": float(x.quantile(0.95)),
        "MaxOutcome": float(x.max()),
        "Top5PctShareOfPositiveSum": tail_share(x, 0.05),
        "Top10PctShareOfPositiveSum": tail_share(x, 0.10),
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/policy_surface_es_compression_entropy_mid_presence_zero")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--pressure", default="CompressionPressure")
    p.add_argument("--entropy-min", type=float, default=0.88)
    p.add_argument("--entropy-max", type=float, default=1.30)
    p.add_argument("--directional-presence", type=float, default=0.0)
    p.add_argument("--stops", type=parse_grid, default=parse_grid("2,4,6,8,10,12"))
    p.add_argument("--targets", type=parse_grid, default=parse_grid("2,4,6,8,10,12,16,20,24,32"))
    p.add_argument("--collision", choices=["conservative", "optimistic", "mfe_priority"], default="conservative")
    args = p.parse_known_args(argv)[0]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    state = load_state(args)
    rows = []
    detail_frames = []
    for stop in args.stops:
        for target in args.targets:
            outcomes = state.apply(policy_outcome, axis=1, stop=stop, target=target, collision=args.collision)
            rows.append(summarize_policy(outcomes, stop, target, args.collision))
            if stop in (4.0, 6.0) and target in (8.0, 12.0, 16.0, 20.0):
                tmp = state[[c for c in ["Time", "File", "SignedReturn", "DirectionalMFE", "DirectionalMAE", "RollingEntropy", "RollingDirectionalPresence"] if c in state.columns]].copy()
                tmp["Stop"] = stop
                tmp["Target"] = target
                tmp["PolicyOutcome"] = outcomes.values
                detail_frames.append(tmp)

    surface = pd.DataFrame(rows).sort_values(["MeanOutcome", "ProfitFactor", "TStat"], ascending=False)
    best = surface.iloc[0]
    score = pd.DataFrame([
        {"Metric": "state_count", "Value": float(len(state))},
        {"Metric": "collision_mode", "Value": args.collision},
        {"Metric": "best_stop", "Value": float(best["Stop"])},
        {"Metric": "best_target", "Value": float(best["Target"])},
        {"Metric": "best_mean_outcome", "Value": float(best["MeanOutcome"])},
        {"Metric": "best_median_outcome", "Value": float(best["MedianOutcome"])},
        {"Metric": "best_t_stat", "Value": float(best["TStat"])},
        {"Metric": "best_win_rate", "Value": float(best["WinRate"])},
        {"Metric": "best_profit_factor", "Value": float(best["ProfitFactor"])},
        {"Metric": "best_top5_share_positive_sum", "Value": float(best["Top5PctShareOfPositiveSum"])},
    ])

    state.to_csv(outdir / "selected_policy_rows.csv", index=False)
    surface.to_csv(outdir / "policy_surface.csv", index=False)
    score.to_csv(outdir / "policy_scorecard.csv", index=False)
    if detail_frames:
        pd.concat(detail_frames, ignore_index=True).to_csv(outdir / "policy_detail_examples.csv", index=False)

    print("APVA policy surface complete")
    print(score.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
