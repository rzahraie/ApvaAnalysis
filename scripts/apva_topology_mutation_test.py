#!/usr/bin/env python3
"""
APVA topology mutation test - optimized.

This version precomputes prior pressure paths once per row instead of repeatedly
scanning the full dataframe.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


INDEX_INSTRUMENTS = {"ES", "NQ"}

BASE_PATH = [
    "RotationalPressure",
    "RotationalPressure",
    "CompressionPressure",
    "CompressionPressure",
    "CompressionPressure",
]


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


def prepare_df(path: str, max_lookback: int) -> pd.DataFrame:
    df = pd.read_csv(path)

    for c in [
        "BarIndex",
        "HorizonBars",
        "RollingEntropy",
        "RollingDirectionalPresence",
        "SignedNormalizedReturn",
        "DirectionalNormalizedMAE",
        "RollingVelocity",
        "DominantPressureValue",
    ]:
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

    # Rolling apply above is useless for strings, so build paths explicitly per group.
    for lb in range(1, max_lookback + 1):
        df[f"PriorPath_{lb}"] = ""

    for _, idxs in df.groupby(["Instrument", "File"], sort=False).groups.items():
        idxs = list(idxs)
        pressures = df.loc[idxs, "DominantPressure"].astype(str).tolist()
        for pos, idx in enumerate(idxs):
            for lb in range(1, max_lookback + 1):
                if pos >= lb:
                    df.at[idx, f"PriorPath_{lb}"] = " > ".join(pressures[pos - lb:pos])

    return df


def make_mutations() -> list[dict]:
    muts = []

    explicit_paths = [
        ("base_RRCCC", BASE_PATH),
        ("RCCC", ["RotationalPressure", "CompressionPressure", "CompressionPressure", "CompressionPressure"]),
        ("RRCC", ["RotationalPressure", "RotationalPressure", "CompressionPressure", "CompressionPressure"]),
        ("RCC", ["RotationalPressure", "CompressionPressure", "CompressionPressure"]),
        ("RRCCCC", ["RotationalPressure", "RotationalPressure", "CompressionPressure", "CompressionPressure", "CompressionPressure", "CompressionPressure"]),
        ("RRRCCC", ["RotationalPressure", "RotationalPressure", "RotationalPressure", "CompressionPressure", "CompressionPressure", "CompressionPressure"]),
        ("RRC", ["RotationalPressure", "RotationalPressure", "CompressionPressure"]),
        ("CCC", ["CompressionPressure", "CompressionPressure", "CompressionPressure"]),
    ]

    for name, path in explicit_paths:
        muts.append({
            "Mutation": name,
            "Mode": "exact_path",
            "Lookback": len(path),
            "PathString": " > ".join(path),
            "MinCompressionCount": np.nan,
            "MinRotationCount": np.nan,
        })

    for lookback in [3, 4, 5, 6, 7]:
        for min_rot in [1, 2]:
            for min_comp in [2, 3, 4]:
                if min_rot + min_comp > lookback:
                    continue
                muts.append({
                    "Mutation": f"relaxed_L{lookback}_R{min_rot}_C{min_comp}",
                    "Mode": "relaxed_counts",
                    "Lookback": lookback,
                    "PathString": "",
                    "MinCompressionCount": min_comp,
                    "MinRotationCount": min_rot,
                })

    for dwell in [2, 3, 4, 5, 6]:
        muts.append({
            "Mutation": f"compression_dwell_{dwell}",
            "Mode": "compression_dwell",
            "Lookback": dwell,
            "PathString": "",
            "MinCompressionCount": dwell,
            "MinRotationCount": 0,
        })

    return muts


def find_state_entries(df: pd.DataFrame, args) -> pd.DataFrame:
    state = (
        df["Instrument"].astype(str).str.upper().isin(INDEX_INSTRUMENTS)
        & df["HorizonBars"].eq(args.horizon)
        & df["DominantPressure"].astype(str).eq(args.pressure)
        & df["RollingEntropy"].between(args.entropy_min, args.entropy_max, inclusive="both")
        & df["RollingDirectionalPresence"].eq(args.directional_presence)
    )

    df["InTargetState"] = state
    prev = (
        df.groupby(["Instrument", "File"])["InTargetState"]
        .shift(1)
        .astype("boolean")
        .fillna(False)
    )
    df["TargetEntry"] = df["InTargetState"] & ~prev

    entries = df.loc[df["TargetEntry"]].copy().reset_index(drop=True)
    if entries.empty:
        raise RuntimeError("No base state entries found")

    return entries


def mutation_mask(entries: pd.DataFrame, mutation: dict) -> pd.Series:
    lb = int(mutation["Lookback"])
    col = f"PriorPath_{lb}"
    paths = entries[col].astype(str)

    if mutation["Mode"] == "exact_path":
        return paths.eq(mutation["PathString"])

    if mutation["Mode"] == "compression_dwell":
        target = " > ".join(["CompressionPressure"] * lb)
        return paths.eq(target)

    if mutation["Mode"] == "relaxed_counts":
        min_comp = int(mutation["MinCompressionCount"])
        min_rot = int(mutation["MinRotationCount"])

        return paths.map(
            lambda p: (
                bool(p)
                and p.split(" > ")[-1] == "CompressionPressure"
                and p.split(" > ").count("CompressionPressure") >= min_comp
                and p.split(" > ").count("RotationalPressure") >= min_rot
            )
        )

    raise ValueError(f"Unknown mutation mode: {mutation['Mode']}")


def apply_policy(entries: pd.DataFrame, disaster_stop: float) -> pd.DataFrame:
    out = entries.copy()
    stopped = out["DirectionalNormalizedMAE"] <= -abs(disaster_stop)
    out["NormalizedPolicyOutcome"] = out["SignedNormalizedReturn"].where(~stopped, -abs(disaster_stop))
    out["DisasterStopped"] = stopped
    return out


def evaluate_mutations(entries: pd.DataFrame, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    summary_rows = []

    for mut in make_mutations():
        mask = mutation_mask(entries, mut)
        g = entries.loc[mask].copy()

        if g.empty:
            continue

        g = apply_policy(g, args.disaster_stop)
        g["Mutation"] = mut["Mutation"]
        g["MutationMode"] = mut["Mode"]
        g["Lookback"] = mut["Lookback"]
        g["PriorDominantPressureSeq"] = g[f"PriorPath_{int(mut['Lookback'])}"]
        all_rows.append(g)

        if len(g) < args.min_count:
            continue

        s = summarize(g["NormalizedPolicyOutcome"])
        inst_means = []
        for _, ig in g.groupby("Instrument"):
            if len(ig) >= 5:
                inst_means.append(float(ig["NormalizedPolicyOutcome"].mean()))

        row = {
            "Mutation": mut["Mutation"],
            "MutationMode": mut["Mode"],
            "Lookback": int(mut["Lookback"]),
            "Count": int(len(g)),
            "ES_Count": int((g["Instrument"].astype(str).str.upper() == "ES").sum()),
            "NQ_Count": int((g["Instrument"].astype(str).str.upper() == "NQ").sum()),
            "StopRate": float(g["DisasterStopped"].mean()),
            "PriorSeqExample": g["PriorDominantPressureSeq"].iloc[0],
            "InstrumentPositiveFraction": float(np.mean(np.array(inst_means) > 0)) if inst_means else np.nan,
            "InstrumentMedianMean": float(np.median(inst_means)) if inst_means else np.nan,
        }
        row.update(s)
        summary_rows.append(row)

    if not all_rows:
        raise RuntimeError("No mutation entries produced")

    all_df = pd.concat(all_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    if not summary.empty:
        summary = summary.sort_values(
            ["InstrumentPositiveFraction", "Median", "Mean", "ProfitFactor", "TStat", "Count"],
            ascending=False,
        )

    return all_df, summary


def build_scorecard(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame([{"Metric": "error", "Value": "empty_summary"}])

    best = summary.iloc[0]
    rows = []

    for k in [
        "Mutation", "MutationMode", "Lookback", "Count", "Mean", "Median",
        "TStat", "WinRate", "ProfitFactor", "InstrumentPositiveFraction",
        "InstrumentMedianMean", "ES_Count", "NQ_Count", "StopRate", "PriorSeqExample",
    ]:
        rows.append({"Metric": f"best_{k}", "Value": best[k]})

    base = summary[summary["Mutation"].eq("base_RRCCC")]
    if not base.empty:
        b = base.iloc[0]
        for k in ["Count", "Mean", "Median", "TStat", "WinRate", "ProfitFactor", "InstrumentPositiveFraction"]:
            rows.append({"Metric": f"base_{k}", "Value": b[k]})

    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/topology_mutation_test")
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

    print("loading dataset...", flush=True)
    df = prepare_df(args.input, max_lookback=7)

    print("finding state entries...", flush=True)
    entries = find_state_entries(df, args)
    print("state entries:", len(entries), flush=True)

    print("evaluating mutations...", flush=True)
    all_rows, summary = evaluate_mutations(entries, args)

    scorecard = build_scorecard(summary)

    all_rows.to_csv(outdir / "topology_mutation_entries.csv", index=False)
    summary.to_csv(outdir / "topology_mutation_summary.csv", index=False)
    scorecard.to_csv(outdir / "topology_mutation_scorecard.csv", index=False)

    print("APVA topology mutation test complete")
    print(scorecard.to_string(index=False))
    print()
    if not summary.empty:
        print(summary.head(25).to_string(index=False))
    return 0


if __name__ == "__main__":
    main()