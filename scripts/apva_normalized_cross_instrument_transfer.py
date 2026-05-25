#!/usr/bin/env python3
"""
APVA normalized cross-instrument transfer test.

Uses normalized returns/excursions instead of raw point units so ES, NQ, and 6E
can be compared more fairly.
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
        "Max": float(x.max()) if len(x) else np.nan,
        "Sum": float(x.sum()) if len(x) else np.nan,
    }


def prepare_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    for c in [
        "BarIndex", "HorizonBars",
        "SignedReturn", "SignedNormalizedReturn",
        "DirectionalMFE", "DirectionalMAE",
        "DirectionalNormalizedMFE", "DirectionalNormalizedMAE",
        "RollingEntropy", "RollingDirectionalPresence",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    required = [
        "Instrument", "File", "BarIndex", "HorizonBars", "DominantPressure",
        "RollingEntropy", "RollingDirectionalPresence",
        "SignedNormalizedReturn",
        "DirectionalNormalizedMFE", "DirectionalNormalizedMAE",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    return df.sort_values(["Instrument", "File", "BarIndex"]).reset_index(drop=True)


def find_entries(df: pd.DataFrame, args) -> pd.DataFrame:
    state = (
        df["HorizonBars"].eq(args.horizon)
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

    rows = []
    for idx in df.index[df["TargetEntry"]]:
        e = df.loc[idx]
        prior = df[
            df["Instrument"].eq(e["Instrument"])
            & df["File"].eq(e["File"])
            & (df["BarIndex"] < e["BarIndex"])
        ].tail(args.lookback)

        if len(prior) < args.lookback:
            continue

        if prior["DominantPressure"].astype(str).tolist() != TARGET_PATH:
            continue

        rows.append(e.copy())

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No entries found")

    return out.reset_index(drop=True)


def normalized_policy_outcome(row, disaster_stop: float | None) -> float:
    ret = float(row["SignedNormalizedReturn"])
    mae = float(row["DirectionalNormalizedMAE"])

    if disaster_stop is not None and mae <= -abs(disaster_stop):
        return -abs(disaster_stop)

    return ret


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/normalized_cross_instrument_transfer")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--pressure", default="CompressionPressure")
    p.add_argument("--entropy-min", type=float, default=0.88)
    p.add_argument("--entropy-max", type=float, default=1.30)
    p.add_argument("--directional-presence", type=float, default=0.0)
    p.add_argument("--lookback", type=int, default=5)
    p.add_argument("--disaster-stop", type=float, default=2.0)
    args = p.parse_known_args(argv)[0]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = prepare_df(args.input)
    entries = find_entries(df, args)

    entries["NormalizedPolicyOutcome"] = entries.apply(
        normalized_policy_outcome,
        axis=1,
        disaster_stop=args.disaster_stop,
    )

    inst_rows = []
    for inst, g in entries.groupby("Instrument", sort=True):
        row = {"Instrument": inst}
        row.update({f"Policy_{k}": v for k, v in summarize(g["NormalizedPolicyOutcome"]).items()})
        row.update({f"RawSignedNorm_{k}": v for k, v in summarize(g["SignedNormalizedReturn"]).items()})
        row["StopRate"] = float(
            (g["DirectionalNormalizedMAE"] <= -abs(args.disaster_stop)).mean()
        )
        inst_rows.append(row)

    by_inst = pd.DataFrame(inst_rows).sort_values("Policy_Mean", ascending=False)

    score = pd.DataFrame([
        {"Metric": "entry_count", "Value": float(len(entries))},
        {"Metric": "instrument_count", "Value": float(by_inst["Instrument"].nunique())},
        {"Metric": "instrument_positive_fraction", "Value": float((by_inst["Policy_Mean"] > 0).mean())},
        {"Metric": "instrument_median_policy_mean", "Value": float(by_inst["Policy_Mean"].median())},
        {"Metric": "best_instrument", "Value": str(by_inst.iloc[0]["Instrument"])},
        {"Metric": "best_policy_mean", "Value": float(by_inst.iloc[0]["Policy_Mean"])},
        {"Metric": "worst_instrument", "Value": str(by_inst.iloc[-1]["Instrument"])},
        {"Metric": "worst_policy_mean", "Value": float(by_inst.iloc[-1]["Policy_Mean"])},
    ])

    entries.to_csv(outdir / "normalized_transfer_entries.csv", index=False)
    by_inst.to_csv(outdir / "normalized_transfer_by_instrument.csv", index=False)
    score.to_csv(outdir / "normalized_transfer_scorecard.csv", index=False)

    print("APVA normalized cross-instrument transfer complete")
    print(score.to_string(index=False))
    print(by_inst.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()