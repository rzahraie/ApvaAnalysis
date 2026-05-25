#!/usr/bin/env python3
"""
APVA index-futures robustness.

Purpose:
Validate the current APVA topology only on equity index futures.

Universe:
    ES, NQ

Topology:
    Prior 5 DominantPressure path:
        RotationalPressure > RotationalPressure > CompressionPressure > CompressionPressure > CompressionPressure

Entry state:
    CompressionPressure
    RollingDirectionalPresence == 0
    RollingEntropy in [0.88, 1.30]

Policy:
    normalized return policy from prior normalized transfer test
    disaster stop on DirectionalNormalizedMAE, default 2.0

Outputs:
    index_futures_entries.csv
    index_futures_by_instrument.csv
    index_futures_by_week.csv
    index_futures_by_instrument_week.csv
    index_futures_scorecard.csv
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


def top_share(s: pd.Series, frac: float) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna().sort_values(ascending=False)
    if x.empty:
        return float("nan")
    k = max(1, math.ceil(len(x) * frac))
    pos = x[x > 0].sum()
    return float(x.iloc[:k].sum() / pos) if pos != 0 else float("nan")


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
        "Std": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "TStat": tstat(x),
        "WinRate": float((x > 0).mean()) if len(x) else np.nan,
        "LossRate": float((x < 0).mean()) if len(x) else np.nan,
        "ProfitFactor": float(gw / gl) if gl > 0 else float("inf"),
        "GrossWin": float(gw),
        "GrossLoss": float(gl),
        "Min": float(x.min()) if len(x) else np.nan,
        "Q25": float(x.quantile(0.25)) if len(x) else np.nan,
        "Q75": float(x.quantile(0.75)) if len(x) else np.nan,
        "Q90": float(x.quantile(0.90)) if len(x) else np.nan,
        "Max": float(x.max()) if len(x) else np.nan,
        "Top5SharePositiveSum": top_share(x, 0.05),
        "Top10SharePositiveSum": top_share(x, 0.10),
        "Sum": float(x.sum()) if len(x) else np.nan,
    }


def week_key(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    iso = dt.dt.isocalendar()
    out = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    out.loc[dt.isna()] = "UnknownWeek"
    return out


def prepare_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric = [
        "BarIndex",
        "HorizonBars",
        "SignedReturn",
        "SignedNormalizedReturn",
        "DirectionalMFE",
        "DirectionalMAE",
        "DirectionalNormalizedMFE",
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

    return df.sort_values(["Instrument", "File", "BarIndex"]).reset_index(drop=True)


def find_entries(df: pd.DataFrame, args) -> pd.DataFrame:
    allowed = {"ES", "NQ"}

    state = (
        df["Instrument"].astype(str).str.upper().isin(allowed)
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

        r = e.copy()
        r["PriorDominantPressureSeq"] = " > ".join(TARGET_PATH)
        rows.append(r)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No index-futures entries found")

    return out.reset_index(drop=True)


def apply_policy(entries: pd.DataFrame, disaster_stop: float | None) -> pd.DataFrame:
    out = entries.copy()

    if disaster_stop is not None:
        stopped = out["DirectionalNormalizedMAE"] <= -abs(disaster_stop)
        out["NormalizedPolicyOutcome"] = out["SignedNormalizedReturn"].where(
            ~stopped,
            -abs(disaster_stop),
        )
        out["DisasterStopped"] = stopped
    else:
        out["NormalizedPolicyOutcome"] = out["SignedNormalizedReturn"]
        out["DisasterStopped"] = False

    if "Time" in out.columns:
        out["RegimeWeek"] = week_key(out["Time"])
    else:
        out["RegimeWeek"] = "UnknownWeek"

    return out


def group_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []

    for key, g in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)

        row = {col: val for col, val in zip(group_cols, key)}
        row.update({f"Policy_{k}": v for k, v in summarize(g["NormalizedPolicyOutcome"]).items()})
        row.update({f"RawNorm_{k}": v for k, v in summarize(g["SignedNormalizedReturn"]).items()})

        row["StopRate"] = float(g["DisasterStopped"].mean())

        if "DirectionalNormalizedMFE" in g.columns:
            row["MeanNormMFE"] = float(pd.to_numeric(g["DirectionalNormalizedMFE"], errors="coerce").mean())
            row["MedianNormMFE"] = float(pd.to_numeric(g["DirectionalNormalizedMFE"], errors="coerce").median())

        if "DirectionalNormalizedMAE" in g.columns:
            row["MeanNormMAE"] = float(pd.to_numeric(g["DirectionalNormalizedMAE"], errors="coerce").mean())
            row["MedianNormMAE"] = float(pd.to_numeric(g["DirectionalNormalizedMAE"], errors="coerce").median())

        rows.append(row)

    return pd.DataFrame(rows)


def build_scorecard(
    rows: pd.DataFrame,
    by_inst: pd.DataFrame,
    by_week: pd.DataFrame,
    by_inst_week: pd.DataFrame,
) -> pd.DataFrame:
    overall = summarize(rows["NormalizedPolicyOutcome"])

    score = [
        {"Metric": "entry_count", "Value": float(len(rows))},
        {"Metric": "instrument_count", "Value": float(by_inst["Instrument"].nunique())},
        {"Metric": "overall_mean", "Value": overall["Mean"]},
        {"Metric": "overall_median", "Value": overall["Median"]},
        {"Metric": "overall_tstat", "Value": overall["TStat"]},
        {"Metric": "overall_win_rate", "Value": overall["WinRate"]},
        {"Metric": "overall_profit_factor", "Value": overall["ProfitFactor"]},
        {"Metric": "overall_stop_rate", "Value": float(rows["DisasterStopped"].mean())},
        {"Metric": "overall_top5_share_positive_sum", "Value": overall["Top5SharePositiveSum"]},
        {
            "Metric": "instrument_positive_fraction",
            "Value": float((by_inst["Policy_Mean"] > 0).mean()) if not by_inst.empty else np.nan,
        },
        {
            "Metric": "instrument_median_mean",
            "Value": float(by_inst["Policy_Mean"].median()) if not by_inst.empty else np.nan,
        },
        {
            "Metric": "week_positive_fraction",
            "Value": float((by_week["Policy_Mean"] > 0).mean()) if not by_week.empty else np.nan,
        },
        {
            "Metric": "week_median_mean",
            "Value": float(by_week["Policy_Mean"].median()) if not by_week.empty else np.nan,
        },
        {
            "Metric": "instrument_week_positive_fraction",
            "Value": float((by_inst_week["Policy_Mean"] > 0).mean()) if not by_inst_week.empty else np.nan,
        },
        {
            "Metric": "instrument_week_median_mean",
            "Value": float(by_inst_week["Policy_Mean"].median()) if not by_inst_week.empty else np.nan,
        },
    ]

    for _, r in by_inst.sort_values("Policy_Mean", ascending=False).iterrows():
        inst = str(r["Instrument"])
        score.extend([
            {"Metric": f"{inst}_count", "Value": float(r["Policy_Count"])},
            {"Metric": f"{inst}_mean", "Value": float(r["Policy_Mean"])},
            {"Metric": f"{inst}_median", "Value": float(r["Policy_Median"])},
            {"Metric": f"{inst}_tstat", "Value": float(r["Policy_TStat"])},
            {"Metric": f"{inst}_win_rate", "Value": float(r["Policy_WinRate"])},
            {"Metric": f"{inst}_profit_factor", "Value": float(r["Policy_ProfitFactor"])},
            {"Metric": f"{inst}_stop_rate", "Value": float(r["StopRate"])},
        ])

    return pd.DataFrame(score)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/index_futures_robustness")
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
    rows = apply_policy(entries, args.disaster_stop)

    by_inst = group_summary(rows, ["Instrument"])
    by_week = group_summary(rows, ["RegimeWeek"])
    by_inst_week = group_summary(rows, ["Instrument", "RegimeWeek"])

    score = build_scorecard(rows, by_inst, by_week, by_inst_week)

    rows.to_csv(outdir / "index_futures_entries.csv", index=False)
    by_inst.to_csv(outdir / "index_futures_by_instrument.csv", index=False)
    by_week.to_csv(outdir / "index_futures_by_week.csv", index=False)
    by_inst_week.to_csv(outdir / "index_futures_by_instrument_week.csv", index=False)
    score.to_csv(outdir / "index_futures_scorecard.csv", index=False)

    print("APVA index-futures robustness complete")
    print(score.to_string(index=False))
    print()
    print(by_inst.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()