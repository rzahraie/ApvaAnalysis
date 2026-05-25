#!/usr/bin/env python3
"""
APVA compression persistence analysis.

Purpose:
Replace brittle symbolic pressure grammars with continuous topology statistics.

Universe:
    ES, NQ

Entry state:
    CompressionPressure
    RollingDirectionalPresence == 0
    RollingEntropy in [0.88, 1.30]

Policy:
    NormalizedPolicyOutcome = SignedNormalizedReturn,
    capped by disaster stop using DirectionalNormalizedMAE.

This script computes prior-window topology features:
    - compression streak length
    - compression count/fraction
    - rotation count/fraction
    - compression-after-rotation flag
    - entropy slope
    - directional-presence slope
    - velocity slope
    - pressure-value slope

Then it tests which features condition the APVA edge.
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

    return df.sort_values(["Instrument", "File", "BarIndex"]).reset_index(drop=True)


def apply_policy(entries: pd.DataFrame, disaster_stop: float) -> pd.DataFrame:
    out = entries.copy()
    stopped = out["DirectionalNormalizedMAE"] <= -abs(disaster_stop)
    out["NormalizedPolicyOutcome"] = out["SignedNormalizedReturn"].where(
        ~stopped,
        -abs(disaster_stop),
    )
    out["DisasterStopped"] = stopped
    return out


def slope(vals: pd.Series) -> float:
    x = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(float)
    if len(x) < 2:
        return np.nan
    return float(x[-1] - x[0])


def compression_streak(prior_pressures: list[str]) -> int:
    n = 0
    for p in reversed(prior_pressures):
        if p == "CompressionPressure":
            n += 1
        else:
            break
    return n


def build_entries(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
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

        pressures = prior["DominantPressure"].astype(str).tolist()

        comp_count = pressures.count("CompressionPressure")
        rot_count = pressures.count("RotationalPressure")

        row = e.copy()
        row["PriorPressureSeq"] = " > ".join(pressures)
        row["PriorCompressionCount"] = comp_count
        row["PriorRotationCount"] = rot_count
        row["PriorCompressionFraction"] = comp_count / args.lookback
        row["PriorRotationFraction"] = rot_count / args.lookback
        row["PriorCompressionStreak"] = compression_streak(pressures)
        row["CompressionAfterRotation"] = bool(
            len(pressures) >= 2
            and pressures[-1] == "CompressionPressure"
            and "RotationalPressure" in pressures[:-1]
        )
        row["LastPriorPressure"] = pressures[-1]
        row["FirstPriorPressure"] = pressures[0]

        for col in [
            "RollingEntropy",
            "RollingDirectionalPresence",
            "RollingVelocity",
            "DominantPressureValue",
        ]:
            if col in prior.columns:
                row[f"PriorMean_{col}"] = float(pd.to_numeric(prior[col], errors="coerce").mean())
                row[f"PriorLast_{col}"] = float(pd.to_numeric(prior[col], errors="coerce").iloc[-1])
                row[f"PriorSlope_{col}"] = slope(prior[col])

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No entries found")

    return out.reset_index(drop=True)


def numeric_gates(entries: pd.DataFrame, feature: str, min_count: int) -> pd.DataFrame:
    vals = pd.to_numeric(entries[feature], errors="coerce")
    valid = entries.loc[vals.notna()].copy()

    if len(valid) < min_count * 2:
        return pd.DataFrame()

    rows = []

    valid["_rank"] = vals.loc[valid.index].rank(method="first")
    valid["_q"] = pd.qcut(valid["_rank"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

    for q, g in valid.groupby("_q", observed=False):
        if len(g) < min_count:
            continue

        row = {
            "Feature": feature,
            "GateType": "quantile",
            "Gate": str(q),
            "Direction": "inside",
            "FeatureMin": float(pd.to_numeric(g[feature], errors="coerce").min()),
            "FeatureMax": float(pd.to_numeric(g[feature], errors="coerce").max()),
            "FeatureMean": float(pd.to_numeric(g[feature], errors="coerce").mean()),
        }
        row.update(summarize(g["NormalizedPolicyOutcome"]))
        rows.append(row)

    qs = sorted(set(float(vals.quantile(q)) for q in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))

    for th in qs:
        for direction, mask in [
            ("le", vals <= th),
            ("ge", vals >= th),
        ]:
            g = entries.loc[mask].copy()
            if len(g) < min_count:
                continue

            row = {
                "Feature": feature,
                "GateType": "threshold",
                "Gate": th,
                "Direction": direction,
                "FeatureMin": float(pd.to_numeric(g[feature], errors="coerce").min()),
                "FeatureMax": float(pd.to_numeric(g[feature], errors="coerce").max()),
                "FeatureMean": float(pd.to_numeric(g[feature], errors="coerce").mean()),
            }
            row.update(summarize(g["NormalizedPolicyOutcome"]))
            rows.append(row)

    return pd.DataFrame(rows)


def categorical_gates(entries: pd.DataFrame, feature: str, min_count: int) -> pd.DataFrame:
    rows = []

    for value, g in entries.groupby(feature, dropna=False, sort=True):
        if len(g) < min_count:
            continue

        row = {
            "Feature": feature,
            "GateType": "category",
            "Gate": str(value),
            "Direction": "equals",
            "FeatureMin": "",
            "FeatureMax": "",
            "FeatureMean": "",
        }
        row.update(summarize(g["NormalizedPolicyOutcome"]))
        rows.append(row)

    return pd.DataFrame(rows)


def build_feature_tests(entries: pd.DataFrame, min_count: int) -> pd.DataFrame:
    numeric_features = [
        "PriorCompressionCount",
        "PriorRotationCount",
        "PriorCompressionFraction",
        "PriorRotationFraction",
        "PriorCompressionStreak",
        "PriorMean_RollingEntropy",
        "PriorLast_RollingEntropy",
        "PriorSlope_RollingEntropy",
        "PriorMean_RollingDirectionalPresence",
        "PriorLast_RollingDirectionalPresence",
        "PriorSlope_RollingDirectionalPresence",
        "PriorMean_RollingVelocity",
        "PriorLast_RollingVelocity",
        "PriorSlope_RollingVelocity",
        "PriorMean_DominantPressureValue",
        "PriorLast_DominantPressureValue",
        "PriorSlope_DominantPressureValue",
    ]

    categorical_features = [
        "Instrument",
        "PriorPressureSeq",
        "LastPriorPressure",
        "FirstPriorPressure",
        "CompressionAfterRotation",
    ]

    frames = []

    for f in numeric_features:
        if f in entries.columns:
            frames.append(numeric_gates(entries, f, min_count))

    for f in categorical_features:
        if f in entries.columns:
            frames.append(categorical_gates(entries, f, min_count))

    frames = [f for f in frames if f is not None and not f.empty]

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(
        ["Mean", "ProfitFactor", "TStat", "Count"],
        ascending=False,
    )


def group_summary(entries: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []

    for key, g in entries.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)

        row = {col: val for col, val in zip(group_cols, key)}
        row.update(summarize(g["NormalizedPolicyOutcome"]))
        row["StopRate"] = float(g["DisasterStopped"].mean())
        rows.append(row)

    return pd.DataFrame(rows)


def build_scorecard(entries: pd.DataFrame, feature_tests: pd.DataFrame) -> pd.DataFrame:
    base = summarize(entries["NormalizedPolicyOutcome"])

    rows = [
        {"Metric": "base_count", "Value": float(len(entries))},
        {"Metric": "base_mean", "Value": base["Mean"]},
        {"Metric": "base_median", "Value": base["Median"]},
        {"Metric": "base_tstat", "Value": base["TStat"]},
        {"Metric": "base_profit_factor", "Value": base["ProfitFactor"]},
        {"Metric": "base_win_rate", "Value": base["WinRate"]},
        {"Metric": "base_stop_rate", "Value": float(entries["DisasterStopped"].mean())},
    ]

    if not feature_tests.empty:
        best = feature_tests.iloc[0]
        rows.extend([
            {"Metric": "best_feature", "Value": best["Feature"]},
            {"Metric": "best_gate_type", "Value": best["GateType"]},
            {"Metric": "best_gate", "Value": best["Gate"]},
            {"Metric": "best_direction", "Value": best["Direction"]},
            {"Metric": "best_count", "Value": float(best["Count"])},
            {"Metric": "best_mean", "Value": float(best["Mean"])},
            {"Metric": "best_median", "Value": float(best["Median"])},
            {"Metric": "best_tstat", "Value": float(best["TStat"])},
            {"Metric": "best_profit_factor", "Value": float(best["ProfitFactor"])},
            {"Metric": "best_win_rate", "Value": float(best["WinRate"])},
        ])

    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/compression_persistence_analysis")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--pressure", default="CompressionPressure")
    p.add_argument("--entropy-min", type=float, default=0.88)
    p.add_argument("--entropy-max", type=float, default=1.30)
    p.add_argument("--directional-presence", type=float, default=0.0)
    p.add_argument("--lookback", type=int, default=5)
    p.add_argument("--disaster-stop", type=float, default=2.0)
    p.add_argument("--min-count", type=int, default=30)
    args = p.parse_known_args(argv)[0]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = prepare_df(args.input)
    entries = build_entries(df, args)
    entries = apply_policy(entries, args.disaster_stop)

    feature_tests = build_feature_tests(entries, args.min_count)
    by_instrument = group_summary(entries, ["Instrument"])
    by_sequence = group_summary(entries, ["PriorPressureSeq"])

    score = build_scorecard(entries, feature_tests)

    entries.to_csv(outdir / "compression_persistence_entries.csv", index=False)
    feature_tests.to_csv(outdir / "compression_persistence_feature_tests.csv", index=False)
    by_instrument.to_csv(outdir / "compression_persistence_by_instrument.csv", index=False)
    by_sequence.to_csv(outdir / "compression_persistence_by_sequence.csv", index=False)
    score.to_csv(outdir / "compression_persistence_scorecard.csv", index=False)

    print("APVA compression persistence analysis complete")
    print(score.to_string(index=False))
    print()
    if not feature_tests.empty:
        print(feature_tests.head(30).to_string(index=False))

    return 0


if __name__ == "__main__":
    main()