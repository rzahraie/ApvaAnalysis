#!/usr/bin/env python3
"""
APVA time-exit robustness.

Purpose
-------
Validate the current best APVA policy shape:

    refined precursor entry
    fixed 5-bar time exit
    wide disaster stop
    no profit cap

Prior result:
    Exit horizon 5, disaster stop 20, no profit cap produced better payoff than
    fixed stop/target truncation.

This script asks whether that policy is robust across file/week/session regimes
or dominated by a small number of clusters.

Outputs
-------
- robustness_entries.csv
- robustness_policy_rows.csv
- robustness_by_file.csv
- robustness_by_week.csv
- robustness_by_session.csv, if session-like column exists
- robustness_scorecard.csv

Recommended Jupyter usage
-------------------------

    import runpy
    ns = runpy.run_path("scripts/apva_time_exit_robustness.py", run_name="not_main")
    ns["main"]([
        "--input", "tables/apva_forward_signed_return_dataset_v1.csv",
        "--outdir", "outputs/time_exit_robustness_es_compression_precursor",
        "--exit-horizon", "5",
        "--disaster-stop", "20"
    ])
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


def tail_share(s: pd.Series, frac: float) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna().sort_values(ascending=False)
    if x.empty:
        return float("nan")
    k = max(1, math.ceil(len(x) * frac))
    pos = x[x > 0].sum()
    return float(x.iloc[:k].sum() / pos) if pos != 0 else float("nan")


def week_key(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    iso = dt.dt.isocalendar()
    out = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    out.loc[dt.isna()] = "UnknownWeek"
    return out


def signed_direction(row: pd.Series) -> float:
    if "DirectionSign" in row and pd.notna(row["DirectionSign"]):
        v = float(row["DirectionSign"])
        if v != 0:
            return 1.0 if v > 0 else -1.0
    if "ActiveDirection" in row:
        s = str(row["ActiveDirection"]).lower()
        if "up" in s or "long" in s or s == "1":
            return 1.0
        if "down" in s or "short" in s or s == "-1":
            return -1.0
    sr = float(row.get("SignedReturn", 0.0))
    return 1.0 if sr >= 0 else -1.0


def summarize(vals: pd.Series) -> dict:
    x = pd.to_numeric(vals, errors="coerce").dropna()
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
        "Top5SharePositiveSum": tail_share(x, 0.05),
        "Top10SharePositiveSum": tail_share(x, 0.10),
        "Sum": float(x.sum()) if len(x) else np.nan,
    }


def prepare_df(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.input)
    numeric = [
        "BarIndex", "HorizonBars", "Open", "High", "Low", "Close",
        "SignedReturn", "RawReturn", "NormalizedReturn", "SignedNormalizedReturn",
        "DirectionalMFE", "DirectionalMAE", "RollingEntropy", "RollingDirectionalPresence",
        "RollingVelocity", "DominantPressureValue", "DirectionSign",
    ]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    required = [
        "Instrument", "File", "BarIndex", "HorizonBars", "DominantPressure", "Close",
        "RollingEntropy", "RollingDirectionalPresence", "SignedReturn"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    return df.sort_values(["File", "BarIndex"]).reset_index(drop=True)


def find_entries(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    state = (
        df["Instrument"].astype(str).str.upper().eq(args.instrument.upper())
        & df["HorizonBars"].eq(args.horizon)
        & df["DominantPressure"].astype(str).eq(args.pressure)
        & df["RollingEntropy"].between(args.entropy_min, args.entropy_max, inclusive="both")
        & df["RollingDirectionalPresence"].eq(args.directional_presence)
    )
    df["InTargetState"] = state
    prev = (
        df.groupby("File")["InTargetState"]
        .shift(1)
        .astype("boolean")
        .fillna(False)
    )
    df["TargetEntry"] = df["InTargetState"] & ~prev

    rows = []
    for idx in df.index[df["TargetEntry"]]:
        e = df.loc[idx]
        prior = df[(df["File"].eq(e["File"])) & (df["BarIndex"] < e["BarIndex"])].tail(args.lookback)
        if len(prior) < args.lookback:
            continue
        if prior["DominantPressure"].astype(str).tolist() != TARGET_PATH:
            continue
        r = e.copy()
        r["EntryDirectionSign"] = signed_direction(e)
        r["PriorDominantPressureSeq"] = " > ".join(TARGET_PATH)
        rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No refined entries found")
    return out.reset_index(drop=True)


def apply_policy(df: pd.DataFrame, entries: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    lookup = {(r.File, int(r.BarIndex)): r for r in df.itertuples(index=False)}
    rows = []
    for eid, e in entries.iterrows():
        file = e["File"]
        entry_bar = int(e["BarIndex"])
        entry_close = float(e["Close"])
        direction = float(e["EntryDirectionSign"])
        path = []
        for offset in range(0, args.exit_horizon + 1):
            hit = lookup.get((file, entry_bar + offset))
            if hit is None:
                continue
            close = float(getattr(hit, "Close"))
            high = float(getattr(hit, "High", np.nan)) if hasattr(hit, "High") else np.nan
            low = float(getattr(hit, "Low", np.nan)) if hasattr(hit, "Low") else np.nan
            signed_close = (close - entry_close) * direction
            if direction > 0:
                fav = high - entry_close if np.isfinite(high) else signed_close
                adv = low - entry_close if np.isfinite(low) else signed_close
            else:
                fav = entry_close - low if np.isfinite(low) else signed_close
                adv = entry_close - high if np.isfinite(high) else signed_close
            path.append({"Offset": offset, "SignedCloseMove": signed_close, "FavorableMove": fav, "AdverseMove": adv})
        if not path:
            continue
        p = pd.DataFrame(path)
        final = float(p.loc[p["Offset"].eq(p["Offset"].max()), "SignedCloseMove"].iloc[0])
        max_fav = float(p["FavorableMove"].max())
        max_adv = float(p["AdverseMove"].min())
        stopped = max_adv <= -abs(args.disaster_stop) if args.disaster_stop is not None else False
        outcome = -abs(args.disaster_stop) if stopped else final
        rows.append({
            "EntryId": eid,
            "File": file,
            "EntryBarIndex": entry_bar,
            "EntryTime": e.get("Time", ""),
            "SessionContext": e.get("SessionContext", e.get("Session", "")),
            "EntryDirectionSign": direction,
            "ExitHorizon": args.exit_horizon,
            "ObservedHorizon": int(p["Offset"].max()),
            "PolicyOutcome": outcome,
            "TimeExitOutcome": final,
            "MaxFavorableMove": max_fav,
            "MaxAdverseMove": max_adv,
            "DisasterStopped": bool(stopped),
            "RollingEntropy": float(e["RollingEntropy"]),
            "RollingDirectionalPresence": float(e["RollingDirectionalPresence"]),
            "PriorDominantPressureSeq": e["PriorDominantPressureSeq"],
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No policy rows built")
    if "EntryTime" in out.columns:
        out["RegimeWeek"] = week_key(out["EntryTime"])
    return out


def group_summary(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = []
    for key, g in rows.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        d = {col: val for col, val in zip(group_cols, key)}
        d.update(summarize(g["PolicyOutcome"]))
        d["StopRate"] = float(g["DisasterStopped"].mean())
        d["MeanMaxFavorableMove"] = float(g["MaxFavorableMove"].mean())
        d["MedianMaxFavorableMove"] = float(g["MaxFavorableMove"].median())
        d["MeanMaxAdverseMove"] = float(g["MaxAdverseMove"].mean())
        d["MedianMaxAdverseMove"] = float(g["MaxAdverseMove"].median())
        out.append(d)
    return pd.DataFrame(out)


def build_scorecard(rows: pd.DataFrame, by_file: pd.DataFrame, by_week: pd.DataFrame, by_session: pd.DataFrame) -> pd.DataFrame:
    overall = summarize(rows["PolicyOutcome"])
    out = [
        {"Metric": "entry_count", "Value": float(len(rows))},
        {"Metric": "overall_mean", "Value": overall["Mean"]},
        {"Metric": "overall_median", "Value": overall["Median"]},
        {"Metric": "overall_tstat", "Value": overall["TStat"]},
        {"Metric": "overall_win_rate", "Value": overall["WinRate"]},
        {"Metric": "overall_profit_factor", "Value": overall["ProfitFactor"]},
        {"Metric": "overall_stop_rate", "Value": float(rows["DisasterStopped"].mean())},
        {"Metric": "overall_top5_share_positive_sum", "Value": overall["Top5SharePositiveSum"]},
        {"Metric": "file_regime_count", "Value": float(len(by_file))},
        {"Metric": "file_positive_fraction", "Value": float((by_file["Mean"] > 0).mean()) if not by_file.empty else np.nan},
        {"Metric": "file_median_mean", "Value": float(by_file["Mean"].median()) if not by_file.empty else np.nan},
        {"Metric": "week_regime_count", "Value": float(len(by_week))},
        {"Metric": "week_positive_fraction", "Value": float((by_week["Mean"] > 0).mean()) if not by_week.empty else np.nan},
        {"Metric": "week_median_mean", "Value": float(by_week["Mean"].median()) if not by_week.empty else np.nan},
    ]
    if not by_session.empty:
        out.extend([
            {"Metric": "session_regime_count", "Value": float(len(by_session))},
            {"Metric": "session_positive_fraction", "Value": float((by_session["Mean"] > 0).mean())},
            {"Metric": "session_median_mean", "Value": float(by_session["Mean"].median())},
        ])
    return pd.DataFrame(out)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/time_exit_robustness_es_compression_precursor")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--pressure", default="CompressionPressure")
    p.add_argument("--entropy-min", type=float, default=0.88)
    p.add_argument("--entropy-max", type=float, default=1.30)
    p.add_argument("--directional-presence", type=float, default=0.0)
    p.add_argument("--lookback", type=int, default=5)
    p.add_argument("--exit-horizon", type=int, default=5)
    p.add_argument("--disaster-stop", type=float, default=20.0)
    args = p.parse_known_args(argv)[0]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = prepare_df(args)
    entries = find_entries(df, args)
    rows = apply_policy(df, entries, args)

    by_file = group_summary(rows, ["File"])
    by_week = group_summary(rows, ["RegimeWeek"]) if "RegimeWeek" in rows.columns else pd.DataFrame()
    if "SessionContext" in rows.columns and rows["SessionContext"].astype(str).str.len().sum() > 0:
        by_session = group_summary(rows, ["SessionContext"])
    else:
        by_session = pd.DataFrame()
    score = build_scorecard(rows, by_file, by_week, by_session)

    entries.to_csv(outdir / "robustness_entries.csv", index=False)
    rows.to_csv(outdir / "robustness_policy_rows.csv", index=False)
    by_file.to_csv(outdir / "robustness_by_file.csv", index=False)
    by_week.to_csv(outdir / "robustness_by_week.csv", index=False)
    by_session.to_csv(outdir / "robustness_by_session.csv", index=False)
    score.to_csv(outdir / "robustness_scorecard.csv", index=False)

    print("APVA time-exit robustness complete")
    print(score.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
