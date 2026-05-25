#!/usr/bin/env python3
"""
APVA time-exit policy.

Purpose
-------
The refined APVA precursor state showed favorable 5-bar raw movement, but fixed
stop/target truncation damaged the payoff because adverse excursion is common.

This script tests time-exit policies:
    - exit after N bars
    - optional disaster stop only
    - optional profit cap only

Primary hypothesis:
    The state behaves like a convex expansion window where tight stops are harmful.

Recommended Jupyter usage:

    import runpy
    ns = runpy.run_path("scripts/apva_time_exit_policy.py", run_name="not_main")
    ns["main"]([
        "--input", "tables/apva_forward_signed_return_dataset_v1.csv",
        "--outdir", "outputs/time_exit_policy_es_compression_precursor"
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


def parse_grid(s: str) -> list[float | None]:
    vals: list[float | None] = []
    for part in s.split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p in {"none", "na", "null"}:
            vals.append(None)
        else:
            vals.append(float(p))
    return vals


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


def build_paths(df: pd.DataFrame, entries: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    lookup = {(r.File, int(r.BarIndex)): r for r in df.itertuples(index=False)}
    rows = []
    for eid, e in entries.iterrows():
        file = e["File"]
        entry_bar = int(e["BarIndex"])
        entry_close = float(e["Close"])
        direction = float(e["EntryDirectionSign"])
        for horizon in args.exit_horizons:
            path = []
            for offset in range(0, int(horizon) + 1):
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
            final = p.loc[p["Offset"].eq(p["Offset"].max())].iloc[0]
            rows.append({
                "EntryId": eid,
                "File": file,
                "EntryBarIndex": entry_bar,
                "EntryTime": e.get("Time", ""),
                "ExitHorizon": int(horizon),
                "ObservedHorizon": int(p["Offset"].max()),
                "HorizonSignedMove": float(final["SignedCloseMove"]),
                "MaxFavorableMove": float(p["FavorableMove"].max()),
                "MaxAdverseMove": float(p["AdverseMove"].min()),
                "EntryDirectionSign": direction,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No paths built")
    return out


def apply_time_policy(row: pd.Series, disaster_stop: float | None, profit_cap: float | None) -> float:
    final = float(row["HorizonSignedMove"])
    mae = float(row["MaxAdverseMove"])
    mfe = float(row["MaxFavorableMove"])

    # Conservative path approximation: if disaster stop was touched at any time,
    # assume stop-out. Profit cap is applied only if no disaster stop fired.
    if disaster_stop is not None and mae <= -abs(disaster_stop):
        return -abs(disaster_stop)
    if profit_cap is not None and mfe >= profit_cap:
        return profit_cap
    if profit_cap is not None:
        final = min(final, profit_cap)
    if disaster_stop is not None:
        final = max(final, -abs(disaster_stop))
    return final


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
        "TStat": tstat(x),
        "WinRate": float((x > 0).mean()) if len(x) else np.nan,
        "LossRate": float((x < 0).mean()) if len(x) else np.nan,
        "ProfitFactor": float(gw / gl) if gl > 0 else float("inf"),
        "Min": float(x.min()) if len(x) else np.nan,
        "Q25": float(x.quantile(0.25)) if len(x) else np.nan,
        "Q75": float(x.quantile(0.75)) if len(x) else np.nan,
        "Q90": float(x.quantile(0.90)) if len(x) else np.nan,
        "Max": float(x.max()) if len(x) else np.nan,
        "Top5SharePositiveSum": tail_share(x, 0.05),
        "Top10SharePositiveSum": tail_share(x, 0.10),
    }


def build_surface(paths: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for horizon in sorted(paths["ExitHorizon"].unique()):
        hp = paths[paths["ExitHorizon"].eq(horizon)]
        for stop in args.disaster_stops:
            for cap in args.profit_caps:
                outcomes = hp.apply(apply_time_policy, axis=1, disaster_stop=stop, profit_cap=cap)
                d = summarize(outcomes)
                d.update({
                    "ExitHorizon": int(horizon),
                    "DisasterStop": "None" if stop is None else stop,
                    "ProfitCap": "None" if cap is None else cap,
                })
                rows.append(d)
    out = pd.DataFrame(rows)
    return out.sort_values(["Mean", "ProfitFactor", "TStat"], ascending=False)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/time_exit_policy_es_compression_precursor")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--pressure", default="CompressionPressure")
    p.add_argument("--entropy-min", type=float, default=0.88)
    p.add_argument("--entropy-max", type=float, default=1.30)
    p.add_argument("--directional-presence", type=float, default=0.0)
    p.add_argument("--lookback", type=int, default=5)
    p.add_argument("--exit-horizons", type=parse_grid, default=parse_grid("3,4,5,6,8,10"))
    p.add_argument("--disaster-stops", type=parse_grid, default=parse_grid("None,12,16,20,24"))
    p.add_argument("--profit-caps", type=parse_grid, default=parse_grid("None,16,24,32,48"))
    args = p.parse_known_args(argv)[0]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = prepare_df(args)
    entries = find_entries(df, args)
    paths = build_paths(df, entries, args)
    surface = build_surface(paths, args)
    best = surface.iloc[0]

    raw5 = paths[paths["ExitHorizon"].eq(5)]
    raw_summary = summarize(raw5["HorizonSignedMove"])
    score = pd.DataFrame([
        {"Metric": "entry_count", "Value": float(len(entries))},
        {"Metric": "raw_5bar_mean", "Value": raw_summary["Mean"]},
        {"Metric": "raw_5bar_median", "Value": raw_summary["Median"]},
        {"Metric": "raw_5bar_tstat", "Value": raw_summary["TStat"]},
        {"Metric": "raw_5bar_win_rate", "Value": raw_summary["WinRate"]},
        {"Metric": "best_exit_horizon", "Value": float(best["ExitHorizon"])},
        {"Metric": "best_disaster_stop", "Value": best["DisasterStop"]},
        {"Metric": "best_profit_cap", "Value": best["ProfitCap"]},
        {"Metric": "best_mean", "Value": float(best["Mean"])},
        {"Metric": "best_median", "Value": float(best["Median"])},
        {"Metric": "best_tstat", "Value": float(best["TStat"])},
        {"Metric": "best_win_rate", "Value": float(best["WinRate"])},
        {"Metric": "best_profit_factor", "Value": float(best["ProfitFactor"])},
        {"Metric": "best_top5_share_positive_sum", "Value": float(best["Top5SharePositiveSum"])},
    ])

    entries.to_csv(outdir / "time_exit_entries.csv", index=False)
    paths.to_csv(outdir / "time_exit_paths.csv", index=False)
    surface.to_csv(outdir / "time_exit_policy_surface.csv", index=False)
    score.to_csv(outdir / "time_exit_scorecard.csv", index=False)

    print("APVA time-exit policy complete")
    print(score.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
