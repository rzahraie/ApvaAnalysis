#!/usr/bin/env python3
"""
APVA five-bar expansion capture.

Purpose
-------
The refined APVA precursor state showed its best followthrough around offset 5
and decayed by offset 20. This script treats the discovered state as a short-lived
5-bar expansion window rather than an open-ended trend signal.

Entry state:
    ES / 5-bar horizon / CompressionPressure
    RollingDirectionalPresence == 0
    RollingEntropy in [0.88, 1.30]
    Prior 5 DominantPressure path:
        RotationalPressure > RotationalPressure > CompressionPressure > CompressionPressure > CompressionPressure

This script compares fixed-horizon exits and target/stop policies constrained to
that 5-bar expansion window.

Outputs
-------
- five_bar_entries.csv
- five_bar_horizon_summary.csv
- five_bar_policy_surface.csv
- five_bar_scorecard.csv

Recommended Jupyter usage
-------------------------

    import runpy
    ns = runpy.run_path("scripts/apva_five_bar_expansion_capture.py", run_name="not_main")
    ns["main"]([
        "--input", "tables/apva_forward_signed_return_dataset_v1.csv",
        "--outdir", "outputs/five_bar_expansion_capture_es_compression_precursor"
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
        "RollingEntropy", "RollingDirectionalPresence", "SignedReturn", "DirectionalMFE", "DirectionalMAE"
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
        path = prior["DominantPressure"].astype(str).tolist()
        if path != TARGET_PATH:
            continue
        r = e.copy()
        r["EntryDirectionSign"] = signed_direction(e)
        r["PriorDominantPressureSeq"] = " > ".join(path)
        rows.append(r)
    entries = pd.DataFrame(rows)
    if entries.empty:
        raise RuntimeError("No refined entries found")
    return entries.reset_index(drop=True)


def build_entry_paths(df: pd.DataFrame, entries: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    lookup = {(r.File, int(r.BarIndex)): r for r in df.itertuples(index=False)}
    rows = []
    for eid, e in entries.iterrows():
        file = e["File"]
        entry_bar = int(e["BarIndex"])
        entry_close = float(e["Close"])
        direction = float(e["EntryDirectionSign"])
        path_rows = []
        for offset in range(0, args.exit_horizon + 1):
            hit = lookup.get((file, entry_bar + offset))
            if hit is None:
                continue
            close = float(getattr(hit, "Close"))
            signed_move = (close - entry_close) * direction
            high = float(getattr(hit, "High", np.nan)) if hasattr(hit, "High") else np.nan
            low = float(getattr(hit, "Low", np.nan)) if hasattr(hit, "Low") else np.nan
            if direction > 0:
                fav = high - entry_close if np.isfinite(high) else signed_move
                adv = low - entry_close if np.isfinite(low) else signed_move
            else:
                fav = entry_close - low if np.isfinite(low) else signed_move
                adv = entry_close - high if np.isfinite(high) else signed_move
            path_rows.append({
                "Offset": offset,
                "Close": close,
                "SignedMove": signed_move,
                "FavorableMove": fav,
                "AdverseMove": adv,
            })
        if not path_rows:
            continue
        p = pd.DataFrame(path_rows)
        final = p.loc[p["Offset"].eq(p["Offset"].max())].iloc[0]
        rows.append({
            "EntryId": eid,
            "File": file,
            "EntryBarIndex": entry_bar,
            "EntryTime": e.get("Time", ""),
            "EntryDirectionSign": direction,
            "ObservedHorizon": int(p["Offset"].max()),
            "HorizonSignedMove": float(final["SignedMove"]),
            "MaxFavorableMove": float(p["FavorableMove"].max()),
            "MaxAdverseMove": float(p["AdverseMove"].min()),
            "EntrySignedReturn": float(e["SignedReturn"]),
            "EntryDirectionalMFE": float(e["DirectionalMFE"]),
            "EntryDirectionalMAE": float(e["DirectionalMAE"]),
            "RollingEntropy": float(e["RollingEntropy"]),
            "RollingDirectionalPresence": float(e["RollingDirectionalPresence"]),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No entry paths built")
    return out


def policy_result(row: pd.Series, stop: float, target: float, collision: str) -> float:
    mfe = float(row["MaxFavorableMove"])
    mae = float(row["MaxAdverseMove"])
    final = float(row["HorizonSignedMove"])
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
    return float(np.clip(final, -abs(stop), target))


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
        "ProfitFactor": float(gw / gl) if gl > 0 else float("inf"),
        "Min": float(x.min()) if len(x) else np.nan,
        "Max": float(x.max()) if len(x) else np.nan,
        "Top5SharePositiveSum": tail_share(x, 0.05),
        "Top10SharePositiveSum": tail_share(x, 0.10),
    }


def build_horizon_summary(paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["HorizonSignedMove", "MaxFavorableMove", "MaxAdverseMove", "EntrySignedReturn", "EntryDirectionalMFE", "EntryDirectionalMAE"]:
        d = summarize(paths[col])
        d["Metric"] = col
        rows.append(d)
    return pd.DataFrame(rows)[["Metric", "Count", "Mean", "Median", "TStat", "WinRate", "ProfitFactor", "Min", "Max", "Top5SharePositiveSum", "Top10SharePositiveSum"]]


def build_policy_surface(paths: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for stop in args.stops:
        for target in args.targets:
            outcomes = paths.apply(policy_result, axis=1, stop=stop, target=target, collision=args.collision)
            d = summarize(outcomes)
            d.update({"Stop": stop, "Target": target, "CollisionMode": args.collision})
            rows.append(d)
    out = pd.DataFrame(rows)
    return out.sort_values(["Mean", "ProfitFactor", "TStat"], ascending=False)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/five_bar_expansion_capture_es_compression_precursor")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--pressure", default="CompressionPressure")
    p.add_argument("--entropy-min", type=float, default=0.88)
    p.add_argument("--entropy-max", type=float, default=1.30)
    p.add_argument("--directional-presence", type=float, default=0.0)
    p.add_argument("--lookback", type=int, default=5)
    p.add_argument("--exit-horizon", type=int, default=5)
    p.add_argument("--stops", type=parse_grid, default=parse_grid("2,4,6,8,10,12"))
    p.add_argument("--targets", type=parse_grid, default=parse_grid("2,4,6,8,10,12,16,20,24,32"))
    p.add_argument("--collision", choices=["conservative", "optimistic", "mfe_priority"], default="conservative")
    args = p.parse_known_args(argv)[0]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = prepare_df(args)
    entries = find_entries(df, args)
    paths = build_entry_paths(df, entries, args)
    horizon = build_horizon_summary(paths)
    policy = build_policy_surface(paths, args)
    best = policy.iloc[0]
    hs = horizon.loc[horizon["Metric"].eq("HorizonSignedMove")].iloc[0]

    score = pd.DataFrame([
        {"Metric": "entry_count", "Value": float(len(paths))},
        {"Metric": "exit_horizon", "Value": float(args.exit_horizon)},
        {"Metric": "horizon_mean", "Value": float(hs["Mean"])},
        {"Metric": "horizon_median", "Value": float(hs["Median"])},
        {"Metric": "horizon_tstat", "Value": float(hs["TStat"])},
        {"Metric": "horizon_win_rate", "Value": float(hs["WinRate"])},
        {"Metric": "best_stop", "Value": float(best["Stop"])},
        {"Metric": "best_target", "Value": float(best["Target"])},
        {"Metric": "best_mean", "Value": float(best["Mean"])},
        {"Metric": "best_median", "Value": float(best["Median"])},
        {"Metric": "best_tstat", "Value": float(best["TStat"])},
        {"Metric": "best_win_rate", "Value": float(best["WinRate"])},
        {"Metric": "best_profit_factor", "Value": float(best["ProfitFactor"])},
        {"Metric": "best_top5_share_positive_sum", "Value": float(best["Top5SharePositiveSum"])},
    ])

    entries.to_csv(outdir / "five_bar_entries.csv", index=False)
    paths.to_csv(outdir / "five_bar_entry_paths.csv", index=False)
    horizon.to_csv(outdir / "five_bar_horizon_summary.csv", index=False)
    policy.to_csv(outdir / "five_bar_policy_surface.csv", index=False)
    score.to_csv(outdir / "five_bar_scorecard.csv", index=False)

    print("APVA five-bar expansion capture complete")
    print(score.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
