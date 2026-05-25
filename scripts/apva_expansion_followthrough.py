#!/usr/bin/env python3
"""
APVA expansion followthrough analysis.

Purpose
-------
After the refined APVA precursor entry, measure whether the market shows real
post-transition followthrough or merely a one-bar/statistical tail artifact.

Entry condition:
    ES / 5-bar horizon / CompressionPressure
    RollingDirectionalPresence == 0
    RollingEntropy in [0.88, 1.30]
    Prior 5 DominantPressure path:
        RotationalPressure > RotationalPressure > CompressionPressure > CompressionPressure > CompressionPressure

For each entry, this script inspects future rows from offset 0..N in the row-level
forward-return dataset and summarizes state evolution and realized path changes.

Outputs
-------
- followthrough_curve.csv
- followthrough_entry_paths.csv
- followthrough_state_evolution.csv
- followthrough_scorecard.csv

Recommended Jupyter usage
-------------------------

    import runpy
    ns = runpy.run_path("scripts/apva_expansion_followthrough.py", run_name="not_main")
    ns["main"]([
        "--input", "tables/apva_forward_signed_return_dataset_v1.csv",
        "--outdir", "outputs/expansion_followthrough_es_compression_precursor",
        "--max-forward", "20"
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


def signed_direction(row: pd.Series) -> float:
    if "DirectionSign" in row and pd.notna(row["DirectionSign"]):
        try:
            v = float(row["DirectionSign"])
            if v != 0:
                return 1.0 if v > 0 else -1.0
        except Exception:
            pass
    if "ActiveDirection" in row:
        s = str(row["ActiveDirection"]).lower()
        if "up" in s or "long" in s or s == "1":
            return 1.0
        if "down" in s or "short" in s or s == "-1":
            return -1.0
    return 0.0


def load_entries(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
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
        if path == TARGET_PATH:
            r = e.copy()
            r["EntryDirectionSign"] = signed_direction(e)
            r["PriorDominantPressureSeq"] = " > ".join(path)
            rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No precursor-filtered entries found")
    return out.reset_index(drop=True)


def prepare_df(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.input)
    numeric = [
        "BarIndex", "HorizonBars", "Open", "High", "Low", "Close", "FutureClose",
        "RawReturn", "SignedReturn", "NormalizedReturn", "SignedNormalizedReturn",
        "DirectionalMFE", "DirectionalMAE", "RollingEntropy", "RollingDirectionalPresence",
        "RollingVelocity", "DominantPressureValue", "DirectionSign",
    ]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    required = [
        "Instrument", "File", "BarIndex", "HorizonBars", "DominantPressure",
        "RollingEntropy", "RollingDirectionalPresence", "SignedReturn", "Close"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    return df.sort_values(["File", "BarIndex"]).reset_index(drop=True)


def build_followthrough_rows(df: pd.DataFrame, entries: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    index = {(r.File, int(r.BarIndex)): r for r in df.itertuples(index=False)}

    for eid, e in entries.iterrows():
        file = e["File"]
        entry_bar = int(e["BarIndex"])
        entry_close = float(e["Close"])
        direction = float(e.get("EntryDirectionSign", 0.0))
        if direction == 0:
            # fallback: use sign of entry SignedReturn if available, otherwise raw direction unknown
            sr = float(e.get("SignedReturn", 0.0))
            direction = 1.0 if sr >= 0 else -1.0

        for offset in range(0, args.max_forward + 1):
            hit = index.get((file, entry_bar + offset))
            if hit is None:
                continue
            close = float(getattr(hit, "Close")) if pd.notna(getattr(hit, "Close")) else np.nan
            raw_move = close - entry_close if np.isfinite(close) else np.nan
            signed_path_move = raw_move * direction if np.isfinite(raw_move) else np.nan
            rows.append({
                "EntryId": eid,
                "File": file,
                "EntryBarIndex": entry_bar,
                "Offset": offset,
                "BarIndex": int(getattr(hit, "BarIndex")),
                "Time": getattr(hit, "Time", ""),
                "EntryDirectionSign": direction,
                "Close": close,
                "RawMoveFromEntry": raw_move,
                "SignedMoveFromEntry": signed_path_move,
                "MacroState": getattr(hit, "MacroState", ""),
                "ResolvedArchetype": getattr(hit, "ResolvedArchetype", ""),
                "DominantPressure": getattr(hit, "DominantPressure", ""),
                "RollingEntropy": getattr(hit, "RollingEntropy", np.nan),
                "RollingDirectionalPresence": getattr(hit, "RollingDirectionalPresence", np.nan),
                "RollingVelocity": getattr(hit, "RollingVelocity", np.nan),
                "DominantPressureValue": getattr(hit, "DominantPressureValue", np.nan),
                "SignedReturnAtOffset": getattr(hit, "SignedReturn", np.nan),
                "DirectionalMFEAtOffset": getattr(hit, "DirectionalMFE", np.nan),
                "DirectionalMAEAtOffset": getattr(hit, "DirectionalMAE", np.nan),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No followthrough rows built")
    return out


def summarize_curve(ft: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for offset, g in ft.groupby("Offset", sort=True):
        move = pd.to_numeric(g["SignedMoveFromEntry"], errors="coerce").dropna()
        rows.append({
            "Offset": int(offset),
            "Count": int(len(move)),
            "MeanSignedMoveFromEntry": float(move.mean()) if len(move) else np.nan,
            "MedianSignedMoveFromEntry": float(move.median()) if len(move) else np.nan,
            "TStatSignedMove": tstat(move),
            "PositiveMoveFraction": float((move > 0).mean()) if len(move) else np.nan,
            "Q25SignedMove": float(move.quantile(0.25)) if len(move) else np.nan,
            "Q75SignedMove": float(move.quantile(0.75)) if len(move) else np.nan,
            "Q90SignedMove": float(move.quantile(0.90)) if len(move) else np.nan,
            "MeanRollingEntropy": float(pd.to_numeric(g["RollingEntropy"], errors="coerce").mean()),
            "MeanRollingDirectionalPresence": float(pd.to_numeric(g["RollingDirectionalPresence"], errors="coerce").mean()),
            "MeanRollingVelocity": float(pd.to_numeric(g["RollingVelocity"], errors="coerce").mean()),
            "MeanDominantPressureValue": float(pd.to_numeric(g["DominantPressureValue"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows)


def summarize_state_evolution(ft: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for offset, g in ft.groupby("Offset", sort=True):
        total = len(g)
        if total == 0:
            continue
        for col in ["DominantPressure", "MacroState", "ResolvedArchetype"]:
            if col not in g.columns:
                continue
            counts = g[col].astype(str).value_counts(dropna=False)
            for val, cnt in counts.head(10).items():
                rows.append({
                    "Offset": int(offset),
                    "StateColumn": col,
                    "StateValue": val,
                    "Count": int(cnt),
                    "Fraction": float(cnt / total),
                })
    return pd.DataFrame(rows)


def build_entry_paths(ft: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for eid, g in ft.groupby("EntryId", sort=True):
        g = g.sort_values("Offset")
        max_move = pd.to_numeric(g["SignedMoveFromEntry"], errors="coerce").max()
        min_move = pd.to_numeric(g["SignedMoveFromEntry"], errors="coerce").min()
        final_row = g[g["Offset"].eq(g["Offset"].max())].iloc[0]
        rows.append({
            "EntryId": eid,
            "File": final_row["File"],
            "EntryBarIndex": int(final_row["EntryBarIndex"]),
            "ObservedMaxOffset": int(g["Offset"].max()),
            "FinalSignedMove": float(final_row["SignedMoveFromEntry"]),
            "MaxSignedMove": float(max_move),
            "MinSignedMove": float(min_move),
            "ExpansionGE8": bool(max_move >= 8),
            "ExpansionGE16": bool(max_move >= 16),
            "ExpansionGE32": bool(max_move >= 32),
            "AdverseLEMinus6": bool(min_move <= -6),
            "AdverseLEMinus12": bool(min_move <= -12),
        })
    return pd.DataFrame(rows)


def build_scorecard(curve: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    best = curve.sort_values(["MeanSignedMoveFromEntry", "TStatSignedMove"], ascending=False).iloc[0]
    final = curve.sort_values("Offset").iloc[-1]
    rows = [
        {"Metric": "entry_count", "Value": float(len(paths))},
        {"Metric": "best_offset", "Value": float(best["Offset"])},
        {"Metric": "best_offset_mean_signed_move", "Value": float(best["MeanSignedMoveFromEntry"])},
        {"Metric": "best_offset_median_signed_move", "Value": float(best["MedianSignedMoveFromEntry"])},
        {"Metric": "best_offset_tstat", "Value": float(best["TStatSignedMove"])},
        {"Metric": "best_offset_positive_fraction", "Value": float(best["PositiveMoveFraction"])},
        {"Metric": "final_offset", "Value": float(final["Offset"])},
        {"Metric": "final_mean_signed_move", "Value": float(final["MeanSignedMoveFromEntry"])},
        {"Metric": "final_median_signed_move", "Value": float(final["MedianSignedMoveFromEntry"])},
        {"Metric": "fraction_expansion_ge_8", "Value": float(paths["ExpansionGE8"].mean())},
        {"Metric": "fraction_expansion_ge_16", "Value": float(paths["ExpansionGE16"].mean())},
        {"Metric": "fraction_expansion_ge_32", "Value": float(paths["ExpansionGE32"].mean())},
        {"Metric": "fraction_adverse_le_minus_6", "Value": float(paths["AdverseLEMinus6"].mean())},
        {"Metric": "fraction_adverse_le_minus_12", "Value": float(paths["AdverseLEMinus12"].mean())},
        {"Metric": "mean_max_signed_move", "Value": float(paths["MaxSignedMove"].mean())},
        {"Metric": "median_max_signed_move", "Value": float(paths["MaxSignedMove"].median())},
    ]
    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/expansion_followthrough_es_compression_precursor")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--pressure", default="CompressionPressure")
    p.add_argument("--entropy-min", type=float, default=0.88)
    p.add_argument("--entropy-max", type=float, default=1.30)
    p.add_argument("--directional-presence", type=float, default=0.0)
    p.add_argument("--lookback", type=int, default=5)
    p.add_argument("--max-forward", type=int, default=20)
    args = p.parse_known_args(argv)[0]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = prepare_df(args)
    entries = load_entries(df, args)
    ft = build_followthrough_rows(df, entries, args)
    curve = summarize_curve(ft)
    states = summarize_state_evolution(ft)
    paths = build_entry_paths(ft, args)
    score = build_scorecard(curve, paths)

    entries.to_csv(outdir / "followthrough_entries.csv", index=False)
    ft.to_csv(outdir / "followthrough_rows.csv", index=False)
    curve.to_csv(outdir / "followthrough_curve.csv", index=False)
    states.to_csv(outdir / "followthrough_state_evolution.csv", index=False)
    paths.to_csv(outdir / "followthrough_entry_paths.csv", index=False)
    score.to_csv(outdir / "followthrough_scorecard.csv", index=False)

    print("APVA expansion followthrough complete")
    print(score.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
