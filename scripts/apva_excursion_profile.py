#!/usr/bin/env python3
"""APVA excursion profile for the discovered compression state."""

from __future__ import annotations

import argparse
import math
import sys
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


def week_key(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    iso = dt.dt.isocalendar()
    out = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    out.loc[dt.isna()] = "UnknownWeek"
    return out


def dist_row(df: pd.DataFrame, col: str, label: str) -> dict:
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    if x.empty:
        return {"Metric": label, "Count": 0}
    xs = x.sort_values(ascending=False)
    top5n = max(1, math.ceil(len(xs) * 0.05))
    top10n = max(1, math.ceil(len(xs) * 0.10))
    pos_sum = xs[xs > 0].sum()
    total_sum = xs.sum()
    def share(part, denom):
        return float(part / denom) if denom != 0 else float("nan")
    return {
        "Metric": label,
        "Count": int(len(x)),
        "Mean": float(x.mean()),
        "Median": float(x.median()),
        "Std": float(x.std(ddof=1)) if len(x) > 1 else float("nan"),
        "TStat": tstat(x),
        "Min": float(x.min()),
        "Q05": float(x.quantile(0.05)),
        "Q25": float(x.quantile(0.25)),
        "Q75": float(x.quantile(0.75)),
        "Q90": float(x.quantile(0.90)),
        "Q95": float(x.quantile(0.95)),
        "Max": float(x.max()),
        "PositiveFraction": float((x > 0).mean()),
        "Top5PctCount": int(top5n),
        "Top5PctSum": float(xs.iloc[:top5n].sum()),
        "Top5PctShareOfPositiveSum": share(xs.iloc[:top5n].sum(), pos_sum),
        "Top5PctShareOfTotalSum": share(xs.iloc[:top5n].sum(), total_sum),
        "Top10PctCount": int(top10n),
        "Top10PctSum": float(xs.iloc[:top10n].sum()),
        "Top10PctShareOfPositiveSum": share(xs.iloc[:top10n].sum(), pos_sum),
        "Top10PctShareOfTotalSum": share(xs.iloc[:top10n].sum(), total_sum),
        "Sum": float(total_sum),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/excursion_profile_es_compression_entropy_mid_presence_zero")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--pressure", default="CompressionPressure")
    p.add_argument("--entropy-min", type=float, default=0.88)
    p.add_argument("--entropy-max", type=float, default=1.30)
    p.add_argument("--directional-presence", type=float, default=0.0)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_known_args(argv)[0]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(args.input)
        for col in ["HorizonBars", "RollingEntropy", "RollingDirectionalPresence", "SignedReturn", "RawReturn", "NormalizedReturn", "DirectionalMFE", "DirectionalMAE"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        mask = (
            df["Instrument"].astype(str).str.upper().eq(args.instrument.upper())
            & df["HorizonBars"].eq(args.horizon)
            & df["DominantPressure"].astype(str).eq(args.pressure)
            & df["RollingEntropy"].between(args.entropy_min, args.entropy_max, inclusive="both")
            & df["RollingDirectionalPresence"].eq(args.directional_presence)
        )
        sel = df.loc[mask].dropna(subset=["SignedReturn"]).copy()
        if sel.empty:
            raise RuntimeError("No rows selected")

        metrics = ["SignedReturn", "RawReturn", "NormalizedReturn", "DirectionalMFE", "DirectionalMAE"]
        summary = pd.DataFrame([dist_row(sel, c, c) for c in metrics if c in sel.columns])

        if "DirectionalMFE" in sel.columns and "DirectionalMAE" in sel.columns:
            sel["MFE_to_abs_MAE"] = sel["DirectionalMFE"] / sel["DirectionalMAE"].abs().replace(0, np.nan)
            summary = pd.concat([summary, pd.DataFrame([dist_row(sel, "MFE_to_abs_MAE", "MFE_to_abs_MAE")])], ignore_index=True)

        if "Time" in sel.columns:
            sel["RegimeWeek"] = week_key(sel["Time"])
            week = sel.groupby("RegimeWeek", dropna=False).agg(
                Count=("SignedReturn", "size"),
                MeanSignedReturn=("SignedReturn", "mean"),
                MedianSignedReturn=("SignedReturn", "median"),
                MaxSignedReturn=("SignedReturn", "max"),
                MinSignedReturn=("SignedReturn", "min"),
                MeanMFE=("DirectionalMFE", "mean") if "DirectionalMFE" in sel.columns else ("SignedReturn", "mean"),
                MeanMAE=("DirectionalMAE", "mean") if "DirectionalMAE" in sel.columns else ("SignedReturn", "mean"),
            ).reset_index()
        else:
            week = pd.DataFrame()

        top = sel.sort_values("SignedReturn", ascending=False).head(20)
        cols = [c for c in ["Time", "File", "SignedReturn", "RawReturn", "NormalizedReturn", "DirectionalMFE", "DirectionalMAE", "RollingEntropy", "RollingDirectionalPresence"] if c in top.columns]

        sr = summary.loc[summary["Metric"].eq("SignedReturn")].iloc[0]
        classification = "convex_tail_dominated" if sr.get("Top5PctShareOfPositiveSum", 0) >= 0.5 and sr["Mean"] > sr["Median"] * 3 else "not_tail_dominated"
        score = pd.DataFrame([
            {"Metric": "selected_count", "Value": float(len(sel))},
            {"Metric": "mean_signed_return", "Value": float(sr["Mean"])},
            {"Metric": "median_signed_return", "Value": float(sr["Median"])},
            {"Metric": "t_stat", "Value": float(sr["TStat"])},
            {"Metric": "positive_fraction", "Value": float(sr["PositiveFraction"])},
            {"Metric": "max_signed_return", "Value": float(sr["Max"])},
            {"Metric": "top5_share_positive_sum", "Value": float(sr["Top5PctShareOfPositiveSum"])},
            {"Metric": "top10_share_positive_sum", "Value": float(sr["Top10PctShareOfPositiveSum"])},
            {"Metric": "edge_shape_classification", "Value": classification},
        ])

        sel.to_csv(outdir / "selected_excursion_rows.csv", index=False)
        summary.to_csv(outdir / "excursion_profile_summary.csv", index=False)
        week.to_csv(outdir / "excursion_by_week.csv", index=False)
        top[cols].to_csv(outdir / "tail_event_examples.csv", index=False)
        score.to_csv(outdir / "excursion_scorecard.csv", index=False)

        print("APVA excursion profile complete")
        print(score.to_string(index=False))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    main()
