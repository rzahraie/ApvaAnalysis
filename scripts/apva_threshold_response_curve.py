#!/usr/bin/env python3
"""
APVA threshold response curve.

Purpose
-------
Test whether a conditioned APVA edge is tied to a real threshold boundary rather
than an arbitrary quantile artifact.

Primary target:

    ES / 5-bar horizon / CompressionPressure
    conditioned on RollingDirectionalPresence <= threshold

Recommended Jupyter usage
-------------------------

    import runpy
    ns = runpy.run_path("scripts/apva_threshold_response_curve.py", run_name="not_main")
    ns["main"]([
        "--input", "tables/apva_forward_signed_return_dataset_v1.csv",
        "--instrument", "ES",
        "--horizon", "5",
        "--pressure", "CompressionPressure",
        "--condition-col", "RollingDirectionalPresence",
        "--mode", "le",
        "--threshold-start", "0.00",
        "--threshold-stop", "1.00",
        "--threshold-step", "0.02",
        "--min-count", "30",
        "--outdir", "outputs/threshold_response_es_compression_low_directional_presence",
    ])

Outputs
-------
- threshold_response_curve.csv
- threshold_week_summary.csv
- threshold_scorecard.csv
- selected_threshold_base_rows.csv

Interpretation
--------------
Look for a plateau, not a single lucky threshold.

Good behavior:
    low thresholds produce stable positive expectancy and positive weekly fraction,
    then the edge decays as the threshold admits noisier regimes.

Bad behavior:
    results jump chaotically from threshold to threshold.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def infer_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    by_norm = {_norm(c): c for c in columns}
    for cand in candidates:
        hit = by_norm.get(_norm(cand))
        if hit is not None:
            return hit
    cols = list(columns)
    for cand in sorted(candidates, key=len, reverse=True):
        nc = _norm(cand)
        for col in cols:
            if nc and nc in _norm(col):
                return col
    return None


def safe_t_stat(vals: pd.Series) -> float:
    x = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 2:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return float("nan")
    return float(np.mean(x) / (sd / math.sqrt(len(x))))


def add_week_column(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[time_col], errors="coerce")
    iso = dt.dt.isocalendar()
    out["RegimeWeek"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    out.loc[dt.isna(), "RegimeWeek"] = "UnknownWeek"
    return out


def thresholds(start: float, stop: float, step: float) -> list[float]:
    vals = []
    x = start
    while x <= stop + step * 0.5:
        vals.append(round(x, 10))
        x += step
    return vals


def select_by_threshold(df: pd.DataFrame, col: str, threshold: float, mode: str) -> pd.DataFrame:
    if mode == "le":
        return df.loc[df[col] <= threshold].copy()
    if mode == "ge":
        return df.loc[df[col] >= threshold].copy()
    raise ValueError(f"Unsupported mode: {mode}")


def summarize_returns(vals: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(vals, errors="coerce").dropna()
    if x.empty:
        return {
            "Count": 0,
            "MeanSignedReturn": float("nan"),
            "MedianSignedReturn": float("nan"),
            "StdSignedReturn": float("nan"),
            "TStat": float("nan"),
            "PositiveRowFraction": float("nan"),
            "ZeroRowFraction": float("nan"),
            "NegativeRowFraction": float("nan"),
        }
    return {
        "Count": int(len(x)),
        "MeanSignedReturn": float(x.mean()),
        "MedianSignedReturn": float(x.median()),
        "StdSignedReturn": float(x.std(ddof=1)) if len(x) > 1 else float("nan"),
        "TStat": safe_t_stat(x),
        "PositiveRowFraction": float((x > 0).mean()),
        "ZeroRowFraction": float((x == 0).mean()),
        "NegativeRowFraction": float((x < 0).mean()),
    }


def load_base(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, str]]:
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    cols = list(df.columns)

    instrument_col = args.instrument_col or infer_column(cols, ["Instrument", "Symbol", "Market"])
    horizon_col = args.horizon_col or infer_column(cols, ["HorizonBars", "Horizon", "ForwardBars", "LookaheadBars", "BarsForward"])
    pressure_col = args.pressure_col or infer_column(cols, ["DominantPressure", "PressureState", "Pressure", "PressureClass"])
    return_col = args.return_col or infer_column(cols, ["SignedReturn", "SignedNormalizedReturn", "MeanSignedReturn", "ForwardSignedReturn", "SignedForwardReturn"])
    time_col = args.time_col or infer_column(cols, ["Time", "Timestamp", "DateTime", "BarTime", "StartTime"])
    condition_col = args.condition_col or infer_column(cols, ["RollingDirectionalPresence", "CurrentDirectionalEmergence", "RollingEntropy", "RollingVelocity"])

    required = {
        "instrument_col": instrument_col,
        "horizon_col": horizon_col,
        "pressure_col": pressure_col,
        "return_col": return_col,
        "condition_col": condition_col,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required columns {missing}. Columns found: {cols}")

    assert instrument_col and horizon_col and pressure_col and return_col and condition_col
    mask = pd.Series(True, index=df.index)
    mask &= df[instrument_col].astype(str).str.upper().eq(args.instrument.upper())
    mask &= pd.to_numeric(df[horizon_col], errors="coerce").eq(float(args.horizon))
    mask &= df[pressure_col].astype(str).str.strip().eq(args.pressure)

    base = df.loc[mask].copy()
    base[return_col] = pd.to_numeric(base[return_col], errors="coerce")
    base[condition_col] = pd.to_numeric(base[condition_col], errors="coerce")
    base = base.dropna(subset=[return_col, condition_col]).reset_index(drop=True)

    if base.empty:
        raise RuntimeError("Base filter selected zero usable rows.")

    mapping = {
        "instrument_col": instrument_col,
        "horizon_col": horizon_col,
        "pressure_col": pressure_col,
        "return_col": return_col,
        "condition_col": condition_col,
        "time_col": time_col or "",
    }
    return base, mapping


def run_threshold_sweep(base: pd.DataFrame, mapping: dict[str, str], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rcol = mapping["return_col"]
    ccol = mapping["condition_col"]
    time_col = mapping.get("time_col")
    base_week = add_week_column(base, time_col) if time_col and time_col in base.columns else None

    curve_rows = []
    week_rows = []

    for th in thresholds(args.threshold_start, args.threshold_stop, args.threshold_step):
        subset = select_by_threshold(base, ccol, th, args.mode)
        if len(subset) < args.min_count:
            continue

        ret = summarize_returns(subset[rcol])
        ret.update({
            "Threshold": th,
            "Mode": args.mode,
            "ConditionMin": float(subset[ccol].min()),
            "ConditionMax": float(subset[ccol].max()),
            "ConditionMean": float(subset[ccol].mean()),
        })

        if base_week is not None:
            wsub = select_by_threshold(base_week, ccol, th, args.mode)
            week_means = []
            for week, wg in wsub.groupby("RegimeWeek", dropna=False, sort=True, observed=False):
                vals = pd.to_numeric(wg[rcol], errors="coerce").dropna()
                if len(vals) < args.min_week_count:
                    continue
                wm = float(vals.mean())
                week_means.append(wm)
                week_rows.append({
                    "Threshold": th,
                    "Mode": args.mode,
                    "RegimeWeek": week,
                    "Count": int(len(vals)),
                    "MeanSignedReturn": wm,
                    "MedianSignedReturn": float(vals.median()),
                    "TStat": safe_t_stat(vals),
                    "PositiveRowFraction": float((vals > 0).mean()),
                })
            if week_means:
                week_arr = np.array(week_means, dtype=float)
                ret.update({
                    "WeekRegimeCount": int(len(week_arr)),
                    "WeeklyPositiveFraction": float(np.mean(week_arr > 0)),
                    "WeeklyMeanOfMeans": float(np.mean(week_arr)),
                    "WeeklyMedianOfMeans": float(np.median(week_arr)),
                    "WeeklyMinMean": float(np.min(week_arr)),
                    "WeeklyMaxMean": float(np.max(week_arr)),
                })
            else:
                ret.update({
                    "WeekRegimeCount": 0,
                    "WeeklyPositiveFraction": float("nan"),
                    "WeeklyMeanOfMeans": float("nan"),
                    "WeeklyMedianOfMeans": float("nan"),
                    "WeeklyMinMean": float("nan"),
                    "WeeklyMaxMean": float("nan"),
                })

        curve_rows.append(ret)

    curve = pd.DataFrame(curve_rows)
    week = pd.DataFrame(week_rows)
    if not curve.empty:
        front = ["Threshold", "Mode", "Count", "MeanSignedReturn", "MedianSignedReturn", "TStat", "PositiveRowFraction"]
        rest = [c for c in curve.columns if c not in front]
        curve = curve[front + rest]
    return curve, week


def build_scorecard(curve: pd.DataFrame, base: pd.DataFrame, mapping: dict[str, str], args: argparse.Namespace) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame([{"Metric": "error", "Value": "empty_curve"}])

    # Require enough regimes where possible and avoid just optimizing tiny support.
    eligible = curve.copy()
    if "WeekRegimeCount" in eligible.columns:
        eligible = eligible.loc[eligible["WeekRegimeCount"] >= args.min_week_regimes].copy()
    if eligible.empty:
        eligible = curve.copy()

    # Stability-first ranking: weekly persistence, then median weekly return, then t-stat.
    sort_cols = []
    ascending = []
    for col in ["WeeklyPositiveFraction", "WeeklyMedianOfMeans", "MeanSignedReturn", "TStat", "Count"]:
        if col in eligible.columns:
            sort_cols.append(col)
            ascending.append(False)
    best = eligible.sort_values(sort_cols, ascending=ascending).iloc[0]

    rcol = mapping["return_col"]
    ccol = mapping["condition_col"]
    vals = base[rcol]
    rows = [
        {"Metric": "base_count", "Value": float(len(base))},
        {"Metric": "base_mean_signed_return", "Value": float(vals.mean())},
        {"Metric": "base_median_signed_return", "Value": float(vals.median())},
        {"Metric": "base_t_stat", "Value": safe_t_stat(vals)},
        {"Metric": "condition_col", "Value": ccol},
        {"Metric": "condition_min", "Value": float(base[ccol].min())},
        {"Metric": "condition_max", "Value": float(base[ccol].max())},
        {"Metric": "best_threshold", "Value": float(best["Threshold"])},
        {"Metric": "best_threshold_mode", "Value": str(best["Mode"])},
        {"Metric": "best_threshold_count", "Value": float(best["Count"])},
        {"Metric": "best_threshold_mean_signed_return", "Value": float(best["MeanSignedReturn"])},
        {"Metric": "best_threshold_median_signed_return", "Value": float(best["MedianSignedReturn"])},
        {"Metric": "best_threshold_t_stat", "Value": float(best["TStat"])},
        {"Metric": "best_threshold_positive_row_fraction", "Value": float(best["PositiveRowFraction"])},
    ]
    for col in ["WeekRegimeCount", "WeeklyPositiveFraction", "WeeklyMedianOfMeans", "WeeklyMeanOfMeans", "WeeklyMinMean", "WeeklyMaxMean"]:
        if col in best.index:
            rows.append({"Metric": f"best_threshold_{col}", "Value": float(best[col])})
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="APVA threshold response curve")
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/threshold_response_es_compression_low_directional_presence")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--pressure", default="CompressionPressure")
    p.add_argument("--condition-col", default="RollingDirectionalPresence")
    p.add_argument("--mode", choices=["le", "ge"], default="le")
    p.add_argument("--threshold-start", type=float, default=0.0)
    p.add_argument("--threshold-stop", type=float, default=1.0)
    p.add_argument("--threshold-step", type=float, default=0.02)
    p.add_argument("--min-count", type=int, default=30)
    p.add_argument("--min-week-count", type=int, default=10)
    p.add_argument("--min-week-regimes", type=int, default=3)

    p.add_argument("--instrument-col", default=None)
    p.add_argument("--horizon-col", default=None)
    p.add_argument("--pressure-col", default=None)
    p.add_argument("--return-col", default=None)
    p.add_argument("--time-col", default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_known_args(argv)[0]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        base, mapping = load_base(args)
        curve, week = run_threshold_sweep(base, mapping, args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if curve.empty:
        print("ERROR: threshold curve is empty; lower --min-count or check filters", file=sys.stderr)
        return 2

    base.to_csv(outdir / "selected_threshold_base_rows.csv", index=False)
    pd.DataFrame([mapping]).to_csv(outdir / "column_mapping.csv", index=False)
    curve.to_csv(outdir / "threshold_response_curve.csv", index=False)
    week.to_csv(outdir / "threshold_week_summary.csv", index=False)

    score = build_scorecard(curve, base, mapping, args)
    score.to_csv(outdir / "threshold_scorecard.csv", index=False)

    print("APVA threshold response curve complete")
    print(f"Base rows: {len(base)}")
    print(f"Condition column: {mapping['condition_col']}")
    print(f"Threshold rows: {len(curve)}")
    print(f"Output directory: {outdir}")
    print(score.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
