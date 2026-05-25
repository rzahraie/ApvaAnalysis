#!/usr/bin/env python3
"""
APVA conditioned compression validation.

Purpose
-------
CompressionPressure alone appears informative, but regime-dependent. This script
conditions CompressionPressure on an additional continuous context variable,
defaulting to RollingDirectionalPresence.

Primary question:

    Does ES / 5-bar / CompressionPressure become more stable when segmented by
    RollingDirectionalPresence quantiles?

Recommended Jupyter usage
-------------------------

    import runpy
    ns = runpy.run_path("scripts/apva_conditioned_compression_validation.py", run_name="not_main")
    ns["main"]([
        "--input", "tables/apva_forward_signed_return_dataset_v1.csv",
        "--instrument", "ES",
        "--horizon", "5",
        "--pressure", "CompressionPressure",
        "--condition-col", "RollingDirectionalPresence",
        "--outdir", "outputs/conditioned_compression_es_directional_presence",
    ])

Outputs
-------
- conditioned_quantile_summary.csv
- conditioned_week_summary.csv
- conditioned_scorecard.csv
- selected_conditioned_rows.csv

Interpretation
--------------
Look for bins where:

- mean signed return is positive,
- positive row fraction improves,
- weekly positive fraction improves,
- median weekly return improves,
- support is not tiny.
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


def summarize(df: pd.DataFrame, group_cols: list[str], return_col: str, condition_col: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        vals = pd.to_numeric(g[return_col], errors="coerce").dropna()
        cvals = pd.to_numeric(g[condition_col], errors="coerce").dropna()
        if vals.empty:
            continue
        row = {col: val for col, val in zip(group_cols, key)}
        row.update({
            "Count": int(len(vals)),
            "MeanSignedReturn": float(vals.mean()),
            "MedianSignedReturn": float(vals.median()),
            "StdSignedReturn": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
            "TStat": safe_t_stat(vals),
            "PositiveRowFraction": float((vals > 0).mean()),
            "ZeroRowFraction": float((vals == 0).mean()),
            "NegativeRowFraction": float((vals < 0).mean()),
            "MeanCondition": float(cvals.mean()) if len(cvals) else float("nan"),
            "MinCondition": float(cvals.min()) if len(cvals) else float("nan"),
            "MaxCondition": float(cvals.max()) if len(cvals) else float("nan"),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def load_selected(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, str]]:
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
    file_col = args.file_col or infer_column(cols, ["File", "SourceFile", "SessionFile"])
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

    selected = df.loc[mask].copy()
    selected[return_col] = pd.to_numeric(selected[return_col], errors="coerce")
    selected[condition_col] = pd.to_numeric(selected[condition_col], errors="coerce")
    selected = selected.dropna(subset=[return_col, condition_col]).reset_index(drop=True)

    if selected.empty:
        raise RuntimeError("Filter selected zero usable rows.")

    # qcut can fail when many duplicate values exist. Rank first to force stable quantiles,
    # but retain the original condition values for min/max reporting.
    ranks = selected[condition_col].rank(method="first")
    q = min(args.quantiles, len(selected))
    selected["ConditionQuantile"] = pd.qcut(ranks, q=q, labels=[f"Q{i+1}" for i in range(q)])

    mapping = {
        "instrument_col": instrument_col,
        "horizon_col": horizon_col,
        "pressure_col": pressure_col,
        "return_col": return_col,
        "condition_col": condition_col,
        "time_col": time_col or "",
        "file_col": file_col or "",
    }
    return selected, mapping


def build_scorecard(selected: pd.DataFrame, quantile_summary: pd.DataFrame, week_summary: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    rcol = mapping["return_col"]
    ccol = mapping["condition_col"]
    vals = selected[rcol]
    rows = [{
        "Metric": "global_count",
        "Value": float(len(selected)),
    }, {
        "Metric": "global_mean_signed_return",
        "Value": float(vals.mean()),
    }, {
        "Metric": "global_median_signed_return",
        "Value": float(vals.median()),
    }, {
        "Metric": "global_positive_row_fraction",
        "Value": float((vals > 0).mean()),
    }, {
        "Metric": "global_t_stat",
        "Value": safe_t_stat(vals),
    }, {
        "Metric": "condition_min",
        "Value": float(selected[ccol].min()),
    }, {
        "Metric": "condition_max",
        "Value": float(selected[ccol].max()),
    }]

    if not quantile_summary.empty:
        best = quantile_summary.sort_values(["MeanSignedReturn", "PositiveRowFraction"], ascending=False).iloc[0]
        rows.extend([
            {"Metric": "best_quantile", "Value": str(best["ConditionQuantile"])},
            {"Metric": "best_quantile_count", "Value": float(best["Count"])},
            {"Metric": "best_quantile_mean_signed_return", "Value": float(best["MeanSignedReturn"])},
            {"Metric": "best_quantile_positive_row_fraction", "Value": float(best["PositiveRowFraction"])},
            {"Metric": "best_quantile_t_stat", "Value": float(best["TStat"])},
        ])

    if not week_summary.empty:
        # Per quantile: fraction of weeks with positive expectancy.
        by_q = []
        for q, g in week_summary.groupby("ConditionQuantile", dropna=False):
            by_q.append({
                "ConditionQuantile": q,
                "WeekCount": len(g),
                "PositiveWeekFraction": float((g["MeanSignedReturn"] > 0).mean()),
                "MedianWeekMean": float(g["MeanSignedReturn"].median()),
                "MeanWeekMean": float(g["MeanSignedReturn"].mean()),
            })
        qweek = pd.DataFrame(by_q)
        bestw = qweek.sort_values(["PositiveWeekFraction", "MedianWeekMean", "MeanWeekMean"], ascending=False).iloc[0]
        rows.extend([
            {"Metric": "best_week_stability_quantile", "Value": str(bestw["ConditionQuantile"])},
            {"Metric": "best_week_stability_week_count", "Value": float(bestw["WeekCount"])},
            {"Metric": "best_week_stability_positive_week_fraction", "Value": float(bestw["PositiveWeekFraction"])},
            {"Metric": "best_week_stability_median_week_mean", "Value": float(bestw["MedianWeekMean"])},
            {"Metric": "best_week_stability_mean_week_mean", "Value": float(bestw["MeanWeekMean"])},
        ])

    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="APVA conditioned compression validation")
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/conditioned_compression_es_directional_presence")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--pressure", default="CompressionPressure")
    p.add_argument("--condition-col", default="RollingDirectionalPresence")
    p.add_argument("--quantiles", type=int, default=4)

    p.add_argument("--instrument-col", default=None)
    p.add_argument("--horizon-col", default=None)
    p.add_argument("--pressure-col", default=None)
    p.add_argument("--return-col", default=None)
    p.add_argument("--time-col", default=None)
    p.add_argument("--file-col", default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_known_args(argv)[0]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        selected, mapping = load_selected(args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    rcol = mapping["return_col"]
    ccol = mapping["condition_col"]
    time_col = mapping.get("time_col")
    file_col = mapping.get("file_col")

    selected.to_csv(outdir / "selected_conditioned_rows.csv", index=False)
    pd.DataFrame([mapping]).to_csv(outdir / "column_mapping.csv", index=False)

    quantile_summary = summarize(selected, ["ConditionQuantile"], rcol, ccol)
    quantile_summary.to_csv(outdir / "conditioned_quantile_summary.csv", index=False)

    if time_col and time_col in selected.columns:
        with_week = add_week_column(selected, time_col)
        week_summary = summarize(with_week, ["ConditionQuantile", "RegimeWeek"], rcol, ccol)
        week_summary.to_csv(outdir / "conditioned_week_summary.csv", index=False)
    else:
        week_summary = pd.DataFrame()
        week_summary.to_csv(outdir / "conditioned_week_summary.csv", index=False)

    if file_col and file_col in selected.columns:
        file_summary = summarize(selected, ["ConditionQuantile", file_col], rcol, ccol)
        file_summary.to_csv(outdir / "conditioned_file_summary.csv", index=False)

    score = build_scorecard(selected, quantile_summary, week_summary, mapping)
    score.to_csv(outdir / "conditioned_scorecard.csv", index=False)

    print("APVA conditioned compression validation complete")
    print(f"Selected rows: {len(selected)}")
    print(f"Condition column: {ccol}")
    print(f"Global mean: {selected[rcol].mean():.8f}")
    print(f"Output directory: {outdir}")
    print(quantile_summary.to_string(index=False))
    print(score.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
