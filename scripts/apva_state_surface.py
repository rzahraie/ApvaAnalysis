#!/usr/bin/env python3
"""
APVA 2D state surface.

Purpose
-------
Map where APVA forward asymmetry lives inside a two-variable state manifold.

Primary target:

    ES / 5-bar horizon / CompressionPressure
    X = RollingEntropy
    Y = RollingDirectionalPresence

This moves the research from univariate thresholds into interaction topology.

Recommended Jupyter usage
-------------------------

    import runpy
    ns = runpy.run_path("scripts/apva_state_surface.py", run_name="not_main")
    ns["main"]([
        "--input", "tables/apva_forward_signed_return_dataset_v1.csv",
        "--instrument", "ES",
        "--horizon", "5",
        "--pressure", "CompressionPressure",
        "--x-col", "RollingEntropy",
        "--y-col", "RollingDirectionalPresence",
        "--bins", "5",
        "--binning", "quantile",
        "--min-cell-count", "30",
        "--outdir", "outputs/state_surface_es_compression_entropy_x_directional_presence",
    ])

Outputs
-------
- state_surface_cells.csv
- state_surface_week_cells.csv
- state_surface_scorecard.csv
- selected_state_surface_rows.csv
- column_mapping.csv

Interpretation
--------------
Look for cells with:

- adequate support,
- positive mean signed return,
- acceptable t-stat,
- positive weekly regime fraction,
- positive median weekly return.

A real surface should show coherent neighborhoods, not isolated lucky cells.
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


def make_bins(series: pd.Series, bins: int, method: str, prefix: str) -> tuple[pd.Series, pd.DataFrame]:
    x = pd.to_numeric(series, errors="coerce")
    if method == "quantile":
        # Use ranks to avoid qcut failure from duplicate values while preserving quantile support.
        ranked = x.rank(method="first")
        labels = [f"{prefix}Q{i+1}" for i in range(bins)]
        binned = pd.qcut(ranked, q=bins, labels=labels)
    elif method == "uniform":
        labels = [f"{prefix}B{i+1}" for i in range(bins)]
        binned = pd.cut(x, bins=bins, labels=labels, include_lowest=True)
    else:
        raise ValueError(f"Unsupported binning method: {method}")

    meta_rows = []
    tmp = pd.DataFrame({"value": x, "bin": binned})
    for label, g in tmp.groupby("bin", dropna=False, observed=False):
        vals = g["value"].dropna()
        if vals.empty:
            continue
        meta_rows.append({
            "Bin": str(label),
            "Count": int(len(vals)),
            "Min": float(vals.min()),
            "Max": float(vals.max()),
            "Mean": float(vals.mean()),
            "Median": float(vals.median()),
        })
    return binned.astype(str), pd.DataFrame(meta_rows)


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
    x_col = args.x_col or infer_column(cols, ["RollingEntropy"])
    y_col = args.y_col or infer_column(cols, ["RollingDirectionalPresence"])

    required = {
        "instrument_col": instrument_col,
        "horizon_col": horizon_col,
        "pressure_col": pressure_col,
        "return_col": return_col,
        "x_col": x_col,
        "y_col": y_col,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required columns {missing}. Columns found: {cols}")

    assert instrument_col and horizon_col and pressure_col and return_col and x_col and y_col
    mask = pd.Series(True, index=df.index)
    mask &= df[instrument_col].astype(str).str.upper().eq(args.instrument.upper())
    mask &= pd.to_numeric(df[horizon_col], errors="coerce").eq(float(args.horizon))
    mask &= df[pressure_col].astype(str).str.strip().eq(args.pressure)

    selected = df.loc[mask].copy()
    for col in [return_col, x_col, y_col]:
        selected[col] = pd.to_numeric(selected[col], errors="coerce")
    selected = selected.dropna(subset=[return_col, x_col, y_col]).reset_index(drop=True)

    if selected.empty:
        raise RuntimeError("Filter selected zero usable rows.")

    selected["XBin"], x_meta = make_bins(selected[x_col], args.bins, args.binning, "X")
    selected["YBin"], y_meta = make_bins(selected[y_col], args.bins, args.binning, "Y")
    selected["SurfaceCell"] = selected["XBin"] + "|" + selected["YBin"]

    mapping = {
        "instrument_col": instrument_col,
        "horizon_col": horizon_col,
        "pressure_col": pressure_col,
        "return_col": return_col,
        "time_col": time_col or "",
        "x_col": x_col,
        "y_col": y_col,
        "binning": args.binning,
        "bins": str(args.bins),
    }
    return selected, mapping, x_meta, y_meta


def summarize_cell(g: pd.DataFrame, rcol: str, xcol: str, ycol: str) -> dict[str, float]:
    vals = pd.to_numeric(g[rcol], errors="coerce").dropna()
    return {
        "Count": int(len(vals)),
        "MeanSignedReturn": float(vals.mean()) if len(vals) else float("nan"),
        "MedianSignedReturn": float(vals.median()) if len(vals) else float("nan"),
        "StdSignedReturn": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
        "TStat": safe_t_stat(vals),
        "PositiveRowFraction": float((vals > 0).mean()) if len(vals) else float("nan"),
        "ZeroRowFraction": float((vals == 0).mean()) if len(vals) else float("nan"),
        "NegativeRowFraction": float((vals < 0).mean()) if len(vals) else float("nan"),
        "XMean": float(g[xcol].mean()),
        "XMin": float(g[xcol].min()),
        "XMax": float(g[xcol].max()),
        "YMean": float(g[ycol].mean()),
        "YMin": float(g[ycol].min()),
        "YMax": float(g[ycol].max()),
    }


def build_surface(selected: pd.DataFrame, mapping: dict[str, str], args: argparse.Namespace) -> pd.DataFrame:
    rcol = mapping["return_col"]
    xcol = mapping["x_col"]
    ycol = mapping["y_col"]
    rows = []
    for (xb, yb), g in selected.groupby(["XBin", "YBin"], dropna=False, sort=True, observed=False):
        if len(g) < args.min_cell_count:
            continue
        row = {"XBin": xb, "YBin": yb, "SurfaceCell": f"{xb}|{yb}"}
        row.update(summarize_cell(g, rcol, xcol, ycol))
        rows.append(row)
    return pd.DataFrame(rows)


def build_week_surface(selected: pd.DataFrame, mapping: dict[str, str], args: argparse.Namespace) -> pd.DataFrame:
    time_col = mapping.get("time_col")
    if not time_col or time_col not in selected.columns:
        return pd.DataFrame()

    rcol = mapping["return_col"]
    xcol = mapping["x_col"]
    ycol = mapping["y_col"]
    with_week = add_week_column(selected, time_col)
    rows = []
    for (xb, yb, week), g in with_week.groupby(["XBin", "YBin", "RegimeWeek"], dropna=False, sort=True, observed=False):
        if len(g) < args.min_week_cell_count:
            continue
        row = {"XBin": xb, "YBin": yb, "SurfaceCell": f"{xb}|{yb}", "RegimeWeek": week}
        row.update(summarize_cell(g, rcol, xcol, ycol))
        rows.append(row)
    return pd.DataFrame(rows)


def attach_week_stability(surface: pd.DataFrame, week_surface: pd.DataFrame) -> pd.DataFrame:
    if surface.empty or week_surface.empty:
        return surface
    rows = []
    for cell, g in week_surface.groupby("SurfaceCell", dropna=False, sort=True):
        rows.append({
            "SurfaceCell": cell,
            "WeekRegimeCount": int(len(g)),
            "WeeklyPositiveFraction": float((g["MeanSignedReturn"] > 0).mean()),
            "WeeklyMeanOfMeans": float(g["MeanSignedReturn"].mean()),
            "WeeklyMedianOfMeans": float(g["MeanSignedReturn"].median()),
            "WeeklyMinMean": float(g["MeanSignedReturn"].min()),
            "WeeklyMaxMean": float(g["MeanSignedReturn"].max()),
        })
    w = pd.DataFrame(rows)
    return surface.merge(w, on="SurfaceCell", how="left")


def build_scorecard(surface: pd.DataFrame, selected: pd.DataFrame, mapping: dict[str, str], args: argparse.Namespace) -> pd.DataFrame:
    rcol = mapping["return_col"]
    vals = selected[rcol]
    eligible = surface.copy()
    if "WeekRegimeCount" in eligible.columns:
        eligible = eligible.loc[eligible["WeekRegimeCount"].fillna(0) >= args.min_week_regimes].copy()
    if eligible.empty:
        eligible = surface.copy()

    sort_cols = []
    ascending = []
    for col in ["WeeklyPositiveFraction", "WeeklyMedianOfMeans", "MeanSignedReturn", "TStat", "Count"]:
        if col in eligible.columns:
            sort_cols.append(col)
            ascending.append(False)
    if sort_cols:
        best = eligible.sort_values(sort_cols, ascending=ascending).iloc[0]
    else:
        best = eligible.sort_values(["MeanSignedReturn", "TStat"], ascending=False).iloc[0]

    rows = [
        {"Metric": "base_count", "Value": float(len(selected))},
        {"Metric": "base_mean_signed_return", "Value": float(vals.mean())},
        {"Metric": "base_median_signed_return", "Value": float(vals.median())},
        {"Metric": "base_t_stat", "Value": safe_t_stat(vals)},
        {"Metric": "surface_cell_count", "Value": float(len(surface))},
        {"Metric": "x_col", "Value": mapping["x_col"]},
        {"Metric": "y_col", "Value": mapping["y_col"]},
        {"Metric": "best_cell", "Value": str(best["SurfaceCell"])},
        {"Metric": "best_x_bin", "Value": str(best["XBin"])},
        {"Metric": "best_y_bin", "Value": str(best["YBin"])},
        {"Metric": "best_count", "Value": float(best["Count"])},
        {"Metric": "best_mean_signed_return", "Value": float(best["MeanSignedReturn"])},
        {"Metric": "best_median_signed_return", "Value": float(best["MedianSignedReturn"])},
        {"Metric": "best_t_stat", "Value": float(best["TStat"])},
        {"Metric": "best_positive_row_fraction", "Value": float(best["PositiveRowFraction"])},
        {"Metric": "best_x_mean", "Value": float(best["XMean"])},
        {"Metric": "best_y_mean", "Value": float(best["YMean"])},
    ]
    for col in ["WeekRegimeCount", "WeeklyPositiveFraction", "WeeklyMeanOfMeans", "WeeklyMedianOfMeans", "WeeklyMinMean", "WeeklyMaxMean"]:
        if col in best.index and pd.notna(best[col]):
            rows.append({"Metric": f"best_{col}", "Value": float(best[col])})
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="APVA 2D state surface")
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/state_surface_es_compression_entropy_x_directional_presence")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--pressure", default="CompressionPressure")
    p.add_argument("--x-col", default="RollingEntropy")
    p.add_argument("--y-col", default="RollingDirectionalPresence")
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--binning", choices=["quantile", "uniform"], default="quantile")
    p.add_argument("--min-cell-count", type=int, default=30)
    p.add_argument("--min-week-cell-count", type=int, default=5)
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
        selected, mapping, x_meta, y_meta = load_selected(args)
        surface = build_surface(selected, mapping, args)
        week_surface = build_week_surface(selected, mapping, args)
        surface = attach_week_stability(surface, week_surface)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if surface.empty:
        print("ERROR: state surface is empty; lower --min-cell-count or check filters", file=sys.stderr)
        return 2

    selected.to_csv(outdir / "selected_state_surface_rows.csv", index=False)
    pd.DataFrame([mapping]).to_csv(outdir / "column_mapping.csv", index=False)
    x_meta.to_csv(outdir / "x_bin_metadata.csv", index=False)
    y_meta.to_csv(outdir / "y_bin_metadata.csv", index=False)
    surface.to_csv(outdir / "state_surface_cells.csv", index=False)
    week_surface.to_csv(outdir / "state_surface_week_cells.csv", index=False)

    score = build_scorecard(surface, selected, mapping, args)
    score.to_csv(outdir / "state_surface_scorecard.csv", index=False)

    print("APVA state surface complete")
    print(f"Base rows: {len(selected)}")
    print(f"Surface cells: {len(surface)}")
    print(f"X: {mapping['x_col']} | Y: {mapping['y_col']}")
    print(f"Output directory: {outdir}")
    print(score.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
