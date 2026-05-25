#!/usr/bin/env python3
"""
APVA regime stability validation.

Purpose
-------
Test whether an APVA edge is persistent across time/file regimes rather than
being produced by one lucky cluster.

Primary target:

    ES / 5-bar horizon / CompressionPressure

Recommended Jupyter usage
-------------------------
Use runpy so the script does not auto-run accidentally:

    import runpy
    ns = runpy.run_path("scripts/apva_regime_stability_validation.py", run_name="not_main")
    ns["main"]([
        "--input", "tables/apva_forward_signed_return_dataset_v1.csv",
        "--instrument", "ES",
        "--horizon", "5",
        "--pressure", "CompressionPressure",
        "--outdir", "outputs/regime_stability_es_compression_pressure",
    ])

Outputs
-------
- regime_file_summary.csv
- regime_week_summary.csv
- regime_session_summary.csv, if SessionContext exists
- rolling_edge_summary.csv
- stability_scorecard.csv
- selected_candidate_rows.csv

Interpretation
--------------
The key metric is not only the global mean. The key question is whether the
edge appears across many independent regimes:

    fraction_positive_file_regimes
    fraction_positive_week_regimes

This is validation infrastructure, not a trading system.
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


def safe_t_stat(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(vals)
    if n < 2:
        return float("nan")
    sd = float(np.std(vals, ddof=1))
    if sd <= 0:
        return float("nan")
    return float(np.mean(vals) / (sd / math.sqrt(n)))


def summarize_group(df: pd.DataFrame, group_cols: list[str], return_col: str) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(group_cols, dropna=False, sort=True)
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        vals = pd.to_numeric(g[return_col], errors="coerce").dropna()
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
            "MeanDirectionalMFE": float(pd.to_numeric(g.get("DirectionalMFE", pd.Series(dtype=float)), errors="coerce").mean()) if "DirectionalMFE" in g else float("nan"),
            "MeanDirectionalMAE": float(pd.to_numeric(g.get("DirectionalMAE", pd.Series(dtype=float)), errors="coerce").mean()) if "DirectionalMAE" in g else float("nan"),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def add_week_column(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[time_col], errors="coerce")
    # ISO year-week avoids ambiguous week numbering at year boundaries.
    iso = dt.dt.isocalendar()
    out["RegimeWeek"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    out.loc[dt.isna(), "RegimeWeek"] = "UnknownWeek"
    return out


def rolling_summary(df: pd.DataFrame, return_col: str, time_col: Optional[str], windows: list[int]) -> pd.DataFrame:
    work = df.copy()
    if time_col and time_col in work.columns:
        work["__time"] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.sort_values(["__time"]).reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)

    vals = pd.to_numeric(work[return_col], errors="coerce")
    rows = []
    for w in windows:
        if w <= 0:
            continue
        roll_mean = vals.rolling(w, min_periods=max(3, min(w, 5))).mean()
        roll_hit = (vals > 0).astype(float).rolling(w, min_periods=max(3, min(w, 5))).mean()
        valid_mean = roll_mean.dropna()
        valid_hit = roll_hit.dropna()
        rows.append({
            "Window": w,
            "RollingPointCount": int(len(valid_mean)),
            "MeanOfRollingMean": float(valid_mean.mean()) if len(valid_mean) else float("nan"),
            "MedianOfRollingMean": float(valid_mean.median()) if len(valid_mean) else float("nan"),
            "MinRollingMean": float(valid_mean.min()) if len(valid_mean) else float("nan"),
            "MaxRollingMean": float(valid_mean.max()) if len(valid_mean) else float("nan"),
            "FractionRollingMeanPositive": float((valid_mean > 0).mean()) if len(valid_mean) else float("nan"),
            "MeanRollingHitRate": float(valid_hit.mean()) if len(valid_hit) else float("nan"),
        })
    return pd.DataFrame(rows)


def load_and_filter(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, str]]:
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    cols = list(df.columns)

    instrument_col = args.instrument_col or infer_column(cols, ["Instrument", "Symbol", "Market"])
    horizon_col = args.horizon_col or infer_column(cols, ["HorizonBars", "Horizon", "ForwardBars", "LookaheadBars", "BarsForward", "NForwardBars"])
    pressure_col = args.pressure_col or infer_column(cols, ["DominantPressure", "PressureState", "Pressure", "PressureClass"])
    return_col = args.return_col or infer_column(cols, ["SignedReturn", "SignedNormalizedReturn", "MeanSignedReturn", "ForwardSignedReturn", "SignedForwardReturn"])
    archetype_col = args.archetype_col or infer_column(cols, ["ResolvedArchetype", "Archetype", "Topology", "AuctionState", "StructuralState"])
    file_col = args.file_col or infer_column(cols, ["File", "SourceFile", "SessionFile"])
    time_col = args.time_col or infer_column(cols, ["Time", "Timestamp", "DateTime", "BarTime", "StartTime"])
    session_col = args.session_col or infer_column(cols, ["SessionContext", "Session", "TradingSession"])

    required = {
        "instrument_col": instrument_col,
        "horizon_col": horizon_col,
        "pressure_col": pressure_col,
        "return_col": return_col,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required columns {missing}. Columns found: {cols}")

    assert instrument_col and horizon_col and pressure_col and return_col
    mask = pd.Series(True, index=df.index)
    mask &= df[instrument_col].astype(str).str.upper().eq(args.instrument.upper())
    mask &= pd.to_numeric(df[horizon_col], errors="coerce").eq(float(args.horizon))
    mask &= df[pressure_col].astype(str).str.strip().eq(args.pressure)

    if args.archetype and archetype_col:
        mask &= df[archetype_col].astype(str).str.strip().eq(args.archetype)

    selected = df.loc[mask].copy()
    selected[return_col] = pd.to_numeric(selected[return_col], errors="coerce")
    selected = selected.dropna(subset=[return_col]).reset_index(drop=True)

    mapping = {
        "instrument_col": instrument_col,
        "horizon_col": horizon_col,
        "pressure_col": pressure_col,
        "return_col": return_col,
        "archetype_col": archetype_col or "",
        "file_col": file_col or "",
        "time_col": time_col or "",
        "session_col": session_col or "",
    }
    return selected, mapping


def scorecard(selected: pd.DataFrame, mapping: dict[str, str], summaries: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rcol = mapping["return_col"]
    vals = selected[rcol].dropna()
    rows = [{
        "Metric": "global_count",
        "Value": float(len(vals)),
    }, {
        "Metric": "global_mean_signed_return",
        "Value": float(vals.mean()) if len(vals) else float("nan"),
    }, {
        "Metric": "global_median_signed_return",
        "Value": float(vals.median()) if len(vals) else float("nan"),
    }, {
        "Metric": "global_positive_row_fraction",
        "Value": float((vals > 0).mean()) if len(vals) else float("nan"),
    }, {
        "Metric": "global_t_stat",
        "Value": safe_t_stat(vals),
    }]

    for name, df in summaries.items():
        if df is None or df.empty:
            continue
        positive = df["MeanSignedReturn"] > 0
        rows.extend([
            {"Metric": f"{name}_regime_count", "Value": float(len(df))},
            {"Metric": f"{name}_positive_regime_fraction", "Value": float(positive.mean())},
            {"Metric": f"{name}_mean_of_regime_means", "Value": float(df["MeanSignedReturn"].mean())},
            {"Metric": f"{name}_median_of_regime_means", "Value": float(df["MeanSignedReturn"].median())},
            {"Metric": f"{name}_min_regime_mean", "Value": float(df["MeanSignedReturn"].min())},
            {"Metric": f"{name}_max_regime_mean", "Value": float(df["MeanSignedReturn"].max())},
        ])

    return pd.DataFrame(rows)


def parse_windows(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="APVA regime stability validation")
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/regime_stability_es_compression_pressure")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--pressure", default="CompressionPressure")
    p.add_argument("--archetype", default="")

    p.add_argument("--instrument-col", default=None)
    p.add_argument("--horizon-col", default=None)
    p.add_argument("--pressure-col", default=None)
    p.add_argument("--return-col", default=None)
    p.add_argument("--archetype-col", default=None)
    p.add_argument("--file-col", default=None)
    p.add_argument("--time-col", default=None)
    p.add_argument("--session-col", default=None)
    p.add_argument("--rolling-windows", type=parse_windows, default=[20, 50, 100])
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_known_args(argv)[0]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        selected, mapping = load_and_filter(args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if selected.empty:
        print("ERROR: filter selected zero rows", file=sys.stderr)
        return 2

    rcol = mapping["return_col"]
    selected.to_csv(outdir / "selected_candidate_rows.csv", index=False)
    pd.DataFrame([mapping]).to_csv(outdir / "column_mapping.csv", index=False)

    summaries: dict[str, pd.DataFrame] = {}

    file_col = mapping.get("file_col")
    if file_col and file_col in selected.columns:
        file_summary = summarize_group(selected, [file_col], rcol)
        file_summary.to_csv(outdir / "regime_file_summary.csv", index=False)
        summaries["file"] = file_summary

    time_col = mapping.get("time_col")
    if time_col and time_col in selected.columns:
        with_week = add_week_column(selected, time_col)
        week_summary = summarize_group(with_week, ["RegimeWeek"], rcol)
        week_summary.to_csv(outdir / "regime_week_summary.csv", index=False)
        summaries["week"] = week_summary

    session_col = mapping.get("session_col")
    if session_col and session_col in selected.columns:
        session_summary = summarize_group(selected, [session_col], rcol)
        session_summary.to_csv(outdir / "regime_session_summary.csv", index=False)
        summaries["session"] = session_summary

    roll = rolling_summary(selected, rcol, time_col if time_col else None, args.rolling_windows)
    roll.to_csv(outdir / "rolling_edge_summary.csv", index=False)

    card = scorecard(selected, mapping, summaries)
    card.to_csv(outdir / "stability_scorecard.csv", index=False)

    print("APVA regime stability validation complete")
    print(f"Selected rows: {len(selected)}")
    print(f"Observed mean: {selected[rcol].mean():.8f}")
    print(f"Observed hit fraction: {(selected[rcol] > 0).mean():.4f}")
    print(f"Output directory: {outdir}")
    print(card.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
