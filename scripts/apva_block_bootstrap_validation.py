#!/usr/bin/env python3
"""
APVA block bootstrap validation.

Purpose
-------
Validate the current best APVA candidate without assuming IID samples:

    ES / 5 bars / Stable Compression Basin + CompressionPressure

This script is intentionally defensive about input schemas. It is designed to run
against Jupyter/CSV research outputs that contain one row per event/bar/candidate
and at least one forward-return column.

Typical usage
-------------
From the ApvaAnalysis repo root:

    python scripts/apva_block_bootstrap_validation.py \
        --input . \
        --instrument ES \
        --horizon 5 \
        --archetype "Stable Compression Basin" \
        --pressure "CompressionPressure" \
        --outdir outputs/block_bootstrap_es5_compression

If auto-detection fails, specify column names explicitly:

    python scripts/apva_block_bootstrap_validation.py \
        --input path/to/your_expectancy_rows.csv \
        --instrument-col Instrument \
        --horizon-col Horizon \
        --archetype-col Archetype \
        --pressure-col PressureState \
        --return-col SignedForwardReturn \
        --instrument ES \
        --horizon 5 \
        --archetype "Stable Compression Basin" \
        --pressure "CompressionPressure"

Outputs
-------
- block_bootstrap_summary.csv
- block_bootstrap_distribution.csv
- selected_candidate_returns.csv
- input_diagnostics.csv

Notes
-----
This is validation infrastructure, not a trading system. It estimates uncertainty
under serial dependence by resampling contiguous return blocks and by stationary
bootstrap resampling. It does not model fills, slippage, commissions, or intrabar
execution.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_BLOCK_SIZES = (5, 10, 20, 50)
DEFAULT_N_BOOT = 10_000
DEFAULT_SEED = 20260524


@dataclass(frozen=True)
class ColumnMap:
    instrument: Optional[str]
    horizon: Optional[str]
    archetype: Optional[str]
    pressure: Optional[str]
    return_col: str
    direction: Optional[str]
    time: Optional[str]
    source_file: str


def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def infer_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    by_norm = {_norm(c): c for c in columns}
    for cand in candidates:
        hit = by_norm.get(_norm(cand))
        if hit is not None:
            return hit

    # substring fallback, longest candidate first to avoid weak accidental matches
    cols = list(columns)
    for cand in sorted(candidates, key=len, reverse=True):
        nc = _norm(cand)
        for col in cols:
            if nc and nc in _norm(col):
                return col
    return None


def discover_csvs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return sorted(p for p in input_path.rglob("*.csv") if p.is_file())


def load_candidate_frames(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    csvs = discover_csvs(Path(args.input))
    diagnostics: list[dict[str, object]] = []
    selected_frames: list[pd.DataFrame] = []

    if not csvs:
        raise RuntimeError(f"No CSV files found under {args.input!r}")

    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # noqa: BLE001 - diagnostics should keep scanning
            diagnostics.append({
                "file": str(csv_path),
                "status": "read_failed",
                "reason": str(exc),
                "rows": 0,
                "columns": "",
            })
            continue

        cols = list(df.columns)
        return_col = args.return_col or infer_column(cols, [
            "SignedForwardReturn",
            "SignedReturn",
            "ForwardSignedReturn",
            "ActiveDirectionReturn",
            "ForwardReturnSigned",
            "FwdSignedReturn",
            "MeanSignedReturn",
            "ReturnSigned",
            "Return",
            "ForwardReturn",
            "FwdReturn",
        ])

        instrument_col = args.instrument_col or infer_column(cols, ["Instrument", "Symbol", "Market"])
        horizon_col = args.horizon_col or infer_column(cols, ["Horizon", "ForwardBars", "LookaheadBars", "BarsForward", "NForwardBars", "TotalBars"])
        archetype_col = args.archetype_col or infer_column(cols, ["Archetype", "Topology", "State", "AuctionState", "StructuralState"])
        pressure_col = args.pressure_col or infer_column(cols, ["PressureState", "Pressure", "PressureClass", "CompressionPressure"])
        direction_col = args.direction_col or infer_column(cols, ["ActiveDirection", "Direction", "Dir", "TradeDirection"])
        time_col = args.time_col or infer_column(cols, ["Time", "Timestamp", "DateTime", "BarTime", "StartTime"])

        missing = []
        if return_col is None:
            missing.append("return_col")
        if args.instrument and instrument_col is None:
            missing.append("instrument_col")
        if args.horizon is not None and horizon_col is None:
            missing.append("horizon_col")
        if args.archetype and archetype_col is None:
            missing.append("archetype_col")
        if args.pressure and pressure_col is None:
            missing.append("pressure_col")

        if missing:
            diagnostics.append({
                "file": str(csv_path),
                "status": "skipped",
                "reason": "missing " + ", ".join(missing),
                "rows": len(df),
                "columns": ",".join(cols),
            })
            continue

        assert return_col is not None
        work = df.copy()
        mask = pd.Series(True, index=work.index)

        if args.instrument and instrument_col:
            mask &= work[instrument_col].astype(str).str.upper().eq(args.instrument.upper())
        if args.horizon is not None and horizon_col:
            h = pd.to_numeric(work[horizon_col], errors="coerce")
            mask &= h.eq(float(args.horizon))
        if args.archetype and archetype_col:
            mask &= work[archetype_col].astype(str).str.strip().eq(args.archetype)
        if args.pressure and pressure_col:
            mask &= work[pressure_col].astype(str).str.strip().eq(args.pressure)

        chosen = work.loc[mask].copy()
        chosen["__source_file"] = str(csv_path)
        chosen["__return_col"] = return_col

        diagnostics.append({
            "file": str(csv_path),
            "status": "selected" if len(chosen) else "no_matching_rows",
            "reason": "",
            "rows": len(df),
            "selected_rows": len(chosen),
            "return_col": return_col,
            "instrument_col": instrument_col or "",
            "horizon_col": horizon_col or "",
            "archetype_col": archetype_col or "",
            "pressure_col": pressure_col or "",
            "direction_col": direction_col or "",
            "time_col": time_col or "",
            "columns": ",".join(cols),
        })

        if len(chosen):
            # Normalize output columns for downstream bootstrap.
            normalized = pd.DataFrame({
                "return": pd.to_numeric(chosen[return_col], errors="coerce"),
                "source_file": str(csv_path),
            })
            if time_col:
                normalized["time"] = chosen[time_col].values
            if direction_col:
                normalized["direction"] = chosen[direction_col].values
            selected_frames.append(normalized)

    diagnostics_df = pd.DataFrame(diagnostics)
    if not selected_frames:
        raise RuntimeError(
            "No matching candidate rows found. Inspect input_diagnostics.csv after rerun with a writable outdir, "
            "or pass explicit --*-col arguments."
        )

    all_selected = pd.concat(selected_frames, ignore_index=True)
    all_selected = all_selected.dropna(subset=["return"]).reset_index(drop=True)
    if all_selected.empty:
        raise RuntimeError("Matching rows were found, but all selected returns were NaN/non-numeric.")
    return all_selected, diagnostics_df


def iid_bootstrap_mean(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    return x[idx].mean(axis=1)


def moving_block_bootstrap_mean(
    x: np.ndarray,
    block_size: int,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(x)
    b = max(1, min(block_size, n))
    if n == 1:
        return np.repeat(x[0], n_boot)

    starts = np.arange(0, n - b + 1)
    blocks_needed = math.ceil(n / b)
    out = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        chosen_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([x[s:s + b] for s in chosen_starts])[:n]
        out[i] = sample.mean()
    return out


def stationary_bootstrap_mean(
    x: np.ndarray,
    avg_block_size: int,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Politis-Romano style stationary bootstrap with circular indexing."""
    n = len(x)
    b = max(1, min(avg_block_size, n))
    p_new_block = 1.0 / b
    out = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(0, n)
        for t in range(1, n):
            if rng.random() < p_new_block:
                idx[t] = rng.integers(0, n)
            else:
                idx[t] = (idx[t - 1] + 1) % n
        out[i] = x[idx].mean()
    return out


def summarize_distribution(
    label: str,
    block_size: Optional[int],
    observed: np.ndarray,
    boot_means: np.ndarray,
) -> dict[str, object]:
    mu = float(np.mean(observed))
    sd = float(np.std(observed, ddof=1)) if len(observed) > 1 else 0.0
    t_stat = mu / (sd / math.sqrt(len(observed))) if sd > 0 and len(observed) > 1 else np.nan
    return {
        "method": label,
        "block_size": block_size if block_size is not None else "",
        "n": len(observed),
        "observed_mean": mu,
        "observed_median": float(np.median(observed)),
        "observed_std": sd,
        "observed_t_stat_naive": t_stat,
        "observed_positive_fraction": float(np.mean(observed > 0)),
        "boot_mean_mean": float(np.mean(boot_means)),
        "boot_mean_std": float(np.std(boot_means, ddof=1)),
        "ci_01": float(np.quantile(boot_means, 0.01)),
        "ci_05": float(np.quantile(boot_means, 0.05)),
        "ci_50": float(np.quantile(boot_means, 0.50)),
        "ci_95": float(np.quantile(boot_means, 0.95)),
        "ci_99": float(np.quantile(boot_means, 0.99)),
        "prob_mean_gt_0": float(np.mean(boot_means > 0)),
    }


def parse_block_sizes(s: str) -> list[int]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    if not vals:
        raise argparse.ArgumentTypeError("block size list cannot be empty")
    return vals


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="APVA block bootstrap validation")
    p.add_argument("--input", default=".", help="CSV file or directory to scan recursively")
    p.add_argument("--outdir", default="outputs/block_bootstrap", help="Output directory")

    p.add_argument("--instrument", default="ES")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--archetype", default="Stable Compression Basin")
    p.add_argument("--pressure", default="CompressionPressure")

    p.add_argument("--instrument-col", default=None)
    p.add_argument("--horizon-col", default=None)
    p.add_argument("--archetype-col", default=None)
    p.add_argument("--pressure-col", default=None)
    p.add_argument("--return-col", default=None)
    p.add_argument("--direction-col", default=None)
    p.add_argument("--time-col", default=None)

    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--block-sizes", type=parse_block_sizes, default=list(DEFAULT_BLOCK_SIZES))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_known_args(argv)[0]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        selected, diagnostics = load_candidate_frames(args)
    except Exception as exc:  # noqa: BLE001 - still write partial diagnostics if possible
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    diagnostics.to_csv(outdir / "input_diagnostics.csv", index=False)
    selected.to_csv(outdir / "selected_candidate_returns.csv", index=False)

    x = selected["return"].to_numpy(dtype=float)
    rng = np.random.default_rng(args.seed)

    summary_rows: list[dict[str, object]] = []
    distribution_frames: list[pd.DataFrame] = []

    iid = iid_bootstrap_mean(x, args.n_boot, rng)
    summary_rows.append(summarize_distribution("iid", None, x, iid))
    distribution_frames.append(pd.DataFrame({"method": "iid", "block_size": np.nan, "boot_mean": iid}))

    for b in args.block_sizes:
        mbb = moving_block_bootstrap_mean(x, b, args.n_boot, rng)
        summary_rows.append(summarize_distribution("moving_block", b, x, mbb))
        distribution_frames.append(pd.DataFrame({"method": "moving_block", "block_size": b, "boot_mean": mbb}))

        sb = stationary_bootstrap_mean(x, b, args.n_boot, rng)
        summary_rows.append(summarize_distribution("stationary", b, x, sb))
        distribution_frames.append(pd.DataFrame({"method": "stationary", "block_size": b, "boot_mean": sb}))

    summary = pd.DataFrame(summary_rows)
    dist = pd.concat(distribution_frames, ignore_index=True)

    summary.to_csv(outdir / "block_bootstrap_summary.csv", index=False)
    dist.to_csv(outdir / "block_bootstrap_distribution.csv", index=False)

    print("APVA block bootstrap complete")
    print(f"Selected rows: {len(selected)}")
    print(f"Observed mean: {np.mean(x):.8f}")
    print(f"Output directory: {outdir}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    main([])
