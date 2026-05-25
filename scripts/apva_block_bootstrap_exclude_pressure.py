#!/usr/bin/env python3
"""
APVA inverse-pressure block bootstrap validation.

This companion script tests whether a pressure edge is special by selecting rows
where DominantPressure/PressureState is NOT equal to the excluded pressure.

Primary use case:

    ES / 5-bar horizon / all pressures except CompressionPressure

Run from ApvaAnalysis repo root, or from Jupyter via:

    main([
        "--input", "tables",
        "--instrument", "ES",
        "--horizon", "5",
        "--exclude-pressure", "CompressionPressure",
        "--outdir", "outputs/block_bootstrap_es_ex_compression_pressure",
    ])
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_BLOCK_SIZES = (5, 10, 20, 50)
DEFAULT_N_BOOT = 10_000
DEFAULT_SEED = 20260524


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


def discover_csvs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return sorted(p for p in input_path.rglob("*.csv") if p.is_file())


def parse_block_sizes(s: str) -> list[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("block size list cannot be empty")
    return vals


def iid_bootstrap_mean(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    return x[idx].mean(axis=1)


def moving_block_bootstrap_mean(x: np.ndarray, block_size: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
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


def stationary_bootstrap_mean(x: np.ndarray, avg_block_size: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
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


def summarize_distribution(label: str, block_size: Optional[int], observed: np.ndarray, boot_means: np.ndarray) -> dict[str, object]:
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


def load_inverse_pressure_frames(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []

    for csv_path in discover_csvs(Path(args.input)):
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # noqa: BLE001
            diagnostics.append({"file": str(csv_path), "status": "read_failed", "reason": str(exc), "rows": 0})
            continue

        cols = list(df.columns)
        return_col = args.return_col or infer_column(cols, [
            "SignedForwardReturn", "SignedReturn", "ForwardSignedReturn", "ActiveDirectionReturn",
            "ForwardReturnSigned", "FwdSignedReturn", "MeanSignedReturn", "ReturnSigned",
            "Return", "ForwardReturn", "FwdReturn",
        ])
        instrument_col = args.instrument_col or infer_column(cols, ["Instrument", "Symbol", "Market"])
        horizon_col = args.horizon_col or infer_column(cols, ["HorizonBars", "Horizon", "ForwardBars", "LookaheadBars", "BarsForward", "NForwardBars", "TotalBars"])
        pressure_col = args.pressure_col or infer_column(cols, ["DominantPressure", "PressureState", "Pressure", "PressureClass"])
        archetype_col = args.archetype_col or infer_column(cols, ["ResolvedArchetype", "Archetype", "Topology", "State", "AuctionState", "StructuralState"])
        time_col = args.time_col or infer_column(cols, ["Time", "Timestamp", "DateTime", "BarTime", "StartTime"])

        missing = []
        if return_col is None:
            missing.append("return_col")
        if instrument_col is None:
            missing.append("instrument_col")
        if horizon_col is None:
            missing.append("horizon_col")
        if pressure_col is None:
            missing.append("pressure_col")
        if missing:
            diagnostics.append({
                "file": str(csv_path), "status": "skipped", "reason": "missing " + ", ".join(missing),
                "rows": len(df), "columns": ",".join(cols)
            })
            continue

        mask = pd.Series(True, index=df.index)
        mask &= df[instrument_col].astype(str).str.upper().eq(args.instrument.upper())
        mask &= pd.to_numeric(df[horizon_col], errors="coerce").eq(float(args.horizon))
        mask &= ~df[pressure_col].astype(str).str.strip().eq(args.exclude_pressure)

        if args.archetype is not None and str(args.archetype).strip() != "" and archetype_col:
            mask &= df[archetype_col].astype(str).str.strip().eq(args.archetype)

        chosen = df.loc[mask].copy()
        diagnostics.append({
            "file": str(csv_path), "status": "selected" if len(chosen) else "no_matching_rows",
            "reason": "", "rows": len(df), "selected_rows": len(chosen),
            "return_col": return_col, "instrument_col": instrument_col, "horizon_col": horizon_col,
            "pressure_col": pressure_col, "archetype_col": archetype_col or "", "time_col": time_col or "",
            "columns": ",".join(cols),
        })

        if len(chosen):
            normalized = pd.DataFrame({
                "return": pd.to_numeric(chosen[return_col], errors="coerce"),
                "pressure": chosen[pressure_col].astype(str).values,
                "source_file": str(csv_path),
            })
            if archetype_col:
                normalized["archetype"] = chosen[archetype_col].astype(str).values
            if time_col:
                normalized["time"] = chosen[time_col].values
            selected_frames.append(normalized)

    diagnostics_df = pd.DataFrame(diagnostics)
    if not selected_frames:
        raise RuntimeError("No inverse-pressure rows found. Check input_diagnostics.csv and explicit column args.")

    selected = pd.concat(selected_frames, ignore_index=True).dropna(subset=["return"]).reset_index(drop=True)
    if selected.empty:
        raise RuntimeError("Inverse-pressure rows found, but all selected returns are NaN/non-numeric.")
    return selected, diagnostics_df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="APVA inverse-pressure block bootstrap validation")
    p.add_argument("--input", default="tables")
    p.add_argument("--outdir", default="outputs/block_bootstrap_exclude_pressure")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--exclude-pressure", default="CompressionPressure")
    p.add_argument("--archetype", default="")
    p.add_argument("--instrument-col", default=None)
    p.add_argument("--horizon-col", default=None)
    p.add_argument("--pressure-col", default=None)
    p.add_argument("--archetype-col", default=None)
    p.add_argument("--return-col", default=None)
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
        selected, diagnostics = load_inverse_pressure_frames(args)
    except Exception as exc:  # noqa: BLE001
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

    pressure_counts = selected["pressure"].value_counts().rename_axis("pressure").reset_index(name="count")

    summary.to_csv(outdir / "block_bootstrap_summary.csv", index=False)
    dist.to_csv(outdir / "block_bootstrap_distribution.csv", index=False)
    pressure_counts.to_csv(outdir / "selected_pressure_counts.csv", index=False)

    print("APVA inverse-pressure block bootstrap complete")
    print(f"Excluded pressure: {args.exclude_pressure}")
    print(f"Selected rows: {len(selected)}")
    print(f"Observed mean: {np.mean(x):.8f}")
    print(f"Output directory: {outdir}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
