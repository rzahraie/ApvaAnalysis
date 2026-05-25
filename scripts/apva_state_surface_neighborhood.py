#!/usr/bin/env python3
"""
APVA state-surface neighborhood validation.

Purpose
-------
Given a 2D APVA state surface, test whether strong cells are part of coherent
neighborhoods or isolated statistical spikes.

Primary input:

    outputs/state_surface_es_compression_entropy_x_directional_presence/state_surface_cells.csv

Core idea
---------
A real state-space region should show spatial coherence:

    strong cell + supportive neighbors = plausible topology
    strong cell + weak/opposite neighbors = likely spike/noise

Outputs
-------
- neighborhood_cells.csv
- neighborhood_scorecard.csv

Recommended Jupyter usage
-------------------------

    import runpy
    ns = runpy.run_path("scripts/apva_state_surface_neighborhood.py", run_name="not_main")
    ns["main"]([
        "--surface", "outputs/state_surface_es_compression_entropy_x_directional_presence/state_surface_cells.csv",
        "--outdir", "outputs/state_surface_neighborhood_es_compression_entropy_x_directional_presence",
    ])
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def parse_bin_index(label: object) -> Optional[int]:
    if pd.isna(label):
        return None
    m = re.search(r"(\d+)$", str(label))
    if not m:
        return None
    return int(m.group(1))


def safe_float(v: object) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def load_surface(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Surface file not found: {path}")
    df = pd.read_csv(path)
    required = ["XBin", "YBin", "SurfaceCell", "Count", "MeanSignedReturn", "TStat", "PositiveRowFraction"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns {missing}. Columns found: {list(df.columns)}")
    df = df.copy()
    df["XIndex"] = df["XBin"].map(parse_bin_index)
    df["YIndex"] = df["YBin"].map(parse_bin_index)
    df = df.dropna(subset=["XIndex", "YIndex"]).copy()
    df["XIndex"] = df["XIndex"].astype(int)
    df["YIndex"] = df["YIndex"].astype(int)
    return df


def neighbor_offsets(radius: int, include_diagonal: bool) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            if not include_diagonal and abs(dx) + abs(dy) != 1:
                continue
            if include_diagonal and max(abs(dx), abs(dy)) > radius:
                continue
            offsets.append((dx, dy))
    return offsets


def build_lookup(df: pd.DataFrame) -> dict[tuple[int, int], pd.Series]:
    return {(int(r.XIndex), int(r.YIndex)): r for r in df.itertuples(index=False)}


def analyze_neighborhood(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    lookup = build_lookup(df)
    offsets = neighbor_offsets(args.radius, args.include_diagonal)
    rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        xi = int(row["XIndex"])
        yi = int(row["YIndex"])
        neighbors = []
        for dx, dy in offsets:
            hit = lookup.get((xi + dx, yi + dy))
            if hit is not None:
                neighbors.append(hit)

        n_count = len(neighbors)
        means = np.array([safe_float(n.MeanSignedReturn) for n in neighbors], dtype=float)
        counts = np.array([safe_float(n.Count) for n in neighbors], dtype=float)
        tstats = np.array([safe_float(n.TStat) for n in neighbors], dtype=float)
        hit_rates = np.array([safe_float(n.PositiveRowFraction) for n in neighbors], dtype=float)

        valid = np.isfinite(means)
        if valid.any():
            neighbor_mean = float(np.nanmean(means))
            neighbor_median = float(np.nanmedian(means))
            neighbor_positive_fraction = float(np.nanmean(means > 0))
            if np.nansum(counts) > 0:
                weighted_neighbor_mean = float(np.nansum(means * counts) / np.nansum(counts))
            else:
                weighted_neighbor_mean = float("nan")
        else:
            neighbor_mean = neighbor_median = neighbor_positive_fraction = weighted_neighbor_mean = float("nan")

        cell_mean = safe_float(row["MeanSignedReturn"])
        cell_count = safe_float(row["Count"])
        cell_t = safe_float(row["TStat"])
        weekly_pos = safe_float(row.get("WeeklyPositiveFraction", float("nan")))
        weekly_median = safe_float(row.get("WeeklyMedianOfMeans", float("nan")))

        # Coherence score is intentionally simple and interpretable.
        # Positive when cell is strong and neighbors support it.
        coherence_score = (
            cell_mean
            + 0.75 * weighted_neighbor_mean
            + 0.50 * neighbor_median
            + 0.25 * cell_t
        )
        if np.isfinite(weekly_pos):
            coherence_score += weekly_pos
        if np.isfinite(weekly_median):
            coherence_score += 0.25 * weekly_median

        rows.append({
            "SurfaceCell": row["SurfaceCell"],
            "XBin": row["XBin"],
            "YBin": row["YBin"],
            "XIndex": xi,
            "YIndex": yi,
            "Count": cell_count,
            "MeanSignedReturn": cell_mean,
            "MedianSignedReturn": safe_float(row.get("MedianSignedReturn", float("nan"))),
            "TStat": cell_t,
            "PositiveRowFraction": safe_float(row["PositiveRowFraction"]),
            "WeeklyPositiveFraction": weekly_pos,
            "WeeklyMedianOfMeans": weekly_median,
            "NeighborCount": n_count,
            "NeighborMeanOfMeans": neighbor_mean,
            "NeighborWeightedMean": weighted_neighbor_mean,
            "NeighborMedianOfMeans": neighbor_median,
            "NeighborPositiveMeanFraction": neighbor_positive_fraction,
            "NeighborMeanTStat": float(np.nanmean(tstats)) if np.isfinite(tstats).any() else float("nan"),
            "NeighborMeanHitRate": float(np.nanmean(hit_rates)) if np.isfinite(hit_rates).any() else float("nan"),
            "IsolationPenalty": float(max(0.0, cell_mean - max(neighbor_mean, weighted_neighbor_mean, neighbor_median))),
            "CoherenceScore": coherence_score,
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["CoherenceScore", "MeanSignedReturn", "Count"], ascending=[False, False, False]).reset_index(drop=True)


def build_scorecard(neighborhood: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    best = neighborhood.iloc[0]
    strong = neighborhood.loc[
        (neighborhood["MeanSignedReturn"] > 0)
        & (neighborhood["NeighborWeightedMean"] > 0)
        & (neighborhood["NeighborPositiveMeanFraction"] >= args.min_neighbor_positive_fraction)
    ].copy()

    rows = [
        {"Metric": "cell_count", "Value": float(len(neighborhood))},
        {"Metric": "coherent_positive_cell_count", "Value": float(len(strong))},
        {"Metric": "coherent_positive_cell_fraction", "Value": float(len(strong) / len(neighborhood)) if len(neighborhood) else float("nan")},
        {"Metric": "best_cell", "Value": str(best["SurfaceCell"])},
        {"Metric": "best_x_bin", "Value": str(best["XBin"])},
        {"Metric": "best_y_bin", "Value": str(best["YBin"])},
        {"Metric": "best_count", "Value": float(best["Count"])},
        {"Metric": "best_mean_signed_return", "Value": float(best["MeanSignedReturn"])},
        {"Metric": "best_t_stat", "Value": float(best["TStat"])},
        {"Metric": "best_neighbor_weighted_mean", "Value": float(best["NeighborWeightedMean"])},
        {"Metric": "best_neighbor_median", "Value": float(best["NeighborMedianOfMeans"])},
        {"Metric": "best_neighbor_positive_mean_fraction", "Value": float(best["NeighborPositiveMeanFraction"])},
        {"Metric": "best_isolation_penalty", "Value": float(best["IsolationPenalty"])},
        {"Metric": "best_coherence_score", "Value": float(best["CoherenceScore"])},
    ]

    if not strong.empty:
        s = strong.iloc[0]
        rows.extend([
            {"Metric": "top_coherent_cell", "Value": str(s["SurfaceCell"])},
            {"Metric": "top_coherent_mean_signed_return", "Value": float(s["MeanSignedReturn"])},
            {"Metric": "top_coherent_neighbor_weighted_mean", "Value": float(s["NeighborWeightedMean"])},
            {"Metric": "top_coherent_neighbor_positive_mean_fraction", "Value": float(s["NeighborPositiveMeanFraction"])},
            {"Metric": "top_coherent_coherence_score", "Value": float(s["CoherenceScore"])},
        ])
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="APVA state-surface neighborhood validation")
    p.add_argument("--surface", default="outputs/state_surface_es_compression_entropy_x_directional_presence/state_surface_cells.csv")
    p.add_argument("--outdir", default="outputs/state_surface_neighborhood_es_compression_entropy_x_directional_presence")
    p.add_argument("--radius", type=int, default=1)
    p.add_argument("--include-diagonal", action="store_true", default=True)
    p.add_argument("--min-neighbor-positive-fraction", type=float, default=0.5)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_known_args(argv)[0]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        surface = load_surface(Path(args.surface))
        neighborhood = analyze_neighborhood(surface, args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if neighborhood.empty:
        print("ERROR: neighborhood result is empty", file=sys.stderr)
        return 2

    neighborhood.to_csv(outdir / "neighborhood_cells.csv", index=False)
    score = build_scorecard(neighborhood, args)
    score.to_csv(outdir / "neighborhood_scorecard.csv", index=False)

    print("APVA state-surface neighborhood validation complete")
    print(f"Surface cells: {len(surface)}")
    print(f"Output directory: {outdir}")
    print(score.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
