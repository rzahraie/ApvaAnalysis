#!/usr/bin/env python3
"""Shared leakage-safe utilities for APVA validation scripts."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd


INDEX_INSTRUMENTS = {"ES", "NQ"}

LEAKY_OUTCOME_FIELDS = [
    "SignedReturn",
    "RawReturn",
    "NormalizedReturn",
    "SignedNormalizedReturn",
    "FutureClose",
    "DirectionalMFE",
    "DirectionalMAE",
    "DirectionalNormalizedMFE",
    "DirectionalNormalizedMAE",
    "DirectionalHit",
]

SAFE_ENTRY_FIELDS = [
    "RollingEntropy",
    "RollingDirectionalPresence",
    "RollingVelocity",
    "DominantPressureValue",
    "DominantPressure",
    "ResolvedArchetype",
    "MacroState",
    "Instrument",
    "File",
    "Time",
    "BarIndex",
    "HorizonBars",
]

COMMON_NUMERIC_FIELDS = [
    "BarIndex",
    "HorizonBars",
    "RollingEntropy",
    "RollingDirectionalPresence",
    "RollingVelocity",
    "DominantPressureValue",
    "SignedReturn",
    "RawReturn",
    "NormalizedReturn",
    "SignedNormalizedReturn",
    "DirectionalMFE",
    "DirectionalMAE",
    "DirectionalNormalizedMFE",
    "DirectionalNormalizedMAE",
]

REQUIRED_NORMALIZED_ENTRY_FIELDS = [
    "Instrument",
    "File",
    "BarIndex",
    "HorizonBars",
    "DominantPressure",
    "RollingEntropy",
    "RollingDirectionalPresence",
    "SignedNormalizedReturn",
    "DirectionalNormalizedMAE",
]


def tstat(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    if len(x) < 2:
        return float("nan")
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / (sd / math.sqrt(len(x)))) if sd > 0 else float("nan")


def summarize(s: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    wins = x[x > 0]
    losses = x[x < 0]
    gross_loss = -losses.sum()
    return {
        "Count": int(len(x)),
        "Mean": float(x.mean()) if len(x) else np.nan,
        "Median": float(x.median()) if len(x) else np.nan,
        "TStat": tstat(x),
        "WinRate": float((x > 0).mean()) if len(x) else np.nan,
        "ProfitFactor": float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf"),
        "Min": float(x.min()) if len(x) else np.nan,
        "Q25": float(x.quantile(0.25)) if len(x) else np.nan,
        "Q75": float(x.quantile(0.75)) if len(x) else np.nan,
        "Q90": float(x.quantile(0.90)) if len(x) else np.nan,
        "Max": float(x.max()) if len(x) else np.nan,
        "Sum": float(x.sum()) if len(x) else np.nan,
    }


def week_key(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    iso = dt.dt.isocalendar()
    out = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    out.loc[dt.isna()] = "UnknownWeek"
    return out


def bin_quantile(series: pd.Series, labels: list[str]) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series("NA", index=series.index, dtype="object")
    valid = x.notna()
    if valid.sum() < len(labels):
        return out
    try:
        bins = pd.qcut(x.loc[valid], q=len(labels), labels=False, duplicates="drop")
        if bins.notna().any():
            out.loc[valid] = bins.map(lambda code: labels[int(code)] if pd.notna(code) else "NA")
        else:
            out.loc[valid] = "ALL_TIED"
    except ValueError:
        out.loc[valid] = "BIN_FAIL"
    return out


def prepare_df(
    path: str,
    required: Iterable[str] | None = None,
    numeric: Iterable[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in numeric or COMMON_NUMERIC_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    required_fields = list(required or REQUIRED_NORMALIZED_ENTRY_FIELDS)
    missing = [col for col in required_fields if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    return df.sort_values(["Instrument", "File", "BarIndex"]).reset_index(drop=True)


def apply_normalized_policy(entries: pd.DataFrame, disaster_stop: float | None) -> pd.DataFrame:
    out = entries.copy()
    if disaster_stop is None:
        out["NormalizedPolicyOutcome"] = out["SignedNormalizedReturn"]
        out["DisasterStopped"] = False
    else:
        stopped = out["DirectionalNormalizedMAE"] <= -abs(disaster_stop)
        out["NormalizedPolicyOutcome"] = out["SignedNormalizedReturn"].where(
            ~stopped,
            -abs(disaster_stop),
        )
        out["DisasterStopped"] = stopped
    out["PositiveOutcome"] = out["NormalizedPolicyOutcome"] > 0
    return out


def target_state_mask(
    df: pd.DataFrame,
    *,
    instruments: set[str] = INDEX_INSTRUMENTS,
    horizon: int = 5,
    pressure: str = "CompressionPressure",
    entropy_min: float = 0.88,
    entropy_max: float = 1.30,
    directional_presence: float = 0.0,
) -> pd.Series:
    return (
        df["Instrument"].astype(str).str.upper().isin(instruments)
        & df["HorizonBars"].eq(horizon)
        & df["DominantPressure"].astype(str).eq(pressure)
        & df["RollingEntropy"].between(entropy_min, entropy_max, inclusive="both")
        & df["RollingDirectionalPresence"].eq(directional_presence)
    )


def mark_target_entries(df: pd.DataFrame, **state_kwargs: object) -> pd.DataFrame:
    out = df.copy()
    out["InTargetState"] = target_state_mask(out, **state_kwargs)
    prev = (
        out.groupby(["Instrument", "File"])["InTargetState"]
        .shift(1)
        .astype("boolean")
        .fillna(False)
    )
    out["TargetEntry"] = out["InTargetState"] & ~prev
    return out


def group_summary(
    entries: pd.DataFrame,
    group_cols: list[str],
    *,
    outcome_col: str = "NormalizedPolicyOutcome",
    min_count: int = 1,
) -> pd.DataFrame:
    rows = []
    for key, group in entries.groupby(group_cols, dropna=False, sort=True):
        if len(group) < min_count:
            continue
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        row.update(summarize(group[outcome_col]))
        if "DisasterStopped" in group.columns:
            row["StopRate"] = float(group["DisasterStopped"].mean())
        rows.append(row)
    return pd.DataFrame(rows)
