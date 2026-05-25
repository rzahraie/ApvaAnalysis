#!/usr/bin/env python3
"""
APVA feature leakage audit.

Purpose:
Audit whether candidate gating features are predictive-safe or likely leaking
future information.

Critical concern:
    DirectionalNormalizedMFE
    DirectionalNormalizedMAE
    DirectionalMFE
    DirectionalMAE

These usually sound like forward-excursion fields. If they are computed from
future bars, they must NOT be used as entry filters.

This script:
    1. Loads the row-level APVA dataset.
    2. Rebuilds the current index-futures entry set.
    3. Compares suspicious features against forward outcomes.
    4. Flags likely leakage fields.
    5. Produces safe/unsafe feature lists.

Outputs:
    leakage_audit_feature_summary.csv
    leakage_audit_scorecard.csv
    leakage_audit_entries.csv
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

INDEX_INSTRUMENTS = {"ES", "NQ"}

SUSPICIOUS_NAME_PARTS = [
    "future",
    "forward",
    "mfe",
    "mae",
    "return",
    "outcome",
    "profit",
    "loss",
    "target",
    "stop",
]

KNOWN_UNSAFE_IF_USED_AS_GATE = {
    "SignedReturn",
    "RawReturn",
    "NormalizedReturn",
    "SignedNormalizedReturn",
    "DirectionalMFE",
    "DirectionalMAE",
    "DirectionalNormalizedMFE",
    "DirectionalNormalizedMAE",
    "FutureClose",
    "ForwardReturn",
    "SignedForwardReturn",
}

PROBABLY_SAFE_FEATURES = {
    "RollingEntropy",
    "RollingDirectionalPresence",
    "RollingVelocity",
    "DominantPressure",
    "DominantPressureValue",
    "ResolvedArchetype",
    "MacroState",
    "Instrument",
    "File",
    "BarIndex",
    "Time",
    "HorizonBars",
}


def tstat(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    if len(x) < 2:
        return float("nan")
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / (sd / math.sqrt(len(x)))) if sd > 0 else float("nan")


def corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return float("nan")
    return float(x[ok].corr(y[ok]))


def summarize_numeric_feature(df: pd.DataFrame, feature: str, outcome: str) -> dict:
    x = pd.to_numeric(df[feature], errors="coerce")
    y = pd.to_numeric(df[outcome], errors="coerce")
    ok = x.notna() & y.notna()
    d = df.loc[ok].copy()

    if d.empty:
        return {
            "Feature": feature,
            "Type": "numeric",
            "Count": 0,
        }

    d["_feature"] = pd.to_numeric(d[feature], errors="coerce")
    d["_outcome"] = pd.to_numeric(d[outcome], errors="coerce")

    try:
        d["_q"] = pd.qcut(
            d["_feature"].rank(method="first"),
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
        )
        q = d.groupby("_q", observed=False)["_outcome"].mean()
        q_spread = float(q.max() - q.min())
        best_q_mean = float(q.max())
        worst_q_mean = float(q.min())
    except Exception:
        q_spread = float("nan")
        best_q_mean = float("nan")
        worst_q_mean = float("nan")

    c = corr(d["_feature"], d["_outcome"])

    lower_name = feature.lower()
    name_suspicious = any(part in lower_name for part in SUSPICIOUS_NAME_PARTS)
    known_unsafe = feature in KNOWN_UNSAFE_IF_USED_AS_GATE

    # Heuristic leakage suspicion:
    # - known forward/excursion/return column names are unsafe as entry gates.
    # - very high absolute correlation to outcome is suspicious.
    # - huge quantile spread is suspicious, especially with future-sounding names.
    leakage_score = 0
    reasons = []

    if known_unsafe:
        leakage_score += 5
        reasons.append("known_forward_or_outcome_field")

    if name_suspicious:
        leakage_score += 2
        reasons.append("suspicious_name")

    if np.isfinite(c) and abs(c) >= 0.50:
        leakage_score += 3
        reasons.append("high_outcome_correlation")

    if np.isfinite(q_spread) and q_spread > abs(d["_outcome"].mean()) * 3 and q_spread > 0.5:
        leakage_score += 2
        reasons.append("large_quantile_outcome_spread")

    if feature in PROBABLY_SAFE_FEATURES:
        leakage_score -= 2
        reasons.append("known_ex_ante_state_feature")

    if leakage_score >= 5:
        verdict = "UNSAFE_AS_ENTRY_GATE"
    elif leakage_score >= 3:
        verdict = "SUSPICIOUS_REVIEW_REQUIRED"
    else:
        verdict = "PROBABLY_SAFE_IF_COMPUTED_EX_ANTE"

    return {
        "Feature": feature,
        "Type": "numeric",
        "Count": int(len(d)),
        "Mean": float(d["_feature"].mean()),
        "Median": float(d["_feature"].median()),
        "Min": float(d["_feature"].min()),
        "Max": float(d["_feature"].max()),
        "OutcomeCorrelation": c,
        "OutcomeMean": float(d["_outcome"].mean()),
        "OutcomeTStat": tstat(d["_outcome"]),
        "QuantileOutcomeSpread": q_spread,
        "BestQuantileMean": best_q_mean,
        "WorstQuantileMean": worst_q_mean,
        "LeakageScore": leakage_score,
        "Verdict": verdict,
        "Reasons": "|".join(reasons),
    }


def summarize_categorical_feature(df: pd.DataFrame, feature: str, outcome: str) -> dict:
    y = pd.to_numeric(df[outcome], errors="coerce")
    d = df.loc[y.notna()].copy()
    d["_outcome"] = y.loc[y.notna()]

    if d.empty:
        return {
            "Feature": feature,
            "Type": "categorical",
            "Count": 0,
        }

    g = d.groupby(feature, dropna=False)["_outcome"].agg(["count", "mean", "median"])
    g = g[g["count"] >= 5]

    if g.empty:
        spread = float("nan")
        best = float("nan")
        worst = float("nan")
    else:
        spread = float(g["mean"].max() - g["mean"].min())
        best = float(g["mean"].max())
        worst = float(g["mean"].min())

    lower_name = feature.lower()
    name_suspicious = any(part in lower_name for part in SUSPICIOUS_NAME_PARTS)
    known_unsafe = feature in KNOWN_UNSAFE_IF_USED_AS_GATE

    leakage_score = 0
    reasons = []

    if known_unsafe:
        leakage_score += 5
        reasons.append("known_forward_or_outcome_field")

    if name_suspicious:
        leakage_score += 2
        reasons.append("suspicious_name")

    if np.isfinite(spread) and spread > abs(d["_outcome"].mean()) * 3 and spread > 0.5:
        leakage_score += 2
        reasons.append("large_category_outcome_spread")

    if feature in PROBABLY_SAFE_FEATURES:
        leakage_score -= 2
        reasons.append("known_ex_ante_state_feature")

    if leakage_score >= 5:
        verdict = "UNSAFE_AS_ENTRY_GATE"
    elif leakage_score >= 3:
        verdict = "SUSPICIOUS_REVIEW_REQUIRED"
    else:
        verdict = "PROBABLY_SAFE_IF_COMPUTED_EX_ANTE"

    return {
        "Feature": feature,
        "Type": "categorical",
        "Count": int(len(d)),
        "OutcomeMean": float(d["_outcome"].mean()),
        "OutcomeTStat": tstat(d["_outcome"]),
        "CategoryOutcomeSpread": spread,
        "BestCategoryMean": best,
        "WorstCategoryMean": worst,
        "LeakageScore": leakage_score,
        "Verdict": verdict,
        "Reasons": "|".join(reasons),
    }


def prepare_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    for c in df.columns:
        if c in {
            "BarIndex",
            "HorizonBars",
            "SignedReturn",
            "SignedNormalizedReturn",
            "RawReturn",
            "NormalizedReturn",
            "DirectionalMFE",
            "DirectionalMAE",
            "DirectionalNormalizedMFE",
            "DirectionalNormalizedMAE",
            "RollingEntropy",
            "RollingDirectionalPresence",
            "RollingVelocity",
            "DominantPressureValue",
        }:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    required = [
        "Instrument",
        "File",
        "BarIndex",
        "HorizonBars",
        "DominantPressure",
        "RollingEntropy",
        "RollingDirectionalPresence",
        "SignedNormalizedReturn",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    return df.sort_values(["Instrument", "File", "BarIndex"]).reset_index(drop=True)


def find_entries(df: pd.DataFrame, args) -> pd.DataFrame:
    state = (
        df["Instrument"].astype(str).str.upper().isin(INDEX_INSTRUMENTS)
        & df["HorizonBars"].eq(args.horizon)
        & df["DominantPressure"].astype(str).eq(args.pressure)
        & df["RollingEntropy"].between(args.entropy_min, args.entropy_max, inclusive="both")
        & df["RollingDirectionalPresence"].eq(args.directional_presence)
    )

    df["InTargetState"] = state
    prev = (
        df.groupby(["Instrument", "File"])["InTargetState"]
        .shift(1)
        .astype("boolean")
        .fillna(False)
    )
    df["TargetEntry"] = df["InTargetState"] & ~prev

    rows = []
    for idx in df.index[df["TargetEntry"]]:
        e = df.loc[idx]
        prior = df[
            df["Instrument"].eq(e["Instrument"])
            & df["File"].eq(e["File"])
            & (df["BarIndex"] < e["BarIndex"])
        ].tail(args.lookback)

        if len(prior) < args.lookback:
            continue

        if prior["DominantPressure"].astype(str).tolist() != TARGET_PATH:
            continue

        rows.append(e.copy())

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No entries found")

    return out.reset_index(drop=True)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    p.add_argument("--outdir", default="outputs/feature_leakage_audit")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--pressure", default="CompressionPressure")
    p.add_argument("--entropy-min", type=float, default=0.88)
    p.add_argument("--entropy-max", type=float, default=1.30)
    p.add_argument("--directional-presence", type=float, default=0.0)
    p.add_argument("--lookback", type=int, default=5)
    p.add_argument("--outcome-col", default="SignedNormalizedReturn")
    args = p.parse_known_args(argv)[0]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = prepare_df(args.input)
    entries = find_entries(df, args)

    if args.outcome_col not in entries.columns:
        raise RuntimeError(f"Outcome column not found: {args.outcome_col}")

    rows = []
    for col in entries.columns:
        if col.startswith("_"):
            continue
        if col == args.outcome_col:
            # Still audit it as unsafe, but avoid self-correlation weirdness as a gate.
            pass

        if pd.api.types.is_numeric_dtype(entries[col]):
            rows.append(summarize_numeric_feature(entries, col, args.outcome_col))
        else:
            rows.append(summarize_categorical_feature(entries, col, args.outcome_col))

    feature_summary = pd.DataFrame(rows).sort_values(
        ["LeakageScore", "OutcomeCorrelation"],
        ascending=[False, False],
        na_position="last",
    )

    unsafe = feature_summary[feature_summary["Verdict"].eq("UNSAFE_AS_ENTRY_GATE")]
    suspicious = feature_summary[feature_summary["Verdict"].eq("SUSPICIOUS_REVIEW_REQUIRED")]
    safe = feature_summary[feature_summary["Verdict"].eq("PROBABLY_SAFE_IF_COMPUTED_EX_ANTE")]

    scorecard = pd.DataFrame([
        {"Metric": "entry_count", "Value": float(len(entries))},
        {"Metric": "feature_count", "Value": float(len(feature_summary))},
        {"Metric": "unsafe_feature_count", "Value": float(len(unsafe))},
        {"Metric": "suspicious_feature_count", "Value": float(len(suspicious))},
        {"Metric": "probably_safe_feature_count", "Value": float(len(safe))},
        {
            "Metric": "unsafe_features",
            "Value": ",".join(unsafe["Feature"].astype(str).tolist()),
        },
        {
            "Metric": "suspicious_features",
            "Value": ",".join(suspicious["Feature"].astype(str).tolist()),
        },
        {
            "Metric": "probably_safe_core_features",
            "Value": ",".join(
                [f for f in safe["Feature"].astype(str).tolist() if f in PROBABLY_SAFE_FEATURES]
            ),
        },
    ])

    entries.to_csv(outdir / "leakage_audit_entries.csv", index=False)
    feature_summary.to_csv(outdir / "leakage_audit_feature_summary.csv", index=False)
    scorecard.to_csv(outdir / "leakage_audit_scorecard.csv", index=False)

    print("APVA feature leakage audit complete")
    print(scorecard.to_string(index=False))
    print()
    print(feature_summary.head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    main()