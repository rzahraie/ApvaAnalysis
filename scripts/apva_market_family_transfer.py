#!/usr/bin/env python3
"""
APVA market-family transfer analysis.

Purpose:
Split normalized transfer results by market family instead of forcing one
universal model.

Default families:
    IndexFutures: ES, NQ
    FXFutures:    6E

Input expected:
    outputs/normalized_cross_instrument_transfer/normalized_transfer_entries.csv

Outputs:
    market_family_summary.csv
    market_family_scorecard.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_FAMILIES = {
    "IndexFutures": {"ES", "NQ"},
    "FXFutures": {"6E"},
}


def tstat(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    if len(x) < 2:
        return float("nan")
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / (sd / math.sqrt(len(x)))) if sd > 0 else float("nan")


def summarize(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce").dropna()
    wins = x[x > 0]
    losses = x[x < 0]
    gw = wins.sum()
    gl = -losses.sum()

    return {
        "Count": int(len(x)),
        "Mean": float(x.mean()) if len(x) else np.nan,
        "Median": float(x.median()) if len(x) else np.nan,
        "TStat": tstat(x),
        "WinRate": float((x > 0).mean()) if len(x) else np.nan,
        "ProfitFactor": float(gw / gl) if gl > 0 else float("inf"),
        "Min": float(x.min()) if len(x) else np.nan,
        "Q25": float(x.quantile(0.25)) if len(x) else np.nan,
        "Q75": float(x.quantile(0.75)) if len(x) else np.nan,
        "Q90": float(x.quantile(0.90)) if len(x) else np.nan,
        "Max": float(x.max()) if len(x) else np.nan,
        "Sum": float(x.sum()) if len(x) else np.nan,
    }


def family_for_instrument(inst: str) -> str:
    u = str(inst).upper()
    for family, members in DEFAULT_FAMILIES.items():
        if u in members:
            return family
    return "Other"


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="outputs/normalized_cross_instrument_transfer/normalized_transfer_entries.csv",
    )
    p.add_argument(
        "--outdir",
        default="outputs/market_family_transfer",
    )
    p.add_argument(
        "--outcome-col",
        default="NormalizedPolicyOutcome",
    )
    args = p.parse_known_args(argv)[0]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    required = ["Instrument", args.outcome_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df[args.outcome_col] = pd.to_numeric(df[args.outcome_col], errors="coerce")
    df["MarketFamily"] = df["Instrument"].map(family_for_instrument)

    rows = []
    for family, g in df.groupby("MarketFamily", sort=True):
        row = {"MarketFamily": family}
        row.update({f"Policy_{k}": v for k, v in summarize(g[args.outcome_col]).items()})
        row["InstrumentCount"] = int(g["Instrument"].nunique())
        row["InstrumentList"] = ",".join(sorted(g["Instrument"].astype(str).unique()))
        rows.append(row)

    family_summary = pd.DataFrame(rows).sort_values("Policy_Mean", ascending=False)

    inst_rows = []
    for inst, g in df.groupby("Instrument", sort=True):
        row = {
            "Instrument": inst,
            "MarketFamily": family_for_instrument(inst),
        }
        row.update({f"Policy_{k}": v for k, v in summarize(g[args.outcome_col]).items()})
        inst_rows.append(row)

    instrument_summary = pd.DataFrame(inst_rows).sort_values("Policy_Mean", ascending=False)

    score_rows = [
        {"Metric": "entry_count", "Value": float(len(df))},
        {"Metric": "family_count", "Value": float(family_summary["MarketFamily"].nunique())},
        {
            "Metric": "positive_family_fraction",
            "Value": float((family_summary["Policy_Mean"] > 0).mean()),
        },
        {
            "Metric": "positive_instrument_fraction",
            "Value": float((instrument_summary["Policy_Mean"] > 0).mean()),
        },
    ]

    for _, r in family_summary.iterrows():
        fam = r["MarketFamily"]
        score_rows.extend([
            {"Metric": f"{fam}_count", "Value": float(r["Policy_Count"])},
            {"Metric": f"{fam}_mean", "Value": float(r["Policy_Mean"])},
            {"Metric": f"{fam}_median", "Value": float(r["Policy_Median"])},
            {"Metric": f"{fam}_tstat", "Value": float(r["Policy_TStat"])},
            {"Metric": f"{fam}_profit_factor", "Value": float(r["Policy_ProfitFactor"])},
            {"Metric": f"{fam}_instruments", "Value": r["InstrumentList"]},
        ])

    scorecard = pd.DataFrame(score_rows)

    df.to_csv(outdir / "market_family_entries.csv", index=False)
    family_summary.to_csv(outdir / "market_family_summary.csv", index=False)
    instrument_summary.to_csv(outdir / "market_family_instrument_summary.csv", index=False)
    scorecard.to_csv(outdir / "market_family_scorecard.csv", index=False)

    print("APVA market-family transfer complete")
    print(scorecard.to_string(index=False))
    print()
    print(family_summary.to_string(index=False))
    print()
    print(instrument_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()