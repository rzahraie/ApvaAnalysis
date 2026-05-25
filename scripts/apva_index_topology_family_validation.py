#!/usr/bin/env python3
"""Validate candidate ES/NQ APVA compression-entry topology families.

Predictor construction is restricted to the entry row and prior rows. Outcome
fields are used only after entries and candidate gates have been identified.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from apva_analysis_utils import (
    INDEX_INSTRUMENTS,
    LEAKY_OUTCOME_FIELDS,
    SAFE_ENTRY_FIELDS,
    apply_normalized_policy,
    bin_quantile,
    group_summary,
    mark_target_entries,
    prepare_df,
    summarize,
    week_key,
)


NUMERIC_FEATURES = [
    "PriorCompressionCount",
    "PriorCompressionFraction",
    "PriorCompressionStreak",
    "PriorRotationCount",
    "PriorRotationFraction",
    "PriorLast_RollingDirectionalPresence",
    "PriorMean_RollingDirectionalPresence",
    "PriorSlope_RollingDirectionalPresence",
    "PriorLast_RollingEntropy",
    "PriorSlope_RollingEntropy",
    "PriorLast_DominantPressureValue",
    "PriorSlope_DominantPressureValue",
]

CATEGORICAL_FEATURES = ["PressureTriplet", "PriorPressureSeq"]
QUARTILE_LABELS = ["Q1", "Q2", "Q3", "Q4"]
PREDICTOR_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def slope(vals: pd.Series) -> float:
    x = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(float)
    return float(x[-1] - x[0]) if len(x) >= 2 else np.nan


def compression_streak(pressures: list[str]) -> int:
    count = 0
    for pressure in reversed(pressures):
        if pressure != "CompressionPressure":
            break
        count += 1
    return count


def build_entries(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    marked = mark_target_entries(
        df,
        instruments=INDEX_INSTRUMENTS,
        horizon=args.horizon,
        pressure=args.pressure,
        entropy_min=args.entropy_min,
        entropy_max=args.entropy_max,
        directional_presence=args.directional_presence,
    )
    rows = []
    for idx in marked.index[marked["TargetEntry"]]:
        entry = marked.loc[idx]
        prior = marked[
            marked["Instrument"].eq(entry["Instrument"])
            & marked["File"].eq(entry["File"])
            & (marked["BarIndex"] < entry["BarIndex"])
        ].tail(args.lookback)
        if len(prior) < args.lookback:
            continue
        pressures = prior["DominantPressure"].astype(str).tolist()
        row = entry.copy()
        row["PriorPressureSeq"] = " > ".join(pressures)
        # The triplet ends at entry; no next-state information is admitted as a gate.
        row["PressureTriplet"] = " > ".join(pressures[-2:] + [str(entry["DominantPressure"])])
        row["PriorCompressionCount"] = pressures.count("CompressionPressure")
        row["PriorCompressionFraction"] = row["PriorCompressionCount"] / args.lookback
        row["PriorCompressionStreak"] = compression_streak(pressures)
        row["PriorRotationCount"] = pressures.count("RotationalPressure")
        row["PriorRotationFraction"] = row["PriorRotationCount"] / args.lookback
        for col in [
            "RollingDirectionalPresence",
            "RollingEntropy",
            "DominantPressureValue",
        ]:
            vals = pd.to_numeric(prior[col], errors="coerce")
            row[f"PriorLast_{col}"] = float(vals.iloc[-1])
            row[f"PriorMean_{col}"] = float(vals.mean())
            row[f"PriorSlope_{col}"] = slope(vals)
        rows.append(row)
    if not rows:
        raise RuntimeError("No target entries with complete prior history found")
    entries = pd.DataFrame(rows).reset_index(drop=True)
    entries = apply_normalized_policy(entries, args.disaster_stop)
    entries["RegimeWeek"] = week_key(entries["Time"]) if "Time" in entries else "UnknownWeek"
    entries["PriorLastDirPresenceQuartile"] = bin_quantile(
        entries["PriorLast_RollingDirectionalPresence"], QUARTILE_LABELS
    )
    return entries


def stability_stats(entries: pd.DataFrame) -> dict[str, float | int]:
    by_inst = group_summary(entries, ["Instrument"])
    by_week = group_summary(entries, ["RegimeWeek"])
    by_inst_week = group_summary(entries, ["Instrument", "RegimeWeek"])
    return {
        "ES_Count": int((entries["Instrument"].astype(str).str.upper() == "ES").sum()),
        "NQ_Count": int((entries["Instrument"].astype(str).str.upper() == "NQ").sum()),
        "InstrumentPositiveFraction": (
            float((by_inst["Mean"] > 0).mean()) if not by_inst.empty else np.nan
        ),
        "WeekPositiveFraction": (
            float((by_week["Mean"] > 0).mean()) if not by_week.empty else np.nan
        ),
        "InstrumentWeekPositiveFraction": (
            float((by_inst_week["Mean"] > 0).mean()) if not by_inst_week.empty else np.nan
        ),
        "WeekCount": int(len(by_week)),
        "InstrumentWeekCount": int(len(by_inst_week)),
    }


def candidate_row(
    entries: pd.DataFrame,
    *,
    feature: str,
    gate_type: str,
    gate: str,
    family: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "Family": family,
        "Feature": feature,
        "GateType": gate_type,
        "Gate": gate,
    }
    row.update(summarize(entries["NormalizedPolicyOutcome"]))
    row.update(stability_stats(entries))
    row["StopRate"] = float(entries["DisasterStopped"].mean())
    return row


def qualify_candidates(
    summary: pd.DataFrame, base_stats: dict[str, float | int], min_count: int
) -> pd.DataFrame:
    out = summary.copy()
    if out.empty:
        return out
    out["MeetsMinCount"] = out["Count"] >= min_count
    out["PreferredCount"] = out["Count"] >= 100
    out["BothInstruments"] = (out["ES_Count"] > 0) & (out["NQ_Count"] > 0)
    out["MedianImproves"] = out["Median"] > float(base_stats["Median"])
    out["PFImproves"] = out["ProfitFactor"] > float(base_stats["ProfitFactor"])
    out["InstrumentStabilityOK"] = (
        out["InstrumentPositiveFraction"]
        >= float(base_stats["InstrumentPositiveFraction"]) - 0.10
    )
    out["WeekStabilityOK"] = (
        out["WeekPositiveFraction"] >= float(base_stats["WeekPositiveFraction"]) - 0.10
    )
    out["InstrumentWeekStabilityOK"] = (
        out["InstrumentWeekPositiveFraction"]
        >= float(base_stats["InstrumentWeekPositiveFraction"]) - 0.10
    )
    out["EligibleCandidate"] = (
        out["MeetsMinCount"]
        & out["BothInstruments"]
        & out["MedianImproves"]
        & out["PFImproves"]
        & out["InstrumentStabilityOK"]
        & out["WeekStabilityOK"]
        & out["InstrumentWeekStabilityOK"]
    )
    return out.sort_values(
        ["EligibleCandidate", "PreferredCount", "Median", "ProfitFactor", "Mean", "TStat", "Count"],
        ascending=False,
    ).reset_index(drop=True)


def build_1d_summary(
    entries: pd.DataFrame, base_stats: dict[str, float | int], min_count: int
) -> pd.DataFrame:
    rows = []
    for feature in NUMERIC_FEATURES:
        quartile_col = f"GateQuartile_{feature}"
        entries[quartile_col] = bin_quantile(entries[feature], QUARTILE_LABELS)
        for gate, group in entries.groupby(quartile_col, dropna=False, sort=True):
            row = candidate_row(
                group, feature=feature, gate_type="quartile", gate=str(gate), family="numeric"
            )
            vals = pd.to_numeric(group[feature], errors="coerce")
            row["FeatureMin"] = float(vals.min())
            row["FeatureMax"] = float(vals.max())
            row["FeatureMean"] = float(vals.mean())
            rows.append(row)
    for feature in CATEGORICAL_FEATURES:
        for gate, group in entries.groupby(feature, dropna=False, sort=True):
            rows.append(candidate_row(
                group, feature=feature, gate_type="category", gate=str(gate), family="grammar"
            ))
    return qualify_candidates(pd.DataFrame(rows), base_stats, min_count)


def build_2d_summary(
    entries: pd.DataFrame, base_stats: dict[str, float | int], min_count: int
) -> pd.DataFrame:
    definitions = [
        ("PriorCompressionStreak", "compression_streak_x_prior_dir_presence"),
        ("PriorCompressionFraction", "compression_fraction_x_prior_dir_presence"),
        ("PressureTriplet", "pressure_triplet_x_prior_dir_presence"),
    ]
    rows = []
    for feature, family in definitions:
        for (gate, presence_q), group in entries.groupby(
            [feature, "PriorLastDirPresenceQuartile"], dropna=False, sort=True
        ):
            row = candidate_row(
                group,
                feature=f"{feature} x PriorLast_RollingDirectionalPresence",
                gate_type="interaction",
                gate=f"{gate} x {presence_q}",
                family=family,
            )
            row["PrimaryGate"] = gate
            row["PriorLastDirPresenceQuartile"] = presence_q
            vals = pd.to_numeric(group["PriorLast_RollingDirectionalPresence"], errors="coerce")
            row["PriorLastDirPresenceMin"] = float(vals.min())
            row["PriorLastDirPresenceMax"] = float(vals.max())
            rows.append(row)
    return qualify_candidates(pd.DataFrame(rows), base_stats, min_count)


def select_best(summary: pd.DataFrame) -> pd.Series | None:
    if summary.empty:
        return None
    eligible = summary.loc[summary["EligibleCandidate"]]
    return eligible.iloc[0] if not eligible.empty else summary.iloc[0]


def gated_entries(entries: pd.DataFrame, selected: pd.Series | None, dimensions: int) -> pd.DataFrame:
    if selected is None:
        return entries.iloc[0:0].copy()
    feature = str(selected["Feature"])
    gate = str(selected["Gate"])
    if dimensions == 1:
        if selected["GateType"] == "quartile":
            return entries.loc[entries[f"GateQuartile_{feature}"].astype(str).eq(gate)].copy()
        return entries.loc[entries[feature].astype(str).eq(gate)].copy()
    base_feature = feature.split(" x ", maxsplit=1)[0]
    return entries.loc[
        entries[base_feature].astype(str).eq(str(selected["PrimaryGate"]))
        & entries["PriorLastDirPresenceQuartile"].astype(str).eq(
            str(selected["PriorLastDirPresenceQuartile"])
        )
    ].copy()


def add_selection_summary(
    frames: list[pd.DataFrame], subset: pd.DataFrame, label: str, grouping: list[str]
) -> None:
    if subset.empty:
        return
    out = group_summary(subset, grouping)
    out.insert(0, "Selection", label)
    frames.append(out)


def build_scorecard(
    entries: pd.DataFrame,
    base_stats: dict[str, float | int],
    summary_1d: pd.DataFrame,
    summary_2d: pd.DataFrame,
    min_count: int,
) -> pd.DataFrame:
    rows = [
        {"Metric": "base_count", "Value": base_stats["Count"]},
        {"Metric": "base_mean", "Value": base_stats["Mean"]},
        {"Metric": "base_median", "Value": base_stats["Median"]},
        {"Metric": "base_tstat", "Value": base_stats["TStat"]},
        {"Metric": "base_profit_factor", "Value": base_stats["ProfitFactor"]},
        {"Metric": "base_win_rate", "Value": base_stats["WinRate"]},
        {"Metric": "base_instrument_positive_fraction", "Value": base_stats["InstrumentPositiveFraction"]},
        {"Metric": "base_week_positive_fraction", "Value": base_stats["WeekPositiveFraction"]},
        {"Metric": "base_instrument_week_positive_fraction", "Value": base_stats["InstrumentWeekPositiveFraction"]},
        {"Metric": "minimum_candidate_count", "Value": min_count},
        {"Metric": "preferred_candidate_count", "Value": 100},
        {"Metric": "predictor_fields_excluded_as_leaky", "Value": ",".join(LEAKY_OUTCOME_FIELDS)},
        {"Metric": "safe_entry_fields", "Value": ",".join(SAFE_ENTRY_FIELDS)},
        {"Metric": "pressure_triplet_definition", "Value": "prior[-2] > prior[-1] > entry"},
    ]
    for label, summary in [("best_1d", summary_1d), ("best_2d", summary_2d)]:
        selected = select_best(summary)
        if selected is None:
            rows.append({"Metric": f"{label}_status", "Value": "no_candidates"})
            continue
        rows.append({
            "Metric": f"{label}_status",
            "Value": "eligible_candidate" if bool(selected["EligibleCandidate"]) else "top_observed_not_eligible",
        })
        for key in [
            "Feature", "GateType", "Gate", "Count", "Mean", "Median", "TStat", "ProfitFactor",
            "WinRate", "ES_Count", "NQ_Count", "InstrumentPositiveFraction",
            "WeekPositiveFraction", "InstrumentWeekPositiveFraction",
        ]:
            rows.append({"Metric": f"{label}_{key.lower()}", "Value": selected[key]})
    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="tables/apva_forward_signed_return_dataset_v1.csv")
    parser.add_argument("--outdir", default="outputs/index_topology_family_validation")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--pressure", default="CompressionPressure")
    parser.add_argument("--entropy-min", type=float, default=0.88)
    parser.add_argument("--entropy-max", type=float, default=1.30)
    parser.add_argument("--directional-presence", type=float, default=0.0)
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--disaster-stop", type=float, default=2.0)
    parser.add_argument("--min-count", type=int, default=50)
    args = parser.parse_known_args(argv)[0]
    leaked_predictors = set(PREDICTOR_FEATURES).intersection(LEAKY_OUTCOME_FIELDS)
    if leaked_predictors:
        raise RuntimeError(f"Leaky predictor fields configured: {sorted(leaked_predictors)}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    entries = build_entries(prepare_df(args.input), args)
    base_stats = summarize(entries["NormalizedPolicyOutcome"])
    base_stats.update(stability_stats(entries))
    summary_1d = build_1d_summary(entries, base_stats, args.min_count)
    summary_2d = build_2d_summary(entries, base_stats, args.min_count)
    best_1d = gated_entries(entries, select_best(summary_1d), dimensions=1)
    best_2d = gated_entries(entries, select_best(summary_2d), dimensions=2)
    by_instrument: list[pd.DataFrame] = []
    by_week: list[pd.DataFrame] = []
    for label, subset in [("base", entries), ("best_1d", best_1d), ("best_2d", best_2d)]:
        add_selection_summary(by_instrument, subset, label, ["Instrument"])
        add_selection_summary(by_week, subset, label, ["RegimeWeek"])
    scorecard = build_scorecard(entries, base_stats, summary_1d, summary_2d, args.min_count)
    entries.to_csv(outdir / "topology_family_entries.csv", index=False)
    summary_1d.to_csv(outdir / "topology_family_1d_summary.csv", index=False)
    summary_2d.to_csv(outdir / "topology_family_2d_summary.csv", index=False)
    pd.concat(by_instrument, ignore_index=True).to_csv(
        outdir / "topology_family_by_instrument.csv", index=False
    )
    pd.concat(by_week, ignore_index=True).to_csv(
        outdir / "topology_family_by_week.csv", index=False
    )
    scorecard.to_csv(outdir / "topology_family_scorecard.csv", index=False)
    print("APVA index topology family validation complete")
    print(scorecard.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
