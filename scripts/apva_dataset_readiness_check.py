#!/usr/bin/env python3
"""Check new CSV datasets for fixed-rule APVA validation readiness.

This is an ingestion/readiness tool. It reports compatibility, frozen
candidate coverage, and overlap with the current reference; it does not
search for candidates or evaluate candidate performance.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from apva_analysis_utils import INDEX_INSTRUMENTS, LEAKY_OUTCOME_FIELDS, mark_target_entries
from apva_topology_walkforward_validation import (
    CCRRR,
    RRCCC,
    PRIOR_SLOPE_Q3_MAX,
    PRIOR_SLOPE_Q3_MIN,
)


REQUIRED_COLUMNS = [
    "Instrument",
    "File",
    "BarIndex",
    "HorizonBars",
    "DominantPressure",
    "RollingDirectionalPresence",
    "RollingEntropy",
    "SignedNormalizedReturn",
    "DirectionalNormalizedMAE",
]
OPTIONAL_COLUMNS = ["Time", "DominantPressureValue"]
ROW_KEY_COLUMNS = ["Instrument", "File", "BarIndex"]
TIME_KEY_COLUMNS = ["Instrument", "File", "Time"]
DUPLICATE_KEY_COLUMNS = ["Instrument", "File", "BarIndex", "HorizonBars"]
FROZEN_CANDIDATES = [
    "Base_ES_NQ",
    "RRCCC",
    "CCRRR",
    "PriorSlope_DominantPressureValue_Q3",
]
PREDICTOR_FIELDS = [
    "Instrument",
    "File",
    "BarIndex",
    "HorizonBars",
    "DominantPressure",
    "RollingDirectionalPresence",
    "RollingEntropy",
    "DominantPressureValue",
    "PriorPressureSeq",
    "PriorSlope_DominantPressureValue",
]


def expand_patterns(patterns: list[str], workspace: Path) -> list[Path]:
    paths: set[Path] = set()
    for raw in patterns:
        path = Path(raw)
        pattern = str(path if path.is_absolute() else workspace / path)
        matches = [Path(match).resolve() for match in glob.glob(pattern, recursive=True)]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern).resolve()]
        paths.update(match for match in matches if match.suffix.lower() == ".csv")
    return sorted(paths)


def display_path(path: Path, workspace: Path) -> str:
    try:
        return path.relative_to(workspace).as_posix()
    except ValueError:
        return str(path)


def numeric_coercion(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "BarIndex",
        "HorizonBars",
        "RollingDirectionalPresence",
        "RollingEntropy",
        "SignedNormalizedReturn",
        "DirectionalNormalizedMAE",
        "DominantPressureValue",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def slope(vals: pd.Series) -> float:
    x = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(float)
    return float(x[-1] - x[0]) if len(x) >= 2 else np.nan


def build_candidate_entries(df: pd.DataFrame, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    marked = mark_target_entries(
        df.sort_values(["Instrument", "File", "BarIndex"]).reset_index(drop=True),
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
        row = entry.copy()
        row["PriorPressureSeq"] = " > ".join(prior["DominantPressure"].astype(str).tolist())
        if "DominantPressureValue" in prior:
            row["PriorSlope_DominantPressureValue"] = slope(prior["DominantPressureValue"])
        rows.append(row)
    base = pd.DataFrame(rows)
    if base.empty:
        base = marked.iloc[0:0].copy()
        base["PriorPressureSeq"] = pd.Series(dtype="object")
        if "DominantPressureValue" in df:
            base["PriorSlope_DominantPressureValue"] = pd.Series(dtype="float64")
    candidates = {
        "Base_ES_NQ": base,
        "RRCCC": base.loc[base["PriorPressureSeq"].eq(RRCCC)].copy(),
        "CCRRR": base.loc[base["PriorPressureSeq"].eq(CCRRR)].copy(),
    }
    if "DominantPressureValue" in df.columns and "PriorSlope_DominantPressureValue" in base:
        candidates["PriorSlope_DominantPressureValue_Q3"] = base.loc[
            base["PriorSlope_DominantPressureValue"].between(
                PRIOR_SLOPE_Q3_MIN, PRIOR_SLOPE_Q3_MAX, inclusive="both"
            )
        ].copy()
    else:
        candidates["PriorSlope_DominantPressureValue_Q3"] = base.iloc[0:0].copy()
    return candidates


def key_overlap(
    df: pd.DataFrame, reference: pd.DataFrame, columns: list[str]
) -> tuple[int, int, float]:
    if not set(columns).issubset(df.columns) or not set(columns).issubset(reference.columns):
        return 0, 0, np.nan
    source_keys = set(map(tuple, df[columns].astype(str).to_numpy().tolist()))
    reference_keys = set(map(tuple, reference[columns].astype(str).to_numpy().tolist()))
    overlap = len(source_keys & reference_keys)
    return len(source_keys), overlap, overlap / len(source_keys) if source_keys else np.nan


def classify_independence(overlap_fraction: float, is_derived_output: bool) -> str:
    if is_derived_output:
        return "ExcludedDerivedOutput"
    if pd.isna(overlap_fraction):
        return "NeedsReviewNoComparableKeys"
    if overlap_fraction > 0.95:
        return "NonIndependent"
    if overlap_fraction < 0.20:
        return "LikelyIndependent"
    return "PartialOverlapNeedsReview"


def readiness_for_file(
    path: Path,
    reference: pd.DataFrame,
    intended_extensions: set[Path],
    workspace: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    dataset = display_path(path, workspace)
    derived_output = dataset.startswith("outputs/")
    intended = path in intended_extensions
    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        summary = {
            "Path": dataset,
            "Location": dataset.split("/", maxsplit=1)[0],
            "RowCount": np.nan,
            "RequiredColumnsPresent": False,
            "MissingRequiredColumns": ",".join(REQUIRED_COLUMNS),
            "OptionalTimePresent": False,
            "OptionalDominantPressureValuePresent": False,
            "InstrumentList": "",
            "ES_RowCount": 0,
            "NQ_RowCount": 0,
            "TimeMin": "",
            "TimeMax": "",
            "BarIndexMin": np.nan,
            "BarIndexMax": np.nan,
            "HorizonBarsValues": "",
            "DominantPressureValues": "",
            "SignedNormalizedReturnExists": False,
            "DirectionalNormalizedMAEExists": False,
            "BaseCandidateRowsPresent": False,
            "CCRRRCandidateRowsPresent": False,
            "RRCCCCandidateRowsPresent": False,
            "PriorSlopeQ3CandidateRowsPresent": False,
            "AnyNonBaseFrozenCandidateRowsPresent": False,
            "AnyCandidateBothESAndNQNonzero": False,
            "DuplicateRowKeyCount": np.nan,
            "DerivedOutputExcluded": derived_output,
            "IntendedExtensionOverride": intended,
            "IndependenceStatus": "ExcludedDerivedOutput" if derived_output else "NotEvaluated",
            "ReadinessStatus": "FAIL_READ_ERROR",
            "ReadinessReason": str(exc),
        }
        candidate_rows = [{
            "Path": dataset,
            "Candidate": candidate,
            "AvailableForCheck": False,
            "Count": 0,
            "ES_Count": 0,
            "NQ_Count": 0,
            "BothESAndNQNonzero": False,
        } for candidate in FROZEN_CANDIDATES]
        overlap = {
            "Path": dataset,
            "ReferencePath": display_path(Path(args.reference).resolve(), workspace),
            "RowKeyColumns": ",".join(ROW_KEY_COLUMNS),
            "UniqueRowKeyCount": np.nan,
            "OverlappingRowKeyCount": np.nan,
            "RowKeyOverlapFraction": np.nan,
            "TimeKeyColumns": "",
            "UniqueTimeKeyCount": np.nan,
            "OverlappingTimeKeyCount": np.nan,
            "TimeKeyOverlapFraction": np.nan,
            "MaximumComparableOverlapFraction": np.nan,
            "IndependenceStatus": summary["IndependenceStatus"],
            "IntendedExtensionOverride": intended,
            "DerivedOutputExcluded": derived_output,
        }
        return summary, candidate_rows, overlap
    columns = set(raw.columns)
    missing = [col for col in REQUIRED_COLUMNS if col not in columns]
    df = numeric_coercion(raw)
    duplicate_row_key_count = (
        int(df.duplicated(DUPLICATE_KEY_COLUMNS).sum())
        if set(DUPLICATE_KEY_COLUMNS).issubset(columns) else np.nan
    )
    row_unique, row_overlap, row_fraction = key_overlap(df, reference, ROW_KEY_COLUMNS)
    time_unique, time_overlap, time_fraction = key_overlap(df, reference, TIME_KEY_COLUMNS)
    fractions = [fraction for fraction in [row_fraction, time_fraction] if pd.notna(fraction)]
    overlap_fraction = max(fractions) if fractions else np.nan
    independence = classify_independence(overlap_fraction, derived_output)
    candidate_tables: dict[str, pd.DataFrame] = {}
    if not missing:
        candidate_tables = build_candidate_entries(df, args)
    candidate_rows = []
    for candidate in FROZEN_CANDIDATES:
        selected = candidate_tables.get(candidate, pd.DataFrame())
        es_count = (
            int(selected["Instrument"].astype(str).str.upper().eq("ES").sum())
            if not selected.empty else 0
        )
        nq_count = (
            int(selected["Instrument"].astype(str).str.upper().eq("NQ").sum())
            if not selected.empty else 0
        )
        available = not (
            candidate == "PriorSlope_DominantPressureValue_Q3"
            and "DominantPressureValue" not in columns
        )
        candidate_rows.append({
            "Path": dataset,
            "Candidate": candidate,
            "AvailableForCheck": available,
            "Count": int(len(selected)),
            "ES_Count": es_count,
            "NQ_Count": nq_count,
            "BothESAndNQNonzero": bool(es_count > 0 and nq_count > 0),
        })
    counts = {row["Candidate"]: int(row["Count"]) for row in candidate_rows}
    frozen_non_base_count = sum(counts[name] for name in FROZEN_CANDIDATES if name != "Base_ES_NQ")
    instruments = sorted(df["Instrument"].dropna().astype(str).unique().tolist()) if "Instrument" in df else []
    instrument_upper = df["Instrument"].astype(str).str.upper() if "Instrument" in df else pd.Series(dtype="object")
    time = pd.to_datetime(df["Time"], errors="coerce") if "Time" in df else pd.Series(dtype="datetime64[ns]")
    schema_pass = not missing
    instrument_pass = bool(instrument_upper.isin(INDEX_INSTRUMENTS).any())
    base_pass = counts.get("Base_ES_NQ", 0) > 0
    candidate_pass = frozen_non_base_count > 0
    overlap_pass = independence == "LikelyIndependent" or intended
    duplicate_pass = bool(pd.isna(duplicate_row_key_count) or duplicate_row_key_count == 0)
    derived_pass = not derived_output
    ready = all([
        schema_pass, instrument_pass, base_pass, candidate_pass, overlap_pass,
        duplicate_pass, derived_pass,
    ])
    failures = []
    for passed, reason in [
        (schema_pass, "missing required columns"),
        (instrument_pass, "no ES or NQ rows"),
        (base_pass, "no base candidate rows"),
        (candidate_pass, "no non-base frozen candidate rows"),
        (overlap_pass, "not independent of reference"),
        (duplicate_pass, "duplicate row keys present"),
        (derived_pass, "prior output artifact excluded from readiness"),
    ]:
        if not passed:
            failures.append(reason)
    summary = {
        "Path": dataset,
        "Location": dataset.split("/", maxsplit=1)[0],
        "RowCount": int(len(df)),
        "RequiredColumnsPresent": schema_pass,
        "MissingRequiredColumns": ",".join(missing),
        "OptionalTimePresent": "Time" in columns,
        "OptionalDominantPressureValuePresent": "DominantPressureValue" in columns,
        "InstrumentList": ",".join(instruments),
        "ES_RowCount": int(instrument_upper.eq("ES").sum()),
        "NQ_RowCount": int(instrument_upper.eq("NQ").sum()),
        "TimeMin": str(time.min()) if not time.empty and time.notna().any() else "",
        "TimeMax": str(time.max()) if not time.empty and time.notna().any() else "",
        "BarIndexMin": float(df["BarIndex"].min()) if "BarIndex" in df else np.nan,
        "BarIndexMax": float(df["BarIndex"].max()) if "BarIndex" in df else np.nan,
        "HorizonBarsValues": ",".join(map(str, sorted(df["HorizonBars"].dropna().unique()))) if "HorizonBars" in df else "",
        "DominantPressureValues": ",".join(sorted(df["DominantPressure"].dropna().astype(str).unique())) if "DominantPressure" in df else "",
        "SignedNormalizedReturnExists": "SignedNormalizedReturn" in columns,
        "DirectionalNormalizedMAEExists": "DirectionalNormalizedMAE" in columns,
        "BaseCandidateRowsPresent": base_pass,
        "CCRRRCandidateRowsPresent": counts.get("CCRRR", 0) > 0,
        "RRCCCCandidateRowsPresent": counts.get("RRCCC", 0) > 0,
        "PriorSlopeQ3CandidateRowsPresent": counts.get("PriorSlope_DominantPressureValue_Q3", 0) > 0,
        "AnyNonBaseFrozenCandidateRowsPresent": candidate_pass,
        "AnyCandidateBothESAndNQNonzero": any(
            row["BothESAndNQNonzero"] for row in candidate_rows if row["Candidate"] != "Base_ES_NQ"
        ),
        "DuplicateRowKeyCount": duplicate_row_key_count,
        "DerivedOutputExcluded": derived_output,
        "IntendedExtensionOverride": intended,
        "IndependenceStatus": independence,
        "ReadinessStatus": "READY" if ready else "NOT_READY",
        "ReadinessReason": "ready for fixed-rule validation" if ready else "; ".join(failures),
    }
    overlap = {
        "Path": dataset,
        "ReferencePath": display_path(Path(args.reference).resolve(), workspace),
        "RowKeyColumns": ",".join(ROW_KEY_COLUMNS),
        "UniqueRowKeyCount": row_unique,
        "OverlappingRowKeyCount": row_overlap,
        "RowKeyOverlapFraction": row_fraction,
        "TimeKeyColumns": ",".join(TIME_KEY_COLUMNS) if "Time" in columns else "",
        "UniqueTimeKeyCount": time_unique,
        "OverlappingTimeKeyCount": time_overlap,
        "TimeKeyOverlapFraction": time_fraction,
        "MaximumComparableOverlapFraction": overlap_fraction,
        "IndependenceStatus": independence,
        "IntendedExtensionOverride": intended,
        "DerivedOutputExcluded": derived_output,
    }
    return summary, candidate_rows, overlap


def build_scorecard(
    summary: pd.DataFrame,
    candidate_counts: pd.DataFrame,
    reference_path: str,
    inputs: list[str],
) -> pd.DataFrame:
    required_columns_present = summary["RequiredColumnsPresent"].fillna(False).astype(bool)
    derived_output_excluded = summary["DerivedOutputExcluded"].fillna(False).astype(bool)
    ready = summary.loc[summary["ReadinessStatus"].eq("READY"), "Path"].tolist()
    schema_bad = summary.loc[~required_columns_present, "Path"].tolist()
    non_independent = summary.loc[
        summary["IndependenceStatus"].isin(["NonIndependent", "PartialOverlapNeedsReview"]),
        "Path",
    ].tolist()
    derived = summary.loc[derived_output_excluded, "Path"].tolist()
    candidate_ready = candidate_counts.loc[
        candidate_counts["BothESAndNQNonzero"] & candidate_counts["Candidate"].ne("Base_ES_NQ")
    ]
    return pd.DataFrame([
        {"Metric": "reference_path", "Value": reference_path},
        {"Metric": "input_patterns", "Value": ",".join(inputs)},
        {"Metric": "file_count_checked", "Value": len(summary)},
        {"Metric": "ready_independent_dataset_count", "Value": len(ready)},
        {"Metric": "ready_independent_datasets", "Value": ",".join(ready) if ready else "none"},
        {"Metric": "schema_incompatible_file_count", "Value": len(schema_bad)},
        {"Metric": "non_independent_or_partial_overlap_count", "Value": len(non_independent)},
        {"Metric": "derived_output_excluded_count", "Value": len(derived)},
        {
            "Metric": "files_with_non_base_candidate_ES_and_NQ_counts",
            "Value": ",".join(sorted(candidate_ready["Path"].unique())) if not candidate_ready.empty else "none",
        },
        {"Metric": "candidate_mode", "Value": "frozen rules only; no candidate search or tuning"},
        {"Metric": "prior_slope_q3_frozen_interval", "Value": f"[{PRIOR_SLOPE_Q3_MIN}, {PRIOR_SLOPE_Q3_MAX}]"},
        {"Metric": "predictor_fields_excluded_as_leaky", "Value": ",".join(LEAKY_OUTCOME_FIELDS)},
    ])


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--reference", default="tables/apva_forward_signed_return_dataset_v1.csv")
    parser.add_argument("--outdir", default="outputs/dataset_readiness_check")
    parser.add_argument("--intended-extension", nargs="*", default=[])
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--pressure", default="CompressionPressure")
    parser.add_argument("--entropy-min", type=float, default=0.88)
    parser.add_argument("--entropy-max", type=float, default=1.30)
    parser.add_argument("--directional-presence", type=float, default=0.0)
    parser.add_argument("--lookback", type=int, default=5)
    args = parser.parse_known_args(argv)[0]

    leaked_predictors = set(PREDICTOR_FIELDS).intersection(LEAKY_OUTCOME_FIELDS)
    if leaked_predictors:
        raise RuntimeError(f"Leaky predictor fields configured: {sorted(leaked_predictors)}")

    workspace = Path(".").resolve()
    reference_path = Path(args.reference)
    reference_path = (reference_path if reference_path.is_absolute() else workspace / reference_path).resolve()
    outdir = Path(args.outdir)
    outdir = (outdir if outdir.is_absolute() else workspace / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    reference = numeric_coercion(pd.read_csv(reference_path))
    inputs = [
        path for path in expand_patterns(args.inputs, workspace)
        if outdir not in path.parents
    ]
    intended = set(expand_patterns(args.intended_extension, workspace))
    if not inputs:
        raise RuntimeError("No input CSV files matched --inputs")
    summary_rows, candidate_rows, overlap_rows = [], [], []
    for path in inputs:
        summary, candidates, overlap = readiness_for_file(path, reference, intended, workspace, args)
        summary_rows.append(summary)
        candidate_rows.extend(candidates)
        overlap_rows.append(overlap)
    summary = pd.DataFrame(summary_rows).sort_values("Path").reset_index(drop=True)
    counts = pd.DataFrame(candidate_rows).sort_values(["Path", "Candidate"]).reset_index(drop=True)
    overlap = pd.DataFrame(overlap_rows).sort_values("Path").reset_index(drop=True)
    scorecard = build_scorecard(summary, counts, display_path(reference_path, workspace), args.inputs)
    summary.to_csv(outdir / "dataset_readiness_summary.csv", index=False)
    counts.to_csv(outdir / "dataset_readiness_candidate_counts.csv", index=False)
    overlap.to_csv(outdir / "dataset_readiness_overlap.csv", index=False)
    scorecard.to_csv(outdir / "dataset_readiness_scorecard.csv", index=False)
    print("APVA dataset readiness check complete")
    print(scorecard.to_string(index=False))
    return 0


if __name__ == "__main__":
    main()
