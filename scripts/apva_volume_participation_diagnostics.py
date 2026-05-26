#!/usr/bin/env python3
"""Describe participation-state proxies for the frozen PriorSlope Q3 candidate.

This diagnostic layer joins evaluated frozen-candidate entries to raw NT8
state-log fields when available. It does not create candidates, alter frozen
thresholds, or evaluate new rules.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from apva_analysis_utils import summarize
from apva_cross_era_validation_report import infer_regime


CANDIDATE = "PriorSlope_DominantPressureValue_Q3"
VALIDATION_MODES = ["Reference", "Spacing_10", "Spacing_20"]
ENTRY_KEY = ["ValidationMode", "Dataset", "Instrument", "File", "BarIndex", "HorizonBars"]
RAW_JOIN_KEY = ["Instrument", "File", "BarIndex"]
NUMERIC_PROXIES = [
    "DominantPressureValue",
    "PriorSlope_DominantPressureValue",
    "RollingEntropy",
    "RollingDirectionalPresence",
    "SponsorConfidence",
    "DominanceScore",
    "DegradationScore",
    "BalanceScore",
    "TransitionScore",
    "AmbiguityScore",
]
CATEGORICAL_PROXIES = ["MacroState", "SponsorState", "SequencePhase"]
VOLUME_NUMERIC_FIELDS = [
    "Volume",
    "VolumeSMA",
    "RelativeVolume",
    "VolumeZScore",
    "SignedVolume",
    "UpVolume",
    "DownVolume",
    "FlatVolume",
    "UpDownVolumeDelta",
]
SPYDER_NUMERIC_FIELDS = [
    "SpyderDominantVolume",
    "SpyderNonDominantVolume",
    "SpyderDominantVolumeShare",
    "SpyderNonDominantVolumeShare",
]
VOLUME_CATEGORICAL_FIELDS = ["BarDirection"]
SPYDER_CATEGORICAL_FIELDS = ["SpyderNonDominantColor", "SpyderSplitMethod"]
EXPLICIT_VOLUME_FIELDS = (
    VOLUME_NUMERIC_FIELDS
    + SPYDER_NUMERIC_FIELDS
    + VOLUME_CATEGORICAL_FIELDS
    + SPYDER_CATEGORICAL_FIELDS
)
RAW_ENRICHMENT_FIELDS = [
    "SponsorState",
    "SponsorConfidence",
    "SequencePhase",
    "DominanceScore",
    "DegradationScore",
    "BalanceScore",
    "TransitionScore",
    "AmbiguityScore",
] + EXPLICIT_VOLUME_FIELDS
FULL_PIPELINE_STATUS = "outputs/full_pipeline_run/full_pipeline_regime_status.csv"
FULL_PIPELINE_INVENTORY = "outputs/full_pipeline_run/full_pipeline_inventory.csv"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required diagnostic input: {path}")
    return pd.read_csv(path)


def display_path(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def annotate_regime(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    parsed = out["Dataset"].map(infer_regime)
    out["RegimeCode"] = parsed.map(lambda pair: pair[0])
    out["Regime"] = parsed.map(lambda pair: pair[1])
    return out


def as_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def source_inventory(
    canonical_paths: list[Path],
    raw_paths: list[Path],
    workspace: Path,
) -> tuple[pd.DataFrame, bool]:
    fields = NUMERIC_PROXIES + CATEGORICAL_PROXIES + EXPLICIT_VOLUME_FIELDS
    rows: list[dict[str, object]] = []
    explicit_volume_present = False
    for source_type, paths in [("CanonicalDataset", canonical_paths), ("RawStateLog", raw_paths)]:
        for path in paths:
            columns = list(pd.read_csv(path, nrows=0).columns)
            column_lookup = {column.lower(): column for column in columns}
            for field in fields:
                present_name = column_lookup.get(field.lower(), "")
                present = bool(present_name)
                if field in EXPLICIT_VOLUME_FIELDS and present:
                    explicit_volume_present = True
                rows.append({
                    "SourceType": source_type,
                    "SourcePath": display_path(path, workspace),
                    "Field": field,
                    "FeatureFamily": (
                        "ExplicitVolume" if field in EXPLICIT_VOLUME_FIELDS
                        else ("CategoricalStateProxy" if field in CATEGORICAL_PROXIES else "NumericStateProxy")
                    ),
                    "Present": present,
                    "ColumnName": present_name,
                })
    return pd.DataFrame(rows), explicit_volume_present


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def resolve_source_path(value: str, workspace: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def selected_raw_sources(
    referenced: pd.DataFrame,
    raw_paths: list[Path],
    status_path: Path,
    inventory_path: Path,
    workspace: Path,
) -> list[tuple[Path, str]]:
    direct_paths: dict[str, list[Path]] = {}
    for path in raw_paths:
        direct_paths.setdefault(path.name, []).append(path)

    status = read_csv(status_path) if status_path.exists() else pd.DataFrame()
    status_by_dataset = (
        status.loc[status["PairStatus"].isin(["PAIRED", "SINGLE_INSTRUMENT_ALLOWED"])]
        .set_index("DatasetPath")
        if not status.empty
        else pd.DataFrame()
    )
    selected_inventory_paths: set[Path] = set()
    selected_inventory_sources: dict[tuple[str, str], Path] = {}
    if inventory_path.exists():
        inventory = read_csv(inventory_path)
        for _, row in inventory.loc[inventory["Selected"].map(is_true)].iterrows():
            if not str(row["RawFile"]).strip():
                continue
            selected_path = resolve_source_path(str(row["RawFile"]), workspace)
            selected_inventory_paths.add(selected_path.resolve())
            selected_inventory_sources[(str(row["Instrument"]).upper(), str(row["Regime"]))] = selected_path

    sources: list[tuple[Path, str]] = []
    requested = referenced[["Dataset", "Instrument", "File"]].drop_duplicates()
    for _, row in requested.sort_values(["Dataset", "Instrument", "File"]).iterrows():
        dataset = str(row["Dataset"])
        instrument = str(row["Instrument"]).upper()
        join_file = str(row["File"])
        source_path: Optional[Path] = None
        if not status_by_dataset.empty and dataset in status_by_dataset.index:
            status_row = status_by_dataset.loc[dataset]
            if isinstance(status_row, pd.DataFrame):
                status_row = status_row.iloc[0]
            source_value = status_row.get(f"{instrument}File", "")
            if pd.notna(source_value) and str(source_value).strip():
                source_path = resolve_source_path(str(source_value), workspace)
        if source_path is None:
            regime_match = re.search(r"(?:^|_)(\d{6})(?:_|$)", Path(dataset).stem)
            if regime_match:
                source_path = selected_inventory_sources.get((instrument, regime_match.group(1)))
        if source_path is None:
            candidates = sorted(direct_paths.get(join_file, []))
            if len(candidates) == 1:
                source_path = candidates[0]
            elif len(candidates) > 1:
                preferred = [path for path in candidates if path.resolve() in selected_inventory_paths]
                source_path = sorted(preferred or candidates)[0]
        if source_path is None or not source_path.exists():
            raise RuntimeError(
                f"Raw enrichment source is missing for evaluated identity: "
                f"{dataset}, {instrument}, {join_file}"
            )
        sources.append((source_path, join_file))
    return sorted(set(sources), key=lambda source: (display_path(source[0], workspace), source[1]))


def load_raw_state_fields(
    raw_sources: list[tuple[Path, str]],
    workspace: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path, join_file in raw_sources:
        raw = pd.read_csv(path)
        required = {"Instrument", "BarIndex"}
        if not required.issubset(raw.columns):
            continue
        raw["BarIndex"] = pd.to_numeric(raw["BarIndex"], errors="coerce")
        raw = raw.loc[raw["BarIndex"].notna()].copy()
        raw["BarIndex"] = raw["BarIndex"].astype(int)
        raw["Instrument"] = raw["Instrument"].astype(str).str.upper()
        raw["File"] = join_file
        raw["RawSourcePath"] = display_path(path, workspace)
        raw["RawSourceRowNumber"] = raw.index + 2
        available = [column for column in RAW_ENRICHMENT_FIELDS if column in raw.columns]
        frames.append(raw[RAW_JOIN_KEY + ["RawSourcePath", "RawSourceRowNumber"] + available])
    if not frames:
        empty = pd.DataFrame(columns=RAW_JOIN_KEY + ["RawSourcePath"] + RAW_ENRICHMENT_FIELDS)
        return empty, pd.DataFrame(columns=RAW_JOIN_KEY + ["RawSourcePath", "RawSourceRowNumber"])
    joined = pd.concat(frames, ignore_index=True)
    duplicate_keys = joined.duplicated(RAW_JOIN_KEY, keep=False)
    duplicates = joined.loc[duplicate_keys].sort_values(
        RAW_JOIN_KEY + ["RawSourcePath", "RawSourceRowNumber"],
        kind="stable",
    )
    deduplicated = joined.sort_values(
        RAW_JOIN_KEY + ["RawSourcePath", "RawSourceRowNumber"],
        kind="stable",
    ).drop_duplicates(RAW_JOIN_KEY, keep="first")
    deduplicated = deduplicated.drop(columns=["RawSourceRowNumber"])
    return (
        as_numeric(deduplicated, [column for column in RAW_ENRICHMENT_FIELDS if column not in CATEGORICAL_PROXIES]),
        duplicates,
    )


def enrich_entries(entries: pd.DataFrame, raw_fields: pd.DataFrame) -> pd.DataFrame:
    out = entries.copy()
    out["Instrument"] = out["Instrument"].astype(str).str.upper()
    out["BarIndex"] = pd.to_numeric(out["BarIndex"], errors="coerce").astype("Int64")
    for field in RAW_ENRICHMENT_FIELDS:
        if field in out.columns:
            out = out.drop(columns=[field])
    out = out.merge(raw_fields, on=RAW_JOIN_KEY, how="left", validate="many_to_one")
    for field in EXPLICIT_VOLUME_FIELDS:
        if field not in out.columns:
            out[field] = np.nan
    requested_raw = [column for column in RAW_ENRICHMENT_FIELDS if column in raw_fields.columns]
    if requested_raw and out["RawSourcePath"].isna().any():
        missing = out.loc[out["RawSourcePath"].isna(), RAW_JOIN_KEY].drop_duplicates().head(5)
        raise RuntimeError(f"Raw state enrichment did not match evaluated entries: {missing.to_dict('records')}")
    return annotate_regime(as_numeric(
        out,
        NUMERIC_PROXIES + VOLUME_NUMERIC_FIELDS + SPYDER_NUMERIC_FIELDS + ["NormalizedPolicyOutcome"],
    ))


def evaluated_base_comparison(extended: pd.DataFrame, prior: pd.DataFrame, raw_fields: pd.DataFrame) -> pd.DataFrame:
    base = extended.loc[extended["Candidate"].eq("Base_ES_NQ")].copy()
    base = enrich_entries(base, raw_fields)
    prior_keys = pd.MultiIndex.from_frame(prior[ENTRY_KEY])
    base["Cohort"] = np.where(
        pd.MultiIndex.from_frame(base[ENTRY_KEY]).isin(prior_keys),
        "PriorSlope_Q3",
        "Base_NotPriorSlope_Q3",
    )
    return base


def outcome_stats(group: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = dict(summarize(group["NormalizedPolicyOutcome"]))
    row["StopRate"] = float(group["DisasterStopped"].astype(bool).mean()) if len(group) else np.nan
    return row


def group_outcomes(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_cols, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = {column: value for column, value in zip(group_cols, keys)}
        row.update(outcome_stats(group))
        row["EntryShareWithinMode"] = float(len(group) / len(frame.loc[frame["ValidationMode"].eq(row["ValidationMode"])]))
        rows.append(row)
    return pd.DataFrame(rows)


def numeric_proxy_summary(base: pd.DataFrame, available: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mode in VALIDATION_MODES:
        for cohort in ["PriorSlope_Q3", "Base_NotPriorSlope_Q3"]:
            group = base.loc[base["ValidationMode"].eq(mode) & base["Cohort"].eq(cohort)]
            for proxy in available:
                values = pd.to_numeric(group[proxy], errors="coerce").dropna()
                rows.append({
                    "ValidationMode": mode,
                    "Cohort": cohort,
                    "Proxy": proxy,
                    "Count": int(len(group)),
                    "AvailableCount": int(len(values)),
                    "Mean": float(values.mean()) if len(values) else np.nan,
                    "Median": float(values.median()) if len(values) else np.nan,
                    "Std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "Q25": float(values.quantile(0.25)) if len(values) else np.nan,
                    "Q75": float(values.quantile(0.75)) if len(values) else np.nan,
                    "PositiveFraction": float((values > 0).mean()) if len(values) else np.nan,
                })
    return pd.DataFrame(rows)


def numeric_subset_summary(frame: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    breakdowns = [
        ("AllAvailableRows", []),
        ("ByInstrument", ["Instrument"]),
        ("ByRegimeInstrument", ["Regime", "Instrument"]),
    ]
    for mode in VALIDATION_MODES:
        mode_frame = frame.loc[frame["ValidationMode"].eq(mode)]
        for breakdown, dimensions in breakdowns:
            groups = [((), mode_frame)] if not dimensions else mode_frame.groupby(dimensions, dropna=False, sort=True)
            for keys, group in groups:
                keys = keys if isinstance(keys, tuple) else (keys,)
                dimensions_values = dict(zip(dimensions, keys))
                for cohort in ["PriorSlope_Q3", "Base_NotPriorSlope_Q3"]:
                    cohort_group = group.loc[group["Cohort"].eq(cohort)]
                    for field in fields:
                        values = pd.to_numeric(cohort_group[field], errors="coerce").dropna()
                        rows.append({
                            "DataScope": "VolumeAvailableSubset",
                            "Breakdown": breakdown,
                            "ValidationMode": mode,
                            "Regime": dimensions_values.get("Regime", "All"),
                            "Instrument": dimensions_values.get("Instrument", "All"),
                            "Cohort": cohort,
                            "Field": field,
                            "Count": int(len(cohort_group)),
                            "AvailableCount": int(len(values)),
                            "Mean": float(values.mean()) if len(values) else np.nan,
                            "Median": float(values.median()) if len(values) else np.nan,
                            "Std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                            "Q25": float(values.quantile(0.25)) if len(values) else np.nan,
                            "Q75": float(values.quantile(0.75)) if len(values) else np.nan,
                            "PositiveFraction": float((values > 0).mean()) if len(values) else np.nan,
                        })
    return pd.DataFrame(rows)


def categorical_subset_summary(frame: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mode in VALIDATION_MODES:
        mode_frame = frame.loc[frame["ValidationMode"].eq(mode)]
        for field in fields:
            for (cohort, value), group in mode_frame.groupby(["Cohort", field], dropna=False, sort=True):
                rows.append({
                    "DataScope": "VolumeAvailableSubset",
                    "Breakdown": "Category",
                    "ValidationMode": mode,
                    "Regime": "All",
                    "Instrument": "All",
                    "Cohort": cohort,
                    "Field": field,
                    "Category": value,
                    "Count": int(len(group)),
                    "MeanOutcome": float(pd.to_numeric(group["NormalizedPolicyOutcome"], errors="coerce").mean()),
                    "EntryShareWithinCohort": float(len(group) / len(mode_frame.loc[mode_frame["Cohort"].eq(cohort)])),
                })
    return pd.DataFrame(rows)


def correlation(left: pd.Series, right: pd.Series, method: str) -> tuple[int, float]:
    data = pd.DataFrame({"x": pd.to_numeric(left, errors="coerce"), "y": pd.to_numeric(right, errors="coerce")}).dropna()
    if len(data) < 3 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return len(data), np.nan
    if method == "spearman":
        return len(data), float(data["x"].rank().corr(data["y"].rank()))
    return len(data), float(data["x"].corr(data["y"]))


def proxy_correlations(base: pd.DataFrame, prior: pd.DataFrame, available: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    reference_base = base.loc[base["ValidationMode"].eq("Reference")].copy()
    reference_base["IsPriorSlope_Q3"] = reference_base["Cohort"].eq("PriorSlope_Q3").astype(int)
    for proxy in available:
        n, pearson = correlation(reference_base[proxy], reference_base["IsPriorSlope_Q3"], "pearson")
        _, spearman = correlation(reference_base[proxy], reference_base["IsPriorSlope_Q3"], "spearman")
        rows.append({
            "Analysis": "CandidateMembershipWithinReferenceBase",
            "ValidationMode": "Reference",
            "Instrument": "All",
            "Proxy": proxy,
            "Count": n,
            "PearsonCorrelation": pearson,
            "SpearmanCorrelation": spearman,
        })
    for mode in VALIDATION_MODES:
        for instrument in ["All", "ES", "NQ"]:
            group = prior.loc[prior["ValidationMode"].eq(mode)]
            if instrument != "All":
                group = group.loc[group["Instrument"].eq(instrument)]
            for proxy in available:
                n, pearson = correlation(group[proxy], group["NormalizedPolicyOutcome"], "pearson")
                _, spearman = correlation(group[proxy], group["NormalizedPolicyOutcome"], "spearman")
                rows.append({
                    "Analysis": "OutcomeWithinPriorSlope_Q3",
                    "ValidationMode": mode,
                    "Instrument": instrument,
                    "Proxy": proxy,
                    "Count": n,
                    "PearsonCorrelation": pearson,
                    "SpearmanCorrelation": spearman,
                })
    return pd.DataFrame(rows)


def volume_correlations(base: pd.DataFrame, prior: pd.DataFrame, available: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    reference_base = base.loc[base["ValidationMode"].eq("Reference")].copy()
    reference_base["IsPriorSlope_Q3"] = reference_base["Cohort"].eq("PriorSlope_Q3").astype(int)
    for instrument in ["All", "ES", "NQ"]:
        group = reference_base if instrument == "All" else reference_base.loc[reference_base["Instrument"].eq(instrument)]
        for field in available:
            n, pearson = correlation(group[field], group["IsPriorSlope_Q3"], "pearson")
            _, spearman = correlation(group[field], group["IsPriorSlope_Q3"], "spearman")
            rows.append({
                "DataScope": "VolumeAvailableSubset",
                "Analysis": "CandidateMembershipWithinReferenceBase",
                "ValidationMode": "Reference",
                "Regime": "All",
                "Instrument": instrument,
                "Field": field,
                "Count": n,
                "PearsonCorrelation": pearson,
                "SpearmanCorrelation": spearman,
            })
    for mode in VALIDATION_MODES:
        mode_prior = prior.loc[prior["ValidationMode"].eq(mode)]
        groupings = [("All", "All", mode_prior)]
        groupings.extend(("All", instrument, group) for instrument, group in mode_prior.groupby("Instrument", sort=True))
        groupings.extend((regime, instrument, group) for (regime, instrument), group in mode_prior.groupby(["Regime", "Instrument"], sort=True))
        for regime, instrument, group in groupings:
            for field in available:
                n, pearson = correlation(group[field], group["NormalizedPolicyOutcome"], "pearson")
                _, spearman = correlation(group[field], group["NormalizedPolicyOutcome"], "spearman")
                rows.append({
                    "DataScope": "VolumeAvailableSubset",
                    "Analysis": "OutcomeWithinPriorSlope_Q3",
                    "ValidationMode": mode,
                    "Regime": regime,
                    "Instrument": instrument,
                    "Field": field,
                    "Count": n,
                    "PearsonCorrelation": pearson,
                    "SpearmanCorrelation": spearman,
                })
    return pd.DataFrame(rows)


def scorecard(
    prior: pd.DataFrame,
    proxy_summary: pd.DataFrame,
    macro: pd.DataFrame,
    sponsor: pd.DataFrame,
    sequence: pd.DataFrame,
    regime_instrument: pd.DataFrame,
    correlations: pd.DataFrame,
    explicit_volume_present: bool,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    ref = prior.loc[prior["ValidationMode"].eq("Reference")]
    ref_corr = correlations.loc[
        correlations["Analysis"].eq("CandidateMembershipWithinReferenceBase")
    ].copy()
    ref_corr["AbsSpearman"] = ref_corr["SpearmanCorrelation"].abs()
    best_proxy = ref_corr.sort_values("AbsSpearman", ascending=False).iloc[0] if not ref_corr.empty else None
    nondef_corr = ref_corr.loc[ref_corr["Proxy"].ne("PriorSlope_DominantPressureValue")]
    best_nondef = nondef_corr.sort_values("AbsSpearman", ascending=False).iloc[0] if not nondef_corr.empty else None
    outcome_corr = correlations.loc[
        correlations["Analysis"].eq("OutcomeWithinPriorSlope_Q3")
        & correlations["ValidationMode"].eq("Reference")
        & correlations["Instrument"].eq("All")
    ].copy()
    outcome_corr["AbsSpearman"] = outcome_corr["SpearmanCorrelation"].abs()
    best_outcome = outcome_corr.sort_values("AbsSpearman", ascending=False).iloc[0] if not outcome_corr.empty else None
    sponsor_ref = sponsor.loc[sponsor["ValidationMode"].eq("Reference")]
    sequence_ref = sequence.loc[sequence["ValidationMode"].eq("Reference")]
    macro_ref = macro.loc[macro["ValidationMode"].eq("Reference")]
    ref_proxy = proxy_summary.loc[proxy_summary["ValidationMode"].eq("Reference")]
    prior_entropy = ref_proxy.loc[
        ref_proxy["Cohort"].eq("PriorSlope_Q3") & ref_proxy["Proxy"].eq("RollingEntropy"), "Mean"
    ].iloc[0]
    nonprior_entropy = ref_proxy.loc[
        ref_proxy["Cohort"].eq("Base_NotPriorSlope_Q3") & ref_proxy["Proxy"].eq("RollingEntropy"), "Mean"
    ].iloc[0]
    rows = [
        {"Metric": "report_scope", "Value": "diagnostic-only participation-state description of frozen PriorSlope_DominantPressureValue_Q3"},
        {"Metric": "frozen_rule", "Value": "0.004718046888521732 <= PriorSlope_DominantPressureValue <= 0.05505235072964343"},
        {"Metric": "candidate_rules_changed", "Value": False},
        {"Metric": "thresholds_changed", "Value": False},
        {"Metric": "explicit_volume_present", "Value": explicit_volume_present},
        {"Metric": "explicit_volume_fields_found", "Value": ",".join(sorted(inventory.loc[inventory["FeatureFamily"].eq("ExplicitVolume") & inventory["Present"], "Field"].unique())) or "none"},
        {"Metric": "available_numeric_participation_proxies", "Value": ",".join(proxy_summary["Proxy"].drop_duplicates())},
        {"Metric": "available_categorical_participation_proxies", "Value": ",".join([column for column in CATEGORICAL_PROXIES if column in prior.columns])},
        {"Metric": "reference_count", "Value": int(len(ref))},
        {"Metric": "prior_slope_direction", "Value": "rising_positive_by_frozen_rule"},
        {"Metric": "reference_prior_slope_min", "Value": float(ref["PriorSlope_DominantPressureValue"].min())},
        {"Metric": "reference_prior_slope_max", "Value": float(ref["PriorSlope_DominantPressureValue"].max())},
        {"Metric": "reference_entropy_mean", "Value": float(ref["RollingEntropy"].mean())},
        {"Metric": "reference_entropy_median", "Value": float(ref["RollingEntropy"].median())},
        {"Metric": "reference_base_not_prior_entropy_mean", "Value": float(nonprior_entropy)},
        {"Metric": "reference_prior_minus_base_not_prior_entropy_mean", "Value": float(prior_entropy - nonprior_entropy)},
        {"Metric": "reference_es_mean_outcome", "Value": float(ref.loc[ref["Instrument"].eq("ES"), "NormalizedPolicyOutcome"].mean())},
        {"Metric": "reference_nq_mean_outcome", "Value": float(ref.loc[ref["Instrument"].eq("NQ"), "NormalizedPolicyOutcome"].mean())},
        {"Metric": "reference_positive_regime_instrument_fraction", "Value": float((regime_instrument.loc[regime_instrument["ValidationMode"].eq("Reference"), "Mean"] > 0).mean())},
        {"Metric": "reference_macrostate_category_count", "Value": int(macro_ref["MacroState"].nunique())},
        {"Metric": "reference_max_macrostate_entry_share", "Value": float(macro_ref["EntryShareWithinMode"].max())},
        {"Metric": "reference_sponsorstate_category_count", "Value": int(sponsor_ref["SponsorState"].nunique()) if not sponsor_ref.empty else 0},
        {"Metric": "reference_max_sponsorstate_entry_share", "Value": float(sponsor_ref["EntryShareWithinMode"].max()) if not sponsor_ref.empty else np.nan},
        {"Metric": "reference_sequencephase_category_count", "Value": int(sequence_ref["SequencePhase"].nunique()) if not sequence_ref.empty else 0},
        {"Metric": "reference_max_sequencephase_entry_share", "Value": float(sequence_ref["EntryShareWithinMode"].max()) if not sequence_ref.empty else np.nan},
        {"Metric": "strongest_membership_proxy_by_absolute_spearman", "Value": best_proxy["Proxy"] if best_proxy is not None else "none"},
        {"Metric": "strongest_membership_proxy_spearman", "Value": best_proxy["SpearmanCorrelation"] if best_proxy is not None else np.nan},
        {"Metric": "strongest_nondefinitional_membership_proxy", "Value": best_nondef["Proxy"] if best_nondef is not None else "none"},
        {"Metric": "strongest_nondefinitional_membership_proxy_spearman", "Value": best_nondef["SpearmanCorrelation"] if best_nondef is not None else np.nan},
        {"Metric": "strongest_reference_outcome_proxy_within_candidate", "Value": best_outcome["Proxy"] if best_outcome is not None else "none"},
        {"Metric": "strongest_reference_outcome_proxy_spearman", "Value": best_outcome["SpearmanCorrelation"] if best_outcome is not None else np.nan},
    ]
    return pd.DataFrame(rows)


def fmt(value: object, digits: int = 3) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.{digits}f}"


def md(value: object) -> str:
    return str(value).replace("|", "\\|")


def subset_mean(summary: pd.DataFrame, field: str, cohort: str) -> float:
    matches = summary.loc[
        summary["Breakdown"].eq("AllAvailableRows")
        & summary["ValidationMode"].eq("Reference")
        & summary["Cohort"].eq(cohort)
        & summary["Field"].eq(field),
        "Mean",
    ]
    return float(matches.iloc[0]) if not matches.empty else np.nan


def subset_correlation(
    correlations: pd.DataFrame,
    analysis: str,
    field: str,
) -> float:
    matches = correlations.loc[
        correlations["Analysis"].eq(analysis)
        & correlations["ValidationMode"].eq("Reference")
        & correlations["Regime"].eq("All")
        & correlations["Instrument"].eq("All")
        & correlations["Field"].eq(field),
        "SpearmanCorrelation",
    ]
    return float(matches.iloc[0]) if not matches.empty else np.nan


def write_handoff(
    path: Path,
    prior: pd.DataFrame,
    proxy_summary: pd.DataFrame,
    macro: pd.DataFrame,
    sponsor: pd.DataFrame,
    sequence: pd.DataFrame,
    regime_instrument: pd.DataFrame,
    correlations: pd.DataFrame,
    explicit_volume_present: bool,
    volume_summary: pd.DataFrame,
    spyder_summary: pd.DataFrame,
    volume_corr: pd.DataFrame,
) -> None:
    ref = prior.loc[prior["ValidationMode"].eq("Reference")]
    slope = ref["PriorSlope_DominantPressureValue"]
    proxy_names = [proxy for proxy in proxy_summary["Proxy"].drop_duplicates().tolist() if proxy in NUMERIC_PROXIES]
    member_corr = correlations.loc[
        correlations["Analysis"].eq("CandidateMembershipWithinReferenceBase")
        & correlations["Proxy"].isin(NUMERIC_PROXIES)
    ].copy()
    member_corr["AbsSpearman"] = member_corr["SpearmanCorrelation"].abs()
    best = member_corr.sort_values("AbsSpearman", ascending=False).iloc[0]
    best_nondef = member_corr.loc[
        member_corr["Proxy"].ne("PriorSlope_DominantPressureValue")
    ].sort_values("AbsSpearman", ascending=False).iloc[0]
    within_outcome = correlations.loc[
        correlations["Analysis"].eq("OutcomeWithinPriorSlope_Q3")
        & correlations["ValidationMode"].eq("Reference")
        & correlations["Instrument"].eq("All")
        & correlations["Proxy"].isin(NUMERIC_PROXIES)
    ].copy()
    within_outcome["AbsSpearman"] = within_outcome["SpearmanCorrelation"].abs()
    best_outcome = within_outcome.sort_values("AbsSpearman", ascending=False).iloc[0]
    inst = group_outcomes(ref.assign(ValidationMode="Reference"), ["ValidationMode", "Instrument"]).set_index("Instrument")
    regime_inst = regime_instrument.loc[regime_instrument["ValidationMode"].eq("Reference")]
    positive_ri = int((regime_inst["Mean"] > 0).sum())
    total_ri = int(len(regime_inst))
    macro_ref = macro.loc[macro["ValidationMode"].eq("Reference")].sort_values("Count", ascending=False)
    sponsor_ref = sponsor.loc[sponsor["ValidationMode"].eq("Reference")].sort_values("Count", ascending=False)
    sequence_ref = sequence.loc[sequence["ValidationMode"].eq("Reference")].sort_values("Count", ascending=False)
    reference_proxy = proxy_summary.loc[proxy_summary["ValidationMode"].eq("Reference")]
    prior_entropy = float(reference_proxy.loc[
        reference_proxy["Cohort"].eq("PriorSlope_Q3") & reference_proxy["Proxy"].eq("RollingEntropy"), "Mean"
    ].iloc[0])
    nonprior_entropy = float(reference_proxy.loc[
        reference_proxy["Cohort"].eq("Base_NotPriorSlope_Q3") & reference_proxy["Proxy"].eq("RollingEntropy"), "Mean"
    ].iloc[0])
    volume_ref = prior.loc[prior["ValidationMode"].eq("Reference") & prior["RelativeVolume"].notna()]
    volume_regimes = sorted(volume_ref["RegimeCode"].dropna().astype(str).unique())
    volume_instruments = sorted(volume_ref["Instrument"].dropna().astype(str).unique())
    relative_prior = subset_mean(volume_summary, "RelativeVolume", "PriorSlope_Q3")
    relative_other = subset_mean(volume_summary, "RelativeVolume", "Base_NotPriorSlope_Q3")
    z_prior = subset_mean(volume_summary, "VolumeZScore", "PriorSlope_Q3")
    z_other = subset_mean(volume_summary, "VolumeZScore", "Base_NotPriorSlope_Q3")
    signed_membership = subset_correlation(volume_corr, "CandidateMembershipWithinReferenceBase", "SignedVolume")
    signed_outcome = subset_correlation(volume_corr, "OutcomeWithinPriorSlope_Q3", "SignedVolume")
    delta_membership = subset_correlation(volume_corr, "CandidateMembershipWithinReferenceBase", "UpDownVolumeDelta")
    delta_outcome = subset_correlation(volume_corr, "OutcomeWithinPriorSlope_Q3", "UpDownVolumeDelta")
    dom_share_membership = subset_correlation(volume_corr, "CandidateMembershipWithinReferenceBase", "SpyderDominantVolumeShare")
    dom_share_outcome = subset_correlation(volume_corr, "OutcomeWithinPriorSlope_Q3", "SpyderDominantVolumeShare")
    nondom_share_membership = subset_correlation(volume_corr, "CandidateMembershipWithinReferenceBase", "SpyderNonDominantVolumeShare")
    nondom_share_outcome = subset_correlation(volume_corr, "OutcomeWithinPriorSlope_Q3", "SpyderNonDominantVolumeShare")
    reference_by_instrument = volume_summary.loc[
        volume_summary["Breakdown"].eq("ByInstrument")
        & volume_summary["ValidationMode"].eq("Reference")
        & volume_summary["Field"].isin(["RelativeVolume", "VolumeZScore"])
    ].pivot_table(index=["Instrument", "Field"], columns="Cohort", values="Mean")
    direction_notes: list[str] = []
    for instrument in volume_instruments:
        for field in ["RelativeVolume", "VolumeZScore"]:
            key = (instrument, field)
            if key in reference_by_instrument.index:
                row = reference_by_instrument.loc[key]
                delta = row.get("PriorSlope_Q3", np.nan) - row.get("Base_NotPriorSlope_Q3", np.nan)
                direction_notes.append(f"{instrument} {field} delta `{fmt(delta)}`")
    lines = [
        "# Volume And Participation Diagnostic Handoff",
        "",
        "## Scope",
        "",
        "This is a descriptive report for the frozen `PriorSlope_DominantPressureValue_Q3` candidate.",
        "No candidate rule or validation threshold was changed.",
        "",
        "## Is Volume Explicitly Present?",
        "",
        (
            "Explicit volume fields are present in the inventoried source files."
            if explicit_volume_present
            else "No explicit raw volume field or derived volume column is present in the current canonical datasets or raw NT8 state-log exports."
        ),
        "",
        f"Direct volume fields are available for a subset of evaluated rows: `{len(volume_ref)}` Reference PriorSlope entries "
        f"from regimes `{', '.join(volume_regimes) or 'none'}` and instruments `{', '.join(volume_instruments) or 'none'}`. "
        "Volume results below apply only to this volume-available subset; all-data state results continue to use every available era.",
        "",
        "## Available Participation Proxies",
        "",
        "- Numeric proxies: `" + "`, `".join(proxy_names) + "`.",
        "- Categorical proxies: `MacroState`, `SponsorState`, `SequencePhase`.",
        "- Direct volume fields: `Volume`, `VolumeSMA`, `RelativeVolume`, `VolumeZScore`, `BarDirection`, `SignedVolume`, `UpVolume`, `DownVolume`, `FlatVolume`, `UpDownVolumeDelta`.",
        "- Spyder proxy fields: `SpyderDominantVolume`, `SpyderNonDominantVolume`, `SpyderDominantVolumeShare`, `SpyderNonDominantVolumeShare`, `SpyderNonDominantColor`, `SpyderSplitMethod`.",
        "",
        "## Pressure-State Interpretation",
        "",
        f"The frozen rule itself selects positive prior pressure-value slopes, so PriorSlope Q3 corresponds to rising pressure-state participation by construction. "
        f"In Reference mode its slope spans `{fmt(slope.min(), 6)}` to `{fmt(slope.max(), 6)}` with median `{fmt(slope.median(), 6)}`.",
        "",
        f"Among available continuous proxies, `{best['Proxy']}` has the largest absolute Spearman association with PriorSlope membership within Reference base entries "
        f"(`{fmt(best['SpearmanCorrelation'])}`), as expected because it defines the frozen candidate. "
        f"Excluding that definitional field, the largest membership association is `{best_nondef['Proxy']}` "
        f"at only `{fmt(best_nondef['SpearmanCorrelation'])}`.",
        "",
        "## Entropy And State Context",
        "",
        f"PriorSlope Q3 Reference entries have RollingEntropy mean `{fmt(prior_entropy)}` and median `{fmt(ref['RollingEntropy'].median())}`; "
        f"Reference base entries outside PriorSlope Q3 have entropy mean `{fmt(nonprior_entropy)}`. "
        f"The difference (`{fmt(prior_entropy - nonprior_entropy)}`) is modestly higher for PriorSlope Q3. "
        "Because the frozen base state already bounds entropy, this report treats entropy as context rather than an independent explanation.",
        "",
        f"The most common MacroState is `{md(macro_ref.iloc[0]['MacroState'])}` with `{int(macro_ref.iloc[0]['Count'])}` of `{len(ref)}` Reference entries "
        f"(`{fmt(macro_ref.iloc[0]['EntryShareWithinMode'])}`). "
        f"The most common SponsorState is `{md(sponsor_ref.iloc[0]['SponsorState'])}` (`{fmt(sponsor_ref.iloc[0]['EntryShareWithinMode'])}` share), and "
        f"the most common SequencePhase is `{md(sequence_ref.iloc[0]['SequencePhase'])}` (`{fmt(sequence_ref.iloc[0]['EntryShareWithinMode'])}` share).",
        "",
        "The candidate is strongly situated in `Unresolved` MacroState, but no specific SponsorState or SequencePhase dominates a majority beyond the stated shares. "
        "No categorical state should be treated as a replacement rule from this diagnostic report.",
        "",
        "## ES, NQ, And Regimes",
        "",
        f"Both instruments remain positive in Reference mode: ES mean `{fmt(inst.loc['ES', 'Mean'])}` on `{int(inst.loc['ES', 'Count'])}` entries and "
        f"NQ mean `{fmt(inst.loc['NQ', 'Mean'])}` on `{int(inst.loc['NQ', 'Count'])}` entries.",
        "",
        f"Across Reference regime/instrument cells, `{positive_ri}` of `{total_ri}` have positive means. "
        "See `prior_slope_q3_by_regime_instrument.csv` for the sparse-cell detail.",
        "",
        "## Volume-Available Subset",
        "",
        f"Within Reference base entries where explicit volume is available, PriorSlope Q3 has mean `RelativeVolume` `{fmt(relative_prior)}` "
        f"versus `{fmt(relative_other)}` outside PriorSlope Q3; it is therefore "
        f"`{'higher' if relative_prior > relative_other else 'lower'}` in this subset.",
        "",
        f"Mean `VolumeZScore` is `{fmt(z_prior)}` for PriorSlope Q3 versus `{fmt(z_other)}` outside it; it is "
        f"`{'higher' if z_prior > z_other else 'lower'}` in this subset.",
        "",
        f"`SignedVolume` has Spearman `{fmt(signed_membership)}` with membership and `{fmt(signed_outcome)}` with outcome. "
        f"`UpDownVolumeDelta` has Spearman `{fmt(delta_membership)}` with membership and `{fmt(delta_outcome)}` with outcome. "
        "These describe direction-sensitive volume behavior; they are not candidate conditions.",
        "",
        f"`SpyderDominantVolumeShare` has Spearman `{fmt(dom_share_membership)}` with membership and `{fmt(dom_share_outcome)}` with outcome. "
        f"`SpyderNonDominantVolumeShare` has Spearman `{fmt(nondom_share_membership)}` with membership and `{fmt(nondom_share_outcome)}` with outcome. "
        "These shares are price-geometry weighted split-volume proxies, not bid/ask volume.",
        "",
        "Instrument and regime splits are provided in `prior_slope_q3_volume_summary.csv`, `prior_slope_q3_spyder_split_summary.csv`, "
        "and `prior_slope_q3_volume_correlations.csv`. "
        + ("Within the available regime: " + "; ".join(direction_notes) + ". " if direction_notes else "")
        + "Because explicit volume currently appears only in one regime, cross-regime consistency cannot yet be established; "
        "ES/NQ comparisons within that available regime are descriptive only.",
        "",
        "## Participation-State Versus Outcome Artifact",
        "",
        "PriorSlope Q3 is defined only from an ex-ante pressure-state slope and is not constructed from return or excursion outcomes. "
        f"Within Reference PriorSlope entries, the strongest numeric-proxy association with outcome is `{best_outcome['Proxy']}` "
        f"with Spearman `{fmt(best_outcome['SpearmanCorrelation'])}` across all-data state proxies. The new volume tables provide "
        "direct-volume diagnostics only for the available subset and do not establish a new rule or threshold.",
        "",
        "## Coverage Boundary",
        "",
        "Continue exporting the explicit volume and Spyder split fields for additional paired ES/NQ regimes, then rerun the unchanged pipeline. "
        "The purpose is to expand direct-volume coverage under the frozen candidate, not to tune or replace it.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended-entries", default="outputs/fixed_candidate_extended_validation/extended_entries.csv")
    parser.add_argument("--prior-entries", default="outputs/prior_slope_q3_diagnostics/prior_slope_q3_entries.csv")
    parser.add_argument("--canonical-glob", default="tables/apva_forward_signed_return_dataset*.csv")
    parser.add_argument("--raw-root", default="data/Validation")
    parser.add_argument("--full-pipeline-status", default=FULL_PIPELINE_STATUS)
    parser.add_argument("--full-pipeline-inventory", default=FULL_PIPELINE_INVENTORY)
    parser.add_argument("--outdir", default="outputs/volume_participation_diagnostics")
    args = parser.parse_args(argv)
    workspace = Path.cwd()
    extended = read_csv(workspace / args.extended_entries)
    prior_source = extended.loc[extended["Candidate"].eq(CANDIDATE)].copy()
    if prior_source.empty:
        prior_source = read_csv(workspace / args.prior_entries)
    canonical_paths = sorted(workspace.glob(args.canonical_glob))
    raw_root = workspace / args.raw_root
    raw_paths = sorted(raw_root.rglob("xApvaV01StateLog*.csv"))
    if not canonical_paths or not raw_paths:
        raise RuntimeError("Canonical dataset or raw-state-log inventory is empty.")
    referenced = pd.concat([extended, prior_source], ignore_index=True)
    raw_sources = selected_raw_sources(
        referenced,
        raw_paths,
        workspace / args.full_pipeline_status,
        workspace / args.full_pipeline_inventory,
        workspace,
    )
    inventory, explicit_volume_present = source_inventory(
        canonical_paths,
        sorted({path for path, _ in raw_sources}),
        workspace,
    )
    raw_fields, duplicate_keys = load_raw_state_fields(raw_sources, workspace)
    prior = enrich_entries(prior_source, raw_fields)
    base = evaluated_base_comparison(extended, prior, raw_fields)
    available_numeric = [
        proxy for proxy in NUMERIC_PROXIES + VOLUME_NUMERIC_FIELDS + SPYDER_NUMERIC_FIELDS
        if proxy in prior.columns and prior[proxy].notna().any()
    ]
    proxy_summary = numeric_proxy_summary(base, available_numeric)
    macro = group_outcomes(prior, ["ValidationMode", "MacroState"])
    sponsor = group_outcomes(prior, ["ValidationMode", "SponsorState"])
    sequence = group_outcomes(prior, ["ValidationMode", "SequencePhase"])
    regime_instrument = group_outcomes(prior, ["ValidationMode", "Regime", "Instrument"])
    correlations = proxy_correlations(base, prior, available_numeric)
    volume_base = base.loc[base["RelativeVolume"].notna()].copy()
    volume_prior = prior.loc[prior["RelativeVolume"].notna()].copy()
    available_volume = [field for field in VOLUME_NUMERIC_FIELDS if volume_prior[field].notna().any()]
    available_spyder = [field for field in SPYDER_NUMERIC_FIELDS if volume_prior[field].notna().any()]
    volume_summary = pd.concat(
        [
            numeric_subset_summary(volume_base, available_volume),
            categorical_subset_summary(volume_base, [
                field for field in VOLUME_CATEGORICAL_FIELDS if volume_base[field].notna().any()
            ]),
        ],
        ignore_index=True,
        sort=False,
    )
    spyder_summary = pd.concat(
        [
            numeric_subset_summary(volume_base, available_spyder),
            categorical_subset_summary(volume_base, [
                field for field in SPYDER_CATEGORICAL_FIELDS if volume_base[field].notna().any()
            ]),
        ],
        ignore_index=True,
        sort=False,
    )
    volume_corr = volume_correlations(
        volume_base,
        volume_prior,
        available_volume + available_spyder,
    )
    output_scorecard = scorecard(
        prior, proxy_summary, macro, sponsor, sequence, regime_instrument,
        correlations, explicit_volume_present, inventory,
    )
    outdir = workspace / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    duplicate_keys.to_csv(outdir / "raw_join_duplicate_keys.csv", index=False)
    inventory.to_csv(outdir / "volume_participation_feature_inventory.csv", index=False)
    proxy_summary.to_csv(outdir / "prior_slope_q3_state_proxy_summary.csv", index=False)
    macro.to_csv(outdir / "prior_slope_q3_by_macrostate.csv", index=False)
    sponsor.to_csv(outdir / "prior_slope_q3_by_sponsorstate.csv", index=False)
    sequence.to_csv(outdir / "prior_slope_q3_by_sequencephase.csv", index=False)
    regime_instrument.to_csv(outdir / "prior_slope_q3_by_regime_instrument.csv", index=False)
    correlations.to_csv(outdir / "prior_slope_q3_proxy_correlation.csv", index=False)
    volume_summary.to_csv(outdir / "prior_slope_q3_volume_summary.csv", index=False)
    spyder_summary.to_csv(outdir / "prior_slope_q3_spyder_split_summary.csv", index=False)
    volume_corr.to_csv(outdir / "prior_slope_q3_volume_correlations.csv", index=False)
    output_scorecard.to_csv(outdir / "volume_participation_scorecard.csv", index=False)
    write_handoff(
        outdir / "chatgpt_master_analysis_handoff.md",
        prior, proxy_summary, macro, sponsor, sequence, regime_instrument,
        correlations, explicit_volume_present, volume_summary, spyder_summary,
        volume_corr,
    )
    print("APVA volume/participation diagnostic report complete")
    print(output_scorecard.loc[output_scorecard["Metric"].isin([
        "explicit_volume_present", "reference_count",
        "strongest_membership_proxy_by_absolute_spearman",
        "strongest_membership_proxy_spearman",
    ])].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
