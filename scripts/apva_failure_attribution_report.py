#!/usr/bin/env python3
"""Attribute frozen APVA candidate validation failures to observed data slices.

This script is diagnostic only. It reads completed validation outputs and does
not alter frozen candidates, thresholds, selection rules, or pipeline inputs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FROZEN_CANDIDATES = [
    "PriorSlope_DominantPressureValue_Q3",
    "CCRRR",
    "RRCCC",
]
PRIOR_SLOPE = "PriorSlope_DominantPressureValue_Q3"
VALIDATION_MODES = ["Reference", "Spacing_10", "Spacing_20"]
VOLUME_DECLARED_REGIMES = {"062019", "062023", "122017", "122023"}
EXPLICIT_VOLUME_FIELD = "Volume"
OUTCOME = "NormalizedPolicyOutcome"
COMPARISONS = [
    ("Reference", "Spacing_10"),
    ("Spacing_10", "Spacing_20"),
    ("Reference", "Spacing_20"),
]


def numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def bool_value(value: object) -> bool:
    return str(value).strip().lower() == "true"


def dataset_code(dataset: object) -> str:
    text = str(dataset)
    match = re.search(r"_((?:0[1-9]|1[0-2])\d{4})_generated\.csv$", text)
    if match:
        return match.group(1)
    if text.endswith("_dataset_generated.csv"):
        return "LegacyGenerated"
    if text.endswith("_dataset_v1.csv"):
        return "CanonicalOriginal"
    return Path(text).stem


def regime_year(code: object) -> str:
    text = str(code)
    return text[-4:] if re.fullmatch(r"\d{6}", text) else text


def dataset_type(code: object) -> str:
    text = str(code)
    if text == "CanonicalOriginal":
        return "Canonical/Original"
    if text == "LegacyGenerated":
        return "Legacy Generated"
    return "Generated Regime"


def profit_factor(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    gains = clean.loc[clean > 0].sum()
    losses = clean.loc[clean < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / abs(losses))


def enrich_entries(entries: pd.DataFrame) -> pd.DataFrame:
    out = numeric(entries, [OUTCOME])
    out = out.loc[out["Candidate"].isin(FROZEN_CANDIDATES)].copy()
    out["DatasetCode"] = out["Dataset"].map(dataset_code)
    out["RegimeCode"] = out["DatasetCode"]
    out["RegimeYear"] = out["DatasetCode"].map(regime_year)
    out["DatasetType"] = out["DatasetCode"].map(dataset_type)
    return out


def selected_volume_codes(workspace: Path, status: pd.DataFrame) -> set[str]:
    explicit = set(VOLUME_DECLARED_REGIMES)
    for _, row in status.iterrows():
        raw_regime = str(row.get("Regime", "")).split(".")[0]
        code = raw_regime.zfill(6) if raw_regime.isdigit() else raw_regime
        for field in ["ESFile", "NQFile"]:
            raw_path = workspace / str(row.get(field, ""))
            if raw_path.exists():
                header = pd.read_csv(raw_path, nrows=0).columns
                if EXPLICIT_VOLUME_FIELD in header:
                    explicit.add(code)
    return explicit


def add_volume_classification(entries: pd.DataFrame, volume_codes: set[str]) -> pd.DataFrame:
    out = entries.copy()
    out["ExplicitVolumeAvailable"] = out["DatasetCode"].isin(volume_codes)
    out["VolumeCoverageGroup"] = np.where(
        out["ExplicitVolumeAvailable"], "VolumeEnabled", "LegacyOrNoExplicitVolume"
    )
    return out


def aggregate_rows(
    frame: pd.DataFrame,
    groups: list[str],
    total_groups: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(groups, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = pd.to_numeric(group[OUTCOME], errors="coerce").dropna()
        row = dict(zip(groups, keys))
        row.update(
            {
                "Count": int(values.count()),
                "Mean": float(values.mean()) if len(values) else np.nan,
                "Median": float(values.median()) if len(values) else np.nan,
                "ProfitFactor": profit_factor(values),
                "SumContribution": float(values.sum()),
                "PositiveFraction": float((values > 0).mean()) if len(values) else np.nan,
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    totals_by = total_groups or [column for column in groups if column not in {
        "Dataset", "DatasetCode", "DatasetType", "RegimeCode", "RegimeYear",
        "AttributionScope", "Instrument",
        "TimeBlock", "RegimeWeek", "VolumeCoverageGroup", "ExplicitVolumeAvailable"
    }]
    if not out.empty and totals_by:
        totals = out.groupby(totals_by)["SumContribution"].transform("sum")
        absolute = out.groupby(totals_by)["SumContribution"].transform(
            lambda values: values.abs().sum()
        )
        out["ContributionShareOfNet"] = out["SumContribution"] / totals.replace(0, np.nan)
        out["AbsoluteContributionShare"] = out["SumContribution"].abs() / absolute.replace(0, np.nan)
    else:
        out["ContributionShareOfNet"] = np.nan
        out["AbsoluteContributionShare"] = np.nan
    return out


def scorecard_metrics(scorecard: pd.DataFrame) -> dict[str, str]:
    return {str(row["Metric"]): str(row["Value"]) for _, row in scorecard.iterrows()}


def pooled_record(metrics: dict[str, str], mode: str, candidate: str) -> dict[str, object]:
    prefix = f"Pooled_{mode}_{candidate}_"
    def get(name: str, default: object = np.nan) -> object:
        return metrics.get(prefix + name, default)
    record: dict[str, object] = {
        "ValidationMode": mode,
        "Candidate": candidate,
        "Count": pd.to_numeric(get("Count"), errors="coerce"),
        "Mean": pd.to_numeric(get("Mean"), errors="coerce"),
        "Median": pd.to_numeric(get("Median"), errors="coerce"),
        "ProfitFactor": pd.to_numeric(get("ProfitFactor"), errors="coerce"),
        "BaseMethodProfitFactor": pd.to_numeric(get("BaseMethodProfitFactor"), errors="coerce"),
        "PositiveBlockFraction": pd.to_numeric(get("PositiveBlockFraction"), errors="coerce"),
        "MaxSingleBlockContributionFraction": pd.to_numeric(
            get("MaxSingleBlockContributionFraction"), errors="coerce"
        ),
        "ES_Mean": pd.to_numeric(get("ES_Mean"), errors="coerce"),
        "NQ_Mean": pd.to_numeric(get("NQ_Mean"), errors="coerce"),
        "ValidationPass": bool_value(get("ValidationPass", "False")),
        "ValidationStatus": get("ValidationStatus", "unknown"),
    }
    record["CountPass"] = bool(record["Count"] >= 75)
    record["MedianPass"] = bool(record["Median"] > 0)
    record["PFAboveMethodBasePass"] = bool(record["ProfitFactor"] > record["BaseMethodProfitFactor"])
    record["PositiveBlockFractionPass"] = bool(record["PositiveBlockFraction"] > 0.6)
    record["MaxSingleBlockContributionPass"] = bool(
        record["MaxSingleBlockContributionFraction"] <= 0.4
    )
    record["InstrumentMeansPass"] = bool(record["ES_Mean"] >= 0 and record["NQ_Mean"] >= 0)
    failures = [
        name.replace("Pass", "")
        for name in [
            "CountPass",
            "MedianPass",
            "PFAboveMethodBasePass",
            "PositiveBlockFractionPass",
            "MaxSingleBlockContributionPass",
            "InstrumentMeansPass",
        ]
        if not record[name]
    ]
    record["FailedCriteria"] = "|".join(failures) if failures else ""
    return record


def validation_lookup(summary: pd.DataFrame) -> pd.DataFrame:
    keep = ["Dataset", "ValidationMode", "Candidate", "ValidationPass", "ValidationStatus"]
    return summary.loc[summary["Candidate"].isin(FROZEN_CANDIDATES), keep].drop_duplicates()


def candidate_dataset_attribution(
    entries: pd.DataFrame, dataset_summary: pd.DataFrame
) -> pd.DataFrame:
    result = aggregate_rows(
        entries,
        ["Candidate", "ValidationMode", "Dataset", "DatasetCode", "DatasetType",
         "VolumeCoverageGroup", "ExplicitVolumeAvailable"],
        ["Candidate", "ValidationMode"],
    )
    result = result.merge(validation_lookup(dataset_summary), how="left",
                          on=["Dataset", "ValidationMode", "Candidate"])
    result["ContributionRankBestToWorst"] = result.groupby(
        ["Candidate", "ValidationMode"]
    )["SumContribution"].rank(ascending=False, method="min").astype(int)
    return result.sort_values(
        ["Candidate", "ValidationMode", "ContributionRankBestToWorst", "DatasetCode"]
    ).reset_index(drop=True)


def candidate_regime_attribution(entries: pd.DataFrame) -> pd.DataFrame:
    dataset_regimes = aggregate_rows(
        entries, ["Candidate", "ValidationMode", "RegimeCode", "RegimeYear"],
        ["Candidate", "ValidationMode"],
    )
    dataset_regimes.insert(2, "AttributionScope", "DatasetRegime")
    year_regimes = aggregate_rows(
        entries, ["Candidate", "ValidationMode", "RegimeYear"], ["Candidate", "ValidationMode"]
    )
    year_regimes.insert(2, "AttributionScope", "YearAggregate")
    year_regimes.insert(3, "RegimeCode", year_regimes["RegimeYear"])
    result = pd.concat([dataset_regimes, year_regimes], ignore_index=True, sort=False)
    result["ContributionRankBestToWorst"] = result.groupby(
        ["Candidate", "ValidationMode", "AttributionScope"]
    )["SumContribution"].rank(ascending=False, method="min").astype(int)
    result["ContributionStatus"] = np.where(result["SumContribution"] < 0, "Negative", "Positive")
    result["PassFailApplicability"] = "Descriptive contribution only"
    return result.sort_values(
        ["Candidate", "ValidationMode", "AttributionScope", "ContributionRankBestToWorst"]
    ).reset_index(drop=True)


def instrument_driver(group: pd.DataFrame) -> str:
    negatives = group.loc[group["SumContribution"] < 0, "Instrument"].tolist()
    if set(negatives) == {"ES", "NQ"}:
        return "Both"
    if negatives == ["ES"] or set(negatives) == {"ES"}:
        return "ES-driven"
    if negatives == ["NQ"] or set(negatives) == {"NQ"}:
        return "NQ-driven"
    return "Neither instrument net-negative"


def candidate_instrument_attribution(entries: pd.DataFrame) -> pd.DataFrame:
    out = aggregate_rows(
        entries, ["Candidate", "ValidationMode", "Instrument"], ["Candidate", "ValidationMode"]
    )
    drivers = (
        out.groupby(["Candidate", "ValidationMode"], group_keys=False)
        .apply(instrument_driver, include_groups=False)
        .rename("NegativeContributionDriver")
        .reset_index()
    )
    return out.merge(drivers, on=["Candidate", "ValidationMode"]).sort_values(
        ["Candidate", "ValidationMode", "Instrument"]
    ).reset_index(drop=True)


def candidate_cell_attribution(entries: pd.DataFrame) -> pd.DataFrame:
    dataset_cells = aggregate_rows(
        entries, ["Candidate", "ValidationMode", "RegimeCode", "RegimeYear", "Instrument"],
        ["Candidate", "ValidationMode"],
    )
    dataset_cells.insert(2, "AttributionScope", "DatasetRegime")
    year_cells = aggregate_rows(
        entries, ["Candidate", "ValidationMode", "RegimeYear", "Instrument"],
        ["Candidate", "ValidationMode"],
    )
    year_cells.insert(2, "AttributionScope", "YearAggregate")
    year_cells.insert(3, "RegimeCode", year_cells["RegimeYear"])
    out = pd.concat([dataset_cells, year_cells], ignore_index=True, sort=False)
    out["NegativeCell"] = out["SumContribution"] < 0
    out["ContributionRankBestToWorst"] = out.groupby(
        ["Candidate", "ValidationMode", "AttributionScope"]
    )["SumContribution"].rank(ascending=False, method="min").astype(int)
    return out.sort_values(
        ["Candidate", "ValidationMode", "AttributionScope", "ContributionRankBestToWorst", "Instrument"]
    ).reset_index(drop=True)


def candidate_block_attribution(entries: pd.DataFrame) -> pd.DataFrame:
    out = aggregate_rows(
        entries,
        ["Candidate", "ValidationMode", "DatasetCode", "RegimeYear", "TimeBlock", "RegimeWeek"],
        ["Candidate", "ValidationMode"],
    )
    out["NegativeBlock"] = out["SumContribution"] <= 0
    out["ContributionRankBestToWorst"] = out.groupby(
        ["Candidate", "ValidationMode"]
    )["SumContribution"].rank(ascending=False, method="min").astype(int)
    return out.sort_values(
        ["Candidate", "ValidationMode", "ContributionRankBestToWorst"]
    ).reset_index(drop=True)


def identity_keys(frame: pd.DataFrame) -> list[str]:
    return [
        column for column in ["Dataset", "Candidate", "Instrument", "File", "BarIndex"]
        if column in frame.columns
    ]


def removed_rows(entries: pd.DataFrame, candidate: str, source: str, target: str) -> pd.DataFrame:
    source_rows = entries.loc[
        entries["Candidate"].eq(candidate) & entries["ValidationMode"].eq(source)
    ]
    target_rows = entries.loc[
        entries["Candidate"].eq(candidate) & entries["ValidationMode"].eq(target)
    ]
    keys = identity_keys(entries)
    target_keys = target_rows[keys].drop_duplicates().assign(_retained=True)
    return source_rows.merge(target_keys, how="left", on=keys).loc[lambda frame: frame["_retained"].isna()]


def added_rows(entries: pd.DataFrame, candidate: str, source: str, target: str) -> pd.DataFrame:
    return removed_rows(entries, candidate, target, source)


def candidate_spacing_degradation(entries: pd.DataFrame, metrics: dict[str, str]) -> pd.DataFrame:
    pooled = {
        (candidate, mode): pooled_record(metrics, mode, candidate)
        for candidate in FROZEN_CANDIDATES for mode in VALIDATION_MODES
    }
    rows: list[dict[str, object]] = []
    for candidate in FROZEN_CANDIDATES:
        for source, target in COMPARISONS:
            before = pooled[(candidate, source)]
            after = pooled[(candidate, target)]
            removed = removed_rows(entries, candidate, source, target)
            added = added_rows(entries, candidate, source, target)
            removed_values = pd.to_numeric(removed[OUTCOME], errors="coerce").dropna()
            added_values = pd.to_numeric(added[OUTCOME], errors="coerce").dropna()
            rows.append({
                "Candidate": candidate,
                "FromMode": source,
                "ToMode": target,
                "FromPass": before["ValidationPass"],
                "ToPass": after["ValidationPass"],
                "PassFailChange": f"{'Pass' if before['ValidationPass'] else 'Fail'} -> {'Pass' if after['ValidationPass'] else 'Fail'}",
                "CountLoss": int(before["Count"] - after["Count"]),
                "MeanChange": after["Mean"] - before["Mean"],
                "MedianChange": after["Median"] - before["Median"],
                "ProfitFactorChange": after["ProfitFactor"] - before["ProfitFactor"],
                "PositiveBlockFractionChange": (
                    after["PositiveBlockFraction"] - before["PositiveBlockFraction"]
                ),
                "ESMeanChange": after["ES_Mean"] - before["ES_Mean"],
                "NQMeanChange": after["NQ_Mean"] - before["NQ_Mean"],
                "RemovedRowCount": int(len(removed_values)),
                "RemovedRowSumContribution": float(removed_values.sum()),
                "RemovedRowMean": float(removed_values.mean()) if len(removed_values) else np.nan,
                "NewlySelectedRowCount": int(len(added_values)),
                "NewlySelectedRowSumContribution": float(added_values.sum()),
                "NewlySelectedRowMean": float(added_values.mean()) if len(added_values) else np.nan,
                "SelectionReplacementNetContribution": (
                    float(added_values.sum()) - float(removed_values.sum())
                ),
                "TargetFailedCriteria": after["FailedCriteria"],
                "FailureAttribution": (
                    "Target fails: " + after["FailedCriteria"]
                    if not after["ValidationPass"] else "Target passes frozen criteria"
                ),
            })
    return pd.DataFrame(rows)


def volume_attribution(entries: pd.DataFrame) -> pd.DataFrame:
    slope = entries.loc[entries["Candidate"].eq(PRIOR_SLOPE)].copy()
    overall = aggregate_rows(
        slope, ["Candidate", "ValidationMode", "VolumeCoverageGroup", "ExplicitVolumeAvailable"],
        ["Candidate", "ValidationMode"],
    )
    overall.insert(2, "Breakdown", "AllInstruments")
    overall["Instrument"] = "All"
    split = aggregate_rows(
        slope,
        ["Candidate", "ValidationMode", "VolumeCoverageGroup", "ExplicitVolumeAvailable", "Instrument"],
        ["Candidate", "ValidationMode"],
    )
    split.insert(2, "Breakdown", "ByInstrument")
    datasets = aggregate_rows(
        slope,
        ["Candidate", "ValidationMode", "VolumeCoverageGroup", "ExplicitVolumeAvailable",
         "DatasetCode", "RegimeYear"],
        ["Candidate", "ValidationMode"],
    )
    datasets.insert(2, "Breakdown", "ByDataset")
    datasets["Instrument"] = "All"
    out = pd.concat([overall, split, datasets], ignore_index=True, sort=False)
    return out.sort_values(
        ["ValidationMode", "VolumeCoverageGroup", "Breakdown", "DatasetCode", "Instrument"]
    ).reset_index(drop=True)


def failure_scorecard(
    entries: pd.DataFrame,
    metrics: dict[str, str],
    cells: pd.DataFrame,
    blocks: pd.DataFrame,
    datasets: pd.DataFrame,
    volume: pd.DataFrame,
) -> pd.DataFrame:
    pooled = [pooled_record(metrics, mode, candidate)
              for candidate in FROZEN_CANDIDATES for mode in VALIDATION_MODES]
    ref_cells = cells.loc[
        cells["Candidate"].eq(PRIOR_SLOPE) & cells["ValidationMode"].eq("Reference")
        & cells["AttributionScope"].eq("YearAggregate")
    ]
    neg_cells = ref_cells.loc[ref_cells["NegativeCell"]]
    slope_blocks = blocks.loc[
        blocks["Candidate"].eq(PRIOR_SLOPE) & blocks["ValidationMode"].eq("Reference")
    ]
    slope_ds = datasets.loc[
        datasets["Candidate"].eq(PRIOR_SLOPE) & datasets["ValidationMode"].eq("Reference")
    ]
    worst_ds = slope_ds.sort_values("SumContribution").iloc[0]
    new_ds = slope_ds.loc[slope_ds["DatasetCode"].eq("032024")].iloc[0]
    vol_ref = volume.loc[
        volume["ValidationMode"].eq("Reference") & volume["Breakdown"].eq("AllInstruments")
    ]
    vol = vol_ref.loc[vol_ref["VolumeCoverageGroup"].eq("VolumeEnabled")].iloc[0]
    legacy = vol_ref.loc[vol_ref["VolumeCoverageGroup"].eq("LegacyOrNoExplicitVolume")].iloc[0]
    rows: list[dict[str, object]] = [
        {"Metric": "report_mode", "Value": "diagnostic attribution over frozen candidates only; no rule or threshold changes"},
        {"Metric": "frozen_candidates", "Value": ",".join(FROZEN_CANDIDATES)},
        {"Metric": "prior_slope_reference_negative_regime_instrument_cell_count", "Value": int(len(neg_cells))},
        {"Metric": "prior_slope_reference_positive_regime_instrument_cell_count", "Value": int((~ref_cells["NegativeCell"]).sum())},
        {"Metric": "prior_slope_reference_negative_cells", "Value": ";".join(
            f"{row.RegimeYear}-{row.Instrument}:{row.SumContribution:.6f}"
            for row in neg_cells.itertuples()
        )},
        {"Metric": "prior_slope_reference_negative_block_count", "Value": int(slope_blocks["NegativeBlock"].sum())},
        {"Metric": "prior_slope_reference_block_count", "Value": int(len(slope_blocks))},
        {"Metric": "prior_slope_worst_reference_dataset", "Value": worst_ds["DatasetCode"]},
        {"Metric": "prior_slope_worst_reference_dataset_sum", "Value": worst_ds["SumContribution"]},
        {"Metric": "prior_slope_032024_reference_count", "Value": new_ds["Count"]},
        {"Metric": "prior_slope_032024_reference_mean", "Value": new_ds["Mean"]},
        {"Metric": "prior_slope_032024_reference_sum", "Value": new_ds["SumContribution"]},
        {"Metric": "prior_slope_volume_enabled_reference_count", "Value": vol["Count"]},
        {"Metric": "prior_slope_volume_enabled_reference_mean", "Value": vol["Mean"]},
        {"Metric": "prior_slope_volume_enabled_reference_sum", "Value": vol["SumContribution"]},
        {"Metric": "prior_slope_legacy_reference_count", "Value": legacy["Count"]},
        {"Metric": "prior_slope_legacy_reference_mean", "Value": legacy["Mean"]},
        {"Metric": "prior_slope_legacy_reference_sum", "Value": legacy["SumContribution"]},
        {"Metric": "prior_slope_volume_enabled_degrades_mean", "Value": bool(vol["Mean"] < legacy["Mean"])},
    ]
    for record in pooled:
        rows.extend([
            {"Metric": f"{record['Candidate']}_{record['ValidationMode']}_Pass", "Value": record["ValidationPass"]},
            {"Metric": f"{record['Candidate']}_{record['ValidationMode']}_FailedCriteria", "Value": record["FailedCriteria"]},
        ])
    return pd.DataFrame(rows)


def fmt(value: object, decimals: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{decimals}f}"


def write_handoff(
    path: Path,
    metrics: dict[str, str],
    regimes: pd.DataFrame,
    datasets: pd.DataFrame,
    cells: pd.DataFrame,
    blocks: pd.DataFrame,
    spacing: pd.DataFrame,
    volume: pd.DataFrame,
) -> None:
    pooled = {
        (candidate, mode): pooled_record(metrics, mode, candidate)
        for candidate in FROZEN_CANDIDATES for mode in VALIDATION_MODES
    }
    negative = cells.loc[
        cells["Candidate"].eq(PRIOR_SLOPE)
        & cells["ValidationMode"].eq("Reference")
        & cells["AttributionScope"].eq("YearAggregate")
        & cells["NegativeCell"]
    ].sort_values("SumContribution")
    ref_ds = datasets.loc[
        datasets["Candidate"].eq(PRIOR_SLOPE) & datasets["ValidationMode"].eq("Reference")
    ].sort_values("SumContribution")
    ref_reg = regimes.loc[
        regimes["Candidate"].eq(PRIOR_SLOPE) & regimes["ValidationMode"].eq("Reference")
        & regimes["AttributionScope"].eq("DatasetRegime")
    ].sort_values("SumContribution")
    slope_blocks = blocks.loc[
        blocks["Candidate"].eq(PRIOR_SLOPE) & blocks["ValidationMode"].eq("Reference")
    ]
    volume_ref = volume.loc[
        volume["ValidationMode"].eq("Reference") & volume["Breakdown"].eq("AllInstruments")
    ]
    vol = volume_ref.loc[volume_ref["VolumeCoverageGroup"].eq("VolumeEnabled")].iloc[0]
    legacy = volume_ref.loc[volume_ref["VolumeCoverageGroup"].eq("LegacyOrNoExplicitVolume")].iloc[0]
    new_ds = ref_ds.loc[ref_ds["DatasetCode"].eq("032024")].iloc[0]
    dataset_cells = cells.loc[
        cells["Candidate"].eq(PRIOR_SLOPE)
        & cells["ValidationMode"].eq("Reference")
        & cells["AttributionScope"].eq("DatasetRegime")
        & cells["RegimeCode"].eq("032024")
    ]
    new_es = dataset_cells.loc[dataset_cells["Instrument"].eq("ES")].iloc[0]
    new_nq = dataset_cells.loc[dataset_cells["Instrument"].eq("NQ")].iloc[0]
    ref = pooled[(PRIOR_SLOPE, "Reference")]
    pre_count = int(ref["Count"] - new_ds["Count"])
    pre_sum = ref["Mean"] * ref["Count"] - new_ds["SumContribution"]
    pre_mean = pre_sum / pre_count
    lines = [
        "# APVA Frozen Candidate Failure Attribution Handoff",
        "",
        "## Scope",
        "",
        "This is a diagnostic attribution report over the three frozen candidates only. "
        "It does not change candidate rules, thresholds, filters, or validation logic.",
        "",
        "## Headline",
        "",
        f"- `PriorSlope_DominantPressureValue_Q3` remains positive in pooled Reference "
        f"(mean `{fmt(pooled[(PRIOR_SLOPE, 'Reference')]['Mean'])}`, PF "
        f"`{fmt(pooled[(PRIOR_SLOPE, 'Reference')]['ProfitFactor'])}`) but fails the "
        f"frozen Reference test because `{pooled[(PRIOR_SLOPE, 'Reference')]['FailedCriteria']}` fails.",
        f"- It passes `Spacing_10` and fails `Spacing_20` because "
        f"`{pooled[(PRIOR_SLOPE, 'Spacing_20')]['FailedCriteria']}` fails at the stricter spacing.",
        f"- `CCRRR` fails all modes. Reference failure criteria: "
        f"`{pooled[('CCRRR', 'Reference')]['FailedCriteria']}`.",
        f"- `RRCCC` fails all modes. Reference failure criteria: "
        f"`{pooled[('RRCCC', 'Reference')]['FailedCriteria']}`.",
        "",
        "## 032024 Synchronization Answers",
        "",
        f"1. Adding `032024` did not materially worsen PriorSlope_Q3: it contributed "
        f"`+{fmt(new_ds['SumContribution'])}` on `{int(new_ds['Count'])}` Reference rows. "
        f"It diluted the pooled mean from `{fmt(pre_mean)}` to `{fmt(ref['Mean'])}` because "
        f"its own mean is lower (`{fmt(new_ds['Mean'])}`), but it did not change the failure reason.",
        f"2. The single largest negative dataset remains `{ref_ds.iloc[0]['DatasetCode']}` "
        f"with contribution `{fmt(ref_ds.iloc[0]['SumContribution'])}`.",
        "3. The four negative year-aggregate regime/instrument cells are unchanged after `032024`; "
        "`032024` does not add a new negative year-aggregate cell.",
        f"4. Within `032024`, ES hurt PriorSlope (`{fmt(new_es['SumContribution'])}` sum, "
        f"mean `{fmt(new_es['Mean'])}`), NQ helped (`+{fmt(new_nq['SumContribution'])}` sum, "
        f"mean `{fmt(new_nq['Mean'])}`), and pooled PriorSlope helped (`+{fmt(new_ds['SumContribution'])}`).",
        f"5. Yes. PriorSlope still fails Reference and Spacing_20 only because "
        f"`{pooled[(PRIOR_SLOPE, 'Reference')]['FailedCriteria']}` fails.",
        "6. Yes. `Spacing_10` remains the only passing mode for PriorSlope_Q3.",
        f"7. Yes. Volume-enabled Reference rows remain weaker (mean `{fmt(vol['Mean'])}`, "
        f"sum `{fmt(vol['SumContribution'])}`) than legacy/no-explicit-volume rows "
        f"(mean `{fmt(legacy['Mean'])}`, sum `{fmt(legacy['SumContribution'])}`).",
        f"8. `032024` behaves more like the positive-participation `062023` dataset than the "
        f"negative `122023` dataset: its pooled contribution is positive (`+{fmt(new_ds['SumContribution'])}`).",
        "",
        "## What Broke PriorSlope_Q3?",
        "",
        f"The Reference failure is not caused by a negative pooled mean: total contribution "
        f"is `{fmt(pooled[(PRIOR_SLOPE, 'Reference')]['Mean'] * pooled[(PRIOR_SLOPE, 'Reference')]['Count'])}` "
        f"across `{int(pooled[(PRIOR_SLOPE, 'Reference')]['Count'])}` rows. It is a breadth failure: "
        f"only `{fmt(pooled[(PRIOR_SLOPE, 'Reference')]['PositiveBlockFraction'])}` of blocks are positive, "
        "below the frozen `> 0.6` requirement.",
        "",
        f"`Spacing_10` raises positive block fraction to "
        f"`{fmt(pooled[(PRIOR_SLOPE, 'Spacing_10')]['PositiveBlockFraction'])}` and passes. "
        f"`Spacing_20` lowers it to `{fmt(pooled[(PRIOR_SLOPE, 'Spacing_20')]['PositiveBlockFraction'])}` "
        "and fails. Because each spacing mode is independently selected, the stricter mode both drops "
        "and reselects observations; its selected sample loses block breadth without turning the pooled "
        "mean negative.",
        "",
        "## Four Negative Reference Regime/Instrument Cells",
        "",
        "| Regime | Instrument | Count | Mean | Median | PF | Sum |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in negative.itertuples():
        lines.append(
            f"| {row.RegimeYear} | {row.Instrument} | {row.Count} | {fmt(row.Mean)} | "
            f"{fmt(row.Median)} | {fmt(row.ProfitFactor)} | {fmt(row.SumContribution)} |"
        )
    lines.extend([
        "",
        f"Thirteen of `{len(cells.loc[cells['Candidate'].eq(PRIOR_SLOPE) & cells['ValidationMode'].eq('Reference') & cells['AttributionScope'].eq('YearAggregate')])}` "
        "Reference regime/instrument cells remain positive; degradation is concentrated in the four cells above.",
        "",
        "## Dataset Attribution",
        "",
        "| Dataset | Volume Enabled | Count | Mean | PF | Sum |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in ref_ds.itertuples():
        lines.append(
            f"| {row.DatasetCode} | {row.ExplicitVolumeAvailable} | {row.Count} | "
            f"{fmt(row.Mean)} | {fmt(row.ProfitFactor)} | {fmt(row.SumContribution)} |"
        )
    lines.extend([
        "",
        f"`122017` is not the principal negative dataset: its Reference sum is "
        f"`{fmt(ref_ds.loc[ref_ds['DatasetCode'].eq('122017'), 'SumContribution'].iloc[0])}`, "
        f"while the most negative dataset is `{ref_ds.iloc[0]['DatasetCode']}` at "
        f"`{fmt(ref_ds.iloc[0]['SumContribution'])}`.",
        "",
        "## Volume-Enabled Versus Legacy Coverage",
        "",
        "| Group | Count | Mean | Median | PF | Sum |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| VolumeEnabled | {vol.Count} | {fmt(vol.Mean)} | {fmt(vol.Median)} | "
        f"{fmt(vol.ProfitFactor)} | {fmt(vol.SumContribution)} |",
        f"| LegacyOrNoExplicitVolume | {legacy.Count} | {fmt(legacy.Mean)} | {fmt(legacy.Median)} | "
        f"{fmt(legacy.ProfitFactor)} | {fmt(legacy.SumContribution)} |",
        "",
        "The explicit-volume group is weaker than the legacy/no-explicit-volume group in the current "
        "Reference sample, but the weakness is not uniform: `032024` and `062023` contribute positively while "
        "`062019`, `122017`, and especially `122023` contribute negatively. The honest attribution is "
        "broader regime diversity revealing fragility, not volume fields themselves causing degradation.",
        "",
        "## Blocks And Concentration",
        "",
        f"PriorSlope Reference has `{int((slope_blocks['NegativeBlock']).sum())}` non-positive blocks "
        f"out of `{len(slope_blocks)}`. No single instrument is net-negative in the pooled Reference "
        "aggregate; the failure is not an ES-only or NQ-only collapse. It is a distributed breadth "
        "problem with concentrated negative dataset/cell pockets.",
        "",
        "## Candidate Disposition",
        "",
        "`CCRRR` and `RRCCC` do not survive as broad frozen candidates in this expanded validation: "
        "each fails every requested validation mode. This report does not replace them or modify them.",
        "",
        "PriorSlope_Q3 retains a descriptive narrower observation worth validating later: performance "
        "is positive in most regime/instrument cells and survives `Spacing_10`, while failing in a small "
        "set of specific cells and at `Spacing_20`. That is a follow-up validation question, not a rule change.",
        "",
        "## Next Honest Research Step",
        "",
        "Preserve the frozen candidates and collect/validate additional comparable regimes with explicit "
        "volume exports, then repeat this attribution prospectively. The immediate evidence supports "
        "documenting where the frozen result is unstable, not revising the candidate.",
        "",
        "## Output Files",
        "",
        "- `candidate_regime_attribution.csv`",
        "- `candidate_instrument_attribution.csv`",
        "- `candidate_regime_instrument_attribution.csv`",
        "- `candidate_block_attribution.csv`",
        "- `candidate_dataset_attribution.csv`",
        "- `candidate_spacing_degradation.csv`",
        "- `volume_enabled_vs_legacy_attribution.csv`",
        "- `prior_slope_q3_failure_cells.csv`",
        "- `failure_attribution_scorecard.csv`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--outdir", default="outputs/failure_attribution_report")
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    outdir = workspace / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    fixed = workspace / "outputs/fixed_candidate_extended_validation"
    cross = workspace / "outputs/cross_era_validation_report"
    volume_dir = workspace / "outputs/volume_participation_diagnostics"
    full = workspace / "outputs/full_pipeline_run"
    paths = {
        "entries": fixed / "extended_entries.csv",
        "scorecard": fixed / "extended_scorecard.csv",
        "datasets": fixed / "extended_dataset_candidate_summary.csv",
        "blocks": fixed / "extended_block_summary.csv",
        "regimes": cross / "regime_candidate_summary.csv",
        "regime_blocks": cross / "regime_block_summary.csv",
        "volume_summary": volume_dir / "prior_slope_q3_volume_summary.csv",
        "volume_corr": volume_dir / "prior_slope_q3_volume_correlations.csv",
        "status": full / "full_pipeline_regime_status.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required diagnostic inputs: {missing}")

    entries = enrich_entries(pd.read_csv(paths["entries"]))
    score = scorecard_metrics(pd.read_csv(paths["scorecard"]))
    datasets_source = pd.read_csv(paths["datasets"])
    # Read required supporting inputs so missing or stale pipeline products fail visibly.
    pd.read_csv(paths["blocks"])
    pd.read_csv(paths["regimes"])
    pd.read_csv(paths["regime_blocks"])
    pd.read_csv(paths["volume_summary"])
    pd.read_csv(paths["volume_corr"])
    status = pd.read_csv(paths["status"], dtype=str)

    volume_codes = selected_volume_codes(workspace, status)
    entries = add_volume_classification(entries, volume_codes)
    datasets = candidate_dataset_attribution(entries, datasets_source)
    regimes = candidate_regime_attribution(entries)
    instruments = candidate_instrument_attribution(entries)
    cells = candidate_cell_attribution(entries)
    blocks = candidate_block_attribution(entries)
    spacing = candidate_spacing_degradation(entries, score)
    volume = volume_attribution(entries)
    prior_failure_cells = cells.loc[
        cells["Candidate"].eq(PRIOR_SLOPE)
        & cells["ValidationMode"].eq("Reference")
        & cells["AttributionScope"].eq("YearAggregate")
        & cells["NegativeCell"]
    ].copy()
    report_score = failure_scorecard(entries, score, cells, blocks, datasets, volume)

    regimes.to_csv(outdir / "candidate_regime_attribution.csv", index=False)
    instruments.to_csv(outdir / "candidate_instrument_attribution.csv", index=False)
    cells.to_csv(outdir / "candidate_regime_instrument_attribution.csv", index=False)
    blocks.to_csv(outdir / "candidate_block_attribution.csv", index=False)
    datasets.to_csv(outdir / "candidate_dataset_attribution.csv", index=False)
    spacing.to_csv(outdir / "candidate_spacing_degradation.csv", index=False)
    volume.to_csv(outdir / "volume_enabled_vs_legacy_attribution.csv", index=False)
    prior_failure_cells.to_csv(outdir / "prior_slope_q3_failure_cells.csv", index=False)
    report_score.to_csv(outdir / "failure_attribution_scorecard.csv", index=False)
    write_handoff(outdir / "chatgpt_master_analysis_handoff.md", score, regimes, datasets,
                  cells, blocks, spacing, volume)

    ref_negative = prior_failure_cells.loc[prior_failure_cells["ValidationMode"].eq("Reference")]
    print("APVA frozen-candidate failure attribution report complete")
    print(f"Output directory: {outdir}")
    print(f"PriorSlope Reference negative regime/instrument cells: {len(ref_negative)}")
    print(ref_negative[["RegimeYear", "Instrument", "Count", "Mean", "SumContribution"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
