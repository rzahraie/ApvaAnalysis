from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# APVA edge inventory / survivorship ranking V1
# ---------------------------------------------------------
# Purpose:
#   Combine multiple validation layers into one ranked inventory:
#
#   1. In-sample signed return expectancy
#   2. Walk-forward validation expectancy
#   3. Non-overlapping offset robustness
#
# This script is intentionally conservative. It penalizes edges that:
#   - appear only in-sample,
#   - fail validation,
#   - fail non-overlap testing,
#   - have low support,
#   - or show poor offset robustness.
#
# Inputs expected:
#   tables/apva_forward_signed_return_expectancy_v1.csv
#   tables/apva_walk_forward_supported_edges_v1.csv
#   tables/apva_nonoverlap_signed_return_edge_summary_v1.csv
#
# Outputs:
#   tables/apva_edge_inventory_survivorship_ranking_v1.csv
#   tables/apva_edge_inventory_top_survivors_v1.csv
#   tables/apva_edge_inventory_failed_edges_v1.csv
#   tables/apva_edge_inventory_summary_v1.csv
#   figures/apva_edge_inventory_survivorship_score_v1.png
#   figures/apva_edge_inventory_validation_vs_nonoverlap_v1.png
#   figures/apva_edge_inventory_class_counts_v1.png

BASE_DIR = Path(r"C:\Users\rz0\Documents\ApvaAnalysis")
TABLE_DIR = BASE_DIR / "tables"
FIG_DIR = BASE_DIR / "figures"

TABLE_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# LOAD INPUTS
# ---------------------------------------------------------

insample_path = TABLE_DIR / "apva_forward_signed_return_expectancy_v1.csv"
wf_path = TABLE_DIR / "apva_walk_forward_supported_edges_v1.csv"
nonoverlap_path = TABLE_DIR / "apva_nonoverlap_signed_return_edge_summary_v1.csv"

missing_paths = [
    str(p) for p in [insample_path, wf_path, nonoverlap_path]
    if not p.exists()
]

if missing_paths:
    raise FileNotFoundError(
        "Missing required input files:\n" + "\n".join(missing_paths)
    )

insample = pd.read_csv(insample_path)
wf = pd.read_csv(wf_path)
nonoverlap = pd.read_csv(nonoverlap_path)

edge_keys = [
    "Instrument",
    "HorizonBars",
    "ResolvedArchetype",
    "DominantPressure",
]

# ---------------------------------------------------------
# NORMALIZE / SELECT INSAMPLE COLUMNS
# ---------------------------------------------------------

insample_cols = edge_keys + [
    "Count",
    "MeanSignedReturn",
    "MedianSignedReturn",
    "DirectionalHitRate",
    "MeanDirectionalMFE",
    "MeanDirectionalMAE",
    "SignedReturnTStat",
    "SignedReturnSignificance",
]

insample_missing = sorted(set(insample_cols) - set(insample.columns))
if insample_missing:
    raise ValueError(f"In-sample table missing columns: {insample_missing}")

ins = insample[insample_cols].copy()
ins = ins.rename(columns={
    "Count": "InSampleSupport",
    "MeanSignedReturn": "InSampleMeanReturn",
    "MedianSignedReturn": "InSampleMedianReturn",
    "DirectionalHitRate": "InSampleHitRate",
    "MeanDirectionalMFE": "InSampleMFE",
    "MeanDirectionalMAE": "InSampleMAE",
    "SignedReturnTStat": "InSampleTStat",
    "SignedReturnSignificance": "InSampleSignificance",
})

# ---------------------------------------------------------
# NORMALIZE / SELECT WALK-FORWARD COLUMNS
# ---------------------------------------------------------

wf_cols = edge_keys + [
    "Count_Train",
    "MeanSignedReturn_Train",
    "DirectionalHitRate_Train",
    "TrainTStat",
    "Count_Validation",
    "MeanSignedReturn_Validation",
    "DirectionalHitRate_Validation",
    "ValidationTStat",
    "GeneralizationClass",
    "ValidationRetainsPositiveEdge",
    "ValidationFlipsNegative",
]

wf_missing = sorted(set(wf_cols) - set(wf.columns))
if wf_missing:
    raise ValueError(f"Walk-forward table missing columns: {wf_missing}")

w = wf[wf_cols].copy()
w = w.rename(columns={
    "Count_Train": "WFTrainSupport",
    "MeanSignedReturn_Train": "WFTrainMeanReturn",
    "DirectionalHitRate_Train": "WFTrainHitRate",
    "TrainTStat": "WFTrainTStat",
    "Count_Validation": "WFValidationSupport",
    "MeanSignedReturn_Validation": "WFValidationMeanReturn",
    "DirectionalHitRate_Validation": "WFValidationHitRate",
    "ValidationTStat": "WFValidationTStat",
    "GeneralizationClass": "WFGeneralizationClass",
})

# ---------------------------------------------------------
# NORMALIZE / SELECT NON-OVERLAP COLUMNS
# ---------------------------------------------------------

non_cols = edge_keys + [
    "OffsetCount",
    "TotalSupport",
    "MeanOfOffsetReturns",
    "MedianOfOffsetReturns",
    "MeanOffsetTStat",
    "MaxOffsetTStat",
    "MinOffsetTStat",
    "MeanDirectionalHitRate",
    "MeanDirectionalMFE",
    "MeanDirectionalMAE",
    "PositiveOffsetFraction",
    "NegativeOffsetFraction",
    "NonOverlapRobustnessClass",
]

non_missing = sorted(set(non_cols) - set(nonoverlap.columns))
if non_missing:
    raise ValueError(f"Non-overlap table missing columns: {non_missing}")

n = nonoverlap[non_cols].copy()
n = n.rename(columns={
    "OffsetCount": "NOOffsetCount",
    "TotalSupport": "NOTotalSupport",
    "MeanOfOffsetReturns": "NOMeanReturn",
    "MedianOfOffsetReturns": "NOMedianReturn",
    "MeanOffsetTStat": "NOMeanTStat",
    "MaxOffsetTStat": "NOMaxTStat",
    "MinOffsetTStat": "NOMinTStat",
    "MeanDirectionalHitRate": "NOHitRate",
    "MeanDirectionalMFE": "NOMFE",
    "MeanDirectionalMAE": "NOMAE",
    "PositiveOffsetFraction": "NOPositiveOffsetFraction",
    "NegativeOffsetFraction": "NONegativeOffsetFraction",
    "NonOverlapRobustnessClass": "NORobustnessClass",
})

# ---------------------------------------------------------
# MERGE INVENTORY
# ---------------------------------------------------------

inventory = ins.merge(w, on=edge_keys, how="outer")
inventory = inventory.merge(n, on=edge_keys, how="outer")

# ---------------------------------------------------------
# FILL DEFAULTS
# ---------------------------------------------------------

numeric_defaults = {
    "InSampleSupport": 0,
    "InSampleMeanReturn": 0.0,
    "InSampleMedianReturn": 0.0,
    "InSampleHitRate": 0.0,
    "InSampleMFE": 0.0,
    "InSampleMAE": 0.0,
    "InSampleTStat": 0.0,
    "WFTrainSupport": 0,
    "WFTrainMeanReturn": 0.0,
    "WFTrainHitRate": 0.0,
    "WFTrainTStat": 0.0,
    "WFValidationSupport": 0,
    "WFValidationMeanReturn": 0.0,
    "WFValidationHitRate": 0.0,
    "WFValidationTStat": 0.0,
    "NOOffsetCount": 0,
    "NOTotalSupport": 0,
    "NOMeanReturn": 0.0,
    "NOMedianReturn": 0.0,
    "NOMeanTStat": 0.0,
    "NOMaxTStat": 0.0,
    "NOMinTStat": 0.0,
    "NOHitRate": 0.0,
    "NOMFE": 0.0,
    "NOMAE": 0.0,
    "NOPositiveOffsetFraction": 0.0,
    "NONegativeOffsetFraction": 0.0,
}

for col, value in numeric_defaults.items():
    if col in inventory.columns:
        inventory[col] = inventory[col].fillna(value)

for col in ["InSampleSignificance", "WFGeneralizationClass", "NORobustnessClass"]:
    if col in inventory.columns:
        inventory[col] = inventory[col].fillna("Missing")

for col in ["ValidationRetainsPositiveEdge", "ValidationFlipsNegative"]:
    if col in inventory.columns:
        inventory[col] = inventory[col].fillna(False).astype(bool)

# ---------------------------------------------------------
# SCORING HELPERS
# ---------------------------------------------------------

def clip_score(x, low=-3.0, high=3.0):
    if pd.isna(x):
        return 0.0
    return float(np.clip(x, low, high))


def support_score(n, cap=500.0):
    if pd.isna(n) or n <= 0:
        return 0.0
    return float(min(1.0, np.log1p(n) / np.log1p(cap)))


def class_score(value, mapping):
    return mapping.get(str(value), 0.0)

wf_map = {
    "ConfirmedPositive": 3.0,
    "WeakPositive": 2.0,
    "PositiveButWeak": 1.0,
    "ConfirmedNegative": -3.0,
    "WeakNegative": -2.0,
    "NegativeButWeak": -1.0,
    "FailedPositive": -3.0,
    "FailedNegative": 1.0,
    "Mixed": 0.0,
    "Missing": 0.0,
}

no_map = {
    "RobustPositive": 4.0,
    "WeakRobustPositive": 2.5,
    "RobustNegative": -4.0,
    "WeakRobustNegative": -2.5,
    "NotRobust": 0.0,
    "Unsupported": 0.0,
    "Missing": 0.0,
}

insample_map = {
    "SignificantPositiveSignedReturn": 1.5,
    "SignificantNegativeSignedReturn": -1.5,
    "WeakEvidence": 0.5,
    "NotSignificant": 0.0,
    "Unsupported": 0.0,
    "Missing": 0.0,
}

# ---------------------------------------------------------
# COMPONENT SCORES
# ---------------------------------------------------------

inventory["InSampleComponent"] = inventory.apply(
    lambda r: (
        class_score(r["InSampleSignificance"], insample_map)
        + 0.25 * clip_score(r["InSampleTStat"])
        + 0.50 * np.sign(r["InSampleMeanReturn"]) * support_score(r["InSampleSupport"])
    ),
    axis=1,
)

inventory["WalkForwardComponent"] = inventory.apply(
    lambda r: (
        class_score(r["WFGeneralizationClass"], wf_map)
        + 0.50 * clip_score(r["WFValidationTStat"])
        + 1.00 * np.sign(r["WFValidationMeanReturn"]) * support_score(r["WFValidationSupport"])
    ),
    axis=1,
)

inventory["NonOverlapComponent"] = inventory.apply(
    lambda r: (
        class_score(r["NORobustnessClass"], no_map)
        + 0.75 * clip_score(r["NOMeanTStat"])
        + 1.25 * np.sign(r["NOMeanReturn"]) * r["NOPositiveOffsetFraction"]
        - 1.25 * (1 if r["NOMeanReturn"] > 0 else 0) * r["NONegativeOffsetFraction"]
    ),
    axis=1,
)

# Penalize failure modes.
inventory["FailurePenalty"] = 0.0
inventory.loc[inventory["WFGeneralizationClass"] == "FailedPositive", "FailurePenalty"] -= 4.0
inventory.loc[inventory["WFGeneralizationClass"] == "FailedNegative", "FailurePenalty"] -= 1.0
inventory.loc[
    (inventory["InSampleMeanReturn"] > 0) & (inventory["NOMeanReturn"] < 0),
    "FailurePenalty"
] -= 2.0
inventory.loc[
    (inventory["WFValidationMeanReturn"] > 0) & (inventory["NOMeanReturn"] < 0),
    "FailurePenalty"
] -= 1.5

# Weighted total: non-overlap and walk-forward matter far more than in-sample.
inventory["SurvivorshipScore"] = (
    0.20 * inventory["InSampleComponent"]
    + 0.35 * inventory["WalkForwardComponent"]
    + 0.45 * inventory["NonOverlapComponent"]
    + inventory["FailurePenalty"]
)

# ---------------------------------------------------------
# SURVIVORSHIP CLASS
# ---------------------------------------------------------

def classify_survivor(row):
    score = row["SurvivorshipScore"]

    if score >= 4.0:
        return "HighConvictionPositive"
    if score >= 2.0:
        return "CandidatePositive"
    if score >= 0.75:
        return "WeakCandidatePositive"
    if score <= -4.0:
        return "HighConvictionNegative"
    if score <= -2.0:
        return "CandidateNegative"
    if score <= -0.75:
        return "WeakCandidateNegative"
    return "NeutralOrUnproven"

inventory["SurvivorshipClass"] = inventory.apply(classify_survivor, axis=1)

inventory = inventory.sort_values(
    ["SurvivorshipScore", "NOMeanReturn", "WFValidationMeanReturn", "InSampleMeanReturn"],
    ascending=[False, False, False, False],
)

# ---------------------------------------------------------
# EXPORTS
# ---------------------------------------------------------

inventory.to_csv(
    TABLE_DIR / "apva_edge_inventory_survivorship_ranking_v1.csv",
    index=False,
)

top_survivors = inventory[
    inventory["SurvivorshipClass"].isin([
        "HighConvictionPositive",
        "CandidatePositive",
        "WeakCandidatePositive",
    ])
].copy()

top_survivors.to_csv(
    TABLE_DIR / "apva_edge_inventory_top_survivors_v1.csv",
    index=False,
)

failed_edges = inventory[
    inventory["SurvivorshipClass"].isin([
        "HighConvictionNegative",
        "CandidateNegative",
        "WeakCandidateNegative",
    ])
].copy()

failed_edges.to_csv(
    TABLE_DIR / "apva_edge_inventory_failed_edges_v1.csv",
    index=False,
)

summary = (
    inventory
    .groupby(["Instrument", "HorizonBars", "SurvivorshipClass"])
    .agg(
        Rows=("SurvivorshipClass", "count"),
        MeanScore=("SurvivorshipScore", "mean"),
        MeanInSampleReturn=("InSampleMeanReturn", "mean"),
        MeanWFValidationReturn=("WFValidationMeanReturn", "mean"),
        MeanNOMeanReturn=("NOMeanReturn", "mean"),
        MeanNOPositiveOffsetFraction=("NOPositiveOffsetFraction", "mean"),
    )
    .reset_index()
)

summary.to_csv(
    TABLE_DIR / "apva_edge_inventory_summary_v1.csv",
    index=False,
)

# ---------------------------------------------------------
# FIGURES
# ---------------------------------------------------------

top_plot = inventory.head(25).copy()

plt.figure(figsize=(14, 7))
labels = (
    top_plot["Instrument"]
    + " | "
    + top_plot["HorizonBars"].astype(str)
    + " | "
    + top_plot["ResolvedArchetype"]
    + " | "
    + top_plot["DominantPressure"]
)

plt.bar(labels, top_plot["SurvivorshipScore"])
plt.xticks(rotation=75, ha="right")
plt.ylabel("SurvivorshipScore")
plt.title("APVA Edge Inventory Survivorship Ranking")
plt.tight_layout()
plt.savefig(
    FIG_DIR / "apva_edge_inventory_survivorship_score_v1.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# ---------------------------------------------------------

plt.figure(figsize=(11, 7))
plt.scatter(
    inventory["WFValidationMeanReturn"],
    inventory["NOMeanReturn"],
    c=inventory["SurvivorshipScore"],
    alpha=0.75,
)
plt.colorbar(label="SurvivorshipScore")
plt.axhline(0, linestyle="--")
plt.axvline(0, linestyle="--")
plt.xlabel("Walk-forward validation mean signed return")
plt.ylabel("Non-overlap mean signed return")
plt.title("APVA Validation vs Non-Overlap Return")
plt.tight_layout()
plt.savefig(
    FIG_DIR / "apva_edge_inventory_validation_vs_nonoverlap_v1.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# ---------------------------------------------------------

class_counts = inventory["SurvivorshipClass"].value_counts().reset_index()
class_counts.columns = ["SurvivorshipClass", "Count"]

plt.figure(figsize=(11, 6))
plt.bar(class_counts["SurvivorshipClass"], class_counts["Count"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("Count")
plt.title("APVA Edge Inventory Class Counts")
plt.tight_layout()
plt.savefig(
    FIG_DIR / "apva_edge_inventory_class_counts_v1.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

print("APVA edge inventory survivorship ranking V1 exported.")
print("Top survivors:")
print(top_survivors.head(20))
print()
print("Failed / negative edges:")
print(failed_edges.head(20))
print()
print("Summary:")
print(summary)
