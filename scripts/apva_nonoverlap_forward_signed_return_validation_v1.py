from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# APVA non-overlapping forward signed return validation V1
# ---------------------------------------------------------
# Purpose:
#   The previous forward-return tests used rolling observations. That is useful
#   for exploration, but overlapping forward windows inflate effective sample
#   size and can overstate significance.
#
#   This script rebuilds a stricter validation set by keeping only bar indices
#   that are spaced at least `HorizonBars` apart within each Instrument/File.
#   This makes the return windows approximately non-overlapping for each horizon.
#
# Inputs expected:
#   tables/apva_forward_signed_return_dataset_v1.csv
#
# Outputs:
#   tables/apva_nonoverlap_forward_signed_return_dataset_v1.csv
#   tables/apva_nonoverlap_signed_return_expectancy_v1.csv
#   tables/apva_nonoverlap_positive_signed_return_edges_v1.csv
#   tables/apva_nonoverlap_negative_signed_return_edges_v1.csv
#   tables/apva_nonoverlap_edge_summary_v1.csv
#   figures/apva_nonoverlap_signed_return_tstats_v1.png
#   figures/apva_nonoverlap_positive_signed_return_edges_v1.png
#   figures/apva_nonoverlap_mfe_mae_v1.png

BASE_DIR = Path(r"C:\Users\rz0\Documents\ApvaAnalysis")
TABLE_DIR = BASE_DIR / "tables"
FIG_DIR = BASE_DIR / "figures"

TABLE_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

input_path = TABLE_DIR / "apva_forward_signed_return_dataset_v1.csv"

if not input_path.exists():
    raise FileNotFoundError(
        f"Missing required input: {input_path}\n"
        "Run the signed forward return validation script first."
    )

forward = pd.read_csv(input_path)

required_cols = {
    "Instrument",
    "File",
    "BarIndex",
    "HorizonBars",
    "ResolvedArchetype",
    "DominantPressure",
    "SignedNormalizedReturn",
    "DirectionalHit",
    "DirectionalNormalizedMFE",
    "DirectionalNormalizedMAE",
}

missing = sorted(required_cols - set(forward.columns))
if missing:
    raise ValueError(f"Input file is missing required columns: {missing}")

forward = forward.sort_values(
    ["Instrument", "File", "HorizonBars", "BarIndex"]
).reset_index(drop=True)

# ---------------------------------------------------------
# NON-OVERLAPPING SUBSAMPLING
# ---------------------------------------------------------
# Keep every h-th bar per Instrument/File/Horizon.
# This is simple and conservative. It does not optimize entry selection.
#
# To reduce phase bias, we compute all possible offsets 0..h-1 and keep
# offset-specific results. This lets us inspect whether an edge depends on a
# lucky sampling phase.
# ---------------------------------------------------------

nonoverlap_frames = []

for (instrument, file, horizon), g in forward.groupby(["Instrument", "File", "HorizonBars"]):
    g = g.sort_values("BarIndex").reset_index(drop=True)
    h = int(horizon)

    if h <= 0:
        continue

    for offset in range(h):
        sampled = g.iloc[offset::h].copy()
        sampled["NonOverlapOffset"] = offset
        sampled["NonOverlapStride"] = h
        sampled["NonOverlapMode"] = "OffsetStride"
        nonoverlap_frames.append(sampled)

nonoverlap = pd.concat(nonoverlap_frames, ignore_index=True)

# ---------------------------------------------------------
# EXPECTANCY BY OFFSET
# ---------------------------------------------------------

group_cols = [
    "Instrument",
    "HorizonBars",
    "NonOverlapOffset",
    "ResolvedArchetype",
    "DominantPressure",
]

def t_stat(mean, std, n):
    if n <= 1:
        return 0.0
    if std is None or std <= 0 or np.isnan(std):
        return 0.0
    return mean / (std / math.sqrt(n))

expectancy_offset = (
    nonoverlap
    .groupby(group_cols)
    .agg(
        Count=("SignedNormalizedReturn", "count"),
        MeanSignedReturn=("SignedNormalizedReturn", "mean"),
        MedianSignedReturn=("SignedNormalizedReturn", "median"),
        StdSignedReturn=("SignedNormalizedReturn", "std"),
        DirectionalHitRate=("DirectionalHit", "mean"),
        MeanDirectionalMFE=("DirectionalNormalizedMFE", "mean"),
        MeanDirectionalMAE=("DirectionalNormalizedMAE", "mean"),
    )
    .reset_index()
)

expectancy_offset["TStat"] = expectancy_offset.apply(
    lambda r: t_stat(r["MeanSignedReturn"], r["StdSignedReturn"], r["Count"]),
    axis=1,
)

# ---------------------------------------------------------
# AGGREGATE OFFSET ROBUSTNESS
# ---------------------------------------------------------
# For each state/pressure condition, summarize across offsets.
# An edge is more credible if its mean return is positive across many offsets,
# not just one lucky phase.
# ---------------------------------------------------------

edge_keys = [
    "Instrument",
    "HorizonBars",
    "ResolvedArchetype",
    "DominantPressure",
]

MIN_OFFSET_SUPPORT = 20
MIN_TOTAL_SUPPORT = 50
MIN_POSITIVE_OFFSET_FRACTION = 0.60

usable_offsets = expectancy_offset[
    expectancy_offset["Count"] >= MIN_OFFSET_SUPPORT
].copy()

edge_summary = (
    usable_offsets
    .groupby(edge_keys)
    .agg(
        OffsetCount=("NonOverlapOffset", "nunique"),
        TotalSupport=("Count", "sum"),
        MeanOfOffsetReturns=("MeanSignedReturn", "mean"),
        MedianOfOffsetReturns=("MeanSignedReturn", "median"),
        StdOfOffsetReturns=("MeanSignedReturn", "std"),
        MeanOffsetTStat=("TStat", "mean"),
        MaxOffsetTStat=("TStat", "max"),
        MinOffsetTStat=("TStat", "min"),
        MeanDirectionalHitRate=("DirectionalHitRate", "mean"),
        MeanDirectionalMFE=("MeanDirectionalMFE", "mean"),
        MeanDirectionalMAE=("MeanDirectionalMAE", "mean"),
        PositiveOffsetFraction=("MeanSignedReturn", lambda s: (s > 0).mean()),
        NegativeOffsetFraction=("MeanSignedReturn", lambda s: (s < 0).mean()),
    )
    .reset_index()
)

# ---------------------------------------------------------
# CLASSIFY ROBUSTNESS
# ---------------------------------------------------------

def classify_edge(row):
    if row["TotalSupport"] < MIN_TOTAL_SUPPORT or row["OffsetCount"] < 2:
        return "Unsupported"

    if (
        row["MeanOfOffsetReturns"] > 0
        and row["PositiveOffsetFraction"] >= MIN_POSITIVE_OFFSET_FRACTION
        and row["MeanOffsetTStat"] >= 1.0
    ):
        if row["MeanOffsetTStat"] >= 2.0:
            return "RobustPositive"
        return "WeakRobustPositive"

    if (
        row["MeanOfOffsetReturns"] < 0
        and row["NegativeOffsetFraction"] >= MIN_POSITIVE_OFFSET_FRACTION
        and row["MeanOffsetTStat"] <= -1.0
    ):
        if row["MeanOffsetTStat"] <= -2.0:
            return "RobustNegative"
        return "WeakRobustNegative"

    return "NotRobust"

edge_summary["NonOverlapRobustnessClass"] = edge_summary.apply(classify_edge, axis=1)

positive_edges = edge_summary[
    edge_summary["NonOverlapRobustnessClass"].isin(["RobustPositive", "WeakRobustPositive"])
].copy()

positive_edges = positive_edges.sort_values(
    ["MeanOfOffsetReturns", "MeanOffsetTStat", "TotalSupport"],
    ascending=[False, False, False],
)

negative_edges = edge_summary[
    edge_summary["NonOverlapRobustnessClass"].isin(["RobustNegative", "WeakRobustNegative"])
].copy()

negative_edges = negative_edges.sort_values(
    ["MeanOfOffsetReturns", "MeanOffsetTStat"],
    ascending=[True, True],
)

# ---------------------------------------------------------
# EXPORT TABLES
# ---------------------------------------------------------

nonoverlap.to_csv(
    TABLE_DIR / "apva_nonoverlap_forward_signed_return_dataset_v1.csv",
    index=False,
)

expectancy_offset.to_csv(
    TABLE_DIR / "apva_nonoverlap_signed_return_expectancy_by_offset_v1.csv",
    index=False,
)

edge_summary.to_csv(
    TABLE_DIR / "apva_nonoverlap_signed_return_edge_summary_v1.csv",
    index=False,
)

positive_edges.to_csv(
    TABLE_DIR / "apva_nonoverlap_positive_signed_return_edges_v1.csv",
    index=False,
)

negative_edges.to_csv(
    TABLE_DIR / "apva_nonoverlap_negative_signed_return_edges_v1.csv",
    index=False,
)

class_summary = (
    edge_summary
    .groupby(["Instrument", "HorizonBars", "NonOverlapRobustnessClass"])
    .agg(
        Rows=("NonOverlapRobustnessClass", "count"),
        MeanOffsetReturn=("MeanOfOffsetReturns", "mean"),
        MeanOffsetTStat=("MeanOffsetTStat", "mean"),
        MeanPositiveOffsetFraction=("PositiveOffsetFraction", "mean"),
        MeanTotalSupport=("TotalSupport", "mean"),
    )
    .reset_index()
)

class_summary.to_csv(
    TABLE_DIR / "apva_nonoverlap_edge_class_summary_v1.csv",
    index=False,
)

# ---------------------------------------------------------
# FIGURES
# ---------------------------------------------------------

plt.figure(figsize=(11, 7))

colors = []
for cls in edge_summary["NonOverlapRobustnessClass"]:
    if cls == "RobustPositive":
        colors.append("green")
    elif cls == "WeakRobustPositive":
        colors.append("lime")
    elif cls == "RobustNegative":
        colors.append("red")
    elif cls == "WeakRobustNegative":
        colors.append("orange")
    else:
        colors.append("gray")

plt.scatter(
    edge_summary["TotalSupport"],
    edge_summary["MeanOffsetTStat"],
    c=colors,
    alpha=0.75,
)

plt.axhline(2.0, linestyle="--")
plt.axhline(-2.0, linestyle="--")
plt.axhline(0.0, linestyle=":")

plt.xlabel("Total non-overlap support across offsets")
plt.ylabel("Mean offset T-stat")
plt.title("APVA Non-Overlapping Signed Return T-stats")

plt.tight_layout()
plt.savefig(
    FIG_DIR / "apva_nonoverlap_signed_return_tstats_v1.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# ---------------------------------------------------------

top_pos = positive_edges.head(20)

if len(top_pos) > 0:
    plt.figure(figsize=(13, 7))

    labels = (
        top_pos["Instrument"]
        + " | "
        + top_pos["HorizonBars"].astype(str)
        + " | "
        + top_pos["ResolvedArchetype"]
        + " | "
        + top_pos["DominantPressure"]
    )

    plt.bar(labels, top_pos["MeanOfOffsetReturns"])

    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Mean signed normalized return across offsets")
    plt.title("APVA Non-Overlapping Positive Signed Return Edges")

    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "apva_nonoverlap_positive_signed_return_edges_v1.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

# ---------------------------------------------------------

plt.figure(figsize=(11, 7))

plt.scatter(
    edge_summary["MeanDirectionalMAE"],
    edge_summary["MeanDirectionalMFE"],
    c=colors,
    alpha=0.75,
)

plt.xlabel("Mean directional normalized MAE")
plt.ylabel("Mean directional normalized MFE")
plt.title("APVA Non-Overlapping MFE vs MAE")

plt.tight_layout()
plt.savefig(
    FIG_DIR / "apva_nonoverlap_mfe_mae_v1.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

print("Non-overlapping forward signed return validation V1 exported.")
print("Positive robust edges:")
print(positive_edges.head(20))
print()
print("Negative robust edges:")
print(negative_edges.head(20))
print()
print("Class summary:")
print(class_summary)
