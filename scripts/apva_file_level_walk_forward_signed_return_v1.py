from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# APVA file/day-level walk-forward signed return validation V1
# ---------------------------------------------------------
# Purpose:
#   A stricter validation than the earlier 60/40 bar split.
#
#   This script treats each source StateLog file as the natural unit of
#   chronology. If an instrument has multiple files, earlier files are used as
#   train-like discovery samples and later files as validation samples.
#
#   If an instrument has only one file, the script falls back to a time-ordered
#   BarIndex split for that instrument, but clearly marks the split mode.
#
# Inputs expected:
#   tables/apva_forward_signed_return_dataset_v1.csv
#
# Outputs:
#   tables/apva_file_level_walk_forward_dataset_v1.csv
#   tables/apva_file_level_walk_forward_expectancy_by_split_v1.csv
#   tables/apva_file_level_walk_forward_supported_edges_v1.csv
#   tables/apva_file_level_walk_forward_positive_validation_edges_v1.csv
#   tables/apva_file_level_walk_forward_failed_positive_edges_v1.csv
#   tables/apva_file_level_walk_forward_generalization_summary_v1.csv
#   figures/apva_file_level_walk_forward_return_generalization_v1.png
#   figures/apva_file_level_walk_forward_positive_validation_edges_v1.png
#   figures/apva_file_level_walk_forward_generalization_classes_v1.png

BASE_DIR = Path(r"C:\Users\rz0\Documents\ApvaAnalysis")
TABLE_DIR = BASE_DIR / "tables"
FIG_DIR = BASE_DIR / "figures"

TABLE_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# LOAD SIGNED FORWARD RETURN DATASET
# ---------------------------------------------------------

forward_path = TABLE_DIR / "apva_forward_signed_return_dataset_v1.csv"

if not forward_path.exists():
    raise FileNotFoundError(
        f"Missing required input: {forward_path}\n"
        "Run the signed forward return validation script first."
    )

forward = pd.read_csv(forward_path)

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
    ["Instrument", "File", "BarIndex", "HorizonBars"]
).reset_index(drop=True)

# ---------------------------------------------------------
# FILE-LEVEL / DAY-LEVEL WALK-FORWARD SPLIT
# ---------------------------------------------------------

split_frames = []

for instrument, g_inst in forward.groupby("Instrument"):
    files = (
        g_inst[["File", "BarIndex"]]
        .groupby("File")
        .agg(FirstBarIndex=("BarIndex", "min"), LastBarIndex=("BarIndex", "max"))
        .reset_index()
        .sort_values(["FirstBarIndex", "File"])
    )

    unique_files = files["File"].tolist()

    if len(unique_files) >= 3:
        # Use file chronology. Keep at least one validation file.
        split_at = max(1, int(math.floor(len(unique_files) * 0.60)))
        split_at = min(split_at, len(unique_files) - 1)

        train_files = set(unique_files[:split_at])
        validation_files = set(unique_files[split_at:])

        g = g_inst.copy()
        g["WalkForwardSplit"] = np.where(
            g["File"].isin(train_files),
            "Train",
            "Validation",
        )
        g["WalkForwardSplitMode"] = "FileLevel"
        g["TrainFileCount"] = len(train_files)
        g["ValidationFileCount"] = len(validation_files)

    else:
        # Fallback for instruments with only one or two files.
        # This is weaker than file-level validation, so mark it explicitly.
        g = g_inst.copy()
        cutoff = g["BarIndex"].quantile(0.60)
        g["WalkForwardSplit"] = np.where(
            g["BarIndex"] <= cutoff,
            "Train",
            "Validation",
        )
        g["WalkForwardSplitMode"] = "BarIndexFallback"
        g["TrainFileCount"] = len(unique_files)
        g["ValidationFileCount"] = len(unique_files)

    split_frames.append(g)

wf = pd.concat(split_frames, ignore_index=True)

# ---------------------------------------------------------
# EXPECTANCY BY SPLIT
# ---------------------------------------------------------

group_cols = [
    "Instrument",
    "HorizonBars",
    "ResolvedArchetype",
    "DominantPressure",
    "WalkForwardSplit",
    "WalkForwardSplitMode",
]

expectancy_split = (
    wf
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

# ---------------------------------------------------------
# TRAIN / VALIDATION PAIRING
# ---------------------------------------------------------

pair_keys = [
    "Instrument",
    "HorizonBars",
    "ResolvedArchetype",
    "DominantPressure",
    "WalkForwardSplitMode",
]

train = expectancy_split[expectancy_split["WalkForwardSplit"] == "Train"].copy()
valid = expectancy_split[expectancy_split["WalkForwardSplit"] == "Validation"].copy()

train = train.drop(columns=["WalkForwardSplit"])
valid = valid.drop(columns=["WalkForwardSplit"])

paired = train.merge(
    valid,
    on=pair_keys,
    how="inner",
    suffixes=("_Train", "_Validation"),
)

# ---------------------------------------------------------
# SUPPORT FILTERS
# ---------------------------------------------------------

MIN_TRAIN_SUPPORT = 50
MIN_VALIDATION_SUPPORT = 30

paired["Supported"] = (
    (paired["Count_Train"] >= MIN_TRAIN_SUPPORT)
    &
    (paired["Count_Validation"] >= MIN_VALIDATION_SUPPORT)
)

supported = paired[paired["Supported"]].copy()

# ---------------------------------------------------------
# GENERALIZATION METRICS
# ---------------------------------------------------------

supported["ReturnSignAgreement"] = (
    np.sign(supported["MeanSignedReturn_Train"])
    ==
    np.sign(supported["MeanSignedReturn_Validation"])
)

supported["HitRateDelta"] = (
    supported["DirectionalHitRate_Validation"]
    - supported["DirectionalHitRate_Train"]
)

supported["ReturnDelta"] = (
    supported["MeanSignedReturn_Validation"]
    - supported["MeanSignedReturn_Train"]
)

supported["ValidationRetainsPositiveEdge"] = (
    (supported["MeanSignedReturn_Train"] > 0)
    &
    (supported["MeanSignedReturn_Validation"] > 0)
)

supported["ValidationFlipsNegative"] = (
    (supported["MeanSignedReturn_Train"] > 0)
    &
    (supported["MeanSignedReturn_Validation"] < 0)
)

# ---------------------------------------------------------
# T-STATS
# ---------------------------------------------------------

def t_stat(mean, std, n):
    if n <= 1:
        return 0.0
    if std is None or std <= 0 or np.isnan(std):
        return 0.0
    return mean / (std / math.sqrt(n))

supported["TrainTStat"] = supported.apply(
    lambda r: t_stat(
        r["MeanSignedReturn_Train"],
        r["StdSignedReturn_Train"],
        r["Count_Train"],
    ),
    axis=1,
)

supported["ValidationTStat"] = supported.apply(
    lambda r: t_stat(
        r["MeanSignedReturn_Validation"],
        r["StdSignedReturn_Validation"],
        r["Count_Validation"],
    ),
    axis=1,
)

# ---------------------------------------------------------
# GENERALIZATION CLASS
# ---------------------------------------------------------

def classify_generalization(row):
    train_ret = row["MeanSignedReturn_Train"]
    valid_ret = row["MeanSignedReturn_Validation"]
    valid_t = row["ValidationTStat"]

    if train_ret > 0 and valid_ret > 0:
        if valid_t >= 2.0:
            return "ConfirmedPositive"
        if valid_t >= 1.0:
            return "WeakPositive"
        return "PositiveButWeak"

    if train_ret < 0 and valid_ret < 0:
        if valid_t <= -2.0:
            return "ConfirmedNegative"
        if valid_t <= -1.0:
            return "WeakNegative"
        return "NegativeButWeak"

    if train_ret > 0 and valid_ret < 0:
        return "FailedPositive"

    if train_ret < 0 and valid_ret > 0:
        return "FailedNegative"

    return "Mixed"

supported["GeneralizationClass"] = supported.apply(
    classify_generalization,
    axis=1,
)

# ---------------------------------------------------------
# EXPORT TABLES
# ---------------------------------------------------------

wf.to_csv(
    TABLE_DIR / "apva_file_level_walk_forward_dataset_v1.csv",
    index=False,
)

expectancy_split.to_csv(
    TABLE_DIR / "apva_file_level_walk_forward_expectancy_by_split_v1.csv",
    index=False,
)

supported.to_csv(
    TABLE_DIR / "apva_file_level_walk_forward_supported_edges_v1.csv",
    index=False,
)

positive_validation = supported[
    supported["GeneralizationClass"].isin(
        ["ConfirmedPositive", "WeakPositive", "PositiveButWeak"]
    )
].copy()

positive_validation = positive_validation.sort_values(
    ["MeanSignedReturn_Validation", "ValidationTStat", "Count_Validation"],
    ascending=[False, False, False],
)

positive_validation.to_csv(
    TABLE_DIR / "apva_file_level_walk_forward_positive_validation_edges_v1.csv",
    index=False,
)

failed_positive = supported[
    supported["GeneralizationClass"] == "FailedPositive"
].copy()

failed_positive = failed_positive.sort_values(
    ["MeanSignedReturn_Train", "MeanSignedReturn_Validation"],
    ascending=[False, True],
)

failed_positive.to_csv(
    TABLE_DIR / "apva_file_level_walk_forward_failed_positive_edges_v1.csv",
    index=False,
)

summary = (
    supported
    .groupby(["Instrument", "HorizonBars", "WalkForwardSplitMode", "GeneralizationClass"])
    .agg(
        Rows=("GeneralizationClass", "count"),
        MeanTrainReturn=("MeanSignedReturn_Train", "mean"),
        MeanValidationReturn=("MeanSignedReturn_Validation", "mean"),
        MeanValidationTStat=("ValidationTStat", "mean"),
        MeanReturnDelta=("ReturnDelta", "mean"),
    )
    .reset_index()
)

summary.to_csv(
    TABLE_DIR / "apva_file_level_walk_forward_generalization_summary_v1.csv",
    index=False,
)

# ---------------------------------------------------------
# FIGURES
# ---------------------------------------------------------

plt.figure(figsize=(11, 7))

plt.scatter(
    supported["MeanSignedReturn_Train"],
    supported["MeanSignedReturn_Validation"],
    alpha=0.75,
)

plt.axhline(0, linestyle="--")
plt.axvline(0, linestyle="--")

plt.xlabel("Train MeanSignedReturn")
plt.ylabel("Validation MeanSignedReturn")
plt.title("APVA File-Level Walk-Forward Return Generalization")

plt.tight_layout()
plt.savefig(
    FIG_DIR / "apva_file_level_walk_forward_return_generalization_v1.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# ---------------------------------------------------------

top = positive_validation.head(20)

if len(top) > 0:
    plt.figure(figsize=(13, 7))

    labels = (
        top["Instrument"]
        + " | "
        + top["HorizonBars"].astype(str)
        + " | "
        + top["ResolvedArchetype"]
        + " | "
        + top["DominantPressure"]
    )

    plt.bar(labels, top["MeanSignedReturn_Validation"])

    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Validation MeanSignedReturn")
    plt.title("APVA File-Level Walk-Forward Positive Validation Edges")

    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "apva_file_level_walk_forward_positive_validation_edges_v1.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

# ---------------------------------------------------------

class_counts = supported["GeneralizationClass"].value_counts().reset_index()
class_counts.columns = ["GeneralizationClass", "Count"]

plt.figure(figsize=(10, 6))

plt.bar(class_counts["GeneralizationClass"], class_counts["Count"])

plt.xticks(rotation=30, ha="right")
plt.ylabel("Count")
plt.title("APVA File-Level Walk-Forward Generalization Classes")

plt.tight_layout()
plt.savefig(
    FIG_DIR / "apva_file_level_walk_forward_generalization_classes_v1.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

print("File-level walk-forward signed return validation V1 exported.")
print("Supported rows:", len(supported))
print()
print(summary)
print()
print(positive_validation.head(20))
