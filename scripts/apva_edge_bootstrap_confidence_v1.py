from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# APVA edge bootstrap confidence V1
# ---------------------------------------------------------
# Purpose:
#   Estimate uncertainty around candidate APVA edges using bootstrap resampling.
#
#   This focuses on the real question:
#       Is the observed mean signed return stable under resampling,
#       or is it dominated by a few lucky bars?
#
# Inputs expected:
#   tables/apva_forward_signed_return_dataset_v1.csv
#   tables/apva_edge_inventory_top_survivors_v1.csv
#
# Outputs:
#   tables/apva_edge_bootstrap_confidence_v1.csv
#   tables/apva_edge_bootstrap_top_survivors_v1.csv
#   tables/apva_edge_bootstrap_failed_candidates_v1.csv
#   figures/apva_edge_bootstrap_mean_ci_v1.png
#   figures/apva_edge_bootstrap_positive_probability_v1.png

BASE_DIR = Path(r"C:\Users\rz0\Documents\ApvaAnalysis")
TABLE_DIR = BASE_DIR / "tables"
FIG_DIR = BASE_DIR / "figures"

TABLE_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

DATA_PATH = TABLE_DIR / "apva_forward_signed_return_dataset_v1.csv"
CANDIDATE_PATH = TABLE_DIR / "apva_edge_inventory_top_survivors_v1.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing required input: {DATA_PATH}")

if not CANDIDATE_PATH.exists():
    raise FileNotFoundError(f"Missing required input: {CANDIDATE_PATH}")

forward = pd.read_csv(DATA_PATH)
candidates = pd.read_csv(CANDIDATE_PATH)

edge_keys = [
    "Instrument",
    "HorizonBars",
    "ResolvedArchetype",
    "DominantPressure",
]

required_forward_cols = set(edge_keys + [
    "SignedNormalizedReturn",
    "DirectionalHit",
    "DirectionalNormalizedMFE",
    "DirectionalNormalizedMAE",
])

missing_forward = sorted(required_forward_cols - set(forward.columns))
if missing_forward:
    raise ValueError(f"Forward dataset missing columns: {missing_forward}")

missing_candidate = sorted(set(edge_keys) - set(candidates.columns))
if missing_candidate:
    raise ValueError(f"Candidate table missing columns: {missing_candidate}")

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

N_BOOTSTRAPS = 5000
RANDOM_SEED = 20260524
MIN_SUPPORT = 50

rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------
# BOOTSTRAP HELPERS
# ---------------------------------------------------------

def bootstrap_stats(values, rng, n_boot=N_BOOTSTRAPS):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]

    n = len(arr)
    if n == 0:
        return {
            "Support": 0,
            "ObservedMean": np.nan,
            "ObservedMedian": np.nan,
            "ObservedStd": np.nan,
            "BootstrapMean": np.nan,
            "BootstrapStd": np.nan,
            "CILow_2p5": np.nan,
            "CIHigh_97p5": np.nan,
            "ProbabilityMeanPositive": np.nan,
            "ProbabilityMeanNegative": np.nan,
        }

    observed_mean = float(np.mean(arr))
    observed_median = float(np.median(arr))
    observed_std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    if n == 1:
        boot_means = np.repeat(arr[0], n_boot)
    else:
        sample_idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = arr[sample_idx].mean(axis=1)

    return {
        "Support": n,
        "ObservedMean": observed_mean,
        "ObservedMedian": observed_median,
        "ObservedStd": observed_std,
        "BootstrapMean": float(np.mean(boot_means)),
        "BootstrapStd": float(np.std(boot_means, ddof=1)),
        "CILow_2p5": float(np.quantile(boot_means, 0.025)),
        "CIHigh_97p5": float(np.quantile(boot_means, 0.975)),
        "ProbabilityMeanPositive": float(np.mean(boot_means > 0)),
        "ProbabilityMeanNegative": float(np.mean(boot_means < 0)),
    }

# ---------------------------------------------------------
# RUN BOOTSTRAP ON CANDIDATES
# ---------------------------------------------------------

rows = []

unique_candidates = candidates[edge_keys + ["SurvivorshipScore", "SurvivorshipClass"]].drop_duplicates()

for _, cand in unique_candidates.iterrows():
    mask = np.ones(len(forward), dtype=bool)

    for key in edge_keys:
        mask &= forward[key].astype(str).values == str(cand[key])

    subset = forward.loc[mask].copy()
    values = subset["SignedNormalizedReturn"].dropna().values

    stats = bootstrap_stats(values, rng)

    hit_rate = float(subset["DirectionalHit"].mean()) if len(subset) else np.nan
    mfe = float(subset["DirectionalNormalizedMFE"].mean()) if len(subset) else np.nan
    mae = float(subset["DirectionalNormalizedMAE"].mean()) if len(subset) else np.nan

    row = {
        **{key: cand[key] for key in edge_keys},
        "SurvivorshipScore": cand.get("SurvivorshipScore", np.nan),
        "SurvivorshipClass": cand.get("SurvivorshipClass", "Unknown"),
        "DirectionalHitRate": hit_rate,
        "MeanDirectionalMFE": mfe,
        "MeanDirectionalMAE": mae,
        **stats,
    }

    rows.append(row)

bootstrap = pd.DataFrame(rows)

# ---------------------------------------------------------
# CONFIDENCE CLASSIFICATION
# ---------------------------------------------------------

def classify_bootstrap(row):
    if row["Support"] < MIN_SUPPORT:
        return "Unsupported"

    if row["CILow_2p5"] > 0 and row["ProbabilityMeanPositive"] >= 0.975:
        return "BootstrapConfirmedPositive"

    if row["CILow_2p5"] > 0 and row["ProbabilityMeanPositive"] >= 0.90:
        return "BootstrapWeakPositive"

    if row["CIHigh_97p5"] < 0 and row["ProbabilityMeanNegative"] >= 0.975:
        return "BootstrapConfirmedNegative"

    if row["CIHigh_97p5"] < 0 and row["ProbabilityMeanNegative"] >= 0.90:
        return "BootstrapWeakNegative"

    if row["ProbabilityMeanPositive"] >= 0.75:
        return "PositiveButUncertain"

    if row["ProbabilityMeanNegative"] >= 0.75:
        return "NegativeButUncertain"

    return "Indeterminate"

bootstrap["BootstrapConfidenceClass"] = bootstrap.apply(classify_bootstrap, axis=1)

bootstrap = bootstrap.sort_values(
    ["ProbabilityMeanPositive", "CILow_2p5", "ObservedMean", "Support"],
    ascending=[False, False, False, False],
)

# ---------------------------------------------------------
# EXPORT TABLES
# ---------------------------------------------------------

bootstrap.to_csv(
    TABLE_DIR / "apva_edge_bootstrap_confidence_v1.csv",
    index=False,
)

bootstrap_top = bootstrap[
    bootstrap["BootstrapConfidenceClass"].isin([
        "BootstrapConfirmedPositive",
        "BootstrapWeakPositive",
        "PositiveButUncertain",
    ])
].copy()

bootstrap_top.to_csv(
    TABLE_DIR / "apva_edge_bootstrap_top_survivors_v1.csv",
    index=False,
)

bootstrap_failed = bootstrap[
    bootstrap["BootstrapConfidenceClass"].isin([
        "BootstrapConfirmedNegative",
        "BootstrapWeakNegative",
        "NegativeButUncertain",
        "Indeterminate",
    ])
].copy()

bootstrap_failed.to_csv(
    TABLE_DIR / "apva_edge_bootstrap_failed_candidates_v1.csv",
    index=False,
)

summary = (
    bootstrap
    .groupby(["BootstrapConfidenceClass"])
    .agg(
        Rows=("BootstrapConfidenceClass", "count"),
        MeanObservedReturn=("ObservedMean", "mean"),
        MeanProbabilityPositive=("ProbabilityMeanPositive", "mean"),
        MeanSupport=("Support", "mean"),
    )
    .reset_index()
)

summary.to_csv(
    TABLE_DIR / "apva_edge_bootstrap_summary_v1.csv",
    index=False,
)

# ---------------------------------------------------------
# FIGURES
# ---------------------------------------------------------

plot_df = bootstrap.head(25).copy()

if len(plot_df) > 0:
    labels = (
        plot_df["Instrument"]
        + " | "
        + plot_df["HorizonBars"].astype(str)
        + " | "
        + plot_df["ResolvedArchetype"]
        + " | "
        + plot_df["DominantPressure"]
    )

    y = plot_df["ObservedMean"].values
    yerr_low = y - plot_df["CILow_2p5"].values
    yerr_high = plot_df["CIHigh_97p5"].values - y

    plt.figure(figsize=(14, 7))
    plt.errorbar(
        labels,
        y,
        yerr=[yerr_low, yerr_high],
        fmt="o",
        capsize=4,
    )
    plt.axhline(0, linestyle="--")
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Signed normalized return")
    plt.title("APVA Bootstrap Mean Return Confidence Intervals")
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "apva_edge_bootstrap_mean_ci_v1.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(14, 7))
    plt.bar(labels, plot_df["ProbabilityMeanPositive"])
    plt.axhline(0.975, linestyle="--")
    plt.axhline(0.90, linestyle=":")
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Bootstrap probability mean > 0")
    plt.title("APVA Bootstrap Positive Mean Probability")
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "apva_edge_bootstrap_positive_probability_v1.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

print("APVA edge bootstrap confidence V1 exported.")
print("Top bootstrap survivors:")
print(bootstrap_top.head(20))
print()
print("Summary:")
print(summary)
