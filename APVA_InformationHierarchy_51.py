#!/usr/bin/env python3
"""
APVA Information Hierarchy Study v0.1

Audit whether each nested APVA layer earns its degrees of freedom by reducing
next-state uncertainty without excessive sample fragmentation. Research only.
"""

from __future__ import annotations

import argparse
import os
import statistics as stats
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from APVA_StructuralLifeCycle_44 import directional_return, ensure_dir, fmt, load_results, pct
from APVA_InterpretationArbitration_48 import build_result as build_arbitration_result, validated_context
from APVA_PersistenceCalibration_50 import run_regime
from APVA_TransitionContext_46 import entropy, normalized_entropy

HORIZONS = (1, 2, 3, 5)
MODELS = (
    ("M1", "State"),
    ("M2", "State + Age"),
    ("M3", "State + Age + ValidatedContext"),
    ("M4", "State + Age + ValidatedContext + Arbitration"),
    ("M5", "State + Age + ValidatedContext + Arbitration + Persistence"),
)
LAYER_NAMES = {
    "M1": "State",
    "M2": "Age",
    "M3": "ValidatedContext",
    "M4": "Arbitration",
    "M5": "Persistence",
}
SPARSE_THRESHOLD = 20
MIN_VALID_INSTRUMENTS = 2
TOP_LIMIT = 25


@dataclass
class Outcome:
    values: List[float] = field(default_factory=list)

    def add(self, value: Optional[float]) -> None:
        if value is not None:
            self.values.append(value)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def mean_dr(self) -> float:
        return mean(self.values)

    @property
    def median_dr(self) -> float:
        return stats.median(self.values) if self.values else 0.0

    @property
    def continuation(self) -> float:
        return sum(x > 0 for x in self.values) / len(self.values) if self.values else 0.0

    @property
    def failure(self) -> float:
        return sum(x < 0 for x in self.values) / len(self.values) if self.values else 0.0

    @property
    def flat(self) -> float:
        return sum(x == 0 for x in self.values) / len(self.values) if self.values else 0.0

    @property
    def skew(self) -> float:
        return self.continuation - self.failure


@dataclass(frozen=True)
class ModelMetrics:
    model: str
    horizon: int
    count: int
    entropy: float
    normalized_entropy: float
    dominant_destination: str
    dominant_probability: float
    mean_dominant_lift: float
    unique_keys: int
    average_per_key: float
    median_per_key: float
    sparse_keys: int
    sparse_rate: float
    adjusted_score: float


@dataclass(frozen=True)
class GainMetrics:
    from_model: str
    to_model: str
    horizon: int
    entropy_reduction: float
    percent_reduction: float
    dominant_probability_increase: float
    information_gain_share: float
    sparse_rate_increase: float


@dataclass
class StudyResult:
    instrument: str
    bars: list
    source_paths: List[str]
    model_keys: Dict[str, List[str]]
    metrics: Dict[Tuple[str, int], ModelMetrics]
    gains: Dict[Tuple[str, str, int], GainMetrics]
    outcomes: Dict[Tuple[str, str], Outcome]


def mean(values: Iterable[float]) -> float:
    xs = list(values)
    return sum(xs) / len(xs) if xs else 0.0


def model_key(
    model: str,
    state: str,
    age: str,
    context_flag: bool,
    arbitration_type: str,
    persistence_type: str,
) -> str:
    parts = [state]
    if model in ("M2", "M3", "M4", "M5"):
        parts.append(f"Age{age}")
    if model in ("M3", "M4", "M5"):
        parts.append(f"Validated={context_flag}")
    if model in ("M4", "M5"):
        parts.append(f"Arbitration={arbitration_type}")
    if model == "M5":
        parts.append(f"Persistence={persistence_type}")
    return " | ".join(parts)


def build_keys(arbitration) -> Dict[str, List[str]]:
    persistence = run_regime(arbitration.arbitrations, "F")
    result = {model: [] for model, _ in MODELS}
    for observation, arbitration_row, persistence_row in zip(
        arbitration.observations, arbitration.arbitrations, persistence.rows
    ):
        context_name, _ = validated_context(observation)
        for model, _ in MODELS:
            result[model].append(
                model_key(
                    model,
                    observation.state,
                    observation.age_bucket,
                    bool(context_name),
                    arbitration_row.winner_type,
                    persistence_row.active_type,
                )
            )
    return result


def transition_metrics(bars, keys: Sequence[str], model: str, horizon: int) -> ModelMetrics:
    groups: Dict[str, Counter[str]] = defaultdict(Counter)
    unconditional: Counter[str] = Counter()
    for index in range(len(bars) - horizon):
        destination = bars[index + horizon].state
        groups[keys[index]][destination] += 1
        unconditional[destination] += 1
    total = sum(unconditional.values())
    key_counts = [sum(counts.values()) for counts in groups.values()]
    weighted_entropy = sum(sum(counts.values()) * entropy(counts) for counts in groups.values()) / total if total else 0.0
    weighted_norm_entropy = sum(sum(counts.values()) * normalized_entropy(counts) for counts in groups.values()) / total if total else 0.0
    dominant_votes: Counter[str] = Counter()
    dominant_probabilities: List[Tuple[int, float]] = []
    dominant_lifts: List[Tuple[int, float]] = []
    for counts in groups.values():
        key_total = sum(counts.values())
        destination, count = counts.most_common(1)[0]
        probability = count / key_total
        baseline = unconditional[destination] / total if total else 0.0
        dominant_votes[destination] += key_total
        dominant_probabilities.append((key_total, probability))
        dominant_lifts.append((key_total, probability / baseline if baseline else 0.0))
    unique = len(groups)
    sparse = sum(count < SPARSE_THRESHOLD for count in key_counts)
    return ModelMetrics(
        model,
        horizon,
        total,
        weighted_entropy,
        weighted_norm_entropy,
        dominant_votes.most_common(1)[0][0] if dominant_votes else "N/A",
        sum(weight * value for weight, value in dominant_probabilities) / total if total else 0.0,
        sum(weight * value for weight, value in dominant_lifts) / total if total else 0.0,
        unique,
        mean(key_counts),
        stats.median(key_counts) if key_counts else 0.0,
        sparse,
        sparse / unique if unique else 0.0,
        0.0,
    )


def with_adjusted_scores(metrics: Dict[Tuple[str, int], ModelMetrics]) -> Dict[Tuple[str, int], ModelMetrics]:
    result = {}
    for key, row in metrics.items():
        baseline = metrics[("M1", row.horizon)]
        reduction = baseline.entropy - row.entropy
        result[key] = replace(row, adjusted_score=reduction * (1.0 - row.sparse_rate))
    return result


def build_gains(metrics: Dict[Tuple[str, int], ModelMetrics]) -> Dict[Tuple[str, str, int], GainMetrics]:
    gains = {}
    pairs = zip(MODELS[:-1], MODELS[1:])
    for (from_model, _), (to_model, _) in pairs:
        for horizon in HORIZONS:
            left = metrics[(from_model, horizon)]
            right = metrics[(to_model, horizon)]
            total_reduction = metrics[("M1", horizon)].entropy - metrics[("M5", horizon)].entropy
            reduction = left.entropy - right.entropy
            gains[(from_model, to_model, horizon)] = GainMetrics(
                from_model,
                to_model,
                horizon,
                reduction,
                reduction / left.entropy if left.entropy else 0.0,
                right.dominant_probability - left.dominant_probability,
                reduction / total_reduction if total_reduction else 0.0,
                right.sparse_rate - left.sparse_rate,
            )
    return gains


def build_outcomes(bars, model_keys: Dict[str, List[str]]) -> Dict[Tuple[str, str], Outcome]:
    outcomes: Dict[Tuple[str, str], Outcome] = defaultdict(Outcome)
    for index in range(len(bars)):
        dr = directional_return(bars, index, 5)
        for model, _ in MODELS:
            outcomes[(model, model_keys[model][index])].add(dr)
    return outcomes


def build_result(loaded) -> StudyResult:
    arbitration = build_arbitration_result(loaded)
    keys = build_keys(arbitration)
    metrics = {
        (model, horizon): transition_metrics(arbitration.bars, keys[model], model, horizon)
        for model, _ in MODELS
        for horizon in HORIZONS
    }
    metrics = with_adjusted_scores(metrics)
    return StudyResult(
        arbitration.instrument,
        arbitration.bars,
        arbitration.source_paths,
        keys,
        metrics,
        build_gains(metrics),
        build_outcomes(arbitration.bars, keys),
    )


def survival_rows(results: Sequence[StudyResult]) -> List[Dict[str, object]]:
    rows = []
    rows.append(
        {
            "layer": "State",
            "from": "",
            "to": "M1",
            "percent": 0.0,
            "sparse_increase": 0.0,
            "valid": len(results),
            "classification": "CoreStructuralLayer",
            "survives": True,
        }
    )
    for (from_model, _), (to_model, _) in zip(MODELS[:-1], MODELS[1:]):
        gains = [result.gains[(from_model, to_model, horizon)] for result in results for horizon in HORIZONS]
        percent = mean(gain.percent_reduction for gain in gains)
        sparse_increase = mean(gain.sparse_rate_increase for gain in gains)
        valid = len(results)
        survives = percent >= 0.05 and sparse_increase <= 0.25 and valid >= MIN_VALID_INSTRUMENTS
        layer = LAYER_NAMES[to_model]
        if to_model == "M2":
            classification = "CoreStructuralLayer" if survives else "RejectedLayer"
        elif to_model == "M3":
            classification = "UsefulContextLayer" if survives else "RejectedLayer"
        else:
            classification = "ImplementationLayer" if not survives else "UsefulContextLayer"
        rows.append(
            {
                "layer": layer,
                "from": from_model,
                "to": to_model,
                "percent": percent,
                "sparse_increase": sparse_increase,
                "valid": valid,
                "classification": classification,
                "survives": survives,
            }
        )
    return rows


def recommendation(results: Sequence[StudyResult]) -> Tuple[str, str, str, str, str]:
    rows = survival_rows(results)
    surviving = [str(row["layer"]) for row in rows if row["survives"]]
    rejected = [str(row["layer"]) for row in rows if not row["survives"] and row["classification"] == "RejectedLayer"]
    implementation = [str(row["layer"]) for row in rows if row["classification"] == "ImplementationLayer"]
    stack = ["State"]
    for row in rows[1:]:
        if row["survives"]:
            stack.append(str(row["layer"]))
        else:
            break
    recommended = " + ".join(stack)
    reason = "Sequentially retain layers only while entropy reduction reaches 5% and sparsity increase stays within 25 percentage points."
    return recommended, ", ".join(surviving), ", ".join(rejected), ", ".join(implementation), reason


def append_metrics(lines: List[str], result: StudyResult) -> None:
    lines.append(
        "Model | Horizon | Count | Entropy | NormalizedEntropy | DominantDestination | "
        "DominantDestinationProbability | MeanLiftOfDominantDestination | UniqueKeyCount | "
        "AverageObservationsPerKey | MedianObservationsPerKey | SparseKeyCount | SparseKeyRate | AdjustedInformationScore"
    )
    for model, _ in MODELS:
        for horizon in HORIZONS:
            row = result.metrics[(model, horizon)]
            lines.append(
                f"{model} | t+{horizon} | {row.count} | {fmt(row.entropy)} | {fmt(row.normalized_entropy)} | "
                f"{row.dominant_destination} | {pct(row.dominant_probability)} | {fmt(row.mean_dominant_lift)} | "
                f"{row.unique_keys} | {fmt(row.average_per_key)} | {fmt(row.median_per_key)} | {row.sparse_keys} | "
                f"{pct(row.sparse_rate)} | {fmt(row.adjusted_score)}"
            )


def append_gains(lines: List[str], result: StudyResult) -> None:
    lines.append(
        "LayerAdded | FromModel | ToModel | Horizon | EntropyReduction | PercentEntropyReduction | "
        "DominantProbabilityIncrease | InformationGainShare | SparseKeyRateIncrease"
    )
    for (from_model, _), (to_model, _) in zip(MODELS[:-1], MODELS[1:]):
        for horizon in HORIZONS:
            row = result.gains[(from_model, to_model, horizon)]
            lines.append(
                f"{LAYER_NAMES[to_model]} | {from_model} | {to_model} | t+{horizon} | {fmt(row.entropy_reduction)} | "
                f"{pct(row.percent_reduction)} | {pct(row.dominant_probability_increase)} | "
                f"{pct(row.information_gain_share)} | {pct(row.sparse_rate_increase)}"
            )


def append_outcomes(lines: List[str], result: StudyResult) -> None:
    lines.append("Model | ModelKey | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew")
    for (model, key), outcome in sorted(result.outcomes.items()):
        lines.append(
            f"{model} | {key} | {outcome.count} | {fmt(outcome.mean_dr)} | {fmt(outcome.median_dr)} | "
            f"{pct(outcome.continuation)} | {pct(outcome.failure)} | {pct(outcome.flat)} | {pct(outcome.skew)}"
        )


def append_audit(lines: List[str]) -> None:
    lines.extend(
        [
            "Variables used:",
            "- StructuralState",
            "- AgeBucket",
            "- ValidatedContextFlag",
            "- ArbitrationWinnerType",
            "- PersistenceWinnerType from frozen Margin 0.20 regime",
            "",
            "No new variables added.",
            "No optimization.",
            "No fitting.",
            "No machine learning.",
            "No forward returns used in model selection.",
            "No outcome-based survival criteria.",
        ]
    )


def write_instrument_report(result: StudyResult, out_root: str) -> str:
    out_dir = os.path.join(out_root, result.instrument)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"InformationHierarchy_{result.instrument}.txt")
    lines = [
        "APVA Information Hierarchy Study v0.1",
        "Pardo-style degree-of-freedom audit. Research only.",
        "",
        "Diagnostics",
        f"Instrument: {result.instrument}",
        f"Input path(s): {'; '.join(result.source_paths)}",
        f"Total rows: {len(result.bars)}",
        "",
        "1. Model key counts",
        "Model | UniqueKeys | SparseKeys | SparseKeyRate",
    ]
    for model, _ in MODELS:
        row = result.metrics[(model, 1)]
        lines.append(f"{model} | {row.unique_keys} | {row.sparse_keys} | {pct(row.sparse_rate)}")
    lines += ["", "2. Entropy by model"]
    append_metrics(lines, result)
    lines += ["", "3. Incremental information gain"]
    append_gains(lines, result)
    lines += ["", "4. Degree-of-freedom penalty"]
    append_metrics(lines, result)
    lines += ["", "5. Layer survival test", "Layer | MeanPercentEntropyReduction | MeanSparseRateIncrease | Classification | Survives"]
    for row in survival_rows([result]):
        lines.append(f"{row['layer']} | {pct(float(row['percent']))} | {pct(float(row['sparse_increase']))} | {row['classification']} | {row['survives']}")
    lines += ["", "6. Outcome diagnostics"]
    append_outcomes(lines, result)
    recommended, surviving, rejected, implementation, reason = recommendation([result])
    lines += [
        "",
        "7. Minimal framework recommendation",
        f"RecommendedStack: {recommended}",
        f"SurvivingLayers: {surviving}",
        f"RejectedLayers: {rejected}",
        f"ImplementationOnlyLayers: {implementation}",
        f"Reason: {reason}",
        "",
        "8. Low-DoF audit",
    ]
    append_audit(lines)
    lines += [
        "",
        "9. Mechanical research notes",
        "- Weighted conditional entropy rewards uncertainty reduction, not table fragmentation.",
        "- Persistence winner uses Study 50 frozen Margin 0.20 regime without retuning.",
        "- Outcome diagnostics are reported separately and cannot affect survival.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def aggregate_outcome(result: StudyResult, model: str) -> Outcome:
    combined = Outcome()
    for (row_model, _), outcome in result.outcomes.items():
        if row_model == model:
            combined.values.extend(outcome.values)
    return combined


def write_aggregate_report(results: Sequence[StudyResult], out_root: str) -> str:
    out_dir = os.path.join(out_root, "InformationHierarchy")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "InformationHierarchy_All.txt")
    instruments = [result.instrument for result in results]
    by_inst = {result.instrument: result for result in results}
    lines = [
        "APVA Information Hierarchy Study v0.1 - Aggregate",
        "Pardo-style degree-of-freedom audit. Research only.",
        f"Instruments: {', '.join(instruments)}",
        "",
        "Aggregate Model Table",
        "Model | Horizon | "
        + " | ".join(field for inst in instruments for field in (f"Entropy_{inst}", f"DominantProb_{inst}", f"UniqueKeys_{inst}", f"SparseRate_{inst}"))
        + " | ValidInstrumentCount | MeanEntropy | MeanDominantProbability | MeanSparseRate | MeanAdjustedInformationScore",
    ]
    model_rows = []
    for model, _ in MODELS:
        for horizon in HORIZONS:
            values = {inst: by_inst[inst].metrics[(model, horizon)] for inst in instruments}
            cols = " | ".join(value for inst in instruments for value in (fmt(values[inst].entropy), pct(values[inst].dominant_probability), str(values[inst].unique_keys), pct(values[inst].sparse_rate)))
            model_rows.append((model, horizon, mean(row.entropy for row in values.values()), mean(row.adjusted_score for row in values.values()), mean(row.sparse_rate for row in values.values())))
            lines.append(
                f"{model} | t+{horizon} | {cols} | {len(values)} | {fmt(mean(row.entropy for row in values.values()))} | "
                f"{pct(mean(row.dominant_probability for row in values.values()))} | {pct(mean(row.sparse_rate for row in values.values()))} | "
                f"{fmt(mean(row.adjusted_score for row in values.values()))}"
            )
    lines += [
        "",
        "Aggregate Incremental Gain Table",
        "LayerAdded | FromModel | ToModel | Horizon | "
        + " | ".join(field for inst in instruments for field in (f"EntropyReduction_{inst}", f"PercentReduction_{inst}"))
        + " | ValidInstrumentCount | MeanEntropyReduction | MeanPercentReduction | MeanSparseRateIncrease | InformationGainShare",
    ]
    gain_rows = []
    for (from_model, _), (to_model, _) in zip(MODELS[:-1], MODELS[1:]):
        for horizon in HORIZONS:
            values = {inst: by_inst[inst].gains[(from_model, to_model, horizon)] for inst in instruments}
            cols = " | ".join(value for inst in instruments for value in (fmt(values[inst].entropy_reduction), pct(values[inst].percent_reduction)))
            gain_rows.append((LAYER_NAMES[to_model], mean(row.entropy_reduction for row in values.values()), mean(row.percent_reduction for row in values.values())))
            lines.append(
                f"{LAYER_NAMES[to_model]} | {from_model} | {to_model} | t+{horizon} | {cols} | {len(values)} | "
                f"{fmt(mean(row.entropy_reduction for row in values.values()))} | {pct(mean(row.percent_reduction for row in values.values()))} | "
                f"{pct(mean(row.sparse_rate_increase for row in values.values()))} | {pct(mean(row.information_gain_share for row in values.values()))}"
            )
    survival = survival_rows(results)
    lines += ["", "Aggregate Survival Table", "Layer | MeanPercentEntropyReduction | MeanSparseRateIncrease | ValidInstrumentCount | Classification | Survives"]
    for row in survival:
        lines.append(f"{row['layer']} | {pct(float(row['percent']))} | {pct(float(row['sparse_increase']))} | {row['valid']} | {row['classification']} | {row['survives']}")
    lines += ["", "Aggregate Outcome Diagnostic Table", "Model | " + " | ".join(field for inst in instruments for field in (f"Count_{inst}", f"Skew_{inst}", f"MeanDR_{inst}")) + " | ValidInstrumentCount | MeanSkew | MeanDR"]
    outcome_rows = []
    for model, _ in MODELS:
        values = {inst: aggregate_outcome(by_inst[inst], model) for inst in instruments}
        cols = " | ".join(value for inst in instruments for value in (str(values[inst].count), pct(values[inst].skew), fmt(values[inst].mean_dr)))
        outcome_rows.append((model, mean(row.skew for row in values.values()), mean(row.mean_dr for row in values.values())))
        lines.append(f"{model} | {cols} | {len(values)} | {pct(mean(row.skew for row in values.values()))} | {fmt(mean(row.mean_dr for row in values.values()))}")
    recommended, surviving, rejected, implementation, reason = recommendation(results)
    lines += [
        "",
        "Aggregate Minimal Framework Recommendation",
        f"RecommendedStack: {recommended}",
        f"SurvivingLayers: {surviving}",
        f"RejectedLayers: {rejected}",
        f"ImplementationOnlyLayers: {implementation}",
        f"Reason: {reason}",
    ]

    def ranking(title: str, rendered: Iterable[str]) -> None:
        lines.extend(["", title])
        lines.extend(list(rendered)[:TOP_LIMIT])

    ranking("1. Most informative model", (f"{model} | t+{horizon} | Entropy={fmt(ent)}" for model, horizon, ent, _, _ in sorted(model_rows, key=lambda row: row[2])))
    ranking("2. Best adjusted information score", (f"{model} | t+{horizon} | AdjustedScore={fmt(score)}" for model, horizon, _, score, _ in sorted(model_rows, key=lambda row: -row[3])))
    ranking("3. Largest entropy reduction", (f"{layer} | EntropyReduction={fmt(reduction)}" for layer, reduction, _ in sorted(gain_rows, key=lambda row: -row[1])))
    ranking("4. Best layer information gain", (f"{layer} | PercentReduction={pct(percent)}" for layer, _, percent in sorted(gain_rows, key=lambda row: -row[2])))
    ranking("5. Worst sparsity penalty", (f"{model} | t+{horizon} | SparseRate={pct(sparse)}" for model, horizon, _, _, sparse in sorted(model_rows, key=lambda row: -row[4])))
    ranking("6. Layers that survive", (f"{row['layer']} | {row['classification']}" for row in survival if row["survives"]))
    ranking("7. Layers that fail", (f"{row['layer']} | {row['classification']}" for row in survival if not row["survives"]))
    ranking("8. Best minimal APVA stack", [recommended])
    ranking("9. Highest outcome skew by model", (f"{model} | OutcomeSkew={pct(skew)}" for model, skew, _ in sorted(outcome_rows, key=lambda row: -row[1])))
    ranking("10. Worst outcome skew by model", (f"{model} | OutcomeSkew={pct(skew)}" for model, skew, _ in sorted(outcome_rows, key=lambda row: row[1])))
    lines += ["", "Low-DoF audit"]
    append_audit(lines)
    lines += [
        "",
        "Mechanical research notes",
        "- Nested weighted conditional entropy measures which layers reduce structural next-state uncertainty.",
        "- Sparsity penalizes fragmentation before recommending additional layers.",
        "- Outcomes remain diagnostic only and do not influence survival or stack recommendation.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="APVA Information Hierarchy Study v0.1")
    parser.add_argument("inputs", nargs="+", help="Evidence CSV files or directories")
    parser.add_argument("--out-root", default="Evidence/Output", help="Output root directory")
    args = parser.parse_args(argv)
    loaded = load_results(args.inputs)
    if not loaded:
        raise SystemExit("No input rows loaded.")
    results = [build_result(result) for result in loaded]
    for result in results:
        write_instrument_report(result, args.out_root)
    aggregate = write_aggregate_report(results, args.out_root)
    print(f"Wrote InformationHierarchy reports under {args.out_root}")
    print(f"Aggregate: {aggregate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
