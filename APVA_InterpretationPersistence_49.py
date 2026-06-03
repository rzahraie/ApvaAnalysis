#!/usr/bin/env python3
"""
APVA Interpretation Persistence Study v0.1

Apply four fixed persistence regimes to Study 48's arbitration stream.
Persistence affects active interpretation labels only. It never changes
structural scoring and never uses forward outcomes. Research only.
"""

from __future__ import annotations

import argparse
import os
import statistics as stats
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from APVA_StructuralLifeCycle_44 import directional_return, ensure_dir, fmt, load_results, pct
from APVA_InterpretationArbitration_48 import (
    ArbitrationRow,
    Outcome,
    StudyResult as ArbitrationResult,
    build_result as build_arbitration_result,
)

REGIMES = (
    ("A", "Immediate Replacement"),
    ("B", "Two-Bar Confirmation"),
    ("C", "Margin Threshold"),
    ("D", "Hybrid"),
)
REGIME_NAME = dict(REGIMES)
MARGIN_THRESHOLD = 0.20
CONFIRMATION_BARS = 2
TOP_LIMIT = 25


@dataclass
class PersistenceRow:
    index: int
    baseline_type: str
    baseline_key: str
    baseline_score: float
    active_type: str
    active_key: str
    active_score: float
    active_run_length: int
    replacement: bool
    replacement_reason: str
    prior_run_length: int
    fractal_jump_risk: bool
    override: bool


@dataclass
class RegimeResult:
    regime: str
    name: str
    rows: List[PersistenceRow]


@dataclass
class StudyResult:
    arbitration: ArbitrationResult
    regimes: Dict[str, RegimeResult]


def mean(values: Iterable[float]) -> float:
    xs = list(values)
    return sum(xs) / len(xs) if xs else 0.0


def scored_key(row: ArbitrationRow, key: str) -> Tuple[str, float]:
    for scored in row.scored:
        if scored.interpretation.key == key:
            return scored.interpretation.interpretation_type, scored.final_score
    return "Unresolved", 0.0


def run_regime(arbitrations: Sequence[ArbitrationRow], regime: str) -> RegimeResult:
    rows: List[PersistenceRow] = []
    active_key = ""
    active_type = "Unresolved"
    active_score = 0.0
    run_length = 0
    pending_key = ""
    pending_count = 0
    for row in arbitrations:
        challenger_key = row.winner_key
        challenger_type = row.winner_type
        challenger_score = row.winner_score
        prior_run_length = run_length
        replacement = False
        reason = "None"
        if not active_key:
            active_key = challenger_key
            active_type = challenger_type
            active_score = challenger_score
            run_length = 1
        elif regime == "A":
            if challenger_key != active_key:
                replacement = True
                reason = "HigherScore"
        else:
            current_type, current_score = scored_key(row, active_key)
            if current_type != "Unresolved":
                active_type = current_type
            active_score = current_score
            different = challenger_key != active_key
            delta = challenger_score - active_score
            if different:
                if challenger_key == pending_key:
                    pending_count += 1
                else:
                    pending_key = challenger_key
                    pending_count = 1
            else:
                pending_key = ""
                pending_count = 0
            if regime == "B" and different and pending_count >= CONFIRMATION_BARS:
                replacement = True
                reason = "ConfirmationSatisfied"
            elif regime == "C" and different and delta >= MARGIN_THRESHOLD:
                replacement = True
                reason = "MarginExceeded"
            elif regime == "D" and different and delta >= MARGIN_THRESHOLD and pending_count >= CONFIRMATION_BARS:
                replacement = True
                reason = "HybridSatisfied"
        if replacement:
            active_key = challenger_key
            active_type = challenger_type
            active_score = challenger_score
            run_length = 1
            pending_key = ""
            pending_count = 0
        elif rows:
            run_length += 1
            if regime == "A":
                active_score = challenger_score
        fractal_jump = replacement and prior_run_length <= 3 and active_type != rows[-1].active_type
        rows.append(
            PersistenceRow(
                row.index,
                challenger_type,
                challenger_key,
                challenger_score,
                active_type,
                active_key,
                active_score,
                run_length,
                replacement,
                reason,
                prior_run_length,
                fractal_jump,
                active_key != challenger_key,
            )
        )
    return RegimeResult(regime, REGIME_NAME[regime], rows)


def build_result(arbitration: ArbitrationResult) -> StudyResult:
    return StudyResult(
        arbitration,
        {regime: run_regime(arbitration.arbitrations, regime) for regime, _ in REGIMES},
    )


def runs(rows: Sequence[PersistenceRow]) -> List[Tuple[str, str, int]]:
    result: List[Tuple[str, str, int]] = []
    if not rows:
        return result
    current_type = rows[0].active_type
    current_key = rows[0].active_key
    length = 1
    for row in rows[1:]:
        if row.active_key == current_key:
            length += 1
        else:
            result.append((current_type, current_key, length))
            current_type = row.active_type
            current_key = row.active_key
            length = 1
    result.append((current_type, current_key, length))
    return result


def lifetime_distribution(rows: Sequence[PersistenceRow]) -> Counter[str]:
    distribution: Counter[str] = Counter()
    for _, _, length in runs(rows):
        if length >= 21:
            bucket = "21+"
        elif length >= 11:
            bucket = "11-20"
        elif length >= 6:
            bucket = "6-10"
        else:
            bucket = str(length)
        distribution[bucket] += 1
    return distribution


def outcome_for(result: StudyResult, rows: Iterable[PersistenceRow]) -> Outcome:
    outcome = Outcome()
    for row in rows:
        outcome.add(directional_return(result.arbitration.bars, row.index, 5))
    return outcome


def override_runs(rows: Sequence[PersistenceRow]) -> List[int]:
    result: List[int] = []
    length = 0
    for row in rows:
        if row.override:
            length += 1
        elif length:
            result.append(length)
            length = 0
    if length:
        result.append(length)
    return result


def regime_summary(result: StudyResult, regime: str) -> Dict[str, float]:
    rows = result.regimes[regime].rows
    run_lengths = [length for _, _, length in runs(rows)]
    replacements = sum(row.replacement for row in rows)
    risks = [row for row in rows if row.fractal_jump_risk]
    overrides = [row for row in rows if row.override]
    override_lengths = override_runs(rows)
    outcome = outcome_for(result, rows)
    risk_outcome = outcome_for(result, risks)
    total = len(rows)
    return {
        "count": total,
        "mean_run": mean(run_lengths),
        "median_run": stats.median(run_lengths) if run_lengths else 0.0,
        "max_run": max(run_lengths) if run_lengths else 0.0,
        "replacement_count": replacements,
        "replacement_rate": replacements / total if total else 0.0,
        "bars_per_replacement": total / replacements if replacements else 0.0,
        "risk_count": len(risks),
        "risk_rate": len(risks) / total if total else 0.0,
        "mean_risk_run": mean(row.prior_run_length for row in risks),
        "risk_replacement_rate": len(risks) / replacements if replacements else 0.0,
        "override_count": len(overrides),
        "override_rate": len(overrides) / total if total else 0.0,
        "override_duration": sum(override_lengths),
        "mean_override_run": mean(override_lengths),
        "outcome_count": outcome.count,
        "mean_dr": outcome.mean_dr,
        "median_dr": outcome.median_dr,
        "continuation": outcome.continuation_rate,
        "failure": outcome.failure_rate,
        "flat": outcome.flat_rate,
        "skew": outcome.skew,
        "risk_skew": risk_outcome.skew,
        "risk_mean_dr": risk_outcome.mean_dr,
    }


def append_summary(lines: List[str], result: StudyResult) -> None:
    lines.append(
        "Regime | Name | MeanRunLength | MedianRunLength | MaxRunLength | ReplacementCount | "
        "ReplacementRate | BarsPerReplacement | ReplacementReasons | LifetimeDistribution"
    )
    for regime, name in REGIMES:
        summary = regime_summary(result, regime)
        distribution = lifetime_distribution(result.regimes[regime].rows)
        reasons = Counter(row.replacement_reason for row in result.regimes[regime].rows if row.replacement)
        rendered_reasons = ", ".join(
            f"{reason}={reasons[reason]}"
            for reason in ("HigherScore", "MarginExceeded", "ConfirmationSatisfied", "HybridSatisfied")
        )
        rendered = ", ".join(f"{bucket}={distribution[bucket]}" for bucket in ("1", "2", "3", "4", "5", "6-10", "11-20", "21+"))
        lines.append(
            f"{regime} | {name} | {fmt(summary['mean_run'])} | {fmt(summary['median_run'])} | "
            f"{int(summary['max_run'])} | {int(summary['replacement_count'])} | {pct(summary['replacement_rate'])} | "
            f"{fmt(summary['bars_per_replacement'])} | {rendered_reasons} | {rendered}"
        )


def append_risks(lines: List[str], result: StudyResult) -> None:
    lines.append("Regime | RiskCount | RiskRate | MeanRiskRunLength | RiskReplacementRate | OutcomeSkewAfterRisk | MeanDRAfterRisk")
    for regime, _ in REGIMES:
        summary = regime_summary(result, regime)
        lines.append(
            f"{regime} | {int(summary['risk_count'])} | {pct(summary['risk_rate'])} | {fmt(summary['mean_risk_run'])} | "
            f"{pct(summary['risk_replacement_rate'])} | {pct(summary['risk_skew'])} | {fmt(summary['risk_mean_dr'])}"
        )


def append_conflicts(lines: List[str], result: StudyResult) -> None:
    lines.append("Regime | OverrideCount | OverrideRate | OverrideDuration | MeanOverrideRunLength")
    for regime, _ in REGIMES:
        summary = regime_summary(result, regime)
        lines.append(
            f"{regime} | {int(summary['override_count'])} | {pct(summary['override_rate'])} | "
            f"{int(summary['override_duration'])} | {fmt(summary['mean_override_run'])}"
        )


def append_outcomes(lines: List[str], result: StudyResult) -> None:
    lines.append("Regime | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew")
    for regime, _ in REGIMES:
        summary = regime_summary(result, regime)
        lines.append(
            f"{regime} | {int(summary['outcome_count'])} | {fmt(summary['mean_dr'])} | {fmt(summary['median_dr'])} | "
            f"{pct(summary['continuation'])} | {pct(summary['failure'])} | {pct(summary['flat'])} | {pct(summary['skew'])}"
        )


def append_comparison(lines: List[str], result: StudyResult) -> None:
    lines.append(
        "Regime | ReplacementRate | MeanRunLength | RiskRate | OutcomeSkew | MeanDR | BarsPerReplacement | OverrideRate | "
        "ReplacementReductionVsImmediate | RunLengthIncreaseVsImmediate | RiskReductionVsImmediate | "
        "SkewImprovementVsImmediate | MeanDRImprovementVsImmediate"
    )
    baseline = regime_summary(result, "A")
    for regime, _ in REGIMES:
        summary = regime_summary(result, regime)
        lines.append(
            f"{regime} | {pct(summary['replacement_rate'])} | {fmt(summary['mean_run'])} | {pct(summary['risk_rate'])} | "
            f"{pct(summary['skew'])} | {fmt(summary['mean_dr'])} | {fmt(summary['bars_per_replacement'])} | "
            f"{pct(summary['override_rate'])} | {pct(baseline['replacement_rate'] - summary['replacement_rate'])} | "
            f"{fmt(summary['mean_run'] - baseline['mean_run'])} | {pct(baseline['risk_rate'] - summary['risk_rate'])} | "
            f"{pct(summary['skew'] - baseline['skew'])} | {fmt(summary['mean_dr'] - baseline['mean_dr'])}"
        )


def append_audit(lines: List[str]) -> None:
    lines.extend(
        [
            "Allowed variables used:",
            "- CurrentState",
            "- AgeBucket",
            "- PreviousDistinctState",
            "- ValidatedContextFlag",
            "- ArbitrationScore",
            "- ScoreMargin",
            "- WinnerRunLength",
            "",
            "No new variables added.",
            "No optimization performed.",
            "No fitting performed.",
            "No machine learning used.",
            "No forward returns used in scoring.",
        ]
    )


def write_instrument_report(result: StudyResult, out_root: str) -> str:
    arbitration = result.arbitration
    out_dir = os.path.join(out_root, arbitration.instrument)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"InterpretationPersistence_{arbitration.instrument}.txt")
    baseline = regime_summary(result, "A")
    lines = [
        "APVA Interpretation Persistence Study v0.1",
        "Decision-stability research only. No optimization, fitting, machine learning, parameter search, or outcome-based scoring.",
        "",
        "Diagnostics",
        f"Instrument: {arbitration.instrument}",
        f"Input path(s): {'; '.join(arbitration.source_paths)}",
        f"Total rows: {len(arbitration.bars)}",
        "",
        "1. Baseline arbitration summary",
        "Regime A reproduces Study 48 immediate replacement.",
        f"BaselineReplacementRate: {pct(baseline['replacement_rate'])}",
        f"BaselineMeanRunLength: {fmt(baseline['mean_run'])}",
        f"BaselineFractalJumpRiskRate: {pct(baseline['risk_rate'])}",
        "",
        "2. Persistence regime summaries",
    ]
    append_summary(lines, result)
    lines += ["", "3. Stability analysis"]
    append_summary(lines, result)
    lines += ["", "4. Fractal-jump proxy"]
    append_risks(lines, result)
    lines += ["", "5. Conflict suppression"]
    append_conflicts(lines, result)
    lines += ["", "6. Outcome diagnostics"]
    append_outcomes(lines, result)
    lines += ["", "7. Regime comparison"]
    append_comparison(lines, result)
    lines += ["", "8. Low-DoF audit"]
    append_audit(lines)
    lines += [
        "",
        "9. Mechanical research notes",
        "- Regime A follows Study 48's highest-scored interpretation immediately.",
        "- Regime B requires two consecutive highest-score bars before replacement.",
        "- Regime C requires the challenger score to exceed the active score by at least 0.20.",
        "- Regime D requires both fixed conditions.",
        "- Persistence changes active labels only. It never changes structural scores or uses outcomes.",
        "- Regime-level overall outcome diagnostics are expected to match because all regimes label the same market bars.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def aggregate_summaries(results: Sequence[StudyResult], regime: str) -> List[Dict[str, float]]:
    return [regime_summary(result, regime) for result in results]


def direction(skew: float) -> str:
    if skew > 0:
        return "Positive"
    if skew < 0:
        return "Negative"
    return "Neutral"


def agreement(results: Sequence[StudyResult], regime: str) -> Tuple[int, int, float, str]:
    directions = [direction(regime_summary(result, regime)["skew"]) for result in results]
    if not directions:
        return 0, 0, 0.0, "N/A"
    label, count = Counter(directions).most_common(1)[0]
    return count, len(directions), count / len(directions), label


def instrument_columns(instruments: Sequence[str], getter) -> str:
    columns: List[str] = []
    for instrument in instruments:
        columns.extend(getter(instrument))
    return " | ".join(columns)


def pooled_stability(results: Sequence[StudyResult], regime: str) -> Dict[str, float]:
    all_lengths = [length for result in results for _, _, length in runs(result.regimes[regime].rows)]
    total = sum(len(result.regimes[regime].rows) for result in results)
    replacements = sum(sum(row.replacement for row in result.regimes[regime].rows) for result in results)
    return {
        "mean_run": mean(all_lengths),
        "median_run": stats.median(all_lengths) if all_lengths else 0.0,
        "max_run": max(all_lengths) if all_lengths else 0.0,
        "replacement_rate": replacements / total if total else 0.0,
        "bars_per_replacement": total / replacements if replacements else 0.0,
    }


def write_aggregate_report(results: Sequence[StudyResult], out_root: str) -> str:
    out_dir = os.path.join(out_root, "InterpretationPersistence")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "InterpretationPersistence_All.txt")
    instruments = [result.arbitration.instrument for result in results]
    by_inst = {result.arbitration.instrument: result for result in results}
    lines = [
        "APVA Interpretation Persistence Study v0.1 - Aggregate",
        "Decision-stability research only. Four fixed persistence regimes applied to Study 48 arbitration.",
        f"Instruments: {', '.join(instruments)}",
        "",
        "Aggregate Regime Table",
        "Regime | Name | "
        + instrument_columns(instruments, lambda inst: [f"ReplacementRate_{inst}", f"RunLength_{inst}", f"RiskRate_{inst}", f"Skew_{inst}", f"MeanDR_{inst}"])
        + " | ValidInstrumentCount | MeanReplacementRate | MeanRunLength | MeanRiskRate | MeanSkew | MeanDR | AgreementCount | AgreementPercent | AgreementDirection",
    ]
    aggregate_rows = []
    for regime, name in REGIMES:
        values = {inst: regime_summary(by_inst[inst], regime) for inst in instruments}
        valid_count = len(values)
        agree_count, agree_total, agree_pct, agree_direction = agreement(results, regime)
        cols = instrument_columns(
            instruments,
            lambda inst: [
                pct(values[inst]["replacement_rate"]),
                fmt(values[inst]["mean_run"]),
                pct(values[inst]["risk_rate"]),
                pct(values[inst]["skew"]),
                fmt(values[inst]["mean_dr"]),
            ],
        )
        row = {
            "regime": regime,
            "name": name,
            "replacement_rate": mean(value["replacement_rate"] for value in values.values()),
            "mean_run": mean(value["mean_run"] for value in values.values()),
            "risk_rate": mean(value["risk_rate"] for value in values.values()),
            "skew": mean(value["skew"] for value in values.values()),
            "mean_dr": mean(value["mean_dr"] for value in values.values()),
            "agreement": agree_pct,
        }
        aggregate_rows.append(row)
        lines.append(
            f"{regime} | {name} | {cols} | {valid_count} | {pct(row['replacement_rate'])} | {fmt(row['mean_run'])} | "
            f"{pct(row['risk_rate'])} | {pct(row['skew'])} | {fmt(row['mean_dr'])} | {agree_count} of {agree_total} | "
            f"{pct(agree_pct)} | {agree_direction}"
        )
    baseline = next(row for row in aggregate_rows if row["regime"] == "A")
    lines += [
        "",
        "Aggregate Improvement Table",
        "Regime | RelativeTo | ReplacementReduction | RunLengthIncrease | RiskReduction | SkewImprovement | MeanDRImprovement",
    ]
    for row in aggregate_rows:
        lines.append(
            f"{row['regime']} | Immediate Replacement | {pct(baseline['replacement_rate'] - row['replacement_rate'])} | "
            f"{fmt(row['mean_run'] - baseline['mean_run'])} | {pct(baseline['risk_rate'] - row['risk_rate'])} | "
            f"{pct(row['skew'] - baseline['skew'])} | {fmt(row['mean_dr'] - baseline['mean_dr'])}"
        )
    lines += ["", "Aggregate Conflict Table", "Regime | OverrideCount | OverrideRate | MeanOverrideRunLength | BarsSuppressed"]
    for regime, _ in REGIMES:
        summaries = aggregate_summaries(results, regime)
        total = sum(summary["count"] for summary in summaries)
        overrides = sum(summary["override_count"] for summary in summaries)
        lines.append(
            f"{regime} | {int(overrides)} | {pct(overrides / total if total else 0.0)} | "
            f"{fmt(mean(summary['mean_override_run'] for summary in summaries))} | {int(sum(summary['override_duration'] for summary in summaries))}"
        )
    lines += ["", "Aggregate Stability Table", "Regime | MeanRunLength | MedianRunLength | MaxRunLength | ReplacementRate | BarsPerReplacement"]
    pooled_rows = {}
    for regime, _ in REGIMES:
        pooled = pooled_stability(results, regime)
        pooled_rows[regime] = pooled
        lines.append(
            f"{regime} | {fmt(pooled['mean_run'])} | {fmt(pooled['median_run'])} | {int(pooled['max_run'])} | "
            f"{pct(pooled['replacement_rate'])} | {fmt(pooled['bars_per_replacement'])}"
        )
    lines += [
        "",
        "Aggregate Outcome Table",
        "Regime | "
        + instrument_columns(instruments, lambda inst: [f"Count_{inst}", f"Skew_{inst}", f"MeanDR_{inst}"])
        + " | ValidInstrumentCount | MeanSkew | MeanDR",
    ]
    for regime, _ in REGIMES:
        values = {inst: regime_summary(by_inst[inst], regime) for inst in instruments}
        cols = instrument_columns(instruments, lambda inst: [str(int(values[inst]["outcome_count"])), pct(values[inst]["skew"]), fmt(values[inst]["mean_dr"])])
        lines.append(f"{regime} | {cols} | {len(values)} | {pct(mean(value['skew'] for value in values.values()))} | {fmt(mean(value['mean_dr'] for value in values.values()))}")
    lines += [
        "",
        "Aggregate Fractal-Jump Table",
        "Regime | "
        + instrument_columns(instruments, lambda inst: [f"RiskCount_{inst}", f"RiskRate_{inst}"])
        + " | ValidInstrumentCount | MeanRiskRate",
    ]
    for regime, _ in REGIMES:
        values = {inst: regime_summary(by_inst[inst], regime) for inst in instruments}
        cols = instrument_columns(instruments, lambda inst: [str(int(values[inst]["risk_count"])), pct(values[inst]["risk_rate"])])
        lines.append(f"{regime} | {cols} | {len(values)} | {pct(mean(value['risk_rate'] for value in values.values()))}")

    def ranking(title: str, rows: Iterable[str]) -> None:
        lines.extend(["", title])
        lines.extend(list(rows)[:TOP_LIMIT])

    ranking("1. Lowest replacement rate", (f"{row['regime']} | {row['name']} | ReplacementRate={pct(row['replacement_rate'])}" for row in sorted(aggregate_rows, key=lambda x: x["replacement_rate"])))
    ranking("2. Longest interpretation persistence", (f"{row['regime']} | {row['name']} | MeanRunLength={fmt(row['mean_run'])}" for row in sorted(aggregate_rows, key=lambda x: -x["mean_run"])))
    ranking("3. Lowest fractal-jump risk", (f"{row['regime']} | {row['name']} | RiskRate={pct(row['risk_rate'])}" for row in sorted(aggregate_rows, key=lambda x: x["risk_rate"])))
    ranking("4. Highest outcome skew", (f"{row['regime']} | {row['name']} | OutcomeSkew={pct(row['skew'])}" for row in sorted(aggregate_rows, key=lambda x: -x["skew"])))
    ranking("5. Highest mean DR", (f"{row['regime']} | {row['name']} | MeanDR={fmt(row['mean_dr'])}" for row in sorted(aggregate_rows, key=lambda x: -x["mean_dr"])))
    ranking("6. Best replacement reduction", (f"{row['regime']} | {row['name']} | Reduction={pct(baseline['replacement_rate'] - row['replacement_rate'])}" for row in sorted(aggregate_rows, key=lambda x: -(baseline["replacement_rate"] - x["replacement_rate"]))))
    ranking("7. Best conflict suppression", (f"{regime} | OverrideRate={pct(mean(regime_summary(result, regime)['override_rate'] for result in results))}" for regime, _ in sorted(REGIMES, key=lambda x: mean(regime_summary(result, x[0])["override_rate"] for result in results), reverse=True)))
    ranking("8. Most stable regime", (f"{row['regime']} | {row['name']} | MeanRunLength={fmt(row['mean_run'])} | RiskRate={pct(row['risk_rate'])}" for row in sorted(aggregate_rows, key=lambda x: (-x["mean_run"], x["risk_rate"]))))
    ranking("9. Most instrument agreement", (f"{row['regime']} | {row['name']} | Agreement={pct(row['agreement'])}" for row in sorted(aggregate_rows, key=lambda x: -x["agreement"])))
    ranking("10. Best overall persistence regime", (f"{row['regime']} | {row['name']} | ReplacementRate={pct(row['replacement_rate'])} | MeanRunLength={fmt(row['mean_run'])} | RiskRate={pct(row['risk_rate'])}" for row in sorted(aggregate_rows, key=lambda x: (x["risk_rate"], x["replacement_rate"], -x["mean_run"]))))
    lines += ["", "Low-DoF audit"]
    append_audit(lines)
    lines += [
        "",
        "Mechanical research notes",
        "- All regimes reuse Study 48 scores exactly and alter only replacement timing.",
        "- Overall regime outcomes are diagnostic and remain equal because each regime labels the same bars.",
        "- Stability, risk, and override metrics measure decision persistence without adding APVA variables.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="APVA Interpretation Persistence Study v0.1")
    parser.add_argument("inputs", nargs="+", help="Evidence CSV files or directories")
    parser.add_argument("--out-root", default="Evidence/Output", help="Output root directory")
    args = parser.parse_args(argv)
    loaded = load_results(args.inputs)
    if not loaded:
        raise SystemExit("No input rows loaded.")
    results = [build_result(build_arbitration_result(result)) for result in loaded]
    for result in results:
        write_instrument_report(result, args.out_root)
    aggregate = write_aggregate_report(results, args.out_root)
    print(f"Wrote InterpretationPersistence reports under {args.out_root}")
    print(f"Aggregate: {aggregate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
