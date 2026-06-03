#!/usr/bin/env python3
"""
APVA Persistence Calibration Study v0.1

Compare eight fixed persistence regimes applied to Study 48's arbitration
stream. This is a decision-calibration study, not prediction, optimization,
fitting, parameter search, or trading research.
"""

from __future__ import annotations

import argparse
import os
import statistics as stats
from collections import Counter
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
    ("A", "Immediate Replacement", 1, 0.00, "Immediate"),
    ("B", "One-Bar Hold", 2, 0.00, "ConfirmationSatisfied"),
    ("C", "Two-Bar Confirmation", 2, 0.00, "ConfirmationSatisfied"),
    ("D", "Three-Bar Confirmation", 3, 0.00, "ConfirmationSatisfied"),
    ("E", "Margin 0.10", 1, 0.10, "MarginSatisfied"),
    ("F", "Margin 0.20", 1, 0.20, "MarginSatisfied"),
    ("G", "Hybrid", 2, 0.20, "HybridSatisfied"),
    ("H", "Hybrid+", 3, 0.20, "HybridPlusSatisfied"),
)
REGIME_BY_NAME = {row[0]: row for row in REGIMES}
FALSE_PERSISTENCE_BARS = 3
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
    pending_key: str
    pending_count: int
    lag_to_replacement: Optional[int]


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
    _, name, confirmation, margin, replacement_reason = REGIME_BY_NAME[regime]
    rows: List[PersistenceRow] = []
    active_key = ""
    active_type = "Unresolved"
    active_score = 0.0
    active_run = 0
    pending_key = ""
    pending_count = 0
    pending_first_index = 0
    for row in arbitrations:
        challenger_key = row.winner_key
        challenger_type = row.winner_type
        challenger_score = row.winner_score
        prior_run = active_run
        replacement = False
        reason = "None"
        lag: Optional[int] = None
        if not active_key:
            active_key = challenger_key
            active_type = challenger_type
            active_score = challenger_score
            active_run = 1
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
                    pending_first_index = row.index
            else:
                pending_key = ""
                pending_count = 0
            if regime == "A" and different:
                replacement = True
            elif different and pending_count >= confirmation and delta >= margin:
                replacement = True
            if replacement:
                reason = replacement_reason
                lag = row.index - pending_first_index
                active_key = challenger_key
                active_type = challenger_type
                active_score = challenger_score
                active_run = 1
                pending_key = ""
                pending_count = 0
            else:
                active_run += 1
        fractal_jump = replacement and prior_run <= 3 and active_type != rows[-1].active_type
        rows.append(
            PersistenceRow(
                row.index,
                challenger_type,
                challenger_key,
                challenger_score,
                active_type,
                active_key,
                active_score,
                active_run,
                replacement,
                reason,
                prior_run,
                fractal_jump,
                active_key != challenger_key,
                pending_key,
                pending_count,
                lag,
            )
        )
    return RegimeResult(regime, name, rows)


def build_result(arbitration: ArbitrationResult) -> StudyResult:
    return StudyResult(
        arbitration,
        {regime: run_regime(arbitration.arbitrations, regime) for regime, *_ in REGIMES},
    )


def runs(rows: Sequence[PersistenceRow], attribute: str = "active_key") -> List[Tuple[str, int]]:
    result: List[Tuple[str, int]] = []
    if not rows:
        return result
    current = str(getattr(rows[0], attribute))
    length = 1
    for row in rows[1:]:
        value = str(getattr(row, attribute))
        if value == current:
            length += 1
        else:
            result.append((current, length))
            current = value
            length = 1
    result.append((current, length))
    return result


def distribution(values: Iterable[int]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for value in values:
        if value >= 21:
            bucket = "21+"
        elif value >= 11:
            bucket = "11-20"
        elif value >= 6:
            bucket = "6-10"
        else:
            bucket = str(value)
        counts[bucket] += 1
    return counts


def half_life(values: Sequence[int]) -> int:
    if not values:
        return 0
    for bars in range(1, max(values) + 1):
        if sum(value <= bars for value in values) / len(values) >= 0.50:
            return bars
    return max(values)


def true_runs(rows: Sequence[PersistenceRow], predicate) -> List[int]:
    result: List[int] = []
    length = 0
    for row in rows:
        if predicate(row):
            length += 1
        elif length:
            result.append(length)
            length = 0
    if length:
        result.append(length)
    return result


def false_persistence_runs(rows: Sequence[PersistenceRow]) -> List[int]:
    return [length for length in true_runs(rows, lambda row: row.override) if length >= FALSE_PERSISTENCE_BARS]


def outcome_for(result: StudyResult, rows: Iterable[PersistenceRow]) -> Outcome:
    outcome = Outcome()
    for row in rows:
        outcome.add(directional_return(result.arbitration.bars, row.index, 5))
    return outcome


def regime_summary(result: StudyResult, regime: str) -> Dict[str, float]:
    rows = result.regimes[regime].rows
    run_lengths = [length for _, length in runs(rows)]
    override_lengths = true_runs(rows, lambda row: row.override)
    false_lengths = false_persistence_runs(rows)
    replacements = [row for row in rows if row.replacement]
    risks = [row for row in rows if row.fractal_jump_risk]
    lags = [row.lag_to_replacement for row in replacements if row.lag_to_replacement is not None]
    outcome = outcome_for(result, rows)
    total = len(rows)
    false_bars = sum(false_lengths)
    return {
        "count": total,
        "replacement_count": len(replacements),
        "replacement_rate": len(replacements) / total if total else 0.0,
        "bars_per_replacement": total / len(replacements) if replacements else 0.0,
        "mean_run": mean(run_lengths),
        "median_run": stats.median(run_lengths) if run_lengths else 0.0,
        "max_run": max(run_lengths) if run_lengths else 0.0,
        "half_life": half_life(run_lengths),
        "risk_count": len(risks),
        "risk_rate": len(risks) / total if total else 0.0,
        "mean_risk_run": mean(row.prior_run_length for row in risks),
        "risk_replacement_rate": len(risks) / len(replacements) if replacements else 0.0,
        "false_count": len(false_lengths),
        "false_rate": false_bars / total if total else 0.0,
        "mean_false_length": mean(false_lengths),
        "max_false_length": max(false_lengths) if false_lengths else 0.0,
        "mean_lag": mean(x for x in lags if x is not None),
        "median_lag": stats.median(lags) if lags else 0.0,
        "max_lag": max(lags) if lags else 0.0,
        "override_count": sum(row.override for row in rows),
        "override_rate": sum(row.override for row in rows) / total if total else 0.0,
        "mean_override_length": mean(override_lengths),
        "max_override_length": max(override_lengths) if override_lengths else 0.0,
        "bars_suppressed": sum(override_lengths),
        "outcome_count": outcome.count,
        "mean_dr": outcome.mean_dr,
        "median_dr": outcome.median_dr,
        "continuation": outcome.continuation_rate,
        "failure": outcome.failure_rate,
        "flat": outcome.flat_rate,
        "skew": outcome.skew,
    }


def normalize_lower(value: float, values: Sequence[float]) -> float:
    maximum = max(values) if values else 0.0
    return value / maximum if maximum > 0 else 0.0


def calibration_rows(result: StudyResult) -> List[Dict[str, float]]:
    summaries = {regime: regime_summary(result, regime) for regime, *_ in REGIMES}
    replacements = [summary["replacement_rate"] for summary in summaries.values()]
    risks = [summary["risk_rate"] for summary in summaries.values()]
    false_rates = [summary["false_rate"] for summary in summaries.values()]
    lags = [summary["mean_lag"] for summary in summaries.values()]
    result_rows = []
    for regime, name, *_ in REGIMES:
        summary = summaries[regime]
        replacement_score = 1.0 - normalize_lower(summary["replacement_rate"], replacements)
        risk_score = 1.0 - normalize_lower(summary["risk_rate"], risks)
        false_score = 1.0 - normalize_lower(summary["false_rate"], false_rates)
        lag_score = 1.0 - normalize_lower(summary["mean_lag"], lags)
        result_rows.append(
            {
                "regime": regime,
                "name": name,
                "replacement_score": replacement_score,
                "risk_score": risk_score,
                "false_score": false_score,
                "lag_score": lag_score,
                "score": replacement_score + risk_score + false_score + lag_score,
            }
        )
    return result_rows


def rendered_distribution(values: Iterable[int]) -> str:
    counts = distribution(values)
    return ", ".join(f"{bucket}={counts[bucket]}" for bucket in ("1", "2", "3", "4", "5", "6-10", "11-20", "21+"))


def append_stability(lines: List[str], result: StudyResult) -> None:
    lines.append(
        "Regime | Name | ReplacementRate | ReplacementCount | BarsPerReplacement | MeanRunLength | "
        "MedianRunLength | MaxRunLength | PersistenceHalfLife | RunLengthDistribution"
    )
    for regime, name, *_ in REGIMES:
        summary = regime_summary(result, regime)
        lines.append(
            f"{regime} | {name} | {pct(summary['replacement_rate'])} | {int(summary['replacement_count'])} | "
            f"{fmt(summary['bars_per_replacement'])} | {fmt(summary['mean_run'])} | {fmt(summary['median_run'])} | "
            f"{int(summary['max_run'])} | {int(summary['half_life'])} | "
            f"{rendered_distribution(length for _, length in runs(result.regimes[regime].rows))}"
        )


def append_risks(lines: List[str], result: StudyResult) -> None:
    lines.append("Regime | RiskCount | RiskRate | MeanRiskRunLength | RiskReplacementRate")
    for regime, *_ in REGIMES:
        summary = regime_summary(result, regime)
        lines.append(
            f"{regime} | {int(summary['risk_count'])} | {pct(summary['risk_rate'])} | "
            f"{fmt(summary['mean_risk_run'])} | {pct(summary['risk_replacement_rate'])}"
        )


def append_false_persistence(lines: List[str], result: StudyResult) -> None:
    lines.append("Regime | FalsePersistenceCount | FalsePersistenceRate | MeanFalsePersistenceLength | MaxFalsePersistenceLength")
    for regime, *_ in REGIMES:
        summary = regime_summary(result, regime)
        lines.append(
            f"{regime} | {int(summary['false_count'])} | {pct(summary['false_rate'])} | "
            f"{fmt(summary['mean_false_length'])} | {int(summary['max_false_length'])}"
        )


def append_lags(lines: List[str], result: StudyResult) -> None:
    lines.append("Regime | MeanLag | MedianLag | MaxLag | LagDistribution")
    for regime, *_ in REGIMES:
        summary = regime_summary(result, regime)
        lags = [row.lag_to_replacement for row in result.regimes[regime].rows if row.lag_to_replacement is not None]
        lines.append(
            f"{regime} | {fmt(summary['mean_lag'])} | {fmt(summary['median_lag'])} | {int(summary['max_lag'])} | "
            f"{rendered_distribution(int(lag) for lag in lags)}"
        )


def append_conflicts(lines: List[str], result: StudyResult) -> None:
    lines.append("Regime | OverrideCount | OverrideRate | MeanOverrideLength | MaxOverrideLength | BarsSuppressed")
    for regime, *_ in REGIMES:
        summary = regime_summary(result, regime)
        lines.append(
            f"{regime} | {int(summary['override_count'])} | {pct(summary['override_rate'])} | "
            f"{fmt(summary['mean_override_length'])} | {int(summary['max_override_length'])} | {int(summary['bars_suppressed'])}"
        )


def append_outcomes(lines: List[str], result: StudyResult) -> None:
    lines.append("Regime | Count | MeanDRFwd5 | MedianDRFwd5 | ContinuationRate5 | FailureRate5 | FlatRate5 | OutcomeSkew")
    for regime, *_ in REGIMES:
        summary = regime_summary(result, regime)
        lines.append(
            f"{regime} | {int(summary['outcome_count'])} | {fmt(summary['mean_dr'])} | {fmt(summary['median_dr'])} | "
            f"{pct(summary['continuation'])} | {pct(summary['failure'])} | {pct(summary['flat'])} | {pct(summary['skew'])}"
        )


def append_calibration(lines: List[str], result: StudyResult) -> None:
    lines.append("Regime | ReplacementScore | RiskScore | FalsePersistenceScore | LagScore | CalibrationScore")
    for row in calibration_rows(result):
        lines.append(
            f"{row['regime']} | {fmt(row['replacement_score'])} | {fmt(row['risk_score'])} | "
            f"{fmt(row['false_score'])} | {fmt(row['lag_score'])} | {fmt(row['score'])}"
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
            "No optimization.",
            "No fitting.",
            "No machine learning.",
            "No forward returns in scoring.",
            "No outcome-based scoring.",
        ]
    )


def write_instrument_report(result: StudyResult, out_root: str) -> str:
    arbitration = result.arbitration
    out_dir = os.path.join(out_root, arbitration.instrument)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"PersistenceCalibration_{arbitration.instrument}.txt")
    lines = [
        "APVA Persistence Calibration Study v0.1",
        "Decision-calibration research only. Eight fixed regimes. No optimization, fitting, machine learning, or outcome-based scoring.",
        "",
        "Diagnostics",
        f"Instrument: {arbitration.instrument}",
        f"Input path(s): {'; '.join(arbitration.source_paths)}",
        f"Total rows: {len(arbitration.bars)}",
        "",
        "1. Arbitration baseline",
        "Regime A reproduces Study 48 immediate replacement.",
        "",
        "2. Persistence regime summaries",
    ]
    append_stability(lines, result)
    lines += ["", "3. Stability analysis"]
    append_stability(lines, result)
    lines += ["", "4. Fractal-jump analysis"]
    append_risks(lines, result)
    lines += ["", "5. False persistence analysis"]
    append_false_persistence(lines, result)
    lines += ["", "6. Replacement lag analysis"]
    append_lags(lines, result)
    lines += ["", "7. Conflict suppression"]
    append_conflicts(lines, result)
    lines += ["", "8. Outcome diagnostics"]
    append_outcomes(lines, result)
    lines += ["", "9. Calibration analysis"]
    append_calibration(lines, result)
    lines += ["", "10. Low-DoF audit"]
    append_audit(lines)
    lines += [
        "",
        "11. Mechanical research notes",
        "- All persistence regimes are frozen in advance.",
        "- One-Bar Hold means one full additional challenger bar, so it replaces after two consecutive challenger bars.",
        "- False persistence is an override run lasting at least three bars.",
        "- Calibration normalization uses the maximum observed rate across the frozen regime set; lower burden scores higher.",
        "- Outcomes are diagnostics only and cannot affect scores or replacements.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def aggregate_summary(results: Sequence[StudyResult], regime: str) -> Dict[str, float]:
    summaries = [regime_summary(result, regime) for result in results]
    return {
        key: mean(summary[key] for summary in summaries)
        for key in ("replacement_rate", "risk_rate", "false_rate", "mean_lag", "skew", "mean_dr", "continuation", "failure")
    }


def aggregate_calibration(results: Sequence[StudyResult]) -> List[Dict[str, float]]:
    summaries = {regime: aggregate_summary(results, regime) for regime, *_ in REGIMES}
    replacements = [row["replacement_rate"] for row in summaries.values()]
    risks = [row["risk_rate"] for row in summaries.values()]
    false_rates = [row["false_rate"] for row in summaries.values()]
    lags = [row["mean_lag"] for row in summaries.values()]
    rows = []
    for regime, name, *_ in REGIMES:
        summary = summaries[regime]
        replacement_score = 1.0 - normalize_lower(summary["replacement_rate"], replacements)
        risk_score = 1.0 - normalize_lower(summary["risk_rate"], risks)
        false_score = 1.0 - normalize_lower(summary["false_rate"], false_rates)
        lag_score = 1.0 - normalize_lower(summary["mean_lag"], lags)
        rows.append(
            {
                "regime": regime,
                "name": name,
                "replacement_score": replacement_score,
                "risk_score": risk_score,
                "false_score": false_score,
                "lag_score": lag_score,
                "score": replacement_score + risk_score + false_score + lag_score,
            }
        )
    return rows


def pooled_lengths(results: Sequence[StudyResult], regime: str, mode: str) -> List[int]:
    if mode == "active":
        return [length for result in results for _, length in runs(result.regimes[regime].rows)]
    if mode == "false":
        return [length for result in results for length in false_persistence_runs(result.regimes[regime].rows)]
    if mode == "override":
        return [length for result in results for length in true_runs(result.regimes[regime].rows, lambda row: row.override)]
    raise ValueError(mode)


def write_aggregate_report(results: Sequence[StudyResult], out_root: str) -> str:
    out_dir = os.path.join(out_root, "PersistenceCalibration")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "PersistenceCalibration_All.txt")
    instruments = [result.arbitration.instrument for result in results]
    by_inst = {result.arbitration.instrument: result for result in results}
    calibration = aggregate_calibration(results)
    lines = [
        "APVA Persistence Calibration Study v0.1 - Aggregate",
        "Decision-calibration research only. Eight fixed persistence regimes.",
        f"Instruments: {', '.join(instruments)}",
        "",
        "Aggregate Regime Table",
        "Regime | Name | "
        + " | ".join(
            field
            for instrument in instruments
            for field in (
                f"ReplacementRate_{instrument}",
                f"RiskRate_{instrument}",
                f"FalsePersistenceRate_{instrument}",
                f"Lag_{instrument}",
            )
        )
        + " | ValidInstrumentCount | MeanReplacementRate | MeanRiskRate | MeanFalsePersistenceRate | MeanLag",
    ]
    rows = []
    for regime, name, *_ in REGIMES:
        values = {instrument: regime_summary(by_inst[instrument], regime) for instrument in instruments}
        aggregate = aggregate_summary(results, regime)
        cols = " | ".join(
            value
            for instrument in instruments
            for value in (
                pct(values[instrument]["replacement_rate"]),
                pct(values[instrument]["risk_rate"]),
                pct(values[instrument]["false_rate"]),
                fmt(values[instrument]["mean_lag"]),
            )
        )
        rows.append((regime, name, aggregate))
        lines.append(
            f"{regime} | {name} | {cols} | {len(instruments)} | {pct(aggregate['replacement_rate'])} | "
            f"{pct(aggregate['risk_rate'])} | {pct(aggregate['false_rate'])} | {fmt(aggregate['mean_lag'])}"
        )
    lines += ["", "Aggregate Stability Table", "Regime | MeanRunLength | MedianRunLength | MaxRunLength | PersistenceHalfLife | BarsPerReplacement"]
    for regime, *_ in REGIMES:
        lengths = pooled_lengths(results, regime, "active")
        total = sum(len(result.regimes[regime].rows) for result in results)
        replacements = sum(regime_summary(result, regime)["replacement_count"] for result in results)
        lines.append(
            f"{regime} | {fmt(mean(lengths))} | {fmt(stats.median(lengths) if lengths else 0.0)} | "
            f"{max(lengths) if lengths else 0} | {half_life(lengths)} | {fmt(total / replacements if replacements else 0.0)}"
        )
    lines += ["", "Aggregate Fractal-Jump Table", "Regime | " + " | ".join(f"RiskCount_{instrument}" for instrument in instruments) + " | MeanRiskRate"]
    for regime, *_ in REGIMES:
        values = {instrument: regime_summary(by_inst[instrument], regime) for instrument in instruments}
        lines.append(f"{regime} | {' | '.join(str(int(values[instrument]['risk_count'])) for instrument in instruments)} | {pct(mean(value['risk_rate'] for value in values.values()))}")
    lines += ["", "Aggregate False Persistence Table", "Regime | " + " | ".join(f"FalsePersistenceCount_{instrument}" for instrument in instruments) + " | MeanFalsePersistenceRate | MeanFalsePersistenceLength | MaxFalsePersistenceLength"]
    for regime, *_ in REGIMES:
        values = {instrument: regime_summary(by_inst[instrument], regime) for instrument in instruments}
        lengths = pooled_lengths(results, regime, "false")
        lines.append(
            f"{regime} | {' | '.join(str(int(values[instrument]['false_count'])) for instrument in instruments)} | "
            f"{pct(mean(value['false_rate'] for value in values.values()))} | {fmt(mean(lengths))} | {max(lengths) if lengths else 0}"
        )
    lines += ["", "Aggregate Lag Table", "Regime | " + " | ".join(f"MeanLag_{instrument}" for instrument in instruments) + " | MeanLag | MedianLag | MaxLag"]
    for regime, *_ in REGIMES:
        values = {instrument: regime_summary(by_inst[instrument], regime) for instrument in instruments}
        lags = [int(row.lag_to_replacement) for result in results for row in result.regimes[regime].rows if row.lag_to_replacement is not None]
        lines.append(
            f"{regime} | {' | '.join(fmt(values[instrument]['mean_lag']) for instrument in instruments)} | "
            f"{fmt(mean(lags))} | {fmt(stats.median(lags) if lags else 0.0)} | {max(lags) if lags else 0}"
        )
    lines += ["", "Aggregate Conflict Table", "Regime | OverrideCount | OverrideRate | MeanOverrideLength | BarsSuppressed"]
    for regime, *_ in REGIMES:
        lengths = pooled_lengths(results, regime, "override")
        total = sum(len(result.regimes[regime].rows) for result in results)
        suppressed = sum(lengths)
        lines.append(f"{regime} | {suppressed} | {pct(suppressed / total if total else 0.0)} | {fmt(mean(lengths))} | {suppressed}")
    lines += ["", "Aggregate Outcome Table", "Regime | MeanSkew | MeanDR | ContinuationRate | FailureRate"]
    for regime, *_ in REGIMES:
        aggregate = aggregate_summary(results, regime)
        lines.append(f"{regime} | {pct(aggregate['skew'])} | {fmt(aggregate['mean_dr'])} | {pct(aggregate['continuation'])} | {pct(aggregate['failure'])}")
    lines += ["", "Aggregate Calibration Table", "Regime | ReplacementScore | RiskScore | FalsePersistenceScore | LagScore | CalibrationScore"]
    for row in calibration:
        lines.append(f"{row['regime']} | {fmt(row['replacement_score'])} | {fmt(row['risk_score'])} | {fmt(row['false_score'])} | {fmt(row['lag_score'])} | {fmt(row['score'])}")

    def ranking(title: str, rendered: Iterable[str]) -> None:
        lines.extend(["", title])
        lines.extend(list(rendered)[:TOP_LIMIT])

    ranking("1. Lowest replacement rate", (f"{regime} | {name} | ReplacementRate={pct(row['replacement_rate'])}" for regime, name, row in sorted(rows, key=lambda item: item[2]["replacement_rate"])))
    ranking("2. Lowest fractal-jump risk", (f"{regime} | {name} | RiskRate={pct(row['risk_rate'])}" for regime, name, row in sorted(rows, key=lambda item: item[2]["risk_rate"])))
    ranking("3. Lowest false persistence rate", (f"{regime} | {name} | FalsePersistenceRate={pct(row['false_rate'])}" for regime, name, row in sorted(rows, key=lambda item: item[2]["false_rate"])))
    ranking("4. Lowest lag", (f"{regime} | {name} | MeanLag={fmt(row['mean_lag'])}" for regime, name, row in sorted(rows, key=lambda item: item[2]["mean_lag"])))
    ranking("5. Best conflict suppression", (f"{regime} | {name} | OverrideRate={pct(mean(regime_summary(result, regime)['override_rate'] for result in results))}" for regime, name, *_ in sorted(REGIMES, key=lambda item: mean(regime_summary(result, item[0])["override_rate"] for result in results), reverse=True)))
    ranking("6. Longest interpretation stability", (f"{regime} | {name} | MeanRunLength={fmt(mean(pooled_lengths(results, regime, 'active')))}" for regime, name, *_ in sorted(REGIMES, key=lambda item: -mean(pooled_lengths(results, item[0], "active")))))
    ranking("7. Best outcome skew", (f"{regime} | {name} | OutcomeSkew={pct(row['skew'])}" for regime, name, row in sorted(rows, key=lambda item: -item[2]["skew"])))
    ranking("8. Best mean DR", (f"{regime} | {name} | MeanDR={fmt(row['mean_dr'])}" for regime, name, row in sorted(rows, key=lambda item: -item[2]["mean_dr"])))
    ranking("9. Highest calibration score", (f"{row['regime']} | {row['name']} | CalibrationScore={fmt(row['score'])}" for row in sorted(calibration, key=lambda item: -item["score"])))
    ranking("10. Best balanced persistence regime", (f"{row['regime']} | {row['name']} | CalibrationScore={fmt(row['score'])}" for row in sorted(calibration, key=lambda item: -item["score"])))
    lines += ["", "Low-DoF audit"]
    append_audit(lines)
    lines += [
        "",
        "Mechanical research notes",
        "- Calibration is a fixed equal-weight diagnostic across the frozen regime set.",
        "- Lower replacement, risk, false persistence, and lag burdens score higher.",
        "- No regime was selected or tuned from forward return outcomes.",
        "- Outcome tables remain downstream diagnostics only.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="APVA Persistence Calibration Study v0.1")
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
    print(f"Wrote PersistenceCalibration reports under {args.out_root}")
    print(f"Aggregate: {aggregate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
