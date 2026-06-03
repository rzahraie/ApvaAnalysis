#!/usr/bin/env python3
"""APVA Final Specification Study v1.0.

Study 84 freezes the APVA v1.0 framework for implementation handoff.

This is a consolidation study. It summarizes, validates, and freezes findings
from Studies 74-83. No new theory, states, families, optimization, fitting,
machine learning, trading rules, or forward returns in state definitions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from APVA_NeutralFailureTiming_83 import (
    build_result as build_failure_timing_result,
    load_contexts,
    recommended_threshold,
    replication_rows,
)
from APVA_RealTimeStateMachine_82 import (
    build_result as build_realtime_result,
    instrument_replication_rows,
    transition_rows,
)
from APVA_StructuralLifeCycle_44 import ensure_dir, fmt, pct


FINAL_STATES = (
    "NeutralFormation",
    "NeutralMaturation",
    "LateNeutral",
    "DestinationSelection",
    "Excursion",
    "ReturnToNeutral",
)

STUDY_CONCLUSIONS = (
    ("74", "Does the Neutral lifecycle map to real OHLCV behavior?", "NeutralAsLifecycle: Age1 gateway, Age3 stabilization, Age4/5 equilibrium split, Age6-10 stale equilibrium, Age11+ drift.", "Confirmed", "ModerateReplication"),
    ("75", "What causes Neutral equilibrium to fail?", "Neutral failure pressure is mainly range/true-range expansion, with final excursion pressure most associated with volume expansion.", "Confirmed", "ModerateReplication"),
    ("76", "Can destination family be distinguished before excursion?", "Destination signatures exist, especially for major families after Neutral_Age6-10 failure.", "Partially Confirmed", "ModerateReplication"),
    ("77", "Are Study 76 destination signatures robust?", "Major destination signatures survived robustness testing; sparse destinations remain constrained.", "Confirmed", "ModerateReplication"),
    ("78", "What happens after destination formation?", "Destination families have distinct lifecycles; many excursions are brief and return to Neutral.", "Confirmed", "ModerateReplication"),
    ("79", "How is Neutral formed?", "Neutral forms from non-neutral collapse/decay/compression pathways, with Mixed and Compression acting as routing regions.", "Confirmed", "ModerateReplication"),
    ("80", "Can APVA be represented as a closed loop?", "DiffuseClosedLoop with 99.53% Neutral return completion; phase model outperformed full StateAge for next-phase prediction.", "Confirmed", "StrongReplication"),
    ("81", "What is the smallest useful signal set?", "PracticalMinimalSet; state plus selected market observables retained most useful structure; graph-only and market-only were insufficient.", "Confirmed", "StrongReplication"),
    ("82", "Can APVA be represented causally in real time?", "PracticalRealTimeMachine; destination detection is causal and strong, but hard NeutralFailure phase is weak.", "Partially Confirmed", "ModerateReplication"),
    ("83", "Can causal NeutralFailure timing improve?", "WarningOverlayPreferred; NeutralFailure should be an overlay score rather than a hard phase.", "Confirmed", "ModerateReplication"),
)

STATE_DEFINITIONS = (
    ("NeutralFormation", "Initialize or reset the loop at NeutralProcessing_Age1.", "Current StructuralState=NeutralProcessing and AgeBucket=1.", "Age advances to NeutralMaturation or data ends.", "NeutralMaturation"),
    ("NeutralMaturation", "Track early neutral stabilization.", "NeutralProcessing Age2-Age4 after formation.", "Age enters LateNeutral or non-neutral state appears.", "LateNeutral, DestinationSelection"),
    ("LateNeutral", "Represent mature, stale, or long neutral equilibrium.", "NeutralProcessing Age5, Age6-10, Age11-20, or Age21+.", "Non-neutral state appears; FailureWarningScore may indicate pressure before exit.", "DestinationSelection, NeutralMaturation"),
    ("DestinationSelection", "Classify the first non-neutral family after Neutral.", "First non-neutral bar after neutral context.", "Subsequent non-neutral continuation or immediate return routing.", "Excursion, ReturnToNeutral, NeutralFormation"),
    ("Excursion", "Track non-neutral continuation after destination formation.", "Non-neutral state after DestinationSelection.", "DecayToNeutral, NeutralProcessing_Age1, or terminal data.", "ReturnToNeutral, NeutralFormation, Excursion"),
    ("ReturnToNeutral", "Represent final non-neutral routing before Neutral reset.", "DecayToNeutral or final non-neutral context before NeutralProcessing_Age1.", "NeutralProcessing_Age1 appears.", "NeutralFormation"),
)

CORE_SIGNALS = (
    ("LoopPhase", "CoreSignal", "Primary compact representation of closed-loop position."),
    ("DestinationFamily", "Support/Core operational", "Known causally once non-neutral state appears; required for excursion interpretation."),
    ("BodyRelativeToPrevious", "SupportSignal", "Captures directional body expansion/collapse."),
    ("VolumeRelativeToRollingMean", "SupportSignal", "Captures participation versus recent baseline."),
    ("VolumeRelativeToPrevious", "SupportSignal", "Captures immediate participation change."),
    ("EfficiencyRatio", "CoreSignal", "Directional efficiency; removal caused material loss in Study 81."),
    ("CloseLocation", "CoreSignal", "Intrabar close pressure; removal caused material loss in Study 81."),
)

SUPPORT_SIGNALS = (
    ("RangeRelativeToPrevious", "Useful in failure pressure despite low standalone ablation necessity."),
    ("VolumeRelativeToSessionMean", "Useful participation context versus session baseline."),
)

REJECTED_SIGNALS = (
    ("MemoryStrength", "Redundant in Study 81; graph-only model weak."),
    ("BranchEntropy", "Redundant in Study 81; graph-only model weak."),
    ("FormationSourceFamily", "Redundant for minimal signal set."),
    ("MaxNeutralAgeReached", "Future/completed-loop hindsight; not allowed in causal implementation."),
    ("TrueRangeRelativeToPrevious", "Useful diagnostically but redundant in final minimal set."),
)

REJECTED_CONCEPTS = (
    ("GraphFamily importance", "Removed as core implementation driver.", "Graph-only Top1 was 47.59% and MemoryStrength/BranchEntropy were redundant."),
    ("NeutralFailure hard phase", "Removed from final state list.", "Study 82 detection was 25.53%; Study 83 improved timing as overlay."),
    ("Large StateAge hierarchy", "Compressed away from final runtime state machine.", "MinimalLoopPhase Top1 exceeded FullStateAge in Study 80."),
    ("Market-only model", "Rejected as complete representation.", "MarketOnly Top1 was 40.26% in Study 81."),
    ("Graph-only model", "Rejected as complete representation.", "GraphOnly Top1 was 47.59% in Study 81."),
)

RETAINED_CONCEPTS = (
    ("Neutral attractor", "Neutral return completion was 99.53% in Study 80."),
    ("LoopPhase", "Best compact causal backbone."),
    ("Destination families", "Causally known at first non-neutral bar and robustly useful."),
    ("Failure pressure", "Retained as FailureWarningScore overlay."),
    ("Excursion lifecycle", "Studies 78-79 closed destination-to-return behavior."),
    ("State structure", "State-only and state-market hybrid representations retained meaningful structure."),
)

IMPLEMENTATION_MODULES = (
    ("State Classifier", "Read APVA StructuralState/AgeBucket/StateAgeNode from live bars."),
    ("Loop Phase Engine", "Maintain final six-state loop phase machine causally."),
    ("Destination Detector", "Set DestinationFamily when first non-neutral state appears."),
    ("Failure Warning Engine", "Compute FailureWarningScore 0-6 and threshold overlay."),
    ("State Machine Manager", "Apply transition rules, persist current phase, and emit diagnostics."),
    ("Export / Diagnostics", "Write replay-safe state, warning, destination, and transition records."),
)

DATA_STRUCTURES = (
    ("StateRecord", "Timestamp, Instrument, StructuralState, AgeBucket, StateAgeNode, PreviousStateAgeNode."),
    ("PhaseRecord", "Timestamp, LoopPhase, PreviousLoopPhase, EntryReason, ExitReason."),
    ("TransitionRecord", "FromState, ToState, Count, Probability, Instrument."),
    ("WarningRecord", "FailureWarningScore, RangePressure, BodyPressure, VolumePressure, EfficiencyPressure, CloseExtremePressure, PolarityInstability."),
    ("DestinationRecord", "DestinationFamily, FirstDestinationNode, PriorNeutralAge, DetectionTimestamp."),
)

TESTING_PROTOCOL = (
    ("Unit tests", "Each state entry/exit rule and warning component must pass fixed examples."),
    ("Transition tests", "Allowed transition matrix must reject impossible phase jumps."),
    ("Replay tests", "Historical replay must match causal Study 82/83 outputs within expected tolerances."),
    ("Cross-instrument tests", "6E, CL, and NQ reports must all generate and preserve replication class."),
    ("Failure-warning tests", "FailureWarningScore threshold >=3 must reproduce timing metrics from Study 83."),
    ("Real-time tests", "No NextNode, future state, or completed-loop information may be accessed."),
)

LIMITATIONS = (
    ("No DOM data", "APVA v1.0 uses bar evidence only."),
    ("No historical order book", "No depth reconstruction is included."),
    ("No delta", "VolumePolarity is available, but true bid/ask delta is absent."),
    ("No options flow", "External derivative flow is not used."),
    ("No external market structure", "Framework is self-contained in APVA state/OHLCV evidence."),
    ("Diffuse loop behavior", "Study 80 classified the loop as DiffuseClosedLoop, not StableClosedLoop."),
    ("Failure detection limitations", "Warning overlay improves hard phase detection but remains noisy."),
)

FUTURE_RESEARCH = (
    ("State Machine Research", "Complete for APVA v1.0 implementation handoff."),
    ("APVA Dynamics", "What creates failure pressure?"),
    ("APVA Dynamics", "Is Neutral an attractor in a deeper dynamical sense?"),
    ("APVA Dynamics", "What is loop tension?"),
    ("APVA Dynamics", "What generates destination selection?"),
    ("APVA Dynamics", "What quantity, if any, is conserved through the loop?"),
)


@dataclass
class FinalResult:
    instrument: str
    source_paths: list
    realtime: object
    timing: object
    transition_matrix: list
    recommendation: tuple[str, str, str, str]


def transition_matrix_for(records: list, instruments: list[str]) -> list:
    return transition_rows(records)


def replication_class(values: list[float], high: float, moderate: float) -> str:
    if not values:
        return "WeakReplication"
    spread = max(values) - min(values) if len(values) > 1 else 0.0
    if min(values) >= high and spread <= 0.10:
        return "StrongReplication"
    if min(values) >= moderate:
        return "ModerateReplication"
    return "WeakReplication"


def build_final_result(instrument: str, source_paths: list, contexts: list, instrument_results: list[FinalResult] | None = None) -> FinalResult:
    realtime = build_realtime_result(instrument, source_paths, contexts)
    timing = build_failure_timing_result(instrument, source_paths, contexts)
    transition_matrix = transition_matrix_for(realtime.records, [context.instrument for context in contexts])
    if timing.thresholds:
        threshold = recommended_threshold(timing.thresholds)
    else:
        threshold = None
    if realtime.agreement.top1 >= 0.65 and threshold and threshold.recall >= 0.50:
        recommendation = ("StrongResearchSuccess", "ImplementAndValidate", "Recommended", "APVA v1.0 is causally implementable with a warning overlay for failure pressure.")
    elif realtime.agreement.top1 >= 0.50:
        recommendation = ("ResearchSuccess", "PrototypeOnly", "Recommended", "Framework is coherent but needs validation before production use.")
    else:
        recommendation = ("PartialResearchSuccess", "PrototypeOnly", "Optional", "Causal agreement is too weak for implementation beyond prototype.")
    return FinalResult(instrument, source_paths, realtime, timing, transition_matrix, recommendation)


def line_list(items: list[str] | tuple[str, ...]) -> str:
    return ", ".join(items) if items else "None"


def append_summary(lines: list[str], result: FinalResult) -> None:
    threshold = recommended_threshold(result.timing.thresholds) if result.timing.thresholds else None
    lines += [
        "",
        "1. Executive Summary",
        "Original goal | Convert APVA from a retrospective StateAge research framework into a compact causal specification suitable for implementation.",
        "Major discoveries | Neutral is the loop attractor; APVA is best modeled as a diffuse closed loop; destination selection is causal once non-neutral appears; hard NeutralFailure should become a warning overlay.",
        "Rejected ideas | Large StateAge runtime hierarchy, graph-only model, market-only model, and hard NeutralFailure phase.",
        "Retained ideas | Six-state loop model, DestinationFamily, selected OHLCV-derived signals, and FailureWarningScore.",
        "Final architecture | NeutralFormation -> NeutralMaturation -> LateNeutral -> DestinationSelection -> Excursion -> ReturnToNeutral -> NeutralFormation, plus FailureWarningScore 0-6.",
    ]
    if threshold:
        lines.append(f"Operational failure threshold | FailureWarningScore >= {threshold.threshold}; detection {pct(threshold.recall)}, false positives {pct(threshold.false_positive)}, precision {pct(threshold.precision)}.")


def append_common(lines: list[str], result: FinalResult, replication_rows_final: list | None = None) -> None:
    append_summary(lines, result)

    lines += ["", "2. Study Conclusions Table", "Study | Question | Result | Confidence | Replication"]
    for row in STUDY_CONCLUSIONS:
        lines.append(" | ".join(row))

    lines += ["", "3. Final State Table", "State | Purpose | Entry | Exit | AllowedTransitions"]
    for row in STATE_DEFINITIONS:
        lines.append(" | ".join(row))

    lines += ["", "4. Transition Matrix", "FromState | ToState | Count | Probability | ReplicationCount"]
    for row in result.transition_matrix:
        if row.from_phase == "NeutralFailure" or row.to_phase == "NeutralFailure":
            continue
        lines.append(f"{row.from_phase} | {row.to_phase} | {row.count} | {pct(row.probability)} | {row.replication_count}")

    threshold = recommended_threshold(result.timing.thresholds) if result.timing.thresholds else None
    lines += ["", "5. Failure Logic Table", "Component | Rule | ScoreContribution"]
    components = (
        ("RangePressure", "RangeRelativeToPrevious > 1 or TrueRangeRelativeToPrevious > 1", "1"),
        ("BodyPressure", "BodyRelativeToPrevious > 1", "1"),
        ("VolumePressure", "VolumeRelativeToRollingMean > 1 or VolumeRelativeToPrevious > 1", "1"),
        ("EfficiencyPressure", "EfficiencyRatio increasing vs prior bar or EfficiencyRatio >= 0.60", "1"),
        ("CloseExtremePressure", "CloseLocation <= 0.20 or CloseLocation >= 0.80", "1"),
        ("PolarityInstability", "VolumePolarity differs from prior bar", "1"),
    )
    for row in components:
        lines.append(" | ".join(row))
    if threshold:
        lines += [
            "Threshold | DetectionRate | FalsePositiveRate | Precision | Recall",
            f">={threshold.threshold} | {pct(threshold.detection_at_failure)} | {pct(threshold.false_positive)} | {pct(threshold.precision)} | {pct(threshold.recall)}",
        ]

    lines += ["", "6. Core Signal Table", "Signal | Class | Reason"]
    for row in CORE_SIGNALS:
        lines.append(" | ".join(row))
    lines += ["", "7. Support Signal Table", "Signal | Reason"]
    for row in SUPPORT_SIGNALS:
        lines.append(" | ".join(row))
    lines += ["", "8. Rejected Signal Table", "Signal | Reason"]
    for row in REJECTED_SIGNALS:
        lines.append(" | ".join(row))

    lines += ["", "9. Real-Time Performance Table", "Scope | Top1Agreement | Top2Agreement | DestinationAccuracy | FailureThreshold | FailureDetection | FailureFalsePositive | FailurePrecision"]
    failure_threshold = f">={threshold.threshold}" if threshold else "N/A"
    lines.append(
        f"{result.instrument} | {pct(result.realtime.agreement.top1)} | {pct(result.realtime.agreement.top2)} | "
        f"{pct(result.realtime.destination_accuracy)} | {failure_threshold} | "
        f"{pct(threshold.detection_at_failure if threshold else 0.0)} | {pct(threshold.false_positive if threshold else 0.0)} | "
        f"{pct(threshold.precision if threshold else 0.0)}"
    )

    lines += ["", "10. Replication Table", "Area | 6E | CL | NQ | Classification"]
    if replication_rows_final:
        by_inst = {row.instrument: row for row in replication_rows_final}
        agreements = [row.agreement for row in replication_rows_final]
        failures = [row.failure_detection for row in replication_rows_final]
        destinations = [row.destination_accuracy for row in replication_rows_final]
        lines.append(f"State machine | {pct(by_inst.get('6E').agreement if by_inst.get('6E') else 0)} | {pct(by_inst.get('CL').agreement if by_inst.get('CL') else 0)} | {pct(by_inst.get('NQ').agreement if by_inst.get('NQ') else 0)} | {replication_class(agreements, 0.65, 0.50)}")
        lines.append(f"Failure logic | {pct(by_inst.get('6E').failure_detection if by_inst.get('6E') else 0)} | {pct(by_inst.get('CL').failure_detection if by_inst.get('CL') else 0)} | {pct(by_inst.get('NQ').failure_detection if by_inst.get('NQ') else 0)} | {replication_class(failures, 0.50, 0.25)}")
        lines.append(f"Destination detection | {pct(by_inst.get('6E').destination_accuracy if by_inst.get('6E') else 0)} | {pct(by_inst.get('CL').destination_accuracy if by_inst.get('CL') else 0)} | {pct(by_inst.get('NQ').destination_accuracy if by_inst.get('NQ') else 0)} | {replication_class(destinations, 0.95, 0.75)}")
    else:
        lines.append("State machine | N/A | N/A | N/A | Per-instrument only")

    lines += ["", "11. Final Loop Model Table", "Loop | Classification | Evidence"]
    lines.append("NeutralFormation -> NeutralMaturation -> LateNeutral -> DestinationSelection -> Excursion -> ReturnToNeutral -> NeutralFormation | DiffuseClosedLoop | Study 80 completion rate 99.53%; high path entropy prevents StableClosedLoop.")

    lines += ["", "12. Rejected Concepts Table", "Concept | ReasonRemoved | Evidence"]
    for row in REJECTED_CONCEPTS:
        lines.append(" | ".join(row))

    lines += ["", "13. Retained Concepts Table", "Concept | ReasonRetained"]
    for row in RETAINED_CONCEPTS:
        lines.append(" | ".join(row))

    lines += ["", "14. Implementation Module Table", "Module | Responsibility"]
    for row in IMPLEMENTATION_MODULES:
        lines.append(" | ".join(row))

    lines += ["", "15. Data Structure Table", "Structure | Fields"]
    for row in DATA_STRUCTURES:
        lines.append(" | ".join(row))

    lines += ["", "16. Testing Protocol Table", "TestClass | AcceptanceCriteria"]
    for row in TESTING_PROTOCOL:
        lines.append(" | ".join(row))

    lines += ["", "17. Known Limitations Table", "Limitation | Detail"]
    for row in LIMITATIONS:
        lines.append(" | ".join(row))

    lines += ["", "18. Future Research Table", "Branch | QuestionOrStatus"]
    for row in FUTURE_RESEARCH:
        lines.append(" | ".join(row))

    rec = result.recommendation
    lines += ["", "19. Final Recommendation Table", "ResearchConclusion | ImplementationRecommendation | DynamicsRecommendation | Reason"]
    lines.append(" | ".join(rec))


def rankings(result: FinalResult) -> list[str]:
    return [
        "",
        "RANKINGS",
        "1. Most important discoveries: Neutral attractor; DiffuseClosedLoop; causal destination detection; warning overlay failure logic; minimal signal set.",
        "2. Strongest replicated findings: Neutral loop closure; destination detection; practical real-time state machine.",
        "3. Most valuable retained signals: LoopPhase; DestinationFamily; EfficiencyRatio; CloseLocation; BodyRelativeToPrevious; VolumeRelativeToRollingMean.",
        "4. Most important rejected concepts: hard NeutralFailure phase; graph-only model; market-only model; full StateAge runtime hierarchy.",
        "5. Highest-confidence conclusions: Neutral is central; final model is causal; failure pressure must be overlayed.",
        "6. Biggest surprises: MinimalLoopPhase beat FullStateAge; graph signals were redundant; destination accuracy was 100% once non-neutral appeared.",
        "7. Largest simplifications: 40+ StateAge nodes reduced to six operational phases plus one warning score.",
        "8. Most important implementation priorities: causal state classifier; warning engine; replay diagnostics; no-lookahead tests.",
        "9. Highest-priority future research questions: loop tension; source of failure pressure; attractor dynamics; destination generation.",
        "10. Final APVA v1.0 summary: implement and validate a six-state real-time loop machine with FailureWarningScore overlay.",
        "",
        "RESEARCH NOTES",
        "This is the final APVA v1.0 consolidation study.",
        "No discovery.",
        "No expansion.",
        "Freeze the framework.",
        "Create the implementation handoff.",
        "Create the bridge to future APVA Dynamics work.",
    ]


def write_report(result: FinalResult, out_root: Path, aggregate: bool = False, replication_rows_final: list | None = None) -> None:
    if aggregate:
        path = out_root / "FinalSpec84" / "FinalSpec84_All.txt"
        title = "APVA Final Specification Study v1.0 - Aggregate"
    else:
        path = out_root / result.instrument / f"FinalSpec84_{result.instrument}.txt"
        title = f"APVA Final Specification Study v1.0 - {result.instrument}"
    ensure_dir(path.parent)
    lines = [
        title,
        "=" * 96,
        f"Instrument: {result.instrument}",
        "Input path(s): " + "; ".join(str(path) for path in result.source_paths),
        "Purpose: Freeze APVA v1.0 framework for NinjaTrader implementation, real-time operation, validation, and future Dynamics research.",
    ]
    append_common(lines, result, replication_rows_final)
    lines += [
        "",
        "20. Low-DoF Audit",
        "No new states introduced.",
        "No new families introduced.",
        "No machine learning.",
        "No optimization.",
        "No fitting.",
        "No trading rules.",
        "No forward returns used in state definitions.",
    ]
    lines += rankings(result)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(result: FinalResult) -> None:
    if not result.realtime.records:
        raise RuntimeError(f"{result.instrument}: missing real-time records.")
    if not result.timing.thresholds:
        raise RuntimeError(f"{result.instrument}: missing failure timing thresholds.")
    if not result.transition_matrix:
        raise RuntimeError(f"{result.instrument}: missing transition matrix.")
    if not result.recommendation[0]:
        raise RuntimeError(f"{result.instrument}: missing recommendation.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Evidence CSV files or directories")
    parser.add_argument("--out-root", default="Evidence/Output", help="Output root directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _loaded, contexts = load_contexts(args.inputs)
    instrument_results = []
    for context in contexts:
        result = build_final_result(context.instrument, context.source_paths, [context])
        validate(result)
        instrument_results.append(result)

    aggregate_paths = [path for context in contexts for path in context.source_paths]
    aggregate_result = build_final_result("ALL", aggregate_paths, contexts, instrument_results)
    validate(aggregate_result)

    rt_replication = instrument_replication_rows([result.realtime for result in instrument_results])
    failure_rep_by_inst = {}
    for result in instrument_results:
        threshold = recommended_threshold(result.timing.thresholds)
        failure_rep_by_inst[result.instrument] = threshold
    # Enrich Study 82-style rows with Study 83 warning detection for final replication.
    for row in rt_replication:
        threshold = failure_rep_by_inst.get(row.instrument)
        if threshold:
            row.failure_detection = threshold.detection_at_failure

    out_root = Path(args.out_root)
    for result in instrument_results:
        single_rep = instrument_replication_rows([result.realtime])
        if single_rep:
            threshold = recommended_threshold(result.timing.thresholds)
            single_rep[0].failure_detection = threshold.detection_at_failure
        write_report(result, out_root, replication_rows_final=single_rep)
    write_report(aggregate_result, out_root, aggregate=True, replication_rows_final=rt_replication)


if __name__ == "__main__":
    main()
