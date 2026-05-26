#!/usr/bin/env python3
"""Run the frozen-rule APVA dataset and validation pipeline end to end.

This orchestrates existing build, readiness, fixed-validation, and cross-era
report scripts. It does not define candidates, alter thresholds, or search
for new validation rules.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


FROZEN_CANDIDATES = [
    "RRCCC",
    "CCRRR",
    "PriorSlope_DominantPressureValue_Q3",
]
EXACT_STATE_LOG = re.compile(r"^xApvaV01StateLog_(ES|NQ)(\d{6})\.csv$", re.IGNORECASE)
STATE_LOG_PREFIX = re.compile(r"^xApvaV01StateLog_(ES|NQ)(.*)\.csv$", re.IGNORECASE)
REFERENCE_DATASET = "tables/apva_forward_signed_return_dataset_v1.csv"
FIXED_OUTDIR = "outputs/fixed_candidate_extended_validation"
CROSS_ERA_OUTDIR = "outputs/cross_era_validation_report"
FULL_OUTDIR = "outputs/full_pipeline_run"


def now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def display_path(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_mixed_case_path(raw: str, workspace: Path) -> Path:
    path = Path(raw)
    path = path if path.is_absolute() else workspace / path
    if path.exists():
        return path.resolve()
    current = path.anchor and Path(path.anchor) or workspace
    parts = path.parts[1:] if path.anchor else path.relative_to(workspace).parts
    for part in parts:
        if not current.exists():
            break
        matches = [child for child in current.iterdir() if child.name.lower() == part.lower()]
        if len(matches) != 1:
            break
        current = matches[0]
    if current.exists():
        return current.resolve()
    raise FileNotFoundError(f"Raw root not found, including case-insensitive lookup: {raw}")


def format_command(command: list[str], workspace: Path) -> str:
    rendered = []
    for item in command:
        path = Path(item)
        if path.is_absolute():
            rendered.append(display_path(path, workspace))
        else:
            rendered.append(item)
    return " ".join(shlex.quote(item) for item in rendered)


def tail_text(value: str, line_count: int = 12) -> str:
    lines = str(value or "").strip().splitlines()
    return "\n".join(line.rstrip() for line in lines[-line_count:])


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def discover_raw_files(
    raw_root: Path,
    requested_regimes: set[str],
    workspace: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, list[Path]]]]:
    inventory_rows: list[dict[str, object]] = []
    paired_inputs: dict[str, dict[str, list[Path]]] = {}
    for path in sorted(raw_root.rglob("*.csv")):
        prefix = STATE_LOG_PREFIX.match(path.name)
        if not prefix:
            continue
        exact = EXACT_STATE_LOG.match(path.name)
        if not exact:
            inventory_rows.append({
                "RawFile": display_path(path, workspace),
                "Instrument": prefix.group(1).upper(),
                "Regime": "",
                "DiscoveryStatus": "Excluded",
                "Selected": False,
                "Reason": "state-log filename does not exactly match <instrument><six-digit-regime>.csv",
            })
            continue
        instrument, regime = exact.group(1).upper(), exact.group(2)
        selected = not requested_regimes or regime in requested_regimes
        inventory_rows.append({
            "RawFile": display_path(path, workspace),
            "Instrument": instrument,
            "Regime": regime,
            "DiscoveryStatus": "Discovered" if selected else "FilteredOut",
            "Selected": selected,
            "Reason": "matched exact dated raw-state-log name" if selected else "not requested by --regimes",
        })
        if selected:
            paired_inputs.setdefault(regime, {}).setdefault(instrument, []).append(path)
    for regime in sorted(requested_regimes - set(paired_inputs)):
        inventory_rows.append({
            "RawFile": "",
            "Instrument": "",
            "Regime": regime,
            "DiscoveryStatus": "MissingRequestedRegime",
            "Selected": False,
            "Reason": "no exactly named ES or NQ raw-state-log file found",
        })
        paired_inputs.setdefault(regime, {})
    return pd.DataFrame(inventory_rows), paired_inputs


def plan_regimes(
    paired_inputs: dict[str, dict[str, list[Path]]],
    allow_single_instrument: bool,
    workspace: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for regime, instruments in sorted(paired_inputs.items()):
        es_files = instruments.get("ES", [])
        nq_files = instruments.get("NQ", [])
        ambiguous = len(es_files) > 1 or len(nq_files) > 1
        if ambiguous:
            pair_status = "SKIP_AMBIGUOUS_FILES"
        elif es_files and nq_files:
            pair_status = "PAIRED"
        elif allow_single_instrument and (es_files or nq_files):
            pair_status = "SINGLE_INSTRUMENT_ALLOWED"
        else:
            pair_status = "SKIP_MISSING_INSTRUMENT"
        rows.append({
            "Regime": regime,
            "ESFile": display_path(es_files[0], workspace) if len(es_files) == 1 else ",".join(display_path(p, workspace) for p in es_files),
            "NQFile": display_path(nq_files[0], workspace) if len(nq_files) == 1 else ",".join(display_path(p, workspace) for p in nq_files),
            "PairStatus": pair_status,
            "DatasetPath": f"tables/apva_forward_signed_return_dataset_es_nq_{regime}_generated.csv",
            "Built": False,
            "ReadinessStatus": "NOT_RUN",
            "IncludedInFixedValidation": False,
            "RawSelectionNote": "",
            "Error": "",
        })
    return rows


def execute_step(
    step_log: list[dict[str, object]],
    step: str,
    regime: str,
    command: list[str],
    workspace: Path,
    dry_run: bool,
) -> None:
    started = now_text()
    rendered = format_command(command, workspace)
    if dry_run:
        print(f"DRY RUN: {rendered}")
        step_log.append({
            "Step": step,
            "Regime": regime,
            "Command": rendered,
            "Status": "DRY_RUN",
            "StartedAtUTC": started,
            "CompletedAtUTC": now_text(),
            "ReturnCode": "",
            "StdoutTail": "",
            "StderrTail": "",
        })
        return
    result = subprocess.run(command, cwd=workspace, text=True, capture_output=True)
    step_log.append({
        "Step": step,
        "Regime": regime,
        "Command": rendered,
        "Status": "PASS" if result.returncode == 0 else "FAIL",
        "StartedAtUTC": started,
        "CompletedAtUTC": now_text(),
        "ReturnCode": result.returncode,
        "StdoutTail": tail_text(result.stdout),
        "StderrTail": tail_text(result.stderr),
    })
    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline step failed: {step} {regime or 'pooled'}\n"
            f"Command: {rendered}\n{tail_text(result.stderr or result.stdout)}"
        )
    print(f"Completed {step}{' for ' + regime if regime else ''}.")


def metric_values(path: Path) -> dict[str, str]:
    scorecard = pd.read_csv(path)
    return dict(zip(scorecard["Metric"].astype(str), scorecard["Value"].astype(str)))


def reference_counts_from_fixed(fixed_scorecard: Path) -> dict[str, int]:
    metrics = metric_values(fixed_scorecard)
    counts = {}
    for candidate in FROZEN_CANDIDATES:
        key = f"Pooled_Reference_{candidate}_Count"
        if key not in metrics:
            raise RuntimeError(f"Extended scorecard is missing required metric: {key}")
        counts[candidate] = int(float(metrics[key]))
    return counts


def validation_status_from_fixed(fixed_summary: Path) -> pd.DataFrame:
    summary = pd.read_csv(fixed_summary)
    return summary.loc[
        summary["Scope"].eq("Pooled") & summary["Candidate"].isin(FROZEN_CANDIDATES),
        ["ValidationMode", "Candidate", "Count", "ES_Count", "NQ_Count", "Mean", "ProfitFactor", "ValidationStatus"],
    ].sort_values(["ValidationMode", "Candidate"])


def create_cross_era_consistency(
    fixed_counts: dict[str, int],
    rankings_path: Path,
    handoff_path: Path,
) -> tuple[pd.DataFrame, bool]:
    rankings = pd.read_csv(rankings_path).set_index("Candidate")
    handoff = handoff_path.read_text(encoding="utf-8") if handoff_path.exists() else ""
    rows: list[dict[str, object]] = []
    all_match = True
    for candidate in FROZEN_CANDIDATES:
        ranking_count = int(rankings.loc[candidate, "ReferenceTotalCount"]) if candidate in rankings.index else None
        match = re.search(
            rf"`{re.escape(candidate)}` Reference pooled count: `(\d+)`",
            handoff,
        )
        handoff_count = int(match.group(1)) if match else None
        fixed_count = fixed_counts[candidate]
        counts_match = fixed_count == ranking_count == handoff_count
        all_match = all_match and counts_match
        rows.append({
            "Candidate": candidate,
            "ExtendedValidationReferenceCount": fixed_count,
            "CrossEraRankingReferenceCount": ranking_count,
            "CrossEraHandoffReferenceCount": handoff_count,
            "CountsMatch": counts_match,
            "Status": "PASS" if counts_match else "FAIL",
        })
    return pd.DataFrame(rows), all_match


def create_full_consistency(
    cross_consistency: pd.DataFrame,
    pooled: pd.DataFrame,
    entries_path: Path,
) -> tuple[pd.DataFrame, bool]:
    rows: list[dict[str, object]] = []
    for _, row in cross_consistency.iterrows():
        rows.append({
            "Check": "CrossEraReferenceCount",
            "Scope": "Reference",
            "Candidate": row["Candidate"],
            "Expected": row["ExtendedValidationReferenceCount"],
            "Actual": row["CrossEraRankingReferenceCount"],
            "Status": row["Status"],
            "Detail": f"handoff_count={row['CrossEraHandoffReferenceCount']}",
        })
    for _, row in pooled.iterrows():
        expected = int(row["Count"])
        actual = int(row["ES_Count"]) + int(row["NQ_Count"])
        matches = expected == actual
        rows.append({
            "Check": "InstrumentCountsEqualEvaluableCount",
            "Scope": row["ValidationMode"],
            "Candidate": row["Candidate"],
            "Expected": expected,
            "Actual": actual,
            "Status": "PASS" if matches else "FAIL",
            "Detail": f"ES_Count={int(row['ES_Count'])};NQ_Count={int(row['NQ_Count'])}",
        })
    entries = pd.read_csv(entries_path)
    unevaluable = int(entries["NormalizedPolicyOutcome"].isna().sum())
    rows.append({
        "Check": "NoUnevaluablePolicyOutcomeEntries",
        "Scope": "AllModes",
        "Candidate": "AllCandidates",
        "Expected": 0,
        "Actual": unevaluable,
        "Status": "PASS" if unevaluable == 0 else "FAIL",
        "Detail": "NormalizedPolicyOutcome must be present before fixed-rule summarization",
    })
    report = pd.DataFrame(rows)
    return report, bool(report["Status"].eq("PASS").all())


def write_cross_era_handoff(
    path: Path,
    fixed_counts: dict[str, int],
    pooled: pd.DataFrame,
    rankings: pd.DataFrame,
    included_datasets: list[str],
) -> None:
    reference = pooled.loc[pooled["ValidationMode"].eq("Reference")].set_index("Candidate")
    lines = [
        "# APVA / SpyderTrader Cross-Era Validation Report Handoff",
        "",
        "## Scope",
        "",
        "This report is descriptive diagnostics over frozen fixed-rule candidates only.",
        "No candidate rules, thresholds, or topology definitions were changed.",
        "",
        "## Included Datasets",
        "",
    ]
    lines.extend(f"- `{dataset}`" for dataset in included_datasets)
    lines.extend([
        "",
        "## Reference Count Check",
        "",
    ])
    for candidate in FROZEN_CANDIDATES:
        lines.append(f"- `{candidate}` Reference pooled count: `{fixed_counts[candidate]}`")
    lines.extend([
        "",
        "These counts are written from `extended_scorecard.csv` and verified against",
        "`robustness_rankings.csv` by the full-pipeline consistency check.",
        "",
        "## Pooled Validation Status",
        "",
        "| Mode | Candidate | Count | ES | NQ | Mean | PF | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for _, row in pooled.sort_values(["ValidationMode", "Candidate"]).iterrows():
        lines.append(
            f"| {row['ValidationMode']} | `{row['Candidate']}` | {int(row['Count'])} | "
            f"{int(row['ES_Count'])} | {int(row['NQ_Count'])} | {float(row['Mean']):.6f} | "
            f"{float(row['ProfitFactor']):.6f} | {row['ValidationStatus']} |"
        )
    lines.extend([
        "",
        "## Cross-Era Robustness Ranking",
        "",
        "| Rank | Candidate | Score | Spacing Survival | Regime Survival | Concentration Penalty |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for _, row in rankings.sort_values("RobustnessRank").iterrows():
        lines.append(
            f"| {int(row['RobustnessRank'])} | `{row['Candidate']}` | "
            f"{float(row['RobustnessScore']):.6f} | {float(row['SpacingSurvivalScore']):.6f} | "
            f"{float(row['RegimeSurvivalScore']):.6f} | {float(row['ConcentrationPenalty']):.6f} |"
        )
    lines.extend([
        "",
        "## Interpretation Boundary",
        "",
        "Passing rows are provisional fixed-rule validation results. The principal",
        "remaining need is new paired ES/NQ observations outside the represented",
        "eras, with particular value in expanding sparse ES support for `CCRRR`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def readiness_rows(regime_rows: list[dict[str, object]], workspace: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in regime_rows:
        regime = str(row["Regime"])
        path = workspace / f"outputs/dataset_readiness_es_nq_{regime}/dataset_readiness_candidate_counts.csv"
        if not path.exists():
            continue
        counts = pd.read_csv(path)
        for _, candidate in counts.iterrows():
            records.append({
                "Regime": regime,
                "Candidate": candidate["Candidate"],
                "Count": int(candidate["Count"]),
                "ES_Count": int(candidate["ES_Count"]),
                "NQ_Count": int(candidate["NQ_Count"]),
            })
    return pd.DataFrame(records)


def write_final_handoff(
    path: Path,
    raw_root: Path,
    inventory: pd.DataFrame,
    regime_status: pd.DataFrame,
    candidate_counts: pd.DataFrame,
    pooled: pd.DataFrame,
    rankings: pd.DataFrame,
    consistency: pd.DataFrame,
    workspace: Path,
) -> None:
    selected = inventory.loc[inventory["Selected"].astype(bool)]
    skipped = regime_status.loc[~regime_status["PairStatus"].isin(["PAIRED", "SINGLE_INSTRUMENT_ALLOWED"])]
    lines = [
        "# APVA Full Pipeline Run Handoff",
        "",
        "## Scope",
        "",
        "This run rebuilds and validates frozen APVA candidates from raw NT8 state logs.",
        "No candidate definitions or validation thresholds were changed.",
        "",
        "## Raw Discovery",
        "",
        f"- Raw root: `{display_path(raw_root, workspace)}`",
        f"- Selected raw build input files after validation/fallback handling: `{len(selected)}`",
        f"- ES/NQ regimes processed: `{', '.join(regime_status.loc[regime_status['PairStatus'].eq('PAIRED'), 'Regime'].tolist()) or 'none'}`",
        "",
        "| Regime | ES File | NQ File | Pair Status | Built | Readiness | Fixed Validation Included |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in regime_status.iterrows():
        lines.append(
            f"| {row['Regime']} | `{row['ESFile']}` | `{row['NQFile']}` | "
            f"{row['PairStatus']} | {row['Built']} | {row['ReadinessStatus']} | "
            f"{row['IncludedInFixedValidation']} |"
        )
    lines.extend(["", "## Excluded Or Skipped Inputs", ""])
    excluded = inventory.loc[~inventory["Selected"].astype(bool)]
    if excluded.empty and skipped.empty:
        lines.append("- None.")
    else:
        for _, row in excluded.iterrows():
            lines.append(f"- `{row['RawFile'] or row['Regime']}`: {row['Reason']}.")
        for _, row in skipped.iterrows():
            lines.append(f"- Regime `{row['Regime']}`: {row['PairStatus']}.")
    lines.extend([
        "",
        "## Readiness Candidate Coverage By Built Regime",
        "",
        "These coverage counts describe candidate rows found in each built dataset.",
        "Fixed validation below excludes any row without an evaluable policy outcome.",
        "",
        "| Regime | Candidate | Count | ES | NQ |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    if candidate_counts.empty:
        lines.append("| - | - | - | - | - |")
    else:
        for _, row in candidate_counts.sort_values(["Regime", "Candidate"]).iterrows():
            lines.append(
                f"| {row['Regime']} | `{row['Candidate']}` | {int(row['Count'])} | "
                f"{int(row['ES_Count'])} | {int(row['NQ_Count'])} |"
            )
    lines.extend([
        "",
        "## Pooled Fixed Validation",
        "",
        "| Mode | Candidate | Count | ES | NQ | Mean | PF | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for _, row in pooled.sort_values(["ValidationMode", "Candidate"]).iterrows():
        lines.append(
            f"| {row['ValidationMode']} | `{row['Candidate']}` | {int(row['Count'])} | "
            f"{int(row['ES_Count'])} | {int(row['NQ_Count'])} | {float(row['Mean']):.6f} | "
            f"{float(row['ProfitFactor']):.6f} | {row['ValidationStatus']} |"
        )
    lines.extend([
        "",
        "## Cross-Era Robustness Ranking",
        "",
        "| Rank | Candidate | Score | Concentration Penalty |",
        "| ---: | --- | ---: | ---: |",
    ])
    for _, row in rankings.sort_values("RobustnessRank").iterrows():
        lines.append(
            f"| {int(row['RobustnessRank'])} | `{row['Candidate']}` | "
            f"{float(row['RobustnessScore']):.6f} | {float(row['ConcentrationPenalty']):.6f} |"
        )
    lines.extend(["", "## Consistency Checks", ""])
    for _, row in consistency.iterrows():
        lines.append(
            f"- `{row['Candidate']}` Reference pooled count: extended validation "
            f"`{row['ExtendedValidationReferenceCount']}`, cross-era ranking "
            f"`{row['CrossEraRankingReferenceCount']}`, handoff "
            f"`{row['CrossEraHandoffReferenceCount']}`: `{row['Status']}`."
        )
    lines.extend([
        "- Fixed-validation evaluated-entry checks: `PASS` (no missing policy "
        "outcomes and pooled ES/NQ counts reconcile to evaluated counts).",
        "",
        "## Next Data Collection Target",
        "",
        "Collect a new paired ES/NQ `xApvaV01StateLog` regime outside the periods",
        "already listed above, prioritizing enough ES observations to increase",
        "fixed-rule `CCRRR` support without changing its definition. Place the two",
        "raw files under `data/Validation/ES/` and `data/Validation/NQ/`, then run:",
        "",
        "```powershell",
        "python scripts/apva_run_full_pipeline.py --raw-root data/validation",
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> int:
    workspace = Path(__file__).resolve().parent.parent
    raw_root = resolve_mixed_case_path(args.raw_root, workspace)
    outdir = workspace / FULL_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    requested = set(args.regimes or [])
    inventory, paired_inputs = discover_raw_files(raw_root, requested, workspace)
    regime_rows = plan_regimes(paired_inputs, args.allow_single_instrument, workspace)
    runnable = [row for row in regime_rows if row["PairStatus"] in {"PAIRED", "SINGLE_INSTRUMENT_ALLOWED"}]
    if not runnable:
        raise RuntimeError("No unambiguous paired or explicitly allowed single-instrument regimes were discovered.")
    step_log: list[dict[str, object]] = []
    if args.dry_run:
        print(inventory.to_string(index=False))
    for row in runnable:
        regime = str(row["Regime"])
        raw_inputs = [value for value in [row["ESFile"], row["NQFile"]] if value]
        dataset = str(row["DatasetPath"])
        if not args.skip_build:
            command = [
                sys.executable,
                "-B",
                str(workspace / "scripts/apva_build_forward_return_dataset.py"),
                "--inputs",
                *raw_inputs,
                "--out",
                dataset,
                "--outdir",
                f"{FULL_OUTDIR}/build_es_nq_{regime}",
                "--horizons",
                *[str(value) for value in args.horizons],
                "--compare-to",
                REFERENCE_DATASET,
            ]
            try:
                execute_step(step_log, "BuildCanonicalDataset", regime, command, workspace, args.dry_run)
            except RuntimeError as exc:
                clean_replacements: dict[str, str] = {}
                if "embedded repeated header rows" in str(exc):
                    for raw_input in raw_inputs:
                        raw_path = workspace / raw_input
                        clean_path = raw_path.with_name(f"{raw_path.stem}_CLEAN{raw_path.suffix}")
                        if clean_path.exists():
                            clean_replacements[raw_input] = display_path(clean_path, workspace)
                if len(clean_replacements) != 1:
                    raise
                replaced_input, clean_input = next(iter(clean_replacements.items()))
                raw_inputs = [clean_input if item == replaced_input else item for item in raw_inputs]
                command = [
                    sys.executable,
                    "-B",
                    str(workspace / "scripts/apva_build_forward_return_dataset.py"),
                    "--inputs",
                    *raw_inputs,
                    "--out",
                    dataset,
                    "--outdir",
                    f"{FULL_OUTDIR}/build_es_nq_{regime}",
                    "--horizons",
                    *[str(value) for value in args.horizons],
                    "--compare-to",
                    REFERENCE_DATASET,
                ]
                row["RawSelectionNote"] = (
                    f"used {clean_input} after {replaced_input} failed embedded-header validation"
                )
                if row["ESFile"] == replaced_input:
                    row["ESFile"] = clean_input
                if row["NQFile"] == replaced_input:
                    row["NQFile"] = clean_input
                inventory.loc[inventory["RawFile"].eq(replaced_input), ["DiscoveryStatus", "Selected", "Reason"]] = [
                    "RejectedAtBuild",
                    False,
                    "exact dated file failed embedded repeated-header validation",
                ]
                inventory.loc[inventory["RawFile"].eq(clean_input), ["Regime", "DiscoveryStatus", "Selected", "Reason"]] = [
                    regime,
                    "SelectedCleanFallback",
                    True,
                    "unique clean companion used after exact dated file failed embedded-header validation",
                ]
                execute_step(step_log, "BuildCanonicalDatasetCleanFallback", regime, command, workspace, False)
            row["Built"] = not args.dry_run
        elif (workspace / dataset).exists():
            row["Built"] = "SKIPPED_EXISTING"
        else:
            raise RuntimeError(f"--skip-build requested but generated dataset is missing: {dataset}")
        if not args.skip_readiness:
            command = [
                sys.executable,
                "-B",
                str(workspace / "scripts/apva_dataset_readiness_check.py"),
                "--inputs",
                dataset,
                "--outdir",
                f"outputs/dataset_readiness_es_nq_{regime}",
            ]
            execute_step(step_log, "ReadinessCheck", regime, command, workspace, args.dry_run)
        if not args.dry_run:
            readiness_path = workspace / f"outputs/dataset_readiness_es_nq_{regime}/dataset_readiness_summary.csv"
            if readiness_path.exists():
                row["ReadinessStatus"] = str(pd.read_csv(readiness_path).iloc[0]["ReadinessStatus"])
            elif not args.skip_readiness:
                raise RuntimeError(f"Readiness output is missing after execution: {readiness_path}")
    if args.dry_run:
        if not args.skip_fixed_validation:
            execute_step(
                step_log,
                "FixedCandidateExtendedValidation",
                "",
                [sys.executable, "-B", str(workspace / "scripts/apva_fixed_candidate_extended_validation.py")],
                workspace,
                True,
            )
        if not args.skip_cross_era:
            execute_step(
                step_log,
                "CrossEraValidationReport",
                "",
                [sys.executable, "-B", str(workspace / "scripts/apva_cross_era_validation_report.py")],
                workspace,
                True,
            )
        execute_step(
            step_log,
            "VolumeParticipationDiagnostics",
            "",
            [sys.executable, "-B", str(workspace / "scripts/apva_volume_participation_diagnostics.py")],
            workspace,
            True,
        )
        print("Dry run complete; no pipeline artifacts were written.")
        return 0
    if not args.skip_fixed_validation:
        execute_step(
            step_log,
            "FixedCandidateExtendedValidation",
            "",
            [sys.executable, "-B", str(workspace / "scripts/apva_fixed_candidate_extended_validation.py")],
            workspace,
            False,
        )
    fixed_scorecard = workspace / FIXED_OUTDIR / "extended_scorecard.csv"
    fixed_summary = workspace / FIXED_OUTDIR / "extended_candidate_summary.csv"
    if not fixed_scorecard.exists() or not fixed_summary.exists():
        raise RuntimeError("Fixed-validation outputs required for pipeline verification are missing.")
    fixed_metrics = metric_values(fixed_scorecard)
    included_datasets = fixed_metrics.get("included_datasets", "").split(",")
    for row in runnable:
        expected = str(row["DatasetPath"])
        included = expected in included_datasets
        row["IncludedInFixedValidation"] = included
        if not included:
            row["Error"] = "generated dataset missing from fixed-validation included datasets"
            raise RuntimeError(f"Fixed validation did not include generated regime dataset: {expected}")
    if not args.skip_cross_era:
        execute_step(
            step_log,
            "CrossEraValidationReport",
            "",
            [sys.executable, "-B", str(workspace / "scripts/apva_cross_era_validation_report.py")],
            workspace,
            False,
        )
    rankings_path = workspace / CROSS_ERA_OUTDIR / "robustness_rankings.csv"
    cross_handoff_path = workspace / CROSS_ERA_OUTDIR / "chatgpt_master_analysis_handoff.md"
    if not rankings_path.exists():
        raise RuntimeError("Cross-era robustness rankings are missing after pipeline execution.")
    fixed_counts = reference_counts_from_fixed(fixed_scorecard)
    pooled = validation_status_from_fixed(fixed_summary)
    rankings = pd.read_csv(rankings_path)
    write_cross_era_handoff(cross_handoff_path, fixed_counts, pooled, rankings, included_datasets)
    consistency, matched = create_cross_era_consistency(fixed_counts, rankings_path, cross_handoff_path)
    consistency_path = workspace / CROSS_ERA_OUTDIR / "cross_era_consistency_check.csv"
    consistency.to_csv(consistency_path, index=False)
    if not matched:
        raise RuntimeError("Cross-era report stale or inconsistent with extended validation.")
    full_consistency, fully_consistent = create_full_consistency(
        consistency,
        pooled,
        workspace / FIXED_OUTDIR / "extended_entries.csv",
    )
    if not fully_consistent:
        raise RuntimeError("Fixed-validation outputs are internally inconsistent.")
    candidate_counts = readiness_rows(runnable, workspace)
    write_csv(inventory.to_dict("records"), outdir / "full_pipeline_inventory.csv")
    write_csv(regime_rows, outdir / "full_pipeline_regime_status.csv")
    full_consistency.to_csv(outdir / "full_pipeline_consistency_check.csv", index=False)
    write_final_handoff(
        outdir / "chatgpt_master_analysis_handoff.md",
        raw_root,
        inventory,
        pd.DataFrame(regime_rows),
        candidate_counts,
        pooled,
        rankings,
        consistency,
        workspace,
    )
    execute_step(
        step_log,
        "VolumeParticipationDiagnostics",
        "",
        [sys.executable, "-B", str(workspace / "scripts/apva_volume_participation_diagnostics.py")],
        workspace,
        False,
    )
    write_csv(step_log, outdir / "full_pipeline_step_log.csv")
    print("APVA full pipeline completed successfully.")
    print(pd.DataFrame(regime_rows)[["Regime", "PairStatus", "Built", "ReadinessStatus", "IncludedInFixedValidation"]].to_string(index=False))
    print()
    print(rankings[["RobustnessRank", "Candidate", "RobustnessScore"]].sort_values("RobustnessRank").to_string(index=False))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/validation")
    parser.add_argument("--regimes", nargs="*", default=[])
    parser.add_argument("--horizons", nargs="+", type=int, default=[5])
    parser.add_argument("--allow-single-instrument", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-readiness", action="store_true")
    parser.add_argument("--skip-fixed-validation", action="store_true")
    parser.add_argument("--skip-cross-era", action="store_true")
    parser.add_argument("--force", action="store_true", help="Reserved run marker; normal runs always refresh requested artifacts.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if any(horizon <= 0 for horizon in args.horizons):
        parser.error("--horizons values must be positive integers")
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
