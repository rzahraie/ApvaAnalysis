#!/usr/bin/env python3
"""Shared command-line and report-output helpers for APVA evidence studies."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def add_input_arguments(
    parser: argparse.ArgumentParser,
    default_input: Path,
) -> None:
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        help=f"Evidence CSV path (default: {default_input.as_posix()})",
    )
    parser.add_argument(
        "--input",
        dest="input_option",
        type=Path,
        help="Evidence CSV path. Kept for compatibility with earlier study versions.",
    )


def resolve_input(args: argparse.Namespace, default_input: Path) -> Path:
    if args.input_path is not None and args.input_option is not None:
        raise ValueError("Provide the input CSV either positionally or with --input, not both.")
    return args.input_path or args.input_option or default_input


def report_path(prefix: str, input_path: Path) -> Path:
    parent_name = input_path.parent.name.strip()
    if parent_name and parent_name.lower() != "evidence":
        instrument = parent_name
    else:
        instrument = input_path.stem.split("_", 1)[0]

    safe_instrument = re.sub(r"[^A-Za-z0-9._-]+", "_", instrument).strip("_")
    if not safe_instrument:
        safe_instrument = "UnknownInstrument"

    return Path("Evidence") / "Output" / safe_instrument / f"{prefix}_{safe_instrument}.txt"


def write_report(report: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(report, encoding="utf-8")
    except PermissionError:
        # PowerShell may already hold the report path open when stdout is
        # redirected to that same file. Printing below still writes the report.
        pass
    print(report, end="")
