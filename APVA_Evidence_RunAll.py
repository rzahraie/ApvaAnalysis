#!/usr/bin/env python3
"""Run all APVA evidence-layer studies for one CSV input."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_INPUT = Path("Evidence/NQ_5 Minute_apva_bar_evidence_v01.csv")

STUDY_SCRIPTS = (
    "APVA_Evidence_Ecology_01.py",
    "APVA_Evidence_Consequences_02.py",
    "APVA_Evidence_Sequences_03.py",
    "APVA_Evidence_Transitions_04.py",
    "APVA_Evidence_DirectionRelative_05.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all APVA evidence-layer reports for one evidence CSV."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Evidence CSV path (default: {DEFAULT_INPUT.as_posix()})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_path = args.input_path
    if not input_path.is_absolute():
        input_path = script_dir / input_path

    if not input_path.is_file():
        raise FileNotFoundError(f"Evidence CSV not found: {input_path}")

    for script_name in STUDY_SCRIPTS:
        script_path = script_dir / script_name
        print(f"\n=== Running {script_name} ===\n", flush=True)
        subprocess.run(
            [sys.executable, str(script_path), str(input_path)],
            cwd=script_dir,
            check=True,
        )

    print("\n=== All APVA evidence reports completed ===")


if __name__ == "__main__":
    main()
