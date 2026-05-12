#!/usr/bin/env python3
"""
Generate Architectural Genotype Validation and Uniqueness Reports.

Entry-point script for the report pipeline step.  Writes Markdown summary
and detail reports to ``results/reports/pareto/``.

Usage
-----
    python scripts/generate_validation_reports.py [options]

Options
-------
--results-dir DIR   Override results root directory (default: <project>/results)
--output-dir DIR    Override output directory (default: <project>/results/reports/pareto)
--datasets DS ...   Space-separated subset of datasets to process
--hardware HW ...   Space-separated subset of hardware targets to process
--proposed-key KEY  Key in ALGO_SPEC for the Proposed algorithm (default: "Proposed")
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── Ensure project root is on sys.path so src/ can be imported ────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.analysis.data_loader import DATASETS, HARDWARE          # noqa: E402
from src.analysis.report_generators import (
    generate_genotype_catalog,
    generate_uniqueness_reports,
)  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=_PROJECT_ROOT / "results",
        help="Root directory of experiment results.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "results" / "reports" / "pareto",
        help="Directory to write Markdown reports into.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DS",
        help=f"Datasets to process (default: {DATASETS}).",
    )
    p.add_argument(
        "--hardware",
        nargs="+",
        default=None,
        metavar="HW",
        help=f"Hardware targets to process (default: {HARDWARE}).",
    )
    p.add_argument(
        "--proposed-key",
        default="Proposed",
        help='ALGO_SPEC key that identifies the Proposed algorithm (default: "Proposed").',
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    print(f"[generate_validation_reports] Results dir : {args.results_dir}")
    print(f"[generate_validation_reports] Output dir  : {args.output_dir}")
    print(f"[generate_validation_reports] Datasets    : {args.datasets or DATASETS}")
    print(f"[generate_validation_reports] Hardware    : {args.hardware or HARDWARE}")

    generate_uniqueness_reports(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        datasets=args.datasets,
        hardware=args.hardware,
        proposed_key=args.proposed_key,
    )

    generate_genotype_catalog(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        datasets=args.datasets,
        hardware=args.hardware,
        proposed_key=args.proposed_key,
    )

    print("[generate_validation_reports] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
