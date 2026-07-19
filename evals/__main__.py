"""CLI entry point: ``python -m evals``.

Runs the evaluation harness over the golden dataset, prints a human-readable
report, and optionally writes the full result (including per-case detail) to a
JSON file.

Exit code is non-zero when any gated metric falls below its threshold, so the
harness can be used directly in CI.
"""
from __future__ import annotations

import argparse
import json
import sys

from .harness import format_report, run_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Response Studio evaluation harness.")
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Write the full evaluation result (with per-case detail) to this JSON file.",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Always exit 0 even if metrics fall below thresholds (report only).",
    )
    args = parser.parse_args(argv)

    report = run_evaluation()
    print(format_report(report))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"\nWrote JSON report to {args.json}")

    if args.no_gate:
        return 0
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
