"""CLI entry point: ``python -m evals``.

Runs the evaluation suites over the golden dataset, prints human-readable
report(s), and optionally writes full results (with per-case detail) to JSON.

Suites:
* ``pipeline`` — classification / risk / routing / retrieval / grounding.
* ``answers``  — AI answer quality: correctness, faithfulness, refusal behavior.
* ``all``      — both (default).

Exit code is non-zero when any gated metric fails, so the harness can gate CI.
Answer quality uses demo-mode answers by default; set ``OPENAI_API_KEY`` to
evaluate real LLM answers, and add ``--judge`` for an LLM-as-judge score.
"""
from __future__ import annotations

import argparse
import json
import sys

from .answer_quality import format_answer_report, llm_judge, run_answer_eval
from .harness import format_report, run_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Response Studio evaluation suites.")
    parser.add_argument(
        "--suite",
        choices=["pipeline", "answers", "all"],
        default="all",
        help="Which evaluation suite(s) to run (default: all).",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Write the combined result(s) to this JSON file.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Add an LLM-as-judge correctness score (answers suite; requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Always exit 0 even if metrics fall below thresholds (report only).",
    )
    args = parser.parse_args(argv)

    results: dict[str, dict] = {}
    passed = True

    if args.suite in ("pipeline", "all"):
        report = run_evaluation()
        print(format_report(report))
        print()
        results["pipeline"] = report.to_dict()
        passed = passed and report.passed

    if args.suite in ("answers", "all"):
        judge = llm_judge if args.judge else None
        answer_report = run_answer_eval(judge=judge)
        print(format_answer_report(answer_report))
        print()
        results["answers"] = answer_report.to_dict()
        passed = passed and answer_report.passed

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"Wrote JSON report to {args.json}")

    if args.no_gate:
        return 0
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
