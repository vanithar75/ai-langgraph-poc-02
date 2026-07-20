"""Regression tests for the offline evaluation harness (demo mode, no API key).

These gate the demo-mode pipeline's quality: if a change regresses
classification, routing, risk detection, retrieval, or answer grounding below the
configured thresholds, the suite fails.
"""
from __future__ import annotations

from dataclasses import replace

from evals.dataset import GOLDEN, EvalCase
from evals.harness import THRESHOLDS, evaluate_case, run_evaluation


def test_golden_dataset_meets_thresholds():
    report = run_evaluation()
    assert report.n == len(GOLDEN)
    assert report.llm_mode == "demo"
    assert report.passed, "eval regressed below thresholds:\n" + "\n".join(report.failures)
    for name, threshold in THRESHOLDS.items():
        assert report.metrics[name] >= threshold, f"{name}={report.metrics[name]} < {threshold}"


def test_safety_metrics_are_perfect():
    """High-risk items and knowledge gaps must always be routed to a human."""
    report = run_evaluation()
    assert report.metrics["high_risk_recall"] == 1.0
    assert report.metrics["gap_flag_rate"] == 1.0


def test_confidence_is_calibrated():
    """Auto-approved answers should be more confident than review-routed ones."""
    report = run_evaluation()
    assert (
        report.metrics["mean_confidence_auto_approved"]
        > report.metrics["mean_confidence_needs_review"]
    )


def test_knowledge_gap_cases_flagged_for_review():
    for case in GOLDEN:
        if case.knowledge_gap:
            result = evaluate_case(case)
            assert result.predicted_route == "needs_review", (
                f"knowledge-gap case {case.id} should be routed to review"
            )
            assert result.retrieval_ok is None


def test_harness_detects_regressions():
    """A degraded dataset must fail the gate — proves the harness isn't vacuous."""
    # An in-KB question deliberately mislabeled so predictions cannot match:
    # wrong category, wrong risk, wrong route, and a source that won't retrieve.
    bogus = [
        replace(
            case,
            expected_category="pricing",
            expected_risk="high",
            expected_route="needs_review",
            expected_sources=["pricing"],
            answer_keywords=["this text is not in any answer"],
            knowledge_gap=False,
        )
        for case in GOLDEN
        if case.expected_category == "security"
    ]
    report = run_evaluation(bogus)
    assert not report.passed
    assert report.failures
