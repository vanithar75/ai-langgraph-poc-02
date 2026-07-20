"""Tests for the answer-quality (AI performance) harness — offline, no API key.

Uses demo-mode answers plus injected mock models to prove the metrics actually
discriminate good answers from hallucinated / refusing ones.
"""
from __future__ import annotations

from evals.answer_quality import (
    THRESHOLDS,
    evaluate_answer,
    is_refusal,
    run_answer_eval,
)
from evals.dataset import GOLDEN


def _case(case_id: str):
    return next(c for c in GOLDEN if c.id == case_id)


def test_demo_answers_meet_quality_thresholds():
    report = run_answer_eval()
    assert report.n == len(GOLDEN)
    assert report.llm_mode == "demo"
    assert report.metrics["mean_fact_coverage"] >= THRESHOLDS["mean_fact_coverage"]
    assert report.metrics["mean_faithfulness"] >= THRESHOLDS["mean_faithfulness"]
    assert report.metrics["hallucination_rate"] <= THRESHOLDS["max_hallucination_rate"]
    assert report.passed, "answer-quality regressed:\n" + "\n".join(report.failures)


def test_refusal_detection():
    assert is_refusal("A subject-matter expert needs to provide this answer.")
    assert is_refusal("No supporting material was found in the knowledge base.")
    assert not is_refusal("We support SSO via SAML 2.0 and OIDC.")


def test_perfect_model_scores_high():
    """A model that returns the reference answer should score full coverage."""

    def perfect_fn(question, hits):
        case = next(c for c in GOLDEN if c.question == question)
        if case.knowledge_gap:
            draft = "No supporting material was found; a subject-matter expert must answer."
        else:
            draft = case.reference_answer or " ".join(case.facts())
        return {"draft": draft, "citations": [], "confidence": 0.9}

    report = run_answer_eval(answer_fn=perfect_fn)
    assert report.metrics["mean_fact_coverage"] >= 0.85
    assert report.metrics["hallucination_rate"] == 0.0


def test_hallucinating_model_is_caught():
    """An ungrounded, fact-free model must fail coverage and flag hallucinations."""

    def bad_fn(question, hits):
        return {
            "draft": "Absolutely, our platform handles everything flawlessly with zero caveats.",
            "citations": [],
            "confidence": 0.99,
        }

    report = run_answer_eval(answer_fn=bad_fn)
    assert not report.passed
    assert report.metrics["mean_fact_coverage"] < 0.85
    # Ungrounded answers on in-KB questions should be flagged as hallucinations.
    assert report.metrics["hallucination_rate"] > 0.10


def test_faithfulness_distinguishes_grounded_from_ungrounded():
    case = _case("q1")  # encryption question with strong KB coverage

    def grounded_fn(question, hits):
        return {"draft": case.reference_answer, "citations": [], "confidence": 0.9}

    def ungrounded_fn(question, hits):
        return {"draft": "Bananas are an excellent source of potassium.", "citations": [], "confidence": 0.9}

    grounded = evaluate_answer(case, answer_fn=grounded_fn)
    ungrounded = evaluate_answer(case, answer_fn=ungrounded_fn)
    assert grounded.faithfulness > ungrounded.faithfulness
    assert ungrounded.hallucinated
    assert not grounded.hallucinated


def test_gap_case_expects_refusal():
    gap = _case("q18")
    assert gap.knowledge_gap

    def refusing_fn(question, hits):
        return {"draft": "No supporting material was found; a subject-matter expert must answer.", "citations": [], "confidence": 0.1}

    result = evaluate_answer(gap, answer_fn=refusing_fn)
    assert result.is_refusal
    assert result.refusal_ok
    assert not result.hallucinated
