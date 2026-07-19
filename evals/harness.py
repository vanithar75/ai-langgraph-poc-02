"""Evaluation harness: run the demo-mode pipeline over the golden dataset and
compute quality metrics.

The harness exercises the *real* pipeline functions used by the LangGraph nodes
(``classify``, ``assess_risk``, ``retrieve``, ``draft_answer``) plus the exact
routing rule from ``node_score_and_route`` so results reflect production behavior
without needing the full graph or checkpointer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from app.graph import assess_risk, classify
from app.knowledge_base import retrieve
from app.llm import CONFIDENCE_THRESHOLD, draft_answer, llm_mode

from .dataset import GOLDEN, EvalCase

# Regression thresholds. Set just below the measured demo-mode baseline so the
# gate catches real regressions while staying green on the current pipeline.
THRESHOLDS: dict[str, float] = {
    "category_accuracy": 0.80,
    "risk_accuracy": 0.90,
    "high_risk_recall": 1.00,
    "routing_accuracy": 0.85,
    "gap_flag_rate": 1.00,
    "retrieval_hit_rate": 0.90,
    "answer_grounding_rate": 0.85,
}


@dataclass
class CaseResult:
    id: str
    question: str
    expected_category: str
    predicted_category: str
    category_ok: bool
    expected_risk: str
    predicted_risk: str
    risk_ok: bool
    expected_route: str
    predicted_route: str
    route_ok: bool
    confidence: float
    retrieved_sources: list[str]
    retrieval_ok: bool | None  # None when the case is a knowledge gap (N/A)
    grounded: bool | None  # None when there are no expected keywords
    knowledge_gap: bool


@dataclass
class EvalReport:
    llm_mode: str
    n: int
    metrics: dict[str, float]
    thresholds: dict[str, float]
    passed: bool
    failures: list[str]
    cases: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "llm_mode": self.llm_mode,
            "n": self.n,
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "passed": self.passed,
            "failures": self.failures,
            "cases": [asdict(c) for c in self.cases],
        }


def _route(confidence: float, risk: str) -> str:
    """Mirror of ``node_score_and_route``'s routing decision."""
    low_confidence = confidence < CONFIDENCE_THRESHOLD
    high_risk = risk == "high"
    return "needs_review" if (low_confidence or high_risk) else "auto_approved"


def evaluate_case(case: EvalCase) -> CaseResult:
    predicted_category = classify(case.question)
    hits = retrieve(case.question, top_k=3)
    drafted = draft_answer(case.question, hits)
    confidence = float(drafted["confidence"])
    predicted_risk = assess_risk(case.question, predicted_category)
    predicted_route = _route(confidence, predicted_risk)

    retrieved_sources = [h.chunk.category for h in hits]

    if case.expected_sources:
        retrieval_ok: bool | None = any(
            src in retrieved_sources for src in case.expected_sources
        )
    else:
        retrieval_ok = None  # knowledge gap: nothing authoritative to retrieve

    if case.answer_keywords:
        draft_low = drafted["draft"].lower()
        grounded: bool | None = any(kw.lower() in draft_low for kw in case.answer_keywords)
    else:
        grounded = None

    return CaseResult(
        id=case.id,
        question=case.question,
        expected_category=case.expected_category,
        predicted_category=predicted_category,
        category_ok=predicted_category == case.expected_category,
        expected_risk=case.expected_risk,
        predicted_risk=predicted_risk,
        risk_ok=predicted_risk == case.expected_risk,
        expected_route=case.expected_route,
        predicted_route=predicted_route,
        route_ok=predicted_route == case.expected_route,
        confidence=confidence,
        retrieved_sources=retrieved_sources,
        retrieval_ok=retrieval_ok,
        grounded=grounded,
        knowledge_gap=case.knowledge_gap,
    )


def _rate(values: list[bool]) -> float:
    return round(sum(1 for v in values if v) / len(values), 4) if values else 1.0


def run_evaluation(dataset: list[EvalCase] | None = None) -> EvalReport:
    cases = [evaluate_case(c) for c in (dataset or GOLDEN)]

    category_vals = [c.category_ok for c in cases]
    risk_vals = [c.risk_ok for c in cases]
    route_vals = [c.route_ok for c in cases]
    high_risk_vals = [c.predicted_risk == "high" for c in cases if c.expected_risk == "high"]
    gap_vals = [c.predicted_route == "needs_review" for c in cases if c.knowledge_gap]
    retrieval_vals = [bool(c.retrieval_ok) for c in cases if c.retrieval_ok is not None]
    grounding_vals = [bool(c.grounded) for c in cases if c.grounded is not None]

    metrics = {
        "category_accuracy": _rate(category_vals),
        "risk_accuracy": _rate(risk_vals),
        "high_risk_recall": _rate(high_risk_vals),
        "routing_accuracy": _rate(route_vals),
        "gap_flag_rate": _rate(gap_vals),
        "retrieval_hit_rate": _rate(retrieval_vals),
        "answer_grounding_rate": _rate(grounding_vals),
    }

    # Confidence calibration: auto-approved items should be more confident than
    # those routed to review.
    auto_conf = [c.confidence for c in cases if c.predicted_route == "auto_approved"]
    review_conf = [c.confidence for c in cases if c.predicted_route == "needs_review"]
    metrics["mean_confidence_auto_approved"] = (
        round(sum(auto_conf) / len(auto_conf), 4) if auto_conf else 0.0
    )
    metrics["mean_confidence_needs_review"] = (
        round(sum(review_conf) / len(review_conf), 4) if review_conf else 0.0
    )

    failures = [
        f"{name}: {metrics[name]:.2%} < threshold {threshold:.2%}"
        for name, threshold in THRESHOLDS.items()
        if metrics.get(name, 0.0) < threshold
    ]

    return EvalReport(
        llm_mode=llm_mode(),
        n=len(cases),
        metrics=metrics,
        thresholds=THRESHOLDS,
        passed=not failures,
        failures=failures,
        cases=cases,
    )


def format_report(report: EvalReport) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"Response Studio evaluation — {report.n} cases (llm_mode={report.llm_mode})")
    lines.append("=" * 72)

    header = f"{'id':<4} {'cat':<10} {'risk':<6} {'route':<14} {'conf':>5}  {'retr':<5} {'grnd':<5}"
    lines.append(header)
    lines.append("-" * len(header))
    for c in report.cases:
        def mark(ok: bool | None) -> str:
            return "n/a" if ok is None else ("ok" if ok else "MISS")

        cat = ("ok " if c.category_ok else "MISS ") + c.predicted_category
        risk = ("ok" if c.risk_ok else "X") + "/" + c.predicted_risk[0]
        route = ("ok " if c.route_ok else "MISS ") + c.predicted_route
        lines.append(
            f"{c.id:<4} {cat:<10.10} {risk:<6} {route:<14.14} {c.confidence:>5.2f}  "
            f"{mark(c.retrieval_ok):<5} {mark(c.grounded):<5}"
        )

    lines.append("-" * len(header))
    lines.append("")
    lines.append("Metrics:")
    for name, value in report.metrics.items():
        threshold = report.thresholds.get(name)
        if threshold is not None:
            status = "PASS" if value >= threshold else "FAIL"
            lines.append(f"  {name:<32} {value:>7.2%}   (>= {threshold:.0%})  [{status}]")
        else:
            lines.append(f"  {name:<32} {value:>7.3f}")

    lines.append("")
    if report.passed:
        lines.append("RESULT: PASS — all gated metrics meet thresholds.")
    else:
        lines.append("RESULT: FAIL")
        for f in report.failures:
            lines.append(f"  - {f}")
    return "\n".join(lines)
