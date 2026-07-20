"""Answer-quality (AI performance) evaluation.

Where ``harness.py`` scores the *pipeline* (classification / routing / retrieval),
this module scores the *generated answers* — i.e. how well the model actually
answers each question. It is deliberately decoupled from the model via a
pluggable ``answer_fn`` so the same metrics apply to:

* **demo** mode (default): deterministic KB-composed answers — offline baseline.
* **live** mode: real LLM answers when ``OPENAI_API_KEY`` is set.
* a **mock** callable: used by tests to prove the metrics discriminate good from
  hallucinated answers.

Metrics (all computed offline, deterministically):

* **fact_coverage** — recall of a case's required ``key_facts`` in the answer
  (answer correctness).
* **faithfulness** — fraction of the answer's content tokens that are supported by
  the retrieved KB context (anti-hallucination / grounding).
* **refusal behavior** — knowledge-gap questions should be *refused/deferred*, not
  answered; in-KB questions should not be refused.
* **hallucination_rate** — answers that are neither refusals nor sufficiently
  grounded.

An optional **LLM-as-judge** (``--judge``) adds a subjective 1–5 correctness score
when a live model is available; it is skipped offline so tests stay hermetic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

from app.knowledge_base import RetrievalHit, retrieve, tokenize
from app.llm import draft_answer, llm_mode

from .dataset import GOLDEN, EvalCase

# An answer function takes the question + retrieved hits and returns a dict with
# at least "draft" and "citations" (the same shape as app.llm.draft_answer).
AnswerFn = Callable[[str, list[RetrievalHit]], dict]

# Markers that indicate the model declined to answer / deferred to a human.
_REFUSAL_MARKERS = (
    "no supporting material",
    "subject-matter expert",
    "subject matter expert",
    "insufficient",
    "cannot answer",
    "do not have",
    "don't have",
    "not found",
    "unable to",
    "no information",
    "cannot provide",
)

THRESHOLDS: dict[str, float] = {
    "mean_fact_coverage": 0.85,
    "mean_faithfulness": 0.80,
    # NOTE: demo-mode baseline is 88.9% — the two knowledge-gap items (liability,
    # quantum crypto) are answered off-topic rather than refused, a known weakness
    # of KB-composition. The pipeline harness still routes both to human review.
    # Threshold sits just below baseline so it catches regressions (e.g. a covered
    # item wrongly refusing) without masking the finding.
    "refusal_accuracy": 0.85,
    "max_hallucination_rate": 0.10,  # upper bound (lower is better)
}

# An answer is considered grounded when at least this fraction of its content
# tokens appear in the retrieved context.
FAITHFULNESS_MIN = 0.6


@dataclass
class AnswerCaseResult:
    id: str
    question: str
    answer: str
    fact_coverage: float | None  # None when the case has no key_facts (a gap)
    faithfulness: float | None  # None for refusals / no retrieved context
    is_refusal: bool
    refusal_expected: bool
    refusal_ok: bool
    hallucinated: bool
    judge_score: float | None = None  # 1..5 from optional LLM judge


@dataclass
class AnswerEvalReport:
    llm_mode: str
    n: int
    metrics: dict[str, float]
    thresholds: dict[str, float]
    passed: bool
    failures: list[str]
    cases: list[AnswerCaseResult] = field(default_factory=list)

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


def is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(marker in low for marker in _REFUSAL_MARKERS)


def _fact_coverage(answer: str, facts: list[str]) -> float | None:
    if not facts:
        return None
    low = answer.lower()
    hits = sum(1 for fact in facts if fact.lower() in low)
    return round(hits / len(facts), 4)


def _faithfulness(answer: str, hits: list[RetrievalHit]) -> float | None:
    """Fraction of the answer's content tokens supported by retrieved context."""
    ans_tokens = tokenize(answer)
    if not ans_tokens or not hits:
        return None
    ctx_tokens: set[str] = set()
    for h in hits:
        ctx_tokens |= set(tokenize(f"{h.chunk.title} {h.chunk.text}"))
    supported = sum(1 for t in ans_tokens if t in ctx_tokens)
    return round(supported / len(ans_tokens), 4)


def evaluate_answer(
    case: EvalCase,
    answer_fn: AnswerFn | None = None,
    judge: Callable[[EvalCase, str], float] | None = None,
) -> AnswerCaseResult:
    answer_fn = answer_fn or draft_answer
    hits = retrieve(case.question, top_k=3)
    drafted = answer_fn(case.question, hits)
    answer = drafted.get("draft", "")

    refusal = is_refusal(answer)
    refusal_expected = case.knowledge_gap
    refusal_ok = refusal == refusal_expected

    coverage = _fact_coverage(answer, case.facts())
    faithfulness = _faithfulness(answer, hits)

    # A hallucination = a non-refusal, ungrounded answer. Refusals are safe;
    # gap items are expected to refuse and are not counted as hallucinations.
    hallucinated = (
        not refusal
        and not refusal_expected
        and faithfulness is not None
        and faithfulness < FAITHFULNESS_MIN
    )

    judge_score = judge(case, answer) if judge is not None else None

    return AnswerCaseResult(
        id=case.id,
        question=case.question,
        answer=answer,
        fact_coverage=coverage,
        faithfulness=faithfulness,
        is_refusal=refusal,
        refusal_expected=refusal_expected,
        refusal_ok=refusal_ok,
        hallucinated=hallucinated,
        judge_score=judge_score,
    )


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 1.0


def run_answer_eval(
    answer_fn: AnswerFn | None = None,
    dataset: list[EvalCase] | None = None,
    judge: Callable[[EvalCase, str], float] | None = None,
) -> AnswerEvalReport:
    cases = [evaluate_answer(c, answer_fn=answer_fn, judge=judge) for c in (dataset or GOLDEN)]

    coverage_vals = [c.fact_coverage for c in cases if c.fact_coverage is not None]
    faith_vals = [c.faithfulness for c in cases if c.faithfulness is not None]
    refusal_vals = [c.refusal_ok for c in cases]
    hallucination_rate = round(sum(1 for c in cases if c.hallucinated) / len(cases), 4)

    metrics = {
        "mean_fact_coverage": _mean(coverage_vals),
        "mean_faithfulness": _mean(faith_vals),
        "refusal_accuracy": round(sum(1 for v in refusal_vals if v) / len(refusal_vals), 4),
        "hallucination_rate": hallucination_rate,
    }
    judge_scores = [c.judge_score for c in cases if c.judge_score is not None]
    if judge_scores:
        metrics["mean_judge_score"] = _mean(judge_scores)

    failures: list[str] = []
    for name, threshold in THRESHOLDS.items():
        if name == "max_hallucination_rate":
            if metrics["hallucination_rate"] > threshold:
                failures.append(
                    f"hallucination_rate: {metrics['hallucination_rate']:.2%} > max {threshold:.2%}"
                )
        elif metrics.get(name, 0.0) < threshold:
            failures.append(f"{name}: {metrics[name]:.2%} < threshold {threshold:.2%}")

    return AnswerEvalReport(
        llm_mode=llm_mode(),
        n=len(cases),
        metrics=metrics,
        thresholds=THRESHOLDS,
        passed=not failures,
        failures=failures,
        cases=cases,
    )


def llm_judge(case: EvalCase, answer: str) -> float:
    """Optional LLM-as-judge: score answer correctness 1..5 vs the reference.

    Requires ``OPENAI_API_KEY`` (live mode). Raises if unavailable so callers can
    decide whether to skip. Kept out of the default path so offline runs/tests
    stay hermetic.
    """
    from langchain_openai import ChatOpenAI

    prompt = (
        "You are grading an RFP answer for factual correctness against a reference. "
        "Respond with ONLY an integer 1-5 (5 = fully correct and complete, "
        "1 = wrong or irrelevant).\n\n"
        f"Question: {case.question}\n"
        f"Reference answer: {case.reference_answer or '(none)'}\n"
        f"Candidate answer: {answer}\n\nScore:"
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    raw = model.invoke(prompt).content.strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    return float(digits[0]) if digits else 0.0


def format_answer_report(report: AnswerEvalReport) -> str:
    lines: list[str] = []
    lines.append("=" * 74)
    lines.append(
        f"Response Studio answer-quality eval — {report.n} cases (llm_mode={report.llm_mode})"
    )
    lines.append("=" * 74)

    header = f"{'id':<4} {'cover':>6} {'faith':>6} {'refuse':>7} {'halluc':>7}  answer"
    lines.append(header)
    lines.append("-" * len(header))
    for c in report.cases:
        cover = "n/a" if c.fact_coverage is None else f"{c.fact_coverage:.0%}"
        faith = "n/a" if c.faithfulness is None else f"{c.faithfulness:.0%}"
        refuse = ("ok" if c.refusal_ok else "MISS") + ("*" if c.refusal_expected else "")
        halluc = "YES" if c.hallucinated else "-"
        snippet = c.answer.replace("\n", " ")[:40]
        lines.append(
            f"{c.id:<4} {cover:>6} {faith:>6} {refuse:>7} {halluc:>7}  {snippet}"
        )
    lines.append("-" * len(header))
    lines.append("  (* = knowledge-gap item; a refusal is the correct behavior)")
    lines.append("")

    lines.append("Metrics:")
    for name, value in report.metrics.items():
        threshold = report.thresholds.get(name)
        if name == "hallucination_rate":
            cap = report.thresholds["max_hallucination_rate"]
            status = "PASS" if value <= cap else "FAIL"
            lines.append(f"  {name:<28} {value:>7.2%}   (<= {cap:.0%})  [{status}]")
        elif threshold is not None:
            status = "PASS" if value >= threshold else "FAIL"
            lines.append(f"  {name:<28} {value:>7.2%}   (>= {threshold:.0%})  [{status}]")
        else:
            lines.append(f"  {name:<28} {value:>7.3f}")

    lines.append("")
    if report.passed:
        lines.append("RESULT: PASS — answer quality meets thresholds.")
    else:
        lines.append("RESULT: FAIL")
        for f in report.failures:
            lines.append(f"  - {f}")
    return "\n".join(lines)
