"""Labeled golden dataset for the Response Studio evaluation harness.

Each :class:`EvalCase` carries human-authored ground truth for one questionnaire
item. Labels reflect domain intent, not the current implementation's output, so
the harness surfaces genuine quality gaps rather than tautologically passing.

Notes on the labels:

* ``expected_category`` is the human-judged category. A few items (e.g. encrypted
  backups) are genuinely ambiguous; the label follows where the supporting
  knowledge lives in ``data/knowledge_base``.
* ``expected_route`` is ``needs_review`` when a human *should* see the item before
  it ships — either because it is high-risk (legal) or because the knowledge base
  has no authoritative answer (a "knowledge gap").
* ``expected_sources`` lists the knowledge-base category files that *should* be
  retrieved. An empty list marks a deliberate knowledge gap (no good source).
* ``answer_keywords`` are substrings that a correct, grounded answer should
  contain. Empty for knowledge-gap items.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expected_category: str
    expected_risk: str  # "low" | "high"
    expected_route: str  # "auto_approved" | "needs_review"
    expected_sources: list[str] = field(default_factory=list)
    answer_keywords: list[str] = field(default_factory=list)
    # True when there is intentionally no authoritative KB answer.
    knowledge_gap: bool = False


GOLDEN: list[EvalCase] = [
    EvalCase(
        id="q1",
        question="Does your platform encrypt customer data at rest and in transit?",
        expected_category="security",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["security"],
        answer_keywords=["AES-256", "TLS"],
    ),
    EvalCase(
        id="q2",
        question="Do you support single sign-on (SSO) via SAML 2.0 and OIDC?",
        expected_category="product",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["product"],
        answer_keywords=["SAML", "OIDC"],
    ),
    EvalCase(
        id="q3",
        question="Do you enforce multi-factor authentication (MFA) for administrative access?",
        expected_category="security",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["security"],
        answer_keywords=["MFA"],
    ),
    EvalCase(
        id="q4",
        question="Are you SOC 2 Type II certified, and can you share the report?",
        expected_category="compliance",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["compliance"],
        answer_keywords=["SOC 2"],
    ),
    EvalCase(
        id="q5",
        question="Do you hold ISO 27001 certification?",
        expected_category="compliance",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["compliance"],
        answer_keywords=["27001"],
    ),
    EvalCase(
        id="q6",
        question="How do you handle GDPR data subject requests and data residency in the EU?",
        expected_category="compliance",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["compliance"],
        answer_keywords=["30 days"],
    ),
    EvalCase(
        id="q7",
        question="Describe your role-based access control (RBAC) model.",
        expected_category="product",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["product"],
        answer_keywords=["RBAC"],
    ),
    EvalCase(
        id="q8",
        question="What is your platform's contractual uptime SLA?",
        expected_category="product",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["product"],
        answer_keywords=["99.9%"],
    ),
    EvalCase(
        id="q9",
        question="How frequently do you perform third-party penetration testing?",
        expected_category="security",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["security"],
        answer_keywords=["annually"],
    ),
    EvalCase(
        id="q10",
        question="Describe your security incident response process and notification timelines.",
        expected_category="security",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["security"],
        answer_keywords=["72 hours"],
    ),
    EvalCase(
        id="q11",
        question="Do you maintain encrypted, geographically redundant backups?",
        expected_category="product",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["product"],
        answer_keywords=["backups"],
    ),
    EvalCase(
        id="q12",
        question="What public APIs and integrations do you provide?",
        expected_category="product",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["product"],
        answer_keywords=["REST API"],
    ),
    EvalCase(
        id="q13",
        question="What are your pricing tiers and how is usage metered?",
        expected_category="pricing",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["pricing"],
        answer_keywords=["Enterprise"],
    ),
    EvalCase(
        id="q14",
        question="Do you offer volume discounts for enterprise agreements?",
        expected_category="pricing",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["pricing"],
        answer_keywords=["discount"],
    ),
    EvalCase(
        id="q15",
        question="What is your standard limitation of liability in the master agreement?",
        expected_category="legal",
        expected_risk="high",
        expected_route="needs_review",
        expected_sources=[],
        answer_keywords=[],
        knowledge_gap=True,
    ),
    EvalCase(
        id="q16",
        question="Do you sign a Data Processing Agreement (DPA) and maintain a subprocessor list?",
        expected_category="legal",
        expected_risk="high",
        expected_route="needs_review",
        expected_sources=["legal"],
        answer_keywords=["DPA", "subprocessor"],
    ),
    EvalCase(
        id="q17",
        question="What are the termination and data return / deletion provisions?",
        expected_category="legal",
        expected_risk="high",
        expected_route="needs_review",
        expected_sources=["legal"],
        answer_keywords=["deleted"],
    ),
    EvalCase(
        id="q18",
        question="Do you provide a quantum-resistant cryptography roadmap for post-2030 compliance?",
        expected_category="security",
        expected_risk="low",
        expected_route="needs_review",
        expected_sources=[],
        answer_keywords=[],
        knowledge_gap=True,
    ),
]
