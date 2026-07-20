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
    # --- answer-quality (AI performance) labels --------------------------
    # ``key_facts`` are the facts a correct answer MUST include (used to score
    # answer correctness / recall). ``reference_answer`` is a short gold answer
    # used for documentation and by the optional LLM-as-judge. Both are empty for
    # knowledge-gap items, where the model is expected to refuse instead.
    key_facts: list[str] = field(default_factory=list)
    reference_answer: str = ""

    def facts(self) -> list[str]:
        """Facts to score correctness against (falls back to answer_keywords)."""
        return self.key_facts or self.answer_keywords


GOLDEN: list[EvalCase] = [
    EvalCase(
        id="q1",
        question="Does your platform encrypt customer data at rest and in transit?",
        expected_category="security",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["security"],
        answer_keywords=["AES-256", "TLS"],
        key_facts=["AES-256", "TLS", "rotation"],
        reference_answer="Data is encrypted at rest with AES-256 and in transit with TLS 1.2+, using a FIPS 140-2 KMS with 90-day key rotation.",
    ),
    EvalCase(
        id="q2",
        question="Do you support single sign-on (SSO) via SAML 2.0 and OIDC?",
        expected_category="product",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["product"],
        answer_keywords=["SAML", "OIDC"],
        key_facts=["SAML", "OIDC", "SCIM"],
        reference_answer="SSO is supported via SAML 2.0 and OIDC, with SCIM 2.0 for automated provisioning.",
    ),
    EvalCase(
        id="q3",
        question="Do you enforce multi-factor authentication (MFA) for administrative access?",
        expected_category="security",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["security"],
        answer_keywords=["MFA"],
        key_facts=["MFA", "TOTP", "WebAuthn"],
        reference_answer="MFA is enforced for all administrative accounts, supporting TOTP apps and WebAuthn/FIDO2 keys.",
    ),
    EvalCase(
        id="q4",
        question="Are you SOC 2 Type II certified, and can you share the report?",
        expected_category="compliance",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["compliance"],
        answer_keywords=["SOC 2"],
        key_facts=["SOC 2 Type II", "annual"],
        reference_answer="We maintain an annual SOC 2 Type II attestation available under NDA via our trust portal.",
    ),
    EvalCase(
        id="q5",
        question="Do you hold ISO 27001 certification?",
        expected_category="compliance",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["compliance"],
        answer_keywords=["27001"],
        key_facts=["27001", "certified"],
        reference_answer="We are ISO/IEC 27001:2022 certified; the certificate and Statement of Applicability are available on request.",
    ),
    EvalCase(
        id="q6",
        question="How do you handle GDPR data subject requests and data residency in the EU?",
        expected_category="compliance",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["compliance"],
        answer_keywords=["30 days"],
        key_facts=["data processor", "EU", "30 days"],
        reference_answer="As a GDPR data processor we offer EU data residency and fulfill data subject requests within 30 days.",
    ),
    EvalCase(
        id="q7",
        question="Describe your role-based access control (RBAC) model.",
        expected_category="product",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["product"],
        answer_keywords=["RBAC"],
        key_facts=["roles", "permissions"],
        reference_answer="Granular RBAC ships with predefined and custom roles and per-resource permissions scoped by workspace, project, and environment.",
    ),
    EvalCase(
        id="q8",
        question="What is your platform's contractual uptime SLA?",
        expected_category="product",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["product"],
        answer_keywords=["99.9%"],
        key_facts=["99.9%", "uptime"],
        reference_answer="Our standard enterprise SLA guarantees 99.9% monthly uptime with service credits for missed targets.",
    ),
    EvalCase(
        id="q9",
        question="How frequently do you perform third-party penetration testing?",
        expected_category="security",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["security"],
        answer_keywords=["annually"],
        key_facts=["annually", "third party"],
        reference_answer="An independent third party performs full-scope penetration tests at least annually and after major changes.",
    ),
    EvalCase(
        id="q10",
        question="Describe your security incident response process and notification timelines.",
        expected_category="security",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["security"],
        answer_keywords=["72 hours"],
        key_facts=["72 hours", "NIST"],
        reference_answer="We run a 24/7 incident response process aligned to NIST 800-61 and notify affected customers within 72 hours of confirmation.",
    ),
    EvalCase(
        id="q11",
        question="Do you maintain encrypted, geographically redundant backups?",
        expected_category="product",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["product"],
        answer_keywords=["backups"],
        key_facts=["encrypted", "redundant", "backups"],
        reference_answer="Encrypted backups are stored in geographically redundant regions and validated via quarterly restore drills.",
    ),
    EvalCase(
        id="q12",
        question="What public APIs and integrations do you provide?",
        expected_category="product",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["product"],
        answer_keywords=["REST API"],
        key_facts=["REST API", "webhooks"],
        reference_answer="We provide a documented REST API, webhooks, and prebuilt integrations (Slack, Salesforce, Jira, Google Workspace).",
    ),
    EvalCase(
        id="q13",
        question="What are your pricing tiers and how is usage metered?",
        expected_category="pricing",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["pricing"],
        answer_keywords=["Enterprise"],
        key_facts=["Team", "Business", "Enterprise"],
        reference_answer="Three tiers (Team, Business, Enterprise); Team/Business are per-seat, Enterprise is a custom annual agreement with usage metering.",
    ),
    EvalCase(
        id="q14",
        question="Do you offer volume discounts for enterprise agreements?",
        expected_category="pricing",
        expected_risk="low",
        expected_route="auto_approved",
        expected_sources=["pricing"],
        answer_keywords=["discount"],
        key_facts=["volume", "discount"],
        reference_answer="Enterprise agreements include tiered volume discounts based on committed annual seats or usage, with multi-year discounts.",
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
        key_facts=["DPA", "subprocessor", "Standard Contractual Clauses"],
        reference_answer="We execute a GDPR-compliant DPA with EU Standard Contractual Clauses and publish a subprocessor list with a 30-day objection window.",
    ),
    EvalCase(
        id="q17",
        question="What are the termination and data return / deletion provisions?",
        expected_category="legal",
        expected_risk="high",
        expected_route="needs_review",
        expected_sources=["legal"],
        answer_keywords=["deleted"],
        key_facts=["30 days", "deleted", "90 days"],
        reference_answer="On termination, data is exportable for 30 days and then securely deleted within 90 days, with certified deletion evidence on request.",
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
