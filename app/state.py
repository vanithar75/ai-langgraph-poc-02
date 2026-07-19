"""Typed state models for the Response Studio graph.

Domain objects use Pydantic for validation; the LangGraph state itself is a
``TypedDict`` of plain JSON-serializable values so it checkpoints cleanly to
SQLite.
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

Category = Literal["security", "compliance", "product", "pricing", "legal", "general"]
RiskLevel = Literal["low", "medium", "high"]
AnswerStatus = Literal[
    "pending",
    "auto_approved",
    "needs_review",
    "approved",
    "rewritten",
    "escalated",
]
ReviewAction = Literal["approve", "rewrite", "escalate"]


class Question(BaseModel):
    """A single decomposed requirement / question from the RFP."""

    id: str
    text: str
    category: Category = "general"


class Answer(BaseModel):
    """A drafted answer for a question, with confidence and routing metadata."""

    question_id: str
    question_text: str
    category: Category = "general"
    draft: str = ""
    citations: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    risk: RiskLevel = "low"
    status: AnswerStatus = "pending"
    reviewer_note: str = ""

    @property
    def needs_human(self) -> bool:
        return self.status in ("needs_review", "escalated")


class ReviewDecision(BaseModel):
    """A human decision applied to one flagged answer during HITL review."""

    question_id: str
    action: ReviewAction
    new_answer: str | None = None
    note: str = ""


class FinalDecision(BaseModel):
    """Human sign-off (or rejection) on the compiled response package."""

    action: Literal["approve", "reject"]
    note: str = ""


class ResponseStudioState(TypedDict, total=False):
    """LangGraph state for a single questionnaire run."""

    rfp_title: str
    source_text: str
    questions: list[dict[str, Any]]
    answers: list[dict[str, Any]]
    review_queue: list[str]
    stats: dict[str, Any]
    compiled_markdown: str
    export_json: dict[str, Any]
    status: str
    timeline: list[dict[str, Any]]
    llm_mode: str
