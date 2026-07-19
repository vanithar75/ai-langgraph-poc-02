"""LangGraph StateGraph for Response Studio.

Flow:

    decompose -> draft_answers -> score_and_route -> human_review(interrupt #1)
      -> compile_package -> final_approval(interrupt #2) -> export

Two human-in-the-loop gates use ``interrupt()``; the run pauses and is resumed
with ``Command(resume=...)`` carrying the reviewer's decisions. State is durably
checkpointed to SQLite per ``thread_id`` so a run survives process restarts.
"""
from __future__ import annotations

import datetime as _dt
import re
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .knowledge_base import retrieve
from .llm import CONFIDENCE_THRESHOLD, draft_answer, llm_mode
from .state import Answer, Question, ResponseStudioState

# --- classification -------------------------------------------------------

_CATEGORY_KEYWORDS = {
    "security": [
        "encrypt", "mfa", "multi-factor", "penetration", "incident", "vulnerab",
        "endpoint", "malware", "siem", "breach",
    ],
    "compliance": [
        "soc 2", "soc2", "iso 27001", "iso27001", "gdpr", "hipaa", "pci",
        "audit", "certif", "data residency", "compliance",
    ],
    "product": [
        "sso", "saml", "oidc", "scim", "rbac", "role-based", "api", "integration",
        "uptime", "sla", "backup", "disaster", "provisioning", "webhook",
    ],
    "pricing": ["pricing", "price", "cost", "tier", "license", "discount", "metered", "usage"],
    "legal": [
        "liability", "indemn", "dpa", "data processing agreement", "subprocessor",
        "termination", "warranty", "contract", "legal",
    ],
}

_HIGH_RISK_KEYWORDS = [
    "liability", "indemn", "warranty", "termination", "penalty", "breach",
    "guarantee", "dpa", "data processing agreement", "subprocessor",
]


def classify(text: str) -> str:
    low = text.lower()
    best_cat, best_hits = "general", 0
    for cat, kws in _CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in low)
        if hits > best_hits:
            best_cat, best_hits = cat, hits
    return best_cat


def assess_risk(text: str, category: str) -> str:
    low = text.lower()
    if any(kw in low for kw in _HIGH_RISK_KEYWORDS):
        return "high"
    if category == "legal":
        return "high"
    return "low"


def decompose_questions(source_text: str) -> list[Question]:
    """Parse an RFP into individual questions.

    Recognizes numbered lists (``1.``/``1)``), ``Q:`` prefixes, and lines that
    end in a question mark.
    """
    questions: list[Question] = []
    seen: set[str] = set()
    for raw_line in source_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"^(?:\d+[.)]|[-*]|Q[:.)]?)\s+(.*)$", line, flags=re.IGNORECASE)
        if m:
            text = m.group(1).strip()
        elif line.endswith("?"):
            text = line
        else:
            continue
        if len(text) < 8 or text in seen:
            continue
        seen.add(text)
        qid = f"q{len(questions) + 1}"
        questions.append(Question(id=qid, text=text, category=classify(text)))
    return questions


# --- timeline helper ------------------------------------------------------

def _event(step: str, detail: str) -> dict:
    return {
        "step": step,
        "detail": detail,
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


def _push(state: ResponseStudioState, step: str, detail: str) -> list[dict]:
    return list(state.get("timeline", [])) + [_event(step, detail)]


# --- nodes ----------------------------------------------------------------

def node_decompose(state: ResponseStudioState) -> dict:
    questions = decompose_questions(state.get("source_text", ""))
    return {
        "questions": [q.model_dump() for q in questions],
        "status": "decomposed",
        "llm_mode": llm_mode(),
        "timeline": _push(state, "decompose", f"Extracted {len(questions)} questions"),
    }


def node_draft(state: ResponseStudioState) -> dict:
    answers: list[dict] = []
    for q in state.get("questions", []):
        question = Question(**q)
        hits = retrieve(question.text, top_k=3)
        drafted = draft_answer(question.text, hits)
        answer = Answer(
            question_id=question.id,
            question_text=question.text,
            category=question.category,
            draft=drafted["draft"],
            citations=drafted["citations"],
            confidence=drafted["confidence"],
            risk=assess_risk(question.text, question.category),
            status="pending",
        )
        answers.append(answer.model_dump())
    return {
        "answers": answers,
        "status": "drafted",
        "timeline": _push(state, "draft", f"Drafted {len(answers)} answers ({llm_mode()} mode)"),
    }


def node_score_and_route(state: ResponseStudioState) -> dict:
    answers = [Answer(**a) for a in state.get("answers", [])]
    review_queue: list[str] = []
    for a in answers:
        low_confidence = a.confidence < CONFIDENCE_THRESHOLD
        high_risk = a.risk == "high"
        if low_confidence or high_risk:
            a.status = "needs_review"
            review_queue.append(a.question_id)
        else:
            a.status = "auto_approved"
    stats = {
        "total": len(answers),
        "auto_approved": sum(1 for a in answers if a.status == "auto_approved"),
        "needs_review": len(review_queue),
    }
    return {
        "answers": [a.model_dump() for a in answers],
        "review_queue": review_queue,
        "stats": stats,
        "status": "scored",
        "timeline": _push(
            state,
            "score",
            f"{stats['auto_approved']} auto-approved, {stats['needs_review']} flagged for review",
        ),
    }


def node_human_review(state: ResponseStudioState) -> dict:
    review_queue = state.get("review_queue", [])
    answers_by_id = {a["question_id"]: a for a in state.get("answers", [])}

    if not review_queue:
        return {
            "status": "reviewed",
            "timeline": _push(state, "review", "No items required human review"),
        }

    flagged = [answers_by_id[qid] for qid in review_queue if qid in answers_by_id]

    # HITL GATE #1 — pause for the reviewer. Resumes with a list of decisions.
    decisions = interrupt(
        {
            "gate": "review_queue",
            "message": "Review low-confidence / high-risk answers before compiling.",
            "items": flagged,
        }
    )

    decisions = decisions or []
    applied = 0
    for d in decisions:
        qid = d.get("question_id")
        action = d.get("action")
        target = answers_by_id.get(qid)
        if not target:
            continue
        applied += 1
        target["reviewer_note"] = d.get("note", "")
        if action == "rewrite":
            target["draft"] = d.get("new_answer", target["draft"])
            target["status"] = "rewritten"
            target["confidence"] = max(target.get("confidence", 0.0), 0.95)
        elif action == "escalate":
            target["status"] = "escalated"
        else:  # approve
            target["status"] = "approved"

    return {
        "answers": list(answers_by_id.values()),
        "status": "reviewed",
        "timeline": _push(state, "review", f"Applied {applied} human decision(s)"),
    }


def _render_markdown(state: ResponseStudioState) -> str:
    title = state.get("rfp_title", "RFP Response")
    answers = [Answer(**a) for a in state.get("answers", [])]
    lines = [f"# {title} — Response Package", ""]
    escalated = [a for a in answers if a.status == "escalated"]
    if escalated:
        lines.append(f"> ⚠️ {len(escalated)} item(s) escalated to a subject-matter expert and excluded from the final answers below.")
        lines.append("")
    for a in answers:
        if a.status == "escalated":
            continue
        lines.append(f"## {a.question_text}")
        lines.append(f"*Category: {a.category} · confidence: {a.confidence:.0%} · status: {a.status}*")
        lines.append("")
        lines.append(a.draft)
        if a.citations:
            lines.append("")
            lines.append("Sources: " + "; ".join(a.citations))
        if a.reviewer_note:
            lines.append("")
            lines.append(f"Reviewer note: {a.reviewer_note}")
        lines.append("")
    if escalated:
        lines.append("## Escalated — pending SME input")
        for a in escalated:
            note = f" ({a.reviewer_note})" if a.reviewer_note else ""
            lines.append(f"- {a.question_text}{note}")
    return "\n".join(lines).strip() + "\n"


def node_compile(state: ResponseStudioState) -> dict:
    answers = [Answer(**a) for a in state.get("answers", [])]
    markdown = _render_markdown(state)
    export_json = {
        "rfp_title": state.get("rfp_title", "RFP Response"),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "llm_mode": state.get("llm_mode", llm_mode()),
        "answers": [a.model_dump() for a in answers if a.status != "escalated"],
        "escalated": [a.question_id for a in answers if a.status == "escalated"],
    }
    return {
        "compiled_markdown": markdown,
        "export_json": export_json,
        "status": "compiled",
        "timeline": _push(state, "compile", "Compiled response package"),
    }


def node_final_approval(state: ResponseStudioState) -> dict:
    # HITL GATE #2 — final sign-off before the package is marked ready to export.
    decision = interrupt(
        {
            "gate": "final_approval",
            "message": "Approve the compiled response package for export.",
            "preview": state.get("compiled_markdown", ""),
        }
    )
    decision = decision or {}
    action = decision.get("action", "approve")
    note = decision.get("note", "")
    if action == "reject":
        return {
            "status": "rejected",
            "timeline": _push(state, "final_approval", f"Package rejected: {note}"),
        }
    return {
        "status": "approved",
        "timeline": _push(state, "final_approval", "Package approved for export"),
    }


def node_export(state: ResponseStudioState) -> dict:
    if state.get("status") == "rejected":
        return {
            "status": "rejected",
            "timeline": _push(state, "export", "Export skipped (package rejected)"),
        }
    return {
        "status": "exported",
        "timeline": _push(state, "export", "Response package ready for download"),
    }


# --- graph assembly -------------------------------------------------------

def build_graph(checkpointer):
    g = StateGraph(ResponseStudioState)
    g.add_node("decompose", node_decompose)
    g.add_node("draft", node_draft)
    g.add_node("score_and_route", node_score_and_route)
    g.add_node("human_review", node_human_review)
    g.add_node("compile", node_compile)
    g.add_node("final_approval", node_final_approval)
    g.add_node("export", node_export)

    g.add_edge(START, "decompose")
    g.add_edge("decompose", "draft")
    g.add_edge("draft", "score_and_route")
    g.add_edge("score_and_route", "human_review")
    g.add_edge("human_review", "compile")
    g.add_edge("compile", "final_approval")
    g.add_edge("final_approval", "export")
    g.add_edge("export", END)

    return g.compile(checkpointer=checkpointer)


def make_sqlite_checkpointer(db_path: str | Path) -> SqliteSaver:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)
