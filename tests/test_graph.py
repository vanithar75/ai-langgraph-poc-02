"""Unit tests for the Response Studio graph interrupt/resume path (demo mode, no LLM)."""
from __future__ import annotations

from pathlib import Path

from langgraph.types import Command

from app.graph import build_graph, classify, decompose_questions, make_sqlite_checkpointer

SAMPLE = (Path(__file__).resolve().parent.parent / "data" / "sample_rfp.md").read_text()


def _pending(snapshot):
    for task in snapshot.tasks:
        for itr in task.interrupts:
            return itr.value
    return None


def _make(tmp_path):
    cp = make_sqlite_checkpointer(tmp_path / "cp.sqlite")
    return build_graph(cp)


def test_decompose_and_classify():
    questions = decompose_questions(SAMPLE)
    # The sample questionnaire has 18 numbered questions.
    assert len(questions) == 18
    joined = {q.text: q.category for q in questions}
    assert classify("Do you encrypt data at rest and in transit?") == "security"
    assert classify("Are you SOC 2 Type II certified?") == "compliance"
    assert classify("What are your pricing tiers?") == "pricing"
    assert any(cat == "legal" for cat in joined.values())


def test_graph_pauses_at_review_gate(tmp_path):
    graph = _make(tmp_path)
    config = {"configurable": {"thread_id": "run-review"}}
    graph.invoke(
        {"rfp_title": "T", "source_text": SAMPLE, "status": "intake", "timeline": []},
        config,
    )
    snap = graph.get_state(config)
    pending = _pending(snap)
    assert pending is not None, "expected an interrupt at the review gate"
    assert pending["gate"] == "review_queue"
    assert len(pending["items"]) >= 1
    # High-risk legal questions (e.g. liability) must be routed to review.
    flagged_text = " ".join(i["question_text"].lower() for i in pending["items"])
    assert "liability" in flagged_text


def test_full_hitl_resume_to_export(tmp_path):
    graph = _make(tmp_path)
    config = {"configurable": {"thread_id": "run-full"}}
    graph.invoke(
        {"rfp_title": "Acme", "source_text": SAMPLE, "status": "intake", "timeline": []},
        config,
    )

    pending = _pending(graph.get_state(config))
    assert pending["gate"] == "review_queue"
    items = pending["items"]

    # Build a mix of decisions: escalate liability items, rewrite one non-liability
    # item, approve the rest.
    decisions = []
    rewritten_once = False
    for item in items:
        if "liability" in item["question_text"].lower():
            decisions.append({"question_id": item["question_id"], "action": "escalate", "note": "legal to review"})
        elif not rewritten_once:
            rewritten_once = True
            decisions.append({"question_id": item["question_id"], "action": "rewrite",
                              "new_answer": "Human-authored answer.", "note": "clarified"})
        else:
            decisions.append({"question_id": item["question_id"], "action": "approve"})
    assert rewritten_once, "sample should flag at least one non-liability item for review"

    graph.invoke(Command(resume=decisions), config)

    # Now paused at the final approval gate.
    pending2 = _pending(graph.get_state(config))
    assert pending2 is not None
    assert pending2["gate"] == "final_approval"
    assert pending2["preview"].strip()

    graph.invoke(Command(resume={"action": "approve", "note": ""}), config)

    values = graph.get_state(config).values
    assert values["status"] == "exported"
    assert values["compiled_markdown"].strip()
    assert values["export_json"]["answers"], "expected compiled answers"

    answers = {a["question_id"]: a for a in values["answers"]}
    # Verify each decision type was applied.
    statuses = {a["status"] for a in answers.values()}
    assert "rewritten" in statuses
    assert "escalated" in statuses
    # Escalated items are excluded from the exported answers.
    exported_ids = {a["question_id"] for a in values["export_json"]["answers"]}
    escalated_ids = set(values["export_json"]["escalated"])
    assert escalated_ids and not (exported_ids & escalated_ids)


def test_final_reject_path(tmp_path):
    graph = _make(tmp_path)
    config = {"configurable": {"thread_id": "run-reject"}}
    graph.invoke(
        {"rfp_title": "Acme", "source_text": SAMPLE, "status": "intake", "timeline": []},
        config,
    )
    items = _pending(graph.get_state(config))["items"]
    graph.invoke(Command(resume=[{"question_id": i["question_id"], "action": "approve"} for i in items]), config)
    graph.invoke(Command(resume={"action": "reject", "note": "needs work"}), config)
    values = graph.get_state(config).values
    assert values["status"] == "rejected"
