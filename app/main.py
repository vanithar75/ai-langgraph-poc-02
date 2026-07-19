"""FastAPI app for Response Studio.

Exposes a thin HTTP API over the LangGraph run + HITL resume cycle and serves a
single static HTML page. State lives entirely in the SQLite checkpointer keyed by
``thread_id`` (the run id).
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel, Field

from .graph import build_graph, make_sqlite_checkpointer
from .knowledge_base import DATA_DIR
from .llm import llm_mode

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
DB_PATH = os.environ.get("RESPONSE_STUDIO_DB", str(BASE_DIR / ".data" / "checkpoints.sqlite"))

app = FastAPI(title="Response Studio", version="0.1.0")

_checkpointer = make_sqlite_checkpointer(DB_PATH)
_graph = build_graph(_checkpointer)
_run_ids: set[str] = set()


# --- request models -------------------------------------------------------

class StartRunRequest(BaseModel):
    rfp_title: str | None = None
    source_text: str | None = None
    use_sample: bool = True


class ReviewDecisionIn(BaseModel):
    question_id: str
    action: str = Field(pattern="^(approve|rewrite|escalate)$")
    new_answer: str | None = None
    note: str = ""


class ResumeRequest(BaseModel):
    decisions: list[ReviewDecisionIn] | None = None
    action: str | None = None  # for the final approval gate
    note: str = ""


# --- helpers --------------------------------------------------------------

def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _pending_interrupt(snapshot) -> dict | None:
    for task in snapshot.tasks:
        for itr in task.interrupts:
            return itr.value
    return None


def _run_view(run_id: str) -> dict:
    snapshot = _graph.get_state(_config(run_id))
    values = snapshot.values or {}
    if not values:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    pending = _pending_interrupt(snapshot)
    return {
        "run_id": run_id,
        "status": values.get("status", "unknown"),
        "llm_mode": values.get("llm_mode", llm_mode()),
        "rfp_title": values.get("rfp_title", ""),
        "questions": values.get("questions", []),
        "answers": values.get("answers", []),
        "review_queue": values.get("review_queue", []),
        "stats": values.get("stats", {}),
        "compiled_markdown": values.get("compiled_markdown", ""),
        "export_json": values.get("export_json", {}),
        "timeline": values.get("timeline", []),
        "next": list(snapshot.next),
        "pending_interrupt": pending,
        "done": not pending and values.get("status") in ("exported", "rejected"),
    }


def _load_sample() -> tuple[str, str]:
    sample = DATA_DIR / "sample_rfp.md"
    text = sample.read_text(encoding="utf-8") if sample.exists() else ""
    title = "Acme Corp Security Questionnaire"
    return title, text


# --- routes ---------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "llm_mode": llm_mode()}


@app.get("/api/sample")
def sample() -> dict:
    title, text = _load_sample()
    return {"rfp_title": title, "source_text": text}


@app.post("/api/runs")
def start_run(req: StartRunRequest) -> JSONResponse:
    if req.use_sample or not (req.source_text and req.source_text.strip()):
        title, text = _load_sample()
        rfp_title = req.rfp_title or title
        source_text = text
    else:
        source_text = req.source_text
        rfp_title = req.rfp_title or "Custom RFP"

    if not source_text.strip():
        raise HTTPException(status_code=400, detail="No RFP content provided")

    run_id = uuid.uuid4().hex[:12]
    _run_ids.add(run_id)
    _graph.invoke(
        {"rfp_title": rfp_title, "source_text": source_text, "status": "intake", "timeline": []},
        _config(run_id),
    )
    return JSONResponse(_run_view(run_id), status_code=201)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return _run_view(run_id)


@app.post("/api/runs/{run_id}/resume")
def resume_run(run_id: str, req: ResumeRequest) -> dict:
    snapshot = _graph.get_state(_config(run_id))
    if not (snapshot.values or {}):
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    pending = _pending_interrupt(snapshot)
    if pending is None:
        raise HTTPException(status_code=409, detail="Run is not waiting for input")

    gate = pending.get("gate")
    if gate == "review_queue":
        decisions = [d.model_dump() for d in (req.decisions or [])]
        resume_value: object = decisions
    else:  # final_approval
        resume_value = {"action": req.action or "approve", "note": req.note}

    _graph.invoke(Command(resume=resume_value), _config(run_id))
    return _run_view(run_id)


# --- static UI ------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
