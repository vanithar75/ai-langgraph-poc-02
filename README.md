# ai-langgraph-poc-02 — Response Studio (Option B)

**Response Studio** is a LangGraph proof-of-concept for a **presales RFP / security
questionnaire assistant with human-in-the-loop (HITL) control** (Option B from the
POC options brief).

Upload / select an RFP or security questionnaire and the stateful agent:

1. **Decomposes** it into individual questions and classifies each
   (security / compliance / product / pricing / legal).
2. **Drafts** an answer for every question from a local markdown knowledge base.
3. **Scores confidence + flags risk**, routing low-confidence or high-risk items
   to a human review queue.
4. **Pauses at HITL Gate 1** (`interrupt()`): a reviewer can **approve / rewrite /
   escalate** each flagged answer.
5. **Compiles** the response package and **pauses at HITL Gate 2** for final
   sign-off before anything is exported.
6. **Exports** the compiled response as Markdown + JSON.

State is durably checkpointed to **SQLite** per run, so a paused run survives a
process restart and resumes exactly where it left off.

## Architecture

```
decompose -> draft -> score_and_route -> human_review(interrupt #1)
   -> compile -> final_approval(interrupt #2) -> export
```

- **Runtime:** Python 3.11+, FastAPI, LangGraph, Pydantic
- **Persistence:** `SqliteSaver` checkpointer (`thread_id` = run id)
- **HITL:** `interrupt()` / `Command(resume=...)` at 2 gates
- **LLM:** uses OpenAI when `OPENAI_API_KEY` is set; otherwise a **deterministic,
  offline demo mode** composes answers from the knowledge base (budget-safe, no
  network). Confidence/routing derive from retrieval quality either way.
- **UI:** a single static HTML page served by FastAPI (no separate frontend build).

## Layout

```
app/            FastAPI app, LangGraph graph, state, KB retrieval, LLM wrapper
static/         Single-page UI (timeline, review queue, export preview)
data/           Sample RFP + markdown knowledge base
evals/          Offline eval harness + labeled golden dataset (quality metrics)
tests/          Graph interrupt/resume + API tests (run fully offline)
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run (development)

```bash
.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 and click **Start sample run**.

Optional live LLM mode:

```bash
export OPENAI_API_KEY=sk-...     # optional; demo mode is the default
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Test

```bash
.venv/bin/python -m pytest -q
```

## Evaluate

An offline evaluation harness (`evals/`) scores the demo-mode pipeline against a
labeled golden dataset (`evals/dataset.py`) — no API key required. It measures
question classification, risk assessment, HITL routing, knowledge-base retrieval,
answer grounding, and confidence calibration, and gates on per-metric thresholds.

```bash
.venv/bin/python -m evals                       # print report; exit 1 if below thresholds
.venv/bin/python -m evals --json evals/report.json   # also write full per-case JSON
.venv/bin/python -m evals --no-gate             # report only, always exit 0
```

Metrics gated: `category_accuracy`, `risk_accuracy`, `high_risk_recall`,
`routing_accuracy`, `gap_flag_rate`, `retrieval_hit_rate`, `answer_grounding_rate`
(thresholds live in `evals/harness.py`). The gate is also exercised by
`tests/test_eval.py`, so `pytest` fails on a quality regression. Safety-critical
metrics (high-risk items and knowledge gaps must always reach a human) are held
at 100%.

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/api/health` | Liveness + current `llm_mode` |
| `GET`  | `/api/sample` | Bundled sample questionnaire |
| `POST` | `/api/runs` | Start a run (decompose → draft → score → pause at Gate 1) |
| `GET`  | `/api/runs/{id}` | Current run state + pending interrupt |
| `POST` | `/api/runs/{id}/resume` | Resume a HITL gate with decisions / final approval |

## Scope / non-goals (POC)

No auth, no multi-tenant, no real document upload/parsing (paste or sample text),
no external services. Kept to ~15–25 questions per the POC guardrails.
