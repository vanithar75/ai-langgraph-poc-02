# ai-langgraph-poc-02 — Response Studio (Option B)

## Cursor Cloud specific instructions

### What this is

**Response Studio** — a LangGraph presales RFP / security-questionnaire assistant
with human-in-the-loop gates (Option B of the POC brief). Python + FastAPI +
LangGraph + Pydantic, single served HTML UI, SQLite checkpointer. See `README.md`
for architecture, the API table, and standard setup/run/test commands.

### Environment / running

- The project uses a Python venv at `.venv` (Python 3.12 on the VM). The startup
  update script creates `.venv` and installs `requirements.txt` into it. Always
  use the venv binaries: `.venv/bin/uvicorn`, `.venv/bin/python`, `.venv/bin/pytest`.
- Dev server: `.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`,
  then open the root URL and click **Start sample run**.
- Tests: `.venv/bin/python -m pytest -q` (fully offline; no API key needed).

### Non-obvious gotchas

- **Demo mode is the default and is intentional** (budget guardrail). Answers are
  composed deterministically from the local KB unless `OPENAI_API_KEY` is set,
  which flips the app to live LLM generation. Confidence/HITL routing come from KB
  retrieval quality in *both* modes, so review-queue behavior is stable/testable
  without an API key. Do not "fix" demo mode by hard-wiring an LLM.
- **State lives only in the SQLite checkpointer** (`.data/checkpoints.sqlite`,
  gitignored), keyed by `thread_id` = run id. There is no separate DB or in-memory
  store of record. Deleting that file resets all runs. Run ids are ephemeral in a
  browser session; a fresh checkpoint file means old run ids 404.
- **HITL resume contract:** the review gate (`gate: "review_queue"`) expects
  `{"decisions": [...]}`; the final gate (`gate: "final_approval"`) expects
  `{"action": "approve"|"reject"}`. Resuming when no interrupt is pending returns
  HTTP 409 by design.
- Escalated answers are intentionally excluded from the exported package and listed
  separately as pending SME input.

### Update script behavior

The startup update script creates/reuses `.venv` and installs `requirements.txt`
into it only if that file exists (guarded, idempotent, safe even if this PR's code
is not merged). It intentionally does NOT start the server or run migrations.
