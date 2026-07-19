"""End-to-end API test using FastAPI's TestClient (demo mode, no LLM)."""
from __future__ import annotations

import importlib
import os

from fastapi.testclient import TestClient


def _client(tmp_path):
    os.environ["RESPONSE_STUDIO_DB"] = str(tmp_path / "api_cp.sqlite")
    os.environ.pop("OPENAI_API_KEY", None)
    import app.main as main
    importlib.reload(main)
    return TestClient(main.app)


def test_api_full_flow(tmp_path):
    client = _client(tmp_path)

    assert client.get("/api/health").json()["llm_mode"] == "demo"
    assert client.get("/").status_code == 200

    r = client.post("/api/runs", json={"use_sample": True})
    assert r.status_code == 201
    run = r.json()
    run_id = run["run_id"]
    assert run["pending_interrupt"]["gate"] == "review_queue"
    assert run["stats"]["total"] == 18

    items = run["pending_interrupt"]["items"]
    decisions = [{"question_id": i["question_id"], "action": "approve"} for i in items]
    r2 = client.post(f"/api/runs/{run_id}/resume", json={"decisions": decisions})
    assert r2.status_code == 200
    assert r2.json()["pending_interrupt"]["gate"] == "final_approval"

    r3 = client.post(f"/api/runs/{run_id}/resume", json={"action": "approve"})
    body = r3.json()
    assert body["done"] is True
    assert body["status"] == "exported"
    assert body["compiled_markdown"].strip()

    # Fetching the run again returns the persisted final state.
    again = client.get(f"/api/runs/{run_id}").json()
    assert again["status"] == "exported"


def test_api_resume_without_interrupt_conflicts(tmp_path):
    client = _client(tmp_path)
    run = client.post("/api/runs", json={"use_sample": True}).json()
    run_id = run["run_id"]
    items = run["pending_interrupt"]["items"]
    client.post(f"/api/runs/{run_id}/resume",
                json={"decisions": [{"question_id": i["question_id"], "action": "approve"} for i in items]})
    client.post(f"/api/runs/{run_id}/resume", json={"action": "approve"})
    # No interrupt pending now -> 409.
    conflict = client.post(f"/api/runs/{run_id}/resume", json={"action": "approve"})
    assert conflict.status_code == 409
