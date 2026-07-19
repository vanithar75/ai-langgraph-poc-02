# ai-langgraph-poc-02

## Cursor Cloud specific instructions

### Current repository state

This repository is currently a **greenfield placeholder**: the only tracked file is `README.md`. There is no application code, no dependency manifest (`requirements.txt`, `pyproject.toml`, `package.json`, etc.), no services, no tests, and no build tooling yet. As a result there is nothing to lint, build, run, or test until code is added.

The name suggests an intended **LangGraph** proof of concept (LangChain's graph-based agent orchestration framework), which typically implies a Python stack and an LLM provider API key (e.g. `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`). None of that is wired up yet.

### Environment

The VM already provides the common toolchains needed to bootstrap either a Python or Node project:

- Python `3.12` with `pip`
- Node `22` with `npm`

`uv` and `docker` are not installed by default; add them only if the project adopts them.

### Update script behavior

The configured startup update script is intentionally guarded/idempotent: it installs dependencies only if a recognized manifest exists (`requirements.txt` → `pip install`, `package.json` → `npm install`). While the repo is empty it is a safe no-op. Once real dependencies land, revisit this section and the update script to match the chosen stack (and add lint/test/build/run notes here).
