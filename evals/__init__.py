"""Offline evaluation harness for Response Studio.

Measures the quality of the deterministic (demo-mode) pipeline against a labeled
golden dataset: question classification, risk assessment, HITL routing, knowledge
-base retrieval, and answer grounding. Runs fully offline (no API key) so it is
reproducible and safe to gate in CI.
"""
from .harness import EvalReport, run_evaluation

__all__ = ["EvalReport", "run_evaluation"]
