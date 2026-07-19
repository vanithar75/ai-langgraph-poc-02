"""Answer generation.

Two modes:

* **demo** (default, budget-safe): deterministic answers composed from the best
  matching knowledge-base chunks. No network, no API key, fully reproducible.
* **live**: if ``OPENAI_API_KEY`` is set, an LLM synthesizes the answer from the
  retrieved KB context. Confidence/routing still derive from retrieval quality so
  HITL behavior stays consistent and testable.
"""
from __future__ import annotations

import os
import textwrap

from .knowledge_base import RetrievalHit

CONFIDENCE_THRESHOLD = 0.45
_MODEL = os.environ.get("RESPONSE_STUDIO_MODEL", "gpt-4o-mini")


def is_live_mode() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def llm_mode() -> str:
    return "live" if is_live_mode() else "demo"


def _confidence_from_hits(hits: list[RetrievalHit]) -> float:
    if not hits:
        return 0.1
    top = hits[0].score
    return round(min(0.98, max(0.1, top * 1.7)), 3)


def _demo_answer(question_text: str, hits: list[RetrievalHit]) -> str:
    if not hits:
        return (
            "No supporting material was found in the knowledge base for this "
            "question. A subject-matter expert needs to provide an authoritative "
            "answer before this item can be submitted."
        )
    parts = [h.chunk.text.strip() for h in hits[:2]]
    body = " ".join(parts)
    return textwrap.shorten(body, width=600, placeholder=" …")


def _live_answer(question_text: str, hits: list[RetrievalHit]) -> str:
    from langchain_openai import ChatOpenAI

    context = "\n\n".join(
        f"[{h.chunk.category}:{h.chunk.title}] {h.chunk.text}" for h in hits
    ) or "(no relevant knowledge base context found)"
    prompt = (
        "You are a presales RFP specialist. Using ONLY the context below, write a "
        "concise, professional answer (2-4 sentences) to the questionnaire item. "
        "If the context is insufficient, say so explicitly and do not invent facts.\n\n"
        f"Context:\n{context}\n\nQuestion: {question_text}\n\nAnswer:"
    )
    model = ChatOpenAI(model=_MODEL, temperature=0)
    return model.invoke(prompt).content.strip()


def draft_answer(question_text: str, hits: list[RetrievalHit]) -> dict:
    """Return ``{draft, citations, confidence, mode}`` for one question."""
    citations = [f"{h.chunk.category}: {h.chunk.title}" for h in hits]
    confidence = _confidence_from_hits(hits)
    if is_live_mode():
        try:
            draft = _live_answer(question_text, hits)
        except Exception as exc:  # pragma: no cover - network/credential failure
            draft = _demo_answer(question_text, hits)
            draft += f"\n\n[fell back to demo mode: {exc}]"
    else:
        draft = _demo_answer(question_text, hits)
    return {
        "draft": draft,
        "citations": citations,
        "confidence": confidence,
        "mode": llm_mode(),
    }
