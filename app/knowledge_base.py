"""Markdown knowledge base loading + lightweight lexical retrieval.

No external services or embeddings are required — retrieval uses simple token
overlap so the POC runs fully offline and deterministically in demo mode.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KB_DIR = DATA_DIR / "knowledge_base"

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "is", "are",
    "do", "does", "you", "your", "we", "our", "how", "what", "with", "at",
    "can", "your", "have", "has", "be", "by", "per", "as", "it", "that", "this",
    "describe", "provide", "support", "offer", "please",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass
class KBChunk:
    id: str
    category: str
    title: str
    text: str

    @property
    def tokens(self) -> set[str]:
        return set(tokenize(f"{self.title} {self.text}"))


@lru_cache(maxsize=1)
def load_chunks() -> tuple[KBChunk, ...]:
    """Load and cache KB chunks split on level-2 markdown headings."""
    chunks: list[KBChunk] = []
    if not KB_DIR.exists():
        return tuple()
    for md_file in sorted(KB_DIR.glob("*.md")):
        category = md_file.stem
        raw = md_file.read_text(encoding="utf-8")
        # Split into (heading, body) sections on "## " headings.
        sections = re.split(r"^##\s+", raw, flags=re.MULTILINE)
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            lines = section.splitlines()
            title = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            if not body:
                continue
            chunks.append(
                KBChunk(
                    id=f"{category}#{i}",
                    category=category,
                    title=title,
                    text=body,
                )
            )
    return tuple(chunks)


@dataclass
class RetrievalHit:
    chunk: KBChunk
    score: float


def retrieve(query: str, top_k: int = 3) -> list[RetrievalHit]:
    """Return the top_k KB chunks scored by token-overlap (Jaccard-ish)."""
    chunks = load_chunks()
    q_tokens = set(tokenize(query))
    if not q_tokens or not chunks:
        return []
    hits: list[RetrievalHit] = []
    for chunk in chunks:
        c_tokens = chunk.tokens
        if not c_tokens:
            continue
        overlap = q_tokens & c_tokens
        if not overlap:
            continue
        # Overlap coverage of the query, damped by chunk size to avoid huge
        # chunks always winning.
        coverage = len(overlap) / len(q_tokens)
        size_penalty = 1 / (1 + math.log(1 + len(c_tokens) / 20))
        score = coverage * (0.6 + 0.4 * size_penalty)
        hits.append(RetrievalHit(chunk=chunk, score=round(score, 4)))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]
