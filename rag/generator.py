"""Build a grounded prompt from retrieved chunks and ask an LLM (Google Gemini)."""
from __future__ import annotations

import os

from rag.ingest import Chunk

NOT_FOUND = "I couldn't find that in the provided documents."

SYSTEM_PROMPT = f"""You are a helpful assistant that answers questions about mortgage and
home-loan documents for borrowers and loan officers.

Rules:
1. Answer ONLY using the numbered context passages below. Do not use outside knowledge.
2. After each fact, cite the passage it came from like [1] or [2][3].
3. If the context does not contain the answer, reply exactly: "{NOT_FOUND}"
4. Keep answers short and in plain language (3-6 sentences or a short list).
5. Never give personalised financial or legal advice; suggest talking to a loan officer instead."""


def build_prompt(question: str, retrieved: list[tuple[Chunk, float]]) -> str:
    context = "\n\n".join(
        f"[{i}] (source: {c.source}, page {c.page})\n{c.text}"
        for i, (c, _score) in enumerate(retrieved, start=1))
    return f"{SYSTEM_PROMPT}\n\nContext passages:\n{context}\n\nQuestion: {question}\nAnswer:"


class GeminiGenerator:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        from google import genai
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.client = genai.Client(api_key=key)
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def generate(self, prompt: str) -> str:
        resp = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config={"temperature": 0.1})        # low temperature = fewer made-up answers
        return (resp.text or "").strip()


def answer(question: str, index, generator, k: int = 4, min_score: float = 0.2) -> dict:
    """Retrieve, then generate. If nothing relevant is retrieved, refuse without
    calling the LLM (saves cost and avoids hallucinated answers)."""
    retrieved = index.search(question, k=k)
    relevant = [(c, s) for c, s in retrieved if s >= min_score]
    if not relevant:
        return {"answer": NOT_FOUND, "sources": [], "used_llm": False}
    text = generator.generate(build_prompt(question, relevant))
    return {"answer": text,
            "sources": [{"source": c.source, "page": c.page, "score": round(s, 3), "text": c.text}
                        for c, s in relevant],
            "used_llm": True}
