"""Tests run offline: a tiny bag-of-words embedder and a fake LLM stand in for the real models."""
import re

import numpy as np
import pytest

from rag.generator import NOT_FOUND, answer, build_prompt
from rag.ingest import Chunk, clean_text, split_into_chunks
from rag.retriever import VectorIndex
from eval.evaluate import retrieval_metrics

VOCAB = ["loan", "estimate", "escrow", "insurance", "closing", "disclosure", "rate", "pizza"]


class BowEmbedder:
    def encode(self, texts):
        return np.array([[len(re.findall(w, t.lower())) + 1e-3 * (i + 1) for i, w in enumerate(VOCAB)]
                         for t in texts], dtype=np.float32)


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return "A Loan Estimate lists your loan terms [1]."


CHUNKS = [Chunk("The Loan Estimate shows your loan terms and estimated costs.", "a.pdf", 1),
          Chunk("An escrow account pays taxes and insurance.", "a.pdf", 2),
          Chunk("The Closing Disclosure lists final closing costs.", "a.pdf", 3)]


def test_clean_text_joins_hyphenation():
    assert clean_text("mort-\ngage   insurance\n") == "mortgage insurance"


def test_chunks_have_overlap_and_cover_text():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = split_into_chunks(text, chunk_size=300, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0].split()[-1] in chunks[1]           # overlap carries text forward
    assert "Sentence number 199 is here." in chunks[-1]  # nothing lost at the end


def test_overlap_must_be_smaller():
    with pytest.raises(ValueError):
        split_into_chunks("abc", chunk_size=10, overlap=10)


def test_search_ranks_relevant_chunk_first(tmp_path):
    idx = VectorIndex(BowEmbedder()).build(CHUNKS)
    assert idx.search("what is escrow", k=2)[0][0].page == 2
    idx.save(tmp_path)
    again = VectorIndex.load(tmp_path, BowEmbedder())
    assert again.search("closing disclosure", k=1)[0][0].page == 3


def test_prompt_contains_sources_and_rules():
    p = build_prompt("q?", [(CHUNKS[0], 0.9)])
    assert "[1] (source: a.pdf, page 1)" in p and "ONLY" in p


def test_refuses_without_calling_llm_when_nothing_relevant():
    idx = VectorIndex(BowEmbedder()).build(CHUNKS)
    llm = FakeLLM()
    out = answer("best pizza?", idx, llm, min_score=0.9)
    assert out["answer"] == NOT_FOUND and llm.calls == 0


def test_answer_returns_sources():
    idx = VectorIndex(BowEmbedder()).build(CHUNKS)
    out = answer("loan estimate", idx, FakeLLM(), k=2, min_score=0.1)
    assert out["used_llm"] and out["sources"][0]["page"] == 1


def test_retrieval_metrics():
    idx = VectorIndex(BowEmbedder()).build(CHUNKS)
    qs = [{"question": "escrow", "expected_keywords": ["escrow"]},
          {"question": "closing disclosure", "expected_keywords": ["Closing Disclosure"]},
          {"question": "pizza", "expected_keywords": [], "out_of_scope": True}]
    m = retrieval_metrics(idx, qs)
    assert m["questions"] == 2 and m["hit@1"] == 1.0 and m["mrr"] == 1.0
