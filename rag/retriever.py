"""Embed chunks, save/load a vector index, and retrieve the most relevant chunks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from rag.ingest import Chunk

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Free, local embedding model (runs on CPU, no API key needed)."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, show_progress_bar=False,
                                 normalize_embeddings=True).astype(np.float32)


def _normalise(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norms, 1e-12, None)


class VectorIndex:
    def __init__(self, embedder: Embedder, chunks: list[Chunk] | None = None,
                 vectors: np.ndarray | None = None):
        self.embedder = embedder
        self.chunks = chunks or []
        self.vectors = vectors

    def build(self, chunks: list[Chunk]) -> "VectorIndex":
        self.chunks = chunks
        self.vectors = _normalise(self.embedder.encode([c.text for c in chunks]))
        return self

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if self.vectors is None or not self.chunks:
            raise RuntimeError("Index is empty. Run build_index.py first.")
        q = _normalise(self.embedder.encode([query]))[0]
        scores = self.vectors @ q                      # cosine similarity
        k = min(k, len(self.chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        (folder / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=1),
            encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, embedder: Embedder) -> "VectorIndex":
        if not (folder / "vectors.npy").exists():
            raise FileNotFoundError(f"No index in {folder}. Run: python build_index.py")
        vectors = np.load(folder / "vectors.npy")
        chunks = [Chunk(**d) for d in json.loads((folder / "chunks.json").read_text(encoding="utf-8"))]
        return cls(embedder, chunks, vectors)
