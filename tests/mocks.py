"""Lightweight test doubles for the embedding model and LLM.

These are deterministic, dependency-free stubs that let the test suite run
without sentence-transformers, CUDA, or a running Ollama instance.
"""

from __future__ import annotations

import numpy as np

from kb.embedder import BaseEmbedder
from kb.generator import BaseLLM

# Small dimension keeps FAISS operations fast in tests.
EMBED_DIM = 8


class MockEmbedder(BaseEmbedder):
    """Deterministic embedder that derives vectors from text hashes."""

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        vectors: list[np.ndarray] = []
        for text in texts:
            # Seed from hash so the same text always yields the same vector.
            seed = abs(hash(text)) % (2**31)
            rng = np.random.RandomState(seed)
            v = rng.randn(EMBED_DIM).astype(np.float32)
            norm = np.linalg.norm(v)
            if norm > 0:
                v /= norm
            vectors.append(v)
        return np.array(vectors, dtype=np.float32)

    @property
    def dim(self) -> int:
        return EMBED_DIM


class MockLLM(BaseLLM):
    """LLM stub that always returns a fixed canned response."""

    def __init__(self, response: str = "Mock answer from the knowledge base.") -> None:
        self._response = response

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return self._response
