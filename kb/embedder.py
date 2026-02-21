"""Embedding model wrappers.

The abstract ``BaseEmbedder`` interface lets tests inject a fast mock
without touching the heavy ``sentence-transformers`` library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """Common interface for text embedding models."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a float32 array of shape ``(len(texts), dim)``."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""


class SentenceTransformerEmbedder(BaseEmbedder):
    """Wraps a ``sentence-transformers`` model for local embedding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Lazy import so the heavy library is only loaded when actually used.
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, convert_to_numpy=True)  # type: ignore[return-value]

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()
