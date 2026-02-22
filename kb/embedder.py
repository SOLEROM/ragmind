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
    """Wraps a ``sentence-transformers`` model for local embedding.

    The underlying model is loaded lazily on the first call to
    :meth:`embed` or :attr:`dim`, so constructing this object is cheap
    and does *not* pay the 2–5 s model-load cost up front.  The model
    is loaded at most once per process lifetime.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None  # loaded on first use

    def _load(self) -> None:
        """Load model weights from disk if not already in memory."""
        if self._model is None:
            # Lazy import — keeps startup fast when the embedder is
            # constructed but never used (e.g. query on an empty store).
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        self._load()
        return self._model.encode(texts, convert_to_numpy=True)  # type: ignore[return-value]

    @property
    def dim(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()
