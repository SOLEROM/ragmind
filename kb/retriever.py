"""Retrieval layer: embed the query and search the vector store."""

from __future__ import annotations

from .chunker import Chunk
from .embedder import BaseEmbedder
from .store import KnowledgeStore


def retrieve(
    question: str,
    store: KnowledgeStore,
    embedder: BaseEmbedder,
    top_k: int = 5,
) -> list[Chunk]:
    """Return the *top_k* most relevant chunks for *question*."""
    vector = embedder.embed([question])[0]
    results = store.search(vector, top_k=top_k)
    return [chunk for chunk, _dist in results]
