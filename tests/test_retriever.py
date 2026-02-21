"""Tests for kb.retriever."""

from __future__ import annotations

import pytest

from kb.chunker import Chunk
from kb.retriever import retrieve
from kb.store import KnowledgeStore
from tests.mocks import EMBED_DIM, MockEmbedder


@pytest.fixture
def populated_store(tmp_path, mock_embedder):
    store = KnowledgeStore(tmp_path / "store", embedding_dim=EMBED_DIM)

    python_chunks = [
        Chunk("Python is a high-level programming language.", "python.md", ["# Python"]),
        Chunk("Python supports multiple programming paradigms.", "python.md", ["# Python", "## Paradigms"]),
    ]
    store.add_file(
        "python.md",
        1.0,
        python_chunks,
        mock_embedder.embed([c.content for c in python_chunks]),
    )

    rust_chunks = [
        Chunk("Rust is a systems programming language focused on safety.", "rust.md", ["# Rust"]),
        Chunk("Rust achieves memory safety without a garbage collector.", "rust.md", ["# Rust", "## Memory"]),
    ]
    store.add_file(
        "rust.md",
        1.0,
        rust_chunks,
        mock_embedder.embed([c.content for c in rust_chunks]),
    )

    return store


def test_retrieve_returns_chunks(populated_store, mock_embedder):
    results = retrieve("programming language", populated_store, mock_embedder, top_k=2)
    assert len(results) > 0
    assert all(isinstance(c, Chunk) for c in results)


def test_retrieve_respects_top_k(populated_store, mock_embedder):
    results = retrieve("anything", populated_store, mock_embedder, top_k=1)
    assert len(results) <= 1


def test_retrieve_empty_store(tmp_path, mock_embedder):
    store = KnowledgeStore(tmp_path / "store", embedding_dim=EMBED_DIM)
    results = retrieve("question", store, mock_embedder, top_k=5)
    assert results == []


def test_retrieve_chunk_ids_set(populated_store, mock_embedder):
    results = retrieve("language", populated_store, mock_embedder, top_k=3)
    for chunk in results:
        assert chunk.chunk_id >= 0
