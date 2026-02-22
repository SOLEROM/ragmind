"""Tests for kb.store.KnowledgeStore."""

from __future__ import annotations

import numpy as np
import pytest

from kb.chunker import Chunk
from kb.store import KnowledgeStore
from tests.mocks import EMBED_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunks(file_path: str, n: int) -> list[Chunk]:
    return [
        Chunk(
            content=f"Content {i} from {file_path} — long enough to be useful.",
            file_path=file_path,
            section_path=[f"# Section {i}"],
        )
        for i in range(n)
    ]


def _vecs(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n, EMBED_DIM).astype(np.float32)


@pytest.fixture
def store(tmp_path) -> KnowledgeStore:
    return KnowledgeStore(tmp_path / "store", embedding_dim=EMBED_DIM)


# ---------------------------------------------------------------------------
# Basic add + search
# ---------------------------------------------------------------------------


def test_add_and_search(store):
    ch = _chunks("a.md", 3)
    vecs = _vecs(3, seed=1)
    store.add_file("a.md", 1.0, ch, vecs)

    results = store.search(vecs[0], top_k=1)
    assert len(results) == 1
    chunk, dist = results[0]
    assert chunk.file_path == "a.md"
    assert isinstance(dist, float)


def test_search_top_k_respected(store):
    store.add_file("a.md", 1.0, _chunks("a.md", 10), _vecs(10, seed=1))
    results = store.search(_vecs(1, seed=99)[0], top_k=3)
    assert len(results) <= 3


def test_search_empty_store(store):
    results = store.search(_vecs(1)[0], top_k=5)
    assert results == []


def test_search_result_order(store):
    """The nearest vector should be first."""
    vecs = _vecs(5, seed=42)
    store.add_file("a.md", 1.0, _chunks("a.md", 5), vecs)
    results = store.search(vecs[2], top_k=5)
    # The exact match (distance ≈ 0) should come first.
    assert results[0][1] < results[-1][1]


# ---------------------------------------------------------------------------
# needs_update
# ---------------------------------------------------------------------------


def test_needs_update_new_file(store):
    assert store.needs_update("new.md", 1000.0) is True


def test_needs_update_unchanged(store):
    store.add_file("a.md", 1000.0, _chunks("a.md", 2), _vecs(2))
    assert store.needs_update("a.md", 1000.0) is False


def test_needs_update_older_mtime(store):
    store.add_file("a.md", 1000.0, _chunks("a.md", 2), _vecs(2))
    # Same mtime — not newer, so no update needed.
    assert store.needs_update("a.md", 999.0) is False


def test_needs_update_newer_mtime(store):
    store.add_file("a.md", 1000.0, _chunks("a.md", 2), _vecs(2))
    assert store.needs_update("a.md", 2000.0) is True


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------


def test_incremental_update_replaces_chunks(store):
    store.add_file("a.md", 1.0, _chunks("a.md", 2), _vecs(2, seed=1))
    assert store.total_chunks == 2

    store.add_file("a.md", 2.0, _chunks("a.md", 4), _vecs(4, seed=2))
    assert store.total_chunks == 4


def test_incremental_update_old_chunks_gone(store):
    old_chunks = _chunks("a.md", 2)
    store.add_file("a.md", 1.0, old_chunks, _vecs(2, seed=1))

    # The new content is very different — old chunks should not appear.
    new_vecs = _vecs(3, seed=77)
    new_chunks = [
        Chunk("Brand new content for testing.", "a.md", ["# New"])
        for _ in range(3)
    ]
    store.add_file("a.md", 2.0, new_chunks, new_vecs)

    # Only the new chunks should be retrievable via the public search API.
    assert store.total_chunks == 3
    results = store.search(new_vecs[0], top_k=10)
    assert all("Brand new content" in chunk.content for chunk, _ in results)


# ---------------------------------------------------------------------------
# remove_file
# ---------------------------------------------------------------------------


def test_remove_file_updates_counts(store):
    store.add_file("a.md", 1.0, _chunks("a.md", 2), _vecs(2, seed=1))
    store.add_file("b.md", 1.0, _chunks("b.md", 3), _vecs(3, seed=2))

    store.remove_file("a.md")

    assert store.total_files == 1
    assert store.total_chunks == 3
    assert "a.md" not in store.indexed_files()
    assert "b.md" in store.indexed_files()


def test_remove_nonexistent_file(store):
    # Should not raise.
    store.remove_file("ghost.md")


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


def test_total_counts_multi_file(store):
    store.add_file("a.md", 1.0, _chunks("a.md", 2), _vecs(2, seed=1))
    store.add_file("b.md", 1.0, _chunks("b.md", 3), _vecs(3, seed=2))
    assert store.total_files == 2
    assert store.total_chunks == 5


def test_indexed_files(store):
    store.add_file("x.md", 1.0, _chunks("x.md", 1), _vecs(1, seed=1))
    store.add_file("y.md", 1.0, _chunks("y.md", 1), _vecs(1, seed=2))
    assert set(store.indexed_files()) == {"x.md", "y.md"}


# ---------------------------------------------------------------------------
# Persistence (save / load)
# ---------------------------------------------------------------------------


def test_save_and_load_preserves_counts(store, tmp_path):
    store.add_file("a.md", 1.0, _chunks("a.md", 3), _vecs(3, seed=5))
    store.save()

    store2 = KnowledgeStore(tmp_path / "store", embedding_dim=EMBED_DIM)
    store2.load()
    assert store2.total_files == 1
    assert store2.total_chunks == 3


def test_save_and_load_search_works(store, tmp_path):
    vecs = _vecs(4, seed=7)
    store.add_file("a.md", 1.0, _chunks("a.md", 4), vecs)
    store.save()

    store2 = KnowledgeStore(tmp_path / "store", embedding_dim=EMBED_DIM)
    store2.load()
    results = store2.search(vecs[0], top_k=2)
    assert len(results) > 0
    assert results[0][0].file_path == "a.md"


def test_load_empty_store(tmp_path):
    store = KnowledgeStore(tmp_path / "empty", embedding_dim=EMBED_DIM)
    store.load()  # should not raise
    assert store.total_chunks == 0


def test_dimension_mismatch_raises(store, tmp_path):
    store.add_file("a.md", 1.0, _chunks("a.md", 1), _vecs(1))
    store.save()

    store_wrong = KnowledgeStore(tmp_path / "store", embedding_dim=EMBED_DIM + 4)
    with pytest.raises(ValueError, match="mismatch"):
        store_wrong.load()
