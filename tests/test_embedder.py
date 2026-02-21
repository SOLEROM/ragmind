"""Tests for the MockEmbedder (and by extension the BaseEmbedder contract)."""

import numpy as np

from tests.mocks import EMBED_DIM, MockEmbedder


def test_embed_shape():
    embedder = MockEmbedder()
    vectors = embedder.embed(["hello world", "foo bar baz"])
    assert vectors.shape == (2, EMBED_DIM)


def test_embed_dtype():
    embedder = MockEmbedder()
    vectors = embedder.embed(["test"])
    assert vectors.dtype == np.float32


def test_embed_deterministic():
    embedder = MockEmbedder()
    v1 = embedder.embed(["hello"])
    v2 = embedder.embed(["hello"])
    np.testing.assert_array_equal(v1, v2)


def test_embed_different_texts_differ():
    embedder = MockEmbedder()
    v1 = embedder.embed(["hello"])
    v2 = embedder.embed(["world"])
    assert not np.allclose(v1, v2)


def test_embed_dim_property():
    embedder = MockEmbedder()
    assert embedder.dim == EMBED_DIM


def test_embed_single_text():
    embedder = MockEmbedder()
    vectors = embedder.embed(["single"])
    assert vectors.shape == (1, EMBED_DIM)


def test_embed_empty_list():
    embedder = MockEmbedder()
    vectors = embedder.embed([])
    assert vectors.shape == (0, EMBED_DIM)
