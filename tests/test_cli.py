"""Integration tests for the kb CLI (kb.cli)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from kb.cli import cli
from tests.mocks import MockEmbedder, MockLLM


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def mock_embedder_cls():
    """Patch SentenceTransformerEmbedder so it returns a MockEmbedder."""
    with patch("kb.cli.SentenceTransformerEmbedder") as mock_cls:
        mock_cls.return_value = MockEmbedder()
        yield mock_cls


@pytest.fixture
def mock_llm_cls():
    """Patch OllamaLLM so it returns a MockLLM."""
    with patch("kb.cli.OllamaLLM") as mock_cls:
        mock_cls.return_value = MockLLM("The knowledge base says: 42.")
        yield mock_cls


# ---------------------------------------------------------------------------
# kb index
# ---------------------------------------------------------------------------


def test_index_finds_and_indexes_files(runner, md_dir, tmp_path, mock_embedder_cls):
    result = runner.invoke(
        cli, ["--store", str(tmp_path / "store"), "index", str(md_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "Indexed" in result.output


def test_index_reports_chunk_count(runner, md_dir, tmp_path, mock_embedder_cls):
    result = runner.invoke(
        cli, ["--store", str(tmp_path / "store"), "index", str(md_dir)]
    )
    assert "chunk" in result.output.lower()


def test_index_skips_unchanged_on_second_run(runner, md_dir, tmp_path, mock_embedder_cls):
    store_path = str(tmp_path / "store")
    runner.invoke(cli, ["--store", store_path, "index", str(md_dir)])
    result = runner.invoke(cli, ["--store", store_path, "index", str(md_dir)])
    assert result.exit_code == 0, result.output
    assert "Skipped" in result.output


def test_index_force_reindexes(runner, md_dir, tmp_path, mock_embedder_cls):
    store_path = str(tmp_path / "store")
    runner.invoke(cli, ["--store", store_path, "index", str(md_dir)])
    result = runner.invoke(
        cli, ["--store", store_path, "index", "--force", str(md_dir)]
    )
    assert result.exit_code == 0, result.output
    # With --force all files are re-indexed; skipped count should be 0.
    assert "Skipped (unchanged): 0" in result.output


def test_index_nonexistent_directory(runner, tmp_path, mock_embedder_cls):
    result = runner.invoke(
        cli, ["--store", str(tmp_path / "store"), "index", "/no/such/dir"]
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# kb status
# ---------------------------------------------------------------------------


def test_status_empty_store(runner, tmp_path):
    result = runner.invoke(cli, ["--store", str(tmp_path / "store"), "status"])
    assert result.exit_code == 0, result.output
    assert "Files" in result.output
    assert "0" in result.output


def test_status_after_index(runner, md_dir, tmp_path, mock_embedder_cls):
    store_path = str(tmp_path / "store")
    runner.invoke(cli, ["--store", store_path, "index", str(md_dir)])

    result = runner.invoke(cli, ["--store", store_path, "status"])
    assert result.exit_code == 0, result.output
    assert "Files" in result.output
    # Should list at least one indexed file.
    assert ".md" in result.output


# ---------------------------------------------------------------------------
# kb query
# ---------------------------------------------------------------------------


def test_query_empty_store_warns(runner, tmp_path, mock_embedder_cls):
    result = runner.invoke(
        cli, ["--store", str(tmp_path / "store"), "query", "what is this?"]
    )
    assert result.exit_code == 0, result.output
    assert "empty" in result.output.lower()


def test_query_no_llm_shows_chunks(runner, md_dir, tmp_path, mock_embedder_cls):
    store_path = str(tmp_path / "store")
    runner.invoke(cli, ["--store", store_path, "index", str(md_dir)])

    result = runner.invoke(
        cli, ["--store", store_path, "query", "--no-llm", "introduction"]
    )
    assert result.exit_code == 0, result.output
    assert "Chunk" in result.output or "Source" in result.output


def test_query_with_llm_shows_answer(
    runner, md_dir, tmp_path, mock_embedder_cls, mock_llm_cls
):
    store_path = str(tmp_path / "store")
    runner.invoke(cli, ["--store", store_path, "index", str(md_dir)])

    result = runner.invoke(
        cli, ["--store", store_path, "query", "what is covered?"]
    )
    assert result.exit_code == 0, result.output
    assert "Answer:" in result.output
    assert "42" in result.output  # from MockLLM response


def test_query_with_llm_shows_sources(
    runner, md_dir, tmp_path, mock_embedder_cls, mock_llm_cls
):
    store_path = str(tmp_path / "store")
    runner.invoke(cli, ["--store", store_path, "index", str(md_dir)])

    result = runner.invoke(
        cli, ["--store", store_path, "query", "configuration"]
    )
    assert result.exit_code == 0, result.output
    assert "Sources:" in result.output


def test_query_top_k_option(runner, md_dir, tmp_path, mock_embedder_cls, mock_llm_cls):
    store_path = str(tmp_path / "store")
    runner.invoke(cli, ["--store", store_path, "index", str(md_dir)])

    result = runner.invoke(
        cli, ["--store", store_path, "query", "--top-k", "1", "--no-llm", "help"]
    )
    assert result.exit_code == 0, result.output
