"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from tests.mocks import MockEmbedder, MockLLM


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def md_dir(tmp_path):
    """Temporary directory with three sample markdown files."""
    (tmp_path / "doc1.md").write_text(
        """\
# Introduction

This is the introduction section. It covers the basics of the system
and gives a high level overview of all components involved.

## Getting Started

To get started, install the required dependencies using the package manager.
Run the setup command to initialise the environment before first use.

## Configuration

Place the config file in your home directory under `.config/kb/`.
All settings can be overridden with environment variables.
"""
    )

    (tmp_path / "doc2.md").write_text(
        """\
# Advanced Usage

This section covers advanced topics for power users and developers.

## Performance Tuning

Increase the batch size for better throughput on large document sets.
Use GPU acceleration when a compatible device is available.

## Troubleshooting

If the system fails to start, check the application logs first.
Ensure all required dependencies are installed at the correct version.
"""
    )

    sub = tmp_path / "notes"
    sub.mkdir()
    (sub / "notes.md").write_text(
        """\
# Project Notes

Quick notes about ongoing work in the project.
Remember to update the changelog before each release.
"""
    )

    return tmp_path
