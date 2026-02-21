"""Tests for kb.chunker."""

from pathlib import Path

import pytest

from kb.chunker import Chunk, chunk_markdown, chunk_text


# ---------------------------------------------------------------------------
# chunk_text unit tests
# ---------------------------------------------------------------------------


def test_no_headers_single_chunk():
    text = "Just plain text without any headers at all. Long enough to keep."
    chunks = chunk_text(text, "f.md", min_chars=10)
    assert len(chunks) == 1
    assert chunks[0].section_path == []
    assert "plain text" in chunks[0].content


def test_basic_headers():
    text = """\
# Title
Introduction content long enough to pass the filter.

## Section
Section content that is also long enough to pass.
"""
    chunks = chunk_text(text, "f.md", min_chars=20)
    assert len(chunks) == 2


def test_section_path_single_level():
    text = """\
# Alpha
Alpha content here for testing the section path.
"""
    chunks = chunk_text(text, "f.md", min_chars=10)
    assert chunks[0].section_path == ["# Alpha"]


def test_section_path_nested():
    text = """\
# H1
h1 content text here for testing.

## H2
h2 content text here for testing.

### H3
h3 content text here for testing.
"""
    chunks = chunk_text(text, "f.md", min_chars=10)
    h3_chunks = [c for c in chunks if any("H3" in h for h in c.section_path)]
    assert len(h3_chunks) == 1
    # Path should include H1 > H2 > H3.
    assert len(h3_chunks[0].section_path) == 3


def test_section_path_resets_on_same_level():
    text = """\
# First
Content of first section, long enough.

# Second
Content of second section, long enough.
"""
    chunks = chunk_text(text, "f.md", min_chars=10)
    second = [c for c in chunks if "# Second" in c.section_path]
    assert len(second) == 1
    assert second[0].section_path == ["# Second"]


def test_pre_header_content():
    text = """\
This is pre-header content, long enough.

# Section
Section content here.
"""
    chunks = chunk_text(text, "f.md", min_chars=10)
    pre = [c for c in chunks if c.section_path == []]
    assert len(pre) == 1
    assert "pre-header" in pre[0].content


def test_short_chunks_filtered(tmp_path):
    text = """\
# Title
Hi.

## Long Section
This section has enough content to survive the minimum character filter.
"""
    chunks = chunk_text(text, "f.md", min_chars=40)
    # "# Title\nHi." is fewer than 40 chars and should be dropped.
    assert all(len(c.content) >= 40 for c in chunks)


def test_file_path_stored(tmp_path):
    md = tmp_path / "test.md"
    md.write_text("# Header\nLong enough content to be included in this chunk.")
    chunks = chunk_markdown(md)
    for chunk in chunks:
        assert chunk.file_path == str(md)


def test_chunk_serialisation_roundtrip():
    original = Chunk(
        content="hello world",
        file_path="/docs/foo.md",
        section_path=["# Title", "## Sub"],
        chunk_id=7,
    )
    restored = Chunk.from_dict(original.to_dict())
    assert restored.content == original.content
    assert restored.file_path == original.file_path
    assert restored.section_path == original.section_path
    assert restored.chunk_id == original.chunk_id


def test_display_source():
    chunk = Chunk("text", "doc.md", ["# H1", "## H2"])
    assert "doc.md" in chunk.display_source
    assert "H1" in chunk.display_source
    assert "H2" in chunk.display_source


def test_empty_file(tmp_path):
    md = tmp_path / "empty.md"
    md.write_text("")
    chunks = chunk_markdown(md)
    assert chunks == []


def test_whitespace_only_file(tmp_path):
    md = tmp_path / "ws.md"
    md.write_text("   \n\n   \n")
    chunks = chunk_markdown(md, min_chars=5)
    assert chunks == []
