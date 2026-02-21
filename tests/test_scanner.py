"""Tests for kb.scanner."""

from kb.scanner import scan_markdown_files


def test_finds_md_files(md_dir):
    names = {f.name for f in scan_markdown_files(md_dir)}
    assert "doc1.md" in names
    assert "doc2.md" in names
    assert "notes.md" in names


def test_ignores_non_md_files(tmp_path):
    (tmp_path / "readme.txt").write_text("plain text")
    (tmp_path / "data.json").write_text("{}")
    (tmp_path / "doc.md").write_text("# Hello\nContent here.")

    files = list(scan_markdown_files(tmp_path))
    assert len(files) == 1
    assert files[0].name == "doc.md"


def test_recursive_scan(md_dir):
    paths = [str(f) for f in scan_markdown_files(md_dir)]
    assert any("notes" in p for p in paths)


def test_empty_directory(tmp_path):
    assert list(scan_markdown_files(tmp_path)) == []


def test_returns_path_objects(md_dir):
    from pathlib import Path

    files = list(scan_markdown_files(md_dir))
    assert all(isinstance(f, Path) for f in files)


def test_only_files_not_dirs(tmp_path):
    # A directory named 'tricky.md' should not be yielded.
    (tmp_path / "tricky.md").mkdir()
    (tmp_path / "real.md").write_text("# Real\nContent.")
    files = list(scan_markdown_files(tmp_path))
    assert len(files) == 1
    assert files[0].name == "real.md"
