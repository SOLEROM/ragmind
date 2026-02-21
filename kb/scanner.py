"""Scan directories for markdown files."""

from pathlib import Path
from typing import Iterator


def scan_markdown_files(root: str | Path) -> Iterator[Path]:
    """Recursively yield all .md files under *root*."""
    root = Path(root)
    for path in sorted(root.rglob("*.md")):
        if path.is_file():
            yield path
