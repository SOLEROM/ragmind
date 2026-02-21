"""Split markdown files into logical chunks for indexing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Matches ATX-style headers: # … through ###### …
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class Chunk:
    """A text chunk with source provenance."""

    content: str
    file_path: str
    section_path: list[str]  # breadcrumb of header lines, e.g. ["# Title", "## Sub"]
    chunk_id: int = field(default=-1)  # assigned by KnowledgeStore

    @property
    def display_source(self) -> str:
        parts = [self.file_path] + self.section_path
        return " > ".join(parts)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "file_path": self.file_path,
            "section_path": self.section_path,
            "chunk_id": self.chunk_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            content=d["content"],
            file_path=d["file_path"],
            section_path=d["section_path"],
            chunk_id=d["chunk_id"],
        )


def chunk_text(text: str, file_path: str, min_chars: int = 50) -> list[Chunk]:
    """Split *text* into chunks by markdown headers.

    Each chunk spans from one header to the start of the next (or EOF).
    The ``section_path`` records the breadcrumb of enclosing headers so
    retrieval results are easy to trace back.
    """
    chunks: list[Chunk] = []
    header_matches = list(_HEADER_RE.finditer(text))

    if not header_matches:
        # No headers — treat the whole file as one chunk.
        content = text.strip()
        if len(content) >= min_chars:
            chunks.append(Chunk(content=content, file_path=file_path, section_path=[]))
        return chunks

    # Collect boundary positions: start of each header + sentinel at EOF.
    boundaries = [m.start() for m in header_matches] + [len(text)]

    # Content before the first header (e.g. front-matter, intro paragraph).
    pre = text[: boundaries[0]].strip()
    if len(pre) >= min_chars:
        chunks.append(Chunk(content=pre, file_path=file_path, section_path=[]))

    # Stack tracks (level, header_line) for building the breadcrumb path.
    stack: list[tuple[int, str]] = []

    for i, match in enumerate(header_matches):
        level = len(match.group(1))  # number of '#' characters
        header_line = match.group(0).strip()

        # Pop headers at the same or deeper level to maintain hierarchy.
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, header_line))

        section_path = [h for _, h in stack]
        content = text[boundaries[i] : boundaries[i + 1]].strip()

        if len(content) >= min_chars:
            chunks.append(
                Chunk(content=content, file_path=file_path, section_path=section_path)
            )

    return chunks


def chunk_markdown(file_path: str | Path, min_chars: int = 50) -> list[Chunk]:
    """Read *file_path* and return its chunks."""
    file_path = Path(file_path)
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return chunk_text(text, str(file_path), min_chars)
