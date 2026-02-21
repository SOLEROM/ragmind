"""Local vector store backed by FAISS + JSON metadata.

Design
------
* ``IndexFlatL2`` stores embedding vectors sequentially.  Each position *i*
  in the FAISS flat array corresponds exactly to ``_chunks[i]``.
* Deletion is logical: the Python slot is set to ``None`` while the FAISS
  vector stays in place.  Search results whose slot is ``None`` are skipped.
* ``save`` / ``load`` persist both the FAISS binary and the JSON metadata so
  the index survives process restarts.
* ``needs_update`` compares stored file mtimes to decide whether to re-index.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss  # type: ignore
import numpy as np

from .chunker import Chunk


@dataclass
class _FileRecord:
    file_path: str
    mtime: float
    positions: list[int]  # indices into _chunks / FAISS flat array


class KnowledgeStore:
    """Manages the vector index and chunk metadata for the knowledge base."""

    def __init__(self, store_dir: str | Path, embedding_dim: int = 384) -> None:
        self.store_dir = Path(store_dir)
        self.embedding_dim = embedding_dim
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self._chunks: list[Chunk | None] = []
        self._files: dict[str, _FileRecord] = {}
        self._index: faiss.Index = faiss.IndexFlatL2(embedding_dim)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def _index_path(self) -> Path:
        return self.store_dir / "index.faiss"

    @property
    def _meta_path(self) -> Path:
        return self.store_dir / "meta.json"

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_file(
        self,
        file_path: str,
        mtime: float,
        chunks: list[Chunk],
        vectors: np.ndarray,
    ) -> None:
        """Index (or re-index) *file_path* with its chunks and vectors."""
        if file_path in self._files:
            self._remove_file_internal(file_path)

        if not chunks:
            return

        start = len(self._chunks)
        positions = list(range(start, start + len(chunks)))

        for pos, chunk in zip(positions, chunks):
            chunk.chunk_id = pos
            self._chunks.append(chunk)

        self._files[file_path] = _FileRecord(
            file_path=file_path, mtime=mtime, positions=positions
        )
        self._index.add(vectors.astype(np.float32))

    def remove_file(self, file_path: str) -> None:
        """Remove all chunks belonging to *file_path* from the store."""
        self._remove_file_internal(file_path)

    def _remove_file_internal(self, file_path: str) -> None:
        record = self._files.pop(file_path, None)
        if record is None:
            return
        for pos in record.positions:
            if pos < len(self._chunks):
                self._chunks[pos] = None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """Return up to *top_k* (chunk, distance) pairs, nearest first."""
        if self._index.ntotal == 0:
            return []

        # Fetch extra candidates to compensate for logically-deleted slots.
        k = min(top_k * 5, self._index.ntotal)
        q = query_vector.astype(np.float32).reshape(1, -1)
        distances, indices = self._index.search(q, k)

        results: list[tuple[Chunk, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            if chunk is not None:
                results.append((chunk, float(dist)))
            if len(results) >= top_k:
                break
        return results

    def needs_update(self, file_path: str, mtime: float) -> bool:
        """Return ``True`` if the file is new or its mtime has changed."""
        record = self._files.get(file_path)
        return record is None or record.mtime < mtime

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write the FAISS index and JSON metadata to disk."""
        faiss.write_index(self._index, str(self._index_path))
        meta = {
            "embedding_dim": self.embedding_dim,
            "chunks": [
                c.to_dict() if c is not None else None for c in self._chunks
            ],
            "files": {
                k: {
                    "file_path": v.file_path,
                    "mtime": v.mtime,
                    "positions": v.positions,
                }
                for k, v in self._files.items()
            },
        }
        self._meta_path.write_text(json.dumps(meta, indent=2))

    def load(self) -> None:
        """Load the FAISS index and JSON metadata from disk (if present)."""
        if not self._meta_path.exists():
            return  # fresh store, nothing to restore

        meta = json.loads(self._meta_path.read_text())
        stored_dim: int = meta.get("embedding_dim", self.embedding_dim)
        if stored_dim != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: stored={stored_dim}, "
                f"requested={self.embedding_dim}.  "
                f"Delete the store or re-index with the same model."
            )

        if self._index_path.exists():
            self._index = faiss.read_index(str(self._index_path))
        else:
            self._index = faiss.IndexFlatL2(self.embedding_dim)

        self._chunks = [
            Chunk.from_dict(c) if c is not None else None
            for c in meta.get("chunks", [])
        ]
        self._files = {
            k: _FileRecord(
                file_path=v["file_path"],
                mtime=v["mtime"],
                positions=v["positions"],
            )
            for k, v in meta.get("files", {}).items()
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def total_chunks(self) -> int:
        return sum(1 for c in self._chunks if c is not None)

    @property
    def total_files(self) -> int:
        return len(self._files)

    def indexed_files(self) -> list[str]:
        return list(self._files.keys())
