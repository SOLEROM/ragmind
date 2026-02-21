"""Prompt construction and LLM generation.

The ``BaseLLM`` interface lets tests inject a lightweight mock without
requiring a running Ollama instance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .chunker import Chunk


class BaseLLM(ABC):
    """Common interface for language model backends."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the model's response for *prompt*."""


class OllamaLLM(BaseLLM):
    """Calls a locally running Ollama model."""

    def __init__(self, model: str = "mistral") -> None:
        import ollama  # type: ignore  # lazy import

        self._client = ollama.Client()
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.generate(model=self._model, prompt=prompt)
        return response.response  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a knowledge assistant. Answer the question using ONLY the "
    "provided context. If the context does not contain enough information "
    "say so clearly. Always cite the source file and section when you "
    "refer to specific information."
)


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    """Assemble the RAG prompt from *question* and retrieved *chunks*."""
    parts: list[str] = []
    for chunk in chunks:
        source = chunk.file_path
        if chunk.section_path:
            source += " > " + " > ".join(chunk.section_path)
        parts.append(f"[Source: {source}]\n{chunk.content}")

    context = "\n\n---\n\n".join(parts)
    return (
        f"{_SYSTEM}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def generate_answer(question: str, chunks: list[Chunk], llm: BaseLLM) -> str:
    """Generate an answer for *question* from *chunks* using *llm*."""
    if not chunks:
        return "No relevant information found in the knowledge base."
    prompt = build_prompt(question, chunks)
    return llm.generate(prompt)
