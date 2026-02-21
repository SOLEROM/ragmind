"""Tests for kb.generator."""

from kb.chunker import Chunk
from kb.generator import build_prompt, generate_answer
from tests.mocks import MockLLM


def _chunk(content: str, path: str = "doc.md", section=None) -> Chunk:
    return Chunk(content=content, file_path=path, section_path=section or [])


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


def test_prompt_contains_question():
    chunks = [_chunk("Some context.")]
    prompt = build_prompt("What is this about?", chunks)
    assert "What is this about?" in prompt


def test_prompt_contains_chunk_content():
    chunks = [_chunk("The answer is forty-two.")]
    prompt = build_prompt("What is the answer?", chunks)
    assert "The answer is forty-two." in prompt


def test_prompt_contains_source_file():
    chunks = [_chunk("Data.", "important.md", ["# Results"])]
    prompt = build_prompt("question", chunks)
    assert "important.md" in prompt


def test_prompt_contains_section():
    chunks = [_chunk("Data.", "doc.md", ["# Title", "## Sub"])]
    prompt = build_prompt("question", chunks)
    assert "Title" in prompt
    assert "Sub" in prompt


def test_prompt_includes_all_chunks():
    chunks = [
        _chunk("First piece of information.", "a.md", ["# A"]),
        _chunk("Second piece of information.", "b.md", ["# B"]),
    ]
    prompt = build_prompt("question", chunks)
    assert "First piece" in prompt
    assert "Second piece" in prompt
    assert "a.md" in prompt
    assert "b.md" in prompt


def test_prompt_has_answer_marker():
    prompt = build_prompt("q?", [_chunk("ctx")])
    assert "Answer:" in prompt


# ---------------------------------------------------------------------------
# generate_answer
# ---------------------------------------------------------------------------


def test_generate_uses_llm_response():
    llm = MockLLM("The answer is here.")
    chunks = [_chunk("Some context data.")]
    answer = generate_answer("question", chunks, llm)
    assert answer == "The answer is here."


def test_generate_empty_chunks():
    llm = MockLLM("Should not be called.")
    answer = generate_answer("question", [], llm)
    assert "No relevant information" in answer


def test_generate_passes_all_chunks_to_prompt():
    """The prompt seen by the LLM must contain content from every chunk."""
    received: list[str] = []

    class CaptureLLM(MockLLM):
        def generate(self, prompt: str) -> str:
            received.append(prompt)
            return "ok"

    chunks = [_chunk("Alpha text."), _chunk("Beta text.")]
    generate_answer("q?", chunks, CaptureLLM())
    assert received, "LLM.generate was not called"
    assert "Alpha text." in received[0]
    assert "Beta text." in received[0]
