"""Command-line interface for the kb knowledge assistant.

Commands
--------
  kb index <directory>   Scan and index markdown files.
  kb query <question>    Ask a question against the knowledge base.
  kb status              Show index statistics.
"""

from __future__ import annotations

from pathlib import Path

import click

from .chunker import chunk_markdown
from .embedder import SentenceTransformerEmbedder
from .generator import OllamaLLM, generate_answer
from .retriever import retrieve
from .scanner import scan_markdown_files
from .store import KnowledgeStore

DEFAULT_STORE = Path.home() / ".kb" / "store"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "mistral"


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.option(
    "--store",
    default=None,
    envvar="KB_STORE",
    help="Path to the store directory (default: ~/.kb/store).",
)
@click.pass_context
def cli(ctx: click.Context, store: str | None) -> None:
    """kb — local offline knowledge assistant for markdown files."""
    ctx.ensure_object(dict)
    ctx.obj["store_dir"] = store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(ctx_obj: dict, embedder: SentenceTransformerEmbedder) -> KnowledgeStore:
    raw = ctx_obj.get("store_dir")
    store_path = Path(raw) if raw else DEFAULT_STORE
    store = KnowledgeStore(store_path, embedding_dim=embedder.dim)
    store.load()
    return store


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--embed-model",
    default=DEFAULT_EMBED_MODEL,
    show_default=True,
    help="Sentence-Transformers model name.",
)
@click.option("--force", is_flag=True, help="Re-index even unchanged files.")
@click.pass_context
def index(
    ctx: click.Context, directory: str, embed_model: str, force: bool
) -> None:
    """Scan DIRECTORY and index all markdown files."""
    embedder = SentenceTransformerEmbedder(embed_model)
    store = _make_store(ctx.obj, embedder)

    md_files = list(scan_markdown_files(directory))
    click.echo(f"Found {len(md_files)} markdown file(s) in '{directory}'.")

    indexed = skipped = 0
    for fp in md_files:
        mtime = fp.stat().st_mtime
        if not force and not store.needs_update(str(fp), mtime):
            skipped += 1
            continue

        chunks = chunk_markdown(fp)
        if not chunks:
            click.echo(f"  Skipped (no content): {fp}")
            continue

        texts = [c.content for c in chunks]
        vectors = embedder.embed(texts)
        store.add_file(str(fp), mtime, chunks, vectors)
        indexed += 1
        click.echo(f"  Indexed: {fp}  ({len(chunks)} chunks)")

    store.save()
    click.echo(
        f"\nDone.  Indexed: {indexed}  |  Skipped (unchanged): {skipped}"
    )
    click.echo(
        f"Total in store: {store.total_files} file(s), {store.total_chunks} chunk(s)."
    )


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("question")
@click.option("--top-k", default=5, show_default=True, help="Chunks to retrieve.")
@click.option(
    "--embed-model",
    default=DEFAULT_EMBED_MODEL,
    show_default=True,
    help="Sentence-Transformers model name.",
)
@click.option(
    "--llm-model",
    default=DEFAULT_LLM_MODEL,
    show_default=True,
    help="Ollama model name.",
)
@click.option(
    "--no-llm",
    is_flag=True,
    help="Print retrieved chunks without calling the LLM.",
)
@click.pass_context
def query(
    ctx: click.Context,
    question: str,
    top_k: int,
    embed_model: str,
    llm_model: str,
    no_llm: bool,
) -> None:
    """Ask QUESTION against the knowledge base."""
    embedder = SentenceTransformerEmbedder(embed_model)
    store = _make_store(ctx.obj, embedder)

    if store.total_chunks == 0:
        click.echo(
            "Knowledge base is empty. Run `kb index <directory>` first."
        )
        return

    chunks = retrieve(question, store, embedder, top_k=top_k)

    if not chunks:
        click.echo("No relevant chunks found for that question.")
        return

    if no_llm:
        click.echo(f"\nQuestion: {question}\n")
        click.echo(f"Top {len(chunks)} chunk(s):\n")
        for i, chunk in enumerate(chunks, 1):
            click.echo(f"--- Chunk {i} ---")
            click.echo(f"Source: {chunk.display_source}")
            click.echo(chunk.content[:500])
            click.echo()
        return

    llm = OllamaLLM(llm_model)
    answer = generate_answer(question, chunks, llm)

    click.echo(f"\nQuestion: {question}\n")
    click.echo("Answer:")
    click.echo(answer)
    click.echo("\nSources:")
    for chunk in chunks:
        click.echo(f"  - {chunk.display_source}")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def _stored_embedding_dim(store_path: Path, default: int = 384) -> int:
    """Read the embedding dimension previously persisted in the store."""
    import json

    meta = store_path / "meta.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text()).get("embedding_dim", default)
        except Exception:
            pass
    return default


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show knowledge base statistics."""
    store_path_raw = ctx.obj.get("store_dir")
    store_path = Path(store_path_raw) if store_path_raw else DEFAULT_STORE
    dim = _stored_embedding_dim(store_path)
    store = KnowledgeStore(store_path, embedding_dim=dim)
    store.load()

    click.echo(f"Store : {store.store_dir}")
    click.echo(f"Files : {store.total_files}")
    click.echo(f"Chunks: {store.total_chunks}")
    if store.total_files > 0:
        click.echo("\nIndexed files:")
        for f in store.indexed_files():
            click.echo(f"  {f}")
