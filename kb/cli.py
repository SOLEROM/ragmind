"""Command-line interface for the kb knowledge assistant.

Commands
--------
  kb index <directory>   Scan and index markdown files.
  kb query <question>    Ask a question against the knowledge base.
  kb status              Show index statistics.
  kb serve               Start a warm-embedder daemon for faster queries.
"""

from __future__ import annotations

from pathlib import Path

import click

from .chunker import Chunk, chunk_markdown
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


def _stored_embedding_dim(store_path: Path) -> int | None:
    """Return the embedding dimension recorded in *store_path*/meta.json.

    Returns ``None`` when no meta file exists (fresh store).
    """
    import json

    meta = store_path / "meta.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text()).get("embedding_dim")
        except Exception:
            pass
    return None


def _make_store(
    ctx_obj: dict, embed_model: str
) -> tuple[KnowledgeStore, SentenceTransformerEmbedder]:
    """Create and load a :class:`KnowledgeStore` and a lazy embedder.

    The embedding dimension is read from the persisted ``meta.json`` when
    available, so the sentence-transformer model is **not** loaded just to
    query the dimension.  The model is loaded only when
    :meth:`~.embedder.SentenceTransformerEmbedder.embed` is first called.
    """
    raw = ctx_obj.get("store_dir")
    store_path = Path(raw) if raw else DEFAULT_STORE

    # Create the embedder object cheaply — model weights are NOT loaded yet.
    embedder = SentenceTransformerEmbedder(embed_model)

    # Try to read the dimension from the persisted store first.
    dim = _stored_embedding_dim(store_path)
    if dim is None:
        # Fresh store: we must know the dimension before building the FAISS
        # index.  This is the one unavoidable model-load during first index.
        dim = embedder.dim

    store = KnowledgeStore(store_path, embedding_dim=dim)
    store.load()
    return store, embedder


def _resolve_store_path(ctx_obj: dict) -> Path:
    raw = ctx_obj.get("store_dir")
    return Path(raw) if raw else DEFAULT_STORE


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
    store, embedder = _make_store(ctx.obj, embed_model)

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


def _try_daemon_query(
    store_path: Path,
    question: str,
    top_k: int,
    embed_model: str,
    llm_model: str,
    no_llm: bool,
) -> dict | None:
    """Attempt to satisfy the query via the warm-embedder daemon.

    Returns the daemon's response dict on success, or ``None`` if no daemon
    is running (caller should fall back to direct mode).
    """
    from .daemon import is_running, send_request

    if not is_running(store_path):
        return None

    return send_request(
        store_path,
        {
            "question": question,
            "top_k": top_k,
            "embed_model": embed_model,
            "no_llm": no_llm,
            "llm_model": llm_model,
        },
    )


def _render_query_result(
    question: str,
    chunks: list[Chunk],
    answer: str | None,
    no_llm: bool,
) -> None:
    """Print the query result (shared by daemon and direct paths)."""
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

    click.echo(f"\nQuestion: {question}\n")
    click.echo("Answer:")
    click.echo(answer)
    click.echo("\nSources:")
    for chunk in chunks:
        click.echo(f"  - {chunk.display_source}")


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
    store_path = _resolve_store_path(ctx.obj)

    # ------------------------------------------------------------------ #
    # Fast path: delegate to the warm-embedder daemon when it is running. #
    # ------------------------------------------------------------------ #
    daemon_resp = _try_daemon_query(
        store_path, question, top_k, embed_model, llm_model, no_llm
    )
    if daemon_resp is not None:
        if daemon_resp.get("error") == "empty":
            click.echo(
                "Knowledge base is empty. Run `kb index <directory>` first."
            )
            return
        if daemon_resp.get("error"):
            # Unexpected daemon error — fall through to direct mode below.
            pass
        else:
            chunks = [Chunk.from_dict(d) for d in daemon_resp.get("chunks", [])]
            _render_query_result(
                question, chunks, daemon_resp.get("answer"), no_llm
            )
            return

    # ------------------------------------------------------------------ #
    # Direct (in-process) path.                                           #
    # ------------------------------------------------------------------ #
    store, embedder = _make_store(ctx.obj, embed_model)

    if store.total_chunks == 0:
        click.echo(
            "Knowledge base is empty. Run `kb index <directory>` first."
        )
        return

    chunks = retrieve(question, store, embedder, top_k=top_k)

    if not no_llm and chunks:
        llm = OllamaLLM(llm_model)
        answer: str | None = generate_answer(question, chunks, llm)
    else:
        answer = None

    _render_query_result(question, chunks, answer, no_llm)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show knowledge base statistics."""
    store_path = _resolve_store_path(ctx.obj)
    dim = _stored_embedding_dim(store_path) or 384
    store = KnowledgeStore(store_path, embedding_dim=dim)
    store.load()

    click.echo(f"Store : {store.store_dir}")
    click.echo(f"Files : {store.total_files}")
    click.echo(f"Chunks: {store.total_chunks}")
    if store.total_files > 0:
        click.echo("\nIndexed files:")
        for f in store.indexed_files():
            click.echo(f"  {f}")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--embed-model",
    default=DEFAULT_EMBED_MODEL,
    show_default=True,
    help="Sentence-Transformers model name (must match the indexed model).",
)
@click.pass_context
def serve(ctx: click.Context, embed_model: str) -> None:
    """Start a daemon that keeps the embedding model warm.

    \b
    The daemon listens on a Unix socket inside the store directory.
    While it is running, ``kb query`` automatically routes requests
    through the daemon, avoiding the 2-5 s model cold-start on every
    invocation.

    Run in a terminal and leave it open, or background it with nohup:

    \b
        kb serve &
        kb query "what is the deployment process?"
    """
    from .daemon import DaemonServer

    store_path = _resolve_store_path(ctx.obj)
    dim = _stored_embedding_dim(store_path) or 384
    server = DaemonServer(store_path, embed_model, dim)
    server.run()
