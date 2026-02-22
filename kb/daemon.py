"""Background daemon that keeps the embedder warm for fast queries.

Start with::

    kb serve                          # foreground, Ctrl-C to stop
    kb serve --embed-model <model>    # non-default model
    kb --store /path serve            # non-default store

The daemon listens on a Unix-domain socket at ``<store>/daemon.sock``.
``kb query`` checks for a live socket first and, when found, delegates the
entire embed + search + (optional) LLM step to the daemon process.  If no
daemon is running the command falls back to the normal in-process path.

Protocol
--------
Both directions are newline-terminated JSON objects.

Request fields
~~~~~~~~~~~~~~
* ``question``  (str, required)
* ``top_k``     (int, default 5)
* ``no_llm``    (bool, default false)
* ``llm_model`` (str, default "mistral")

Response fields
~~~~~~~~~~~~~~~
* ``error``   — ``null`` on success, short string on failure
* ``chunks``  — list of chunk dicts (``Chunk.to_dict()``)
* ``answer``  — generated string, or ``null`` when ``no_llm`` is true
"""

from __future__ import annotations

import json
import os
import signal
import socket
import threading
from pathlib import Path

from .embedder import SentenceTransformerEmbedder
from .generator import OllamaLLM, generate_answer
from .retriever import retrieve
from .store import KnowledgeStore

_SOCKET_NAME = "daemon.sock"
_PID_NAME = "daemon.pid"


# ---------------------------------------------------------------------------
# Client-side helpers (used by cli.py)
# ---------------------------------------------------------------------------


def socket_path(store_dir: Path) -> Path:
    return store_dir / _SOCKET_NAME


def is_running(store_dir: Path) -> bool:
    """Return ``True`` if a daemon is accepting connections on the socket."""
    sock = socket_path(store_dir)
    if not sock.exists():
        return False
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect(str(sock))
        s.close()
        return True
    except OSError:
        # Stale socket — clean it up.
        try:
            sock.unlink()
        except OSError:
            pass
        return False


def send_request(
    store_dir: Path, request: dict, timeout: float = 120.0
) -> dict | None:
    """Send a JSON request to the daemon; return the response or ``None``."""
    sock_path = socket_path(store_dir)
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(str(sock_path))
        s.sendall(json.dumps(request).encode() + b"\n")

        buf = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
            if buf.endswith(b"\n"):
                break
        s.close()
        return json.loads(buf.decode())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Server-side
# ---------------------------------------------------------------------------


class _Handler(threading.Thread):
    """Handles one client connection in its own thread."""

    def __init__(self, conn: socket.socket, server: "DaemonServer") -> None:
        super().__init__(daemon=True)
        self._conn = conn
        self._server = server

    def run(self) -> None:
        try:
            buf = b""
            while True:
                data = self._conn.recv(65536)
                if not data:
                    break
                buf += data
                if buf.endswith(b"\n"):
                    break
            request = json.loads(buf.decode())
            response = self._server.handle(request)
            self._conn.sendall(json.dumps(response).encode() + b"\n")
        except Exception:
            pass
        finally:
            self._conn.close()


class DaemonServer:
    """Keeps the embedder in memory; serves JSON query requests over a socket."""

    def __init__(
        self,
        store_dir: Path,
        embed_model: str,
        embedding_dim: int,
    ) -> None:
        self.store_dir = store_dir
        self.embed_model = embed_model
        self.embedding_dim = embedding_dim
        self._embedder = SentenceTransformerEmbedder(embed_model)
        self._store: KnowledgeStore | None = None
        self._store_mtime: float = -1.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Store management — auto-reload when the index file changes
    # ------------------------------------------------------------------

    def _get_store(self) -> KnowledgeStore:
        index_file = self.store_dir / "index.faiss"
        try:
            mtime = index_file.stat().st_mtime
        except FileNotFoundError:
            mtime = -1.0

        if self._store is None or mtime != self._store_mtime:
            store = KnowledgeStore(self.store_dir, embedding_dim=self.embedding_dim)
            store.load()
            self._store = store
            self._store_mtime = mtime

        return self._store

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    def handle(self, request: dict) -> dict:
        try:
            question: str = request["question"]
            top_k: int = request.get("top_k", 5)
            no_llm: bool = request.get("no_llm", False)
            llm_model: str = request.get("llm_model", "mistral")

            with self._lock:
                store = self._get_store()
                if store.total_chunks == 0:
                    return {"error": "empty", "chunks": [], "answer": None}
                chunks = retrieve(question, store, self._embedder, top_k=top_k)

            if not chunks:
                return {"error": None, "chunks": [], "answer": None}

            chunk_dicts = [c.to_dict() for c in chunks]

            if no_llm:
                return {"error": None, "chunks": chunk_dicts, "answer": None}

            llm = OllamaLLM(llm_model)
            answer = generate_answer(question, chunks, llm)
            return {"error": None, "chunks": chunk_dicts, "answer": answer}

        except Exception as exc:
            return {"error": str(exc), "chunks": [], "answer": None}

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        sock_path = socket_path(self.store_dir)

        # Remove any stale socket file from a previous (crashed) daemon.
        try:
            sock_path.unlink()
        except FileNotFoundError:
            pass

        # Record PID so the user can kill the daemon if needed.
        (self.store_dir / _PID_NAME).write_text(str(os.getpid()))

        # Warm the embedder *now* — this is the whole point of the daemon.
        self._embedder._load()

        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(str(sock_path))
        server_sock.listen(8)

        def _shutdown(sig, frame):  # noqa: ARG001
            server_sock.close()
            for name in (_SOCKET_NAME, _PID_NAME):
                try:
                    (self.store_dir / name).unlink()
                except OSError:
                    pass
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        print(f"kb daemon ready  (pid {os.getpid()})", flush=True)
        print(f"  socket : {sock_path}", flush=True)
        print(f"  model  : {self.embed_model}", flush=True)
        print(f"  store  : {self.store_dir}", flush=True)
        print("Press Ctrl-C to stop.", flush=True)

        while True:
            try:
                conn, _ = server_sock.accept()
            except OSError:
                break
            _Handler(conn, self).start()
