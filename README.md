# kb — Local Offline Knowledge Assistant

A command-line tool that turns a folder of markdown files into a searchable,
AI-powered knowledge base — entirely on your own machine, no cloud required.

You ask a question in plain English. The tool finds the most relevant sections
from your notes and feeds them to a local language model that writes a focused
answer with source citations.

---

## How it works

```
Your markdown files
        │
        ▼
  [ Scanner ]  ──  walks directories, finds all .md files
        │
        ▼
  [ Chunker ]  ──  splits each file into sections by header hierarchy
        │
        ▼
  [ Embedder ] ──  converts each chunk to a vector (sentence-transformers)
        │
        ▼
  [ Store ]    ──  saves vectors in a local FAISS index + JSON metadata
                   (incremental: unchanged files are skipped on re-index)

  ── at query time ──

  Your question
        │
        ▼
  [ Embedder ] ──  embeds the question
        │
        ▼
  [ Store ]    ──  nearest-neighbour search → top-k relevant chunks
        │
        ▼
  [ Generator ]──  builds a focused prompt (question + retrieved context)
        │
        ▼
  [ Ollama LLM]──  generates an answer that cites source files & sections
```

All steps run locally. The language model only ever sees the small slice of
your notes that is relevant to the question.

---

## Requirements

| Dependency | Purpose |
|---|---|
| Python ≥ 3.10 | runtime |
| [sentence-transformers](https://www.sbert.net/) | local text embeddings |
| [faiss-cpu](https://github.com/facebookresearch/faiss) | vector search |
| [Ollama](https://ollama.com/) | local language model server |
| click, numpy | CLI + numerics |

---

## Installation

```bash
# 1. Clone or enter the project directory
cd kb

# 2. Install the package (adds the `kb` command to your PATH)
pip install -e .

# 3. Pull a model into Ollama (one-time, ~4 GB)
ollama pull mistral
```

> The first time you run `kb index` the embedding model
> (`all-MiniLM-L6-v2`, ~80 MB) is downloaded automatically by
> sentence-transformers and cached locally.

---

## Quick start

```bash
# Index your notes folder
kb index ~/notes

# Ask a question
kb query "how do I set up SSH key authentication?"

# See what is in the index
kb status
```

---

## Commands

### `kb index <directory>`

Scans `<directory>` recursively for `.md` files and indexes any that are new
or have changed since the last run.

```
kb index ~/notes
kb index ~/notes --force          # re-index everything, ignore mtimes
kb index ~/notes --embed-model sentence-transformers/all-mpnet-base-v2
```

| Option | Default | Description |
|---|---|---|
| `--embed-model` | `all-MiniLM-L6-v2` | Sentence-Transformers model |
| `--force` | off | Re-index all files regardless of modification time |

---

### `kb query "<question>"`

Retrieves the most relevant chunks from the index and generates an answer.

```
kb query "what is the project deadline?"
kb query "explain the deployment process" --top-k 8
kb query "authentication flow" --no-llm        # show raw chunks, skip LLM
kb query "how to reset password" --llm-model llama3
```

| Option | Default | Description |
|---|---|---|
| `--top-k` | `5` | Number of chunks to retrieve |
| `--embed-model` | `all-MiniLM-L6-v2` | Must match the model used during index |
| `--llm-model` | `mistral` | Ollama model for answer generation |
| `--no-llm` | off | Print retrieved chunks without calling the LLM |

Example output:

```
Question: how do I configure the database connection?

Answer:
According to the configuration guide, the database connection is set
in `~/.config/myapp/config.yaml` under the `database` key. You must
provide `host`, `port`, and `credentials_file` fields. [Source:
docs/setup.md > # Setup > ## Database]

Sources:
  - docs/setup.md > # Setup > ## Database
  - docs/setup.md > # Setup > ## Environment Variables
  - notes/ops.md > # Operations > ## Troubleshooting
```

---

### `kb status`

Shows the current state of the index.

```
kb status
```

```
Store : /home/user/.kb/store
Files : 42
Chunks: 318

Indexed files:
  /home/user/notes/project.md
  /home/user/notes/ops.md
  ...
```

---

## Store location

By default the index is stored in `~/.kb/store/`. You can override this
per-command or via an environment variable:

```bash
# Per-command
kb --store /data/my-kb index ~/notes
kb --store /data/my-kb query "..."

# Via environment variable
export KB_STORE=/data/my-kb
kb index ~/notes
kb query "..."
```

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

The test suite uses lightweight stubs (`MockEmbedder`, `MockLLM`) so it runs
in under a second without a GPU, without Ollama, and without downloading any
model weights.

```
68 passed in 0.19s
```

---

## Project structure

```
kb/
├── kb/
│   ├── scanner.py      # find .md files
│   ├── chunker.py      # split by markdown headers → Chunk objects
│   ├── embedder.py     # BaseEmbedder + SentenceTransformerEmbedder
│   ├── store.py        # FAISS vector store + JSON metadata + save/load
│   ├── retriever.py    # embed query, search store, return top-k chunks
│   ├── generator.py    # build RAG prompt, call LLM, return answer
│   └── cli.py          # Click CLI (index / query / status)
└── tests/
    ├── mocks.py        # MockEmbedder (deterministic) + MockLLM
    ├── conftest.py     # shared fixtures
    ├── test_scanner.py
    ├── test_chunker.py
    ├── test_embedder.py
    ├── test_store.py
    ├── test_retriever.py
    ├── test_generator.py
    └── test_cli.py
```

Each layer (scan → chunk → embed → store → retrieve → generate) is an
independent module. You can swap the embedding model or the LLM backend
without touching any other part of the system.
