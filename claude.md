The system is a local knowledge assistant built around markdown files.
The markdown files are the main source of truth.
The language model is only used to read and summarize retrieved content.
A scanner walks through selected folders and finds all markdown files.
Each file is split into logical chunks based on headers and section boundaries.
Chunks keep metadata like file name and section path.
Each chunk is converted into an embedding vector using a local embedding model.
All vectors are stored in a local vector database.
The original text chunks are stored in a local index for retrieval.
When a user asks a question, the question is also converted into an embedding.
The system searches the vector database for the most similar chunks.
Only the top relevant chunks are selected to reduce prompt size.
The system builds a prompt that includes the user question and retrieved context.
The local language model receives only this focused context.
The model generates an answer based mainly on the provided chunks.
The answer includes references to the source files and sections.
If no relevant context is found, the system reports that the answer is missing from the knowledge base.
The indexer supports incremental updates when markdown files change.
The system avoids reprocessing unchanged files for speed.
All processing runs locally without cloud dependency.
The architecture separates indexing, retrieval, and generation into independent modules.
The retrieval layer can be improved later without changing the language model.
The model can be replaced or upgraded without rebuilding the index.
The system is designed to be fast, predictable, and reproducible on normal hardware.
The final result is a small offline assistant that answers using personal markdown knowledge instead of general internet knowledge.
