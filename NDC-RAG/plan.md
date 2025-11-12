## Plan
- **Project structure**: create `data/` for raw sources, `artifacts/` for serialized vector store, `src/` modules `ingest.py`, `chunk.py`, `embed.py`, `store.py`, `retrieve.py`, `evaluate.py`, plus `main.py` orchestrator and `config.py` for paths/keys.
- **Ingestion**: implement `load_documents(data_dir)` supporting `.txt` (extendable to PDF/MD). Return list of `Document` dataclass instances with metadata.
- **Chunking**: add `chunk_documents(docs, chunk_size=500, overlap=50)` using sliding window by word count; preserve metadata.
- **Embedding**: create `embed_chunks(chunks, model="text-embedding-3-small")` wrapper around OpenAI client, batching requests and reading API key from env.
- **Storage**: design `VectorStore` class storing embeddings + metadata in memory with `save(path)`/`load(path)` methods persisting to JSON lists; assemble via `build_knowledge_base()` running steps 1–4.
- **Retrieval**: implement `retrieve(query, store, top_k=5)` computing cosine similarity (NumPy) against stored vectors; include placeholder `rerank(results)` returning input unchanged.
- **Evaluation**: add `evaluate(query, expected_answer, store)` scaffold (e.g., precision@k or context overlap) logging metrics for future refinement.
- **Entrypoints & tests**: expose `build_kb()` (ingest→store) and `answer_query(query)` (retrieve→evaluate) in `main.py`, plus CLI or notebook example and unit tests for chunking and similarity utilities.

