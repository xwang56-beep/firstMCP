from __future__ import annotations

import argparse
from typing import Iterable, Optional, Sequence

from .chunk import chunk_documents
from .config import AppConfig, load_config
from .embed import embed_chunks, embed_texts
from .evaluate import evaluate_precision_at_k
from .ingest import load_documents
from .retrieve import rerank, retrieve
from .store import VectorStore


def build_knowledge_base(config: Optional[AppConfig] = None) -> VectorStore:
    """Run ingestion through storage (steps 1-4) and persist the vector store."""
    config = config or load_config()

    documents = load_documents(config.data_dir)
    chunks = chunk_documents(documents)
    embeddings = embed_chunks(
        chunks,
        model=config.embedding_model,
        api_key=config.openai_api_key,
    )
    store = VectorStore.from_embeddings(embeddings)
    store.save(config.vector_store_path)
    return store


def load_or_build_store(config: Optional[AppConfig] = None) -> VectorStore:
    config = config or load_config()
    try:
        return VectorStore.load(config.vector_store_path)
    except FileNotFoundError:
        return build_knowledge_base(config)


def query_knowledge_base(
    query: str,
    *,
    config: Optional[AppConfig] = None,
    store: Optional[VectorStore] = None,
    top_k: int = 5,
    expected_sources: Optional[Sequence[str]] = None,
) -> dict:
    config = config or load_config()
    store = store or load_or_build_store(config)

    query_vector = embed_texts(
        [query],
        model=config.embedding_model,
        api_key=config.openai_api_key,
    )[0]

    retrieval_results = retrieve(query_vector, store, top_k=top_k)
    reranked = rerank(retrieval_results)

    evaluation = None
    if expected_sources:
        evaluation = evaluate_precision_at_k(reranked, expected_sources)

    return {
        "results": reranked,
        "evaluation": evaluation,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Simple RAG pipeline CLI.")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run ingestion through storage and persist the vector store.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to run against the knowledge base.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to retrieve.",
    )

    args = parser.parse_args(argv)
    config = load_config()

    if args.build:
        build_knowledge_base(config)

    if args.query:
        response = query_knowledge_base(
            args.query,
            config=config,
            top_k=args.top_k,
        )
        for result in response["results"]:
            content = result.metadata.get("content", "").strip()
            print(f"Score: {result.score:.4f}")
            print(f"Source: {result.metadata}")
            if content:
                print("Content:")
                print(content)
            print("-" * 40)

        if response["evaluation"]:
            print(
                f"Precision@k: {response['evaluation'].precision_at_k:.2f}"
            )

    if not args.build and not args.query:
        parser.print_help()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

