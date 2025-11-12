from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .store import VectorStore


@dataclass
class RetrievalResult:
    score: float
    metadata: dict


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero; treat zero vectors as zero after normalization
    norms[norms == 0] = 1.0
    return vectors / norms


def cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        raise ValueError("Query vector must not be the zero vector.")
    normalized_query = query / query_norm
    normalized_vectors = normalize_vectors(vectors)
    return normalized_vectors @ normalized_query


def retrieve(
    query_vector: Sequence[float],
    store: VectorStore,
    *,
    top_k: int = 5,
) -> List[RetrievalResult]:
    if len(store) == 0:
        return []

    vectors = store.vectors
    scores = cosine_similarity(np.array(query_vector, dtype=float), vectors)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results: List[RetrievalResult] = []
    for index in top_indices:
        results.append(
            RetrievalResult(
                score=float(scores[index]),
                metadata=store.metadata[index],
            )
        )
    return results


def rerank(results: Iterable[RetrievalResult]) -> List[RetrievalResult]:
    """Placeholder rerank function."""
    return list(results)


__all__ = ["RetrievalResult", "cosine_similarity", "normalize_vectors", "retrieve", "rerank"]

