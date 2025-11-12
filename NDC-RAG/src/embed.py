from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, TYPE_CHECKING, Callable, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    from openai import OpenAI as OpenAIType

from .chunk import Chunk


@dataclass
class Embedding:
    """Represents an embedding vector associated with a chunk."""

    chunk: Chunk
    vector: List[float]


_st_cache: dict[str, "SentenceTransformer"] = {}


def get_openai_client(api_key: str | None = None):
    """Instantiate an OpenAI client, pulling the API key from the environment."""
    if OpenAI is None:
        raise RuntimeError(
            "openai package is not installed. Install it to use embedding features."
        )
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError(
            "OpenAI API key is required. Set the OPENAI_API_KEY environment variable."
        )
    return OpenAI(api_key=key)


def get_sentence_transformer(model: str):
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers package is required for local embeddings. "
            "Install it via `pip install sentence-transformers`."
        )
    if model not in _st_cache:
        _st_cache[model] = SentenceTransformer(model)
    return _st_cache[model]


def _embed_with_openai(
    texts: Sequence[str],
    *,
    model: str,
    batch_size: int,
    api_key: str | None,
) -> List[List[float]]:
    client = get_openai_client(api_key=api_key)
    embeddings: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        for data in response.data:
            embeddings.append(data.embedding)

    return embeddings


def _embed_with_sentence_transformer(
    texts: Sequence[str],
    *,
    model: str,
    batch_size: int,
) -> List[List[float]]:
    transformer = get_sentence_transformer(model)
    vectors = transformer.encode(
        list(texts),  # ensure list for compatibility
        batch_size=batch_size,
        convert_to_numpy=False,
        normalize_embeddings=False,
    )
    return [vector.tolist() if hasattr(vector, "tolist") else list(vector) for vector in vectors]


def _select_backend(model: str) -> Callable[..., List[List[float]]]:
    if model.startswith("text-embedding"):
        return lambda texts, batch_size, api_key: _embed_with_openai(
            texts, model=model, batch_size=batch_size, api_key=api_key
        )
    return lambda texts, batch_size, api_key: _embed_with_sentence_transformer(
        texts, model=model, batch_size=batch_size
    )


def embed_texts(
    texts: Sequence[str],
    *,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 100,
    api_key: str | None = None,
) -> List[List[float]]:
    """Generate embeddings for raw texts."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if not texts:
        return []

    backend = _select_backend(model)
    return backend(texts, batch_size=batch_size, api_key=api_key)


def embed_chunks(
    chunks: Sequence[Chunk],
    *,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 100,
    api_key: str | None = None,
) -> List[Embedding]:
    """Generate embeddings for a sequence of chunks."""
    if not chunks:
        return []

    vectors = embed_texts(
        [chunk.content for chunk in chunks],
        model=model,
        batch_size=batch_size,
        api_key=api_key,
    )
    return [Embedding(chunk=chunk, vector=vector) for chunk, vector in zip(chunks, vectors)]


def embed_chunks_stream(
    chunks: Iterable[Chunk],
    *,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 100,
    api_key: str | None = None,
) -> Iterable[Embedding]:
    """Lazy embedding generator for chunk iterables."""
    buffer: List[Chunk] = []
    for chunk in chunks:
        buffer.append(chunk)
        if len(buffer) >= batch_size:
            yield from embed_chunks(
                buffer, model=model, batch_size=batch_size, api_key=api_key
            )
            buffer.clear()

    if buffer:
        yield from embed_chunks(buffer, model=model, batch_size=batch_size, api_key=api_key)


__all__ = [
    "Embedding",
    "embed_chunks",
    "embed_chunks_stream",
    "embed_texts",
    "get_openai_client",
]

