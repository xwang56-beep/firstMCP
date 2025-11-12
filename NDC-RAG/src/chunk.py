from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .ingest import Document


@dataclass
class Chunk:
    """Represents a chunked portion of a document."""

    content: str
    document: Document
    index: int
    metadata: dict


def split_text(
    text: str,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """Split text into chunks with a simple word-based sliding window."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap

    return chunks


def chunk_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """Chunk each document into sliding windows."""
    chunks: List[Chunk] = []

    for document in documents:
        text_chunks = split_text(
            document.content, chunk_size=chunk_size, overlap=overlap
        )
        for index, chunk_text in enumerate(text_chunks):
            chunks.append(
                Chunk(
                    content=chunk_text,
                    document=document,
                    index=index,
                    metadata={
                        **document.metadata,
                        "chunk_index": index,
                        "total_chunks": len(text_chunks),
                    },
                )
            )

    return chunks


def chunk_texts(
    texts: Iterable[str], *, chunk_size: int = 500, overlap: int = 50
) -> List[str]:
    """Convenience helper to chunk raw strings without document objects."""
    result: List[str] = []
    for text in texts:
        result.extend(split_text(text, chunk_size=chunk_size, overlap=overlap))
    return result


__all__ = ["Chunk", "chunk_documents", "chunk_texts", "split_text"]

