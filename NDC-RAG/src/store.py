from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from .embed import Embedding


@dataclass
class VectorRecord:
    vector: List[float]
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """Simple in-memory vector store with JSON persistence."""

    def __init__(self, records: Sequence[VectorRecord] | None = None) -> None:
        self._vectors = np.array([r.vector for r in records] if records else [])
        self._metadata = [r.metadata for r in records] if records else []

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors

    @property
    def metadata(self) -> List[dict]:
        return self._metadata

    def __len__(self) -> int:
        return len(self._metadata)

    def add(self, record: VectorRecord) -> None:
        vector_array = np.array(record.vector, dtype=float)
        if self._vectors.size == 0:
            self._vectors = vector_array.reshape(1, -1)
        else:
            self._vectors = np.vstack([self._vectors, vector_array])
        self._metadata.append(record.metadata)

    def extend(self, records: Iterable[VectorRecord]) -> None:
        for record in records:
            self.add(record)

    def save(self, path: str | Path) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"vector": metadata["vector"], "metadata": metadata["metadata"]}
            for metadata in (
                {"vector": vec.tolist(), "metadata": meta}
                for vec, meta in zip(self._vectors, self._metadata)
            )
        ]
        path_obj.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Vector store file does not exist: {path}")
        data = json.loads(path_obj.read_text())
        records = [
            VectorRecord(vector=item["vector"], metadata=item["metadata"])
            for item in data
        ]
        return cls(records)

    @classmethod
    def from_embeddings(cls, embeddings: Sequence[Embedding]) -> "VectorStore":
        records = [
            VectorRecord(
                vector=embedding.vector,
                metadata={
                    "content": embedding.chunk.content,
                    "chunk_index": embedding.chunk.index,
                    "document_path": embedding.chunk.document.metadata.get(
                        "relative_path"
                    ),
                    "source_path": str(embedding.chunk.document.source_path),
                    **{
                        k: v
                        for k, v in embedding.chunk.metadata.items()
                        if k not in {"chunk_index", "total_chunks"}
                    },
                    "total_chunks": embedding.chunk.metadata.get("total_chunks"),
                },
            )
            for embedding in embeddings
        ]
        return cls(records)


__all__ = ["VectorRecord", "VectorStore"]

