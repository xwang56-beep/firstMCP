from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class Document:
    """Simple container for an ingested document."""

    content: str
    source_path: Path
    metadata: dict


SUPPORTED_EXTENSIONS = {".txt"}


def iter_source_files(data_dir: Path) -> Iterable[Path]:
    """Yield supported document files from ``data_dir``."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def load_documents(data_dir: str | Path) -> List[Document]:
    """Load documents from the given directory into memory.

    Parameters
    ----------
    data_dir:
        Directory containing source documents. Only supported extensions are ingested.

    Returns
    -------
    list[Document]
        Documents ready for downstream processing.
    """
    directory = Path(data_dir).expanduser().resolve()
    documents: List[Document] = []

    for file_path in iter_source_files(directory):
        content = file_path.read_text(encoding="utf-8")
        documents.append(
            Document(
                content=content,
                source_path=file_path,
                metadata={
                    "relative_path": file_path.relative_to(directory).as_posix(),
                    "extension": file_path.suffix.lower(),
                    "size_bytes": file_path.stat().st_size,
                },
            )
        )

    return documents


__all__ = ["Document", "load_documents", "iter_source_files", "SUPPORTED_EXTENSIONS"]

