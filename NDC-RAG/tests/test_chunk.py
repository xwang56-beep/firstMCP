from pathlib import Path

from src.chunk import chunk_documents, chunk_texts, split_text
from src.ingest import Document


def test_split_text_produces_expected_number_of_chunks():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = split_text(text, chunk_size=20, overlap=5)
    assert len(chunks) == 7
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_chunk_documents_preserves_metadata():
    document = Document(content=" ".join("hello" for _ in range(30)), source_path=Path("doc.txt"), metadata={"relative_path": "doc.txt"})
    chunks = chunk_documents([document], chunk_size=10, overlap=2)
    assert len(chunks) == 4
    assert all(chunk.document is document for chunk in chunks)
    assert all(chunk.metadata["relative_path"] == "doc.txt" for chunk in chunks)


def test_chunk_texts_matches_split_text_output():
    text = " ".join(f"token{i}" for i in range(15))
    expected = split_text(text, chunk_size=5, overlap=2)
    actual = chunk_texts([text], chunk_size=5, overlap=2)
    assert actual == expected

