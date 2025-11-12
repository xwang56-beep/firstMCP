import numpy as np

from src.retrieve import cosine_similarity, retrieve
from src.store import VectorRecord, VectorStore


def test_cosine_similarity_identifies_best_match():
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    query = np.array([1.0, 0.5])
    scores = cosine_similarity(query, vectors)
    assert np.argmax(scores) == 2


def test_retrieve_returns_sorted_results():
    store = VectorStore()
    store.extend(
        [
            VectorRecord(vector=[1.0, 0.0], metadata={"id": "a"}),
            VectorRecord(vector=[0.0, 1.0], metadata={"id": "b"}),
            VectorRecord(vector=[1.0, 1.0], metadata={"id": "c"}),
        ]
    )
    query = [1.0, 0.5]
    results = retrieve(query, store, top_k=2)
    assert len(results) == 2
    assert results[0].metadata["id"] == "c"

