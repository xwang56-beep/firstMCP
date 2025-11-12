from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .retrieve import RetrievalResult


@dataclass
class EvaluationResult:
    precision_at_k: float
    retrieved: Sequence[RetrievalResult]


def evaluate_precision_at_k(
    retrieved: Sequence[RetrievalResult],
    expected_sources: Sequence[str],
    *,
    k: int | None = None,
) -> EvaluationResult:
    """Compute a basic precision@k based on expected source identifiers."""
    if k is None:
        k = len(retrieved)
    k = max(1, min(k, len(retrieved)))

    normalized_expected = {item.lower() for item in expected_sources}
    hits = 0
    for result in retrieved[:k]:
        source = str(result.metadata.get("document_path", "")).lower()
        if source in normalized_expected:
            hits += 1

    precision = hits / k if k else 0.0
    return EvaluationResult(precision_at_k=precision, retrieved=retrieved)


__all__ = ["EvaluationResult", "evaluate_precision_at_k"]

