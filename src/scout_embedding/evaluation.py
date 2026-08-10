"""Dependency-light dense retrieval evaluation."""

from __future__ import annotations

import math

import numpy as np


def _dcg(relevances: list[int]) -> float:
    return sum((2**rel - 1) / math.log2(index + 2) for index, rel in enumerate(relevances))


def evaluate_ndcg(
    queries: np.ndarray,
    documents: np.ndarray,
    query_ids: list[str],
    document_ids: list[str],
    qrels: dict[str, dict[str, int]],
    cutoffs: tuple[int, ...] = (1, 10),
) -> dict[str, float]:
    similarities = queries @ documents.T
    results = {f"ndcg@{cutoff}": [] for cutoff in cutoffs}
    for row, query_id in enumerate(query_ids):
        relevance = qrels.get(query_id, {})
        order = np.argsort(-similarities[row])
        for cutoff in cutoffs:
            observed = [relevance.get(document_ids[index], 0) for index in order[:cutoff]]
            ideal = sorted(relevance.values(), reverse=True)[:cutoff]
            denominator = _dcg(ideal)
            results[f"ndcg@{cutoff}"].append(
                _dcg(observed) / denominator if denominator else 0.0
            )
    return {key: float(np.mean(values)) for key, values in results.items()}
