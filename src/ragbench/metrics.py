"""RAG evaluation metrics."""

import numpy as np
from typing import List


def precision_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    """Compute Precision@K."""
    if not retrieved or k <= 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for doc in retrieved_k if doc in relevant_set)
    return hits / k


def recall_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    """Compute Recall@K."""
    if not relevant:
        return 0.0
    if not retrieved or k <= 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for doc in retrieved_k if doc in relevant_set)
    return hits / len(relevant)


def mrr(relevant: List[str], retrieved: List[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    if not relevant or not retrieved:
        return 0.0
    relevant_set = set(relevant)
    for i, doc in enumerate(retrieved):
        if doc in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(relevant: List[str], retrieved: List[str], k: int,
              relevance_scores: dict = None) -> float:
    """Compute Normalized Discounted Cumulative Gain at K."""
    if not retrieved or k <= 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    scores = relevance_scores or {d: 1.0 for d in relevant_set}

    dcg = 0.0
    for i, doc in enumerate(retrieved_k):
        rel = scores.get(doc, 0.0)
        dcg += rel / np.log2(i + 2)

    ideal_scores = sorted([scores.get(d, 0.0) for d in relevant_set], reverse=True)
    ideal_dcg = 0.0
    for i, rel in enumerate(ideal_scores[:k]):
        ideal_dcg += rel / np.log2(i + 2)

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def hit_rate(relevant: List[str], retrieved: List[str], k: int) -> float:
    """Compute Hit Rate at K — 1 if any relevant doc in top K, else 0."""
    if not relevant or not retrieved:
        return 0.0
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    hits = retrieved_k & relevant_set
    return 1.0 if hits else 0.0


def average_precision(relevant: List[str], retrieved: List[str]) -> float:
    """Compute Average Precision."""
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = 0
    sum_precisions = 0.0
    for i, doc in enumerate(retrieved):
        if doc in relevant_set:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(relevant)
