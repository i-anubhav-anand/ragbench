"""RAG evaluation orchestrator."""

import json
import csv
from typing import Iterator

from .config import EvalConfig
from .dataset import Query, load_jsonl
from .metrics import (
    precision_at_k, recall_at_k, mrr, ndcg_at_k, hit_rate_at_k, average_precision
)


METRIC_FUNCTIONS = {
    "precision_at_5": lambda r, ret: precision_at_k(r, ret, 5),
    "precision_at_10": lambda r, ret: precision_at_k(r, ret, 10),
    "recall_at_5": lambda r, ret: recall_at_k(r, ret, 5),
    "recall_at_10": lambda r, ret: recall_at_k(r, ret, 10),
    "mrr": mrr,
    "ndcg_at_5": lambda r, ret: ndcg_at_k(r, ret, 5),
    "ndcg_at_10": lambda r, ret: ndcg_at_k(r, ret, 10),
    "hit_rate_at_5": lambda r, ret: hit_rate_at_k(r, ret, 5),
    "hit_rate_at_10": lambda r, ret: hit_rate_at_k(r, ret, 10),
    "average_precision": average_precision,
}


def run_evaluation(config: EvalConfig, queries: Iterator[Query]) -> dict:
    """Run a full RAG evaluation benchmark.

    Args:
        config: Evaluation configuration.
        queries: Iterator over Query objects.

    Returns:
        Dictionary with aggregated results.
    """
    results = {
        "config": {
            "retriever": {
                "type": config.retriever.type,
                "model": config.retriever.model,
                "top_k": config.retriever.top_k,
            },
            "generator": {
                "type": config.generator.type,
                "model": config.generator.model,
            },
            "metrics": config.metrics,
        },
        "query_results": [],
        "aggregates": {},
    }

    metric_scores = {m: [] for m in config.metrics}

    for query in queries:
        retrieved = _simulate_retrieval(query, config.retriever.top_k)
        relevant = query["relevant_docs"]

        query_result = {
            "query_id": query["id"],
            "question": query["question"],
            "num_relevant": len(relevant),
            "retrieved": retrieved,
        }

        for metric_name in config.metrics:
            if metric_name in METRIC_FUNCTIONS:
                score = METRIC_FUNCTIONS[metric_name](relevant, retrieved)
                query_result[metric_name] = score
                metric_scores[metric_name].append(score)

        results["query_results"].append(query_result)

    for metric_name in config.metrics:
        scores = metric_scores[metric_name]
        results["aggregates"][metric_name] = {
            "mean": sum(scores) / len(scores) if scores else 0.0,
            "min": min(scores) if scores else 0.0,
            "max": max(scores) if scores else 0.0,
        }

    return results


def _simulate_retrieval(query: Query, top_k: int) -> list[str]:
    """Simulate retrieval for benchmarking purposes.

    In production, this would call the actual retriever.
    """
    return [f"doc_{i}" for i in range(top_k)]


def save_results(results: dict, path: str, fmt: str = "json") -> None:
    """Save evaluation results to file."""
    if fmt == "json":
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
    elif fmt == "csv":
        _save_csv(results, path)


def _save_csv(results: dict, path: str) -> None:
    """Save aggregate results as CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "min", "max"])
        for metric, stats in results["aggregates"].items():
            writer.writerow([
                metric,
                round(stats["mean"], 4),
                round(stats["min"], 4),
                round(stats["max"], 4),
            ])
