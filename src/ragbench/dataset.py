"""Dataset loading and synthetic generation for RAG evaluation."""

import json
import random
from typing import Iterator, TypedDict


class Query(TypedDict):
    id: str
    question: str
    relevant_docs: list[str]
    ground_truth: str


def load_jsonl(path: str) -> Iterator[Query]:
    """Load evaluation dataset from JSONL file.

    Expected format per line:
    {"id": "q1", "question": "...", "relevant_docs": ["d1","d2"], "ground_truth": "..."}
    """
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}")


def generate_synthetic_dataset(num_queries: int, num_docs: int,
                                seed: int = 42) -> list[Query]:
    """Generate synthetic queries and documents for testing RAG pipelines.

    Returns a list of Query dicts with randomly assigned relevant documents.
    """
    random.seed(seed)
    queries = []

    doc_ids = [f"doc_{i}" for i in range(num_docs)]

    templates = [
        "What is the impact of {topic} on {field}?",
        "How does {topic} relate to {field}?",
        "Explain the concept of {topic} in {field}.",
        "Compare {topic} and {field} in terms of performance.",
        "What are the best practices for {topic} in {field}?",
    ]

    topics = [
        "vector search", "embedding models", "chunking strategies",
        "reranking", "hybrid search", "metadata filtering",
        "semantic caching", "query expansion", "contrastive learning",
    ]

    fields = [
        "RAG systems", "information retrieval", "NLP pipelines",
        "search engines", "knowledge bases", "recommendation systems",
    ]

    for i in range(num_queries):
        topic = random.choice(topics)
        field = random.choice(fields)
        template = random.choice(templates)
        question = template.format(topic=topic, field=field)

        num_relevant = random.randint(1, min(5, num_docs))
        relevant = random.sample(doc_ids, num_relevant)

        queries.append({
            "id": f"q_{i:04d}",
            "question": question,
            "relevant_docs": relevant,
            "ground_truth": f"Answer about {topic} in {field}",
        })

    return queries


def save_jsonl(queries: list[Query], path: str) -> None:
    """Save queries to JSONL file."""
    with open(path, "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
