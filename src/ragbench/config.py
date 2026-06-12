"""Configuration parsing for ragbench."""

import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrieverConfig:
    type: str = "dense"
    model: str = "all-MiniLM-L6-v2"
    top_k: int = 10
    index_path: Optional[str] = None


@dataclass
class GeneratorConfig:
    type: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512


@dataclass
class EvalConfig:
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    metrics: list[str] = field(default_factory=lambda: [
        "precision_at_5", "recall_at_5", "mrr", "ndcg_at_10", "hit_rate_at_5"
    ])
    output_dir: str = "./results"


def load_config(path: str) -> EvalConfig:
    """Load evaluation configuration from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    retriever = RetrieverConfig(**raw.get("retriever", {}))
    generator = GeneratorConfig(**raw.get("generator", {}))
    metrics = raw.get("metrics", EvalConfig.__dataclass_fields__["metrics"].default_factory())
    output_dir = raw.get("output_dir", "./results")

    return EvalConfig(
        retriever=retriever,
        generator=generator,
        metrics=list(metrics),
        output_dir=output_dir,
    )
