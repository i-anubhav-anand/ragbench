# ragbench

A lightweight CLI tool for benchmarking RAG (Retrieval-Augmented Generation) pipelines.

Built on June 12, 2026.

## Features

- Measure retrieval accuracy (MRR, Recall@K, NDCG)
- Evaluate generation quality against ground truth
- Compare multiple RAG configurations side by side
- CSV and JSON output formats
- Built-in synthetic test data generator

## Quick Start

```bash
pip install ragbench
ragbench eval --config config.yaml --dataset my-dataset.jsonl
```

## Development

```bash
uv pip install -e ".[dev]"
pytest
```
