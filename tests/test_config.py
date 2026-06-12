"""Tests for configuration parsing."""

import tempfile
from pathlib import Path

from ragbench.config import load_config, EvalConfig, RetrieverConfig, GeneratorConfig


class TestLoadConfig:
    def test_minimal_yaml(self):
        yaml_content = """
retriever:
  type: sparse
  model: bm25
  top_k: 20
generator:
  model: gpt-4o
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)

        try:
            config = load_config(f.name)
            assert config.retriever.type == "sparse"
            assert config.retriever.model == "bm25"
            assert config.retriever.top_k == 20
            assert config.generator.model == "gpt-4o"
        finally:
            Path(f.name).unlink()

    def test_empty_yaml_uses_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")

        try:
            config = load_config(f.name)
            assert isinstance(config, EvalConfig)
            assert config.retriever.type == "dense"
            assert config.generator.type == "openai"
            assert len(config.metrics) == 5
        finally:
            Path(f.name).unlink()

    def test_partial_config(self):
        yaml_content = """
retriever:
  top_k: 50
metrics:
  - precision_at_5
  - mrr
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)

        try:
            config = load_config(f.name)
            assert config.retriever.top_k == 50
            assert config.retriever.type == "dense"
            assert config.metrics == ["precision_at_5", "mrr"]
        finally:
            Path(f.name).unlink()

    def test_output_dir(self):
        yaml_content = "output_dir: /custom/output"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)

        try:
            config = load_config(f.name)
            assert config.output_dir == "/custom/output"
        finally:
            Path(f.name).unlink()
