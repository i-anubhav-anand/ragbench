"""Tests for dataset loading and generation."""

import json
import tempfile
from pathlib import Path

import pytest
from ragbench.dataset import load_jsonl, generate_synthetic_dataset, save_jsonl


class TestLoadJSONL:
    def test_valid_file(self):
        data = [
            {"id": "q1", "question": "What?", "relevant_docs": ["d1"], "ground_truth": "A1"},
            {"id": "q2", "question": "Why?", "relevant_docs": ["d2"], "ground_truth": "A2"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

        try:
            queries = list(load_jsonl(f.name))
            assert len(queries) == 2
            assert queries[0]["id"] == "q1"
            assert queries[0]["relevant_docs"] == ["d1"]
        finally:
            Path(f.name).unlink()

    def test_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("not json\n")

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                list(load_jsonl(f.name))
        finally:
            Path(f.name).unlink()

    def test_empty_lines_skipped(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('\n{"id": "q1", "question": "X", "relevant_docs": [], "ground_truth": ""}\n\n')

        try:
            queries = list(load_jsonl(f.name))
            assert len(queries) == 1
        finally:
            Path(f.name).unlink()


class TestGenerateSynthetic:
    def test_default_params(self):
        queries = generate_synthetic_dataset(num_queries=10, num_docs=100, seed=42)
        assert len(queries) == 10
        for q in queries:
            assert "id" in q
            assert "question" in q
            assert len(q["relevant_docs"]) >= 1
            assert q["ground_truth"]

    def test_reproducible(self):
        q1 = generate_synthetic_dataset(5, 50, seed=42)
        q2 = generate_synthetic_dataset(5, 50, seed=42)
        assert [q["id"] for q in q1] == [q["id"] for q in q2]
        assert [q["question"] for q in q1] == [q["question"] for q in q2]

    def test_different_seeds_differ(self):
        q1 = generate_synthetic_dataset(5, 50, seed=1)
        q2 = generate_synthetic_dataset(5, 50, seed=2)
        assert [q["question"] for q in q1] != [q["question"] for q in q2]


class TestSaveJSONL:
    def test_roundtrip(self):
        queries = generate_synthetic_dataset(5, 50, seed=42)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            save_jsonl(queries, f.name)

        try:
            loaded = list(load_jsonl(f.name))
            assert len(loaded) == len(queries)
            assert loaded[0]["id"] == queries[0]["id"]
        finally:
            Path(f.name).unlink()
