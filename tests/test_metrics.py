"""Tests for RAG evaluation metrics."""

import pytest
from ragbench.metrics import (
    precision_at_k, recall_at_k, mrr, ndcg_at_k, hit_rate_at_k,
    average_precision
)


class TestPrecisionAtK:
    def test_perfect(self):
        assert precision_at_k(["a", "b"], ["a", "b", "c"], 2) == 1.0

    def test_partial(self):
        assert precision_at_k(["a", "b"], ["a", "c", "d"], 2) == 0.5

    def test_empty_relevant(self):
        assert precision_at_k([], ["a", "b"], 2) == 0.0

    def test_empty_retrieved(self):
        assert precision_at_k(["a"], [], 5) == 0.0

    def test_k_zero(self):
        assert precision_at_k(["a"], ["a"], 0) == 0.0


class TestRecallAtK:
    def test_full_recall(self):
        assert recall_at_k(["a", "b"], ["a", "b", "c"], 5) == 1.0

    def test_half_recall(self):
        assert recall_at_k(["a", "b", "c", "d"], ["a", "e", "f"], 3) == 0.25

    def test_empty_relevant(self):
        assert recall_at_k([], ["a"], 5) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mrr(["a"], ["a", "b", "c"]) == 1.0

    def test_second_position(self):
        assert mrr(["b"], ["a", "b", "c"]) == 0.5

    def test_not_found(self):
        assert mrr(["z"], ["a", "b", "c"]) == 0.0

    def test_empty(self):
        assert mrr([], []) == 0.0


class TestNDCG:
    def test_perfect_ordering(self):
        scores = {"a": 3.0, "b": 2.0, "c": 1.0}
        assert ndcg_at_k(["a", "b", "c"], ["a", "b", "c"], 3, scores) == 1.0

    def test_reversed_ordering(self):
        scores = {"a": 3.0, "b": 2.0, "c": 1.0}
        result = ndcg_at_k(["a", "b", "c"], ["c", "b", "a"], 3, scores)
        assert result < 1.0
        assert result > 0.0

    def test_empty_retrieved(self):
        assert ndcg_at_k(["a"], [], 5) == 0.0


class TestHitRate:
    def test_hit_found(self):
        assert hit_rate_at_k(["a"], ["a", "b", "c"], 3) == 1.0

    def test_hit_not_found(self):
        assert hit_rate_at_k(["z"], ["a", "b", "c"], 3) == 0.0


class TestAveragePrecision:
    def test_perfect(self):
        assert average_precision(["a", "b"], ["a", "b"]) == 1.0

    def test_interleaved(self):
        ap = average_precision(["a", "b"], ["c", "a", "d", "b"])
        assert 0.0 < ap < 1.0

    def test_empty_relevant(self):
        assert average_precision([], ["a", "b"]) == 0.0
