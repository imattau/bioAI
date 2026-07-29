"""Tests for ANN-indexed lookup in AssociativeStore (Phase 2).

Verifies correctness vs exact search at scale.
"""

import torch

from src.vsa import VSA
from src.vsa.store import AssociativeStore


class TestANNCorrectness:
    def setup_method(self):
        self.vsa = VSA(dim=500, device="cpu")
        self.dim = 500

    def test_lsh_fallback_to_exact_when_empty(self):
        store = AssociativeStore(dim=self.dim, lsh_bits=16, lsh_threshold=10)
        result = store.lookup(torch.randn(self.dim), k=1)
        assert result == []

    def test_exact_below_threshold(self):
        store = AssociativeStore(dim=self.dim, lsh_bits=16, lsh_threshold=100)
        for i in range(10):
            store.insert(self.vsa.make_vector(), self.vsa.make_vector())
        q = self.vsa.make_vector()
        result = store.lookup(q, k=1)
        assert len(result) == 1

    def test_ann_matches_exact_top1(self):
        store = AssociativeStore(dim=self.dim, lsh_bits=16, lsh_threshold=5)
        keys = [self.vsa.make_vector() for _ in range(200)]
        for k in keys:
            store.insert(k, k)

        # Query with one of the stored keys — ANN should find it
        q = keys[100]
        ann = store.lookup(q, k=1)
        assert len(ann) == 1
        found, sim = ann[0]
        assert sim > 0.99, f"ANN should find exact match (sim={sim})"

    def test_ann_retrieves_similar_vectors(self):
        store = AssociativeStore(dim=self.dim, lsh_bits=12, lsh_threshold=5)
        base = self.vsa.make_vector()
        for i in range(300):
            store.insert(base + 0.1 * torch.randn(self.dim), self.vsa.make_vector())

        result = store.lookup(base, k=1)
        assert len(result) == 1
        assert result[0][1] > 0.5

    def test_ann_disabled_uses_exact(self):
        store = AssociativeStore(dim=self.dim, lsh_bits=0, lsh_threshold=0)
        for i in range(200):
            store.insert(self.vsa.make_vector(), self.vsa.make_vector())
        q = self.vsa.make_vector()
        result = store.lookup(q, k=1)
        assert len(result) == 1

    def test_ann_no_false_negatives_for_exact_match(self):
        store = AssociativeStore(dim=self.dim, lsh_bits=20, lsh_threshold=5)
        target = self.vsa.make_vector()
        store.insert(target, target)
        for i in range(499):
            store.insert(self.vsa.make_vector(), self.vsa.make_vector())

        result = store.lookup(target, k=5)
        found = any((r[0] - target).abs().max().item() < 1e-4 for r in result)
        assert found, "ANN must find exact match in top-5"


