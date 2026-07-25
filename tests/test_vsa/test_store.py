import torch
from src.vsa import VSA, AssociativeStore


class TestAssociativeStore:
    def setup_method(self):
        self.vsa = VSA(dim=1000)
        self.store = AssociativeStore(dim=1000, capacity=10)

    def test_insert_and_lookup(self):
        k = self.vsa.make_vector()
        self.store.insert(k)
        results = self.store.lookup(k, k=1)
        assert len(results) == 1
        sim = self.vsa.similarity(results[0][0], k)
        assert sim > 0.99

    def test_capacity_eviction(self):
        keys = [self.vsa.make_vector() for _ in range(15)]
        for k in keys:
            self.store.insert(k)
        assert len(self.store) == 10
        first = self.store.lookup(keys[0], k=1)
        assert len(first) == 0 or first[0][1] < 0.5
