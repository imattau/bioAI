import torch
from src.vsa import VSA, HopfieldNet


class TestHopfieldNet:
    def setup_method(self):
        self.vsa = VSA(dim=100)
        self.hop = HopfieldNet(dim=100)
        self.hop.to(self.vsa.device)

    def test_store_and_recall(self):
        p = self.vsa.make_vector()
        self.hop.store(p)
        noise = 0.1 * torch.randn(100, device=p.device)
        recalled = self.hop.recall(p + noise, steps=20)
        sim = self.vsa.similarity(p, recalled)
        assert sim.item() > 0.8

    def test_energy_decreases(self):
        p = self.vsa.make_vector()
        self.hop.store(p)
        e0 = self.hop.energy(p)
        noise = 0.1 * torch.randn(100, device=p.device)
        recalled = self.hop.recall(p + noise, steps=5)
        e1 = self.hop.energy(recalled)
        assert e1 <= e0 + 1e-4

    def test_modern_recall_separates_correlated_patterns(self):
        hop = HopfieldNet(dim=100, retrieval_mode="modern")
        hop.to(self.vsa.device)
        common = self.vsa.make_vector()
        patterns = []
        for _ in range(20):
            pattern = common.clone()
            flip_indices = torch.randperm(100)[:25]
            pattern[flip_indices] *= -1
            patterns.append(pattern)
        hop.store_batch(patterns)
        for index, pattern in enumerate(patterns):
            noise = 0.5 * torch.randn(100, device=pattern.device)
            recalled = hop.recall(pattern + noise)
            predicted = torch.stack(patterns) @ recalled
            assert predicted.argmax().item() == index
