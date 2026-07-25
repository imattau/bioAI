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
