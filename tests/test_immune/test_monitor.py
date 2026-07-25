import torch
from src.vsa import VSA, HopfieldNet
from src.immune import SelfMonitor


class TestSelfMonitor:
    def setup_method(self):
        self.vsa = VSA(dim=32, device="cpu")
        self.hop = HopfieldNet(dim=32)
        patterns = [self.vsa.make_vector() for _ in range(5)]
        for p in patterns:
            self.hop.store(p)
        self.monitor = SelfMonitor(self.hop, energy_threshold=3.0)

    def test_calibrate_and_score(self):
        normal = [self.hop.recall(p + 0.1 * torch.randn(32), steps=10)
                  for p in [self.vsa.make_vector() for _ in range(5)]]
        for p in normal:
            self.hop.store(p)
        self.monitor.calibrate(normal)
        assert self.monitor.calibrated

    def test_normal_vs_anomaly(self):
        stored = [self.vsa.make_vector() for _ in range(5)]
        for p in stored:
            self.hop.store(p)
        normal = [self.hop.recall(p + 0.1 * torch.randn(32), steps=10) for p in stored]
        self.monitor.calibrate(normal)
        normal_result = self.monitor.score(self.hop.recall(stored[0] + 0.1 * torch.randn(32), steps=10))
        anomaly_result = self.monitor.score(torch.randn(32))
        assert normal_result["energy_z"] < anomaly_result["energy_z"]
