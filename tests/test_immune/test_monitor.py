import torch
from src.immune import SelfMonitor


class TestSelfMonitor:
    def setup_method(self):
        self.monitor = SelfMonitor(activation_dim=32, n_detectors=3)

    def test_calibrate_and_score(self):
        normal = [torch.randn(32) * 0.5 for _ in range(50)]
        self.monitor.calibrate(normal)
        assert self.monitor.calibrated

    def test_normal_vs_anomaly(self):
        normal = [torch.randn(32) * 0.5 for _ in range(50)]
        self.monitor.calibrate(normal)
        normal_result = self.monitor.score(torch.randn(32) * 0.5)
        anomaly_result = self.monitor.score(torch.randn(32) * 10.0)
        assert normal_result["drift"] < anomaly_result["drift"]
