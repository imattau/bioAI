import torch
from src.immune import NegativeSelectionDetector, DetectorEnsemble


class TestNegativeSelectionDetector:
    def setup_method(self):
        self.detector = NegativeSelectionDetector(nu=0.1)

    def test_fit_and_score(self):
        normal = torch.randn(100, 50) * 0.5
        self.detector.fit(normal)
        assert self.detector.fitted

    def test_ood_detection(self):
        normal = torch.randn(100, 50) * 0.5
        ood = torch.randn(10, 50) * 5.0
        self.detector.fit(normal)
        normal_score = self.detector.score(torch.randn(1, 50) * 0.5)
        ood_score = self.detector.score(torch.randn(1, 50) * 5.0)
        assert ood_score.item() < normal_score.item()


class TestDetectorEnsemble:
    def setup_method(self):
        self.ensemble = DetectorEnsemble(n_detectors=3, nu=0.1)

    def test_ensemble_fit_and_predict(self):
        batches = [torch.randn(50, 20) for _ in range(3)]
        self.ensemble.fit(batches)
        scores = self.ensemble.ensemble_score(torch.randn(5, 20))
        assert scores.shape == (5,)
