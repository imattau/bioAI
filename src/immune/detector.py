import torch
from sklearn.svm import OneClassSVM


class NegativeSelectionDetector:
    def __init__(self, nu: float = 0.1, kernel: str = "rbf", gamma: str = "auto"):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.fitted = False

    def fit(self, activations: torch.Tensor):
        self.model.fit(activations.cpu().numpy())
        self.fitted = True

    def score(self, activations: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            return torch.full((activations.shape[0],), -1.0)
        return torch.from_numpy(self.model.score_samples(activations.cpu().numpy()))

    def predict(self, activations: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            return torch.ones(activations.shape[0], dtype=torch.long)
        return torch.from_numpy(self.model.predict(activations.cpu().numpy()))


class DetectorEnsemble:
    def __init__(self, n_detectors: int = 5, nu: float = 0.1):
        self.detectors = [NegativeSelectionDetector(nu=nu) for _ in range(n_detectors)]

    def fit(self, activations_batches: list[torch.Tensor]):
        for detector, batch in zip(self.detectors, activations_batches):
            detector.fit(batch)

    def ensemble_score(self, activations: torch.Tensor) -> torch.Tensor:
        scores = torch.stack([d.score(activations) for d in self.detectors])
        return scores.mean(dim=0)

    def ensemble_predict(self, activations: torch.Tensor,
                          threshold: float = 0.0) -> torch.Tensor:
        scores = self.ensemble_score(activations)
        return (scores < threshold).long()
