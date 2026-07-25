import torch
from .detector import DetectorEnsemble


class SelfMonitor:
    def __init__(self, activation_dim: int, n_detectors: int = 5, nu: float = 0.05):
        self.activation_dim = activation_dim
        self.ensemble = DetectorEnsemble(n_detectors=n_detectors, nu=nu)
        self.calibrated = False
        self.baseline_mean: torch.Tensor | None = None
        self.baseline_std: torch.Tensor | None = None

    def calibrate(self, activations: list[torch.Tensor]):
        stacked = torch.stack(activations)
        self.baseline_mean = stacked.mean(dim=0)
        self.baseline_std = stacked.std(dim=0).clamp(min=1e-8)
        self.ensemble.fit([stacked for _ in self.ensemble.detectors])
        self.calibrated = True

    def score(self, activation: torch.Tensor) -> dict:
        if not self.calibrated:
            return {"anomaly_score": 0.0, "drift": 0.0, "is_anomaly": False}
        z = (activation - self.baseline_mean) / self.baseline_std
        drift = z.square().mean().sqrt().item()
        ano_score = self.ensemble.ensemble_score(activation.unsqueeze(0)).item()
        return {"anomaly_score": ano_score, "drift": drift,
                "is_anomaly": ano_score < -0.5 or drift > 2.0}
